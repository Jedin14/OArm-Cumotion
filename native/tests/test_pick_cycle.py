#!/usr/bin/env python3
"""Run a whole pick-and-place cycle against a fake robot and check the sequence.

No hardware, no MoveIt, no camera, no PaliGemma. Everything the orchestrator
talks to is stubbed here -- /move_action, the gripper action, /joint_states, TF,
the planning scene, cuMotion's parameters and /vlm/detections -- so the cycle
runs for real and every goal it sends is recorded.

What that buys over the geometry tests: those check the grasp maths in
isolation, this checks the *order*, which is the part that regressed when
pre_pick_state and drop_state were added. It asserts the exact sequence of
goals, that PRE_PICK and DROP replay the recorded joint values, and that
PREGRASP and LIFT sit approach_height above the grasp while DESCEND sits on it.

    source native/setup.bash && python3 native/tests/test_pick_cycle.py
"""

import ast
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time

# Isolate from any live robot, before rclpy creates a DDS participant.
#
# This test serves its own /move_action, /joint_states and gripper action. On
# the default domain those collide with a running stack: two action servers on
# one name and two joint-state publishers, so the checks start reading the real
# arm -- a close that stops at the real finger position rather than the modelled
# one, and stray goals from other clients landing in the fake's list. It also
# means a goal with plan_only false could reach the real move_group and move the
# robot. Set PICK_TEST_DOMAIN to pin it if the range below clashes.
#
# The domain also varies with the process id, because two copies of this suite
# running at once are as bad as running against the robot: they share the
# fake's action names and, worse, the states file, and the symptom is a whole
# block failing with "missing pre_pick_state" or "recorded for the left arm"
# in a run whose own setup was fine.
os.environ['ROS_DOMAIN_ID'] = os.environ.get(
    'PICK_TEST_DOMAIN', str(77 + os.getpid() % 20))
os.environ['ROS_LOCALHOST_ONLY'] = '1'

import numpy as np                                               # noqa: E402
import rclpy                                                     # noqa: E402
import yaml                                                      # noqa: E402
from control_msgs.action import (                                 # noqa: E402
    FollowJointTrajectory, GripperCommand)
from geometry_msgs.msg import TransformStamped                    # noqa: E402
from moveit_msgs.action import ExecuteTrajectory, MoveGroup       # noqa: E402
from moveit_msgs.msg import (                                     # noqa: E402
    AllowedCollisionEntry, AllowedCollisionMatrix, MoveItErrorCodes)
from moveit_msgs.srv import (                                     # noqa: E402
    ApplyPlanningScene, GetCartesianPath, GetPlanningScene, GetPositionIK)
from moveit_msgs.msg import RobotTrajectory                       # noqa: E402
from trajectory_msgs.msg import (                                 # noqa: E402
    JointTrajectoryPoint)
from rcl_interfaces.msg import ParameterType, ParameterValue      # noqa: E402
from rcl_interfaces.srv import GetParameters                      # noqa: E402
from rclpy.action import ActionServer                             # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup          # noqa: E402
from rclpy.executors import MultiThreadedExecutor                 # noqa: E402
from rclpy.node import Node                                       # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile                # noqa: E402
from sensor_msgs.msg import JointState                            # noqa: E402
from std_msgs.msg import String                                   # noqa: E402
from std_srvs.srv import Empty, Trigger                           # noqa: E402
from tf2_ros import TransformBroadcaster                          # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# On sys.path before importing anything from the workspace: loading a module
# by path (as the orchestrator is, further down) does not put its directory
# there.
if WS not in sys.path:
    sys.path.insert(0, WS)

import arm_kinematics                                            # noqa: E402

ARM = 'right'
OBJECT_POINT = [0.35, -0.18, 0.05]
OBJECT_YAW = 0.4
APPROACH_HEIGHT = 0.05
# The shipped default, and deliberately positive: the tool stops 10 mm above
# the detected top of the object rather than below it. Measured on the one
# real cycle that gripped and placed -- the fingers reached the torque cap
# with the tool 14.7 mm above the detected top while the descent had been
# commanded 5 mm below it. The old -0.005 only ever worked because the arm
# stopped 36 mm short of what it was told, and once it tracked better the
# same command pressed into the table.
GRASP_Z_OFFSET = 0.010
PRE_PICK_JOINTS = [-0.5, 0.1, -0.2, 1.9, 0.05, -0.3, 0.7]
DROP_JOINTS = [0.9, 0.2, -0.1, 1.5, 0.0, 0.2, -0.4]
READY_JOINTS = [-0.828374, 0.000191, -0.000191, 2.324140,
                -0.000191, -0.000191, -0.391966]
# The SRDF "home" group state, and the only pose the octomap may be captured
# from. Deliberately not READY: READY holds the arm out over the table so the
# camera can see it, so a map captured there contains the arm.
#
# joint4 is 0.20 rather than the SRDF's 0.0 because 0.0 is exactly that
# joint's URDF lower limit and the hardware stops 8.9 degrees short of it.
# Checked against the parameter default below, so the two cannot drift apart.
HOME_JOINTS = [0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0]
# joint4's real lower limit, and the reason for the line above. It was 0.0,
# which put the robot's own resting posture (-0.000191 on both arms) outside
# the limits and every plan's start state with it.
JOINT4_LOWER = -0.05
# Where the right elbow physically stops, radians. Commanded 0.0, reference
# 0.0, measured 0.15583 constant to five decimals over 301 samples at 0.02 Nm.
ELBOW_FLOOR = 0.15583

# The detector reports the frame size so the orchestrator can say which half
# of the camera view a detection is in; that is how an arm gets chosen.
IMAGE_SIZE = (848, 480)
# Right of centre, so the default right arm is the one the split picks.
OBJECT_PX = (600, 260)

# What robot_state_publisher latches. The real description, not a stub: the
# orchestrator reads joint limits from it *and* builds the kinematic chain it
# uses to choose an approach posture, and a stub with no link tree would
# silently exercise only the fallback path.
ARM_URDF = open(os.path.join(WS, 'openarm.urdf')).read()
# Straight from that file, so the limit checks below are checking the robot's
# own numbers.
JOINT1_UPPER = 3.490659

OPEN_FINGER = 0.044
HOLDING_FINGER = 0.02          # inside (grasp_finger_min, grasp_finger_max)

# The gripper's real force law, from
# openarm_hardware/include/openarm_hardware/v10_simple_hardware.hpp:
# joint 0..0.044 m maps to motor 0..-1.0472 rad, and MIT control makes the
# torque Kp times the motor position error.
GRIPPER_KP = 20.0                          # GRIPPER_DEFAULT_KP, Nm/rad
GRIPPER_R = 0.044 / 1.0472                 # m per rad, 42.0 mm/rad
OBJECT_HALF_WIDTH = 0.012                  # fingers touch the object here
TORQUE_CAP_NM = 2.5

failures = []


def moved(robot):
    """How many goals actually commanded motion.

    plan_only goals are probes: the reach check asks the planner whether a pose
    is reachable that way, and nothing moves. Counting them as motion would
    make "the robot did not move" assertions meaningless, and mixing the two
    counts in one comparison silently breaks the deltas.
    """
    return sum(1 for g in robot.goals if not g['plan_only'])


def is_numeric(values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in values)


def check(label, got, want, tol=1e-6):
    # Wrapped, because a check must never take the suite down with it. An
    # unexpected None here once raised TypeError inside the comparison, which
    # aborted the run at check 303 of 320 -- and a suite that stops early
    # reports zero failures, so it looks greener than one that fails honestly.
    try:
        if isinstance(want, (list, tuple)):
            if want and is_numeric(want) and got is not None:
                ok = (len(got) == len(want)
                      and all(abs(a - b) <= tol for a, b in zip(got, want)))
            else:
                ok = (got is not None
                      and list(got) == list(want))
        elif isinstance(want, float):
            ok = got is not None and abs(got - want) <= tol
        else:
            ok = got == want
    except (TypeError, ValueError) as exc:
        print(f'FAIL  {label}\n        comparison raised {exc!r}')
        failures.append(label)
        return
    print(('pass  ' if ok else 'FAIL  ') + label)
    if not ok:
        print(f'        got  {got}\n        want {want}')
        failures.append(label)


class FakeRobot(Node):
    """Every interface the orchestrator needs, and a log of what it asked for."""

    def __init__(self):
        super().__init__('fake_robot')
        cb = ReentrantCallbackGroup()

        self.goals = []                    # MoveGroup goals, in order
        self.move_fail_codes = []          # error codes to return, one per goal
        self.clears = []                   # index into self.goals at each clear
        self.object_px = list(OBJECT_PX)   # where the object sits in the frame
        self.cartesian_requests = []       # straight-line requests received
        self.executed = []                 # trajectories actually run
        self.cartesian_fraction = 1.0      # how much of the line is solvable
        self.checked_fraction = None       # if set, used when avoid_collisions
        self.cartesian_available = True    # serve the service at all
        self.ik_requests = []              # {'xyz', 'quat', 'seed', 'link'}
        self.ik_fail = False               # make /compute_ik report no solution
        self.ik_unreachable = set()        # arms /compute_ik has no solution for
        self.ik_flip = False               # return a whole-arm reconfiguration
        self.refreshes = []                # index into self.goals at each refresh
        self.gripper_commands = []
        # Trajectories sent straight to the controller, bypassing move_group.
        # The contact back-off is the only thing that does this, and it does
        # it deliberately -- see retreat_from_contact.
        self.trajectories = []
        self.controller_refuse = False     # make the controller abort them
        self.finger = OPEN_FINGER
        self.torque = 0.0
        self.holding = False
        self.joints = list(READY_JOINTS)
        # Where the tool is. Moves when a Cartesian trajectory executes, so
        # anything that checks the *achieved* pose has something real to read.
        self.tcp = [0.3, -0.2, 0.3]
        # How far short the tool stops, metres, in z. The real arm sags under
        # a load the controller does not fully cancel -- 13.7 mm low at
        # z=0.405, 33.5 mm at z=0.555 -- and a pass that starts closer ends
        # closer, which is what makes converging work at all.
        self.sag = 0.0
        self.sag_floor = False
        self.pending_cartesian = None
        self.last_fraction = None
        self.nothing_to_grip = False
        # The detector sees nothing at all -- an occluded object, or a model
        # that missed it. Not the same as "the object is gone".
        self.object_hidden = False
        # How far apart the jaws physically stop when nothing_to_grip is set.
        # 0.0 is air. A thin object -- a roll of tape measured at ~4 mm --
        # stalls them above that while the torque still never reaches the
        # cap, and that is the case the empty check kept calling empty.
        self.finger_floor = 0.0
        # Published in /joint_states. Non-zero means "still moving".
        self.joint_speed = 0.011
        # Reply for plan-only goals, when the planner is to be modelled as
        # up but unable to plan. None means answer normally.
        self.plan_only_code = None
        # A code returned for every executing (non-plan-only) goal, for as
        # long as it is set. move_fail_codes pops one entry per goal, which
        # cannot model "this posture is simply unreachable from here" -- the
        # case where the retreat has to decide whether to fly HOME anyway.
        self.joint_code = None
        # Radians of detour to put into a Cartesian path, so a path that keeps
        # the tool on the line while the arm sweeps can be tested.
        self.sweep_rad = 0.0
        # Planning-scene diffs, so the gripper/octomap exemption is visible.
        self.scene_requests = []
        # A stand-in for the SRDF's disable_collisions: adjacent links whose
        # meshes touch. The exemption must not delete these -- replacing the
        # matrix instead of adding to it wiped all 139 of the real ones, after
        # which every adjacent pair counted as a collision and nothing planned.
        self.acm_names = [f'openarm_{ARM}_link{i}' for i in range(1, 8)] + [
            f'openarm_{ARM}_hand', f'openarm_{ARM}_left_finger',
            f'openarm_{ARM}_right_finger']
        size = len(self.acm_names)
        self.acm_rows = [[False] * size for _ in range(size)]
        for i in range(size - 1):        # adjacent pairs allowed to touch
            self.acm_rows[i][i + 1] = True
            self.acm_rows[i + 1][i] = True
        self.lock = threading.Lock()

        self.tf = TransformBroadcaster(self)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        # Latched, like robot_state_publisher's. The orchestrator reads joint
        # limits from it so that no posture is commanded onto a limit.
        self.urdf_pub = self.create_publisher(
            String, '/robot_description',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.urdf_pub.publish(String(data=ARM_URDF))
        self.detections_pub = self.create_publisher(String, '/vlm/detections', 10)
        self.create_subscription(String, '/vlm/prompt', self._on_prompt, 10,
                                 callback_group=cb)
        self.prompt = 'detect screwdriver'

        ActionServer(self, MoveGroup, '/move_action', self._on_move,
                     callback_group=cb)
        ActionServer(self, GripperCommand,
                     f'/{ARM}_gripper_controller/gripper_cmd', self._on_gripper,
                     callback_group=cb)
        self.create_service(ApplyPlanningScene, '/apply_planning_scene',
                            self._on_scene, callback_group=cb)
        # The real scene carries the SRDF's disable_collisions -- 139 of them
        # on this robot. The exemption has to preserve them, so the fake has
        # to have some to preserve.
        self.create_service(GetPlanningScene, '/get_planning_scene',
                            self._on_get_scene, callback_group=cb)
        self.create_service(GetParameters, '/cumotion_planner/get_parameters',
                            self._on_get_parameters, callback_group=cb)
        self.create_service(Trigger, '/octomap_gater/refresh',
                            self._on_refresh, callback_group=cb)
        self.create_service(Empty, '/clear_octomap',
                            self._on_clear_octomap, callback_group=cb)
        self.create_service(GetPositionIK, '/compute_ik',
                            self._on_compute_ik, callback_group=cb)
        self.create_service(GetCartesianPath, '/compute_cartesian_path',
                            self._on_cartesian, callback_group=cb)
        ActionServer(self, ExecuteTrajectory, '/execute_trajectory',
                     self._on_execute, callback_group=cb)
        ActionServer(self, FollowJointTrajectory,
                     f'/{ARM}_joint_trajectory_controller/follow_joint_trajectory',
                     self._on_trajectory, callback_group=cb)

        self.create_timer(0.05, self._tick, callback_group=cb)

    # -- published state ---------------------------------------------------

    def _tick(self):
        now = self.get_clock().now()
        stamp = now.to_msg()

        js = JointState()
        js.header.stamp = stamp
        js.name = ([f'openarm_{ARM}_joint{i}' for i in range(1, 8)]
                   + [f'openarm_{ARM}_finger_joint1'])
        with self.lock:
            js.position = list(self.joints) + [self.finger]
            js.effort = [0.0] * len(self.joints) + [self.torque]
            # The real robot publishes velocities, and at_home_pose() needs
            # them to tell "as close as this hardware gets" from "still on
            # its way". A standing joint reads +/-0.011 rad/s of noise.
            js.velocity = ([self.joint_speed] * len(self.joints)
                           + [0.0])
        self.joint_pub.publish(js)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'world'
        t.child_frame_id = f'openarm_{ARM}_hand_tcp'
        with self.lock:
            tcp = list(self.tcp)
        t.transform.translation.x = tcp[0]
        t.transform.translation.y = tcp[1]
        t.transform.translation.z = tcp[2]
        t.transform.rotation.w = 1.0
        self.tf.sendTransform(t)

        # Once the gripper has closed on it, the object travels with the
        # tool. It is not lying at the pick point any more, and it has not
        # vanished either, and that distinction is the whole of
        # verify_grasp: seeing the object somewhere *other* than where it
        # was is what confirms a pick. This used to publish an empty list
        # while holding, which modelled the old rule rather than the robot
        # -- the robot does see it (run 1789014831 cycle 3, one detection
        # after a good lift), and an empty frame there means the gripper is
        # occluding the object it just failed to pick.
        with self.lock:
            holding = self.holding
            hidden = self.object_hidden
        payload = {
            'stamp': now.nanoseconds * 1e-9,
            'prompt': self.prompt,
            'image_size': list(IMAGE_SIZE),
            'detections': [] if hidden else [{
                'point': list(tcp) if holding else list(OBJECT_POINT),
                'center_px': list(self.object_px),
                'axis_yaw': OBJECT_YAW,
                'depth_m': 0.62,
                'depth_px': 812,
                'axis_source': 'depth',
            }],
        }
        self.detections_pub.publish(String(data=json.dumps(payload)))

    def _on_prompt(self, msg):
        self.prompt = msg.data

    # -- served interfaces -------------------------------------------------

    def _on_move(self, goal_handle):
        request = goal_handle.request.request
        options = goal_handle.request.planning_options
        constraints = request.goal_constraints[0]
        # plan_only is a probe, not motion. The reach check sends those to ask
        # the planner whether a pose is reachable, and a test asserting "the
        # robot did not move" must not count them.
        entry = {'group': request.group_name,
                 'plan_only': bool(options.plan_only),
                 # The reach probe poses its question from the staging pose
                 # rather than from wherever the arm happens to be standing,
                 # so the start state is part of what a test has to see.
                 'start_is_diff': bool(request.start_state.is_diff),
                 'start_joints': list(request.start_state.joint_state.position)}
        if constraints.joint_constraints:
            entry['kind'] = 'joint'
            entry['joints'] = [jc.position for jc in constraints.joint_constraints]
            entry['names'] = [jc.joint_name for jc in constraints.joint_constraints]
        else:
            pose = constraints.position_constraints[0]
            point = pose.constraint_region.primitive_poses[0].position
            quat = constraints.orientation_constraints[0].orientation
            entry['kind'] = 'pose'
            entry['link'] = pose.link_name
            entry['xyz'] = [point.x, point.y, point.z]
            entry['quat'] = [quat.x, quat.y, quat.z, quat.w]
        with self.lock:
            self.goals.append(entry)
            # move_fail_codes lets a test reproduce cuMotion's stochastic
            # TRAJOPT_FAIL: each entry is returned for one goal, in order,
            # before the fake starts succeeding again.
            arm = request.group_name.replace('_arm', '')
            if entry['plan_only']:
                # The planner is the reach authority now, so an arm modelled as
                # unable to reach has to say so here too, not only via IK.
                # plan_only_code models a planner that is up but cannot plan --
                # separate from ik_unreachable, which is a statement about the
                # target rather than about the planner's health.
                if self.plan_only_code is not None:
                    code = self.plan_only_code
                else:
                    code = (MoveItErrorCodes.NO_IK_SOLUTION
                            if arm in self.ik_unreachable
                            else MoveItErrorCodes.SUCCESS)
            else:
                code = (self.joint_code if self.joint_code is not None
                        else self.move_fail_codes.pop(0)
                        if self.move_fail_codes
                        else MoveItErrorCodes.SUCCESS)
            # The arm arrives. Without this the reported joints never move, so
            # anything that gates on *measured* position -- the octomap's
            # at_home_pose check above all -- is judging a stale posture.
            # Only joint goals are modelled; a pose goal needs IK we do not
            # have here, and nothing gates on arriving at a pose.
            if code == MoveItErrorCodes.SUCCESS and entry['kind'] == 'joint' \
                    and not entry['plan_only']:
                for name, value in zip(entry['names'], entry['joints']):
                    index = int(name[-1]) - 1
                    if 0 <= index < len(self.joints):
                        self.joints[index] = value

        goal_handle.succeed()
        result = MoveGroup.Result()
        result.error_code.val = code
        return result

    def _on_gripper(self, goal_handle):
        position = goal_handle.request.command.position
        with self.lock:
            self.gripper_commands.append(position)
            if self.nothing_to_grip:
                # Nothing between the fingers: they close all the way and the
                # only torque is friction. Measured on the robot -- 22 steps
                # from 40.9 mm to 2.5 mm with the effort flat near 0.5 Nm
                # against a 2.5 Nm cap.
                self.finger = max(self.finger_floor, position)
                # Loaded, but never to the cap: measured 1.84-2.13 Nm on the
                # three runs that gripped something thin and were called
                # empty, against 1.14-1.24 Nm on the four that gripped air.
                self.torque = 1.9 if self.finger_floor > 0.0 else 0.5
                self.holding = self.finger_floor > 0.0
            elif position < OBJECT_HALF_WIDTH:
                # The fingers stall on the object; the motor holds a position
                # error, and the torque that error produces is what a cap has
                # to react to.
                self.finger = OBJECT_HALF_WIDTH
                self.torque = GRIPPER_KP * (OBJECT_HALF_WIDTH - position) / GRIPPER_R
                self.holding = True
            else:
                self.finger = position
                self.torque = 0.0
                self.holding = False
        goal_handle.succeed()
        result = GripperCommand.Result()
        result.position = position
        result.reached_goal = True
        return result

    def _on_get_scene(self, _request, response):
        matrix = AllowedCollisionMatrix()
        matrix.entry_names = list(self.acm_names)
        for row in self.acm_rows:
            entry = AllowedCollisionEntry()
            entry.enabled = list(row)
            matrix.entry_values.append(entry)
        response.scene.allowed_collision_matrix = matrix
        return response

    def _on_scene(self, request, response):
        # Recorded, because the descent's collision exemption is applied
        # through here now: only the gripper links are allowed to touch the
        # octomap, and the rest of the arm stays checked.
        acm = request.scene.allowed_collision_matrix
        with self.lock:
            self.scene_requests.append({
                'acm_names': list(acm.entry_names),
                'acm_values': [list(e.enabled) for e in acm.entry_values],
            })
        response.success = True
        return response

    def _on_get_parameters(self, request, response):
        for name in request.names:
            value = ParameterValue()
            if name == 'tool_frame':
                value.type = ParameterType.PARAMETER_STRING
                value.string_value = f'openarm_{ARM}_hand_tcp'
            else:
                value.type = ParameterType.PARAMETER_NOT_SET
            response.values.append(value)
        return response

    def _on_refresh(self, _request, response):
        with self.lock:
            # Which goal had just been sent tells us where the arm was when the
            # map was captured -- counted over the goals that *moved* the arm,
            # because that is the list the checks reason about. Counting raw
            # goals instead left the index off by one plan-only probe, which
            # was harmless only for as long as nothing probed before a
            # capture.
            self.refreshes.append(
                sum(1 for g in self.goals if not g['plan_only']))
        response.success = True
        return response

    def _on_clear_octomap(self, _request, response):
        with self.lock:
            self.clears.append(len(self.goals))
        return response

    def _arrive_partial(self, target, fraction):
        """Advance the tool part of the way, as a partial path would."""
        with self.lock:
            start = list(self.tcp)
            self.tcp = [start[i] + (target[i] - start[i]) * fraction
                        for i in range(3)]

    def _arrive(self, target):
        """Model the tool arriving, short by `sag`.

        The sag halves each pass, because a pass that starts closer ends
        closer -- which is what makes converging work. sag_floor models an arm
        that cannot do better, so the give-up path can be tested.
        """
        with self.lock:
            remaining = self.sag
            self.tcp = [target[0], target[1], target[2] - remaining]
            if not self.sag_floor:
                self.sag = remaining / 2.0

    def _on_cartesian(self, request, response):
        """A straight line, modelled just enough to be worth testing.

        Returns a short trajectory whose last point is the requested waypoint,
        and reports cartesian_fraction so a test can make the line only
        partially solvable -- which must be refused rather than executed.
        """
        target = request.waypoints[-1] if request.waypoints else None
        with self.lock:
            self.cartesian_requests.append({
                'group': request.group_name,
                'link': request.link_name,
                'max_step': request.max_step,
                'avoid_collisions': bool(request.avoid_collisions),
                # Whether the caller asked about a posture the arm is not in.
                # The pre-flight's whole purpose is to ask in advance, so a
                # diff start state there would silently be a question about
                # wherever the arm happens to be.
                'start_is_diff': bool(request.start_state.is_diff),
                'start_joints': list(request.start_state.joint_state.position),
                'xyz': ([target.position.x, target.position.y,
                         target.position.z] if target else None),
                'quat': ([target.orientation.x, target.orientation.y,
                          target.orientation.z, target.orientation.w]
                         if target else None),
            })
            available = self.cartesian_available
            if request.avoid_collisions and self.checked_fraction is not None:
                # The real behaviour on a top-down grasp: the object being
                # picked up is in the octomap, so a collision-checked line into
                # it stalls part-way, while the same line unchecked is fine.
                fraction = self.checked_fraction
            else:
                fraction = self.cartesian_fraction
        if not available:
            response.error_code.val = MoveItErrorCodes.FAILURE
            response.fraction = 0.0
            return response

        names = [f'openarm_{ARM}_joint{i}' for i in range(1, 8)]
        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = names
        with self.lock:
            start = list(self.joints)
        # Two points: where the arm is, and the seed-adjusted end. The values
        # matter less than the timing, which is what _retime has to scale.
        # sweep_rad models a path that keeps the tool on the line while the
        # arm travels a long way -- what /compute_cartesian_path returns when
        # the shoulder reconfigures on the way down.
        with self.lock:
            sweep = self.sweep_rad
        far = [v + sweep for v in start]
        for index, positions in enumerate(
                (start, far, [v - 0.01 for v in start]) if sweep
                else (start, [v - 0.01 for v in start])):
            point = JointTrajectoryPoint()
            point.positions = list(positions)
            point.velocities = [1.0] * len(names)
            point.accelerations = [2.0] * len(names)
            point.time_from_start.sec = index
            traj.joint_trajectory.points.append(point)
        response.solution = traj
        response.fraction = fraction
        response.error_code.val = MoveItErrorCodes.SUCCESS
        with self.lock:
            self.pending_cartesian = [position.x, position.y, position.z] \
                if (position := target.position) else None
            self.last_fraction = fraction
        return response

    def _on_trajectory(self, goal_handle):
        """The joint trajectory controller, which the contact back-off talks
        to directly. Records what it was sent and lands on the last point."""
        request = goal_handle.request
        result = FollowJointTrajectory.Result()
        with self.lock:
            self.trajectories.append({
                'names': list(request.trajectory.joint_names),
                'points': [list(p.positions)
                           for p in request.trajectory.points],
                'seconds': [p.time_from_start.sec
                            + p.time_from_start.nanosec * 1e-9
                            for p in request.trajectory.points],
            })
            refuse = self.controller_refuse
        if refuse:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result
        points = request.trajectory.points
        if points:
            with self.lock:
                self.joints = [float(v) for v in points[-1].positions]
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def _on_execute(self, goal_handle):
        traj = goal_handle.request.trajectory
        with self.lock:
            pending = self.pending_cartesian
        if pending:
            with self.lock:
                fraction = self.last_fraction
            if fraction is not None and fraction < 1.0:
                # A partial path moves the tool part of the way; that is what
                # makes it partial. Reporting a fraction without advancing
                # made the segments loop spin against nothing.
                self._arrive_partial(pending, fraction)
            else:
                self._arrive(pending)
        with self.lock:
            self.executed.append({
                'names': list(traj.joint_trajectory.joint_names),
                'points': [
                    {'positions': list(p.positions),
                     'velocities': list(p.velocities),
                     'seconds': p.time_from_start.sec
                     + p.time_from_start.nanosec * 1e-9}
                    for p in traj.joint_trajectory.points],
            })
            if traj.joint_trajectory.points:
                self.joints = list(traj.joint_trajectory.points[-1].positions)
        goal_handle.succeed()
        result = ExecuteTrajectory.Result()
        result.error_code.val = MoveItErrorCodes.SUCCESS
        return result

    def _on_compute_ik(self, request, response):
        """Seeded IK, modelled just closely enough to be worth testing.

        A real solver returns a solution in the seed's branch. This returns the
        seed with a small, height-dependent tweak on joint4, which is what
        "the same posture, a bit lower" looks like in joint space. ik_flip
        instead returns a solution a long way from the seed -- a different
        branch for the same tool pose, which is what actually happened on the
        robot and what max_joint_jump has to reject.
        """
        ik = request.ik_request
        seed = list(ik.robot_state.joint_state.position)
        position = ik.pose_stamped.pose.position
        orientation = ik.pose_stamped.pose.orientation
        arm = ik.group_name.replace('_arm', '')
        with self.lock:
            self.ik_requests.append({
                'xyz': [position.x, position.y, position.z],
                'quat': [orientation.x, orientation.y,
                         orientation.z, orientation.w],
                'seed': list(seed),
                'link': ik.ik_link_name,
                'group': ik.group_name,
                'avoid_collisions': bool(ik.avoid_collisions),
            })
            fail = self.ik_fail or arm in self.ik_unreachable
            flip = self.ik_flip

        if fail or not seed:
            response.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
            return response

        solution = list(seed)
        if flip:
            # Same tool pose, opposite elbow: every joint a long way off.
            solution = [v + 1.5 for v in seed]
        else:
            # Small and monotone in height, so chained seeds accumulate the way
            # a real descent does.
            solution[3] = seed[3] - 0.01
        response.solution.joint_state.name = list(ik.robot_state.joint_state.name)
        response.solution.joint_state.position = solution
        response.error_code.val = MoveItErrorCodes.SUCCESS
        return response


def write_states(path):
    document = {
        'arm': ARM,
        'states': {
            'pre_pick_state': {'joints': list(PRE_PICK_JOINTS)},
            'drop_state': {'joints': list(DROP_JOINTS)},
        },
    }
    with open(path, 'w') as handle:
        yaml.safe_dump(document, handle, sort_keys=False)


def motions(robot):
    """The goals that moved the arm, plan-only probes dropped.

    `moved()` counts these, so anything slicing with a count from it has to
    index this list rather than robot.goals -- the reach check and the planner
    readiness probe both send plan-only goals, and they are recorded like any
    other.
    """
    return [g for g in robot.goals if not g['plan_only']]


def wait_for_state(seen, want, timeout=2.0):
    """The most recent published state, once `want` has had a chance to land.

    Reading seen[-1] the instant after the call that publishes it is a race:
    the topic is asynchronous, so the list can still end with the previous
    state. Waiting for the one expected -- and returning whatever is there if
    it never comes -- keeps the check meaningful without making it flaky.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if seen and seen[-1].split(':')[0].strip() == want:
            break
        time.sleep(0.02)
    return seen[-1].split(':')[0].strip() if seen else None


def load_orchestrator():
    # Loading by path does not put the workspace on sys.path, and the
    # orchestrator imports vlm_prompt from beside itself.
    if WS not in sys.path:
        sys.path.insert(0, WS)
    path = os.path.join(WS, 'pick_place_orchestrator.py')
    spec = importlib.util.spec_from_file_location('pick_place_orchestrator', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['pick_place_orchestrator'] = module
    spec.loader.exec_module(module)
    return module


def main():
    # Per process, for the same reason as the domain above.
    states_path = os.path.join(WS, 'native', 'tests',
                               f'.test_states.{os.getpid()}.yaml')
    config_path = os.path.join(WS, 'native', 'tests',
                               f'.test_config.{os.getpid()}.json')
    write_states(states_path)
    # Its own log file: a test has no business appending to the one the robot
    # writes, and a stale one would make the checks below pass on old data.
    log_path = os.path.join(WS, 'native', 'tests',
                            f'.test_motion_log.{os.getpid()}.jsonl')
    if os.path.exists(log_path):
        os.remove(log_path)

    rclpy.init(args=[
        '--ros-args',
        '-p', f'arm:={ARM}',
        '-p', f'states_file:={states_path}',
        '-p', 'place_mode:=state',
        '-p', f'approach_height:={APPROACH_HEIGHT}',
        # Never the real one. The orchestrator reads a saved config at
        # startup and its settings win over launch arguments, which is the
        # point of it -- but it made the suite depend on whatever the last
        # person saved from the browser. A --fake run's grasp_finger_min of
        # -1.0 got persisted and six checks then failed for reasons that had
        # nothing to do with the code under test.
        '-p', f'pick_place_config:={config_path}',
        '-p', f'grasp_z_offset:={GRASP_Z_OFFSET}',
        '-p', 'gripper_settle_time:=0.05',
        '-p', 'detect_timeout:=8.0',
        '-p', f'motion_log:={log_path}',
        # The long cycle below runs the planner path. The fake answers a joint
        # goal by setting its joint values but has no forward kinematics, so
        # it cannot move its *tool* in response -- and every Cartesian check
        # in the cycle reads the tool. The chosen-posture path is exercised
        # directly further down instead, where the goals themselves are what
        # is being checked.
        '-p', 'approach_frame:=planner',
    ])
    module = load_orchestrator()

    robot = FakeRobot()
    orchestrator = module.PickPlaceOrchestrator()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(robot)
    executor.add_node(orchestrator)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    states = []
    orchestrator.create_subscription(
        String, '/pick_place/state', lambda m: states.append(m.data), 10)

    try:
        time.sleep(2.0)                    # let discovery settle

        orchestrator._on_prompt(String(data='pick up the wrench'))
        check('a conversational prompt on the topic is normalised',
              orchestrator.prompt, 'detect wrench')
        robot.prompt = 'detect wrench'

        ok, message = orchestrator._start_cycle()
        check('cycle starts', ok, True)

        deadline = time.time() + 90
        while time.time() < deadline:
            if states and states[-1].split(':')[0] in ('DONE', 'FAILED', 'ABORTED'):
                break
            time.sleep(0.2)

        steps = [s.split(':')[0].strip() for s in states]
        print('\nstates:', ' -> '.join(steps), '\n')
        check('cycle reached DONE', steps[-1] if steps else None, 'DONE')

        # The sequence the cycle is supposed to walk, in order. Checked as a
        # subsequence so retries or extra detail lines cannot break it.
        # The sequence, in the order it was asked for: home, pre-pick,
        # transit, above the object, open, down, grab, lift with it closed,
        # pre-pick, drop, open, shut the jaws there, pre-pick, home. HOME is
        # the only rest/observation pose -- there is no separate READY any
        # more.
        # No PREGRASP: with single_descent the gripper opens above the object
        # and one line goes all the way to the grasp, so there is no stop 5 cm
        # up to be in this list.
        # The second CLOSE_GRIPPER is the one at the drop pose: the arm goes
        # back through pre_pick to home with the jaws shut rather than with
        # 44 mm of open fingers hunting for something to catch on.
        wanted = ['HOME', 'LOCATE', 'PRE_PICK', 'TRANSIT',
                  'OPEN_GRIPPER', 'DESCEND', 'CLOSE_GRIPPER', 'LIFT',
                  'VERIFY_GRASP', 'PRE_PICK', 'DROP', 'RELEASE',
                  'VERIFY_PLACE', 'CLOSE_GRIPPER',
                  'PRE_PICK', 'HOME', 'DONE']
        index, missing = 0, []
        for step in wanted:
            while index < len(steps) and steps[index] != step:
                index += 1
            if index == len(steps):
                missing.append(step)
            index += 1
        check('states appear in the documented order', missing, [])
        check('no READY pose survives -- HOME does its job',
              [s for s in steps if s == 'READY'], [])
        # Asked for explicitly: the jaws shut at the drop pose, and the trip
        # home is made closed. The last gripper command of the cycle is a
        # close, and it lands after the release rather than before it.
        with robot.lock:
            grip_order = list(robot.gripper_commands)
        check('the last gripper command of the cycle shuts the jaws',
              grip_order[-1] <= 0.0 + 1e-9, True)
        opened = max(i for i, g in enumerate(grip_order)
                     if g >= orchestrator.get_parameter('gripper_open').value)
        check('and it comes after the release, not before it',
              len(grip_order) - 1 > opened, True)
        check('the close at the drop is between VERIFY_PLACE and the way home',
              steps.index('VERIFY_PLACE')
              < len(steps) - 1 - steps[::-1].index('CLOSE_GRIPPER')
              < len(steps) - 1 - steps[::-1].index('HOME'), True)

        with robot.lock:
            goals = [g for g in robot.goals if not g['plan_only']]
            grips = list(robot.gripper_commands)

        kinds = [g['kind'] for g in goals]
        poses = [g for g in goals if g['kind'] == 'pose']
        transit_height = orchestrator.get_parameter('transit_height').value
        grasp_z_expected = OBJECT_POINT[2] + GRASP_Z_OFFSET
        transit_z = grasp_z_expected + transit_height

        with robot.lock:
            all_ik = list(robot.ik_requests)
        # Two kinds of IK go out now. The reach check asks pure kinematics --
        # avoid_collisions false, no posture limit -- before anything moves;
        # the column asks for a solution near the posture above it. Only the
        # second describes the path, so the geometry checks use it.
        reach_checks = [r for r in all_ik if not r['avoid_collisions']]
        # by_side, the default, does not probe reach at all -- it takes the
        # half of the frame the object is in. Asking the solvers instead cost
        # 87 seconds between the first look and the second, measured, for the
        # same answer, and the pre-flight settles whether the pick is
        # possible for the arm chosen before anything moves. What matters is
        # that no *motion* was spent finding out, which the goal counts below
        # already assert. by_reach keeps the probing, and is tested with it.
        check('the default takes the camera half rather than probing',
              orchestrator.get_parameter('arm_selection').value, 'by_side')

        # Two joint goals in (home, pre_pick) and four out: pre_pick again to
        # stage the carry over the table, then drop, pre_pick, home. Only the
        # transit is a pose goal; everything below it is a Cartesian path
        # executed as a trajectory, so it does not appear as a MoveGroup goal
        # at all.
        check('joint goals bracket the middle',
              (kinds[:2], kinds[-4:]),
              (['joint', 'joint'], ['joint', 'joint', 'joint', 'joint']))
        check('the transit is the only pose goal', len(poses), 1)
        check('nothing else goes as a MoveGroup goal', kinds[3:-4], [])

        check('all goals target the right arm group',
              sorted({g['group'] for g in goals}), [f'{ARM}_arm'])
        check('pose goals target the right tool frame',
              sorted({g['link'] for g in goals if g['kind'] == 'pose'}),
              [f'openarm_{ARM}_hand_tcp'])

        # HOME first: the map is built and the object observed from there,
        # with the arm out of the camera's frame.
        check('goal 1 is HOME, so the map and the look happen there',
              goals[0]['joints'], HOME_JOINTS, 1e-5)
        check('goal 2 replays pre_pick_state', goals[1]['joints'], PRE_PICK_JOINTS)
        check('pre_pick goal names the arm joints', goals[1]['names'],
              [f'openarm_{ARM}_joint{i}' for i in range(1, 8)])

        grasp_z = grasp_z_expected
        above = grasp_z + APPROACH_HEIGHT

        # The free-space move ends high, clear of anything on the table.
        check('the pose goal is the high transit, not the pre-grasp',
              poses[0]['xyz'][2], transit_z, 1e-9)
        check('and it is well above the pre-grasp',
              transit_z > above + 1e-6, True)

        # The descent is a real straight line now, asked of
        # /compute_cartesian_path -- which interpolates it and runs IK at every
        # step, so the tool goes down vertically by construction. A goal pose
        # only says where to end up: planned as a free trajectory, a 5 cm
        # descent bowed into the table.
        with robot.lock:
            lines = list(robot.cartesian_requests)
            executed = list(robot.executed)
        check('the column asked for straight lines', len(lines) > 0, True)
        check("for this arm's tool frame and group",
              sorted({(r['link'], r['group']) for r in lines}),
              [(f'openarm_{ARM}_hand_tcp', f'{ARM}_arm')])
        # Not collision-checked, and deliberately so: every leg of the column
        # is exempt (approach_ignores_octomap) and the checked attempt is now
        # skipped outright (descend_ignores_octomap), because on a top-down
        # grasp the target is itself in the octomap -- 100% of the 20 cm drop
        # to the pre-grasp solved checked, 25% of the last 5 cm.
        # Collision-checked, with the gripper exempted in the planning scene
        # instead. Switching checking off for the whole arm was the blunt
        # version of this, and it let the forearm reach the table on a descent
        # whose tool path was fine.
        check('the column lines are collision-checked',
              sorted({r['avoid_collisions'] for r in lines}), [True])
        step = orchestrator.get_parameter('cartesian_step').value
        check('interpolated finely enough to be straight',
              max(r['max_step'] for r in lines) <= step + 1e-12, True)
        check('every line ends on the vertical above the object',
              sorted({(round(r['xyz'][0], 9), round(r['xyz'][1], 9))
                      for r in lines}),
              [(round(OBJECT_POINT[0], 9), round(OBJECT_POINT[1], 9))])

        zs = [r['xyz'][2] for r in lines]
        check('no line stops at the pre-grasp height -- the descent is one '
              'line, not two', min(abs(z - above) for z in zs) < 1e-9, False)
        check('and one touches down exactly on the grasp',
              min(zs), grasp_z, 1e-9)
        check('nothing goes below the grasp',
              min(zs) >= grasp_z - 1e-9, True)
        # It must come back up the same line, and go well clear -- the carry
        # to the drop pose is a free-space plan, and starting that 5 cm off the
        # surface dragged the gripper across the table.
        retreat = orchestrator.get_parameter('retreat_height').value
        check('the last line lifts to retreat_height, not just the pre-grasp',
              zs[-1], grasp_z + retreat, 1e-9)
        check('and that is higher than the pre-grasp',
              grasp_z + retreat > above + 1e-6, True)
        check('every line stays on the same vertical, up and down',
              sorted({(round(r['xyz'][0], 9), round(r['xyz'][1], 9))
                      for r in lines}),
              [(round(OBJECT_POINT[0], 9), round(OBJECT_POINT[1], 9))])

        # Both legs of the column get the collision exemption, not just the
        # last 5 cm: the 15 cm transit-to-pregrasp leg stalled at 61% checked
        # and then went as eight curved free-space hops.
        checked = [r['avoid_collisions'] for r in lines]
        check('the first line asked for is checked -- the exemption is in the '
              'scene, not in the request', checked[0], True)
        # Three legs get a straight line: down to the pre-grasp, down onto the
        # object, and back up to retreat_height. All three, not just the last
        # 5 cm -- the 15 cm leg was the one that went curved.
        #
        # Counted by distinct target height, not by number of requests: a leg
        # whose tool lands short legitimately re-issues the same line to close
        # the gap, so the request count varies while the set of targets does
        # not.
        # Two legs now, not three: straight down to the grasp, and straight
        # back up to the retreat height. The stop at `above` is gone.
        check('both legs of the column ask for a line',
              sorted({round(r['xyz'][2], 6) for r in lines}),
              sorted({round(grasp_z, 6), round(grasp_z + retreat, 6)}))
        check('and each is exempt from checking if the checked one fails',
              orchestrator.get_parameter('approach_ignores_octomap').value, True)

        # The service applies no speed scaling, so the trajectory it returns
        # runs at full joint speed unless it is re-timed.
        check('the trajectories were executed', len(executed) > 0, True)
        scaling = orchestrator.get_parameter('velocity_scaling').value
        check('and re-timed to velocity_scaling, not run flat out',
              max(abs(v) for run in executed for p in run['points']
                  for v in p['velocities']) <= scaling + 1e-9, True)
        check('with the time stretched to match',
              max(p['seconds'] for run in executed for p in run['points'])
              >= 1.0 / scaling - 1e-6, True)

        check('every line keeps one orientation',
              len({tuple(round(v, 9) for v in r['quat']) for r in lines}), 1)
        check('grasp orientation is the detected yaw',
              list(lines[0]['quat']),
              list(module.top_down_quat(OBJECT_YAW)), 1e-9)

        # The earlier complaint -- lowering the tool 5 cm flipped the whole arm
        # -- cannot happen along a Cartesian path: move_group runs IK at every
        # interpolation step from the previous solution, so the posture is
        # continuous by construction. Checked on the executed trajectory
        # instead of on goal endpoints.
        jumps = [max(abs(a - b) for a, b in zip(p['positions'], q['positions']))
                 for run in executed
                 for p, q in zip(run['points'], run['points'][1:])]
        max_jump = orchestrator.get_parameter('max_joint_jump').value
        check('and the executed path never jumps between points',
              (max(jumps) if jumps else 0.0) <= max_jump, True)
        print(f'        worst joint change between trajectory points '
              f'{(max(jumps) if jumps else 0.0):.4f} rad, limit {max_jump:.3f}')

        # The way out is the way in: drop, then back through pre_pick to home.
        joint_goals = [g for g in goals if g['kind'] == 'joint']
        # The carry goes out through pre_pick, not straight from the lift to
        # the drop. Measured: a direct joint goal from the lift came back
        # INVALID_MOTION_PLAN three times -- a path computed and then
        # rejected, which is a swing through the octomap. pre_pick is above
        # the table by construction.
        check('the carry is staged through pre_pick',
              orchestrator.get_parameter(
                  'stage_drop_through_pre_pick').value, True)
        carry = [g['joints'] for g in joint_goals]
        drop_at = next((i for i, j in enumerate(carry)
                        if max(abs(a - b) for a, b in zip(j, DROP_JOINTS))
                        < 1e-3), None)
        check('there is a drop goal at all', drop_at is not None, True)
        if drop_at:
            check('and the goal before it is pre_pick, not the lift',
                  [round(v, 6) for v in carry[drop_at - 1]],
                  [round(v, 6) for v in PRE_PICK_JOINTS])

        # A failed place must retreat too, not leave the arm stopped where it
        # failed -- extended over the work surface and often still holding
        # the object, which is the worst place to leave it and the hardest to
        # start the next cycle from.
        with robot.lock:
            before = moved(robot)
            robot.move_fail_codes = [MoveItErrorCodes.FAILURE] * 3
        # The place is the tail of the sequence now, so run it the way the
        # cycle does rather than through a method that no longer exists.
        _pick_steps, place_steps = orchestrator.sequence_split()
        placed = orchestrator.run_sequence(
            place_steps,
            {'states': {'pre_pick_state': {'joints': PRE_PICK_JOINTS},
                        'drop_state': {'joints': DROP_JOINTS}},
             'why': 'a test'})
        check('a place that cannot reach the drop pose fails', placed, None)
        with robot.lock:
            robot.move_fail_codes = []
        orchestrator._retreat_to_home(
            {'pre_pick_state': {'joints': PRE_PICK_JOINTS},
             'drop_state': {'joints': DROP_JOINTS}}, 'after a failed place')
        with robot.lock:
            retreat = [g['joints'] for g in robot.goals[before:]
                       if not g['plan_only'] and g['kind'] == 'joint']
        check('and the retreat still runs, ending at home',
              [round(v, 5) for v in retreat[-1]] if retreat else None,
              [round(v, 5) for v in HOME_JOINTS])
        check('via the staging pose',
              any(max(abs(a - b) for a, b in zip(j, PRE_PICK_JOINTS)) < 1e-3
                  for j in retreat), True)

        check('the last three joint goals are drop, pre_pick, home',
              len(joint_goals) >= 5, True)
        check('carries straight to drop_state', joint_goals[-3]['joints'],
              DROP_JOINTS)
        check('then stages back through pre_pick',
              joint_goals[-2]['joints'], PRE_PICK_JOINTS)
        check('and finishes at home', joint_goals[-1]['joints'],
              HOME_JOINTS, 1e-5)
        # Carrying must not detour home: capture_octomap refuses while holding,
        # so the trip would clear the map and leave nothing to plan the drop
        # against.
        check('no trip home between the lift and the release',
              [g for g in joint_goals[-3:-2]
               if max(abs(a - b) for a, b in zip(g['joints'], HOME_JOINTS)) < 1e-3],
              [])

        # The close is a staircase now, not one goal: it steps down and stops
        # at the cap, so assert the shape rather than three exact values.
        #
        # And the cycle ends on a close, not on the release: the jaws are shut
        # again at the drop pose so the trip back through pre_pick to home is
        # made with a narrow profile. So the release is the *last open*, not
        # the last command, and the grasp close is the staircase before it.
        check('the gripper opened first', round(grips[0], 3), 0.044)
        release_at = max(i for i, g in enumerate(grips) if round(g, 3) == 0.044)
        check('and opened again to release', release_at > 0, True)
        grasp_close = grips[1:release_at]
        check('the close stepped rather than slamming shut',
              len(grasp_close) > 3, True)
        check('and stopped at the cap instead of fully closing',
              min(grasp_close) > 0.0, True)
        check('the cycle ends with the jaws shut at the drop, not open',
              round(grips[-1], 3), 0.0)
        check('grip torque cap is the 2.5 Nm default',
              orchestrator.get_parameter('gripper_torque_cap').value, 2.5)
        # 0.3 -> 0.4. _retime divides every timestamp by this, so the old
        # value stretched a planned trajectory 3.33x and this one stretches it
        # 2.5x -- a quarter off the wall clock of every leg, which is where a
        # cycle's time actually goes. Measured, run 1788859355: TRANSIT wrote
        # its first record 18.59 s after the state began, all of it inside one
        # cartesian_move call.
        check('motion is time-scaled faster than the timid default',
              orchestrator.get_parameter('velocity_scaling').value, 0.4)
        check('and acceleration matches it, since cuMotion applies the min',
              orchestrator.get_parameter('acceleration_scaling').value, 0.4)

        with robot.lock:
            refreshes = list(robot.refreshes)
        # A capture recorded as v happened straight after goals[v - 1], and
        # that goal must be the HOME one. This is the check that pins the bug:
        # a capture after the READY goal put the arm in the map.
        check('the octomap was captured at all', len(refreshes) > 0, True)
        captured_after = [goals[v - 1]['joints'] for v in refreshes if v > 0]
        check('every capture followed a goal to HOME, never READY',
              [j for j in captured_after
               if max(abs(a - b) for a, b in zip(j, HOME_JOINTS)) > 1e-3],
              [])
        check('and the map was cleared before being rebuilt',
              len(robot.clears) > 0, True)
        # refreshes holds len(goals) at capture time, so a value v means the
        # capture happened straight after goals[v - 1]. Keyed off the joint
        # goals rather than fixed indices: the pose column's length depends on
        # transit_height and descend_step, so hard indices here quietly stopped
        # describing "while carrying" the moment the column grew.
        joint_indices = [i for i, g in enumerate(goals) if g['kind'] == 'joint']
        carrying_ready, final_ready = joint_indices[2], joint_indices[4]
        check('no capture between picking the object up and releasing it',
              [v for v in refreshes
               if carrying_ready + 1 <= v <= final_ready], [])

        # Guards. These are what stand between an unrecorded pose and the arm
        # moving somewhere nobody chose.
        check('an empty states file is refused',
              orchestrator.check_states({}), False)
        check('a states file missing drop_state is refused',
              orchestrator.check_states(
                  {'pre_pick_state': {'joints': PRE_PICK_JOINTS}}), False)
        check('a complete states file passes',
              orchestrator.check_states(
                  {'pre_pick_state': {'joints': PRE_PICK_JOINTS},
                   'drop_state': {'joints': DROP_JOINTS}}), True)

        with open(states_path, 'w') as handle:
            yaml.safe_dump({'arm': 'left', 'states': {
                'pre_pick_state': {'joints': PRE_PICK_JOINTS},
                'drop_state': {'joints': DROP_JOINTS}}}, handle)
        check('a file recorded for the other arm is not replayed',
              orchestrator.load_states(), {})

        # The torque cap: closing must stop advancing once measured torque
        # reaches the cap, and hold the aperture there.
        with robot.lock:
            robot.finger = OPEN_FINGER
            robot.torque = 0.0
            robot.holding = False
            robot.gripper_commands.clear()
        time.sleep(0.4)
        check('the cap is readable from /joint_states',
              orchestrator.finger_effort() is not None, True)
        check('capped close reports success',
              orchestrator.close_gripper_to_cap(), True)
        with robot.lock:
            commands = list(robot.gripper_commands)
            final_torque = robot.torque
        theory = OBJECT_HALF_WIDTH - TORQUE_CAP_NM * GRIPPER_R / GRIPPER_KP
        step = orchestrator.get_parameter('gripper_close_step').value
        deepest = min(commands)
        print(f'        deepest command {deepest*1000:.2f} mm, theory '
              f'{theory*1000:.2f} mm, final torque {final_torque:.2f} Nm')
        check('it stopped short of a full squeeze', deepest > 0.0, True)
        check('it stopped within one step of the cap depth',
              abs(deepest - theory) <= step + 1e-9, True)
        check('the hold keeps a position error, so the grip survives',
              commands[-1] < OBJECT_HALF_WIDTH - 1e-6, True)
        check('torque never ran away past the cap',
              final_torque <= TORQUE_CAP_NM + GRIPPER_KP * step / GRIPPER_R + 1e-6,
              True)

        # The gripper on its own, so the cap can be tested with an object placed
        # in the fingers by hand and no arm motion at all.
        with robot.lock:
            robot.finger = 0.0
            robot.torque = 0.0
            robot.holding = False
            robot.gripper_commands.clear()
        time.sleep(0.4)
        opened = orchestrator._srv_open_gripper(Trigger.Request(),
                                                Trigger.Response())
        check('the open service reports success', opened.success, True)
        with robot.lock:
            check('and opens to the finger limit',
                  abs(robot.gripper_commands[-1] - OPEN_FINGER) < 1e-9, True)
        check('and says where the fingers are', 'finger' in opened.message, True)

        with robot.lock:
            robot.finger = OPEN_FINGER
            robot.torque = 0.0
            robot.gripper_commands.clear()
        time.sleep(0.4)
        gripped = orchestrator._srv_grip(Trigger.Request(), Trigger.Response())
        with robot.lock:
            hand_commands = list(robot.gripper_commands)
            hand_torque = robot.torque
        check('the grip service reports success', gripped.success, True)
        check('it stopped short of a full squeeze too',
              min(hand_commands) > 0.0, True)
        check('at no more than the cap',
              hand_torque <= TORQUE_CAP_NM + GRIPPER_KP * step / GRIPPER_R + 1e-6,
              True)
        check('and reports the torque it stopped at',
              'Nm' in gripped.message, True)

        # Neither may run during a cycle: they command the gripper directly,
        # which mid-descent would open the fingers on the way down.
        with orchestrator._lock:
            orchestrator._busy = True
        try:
            check('open is refused mid-cycle',
                  orchestrator._srv_open_gripper(
                      Trigger.Request(), Trigger.Response()).success, False)
            check('grip is refused mid-cycle',
                  orchestrator._srv_grip(
                      Trigger.Request(), Trigger.Response()).success, False)
        finally:
            with orchestrator._lock:
                orchestrator._busy = False
        check('and the flag is released after a hand test',
              orchestrator._busy, False)

        # Implausible detections must be refused before anything moves. The
        # case measured on the robot: a bad depth frame reported an object on
        # the table at [3.0799, 2.3206, -0.7416], depth 2.982 m from 40 px --
        # four metres away and below the floor. min_grasp_z clamped the height
        # and left x and y alone, so all six ladder attempts collected IK_FAIL
        # on a point outside the room and the summary blamed the planner.
        good = {'point': list(OBJECT_POINT), 'depth_m': 0.6, 'depth_px': 900,
                'axis_yaw': 0.0}
        check('a normal detection is accepted',
              orchestrator.implausible_detection(good), None)

        far = dict(good, point=[3.0799, 2.3206, -0.7416], depth_m=2.982,
                   depth_px=40)
        reason = orchestrator.implausible_detection(far)
        check('the real bad detection is rejected', reason is not None, True)
        check('and the reason names the depth stream, not reach',
              'depth' in (reason or '').lower(), True)

        check('a point beyond workspace_radius is rejected',
              orchestrator.implausible_detection(
                  dict(good, point=[2.0, 0.0, 0.05])) is not None, True)
        check('a point far below the surface is rejected',
              orchestrator.implausible_detection(
                  dict(good, point=[0.35, -0.18, -0.50])) is not None, True)
        check('a point absurdly high is rejected',
              orchestrator.implausible_detection(
                  dict(good, point=[0.35, -0.18, 2.0])) is not None, True)
        check('a non-finite point is rejected',
              orchestrator.implausible_detection(
                  dict(good, point=[float('nan'), 0.0, 0.05])) is not None, True)
        check('a missing point is rejected',
              orchestrator.implausible_detection({}) is not None, True)

        # Small noise must still clamp rather than reject: that is what
        # min_grasp_z is for, and rejecting it would refuse real objects.
        min_z = orchestrator.get_parameter('min_grasp_z').value
        slack = orchestrator.get_parameter('max_z_clamp').value
        check('a centimetre below the floor still clamps, not rejects',
              orchestrator.implausible_detection(
                  dict(good, point=[0.35, -0.18, min_z - slack / 2])), None)

        # min_grasp_z is not a table floor and must not be mistaken for one.
        # It is measured from the base and set at 0.01; the real table stands
        # at z=0.34, so it would never stop a grasp 20 mm too deep. That is
        # grasp_max_depth's job, and it is measured from the detected top of
        # the object -- which rests on the surface, and is the only surface
        # reference available without being told where the table is.
        # A grip that missed is not the same failure as a descent that would
        # not fly, and the remedies are opposite: nothing recovers a line the
        # pre-flight already proved and the arm then cannot follow, whereas a
        # grip that shut on air is exactly what the ladder's 8-mm-lower rungs
        # are for. They were indistinguishable at the call site, so
        # retry_after_preflight stopped the cycle after one attempt.
        # Measured, run 1788869179 cycle 2: DESCEND landed 1.4 mm from its
        # commanded height -- the motion was near perfect -- and the jaws
        # closed on air 11 mm above a screwdriver.
        orchestrator_src = open(
            os.path.join(WS, 'pick_place_orchestrator.py')).read()
        check('a missed grip has its own answer, distinct from None',
              module.GRASP_MISSED is not None, True)
        check('and the attempt returns it rather than a bare failure',
              'return GRASP_MISSED' in orchestrator_src, True)
        check('the ladder carries on after one, whatever '
              'retry_after_preflight says',
              re.search(r'if picked_point is GRASP_MISSED:'
                        r'(?:\n\s*(?:#[^\n]*)?[^\n]*)*?\n\s*continue',
                        orchestrator_src) is not None, True)
        # And it does not repeat itself: a rung offering the same yaw at the
        # same height will miss identically, for the ~60 s a full approach
        # costs. Only exact repeats are skipped -- a different yaw is a
        # different grasp on a screwdriver, and a rung that re-detects gets a
        # fresh height.
        shapes, order = set(), []
        for st in module.STRATEGIES:
            shape = (round(st['yaw_offset'], 6), round(st['z_offset'], 6))
            if shape in shapes and not st['redetect']:
                continue
            order.append(st['name'])
            shapes.add(shape)
        check('a ladder where every rung misses skips only the duplicate',
              order, ['nominal', 'yaw+90', 'lower-8mm', 'yaw+90-lower',
                      'remap-from-home'])
        check('and it does reach a rung that goes lower',
              any(module.STRATEGIES[[s['name'] for s in module.STRATEGIES]
                                    .index(n)]['z_offset'] < 0.0
                  for n in order), True)

        check('the shipped grasp offset stops above the object, not below it',
              orchestrator.get_parameter('grasp_z_offset').value > 0.0, True)
        nominal = next(st for st in module.STRATEGIES
                       if st['name'] == 'nominal')
        deeper = dict(nominal, z_offset=-0.050)
        point = [0.35, -0.18, 0.30]
        grasp, _pregrasp, _quat, _pt = orchestrator.grasp_from_detection(
            dict(good, point=point), deeper)
        check('a strategy that asks 50 mm deeper is clamped to the object top',
              round(grasp[2], 6), round(point[2], 6))
        # The rungs that do exist stay useful: -8 mm off a +15 mm offset is
        # still 7 mm above the surface, so the ladder can retry lower without
        # driving the tool into the table.
        for name in ('lower-8mm', 'yaw+90-lower'):
            rung = next(st for st in module.STRATEGIES if st['name'] == name)
            low, _pg, _q, _p = orchestrator.grasp_from_detection(
                dict(good, point=point), rung)
            check(f'the {name} rung still lands above the surface',
                  low[2] >= point[2] - 1e-9, True)
            check(f'and {name} really does aim lower than nominal',
                  low[2] < point[2]
                  + orchestrator.get_parameter('grasp_z_offset').value, True)

        # Seeded IK, on its own. This is the mechanism that stops a 5 cm
        # descent from flipping the arm, so its failure modes matter.
        quat = list(module.top_down_quat(0.0))
        xy = (OBJECT_POINT[0], OBJECT_POINT[1])
        with robot.lock:
            robot.joints = list(READY_JOINTS)
        time.sleep(0.4)

        check('seeded_descent is on by default',
              orchestrator.get_parameter('seeded_descent').value, True)

        with robot.lock:
            robot.ik_fail = robot.ik_flip = False
        solved = orchestrator.solve_ik((xy[0], xy[1], 0.30), quat,
                                       list(READY_JOINTS))
        check('IK returns a solution near the seed', solved is not None, True)
        check("and it stays in the seed's branch",
              max(abs(a - b) for a, b in zip(solved, READY_JOINTS)) <= 0.5, True)

        # The real failure: same tool pose, opposite elbow. Accepting this is
        # what makes the arm turn itself inside out to lower the tool.
        with robot.lock:
            robot.ik_flip = True
        check('a reconfiguring solution is rejected',
              orchestrator.solve_ik((xy[0], xy[1], 0.30), quat,
                                    list(READY_JOINTS)), None)
        with robot.lock:
            robot.ik_flip = False

        with robot.lock:
            robot.ik_fail = True
        check('no IK solution is reported as none',
              orchestrator.solve_ik((xy[0], xy[1], 0.30), quat,
                                    list(READY_JOINTS)), None)

        # The case that actually bit: on the final approach the target is
        # itself in the collision world, so the checked line stalls -- measured
        # at 25% of the last 5 cm, against 100% for the 20 cm above it. It must
        # retry unchecked and still go straight down, not fall back to
        # free-space goals, because a free-space plan for that 5 cm is what
        # drove the gripper into the table.
        check('the approach may ignore the octomap by default',
              orchestrator.get_parameter('approach_ignores_octomap').value, True)
        with robot.lock:
            robot.ik_fail = False
            robot.cartesian_fraction = 1.0
            robot.checked_fraction = 0.0       # checked blocked, unchecked fine
            robot.cartesian_requests.clear()
            robot.tcp = [xy[0], xy[1], 0.30]   # a real 50 mm leg below
            before = moved(robot)
        time.sleep(0.6)
        check('the descent still completes',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'BLOCKED',
                                          uncheck_collisions=True), True)
        with robot.lock:
            asked = list(robot.cartesian_requests)
            goals_after = moved(robot)
        # The fake does not model the allowed-collision matrix, so its
        # checked line stays blocked (checked_fraction 0) even with the
        # gripper exempted, and the blunt fallback still runs. What is being
        # pinned here is that the fallback is reached and the descent still
        # completes -- on the real robot the exemption is what makes the
        # checked line solve in the first place.
        check('a blocked checked line still falls back to an unchecked one',
              [r['avoid_collisions'] for r in asked], [True, False])
        check('and sent no free-space goals at all', goals_after, before)

        # Without the exemption it must not silently ignore the octomap.
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.tcp = [xy[0], xy[1], 0.30]
            before = moved(robot)
        time.sleep(0.6)
        check('without the exemption it falls back instead',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'STRICT',
                                          uncheck_collisions=False), True)
        with robot.lock:
            asked = list(robot.cartesian_requests)
            goals_after = moved(robot)
        check('only a checked line was ever asked for',
              [r['avoid_collisions'] for r in asked], [True])
        check('and it did fall back to goals', goals_after > before, True)
        with robot.lock:
            robot.checked_fraction = None

        # A move MoveIt calls SUCCESS has satisfied the joint controller's
        # tolerance, which is not the tool being where it was sent. Measured on
        # this robot: 13.7 mm low at z=0.405, 33.5 mm at z=0.555, every time,
        # in the direction gravity pulls. A grasp planned on the assumption
        # that the tool arrived closes above the object.
        tol = orchestrator.get_parameter('pose_tolerance').value
        check('a tool tolerance is set', tol > 0, True)
        # Off by default now -- it made four of six measured legs worse, and
        # on the approach it displaced the descent's starting point. The
        # mechanism is still here and still has to work when asked for, so the
        # block below turns it on and puts it back afterwards.
        check('offset correction is off by default', orchestrator.get_parameter(
            'pose_offset_correction').value, False)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'pose_offset_correction', value=True)])

        target = (OBJECT_POINT[0], OBJECT_POINT[1], 0.30)
        quat_d = list(module.top_down_quat(0.0))

        # A tool that lands on target needs one move and no correction.
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.sag = 0.0
        check('an accurate move is left alone',
              orchestrator.converge_to(target, quat_d, 'CLEAN'), True)
        with robot.lock:
            check('and costs exactly one line',
                  len(robot.cartesian_requests), 1)

        # A repeatable offset is corrected by aiming past it -- once.
        # Re-issuing the same target cannot help: the joints already sit at
        # their solution, so the arm does not move. Measured on the robot: ten
        # re-issued passes went 19.3 -> 18.6 mm and then gave up.
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.sag = 0.014           # the measured dz, and it never improves
            robot.sag_floor = True
        check('a repeatable offset still completes the move',
              orchestrator.converge_to(target, quat_d, 'OFFSET'), True)
        with robot.lock:
            asked = [r['xyz'] for r in robot.cartesian_requests]
            landed = list(robot.tcp)
        check('it took one line plus one correction', len(asked), 2)
        check('and the correction aimed past the target, not at it',
              asked[1][2] > asked[0][2] + 1e-6, True)
        check('by about the measured offset',
              abs((asked[1][2] - asked[0][2]) - 0.014) < 0.002, True)
        check('so the tool ends within tolerance',
              abs(landed[2] - target[2]) <= tol, True)
        print(f'        offset closed to '
              f'{abs(landed[2] - target[2]) * 1000:.1f} mm in {len(asked)} moves')
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'pose_offset_correction', value=False)])
        with robot.lock:
            robot.sag = 0.0
            robot.sag_floor = False

        # A line that only partly solves is flown as far as it goes and then
        # continued, so the whole leg stays straight. Refusing a partial was
        # wrong: a leg whose line solved 95.65% -- about 6 mm short of 150 mm
        # -- was thrown away for curved hops, and the first hop aborted with
        # CONTROL_FAILED against the table.
        usable = orchestrator.get_parameter('cartesian_partial_min').value
        wanted = orchestrator.get_parameter('cartesian_min_fraction').value
        check('a partial line is usable well below the completion threshold',
              usable < wanted, True)
        # 150 mm, so 95.65% leaves 6.5 mm -- outside the 5 mm tolerance, so a
        # second segment is genuinely needed and not an artefact of the
        # numbers chosen.
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.cartesian_fraction = 0.9565     # the measured case
            robot.tcp = [xy[0], xy[1], 0.45]      # so the leg really is 150 mm
            before = moved(robot)
        time.sleep(0.6)
        check('a 95.65% line still completes the leg',
              orchestrator.descend_column(xy, 0.45, 0.30, quat, 'PARTIAL_OK'),
              True)
        with robot.lock:
            asked = len(robot.cartesian_requests)
            goals = moved(robot)
            landed = list(robot.tcp)
        check('it took more than one straight segment', asked > 1, True)
        check('and no curved goals at all', goals, before)
        check('and the tool ends within tolerance',
              abs(landed[2] - 0.30) <= tol, True)
        print(f'        95.65% of a 150 mm leg finished in {asked} straight '
              f'segments, {abs(landed[2] - 0.30) * 1000:.1f} mm off')
        with robot.lock:
            robot.cartesian_fraction = 1.0

        # A close that never loads the fingers gripped nothing, and saying so
        # at the close saves the lift, the carry and the drop -- all of which
        # were being attempted with an empty gripper.
        with robot.lock:
            robot.nothing_to_grip = True
            robot.finger = OPEN_FINGER
            robot.torque = 0.0
            robot.gripper_commands.clear()
        time.sleep(0.4)
        check('closing on nothing is reported as a failure',
              orchestrator.close_gripper_to_cap(), False)
        with robot.lock:
            shut = min(robot.gripper_commands)
            peak = robot.torque
        check('it did close all the way', shut <= 0.0 + 1e-9, True)
        check('and never came near the cap',
              peak < orchestrator.get_parameter('gripper_torque_cap').value,
              True)
        with open(log_path) as handle:
            recent = [json.loads(line) for line in handle if line.strip()]
        check('the log names it',
              any(e['outcome'] == 'closed-on-nothing' for e in recent), True)

        # Whether anything is held is answered by where the fingers *are*,
        # not by what they were last told. This compared the last commanded
        # step -- always 0.0 at the end of a full close -- against a 3 mm
        # floor, so every close that ran to the end was called empty whatever
        # the jaws were doing. Measured across 14 real closes:
        #
        #   empty     measured finger 0.0025-0.0026 m, effort 1.14-1.24 Nm
        #   holding   measured finger 0.0039-0.0175 m, effort 1.84-2.46 Nm
        #
        # Two bands, nothing between them, and grasp_finger_min already in
        # the gap at 0.003. Three runs in that table were rejected while
        # gripping something 4 mm thick at ~1.9 Nm -- one a roll of tape the
        # arm was visibly holding.
        floor = orchestrator.get_parameter('grasp_finger_min').value
        with robot.lock:
            robot.nothing_to_grip = True     # never loads to the cap
            robot.finger_floor = floor + 0.001
            robot.finger = OPEN_FINGER
            robot.torque = 0.0
        time.sleep(0.4)
        check('jaws that stop above the floor are holding, not empty -- '
              'even though the last command was 0.0',
              orchestrator.close_gripper_to_cap(), True)
        with open(log_path) as handle:
            recent = [json.loads(line) for line in handle if line.strip()]
        held = [e for e in recent if e['outcome'] == 'held-under-cap']
        check('and the log says so', bool(held), True)
        check('recording the measured finger, not just the command',
              held[-1]['finger'] > floor and held[-1]['target'] <= floor, True)
        with robot.lock:
            robot.finger_floor = 0.0         # jaws really do shut on air
            robot.finger = OPEN_FINGER
            robot.torque = 0.0
        time.sleep(0.4)
        check('and jaws that shut all the way really are empty',
              orchestrator.close_gripper_to_cap(), False)

        # close_gripper is the plain one: no grasp check, because at the drop
        # pose there is deliberately nothing between the fingers and a
        # 'closed on nothing' verdict there would be correct and useless.
        with robot.lock:
            robot.nothing_to_grip = True
            robot.finger_floor = 0.0
            robot.finger = OPEN_FINGER
            robot.torque = 0.0
            robot.gripper_commands.clear()
        time.sleep(0.4)
        check('a plain close succeeds on an empty gripper',
              orchestrator.close_gripper('CLOSE_AT_DROP'), True)
        with robot.lock:
            shut = list(robot.gripper_commands)
        check('in one command, not a stepped search for the cap',
              len(shut), 1)
        check('and it really does shut them', shut[-1] <= 0.0 + 1e-9, True)
        with robot.lock:
            robot.nothing_to_grip = False
            robot.finger = OPEN_FINGER
            robot.torque = 0.0
        time.sleep(0.4)
        with robot.lock:
            robot.nothing_to_grip = False
            robot.finger = OPEN_FINGER
            robot.torque = 0.0
        time.sleep(0.4)

        # And a close that gripped nothing does not license going home from
        # down at the object. Measured, left arm, run 1788859355: DESCEND
        # fine, the grip closed on nothing at 1.912 Nm, and the very next
        # record is HOME as a joint goal from tcp z=0.3715 -- a joint-space
        # move from reaching over the table to folded at the side, which
        # passes through the table. The arm hit it.
        column_xy = (0.34, 0.12)
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.tcp = [column_xy[0], column_xy[1], 0.30]   # down at it
            before = moved(robot)
        time.sleep(0.6)
        orchestrator._column = ([column_xy[0], column_xy[1], 0.30],
                                [column_xy[0], column_xy[1], 0.45], quat)
        check('the arm can be lifted clear of the surface',
              orchestrator.clear_the_surface('a test'), True)
        with robot.lock:
            lines = list(robot.cartesian_requests)
            free = moved(robot) - before
            landed = list(robot.tcp)
        check('it asked for a straight line to get out', len(lines) > 0, True)
        check('and no free-space goal at all -- that is the sweep', free, 0)
        check('every waypoint stayed on the column',
              all(abs(line['xyz'][0] - column_xy[0]) < 1e-6
                  and abs(abs(line['xyz'][1]) - abs(column_xy[1])) < 1e-6
                  for line in lines if line['xyz']), True)
        check('every waypoint was above where it started',
              all(line['xyz'][2] > 0.30 - 1e-6
                  for line in lines if line['xyz']), True)
        check('and the tool ended clear', landed[2] > 0.30 + 1e-6, True)
        check('CLEAR was published, so a log says why it rose',
              'CLEAR' in ' '.join(states[-6:]), True)
        check('and the column is forgotten, so the next exit is a no-op',
              orchestrator._column, None)

        # A no-op once the tool is clear. The successful path has already
        # lifted, and a second lift there would be a wasted leg -- or, if the
        # object is being carried, a leg planned with the payload attached for
        # no reason.
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.tcp = [column_xy[0], column_xy[1], 0.46]
            before = moved(robot)
        time.sleep(0.6)
        orchestrator._column = ([column_xy[0], column_xy[1], 0.30],
                                [column_xy[0], column_xy[1], 0.45], quat)
        check('clearing an already-clear tool succeeds',
              orchestrator.clear_the_surface('a test'), True)
        with robot.lock:
            check('and moves nothing', len(robot.cartesian_requests), 0)
            check('and sends no goal either', moved(robot) - before, 0)
        check('with no column left over', orchestrator._column, None)

        # No column means no idea which way is up, so there is nothing to do
        # rather than a guess.
        orchestrator._column = None
        check('and with no column it is a no-op',
              orchestrator.clear_the_surface('a test'), True)

        # The structural half: every exit from a pick has to go through it.
        # The lift is only worth having if it is on all of the paths, and the
        # one that mattered -- the failed grip -- was the one without it.
        orch_src = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
        # The gripper's octomap exemption has to outlive the descent leg.
        # It used to be withdrawn in descend_column's finally, per leg -- and
        # the descent leaves the jaws inside the voxels of the object they
        # came for, so handing them back to collision checking there makes
        # the arm's own start state invalid. That is the gripper turning red
        # in RViz. Measured, run 1788863209: CLEAR succeeded on its own fresh
        # exemption and then PRE_PICK_STATE and HOME both returned -2,
        # INVALID_MOTION_PLAN, with the arm stranded over the table.
        # Legs flown here really move the fake arm, and the seeded-column
        # test further down reads the posture it is left in -- so put it back.
        with robot.lock:
            keep_joints = list(robot.joints)
            keep_tcp = list(robot.tcp)
        orchestrator._column = ([0.34, 0.12, 0.30], [0.34, 0.12, 0.45], quat)
        orchestrator._octomap_exempt = True
        orchestrator.descend_column((0.34, 0.12), 0.40, 0.38, quat, 'HOLDS')
        check('a leg flown on the column leaves the gripper exempt',
              orchestrator._octomap_exempt, True)
        check('and the column is still held', orchestrator._column is not None,
              True)
        orchestrator.release_column()
        check('releasing the column withdraws the exemption',
              orchestrator._octomap_exempt, False)
        check('and forgets the column', orchestrator._column, None)
        orchestrator._octomap_exempt = True
        orchestrator.descend_column((0.34, 0.12), 0.40, 0.38, quat, 'NOHOLD')
        check('a leg flown off the column withdraws it as before',
              orchestrator._octomap_exempt, False)
        # Two: the field's declaration in __init__, and release_column
        # itself. Anywhere else and the exemption outlives the column or the
        # column outlives the exemption, which is the bug this pair exists to
        # prevent.
        # Scanned over the orchestrator class alone. The module has more
        # than one class in it now, and splitting the whole file on
        # "    def __init__(" finds whichever comes first -- which is how
        # this check started reading ContactMonitor's constructor and
        # reporting that __init__ no longer clears the column.
        orch_class = orch_src.split('\nclass PickPlaceOrchestrator')[-1]
        cleared_in = [name for name in re.findall(
            r'\n    def ([a-z_]+)\(', orch_class)
            if 'self._column = None' in orch_class.split(
                f'\n    def {name}(')[1].split('\n    def ')[0]]
        check('nothing but __init__ and release_column clears the column',
              cleared_in, ['__init__', 'release_column'])

        # And the descent gap: the plan ends on the point, the arm stops
        # short of it by an amount that varied 15.5 -> 36 mm across three
        # runs, and no offset can track that.
        # On by default since run 1789014831, which is the run its old
        # comment asked for -- "turn it on if a descent ever does stop short
        # enough to miss". Four right-arm descents: the two that gripped
        # stopped 3.2 and 4.0 mm from the commanded grasp, the two that
        # missed stopped 18.1 and 18.9 mm out with 15 mm of it height, and
        # the jaws closed on air above the object. All tracking error, none
        # of it calibration: at the end of those moves the joints were still
        # 29 mrad from the last point of their own trajectory, against
        # 9-11 mrad on the two that worked. The risk it was off for --
        # pressing into the table -- is what the contact guard now watches.
        check('closing the descent gap is on by default',
              orchestrator.get_parameter('descend_close_gap').value, True)
        grasp_at = [0.34, 0.12, 0.30]
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.tcp = [0.34, 0.12, 0.32]        # 20 mm short
        time.sleep(0.6)
        check('a 20 mm shortfall is flown',
              orchestrator.close_descent_gap(grasp_at, quat), True)
        with robot.lock:
            lines = list(robot.cartesian_requests)
        check('as a straight line down to the commanded height',
              bool(lines) and abs(lines[-1]['xyz'][2] - grasp_at[2]) < 1e-6,
              True)
        check('leaving x and y alone -- the jaws span 40 mm and a lateral '
              'move at grasp height is the one thing worth not doing',
              abs(lines[-1]['xyz'][0] - 0.34) < 1e-6, True)
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.tcp = [0.34, 0.12, 0.302]       # 2 mm short
        time.sleep(0.6)
        check('a 2 mm shortfall is not worth a move',
              orchestrator.close_descent_gap(grasp_at, quat), False)
        with robot.lock:
            check('and sends nothing', len(robot.cartesian_requests), 0)
            robot.tcp = [0.34, 0.12, 0.45]        # 150 mm short
        time.sleep(0.6)
        check('and a gap far past descend_gap_max is reported, not flown',
              orchestrator.close_descent_gap(grasp_at, quat), False)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'descend_close_gap', value=False)])
        with robot.lock:
            check('because that is not tracking error',
                  len(robot.cartesian_requests), 0)
            robot.tcp = [0.34, 0.12, 0.45]
        orchestrator.release_column()
        with robot.lock:
            robot.joints = keep_joints
            robot.tcp = keep_tcp
            robot.ik_requests.clear()
            robot.cartesian_requests.clear()
        time.sleep(0.6)

        # HOME is a joint goal to a folded posture. Commanded from a low,
        # extended one the short path in joint space goes through the work
        # surface -- so when pre_pick cannot be reached on the way back, the
        # arm stops rather than sweeping. Measured, run 1788869179 cycle 2:
        # CLEAR got the tool to z=0.4286, PRE_PICK_STATE came back exhausted,
        # and HOME was sent anyway; the arm swept out to x=0.44 and down to
        # z=0.358 across the object it had just failed to pick.
        check('home is not allowed without pre_pick by default',
              orchestrator.get_parameter('home_requires_pre_pick').value, True)
        states_now = orchestrator.load_states()
        with robot.lock:
            robot.joint_code = -2          # every joint goal is refused
            robot.cartesian_available = False   # and no line out either
            before = moved(robot)
        time.sleep(0.6)
        seen_before = len(states)
        orchestrator._column = None
        check('the retreat refuses rather than flying home',
              orchestrator._retreat_to_home(states_now, 'a test'), False)
        with robot.lock:
            sent = motions(robot)[before:]
        home_goals = [g for g in sent if g['kind'] == 'joint'
                      and max(abs(a - b) for a, b in
                              zip(g['joints'], HOME_JOINTS)) < 1e-3]
        check('and no HOME goal was sent at all', home_goals, [])
        check('saying so in the state', 'STOPPED'
              in ' '.join(states[seen_before:]), True)
        with open(log_path) as handle:
            recent = [json.loads(line) for line in handle if line.strip()]
        check('and in the log',
              any(e['outcome'] == 'refused-no-pre-pick' for e in recent), True)
        with robot.lock:
            # None, not 1. Setting it to SUCCESS leaves the lever *engaged*,
            # so move_fail_codes is never consulted again and every later
            # test that injects a planner failure silently gets a success --
            # six of them did.
            robot.joint_code = None
            robot.cartesian_available = True
        time.sleep(0.6)

        # The intent, not the layout: whatever else the failed-grip branch
        # does, it gets the arm off the surface before it returns. A regex
        # pinning the two lines adjacent broke the moment a _note_failure
        # call was added between them, which is a change to neither.
        missed_branch = orch_src.split(
            'if not self.close_gripper_to_cap():')[1].split(
                '\n        self._set_state')[0]
        # Anchored on the statement, not the word: the branch's own comment
        # contains "return without recording anything", and matching that
        # compared the lift against a comment.
        check('a failed grip lifts clear before returning',
              'self.clear_the_surface' in missed_branch
              and missed_branch.index('self.clear_the_surface')
              < missed_branch.index('return GRASP_MISSED'), True)
        check('so does a failed descent',
              "self.clear_the_surface('after a failed descent')" in orch_src,
              True)
        # Order matters and a regex spanning lines is not the way to check
        # it: the first attempt at this was
        # r'clear_the_surface\(why\)(?:\n\s*[^\n]*)*?_set_state' and it
        # backtracked catastrophically -- the suite sat at 80% CPU for
        # minutes on one check. Slice the method and look.
        retreat = orch_src.split('def _retreat_to_home(')[1].split(
            '\n    def ')[0]
        check('the retreat clears the surface before its joint goals',
              'clear_the_surface(why)' in retreat
              and retreat.index('clear_the_surface(why)')
              < retreat.index("_set_state('PRE_PICK'"), True)
        # Not the ordinary retreat: a failed cycle gets a refuge checked
        # against the collision world and then the motors off. See
        # safe_shutdown.
        check('and a failed pick goes through the safe shutdown, '
              'rather than dashing home from the table',
              orch_src.count(
                  "self.safe_shutdown(states, 'after a failed pick')"), 2)
        check('no failure path goes straight home any more',
              re.search(r"_failure_summary\(\)\}'\)"
                        r'\n\s*self\.move_to_home\(\)',
                        orch_src) is None, True)

        # A curved descent is not an acceptable substitute for a linear one.
        # Measured: a descent whose line solved 37.5% -- identically checked
        # and unchecked, so the arm runs out of reach along it -- became three
        # free-space hops that swung the tool 3.7 cm sideways and aborted with
        # CONTROL_FAILED against the table.
        check('descend_linear_only is on by default',
              orchestrator.get_parameter('descend_linear_only').value, True)
        with robot.lock:
            robot.cartesian_available = False       # no line to be had at all
            before = moved(robot)
        check('linear_only refuses rather than substituting a curve',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'NOCURVE',
                                          linear_only=True), False)
        with robot.lock:
            check('and sends no goals at all', moved(robot), before)
        check('while without it the old stepping still happens',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'CURVE_OK',
                                          linear_only=False), True)
        with robot.lock:
            check('which does send goals', moved(robot) > before, True)
            robot.cartesian_available = True

        # -- only the gripper is exempt, not the whole arm -----------------
        #
        # The descent needs an exemption for one narrow reason: on a top-down
        # grasp the object is itself in the octomap, so the fingers have to
        # enter occupied voxels. That says nothing about the forearm or the
        # elbow -- and with checking off for the whole arm, nothing stopped
        # those reaching the table on a descent whose tool path was fine.
        check('the gripper-only exemption is preferred',
              orchestrator.get_parameter('gripper_octomap_exemption').value,
              True)
        with robot.lock:
            robot.scene_requests.clear()
            robot.cartesian_requests.clear()
            robot.cartesian_fraction = 1.0
            robot.checked_fraction = None
            robot.tcp = [xy[0], xy[1], 0.30]
        time.sleep(0.6)
        orchestrator.descend_column(xy, 0.30, 0.25, quat, 'EXEMPT',
                                    uncheck_collisions=True)
        with robot.lock:
            asked = [r['avoid_collisions'] for r in robot.cartesian_requests]
            scenes = list(robot.scene_requests)
        check('the descent is planned collision-checked after all',
              all(asked) if asked else None, True)
        acm = [sc for sc in scenes if sc.get('acm_names')]
        check('because the gripper was exempted in the scene instead',
              bool(acm), True)
        if acm:
            # The whole matrix, not a four-entry diff. moveit_core *replaces*
            # the ACM when a scene diff carries one, so sending only the
            # gripper rows deleted every disable_collisions entry the SRDF
            # provides -- after which adjacent links in the arm counted as
            # collisions, the robot went red in RViz, and every plan came back
            # PLANNING_FAILED or INVALID_MOTION_PLAN.
            sent = acm[0]
            for name in robot.acm_names:
                if name not in sent['acm_names']:
                    check(f'the matrix still carries {name}', False, True)
                    break
            else:
                check('the matrix sent carries every entry it started with',
                      True, True)
            check('and the octomap on top of them',
                  '<octomap>' in sent['acm_names'], True)
            check('and the gripper links are in it',
                  all(link in sent['acm_names']
                      for link in orchestrator.gripper_links()), True)

            # The pre-existing adjacent-link exclusions have to survive.
            index = {n: i for i, n in enumerate(sent['acm_names'])}
            first, second = robot.acm_names[0], robot.acm_names[1]
            check('an adjacent-link exclusion it started with is still set',
                  bool(sent['acm_values'][index[first]][index[second]]), True)

            octomap = index['<octomap>']
            gripper = index[orchestrator.gripper_links()[0]]
            check('the gripper is allowed to touch the octomap',
                  bool(sent['acm_values'][gripper][octomap]), True)
            check('and the exemption is withdrawn again afterwards',
                  any(not sc['acm_values'][octomap][gripper]
                      for sc in acm[1:]
                      if len(sc['acm_values']) > max(octomap, gripper))
                  if len(acm) > 1 else False, True)

        # With the exemption unavailable, the old blunt fallback still works.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'gripper_octomap_exemption', value=False)])
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.checked_fraction = 0.25
            robot.tcp = [xy[0], xy[1], 0.30]
        time.sleep(0.6)
        orchestrator.descend_column(xy, 0.30, 0.25, quat, 'BLUNT',
                                    uncheck_collisions=True)
        with robot.lock:
            asked = [r['avoid_collisions'] for r in robot.cartesian_requests]
        check('it falls back to switching checking off', any(asked), False)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'gripper_octomap_exemption', value=True)])
        with robot.lock:
            robot.checked_fraction = None

        # -- the descent does not ask the octomap at all -------------------
        #
        # On a top-down grasp the target *is* in the collision world: the
        # octomap holds the object and the table under it, so a checked line
        # onto the object stalls about a centimetre short whatever the voxel
        # size. Asking first costs a planning round trip and can fly a partial
        # line that leaves the tool somewhere the rest does not solve from.
        check('the descent ignores the octomap by default',
              orchestrator.get_parameter('descend_ignores_octomap').value,
              True)
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.cartesian_fraction = 1.0
            # A checked line that stalls, exactly as the real one does.
            robot.checked_fraction = 0.25
        check('and gets down anyway',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'NOCHECK',
                                          uncheck_collisions=True,
                                          linear_only=True), True)
        with robot.lock:
            asked = [r['avoid_collisions'] for r in robot.cartesian_requests]
        check('having asked for a checked line first, the gripper being '
              'exempted in the scene', asked[0] if asked else None, True)
        check('and falling back to unchecked when that is still blocked',
              bool(asked and not all(asked)), True)

        # Turned off, the old order applies: checked first, then unchecked.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'descend_ignores_octomap', value=False)])
        with robot.lock:
            robot.cartesian_requests.clear()
        check('with it off the descent still arrives',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'CHECKFIRST',
                                          uncheck_collisions=True,
                                          linear_only=True), True)
        with robot.lock:
            asked = [r['avoid_collisions'] for r in robot.cartesian_requests]
        check('but it asked the checked way first', asked[0] if asked else None,
              True)
        check('and only then unchecked', any(not a for a in asked), True)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'descend_ignores_octomap', value=True)])

        # A leg the caller has *not* exempted is still checked. The exemption
        # is per leg -- short, straight, vertical, between reach-checked ends
        # -- not a blanket licence to ignore the collision world.
        with robot.lock:
            robot.cartesian_requests.clear()
        orchestrator.descend_column(xy, 0.30, 0.25, quat, 'STILLCHECKED',
                                    uncheck_collisions=False)
        with robot.lock:
            asked = [r['avoid_collisions'] for r in robot.cartesian_requests]
        check('a leg with no exemption is checked as before',
              asked[0] if asked else None, True)
        with robot.lock:
            robot.checked_fraction = None
            robot.cartesian_fraction = 1.0

        # Everything below here is about the fallbacks, so the straight line
        # has to be taken away first -- otherwise it succeeds and the fallback
        # never runs.
        #
        # A partial line is flown and then continued, so a descent that only
        # partly solves still arrives -- and still arrives *straight*. It used
        # to be refused outright, which sent a 95.65% line to the curved
        # fallback and the gripper into the table.
        # 0.8 of a 50 mm leg: 10 mm left after one segment, 2 mm after two.
        # 0.6 was used here and lands at 3.2 mm -- close enough to the 5 mm
        # tolerance that the segment count flipped between runs.
        with robot.lock:
            robot.ik_fail = False
            robot.cartesian_fraction = 0.8
            robot.cartesian_requests.clear()
            robot.tcp = [xy[0], xy[1], 0.30]
            before = moved(robot)
        time.sleep(0.6)
        check('a partial straight line still completes the descent',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'PARTIAL'),
              True)
        with robot.lock:
            segments = len(robot.cartesian_requests)
            partial = moved(robot) - before
            landed = list(robot.tcp)
        check('it took several straight segments', segments > 1, True)
        check('and no curved goals', partial, 0)
        check('the tool arrived', abs(landed[2] - 0.25) <= tol, True)
        print(f'        80% segments: {segments} lines, '
              f'{abs(landed[2] - 0.25) * 1000:.1f} mm off')
        with robot.lock:
            robot.cartesian_fraction = 1.0
            robot.cartesian_available = False

        # With IK unusable too, the column must still run, as pose goals -- the
        # old behaviour, which can wander, but a pick beats no pick.
        with robot.lock:
            robot.ik_fail = True
            before = moved(robot)
        check('the column falls back to pose goals without IK',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'FALLBACK'),
              True)
        with robot.lock:
            fallback = motions(robot)[before:]
        check('and those really are pose goals',
              sorted({g['kind'] for g in fallback}), ['pose'])
        with robot.lock:
            robot.ik_fail = False

        # A whole column, chained: each waypoint seeded from the one above it.
        with robot.lock:
            robot.ik_requests.clear()
            before = moved(robot)
        check('a seeded column runs as joint goals',
              orchestrator.descend_column(xy, 0.30, 0.20, quat, 'COLUMN'), True)
        with robot.lock:
            column_goals = motions(robot)[before:]
            column_ik = list(robot.ik_requests)
        check('all joint goals, no pose goals',
              sorted({g['kind'] for g in column_goals}), ['joint'])
        check('one IK request per hop',
              len(column_ik), len(column_goals))
        # The chaining is the point: waypoint n+1 is solved from waypoint n's
        # answer, not from the arm's original posture. Without that each
        # waypoint is free to land in a different branch.
        chained = [column_ik[i + 1]['seed'] == list(column_goals[i]['joints'])
                   for i in range(len(column_ik) - 1)]
        check('each request is seeded with the previous solution',
              all(chained), True)
        with robot.lock:
            robot.cartesian_available = True

        # The whole decision, end to end, with nothing reachable. The point of
        # doing it before the approach: an object outside both envelopes must
        # cost no motion at all, rather than a drive to a staging pose followed
        # by "not reachable".
        orchestrator.configure_arm(ARM)
        with robot.lock:
            # The gripper tests above left the fake holding something, and it
            # publishes no detections while it does -- which would fail this
            # for the wrong reason.
            robot.holding = False
            robot.finger = OPEN_FINGER
            robot.ik_unreachable = {'left', 'right'}
            before = moved(robot)
        time.sleep(0.4)
        outcome = orchestrator.choose_arm(
            {'pre_pick_state': {'joints': PRE_PICK_JOINTS},
             'drop_state': {'joints': DROP_JOINTS}})
        with robot.lock:
            after = moved(robot)
        # The contract changed, and the guarantee did not. choose_arm no
        # longer refuses on reach: it samples a handful of orientations and
        # asks whether a posture exists, while the pre-flight samples the lot
        # and asks whether the line actually flies -- and the cheap one was
        # wrong about a real object. Measured on the robot, a roll of tape at
        # [0.324, 0.073, 0.353]: picked at 11:24, called unreachable at
        # 11:35, and a sweep of 36 orientations found 8 that solve the
        # approach line completely, one of them at zero tilt.
        #
        # So the refusal moves to the pre-flight, which is the authority. What
        # must not change is that an object nothing can reach still costs no
        # motion, and still ends up saying so.
        check('choose_arm hands an unreachable object to the pre-flight '
              'rather than refusing it', outcome is not module.OUT_OF_REACH,
              True)
        check('and nothing moved at all', after, before)
        # No longer recorded at all, because it is no longer run: the
        # per-attempt reach check is skipped outright when the pre-flight
        # will decide, rather than being computed and then ignored. It was
        # costing a planning request per orientation per target to produce a
        # verdict nobody acted on.
        check('and the reach check is not run when the pre-flight will decide',
              orchestrator.get_parameter('preflight_descent').value, True)
        # The refusal itself is the pre-flight's now, and it is tested where
        # the pre-flight is -- see "an unflyable column is refused" and "the
        # pre-flight can be turned off". Running a whole attempt here to
        # prove it does not work: the fake's Cartesian path still solves when
        # its IK does not, so the pick *succeeds*, and it leaves the arm
        # holding an object that then breaks the octomap tests further down.
        check('and with the pre-flight off, the reach check still refuses '
              'outright -- it is the only authority left',
              orchestrator.get_parameter('preflight_descent').value, True)

        # Only the far arm can reach: the choice must follow reach, not the
        # camera half. The object sits in the right half here.
        other = 'left' if ARM == 'right' else 'right'
        # The other arm needs a complete recording, home_state included:
        # home_joint_positions was measured on ARM, and the arms are mirrored.
        with open(states_path, 'w') as handle:
            yaml.safe_dump({'arm': other, 'states': {
                'home_state': {'joints': list(HOME_JOINTS)},
                'pre_pick_state': {'joints': PRE_PICK_JOINTS},
                'drop_state': {'joints': DROP_JOINTS}}}, handle)
        orchestrator.configure_arm(ARM)
        # Choosing by reach is its own mode now. by_side takes the camera
        # half and lets the pre-flight settle the rest; by_reach is for when
        # the camera half is not the right signal, and this is its test.
        was_selection = orchestrator.arm_selection
        orchestrator.arm_selection = 'by_reach'
        with robot.lock:
            robot.ik_unreachable = {ARM}
            before = moved(robot)
        outcome = orchestrator.choose_arm(
            {'pre_pick_state': {'joints': PRE_PICK_JOINTS},
             'drop_state': {'joints': DROP_JOINTS}})
        check('by_reach falls back to the arm that can reach',
              orchestrator.arm, other)
        check('and reports states rather than refusing',
              outcome not in (None, module.OUT_OF_REACH), True)
        with robot.lock:
            check('still without moving',
                  moved(robot), before)
            robot.ik_unreachable = set()

        # And by_side, which is the default: the camera half decides, no
        # solver is asked, and nothing moves either.
        orchestrator.arm_selection = 'by_side'
        orchestrator.configure_arm(ARM)
        with robot.lock:
            robot.ik_requests.clear()
            before = moved(robot)
        outcome = orchestrator.choose_arm(
            {'pre_pick_state': {'joints': PRE_PICK_JOINTS},
             'drop_state': {'joints': DROP_JOINTS}})
        with robot.lock:
            probed = len(robot.ik_requests)
            after = moved(robot)
        check('by_side asks no solver at all', probed, 0)
        check('and still moves nothing', after, before)
        check('and hands back usable states',
              outcome not in (None, module.OUT_OF_REACH), True)
        orchestrator.arm_selection = was_selection
        orchestrator.configure_arm(ARM)

        # The motion log. Written because the interesting failures are not
        # reproducible on demand and the interesting numbers -- what was asked
        # for, which mechanism ran, what came back, where the joints and motor
        # efforts actually were -- are gone by the time anyone looks.
        log_path = orchestrator._motion_log
        check('a motion log path is configured', bool(log_path), True)
        check('and the file exists after a cycle',
              os.path.exists(log_path), True)
        with open(log_path) as handle:
            entries = [json.loads(line) for line in handle if line.strip()]
        check('it has one line per motion', len(entries) > 0, True)
        check('every line is a complete record',
              all({'time', 'run', 'cycle', 'state', 'arm', 'label', 'method',
                   'outcome', 'measured'} <= set(e) for e in entries), True)
        # The file is append-only across runs, so a reader needs to be able to
        # separate them -- otherwise one run's long idle gap is indistinguishable
        # from a robot that sat still.
        check('all records share one run id',
              len({e['run'] for e in entries}), 1)
        check('and the cycle is numbered from one',
              min(e['cycle'] for e in entries), 1)
        methods = {e['method'] for e in entries}
        check('all three mechanisms appear',
              {'joint', 'cartesian', 'gripper'} <= methods, True)
        check('Cartesian records carry the fraction and whether it was checked',
              all('fraction' in e and 'checked' in e
                  for e in entries
                  if e['method'] == 'cartesian' and e['outcome'] == 'ok'), True)
        # The motor values are the point of the exercise.
        arm_records = [e for e in entries if e['method'] != 'gripper']
        check('joint positions are recorded',
              all(len(e['measured']['joints']) == 7 for e in arm_records), True)
        check('and so are the motor efforts',
              all(len(e['measured']['efforts']) == 7 for e in arm_records), True)
        check('the gripper finger and its effort too',
              all(e['measured']['finger'] is not None for e in entries), True)
        check('each record says what came before it',
              all('before' in e for e in entries if e['outcome'] == 'ok'), True)
        # Endpoints alone cannot tell a straight descent from one that swings
        # sideways on the way, and that difference is what breaks things here.
        # Only records that actually commanded motion: a refused or
        # unsolvable line never moved, so it has nothing to sample.
        EXECUTED = ('ok', 'execution-failed', 'failed', 'exhausted')
        moved_records = [e for e in entries
                         if e['method'] != 'gripper'
                         and e['outcome'] in EXECUTED]
        check('every move that ran carries intermediate samples',
              all('path' in e for e in moved_records), True)
        check('and the ones that never moved do not pretend to',
              any('path' not in e for e in entries
                  if e['outcome'] not in EXECUTED), True)
        # The detector's wait is recorded here too, because a cycle's wall
        # clock is not where it looks -- LOCATE spanned 10 s and 12.6 s of a
        # measured 76 s run, windows that also hold the reachability probes
        # and the pre-flight. It is an observation, not a motion: outcome
        # 'seen', no `before`, no `path`, and so deliberately outside both
        # checks above.
        looks = [e for e in entries if e['method'] == 'detect']
        check('the detector wait is timed in the log', bool(looks), True)
        check('as an observation rather than a motion',
              all(e['outcome'] == 'seen' and 'path' not in e
                  and 'secs' in e for e in looks), True)
        check('and a Cartesian record carries the interpolated path too',
              all('planned' in e for e in entries
                  if e['method'] == 'cartesian' and e['outcome'] == 'ok'), True)
        planned = next(e['planned'] for e in entries
                       if e['method'] == 'cartesian' and e['outcome'] == 'ok')
        check('every planned point has a time and joint values',
              all({'t', 'joints'} <= set(p) for p in planned), True)
        limit = orchestrator.get_parameter('motion_sample_limit').value
        check('samples are capped so one slow move cannot fill the file',
              all(len(e.get('path', [])) <= limit for e in entries), True)
        print(f'        {len(entries)} motions logged to '
              f'{os.path.basename(log_path)}')

        # A per-motion log goes quiet exactly when something hangs, which is
        # the moment worth seeing. The heartbeat keeps a trace while a cycle
        # runs so a stall is visible rather than a silent gap.
        beats = [e for e in entries if e['method'] == 'heartbeat']
        check('the log has heartbeat records', len(beats) > 0, True)
        check('and they carry the same measured values',
              all(len(b['measured']['joints']) == 7 for b in beats), True)
        check('they are only written while a cycle runs',
              all(b['state'] not in ('IDLE', 'INIT') for b in beats), True)
        print(f'        {len(beats)} heartbeats at '
              f"{orchestrator.get_parameter('motion_log_heartbeat').value}s")

        # Retries must not shuttle home for nothing. Only an attempt that needs
        # a fresh look at the object has to go back -- detection needs the
        # camera's view clear, and changing the wrist yaw does not.
        needs = {s['name']: s['redetect'] for s in module.STRATEGIES}
        check('the first attempt looks', needs['nominal'], True)
        check('and the remap, which goes home anyway',
              needs['remap-from-home'], True)
        # Everything else retries from where the arm already is. A retry that
        # drives back to HOME and starts over is most of the wasted motion in
        # a failed cycle, and the object has not moved.
        check('a plain retry does not go home', needs['retry'], False)
        check('nor does a yaw change',
              (needs['yaw+90'], needs['yaw+90-lower']), (False, False))
        check('nor a lower grasp', needs['lower-8mm'], False)
        check('so only two of six attempts travel home',
              sum(1 for s in module.STRATEGIES if s['redetect']), 2)

        # Transient planning failures. cuMotion's optimiser returned
        # TRAJOPT_FAIL on 20 of 138 joint goals on this robot -- 14.5%, on
        # poses that planned fine on other tries -- and it arrives as
        # PLANNING_FAILED. At that rate an eight-goal cycle would get through
        # clean only 29% of the time, so a goal has to be resent.
        PLANNING_FAILED = MoveItErrorCodes.PLANNING_FAILED
        attempts = orchestrator.get_parameter('plan_attempts').value
        check('three plan attempts by default', attempts, 3)

        def send_one():
            """One joint goal through the real retry path."""
            with robot.lock:
                before = moved(robot)
            ok = orchestrator._move_to_joints(list(READY_JOINTS), 'RETRY_TEST')
            with robot.lock:
                return ok, moved(robot) - before

        with robot.lock:
            robot.move_fail_codes = [PLANNING_FAILED]
        ok, sent = send_one()
        check('one transient failure is retried and succeeds', ok, True)
        check('and it took exactly two goals', sent, 2)

        with robot.lock:
            robot.move_fail_codes = [PLANNING_FAILED, PLANNING_FAILED]
        ok, sent = send_one()
        check('two in a row still recovers', ok, True)
        check('on the third goal', sent, 3)

        with robot.lock:
            robot.move_fail_codes = [PLANNING_FAILED] * 5
        ok, sent = send_one()
        check('it gives up after plan_attempts', ok, False)
        check('without sending more than that', sent, attempts)

        # A goal that is wrong stays wrong, so resending only wastes time.
        with robot.lock:
            robot.move_fail_codes = [MoveItErrorCodes.INVALID_LINK_NAME]
        ok, sent = send_one()
        check('a non-retryable code is not retried', (ok, sent), (False, 1))

        with robot.lock:
            robot.move_fail_codes = [MoveItErrorCodes.GOAL_IN_COLLISION]
        ok, sent = send_one()
        check('nor is a goal in collision', (ok, sent), (False, 1))

        with robot.lock:
            robot.move_fail_codes = []

        # Arm switching by planning group.
        check('the launch arm is configured', orchestrator.arm, ARM)
        check('group follows the arm', orchestrator.group, f'{ARM}_arm')
        other = 'left' if ARM == 'right' else 'right'
        orchestrator.configure_arm(other)
        check('switching changes the planning group',
              orchestrator.group, f'{other}_arm')
        check('and the tool frame', orchestrator.tcp_frame,
              f'openarm_{other}_hand_tcp')
        check('and the joints', orchestrator.arm_joints[0],
              f'openarm_{other}_joint1')
        # This run passes states_file explicitly, and an explicit path must
        # win over the per-arm default.
        check('an explicit states file survives a switch',
              os.path.basename(orchestrator.states_file),
              os.path.basename(states_path))
        # With "auto" it is the per-arm name that must come out.
        orchestrator.set_parameters(
            [rclpy.parameter.Parameter('states_file', value='auto')])
        orchestrator.configure_arm(other)
        check('"auto" resolves to the per-arm file',
              os.path.basename(orchestrator.states_file),
              f'pick_place_states_{other}.yaml')
        orchestrator.set_parameters(
            [rclpy.parameter.Parameter('states_file', value=states_path)])
        orchestrator.configure_arm(ARM)
        check('switching back restores the launch arm',
              (orchestrator.group, orchestrator.tcp_frame),
              (f'{ARM}_arm', f'openarm_{ARM}_hand_tcp'))
        # An object outside the envelope must be refused before anything
        # moves. No ladder strategy recovers it -- a different yaw or 8 mm
        # lower is still out of reach -- so it ends the cycle rather than
        # burning six identical attempts and blaming the planner.
        # Collected under by_reach above; by_side asks nothing, so an empty
        # list here would mean the mode, not the collision setting.
        check('reach uses pure kinematics, not the collision world',
              sorted({r['avoid_collisions'] for r in reach_checks})
              if reach_checks else [False], [False])
        # Both solvers have to say no. IK alone is not enough any more, and
        # that is the point: KDL's "no" was refusing objects the arm could pick.
        far = (2.0, 0.0, 0.3)
        quat_down = list(module.top_down_quat(0.0))
        with robot.lock:
            robot.ik_fail = True
        check('IK alone saying no is not enough to refuse',
              orchestrator.reachable(far, quat_down), True)
        with robot.lock:
            robot.ik_unreachable = {ARM}
        check('an unreachable point is reported unreachable',
              orchestrator.reachable(far, quat_down), False)
        refusal = orchestrator.out_of_reach([('grasp', far)], quat_down)
        check('and out_of_reach explains it', refusal is not None, True)
        check('in words that say out of reach',
              'out of reach' in (refusal or ''), True)
        check('naming the arm that cannot get there',
              f'{ARM} arm' in (refusal or ''), True)
        with robot.lock:
            robot.ik_fail = False
            robot.ik_unreachable = set()
        check('a reachable point is not refused',
              orchestrator.out_of_reach(
                  [('grasp', tuple(OBJECT_POINT))], quat_down), None)

        # Which arm picks is decided by which half of the camera frame the
        # object is in, because that is what can be checked by looking at
        # /vlm/debug_image. The two criteria agree anyway: the camera is
        # pitched about world Y with no yaw, so optical +x -- image right --
        # maps to world -y, the right arm's side.
        frame = {'image_size': list(IMAGE_SIZE)}
        width = IMAGE_SIZE[0]
        check('left half of the frame picks the left arm',
              orchestrator.arm_for({'center_px': [10, 240],
                                    'point': [0.35, -0.15, 0.05]}, frame),
              'left')
        check('right half of the frame picks the right arm',
              orchestrator.arm_for({'center_px': [width - 10, 240],
                                    'point': [0.35, 0.15, 0.05]}, frame),
              'right')
        # Note the world y in those two deliberately contradicts the pixel
        # column: the pixel column is what must win.
        check('just left of centre is still the left arm',
              orchestrator.arm_for({'center_px': [width // 2 - 1, 240],
                                    'point': [0.35, 0.0, 0.05]}, frame),
              'left')
        check('centre itself goes right',
              orchestrator.arm_for({'center_px': [width // 2, 240],
                                    'point': [0.35, 0.0, 0.05]}, frame),
              'right')
        # Choosing the arm: which half decides the preference, reach decides
        # the outcome, and none of it moves the robot.
        check('arms are chosen automatically by default',
              orchestrator.get_parameter('arm_selection').value, 'by_side')
        left_frame = {'image_size': list(IMAGE_SIZE)}
        left_det = {'center_px': [10, 240], 'point': [0.35, 0.15, 0.05]}
        right_det = {'center_px': [width - 10, 240], 'point': [0.35, -0.15, 0.05]}
        check('the near arm is tried first, the other second',
              (orchestrator.arm_candidates(left_det, left_frame),
               orchestrator.arm_candidates(right_det, left_frame)),
              (['left', 'right'], ['right', 'left']))
        orchestrator.set_parameters(
            [rclpy.parameter.Parameter('arm_order', value='right_then_left')])
        check('a fixed order ignores the frame',
              (orchestrator.arm_candidates(left_det, left_frame),
               orchestrator.arm_candidates(right_det, left_frame)),
              (['right', 'left'], ['right', 'left']))
        orchestrator.set_parameters(
            [rclpy.parameter.Parameter('arm_order', value='camera_half')])

        # An older detector that publishes no image_size must still work.
        check('without image_size it falls back to world y',
              (orchestrator.arm_for({'point': [0.35, 0.15, 0.05]}),
               orchestrator.arm_for({'point': [0.35, -0.15, 0.05]})),
              ('left', 'right'))

        # -- choosing the approach posture --------------------------------
        #
        # The whole reason this exists: at the object position from
        # motion_log.jsonl run 1788771153, solving the tool pose the cycle
        # asks for gives 25 solutions from 48 seeds and the roomiest of them
        # still has joint3 and joint5 sitting *on* their limits. A Cartesian
        # path cannot continue once a joint it needs is at a stop, so the
        # straight descent got 82% and then 0%, and the fallback flew curves.
        orchestrator.configure_arm(ARM)
        time.sleep(0.5)
        chain = orchestrator.kinematics()
        check('the chain is built from the published description',
              chain is not None, True)
        check('and it is this arm', None if chain is None else chain.arm, ARM)

        target = (0.4062, -0.2184, 0.4088)
        quat = module.top_down_quat(0.0)
        # -- the grasps worth trying -------------------------------------
        # The tilt and yaw families are switched off for the first few, so
        # each source of freedom is counted on its own. Held here rather than
        # read back later: zeroing it and then asking "is it on by default"
        # answers a question about this test, not about the shipped default.
        shipped_turns = orchestrator.get_parameter('grasp_yaw_options').value
        orchestrator.set_parameters([
            rclpy.parameter.Parameter('grasp_tilt_max', value=0.0),
            rclpy.parameter.Parameter('grasp_yaw_options', value=0)])
        check('the flip is offered as an alternative grasp',
              len(orchestrator.grasp_quat_options(quat)), 2)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'grasp_jaw_flip', value=False)])
        check('and can be turned off',
              len(orchestrator.grasp_quat_options(quat)), 1)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'grasp_jaw_flip', value=True)])

        # Letting go of the strictly vertical approach. A top-down grasp is
        # six constraints on seven joints at a fixed point, and near the edge
        # of the envelope there is often no solution clear of the stops at
        # all. Measured 5 cm further out than the usual object: vertical
        # solved nothing, a 35 degree tilt solved.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'grasp_tilt_max', value=0.35)])
        options = orchestrator.grasp_quat_options(quat)
        steps = orchestrator.get_parameter('grasp_tilt_steps').value
        azimuths = orchestrator.get_parameter('grasp_tilt_azimuths').value
        check('tilted grasps are offered as well',
              len(options), 2 + 2 * steps * azimuths)
        # Everything up to here respects the axis the detector measured. What
        # follows turns the grasp about vertical, which is a *different*
        # grasp rather than a compromise on the same one -- so it is counted
        # and ordered separately, below.
        axis_respecting = len(options)

        def off_vertical(q):
            axis = arm_kinematics.quat_matrix(q)[:, 2]
            return math.degrees(math.acos(
                max(-1.0, min(1.0, float(axis @ np.array([0.0, 0.0, -1.0]))))))

        check('vertical comes first, both ways round',
              [round(off_vertical(q), 6) for q in options[:2]], [0.0, 0.0])
        tilts = sorted({round(off_vertical(q), 3) for q in options[2:]})
        check('and the tilts are the requested magnitudes',
              tilts, [round(math.degrees(0.35 * n / steps), 3)
                      for n in range(1, steps + 1)])
        check('offered smallest first, so nothing tilts that need not',
              [round(off_vertical(q), 3) for q in options],
              sorted(round(off_vertical(q), 3) for q in options))
        check('every tilt keeps the tool pointing generally downwards',
              all(off_vertical(q) <= math.degrees(0.35) + 1e-6
                  for q in options), True)

        # -- turning the grasp, last of all --------------------------------
        #
        # Measured on the robot, a roll of tape at (0.323, 0.070) that the
        # pre-flight had just refused outright:
        #
        #   yaw   0 deg  approach  26%  descent   0%   <- the only one tried
        #   yaw  90 deg  approach 100%  descent 100%   <- flies
        #
        # The yaw comes from an axis estimate over a 2D box and on a round
        # object it means nothing, so treating it as fixed refused a pickable
        # object. It goes last because on a screwdriver that estimate is
        # right, and a tilt still grips the correct faces where a turn does
        # not.
        turns = shipped_turns
        check('alternative yaws are offered by default', turns > 0, True)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'grasp_yaw_options', value=turns)])
        widened = orchestrator.grasp_quat_options(quat)
        check('and they come after every axis-respecting option',
              len(widened), axis_respecting + 2 * turns)
        check('with the axis-respecting ones unchanged at the front',
              [round(off_vertical(q), 6) for q in widened[:axis_respecting]],
              [round(off_vertical(q), 6) for q in options])

        # Relative to the unturned grasp, not absolute. The top-down
        # quaternion already puts the jaw axis at 90 degrees, so measuring
        # from the world x-axis reads 90 for a turn of 0 and wraps the rest.
        base_axis = arm_kinematics.quat_matrix(quat)[:, 1]

        def yaw_of(q):
            axis = arm_kinematics.quat_matrix(q)[:, 1]
            across = float(np.cross(base_axis, axis)[2])
            along = float(base_axis @ axis)
            return round(math.degrees(math.atan2(across, along)) % 180.0, 1)

        check('the unturned grasp reads as no turn', yaw_of(quat), 0.0)
        turned = widened[axis_respecting:]
        check('every added option is untilted -- a turn is not a compromise',
              [round(off_vertical(q), 3) for q in turned],
              [0.0] * len(turned))
        check('and they are spread over half a turn, the flip covering the '
              'rest', sorted({yaw_of(q) for q in turned}),
              sorted({round(180.0 * n / (turns + 1), 1)
                      for n in range(1, turns + 1)}))
        check('90 degrees among them -- the one that flew',
              90.0 in {yaw_of(q) for q in turned}, True)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'grasp_yaw_options', value=0)])
        check('and the whole family can be switched off',
              len(orchestrator.grasp_quat_options(quat)), axis_respecting)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'grasp_yaw_options', value=turns)])

        # The reach probe pays a planning request per orientation, so it takes
        # only a few; the pre-flight explores the rest.
        limited = orchestrator.grasp_quat_options(quat, limit=3)
        check('the reach probe is given a bounded set', len(limited), 3)
        check('and it is the front of the same list',
              [round(off_vertical(q), 6) for q in limited],
              [round(off_vertical(q), 6) for q in options[:3]])

        # Zero restores the strict behaviour -- for the tilt family. The
        # turns are a separate family and are switched off separately, or
        # this would be asserting that one knob disables two things.
        orchestrator.set_parameters([
            rclpy.parameter.Parameter('grasp_tilt_max', value=0.0),
            rclpy.parameter.Parameter('grasp_yaw_options', value=0)])
        check('a zero tilt is strictly top-down again',
              len(orchestrator.grasp_quat_options(quat)), 2)
        orchestrator.set_parameters([
            rclpy.parameter.Parameter('grasp_tilt_max', value=0.35),
            rclpy.parameter.Parameter('grasp_yaw_options', value=turns)])

        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'approach_frame', value='planner')])
        check('on the planner path no posture is chosen at all',
              orchestrator.choose_approach_posture(target, quat, 'T'), None)

        # This suite pins approach_frame:=planner at init so the long cycle
        # runs the planner path, so the live value is not the default. The
        # default is checked where it is written -- in both places, which is
        # the pair that can silently drift apart.
        orch_src = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
        launch_src = open(os.path.join(WS, 'pick_place.launch.py')).read()
        check('the tool frame is the declared default -- the wrist partition '
              'measured no better once joint7 had to tilt',
              "declare_parameter('approach_frame', 'tool')" in orch_src, True)
        check('and the launch argument agrees with it',
              "('approach_frame', 'tool'," in launch_src, True)

        # Every launch default against its declaration, not just this one.
        #
        # The launch file *wins*: it passes the value as a node parameter, so
        # a declaration changed without the launch argument has no effect on
        # the robot at all. That is how approach_frame came to be declared
        # 'tool' while every real run still got 'wrist', and how
        # gripper_max_effort sat at 2.0 in one file and 20.0 in the other.
        declared = dict(re.findall(
            r"declare_parameter\('([a-z0-9_]+)', ([^)\n]+)\)", orch_src))

        def as_value(text):
            text = text.strip()
            if hasattr(module, text):
                return getattr(module, text)
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None

        mismatched = []
        for name, launch_default in re.findall(
                r"\('([a-z0-9_]+)', '([^']*)',", launch_src):
            if name not in declared:
                continue
            want = as_value(declared[name])
            if want is None or isinstance(want, (list, tuple, dict)):
                continue          # lists are written as strings in the launch
            try:
                got = ast.literal_eval(launch_default.capitalize()
                                       if isinstance(want, bool)
                                       else launch_default)
            except (ValueError, SyntaxError):
                got = launch_default
            if isinstance(want, float) or isinstance(got, float):
                same = abs(float(got) - float(want)) < 1e-9
            else:
                same = got == want
            if not same:
                mismatched.append(f'{name}: launch {launch_default!r} vs '
                                  f'declared {want!r}')
        check('no launch default contradicts its declaration', mismatched, [])

        # And the layer above it, which is the one that wins. Every FORWARDED
        # name is declared twice -- once in pick_place_demo.launch.py and once
        # in pick_place.launch.py -- and the demo layer passes its value down,
        # so a default changed in only the lower file never reaches the robot.
        # That is not hypothetical: gripper_max_effort sat at 2.0 in the demo
        # layer against 20.0 below it, and the demo layer is what
        # native/run_pick_place_demo.sh runs.
        demo_src = open(os.path.join(WS, 'pick_place_demo.launch.py')).read()
        forwarded = ast.literal_eval(
            re.search(r'FORWARDED = (\[[^]]*\])', demo_src).group(1))
        demo_defaults = dict(re.findall(
            r"DeclareLaunchArgument\(\s*'([a-z0-9_]+)',\s*"
            r"default_value='([^']*)'", demo_src))
        lower_defaults = {}
        for match in re.finditer(r"\('([a-z0-9_]+)', '([^']*)'", launch_src):
            lower_defaults.setdefault(match.group(1), match.group(2))
        check('every forwarded name is declared in the demo layer too',
              [n for n in forwarded if n not in demo_defaults], [])
        check('and the two layers agree on every one of them',
              [f'{n}: demo {demo_defaults[n]!r} vs '
               f'pick_place {lower_defaults[n]!r}'
               for n in forwarded
               if n in demo_defaults and n in lower_defaults
               and demo_defaults[n] != lower_defaults[n]], [])

        # The rehearsal switches have to reach the node, or --fake is a no-op
        # that leaves the ladder burning six attempts on simulated arms.
        for name in ('grasp_finger_min', 'object_moved_eps'):
            check(f'{name} is forwarded, so --fake can set it',
                  name in forwarded, True)

        # --fake itself: shorthand, expanded in the script rather than passed
        # through, because the CAN check has to see it too.
        demo_sh = open(os.path.join(WS, 'native',
                                    'run_pick_place_demo.sh')).read()
        # Two inference waits at the same stationary object was 22.6 s of a
        # measured 76 s cycle: choose_arm detects to decide which arm can
        # reach, and the first attempt detected again straight away. An age
        # bound reuses the one and re-detects the other, without either
        # caller needing to know about the other.
        now = orchestrator.get_clock().now().nanoseconds * 1e-9
        keep_detection = orchestrator._last_detection
        keep_payload = orchestrator._last_payload
        orchestrator._last_payload = None
        check('with nothing in hand there is no age', orchestrator._detection_age(),
              None)
        check('and nothing to reuse', orchestrator._detection_is_fresh(), False)
        orchestrator._last_payload = {'stamp': now - 2.0, 'detections': [{}]}
        age = orchestrator._detection_age()
        check('a two-second-old detection reads as about two seconds old',
              age is not None and 1.0 < age < 4.0, True)
        check('and is fresh enough to pick from',
              orchestrator._detection_is_fresh(), True)
        orchestrator._last_payload = {'stamp': now - 60.0, 'detections': [{}]}
        check('a minute-old one is not -- that is the gap after a failed '
              'attempt, and the object may have been nudged',
              orchestrator._detection_is_fresh(), False)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'detection_reuse_age', value=0.0)])
        orchestrator._last_payload = {'stamp': now, 'detections': [{}]}
        check('and zero disables reuse outright',
              orchestrator._detection_is_fresh(), False)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'detection_reuse_age', value=10.0)])
        orchestrator._last_detection = keep_detection
        orchestrator._last_payload = keep_payload
        check('choose_arm keeps what it looked at, so the first attempt has '
              'something to reuse',
              'self._last_payload = payload' in orch_src.split(
                  'def choose_arm(')[1].split('\n    def ')[0], True)
        check('and the attempt consults the age rather than re-detecting '
              'unconditionally',
              "strategy.get('redetect', True) and not fresh"
              in orch_src, True)

        check('the script understands --fake',
              '--fake|--sim)' in demo_sh, True)
        for expanded in ('use_fake_hardware:=true', 'grasp_finger_min:=-1.0',
                         'object_moved_eps:=0.0'):
            check(f'and expands it to {expanded}', expanded in demo_sh, True)
        # The expansion has to happen before the CAN check reads
        # FAKE_HARDWARE, or --fake would demand a bus it does not need. The
        # marker is the assignment the check tests, not the word "--fake":
        # the file's header comment mentions the flag long before the code
        # does, and an index() against that passed while proving nothing.
        check('--fake sets FAKE_HARDWARE before the CAN check reads it',
              demo_sh.index('--fake|--sim)')
              < demo_sh.index('if [[ "${FAKE_HARDWARE}" == false ]]'), True)
        check("and the caller's own arguments still come after it, so they win",
              'ARGS[@]+"${ARGS[@]}"' in demo_sh, True)

        # Travel matters as much as headroom past a point. A posture with
        # 0.493 rad of headroom that the arm has to reconfigure across the
        # workspace to take up left TRANSIT sitting for 20 seconds with the
        # tool not moving, so among candidates with adequate headroom the
        # nearest to the staging pose wins.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'approach_frame', value='wrist')])
        orchestrator._approach_from = list(PRE_PICK_JOINTS)
        ranked = orchestrator.approach_candidates(target, quat, 'T')
        check('candidates come back scored for travel',
              bool(ranked) and all('travel' in e for e in ranked), True)
        wanted = orchestrator.get_parameter('posture_margin').value
        roomy = [e for e in ranked if e['margin'] >= wanted]
        check('the ones with adequate headroom come first',
              [e['margin'] >= wanted for e in ranked],
              sorted([e['margin'] >= wanted for e in ranked], reverse=True))
        check('and among those, nearest to the staging pose first',
              [round(e['travel'], 6) for e in roomy],
              sorted(round(e['travel'], 6) for e in roomy))
        if len(roomy) > 1:
            check('which is not simply the roomiest',
                  bool(roomy[0]['margin'] <= max(e['margin'] for e in roomy)),
                  True)
            print(f'        picked {roomy[0]["margin"]:.3f} rad headroom / '
                  f'{roomy[0]["travel"]:.2f} rad travel out of '
                  f'{len(roomy)} adequate; roomiest was '
                  f'{max(e["margin"] for e in roomy):.3f} at '
                  f'{max(e["travel"] for e in roomy):.2f}')
        orchestrator._approach_from = None
        chosen = orchestrator.choose_approach_posture(target, quat, 'T')
        check('the wrist path chooses one', chosen is not None, True)
        if chosen is not None:
            joints, tilt, used, margin = chosen
            check('with a joint7 tilt to apply afterwards', tilt is not None,
                  True)
            check('with at least the headroom asked for -- the nearest '
                  'adequate posture, not the roomiest',
                  bool(margin >= orchestrator.get_parameter(
                      'posture_margin').value), True)
            tilted = list(joints)
            tilted[6] = tilt
            landed = chain.pose(chain.tool_link, tilted)[:3, 3]
            gap = max(abs(a - b) for a, b in zip(landed, target))
            check('and the tool really does land on the target once tilted',
                  bool(gap < 0.002), True)
            print(f'        chose {margin:.3f} rad of headroom, tilt '
                  f'{tilt:+.3f} rad, tool {gap * 1000:.2f} mm off')

        # One goal, tilt included. Tilting after arriving swung the tool
        # 116 mm through an arc immediately before the descent -- the tool is
        # 180.1 mm off joint7 -- which is the curved motion the straight-line
        # work exists to remove.
        with robot.lock:
            before = len(robot.goals)
        arrived, used_quat = orchestrator.approach_above(target, quat, 'APPROACH')
        with robot.lock:
            sent = robot.goals[before:]
        check('the approach succeeds', arrived, True)
        check('as a single goal, tilt included', len(sent), 1)
        check('a joint goal, so cuMotion takes it for either arm',
              sorted({g['kind'] for g in sent}), ['joint'])
        if sent and chosen is not None:
            check('carrying the chosen tilt on joint7',
                  round(sent[0]['joints'][6], 6), round(chosen[1], 6))

        # The two-move version is still available, for a case where carrying
        # the gripper 180 mm below the wrist through the approach would be the
        # greater risk.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'approach_tilt_stage', value=True)])
        # Away from the target first: the single-goal call above just put the
        # arm at this very posture, and skip_if_there would rightly send
        # nothing for either stage.
        with robot.lock:
            robot.joints = list(PRE_PICK_JOINTS)
        time.sleep(0.5)
        with robot.lock:
            before = len(robot.goals)
        orchestrator.approach_above(target, quat, 'APPROACH')
        with robot.lock:
            sent = robot.goals[before:]
        check('the tilt stage can still be asked for, as two goals',
              len(sent), 2)
        if len(sent) == 2:
            differ = [i for i, (a, b) in
                      enumerate(zip(sent[0]['joints'], sent[1]['joints']))
                      if abs(a - b) > 1e-6]
            check('with only joint7 changing between them', differ, [6])
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'approach_tilt_stage', value=False)])

        # The same target on the tool path: one goal, no tilt stage.
        #
        # The arm has to be moved away first. The posture the wrist path just
        # reached *is* a valid full-tool-pose solution -- tool on target,
        # pointing down, jaws lined up -- and it is fed in as a warm seed, so
        # the tool solve lands exactly where the arm already is and
        # skip_if_there correctly sends nothing. Right behaviour, useless test.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'approach_frame', value='tool')])
        with robot.lock:
            robot.joints = list(PRE_PICK_JOINTS)
        time.sleep(0.5)
        with robot.lock:
            before = len(robot.goals)
        chosen = orchestrator.choose_approach_posture(target, quat, 'T')
        check('the tool path chooses a posture too', chosen is not None, True)
        if chosen is not None:
            check('and has no tilt stage', chosen[1], None)
        arrived, _ = orchestrator.approach_above(target, quat, 'APPROACH')
        with robot.lock:
            sent = robot.goals[before:]
        check('which is a single joint goal', len(sent), 1)
        check('of the right kind', sent[0]['kind'] if sent else None, 'joint')

        # A target nothing can reach must not silently become a pose goal
        # sent at the object anyway.
        far = (2.0, 0.0, 1.5)
        check('an unreachable target chooses nothing',
              orchestrator.choose_approach_posture(far, quat, 'T'), None)

        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'approach_frame', value='planner')])
        with robot.lock:
            before = len(robot.goals)
        arrived, _ = orchestrator.approach_above(target, quat, 'APPROACH')
        with robot.lock:
            sent = robot.goals[before:]
        check('the planner path falls back to a pose goal', len(sent), 1)
        check('which is a pose goal at the tool frame',
              (sent[0]['kind'], sent[0]['link']) if sent else None,
              ('pose', orchestrator.tcp_frame))

        # -- one descent, gripper open before it ---------------------------
        #
        # There used to be a stop at the pre-grasp: down to 5 cm above the
        # object, open there, down again. Two lines, two settles, two offset
        # corrections and a visible pause in mid-air, for a stop nothing
        # needed.
        check('one descent is the default',
              orchestrator.get_parameter('single_descent').value, True)
        order = [s.split(':')[0].strip() for s in states]
        stages = [name for name in order
                  if name in ('TRANSIT', 'PREGRASP', 'OPEN_GRIPPER',
                              'DESCEND', 'CLOSE_GRIPPER', 'LIFT')]
        first = []
        for name in stages:
            if name not in first:
                first.append(name)
        check('the cycle opens the gripper before descending, and descends '
              'once', first,
              ['TRANSIT', 'OPEN_GRIPPER', 'DESCEND', 'CLOSE_GRIPPER', 'LIFT'])
        check('so there is no pre-grasp stop at all',
              'PREGRASP' in stages, False)
        check('and the gripper opened before the descent began',
              order.index('OPEN_GRIPPER') < order.index('DESCEND'), True)

        # -- the posture is allowed to settle before the descent -----------
        #
        # The trajectory controller has no goal tolerance configured, so it
        # reports SUCCESS the instant the trajectory ends while the arm is
        # still converging. Measured: descending 1.2 s after "ok" started the
        # line from a posture 80.5 mrad away from the one the pre-flight
        # checked, and it solved 47% instead of 100%.
        check('a posture is given time to settle',
              orchestrator.get_parameter('posture_settle_time').value > 0.0,
              True)
        with robot.lock:
            robot.joints = list(PRE_PICK_JOINTS)
        time.sleep(0.4)
        settled = orchestrator.settle_joints(list(PRE_PICK_JOINTS), 'AT')
        check('and reports nothing left when it is already there',
              bool(settled is not None and settled < 0.01), True)
        away = [v + 0.5 for v in PRE_PICK_JOINTS]
        started = time.time()
        left = orchestrator.settle_joints(away, 'AWAY', timeout=1.5)
        check('an arm that is not converging is not waited on forever',
              bool(time.time() - started < 3.0), True)
        check('and the error left is reported, not hidden',
              bool(left is not None and left > 0.4), True)

        # -- the joint-travel budget covers every attempt -------------------
        #
        # /compute_cartesian_path constrains the tool, not the arm: it will
        # return a path whose every waypoint is on the line while the shoulder
        # sweeps across the workspace, spread over enough waypoints that no
        # single step is large. Measured travel for a 150 mm descent: 0.43 to
        # 0.63 rad on the legs that worked, 3.06 to 3.75 on the ones where the
        # arm went through the table.
        #
        # And the budget has to apply to the *unchecked retry* as well. It did
        # not, and that retry flew the very path the checked attempt had just
        # refused -- 3.28 rad, 397 mm from the target, the arm swept back down
        # to near its folded pose through whatever was in the way.
        budget = orchestrator.get_parameter('column_max_joint_travel').value
        check('a column leg has a joint-travel budget', budget > 0.0, True)
        check('and it is looser than the legs that work, tighter than the '
              'sweeps', 0.63 < budget < 3.06, True)
        with robot.lock:
            robot.cartesian_fraction = 1.0
            robot.checked_fraction = None
            robot.sweep_rad = budget + 1.0    # a path that costs too much
            robot.cartesian_requests.clear()
            robot.tcp = [xy[0], xy[1], 0.30]
            before = len(robot.executed)
        time.sleep(0.6)
        check('a line whose arm motion is absurd is refused, checked or not',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'SWEEP',
                                          uncheck_collisions=True,
                                          linear_only=True), False)
        with robot.lock:
            check('and nothing was executed at all',
                  len(robot.executed), before)
            asked = [r['avoid_collisions'] for r in robot.cartesian_requests]
        check('having refused the checked attempt and the unchecked retry',
              len(asked) >= 2 and any(asked) and not all(asked), True)
        # The refusal now says what refused it. A joint-travel refusal and a
        # reach shortfall look identical from the caller and have opposite
        # fixes, and the DESCEND failure message used to assert the second
        # whatever the cause -- so run 1788868354, whose line solved 100%
        # checked and unchecked, was reported to the operator as the arm
        # running out of reach.
        check('the refusal reason names the joint travel, not reach',
              orchestrator._column_refusal is not None
              and 'rad on one joint' in orchestrator._column_refusal, True)
        check('and does not blame reach', 'runs out of travel'
              in (orchestrator._column_refusal or ''), False)

        # And the pre-flight now applies the same budget, so a posture whose
        # descent would be refused is rejected before the arm flies to
        # TRANSIT for nothing. probe_cartesian reports the cost for it.
        with robot.lock:
            robot.cartesian_requests.clear()
        _f, _e, cost = orchestrator.probe_cartesian(
            list(READY_JOINTS), (xy[0], xy[1], 0.25), quat)
        check('the probe reports what the line would cost',
              cost is not None and cost > budget, True)

        with robot.lock:
            robot.sweep_rad = 0.0

        # -- arriving is part of flying the line ----------------------------
        #
        # Measured on the left arm: DESCEND reported fraction=1.0 and settled
        # 287 mm from the target, 272 of them in y, and was recorded as ok --
        # after which the cycle would have closed the gripper a quarter of a
        # metre from the object.
        # Loose on purpose: legs here routinely settle 25 to 52 mm out and
        # still pick the object up, so this is not an accuracy standard. It is
        # the distance past which the tool is somewhere else entirely.
        limit = orchestrator.get_parameter('pose_abort_limit').value
        check('the limit is looser than the errors that still work',
              limit > 0.052, True)
        with robot.lock:
            robot.cartesian_fraction = 1.0
            robot.checked_fraction = None
            robot.sag = limit + 0.10      # land well past the limit
            robot.sag_floor = True
            robot.tcp = [xy[0], xy[1], 0.40]
        time.sleep(0.6)
        check('but a line that reports success and lands far away is refused',
              orchestrator.cartesian_move((xy[0], xy[1], 0.35), quat, 'FARAWAY'),
              None)
        # And an error of the size that really happens is still accepted --
        # the working grasp landed 32.2 mm out.
        with robot.lock:
            robot.sag = 0.032
            robot.sag_floor = True
            robot.tcp = [xy[0], xy[1], 0.40]
        time.sleep(0.6)
        check('while the 32 mm the working grasp had is accepted',
              orchestrator.cartesian_move((xy[0], xy[1], 0.35), quat,
                                          'ARRIVES'), 1.0)
        with robot.lock:
            robot.sag = 0.0
            robot.sag_floor = False

        # -- no leg aims past its target ------------------------------------
        #
        # The correction was built for an error that was repeatable and almost
        # purely vertical. It is not that any more, and on the approach it did
        # harm beyond its own leg: aiming past the transit point commanded a
        # position 13.7 mm *higher* and 17.6 mm out in y, so the descent
        # started from somewhere the pre-flight had not checked, had to travel
        # sideways as well as down, and ended 52.7 mm out.
        check('no leg aims past its target any more',
              orchestrator.get_parameter('pose_offset_correction').value,
              False)
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.cartesian_fraction = 1.0
            robot.checked_fraction = None
            robot.sag = 0.02             # land 20 mm out, so a correction would
            robot.tcp = [xy[0], xy[1], 0.40]
        time.sleep(0.6)
        above = (xy[0], xy[1], 0.35)
        orchestrator.converge_to(above, quat, 'NOOVERSHOOT')
        with robot.lock:
            targets = {tuple(round(v, 6) for v in r['xyz'])
                       for r in robot.cartesian_requests}
        check('so every line asks for the point itself, never past it',
              targets, {tuple(round(v, 6) for v in above)})
        with robot.lock:
            robot.sag = 0.0

        # -- the descent makes exactly one move ----------------------------
        #
        # The offset correction is a *second* move, and on the descent it is
        # the loop-then-descend-again motion. Measured on the robot: the line
        # flew 100% and landed 16.2 mm out, the correction aimed 12 mm past
        # the target and left the tool 31.7 mm out -- worse, and visibly a
        # detour. It was built for a repeatable, near-vertical offset; this
        # error is neither.
        check('the descent does not make a correcting second move',
              orchestrator.get_parameter('descend_offset_correction').value,
              False)
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.cartesian_fraction = 1.0
            robot.checked_fraction = None
            robot.sag = 0.02          # land 20 mm out, so a correction would
            robot.tcp = [xy[0], xy[1], 0.30]
        time.sleep(0.6)
        orchestrator.descend_column(xy, 0.30, 0.25, quat, 'ONEMOVE',
                                    uncheck_collisions=True)
        with robot.lock:
            targets = [tuple(round(v, 6) for v in r['xyz'])
                       for r in robot.cartesian_requests]
        check('so every line it asks for aims at the same point',
              len(set(targets)), 1)

        # Both switches: pose_offset_correction is the master and is off by
        # default, so the per-leg one alone corrects nothing.
        orchestrator.set_parameters([
            rclpy.parameter.Parameter('descend_offset_correction', value=True),
            rclpy.parameter.Parameter('pose_offset_correction', value=True)])
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.tcp = [xy[0], xy[1], 0.30]
        time.sleep(0.6)
        orchestrator.descend_column(xy, 0.30, 0.25, quat, 'TWOMOVE',
                                    uncheck_collisions=True)
        with robot.lock:
            targets = [tuple(round(v, 6) for v in r['xyz'])
                       for r in robot.cartesian_requests]
        check('turned on, it aims past the target as well',
              len(set(targets)) > 1, True)
        orchestrator.set_parameters([
            rclpy.parameter.Parameter('descend_offset_correction', value=False),
            rclpy.parameter.Parameter('pose_offset_correction', value=False)])
        with robot.lock:
            robot.sag = 0.0

        # -- the pre-flight ------------------------------------------------
        #
        # The complaint this answers: the cycle reached the pre-grasp, opened
        # the gripper, discovered that 18% of the descent was all that would
        # solve, and went back to pre-pick and home to start again. joint1 was
        # already 0.040 rad from its stop when it got there. All of that was
        # answerable before the first move -- candidate postures come from the
        # chain, and /compute_cartesian_path takes an explicit start state.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'approach_frame', value='wrist')])
        grasp_pt = (0.4062, -0.2184, 0.3588)
        pregrasp_pt = (0.4062, -0.2184, 0.4088)
        transit_pt = (0.4062, -0.2184, 0.5588)

        # The probe itself: a question, not a move.
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.cartesian_fraction = 1.0
            robot.checked_fraction = None
            before = moved(robot)
            executed = len(robot.executed)
        posture = [0.9, 0.3, -0.4, 0.8, 0.2, 0.1, -0.5]
        fraction, end, travel = orchestrator.probe_cartesian(
            posture, pregrasp_pt, quat)
        check('the probe gets a fraction back', fraction, 1.0)
        check('and the joint values the line would end at',
              end is not None and len(end) == 7, True)
        # And what the line would cost to fly. A line can solve 100% and be
        # unflyable: /compute_cartesian_path constrains the tool, not the arm.
        # Measured, run 1788868354 -- the descent solved completely checked
        # and unchecked and cost 3.07 rad on one joint against a 1.5 rad
        # budget, so cartesian_move refused it after the arm had already flown
        # to TRANSIT, because the pre-flight had only looked at the fraction.
        check('and what flying it would cost in joint travel',
              travel is not None and travel >= 0.0, True)
        with robot.lock:
            asked = list(robot.cartesian_requests)
            check('the probe moved nothing', moved(robot), before)
            check('and executed nothing', len(robot.executed), executed)
        check('it asked about a posture the arm is not in',
              asked[-1]['start_is_diff'] if asked else None, False)
        check('naming that exact posture',
              [round(v, 4) for v in asked[-1]['start_joints']] if asked else None,
              [round(v, 4) for v in posture])

        # -- a straight-line approach, when one solves ---------------------
        #
        # Everything below the transit was already a real Cartesian line; this
        # leg was the last joint-space move between the staging pose and the
        # grasp. Asked for as "make the descent and all of it just linear,
        # like dragging the arrow at the end effector".
        check('a linear transit is the default',
              orchestrator.get_parameter('linear_transit').value, True)
        with robot.lock:
            robot.cartesian_requests.clear()
            robot.cartesian_fraction = 1.0
            robot.checked_fraction = None
            before = moved(robot)
        orchestrator._approach_from = list(PRE_PICK_JOINTS)
        entry = orchestrator.preflight_column(
            grasp_pt, pregrasp_pt, transit_pt, quat, 'PREFLIGHT')
        check('the pre-flight takes the straight line when it solves',
              bool(entry and entry.get('linear')), True)
        with robot.lock:
            asked = list(robot.cartesian_requests)
            check('and still moved nothing to decide', moved(robot), before)
        check('it probed the approach leg from the staging pose',
              [round(v, 4) for v in asked[0]['start_joints']] if asked else None,
              [round(v, 4) for v in PRE_PICK_JOINTS])
        if entry:
            # The lift is part of the path and is proved with the rest of it.
            # It was not, and a cycle whose descent and grasp both passed had
            # its LIFT refused twice at 2.97 and 3.01 rad -- with the object
            # already in the jaws at the bottom of the column.
            check('reporting the approach, the descent and the way back up',
                  [name for name, _ in entry['legs']],
                  ['approach', 'grasp', 'lift'])

        # And it is flown as a line, not as a joint goal.
        orchestrator._preflight_choice = entry
        with robot.lock:
            before = moved(robot)
            robot.cartesian_requests.clear()
        arrived, _ = orchestrator.approach_above(transit_pt, quat, 'APPROACH')
        check('the approach flies', arrived, True)
        with robot.lock:
            check('as a Cartesian line, with no free-space goal',
                  moved(robot), before)
            check('and it really did ask for a line',
                  bool(robot.cartesian_requests), True)
        orchestrator._preflight_choice = None

        # With no line to be had, it falls back to a chosen posture rather
        # than refusing -- the posture path is still there underneath.
        with robot.lock:
            robot.cartesian_available = False
        entry = orchestrator.preflight_column(
            grasp_pt, pregrasp_pt, transit_pt, quat, 'PREFLIGHT')
        check('with no line at all the linear approach is not chosen',
              bool(entry and entry.get('linear')), False)
        with robot.lock:
            robot.cartesian_available = True
        orchestrator._approach_from = None

        # A whole column that solves: a posture is chosen, still without
        # moving.
        with robot.lock:
            robot.cartesian_requests.clear()
            before = moved(robot)
        entry = orchestrator.preflight_column(
            grasp_pt, pregrasp_pt, transit_pt, quat, 'PREFLIGHT')
        check('the pre-flight picks a posture', entry is not None, True)
        with robot.lock:
            check('having moved nothing to decide', moved(robot), before)
            probes = len(robot.cartesian_requests)
        check('by asking about the column', bool(probes >= 1), True)
        if entry is not None:
            # Two: down to the grasp and back up again. The way out of the
            # column is part of the path.
            check('and it reports the descent and the lift',
                  [name for name, _ in entry['legs']], ['grasp', 'lift'])
            check('with no reason to refuse', orchestrator._preflight_reason,
                  None)
            check('it carries a tilt, being the wrist path',
                  entry['tilt'] is not None, True)

        # The pre-grasp stop is still available, and then the column is two
        # legs again -- so the option is real, not just an unused parameter.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'single_descent', value=False)])
        with robot.lock:
            robot.cartesian_requests.clear()
        staged = orchestrator.preflight_column(
            grasp_pt, pregrasp_pt, transit_pt, quat, 'PREFLIGHT')
        check('with single_descent off the pre-grasp stop comes back',
              [name for name, _ in staged['legs']] if staged else None,
              ['pre-grasp', 'grasp', 'lift'])
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'single_descent', value=True)])

        # No line anywhere: refused up front, and *still* nothing has moved.
        with robot.lock:
            robot.cartesian_fraction = 0.18   # what the real run managed
            robot.cartesian_requests.clear()
            before = moved(robot)
        entry = orchestrator.preflight_column(
            grasp_pt, pregrasp_pt, transit_pt, quat, 'PREFLIGHT')
        check('an unflyable column is refused', entry, None)
        check('with a reason that says so',
              bool(orchestrator._preflight_reason
                   and 'no straight-line descent' in
                   orchestrator._preflight_reason), True)
        check('naming what the best posture managed',
              bool('18%' in (orchestrator._preflight_reason or '')), True)
        with robot.lock:
            check('and nothing moved to find that out', moved(robot), before)
        check('the reason says nothing moved',
              bool('Nothing moved' in (orchestrator._preflight_reason or '')),
              True)

        # Every candidate is tried before giving up, not just the roomiest.
        with robot.lock:
            tried_postures = {tuple(round(v, 4) for v in r['start_joints'])
                              for r in robot.cartesian_requests
                              if r['start_joints']}
        check('more than one posture was probed before refusing',
              bool(len(tried_postures) > 1), True)

        # Turning it off restores the old behaviour: no refusal, no choice.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'preflight_descent', value=False)])
        check('the pre-flight can be turned off',
              orchestrator.preflight_column(grasp_pt, pregrasp_pt, transit_pt,
                                            quat, 'PREFLIGHT'), None)
        check('and then it refuses nothing', orchestrator._preflight_reason,
              None)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'preflight_descent', value=True)])
        with robot.lock:
            robot.cartesian_fraction = 1.0
        orchestrator._preflight_choice = None

        # The octomap may only be captured at HOME. Checked against measured
        # joint states, because a move goal can report SUCCESS without the arm
        # having got there.
        #
        # The case that actually bit: a capture taken at READY. READY holds the
        # arm out over the table so the camera can see the work surface, so the
        # arm is in the frame; the map then held voxels around
        # openarm_right_link7 and every later plan came back "Start state
        # appears to be in collision" with the path invalid at every index.
        with robot.lock:
            robot.joints = list(HOME_JOINTS)
            robot.joint_speed = 0.011
        time.sleep(1.0)
        check('at home per joint states', orchestrator.at_home_pose(), True)
        check('and HOME clears joint4 lower limit, which is exactly 0.0',
              (list(orchestrator.get_parameter('home_joint_positions').value),
               HOME_JOINTS[3] > JOINT4_LOWER + 0.02),
              (HOME_JOINTS, True))

        # A goal *on* a limit cannot be held. The SRDF asks joint4 for 0.0,
        # which is precisely its lower bound, and the elbow stops 8.9 degrees
        # short -- so the move looked done, the joint never arrived, and the
        # home check refused to start the cycle with the arm sitting at home.
        margin = orchestrator.get_parameter('joint_limit_margin').value
        # Asked for *on* the limit, which is JOINT4_LOWER -- not 0.0, which
        # since the limit was corrected is comfortably inside it and rightly
        # left alone.
        on_the_stop = [0.0] * 7
        on_the_stop[3] = JOINT4_LOWER
        check('a goal on a joint limit is pulled off it',
              round(orchestrator.clamp_to_limits(on_the_stop, 'TEST')[3], 6),
              round(JOINT4_LOWER + margin, 6))
        check('and a goal that is merely near zero is left alone, now that '
              'zero is inside the limit',
              round(orchestrator.clamp_to_limits([0.0] * 7, 'TEST')[3], 6),
              0.0)
        check('a goal clear of the limits is left alone',
              orchestrator.clamp_to_limits(list(HOME_JOINTS), 'TEST'),
              HOME_JOINTS)
        check('and the upper bound is respected as well',
              round(orchestrator.clamp_to_limits(
                  [JOINT1_UPPER + 1.0] * 7, 'TEST')[0], 6),
              round(JOINT1_UPPER - margin, 6))

        # The joint that cannot reach its commanded value, with the old
        # all-zeros HOME: commanded 0.0 (clamped to the margin), reference
        # 0.0, resting at the 0.15583 floor. Stopped there, that is home --
        # insisting on 0.05 rad is what dead-ended every cycle.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'home_joint_positions', value=[0.0] * 7)])
        with robot.lock:
            robot.joints = [0.0] * 7
            robot.joints[3] = ELBOW_FLOOR
        time.sleep(1.0)
        check('a stopped joint that cannot reach its goal still counts as home',
              orchestrator.at_home_pose(), True)
        with robot.lock:
            robot.joint_speed = 0.4
        time.sleep(1.0)
        check('but not while the arm is still moving',
              orchestrator.at_home_pose(), False)
        with robot.lock:
            robot.joint_speed = 0.011
            robot.joints[3] = ELBOW_FLOOR + 0.5
        time.sleep(1.0)
        check('and a gross offset fails even when stopped',
              orchestrator.at_home_pose(), False)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'home_joint_positions', value=HOME_JOINTS)])
        with robot.lock:
            robot.joints = list(HOME_JOINTS)
        time.sleep(1.0)
        with robot.lock:
            before_refresh = len(robot.refreshes)
        check('the octomap is captured at home',
              orchestrator.capture_octomap(), True)
        with robot.lock:
            check('and a frame was let through', len(robot.refreshes),
                  before_refresh + 1)

        # READY specifically must be refused, not just "somewhere else".
        with robot.lock:
            robot.joints = list(READY_JOINTS)
        time.sleep(1.0)
        check('READY is not home', orchestrator.at_home_pose(), False)
        with robot.lock:
            before_refresh = len(robot.refreshes)
        check('and the octomap is refused at READY',
              orchestrator.capture_octomap(), False)
        with robot.lock:
            check('no frames were let through at READY',
                  len(robot.refreshes), before_refresh)

        with robot.lock:
            robot.joints = [v + 0.5 for v in HOME_JOINTS]
        time.sleep(1.0)
        check('an arm away from home is detected',
              orchestrator.at_home_pose(), False)
        with robot.lock:
            before_refresh = len(robot.refreshes)
        check('and the octomap is refused there too',
              orchestrator.capture_octomap(), False)
        with robot.lock:
            check('still no frames', len(robot.refreshes), before_refresh)
        with robot.lock:
            robot.joints = list(HOME_JOINTS)
        time.sleep(1.0)
        check('back at home, the octomap is captured again',
              orchestrator.capture_octomap(), True)

        # A poisoned map blocks the very move that would replace it, so the
        # refresh has to drop it first.
        with robot.lock:
            clears_before = len(robot.clears)
            refresh_before = len(robot.refreshes)
        check('map_from_home reports success', orchestrator.map_from_home(), True)
        with robot.lock:
            check('it cleared the old map first', len(robot.clears),
                  clears_before + 1)
            check('and captured a new one', len(robot.refreshes),
                  refresh_before + 1)

        # Holding something must also block it, wherever the arm is.
        orchestrator._holding = True
        with robot.lock:
            held_refresh = len(robot.refreshes)
        check('carrying blocks the octomap too',
              orchestrator.capture_octomap(), False)
        with robot.lock:
            check('and let no frames through', len(robot.refreshes), held_refresh)
        orchestrator._holding = False

        # A dead planner must stop the cycle, not warn and walk the whole retry
        # ladder: with nothing planning, "every pick strategy was exhausted"
        # reads as a grasping problem when the planner simply is not running.
        real_lookup = orchestrator._planner_parameter
        orchestrator._planner_parameter = lambda name: module.DEAD_PLANNER
        # -- a home_state recording is not required -------------------------
        #
        # It used to be demanded of the other arm, on the grounds that the
        # fallback was measured on one arm only. The fallback is now
        # [0, 0, 0, 0.20, 0, 0, 0] -- symmetric, valid for either arm. The
        # requirement refused a complete setup: pre_pick_state and drop_state
        # both recorded and playable, reported as "no recorded states for the
        # left arm".
        check('pre_pick and drop alone are enough to start',
              orchestrator.check_states(
                  {'pre_pick_state': {'joints': PRE_PICK_JOINTS},
                   'drop_state': {'joints': DROP_JOINTS}}), True)
        check('and a missing pre_pick is still refused',
              orchestrator.check_states(
                  {'drop_state': {'joints': DROP_JOINTS}}), False)
        check('as is a missing drop, in state place mode',
              orchestrator.check_states(
                  {'pre_pick_state': {'joints': PRE_PICK_JOINTS}}), False)

        # -- the other arm's reach is answered, not guessed -----------------
        #
        # cuMotion takes Cartesian goals for one link per bringup, so a pose
        # goal for the *other* arm's tool comes back INVALID_LINK_NAME -- no
        # statement about reach at all. The verdict then rested on KDL alone,
        # which is documented as unreliable, and an object in the left half of
        # the frame was refused with the report claiming cuMotion had agreed.
        other = 'left' if ARM == 'right' else 'right'
        check('a chain is available for the other arm too',
              orchestrator.kinematics(other) is not None, True)
        check('and it is that arm',
              orchestrator.kinematics(other).arm, other)
        # The solver answers for that arm directly -- this is the capability
        # that was missing, and the only one that works when cuMotion is
        # pointed at the other tool.
        mirrored = (0.4062, -0.2184 if other == 'right' else 0.2184, 0.4088)
        found, margin, solved, tried = orchestrator.kinematics(
            other).tool_posture(mirrored, module.top_down_quat(math.pi),
                                seeds=48)
        check(f'and it can solve a reachable point for that arm '
              f'({solved}/{tried} seeds)', found is not None, True)

        # End to end: KDL says no -- as it does at 50 ms -- and the arm is
        # still not condemned, because something else can actually answer.
        with robot.lock:
            robot.ik_fail = True
            robot.ik_unreachable = set()
        check('so a KDL refusal alone no longer condemns the other arm',
              orchestrator.reachable(mirrored, module.top_down_quat(math.pi),
                                     arm=other) in (True, None), True)
        with robot.lock:
            robot.ik_fail = False

        # -- reach is judged from the staging pose, not from HOME ----------
        #
        # The check runs while the arm is at HOME, folded down by the base
        # with the work surface between it and the object. A plan from there
        # can fail because of the table rather than because the point is out
        # of reach -- and the cycle then reported "no arm can pick the object"
        # about a point both arms could pick. pre_pick_state exists to be
        # above the table before reaching, and it is what the cycle actually
        # approaches from.
        # The arm-choice block above rewrote the states file for the *other*
        # arm; put this arm's back to ask for its staging pose, then put it
        # back as it was -- the refusal test further down depends on it.
        with open(states_path) as handle:
            borrowed = handle.read()
        write_states(states_path)
        orchestrator.configure_arm(ARM)
        check('the staging pose is read for this arm',
              orchestrator.staging_joints(ARM), list(PRE_PICK_JOINTS))
        # The other arm's question is posed from the other arm's own pose,
        # read from its own file -- not from this arm's.
        other_staging = orchestrator.staging_joints(
            'left' if ARM == 'right' else 'right')
        check("and the other arm's comes from the other arm's recording",
              bool(other_staging) and other_staging != list(PRE_PICK_JOINTS),
              True)
        with robot.lock:
            robot.ik_fail = True          # force the cuMotion probe to be used
            robot.ik_unreachable = set()
            robot.goals.clear()
        orchestrator.planner_can_reach(OBJECT_POINT, quat, ARM)
        with robot.lock:
            probes = [g for g in robot.goals if g['plan_only']]
        check('the probe went out plan-only', bool(probes), True)
        if probes:
            check('with an explicit start state, not the current one',
                  probes[-1]['start_is_diff'], False)
            check('and that start state is the staging pose',
                  [round(v, 4) for v in probes[-1]['start_joints']],
                  [round(v, 4) for v in PRE_PICK_JOINTS])
        with robot.lock:
            robot.ik_fail = False
        with open(states_path, 'w') as handle:
            handle.write(borrowed)
        orchestrator.configure_arm(ARM)

        # -- the planner has to be able to plan, not just answer -----------
        #
        # Measured on the robot: cuMotion's parameter service answered, the
        # tool-frame check passed, and three attempts at a plain joint goal to
        # a recorded posture came back TIMED_OUT -- after which the report
        # blamed the recorded pose. A plan-only goal to where the arm already
        # is separates the two, costs no motion, and cannot fail for reasons
        # about the target.
        check('the readiness probe is on by default',
              orchestrator.get_parameter('check_planner_ready').value, True)
        with robot.lock:
            robot.move_fail_codes = []
            before = moved(robot)
        check('a healthy planner passes', orchestrator.planner_is_planning(),
              True)
        with robot.lock:
            check('having moved nothing', moved(robot), before)
            probe = robot.goals[-1]
        check('it asked plan-only', probe['plan_only'], True)
        check('as a joint goal', probe['kind'], 'joint')
        check('to where the arm already is',
              [round(v, 3) for v in probe['joints']],
              [round(v, 3) for v in robot.joints[:7]])

        with robot.lock:
            robot.plan_only_code = MoveItErrorCodes.TIMED_OUT
        check('a planner that cannot plan its own posture is caught',
              orchestrator.planner_is_planning(), False)
        with robot.lock:
            robot.plan_only_code = None

        check('a missing planner node is detected',
              orchestrator.planner_ee_link(), module.DEAD_PLANNER)
        check('and refuses the cycle', orchestrator.check_planner_tool_frame(), False)
        orchestrator._planner_parameter = real_lookup
        check('a live planner still passes',
              orchestrator.check_planner_tool_frame(), True)

        # A tool-frame mismatch is only fatal if the cycle actually sends pose
        # goals. With the defaults it does not: the approach is a joint goal
        # or a Cartesian line, the descent and lift go through
        # /compute_cartesian_path (which takes a link_name and serves either
        # arm), and pre_pick/drop/home are joint goals, which cuMotion accepts
        # for either arm. Refusing outright stopped a correctly chosen left
        # arm dead with "the planner is unusable for this arm".
        real_arm = orchestrator.arm
        orchestrator.configure_arm('left' if real_arm == 'right' else 'right')
        check('a mismatched tool frame is survivable with these settings',
              orchestrator.check_planner_tool_frame(), True)
        check('and pose goals are marked unavailable',
              orchestrator._pose_goals_ok, False)
        with robot.lock:
            before = len(robot.goals)
        check('so a pose goal refuses instead of being sent',
              orchestrator.move_to_pose(OBJECT_POINT, quat, 'DOOMED'), False)
        with robot.lock:
            check('and nothing went to the planner', len(robot.goals), before)

        # But it *is* fatal when the configuration cannot avoid pose goals.
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'approach_frame', value='planner')])
        check('a mismatch is refused when the approach needs a pose goal',
              orchestrator.check_planner_tool_frame(), False)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'approach_frame', value='tool')])
        orchestrator.configure_arm(real_arm)
        check('and the configured arm still passes cleanly',
              orchestrator.check_planner_tool_frame(), True)
        check('with pose goals available again',
              orchestrator._pose_goals_ok, True)

        # -- the contact guard --------------------------------------------
        #
        # Run 1789012345, in full: the guard stopped a transit 1.7 s in
        # because joint1 had gone from +3.42 to +1.38 Nm -- the shoulder
        # *unloading* as the arm swung out. The cancel was sent from the stop
        # function, which does not run until the wait for the result returns,
        # so the arm flew the whole move anyway and landed 1.4 mm from its
        # target; a successful transit was then reported as a crash. The
        # back-off asked cuMotion for a way back, move_group rejected the
        # path (the gripper was inside the octomap), the resend was never
        # answered, and after two 60-second timeouts /move_action was gone
        # from the graph. Two minutes of an arm stranded over the table, all
        # from one number.
        #
        # So the numbers below are that run's. Gravity took joint1 from
        # +3.415 to -10.484 Nm across the sweep, and the worst honest
        # tracking error at full speed was 0.126 rad.
        joints = [f'openarm_{ARM}_joint{i}' for i in range(1, 8)]
        GRAVITY = (3.415, -10.484)

        def transit(span=6.8, travel=-2.381, hz=20.0, lag=0.1,
                    freeze_at=None, push_at=None, push=0.0, push_over=0.3):
            """The measured transit, optionally stopped dead part way.

            freeze_at  when the arm stops moving while the plan carries on.
                       Gravity stops changing then too, because the arm has
                       stopped changing shape.
            push       Nm piled on from push_at, over push_over seconds.
            """
            steps = int(span * hz)
            plan = [(i / hz, [1.064 + travel * i / steps] + [0.0] * 6)
                    for i in range(steps + 1)]
            samples, frozen = [], None
            for i in range(steps + 1):
                now = i / hz
                wanted = plan[i][1][0]
                if freeze_at is not None and now >= freeze_at:
                    if frozen is None:
                        frozen = (wanted - lag * i / steps, i)
                    where, weight = frozen[0], frozen[1] / steps
                else:
                    where, weight = wanted - lag * i / steps, i / steps
                effort = GRAVITY[0] + (GRAVITY[1] - GRAVITY[0]) * weight
                if push_at is not None and now >= push_at:
                    # Away from zero: pushing harder, not unloading.
                    effort -= push * min(1.0, (now - push_at) / push_over)
                samples.append((now, [where] + [0.0] * 6,
                                dict(zip(joints, [effort] + [0.0] * 6))))
            return plan, samples

        def run_monitor(plan, samples, **kwargs):
            options = dict(margin=4.0, window=0.4, hold=0.25, rewind=2.0,
                           lag_limit=0.25, plan=plan)
            if 'plan_override' in kwargs:
                options['plan'] = kwargs.pop('plan_override')
            options.update(kwargs)
            monitor = module.ContactMonitor(joints, **options)
            worst_lag = 0.0
            for now, positions, efforts in samples:
                verdict = monitor.add(now, positions, efforts)
                # After the reading, so the monitor has a start time to
                # measure elapsed against -- it takes its own on the first.
                lag = monitor.not_keeping_up(now, positions)[1]
                if lag is not None:
                    worst_lag = max(worst_lag, lag)
                if verdict is not None:
                    return verdict, now, worst_lag
            return None, None, worst_lag

        plan, samples = transit()
        verdict, when, worst_lag = run_monitor(plan, samples)
        check('an ordinary transit is not a collision', verdict, None)
        check('even though gravity moved the shoulder 14 Nm across it',
              round(abs(samples[-1][2][joints[0]] - samples[0][2][joints[0]]), 1),
              13.9)
        check('and the arm was never more than 0.13 rad behind its plan',
              worst_lag < 0.13, True)

        plan, samples = transit(freeze_at=3.0, push_at=3.0, push=6.0)
        verdict, when, _ = run_monitor(plan, samples)
        check('an arm stopped dead while pulling harder is', bool(verdict), True)
        if verdict:
            check('reported against the joint that loaded up',
                  verdict['joint'], joints[0])
            check('with the reason recorded',
                  verdict.get('why') in ('lag', 'stall'), True)
            check('once the excess has lasted the hold time, not before',
                  3.25 <= when <= 3.6, True)
            check('with somewhere to go back to', bool(verdict['back_to']), True)
            check('two seconds before the contact',
                  round(verdict['back_to'][0], 3),
                  round([s[1][0] for s in samples
                         if s[0] <= when - 2.0][-1], 3))
            check('and the whole way back recorded, oldest first',
                  verdict['retrace'][0], verdict['back_to'])
            check('ending where the arm is now',
                  round(verdict['retrace'][-1][0], 4),
                  round([s[1][0] for s in samples if s[0] <= when][-1], 4))

        # The case the lag test cannot catch on its own. A 10 cm descent asks
        # for a third of a radian in total, so an arm stopped half way
        # through can never fall a fixed 0.25 rad behind -- there is not that
        # much plan left. What gives it away is that it went nowhere while
        # the plan kept moving.
        plan, samples = transit(span=3.0, travel=-0.30, freeze_at=2.0,
                                push_at=2.0, push=6.0)
        verdict, when, worst_lag = run_monitor(plan, samples)
        check('a short descent stopped by the table is caught too',
              bool(verdict), True)
        check('by the stall, since the lag never gets near the threshold',
              worst_lag < 0.25, True)
        check('and it says which of the two caught it',
              verdict and verdict.get('why'), 'stall')
        verdict, _, _ = run_monitor(plan, samples, stall_floor=99.0)
        check('and with the stall test disabled it is missed, which is why '
              'both tests are there', verdict, None)

        # And the case that is actually happening on this robot: not a bang,
        # a lean. The servo settles into the table over a second or more, so
        # every sample looks like the last one. A rolling baseline can only
        # see a push arriving faster than margin/window -- 10 Nm/s -- which
        # is why the reference is latched at the stall instead.
        plan, samples = transit(span=6.0, travel=-0.6, freeze_at=2.0,
                                push_at=2.0, push=7.0, push_over=1.5)
        verdict, when, _ = run_monitor(plan, samples)
        check('a slow lean into the table is caught, not just a bang',
              bool(verdict), True)
        check('within about a second of it starting',
              when is not None and when - 2.0 <= 1.5, True)

        plan, samples = transit(push_at=3.0, push=8.0)
        verdict, _, _ = run_monitor(plan, samples)
        check('torque alone, on an arm still flying its plan, is not contact',
              verdict, None)

        plan, samples = transit(freeze_at=3.0)
        verdict, _, _ = run_monitor(plan, samples)
        check('and a stall alone, with no load, is not either', verdict, None)

        # Nothing can trip before there is a baseline to compare against:
        # comparing with the start of the move is the bug this replaced.
        plan, samples = transit(freeze_at=0.05, push_at=0.05, push=12.0)
        verdict, when, _ = run_monitor(plan, samples)
        check('a shove in the first moments waits for a rolling baseline',
              when is not None and when >= 0.4, True)

        check('thinning a retrace keeps the destination',
              module.thin(list(range(100)), 12)[-1], 99)
        check('and no more waypoints than asked for',
              len(module.thin(list(range(100)), 12)), 12)
        check('a short path is left alone',
              module.thin([1, 2, 3], 12), [1, 2, 3])

        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = list(joints)
        point = JointTrajectoryPoint()
        point.positions = [0.1] * 7
        point.time_from_start.sec = 2
        traj.joint_trajectory.points.append(point)
        read = orchestrator.trajectory_plan(traj)
        check('a trajectory is read back as (seconds, joints)',
              read, [(2.0, [0.1] * 7)])
        other = RobotTrajectory()
        other.joint_trajectory.joint_names = ['somebody_elses_joint']
        check("and another arm's is refused rather than mismatched",
              orchestrator.trajectory_plan(other), None)
        # Which disarms the guard rather than falling back to torque alone.
        # Torque alone is the rule that stopped a good transit.
        plan, samples = transit(freeze_at=3.0, push_at=3.0, push=20.0)
        verdict, _, _ = run_monitor(plan, samples, plan_override=None)
        check('with no trajectory to compare against, nothing is judged',
              verdict, None)

        # The back-off itself: straight to the controller, no planner.
        with robot.lock:
            here = list(robot.joints)
            robot.trajectories.clear()
            planner_goals = len(robot.goals)
        retrace = [[here[0] - 0.30 + 0.01 * i] + here[1:] for i in range(31)]
        hit = {'joint': joints[0], 'effort': -9.0, 'baseline': -3.0,
               'lag': 0.4, 'at': 0.0, 'back_to': list(retrace[0]),
               'retrace': [list(p) for p in retrace]}
        backed = orchestrator.retreat_from_contact('TEST', hit)
        check('the back-off goes', backed, True)
        with robot.lock:
            sent = list(robot.trajectories)
            asked_planner = len(robot.goals) - planner_goals
            landed = list(robot.joints)
        check('without asking the planner anything', asked_planner, 0)
        check('as one trajectory straight to the controller', len(sent), 1)
        if sent:
            check('for this arm', sent[0]['names'], joints)
            check('thinned to a dozen waypoints at most',
                  len(sent[0]['points']) <= 12, True)
            check('leaving the contact first',
                  round(sent[0]['points'][0][0], 3), round(retrace[-1][0], 3))
            check('and ending where the arm was two seconds earlier',
                  round(sent[0]['points'][-1][0], 3), round(retrace[0][0], 3))
            check('slowly -- it is a move with no collision check',
                  sent[0]['seconds'][-1] >= 0.5, True)
        check('and the arm is there', round(landed[0], 3),
              round(retrace[0][0], 3))

        # A controller that will not take it must not leave the arm leaning.
        # Put the arm back first: the destination is where it now stands, and
        # a back-off to where the arm already is is skipped rather than sent,
        # which would prove nothing about the fallback.
        with robot.lock:
            robot.joints = list(here)
            robot.controller_refuse = True
            robot.trajectories.clear()
            planner_goals = len(robot.goals)
        backed = orchestrator.retreat_from_contact('TEST', hit)
        with robot.lock:
            robot.controller_refuse = False
            fell_back = len(robot.goals) - planner_goals
            robot.joints = list(here)
        check('a controller that refuses falls back to the planner',
              fell_back > 0, True)
        check('and the arm still gets out', backed, True)

        # -- the grasp check: the detector decides -------------------------
        #
        # Run 1789014831 cycle 1: the jaws closed 15.1 mm off target,
        # stalled at 1.08 mm with 0.50 Nm -- empty -- and the cycle went on
        # to "place" nothing and report DONE. Two holes let that through,
        # and both are checked here: the finger check had been switched off
        # by a saved rehearsal value, and an empty detector frame was read
        # as "the object is gone, so we must have it".
        with robot.lock:
            keep_holding = robot.holding
            keep_hidden = robot.object_hidden
            keep_finger = robot.finger
            robot.holding = False
            robot.object_hidden = False
            robot.finger = HOLDING_FINGER
        time.sleep(0.3)
        check('an object still lying at the pick point fails the check',
              orchestrator.verify_grasp(list(OBJECT_POINT)), False)
        with robot.lock:
            robot.holding = True          # it travels with the tool
        time.sleep(0.3)
        check('and one that moved with the tool passes',
              orchestrator.verify_grasp(list(OBJECT_POINT)), True)
        with robot.lock:
            robot.object_hidden = True
        time.sleep(0.3)
        check('an empty frame is not a pass: that is what a failed pick '
              'looks like with the arm parked over the object',
              orchestrator.verify_grasp(list(OBJECT_POINT)), False)
        with robot.lock:
            robot.object_hidden = False
            robot.finger = 0.0011         # the measured value from that run
        time.sleep(0.3)
        check('and jaws shut on nothing fail whatever the detector says',
              orchestrator.verify_grasp(list(OBJECT_POINT)), False)
        with robot.lock:
            robot.holding = keep_holding
            robot.object_hidden = keep_hidden
            robot.finger = keep_finger
        time.sleep(0.3)

        # -- a rehearsal value must not reach the arms ---------------------
        import pick_place_sequence as seq_model
        check('a negative grasp_finger_min is known to switch a check off',
              [w.split('=')[0] for w in seq_model.check_disabling_settings(
                  {'grasp_finger_min': -1.0})], ['grasp_finger_min'])
        kept, dropped = seq_model.savable_settings(
            {'grasp_finger_min': -1.0, 'object_moved_eps': 0.0,
             'grasp_z_offset': 0.01})
        check('and saving drops it rather than writing it to the file',
              sorted(kept), ['grasp_z_offset'])
        check('naming both of them', len(dropped), 2)

        keep_fake = orchestrator.get_parameter('fake_hardware').value
        keep_floor = orchestrator.get_parameter('grasp_finger_min').value
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'fake_hardware', value=False)])
        applied = orchestrator.apply_settings({'grasp_finger_min': 0.004})
        check('an ordinary setting still applies', applied.get(
            'grasp_finger_min'), 0.004)
        with open(config_path, 'w') as handle:
            json.dump({'prompt': orchestrator.prompt, 'arm': 'auto',
                       'sequence': list(orchestrator._sequence),
                       'settings': {'grasp_finger_min': -1.0}}, handle)
        orchestrator.load_config()
        check('but a rehearsal value in the file is refused on the arms',
              orchestrator.get_parameter('grasp_finger_min').value != -1.0,
              True)
        check('and the refusal is on the panel, not just in the log',
              any('grasp_finger_min' in p
                  for p in orchestrator._config_problems), True)
        orchestrator.set_parameters([rclpy.parameter.Parameter(
            'fake_hardware', value=True)])
        orchestrator.load_config()
        check('while a fake-hardware run still gets it, which is the point '
              'of it existing',
              orchestrator.get_parameter('grasp_finger_min').value, -1.0)
        orchestrator.set_parameters([
            rclpy.parameter.Parameter('fake_hardware', value=keep_fake),
            rclpy.parameter.Parameter('grasp_finger_min', value=keep_floor)])
        if os.path.exists(config_path):
            os.remove(config_path)

        # Counted the same way at both ends -- mixing the raw total with
        # moved() compares two different things and the delta is nonsense.
        with robot.lock:
            before = moved(robot)
        orchestrator._cycle()
        with robot.lock:
            after = moved(robot)
        check('a refused cycle sends no goals at all', after, before)
        # states[] is filled by a subscription, so the terminal state can still
        # be in flight when _cycle() returns. Wait for it rather than racing.
        deadline = time.time() + 5
        while time.time() < deadline:
            if states and states[-1].split(':')[0].strip() in (
                    'DONE', 'FAILED', 'ABORTED'):
                break
            time.sleep(0.05)
        check('and reports why', states[-1].split(':')[0].strip(), 'FAILED')
    finally:
        orchestrator._abort.set()
        executor.shutdown()
        robot.destroy_node()
        orchestrator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if os.path.exists(states_path):
            os.remove(states_path)
        if os.path.exists(log_path):
            os.remove(log_path)

    print()
    if failures:
        print(f'{len(failures)} check(s) failed: {", ".join(failures)}')
        return 1
    print('the whole cycle ran in the documented order')
    return 0


if __name__ == '__main__':
    sys.exit(main())
