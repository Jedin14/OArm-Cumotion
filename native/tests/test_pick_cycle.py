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

import importlib.util
import json
import os
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
# robot. Set PICK_TEST_DOMAIN if 77 clashes with something.
os.environ['ROS_DOMAIN_ID'] = os.environ.get('PICK_TEST_DOMAIN', '77')
os.environ['ROS_LOCALHOST_ONLY'] = '1'

import rclpy                                                     # noqa: E402
import yaml                                                      # noqa: E402
from control_msgs.action import GripperCommand                   # noqa: E402
from geometry_msgs.msg import TransformStamped                    # noqa: E402
from moveit_msgs.action import ExecuteTrajectory, MoveGroup       # noqa: E402
from moveit_msgs.msg import MoveItErrorCodes                      # noqa: E402
from moveit_msgs.srv import (                                     # noqa: E402
    ApplyPlanningScene, GetCartesianPath, GetPositionIK)
from moveit_msgs.msg import RobotTrajectory                       # noqa: E402
from trajectory_msgs.msg import (                                 # noqa: E402
    JointTrajectoryPoint)
from rcl_interfaces.msg import ParameterType, ParameterValue      # noqa: E402
from rcl_interfaces.srv import GetParameters                      # noqa: E402
from rclpy.action import ActionServer                             # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup          # noqa: E402
from rclpy.executors import MultiThreadedExecutor                 # noqa: E402
from rclpy.node import Node                                       # noqa: E402
from sensor_msgs.msg import JointState                            # noqa: E402
from std_msgs.msg import String                                   # noqa: E402
from std_srvs.srv import Empty, Trigger                           # noqa: E402
from tf2_ros import TransformBroadcaster                          # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

ARM = 'right'
OBJECT_POINT = [0.35, -0.18, 0.05]
OBJECT_YAW = 0.4
APPROACH_HEIGHT = 0.05
GRASP_Z_OFFSET = -0.005
PRE_PICK_JOINTS = [-0.5, 0.1, -0.2, 1.9, 0.05, -0.3, 0.7]
DROP_JOINTS = [0.9, 0.2, -0.1, 1.5, 0.0, 0.2, -0.4]
READY_JOINTS = [-0.828374, 0.000191, -0.000191, 2.324140,
                -0.000191, -0.000191, -0.391966]
# The SRDF "home" group state, and the only pose the octomap may be captured
# from. Deliberately not READY: READY holds the arm out over the table so the
# camera can see it, so a map captured there contains the arm.
HOME_JOINTS = [0.0] * 7

# The detector reports the frame size so the orchestrator can say which half
# of the camera view a detection is in; that is how an arm gets chosen.
IMAGE_SIZE = (848, 480)
# Right of centre, so the default right arm is the one the split picks.
OBJECT_PX = (600, 260)

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
    if isinstance(want, (list, tuple)):
        if want and is_numeric(want):
            ok = (len(got) == len(want)
                  and all(abs(a - b) <= tol for a, b in zip(got, want)))
        else:
            ok = list(got) == list(want)
    elif isinstance(want, float):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
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
        self.finger = OPEN_FINGER
        self.torque = 0.0
        self.holding = False
        self.joints = list(READY_JOINTS)
        self.lock = threading.Lock()

        self.tf = TransformBroadcaster(self)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
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
        self.joint_pub.publish(js)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'world'
        t.child_frame_id = f'openarm_{ARM}_hand_tcp'
        t.transform.translation.x = 0.3
        t.transform.translation.y = -0.2
        t.transform.translation.z = 0.3
        t.transform.rotation.w = 1.0
        self.tf.sendTransform(t)

        # Once the gripper has closed on it, the object is no longer lying at
        # the pick point -- which is exactly what verify_grasp checks for.
        with self.lock:
            holding = self.holding
        payload = {
            'stamp': now.nanoseconds * 1e-9,
            'prompt': self.prompt,
            'image_size': list(IMAGE_SIZE),
            'detections': [] if holding else [{
                'point': list(OBJECT_POINT),
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
                 'plan_only': bool(options.plan_only)}
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
                code = (MoveItErrorCodes.NO_IK_SOLUTION
                        if arm in self.ik_unreachable
                        else MoveItErrorCodes.SUCCESS)
            else:
                code = (self.move_fail_codes.pop(0) if self.move_fail_codes
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
            if position < OBJECT_HALF_WIDTH:
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

    def _on_scene(self, _request, response):
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
            # map was captured.
            self.refreshes.append(len(self.goals))
        response.success = True
        return response

    def _on_clear_octomap(self, _request, response):
        with self.lock:
            self.clears.append(len(self.goals))
        return response

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
        for index, positions in enumerate((start, [v - 0.01 for v in start])):
            point = JointTrajectoryPoint()
            point.positions = list(positions)
            point.velocities = [1.0] * len(names)
            point.accelerations = [2.0] * len(names)
            point.time_from_start.sec = index
            traj.joint_trajectory.points.append(point)
        response.solution = traj
        response.fraction = fraction
        response.error_code.val = MoveItErrorCodes.SUCCESS
        return response

    def _on_execute(self, goal_handle):
        traj = goal_handle.request.trajectory
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
    states_path = os.path.join(WS, 'native', 'tests', '.test_states.yaml')
    write_states(states_path)
    # Its own log file: a test has no business appending to the one the robot
    # writes, and a stale one would make the checks below pass on old data.
    log_path = os.path.join(WS, 'native', 'tests', '.test_motion_log.jsonl')
    if os.path.exists(log_path):
        os.remove(log_path)

    rclpy.init(args=[
        '--ros-args',
        '-p', f'arm:={ARM}',
        '-p', f'states_file:={states_path}',
        '-p', 'place_mode:=state',
        '-p', f'approach_height:={APPROACH_HEIGHT}',
        '-p', f'grasp_z_offset:={GRASP_Z_OFFSET}',
        '-p', 'gripper_settle_time:=0.05',
        '-p', 'detect_timeout:=8.0',
        '-p', f'motion_log:={log_path}',
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
        # transit, above the object, open, down, grab, back above, drop, open,
        # pre-pick, home. HOME is the only rest/observation pose -- there is no
        # separate READY any more.
        wanted = ['HOME', 'LOCATE', 'PRE_PICK', 'TRANSIT', 'PREGRASP',
                  'OPEN_GRIPPER', 'DESCEND', 'CLOSE_GRIPPER', 'LIFT',
                  'VERIFY_GRASP', 'DROP', 'RELEASE', 'VERIFY_PLACE',
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
        check('the reach of the object was checked before moving',
              len(reach_checks) > 0, True)

        # Two joint goals in (home, pre_pick) and three out (drop, pre_pick,
        # home). Only the transit is a pose goal; everything below it is a
        # Cartesian path executed as a trajectory, so it does not appear as a
        # MoveGroup goal at all.
        check('joint goals bracket the middle',
              (kinds[:2], kinds[-3:]),
              (['joint', 'joint'], ['joint', 'joint', 'joint']))
        check('the transit is the only pose goal', len(poses), 1)
        check('nothing else goes as a MoveGroup goal', kinds[3:-3], [])

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
        check('collision-checked', sorted({r['avoid_collisions'] for r in lines}),
              [True])
        step = orchestrator.get_parameter('cartesian_step').value
        check('interpolated finely enough to be straight',
              max(r['max_step'] for r in lines) <= step + 1e-12, True)
        check('every line ends on the vertical above the object',
              sorted({(round(r['xyz'][0], 9), round(r['xyz'][1], 9))
                      for r in lines}),
              [(round(OBJECT_POINT[0], 9), round(OBJECT_POINT[1], 9))])

        zs = [r['xyz'][2] for r in lines]
        check('one line reaches the pre-grasp height exactly',
              min(abs(z - above) for z in zs) < 1e-9, True)
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
        check('the column asks collision-checked first',
              checked[0], True)
        # Three legs get a straight line: down to the pre-grasp, down onto the
        # object, and back up to retreat_height. All three, not just the last
        # 5 cm -- the 15 cm leg was the one that went curved.
        check('all three legs of the column ask for a line', len(lines), 3)
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
        check('the gripper opened first', round(grips[0], 3), 0.044)
        check('and opened again to release', round(grips[-1], 3), 0.044)
        check('the close stepped rather than slamming shut', len(grips) > 3, True)
        check('and stopped at the cap instead of fully closing',
              min(grips) > 0.0, True)
        check('grip torque cap is the 2.5 Nm default',
              orchestrator.get_parameter('gripper_torque_cap').value, 2.5)
        check('motion is time-scaled faster than the timid default',
              orchestrator.get_parameter('velocity_scaling').value, 0.3)

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
            robot.checked_fraction = 0.25      # checked stalls, unchecked fine
            robot.cartesian_requests.clear()
            before = moved(robot)
        check('the descent still completes',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'BLOCKED',
                                          uncheck_collisions=True), True)
        with robot.lock:
            asked = list(robot.cartesian_requests)
            goals_after = moved(robot)
        check('it asked for a checked line first, then an unchecked one',
              [r['avoid_collisions'] for r in asked], [True, False])
        check('and sent no free-space goals at all', goals_after, before)

        # Without the exemption it must not silently ignore the octomap.
        with robot.lock:
            robot.cartesian_requests.clear()
            before = moved(robot)
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

        # Everything below here is about the fallbacks, so the straight line
        # has to be taken away first -- otherwise it succeeds and the fallback
        # never runs.
        #
        # A partial line must be refused, not executed: a descent that stops
        # at 60% leaves the gripper closing on air.
        with robot.lock:
            robot.ik_fail = False
            robot.cartesian_fraction = 0.6
            lines_before = len(robot.cartesian_requests)
            before = moved(robot)
        check('a partial straight line still completes the descent',
              orchestrator.descend_column(xy, 0.30, 0.25, quat, 'PARTIAL'),
              True)
        with robot.lock:
            check('the line was asked for',
                  len(robot.cartesian_requests) > lines_before, True)
            partial = robot.goals[before:]
        check('but not executed -- it fell back to goals instead',
              len(partial) > 0, True)
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
            fallback = robot.goals[before:]
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
            column_goals = robot.goals[before:]
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
        check('no arm reachable gives OUT_OF_REACH', outcome, module.OUT_OF_REACH)
        check('and nothing moved at all', after, before)
        check('the state says out of reach',
              states[-1].split(':')[0].strip(), 'OUT_OF_REACH')

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
        with robot.lock:
            robot.ik_unreachable = {ARM}
            before = moved(robot)
        outcome = orchestrator.choose_arm(
            {'pre_pick_state': {'joints': PRE_PICK_JOINTS},
             'drop_state': {'joints': DROP_JOINTS}})
        check('it falls back to the arm that can reach',
              orchestrator.arm, other)
        check('and reports states rather than refusing',
              outcome not in (None, module.OUT_OF_REACH), True)
        with robot.lock:
            check('still without moving',
                  moved(robot), before)
            robot.ik_unreachable = set()
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
              all({'time', 'state', 'arm', 'label', 'method', 'outcome',
                   'measured'} <= set(e) for e in entries), True)
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

        # Retries must not shuttle home for nothing. Only an attempt that needs
        # a fresh look at the object has to go back -- detection needs the
        # camera's view clear, and changing the wrist yaw does not.
        needs = {s['name']: s['redetect'] for s in module.STRATEGIES}
        check('the first attempt looks',
              needs['nominal'], True)
        check('so does the explicit redetect', needs['redetect'], True)
        check('and the remap, which goes home anyway',
              needs['remap-from-home'], True)
        check('but a yaw change does not',
              (needs['yaw+90'], needs['yaw+90-lower']), (False, False))
        check('nor does a lower grasp', needs['lower-8mm'], False)

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
        check('reach uses pure kinematics, not the collision world',
              sorted({r['avoid_collisions'] for r in reach_checks}), [False])
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
        time.sleep(1.0)
        check('at home per joint states', orchestrator.at_home_pose(), True)
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
        check('a missing planner node is detected',
              orchestrator.planner_ee_link(), module.DEAD_PLANNER)
        check('and refuses the cycle', orchestrator.check_planner_tool_frame(), False)
        orchestrator._planner_parameter = real_lookup
        check('a live planner still passes',
              orchestrator.check_planner_tool_frame(), True)

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
