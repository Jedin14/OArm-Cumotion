#!/usr/bin/env python3
"""VLM-guided pick and place for the 7DOF-OArm, with VLM-checked retries.

    HOME -> LOCATE -> PRE_PICK -> TRANSIT -> PREGRASP -> OPEN -> DESCEND
      ^                                                             |
      |                                                             v
      +---------------------- failed ------------------ VERIFY_GRASP <- CLOSE
                                                          |
      HOME <- PRE_PICK <- VERIFY_PLACE <- RELEASE <- DROP <+- LIFT

HOME is the only pose the arm rests, observes and maps from. There used to be a
separate READY as well, which held the arm out over the table so the camera
could see the work surface -- and that is exactly why the octomap kept being
captured with the arm inside it. One pose, out of the camera's frame, does the
job of both.

TRANSIT, PREGRASP, DESCEND and LIFT are all on one vertical line above the
object. The move up to TRANSIT is a free-space plan; everything below that
height is a straight Cartesian line from /compute_cartesian_path. That matters
because a goal pose says where to end up, not how to get there: a 5 cm descent
planned as a free trajectory bows away from the vertical, and a single move to
a point above the object can arrive from the side and low, sweeping the object
off the table before the gripper is over it.

The way out is the way in: LIFT, then DROP, then back through PRE_PICK to HOME.
PRE_PICK is a posture reachable from both ends, which is what makes it a safe
waypoint rather than a straight dash home across the workspace.

Three named postures, all joint-space goals:

  READY            ready_joint_positions, from the parameter. The observation
                   pose: the arm starts here and the object is located from
                   here, so it has to leave the camera a clear view.
  PRE_PICK         pre_pick_state, from the states file. Staging pose entered
                   after the object is located, so the approach to the object
                   starts from a known posture.
  DROP             drop_state, from the states file. Where the object is
                   released.

PRE_PICK and DROP are recorded with record_states.py rather than typed in; the
orchestrator re-reads that file at the start of every cycle, so re-recording a
pose takes effect without restarting anything. place_mode selects what DROP
means: "state" (the recorded drop_state), "ready" (the observation pose), or
"position" (a place_position in world coordinates).

Consumes /vlm/detections from VLM/vlm_detector_node.py (world-frame 3D points)
and drives the arm through MoveIt's /move_action with pipeline_id "cumotion".

Four things about this workspace decide how it talks to the robot:

* Arm motion goes through /move_action, not moveit_py -- there is no moveit_py
  in Humble. pipeline_id "cumotion" is the default pipeline registered by
  openarm_bimanual_moveit_config's demo.launch.py.

* Pose goals target openarm_<arm>_hand_tcp in `world`, because that is exactly
  what cuMotion is configured for: openarm.yml sets base_link "world" and
  ee_link "openarm_left_hand_tcp".

* The gripper is NOT commanded through MoveIt. cuMotion rejects 1-DOF grippers
  (see README), so grasping uses the controller's own GripperCommand action.

* That controller is position_controllers/GripperActionController with
  allow_stalling unset, i.e. false -- so closing onto an object ABORTS the
  action even though the grasp succeeded. The action result is therefore
  ignored, and success is judged from the finger position in /joint_states plus
  a VLM re-detection.

Verification is deliberately geometric rather than a question to the VLM: the
model here is paligemma-3b-pt-224, a pretrained checkpoint where "detect X" is
well-formed but yes/no VQA is not reliable. So instead of asking "is it in the
gripper?", it re-runs the same detection and checks where the object went:
still at the pick point means the grasp failed, near the tool frame means it
came along.

Run it via pick_place.launch.py, or standalone:

    source native/setup.bash
    python3 pick_place_orchestrator.py --ros-args \
        -p prompt:="detect screwdriver" -p arm:=right

Then start a cycle with:

    ros2 service call /pick_place/start std_srvs/srv/Trigger
"""

import math
import json
import os
import threading

import rclpy
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    MoveItErrorCodes,
    JointConstraint,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
)
import yaml
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger
from tf2_ros import Buffer, TransformListener

from vlm_prompt import to_detection_prompt

MOVEIT_SUCCESS = 1

# MoveIt codes that mean "ask again", as opposed to "this goal is wrong".
#
# cuMotion's trajectory optimiser is stochastic and does not converge every
# time: measured over 138 joint-goal queries on this robot, 20 came back
# MotionGenStatus.TRAJOPT_FAIL -- 14.5% -- on goals that succeeded on other
# tries. Those arrive here as PLANNING_FAILED, and the node reseeds between
# queries, so resending the identical goal usually works. In that sample the
# failures came in runs of one (16 times) and two (twice) and never three, so
# three attempts covered every one of them.
#
# Retrying matters most for a cycle: at 14.5% per goal, eight goals means only
# 0.855^8 = 29% of cycles would get through without a single transient failure.
# TIMED_OUT is included because a busy planner can miss the window; the codes
# that describe the goal itself -- in collision, bad link name, out of limits --
# are deliberately absent, because asking again cannot change them.
RETRYABLE_MOVEIT_CODES = (
    -1,   # PLANNING_FAILED           -- cuMotion TRAJOPT_FAIL lands here
    -2,   # INVALID_MOTION_PLAN
    -6,   # TIMED_OUT
)

# Distinguishes "the planner node is not in the graph" from "it did not answer",
# because only the first is worth refusing to start over.
DEAD_PLANNER = 'planner-not-running'
# _attempt_pick returns this instead of None when the object is simply not
# reachable. None means "this attempt failed, try the next strategy"; this
# means "no strategy will help", and the ladder stops.
OUT_OF_REACH = 'out-of-reach'

# Every state a cycle can end on. The panel imports this to decide when to
# re-enable its buttons: a terminal state it does not recognise leaves Pick
# greyed out with no way back, which is what adding OUT_OF_REACH did.
TERMINAL_STATES = ('DONE', 'FAILED', 'ABORTED', 'OUT_OF_REACH', 'IDLE')

ATTACHED_OBJECT_ID = 'vlm_target'
TABLE_OBJECT_ID = 'work_surface'

WS = os.path.dirname(os.path.realpath(__file__))
# 'auto' means pick_place_states_<arm>.yaml. The arms are mirrored, so a pose
# recorded on one is a different posture on the other and they cannot share a
# file; load_states() refuses a recording made for the other arm anyway.
DEFAULT_STATES_FILE = 'auto'
LEGACY_STATES_FILE = os.path.join(WS, 'pick_place_states.yaml')
HOME_STATE = 'home_state'
# Recordings made before HOME and READY were merged. The old ready_state was a
# second, separate observation pose, and having two was the whole problem: the
# octomap has to be captured with the arm out of the camera's frame, and the
# old READY deliberately held it out over the table -- in frame.
LEGACY_READY_STATE = 'ready_state'
PRE_PICK_STATE = 'pre_pick_state'
DROP_STATE = 'drop_state'

# Default READY pose, joint1..joint7 in radians: elbow up with the tool clear of
# the work area, so the camera sees the table unobstructed. Captured off the
# right arm while it was held in that pose, which is why the numbers are not
# round -- the RViz Joints tab showed -47, 0, 0, 133, 0, 0, -23 degrees, and
# these are the same pose to better than half a degree. Every value is inside
# the URDF limits (joint4 is 2.324 against an upper limit of 2.443).
# Jog to whatever pose you want and call /pick_place/capture_ready for its list.
READY_JOINT_POSITIONS = [
    -0.828374,
    0.000191,
    -0.000191,
    2.324140,
    -0.000191,
    -0.000191,
    -0.391966,
]

# Retry ladder. Each entry is a whole fresh attempt: re-detect, then apply these
# modifiers. Escalating beats repeating -- a failed grasp usually has a single
# cause, and the two most common ones here are minAreaRect's 90-degree axis
# ambiguity and depth read off the object's top surface sitting too high.
# `redetect` says whether an attempt needs a fresh look at the object, and that
# is what decides whether the arm travels back to HOME first -- detection needs
# the camera's view of the table unobstructed, and nothing else here does.
#
# Only three attempts need it. Changing the wrist yaw or dropping the grasp
# 8 mm changes the grasp, not the object, so those retry from wherever the arm
# already is: it was visibly shuttling pre_pick -> home -> pre_pick between
# attempts for nothing.
STRATEGIES = [
    {'name': 'nominal', 'yaw_offset': 0.0, 'z_offset': 0.0,
     'refresh_octomap': False, 'redetect': True},
    {'name': 'redetect', 'yaw_offset': 0.0, 'z_offset': 0.0,
     'refresh_octomap': False, 'redetect': True},
    {'name': 'yaw+90', 'yaw_offset': math.pi / 2, 'z_offset': 0.0,
     'refresh_octomap': False, 'redetect': False},
    {'name': 'lower-8mm', 'yaw_offset': 0.0, 'z_offset': -0.008,
     'refresh_octomap': False, 'redetect': False},
    {'name': 'yaw+90-lower', 'yaw_offset': math.pi / 2, 'z_offset': -0.008,
     'refresh_octomap': False, 'redetect': False},
    # Last resort: drop the map and rebuild it from home. Worth trying because
    # a map holding the arm or the target makes every plan fail identically,
    # and no change of yaw or height can get round it.
    {'name': 'remap-from-home', 'yaw_offset': 0.0, 'z_offset': 0.0,
     'refresh_octomap': True, 'redetect': True},
]


def quat_mul(a, b):
    """(x, y, z, w) * (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def other_arm(arm):
    return 'left' if arm == 'right' else 'right'


def top_down_quat(yaw):
    """Tool orientation with hand_tcp's +Z pointing straight down.

    hand_tcp sits 0.08 m along +Z of openarm_<arm>_hand (see openarm.urdf), so
    +Z is the approach axis and the fingers close along +/-Y (finger_joint1's
    axis is 0 -1 0 in the hand frame).

    R = Rz(yaw) * Ry(pi) puts the tool Z on world -Z and the finger closing
    direction at yaw + 90deg. Feeding the detected object axis in as `yaw`
    therefore closes the fingers across the object, which is what we want.
    """
    q_pitch = (0.0, 1.0, 0.0, 0.0)                    # pi about Y
    q_yaw = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    return quat_mul(q_yaw, q_pitch)


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class PickPlaceOrchestrator(Node):

    def __init__(self):
        super().__init__('pick_place_orchestrator')
        self.cb = ReentrantCallbackGroup()

        self.declare_parameter('arm', 'right')
        # by_side by default: with "fixed" the camera-half rule never runs and
        # the launch arm moves whatever half the object is in, which is not
        # what anyone means by a two-armed robot. "fixed" is still there for
        # single-arm work.
        self.declare_parameter('arm_selection', 'by_side')
        # Order the arms are considered in. The first that can reach the object
        # gets it.
        #
        #   camera_half     the half of the frame the object is in decides,
        #                   with the other arm as the fallback. The two orders
        #                   only differ when *both* arms can reach, and then
        #                   the near arm is the right answer -- a cross-body
        #                   reach is worse in every way.
        #   right_then_left / left_then_right
        #                   a fixed order, ignoring the frame.
        self.declare_parameter('arm_order', 'camera_half')
        self.declare_parameter('arm_split_y', 0.0)
        # Nudge for the camera-half split, in pixels, added to the middle
        # column. Positive widens the left arm's half. Only needed if the
        # camera is not centred on the robot.
        self.declare_parameter('arm_split_px', 0.0)
        self.declare_parameter('prompt', 'detect screwdriver')
        self.declare_parameter('pipeline_id', 'cumotion')
        self.declare_parameter('planning_time', 5.0)
        # cuMotion takes min(velocity, acceleration) scaling as a *time
        # dilation* of the trajectory it already optimised, so raising these
        # replays the same geometric path faster -- it costs tracking margin,
        # not accuracy. 0.15 was a deliberately timid first-hardware-run value.
        self.declare_parameter('velocity_scaling', 0.3)
        self.declare_parameter('acceleration_scaling', 0.3)
        self.declare_parameter('motion_timeout', 60.0)
        # cuMotion's optimiser misses roughly one goal in seven and succeeds on
        # a resend -- see RETRYABLE_MOVEIT_CODES. Three covered every transient
        # failure in a 138-query sample.
        self.declare_parameter('plan_attempts', 3)

        self.declare_parameter('ready_joint_positions', READY_JOINT_POSITIONS)
        # Which arm ready_joint_positions was measured on. The other arm needs
        # its own recorded ready_state, because the arms are mirrored.
        self.declare_parameter('ready_joint_positions_arm', 'right')
        self.declare_parameter('approach_height', 0.05)
        # Height above the grasp at which the long free-space move ends.
        #
        # Every leg here is a free-space plan, not a straight line: cuMotion is
        # given a goal pose and optimises a smooth trajectory to it. So a single
        # move from the staging pose to a point 5 cm over the object is free to
        # arrive from the side and low, which is how a gripper sweeps a
        # screwdriver off the table on its way to sitting above it.
        #
        # Ending that move well clear of the table instead, and only then
        # coming down the vertical line above the object, keeps the free part
        # of the motion away from anything on the surface. Set to 0 to go
        # straight to the pre-grasp, which is the old behaviour.
        self.declare_parameter('transit_height', 0.20)
        # Longest vertical hop allowed below transit_height. The waypoints are
        # collinear above the object, so short hops cannot bow far off that
        # line; one long hop can. 0 disables stepping.
        # Ask move_group for a straight Cartesian line for the legs above the
        # object, rather than a goal pose it may reach any way it likes.
        #
        # This is what stops the gripper taking a detour into the table on a
        # 5 cm descent. A goal pose says where to end up, not how to get
        # there, and cuMotion optimises for a smooth trajectory -- which over a
        # short drop can bow well away from the vertical.
        self.declare_parameter('linear_descent', True)
        # Interpolation step for that line, metres. Smaller is straighter.
        self.declare_parameter('cartesian_step', 0.005)
        # Reject a partial path. compute_cartesian_path returns the fraction it
        # managed; executing 60% of a descent stops the tool in mid-air and the
        # gripper then closes on nothing.
        self.declare_parameter('cartesian_min_fraction', 0.98)
        # Let the final approach and the retreat ignore the collision world
        # when a checked straight line cannot be had.
        #
        # Not a shortcut: on a top-down grasp the *target* is in the octomap.
        # The gripper has to enter the voxels of the object it is picking up,
        # and of the table beneath it, so a collision-checked descent onto an
        # object can never complete. Measured -- the 20 cm descent to the
        # pre-grasp solved 100% of its line, the last 5 cm onto the object
        # solved 25%.
        #
        # What keeps it safe is that the leg is short, straight, vertical,
        # between two points whose reach was already checked, with the gripper
        # open and min_grasp_z as a hard floor. The alternative is strictly
        # worse: a free-space plan for the same 5 cm, which is what drove the
        # gripper into the table.
        self.declare_parameter('approach_ignores_octomap', True)
        # How high the object is lifted before being carried anywhere. Defaults
        # to transit_height: the carry to the drop pose is a free-space plan,
        # and starting it 5 cm above the surface dragged the gripper across the
        # table. 0 means "back to the pre-grasp", the old behaviour.
        self.declare_parameter('retreat_height', 0.20)
        # Refuse the descent and the retreat outright when no straight line can
        # be had, rather than substituting free-space hops.
        #
        # The hops are not a milder version of the same motion. Measured: a
        # descent whose line solved 37.5% -- identically checked and unchecked,
        # so the arm runs out of reach along it -- became three free-space hops
        # that swung the tool 3.7 cm sideways and aborted with CONTROL_FAILED
        # against the table. Set false to allow the old behaviour.
        self.declare_parameter('descend_linear_only', True)
        # Append-only record of every motion, one JSON object per line.
        #
        # Written because the interesting failures are not reproducible on
        # demand and the interesting numbers are gone by the time anyone looks:
        # what the arm was asked for, which mechanism was used, what came back,
        # and where the joints and motor efforts actually were before and
        # after. Empty disables it.
        self.declare_parameter('motion_log', 'motion_log.jsonl')
        # How often to sample the arm during a move, seconds. 0 disables the
        # intermediate samples and keeps only the endpoints.
        self.declare_parameter('motion_sample_period', 0.1)
        # Most samples kept per motion. Beyond this the middle is thinned and
        # the ends preserved, so one slow descent cannot fill the file.
        self.declare_parameter('motion_sample_limit', 40)
        # Fallback stepping, when no straight line could be had. 0.02 rather
        # than 0.05 because the descent from the pre-grasp *is* 0.05: at 0.05
        # the span never exceeded the step, so it was never subdivided at all
        # and the whole descent went as one free-space goal.
        self.declare_parameter('descend_step', 0.02)
        # Descend by joint goals from seeded IK rather than by pose goals.
        #
        # This arm has 7 joints for a 6-DOF pose, so a tool pose does not pick
        # a posture -- there is a whole null space of solutions, and elbow-up
        # and elbow-flipped are both valid answers for "5 cm lower". A pose
        # goal lets the planner choose freely, and it will happily reconfigure
        # the entire arm to lower the tool 5 cm.
        #
        # So each waypoint's joints are solved with the previous waypoint's
        # joints as the IK seed, which keeps the solution in the same branch,
        # and the goal sent is a joint goal. The tool still tracks the vertical
        # line because the waypoints are collinear and close together.
        self.declare_parameter('seeded_descent', True)
        # Reject an IK solution that moves any joint more than this from the
        # seed: that is not "the same posture, lower", it is a reconfiguration.
        self.declare_parameter('max_joint_jump', 0.5)
        self.declare_parameter('ik_timeout', 1.0)
        # KDL is randomly seeded, so a failure is worth repeating before it is
        # believed. Its "yes" is conclusive; its "no" is not.
        self.declare_parameter('ik_attempts', 3)
        self.declare_parameter('grasp_z_offset', -0.005)
        self.declare_parameter('min_grasp_z', 0.01)
        # Bounds a detection has to satisfy before the arm is asked to move.
        #
        # A bad depth reading does not produce a slightly-off target, it
        # produces a wild one. Measured here: an object on the table came back
        # at [3.0799, 2.3206, -0.7416] with depth 2.982 m from 40 pixels --
        # nearly four metres away and three quarters of a metre below the
        # floor. min_grasp_z clamped the height and left x and y alone, so the
        # cycle then spent all six ladder attempts collecting IK_FAIL on a
        # point outside the room.
        #
        # Clamping is right for a centimetre of noise and wrong for this. A z
        # more than max_z_clamp below min_grasp_z means the depth is not to be
        # trusted at all, so x and y are not either.
        self.declare_parameter('max_z_clamp', 0.05)
        self.declare_parameter('workspace_radius', 1.0)
        self.declare_parameter('max_grasp_z', 0.80)
        # Ask /compute_ik whether the object is reachable before moving.
        self.declare_parameter('check_reach', True)
        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('orientation_tolerance', 0.05)

        self.declare_parameter('states_file', DEFAULT_STATES_FILE)
        self.declare_parameter('place_mode', 'state')
        self.declare_parameter('place_position', [0.35, 0.30, 0.25])
        self.declare_parameter('place_yaw', 0.0)
        self.declare_parameter('place_approach_height', 0.15)
        self.declare_parameter('place_radius', 0.15)

        self.declare_parameter('gripper_open', 0.044)      # finger_joint1 upper limit
        self.declare_parameter('gripper_close', 0.0)
        # Passed to the gripper action as max_effort. The v10 hardware
        # discards it (the controller's position adapter never forwards it to a
        # command interface), so it is the cap below that actually bounds the
        # grip -- this is left at the controller's own stall-detection value.
        self.declare_parameter('gripper_max_effort', 20.0)
        # Grip torque cap, Nm at the gripper motor -- the same units and
        # default as DEFAULT_GRIPPER_TORQUE_CAP_NM in the exoskeleton bridge.
        # 2.5 Nm is about 59.5 N at the finger over the 42.0 mm/rad
        # transmission, held by 5.25 mm of finger overshoot.
        # Adjustable from the panel; read per close.
        self.declare_parameter('gripper_torque_cap', 2.5)
        self.declare_parameter('gripper_close_step', 0.002)
        self.declare_parameter('gripper_step_settle', 0.08)
        self.declare_parameter('gripper_settle_time', 1.0)
        self.declare_parameter('grasp_finger_min', 0.003)
        self.declare_parameter('grasp_finger_max', 0.040)

        self.declare_parameter('detect_timeout', 15.0)
        self.declare_parameter('object_moved_eps', 0.05)
        self.declare_parameter('object_hold_radius', 0.15)
        self.declare_parameter('attached_object_size', [0.05, 0.05, 0.05])

        self.declare_parameter('use_table_collision', False)
        self.declare_parameter('table_z', 0.0)
        self.declare_parameter('table_size', [1.2, 1.2, 0.02])
        self.declare_parameter('refresh_octomap_frames', 3)
        self.declare_parameter('refresh_octomap_at_home', True)
        # HOME is the only pose the octomap may be captured from, and it is not
        # READY. READY holds the arm out over the table so the camera can see
        # the work surface -- which means the arm is *in the frame*, and a map
        # captured there contains the arm. That is not theoretical: it put
        # voxels around openarm_right_link7, every later plan then reported
        # "Start state appears to be in collision", and MoveIt rejected
        # cuMotion's path at every index with "Computed path is not valid".
        #
        # Seven zeros is the "home" group state in openarm_bimanual.srdf, which
        # folds the arm down by the base and out of the camera's view of the
        # table.
        self.declare_parameter('home_joint_positions', [0.0] * 7)
        self.declare_parameter('octomap_settle_time', 1.5)
        self.declare_parameter('home_pose_tolerance', 0.05)
        # How close counts as "already at" a named posture, radians.
        self.declare_parameter('at_goal_tolerance', 0.02)
        self.declare_parameter('auto_start', False)

        p = self.get_parameter
        self.base_frame = 'world'
        self.launch_arm = p('arm').value
        if self.launch_arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'")
        self.gripper_clients = {}
        self.configure_arm(self.launch_arm)

        self.arm_selection = p('arm_selection').value
        if self.arm_selection not in ('fixed', 'by_side'):
            raise ValueError("arm_selection must be 'fixed' or 'by_side'")

        self.place_mode = p('place_mode').value
        if self.place_mode == 'ready':
            # Renamed with the pose it referred to. Accepted rather than
            # rejected so an old launch line still starts.
            self.get_logger().warn(
                "place_mode 'ready' is now 'home': READY and HOME are one pose")
            self.place_mode = 'home'
        if self.place_mode not in ('state', 'home', 'position'):
            raise ValueError("place_mode must be 'state', 'home' or 'position'")

        self.prompt = p('prompt').value
        if not self.prompt.lower().startswith('detect'):
            self.prompt = f'detect {self.prompt}'

        self._lock = threading.Lock()
        self._latest_detections = None
        self._finger_position = None
        self._finger_effort = None
        self._failures = []
        self._arm_positions = {}
        self._arm_efforts = {}
        self._arm_velocities = {}
        self._arm_positions_at = 0.0
        self._holding = False
        self._last_detection = None
        self._last_payload = None
        self.state = 'INIT'
        motion_log = self.get_parameter('motion_log').value
        self._motion_log = (os.path.join(WS, motion_log)
                            if motion_log and not os.path.isabs(motion_log)
                            else motion_log)
        self._busy = False
        self._abort = threading.Event()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.move_client = ActionClient(self, MoveGroup, '/move_action',
                                        callback_group=self.cb)
        self.scene_client = self.create_client(ApplyPlanningScene,
                                               '/apply_planning_scene',
                                               callback_group=self.cb)
        self.octomap_client = self.create_client(Trigger, '/octomap_gater/refresh',
                                                 callback_group=self.cb)
        # move_group's own service. Needed because a map with the arm in it
        # blocks every plan, including the one that would replace it.
        self.clear_octomap_client = self.create_client(
            Empty, '/clear_octomap', callback_group=self.cb)
        # Seeded IK, so the descent keeps the posture it approached in.
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik',
                                            callback_group=self.cb)
        # A genuine straight line, for the descent onto the object and the
        # retreat off it.
        self.cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=self.cb)
        self.execute_client = ActionClient(self, ExecuteTrajectory,
                                           '/execute_trajectory',
                                           callback_group=self.cb)
        # Long-lived, like the two above. Creating one per query and destroying
        # it in a finally: looks tidier but destroys it out from under the
        # spinning executor, which raises InvalidHandle inside spin() and kills
        # the spin thread -- after which the node is deaf: no joint states, no
        # detections, no action results, and every subsequent goal "rejected".
        self.planner_param_client = self.create_client(
            GetParameters, '/cumotion_planner/get_parameters',
            callback_group=self.cb)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.prompt_pub = self.create_publisher(String, '/vlm/prompt', latched)
        self.state_pub = self.create_publisher(String, '/pick_place/state', latched)

        self.create_subscription(String, '/pick_place/prompt', self._on_prompt, 10,
                                 callback_group=self.cb)
        self.create_subscription(String, '/vlm/detections', self._on_detections, 10,
                                 callback_group=self.cb)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10,
                                 callback_group=self.cb)

        self.create_service(Trigger, '/pick_place/start', self._srv_start,
                            callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/abort', self._srv_abort,
                            callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/capture_ready',
                            self._srv_capture_ready, callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/open_gripper',
                            self._srv_open_gripper, callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/grip',
                            self._srv_grip, callback_group=self.cb)

        self._set_state('IDLE')
        self.get_logger().info(
            f'arm={self.arm} group={self.group} tcp={self.tcp_frame} '
            f'pipeline={p("pipeline_id").value}')
        ready = [round(v, 4) for v in p('ready_joint_positions').value]
        self.get_logger().info(f'ready pose (rad) {ready}')
        if self.place_mode == 'state':
            self.get_logger().info(f'states file {self.states_file}')
            states = self.load_states()
            for name in (PRE_PICK_STATE, DROP_STATE):
                entry = states.get(name)
                if entry is None:
                    self.get_logger().warn(
                        f'{name} is not recorded yet -- run '
                        f'"python3 record_states.py {name}" before starting a '
                        'cycle')
                else:
                    self.get_logger().info(
                        f'{name} tool at {entry.get("tcp_xyz")}')
        elif self.place_mode == 'home':
            self.get_logger().warn(
                'place_mode is "home": the object is released at HOME, so it '
                'drops from whatever height that pose holds the tool at. Check '
                'what is underneath before the first run.')
        else:
            place = p('place_position').value
            self.get_logger().warn(
                f'place_position is {list(place)} -- verify this is reachable for '
                f'the {self.arm} arm in RViz before running on hardware')

        if p('auto_start').value:
            self.create_timer(3.0, self._auto_start_once, callback_group=self.cb)

    # -- plumbing ------------------------------------------------------------

    def _note_failure(self, stage, message):
        self._failures.append((stage, message))

    def _failure_summary(self):
        """Which stage failed most often -- not merely which failed last.

        Attempts can fail at different stages: a marginal pre_pick plan on one,
        an out-of-reach object on the next. Reporting only the last one is
        actively misleading -- it once pointed at pre_pick_state, which failed
        twice, while the pre-grasp failed three times because the object was
        90 mm beyond the arm. So report the dominant stage, and count the rest.
        """
        if not self._failures:
            return 'no attempt ran'
        counts = {}
        for stage, message in self._failures:
            entry = counts.setdefault(stage, [0, message])
            entry[0] += 1
        stage, (count, message) = max(counts.items(), key=lambda item: item[1][0])
        summary = f'{count} of {len(self._failures)} attempts failed at {stage}'
        others = ', '.join(f'{s} x{c}' for s, (c, _m) in sorted(counts.items())
                           if s != stage)
        if others:
            summary += f' (also {others})'
        return f'{summary}. {message}'

    def configure_arm(self, arm):
        """Point every arm-dependent name at `arm`.

        Everything the cycle touches is derived here rather than fixed at
        construction, so arm_selection "by_side" can swap arms between cycles:
        the planning group, the tool and hand frames, the seven joints, the
        touch links for the attached object, the gripper action, and the states
        file. Gripper action clients are cached per arm -- creating one per
        cycle leaks them.
        """
        self.arm = arm
        self.group = f'{arm}_arm'
        self.tcp_frame = f'openarm_{arm}_hand_tcp'
        self.hand_link = f'openarm_{arm}_hand'
        self.finger_joint = f'openarm_{arm}_finger_joint1'
        self.arm_joints = [f'openarm_{arm}_joint{i}' for i in range(1, 8)]
        self.touch_links = [
            self.hand_link,
            f'openarm_{arm}_left_finger',
            f'openarm_{arm}_right_finger',
        ]

        if arm not in self.gripper_clients:
            self.gripper_clients[arm] = ActionClient(
                self, GripperCommand, f'/{arm}_gripper_controller/gripper_cmd',
                callback_group=self.cb)
        self.gripper_client = self.gripper_clients[arm]

        # Joint readings are per-arm, so a switch must not compare the new
        # arm's ready pose against the old arm's measurements.
        self._arm_positions = {}
        self._arm_efforts = {}
        self._arm_velocities = {}
        self._arm_positions_at = 0.0
        self._finger_position = None
        self._finger_effort = None

        self.states_file = self.resolve_states_file()

    def resolve_states_file(self):
        """The states file for the current arm."""
        configured = self.get_parameter('states_file').value
        if configured not in ('', 'auto'):
            return configured
        path = os.path.join(WS, f'pick_place_states_{self.arm}.yaml')
        if os.path.exists(path) or not os.path.exists(LEGACY_STATES_FILE):
            return path
        # Fall back to the pre-split file only when it was recorded for this
        # arm. Handing over the other arm's recording would just be rejected by
        # load_states(), with a confusing message about the wrong file.
        try:
            with open(LEGACY_STATES_FILE) as handle:
                recorded_arm = (yaml.safe_load(handle) or {}).get('arm')
        except (OSError, yaml.YAMLError):
            return path
        if recorded_arm != self.arm:
            return path
        self.get_logger().warn(
            f'{path} does not exist; falling back to the pre-split '
            f'{LEGACY_STATES_FILE}')
        return LEGACY_STATES_FILE

    def _set_state(self, state, detail=''):
        self.state = state
        text = state if not detail else f'{state}: {detail}'
        self.state_pub.publish(String(data=text))
        self.get_logger().info(f'[{state}] {detail}'.rstrip())

    def _await(self, future, timeout):
        """Block a worker thread on an rclpy future.

        spin_until_future_complete cannot be used here: the executor is already
        spinning on the main thread. add_done_callback plus an Event is the
        thread-safe equivalent.
        """
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return None
        return future.result()

    def _on_prompt(self, msg):
        """Retarget the next cycle, so the UI can change what to pick.

        Refused while a cycle is running: self.prompt is what detect() matches
        replies against, and swapping it mid-cycle would leave the retry ladder
        hunting for a different object than the one it started on.
        """
        # Normalised the same way the panel does, so that publishing
        # "pick up the wrench" straight onto this topic behaves identically to
        # typing it into the panel.
        wanted = to_detection_prompt(msg.data)
        if not wanted:
            self.get_logger().warn(f'ignoring prompt "{msg.data}": no object in it')
            return
        with self._lock:
            if self._busy:
                self.get_logger().warn(
                    f'ignoring prompt "{wanted}": a cycle is already running')
                return
            self.prompt = wanted
        self.get_logger().info(f'prompt is now "{wanted}"')
        self.prompt_pub.publish(String(data=wanted))

    def _on_detections(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('malformed JSON on /vlm/detections')
            return
        with self._lock:
            self._latest_detections = payload

    def joint_snapshot(self):
        """Positions, velocities and efforts of this arm plus the gripper.

        Efforts are the motor values: for the arm joints they come straight
        from the hardware interface, and for the finger they are what the grip
        torque cap reacts to.
        """
        with self._lock:
            return {
                'joints': {j: round(self._arm_positions[j], 6)
                           for j in self.arm_joints
                           if j in self._arm_positions},
                'velocities': {j: round(self._arm_velocities[j], 6)
                               for j in self.arm_joints
                               if j in self._arm_velocities},
                'efforts': {j: round(self._arm_efforts[j], 4)
                            for j in self.arm_joints
                            if j in self._arm_efforts},
                'finger': (None if self._finger_position is None
                           else round(self._finger_position, 6)),
                'finger_effort': (None if self._finger_effort is None
                                  else round(self._finger_effort, 4)),
            }

    def sample_motion(self):
        """Record where the arm goes *while* it moves, not just at the ends.

        A start and an end point cannot tell a straight descent from one that
        swings 3.7 cm sideways on the way -- and that difference is exactly
        what breaks things here. Returns a stop function that hands back the
        samples.
        """
        period = self.get_parameter('motion_sample_period').value
        if not self._motion_log or period <= 0.0:
            return lambda: []

        samples = []
        stop = threading.Event()
        started = self.get_clock().now().nanoseconds * 1e-9

        def poll():
            while not stop.is_set():
                now = self.get_clock().now().nanoseconds * 1e-9
                with self._lock:
                    joints = [self._arm_positions.get(j)
                              for j in self.arm_joints]
                entry = {'t': round(now - started, 3)}
                if all(v is not None for v in joints):
                    entry['joints'] = [round(v, 5) for v in joints]
                tcp = self.tcp_position()
                if tcp is not None:
                    entry['tcp'] = [round(v, 5) for v in tcp]
                samples.append(entry)
                stop.wait(period)

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()

        def finish():
            stop.set()
            thread.join(timeout=2.0)
            # Cap it: a slow move at 20 Hz would otherwise write thousands of
            # lines for one descent. Keep the ends and thin the middle.
            limit = int(self.get_parameter('motion_sample_limit').value)
            if limit > 2 and len(samples) > limit:
                step = len(samples) / float(limit - 1)
                kept = [samples[int(i * step)] for i in range(limit - 1)]
                kept.append(samples[-1])
                return kept
            return samples

        return finish

    def log_motion(self, label, method, outcome, **extra):
        """Append one motion to the motion log. Never raises.

        Called for every commanded motion, successful or not. The file is
        opened and closed per record and flushed: a cycle that ends in a
        segfault -- which has happened here more than once -- must still leave
        everything up to that point on disk.
        """
        path = self._motion_log
        if not path:
            return
        record = {
            'time': round(self.get_clock().now().nanoseconds * 1e-9, 4),
            'state': self.state,
            'arm': self.arm,
            'label': label,
            'method': method,
            'outcome': outcome,
        }
        record.update(extra)
        record['measured'] = self.joint_snapshot()
        tcp = self.tcp_position()
        if tcp is not None:
            record['tcp'] = [round(v, 5) for v in tcp]
        try:
            with open(path, 'a') as handle:
                handle.write(json.dumps(record) + '\n')
                handle.flush()
        except OSError as exc:                     # never break a cycle for a log
            self.get_logger().warn(f'could not write {path}: {exc}',
                                   throttle_duration_sec=60.0)

    def _on_joint_states(self, msg):
        with self._lock:
            if self.finger_joint in msg.name:
                index = msg.name.index(self.finger_joint)
                self._finger_position = msg.position[index]
                if index < len(msg.effort):
                    self._finger_effort = msg.effort[index]
            for joint in self.arm_joints:
                if joint in msg.name:
                    index = msg.name.index(joint)
                    self._arm_positions[joint] = msg.position[index]
                    if index < len(msg.effort):
                        self._arm_efforts[joint] = msg.effort[index]
                    if index < len(msg.velocity):
                        self._arm_velocities[joint] = msg.velocity[index]
            # When this arrived, not when it was stamped. Used to tell a
            # reading of where the arm is *now* from one cached before the last
            # move -- see joint_error(fresh=True).
            self._arm_positions_at = self.get_clock().now().nanoseconds * 1e-9

    def _auto_start_once(self):
        for timer in list(self.timers):
            timer.cancel()
        self._start_cycle()

    def _srv_start(self, _request, response):
        response.success, response.message = self._start_cycle()
        return response

    def _srv_capture_ready(self, _request, response):
        """Report the arm's current joint values as a ready_joint_positions list.

        Jog the arm to the pose you want in RViz, call this, and paste the list
        back as -p ready_joint_positions:="[...]". Beats reading angles off the
        Joints tab, which only shows them rounded to the degree.
        """
        with self._lock:
            missing = [j for j in self.arm_joints if j not in self._arm_positions]
            values = [self._arm_positions.get(j) for j in self.arm_joints]
        if missing:
            response.success = False
            response.message = f'no /joint_states yet for {missing}'
            return response
        formatted = '[' + ', '.join(f'{v:.6f}' for v in values) + ']'
        self.get_logger().info(f'ready_joint_positions:="{formatted}"')
        response.success = True
        response.message = formatted
        return response

    def _srv_abort(self, _request, response):
        self._abort.set()
        response.success = True
        response.message = 'abort requested'
        return response

    def _gripper_report(self):
        """Finger opening and measured torque, for a service response."""
        position = self.finger_position()
        torque = self.finger_effort()
        where = 'unknown' if position is None else f'{position * 1000:.1f} mm'
        force = 'not reported' if torque is None else f'{abs(torque):.3f} Nm'
        return f'finger {where}, torque {force}'

    def _srv_open_gripper(self, _request, response):
        """Open the gripper on its own, so an object can be placed in it.

        Separate from a cycle on purpose: the cap is worth testing by hand
        before trusting it on a real pick, and that means opening the fingers,
        putting something between them and closing -- with no arm motion at all.
        """
        if not self._claim_gripper(response):
            return response
        self._abort.clear()
        try:
            opened = self.open_gripper()
        finally:
            self._release_gripper()
        response.success = opened
        response.message = (f'opened -- {self._gripper_report()}' if opened
                            else f'open failed -- {self._gripper_report()}')
        return response

    def _srv_grip(self, _request, response):
        """Close to the torque cap and hold, with nothing else moving.

        Reports where the fingers stopped and the torque they stopped at, which
        is the number to compare against gripper_torque_cap.
        """
        if not self._claim_gripper(response):
            return response
        self._abort.clear()
        try:
            gripped = self.close_gripper_to_cap()
        finally:
            self._release_gripper()
        cap = self.get_parameter('gripper_torque_cap').value
        response.success = gripped
        response.message = (f'{"gripped" if gripped else "grip failed"} at a '
                            f'{cap:.3f} Nm cap -- {self._gripper_report()}')
        self.get_logger().info(response.message)
        return response

    def _claim_gripper(self, response):
        """Take the busy flag, so a hand test cannot overlap a cycle."""
        with self._lock:
            if self._busy:
                response.success = False
                response.message = 'a cycle is already running'
                return False
            self._busy = True
        return True

    def _release_gripper(self):
        with self._lock:
            self._busy = False

    def _start_cycle(self):
        with self._lock:
            if self._busy:
                return False, 'a cycle is already running'
            self._busy = True
        self._abort.clear()
        threading.Thread(target=self._run_cycle, daemon=True).start()
        return True, 'cycle started'

    # -- motion primitives ---------------------------------------------------

    def _base_request(self):
        req = MoveGroup.Goal().request
        req.group_name = self.group
        req.pipeline_id = self.get_parameter('pipeline_id').value
        req.num_planning_attempts = 1
        req.allowed_planning_time = self.get_parameter('planning_time').value
        req.max_velocity_scaling_factor = self.get_parameter('velocity_scaling').value
        req.max_acceleration_scaling_factor = \
            self.get_parameter('acceleration_scaling').value
        req.start_state.is_diff = True
        req.workspace_parameters.header.frame_id = self.base_frame
        for corner, sign in ((req.workspace_parameters.min_corner, -1.0),
                             (req.workspace_parameters.max_corner, 1.0)):
            corner.x = corner.y = corner.z = sign * 1.5
        return req

    def _send_move_goal(self, request, label):
        """Send one goal, resending it if the planner merely failed to converge.

        See RETRYABLE_MOVEIT_CODES: cuMotion's optimiser is stochastic and
        misses roughly one goal in seven, and the same goal sent again normally
        succeeds. Without this a cycle inherits that rate eight times over.
        """
        if not self.move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('/move_action unavailable -- is move_group running?')
            return False

        attempts = max(1, int(self.get_parameter('plan_attempts').value))
        before = self.joint_snapshot()
        constraints = request.goal_constraints[0]
        if constraints.joint_constraints:
            method, target = 'joint', [round(jc.position, 6)
                                       for jc in constraints.joint_constraints]
        else:
            pose = constraints.position_constraints[0]
            point = pose.constraint_region.primitive_poses[0].position
            method = 'pose'
            target = [round(point.x, 5), round(point.y, 5), round(point.z, 5)]
        for attempt in range(1, attempts + 1):
            if self._abort.is_set():
                self.get_logger().warn(f'{label}: aborted')
                return False
            finish = self.sample_motion()
            code = self._send_move_goal_once(request, label)
            path = finish()
            if code == MOVEIT_SUCCESS:
                if attempt > 1:
                    self.get_logger().info(
                        f'{label}: planned on attempt {attempt} of {attempts}')
                self.log_motion(label, method, 'ok', target=target,
                                attempt=attempt, before=before, path=path)
                return True
            if code not in RETRYABLE_MOVEIT_CODES:
                # Nothing about the goal changes by asking again.
                self.get_logger().error(
                    f'{label}: MoveIt error code {code}, not retryable')
                self.log_motion(label, method, 'failed', target=target,
                                attempt=attempt, code=code, before=before,
                                path=path)
                return False
            if attempt < attempts:
                self.get_logger().warn(
                    f'{label}: MoveIt error code {code} on attempt {attempt} '
                    f'of {attempts}, resending')
        self.get_logger().error(
            f'{label}: still failing after {attempts} attempts (last code '
            f'{code})')
        self.log_motion(label, method, 'exhausted', target=target,
                        attempt=attempts, code=code, before=before)
        if code == MoveItErrorCodes.PLANNING_FAILED:
            # Worth spelling out, because the two causes need opposite
            # responses and the code cannot tell them apart: the cuMotion
            # MoveIt plugin reports "No trajectory" and the pipeline turns
            # every cuMotion failure into PLANNING_FAILED, whatever the
            # planner's own status was.
            self.get_logger().error(
                'PLANNING_FAILED covers two very different things here. '
                'TRAJOPT_FAIL is the optimiser not converging and a resend '
                'usually fixes it; IK_FAIL means the pose is not reachable at '
                'all and no number of resends will help. The planner log says '
                'which:  grep MotionGenStatus '
                '~/.ros/log/python3_*.log | tail')
        return False

    def _send_move_goal_once(self, request, label):
        """One attempt. Returns a MoveIt error code, never raises."""
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        handle = self._await(self.move_client.send_goal_async(goal), 15.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: goal rejected by move_group')
            # Not retryable: move_group refused to take it at all.
            return MoveItErrorCodes.FAILURE

        result = self._await(handle.get_result_async(),
                             self.get_parameter('motion_timeout').value)
        if result is None:
            self.get_logger().error(f'{label}: timed out waiting for execution')
            # A goal that never came back may still be executing; resending
            # would race with it.
            return MoveItErrorCodes.FAILURE
        return result.result.error_code.val

    def pose_constraints(self, position, quat, link):
        """A Cartesian goal for `link`. Shared so a reach probe and a real
        move ask for exactly the same thing."""
        pos_tol = self.get_parameter('position_tolerance').value
        ori_tol = self.get_parameter('orientation_tolerance').value

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = link
        pc.weight = 1.0
        region = SolidPrimitive()
        region.type = SolidPrimitive.BOX
        region.dimensions = [2 * pos_tol] * 3
        pc.constraint_region.primitives.append(region)
        target = PoseStamped().pose
        target.position.x, target.position.y, target.position.z = position
        target.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(target)

        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = link
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = quat
        oc.absolute_x_axis_tolerance = ori_tol
        oc.absolute_y_axis_tolerance = ori_tol
        oc.absolute_z_axis_tolerance = ori_tol
        oc.weight = 1.0

        return Constraints(position_constraints=[pc],
                           orientation_constraints=[oc])

    def move_to_pose(self, position, quat, label):
        req = self._base_request()
        req.goal_constraints = [
            self.pose_constraints(position, quat, self.tcp_frame)]
        self.get_logger().info(
            f'{label}: xyz=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})')
        return self._send_move_goal(req, label)

    def _move_to_joints(self, positions, label, skip_if_there=False):
        """Joint-space goal, used for every named posture.

        A joint goal is the one kind cuMotion takes for either arm regardless of
        its ee_link: it is turned into a pose by running FK on the merged goal
        state, so no link name is compared. It is also repeatable -- the same
        seven numbers give the same posture every time, which a pose goal on a
        redundant 7-DOF arm does not.

        skip_if_there returns success without sending a goal when the arm is
        already at the posture. Used for the named states, and it is why a
        retry no longer shuttles: the ladder re-enters PRE_PICK on every
        attempt, and re-planning a move the arm has already made is a wasted
        several seconds and a wasted roll against cuMotion's ~14.5% failure
        rate -- the arm visibly steps back and forth for nothing.
        """
        if skip_if_there:
            worst = self.joint_error(positions, fresh=True)
            tolerance = self.get_parameter('at_goal_tolerance').value
            if worst is not None and worst <= tolerance:
                self.get_logger().info(
                    f'{label}: already there ({worst:.4f} rad off, tolerance '
                    f'{tolerance:.3f}); not re-planning')
                return True
        if len(positions) != len(self.arm_joints):
            self.get_logger().error(
                f'{label} needs {len(self.arm_joints)} joint values, got '
                f'{len(positions)}')
            return False
        req = self._base_request()
        constraints = Constraints()
        for name, value in zip(self.arm_joints, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints = [constraints]
        return self._send_move_goal(req, label)

    def await_joint_states(self, timeout=2.0):
        """Wait for a reading of the arm now configured.

        configure_arm() drops the previous arm's readings, so anything asked
        immediately after a switch would otherwise be judging on no data --
        which reads as "not at the ready pose" and silently skips the octomap.
        """
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        while self.get_clock().now().nanoseconds * 1e-9 < deadline:
            with self._lock:
                if all(j in self._arm_positions for j in self.arm_joints):
                    return True
            if self._abort.wait(0.05):
                return False
        return False

    def home_positions(self):
        """HOME: where the arm starts, observes from, and returns to.

        Taken from home_state in this arm's states file when it is recorded
        there, because the arms are mirrored and one parameter cannot describe
        both. home_joint_positions is the fallback.
        """
        states = self.load_states()
        joints = (states.get(HOME_STATE) or {}).get('joints')
        if joints:
            return list(joints)
        if (states.get(LEGACY_READY_STATE) or {}).get('joints'):
            # Deliberately *not* used. The old ready_state was an observation
            # pose that held the arm out over the table so the camera could see
            # it -- in frame, which is what kept getting the arm captured into
            # the octomap. Replaying it as HOME would reintroduce exactly that.
            self.get_logger().warn(
                f'{self.states_file} has a ready_state but no home_state. '
                'Ignoring it: the old ready pose held the arm in the camera '
                'view, which is why the map kept containing the arm. Record a '
                f'home_state clear of the camera:  python3 record_states.py '
                f'--arm {self.arm} home_state',
                throttle_duration_sec=60.0)
        return list(self.get_parameter('home_joint_positions').value)

    def joint_error(self, wanted, fresh=False):
        """Worst per-joint distance from `wanted`, or None with no readings.

        Measured from /joint_states rather than inferred from "the move goal
        succeeded", because those are not the same thing: a goal can come back
        SUCCESS while execution was cut short.

        fresh=True waits for a reading that arrived *after* this call, and is
        required by anything that decides not to move. The cached reading can
        still describe the posture from before the previous goal -- the
        callback and the caller are different threads and the publisher runs at
        its own rate -- and acting on that means skipping a move because the
        arm was at the target a moment ago, which is exactly backwards.
        """
        if fresh:
            since = self.get_clock().now().nanoseconds * 1e-9
            deadline = since + 1.0
            while True:
                with self._lock:
                    if self._arm_positions_at > since:
                        break
                if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                    self.get_logger().warn(
                        'no fresh joint state within 1 s; treating the arm as '
                        'not at the goal')
                    return None
                if self._abort.wait(0.02):
                    return None
        else:
            self.await_joint_states()
        with self._lock:
            measured = [self._arm_positions.get(j) for j in self.arm_joints]
        if any(v is None for v in measured) or len(wanted) != len(measured):
            return None
        return max(abs(have - want) for have, want in zip(measured, wanted))

    def at_home_pose(self):
        """Is the arm measurably at HOME right now?

        Insists on a reading that arrived after the question was asked. The
        cached one can still describe the posture from before the move to
        home, and capturing a map on that is how the arm ends up inside its
        own octomap -- voxels around link7, every later plan reporting the
        start state in collision.
        """
        tolerance = self.get_parameter('home_pose_tolerance').value
        worst = self.joint_error(self.home_positions(), fresh=True)
        if worst is None:
            self.get_logger().warn('no joint states for the arm; assuming not at home')
            return False
        if worst > tolerance:
            self.get_logger().info(
                f'not at home: worst joint is {worst:.3f} rad off '
                f'(tolerance {tolerance:.3f})')
            return False
        return True

    def move_to_home(self):
        return self._move_to_joints(self.home_positions(), 'HOME',
                                    skip_if_there=True)

    def capture_octomap(self):
        """Update the octomap -- only from HOME, only empty-handed.

        Three ways this is refused, all for the same reason: whatever the camera
        can see becomes a permanent obstacle.

        * Not at HOME. This is the important one, and READY is not good enough:
          READY deliberately holds the arm out over the table so the camera can
          see the work surface, so the arm is in the frame and gets captured as
          an obstacle sitting exactly where the arm is about to plan from.
          Measured, not inferred from a goal result.
        * Holding something. The payload would be captured as an obstacle that
          then travels with the tool.
        * Turned off with refresh_octomap_at_home.
        """
        if not self.get_parameter('refresh_octomap_at_home').value:
            return False
        if self._holding:
            self.get_logger().info(
                'carrying the object: leaving the octomap alone so the payload '
                'is not captured as an obstacle')
            return False
        if not self.at_home_pose():
            self.get_logger().warn(
                'skipping the octomap update: the arm is not at home')
            return False
        if not self.refresh_octomap():
            return False
        # The updater integrates asynchronously; planning against a half-built
        # map is worse than planning against the previous one.
        self._abort.wait(self.get_parameter('octomap_settle_time').value)
        return True

    def map_from_home(self):
        """Clear the map, go home, and rebuild it from there.

        The clear comes first and is not optional. A map with the arm in it
        makes the arm's own start state collide, and then *nothing* plans --
        including the move to home that would let a good map be captured. So a
        poisoned map has to be dropped before it can be replaced, or the cell
        is stuck needing a plan it cannot get.

        Which is exactly why this refuses while holding something, before it
        clears anything: capture_octomap would refuse at the end, and the cycle
        would be left with no map at all for the drop.
        """
        if self._holding:
            self.get_logger().info(
                'carrying the object: not remapping, so the payload is not '
                'captured and the existing map is kept')
            return False
        self.clear_octomap()
        self._set_state('HOME')
        if not self.move_to_home():
            self.get_logger().error(
                'could not reach home, so the octomap cannot be refreshed')
            return False
        return self.capture_octomap()

    def arrive_at_home(self):
        """Refresh the map from home, then take up the observation pose.

        In that order: the map has to be built with the arm out of the frame,
        and READY is in the frame by design.
        """
        if not self.map_from_home():
            # Worth continuing on the previous map rather than refusing to
            # pick: it may well be usable, and the failure is already logged.
            self.get_logger().warn('continuing on the existing octomap')
        # map_from_home() has already driven the arm to HOME, and HOME is
        # where the object is observed from -- there is nowhere else to go.
        return self.at_home_pose()

    def move_to_state(self, name, states):
        """Replay a pose recorded by record_states.py."""
        entry = states.get(name)
        if not entry or not entry.get('joints'):
            self.get_logger().error(
                f'{name} is not in {self.states_file}; record it with '
                f'"python3 record_states.py {name}"')
            return False
        return self._move_to_joints(list(entry['joints']), name.upper(),
                                    skip_if_there=True)

    def command_gripper(self, position, label, settle=None):
        """Drive the gripper and wait for it to settle.

        The result status is intentionally discarded: this controller has
        allow_stalling false, so a successful grasp -- fingers stopped by the
        object -- comes back ABORTED. finger_position() is the real signal.
        """
        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f'/{self.arm}_gripper_controller/gripper_cmd unavailable')
            return False
        before = self.joint_snapshot()
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = self.get_parameter('gripper_max_effort').value
        handle = self._await(self.gripper_client.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: gripper goal rejected')
            return False
        self._await(handle.get_result_async(), 10.0)
        if settle is None:
            settle = self.get_parameter('gripper_settle_time').value
        self._abort.wait(settle)
        self.get_logger().info(f'{label}: finger at {self.finger_position()}')
        self.log_motion(label, 'gripper', 'ok',
                        target=round(float(position), 6), before=before)
        return True

    def open_gripper(self, label='OPEN'):
        """Open to gripper_open, which defaults to finger_joint1's upper limit.

        0.044 m is the limit in openarm_hand.xacro, so this is as wide as the
        fingers mechanically go -- there is no more travel to ask for.
        """
        return self.command_gripper(
            self.get_parameter('gripper_open').value, label)

    def close_gripper_to_cap(self):
        """Close until measured torque reaches the cap, then stop advancing.

        The same enforcement the exoskeleton bridge does, for the same reason:
        the gripper is commanded as a position and its closing force is position
        error times the hardware's fixed GRIPPER_DEFAULT_KP, so nothing in
        ros2_control bounds it -- max_effort never reaches a command interface.
        The only way to bound the grip is to stop advancing the position
        command once torque says the cap is reached.

        Two things this must not do. It must not command the fingers' *measured*
        position as a "hold": that would zero the position error and therefore
        the grip force, and the object would drop. What holds the object is
        precisely the last command that was still under the cap. And it must not
        command a full close in one goal, which would be at full squeeze before
        the first torque reading came back -- hence stepping, coarse until
        torque appears and fine near the cap, so the step does not overshoot it
        by more than it has to.
        """
        cap = self.get_parameter('gripper_torque_cap').value
        coarse = self.get_parameter('gripper_close_step').value
        settle = self.get_parameter('gripper_step_settle').value
        target = self.get_parameter('gripper_close').value
        fine = max(coarse / 4.0, 1e-4)

        position = self.finger_position()
        if position is None:
            self.get_logger().warn(
                'no finger position; falling back to an uncapped close')
            return self.command_gripper(target, 'CLOSE')

        if self.finger_effort() is None:
            # A build from before the torque readback landed. Refusing would be
            # worse than the behaviour that has been running all along, but it
            # must be said out loud.
            self.get_logger().warn(
                'no gripper torque on /joint_states -- the cap cannot be '
                'enforced, closing uncapped. Rebuild with native/build_ws.sh '
                'so v10_simple_hardware reads gripper_motors[0].get_torque().')
            return self.command_gripper(target, 'CLOSE')

        while position > target and not self._abort.is_set():
            torque = abs(self.finger_effort() or 0.0)
            step = fine if torque > 0.5 * cap else coarse
            advanced = max(target, position - step)
            if not self.command_gripper(advanced, 'CLOSE', settle=settle):
                return False
            torque = abs(self.finger_effort() or 0.0)
            if torque >= cap:
                # Back off to the last command that was under the cap: that
                # position error is what holds the object at the cap force.
                self.get_logger().info(
                    f'torque cap reached at {advanced:.4f} m '
                    f'({torque:.3f} Nm >= {cap:.3f} Nm); holding at '
                    f'{position:.4f} m')
                return self.command_gripper(position, 'HOLD', settle=settle)
            position = advanced

        self.get_logger().info(
            f'closed to {position:.4f} m without reaching {cap:.3f} Nm '
            f'(torque {abs(self.finger_effort() or 0.0):.3f} Nm)')
        return True

    def finger_position(self):
        with self._lock:
            return self._finger_position

    def finger_effort(self):
        """Measured gripper motor torque in Nm, or None if the driver is silent.

        Reported only since the v10 hardware started reading
        gripper_motors[0].get_torque(); before that it was hardcoded to 0, and a
        cap is not enforceable without it.
        """
        with self._lock:
            return self._finger_effort

    # -- planning scene ------------------------------------------------------

    def _apply_scene(self, scene, label):
        if not self.scene_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'{label}: /apply_planning_scene unavailable')
            return False
        request = ApplyPlanningScene.Request()
        request.scene = scene
        result = self._await(self.scene_client.call_async(request), 5.0)
        if result is None or not result.success:
            self.get_logger().warn(f'{label}: planning scene update rejected')
            return False
        return True

    def attach_object(self):
        """Tell cuMotion the gripper is now holding something.

        Without this the planner drags an invisible object through the octomap
        on the way to the box.
        """
        size = list(self.get_parameter('attached_object_size').value)
        aco = AttachedCollisionObject()
        aco.link_name = self.hand_link
        aco.touch_links = self.touch_links
        aco.object.id = ATTACHED_OBJECT_ID
        aco.object.header.frame_id = self.tcp_frame
        aco.object.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = size
        aco.object.primitives.append(primitive)
        pose = PoseStamped().pose
        pose.orientation.w = 1.0
        aco.object.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        return self._apply_scene(scene, 'ATTACH')

    def detach_object(self):
        self._holding = False
        aco = AttachedCollisionObject()
        aco.link_name = self.hand_link
        aco.object.id = ATTACHED_OBJECT_ID
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        return self._apply_scene(scene, 'DETACH')

    def add_table(self):
        """Optional explicit collision box for the work surface.

        Worth turning on with octomap:=static, where the map can legitimately be
        empty and nothing else stops the tool descending through the table.
        """
        if not self.get_parameter('use_table_collision').value:
            self.get_logger().info(
                'use_table_collision is false: relying on the octomap alone for '
                'the work surface')
            return True
        table_z = self.get_parameter('table_z').value
        size = list(self.get_parameter('table_size').value)
        obj = CollisionObject()
        obj.id = TABLE_OBJECT_ID
        obj.header.frame_id = self.base_frame
        obj.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = size
        obj.primitives.append(primitive)
        pose = PoseStamped().pose
        pose.position.z = table_z - size[2] / 2.0     # top face at table_z
        pose.orientation.w = 1.0
        obj.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        return self._apply_scene(scene, 'TABLE')

    def clear_octomap(self):
        """Drop the current map.

        The escape hatch for a poisoned map. Once voxels sit where the arm is,
        MoveIt reports the start state in collision and rejects every path
        cuMotion returns -- "Computed path is not valid. Invalid states at
        index locations: [0 1 2 ...]" -- so no goal succeeds, not even the move
        to home that a good capture needs. Clearing is the only way out.
        """
        if not self.clear_octomap_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(
                '/clear_octomap unavailable -- is move_group running?')
            return False
        if self._await(self.clear_octomap_client.call_async(Empty.Request()),
                       5.0) is None:
            self.get_logger().warn('/clear_octomap did not answer')
            return False
        self.get_logger().info('octomap cleared')
        return True

    def refresh_octomap(self):
        if not self.octomap_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(
                '/octomap_gater/refresh unavailable (octomap:=live, or the '
                'gater is not running) -- skipping refresh')
            return False
        result = self._await(self.octomap_client.call_async(Trigger.Request()), 5.0)
        return bool(result and result.success)

    def _planner_parameter(self, name):
        """One parameter off the cuMotion node.

        Queried one name per call on purpose: if any name in a GetParameters
        request is declared-but-unset, the whole reply comes back with values=[]
        rather than a PARAMETER_NOT_SET entry per name -- measured against the
        running planner. Batching tool_frame (usually unset) with robot (always
        set) therefore returns nothing for either.

        Returns DEAD_PLANNER if the node is not there at all, None if it is
        there but did not answer, '' if the parameter is unset, otherwise its
        value.
        """
        client = self.planner_param_client
        if not client.wait_for_service(timeout_sec=5.0):
            return DEAD_PLANNER
        result = self._await(
            client.call_async(GetParameters.Request(names=[name])), 5.0)
        if result is None:
            return None
        if not result.values:
            return ''
        return result.values[0].string_value

    def planner_ee_link(self):
        """Which link cuMotion will actually accept Cartesian goals for.

        tool_frame wins if it is set; otherwise it is the ee_link in the robot
        config the node was launched with. Returns None if neither could be
        determined.
        """
        tool_frame = self._planner_parameter('tool_frame')
        if tool_frame is DEAD_PLANNER or tool_frame is None:
            return tool_frame
        if tool_frame:
            return tool_frame
        robot_file = self._planner_parameter('robot')
        if robot_file is DEAD_PLANNER or not robot_file:
            return None
        try:
            import yaml
            with open(robot_file) as handle:
                config = yaml.safe_load(handle)
            return config['robot_cfg']['kinematics']['ee_link']
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f'could not read ee_link from {robot_file}: {exc}')
            return None

    def load_states(self):
        """Read the recorded poses. Re-read every cycle, on purpose.

        That way re-recording a pose with record_states.py takes effect on the
        next pick without restarting the stack -- which matters because the
        stack takes a PaliGemma load to come back up.
        """
        path = self.states_file
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().error(f'could not read {path}: {exc}')
            return {}
        recorded_arm = data.get('arm')
        if recorded_arm and recorded_arm != self.arm:
            # Joint values are per-arm: the arms are mirrored, so replaying a
            # left-arm recording on the right arm is a different posture, not a
            # mirrored one.
            self.get_logger().error(
                f'{path} was recorded for the {recorded_arm} arm but this is the '
                f'{self.arm} arm; re-record it with '
                f'"python3 record_states.py --arm {self.arm}"')
            return {}
        return data.get('states') or {}

    def check_states(self, states):
        """Every pose the cycle will need, before it moves anything."""
        needed = [PRE_PICK_STATE]
        if self.place_mode == 'state':
            needed.append(DROP_STATE)
        if self.arm_selection == 'by_side' \
                and self.arm != self.get_parameter(
                    'ready_joint_positions_arm').value:
            # ready_joint_positions is one list measured on one arm, and the
            # arms are mirrored -- driving the other arm to it would be a
            # different posture entirely. So the *other* arm must carry its own
            # recorded ready pose. Requiring it from both was over-strict and
            # refused to start on a setup that was actually complete.
            needed.append(HOME_STATE)
        missing = [n for n in needed if not (states.get(n) or {}).get('joints')]
        if missing:
            self.get_logger().error(
                f'missing {", ".join(missing)} in {self.states_file}. Record with: '
                f'python3 record_states.py --arm {self.arm} {" ".join(missing)}')
            return False
        return True

    def check_planner_tool_frame(self):
        """cuMotion has exactly one ee_link and rejects pose goals aimed elsewhere.

        openarm.yml sets ee_link to openarm_left_hand_tcp, so with the default
        config every right-arm pose goal comes back INVALID_LINK_NAME -- but only
        at PREGRASP, after the gripper has already opened. Catching it up front
        turns that into one line of text instead of a half-executed cycle.

        The node reads tool_frame once at construction, so this cannot be fixed
        at runtime: it needs a relaunch of the robot.
        """
        if self.get_parameter('pipeline_id').value != 'cumotion':
            return True
        ee_link = self.planner_ee_link()
        if ee_link is DEAD_PLANNER:
            # Worth a hard stop rather than a warning: with no planner every
            # pose goal fails, so the cycle would walk the whole retry ladder
            # and report "every pick strategy was exhausted" -- which reads as
            # a grasping problem when the planner simply is not running. It has
            # crashed here before (SIGFPE, exit code -8), so this is a state
            # the stack really does reach.
            self.get_logger().error(
                '/cumotion_planner is not running -- nothing can be planned. '
                'Check the launch output for "cumotion_goal_set_planner_node ... '
                'process has died", and restart the robot.')
            return False
        if ee_link is None:
            self.get_logger().warn(
                'could not determine cuMotion\'s ee_link; continuing without the '
                'check. A mismatch shows up as INVALID_LINK_NAME at PREGRASP.')
            return True
        if ee_link != self.tcp_frame:
            self.get_logger().error(
                f'cuMotion takes Cartesian goals for {ee_link}, but this cycle '
                f'needs {self.tcp_frame}. Relaunch the robot with '
                f'tool_frame:={self.tcp_frame} '
                '(native/run_launch_everything.sh '
                f'tool_frame:={self.tcp_frame}), or run this with arm:='
                f'{"left" if self.arm == "right" else "right"}.')
            return False
        self.get_logger().info(f'cuMotion plans Cartesian goals for {ee_link}')
        return True

    # -- perception ----------------------------------------------------------

    def detect(self, min_count=1):
        """Wait for a detection message newer than now and matching our prompt."""
        self.prompt_pub.publish(String(data=self.prompt))
        started = self.get_clock().now().nanoseconds * 1e-9
        timeout = self.get_parameter('detect_timeout').value
        deadline = started + timeout
        while not self._abort.is_set():
            with self._lock:
                payload = self._latest_detections
            fresh = (payload is not None
                     and payload.get('stamp', 0.0) > started
                     and payload.get('prompt') == self.prompt)
            if fresh and len(payload.get('detections', [])) >= min_count:
                return payload
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                if fresh:
                    return payload          # fresh but empty: nothing was seen
                self.get_logger().error(
                    'no fresh /vlm/detections -- is VLM/run_vlm_detector.sh running?')
                return None
            self._abort.wait(0.1)
        return None

    def tcp_position(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as exc:
            self.get_logger().warn(f'no {self.tcp_frame} transform: {exc}')
            return None
        t = tf.transform.translation
        return (t.x, t.y, t.z)

    # -- grasp geometry ------------------------------------------------------

    def _describe_side(self, detection, payload=None):
        """Where the object is, in whichever terms the decision was made in."""
        size = (payload or {}).get('image_size')
        centre = detection.get('center_px')
        if size and centre and len(size) == 2 and size[0]:
            half = 'left' if centre[0] < size[0] / 2.0 else 'right'
            return (f'at pixel column {centre[0]} of {size[0]}, '
                    f'the {half} half of the frame')
        return f'at y={detection["point"][1]:+.3f}'

    def arm_for(self, detection, payload=None):
        """Which arm should pick this object: the half of the camera it is in.

        Left half of the frame -> left arm, right half -> right arm, split at
        the middle column. The two agree with the world-y test anyway: the
        camera is pitched about world Y with no yaw, so optical +x (image
        right) maps to world -y, which is the right arm's side. Splitting on
        pixels is what was asked for and it is what you can see on
        /vlm/debug_image, so the decision is checkable by looking.

        Falls back to world y when the frame width is unknown -- an older
        detector that does not publish image_size, or a hand-made payload.
        """
        size = (payload or {}).get('image_size')
        centre = detection.get('center_px')
        if size and centre and len(size) == 2 and size[0]:
            middle = size[0] / 2.0 + self.get_parameter('arm_split_px').value
            return 'left' if centre[0] < middle else 'right'

        split = self.get_parameter('arm_split_y').value
        return 'left' if detection['point'][1] > split else 'right'

    def solve_ik(self, position, quat, seed, max_jump=None,
                 avoid_collisions=True, group=None, joints=None):
        """Joints for this pose, staying near `seed`.

        The seed is what makes this useful: KDL starts its search from the
        seeded state, so passing the posture the arm is already in returns a
        solution in the same branch instead of an arbitrary one. A solution
        that still moves a joint more than max_joint_jump is rejected -- that
        is a reconfiguration, not a descent.

        Returns None if IK is unavailable, failed, or only offered a flip.
        """
        if not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                '/compute_ik unavailable -- is move_group running?')
            return None

        group = group or self.group
        joint_names = joints or self.arm_joints
        arm = group.replace('_arm', '')

        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = group
        ik.ik_link_name = f'openarm_{arm}_hand_tcp'
        ik.avoid_collisions = avoid_collisions
        timeout = self.get_parameter('ik_timeout').value
        ik.timeout.sec = int(timeout)
        ik.timeout.nanosec = int((timeout % 1.0) * 1e9)
        ik.robot_state.joint_state.name = list(joint_names)
        ik.robot_state.joint_state.position = [float(v) for v in seed]

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        (pose.pose.position.x, pose.pose.position.y,
         pose.pose.position.z) = position
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = quat
        ik.pose_stamped = pose

        result = self._await(self.ik_client.call_async(request), 10.0)
        if result is None:
            self.get_logger().warn('/compute_ik did not answer')
            return None
        if result.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().info(
                f'no IK for ({position[0]:.3f}, {position[1]:.3f}, '
                f'{position[2]:.3f}): error code {result.error_code.val}')
            return None

        found = dict(zip(result.solution.joint_state.name,
                         result.solution.joint_state.position))
        missing = [j for j in joint_names if j not in found]
        if missing:
            self.get_logger().warn(f'IK solution is missing {missing}')
            return None
        solution = [found[j] for j in joint_names]

        limit = (self.get_parameter('max_joint_jump').value
                 if max_jump is None else max_jump)
        worst = max(abs(a - b) for a, b in zip(solution, seed))
        if limit > 0.0 and worst > limit:
            self.get_logger().info(
                f'rejecting an IK solution that moves a joint {worst:.3f} rad '
                f'(limit {limit:.3f}): that is a reconfiguration, not a '
                f'descent')
            return None
        return solution

    def reachable(self, position, quat, arm=None):
        """Can this arm put its tool there? True, False, or None for unknown.

        Two solvers get asked, in cheapness order, and this is the whole point:

        KDL, through /compute_ik, is quick and its "yes" is conclusive. Its
        "no" is not. KDL is a randomly-seeded iterative solver, it was
        configured here with a 5 ms timeout, and it fails on poses that are
        perfectly reachable -- measured: 0 solutions in 10 tries at a point the
        arm can actually pick from, at both a 1 s and a 5 s request timeout.
        Treating that as the verdict refused objects the robot could pick.

        So on a KDL "no", cuMotion is asked with a plan_only request. cuMotion
        is the authority because cuMotion is what executes: cuRobo's IK is far
        stronger than KDL at 5 ms, and if it can plan there, the arm can go
        there. plan_only means nothing moves.

        None -- neither solver could be asked -- is deliberately distinct from
        False. Reporting "out of reach" because a service did not answer is how
        a stack that is merely down looks like a workspace problem.
        """
        arm = arm or self.arm
        joints = [f'openarm_{arm}_joint{i}' for i in range(1, 8)]
        seed = [0.0] * 7
        if arm == self.arm:
            self.await_joint_states()
            with self._lock:
                measured = [self._arm_positions.get(j) for j in self.arm_joints]
            if all(v is not None for v in measured):
                seed = measured

        asked = False
        if self.ik_client.service_is_ready() or \
                self.ik_client.wait_for_service(timeout_sec=2.0):
            asked = True
            attempts = max(1, int(self.get_parameter('ik_attempts').value))
            for _ in range(attempts):
                if self.solve_ik(position, quat, seed, max_jump=0.0,
                                 avoid_collisions=False, group=f'{arm}_arm',
                                 joints=joints) is not None:
                    return True

        verdict = self.planner_can_reach(position, quat, arm)
        if verdict is not None:
            return verdict
        return False if asked else None

    def planner_can_reach(self, position, quat, arm):
        """Ask the planner that will actually execute. Plans only, never moves.

        Returns None if it could not be asked -- a Cartesian goal only works
        for the arm cuMotion's ee_link points at, and for the other arm the
        answer is INVALID_LINK_NAME, which says nothing about reach.
        """
        if self.get_parameter('pipeline_id').value != 'cumotion':
            return None
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            return None

        request = self._base_request()
        request.group_name = f'{arm}_arm'
        request.goal_constraints = [
            self.pose_constraints(position, quat, f'openarm_{arm}_hand_tcp')]
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = True          # nothing moves
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        handle = self._await(self.move_client.send_goal_async(goal), 15.0)
        if handle is None or not handle.accepted:
            return None
        result = self._await(handle.get_result_async(), 30.0)
        if result is None:
            return None
        code = result.result.error_code.val
        if code == MOVEIT_SUCCESS:
            return True
        if code in (MoveItErrorCodes.INVALID_LINK_NAME,
                    MoveItErrorCodes.INVALID_GROUP_NAME):
            # cuMotion is pointed at the other arm; not a statement about reach.
            return None
        return False

    def out_of_reach(self, points, quat):
        """The first of `points` this arm cannot reach, with what to do.

        Asked before anything moves. An object outside the envelope fails every
        goal identically and no retry strategy recovers it -- a different yaw
        or 8 mm lower is still out of reach -- so the ladder is worse than
        useless here: it is six identical failures and a report that blames the
        planner.
        """
        for label, point in points:
            verdict = self.reachable(point, quat)
            if verdict is None:
                self.get_logger().warn(
                    'could not determine whether the object is in reach; '
                    'continuing without the check')
                return None
            if verdict:
                continue

            message = (
                f'out of reach: the {self.arm} arm cannot put its tool at the '
                f'{label} ({point[0]:+.3f}, {point[1]:+.3f}, {point[2]:+.3f}). '
                f'Not moving.')
            other = other_arm(self.arm)
            if self.reachable(point, quat, arm=other):
                message += (f' The {other} arm can reach it -- run with '
                            f'arm_selection:=by_side, or arm:={other}.')
            else:
                message += (' Neither arm can. Move the object closer, or '
                            'check it with:  python3 '
                            f'native/tests/check_reachability.py --arm '
                            f'{self.arm} --point {point[0]:.3f} '
                            f'{point[1]:.3f} {point[2]:.3f}')
            return message
        return None

    def cartesian_move(self, position, quat, label, avoid_collisions=True):
        """Move the tool in a straight line to `position`. False if it cannot.

        move_group interpolates the line itself and runs IK at every
        cartesian_step, so the path is Cartesian by construction rather than by
        hope. It reports the fraction it managed; anything short of
        cartesian_min_fraction is refused rather than executed, because a
        descent that stops at 60% leaves the gripper closing on air.

        The returned trajectory is not speed-scaled by the service, so it is
        re-timed here to velocity_scaling -- otherwise the approach runs at
        full joint speed, which is the opposite of what a careful descent
        wants.
        """
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(
                '/compute_cartesian_path unavailable -- is move_group running?')
            return False

        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.group_name = self.group
        request.link_name = self.tcp_frame
        request.max_step = self.get_parameter('cartesian_step').value
        request.jump_threshold = 0.0
        request.avoid_collisions = avoid_collisions
        request.start_state.is_diff = True

        target = PoseStamped().pose
        target.position.x, target.position.y, target.position.z = position
        (target.orientation.x, target.orientation.y,
         target.orientation.z, target.orientation.w) = quat
        request.waypoints = [target]

        result = self._await(self.cartesian_client.call_async(request), 20.0)
        if result is None:
            self.get_logger().warn(f'{label}: /compute_cartesian_path did not answer')
            self.log_motion(label, 'cartesian', 'no-answer',
                            target=[round(v, 5) for v in position],
                            checked=avoid_collisions)
            return False
        if result.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().info(
                f'{label}: no straight line ({result.error_code.val})')
            self.log_motion(label, 'cartesian', 'refused',
                            target=[round(v, 5) for v in position],
                            checked=avoid_collisions,
                            code=result.error_code.val)
            return False

        wanted = self.get_parameter('cartesian_min_fraction').value
        if result.fraction < wanted:
            self.get_logger().info(
                f'{label}: only {result.fraction * 100:.0f}% of the straight '
                f'line was solvable (need {wanted * 100:.0f}%'
                f'{", collision-checked" if avoid_collisions else ""})')
            if result.fraction <= 0.0 and not avoid_collisions:
                # Not "the line is blocked" -- not even the first 5 mm step was
                # accepted, with nothing being checked. That points at the
                # start state rather than the path: a joint at its limit, or a
                # posture move_group considers invalid.
                self.get_logger().warn(
                    f'{label}: nothing at all was solvable with collision '
                    f'checking off, which points at the start state rather '
                    f'than the path -- check the joints against their limits '
                    f'here')
            self.log_motion(label, 'cartesian', 'short',
                            target=[round(v, 5) for v in position],
                            checked=avoid_collisions,
                            fraction=round(result.fraction, 4))
            return False

        trajectory = self._retime(result.solution)
        if not trajectory.joint_trajectory.points:
            self.get_logger().warn(f'{label}: empty Cartesian trajectory')
            return False

        points = len(trajectory.joint_trajectory.points)
        self.get_logger().info(
            f'{label}: straight line to ({position[0]:.3f}, {position[1]:.3f}, '
            f'{position[2]:.3f}), {points} points')
        before = self.joint_snapshot()
        # The commanded path, as move_group interpolated it.
        planned = [
            {'t': round(p.time_from_start.sec
                        + p.time_from_start.nanosec * 1e-9, 3),
             'joints': [round(v, 5) for v in p.positions]}
            for p in trajectory.joint_trajectory.points]
        finish = self.sample_motion()
        ok = self._execute(trajectory, label)
        self.log_motion(label, 'cartesian', 'ok' if ok else 'execution-failed',
                        target=[round(v, 5) for v in position],
                        checked=avoid_collisions,
                        fraction=round(result.fraction, 4),
                        points=points, before=before,
                        planned=planned, path=finish())
        return ok

    def _retime(self, trajectory):
        """Scale a trajectory in time to velocity_scaling.

        A pure time scaling: t/s, v*s, a*s^2. compute_cartesian_path has no
        scaling field, so without this the descent runs at full joint speed.
        """
        scale = max(1e-3, min(1.0, self.get_parameter('velocity_scaling').value))
        for point in trajectory.joint_trajectory.points:
            seconds = (point.time_from_start.sec
                       + point.time_from_start.nanosec * 1e-9) / scale
            point.time_from_start.sec = int(seconds)
            point.time_from_start.nanosec = int((seconds % 1.0) * 1e9)
            point.velocities = [v * scale for v in point.velocities]
            point.accelerations = [a * scale * scale for a in point.accelerations]
        return trajectory

    def _execute(self, trajectory, label):
        if not self.execute_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('/execute_trajectory unavailable')
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        handle = self._await(self.execute_client.send_goal_async(goal), 15.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: trajectory rejected')
            return False
        result = self._await(handle.get_result_async(),
                             self.get_parameter('motion_timeout').value)
        if result is None:
            self.get_logger().error(f'{label}: trajectory timed out')
            return False
        code = result.result.error_code.val
        if code != MOVEIT_SUCCESS:
            self.get_logger().error(f'{label}: execution failed, code {code}')
            return False
        return True

    def column_heights(self, from_z, to_z):
        """The z values to visit, from just below from_z down to exactly to_z.

        Hops are at most descend_step; the last lands exactly on to_z, so
        rounding cannot leave the tool short of the grasp or low on a retreat.
        """
        step = self.get_parameter('descend_step').value
        span = abs(to_z - from_z)
        if step <= 0.0 or span <= step:
            return [to_z]
        count = int(math.ceil(span / step))
        direction = -1.0 if to_z < from_z else 1.0
        return [to_z if i == count else from_z + direction * step * i
                for i in range(1, count + 1)]

    def descend_column(self, xy, from_z, to_z, quat, label,
                       uncheck_collisions=False, linear_only=False):
        """Move along the vertical line above the object, in short hops.

        Waypoints share x and y and differ only in z, so the tool tracks the
        line above the object rather than taking whatever curve the optimiser
        prefers between two distant poses. Used for both the descent and the
        retreat -- a retreat that bows is how a held object gets dragged across
        whatever it was picked up from.

        Sent as *joint* goals from seeded IK, not as pose goals. Seven joints
        for a 6-DOF pose means a tool pose does not determine a posture, so a
        pose goal leaves the planner free to reconfigure the whole arm in order
        to lower the tool a few centimetres -- which it does. Chaining the IK
        seed from one waypoint to the next pins the solution to the branch the
        arm is already in, so the descent is the same posture, lower.

        Three mechanisms, best first:

        1. A real Cartesian line from /compute_cartesian_path. move_group
           interpolates it and runs IK at every step, so the tool goes
           straight down by construction. This is the one that matters -- a
           goal pose says where to end up, not how to get there, and a 5 cm
           descent planned as a free trajectory can bow into the table.
        2. Failing that, joint goals from seeded IK at intermediate heights.
        3. Failing that, pose goals -- the old behaviour, which can wander.
        """
        target = (xy[0], xy[1], to_z)
        if self.get_parameter('linear_descent').value:
            if self.cartesian_move(target, quat, label):
                return True
            # A collision-checked line can fail for a reason that is not an
            # obstacle at all: on the final approach the *target* is in the
            # collision world. The octomap holds the object being grasped and
            # the table under it, so the gripper is required to enter voxels on
            # its way to the thing it is picking up. Measured: the 20 cm
            # descent to the pre-grasp solved 100%, the last 5 cm onto the
            # object solved 25% -- about a centimetre before the gripper met
            # the object's own voxels.
            #
            # Every MoveIt pick pipeline handles this the same way: the
            # approach and retreat are exempt from collision checking. It is a
            # short straight vertical move, along a line whose ends were
            # already reach-checked, with the gripper open and min_grasp_z as a
            # hard floor -- and the alternative is what happens below, a
            # free-space plan that took the arm into the table.
            if uncheck_collisions:
                self.get_logger().info(
                    f'{label}: retrying the straight line without collision '
                    f'checking -- the object being grasped is itself in the '
                    f'octomap')
                if self.cartesian_move(target, quat, label,
                                       avoid_collisions=False):
                    return True
            if linear_only:
                # Refuse rather than substitute a curved path. Measured on this
                # robot: a descent whose straight line solved only 37.5% --
                # identically with collision checking on and off, so the arm
                # simply runs out of reach part-way down -- was replaced by
                # three free-space hops that swung the tool 3.7 cm sideways and
                # then aborted with CONTROL_FAILED against the table. A
                # curved descent is worse than no descent.
                self.get_logger().error(
                    f'{label}: no straight line available, and a curved '
                    f'descent is not an acceptable substitute -- refusing. '
                    f'A line that fails identically with collision checking '
                    f'on and off means the arm runs out of reach along it, '
                    f'not that something is in the way: bring the object '
                    f'closer.')
                self.log_motion(label, 'cartesian', 'refused-no-line',
                                target=[round(v, 5) for v in target])
                return False
            self.get_logger().warn(
                f'{label}: no straight line available, stepping instead')

        heights = self.column_heights(from_z, to_z)
        total = len(heights)

        if self.get_parameter('seeded_descent').value:
            joints = self.solve_column(xy, heights, quat)
            if joints is not None:
                for index, (z, target) in enumerate(zip(heights, joints), 1):
                    if not self._move_to_joints(
                            target, f'{label} {index}/{total} z={z:.3f}'):
                        return False
                return True
            self.get_logger().warn(
                f'{label}: falling back to pose goals, so the planner may '
                f'change posture on the way')

        for index, z in enumerate(heights, 1):
            if not self.move_to_pose((xy[0], xy[1], z), quat,
                                     f'{label} {index}/{total}'):
                return False
        return True

    def solve_column(self, xy, heights, quat):
        """Joints for every height, each seeded from the one above it.

        All or nothing: a column that is half joint goals and half pose goals
        could still flip at the seam, which is the thing being avoided.
        """
        if not self.await_joint_states():
            self.get_logger().warn('no joint states, so IK cannot be seeded')
            return None
        with self._lock:
            seed = [self._arm_positions.get(j) for j in self.arm_joints]
        if any(v is None for v in seed):
            return None

        solved = []
        for z in heights:
            joints = self.solve_ik((xy[0], xy[1], z), quat, seed)
            if joints is None:
                return None
            solved.append(joints)
            seed = joints           # next waypoint continues from this one
        return solved

    def implausible_detection(self, detection):
        """Why this detection cannot be a real object here, or None if it can.

        The depth stream is the usual culprit, and it fails loudly rather than
        subtly: a missing or stale depth frame turns into a point metres away
        and well below the work surface. Catching that here costs one log line;
        not catching it costs the whole retry ladder, because every goal to an
        unreachable pose fails the same way and the failure summary then reads
        as a planning problem.
        """
        point = detection.get('point')
        if not point or len(point) != 3:
            return f'detection has no usable point: {point!r}'
        x, y, z = point

        if not all(math.isfinite(v) for v in (x, y, z)):
            return f'detection point is not finite: {point}'

        radius = self.get_parameter('workspace_radius').value
        horizontal = math.hypot(x, y)
        if horizontal > radius:
            return (f'the detected object is {horizontal:.2f} m from the base '
                    f'in x-y, outside the {radius:.2f} m workspace_radius. '
                    f'Point {[round(v, 3) for v in point]}, depth '
                    f'{detection.get("depth_m")} m from '
                    f'{detection.get("depth_px")} px -- that is a depth '
                    f'reading, not a reach problem. Check that the depth '
                    f'stream is alive:  ros2 topic hz '
                    f'/camera/camera/depth/image_rect_raw')

        min_z = self.get_parameter('min_grasp_z').value
        slack = self.get_parameter('max_z_clamp').value
        if z < min_z - slack:
            return (f'the detected object is at z={z:.3f}, more than '
                    f'{slack:.3f} m below min_grasp_z {min_z:.3f} -- the depth '
                    f'is wrong, so x and y cannot be trusted either. Depth '
                    f'{detection.get("depth_m")} m from '
                    f'{detection.get("depth_px")} px. Check the depth stream:  '
                    f'ros2 topic hz /camera/camera/depth/image_rect_raw')

        max_z = self.get_parameter('max_grasp_z').value
        if z > max_z:
            return (f'the detected object is at z={z:.3f}, above max_grasp_z '
                    f'{max_z:.3f} -- nothing on the work surface is that high')
        return None

    def grasp_from_detection(self, detection, strategy):
        point = detection['point']
        axis_yaw = detection.get('axis_yaw')
        if axis_yaw is None:
            axis_yaw = 0.0
            self.get_logger().warn('no object axis from the VLM; assuming yaw 0')
        yaw = axis_yaw + strategy['yaw_offset']

        grasp_z = point[2] + self.get_parameter('grasp_z_offset').value \
            + strategy['z_offset']
        min_z = self.get_parameter('min_grasp_z').value
        if grasp_z < min_z:
            self.get_logger().warn(
                f'grasp z {grasp_z:.3f} below min_grasp_z {min_z:.3f}; clamping')
            grasp_z = min_z

        grasp = (point[0], point[1], grasp_z)
        pregrasp = (point[0], point[1],
                    grasp_z + self.get_parameter('approach_height').value)
        return grasp, pregrasp, top_down_quat(yaw), point

    # -- verification --------------------------------------------------------

    def verify_grasp(self, pick_point):
        """Two independent checks, because either alone has a blind spot.

        The finger check misses the case where a fingertip stalls on the
        object's edge without capturing it; the re-detection check misses when
        the object is occluded by the gripper. Requiring the finger check and
        no-object-left-behind together is the combination that actually
        distinguishes "held" from "closed on air".
        """
        finger = self.finger_position()
        lo = self.get_parameter('grasp_finger_min').value
        hi = self.get_parameter('grasp_finger_max').value
        if finger is None:
            self.get_logger().warn('no finger feedback on /joint_states')
            finger_ok = False
        else:
            finger_ok = lo < finger < hi
            self.get_logger().info(
                f'finger {finger:.4f} in ({lo:.4f}, {hi:.4f}) -> '
                f'{"holding" if finger_ok else "empty"}')
        if not finger_ok:
            return False

        payload = self.detect(min_count=0)
        if payload is None:
            self.get_logger().warn('no detection for grasp check; trusting fingers')
            return True

        detections = payload.get('detections', [])
        eps = self.get_parameter('object_moved_eps').value
        left_behind = [d for d in detections if dist(d['point'], pick_point) < eps]
        if left_behind:
            self.get_logger().warn(
                f'object still at the pick point ({dist(left_behind[0]["point"], pick_point):.3f} m) '
                '-- grasp failed')
            return False

        tcp = self.tcp_position()
        if tcp is not None and detections:
            radius = self.get_parameter('object_hold_radius').value
            nearest = min(dist(d['point'], tcp) for d in detections)
            if nearest < radius:
                self.get_logger().info(f'object {nearest:.3f} m from the tool: held')
                return True
            self.get_logger().info(
                f'nearest detection {nearest:.3f} m from the tool; the gripper is '
                'probably occluding it -- trusting fingers')
        return True

    def verify_place(self, reference=None):
        """Was the object seen near where it was released?

        `reference` is where the release actually happened -- the tool position
        in place_mode "ready", place_position otherwise. Advisory either way: a
        box occludes what is inside it.
        """
        if reference is None:
            reference = list(self.get_parameter('place_position').value)
        payload = self.detect(min_count=0)
        if payload is None:
            return False
        radius = self.get_parameter('place_radius').value
        for d in payload.get('detections', []):
            if dist(d['point'][:2], reference[:2]) < radius:
                self.get_logger().info('object seen where it was released')
                return True
        self.get_logger().warn(
            'object not seen at the release point -- it may be occluded, or it '
            'rolled after the drop')
        return False

    # -- the cycle -----------------------------------------------------------

    def _run_cycle(self):
        try:
            self._cycle()
        except Exception as exc:
            self.get_logger().error(f'cycle crashed: {exc}')
            self._set_state('FAILED', str(exc))
        finally:
            with self._lock:
                self._busy = False

    def _cycle(self):
        self._failures = []
        self._holding = False
        self._last_detection = None
        self._last_payload = None
        # Every cycle starts from the launch arm, so a switch on the previous
        # cycle does not decide this one.
        self.configure_arm(self.launch_arm)

        states = self.load_states()
        if not self.check_states(states):
            self._set_state('FAILED', 'recorded states are missing')
            return
        self.add_table()

        # The planner first, before any motion at all. It is a precondition for
        # every goal in the cycle, and finding out after the arm has moved is
        # both wasted motion and a misleading report.
        if not self.check_planner_tool_frame():
            self._set_state('FAILED', 'the planner is unusable for this arm')
            return

        # Decide the arm before moving anything.
        #
        # The camera needs an unobstructed view to detect from, so the arm goes
        # to READY first if it is not already somewhere clear -- but that is
        # one move, not the whole approach. Which arm, and whether either can
        # reach, are settled on that detection while the robot stands still.
        # Doing it the other way round -- drive to a staging pose, then find
        # out the object is outside the envelope -- wastes the motion and
        # reports the wrong cause.
        # HOME first, and HOME is the only observation pose. There used to be
        # a separate READY that held the arm out over the table so the camera
        # could see it -- which is precisely why the map kept getting the arm
        # in it. One pose, out of the camera's frame, used for the map, the
        # detection and the rest position.
        if not self.arrive_at_home():
            self._set_state('FAILED', 'could not reach home')
            return

        if self.arm_selection == 'by_side':
            chosen_from = self.arm
            states = self.choose_arm(states)
            if states is None or states is OUT_OF_REACH:
                return
            # Only if it actually switched. Re-homing regardless would clear
            # and rebuild the octomap a second time every cycle for nothing.
            if self.arm != chosen_from and not self.arrive_at_home():
                self._set_state('FAILED', f'could not reach the {self.arm} home')
                return

        # Again, because the arm may have changed since the first check and
        # cuMotion accepts Cartesian goals for one link per bringup.
        if not self.check_planner_tool_frame():
            self._set_state('FAILED', 'the planner is unusable for this arm')
            return

        picked_point = None
        for attempt, strategy in enumerate(STRATEGIES):
            if self._abort.is_set():
                self._set_state('ABORTED')
                return
            self._set_state('ATTEMPT',
                            f'{attempt + 1}/{len(STRATEGIES)} ({strategy["name"]})')
            if attempt > 0:
                # Home only when this attempt needs a fresh look. A yaw or
                # height variant retries from where the arm already is.
                if strategy['refresh_octomap']:
                    if not self.arrive_at_home():
                        continue
                elif strategy['redetect'] and not self.move_to_home():
                    continue
            picked_point = self._attempt_pick(strategy, states)
            if picked_point is OUT_OF_REACH:
                # No strategy recovers this: a different yaw or 8 mm lower is
                # still outside the envelope. The state already says why.
                return
            if picked_point is not None:
                break
        else:
            self._set_state(
                'FAILED',
                f'every pick strategy was exhausted -- {self._failure_summary()}')
            self.move_to_home()
            return

        # Straight from the lift to the drop, carrying. No detour home:
        # capture_octomap refuses while holding -- the payload would be mapped
        # as an obstacle that then travels with the tool -- so going home would
        # clear the map, fail to replace it, and leave the drop planning
        # against nothing.
        #
        # attach_object() has already put the payload in the planning scene, so
        # these moves are planned with the object on the tool rather than with
        # an invisible thing swinging through the octomap.
        if not self._place(states):
            self._set_state('FAILED', 'place failed')
            return

        # Back the way it came: the staging pose, then home. PRE_PICK is a
        # posture the arm is known to be able to reach from both the drop and
        # from home, which is what makes it a safe waypoint out.
        self._set_state('PRE_PICK', 'on the way back')
        if not self.move_to_state(PRE_PICK_STATE, states):
            self.get_logger().warn(
                'could not stage through pre_pick on the way back; going '
                'straight home')
        self._set_state('HOME', 'cycle complete')
        self.move_to_home()
        self._set_state('DONE')

    def arm_candidates(self, detection, payload):
        """The arms to consider, best first.

        camera_half puts the half the object is in first and the other arm
        second. The orders only disagree when *both* arms can reach, and then
        the near arm wins -- a cross-body reach is slower, more likely to clip
        the other arm, and nearer its joint limits.
        """
        order = self.get_parameter('arm_order').value
        if order == 'right_then_left':
            return ['right', 'left']
        if order == 'left_then_right':
            return ['left', 'right']
        preferred = self.arm_for(detection, payload)
        return [preferred, other_arm(preferred)]

    def choose_arm(self, states):
        """Pick the arm that can actually reach the object, before moving.

        Runs once, on the detection taken when the prompt arrives, and moves
        nothing: reachability is a kinematics question and asking it costs a
        service call, not a trajectory. This is the whole point -- driving an
        arm to a staging pose and only then discovering the object is outside
        its envelope wastes the motion and reports the wrong cause.

        Candidates are tried in arm_candidates() order and the first that can
        reach the object gets it. Returns the states for that arm, None on a
        detection failure, or OUT_OF_REACH when no arm can reach it.
        """
        self._set_state('LOCATE', f'{self.prompt} (choosing an arm)')
        payload = self.detect(min_count=1)
        if payload is None or not payload.get('detections'):
            self._note_failure('LOCATE', (
                f'nothing matched "{self.prompt}", so there is no side to '
                'choose from'))
            self._set_state('FAILED', f'nothing matched "{self.prompt}"')
            return None

        detection = payload['detections'][0]
        reason = self.implausible_detection(detection)
        if reason is not None:
            self._note_failure('LOCATE', reason)
            self._set_state('FAILED', reason)
            return None

        where = self._describe_side(detection, payload)
        candidates = self.arm_candidates(detection, payload)
        self.get_logger().info(
            f'object {where}; trying {" then ".join(candidates)}')

        grasp, pregrasp, quat, _ = self.grasp_from_detection(
            detection, STRATEGIES[0])
        transit = (grasp[0], grasp[1],
                   grasp[2] + self.get_parameter('transit_height').value)
        targets = [('grasp', grasp), ('pre-grasp', pregrasp),
                   ('transit height', transit)]

        unreachable = []
        for arm in candidates:
            verdicts = [self.reachable(point, quat, arm=arm)
                        for _, point in targets]
            if any(v is None for v in verdicts):
                # Nothing could answer. Refusing on that would turn a stack
                # that is merely not up into a workspace problem, so carry on
                # and let the motion itself be the test.
                self.get_logger().warn(
                    'could not determine which arm can reach the object; '
                    f'keeping the {self.arm} arm and trying')
                return states
            if all(verdicts):
                if arm != self.arm:
                    self._set_state('SWITCH_ARM', f'{self.arm} -> {arm} ({where})')
                    self.configure_arm(arm)
                    states = self.load_states()
                    if not self.check_states(states):
                        self._set_state(
                            'FAILED', f'no recorded states for the {arm} arm')
                        return None
                else:
                    self.get_logger().info(f'the {arm} arm can reach it')
                return states
            missed = [label for (label, _), ok in zip(targets, verdicts)
                      if not ok]
            unreachable.append(f'{arm} (cannot reach the {", ".join(missed)})')

        message = (
            f'out of reach: no arm can pick the object {where}. '
            f'Tried {"; ".join(unreachable)}. Both /compute_ik and a plan-only '
            f'cuMotion query said no, so nothing moved. The grip is asked to '
            f'be straight down, which costs reach. Map what is actually '
            f'reachable at this height with:  python3 '
            f'native/tests/check_reachability.py --arm {candidates[0]} '
            f'--sweep --z {grasp[2]:.3f}')
        self._note_failure('REACH', message)
        self._set_state('OUT_OF_REACH', message)
        return OUT_OF_REACH

    def _attempt_pick(self, strategy, states):
        if strategy.get('redetect', True) or self._last_detection is None:
            self._set_state('LOCATE', self.prompt)
            payload = self.detect(min_count=1)
            if payload is None or not payload.get('detections'):
                self._set_state('LOCATE', 'nothing detected')
                self._note_failure('LOCATE', (
                    f'nothing matched "{self.prompt}" -- check '
                    '/vlm/debug_image, and that the detector has finished '
                    'loading'))
                return None
            self._last_detection = payload['detections'][0]
            self._last_payload = payload
        else:
            self.get_logger().info(
                f'{strategy["name"]}: reusing the last detection, so no trip '
                f'home to look again')
            payload = self._last_payload

        detection = self._last_detection
        self.get_logger().info(
            f'target at {[round(v, 4) for v in detection["point"]]} '
            f'axis_yaw={detection.get("axis_yaw")} '
            f'depth={detection["depth_m"]} m from {detection["depth_px"]} px')

        # Checked before anything moves. An unreachable target is not worth a
        # ladder attempt: every goal fails identically and the report blames
        # the planner.
        reason = self.implausible_detection(detection)
        if reason is not None:
            self._set_state('LOCATE', 'detection rejected')
            self._note_failure('LOCATE', reason)
            return None

        grasp, pregrasp, quat, point = self.grasp_from_detection(detection, strategy)

        # Before anything moves. Out of reach is not a planning failure and no
        # strategy on the ladder recovers it, so it ends the cycle instead of
        # burning six identical attempts. Both ends of the column are checked:
        # a grasp the arm can touch but not approach from above is no use.
        transit = (grasp[0], grasp[1],
                   grasp[2] + self.get_parameter('transit_height').value)
        if self.get_parameter('check_reach').value:
            refusal = self.out_of_reach(
                [('grasp', grasp), ('pre-grasp', pregrasp),
                 ('transit height', transit)], quat)
            if refusal is not None:
                self._note_failure('REACH', refusal)
                self._set_state('OUT_OF_REACH', refusal)
                return OUT_OF_REACH

        # Staged through pre_pick_state rather than going straight at the
        # object: the observation pose is chosen to keep the arm out of the
        # camera's view, which is not necessarily a good posture to start a
        # descent from.
        self._set_state('PRE_PICK')
        if not self.move_to_state(PRE_PICK_STATE, states):
            self._note_failure('PRE_PICK', (
                f'could not plan to {PRE_PICK_STATE}; re-record it somewhere the '
                'arm can reach from the ready pose'))
            return None

        # End the long free move high above the object, then come down the
        # vertical line to the pre-grasp. Without this the approach to a point
        # 5 cm up is a single unconstrained plan that can arrive from the side
        # and low, and push the object away before the gripper is over it.
        transit_height = self.get_parameter('transit_height').value
        transit_z = grasp[2] + transit_height
        if transit_height > self.get_parameter('approach_height').value:
            self._set_state('TRANSIT')
            if not self.move_to_pose((grasp[0], grasp[1], transit_z), quat,
                                     'TRANSIT'):
                self._note_failure('TRANSIT', (
                    f'could not plan to {transit_height * 100:.0f} cm above '
                    f'({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) -- most '
                    f"often the object is out of the {self.arm} arm's reach. "
                    f'Verify with: python3 native/tests/check_reachability.py '
                    f'--arm {self.arm} --point {grasp[0]:.3f} {grasp[1]:.3f} '
                    f'{transit_z:.3f}'))
                return None
            self._set_state('PREGRASP')
            reached = self.descend_column(
                grasp, transit_z, pregrasp[2], quat, 'PREGRASP',
                uncheck_collisions=self.get_parameter(
                    'approach_ignores_octomap').value)
        else:
            self._set_state('PREGRASP')
            reached = self.move_to_pose(pregrasp, quat, 'PREGRASP')
        if not reached:
            # By far the most common real cause, and the one that used to hide
            # behind "every pick strategy was exhausted": the object is simply
            # outside the arm's envelope. Say where it was and what the limit is
            # instead of making the reader go and measure it.
            self._note_failure('PREGRASP', (
                f'could not plan to the pre-grasp above '
                f'({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) -- most often '
                f'the object is out of the {self.arm} arm\'s reach. Verify with: '
                f'python3 native/tests/check_reachability.py --arm {self.arm} '
                f'--point {pregrasp[0]:.3f} {pregrasp[1]:.3f} {pregrasp[2]:.3f}'))
            return None

        # Opened here rather than before the approach: the fingers are already
        # over the object, so an open gripper cannot catch anything on the way
        # in, and a gripper that fails to open costs no motion.
        self._set_state('OPEN_GRIPPER')
        if not self.open_gripper():
            return None

        self._set_state('DESCEND')
        if not self.descend_column(
                grasp, pregrasp[2], grasp[2], quat, 'DESCEND',
                uncheck_collisions=self.get_parameter(
                    'approach_ignores_octomap').value,
                linear_only=self.get_parameter('descend_linear_only').value):
            self._note_failure('DESCEND', (
                f'reached the pre-grasp but could not descend to '
                f'z={grasp[2]:.3f} -- the octomap may have the object itself in '
                'it, or the grasp is below the work surface'))
            return None

        self._set_state('CLOSE_GRIPPER')
        if not self.close_gripper_to_cap():
            return None

        # Straight back up, and well clear -- not just back to the pre-grasp.
        # The next move carries the object to the drop pose as a free-space
        # plan, and starting that 5 cm off the surface is what dragged the
        # gripper across the table. retreat_height defaults to transit_height,
        # so the object leaves at the same altitude the approach arrived at.
        retreat_z = grasp[2] + self.get_parameter('retreat_height').value
        self._set_state('LIFT')
        if not self.descend_column(
                grasp, grasp[2], retreat_z, quat, 'LIFT',
                uncheck_collisions=self.get_parameter(
                    'approach_ignores_octomap').value,
                linear_only=self.get_parameter('descend_linear_only').value):
            return None

        self._set_state('VERIFY_GRASP')
        if not self.verify_grasp(point):
            self._note_failure('VERIFY_GRASP', (
                'the gripper closed but nothing was held -- check '
                'grasp_finger_min against your object, and the grasp height'))
            self._set_state('VERIFY_GRASP', 'failed, retrying')
            self.open_gripper()
            return None

        self.attach_object()
        self._holding = True
        self._set_state('VERIFY_GRASP', 'confirmed')
        return point

    def _place(self, states):
        if self.place_mode == 'state':
            return self._place_at_state(states)
        if self.place_mode == 'home':
            return self._place_at_home()
        place = list(self.get_parameter('place_position').value)
        quat = top_down_quat(self.get_parameter('place_yaw').value)
        above = (place[0], place[1],
                 place[2] + self.get_parameter('place_approach_height').value)

        self._set_state('OVER_BOX')
        if not self.move_to_pose(above, quat, 'OVER_BOX'):
            return False
        if not self.move_to_pose(tuple(place), quat, 'PLACE'):
            return False

        self._set_state('RELEASE')
        if not self.open_gripper():
            return False
        self.detach_object()

        self._set_state('RETREAT')
        if not self.move_to_pose(above, quat, 'RETREAT'):
            return False

        self._set_state('VERIFY_PLACE')
        self.verify_place(place)     # advisory: the box usually occludes the object
        return True

    def _place_at_state(self, states):
        """Carry to the recorded drop_state and release there."""
        self._set_state('DROP', 'moving to the recorded drop pose')
        if not self.move_to_state(DROP_STATE, states):
            return False

        release_point = self.tcp_position()
        self._set_state('RELEASE', 'at drop_state')
        if not self.open_gripper():
            return False
        self.detach_object()

        self._set_state('VERIFY_PLACE')
        if release_point is None:
            self.get_logger().warn(
                f'no {self.tcp_frame} transform at release; skipping the check')
        else:
            self.verify_place(release_point)
        return True

    def _place_at_home(self):
        """Carry back to HOME and release there.

        No approach or retreat move: the pose is the drop point, so the object
        falls the distance between the tool and whatever is below it.
        """
        self._set_state('HOME', 'carrying the object back')
        if not self.move_to_home():
            return False
        release_point = self.tcp_position()

        self._set_state('RELEASE', 'at home')
        if not self.open_gripper():
            return False
        self.detach_object()

        self._set_state('VERIFY_PLACE')
        if release_point is None:
            self.get_logger().warn(
                f'no {self.tcp_frame} transform at release; skipping the check')
        else:
            self.verify_place(release_point)
        return True


def main():
    rclpy.init()
    node = PickPlaceOrchestrator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._abort.set()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
