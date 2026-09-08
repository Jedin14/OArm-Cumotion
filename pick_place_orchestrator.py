#!/usr/bin/env python3
"""VLM-guided pick and place for the 7DOF-OArm, with VLM-checked retries.

    HOME -> LOCATE -> PRE_PICK -> TRANSIT -> PREGRASP -> OPEN -> DESCEND
      ^                                                             |
      |                                                             v
      +------------- PRE_PICK <- CLEAR <- failed ------- VERIFY_GRASP <- CLOSE
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

A pick that *fails* leaves by the same door. Every failure below TRANSIT --
the descent stopping short, the jaws closing on nothing, the grasp not
verifying, the cycle crashing outright -- leaves the arm extended down at the
object, and PRE_PICK and HOME are joint goals: the short path in joint space
from reaching down over the table to folded at the side goes through the
table. So the first thing any exit does is CLEAR, straight back up the
descent line, gripper left exactly as it is. clear_the_surface() is a no-op
when the tool is already above the pre-grasp height, which on the successful
path it is.

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
import time

import rclpy
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
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
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath,
                             GetPlanningScene, GetPositionIK)
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

import arm_kinematics

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
# What MoveIt calls the octomap in the collision world and in the allowed
# collision matrix.
OCTOMAP_NAME = '<octomap>'
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
    # Deliberately does not go home. A retry that drives the arm back to HOME
    # and starts over is most of the wasted motion in a failed cycle, and the
    # object has not moved -- the last detection is still good.
    {'name': 'retry', 'yaw_offset': 0.0, 'z_offset': 0.0,
     'refresh_octomap': False, 'redetect': False},
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


def tilt_quat(quat, tilt, azimuth):
    """`quat` tipped `tilt` radians away from its own approach axis.

    The rotation is applied in the *world* frame, about a horizontal axis at
    `azimuth`, so it tips the tool's approach direction off vertical by exactly
    `tilt` while leaving the grasp otherwise as it was.

    Why this exists: a strictly vertical approach is six constraints on seven
    joints at a fixed point, and at the edge of the envelope there is often no
    solution clear of the joint stops -- measured, every one of 25 solutions
    against a limit. Letting the gripper come down at 10 or 20 degrees off
    vertical is a much larger solution set for a grasp that is, on most
    objects, just as good.
    """
    half = tilt / 2.0
    axis = (math.cos(azimuth) * math.sin(half),
            math.sin(azimuth) * math.sin(half),
            0.0,
            math.cos(half))
    return quat_mul(axis, quat)


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
        # 0.3 -> 0.4. A time dilation of a path already planned, so it
        # changes speed and not the route.
        #
        # It is a bigger lever on cycle time than it looks. _retime divides
        # every timestamp by this, so 0.3 stretches a trajectory 3.33x and
        # 0.4 stretches it 2.5x -- a quarter off every leg. Measured, run
        # 1788859355: TRANSIT wrote its first record 18.59 s after the state
        # began, all of it inside one cartesian_move call. Whether that was
        # the plan or the flight was not knowable from the log, which is what
        # plan_s and exec_s were added for; if it was the flight, this alone
        # takes it to about 14 s.
        self.declare_parameter('velocity_scaling', 0.4)
        self.declare_parameter('acceleration_scaling', 0.4)
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
        # 15 cm above the grasp, not 20. Lowered on request: the approach
        # ends closer to the object, so the single descent is a shorter line
        # and there is less of it to go wrong.
        self.declare_parameter('transit_height', 0.15)
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
        # The smallest share of a line worth flying. Anything at or above this
        # is executed and the remainder requested as a further line, so the
        # whole move stays straight.
        #
        # Refusing a partial outright was wrong: a leg whose line solved
        # 95.65% -- about 6 mm short of 150 mm -- was thrown away, replaced
        # with curved free-space hops, and the first hop aborted with
        # CONTROL_FAILED against the table. Flying 95.65% of a straight line
        # and then the last 6 mm is obviously better than not flying it.
        self.declare_parameter('cartesian_partial_min', 0.5)
        # How many such segments one leg may take before giving up.
        self.declare_parameter('cartesian_segments', 4)
        # Least distance a segment must gain, metres, before another is tried.
        # A line that stalls in the same place is stalling against something.
        self.declare_parameter('cartesian_min_gain', 0.002)
        # Distance below which a remaining gap is treated as the arm's
        # standing offset rather than as path left to fly. The measured offset
        # is 12-14 mm, so anything inside this is handed to settle_at and the
        # offset correction; beyond it, another straight segment is tried.
        self.declare_parameter('pose_residual_limit', 0.025)
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
        # How close the *tool* has to end up, metres, and how many passes are
        # allowed to get it there. A move MoveIt calls SUCCESS has satisfied
        # the joint controller's tolerance, which is a different thing: the
        # tool was landing 13.7 mm low at z=0.405 and 33.5 mm low at z=0.555.
        self.declare_parameter('pose_tolerance', 0.005)
        # Seconds to let the tool creep onto its target before judging it. The
        # servo's integral term is slow: held at one target the error went 19.3
        # -> 10.7 mm over about eleven seconds. Waiting is what closes it;
        # commanding again does not move the arm at all.
        # How far out the tool may settle before a leg counts as failed,
        # metres. Distinct from pose_tolerance, which is the accuracy worth
        # correcting: this is the distance past which the arm is not where it
        # was sent at all.
        #
        # Measured: a left-arm descent reported fraction=1.0 and settled
        # 287 mm from the target -- 272 of them in y -- and was recorded as
        # ok, after which the cycle would have closed the gripper there.
        #
        # 0.15 m, not something tighter, and deliberately so. Legs on this
        # robot routinely settle 25 to 52 mm out and still pick the object up
        # -- the one successful grasp so far landed 32.2 mm from its
        # commanded point. A 50 mm limit would have failed that. This is not
        # an accuracy standard; it is the distance past which the tool is
        # somewhere else entirely, and 0.15 m is the whole length of the
        # descent.
        self.declare_parameter('pose_abort_limit', 0.15)
        self.declare_parameter('pose_settle_time', 4.0)
        # Then, once, aim past the target by the measured error. Only useful
        # because that error is repeatable -- dz -13.8, -14.0, -14.0, -14.0 mm
        # across four passes at the same target.
        # Aim past the target by the measured error, once, after settling.
        #
        # Off. It was built for an offset that was repeatable and almost
        # purely vertical -- dz -13.8, -14.0, -14.0, -14.0 mm across four
        # passes -- and the error is no longer that. The record since:
        #
        #   TRANSIT  20.2 -> 16.6 mm   better
        #   TRANSIT  12.6 ->  8.5 mm   better
        #   TRANSIT  18.6 -> 21.5 mm   worse
        #   DESCEND  16.2 -> 31.7 mm   worse
        #   DESCEND  25.0 -> 42.0 mm   worse
        #   DESCEND  38.4 -> 59.7 mm   worse
        #
        # And on the approach it does specific harm beyond its own leg: aiming
        # past the transit point commanded a position 13.7 mm *higher* and
        # left the tool 17.6 mm out in y, so the descent then started from
        # somewhere the pre-flight had not checked and had to travel sideways
        # as well as down -- 52.7 mm out at the grasp.
        #
        # A few millimetres of error above the object is harmless: the descent
        # is a fresh line to the grasp. Overshooting to remove it is not.
        self.declare_parameter('pose_offset_correction', False)

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
        # Seconds between heartbeat records while a cycle is running. Without
        # these a stall is a silent gap in the file and there is no way to tell
        # a wedged planner from an arm crawling somewhere. 0 disables them.
        self.declare_parameter('motion_log_heartbeat', 1.0)
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
        # How far the jaws may be from the commanded grasp before the log
        # says so. Reporting only -- nothing refuses on it. 10 mm is about
        # the point where a jaw that spans 40 mm starts missing a small
        # object; the measured miss that gripped nothing was 29 mm.
        self.declare_parameter('grasp_miss_warn', 0.010)

        self.declare_parameter('detect_timeout', 15.0)
        # How old a detection may be and still be picked from without asking
        # the detector again. Set from the measured gap: choose_arm's answer
        # is a few seconds old when the first attempt wants it, while a
        # failed attempt takes nearer a minute, so 10 s reuses the one and
        # re-detects the other. 0 disables reuse.
        self.declare_parameter('detection_reuse_age', 10.0)
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
        # The "home" group state in openarm_bimanual.srdf is seven zeros,
        # which folds the arm down by the base and out of the camera's view of
        # the table. joint4 is the one departure from it: the URDF gives that
        # joint a lower limit of exactly 0.0, and the hardware stops 8.9
        # degrees short of it -- measured at 0.15583 rad, constant to five
        # decimals, while the trajectory controller's reference sat at 0.0.
        # Commanding zero there therefore asks for a posture the elbow cannot
        # hold: the move reports success or exhausts its retries depending on
        # the run, and every later "is the arm home?" check sees 0.156 rad of
        # error and refuses to start the cycle.
        #
        # 0.20 rad clears that floor. At a folded-down posture it is 11
        # degrees at the elbow -- the arm still hangs by the base, still out
        # of frame -- and it is a goal the joint can actually reach and hold.
        self.declare_parameter('home_joint_positions',
                               [0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0])
        self.declare_parameter('octomap_settle_time', 1.5)
        self.declare_parameter('home_pose_tolerance', 0.05)
        # How far a *settled* joint may stand off HOME and still count as
        # arrived, radians.
        #
        # Not slack for a sloppy move -- it is for the joint that physically
        # cannot reach its commanded value. Measured on this robot: the
        # right joint4 is commanded 0.0 by the "home" group state, the
        # trajectory controller's reference is 0.0, and the joint sits at
        # 0.15583 rad -- constant to five decimals over 301 samples, with
        # 0.02 Nm of effort. The URDF gives joint4 a lower limit of exactly
        # 0.0, so HOME asks the elbow to sit *on* its limit, and the hardware
        # stops 8.9 degrees short of it. joint4 tracks 0.70 to 2.15 rad
        # perfectly elsewhere in the same log, so this is a floor, not a
        # servo fault.
        #
        # With home_pose_tolerance alone the check could therefore never
        # pass: every cycle drove home, arrived as well as it can, and then
        # failed with "could not reach home" -- the arm at home, refusing to
        # move.
        self.declare_parameter('home_settle_tolerance', 0.20)
        # Speed below which a joint counts as stopped, rad/s. The standing
        # reading is +/-0.011 rad/s of noise.
        self.declare_parameter('joint_still_speed', 0.05)
        # Keep commanded postures this far off the position limits, radians.
        # A goal *on* a limit is what created the trouble above; it also gives
        # cuRobo's trajopt no room on that joint.
        self.declare_parameter('joint_limit_margin', 0.02)
        # How the approach above the object is aimed.
        #
        #   planner  a tool pose goal -- cuMotion or MoveIt picks the arm
        #            configuration. What this did before.
        #   tool     solve the same tool pose here and send the *roomiest*
        #            solution as a joint goal.
        #   wrist    solve for joint7's centre instead, leaving joint7 out of
        #            it, then tilt joint7 to point the hand down once the arm
        #            is above the object.
        #
        # Why it is not 'planner' by default. Both cuMotion and /compute_ik
        # return the first solution they find, and at the measured object
        # position almost every solution is jammed against a joint stop:
        # solving the exact tool pose the cycle asks for gives 25 solutions
        # from 48 seeds and the *roomiest of them* still has 0.000 rad of
        # headroom -- joint3 and joint5 both sitting on their limits, which is
        # the posture the robot actually used. A Cartesian path cannot
        # continue once a joint it needs is at its stop, so the straight-line
        # descent died part-way (fraction=0.8182, then 0.0) and the fallback
        # flew curves. Choosing the roomiest solution instead of the first is
        # what fixes that, and it applies to either frame.
        #
        # Why 'tool' rather than 'wrist'. Measured at the same target, the
        # two are within a few thousandths of each other once the tilt joint7
        # needs is accounted for -- 0.172 rad against 0.175 at the pre-grasp,
        # 0.247 against 0.240 at the transit point. The wrist partition
        # therefore buys no measurable headroom, and it costs: the grasp yaw
        # has to be pinned separately (it is not part of a wrist point), the
        # tilt has to be solved and checked, and the postures it likes best
        # are ones joint7 cannot tilt into line at all. The tool pose asks for
        # what is actually wanted in one statement.
        self.declare_parameter('approach_frame', 'tool')
        # A parallel gripper grips the same either way round, so the grasp
        # yaw and the yaw turned by 180 degrees are the same grasp -- but they
        # are very different postures. Trying both took the roomiest solution
        # from 0.000 to 0.172 rad on its own.
        self.declare_parameter('grasp_jaw_flip', True)
        # Leave the grasp yaw to the solver instead of pinning it to the
        # object's axis. Off, and it should stay off for anything that is not
        # round: the jaws close along the tool frame's y, which is link6's
        # y-axis, and joint7 cannot move it -- so an unconstrained solve picks
        # the closing angle arbitrarily. Measured consequence: DESCEND
        # fraction=1.0 onto a point 6.3 mm from target, then the gripper shut
        # to 0 mm at 1.14 Nm against a 2.5 Nm cap. It closed beside the
        # screwdriver, not across it.
        self.declare_parameter('grasp_yaw_free', False)
        # How far off vertical the gripper may come down, radians.
        #
        # A strictly top-down grasp is six constraints on seven joints at a
        # fixed point, and near the edge of the envelope there is frequently no
        # solution clear of the joint stops at all -- measured, all 25 of them
        # against a limit. 0.35 rad is 20 degrees, which on most objects grips
        # just as well and is a far larger solution set to choose from.
        #
        # Vertical is always tried first and tilts are tried in increasing
        # order, so nothing tilts that does not need to. 0 restores the strict
        # top-down behaviour.
        self.declare_parameter('grasp_tilt_max', 0.35)
        # Tilt magnitudes to try between 0 and grasp_tilt_max, and how many
        # directions to try each in (4 = away, left, toward, right).
        self.declare_parameter('grasp_tilt_steps', 2)
        self.declare_parameter('grasp_tilt_azimuths', 4)
        # Orientations the reach probe may try per point. It sends a planning
        # request for each, so the full tilt set would make an unreachable
        # object cost a minute of probing; the pre-flight explores the rest.
        self.declare_parameter('reach_orientations', 3)
        # Prove the planner can plan before the cycle starts, with a plan-only
        # goal to the arm's own current posture. Two seconds, no motion, and it
        # is the difference between "the planner is not planning" and a report
        # blaming the recorded pre-pick pose.
        self.declare_parameter('check_planner_ready', True)
        # One descent instead of two. The cycle used to stop at the pre-grasp,
        # open the gripper there and descend again -- two lines to solve, two
        # settles, two offset corrections, and a visible pause in mid-air.
        # With this the gripper opens above the object and a single straight
        # line goes all the way to the grasp.
        self.declare_parameter('single_descent', True)
        # Whether the vertical column legs may make a *second*, correcting
        # move. Off: on the descent that correction is the loop-then-descend-
        # again motion, and measured it made things worse -- a line that flew
        # 100% and landed 16.2 mm out became 31.7 mm out after it. The
        # correction was built for a repeatable, almost purely vertical offset
        # (dz -13.8, -14.0, -14.0, -14.0 mm); this error has lateral
        # components and is not repeatable, so aiming past it overshoots.
        #
        # The residual it leaves behind is real and is not fixed here: it is
        # the arm not tracking its own trajectory, which is what the
        # openarm_hardware gravity model addresses.
        self.declare_parameter('descend_offset_correction', False)
        # How long to let a commanded posture settle before trusting it,
        # seconds, and how close counts as settled, radians.
        #
        # The controller has no goal tolerance configured, so it reports
        # SUCCESS the instant the trajectory ends; the arm is still converging
        # for seconds afterwards. Descending 1.2 s after "ok" started the line
        # from a posture 80 mrad away from the one the pre-flight checked, and
        # it solved 47% instead of 100%.
        self.declare_parameter('posture_settle_time', 4.0)
        self.declare_parameter('posture_settle_tolerance', 0.02)
        # Send the joint7 tilt as its own move, after the arm is over the
        # object. Off: the tool hangs 180.1 mm off that joint, so tilting it
        # at the end swings the tool through an arc -- measured at 116 mm,
        # right before the descent, which is exactly the curved motion the
        # straight-line work exists to remove. With it off the tilt is part of
        # the same joint goal and the approach ends pointing down, having
        # taken one move rather than two.
        self.declare_parameter('approach_tilt_stage', False)
        # Headroom, radians, a chosen approach posture should have.
        #
        # A threshold, not a score: past it, more headroom buys nothing and
        # the ranking switches to preferring the posture nearest the staging
        # pose. At 0.05 it was picking postures 3 degrees from a stop, which
        # is the condition that kills a Cartesian leg -- including the offset
        # correction, which still has to travel a little after arriving.
        #
        # If no candidate clears it, the ranking falls back to roomiest-first
        # rather than refusing: the pre-flight is the real gate, and it checks
        # the descent from whichever posture is chosen.
        self.declare_parameter('posture_margin', 0.10)
        self.declare_parameter('posture_seeds', 48)
        # Prove the descent before the arm leaves its rest pose.
        #
        # Everything needed is available up front: candidate postures come
        # from the kinematic chain, and /compute_cartesian_path answers
        # questions about a start state the arm is not in yet. So the run
        # where the arm reached the pre-grasp, opened the gripper, discovered
        # 18% of the descent was all that would solve, and went back to start
        # again was avoidable -- the geometry had not changed since before it
        # moved.
        # Do not even ask for a collision-checked line on the vertical
        # column legs -- go straight to the unchecked one. See
        # descend_column: the object being grasped is itself in the octomap,
        # so the checked line stalls a centimetre above it whatever the map
        # resolution, and asking first costs a planning round trip and risks
        # flying a partial line into a worse start.
        self.declare_parameter('descend_ignores_octomap', True)
        # Exempt only the gripper links from the octomap, instead of turning
        # collision checking off for the whole arm. The reason the descent
        # needs an exemption is narrow -- the object being grasped is itself in
        # the map, so the fingers must enter its voxels -- and it says nothing
        # about the forearm or the elbow, which with checking off entirely are
        # free to sweep into the table.
        self.declare_parameter('gripper_octomap_exemption', True)
        # Worst per-joint travel a column leg may cost, radians.
        #
        # /compute_cartesian_path constrains the tool, not the arm. Measured
        # travel for a 150 mm descent: 0.43, 0.44, 0.49, 0.63 rad on the legs
        # that worked; 3.06, 3.26, 3.75 rad on the ones where the tool traced
        # the line while the shoulder swept across the workspace -- one of
        # which reached the table. 1.5 rad sits between with margin either
        # side. 0 disables the check.
        self.declare_parameter('column_max_joint_travel', 1.5)
        # Carry the object out through the pre-pick pose rather than straight
        # from the lift to the drop. The lift ends low over the work surface
        # and the drop pose is across it, and a direct joint goal came back
        # INVALID_MOTION_PLAN three times -- a path computed and then rejected,
        # which is what sweeping through the octomap looks like. pre_pick is
        # above the table by construction and is already the waypoint used on
        # the way back.
        self.declare_parameter('stage_drop_through_pre_pick', True)
        # Fly the long move above the object as a straight line too, when one
        # solves -- the same motion as dragging the end-effector arrow in
        # RViz. Everything below the transit was already a real Cartesian
        # line; this leg was the last joint-space move between the staging
        # pose and the grasp.
        #
        # Pre-flighted like the rest: the line is planned from the staging
        # posture and the descent legs from *its end*, so a linear approach is
        # only used when the whole column still flies from where the line
        # leaves the arm. Otherwise the chosen posture is used and this leg
        # goes as a joint goal.
        self.declare_parameter('linear_transit', True)
        self.declare_parameter('preflight_descent', True)
        # How many postures the pre-flight is allowed to probe. Each one costs
        # up to four /compute_cartesian_path calls and no motion, so this
        # trades a few seconds before moving against minutes of failed
        # attempts after.
        self.declare_parameter('preflight_candidates', 6)
        # After a pre-flight has passed, a later failure is a real fault, not
        # something a different yaw or 8 mm recovers -- so the ladder does not
        # run and the arm does not travel back for another go. Set true for
        # the old behaviour.
        self.declare_parameter('retry_after_preflight', False)
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
        self._urdf = None
        self._joint_limits = None
        # Kinematic chains, per arm, built from the description on demand.
        self._chains = {}
        # The posture the last approach used, as a warm start for the next.
        self._last_approach_posture = None
        # The posture the approach will start from, so a candidate can be
        # scored on how far the arm has to travel to take it up.
        self._approach_from = None
        # Why the pre-flight refused, when it did. Set means "checked and
        # impossible", which no retry recovers.
        self._preflight_reason = None
        # The posture the pre-flight settled on, for approach_above to use
        # instead of solving again.
        self._preflight_choice = None
        # Whether the gripper is currently allowed to touch the octomap. Set
        # for one leg at a time and always withdrawn.
        self._octomap_exempt = False
        # Whether cuMotion will accept a pose goal for this arm's tool.
        # None until checked; False refuses pose goals instead of sending
        # ones that come back INVALID_LINK_NAME.
        self._pose_goals_ok = None
        self._reported_limit_clamp = set()
        # The vertical line the arm is currently on, as
        # (grasp, pregrasp, quat), set the moment the descent starts and
        # cleared once the tool is back clear of the surface. It is what makes
        # it possible to get the arm out of trouble: every way a pick fails
        # leaves it extended down at the object, and this says which way is up
        # and how far.
        self._column = None
        self._holding = False
        self._last_detection = None
        self._last_payload = None
        self.state = 'INIT'
        # Distinguishes runs in the append-only log. Seconds since the epoch,
        # truncated: unique per launch without needing coordination.
        self._run_id = int(time.time())
        self._cycle_count = 0
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
        # Needed to *add* to the allowed-collision matrix rather than replace
        # it -- see allow_gripper_in_octomap.
        self.scene_query_client = self.create_client(
            GetPlanningScene, '/get_planning_scene', callback_group=self.cb)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.prompt_pub = self.create_publisher(String, '/vlm/prompt', latched)
        self.state_pub = self.create_publisher(String, '/pick_place/state', latched)

        self.create_subscription(String, '/pick_place/prompt', self._on_prompt, 10,
                                 callback_group=self.cb)
        self.create_subscription(String, '/vlm/detections', self._on_detections, 10,
                                 callback_group=self.cb)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10,
                                 callback_group=self.cb)
        # Latched by robot_state_publisher, so it arrives even subscribing
        # long after it was published. Read rather than assumed: the two arms'
        # limits are not mirror images (left joint2 runs [-3.316, +0.175]
        # against the right's [-0.175, +3.316]).
        self.create_subscription(String, '/robot_description', self._on_urdf,
                                 latched, callback_group=self.cb)

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

        heartbeat = self.get_parameter('motion_log_heartbeat').value
        if self._motion_log and heartbeat > 0.0:
            self.create_timer(heartbeat, self._log_heartbeat,
                              callback_group=self.cb)

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
            # The file is append-only across runs, which is the point -- but
            # without this a reader cannot tell one run's 240000-second gap
            # from a robot that sat still, and cycles from different runs
            # interleave in the eye.
            'run': self._run_id,
            'cycle': self._cycle_count,
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

    def _log_heartbeat(self):
        """One record a second while a cycle runs, moving or not.

        A per-motion log goes quiet exactly when something hangs, which is the
        moment worth seeing: a 12-second gap in the file could be a slow plan,
        a wedged planner, or an arm creeping. This makes the difference visible.
        Only while busy, so an idle robot does not fill the file.
        """
        with self._lock:
            busy = self._busy
        if busy:
            self.log_motion('-', 'heartbeat', 'running')

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

    def _on_urdf(self, msg):
        with self._lock:
            self._urdf = msg.data
            self._joint_limits = None
            self._chains = {}

    def joint_limits(self):
        """This arm's position limits, {joint: (lower, upper)}, or {}.

        From /robot_description, so what is checked is what the running robot
        was launched with. Parsed once and cached.
        """
        with self._lock:
            if self._joint_limits is not None:
                return self._joint_limits
            urdf = self._urdf
        if not urdf:
            return {}
        import xml.etree.ElementTree as ET
        limits = {}
        try:
            for joint in ET.fromstring(urdf).findall('joint'):
                if joint.get('name') not in self.arm_joints:
                    continue
                limit = joint.find('limit')
                if limit is None or limit.get('lower') is None:
                    continue
                limits[joint.get('name')] = (float(limit.get('lower')),
                                             float(limit.get('upper')))
        except (ET.ParseError, ValueError) as exc:
            self.get_logger().warn(
                f'could not read joint limits from the robot description: {exc}')
            return {}
        with self._lock:
            self._joint_limits = limits
        return limits

    def clamp_to_limits(self, positions, label):
        """Pull a commanded posture off the position limits.

        A goal sitting exactly on a limit is not reachable in practice. The
        "home" group state commands joint4 to 0.0, which *is* its lower limit,
        and the elbow stops 8.9 degrees short -- so the move looks successful,
        the joint never arrives, and anything verifying the posture says the
        arm is not home. Backing the goal off by joint_limit_margin makes the
        goal one the hardware can actually hold.

        Reported once per joint per label; it is a launch-time property of the
        posture, not an event.
        """
        limits = self.joint_limits()
        if not limits:
            return list(positions)
        margin = self.get_parameter('joint_limit_margin').value
        out = []
        for joint, value in zip(self.arm_joints, positions):
            low, high = limits.get(joint, (None, None))
            if low is None:
                out.append(value)
                continue
            # A margin wider than the joint's own range would invert the
            # bounds; centre it instead.
            if high - low <= 2 * margin:
                clamped = min(max(value, low), high)
            else:
                clamped = min(max(value, low + margin), high - margin)
            if abs(clamped - value) > 1e-9:
                key = (label, joint)
                if key not in self._reported_limit_clamp:
                    self._reported_limit_clamp.add(key)
                    self.get_logger().warn(
                        f'{label}: {joint} asked for {value:.4f} rad, which is '
                        f'within {margin:.3f} rad of its limits '
                        f'[{low:.4f}, {high:.4f}]; commanding {clamped:.4f} '
                        'instead. A joint cannot hold a goal on its own limit.')
            out.append(clamped)
        return out

    def kinematics(self, arm=None):
        """A chain for `arm`, built from /robot_description. None if unusable.

        Cached per arm: the description does not change during a run, and
        parsing it per query would be wasteful. None is a normal answer -- the
        caller falls back to a planner pose goal and says so.

        Either arm, not just the configured one, because the reach check asks
        whether the *other* arm could do the job -- and cuMotion cannot answer
        that: it takes Cartesian goals for one link per bringup, so a pose
        goal for the other arm's tool comes back INVALID_LINK_NAME, which says
        nothing about reach.
        """
        arm = arm or self.arm
        with self._lock:
            chain = self._chains.get(arm)
            urdf = self._urdf
            if chain is not None:
                return chain
        if not urdf:
            self.get_logger().warn(
                'no /robot_description yet, so the approach posture cannot be '
                'chosen here; falling back to a pose goal',
                throttle_duration_sec=30.0)
            return None
        chain = arm_kinematics.chain_from_urdf(urdf, arm)
        if chain is None:
            self.get_logger().warn(
                f'could not build a kinematic chain for the {arm} arm '
                f'from /robot_description; falling back to pose goals')
        with self._lock:
            self._chains[arm] = chain
        return chain

    def grasp_quat_options(self, quat, limit=None):
        """The grasps worth trying for this object, best first.

        Three sources of freedom, and none of them changes where the object is
        gripped:

        * The 180 degree jaw flip. A parallel gripper closes on the same two
          faces whichever way round it arrives. Free, and at the measured
          object position worth the difference between 0.000 rad of joint
          headroom and 0.172.
        * Coming down off vertical, up to grasp_tilt_max. A strictly top-down
          approach pins six of seven joints at a fixed point and near the edge
          of the envelope had no solution clear of the stops at all. Tilting
          the approach is a much larger set to choose from and, on most
          objects, the same grasp.
        * Both together.

        Ordered so that nothing is given up that does not have to be: exactly
        vertical first, then vertical flipped, then the smallest tilt in each
        direction, and so on outward. The pre-flight takes the first one whose
        whole column flies, so a tilt is only ever used because vertical could
        not be.
        """
        flips = [quat]
        if self.get_parameter('grasp_jaw_flip').value:
            flips.append(quat_mul(quat, (0.0, 0.0, 1.0, 0.0)))
        options = list(flips)

        most = float(self.get_parameter('grasp_tilt_max').value)
        steps = int(self.get_parameter('grasp_tilt_steps').value)
        azimuths = int(self.get_parameter('grasp_tilt_azimuths').value)
        if most > 0.0 and steps > 0 and azimuths > 0:
            for step in range(1, steps + 1):
                tilt = most * step / steps
                for index in range(azimuths):
                    azimuth = 2.0 * math.pi * index / azimuths
                    for base in flips:
                        options.append(tilt_quat(base, tilt, azimuth))
        # `limit` is for callers that pay per orientation and only need a
        # representative few -- the reach probe sends a planning request each.
        return options[:limit] if limit else options

    def choose_approach_posture(self, position, quat, label):
        """Pick the arm configuration to approach `position` in.

        Returns (joints, tilt_joint7 or None, quat_used, margin) or None.

        The point of doing this here rather than leaving it to the planner is
        that a planner returns *a* solution and this returns the one with the
        most room left in every joint -- see approach_frame. Used when the
        pre-flight is off or could not run; otherwise preflight_column has
        already chosen, having also checked that the descent flies.
        """
        candidates = self.approach_candidates(position, quat, label)
        if not candidates:
            return None
        best = candidates[0]
        wanted = self.get_parameter('posture_margin').value
        flipped = best['quat'] is not self.grasp_quat_options(quat)[0]
        note = (f'{label}: approaching in a posture with '
                f'{best["margin"]:.3f} rad of joint headroom, '
                f'{best.get("travel", 0.0):.2f} rad of travel to take it up '
                f'({best["solved"]}/{best["tried"]} seeds solved'
                + (', jaws turned 180 deg' if flipped else '') + ')')
        if best['margin'] < wanted:
            self.get_logger().warn(
                note + f' -- under {wanted:.3f}, so a straight line may not '
                'run the whole way; the descent will say so')
        else:
            self.get_logger().info(note)
        return best['joints'], best['tilt'], best['quat'], best['margin']

    def approach_candidates(self, position, quat, label):
        """Usable postures for the approach, most joint headroom first.

        A list rather than a single answer, because headroom is only half the
        question -- see preflight_column. The tilt is already applied and
        scored by the chain, so what comes back is ranked by the headroom of
        the posture that will actually be commanded.
        """
        chain = self.kinematics()
        if chain is None:
            return []
        mode = self.get_parameter('approach_frame').value
        if mode not in ('tool', 'wrist'):
            return []
        seeds = int(self.get_parameter('posture_seeds').value)
        with self._lock:
            here = [self._arm_positions.get(j) for j in self.arm_joints]
        extra = [here] if all(v is not None for v in here) else []
        if self._last_approach_posture is not None:
            extra.append(list(self._last_approach_posture))
        keep = max(1, int(self.get_parameter('preflight_candidates').value))

        # Each orientation costs a posture solve of about a second, and the
        # tilted ones can number a dozen or more, so stop as soon as there is
        # something good enough rather than solving them all. Ordered
        # vertical-first, so this also means a tilt is only ever solved for
        # because vertical did not yield one.
        enough = self.get_parameter('posture_margin').value
        out = []
        for candidate in self.grasp_quat_options(quat):
            if any(entry['margin'] >= enough for entry in out):
                break
            # The direction the jaws close in. Fixed by joints 1-6 alone --
            # joint7 turns about that very axis -- so pinning it costs the
            # wrist partition nothing, and leaving it loose cost a grasp: a
            # clean straight-line descent onto the right point, and then the
            # jaws closing 17 degrees out of line with the object.
            jaw = None
            if not self.get_parameter('grasp_yaw_free').value:
                jaw = chain.jaw_axis_for(candidate)
            kept, solved, tried = chain.approach_candidates(
                position, candidate, wrist=(mode == 'wrist'), seeds=seeds,
                extra_seeds=extra, jaw_axis=jaw, keep=keep,
                align_tolerance=self.get_parameter(
                    'orientation_tolerance').value)
            for joints, tilt, margin in kept:
                out.append({'joints': list(joints), 'tilt': tilt,
                            'quat': candidate, 'margin': margin,
                            'solved': solved, 'tried': tried})
        # Ranking. Headroom first, but only as a threshold: past
        # posture_margin more of it buys nothing, while a posture the arm has
        # to reconfigure across the workspace to reach is a much harder
        # free-space plan. One was measured sitting in TRANSIT for 20 seconds
        # with the tool not moving at all -- 0.493 rad of headroom and a plan
        # that never came back.
        #
        # So: everything with adequate headroom first, nearest to where the
        # approach starts from; then the cramped ones, roomiest first.
        wanted = self.get_parameter('posture_margin').value
        reference = self._approach_from
        if reference is None:
            with self._lock:
                here = [self._arm_positions.get(j) for j in self.arm_joints]
            reference = here if all(v is not None for v in here) else None
        for entry in out:
            entry['travel'] = (
                max(abs(a - b) for a, b in zip(entry['joints'], reference))
                if reference else 0.0)

        def rank(entry):
            if entry['margin'] >= wanted:
                return (0, entry['travel'])
            return (1, -entry['margin'])

        out.sort(key=rank)
        if not out:
            self.get_logger().warn(
                f'{label}: no approach posture solved for either grasp '
                f'direction')
        return out[:keep]

    def preflight_column(self, grasp, pregrasp, approach, quat, label):
        """Prove the descent before the arm leaves its rest pose.

        For each candidate posture, ask /compute_cartesian_path whether the
        line down to the pre-grasp and on to the grasp solves *from that
        posture* -- with an explicit start state, so nothing has to move to
        find out. The first candidate whose whole column solves is the one
        used.

        This exists because joint headroom is not the same question. The run
        that prompted it reached the pre-grasp with joint1 0.040 rad from its
        stop, opened the gripper, and only then discovered that 18% of the
        descent was all that would solve -- at which point the cycle was
        standing over the object with the gripper open and nothing to do but
        go back and start again. The information needed to avoid that was
        available before the first move.

        Returns a dict describing the chosen posture, or None. None with
        `self._preflight_reason` set means "checked, and no posture works" --
        which is a refusal, not something a retry can help.
        """
        # Timed: this is a run of Cartesian probes, and it shares the
        # LOCATE window in the log with the detector's own wait, so
        # neither could be attributed. preflight_s is the probe cost.
        preflight_from = self.get_clock().now().nanoseconds * 1e-9
        self._preflight_reason = None
        if not self.get_parameter('preflight_descent').value:
            return None
        candidates = self.approach_candidates(approach, quat, label)
        if not candidates:
            return None                      # no chain: fall back, don't refuse
        candidates = candidates[:max(1, int(self.get_parameter(
            'preflight_candidates').value))]
        wanted = self.get_parameter('cartesian_min_fraction').value
        floor = self.get_parameter('cartesian_partial_min').value
        # The legs the descent will actually fly. With no transit stage the
        # approach *is* the pre-grasp, and probing a leg of zero length would
        # both waste a call and answer a question nobody asked.
        wanted_legs = []
        if not self.get_parameter('single_descent').value \
                and abs(approach[2] - pregrasp[2]) > 1e-6:
            wanted_legs.append(('pre-grasp', tuple(pregrasp)))
        wanted_legs.append(('grasp', tuple(grasp)))
        # Ask exactly the way the descent will, or the pre-flight is answering
        # a different question than the one that matters.
        checked_first = not (
            self.get_parameter('approach_ignores_octomap').value
            and self.get_parameter('descend_ignores_octomap').value)

        # Candidate zero: no posture goal at all, just a straight line from
        # where the approach starts. Tried first because it is the motion
        # asked for -- the arm travelling in a line, as if the end-effector
        # arrow were being dragged -- and taken only if the descent still
        # flies from wherever that line leaves the arm.
        if self.get_parameter('linear_transit').value and self._approach_from:
            for candidate_quat in self.grasp_quat_options(quat):
                flown, landed = self.probe_cartesian(
                    self._approach_from, approach, candidate_quat,
                    avoid_collisions=True)
                if flown is None or flown < wanted or landed is None:
                    continue
                legs = [('approach', flown)]
                reached, worst = landed, flown
                for leg_label, target in wanted_legs:
                    part, end = self.probe_cartesian(
                        reached, target, candidate_quat,
                        avoid_collisions=checked_first)
                    if part is None:
                        break
                    if part < wanted and checked_first:
                        loose, loose_end = self.probe_cartesian(
                            reached, target, candidate_quat,
                            avoid_collisions=False)
                        if loose is not None and loose > part:
                            part, end = loose, loose_end
                    legs.append((leg_label, part))
                    worst = min(worst, part)
                    if end is not None:
                        reached = end
                    if part < wanted:
                        break
                if worst >= wanted:
                    detail = ', '.join(f'{n} {v * 100:.0f}%' for n, v in legs)
                    self.get_logger().info(
                        f'{label}: pre-flight passed on a straight-line '
                        f'approach -- {detail}. No posture goal needed.')
                    self.log_motion(label, 'preflight', 'ok-linear',
                                    secs=round(self.get_clock().now().nanoseconds * 1e-9
                                               - preflight_from, 2),
                                    target=[round(v, 5) for v in approach],
                                    legs={n: round(v, 4) for n, v in legs})
                    return {'joints': list(landed), 'tilt': None,
                            'quat': candidate_quat, 'margin': 0.0,
                            'solved': 0, 'tried': 0, 'linear': True,
                            'legs': legs, 'worst_leg': worst, 'travel': 0.0}

        best = None
        for index, entry in enumerate(candidates):
            start = list(entry['joints'])
            if entry['tilt'] is not None:
                start[6] = entry['tilt']
            legs, worst, reached = [], 1.0, start
            for leg_label, target in wanted_legs:
                fraction, end = self.probe_cartesian(
                    reached, (target[0], target[1], target[2]), entry['quat'],
                    avoid_collisions=checked_first)
                if fraction is None:
                    self.get_logger().warn(
                        f'{label}: /compute_cartesian_path will not answer, so '
                        f'the descent cannot be checked in advance')
                    return None
                if fraction < wanted and checked_first:
                    # The descent retries unchecked when the object it is
                    # reaching for is itself in the octomap, so the pre-flight
                    # has to ask the same way or it would reject every
                    # top-down grasp.
                    unchecked, end_unchecked = self.probe_cartesian(
                        reached, (target[0], target[1], target[2]),
                        entry['quat'], avoid_collisions=False)
                    if unchecked is not None and unchecked > fraction:
                        fraction, end = unchecked, end_unchecked
                legs.append((leg_label, fraction))
                worst = min(worst, fraction)
                if end is not None:
                    reached = end
                if fraction < floor:
                    break
            entry['legs'] = legs
            entry['worst_leg'] = worst
            if best is None or worst > best['worst_leg']:
                best = entry
            detail = ', '.join(f'{name} {value * 100:.0f}%'
                               for name, value in legs)
            if worst >= wanted:
                self.get_logger().info(
                    f'{label}: pre-flight passed on candidate {index + 1} of '
                    f'{len(candidates)} -- {detail}, {entry["margin"]:.3f} rad '
                    f'of joint headroom, {entry.get("travel", 0.0):.2f} rad of '
                    f'travel from the staging pose')
                self.log_motion(label, 'preflight', 'ok',
                                secs=round(self.get_clock().now().nanoseconds * 1e-9
                                           - preflight_from, 2),
                                target=[round(v, 5) for v in approach],
                                candidate=index + 1,
                                of=len(candidates),
                                margin_rad=round(entry['margin'], 4),
                                travel_rad=round(entry.get('travel', 0.0), 3),
                                legs={n: round(v, 4) for n, v in legs})
                return entry
            self.get_logger().info(
                f'{label}: candidate {index + 1} of {len(candidates)} cannot '
                f'fly the column ({detail}); trying the next posture')

        detail = ', '.join(f'{name} {value * 100:.0f}%'
                           for name, value in (best or {}).get('legs', []))
        self._preflight_reason = (
            f'no straight-line descent exists from any of the '
            f'{len(candidates)} postures that reach '
            f'({approach[0]:.3f}, {approach[1]:.3f}, {approach[2]:.3f}). '
            f'The best managed {detail}. Nothing moved. This is the arm '
            f'running out of travel along the descent, not an obstacle -- '
            f'bring the object closer to the base, or try the other arm.')
        self.log_motion(label, 'preflight', 'refused',
                        secs=round(self.get_clock().now().nanoseconds * 1e-9
                                   - preflight_from, 2),
                        target=[round(v, 5) for v in approach],
                        of=len(candidates),
                        legs={n: round(v, 4)
                              for n, v in (best or {}).get('legs', [])})
        return None

    def _report_posture_drift(self, label):
        """How far the arm settled from the posture the pre-flight used.

        The pre-flight proves the descent from a specific posture. The arm
        then has to hold it, and it does not always: 70 mrad on joint1 at the
        transit in one run, which is about 35 mm at the tool, and the line
        that solved 100% from the intended posture solved 48.6% from the real
        one. Silent, until it turns into "could not descend" with the
        pre-flight insisting it had checked.
        """
        entry = self._preflight_choice
        if not entry or entry.get('linear'):
            # A linear approach was never commanded to a posture. entry
            # ['joints'] there is the *probe's predicted* end of the line, not
            # a target the arm was asked for, so comparing against it measures
            # nothing -- it reported 711 mrad of "drift" on joint3 for an
            # approach that had gone exactly as planned.
            return
        wanted = list(entry['joints'])
        if entry.get('tilt') is not None:
            wanted[6] = entry['tilt']
        with self._lock:
            here = [self._arm_positions.get(j) for j in self.arm_joints]
        if any(v is None for v in here):
            return
        drift = [h - w for h, w in zip(here, wanted)]
        worst = max(range(len(drift)), key=lambda i: abs(drift[i]))
        if abs(drift[worst]) < 0.01:
            return
        self.get_logger().warn(
            f'{label}: the arm is {abs(drift[worst]) * 1000:.0f} mrad from the '
            f'posture the descent was checked against, worst on '
            f'{self.arm_joints[worst].rsplit("_", 1)[-1]}. The line was proved '
            f'from where it was told to stand, not from here.')
        self.log_motion(label, 'posture', 'drifted',
                        target=[round(v, 5) for v in wanted],
                        drift_mrad=[round(v * 1000, 1) for v in drift])

    def approach_above(self, position, quat, label):
        """Get the arm above `position`, ready to descend. True/False.

        Two joint goals in wrist mode, because that is the whole idea: bring
        joint7's centre over the object with the hand still out of the way,
        *then* tilt the hand down. Approaching with the hand already pointing
        down means carrying 180 mm of gripper below the wrist through the
        whole move.

        Joint goals, not pose goals, for two reasons: cuMotion accepts them
        for either arm regardless of which link its ee_link is, and the same
        seven numbers give the same posture every time -- which a pose goal on
        a redundant 7-DOF arm does not.
        """
        entry = self._preflight_choice
        if entry is not None:
            # Already solved *and* proved before anything moved. Solving again
            # would risk landing on a different posture than the one the
            # descent was checked against.
            chosen = (entry['joints'], entry['tilt'], entry['quat'],
                      entry['margin'])
        else:
            chosen = self.choose_approach_posture(position, quat, label)
        if chosen is None:
            self.log_motion(label, 'posture', 'fell-back-to-pose-goal',
                            target=[round(v, 5) for v in position])
            self.get_logger().warn(
                f'{label}: no posture was chosen here, so this is a plain pose '
                f'goal and the planner picks the arm configuration -- which is '
                f'what put joint1 0.04 rad from its stop before a descent. '
                f'Check that /robot_description is published and numpy is '
                f'importable.')
            return self.move_to_pose(position, quat, label), quat
        if entry is not None and entry.get('linear'):
            # The pre-flight already proved a straight line here, and the
            # descent below it from where that line ends. So fly it -- the
            # whole way from the staging pose to the grasp is then one
            # continuous set of straight lines, which is the motion asked for.
            used = entry['quat']
            flown = self.converge_to(position, used, label)
            self.log_motion(label, 'posture', 'linear' if flown else 'failed',
                            target=[round(v, 5) for v in position],
                            legs={n: round(v, 4) for n, v in entry['legs']})
            if flown:
                self._last_approach_posture = list(entry['joints'])
                return True, used
            self.get_logger().warn(
                f'{label}: the straight line was checked and passed but did '
                f'not fly; falling back to a chosen posture')
            chosen = self.choose_approach_posture(position, quat, label)
            if chosen is None:
                return self.move_to_pose(position, quat, label), quat
        joints, tilt, used, margin = chosen
        self._last_approach_posture = list(joints)
        if tilt is None:
            ok = self._move_to_joints(list(joints), label, skip_if_there=True)
            self.log_motion(label, 'posture', 'ok' if ok else 'failed',
                            target=[round(v, 5) for v in joints],
                            margin_rad=round(margin, 4))
            return ok, used
        if not self.get_parameter('approach_tilt_stage').value:
            # One move, ending with the hand already pointing down. Tilting
            # afterwards swung the tool 116 mm through an arc immediately
            # before the descent -- the tool is 180.1 mm off joint7 -- and cost
            # a second plan. The partition is in the *solve*, which still
            # leaves joint7 out; the execution does not have to copy it.
            tilted = list(joints)
            tilted[6] = tilt
            ok = self._move_to_joints(tilted, label, skip_if_there=True)
            self.log_motion(label, 'posture', 'ok' if ok else 'failed',
                            target=[round(v, 5) for v in tilted],
                            tilt_deg=round(math.degrees(tilt), 1),
                            margin_rad=round(margin, 4))
            return ok, used
        # Stage one: the wrist over the object, joint7 left where it is.
        with self._lock:
            have = self._arm_positions.get(self.arm_joints[6])
        staged = list(joints)
        staged[6] = have if have is not None else joints[6]
        if not self._move_to_joints(staged, f'{label} wrist',
                                    skip_if_there=True):
            self.log_motion(label, 'posture', 'wrist-failed',
                            target=[round(v, 5) for v in staged])
            return False, used
        # Stage two: tilt the hand onto the grasp axis.
        tilted = list(joints)
        tilted[6] = tilt
        ok = self._move_to_joints(tilted, f'{label} tilt', skip_if_there=True)
        self.log_motion(label, 'posture', 'ok' if ok else 'tilt-failed',
                        target=[round(v, 5) for v in tilted],
                        tilt_deg=round(math.degrees(tilt), 1),
                        margin_rad=round(margin, 4))
        return ok, used

    def arm_is_still(self):
        """Has the arm stopped moving? None with no velocity readings.

        Distinguishes "as close to the goal as this hardware gets" from "still
        on its way there", which is the difference between accepting a
        standing offset and accepting a move that was cut short.
        """
        with self._lock:
            speeds = [self._arm_velocities.get(j) for j in self.arm_joints]
        if any(v is None for v in speeds):
            return None
        return max(abs(v) for v in speeds) <= \
            self.get_parameter('joint_still_speed').value

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
        if self._pose_goals_ok is False:
            self.get_logger().error(
                f'{label}: this would be a pose goal for {self.tcp_frame}, '
                f'which cuMotion will reject -- it plans Cartesian goals for '
                f'one link per bringup and that is not this arm. Refusing '
                f'rather than sending it, so the failure says why. Relaunch '
                f'with tool_frame:={self.tcp_frame} to use pose goals with '
                f'this arm.')
            return False
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
        if len(positions) == len(self.arm_joints):
            positions = self.clamp_to_limits(positions, label)
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

        Clamped off the position limits, and for the same reason the goal is:
        what this returns is what every "are we home?" check compares against,
        so it has to be the posture that was actually commanded, not the one
        that was asked for.
        """
        states = self.load_states()
        joints = (states.get(HOME_STATE) or {}).get('joints')
        if joints:
            return self.clamp_to_limits(list(joints), 'HOME')
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
        return self.clamp_to_limits(
            list(self.get_parameter('home_joint_positions').value), 'HOME')

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

        Two tolerances, because there are two different situations and only
        one of them is a fault:

        * within home_pose_tolerance -- at home, nothing to say.
        * beyond it but within home_settle_tolerance *and stopped* -- the arm
          is as close as this hardware will get. Accepted, and named in the
          log. The right joint4 does exactly this: commanded 0.0, reference
          0.0, resting at 0.15583 rad because the URDF puts its lower limit at
          precisely 0.0 and the elbow stops short of it. Failing here is what
          left the arm sitting at home unable to start a cycle.
        * still moving, or further off than that -- genuinely not home.

        Being a few degrees off at a folded-down posture does not put the arm
        back in the camera's view, which is what the check is really guarding.
        A gross error still fails, so a move that was cut short is still
        caught.
        """
        tolerance = self.get_parameter('home_pose_tolerance').value
        settled = self.get_parameter('home_settle_tolerance').value
        wanted = self.home_positions()
        worst = self.joint_error(wanted, fresh=True)
        if worst is None:
            self.get_logger().warn('no joint states for the arm; assuming not at home')
            return False
        if worst <= tolerance:
            return True
        still = self.arm_is_still()
        if worst <= settled and still:
            self.get_logger().warn(
                f'at home with a standing offset: {self._worst_joint(wanted)} '
                f'(tolerance {tolerance:.3f}, settled limit {settled:.3f}). '
                'The arm has stopped, so this is as close as it gets -- '
                'treating it as home.')
            return True
        moving = '' if still else ', and it is still moving'
        self.get_logger().info(
            f'not at home: {self._worst_joint(wanted)}{moving} '
            f'(settled limit {settled:.3f})')
        return False

    def _worst_joint(self, wanted):
        """"joint4 is 0.156 rad off (0.156 vs 0.000)", for the logs."""
        with self._lock:
            measured = {j: self._arm_positions.get(j) for j in self.arm_joints}
        worst, name, have, want = -1.0, '?', 0.0, 0.0
        for joint, value in zip(self.arm_joints, wanted):
            got = measured.get(joint)
            if got is None:
                continue
            if abs(got - value) > worst:
                worst, name, have, want = abs(got - value), joint, got, value
        if worst < 0:
            return 'no readings'
        return (f'{name.rsplit("_", 1)[-1]} is {worst:.3f} rad off '
                f'({have:+.3f} vs {want:+.3f} commanded)')

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

        torque = abs(self.finger_effort() or 0.0)
        self.get_logger().info(
            f'closed to {position:.4f} m without reaching {cap:.3f} Nm '
            f'(torque {torque:.3f} Nm)')
        # Fully shut with the fingers never loaded means there was nothing
        # between them. Measured: 22 steps from 40.9 mm down to 2.5 mm with
        # the effort flat at about 0.5 Nm -- friction, not an object -- against
        # a 2.5 Nm cap. Saying so here saves the lift, the carry and the drop,
        # all of which were being attempted with an empty gripper.
        floor = self.get_parameter('grasp_finger_min').value
        if position <= floor:
            self.get_logger().error(
                f'the gripper closed to {position * 1000:.1f} mm without the '
                f'torque ever passing {cap:.2f} Nm (peak '
                f'{torque:.2f} Nm) -- there was nothing between the fingers. '
                f'Either the object is not where it was detected, or the '
                f'jaws are not where they were sent. Read the CLOSE_GRIPPER '
                f'line just above: it prints the measured jaw position '
                f'against the commanded grasp, and says off-target when they '
                f'disagree. Measured once at 29 mm high with the descent '
                f'plan ending exactly on the point -- nothing about the '
                f'detector or the fingers was wrong.')
            self.log_motion('CLOSE', 'gripper', 'closed-on-nothing',
                            target=round(float(position), 6),
                            torque=round(torque, 3), cap=cap)
            return False
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
        # A name in the graph is not a running node. When the planner dies its
        # service and action names linger in DDS discovery, so
        # wait_for_service keeps succeeding -- measured: no cumotion process at
        # all, no node in `ros2 node list`, and
        # cumotion_planner/get_parameters still listed. Treating "advertised
        # but silent" as merely unknown is what let a cycle run on a dead
        # planner: PRE_PICK timed out three times and then even the move home
        # failed, with nothing in the log saying why.
        #
        # Asked twice before calling it dead, because a live node that happens
        # to be mid-plan can be slow to answer its parameter service. This
        # check runs before any motion, when it should be idle.
        for timeout in (5.0, 10.0):
            result = self._await(
                client.call_async(GetParameters.Request(names=[name])), timeout)
            if result is not None:
                break
            self.get_logger().warn(
                f'the planner did not answer for {name} within {timeout:.0f} s')
        if result is None:
            return DEAD_PLANNER
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
        if robot_file is DEAD_PLANNER:
            # Propagated, not flattened to None. None means "could not tell,
            # carry on"; this means "there is no planner", and collapsing the
            # two is how a cycle came to run against a dead planner and time
            # out on a plain joint goal to a recorded posture.
            return DEAD_PLANNER
        if not robot_file:
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
        # home_state is NOT required, from either arm.
        #
        # It used to be, for the other arm, on the grounds that the fallback
        # was one list measured on one arm and the arms are mirrored. That is
        # no longer what the fallback is: home_joint_positions is
        # [0, 0, 0, 0.20, 0, 0, 0] -- symmetric, and a valid folded-down
        # posture for either arm. Measured on the left arm at all zeros, the
        # tool sits at [0.0002, +0.1736, 0.0819], the mirror of the right's
        # [-0.0000, -0.1735, 0.0819].
        #
        # So the requirement was refusing complete setups. It cost a cycle
        # that had correctly chosen the left arm for an object in the left
        # half of the frame, with pre_pick_state and drop_state both recorded
        # and playable, and reported it as "no recorded states for the left
        # arm".
        missing = [n for n in needed if not (states.get(n) or {}).get('joints')]
        if not (states.get(HOME_STATE) or {}).get('joints'):
            self.get_logger().info(
                f'no {HOME_STATE} recorded for the {self.arm} arm; using the '
                f'symmetric home_joint_positions fallback. Record one with '
                f'"python3 record_states.py --arm {self.arm} {HOME_STATE}" if '
                f'this arm needs its own rest pose.')
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
                '/cumotion_planner is not answering -- nothing can be planned. '
                'Its service name can still be in the graph after the process '
                'has gone, so "the service exists" is not evidence it is '
                'alive; check with  ros2 node list | grep cumotion  and look '
                'in the launch output for "cumotion_goal_set_planner_node ... '
                'process has died". Restart the robot.')
            return False
        if ee_link is None:
            self.get_logger().warn(
                'could not determine cuMotion\'s ee_link; continuing without the '
                'check. A mismatch shows up as INVALID_LINK_NAME at PREGRASP.')
            return True
        if ee_link != self.tcp_frame:
            # A mismatch is only fatal if the cycle actually sends pose goals.
            #
            # With the defaults it does not. The approach goes as a joint goal
            # or a Cartesian line; the descent and lift go through
            # /compute_cartesian_path, which is move_group's own service and
            # takes a link_name, so it serves either arm; and pre_pick, drop
            # and home are joint goals, which cuMotion accepts for either arm
            # whatever its ee_link. Refusing outright meant an object in the
            # left half of the frame switched the arm correctly and then
            # stopped dead with "the planner is unusable for this arm", when
            # nothing in the cycle was going to ask cuMotion for a left-arm
            # pose.
            self._pose_goals_ok = False
            needs = []
            if self.get_parameter('approach_frame').value == 'planner':
                needs.append('approach_frame is "planner", which aims the '
                             'move above the object as a pose goal')
            if self.place_mode == 'position':
                needs.append('place_mode is "position", which drives to the '
                             'drop point as pose goals')
            if not self.get_parameter('descend_linear_only').value:
                needs.append('descend_linear_only is off, so a failed line '
                             'falls back to pose-goal stepping')
            if needs:
                self.get_logger().error(
                    f'cuMotion takes Cartesian goals for {ee_link}, but this '
                    f'cycle needs {self.tcp_frame} and cannot avoid pose '
                    f'goals: {"; ".join(needs)}. Relaunch with '
                    f'tool_frame:={self.tcp_frame} '
                    f'(native/run_launch_everything.sh '
                    f'tool_frame:={self.tcp_frame}), or change those '
                    f'settings.')
                return False
            self.get_logger().warn(
                f'cuMotion takes Cartesian goals for {ee_link}, not '
                f'{self.tcp_frame} -- so no pose goal can be sent for this '
                f'arm. Continuing: with these settings the cycle uses joint '
                f'goals and /compute_cartesian_path, which both serve either '
                f'arm. A fallback that needs a pose goal will refuse rather '
                f'than send one.')
            return True
        self._pose_goals_ok = True
        self.get_logger().info(f'cuMotion plans Cartesian goals for {ee_link}')
        return True

    # -- perception ----------------------------------------------------------

    def detect(self, min_count=1):
        """Wait for a detection message newer than now and matching our prompt.

        Logged with how long the wait was, because a cycle's time is not
        where it looks: LOCATE spanned 10 s and 12.6 s of a measured 76 s run,
        but that window also holds the reachability probes and the pre-flight,
        and nothing in the log separated the detector's own latency from
        them.
        """
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
                # 'seen', not 'ok': nothing was commanded and nothing
                # moved, so this record carries no `before` and no `path`,
                # and an outcome of 'ok' would put it among the motions that
                # do.
                self.log_motion(
                    'LOCATE', 'detect', 'seen',
                    secs=round(self.get_clock().now().nanoseconds * 1e-9
                               - started, 2),
                    found=len(payload.get('detections', [])))
                return payload
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                if fresh:
                    return payload          # fresh but empty: nothing was seen
                self.get_logger().error(
                    'no fresh /vlm/detections -- is VLM/run_vlm_detector.sh running?')
                return None
            self._abort.wait(0.1)
        return None

    def fresh_tcp(self, max_age=0.5):
        """Where the tool is now, refusing a transform older than max_age.

        Exists because forgetting the freshness argument has gone wrong four
        times in this file -- the joint states, the convergence check, and both
        ends of the progress measurement in converge_to. The default lookup
        returns the latest *available* transform, which straight after a move
        is the pose from before it, and every one of those bugs was a
        comparison against where the arm used to be. Anything asking "where is
        the tool" in order to judge a move should call this.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        return self.tcp_position(newer_than=now - max_age)

    def tcp_position(self, newer_than=None, timeout=2.0):
        """Where the tool is, in the base frame.

        newer_than insists on a transform stamped after that time, and anything
        judging whether a move *arrived* has to pass it. rclpy.time.Time()
        means "latest available", and the latest available immediately after a
        move is still the pose from before it -- so a convergence check reading
        it would compare the target against where the arm used to be, decide it
        had not arrived, and move again. The same staleness that made the
        octomap gate fire off-home.
        """
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        while True:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, self.tcp_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
            except Exception as exc:                 # noqa: BLE001 - reported
                self.get_logger().warn(f'no {self.tcp_frame} transform: {exc}')
                return None
            if newer_than is None:
                break
            stamp = (tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9)
            if stamp > newer_than:
                break
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                self.get_logger().warn(
                    f'no {self.tcp_frame} transform newer than the move; '
                    f'judging on a {newer_than - stamp:.2f} s old one')
                break
            if self._abort.wait(0.02):
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
        # Seeded from the staging pose for the same reason the planner probe
        # is: KDL is a local solver, and seeding it from the folded-down HOME
        # posture asks it to find a solution far from where it starts.
        staging = self.staging_joints(arm)
        if staging:
            seed = staging
        elif arm == self.arm:
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

        # Solved here, for either arm.
        #
        # This is the only authority that works for the arm cuMotion is not
        # pointed at. cuMotion takes Cartesian goals for one link per bringup,
        # so a pose goal for the other arm's tool returns INVALID_LINK_NAME --
        # no statement about reach at all -- and the verdict then rested
        # entirely on KDL, which is documented above as unreliable. Measured
        # consequence: an object in the left half of the frame, the left arm
        # correctly tried first, refused as unreachable on a KDL "no", and the
        # report claiming cuMotion had agreed when it had never been asked.
        #
        # The solver here is the same one that chooses approach postures, and
        # it works from the description rather than from a service.
        solver = self.kinematics(arm)
        if solver is not None:
            for candidate in self.grasp_quat_options(
                    quat, limit=max(1, int(self.get_parameter(
                        'reach_orientations').value))):
                joints_found, _margin, _solved, _tried = solver.tool_posture(
                    position, candidate,
                    seeds=int(self.get_parameter('posture_seeds').value))
                if joints_found is not None:
                    return True

        verdict = self.planner_can_reach(position, quat, arm)
        if verdict is not None:
            return verdict
        # Neither authority could answer. Unknown, not "no": reporting out of
        # reach because a service was unavailable is how a stack that is
        # merely misconfigured looks like a workspace problem.
        return None if arm != self.arm else (False if asked else None)

    def planner_is_planning(self):
        """Can the planner plan *anything* right now? True/False/None.

        A joint goal to where the arm already is, plan_only, so nothing moves
        and the answer cannot be about reach: if this fails, the planner is
        not usable and no target will be.

        Worth the two seconds it costs. cuMotion's parameter service answering
        is not evidence it can plan -- measured on this robot, the node was
        started in the same second the cycle began, logged nothing at all, and
        took SIGFPE (exit code -8) two minutes later, while three attempts at
        a plain joint goal to a recorded posture came back TIMED_OUT. Without
        this the report blames the recorded pose ("re-record it somewhere the
        arm can reach"), which is exactly the wrong place to look.
        """
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            return None
        if not self.await_joint_states():
            return None
        with self._lock:
            here = [self._arm_positions.get(j) for j in self.arm_joints]
        if any(v is None for v in here):
            return None

        request = self._base_request()
        constraints = Constraints()
        for name, value in zip(self.arm_joints, here):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = jc.tolerance_below = 0.05
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        request.goal_constraints = [constraints]
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = True          # nothing moves
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        handle = self._await(self.move_client.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            return None
        result = self._await(handle.get_result_async(),
                             self.get_parameter('motion_timeout').value)
        if result is None:
            return False
        code = result.result.error_code.val
        if code == MOVEIT_SUCCESS:
            return True
        self.get_logger().error(
            f'the planner cannot plan a goal to where the arm already is '
            f'(error {code}). Nothing else will plan either, so this is the '
            f'stack rather than the target: check that /cumotion_planner is '
            f'alive and finished warming up (it logs nothing while it loads), '
            f'and look for "process has died" in the launch output.')
        return False

    def staging_joints(self, arm):
        """That arm's recorded pre-pick posture, or None.

        Read for either arm, not just the configured one: the reach check asks
        whether the *other* arm could do the job, and that question has to be
        posed from where that arm would actually start.
        """
        path = (self.states_file if arm == self.arm
                else os.path.join(WS, f'pick_place_states_{arm}.yaml'))
        if not os.path.exists(path):
            return None
        try:
            with open(path) as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            return None
        if data.get('arm') not in (None, arm):
            return None
        entry = (data.get('states') or {}).get(PRE_PICK_STATE) or {}
        joints = entry.get('joints')
        return list(joints) if joints and len(joints) == 7 else None

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
        # Asked from the staging pose, not from wherever the arm is standing.
        #
        # The check runs at HOME, where the arm is folded down by the base
        # with the work surface between it and the object -- so a plan from
        # there can fail because of the table rather than because the point is
        # out of reach, and the cycle then reports "no arm can pick the object"
        # about a point both arms can pick. pre_pick_state exists exactly to
        # be above the table before reaching, and it is what the cycle
        # actually approaches from.
        staging = self.staging_joints(arm)
        if staging:
            request.start_state.is_diff = False
            state = JointState()
            state.name = [f'openarm_{arm}_joint{i}' for i in range(1, 8)]
            state.position = [float(v) for v in staging]
            request.start_state.joint_state = state
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

        Every grasp direction the cycle would actually use is tried, which
        means the 180-degree jaw flip as well: refusing a point the arm can
        reach perfectly well with the jaws the other way round would be a
        false negative, and this check ends the cycle outright.
        """
        options = self.grasp_quat_options(
            quat, limit=max(1, int(self.get_parameter(
                'reach_orientations').value)))
        for label, point in points:
            verdict = None
            for candidate in options:
                verdict = self.reachable(point, candidate)
                if verdict:
                    break
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
            if any(self.reachable(point, c, arm=other) for c in options):
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

    def _cartesian_request(self, position, quat, avoid_collisions,
                           start_joints=None):
        """The straight-line query, shared by the probe and the real move.

        start_joints makes it a question about a posture the arm is *not* in
        yet, which is what lets the whole column be proved before anything
        moves.
        """
        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.group_name = self.group
        request.link_name = self.tcp_frame
        request.max_step = self.get_parameter('cartesian_step').value
        request.jump_threshold = 0.0
        request.avoid_collisions = avoid_collisions
        if start_joints is None:
            request.start_state.is_diff = True
        else:
            request.start_state.is_diff = False
            state = JointState()
            state.name = list(self.arm_joints)
            state.position = [float(v) for v in start_joints]
            request.start_state.joint_state = state
        target = PoseStamped().pose
        target.position.x, target.position.y, target.position.z = position
        (target.orientation.x, target.orientation.y,
         target.orientation.z, target.orientation.w) = quat
        request.waypoints = [target]
        return request

    def gripper_links(self):
        """The links that are allowed to enter the object's own voxels."""
        arm = self.arm
        return [f'openarm_{arm}_hand',
                f'openarm_{arm}_left_finger',
                f'openarm_{arm}_right_finger']

    def current_acm(self):
        """The planning scene's allowed-collision matrix, or None.

        Fetched because a scene diff carrying an ACM does not merge it --
        moveit_core replaces the matrix wholesale. Sending a four-entry matrix
        to exempt the gripper therefore deleted all 139 disable_collisions
        entries the SRDF provides, after which every adjacent link pair in the
        arm counted as a collision: the robot went red in RViz and every plan
        from that moment was invalid, PLANNING_FAILED then
        INVALID_MOTION_PLAN, with nothing in the log to say the collision
        model had been thrown away.
        """
        if not self.scene_query_client.wait_for_service(timeout_sec=2.0):
            return None
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
        result = self._await(self.scene_query_client.call_async(request), 5.0)
        if result is None:
            return None
        matrix = result.scene.allowed_collision_matrix
        return matrix if matrix.entry_names else None

    def allow_gripper_in_octomap(self, allow):
        """Let *only the gripper* touch the octomap, and nothing else.

        Turning collision checking off for the whole descent was too blunt.
        The reason the descent needs an exemption at all is narrow: on a
        top-down grasp the object being picked up is itself in the map, so the
        fingers have to enter occupied voxels to reach it. That says nothing
        about the forearm, the elbow or the upper arm, and with checking off
        entirely nothing stops those sweeping into the table on the way down.

        So the allowed-collision matrix gets an entry for the gripper links
        against <octomap>, and the descent is planned *checked*. The gripper
        may pass through the object; the rest of the arm may not pass through
        anything.

        The matrix is read, added to, and sent back whole. It cannot be sent
        as a small diff -- see current_acm.
        """
        matrix = self.current_acm()
        if matrix is None:
            self.get_logger().warn(
                'could not read the allowed-collision matrix, so the gripper '
                'cannot be exempted from the octomap on its own; the descent '
                'will fall back to switching collision checking off entirely')
            return False
        if not self.scene_client.wait_for_service(timeout_sec=2.0):
            return False

        links = self.gripper_links()
        names = list(matrix.entry_names)
        rows = [list(entry.enabled) for entry in matrix.entry_values]
        if OCTOMAP_NAME not in names:
            names.append(OCTOMAP_NAME)
            for row in rows:
                row.append(False)
            rows.append([False] * len(names))
        index = {name: i for i, name in enumerate(names)}
        octomap = index[OCTOMAP_NAME]
        touched = 0
        for link in links:
            if link not in index:
                continue
            row = index[link]
            rows[row][octomap] = bool(allow)
            rows[octomap][row] = bool(allow)
            touched += 1
        if not touched:
            self.get_logger().warn(
                'none of the gripper links are in the collision matrix, so '
                'there is nothing to exempt')
            return False

        updated = AllowedCollisionMatrix()
        updated.entry_names = names
        for row in rows:
            entry = AllowedCollisionEntry()
            entry.enabled = row
            updated.entry_values.append(entry)
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = updated
        request = ApplyPlanningScene.Request()
        request.scene = scene
        result = self._await(self.scene_client.call_async(request), 5.0)
        if result is None or not result.success:
            self.get_logger().warn(
                f'the planning scene refused the gripper/octomap exemption '
                f'({"granted" if allow else "withdrawn"} was asked for)')
            return False
        self.get_logger().info(
            f'{"allowing" if allow else "no longer allowing"} {touched} '
            f'gripper links to touch the octomap; the other '
            f'{len(names) - touched - 1} entries in the matrix are untouched')
        return True

    def probe_cartesian(self, start_joints, position, quat,
                        avoid_collisions=True):
        """Would a straight line to `position` solve from `start_joints`?

        Plans and throws the trajectory away. Returns (fraction, end_joints),
        or (None, None) if the service could not answer at all -- note that
        0.0 is a real answer and None is not.

        This is the whole point of the pre-flight. Joint headroom says a
        posture is comfortable; it does not say a line out of it solves. The
        run that prompted this had joint1 at -1.356 with 0.040 rad left before
        its stop *before the descent started*, and got 18% of the line: the
        posture was fine, the direction it then had to travel was not. Asking
        the same question of /compute_cartesian_path with an explicit start
        state costs one service call and no motion at all.
        """
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            return None, None
        request = self._cartesian_request(position, quat, avoid_collisions,
                                          start_joints=start_joints)
        result = self._await(self.cartesian_client.call_async(request), 20.0)
        if result is None or result.error_code.val != MOVEIT_SUCCESS:
            return (0.0, None) if result is not None else (None, None)
        points = result.solution.joint_trajectory.points
        names = list(result.solution.joint_trajectory.joint_names)
        end = None
        if points:
            index = {n: i for i, n in enumerate(names)}
            if all(j in index for j in self.arm_joints):
                end = [points[-1].positions[index[j]] for j in self.arm_joints]
        return float(result.fraction), end

    def joint_travel(self, trajectory):
        """Worst per-joint distance travelled along a trajectory, radians.

        Cumulative, not end-to-end, so a joint that sweeps out and back is
        counted for both halves.
        """
        points = trajectory.joint_trajectory.points
        if len(points) < 2:
            return 0.0
        totals = [0.0] * len(points[0].positions)
        for first, second in zip(points, points[1:]):
            for i, (a, b) in enumerate(zip(first.positions, second.positions)):
                totals[i] += abs(b - a)
        return max(totals) if totals else 0.0

    def cartesian_move(self, position, quat, label, avoid_collisions=True,
                       max_travel=None):
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
            return None

        # Timed because a cycle's wall clock is not where it looks. TRANSIT
        # took 18.6 s of a measured 76 s run while moving one joint 2.58 rad,
        # which at any sane speed scaling is under 2 s of motion -- so the
        # other 17 s went on planning, or waiting, and there was no way to
        # tell which from the log. plan_s and exec_s say.
        planning_from = self.get_clock().now().nanoseconds * 1e-9
        request = self._cartesian_request(position, quat, avoid_collisions)
        result = self._await(self.cartesian_client.call_async(request), 20.0)
        plan_s = round(self.get_clock().now().nanoseconds * 1e-9
                       - planning_from, 2)
        if result is None:
            self.get_logger().warn(f'{label}: /compute_cartesian_path did not answer')
            self.log_motion(label, 'cartesian', 'no-answer',
                            target=[round(v, 5) for v in position],
                            checked=avoid_collisions)
            return None
        if result.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().info(
                f'{label}: no straight line ({result.error_code.val})')
            self.log_motion(label, 'cartesian', 'refused',
                            target=[round(v, 5) for v in position],
                            checked=avoid_collisions,
                            code=result.error_code.val)
            return None

        wanted = self.get_parameter('cartesian_min_fraction').value
        usable = self.get_parameter('cartesian_partial_min').value
        if result.fraction < min(wanted, usable):
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
            return None

        trajectory = self._retime(result.solution)
        if not trajectory.joint_trajectory.points:
            self.get_logger().warn(f'{label}: empty Cartesian trajectory')
            return None

        # A straight tool path is not the same as a safe arm motion.
        #
        # /compute_cartesian_path only constrains the *tool*. It will happily
        # return a path whose every waypoint is on the line while the shoulder
        # and elbow sweep right across the workspace -- spread over enough
        # waypoints that no single step is large, so jump_threshold never
        # fires. Measured on this robot, worst per-joint travel for a 150 mm
        # descent:
        #
        #   right DESCEND, worked, 32.2 mm out    0.44 rad
        #   right DESCEND, worked, 24.8 mm out    0.43 rad
        #   right LIFT,    worked,  7.5 mm out    0.49 rad
        #   left  DESCEND, 287 mm out             3.75 rad
        #   left  DESCEND, 323 mm out             3.26 rad   <- hit the table
        #
        # The tool traced the line in every case. The last two asked joint2 for
        # 178 degrees on the way down, which is the arm going through the
        # table rather than to the object.
        if max_travel is not None and max_travel > 0.0:
            travel = self.joint_travel(trajectory)
            if travel > max_travel:
                self.get_logger().error(
                    f'{label}: the line solves, but flying it costs '
                    f'{travel:.2f} rad on one joint -- more than the '
                    f'{max_travel:.2f} rad a leg this short should need. The '
                    f'tool would follow the line while the arm swings across '
                    f'the workspace. Refusing.')
                self.log_motion(label, 'cartesian', 'refused-joint-travel',
                                target=[round(v, 5) for v in position],
                                checked=avoid_collisions,
                                fraction=round(result.fraction, 4),
                                travel_rad=round(travel, 3),
                                limit_rad=round(max_travel, 3))
                return None

        # Where the plan's own last point puts the tool, by forward
        # kinematics. Without this a leg that reports fraction=1.0 and lands
        # 287 mm away -- measured, left arm, DESCEND -- cannot be attributed:
        # a plan that ends at the target means the arm did not follow it, and
        # a plan that ends elsewhere means the request was wrong.
        planned_tip = None
        chain = self.kinematics()
        last = (trajectory.joint_trajectory.points[-1]
                if trajectory.joint_trajectory.points else None)
        if chain is not None and last is not None:
            names = list(trajectory.joint_trajectory.joint_names)
            order = {n: i for i, n in enumerate(names)}
            if all(j in order for j in self.arm_joints):
                values = [last.positions[order[j]] for j in self.arm_joints]
                planned_tip = [round(float(v), 5) for v in
                               chain.pose(chain.tool_link, values)[:3, 3]]

        points = len(trajectory.joint_trajectory.points)
        share = ('' if result.fraction >= wanted
                 else f' ({result.fraction * 100:.0f}% of it -- the rest '
                      f'follows as another line)')
        self.get_logger().info(
            f'{label}: straight line to ({position[0]:.3f}, {position[1]:.3f}, '
            f'{position[2]:.3f}), {points} points{share}')
        before = self.joint_snapshot()
        # The commanded path, as move_group interpolated it.
        planned = [
            {'t': round(p.time_from_start.sec
                        + p.time_from_start.nanosec * 1e-9, 3),
             'joints': [round(v, 5) for v in p.positions]}
            for p in trajectory.joint_trajectory.points]
        finish = self.sample_motion()
        executing_from = self.get_clock().now().nanoseconds * 1e-9
        ok = self._execute(trajectory, label)
        exec_s = round(self.get_clock().now().nanoseconds * 1e-9
                       - executing_from, 2)
        # Measured from a transform stamped after the move finished. Read
        # without that, the buffer can still hold the pose from before it, and
        # then a leg that went somewhere else entirely looks like it arrived --
        # which is exactly what the arrival check below exists to catch.
        done_at = self.get_clock().now().nanoseconds * 1e-9
        landed = self.tcp_position(newer_than=done_at)
        error = (None if landed is None
                 else math.dist(landed, position))
        # How far the plan itself ended from the target, by FK. If this is
        # small and error_mm is large, the arm did not follow its trajectory;
        # if both are large, the request was wrong.
        plan_error = (None if planned_tip is None
                      else math.dist(planned_tip, position))
        self.log_motion(label, 'cartesian', 'ok' if ok else 'execution-failed',
                        target=[round(v, 5) for v in position],
                        checked=avoid_collisions,
                        fraction=round(result.fraction, 4),
                        plan_s=plan_s, exec_s=exec_s,
                        points=points, before=before,
                        landed=(None if landed is None
                                else [round(v, 5) for v in landed]),
                        error_mm=(None if error is None
                                  else round(error * 1000, 1)),
                        planned_tip=planned_tip,
                        plan_error_mm=(None if plan_error is None
                                       else round(plan_error * 1000, 1)),
                        planned=planned, path=finish())
        if plan_error is not None and plan_error > 0.02:
            self.get_logger().error(
                f'{label}: the plan reported {result.fraction * 100:.0f}% of '
                f'the line but its own last point leaves the tool '
                f'{plan_error * 1000:.0f} mm from the target -- the request '
                f'and the trajectory disagree, which is not something the arm '
                f'can be blamed for')
        if not ok:
            return None
        # Arriving is part of flying the line. A leg that reports the whole
        # line and leaves the tool a quarter of a metre away is not a leg that
        # worked, and treating it as one let the cycle carry on to close the
        # gripper somewhere else entirely.
        limit = self.get_parameter('pose_abort_limit').value
        if error is not None and limit > 0.0 and error > limit:
            self.get_logger().error(
                f'{label}: the line reported '
                f'{result.fraction * 100:.0f}% but the tool settled '
                f'{error * 1000:.0f} mm from the target, past the '
                f'{limit * 1000:.0f} mm this is allowed to be out by. '
                f'Refusing the leg rather than carrying on from the wrong '
                f'place.')
            self.log_motion(label, 'cartesian', 'landed-far',
                            target=[round(v, 5) for v in position],
                            landed=[round(v, 5) for v in landed],
                            error_mm=round(error * 1000, 1),
                            planned_tip=planned_tip)
            return None
        # The share actually flown, so the caller can tell "went most of the
        # way" from "went nowhere". Returning a bool for both is what sent a
        # 95.65% line to the curved fallback.
        return result.fraction

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

    def settle_at(self, position, label, timeout=None):
        """Wait for the tool to stop creeping, and report where it ended up.

        The servo has a slow integral term, and this is what it looks like from
        outside: held at one target the tool error went 19.3 -> 19.0 -> 18.8 ->
        18.6 -> 15.5 -> 13.4 -> 11.5 -> 10.7 mm over about eleven seconds. So
        the way to get the tool onto its target is to *wait*, not to command
        again -- the intermediate samples show a re-issued pass moving the arm
        not at all, because the joints already sit at their solution and the
        error is between that solution and reality.

        Returns (position, error) using the last reading, or (None, None).
        """
        if timeout is None:
            timeout = self.get_parameter('pose_settle_time').value
        tolerance = self.get_parameter('pose_tolerance').value
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        best, error, previous = None, None, None
        while True:
            now = self.get_clock().now().nanoseconds * 1e-9
            landed = self.fresh_tcp()
            if landed is None:
                return None, None
            error = math.dist(landed, position)
            best = landed
            if error <= tolerance:
                return best, error
            # Stop early once it stops improving: creeping 0.2 mm per second is
            # not going to arrive, and waiting the full timeout for that is the
            # sort of dead time that reads as the robot being stuck.
            if previous is not None and previous - error < 0.0005:
                break
            previous = error
            if now > deadline or self._abort.wait(0.5):
                break
        return best, error

    def settle_joints(self, wanted, label, timeout=None):
        """Wait for the arm to stop creeping toward a commanded posture.

        The joint trajectory controller here has no `constraints` block, so it
        enforces no goal tolerance and reports SUCCESS the moment the
        trajectory ends -- whatever the arm is actually doing. What it is
        actually doing is still converging: the hardware runs an integral term
        that pulls in over seconds, which settle_at documents from the tool
        side (19.3 -> 10.7 mm over about eleven seconds).

        Measured consequence of not waiting: the approach reported ok, the
        descent was attempted 1.2 s later with joint1 still 80.5 mrad from its
        commanded value -- 35 mm at the tool -- and the straight line that
        solved 100% from the intended posture solved 47% from the real one.

        Returns the worst per-joint error left, or None with no readings.
        """
        if timeout is None:
            timeout = self.get_parameter('posture_settle_time').value
        tolerance = self.get_parameter('posture_settle_tolerance').value
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        worst, previous = None, None
        while True:
            worst = self.joint_error(wanted)
            if worst is None:
                return None
            if worst <= tolerance:
                return worst
            # Same early exit as settle_at: creeping a fraction of a
            # milliradian per second is not going to arrive, and standing
            # still for the whole timeout reads as the robot being stuck.
            if previous is not None and previous - worst < 0.0005:
                break
            previous = worst
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                break
            if self._abort.wait(0.5):
                break
        self.get_logger().info(
            f'{label}: posture settled {worst * 1000:.0f} mrad from the '
            f'commanded joints')
        return worst

    def converge_to(self, position, quat, label, avoid_collisions=True,
                    correct=True, max_travel=None):
        """Drive to `position` and get the tool onto it: move, settle, correct.

        A move that MoveIt calls SUCCESS has satisfied the *joint* trajectory
        controller's tolerance, which is not the same as the tool being where
        it was sent. Measured on this robot: the tool landed 13.8 mm low and
        12.1 mm short in x, repeatably, at a commanded z of 0.412.

        Three steps, at most two moves:

        1. Fly the line.
        2. Wait for the servo's integral term to pull the tool in -- see
           settle_at.
        3. If it is still out, command the target *offset by the measured
           error* once. That is the only thing that can help a residual, and
           the reason it works is that the offset is repeatable: dz -13.8,
           -14.0, -14.0, -14.0 mm across four passes at the same target.

        Re-issuing the same target instead, which is what this used to do, does
        nothing at all: the joints are already at their solution, so the arm
        does not move. It cost ten moves per leg and arrived no closer.
        """
        tolerance = self.get_parameter('pose_tolerance').value
        segments = max(1, int(self.get_parameter('cartesian_segments').value))
        # A line that only partly solves is flown as far as it goes and then
        # continued, so the whole leg is straight. Distinct from re-commanding
        # an already-achieved pose, which moves nothing: here the arm really
        # does advance each segment, and the fraction says so.
        sent_at = self.get_clock().now().nanoseconds * 1e-9
        wanted = self.get_parameter('cartesian_min_fraction').value
        remaining = None
        for segment in range(1, segments + 1):
            # Both ends of the comparison have to be current. Reading `was`
            # from the stale buffer made a leg that flew 143.5 mm of its 150
            # look like it had lost 6.5 mm, and the stall guard threw it away.
            was = self.fresh_tcp()
            flown = self.cartesian_move(position, quat, label,
                                        avoid_collisions=avoid_collisions,
                                        max_travel=max_travel)
            # Timed *after* the move, not before it. A transform published
            # between sending the goal and the tool actually moving already
            # post-dates the send, so newer_than=<sent> can be satisfied by a
            # pre-move pose -- and then the segment looks like it gained
            # nothing and the stall guard throws away a line that flew. Seen
            # once in three runs: "1 line, 10.0 mm off" where two segments and
            # 2.0 mm was the norm.
            finished = self.get_clock().now().nanoseconds * 1e-9
            if flown is None:
                # No line at all. If the tool is already close, the gap left is
                # the servo's steady-state offset rather than a path problem --
                # asking for a line to a pose the arm cannot hold returns 0%
                # for ever. Hand it to the settle-and-offset path below instead
                # of failing the leg, which is what sent a 12.1 mm residual to
                # eight curved hops.
                if segment > 1:
                    here = self.fresh_tcp()
                    if here is not None and math.dist(here, position) <= \
                            self.get_parameter('pose_residual_limit').value:
                        self.get_logger().info(
                            f'{label}: no line for the last '
                            f'{math.dist(here, position) * 1000:.1f} mm; that '
                            f'is the standing offset, not the path')
                        break
                return False
            if flown >= wanted:
                break

            # Stop unless the tool actually got closer. A line that stalls at
            # the same place every segment is stalling against something --
            # continuing to re-request it walks the gripper into whatever that
            # is, one short segment at a time, which is precisely the failure
            # this whole mechanism exists to avoid.
            #
            # newer_than is not optional here. Read without it, the buffer
            # still holds the pre-move transform, every segment looks like it
            # gained nothing, and a line that was flying perfectly well gets
            # abandoned. Third time this has bitten: joint states, the
            # convergence check, and now this.
            now = self.tcp_position(newer_than=finished)
            if was is not None and now is not None:
                left = math.dist(now, position)
                gained = math.dist(was, position) - left
                # Deliberately no residual shortcut here: while segments are
                # still gaining ground they are flying real distance, and
                # stopping at 20 mm because that happens to be offset-sized
                # would abandon a leg that was working. The residual path
                # applies only when there is no line at all, and the stall
                # guard below covers making no progress.
                if left <= tolerance:
                    # Arrived. Segments after this would gain nothing by
                    # definition, and judging that as a stall would throw away
                    # a leg that had just succeeded.
                    break
                if gained < self.get_parameter('cartesian_min_gain').value:
                    self.get_logger().warn(
                        f'{label}: the line stalls {left * 1000:.1f} mm short '
                        f'and segment {segment} gained only '
                        f'{gained * 1000:.1f} mm -- something is stopping it, '
                        f'so not pushing further')
                    self.log_motion(label, 'cartesian', 'stalled',
                                    target=[round(v, 5) for v in position],
                                    error_mm=round(left * 1000, 1),
                                    gained_mm=round(gained * 1000, 1))
                    return False
                remaining = left

            if segment == segments:
                self.get_logger().warn(
                    f'{label}: {segments} straight segments still did not '
                    f'reach the target'
                    + ('' if remaining is None
                       else f', {remaining * 1000:.1f} mm short'))
                return False
            self.get_logger().info(
                f'{label}: flew {flown * 100:.0f}% of the line; continuing '
                f'straight, segment {segment + 1}')
        # Also timed from the end of the last move rather than from when the
        # first was sent, for the reason above.
        landed = self.tcp_position(newer_than=max(sent_at, finished))
        if landed is None:
            self.get_logger().warn(
                f'{label}: no {self.tcp_frame} transform, so the achieved pose '
                f'cannot be checked')
            return True

        landed, error = self.settle_at(position, label)
        if landed is None:
            return True
        if error <= tolerance:
            return True

        if not correct or not self.get_parameter(
                'pose_offset_correction').value:
            self.get_logger().warn(
                f'{label}: tool settled {error * 1000:.1f} mm from the target '
                f'and offset correction is off here')
            return True

        offset = [position[i] - landed[i] for i in range(3)]
        corrected = tuple(position[i] + offset[i] for i in range(3))
        self.get_logger().info(
            f'{label}: tool settled {error * 1000:.1f} mm out '
            f'(dx {offset[0] * 1000:+.1f} dy {offset[1] * 1000:+.1f} '
            f'dz {offset[2] * 1000:+.1f}); aiming past it by the same amount')
        self.log_motion(label, 'cartesian', 'offset-correction',
                        target=[round(v, 5) for v in position],
                        landed=[round(v, 5) for v in landed],
                        error_mm=round(error * 1000, 1),
                        offset_mm=[round(v * 1000, 1) for v in offset])

        if self.cartesian_move(corrected, quat, f'{label} corrected',
                               avoid_collisions=avoid_collisions,
                               max_travel=max_travel) is None:
            # The correction is a bonus, not a requirement: the first move did
            # land, just not accurately.
            return True
        landed, error = self.settle_at(position, label)
        if landed is not None and error > tolerance:
            self.get_logger().warn(
                f'{label}: still {error * 1000:.1f} mm out after the offset '
                f'correction. The arm has a steady-state error the servo is '
                f'not closing -- rebuild openarm_hardware so the gravity '
                f'model includes the hand, and check joint4 in the motion log.')
        return True

    def descend_column(self, xy, from_z, to_z, quat, label,
                       uncheck_collisions=False, linear_only=False):
        try:
            return self._descend_column(xy, from_z, to_z, quat, label,
                                        uncheck_collisions, linear_only)
        finally:
            # Never leave the gripper exempt. The exemption is for one leg;
            # left in place it would silently apply to the carry, the drop and
            # everything after.
            if self._octomap_exempt:
                self.allow_gripper_in_octomap(False)
                self._octomap_exempt = False

    def _descend_column(self, xy, from_z, to_z, quat, label,
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
        # Whether to bother asking for a collision-checked line first.
        #
        # On a top-down grasp the answer is no. The *target* is in the
        # collision world: the octomap holds the object being picked up and the
        # table under it, so the gripper is required to enter occupied voxels
        # to reach the thing it is grasping. A checked line therefore stalls
        # about a centimetre above the object every time -- measured, 25% of
        # the last 5 cm -- and asking costs a planning round trip and, worse,
        # can fly a partial line that leaves the tool somewhere the rest does
        # not solve from.
        #
        # Only on legs the caller has already marked as exempt
        # (uncheck_collisions), which are the short straight vertical ones
        # between reach-checked ends with min_grasp_z as a hard floor.
        # Preferred: keep checking, and exempt only the gripper from the
        # octomap. Turning checking off for the whole arm is the fallback, and
        # it is what let the forearm reach the table on a descent whose tool
        # path was perfectly fine.
        exempt = False
        if uncheck_collisions and self.get_parameter(
                'gripper_octomap_exemption').value:
            exempt = self.allow_gripper_in_octomap(True)
            self._octomap_exempt = exempt
        skip_checked = (not exempt and uncheck_collisions
                        and self.get_parameter(
                            'descend_ignores_octomap').value)
        if self.get_parameter('linear_descent').value:
            # A correction on this leg is a *second* move, and on the
            # descent that is exactly the extra motion the single-descent work
            # exists to remove: measured, the line flew 100% and landed 16.2 mm
            # out, the correction then aimed 12 mm past the target, moved the
            # tool 2 mm sideways and left it 31.7 mm out -- worse than before,
            # and visible as a loop before a second descent.
            #
            # The offset it was built for was repeatable and almost purely
            # vertical (dz -13.8, -14.0, -14.0, -14.0 mm). This one is not, and
            # aiming past a non-repeatable error just overshoots.
            correct = self.get_parameter('descend_offset_correction').value
            # The column legs are the ones near the table, so they carry the
            # joint-travel limit: a 150 mm line that costs three radians is
            # the arm going through the surface, not down to the object.
            budget = self.get_parameter('column_max_joint_travel').value
            if skip_checked:
                self.get_logger().info(
                    f'{label}: straight line without collision checking -- '
                    f'the object being grasped is itself in the octomap, so a '
                    f'checked descent onto it cannot complete')
                if self.converge_to(target, quat, label,
                                    avoid_collisions=False, correct=correct,
                                    max_travel=budget):
                    return True
            elif self.converge_to(target, quat, label, correct=correct,
                                  max_travel=budget):
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
            if uncheck_collisions and not skip_checked:
                self.get_logger().info(
                    f'{label}: retrying the straight line without collision '
                    f'checking -- the object being grasped is itself in the '
                    f'octomap')
                # The budget applies here too. Without it this retry flew
                # the very path the checked attempt had just refused: 3.28 rad
                # of joint travel for a 150 mm descent, which swept the arm
                # back down to near its folded pose, 397 mm from the target,
                # through whatever was in the way. Turning collision checking
                # off is a licence for the *gripper* to enter the object's
                # voxels, not for the arm to take a different route.
                if self.converge_to(target, quat, label,
                                    avoid_collisions=False,
                                    max_travel=budget):
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
                f'object still at the pick point '
                f'({dist(left_behind[0]["point"], pick_point):.3f} m) '
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
            # A crash mid-descent leaves the arm down at the object as surely
            # as a failure does, and the next cycle starts by going home.
            try:
                self.clear_the_surface('after the cycle crashed')
            except Exception as lift_exc:
                self.get_logger().error(
                    f'could not lift clear after the crash: {lift_exc}')
        finally:
            with self._lock:
                self._busy = False

    def _cycle(self):
        self._failures = []
        self._column = None
        self._holding = False
        self._last_detection = None
        self._last_payload = None
        self._cycle_count += 1
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

        # And that it can actually plan, which the checks above do not show.
        # A goal to where the arm already is, plan_only: no motion, and a
        # failure cannot be about the target.
        if self.get_parameter('check_planner_ready').value:
            planning = self.planner_is_planning()
            if planning is False:
                self._set_state(
                    'FAILED',
                    'the planner is not planning -- a goal to the arm\'s own '
                    'current posture failed, so this is the stack, not the '
                    'target. See the log for what to check.')
                return
            if planning is None:
                self.get_logger().warn(
                    'could not confirm the planner is planning; continuing')

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
            self._set_state(
                'FAILED',
                f'could not reach home -- {self._worst_joint(self.home_positions())}')
            return

        if self.arm_selection == 'by_side':
            chosen_from = self.arm
            states = self.choose_arm(states)
            if states is None or states is OUT_OF_REACH:
                return
            # Only if it actually switched. Re-homing regardless would clear
            # and rebuild the octomap a second time every cycle for nothing.
            if self.arm != chosen_from and not self.arrive_at_home():
                self._set_state(
                    'FAILED',
                    f'could not reach the {self.arm} home -- '
                    f'{self._worst_joint(self.home_positions())}')
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
            # A failure *after* the geometry was proved is not something the
            # ladder fixes. Its rungs change the yaw or drop 8 mm and try
            # again from the staging pose -- worth a roll of the dice when the
            # descent was an unknown, but once the whole column has been
            # checked in advance the next rung will find the same geometry and
            # the only visible effect is the arm travelling back to pre-pick
            # and home for another identical go.
            if self._preflight_choice is not None and not self.get_parameter(
                    'retry_after_preflight').value:
                self._set_state(
                    'FAILED',
                    'the descent was checked and the posture chosen before '
                    'moving, so this is a real fault rather than something a '
                    f'different yaw recovers -- {self._failure_summary()}')
                self._retreat_to_home(states, 'after a failed pick')
                return
        else:
            self._set_state(
                'FAILED',
                f'every pick strategy was exhausted -- {self._failure_summary()}')
            self._retreat_to_home(states, 'after a failed pick')
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
            # Retreat anyway. A failed place used to return here and leave the
            # arm wherever it stopped -- extended over the work surface, often
            # still holding the object -- which is both the worst place to
            # leave it and the hardest place to start the next cycle from.
            # The same way out as a successful place: staging pose, then home.
            self._retreat_to_home(states, 'after a failed place')
            self._set_state('FAILED', 'place failed')
            return

        # Back the way it came: the staging pose, then home. PRE_PICK is a
        # posture the arm is known to be able to reach from both the drop and
        # from home, which is what makes it a safe waypoint out.
        self._retreat_to_home(states, 'cycle complete')
        self._set_state('DONE')

    def _retreat_to_home(self, states, why):
        """Back the way it came: the staging pose, then home.

        PRE_PICK is a posture the arm is known to be able to reach from both
        the drop and from home, which is what makes it a safe waypoint out --
        and it is above the work surface, so the trip home does not sweep
        across it.

        Used after a failed place as well as a successful one. Leaving the arm
        stopped where it failed, extended over the table and often still
        holding the object, is the worst outcome available.
        """
        # Before the staging pose, which is a joint goal: if the arm is still
        # down at the object, that goal is what sweeps it across the surface.
        self.clear_the_surface(why)
        self._set_state('PRE_PICK', f'on the way back, {why}')
        if not self.move_to_state(PRE_PICK_STATE, states):
            self.get_logger().warn(
                'could not stage through pre_pick on the way back; going '
                'straight home')
        self._set_state('HOME', why)
        self.move_to_home()

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

        # Keep it. The first pick attempt used to detect again immediately --
        # two full inference waits back to back at the same stationary object
        # -- and _detection_is_fresh() is what lets that attempt use this
        # answer instead.
        self._last_detection = detection
        self._last_payload = payload

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

        # Say which authorities actually answered. Claiming "cuMotion said
        # no" when cuMotion was never asked -- it takes Cartesian goals for
        # one link per bringup, so the other arm's tool returns
        # INVALID_LINK_NAME -- sent the reader after the workspace when the
        # verdict had rested on KDL alone.
        tool = self.planner_ee_link()
        asked_planner = [arm for arm in candidates
                         if tool == f'openarm_{arm}_hand_tcp']
        who = ('/compute_ik, the kinematic solver here, and a plan-only '
               f'cuMotion query for the {asked_planner[0]} arm'
               if asked_planner else
               '/compute_ik and the kinematic solver here (cuMotion could not '
               f'be asked: it plans Cartesian goals for {tool} only)')
        message = (
            f'out of reach: no arm can pick the object {where}. '
            f'Tried {"; ".join(unreachable)}. {who} all said no, so nothing '
            f'moved. The grip is asked to be within '
            f'{math.degrees(self.get_parameter("grasp_tilt_max").value):.0f} '
            f'degrees of straight down, which costs reach. Map what is '
            f'actually reachable at this height with:  python3 '
            f'native/tests/check_reachability.py --arm {candidates[0]} '
            f'--sweep --z {grasp[2]:.3f}')
        self._note_failure('REACH', message)
        self._set_state('OUT_OF_REACH', message)
        return OUT_OF_REACH

    def _detection_age(self):
        """Seconds since the detection in hand was published, or None."""
        payload = self._last_payload
        stamp = (payload or {}).get('stamp')
        if not stamp:
            return None
        return self.get_clock().now().nanoseconds * 1e-9 - stamp

    def _detection_is_fresh(self):
        """Whether the detection in hand is new enough to pick from again.

        choose_arm() detects in order to decide which arm can reach, and the
        first attempt then detected again straight away -- two full inference
        waits at the same stationary object. Measured, run 1788859355: LOCATE
        from 5.0 s to 15.0 s and again from 19.0 s to 31.6 s. 22.6 s of a 76 s
        cycle spent looking twice.

        Nothing moves the object between those two calls, and HOME is out of
        the camera's frame by design, so the second answer is the first
        answer. A later rung of the ladder is a different matter -- tens of
        seconds have passed and a failed grasp may well have nudged the
        object -- and an age bound separates the two without needing to know
        which caller is asking: the gap after choose_arm is a few seconds, the
        gap after a failed attempt is nearer a minute.
        """
        limit = self.get_parameter('detection_reuse_age').value
        if limit <= 0.0:
            return False
        age = self._detection_age()
        return age is not None and -1.0 <= age <= limit

    def _attempt_pick(self, strategy, states):
        fresh = self._detection_is_fresh()
        if (strategy.get('redetect', True) and not fresh) \
                or self._last_detection is None:
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
            age = self._detection_age()
            self.get_logger().info(
                f'{strategy["name"]}: reusing the detection from '
                f'{age:.1f} s ago rather than waiting for another -- '
                f'{"nothing has moved since" if fresh else "this rung does not re-detect"}')
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

        # Everything that can be worked out without moving is worked out
        # here: which posture to approach in, and whether the straight line
        # down actually solves from it. Both are answerable in advance --
        # /compute_cartesian_path takes an explicit start state -- and finding
        # out mid-descent means standing over the object with the gripper open
        # and no way on.
        # The point the free-space move actually ends at, which is what the
        # posture has to be solved for.
        transit_height = self.get_parameter('transit_height').value
        staged = transit_height > self.get_parameter('approach_height').value
        approach_point = ((grasp[0], grasp[1], grasp[2] + transit_height)
                          if staged else tuple(pregrasp))
        # The approach is made from the staging pose, so that is what a
        # candidate's travel is measured against -- not where the arm happens
        # to be standing while the pre-flight runs, which is HOME.
        staging = (states.get(PRE_PICK_STATE) or {}).get('joints')
        self._approach_from = list(staging) if staging else None
        self._preflight_choice = self.preflight_column(
            grasp, pregrasp, approach_point, quat, 'PREFLIGHT')
        if self._preflight_choice is None and self._preflight_reason:
            self._note_failure('PREFLIGHT', self._preflight_reason)
            self._set_state('OUT_OF_REACH', self._preflight_reason)
            return OUT_OF_REACH
        if self._preflight_choice is not None:
            quat = self._preflight_choice['quat']

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
        transit_z = grasp[2] + transit_height
        if staged:
            self._set_state('TRANSIT')
            # The posture is chosen here, not by the planner, and the grasp
            # direction can come back turned 180 degrees -- the same grasp in a
            # roomier posture. Everything below the transit uses whichever one
            # was picked, or the descent would be fighting the wrist.
            arrived, quat = self.approach_above(
                (grasp[0], grasp[1], transit_z), quat, 'TRANSIT')
            if not arrived:
                self._note_failure('TRANSIT', (
                    f'could not plan to {transit_height * 100:.0f} cm above '
                    f'({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) -- most '
                    f"often the object is out of the {self.arm} arm's reach. "
                    f'Verify with: python3 native/tests/check_reachability.py '
                    f'--arm {self.arm} --point {grasp[0]:.3f} {grasp[1]:.3f} '
                    f'{transit_z:.3f}'))
                return None
            from_z = transit_z
        else:
            # No transit stage, so the pre-grasp itself is the approach and
            # gets the same posture choice.
            self._set_state('PREGRASP')
            arrived, quat = self.approach_above(pregrasp, quat, 'PREGRASP')
            from_z = pregrasp[2]
        if not arrived:
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

        # One descent, not two.
        #
        # There used to be a stop at the pre-grasp: down to 5 cm above the
        # object, open the gripper, then down again. Two legs, each with its
        # own line to solve, its own settle and its own offset correction --
        # and the arm visibly pausing in mid-air. Nothing needed the stop: the
        # gripper can just as well be opened before any of it, at the transit
        # height, where there is nothing to catch on.
        #
        # So the order is: open, then a single straight line all the way down
        # to the grasp, then close. single_descent false restores the stop.
        if not self.get_parameter('single_descent').value \
                and from_z > pregrasp[2] + 1e-6:
            self._set_state('PREGRASP')
            if not self.descend_column(
                    grasp, from_z, pregrasp[2], quat, 'PREGRASP',
                    uncheck_collisions=self.get_parameter(
                        'approach_ignores_octomap').value):
                self._note_failure('PREGRASP', (
                    f'could not descend to the pre-grasp at '
                    f'z={pregrasp[2]:.3f}'))
                return None
            from_z = pregrasp[2]

        # Before the descent, not between two of them: with the tool above the
        # object at transit height there is nothing for open fingers to catch
        # on, and a gripper that fails to open then costs no motion at all.
        self._set_state('OPEN_GRIPPER')
        if not self.open_gripper():
            return None

        # What the pre-flight promised was a line from the posture it chose.
        # The arm does not always hold that posture: measured at TRANSIT,
        # joint1 settled 70 mrad (4 degrees) from its commanded value, which
        # is ~35 mm at the tool -- and from *there* the same descent solved
        # 48.6% instead of 100%. Report the gap, so a descent that fails after
        # a pre-flight that passed is not a mystery.
        # Let it finish arriving first. Without this the drift below is
        # measured mid-convergence and the descent is planned from a posture
        # the arm is still leaving.
        entry = self._preflight_choice
        if entry and not entry.get('linear'):
            settling = list(entry['joints'])
            if entry.get('tilt') is not None:
                settling[6] = entry['tilt']
            self.settle_joints(settling, 'DESCEND')
        self._report_posture_drift('DESCEND')

        # From here until the tool is clear again the arm is on a vertical
        # line over the object, and every way out has to go back up it first.
        self._column = (list(grasp), list(pregrasp), list(quat))
        self._set_state('DESCEND')
        if not self.descend_column(
                grasp, from_z, grasp[2], quat, 'DESCEND',
                uncheck_collisions=self.get_parameter(
                    'approach_ignores_octomap').value,
                linear_only=self.get_parameter('descend_linear_only').value):
            unchecked = (self.get_parameter('approach_ignores_octomap').value
                         and self.get_parameter(
                             'descend_ignores_octomap').value)
            blame = ('the collision world may hold the object itself, or the '
                     'grasp is below the work surface'
                     if not unchecked else
                     'collision checking was already off for this leg, so '
                     'this is reach rather than an obstacle: the arm runs out '
                     'of travel along the line. Most often it is not standing '
                     'where the pre-flight assumed -- compare the commanded '
                     'and measured joints in the motion log at TRANSIT')
            self._note_failure('DESCEND', (
                f'arrived above the object but could not descend to '
                f'z={grasp[2]:.3f} -- {blame}'))
            self.clear_the_surface('after a failed descent')
            return None

        # Where the jaws actually are, before they close on whatever is
        # there. The descent reports its own error, but the number that
        # decides whether the grip works is this one: the plan ended exactly
        # at the commanded point (plan_error_mm 0.0) and the arm stopped
        # 29 mm above it, so the jaws closed on air 29 mm over the object and
        # the only symptom was 'closed-on-nothing' at 1.9 Nm. Say it plainly
        # instead, because no gripper setting fixes a gripper in the wrong
        # place.
        self._set_state('CLOSE_GRIPPER')
        at_jaws = self.tcp_position()
        if at_jaws is not None:
            short = at_jaws[2] - grasp[2]
            sideways = math.hypot(at_jaws[0] - grasp[0],
                                  at_jaws[1] - grasp[1])
            miss = max(abs(short), sideways)
            if miss > self.get_parameter('grasp_miss_warn').value:
                self.get_logger().warn(
                    f'closing at [{at_jaws[0]:.4f}, {at_jaws[1]:.4f}, '
                    f'{at_jaws[2]:.4f}] but the grasp was commanded at '
                    f'[{grasp[0]:.4f}, {grasp[1]:.4f}, {grasp[2]:.4f}] -- '
                    f'{short * 1000:+.0f} mm in height and {sideways * 1000:.0f} '
                    f'mm sideways. The descent plan ended on the point; the '
                    f'arm did not follow it there. If this grip fails it '
                    f'failed here, not at the fingers.')
                self.log_motion('CLOSE_GRIPPER', 'gripper', 'off-target',
                                target=list(grasp), tcp=list(at_jaws),
                                error_mm=round(miss * 1000.0, 1))
        if not self.close_gripper_to_cap():
            # Up first, jaws left closed. The grip failing does not make the
            # arm any less extended over the table, and closed jaws are the
            # slimmest thing to lift back through whatever is down there.
            self.clear_the_surface('the grip closed on nothing')
            return None

        # Straight back up, and well clear -- not just back to the pre-grasp.
        # The next move carries the object to the drop pose as a free-space
        # plan, and starting that 5 cm off the surface is what dragged the
        # gripper across the table. retreat_height defaults to transit_height,
        # so the object leaves at the same altitude the approach arrived at.
        self._set_state('LIFT')
        if not self.lift_column(grasp, pregrasp, quat, 'LIFT'):
            self._note_failure('LIFT', (
                'the object was gripped but the arm could not lift it '
                'straight up, at any height. It is still holding it, at the '
                'grasp.'))
            return None
        # Off the column and clear. Everything after this -- the carry, the
        # drop, the retreat -- is above the surface and has its own way out,
        # so leaving the column set would have the retreat fly one more line
        # straight up from wherever the drop left the tool, for nothing.
        self._column = None

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

    def lift_column(self, start, pregrasp, quat, label='LIFT'):
        """Rise up the line the tool is standing on, as far as it solves.

        Ask for the full retreat, then for less. Measured: 20 cm straight up
        from the grasp solved 12.5% checked and 0% unchecked -- there is
        simply no line that long from down there, and refusing to lift at all
        is worse than lifting less. The pre-grasp height is the floor: that
        much clearance is what the approach already proved reachable, so if
        the arm got down here it can get back up to there.

        start is where the line begins -- the grasp on the way out of a
        successful pick, and the arm's measured position when a leg failed
        partway and the two are not the same place.
        """
        heights = [start[2] + self.get_parameter('retreat_height').value]
        if pregrasp is not None:
            heights.append(pregrasp[2])
        for index, retreat_z in enumerate(heights):
            if retreat_z <= start[2] + 1e-6:
                continue
            if index:
                self.get_logger().warn(
                    f'{label}: no line to {heights[0] - start[2]:.2f} m up; '
                    f'retreating to {retreat_z - start[2]:.2f} m instead. '
                    f'Whatever moves next starts lower, so watch the '
                    f'clearance.')
            if self.descend_column(
                    start, start[2], retreat_z, quat, label,
                    uncheck_collisions=self.get_parameter(
                        'approach_ignores_octomap').value,
                    linear_only=self.get_parameter(
                        'descend_linear_only').value):
                return True
        return False

    def clear_the_surface(self, why):
        """Get the tool up off the work surface before anything else moves.

        Every way out of a pick starts with the arm extended down at the
        object -- holding it, closed on nothing, or stopped partway down --
        and the next thing the cycle does is go home. Home is a joint goal,
        and the short path in joint space from "reaching down over the table"
        to "folded at the side" goes through the table. Measured, left arm,
        run 1788859355: DESCEND fine, the grip closed on nothing at
        torque 1.912 Nm, and the very next record is HOME as a joint goal
        from tcp z=0.3715. It dragged across the surface on the way.

        So the way out is the way in, for a failed pick exactly as for a
        successful one: straight up the descent line first, gripper left as
        it is, and only then home. A no-op once the tool is clear, which it
        already is on the successful path -- LIFT has run.
        """
        column = self._column
        if column is None:
            return True
        grasp, pregrasp, quat = column
        clear_at = (pregrasp[2] if pregrasp is not None
                    else grasp[2] + self.get_parameter('retreat_height').value)
        # Where the arm actually is, not where it was told to be: a descent
        # that failed stopped somewhere between the two, and one that worked
        # still lands tens of millimetres out.
        here = self.tcp_position()
        if here is not None and here[2] >= clear_at - 0.01:
            self._column = None
            return True
        start = list(here) if here is not None else list(grasp)
        self._set_state('CLEAR', why)
        lifted = self.lift_column(start, pregrasp, quat, 'CLEAR')
        if not lifted:
            self.get_logger().error(
                f'could not lift clear of the surface {why}: the arm is '
                f'still down at the object and there is no straight line up '
                f'from where it is standing. Move it clear by hand, or with '
                f'record_states.py --play, before restarting.')
        self._column = None
        return lifted

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
        """Carry to the recorded drop_state and release there.

        Staged through the pre-pick pose on the way, because the carry starts
        low over the work surface and the drop pose is on the other side of
        it. Measured: a direct joint goal from the lift to drop_state came
        back INVALID_MOTION_PLAN (-2) three times -- the path was computed and
        then rejected, which is what a swing through the octomap looks like.
        pre_pick is above the table by construction, is a posture both ends
        are known to reach, and is already the waypoint used on the way back.

        Not fatal if the staging move fails: the direct carry is still tried,
        and it may well be fine for a drop pose on the same side.
        """
        if self.get_parameter('stage_drop_through_pre_pick').value:
            self._set_state('PRE_PICK', 'staging the carry over the table')
            if not self.move_to_state(PRE_PICK_STATE, states):
                self.get_logger().warn(
                    'could not stage the carry through pre_pick; going '
                    'straight to the drop pose, which may sweep low across '
                    'the work surface')

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
