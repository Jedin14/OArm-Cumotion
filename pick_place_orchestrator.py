#!/usr/bin/env python3
"""VLM-guided pick and place for the 7DOF-OArm, with VLM-checked retries.

    HOME -> LOCATE -> PRE_PICK -> TRANSIT -> PREGRASP -> OPEN -> DESCEND
      ^                                                             |
      |                                                             v
      +------------- PRE_PICK <- CLEAR <- failed ------- VERIFY_GRASP <- CLOSE
                                                          |
      HOME <- PRE_PICK <- CLOSE <- VERIFY_PLACE <- RELEASE <- DROP <- PRE_PICK
                                                                        ^
                                                                        +- LIFT

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

The way out is the way in: LIFT with the object held, PRE_PICK, DROP, open,
shut the jaws again at the drop pose, then back through PRE_PICK to HOME. The
trip home is made closed -- 44 mm of open fingers on a moving arm is something
looking for an edge to catch on, and the arm ends the cycle in the shape it
started it.
PRE_PICK is a posture reachable from both ends, which is what makes it a safe
waypoint rather than a straight dash home across the workspace.

HOME is only ever commanded from PRE_PICK. It is a joint goal to a folded
posture, and the short path in joint space from a low, extended one goes
through the work surface -- so when pre_pick cannot be reached on the way back
the arm lifts higher, tries once more, and then stops and reports rather than
sweeping. See home_requires_pre_pick.

A pick that *fails* does not leave by that door at all: see safe_shutdown.
It has an arm somewhere unplanned, often still holding the object and often
with its own start state already in collision, so it gets a refuge checked
against the collision world before anything moves, and the motors off once it
is parked.

A pick that fails below TRANSIT still comes back up the same way first. Every
failure below TRANSIT -- the descent stopping short, the jaws closing on
nothing, the grasp not verifying, the cycle crashing outright -- leaves the
arm extended down at the object, and PRE_PICK and HOME are joint
goals: the short path in joint space
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

import contextlib
import math
import json
import os
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
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
                             GetPlanningScene, GetPositionIK,
                             GetStateValidity)
from controller_manager_msgs.srv import SetHardwareComponentState
from lifecycle_msgs.msg import State as LifecycleState
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from vlm_prompt import to_detection_prompt

import arm_kinematics
import grasp_recipes
import pick_place_sequence as sequence_model

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

# take_up_the_view returns this when the detector answered and there was
# nothing in the frame. Distinct from False, which is "could not get
# somewhere to look from": one is a missing object and the other is a
# missing robot, and driving home to look a second time helps neither.
NOTHING_SEEN = 'nothing-seen'

# And this is "nobody answered at all", which is a different report and a
# different thing to go and do about it. An empty frame means the table is
# empty; this means the detector is not running.
NO_DETECTOR = 'no-detector'

# Distinguishes "the planner node is not in the graph" from "it did not answer",
# because only the first is worth refusing to start over.
DEAD_PLANNER = 'planner-not-running'
# _attempt_pick returns this when the motion all worked and the jaws still
# found nothing. Distinct from None -- a failure of the *motion* -- because
# the remedies are opposite: nothing recovers a descent that will not fly from
# the posture the pre-flight already proved, whereas a grip that missed is
# exactly what the ladder's 8-mm-lower rungs are for. Measured, run
# 1788869179 cycle 2: DESCEND landed 1.4 mm from its commanded height and the
# jaws closed on air 11 mm above a screwdriver, and the cycle gave up after
# one attempt because the two cases were indistinguishable here.
GRASP_MISSED = object()

# _attempt_pick returns this instead of None when the object is simply not
# reachable. None means "this attempt failed, try the next strategy"; this
# means "no strategy will help", and the ladder stops.
OUT_OF_REACH = 'out-of-reach'

# Every state a cycle can end on. The panel imports this to decide when to
# re-enable its buttons: a terminal state it does not recognise leaves Pick
# greyed out with no way back, which is what adding OUT_OF_REACH did.
#
# STOPPED belongs here for the same reason. It is where safe_shutdown ends
# when nothing safe could be reached -- the arm is stranded, the motors are
# still on, and there is nothing further coming. Leaving the buttons dead
# then is the worst moment to do it: clearing the octomap and asking again
# is exactly what the operator needs to be able to do.
# HOLDING is terminal in exactly the same sense: the pick has finished and
# nothing further is coming until the operator presses something. It is the
# *good* resting state of a pick -- object in the jaws, arm at the staging
# pose -- and leaving the buttons dead there would strand the object.
TERMINAL_STATES = ('DONE', 'FAILED', 'ABORTED', 'OUT_OF_REACH', 'STOPPED',
                   'IDLE', 'HOLDING')

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
    # Height before yaw, because the yaw comes from the detector's own
    # measured object axis and the height is the thing that has been
    # wrong. Measured, run at 15:55: a screwdriver with the orientation
    # visibly right closed on nothing, and the ladder spent its next rung
    # turning the wrist 90 degrees instead of going lower.
    {'name': 'lower-8mm', 'yaw_offset': 0.0, 'z_offset': -0.008,
     'refresh_octomap': False, 'redetect': False},
    {'name': 'yaw+90', 'yaw_offset': math.pi / 2, 'z_offset': 0.0,
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


def thin(items, limit):
    """At most `limit` of them, evenly spaced, keeping the first and last.

    Used on a retrace: two seconds of 50 Hz samples is a hundred waypoints
    describing a path a dozen would describe as well, and the last one is
    the destination, so it is the one that must survive.
    """
    if limit < 2 or len(items) <= limit:
        return list(items)
    step = (len(items) - 1) / float(limit - 1)
    kept = [items[int(round(i * step))] for i in range(limit - 1)]
    kept.append(items[-1])
    return kept


class ContactMonitor:
    """Decides, reading by reading, whether the arm is leaning on something.

    Kept apart from the thread that feeds it so the rule can be tested
    against recorded runs, because every wrong answer this has given was the
    rule misreading ordinary motion rather than anything to do with threads,
    and the numbers that produced them are all in motion_log.jsonl.

    Contact has to show in two independent ways before a move is stopped:

      torque   a joint's |effort| is `margin` Nm above what that same joint
               was pulling `window` seconds ago, and has stayed there for
               `hold` seconds. The baseline rolls, so the gravity that takes
               joint1 from +3.4 to -10.5 Nm over one transit is never in the
               comparison -- only a step is.
      motion   the arm is `lag_limit` rad or more behind the trajectory it is
               flying, or it has stopped making progress along it: moving
               less than `stall_share` of what the plan asked for over the
               last `hold` seconds, when the plan asked for at least
               `stall_floor`.

    Neither alone is safe. Torque alone stopped a perfectly good transit --
    on the shoulder *unloading* as it swung out, 3.42 to 1.38 Nm, run
    1789012345 -- and the cycle then spent two minutes failing to recover
    from a move that had worked. Motion alone fires on any servo that is
    merely slow. Together they are what an arm being held up looks like:
    pulling harder and getting nowhere.

    The two motion tests are not redundant. Lag catches a long free-space
    move stopped early, where the plan runs away from the arm. Stall catches
    a short one -- a 10 cm descent asks for maybe 0.3 rad in total, so the
    lag can never grow to a fixed threshold, and what gives it away is that
    the arm went nowhere while the plan kept moving.
    """

    def __init__(self, joint_names, *, margin, window, hold, rewind,
                 lag_limit=0.0, plan=None, stall_floor=0.01,
                 stall_share=0.3):
        self.joint_names = list(joint_names)
        self.margin = margin
        # Never shorter than the hold: the reference has to come from before
        # the arm stopped, and noticing the stall costs `hold` on its own.
        # A window inside that would take its reading from during the push
        # and measure the climb against itself.
        self.window = max(window, hold + 0.1)
        self.hold = hold
        self.rewind = rewind
        self.lag_limit = lag_limit
        # [(seconds from the start of the move, joints in joint_names order)]
        self.plan = list(plan or [])
        self.stall_floor = stall_floor
        self.stall_share = stall_share
        self.started = None
        self.samples = []          # (t, positions, {joint: effort})
        self.since = {}            # joint -> when its excess began
        # joint -> what it was pulling just before the arm stopped
        # following its plan. Latched for as long as it is stopped, and
        # thrown away the moment it is flying again.
        #
        # Latched rather than rolling because a rolling baseline can only
        # see a push that arrives faster than margin/window -- 10 Nm/s on
        # the defaults. A servo settling into a table takes a second or two
        # to build up, and every sample of it looks like the last. Against a
        # fixed pre-stall reading the whole climb counts, and gravity cannot
        # forge it: an arm that is not moving is not changing shape, so its
        # gravity torque is not changing either.
        self.reference = {}

    def add(self, now, positions, efforts):
        """One reading. Returns the verdict when contact is decided, else None."""
        if self.started is None:
            self.started = now
        usable = (positions is not None
                  and all(v is not None for v in positions))
        self.samples.append((now, list(positions) if usable else None,
                             dict(efforts or {})))
        horizon = max(self.rewind, self.window, self.hold) * 2.0 + 1.0
        while len(self.samples) > 2 and now - self.samples[0][0] > horizon:
            self.samples.pop(0)

        why, lag = self.not_keeping_up(now, positions if usable else None)
        if not why:
            # The arm is flying its plan, so whatever the motors are pulling
            # is what it takes to fly it -- however much that is, and it can
            # be a lot. Nothing accumulates while this is true.
            self.since.clear()
            self.reference.clear()
            return None
        for joint in self.joint_names:
            value = (efforts or {}).get(joint)
            if joint not in self.reference:
                # From `window` back, which is before the arm stopped: the
                # stall itself takes `hold` to notice, and hold is shorter
                # than window, so this reading is from before the contact.
                # None while the move is younger than one window, which is
                # what keeps the guard quiet at the start of every move.
                was = self.baseline(joint, now - self.window)
                if was is None:
                    continue
                self.reference[joint] = was
            was = self.reference[joint]
            if value is None or abs(value) - abs(was) < self.margin:
                # Not pulling harder than it was before it stopped. Note the
                # |effort|: this is not the signed change, because the
                # shoulder crosses zero and keeps going negative on an
                # ordinary swing, and the signed difference read that as a
                # 2 Nm collision.
                self.since.pop(joint, None)
                continue
            first = self.since.setdefault(joint, now)
            if now - first < self.hold:
                continue
            return self.verdict(joint, value, was, lag, now, why)
        return None

    def planned_at(self, elapsed):
        """Where the trajectory says the arm should be, `elapsed` seconds in."""
        if not self.plan:
            return None
        for seconds, positions in self.plan:
            if seconds >= elapsed:
                return positions
        return self.plan[-1][1]

    def sample_at(self, when):
        """The last reading taken at or before `when`."""
        found = None
        for sample in self.samples:
            if sample[0] > when:
                break
            if sample[1] is not None:
                found = sample
        return found

    def baseline(self, joint, when):
        """That joint's effort as of `when`.

        None when nothing goes back that far, which is what keeps the guard
        quiet for the first `window` of a move: there is nothing to compare
        against yet, and comparing against the start of the move is the bug
        this replaced.
        """
        found = None
        for stamp, _positions, efforts in self.samples:
            if stamp > when:
                break
            value = efforts.get(joint)
            if value is not None:
                found = value
        return found

    def not_keeping_up(self, now, positions):
        """(why the arm is not flying its trajectory, how far behind it is).

        The reason is 'lag' or 'stall', or None when the arm is keeping up
        -- and None with no plan to compare against, which disarms the guard
        entirely. That last is deliberate: without knowing where the arm was
        supposed to be there is no way to tell an obstacle from the arm's
        own weight, and judging on torque alone is precisely what stopped a
        good transit and cost two minutes of recovery.
        """
        if positions is None or not self.plan:
            return None, None
        elapsed = now - self.started
        wanted = self.planned_at(elapsed)
        lag = max(abs(a - b) for a, b in zip(positions, wanted))
        if self.lag_limit > 0.0 and lag >= self.lag_limit:
            return 'lag', lag
        earlier = self.sample_at(now - self.hold)
        before = self.planned_at(elapsed - self.hold)
        if earlier is None or before is None:
            return None, lag
        asked = max(abs(a - b) for a, b in zip(wanted, before))
        went = max(abs(a - b) for a, b in zip(positions, earlier[1]))
        if asked >= self.stall_floor and went < asked * self.stall_share:
            return 'stall', lag
        return None, lag

    def verdict(self, joint, effort, baseline, lag, now, why='stall'):
        """What tripped, and the way back out.

        `retrace` is every posture the arm actually held over the last
        `rewind` seconds, oldest first. The caller drives it in reverse
        rather than planning a way out: these are postures the arm measured
        itself in seconds ago, so the path is known good without asking a
        planner about a scene the arm is now pressed into.
        """
        target = now - self.rewind
        retrace = [list(s[1]) for s in self.samples
                   if s[1] is not None and s[0] >= target]
        older = [s for s in self.samples if s[1] is not None and s[0] <= target]
        back_to = (list(older[-1][1]) if older
                   else (list(retrace[0]) if retrace else None))
        if back_to is not None and (not retrace or retrace[0] != back_to):
            retrace.insert(0, list(back_to))
        return {'joint': joint, 'effort': effort, 'baseline': baseline,
                'lag': lag, 'why': why, 'at': now, 'back_to': back_to,
                'retrace': retrace}


class PickPlaceOrchestrator(Node):

    def __init__(self):
        super().__init__('pick_place_orchestrator')
        self.cb = ReentrantCallbackGroup()

        self.declare_parameter('arm', 'right')
        # by_side by default: with "fixed" the camera-half rule never runs and
        # the launch arm moves whatever half the object is in, which is not
        # what anyone means by a two-armed robot. "fixed" is still there for
        # single-arm work.
        # 'by_side' takes the half of the camera frame the object is in and
        # gets on with it. 'by_reach' asks the solvers which arm can reach
        # before choosing; 'fixed' always uses arm.
        #
        # by_side is the default because the answer is nearly always the same
        # and the asking is what costs: measured, 87 seconds between the
        # first look and the second, on a decision the camera half made
        # correctly for free. The pre-flight then settles whether the pick is
        # possible for the arm chosen -- properly, and for the arm that will
        # actually do it -- so a wrong guess costs a pre-flight rather than a
        # motion.
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
        # How much faster than /compute_cartesian_path timed it a straight
        # line may be flown. 1.0 is the old behaviour: fly it exactly as
        # timed.
        #
        # This exists because the two halves of a cycle are timed by
        # different things. cuMotion times the joint goals from its own
        # limits; MoveIt times the Cartesian paths from joint_limits.yaml,
        # which on this robot sets `has_acceleration_limits: false` for
        # every joint, so there is no acceleration limit for it to work
        # with. Measured in run 1789640514: TRANSIT took 11.2-11.6 s at a
        # peak of 0.8 rad/s^2 while the run's joint goals peaked at 3.4-5.4
        # rad/s^2 and finished inside 3.5 s.
        #
        # The ceilings below are what actually bounds it; this is only a
        # cap on how much correction is allowed at once.
        self.declare_parameter('cartesian_boost_max', 3.0)
        # The ceilings, and what actually bounds the speed-up. Set against
        # what the joint goals on this arm already reach in an ordinary
        # cycle -- measured peaks of 1.73 rad/s and 5.4 rad/s^2 -- so a
        # Cartesian leg is allowed to work about as hard as a free-space
        # one already does, and no harder. Lower them if the contact guard
        # starts tripping on the robot's own moves: that guard watches
        # torque, and torque follows acceleration.
        self.declare_parameter('cartesian_vel_max', 2.5)
        self.declare_parameter('cartesian_accel_max', 4.0)
        self.declare_parameter('motion_timeout', 60.0)
        # How long a goal move_group has accepted may sit without the arm
        # moving before it is given up on. 0 waits the full motion_timeout,
        # which is what it used to do.
        #
        # motion_timeout is the budget for a *flight*, and spending it on a
        # goal that never started is a minute of a cycle spent on nothing.
        # Measured, run 1789640514: PRE_PICK_STATE sat for 59.9 s with
        # every joint inside 0.01 rad of where it began, then reported
        # FAILURE -- which is not retryable, so the cycle gave up on a goal
        # that had never been tried once.
        self.declare_parameter('motion_start_timeout', 12.0)
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
        # -0.005 -> +0.015: stop 15 mm *above* the detected top of the
        # object rather than 5 mm below it.
        #
        # Measured, run 1788862221, the one cycle that gripped and placed:
        # the detector put the object top at z=0.354, the descent was
        # commanded to 0.349, and the tool was at z=0.36869 when the fingers
        # reached the torque cap -- point + 14.7 mm. So the height that
        # actually grips is a good 20 mm above the height being asked for,
        # and the only reason the old offset ever worked is that the arm
        # stopped 36 mm short of it (error_mm 40.6, plan_error_mm 0.0 -- the
        # plan was right, the arm did not follow it there). Track the
        # trajectory any better and the same command drives into the table,
        # which is what happened.
        #
        # Two more millimetres of the gap are the close itself: the tool sank
        # 0.38452 -> 0.36869, 15.8 mm, while the fingers were closing.
        # 0.015 -> 0.010, asked for: five millimetres lower. Still above the
        # detected top, and still floored by grasp_max_depth, so the descent
        # cannot aim into the surface. For reference the two measured grips
        # came in at point + 14.7 mm and point + 25 mm, both with the arm
        # stopping short of its command -- so the jaws have real tolerance
        # here. What they do not tolerate is being sent below the object.
        # Negative now, and measured from the object's *top*: the jaws
        # close 30 mm down the side of it, or as far down as the surface it
        # is standing on allows -- whichever comes first.
        #
        # It was +10 mm, which put the fingertips above the top face. That
        # is a grip on the corner of a thing, and it is why a screwdriver
        # came back "closed on nothing" twice with the orientation right.
        # It only made sense while the floor was the object's own top and
        # the jaws could not be sent below it; the floor is the measured
        # surface now (grasp_table_clearance), so this can reach down and
        # cannot reach through.
        self.declare_parameter('grasp_z_offset', -0.030)
        # How far *below* the detected top of the object the tool may be
        # commanded, metres. The detected point is the top face of something
        # resting on the work surface, so it is the one surface reference
        # available without being told where the table is -- and 0.0 means
        # the descent never aims below it.
        #
        # This is the floor min_grasp_z is not. min_grasp_z is measured from
        # the base and set at 0.01 to catch a depth reading that comes back
        # metres away and below the floor; this table stands at z=0.34, so it
        # would never have stopped a grasp 20 mm too deep. It also bounds the
        # ladder: the lower-8mm and yaw+90-lower rungs subtract height on a
        # retry, which after a missed grip is precisely how the tool gets
        # pressed into the surface.
        self.declare_parameter('grasp_max_depth', 0.0)
        # Fly the remainder when the descent stops short of the grasp.
        #
        # The descent's plan ends on the point and the arm does not: measured
        # 15.5, 29 and 36 mm above the commanded grasp across three runs,
        # with plan_error_mm 0.0 every time -- the trajectory was right and
        # the controller stopped following it, which nothing catches because
        # the joint trajectory controller has no `constraints` block and so
        # reports SUCCESS the instant a trajectory ends.
        #
        # That cannot be dialled out with grasp_z_offset, because it moves
        # 20 mm between runs: the offset that grips one run closes on air or
        # presses the table the next. Both have now happened. Measuring the
        # gap and flying it is the only thing that makes the grasp height
        # repeatable.
        #
        # It is a continuation of the one descent, not a second one: the same
        # vertical line, straight down from where the tool actually is, same
        # collision settings and same joint-travel budget. x and y are left
        # alone deliberately -- the jaws span 40 mm and the successful grip
        # was 17 mm off laterally, so sideways error is tolerable where
        # 30 mm of height is not, and a lateral move at grasp height is the
        # one thing worth not doing down there.
        #
        # It was off for a long time, on the evidence that every close
        # which reached the object did so despite shortfalls of 15-36 mm --
        # so driving the tool the last centimetre down, toward the table,
        # looked like a fix for a problem that was not there. That comment
        # ended "turn it on if a descent ever does stop short enough to
        # miss", and run 1789014831 is that run.
        #
        # Four descents on the right arm, all from a transit that landed
        # 27-32 mm out. The two that gripped stopped 3.2 and 4.0 mm from the
        # commanded grasp; the two that missed stopped 18.1 and 18.9 mm out,
        # 15 mm of it height, and the jaws closed on air above the object.
        # It is all tracking error and none of it calibration: at the end of
        # those moves the joints were still 29 mrad from the last point of
        # their own trajectory, against 9-11 mrad on the two that worked.
        # The controller reports the move done and the arm is not there yet.
        #
        # The risk it was off for -- pressing into the table -- is now
        # watched for: the contact guard stops a move where a joint loads up
        # while the arm stops following its plan, and backs it out.
        self.declare_parameter('descend_close_gap', True)
        # Re-probe the descent from the arm's measured posture once it has
        # arrived, instead of trusting the pre-flight's model of where the
        # approach would end.
        #
        # Measured, run 1788948709: the pre-flight passed the column, TRANSIT
        # landed 32.6 mm off, and the descent then cost 6.42 rad against a
        # 1.5 rad budget and was refused -- a whole approach flown for a
        # descent that was never the one checked. One service call per
        # orientation, no motion.
        self.declare_parameter('descend_recheck', True)
        # Include the lift in what the pre-flight proves, so the way out of
        # the column is checked before the arm goes into it.
        #
        # Without it the pre-flight blesses a descent whose lift then cannot
        # be flown, and that is found out with the object gripped at the
        # bottom of the column -- measured, twice, at 2.97 and 3.01 rad
        # against a 1.5 rad budget.
        self.declare_parameter('preflight_lift', True)
        # Stop a move when a joint pulls this many Nm harder than it was
        # pulling contact_torque_window ago, and back the arm off to where it
        # was contact_rewind_seconds before.
        #
        # A rolling baseline, not the torque at the start of the move, and a
        # relative one rather than an absolute: the shoulder carries the whole
        # arm and the wrist carries a gripper, so one absolute number would
        # either miss a wrist collision or fire constantly on joint1.
        #
        # Measured, run 1789012345: joint1 read +3.42 Nm at the staging pose
        # and -10.48 Nm seven seconds later at full stretch -- 14 Nm of pure
        # gravity across one ordinary transit, at about 2.5 Nm/s. Against the
        # start of the move a 2 Nm margin tripped 1.7 s in, on the shoulder
        # *unloading*. Over a 0.4 s window that same gravity moves the
        # baseline by about 1 Nm, so 4 leaves a factor of four in hand.
        #
        # 0 disables it. There are no efforts on fake hardware, so it never
        # fires there.
        self.declare_parameter('contact_torque_margin', 4.0)
        # How far back the rolling baseline looks, seconds.
        self.declare_parameter('contact_torque_window', 0.4)
        # How long the excess has to last before it counts, seconds. An
        # acceleration transient passes; a thing the arm is leaning on does
        # not.
        self.declare_parameter('contact_torque_hold', 0.25)
        # How far behind its own trajectory the arm has to fall, rad, for the
        # torque to be read as contact rather than as its own weight.
        #
        # This is the half of the test gravity cannot fake: a servo being
        # held up falls behind its plan; one flying freely tracks it. Measured
        # over that same transit, the worst honest tracking error at full
        # speed was 0.126 rad, so this is roughly double it. 0 drops the lag
        # test -- the stall test still applies.
        self.declare_parameter('contact_lag_rad', 0.25)
        self.declare_parameter('contact_rewind_seconds', 2.0)
        # How fast to retrace those seconds on the way back out, rad/s. Slow:
        # this is the one move in the cycle that is driven straight at the
        # controller without a collision check, and it is only safe because
        # the postures it visits are ones the arm measured itself in moments
        # earlier.
        self.declare_parameter('contact_retreat_speed', 0.5)
        # How many times to go back down and try the same grasp when the
        # detector says the object never moved. The fingers cannot tell a
        # grip from a fingertip on an edge; the object still being there can.
        #
        # Bounded because the same grasp failing the same way three times
        # will not work on the fourth, and each go costs a descent.
        self.declare_parameter('regrasp_attempts', 3)
        # Whether the arms are mock_components rather than motors.
        #
        # Only used to decide whether a saved setting that switches a check
        # off may be honoured. The rehearsal needs those values -- fake
        # fingers never close on anything -- and the arms must never see
        # them, and a file cannot tell which run it is being read by.
        self.declare_parameter('fake_hardware', False)
        # How many times to ask the detector before believing it when it
        # says there is nothing there.
        #
        # One empty answer is not an answer: the arm is hovering directly
        # over the object at that moment, which is the one place guaranteed
        # to hide it from the camera.
        self.declare_parameter('verify_detect_tries', 2)
        # Below this the gap is not worth a move; above it something other
        # than tracking error is wrong and flying blind into it is not the
        # answer.
        self.declare_parameter('descend_gap_tolerance', 0.005)
        self.declare_parameter('descend_gap_max', 0.06)
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
        self.declare_parameter('place_mode', 'detected')
        self.declare_parameter('place_position', [0.35, 0.30, 0.25])
        self.declare_parameter('place_yaw', 0.0)
        self.declare_parameter('place_approach_height', 0.15)
        # How far off the point above the sheet the tool may land and still
        # count as having got there.
        #
        # Deliberately coarse, and coarse for a reason. The leg that runs
        # straight after this one measures where the tool actually is and
        # closes whatever gap is left horizontally, at height, before
        # anything descends -- so a few centimetres of tracking error here
        # costs nothing. What this check is really for is the case the
        # error was written for: the posture succeeds and the tool has not
        # moved at all, still standing in the staging pose 215 mm away,
        # after which every height below is computed from a place the arm
        # is not. Measured, run at 14:24: the joint goal settled 0.115 rad
        # off in joint1 -- about 60 mm at the tool -- and a 50 mm check
        # threw away three otherwise sound places.
        self.declare_parameter('place_above_tolerance', 0.12)
        # ... and, past that, how much of the distance the move had to
        # cover it has to have covered. This is the real test, and the
        # reason there are two numbers: the failure worth stopping for is
        # an arm that has not gone anywhere, not an arm that has gone
        # nearly all the way. Measured, the miss this guard exists for
        # closed 0 mm of 300; the three places it wrongly threw away
        # closed 198 mm of 300.
        self.declare_parameter('arrival_progress', 0.05)
        self.declare_parameter('place_radius', 0.15)
        # place_mode 'detected': what the drop surface is called, asked of
        # the same detector the pick uses. A sheet of paper is what a person
        # can put down to say "here", and the model answers 'piece of paper'
        # where it answers neither 'paper' nor 'white paper' -- measured on
        # this camera, seven phrasings against one frame.
        self.declare_parameter('place_prompt', 'piece of paper')
        # The gap left under the object when it is let go.
        #
        # Under the *object*, not under the tool. How far the object hangs
        # below the tool is not a constant and not a guess: the pick
        # measured it. The tool gripped at a known height above the surface
        # the object was standing on, and that difference is how high the
        # tool has to be above the paper for the object to be resting on
        # it. This is the centimetre on top of that.
        #
        # Without that measurement -- an older detector, or a pick that
        # never saw a surface -- it falls back to being clearance for the
        # tool, which is what it used to mean.
        self.declare_parameter('place_clearance', 0.005)
        # How far inside the sheet's own edge a fallback drop point sits.
        # Only used when the middle of the sheet cannot be reached: the
        # object has to land *on* the paper, and its own width is not
        # known, so the point is pulled well in.
        self.declare_parameter('place_edge_margin', 0.04)
        # The most the object's own base may be above the sheet when the
        # jaws open. Not a tolerance on the descent -- the height of the
        # thing being put down, which is what "do not drop it" means.
        # place_clearance is the part of this that is deliberate; whatever
        # the descent falls short by comes out of the same budget, and the
        # place refuses rather than dropping from higher.
        self.declare_parameter('place_max_drop', 0.010)
        # How finely the sheet is searched for somewhere to put the object:
        # a place_grid x place_grid grid across its footprint, tried from
        # the middle outwards. 4 gives 16 spots across a sheet of paper,
        # which at A4 is one every 6 cm or so -- enough to find a corner
        # the arm can get down to when the middle is at the edge of its
        # envelope, without turning the place into a minute of probes.
        self.declare_parameter('place_grid', 4)
        # A cap on joint speed for the descent onto the drop surface and the
        # release. "Gently" is the whole requirement of a place: the object
        # is already held and there is nothing to gain from arriving fast.
        self.declare_parameter('place_speed', 0.2)
        # Coming back *up* off the drop surface, once the object is down
        # and the jaws are open around it.
        #
        # It was flown at place_speed, which is the speed for lowering
        # something onto a surface and is the wrong answer here: nothing is
        # being set down any more. What the retreat has to be is
        # *vertical*, so the open jaws rise off the object rather than
        # sweeping it along, and that is the column it is flown on, not its
        # speed. Measured, run 1789640514: RETREAT took 7.2 s at a peak of
        # 0.14 rad/s, on a leg that carries nothing.
        self.declare_parameter('place_retreat_speed', 0.5)
        # The drop surface is in the octomap, exactly as the object being
        # picked up is, so a collision-checked line onto it stalls a
        # centimetre short for the same reason. See _descend_column.
        self.declare_parameter('place_ignores_octomap', True)
        # If the drop target cannot be seen from the staging pose, go to
        # HOME and look once more. The staging pose is above the work
        # surface, which is a good place to stand and a plausible place to
        # be standing between the camera and a sheet of paper; HOME is out
        # of the camera's frame by definition -- it is where the object was
        # located from in the first place. Costs one move and one inference,
        # and only when the first look found nothing.
        self.declare_parameter('place_look_from_home', True)
        # Whether one press of Pick also places. Off: pick and place are two
        # operator actions, and between them the arm stands at the staging
        # pose holding the object. On restores the single cycle -- pick,
        # carry, drop, home -- which is what the sequence editor's place half
        # used to run automatically.
        self.declare_parameter('place_after_pick', False)

        # At bringup, walk both arms to their recorded staging pose and stop
        # there. That is where every cycle starts and ends, so it is where an
        # arm should be waiting; the alternative is the posture the last run
        # happened to leave it in, which after a failure is somewhere over
        # the table.
        #
        # This moves the robot without anyone pressing anything, so: slowly,
        # to a *recorded* posture, as a joint goal (the one kind that is
        # repeatable), skipped for an arm with no recording, and refused if
        # anything is already running.
        self.declare_parameter('boot_walk', True)
        # Which pose they wait in. The staging pose, because that is where
        # the approach starts: a Pick pressed from there begins with the
        # transit rather than with a fold down to HOME and back.
        #
        # It is not free. HOME is out of the camera's frame and the staging
        # pose is not, so the map cannot be captured from here and the look
        # has the arm in the picture. Both are handled rather than ignored:
        # the boot walk goes *via* HOME and maps the work area on the way
        # through, and a detection that lands on the robot's own gripper is
        # refused (see implausible_detection) and re-taken from HOME.
        self.declare_parameter('boot_pose', 'pre_pick')
        # Where the arm that is *not* picking waits. Both at the staging
        # pose: that is what they are for, and it is what was asked for.
        #
        # It is not free, and the cost is worth knowing. The two staging
        # poses are mirror images about 35 cm apart, and an arm carrying
        # something to its own staging pose passes the other one -- run at
        # 15:10 came back INVALID_MOTION_PLAN three times at the staging
        # goal with the tape in the jaws. That is the planner doing its
        # job: the other arm is a real robot link in the planning scene,
        # not a voxel, and no amount of self-filtering makes it disappear.
        # If it recurs, the answer is to re-record the two staging poses
        # further apart -- or 'home' here, which folds the idle one down at
        # its own side. 'leave' does not move it at all.
        self.declare_parameter('boot_other_arm', 'pre_pick')
        # Whether the boot walk folds down to HOME to capture the map
        # before going to where the arms wait.
        #
        # Off: straight to where they wait, and the map is taken from
        # there.
        #
        # The detour existed because a map captured at the staging pose
        # caught the grippers, and an arm inside its own map cannot plan at
        # all -- measured at 13:37, all four finger links in the octomap and
        # both arms invalid. That is now handled where it belongs rather
        # than by moving the robot out of the way: the gripper links are
        # exempt from the octomap for the whole session
        # (gripper_never_in_map), so a map that contains them is harmless.
        # The rest of each arm is still checked against it, and
        # capture_octomap still drops a map that caught anything else.
        #
        # On restores the fold down to HOME for a stack where that is not
        # wanted.
        self.declare_parameter('boot_via_home', False)
        # Whether a cycle that starts with the arm already at the staging
        # pose looks from there instead of going to HOME first. The trip is
        # about ten seconds each way and buys a clear view and a fresh map;
        # this trades it for starting where the arm already stands.
        self.declare_parameter('locate_from_staging', True)
        # Whether HOME may be commanded at all. Off: the staging pose is
        # the rest pose, the observation pose and the place the cycle
        # returns to, and HOME is never driven to by anything.
        #
        # HOME is a folded posture on the far side of the workspace. Every
        # trip to it is a long swing that buys nothing left to buy: the map
        # is captured from wherever the arms stand now
        # (gripper_never_in_map covers the gripper, robot_in_map throws
        # away a map that caught anything else), the detector looks from
        # staging, and a cycle already starts and ends at staging.
        #
        # It is also a posture this robot cannot actually reach. joint4's
        # HOME value sits against its lower limit and the elbow stops short
        # of it, which is why the arrival check needed its own slack to
        # pass at all -- and with the arm at staging the miss reads as
        # 1.979 rad, which is not a fault, it is "at pre_pick, not home".
        self.declare_parameter('use_home', False)
        # How close to the tool a detection has to be before it is the
        # robot rather than the object. The jaws span 44 mm and the hand is
        # about 150 mm long, so anything inside 150 mm of the tool centre
        # is the gripper looking at itself.
        self.declare_parameter('self_detect_radius', 0.15)
        # How close to the surface the object is standing on the jaws may
        # close. This is what stops a deeper grasp pressing into the table,
        # and the surface is measured per detection rather than configured:
        # the detector reports the depth of the ring around the object,
        # which is the table right there rather than the table on average.
        self.declare_parameter('grasp_table_clearance', 0.005)
        # When the jaws shut on nothing, step the tool down this far and
        # close again -- right there, without lifting away. The commonest
        # cause of an empty grip is the grasp being computed above a thin
        # object, and this is the cheapest thing that fixes it: one short
        # line and one grip, against a ladder rung's full approach. Bounded
        # below by the surface the object stands on, so it cannot press
        # into the table. 0 tries goes straight to the ladder.
        self.declare_parameter('regrasp_lower_tries', 2)
        self.declare_parameter('regrasp_lower_step', 0.010)
        # Above this, an object is too tall for a top-down grasp to close
        # around it -- the jaws meet its top face instead of its sides. 60
        # mm is the finger length. Reported, not refused: whether that
        # holds depends on the object, and the alternative (coming in
        # horizontally) is not something the vertical approach can fly. 0
        # turns the warning off.
        self.declare_parameter('tall_object_height', 0.06)
        # How far the tool may be off the column, laterally, before the
        # descent is worth correcting. Under this it is not worth a move;
        # over it, the "vertical" descent is a diagonal.
        self.declare_parameter('descend_align_tolerance', 0.005)
        # Carrying back to the staging pose, keep the tool on the half of
        # the workspace the arm itself is on.
        #
        # The two arms stand either side of the centre-line and the other
        # one is parked at its own staging pose, in the air, with nothing
        # in the octomap to say so -- the map is of the table, taken while
        # both arms were out of frame. A 7-DOF arm has more than one way to
        # reach a posture, and the free-space plan back from a pick is
        # under no obligation to pick the one that stays on its own side:
        # it can swing the tool across the middle and back out again, which
        # is a swing through the space the other arm is standing in while
        # something is in the jaws.
        #
        # The goal itself is on the arm's own side -- the staging pose is
        # what it is -- so this is a constraint on the *path*, which is why
        # it cannot be expressed as a goal. cuMotion's MoveIt plugin takes
        # no path constraints, so the plan is asked for, measured with the
        # arm's own kinematics, and flown only if it stays put. Its
        # optimiser is stochastic, so a plan that crosses is worth asking
        # again for rather than giving up on.
        self.declare_parameter('carry_guard', True)
        self.declare_parameter('carry_guard_tries', 3)
        # How far over the line counts as over it. Not a licence to cross:
        # slack for the tool's own width and for the difference between the
        # plan and what the arm actually tracks, which on this robot is
        # tens of millimetres.
        self.declare_parameter('carry_side_slack', 0.02)
        # Past this much of a crossing the guard stops being a preference
        # and refuses.
        #
        # It has to be a preference for anything less, because a guard that
        # vetoes leaves the arm stranded over the table holding something,
        # which is worse than the swing it was avoiding. Measured, run at
        # 15:52: every plan crossed by 85 mm, the goal was refused, the
        # lift that the retry depends on then solved 2.4% of its line, and
        # the cycle ended with the arm out at z=0.357 and the controller
        # reporting CONTROL_FAILED. Nothing about that is better than
        # having flown the 85 mm.
        #
        # So: ask for several plans, fly the one that stays furthest out of
        # the other arm's half, and say by how much it still crosses. Only
        # an excursion past this -- a swing right through where the other
        # arm is parked, rather than a clip of the middle -- is refused.
        self.declare_parameter('carry_side_refuse', 0.25)
        self.declare_parameter('boot_speed', 0.15)
        # Long enough for move_group, the controllers and the joint states to
        # be up. Not a correctness requirement -- the walk waits for the
        # planner to prove it can plan before it sends anything -- it just
        # keeps the log quiet for the first few seconds.
        self.declare_parameter('boot_delay', 8.0)
        # How long the boot walk will wait for the planner to become usable
        # before giving up and leaving the arms where they are.
        #
        # Generous on purpose. move_group being in the graph is not the same
        # as move_group being able to take a goal: measured on a cold
        # bringup, the first goal it accepted came 77 s after the node
        # started, and the three the walk had already sent were refused at
        # 15 s each -- and then executed anyway, a minute later, which is an
        # arm moving while the panel says IDLE. Nothing is sent now until a
        # plan-only probe comes back.
        self.declare_parameter('boot_planner_wait', 180.0)

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
        # What the jaws must be pressing with to count as gripping, Nm at the
        # gripper motor. The grip force is position error times a fixed Kp,
        # so an object that leaves lets the fingers travel in and the torque
        # goes with it. Measured across every close on this robot: shut on
        # nothing reads 0.5-1.2 Nm, holding reads 1.8-2.5 Nm. 1.0 sits below
        # the empty band's top rather than in the middle of the gap on
        # purpose -- this only has to catch a jaw that is clearly slack,
        # because the width catches the rest, and throwing away a real grasp
        # is the more expensive mistake. 0 turns it off.
        self.declare_parameter('grasp_hold_torque_min', 1.0)
        # How much narrower than the close the jaws may sit and still be
        # holding the same thing, metres. This is what catches a drop the
        # width band cannot: 6 mm of finger is a grip if the close ended at
        # 6 mm and an empty jaw if it ended at 14. The close steps 2 mm
        # coarse and 0.5 mm fine, so anything under 2 mm is the step and not
        # a drop. 0 turns it off.
        self.declare_parameter('grasp_width_drop', 0.002)
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
        # Rebuild the map at the top of every pick, not just on the
        # trips that pass through HOME. Measured, run 1790054584: the
        # boot walk mapped the area at 10:53 and the cycle at 10:44 --
        # eleven minutes later -- planned against that same map, because
        # locate_from_staging skips the refresh. Anything moved on the
        # table in between was invisible to the planner.
        self.declare_parameter('map_before_pick', True)
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
        # Whether HOME may be commanded when pre_pick could not be reached on
        # the way back. HOME is a joint goal to a folded posture; from a low,
        # extended one the short path in joint space goes through the work
        # surface. Off means the arm stops and reports instead -- leaving it
        # parked over the table is bad, dragging it across the table is
        # worse.
        self.declare_parameter('home_requires_pre_pick', True)
        # Take the motors off once a failed cycle has parked the arm
        # somewhere the collision world says is free.
        #
        # Off. An ordinary failure -- nothing detected, a plan that would
        # not go, a grip that closed on air -- leaves a perfectly healthy
        # arm, and taking the motors off it means the next run starts with
        # a restart of the whole stack. The two cases where letting go is
        # the right answer have their own switches: disengage_on_contact,
        # for an arm that has run into something, and the Stop button,
        # which parks it deliberately.
        #
        # These are direct-drive motors with no brakes, so wherever this
        # does fire it is only *after* a refuge is reached -- never with
        # the arm stranded, where letting go would drop it onto whatever it
        # is over.
        self.declare_parameter('disengage_on_failure', False)
        # The exception. The contact guard fires when a joint loads up
        # *and* the arm stops following its trajectory -- it has run into
        # something and is leaning on it. Holding position against an
        # obstruction is how a motor cooks itself, and the arm is by then
        # resting on the thing it hit, so letting go is the safe answer
        # rather than the drastic one.
        self.declare_parameter('disengage_on_contact', True)
        # After a torque trip, put the motors back on and glide to the
        # staging pose so the next cycle can just be started. Off leaves
        # the arm limp where it settled, which needs a bringup restart.
        self.declare_parameter('recover_after_contact', True)
        # How long the motors stay off before they are taken up again. The
        # point of letting go is to come off whatever the arm is leaning
        # on, and that takes a moment of gravity.
        self.declare_parameter('contact_relax_seconds', 3.0)
        # Where the editable sequence and the UI-settable parameters live.
        # Empty disables both and the built-in order is used.
        #
        # Not 'config_file'. realsense2_camera's rs_launch.py declares a
        # launch argument by that name and opens it as YAML, and a launch
        # configuration set at the top reaches every included launch file --
        # so naming it that took the whole bringup down with
        # FileNotFoundError on our own JSON path.
        self.declare_parameter('pick_place_config',
                               'pick_place_config.json')
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
        # How many alternative grasp yaws to try when nothing that respects
        # the detector's axis estimate flies. Spread over 180 degrees, since
        # half a turn is the jaw flip and already covered. 3 gives 45, 90 and
        # 135 degrees.
        #
        # Measured: a roll of tape the pre-flight refused outright flew its
        # whole column at 90 degrees, and at no other yaw. The axis estimate
        # is made from a 2D box and on a round object it is arbitrary.
        # 0 restores the old behaviour of treating it as fixed.
        self.declare_parameter('grasp_yaw_options', 3)
        # Ask the grasp server for candidates on reaching the staging pose.
        # Only a topic publish -- nothing acts on the answer yet, so this is
        # free to leave on and is how the candidates get looked at on real
        # data before anything is wired to them.
        self.declare_parameter('request_grasps', True)
        # Take the grasp from the model when it has one, instead of
        # synthesising a top-down pose from the detector's box.
        #
        # It answers a question the synthesised pose only guesses at -- where
        # on the object, at what height, in which direction, and how far the
        # jaws have to open -- all read off the point cloud rather than
        # assumed. What it does *not* answer is whether the arm can get
        # there, which is still the pre-flight's job: the model proposes,
        # ranked, and the pre-flight disposes.
        #
        # Falls back to the synthesised grasp whenever the model has nothing
        # usable, which is not a corner case: GraspNet was trained for a
        # 100 mm gripper and this one spans 44 mm, so on a wide object every
        # proposal can be one this robot cannot make.
        self.declare_parameter('use_grasp_model', True)
        # How old a candidate list may be, and how near its object must be
        # to the one being picked, before it is used for this pick.
        #
        # 0 means no age limit, which is the default, and it is safe for
        # one reason: the list is thrown away at the start of every cycle,
        # so "no limit" means "no limit within this cycle" rather than
        # "whatever the server last said, whenever that was". The clock was
        # the wrong bound anyway -- the grasps are requested at the staging
        # pose and the pre-flight that uses them runs on the *next*
        # attempt, so 30 s meant the model's answer usually expired before
        # anything could act on it, and every pick fell back to the
        # synthesised top-down pose.
        #
        # What still rejects them is the distance from the object actually
        # being picked (grasp_model_max_offset), which is the bound that
        # matters: candidates about something 20 cm away are for something
        # else whatever their age.
        self.declare_parameter('grasp_model_max_age', 0.0)
        self.declare_parameter('grasp_model_max_offset', 0.08)
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
        # The same limit, applied to the model's own grasps, rad.
        #
        # GraspNet proposes full 6-DoF grasps and some of them come in from
        # the side, which is the whole point of asking it. The cycle cannot
        # fly those yet: the approach and the retreat are a *vertical*
        # column -- descend_column and lift_column interpolate in z and hold
        # x and y -- so a tilted tool is fine and a tilted *path* is not.
        # Past a modest tilt the jaws would be driven down across the object
        # rather than onto it.
        #
        # So a steeper proposal is skipped rather than mis-flown, and said
        # out loud, because "the model suggested a side grasp and this cycle
        # cannot make one" is a real answer and worth reading. Flying the
        # column along the grasp's own approach axis is the change that
        # would lift this; it is not a parameter.
        self.declare_parameter('grasp_model_max_tilt', 0.6)
        # Ask the detector for the part of the object worth gripping,
        # rather than for the object.
        #
        # The detector answers referring expressions -- "detect handle of
        # the screwdriver" localises the handle -- so this is the prompt and
        # nothing else. It matters because the reported point is the centre
        # of whatever was found, and the centre of a roll of tape is the
        # hole. See grasp_recipes.py for the table and for what each entry
        # is for.
        #
        # Always a first try: a part prompt that finds nothing falls
        # straight back to the plain object, because a recipe that makes an
        # object unpickable would be worse than no recipe.
        self.declare_parameter('use_grasp_recipes', True)
        self.declare_parameter('grasp_recipes',
                               grasp_recipes.DEFAULT_RECIPE_FILE)
        # Tilt magnitudes to try between 0 and grasp_tilt_max, and how many
        # directions to try each in (4 = away, left, toward, right).
        self.declare_parameter('grasp_tilt_steps', 2)
        self.declare_parameter('grasp_tilt_azimuths', 4)
        # Orientations the reach probe may try per point. It sends a planning
        # request for each, so the full tilt set would make an unreachable
        # object cost a minute of probing; the pre-flight explores the rest.
        # How many grasp orientations the reach check tries before calling a
        # point unreachable.
        #
        # It was 3, and the pre-flight tries the lot -- so a point reachable
        # only at an orientation outside those three was refused before the
        # thorough check ever ran. Measured on the live robot, a roll of tape
        # at [0.324, 0.073, 0.353]: the left arm picked it at 11:24 and was
        # told "out of reach" at 11:35, and a sweep of 36 orientations at the
        # same point found 8 that solve the approach line completely --
        # including one at zero tilt. Three samples of an eight-in-thirty-six
        # target is a coin toss, and it was being reported as geometry.
        #
        # grasp_quat_options caps out at 2 + tilt_steps * azimuths, which is
        # 10 by default, so this is "all of them" rather than a new number.
        self.declare_parameter('reach_orientations', 10)
        # How many orientations the *arm choice* tries. Fewer than the
        # pre-flight's set on purpose: it is deciding which arm, not whether
        # the object can be picked, and the pre-flight settles the second
        # question afterwards for whichever arm it picks. At 10 orientations
        # times three heights times two arms it was 60 planning requests
        # before the robot had moved at all -- measured at 70 seconds.
        self.declare_parameter('arm_choice_orientations', 3)
        # Choose the arm on KDL alone, and take "cannot tell" as "use the
        # half of the frame the object is in".
        #
        # The solvers behind KDL exist to second-guess its "no", and they are
        # where the time goes: 2.9 seconds per orientation per height per
        # arm, measured, whether they find anything or not -- 52 seconds
        # before the robot moved. The pre-flight settles whether the pick is
        # actually possible, so buying a worse version of that answer first
        # is the most expensive thing this cycle did.
        self.declare_parameter('arm_choice_cheap', True)
        # Ask, before accepting a pre-flight candidate, whether the arm can
        # actually be planned into the posture the candidate proves the
        # descent from. One plan-only goal per accepted candidate.
        #
        # Without it the pre-flight checks the wrong half: run 1788945589
        # passed a candidate and TRANSIT -- the joint goal that puts the arm
        # in that posture -- then returned -2, after the arm had already
        # driven to the staging pose for nothing.
        self.declare_parameter('preflight_posture_reachable', True)
        # Prove the planner can plan before the cycle starts, with a plan-only
        # goal to the arm's own current posture. Two seconds, no motion, and it
        # is the difference between "the planner is not planning" and a report
        # blaming the recorded pre-pick pose.
        self.declare_parameter('check_planner_ready', True)
        # How long the pre-flight liveness probe waits for move_group to
        # answer a plan_only goal. Its own budget, not motion_timeout:
        # nothing is moving, so 60 s of silence is never the right
        # answer. Measured, run 1790054584 cycle 2 -- cuMotion planned
        # the probe in 116 ms and logged success, move_group never
        # returned the result, and the cycle sat in STARTING for 46 s
        # saying nothing until it was killed by hand.
        self.declare_parameter('planner_probe_timeout', 10.0)
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
        # Keep that exemption for the whole session rather than just for
        # the descent.
        #
        # The gripper lives in the camera's view: at the staging pose -- the
        # pose both arms wait in and the pose a map is now captured from --
        # it is a foot in front of the lens. Measured at 13:37: the map
        # caught all four finger links, /check_state_validity called both
        # arms invalid, and every goal after that failed with "the planner
        # is not planning", including a goal to the arm's own posture.
        #
        # The gripper's own geometry is never an obstacle worth planning
        # around: it is the part doing the grasping, it is checked against
        # the rest of the robot by the SRDF, and what it runs into is what
        # the contact guard is for. The arm's other links stay checked
        # against the map, so nothing here lets the forearm through the
        # table.
        self.declare_parameter('gripper_never_in_map', True)
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
        self.traj_clients = {}
        self.configure_arm(self.launch_arm)

        self.arm_selection = p('arm_selection').value
        if self.arm_selection not in ('fixed', 'by_side', 'by_reach'):
            raise ValueError(
                "arm_selection must be 'fixed', 'by_side' or 'by_reach'")

        self.place_mode = p('place_mode').value
        if self.place_mode == 'ready':
            # Renamed with the pose it referred to. Accepted rather than
            # rejected so an old launch line still starts.
            self.get_logger().warn(
                "place_mode 'ready' is now 'home': READY and HOME are one pose")
            self.place_mode = 'home'
        if self.place_mode not in ('state', 'home', 'position', 'detected'):
            raise ValueError(
                "place_mode must be 'state', 'home', 'position' or 'detected'")

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
        # Which half of the workspace each arm works in, worked out once
        # from its own recorded staging pose. See own_side.
        self._own_side = {}
        # What the last Cartesian re-timing scaled the path by. Recorded
        # per leg in the motion log; see _retime.
        self._last_boost = 1.0
        # Depth of nested at_speed blocks. Non-zero means the operator has
        # asked for this leg to be gentle, and the re-timing leaves it
        # alone.
        self._gentle = 0
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
        # Whether the gripper is currently allowed to touch the octomap. Held
        # for as long as the arm is on the column; release_column ends it.
        self._octomap_exempt = False
        # Why the last column leg was refused, in words, or None. The DESCEND
        # failure message used to assume every refusal was a reach shortfall
        # and said so -- 'the arm runs out of travel along the line' -- when
        # the real reason was the joint-travel guard, with the line solving
        # 100%. Two different faults with opposite fixes, reported as one.
        self._column_refusal = None
        # The order the cycle runs in, and where it is in it. Loaded from the
        # config file so an edit in the UI survives a restart, which is the
        # whole point of it being a file rather than a topic.
        self._sequence = list(sequence_model.DEFAULT_SEQUENCE)
        self._config_problems = []
        self._ctx = None
        self._step = None
        # The last thing the grasp server proposed, if anything has.
        self._grasps = None
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
        # What each arm is holding, keyed by arm name:
        #
        #   {'left': {'object': 'tape', 'offset': 0.024}}
        #
        # Per arm and not a single flag, because this robot has two of them
        # and they are used one at a time. A pick while the left hand is
        # full is a pick for the *right* hand, not a refusal -- and with
        # both full there are two things to put down, which is two Place
        # buttons rather than one that cannot say which.
        #
        #   object  what is in the jaws, in the operator's own words, so the
        #           panel says "holding tape" rather than "holding"
        #   offset  how far the tool is above the bottom of it, measured at
        #           the grasp. What turns "put it down on the paper" into a
        #           tool height.
        self._held = {}
        # The surface the object was standing on when it was picked up,
        # world z. Pairs with _pick_hold_offset.
        self._pick_surface_z = None
        self._boot_timer = None
        # Whether the contact guard fired during this cycle. The one
        # failure that ends with the motors off, because the arm is leaning
        # on whatever it ran into.
        self._contact_fired = False
        # Where the jaws settled on the close that passed its own checks --
        # the one moment the grip is known good. "Is it still in there" is a
        # question about a change, and a change needs a before.
        self._grasp_close = None
        self._last_detection = None
        self._last_payload = None
        # Which question found the object -- the part, or the object
        # itself. The grasp check reuses it so it compares like with like.
        self._pick_prompt = None
        # Where to grab a named object. A file's entries come before the
        # built-in ones so an operator can override any of them without
        # touching code; a broken file is reported and ignored rather than
        # being allowed to stop the robot coming up.
        recipe_file = self.get_parameter('grasp_recipes').value
        if recipe_file and not os.path.isabs(recipe_file):
            recipe_file = os.path.join(WS, recipe_file)
        self._recipes, recipe_problem = grasp_recipes.load_recipes(recipe_file)
        if recipe_problem:
            self.get_logger().warn(recipe_problem)
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
        # Whether move_group has ever answered, and when it last failed to.
        # Not for deciding anything -- for saying the right thing when it
        # stops, and for not spending ten seconds per goal waiting for a
        # server that has gone away mid-cycle.
        self._planner_answered = False
        self._planner_stalled = None

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
        # Asks the collision world -- octomap included -- whether a posture is
        # legal, without planning to it. The failure path checks a refuge
        # before driving to it, rather than finding out by arriving.
        self.validity_client = self.create_client(
            GetStateValidity, '/check_state_validity',
            callback_group=self.cb)
        # Deactivating a hardware component calls its on_deactivate, and
        # openarm_hardware's calls disable_all() -- which is what taking the
        # motors off actually means here.
        self.hardware_client = self.create_client(
            SetHardwareComponentState,
            '/controller_manager/set_hardware_component_state',
            callback_group=self.cb)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.prompt_pub = self.create_publisher(String, '/vlm/prompt', latched)
        self.state_pub = self.create_publisher(String, '/pick_place/state', latched)
        # The same transitions again as JSON, for anything that wants to act
        # on them rather than print them: the flow chart highlights the live
        # step, and a state string with free text after a colon is no use for
        # that.
        self.status_pub = self.create_publisher(
            String, '/pick_place/status', latched)
        # Asks the grasp server for candidates about a point. Published from
        # the staging pose rather than from home: that is where the arm
        # actually is when the descent is planned, and the camera sees the
        # object from the same place the pre-flight measures from.
        self.grasp_request_pub = self.create_publisher(
            String, '/grasp/request', 10)

        self.create_subscription(String, '/pick_place/prompt', self._on_prompt, 10,
                                 callback_group=self.cb)
        self.create_subscription(String, '/vlm/detections', self._on_detections, 10,
                                 callback_group=self.cb)
        self.create_subscription(String, '/grasp/candidates',
                                 self._on_grasp_candidates, latched,
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
        # One per hand, plus the ambiguous one for the common case of a
        # single full gripper. Trigger carries no fields, so which arm is
        # in the service name rather than in the request.
        self.create_service(Trigger, '/pick_place/place', self._srv_place,
                            callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/place_left',
                            self._srv_place_left, callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/place_right',
                            self._srv_place_right, callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/stop_arm',
                            self._srv_stop_arm, callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/abort', self._srv_abort,
                            callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/capture_ready',
                            self._srv_capture_ready, callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/open_gripper',
                            self._srv_open_gripper, callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/grip',
                            self._srv_grip, callback_group=self.cb)
        # The web UI edits the config file and then asks for it to be read
        # back, rather than pushing a sequence over a custom message type.
        # One source of truth, and it is the file -- so what the UI shows,
        # what the robot runs and what survives a restart cannot drift apart.
        self.create_service(Trigger, '/pick_place/reload_config',
                            self._srv_reload_config, callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/save_config',
                            self._srv_save_config, callback_group=self.cb)

        # Before the first state is published, so the UI's very first frame
        # already carries the saved order rather than the built-in one.
        self.load_config()

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
        elif self.place_mode == 'detected':
            self.get_logger().info(
                f'place_mode is "detected": the drop point is wherever '
                f'"{p("place_prompt").value}" is seen, and the object is let '
                f'go {p("place_clearance").value * 1000:.0f} mm above it. Put '
                f'the sheet down where you want the object, and check the '
                f'camera can see it -- nothing is placed if it cannot.')
        else:
            place = p('place_position').value
            self.get_logger().warn(
                f'place_position is {list(place)} -- verify this is reachable for '
                f'the {self.arm} arm in RViz before running on hardware')

        if p('auto_start').value:
            self.create_timer(3.0, self._auto_start_once, callback_group=self.cb)
        elif p('boot_walk').value:
            # Its own timer, cancelled by its own handler. Not the
            # auto_start one, which cancels every timer the node has --
            # including the motion-log heartbeat.
            self._boot_timer = self.create_timer(
                p('boot_delay').value, self._boot_once, callback_group=self.cb)

    # -- speed ---------------------------------------------------------------

    @contextlib.contextmanager
    def at_speed(self, cap, why):
        """Everything planned inside this block moves no faster than `cap`.

        One lever reaches every kind of move because every kind reads the
        same two parameters at the moment it builds its request: the joint
        goals through _base_request, and the Cartesian legs through _retime,
        which exists precisely because compute_cartesian_path has no scaling
        field. Setting the parameter for the block is therefore the only
        thing that slows a descent as well as a swing.

        A cap and not a factor: `cap` is a share of the joint limits, and a
        cycle already configured slower than that keeps its own speed. Put
        back in a finally, so an exception mid-move cannot leave the robot
        permanently slow.

        It also turns off the Cartesian re-timing boost for the block. The
        boost corrects a base timing that is too slow; these caps were
        measured against that base, with an object in the jaws and a
        surface underneath, and quietly changing what `place_speed: 0.2`
        means is not something a speed-up should do behind the operator's
        back. A gentle leg stays exactly as gentle as it was.
        """
        was_v = self.get_parameter('velocity_scaling').value
        was_a = self.get_parameter('acceleration_scaling').value
        want_v = max(0.01, min(was_v, cap))
        want_a = max(0.01, min(was_a, cap))
        # Set before the early return below, not after it. The flag says
        # "this leg was asked to be gentle", and that is true whether or
        # not the parameters had to change to make it so -- a cycle already
        # configured at 0.2 would otherwise have its place descent sped up
        # by the very correction this is here to keep out of it.
        self._gentle += 1
        try:
            if want_v >= was_v and want_a >= was_a:
                yield                  # already at least this gentle
                return
            self.get_logger().info(
                f'{why}: {want_v:.2f} of joint speed instead of {was_v:.2f}')
            self.set_parameters([
                Parameter('velocity_scaling', value=float(want_v)),
                Parameter('acceleration_scaling', value=float(want_a))])
            try:
                yield
            finally:
                self.set_parameters([
                    Parameter('velocity_scaling', value=float(was_v)),
                    Parameter('acceleration_scaling', value=float(was_a))])
        finally:
            self._gentle -= 1

    # -- the pose the robot waits in -----------------------------------------

    def _boot_once(self):
        """Fired once, `boot_delay` after bringup."""
        timer, self._boot_timer = self._boot_timer, None
        if timer is not None:
            timer.cancel()
        threading.Thread(target=self._boot_walk, daemon=True).start()

    def _boot_walk(self):
        """Both arms to their recorded staging pose, and wait there.

        This is the one move nobody asked for, so every part of it is the
        conservative choice: slow, to a posture that was recorded rather
        than computed, as a joint goal -- the only kind that is repeatable
        on a 7-DOF arm -- skipped entirely for an arm with no recording, and
        refused outright if anything else is running.

        Both arms, because the other one is not parked by anything else. It
        is left wherever the last run put it, and on this robot "wherever"
        after a failure means somewhere over the table with no brakes
        holding it.
        """
        with self._lock:
            if self._busy:
                self.get_logger().info(
                    'something is already running, so the arms are not being '
                    'walked to the staging pose')
                return
            self._busy = True
        self._abort.clear()
        detail = 'the boot move failed -- see the log'
        try:
            detail = self._walk_to_boot_pose()
        except Exception as exc:                     # noqa: BLE001 - reported
            self.get_logger().error(f'the boot move failed: {exc}')
        finally:
            with self._lock:
                self._busy = False
        # After the flag is down, not before: the motion log's heartbeat
        # writes one record a second *while busy*, and a state published
        # inside that window says IDLE while a run is supposedly in
        # progress. The last word on a move belongs to the moment the move
        # is actually over.
        self._set_state('IDLE', detail)

    def boot_destination(self, arm):
        """Where this arm waits: the picking arm and the other one differ.

        The picking arm waits where the approach starts. The other one
        waits out of the way, because the two staging poses are mirror
        images a third of a metre apart and everything the picking arm does
        with something in its jaws passes between them.
        """
        if arm == self.launch_arm:
            return self.get_parameter('boot_pose').value
        return self.get_parameter('boot_other_arm').value

    def boot_joints(self, arm):
        """Where this arm waits, and why it might have nowhere to wait.

        Returns (joints, problem). HOME always has somewhere to go -- a
        recorded home_state, or the symmetric home_joint_positions, which
        is a valid folded posture for either arm. The staging pose has to
        have been recorded, and an arm with no recording is left where it
        is rather than sent to a computed guess.
        """
        where = self.boot_destination(arm)
        if where == 'leave':
            return None, f'the {arm} arm is left where it is (boot_other_arm)'
        if where == 'home':
            return self.home_positions(), None
        entry = (self.load_states() or {}).get(PRE_PICK_STATE) or {}
        if entry.get('joints'):
            return list(entry['joints']), None
        return None, (
            f'no {PRE_PICK_STATE} recorded for the {arm} arm '
            f'({self.states_file}), so it is left where it is. Record one '
            f'with "python3 record_states.py --arm {arm} {PRE_PICK_STATE}".')

    def wait_for_planner(self, limit):
        """Block until the planner can actually plan, or give up saying so.

        Every cycle begins with checks that give cuMotion time to warm up.
        The boot walk has none of them and fires `boot_delay` after the node
        starts, which on a cold bringup is a full minute before move_group
        will take anything -- so it has to wait for itself.

        The probe is plan_only against the arm's own current posture:
        nothing moves, and a failure cannot be about reach.
        """
        started = self.get_clock().now().nanoseconds * 1e-9
        said = False
        while not self._abort.is_set():
            waited = self.get_clock().now().nanoseconds * 1e-9 - started
            if self.move_client.wait_for_server(timeout_sec=2.0):
                if self.planner_is_planning() is True:
                    if said:
                        self.get_logger().info(
                            f'the planner is ready after {waited:.0f} s')
                    return True
            if waited >= limit:
                self.get_logger().error(
                    f'the planner still cannot plan after {waited:.0f} s, so '
                    f'the arms are being left where they are. Nothing has '
                    f'been sent -- a goal sent now would be accepted minutes '
                    f'later and move the arm with nobody expecting it.')
                return False
            if not said:
                self.get_logger().info(
                    'waiting for the planner before moving anything. '
                    'move_group being in the graph is not the same as '
                    'move_group being able to take a goal, and a goal it '
                    'accepts late is an arm that moves late.')
                said = True
            self._abort.wait(2.0)
        return False

    def _walk_to_boot_pose(self):
        launch = self.launch_arm
        # The launch arm last, so the node ends configured for it whatever
        # happened to the other one.
        order = [arm for arm in (other_arm(launch), launch) if arm]
        speed = self.get_parameter('boot_speed').value
        self._set_state('BOOT', 'waiting for the planner')
        if not self.wait_for_planner(
                self.get_parameter('boot_planner_wait').value):
            return 'the planner never became ready, so nothing was moved'
        if self.get_parameter('gripper_never_in_map').value:
            # Before the map is captured, so a capture that catches the
            # gripper is harmless rather than fatal.
            if self.allow_gripper_in_octomap(True):
                self._octomap_exempt = True
                self.get_logger().info(
                    'the grippers are exempt from the octomap for this '
                    'session: they stand in the camera view at the staging '
                    'pose, so they land in any map captured from there, and '
                    'an arm inside its own map cannot plan at all. The rest '
                    'of each arm is still checked against the map.')
        self._set_state('BOOT', 'walking both arms to where they wait')
        reached, skipped, failed = [], [], []
        try:
            with self.at_speed(speed, 'boot move'):
                # Straight to where they wait. Going via HOME to map costs
                # a fold down and back on every bringup, and the whole
                # reason for it -- that the arm would otherwise be carved
                # into the map -- is the self-filter's job, not a detour's.
                # boot_via_home puts the detour back for a stack whose
                # self-filter is not working.
                mapped = (self._map_on_the_way()
                          if self.get_parameter('boot_via_home').value
                          else False)
                for arm in order:
                    if self._abort.is_set():
                        break
                    self.configure_arm(arm)
                    where = self.boot_destination(arm)
                    joints, problem = self.boot_joints(arm)
                    if joints is None:
                        skipped.append(f'{arm} ({problem.split(",")[0]})'
                                       if problem else arm)
                        self.get_logger().warn(problem or f'{arm}: nowhere')
                        continue
                    self._set_state('BOOT', f'{arm} arm to {where}')
                    if self._move_to_joints(list(joints),
                                            f'BOOT_{arm.upper()}',
                                            skip_if_there=True):
                        reached.append(f'{arm} at {where}')
                    else:
                        failed.append(arm)
                        self.get_logger().error(
                            f'the {arm} arm could not be walked to {where}. '
                            f'It is where it is; check it before starting a '
                            f'cycle.')
                if not mapped and not failed:
                    # From where they now stand. The arm is in the camera's
                    # view at the staging pose and that is exactly what the
                    # updater's self-filter is for: it renders the robot
                    # from the URDF at the current TF and drops those
                    # pixels before they become voxels. If the map comes
                    # back empty, capture_octomap says why.
                    self._set_state('BOOT', 'mapping the work area')
                    mapped = bool(self.capture_octomap(anywhere=True))
        finally:
            self.configure_arm(launch)
        parts = []
        if reached:
            parts.append(', '.join(reached))
        if skipped:
            parts.append(f'{" and ".join(skipped)} not moved')
        if failed:
            parts.append(f'{" and ".join(failed)} could not get there')
        if mapped:
            parts.append('the work area mapped')
        return '; '.join(parts) or 'nothing to move'

    def _map_on_the_way(self):
        """Capture the octomap while the arms are still folded at HOME.

        The one moment a map can be taken without the robot in it, and the
        reason the boot walk goes via HOME when it is heading somewhere
        else: the staging pose is over the work surface, so from there
        neither the map nor the look can be taken cleanly. Doing it here
        means the first Pick of the session already has a map and does not
        have to fetch one.

        Best effort. A boot that cannot map still parks the arms, and the
        first cycle refreshes the map itself the moment it goes to HOME.
        """
        launch = self.launch_arm
        try:
            for arm in [a for a in (other_arm(launch), launch) if a]:
                if self._abort.is_set():
                    return False
                self.configure_arm(arm)
                joints = self.home_positions()
                self._set_state('BOOT', f'{arm} arm to home, to map from')
                if not self._move_to_joints(list(joints),
                                            f'BOOT_HOME_{arm.upper()}',
                                            skip_if_there=True):
                    self.get_logger().warn(
                        f'the {arm} arm could not be walked to home, so the '
                        f'work area is not being mapped from here. The first '
                        f'cycle will map it.')
                    return False
        finally:
            self.configure_arm(launch)
        self._set_state('BOOT', 'mapping the work area')
        return bool(self.capture_octomap())
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
        # The controller itself, for the escape from a contact. Everything
        # else in the cycle goes through move_group; this deliberately does
        # not, because move_group is one of the things being escaped from.
        if arm not in self.traj_clients:
            self.traj_clients[arm] = ActionClient(
                self, FollowJointTrajectory,
                f'/{arm}_joint_trajectory_controller/follow_joint_trajectory',
                callback_group=self.cb)
        self.traj_client = self.traj_clients[arm]

        # Joint readings are per-arm, so a switch must not compare the new
        # arm's ready pose against the old arm's measurements.
        self._arm_positions = {}
        self._arm_efforts = {}
        self._arm_velocities = {}
        self._arm_positions_at = 0.0
        self._finger_position = None
        self._finger_effort = None

        self.states_file = self.resolve_states_file()

    def holding(self, arm=None):
        """Is this arm holding something? The configured one by default."""
        return (arm or self.arm) in self._held

    def held_object(self, arm=None):
        """What this arm is holding, or None."""
        return (self._held.get(arm or self.arm) or {}).get('object')

    def hold_offset(self, arm=None):
        """How far the tool is above the bottom of what this arm holds."""
        return (self._held.get(arm or self.arm) or {}).get('offset')

    def busy_arms(self):
        """Every arm with something in its jaws, in a stable order."""
        return sorted(self._held)

    def free_arms(self):
        """Every arm that could take a new object."""
        return [arm for arm in ('left', 'right') if arm not in self._held]

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
        self.publish_status()

    def _srv_reload_config(self, _request, response):
        with self._lock:
            busy = self._busy
        if busy:
            response.success = False
            response.message = ('a cycle is running; the sequence would '
                                'change under it')
            return response
        self.load_config()
        response.success = True
        response.message = ' -> '.join(self._sequence)
        if self._config_problems:
            response.message += ' (' + '; '.join(self._config_problems) + ')'
        return response

    def _srv_save_config(self, _request, response):
        why = self.save_config()
        response.success = why is None
        response.message = why or f'saved {self.config_path()}'
        return response

    # -- the saved config ----------------------------------------------------

    def config_path(self):
        """Where the sequence and the UI-settable parameters live, or None."""
        name = self.get_parameter('pick_place_config').value
        if not name:
            return None
        return name if os.path.isabs(name) else os.path.join(WS, name)

    def load_config(self):
        """Read the saved sequence and settings, and apply them.

        Applied as ROS parameters, so everything downstream reads them the way
        it always has -- there is no second source of truth for the torque cap
        or the grasp height, just a file that sets the same parameters a
        launch argument would.

        A launch argument still wins where the two disagree *and* the file has
        no opinion; where the file has one, the file wins, because it is the
        thing the operator edited most recently. Anything unreadable falls
        back to the defaults and is reported rather than fatal: the robot
        coming up matters more than the last setting being honoured.
        """
        path = self.config_path()
        if path is None:
            self.get_logger().info('pick_place_config is empty; using the built-in '
                                   'sequence and the launch parameters')
            return
        config, problems = sequence_model.load_config(path)
        self._config_problems = problems
        for complaint in problems:
            self.get_logger().warn(f'{os.path.basename(path)}: {complaint}')
        self._sequence = list(config['sequence'])
        # Before anything is applied, and on real hardware *instead* of
        # applying it. A warning is not a guard: this warned exactly as
        # designed on run 1789014831 and the run went ahead with the grasp
        # check off, because grasp_finger_min=-1.0 had been saved by a
        # --fake rehearsal. The jaws then stalled at 1.08 mm and 0.50 Nm --
        # empty -- and the cycle called it a hold, carried nothing to the
        # drop point and finished DONE.
        settings = dict(config.get('settings') or {})
        fake = self.get_parameter('fake_hardware').value
        for warning in sequence_model.check_disabling_settings(settings):
            self._config_problems.append(warning)
            if fake:
                self.get_logger().warn(
                    f'{os.path.basename(path)} switches a check off -- '
                    f'{warning} This is a fake-hardware run, so it stands.')
                continue
            name = warning.split('=', 1)[0]
            settings.pop(name, None)
            self.get_logger().error(
                f'{os.path.basename(path)} SWITCHES A CHECK OFF -- {warning} '
                f'These are the arms, not a rehearsal, so {name} is being '
                f'ignored and the built-in default used instead. Save the '
                f'settings again from the panel to clear it out of the file.')
        applied = self.apply_settings(settings)
        self.get_logger().info(
            f'sequence ({len(self._sequence)} steps): '
            f'{" -> ".join(self._sequence)}')
        if applied:
            self.get_logger().info(
                f'settings from {os.path.basename(path)}: '
                + ', '.join(f'{k}={v}' for k, v in sorted(applied.items())))

    def apply_settings(self, settings):
        """Set the UI-settable parameters. Returns what was actually applied."""
        applied = {}
        for name, value in (settings or {}).items():
            entry = sequence_model.SETTINGS_BY_NAME.get(name)
            if entry is None:
                continue
            coerced, complaint = sequence_model.coerce_setting(entry, value)
            if complaint:
                self.get_logger().warn(complaint)
            if coerced is None:
                continue
            try:
                self.set_parameters([rclpy.parameter.Parameter(
                    name,
                    rclpy.Parameter.Type.BOOL if entry['type'] == 'bool'
                    else rclpy.Parameter.Type.DOUBLE,
                    coerced)])
            except Exception as exc:                 # noqa: BLE001 - reported
                self.get_logger().warn(f'could not set {name}: {exc}')
                continue
            applied[name] = coerced
        return applied

    def current_settings(self):
        """What the settable parameters are right now, for the UI to show."""
        values = {}
        for entry in sequence_model.SETTINGS:
            try:
                values[entry['name']] = self.get_parameter(
                    entry['name']).value
            except Exception:                        # noqa: BLE001
                continue
        return values

    def save_config(self, sequence=None, settings=None, prompt=None):
        """Persist the sequence and settings. Returns None or a reason.

        Applies before it writes, so a saved file and a running robot cannot
        disagree: if the parameters will not take, nothing is written.
        """
        path = self.config_path()
        if path is None:
            return 'pick_place_config is empty, so there is nowhere to save'
        if sequence is not None:
            cleaned, problems = sequence_model.validate_sequence(sequence)
            self._sequence = cleaned
            self._config_problems = problems
            for complaint in problems:
                self.get_logger().warn(f'sequence: {complaint}')
        if settings:
            self.apply_settings(settings)
        if prompt is not None:
            self.prompt = to_detection_prompt(prompt) or self.prompt
        body = {
            'prompt': self.prompt,
            'arm': self.arm,
            'sequence': list(self._sequence),
            'settings': self.current_settings(),
        }
        why = sequence_model.save_config(path, body)
        if why:
            self.get_logger().error(why)
        else:
            self.get_logger().info(f'saved {os.path.basename(path)}')
        self.publish_status()
        return why

    def publish_status(self):
        """Everything the UI needs to draw one frame, as JSON.

        Separate from the human-readable state string because the flow chart
        highlights a *step*, and 'PRE_PICK: on the way back, cycle complete'
        cannot be matched against a step name without parsing prose.
        """
        try:
            self.status_pub.publish(
                String(data=json.dumps(self.status_payload())))
        except Exception as exc:                     # noqa: BLE001 - reported
            self.get_logger().warn(f'could not publish status: {exc}')

    def status_payload(self):
        """The frame itself, apart from publishing it, so it can be read
        without a subscription."""
        return {
            'state': self.state,
            'step': self._step,
            'arm': self.arm,
            'cycle': self._cycle_count,
            'holding': self.holding(),
            'held_object': self.held_object(),
            # Both arms, so the panel can show a Place button per occupied
            # hand rather than one that cannot say which it means.
            'held': {arm: entry.get('object')
                     for arm, entry in self._held.items()},
            'busy': self._busy,
            'prompt': self.prompt,
            'sequence': list(self._sequence),
            'problems': list(self._config_problems),
            'stamp': round(self.get_clock().now().nanoseconds * 1e-9, 3),
        }

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

    def measured_joints(self):
        """This arm's measured positions in joint order, or None."""
        with self._lock:
            values = [self._arm_positions.get(j) for j in self.arm_joints]
        return None if any(v is None for v in values) else values

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

        Efforts are sampled too, and they are the only way to answer "is it
        pressing on the table": the endpoints alone cannot separate an arm
        leaning on something from an arm holding itself out at full stretch.
        They are also what the torque guard watches -- see watch_torque.
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
                with self._lock:
                    efforts = [self._arm_efforts.get(j)
                               for j in self.arm_joints]
                if any(v is not None for v in efforts):
                    entry['efforts'] = [None if v is None else round(v, 3)
                                        for v in efforts]
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
            self._own_side = {}

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
        * Turning the grasp about vertical, last of all -- see below.

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
        # Turning the grasp about vertical, and deliberately last.
        #
        # The yaw comes from the detector's axis estimate over a 2D box, and
        # for a round object that estimate means nothing. Measured on the
        # robot, a roll of tape at (0.323, 0.070) that the pre-flight had just
        # refused outright:
        #
        #     yaw   0 deg   approach  26%   descent   0%   <- the only one tried
        #     yaw  45 deg   approach 100%   descent  76%
        #     yaw  90 deg   approach 100%   descent 100%   <- flies
        #     yaw 135 deg   approach  26%   descent   0%
        #
        # So a pickable object was reported unreachable because one number
        # nobody measured was taken as fixed.
        #
        # Last, though, because turning the grasp is not free on every object:
        # on a screwdriver it is the difference between gripping across the
        # shaft and gripping along it, and the detector's axis is right about
        # that. A tilt is a compromise that still grips the correct faces; a
        # yaw change is a *different* grasp. So every tilt is tried first, and
        # yaw is only reached when nothing that respects the measured axis
        # flies at all. Half a turn is the flip, which is already covered, so
        # the alternatives are spread over 180 degrees.
        turns = int(self.get_parameter('grasp_yaw_options').value)
        if turns > 0:
            for index in range(1, turns + 1):
                yaw = math.pi * index / (turns + 1)
                spin = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
                for base in flips:
                    options.append(quat_mul(spin, base))
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
        # And back up again, chained from the grasp posture, because the way
        # out is part of the path and was not being checked at all.
        #
        # Measured: a cycle whose descent and grasp both passed the
        # pre-flight, and whose LIFT was then refused twice for 2.97 and 3.01
        # rad of joint travel -- with the object already in the jaws, at the
        # bottom of the column, which is the worst place to discover it. It
        # settled for a 50 mm lift and the carry started from there.
        #
        # Checked to the pre-grasp height rather than the full retreat:
        # lift_column asks for retreat_height and falls back to pregrasp, so
        # pregrasp is what it actually needs. Demanding the full 200 mm here
        # would reject candidates that fly perfectly well.
        if self.get_parameter('preflight_lift').value:
            wanted_legs.append(('lift', (grasp[0], grasp[1], pregrasp[2])))
        # Ask exactly the way the descent will, or the pre-flight is answering
        # a different question than the one that matters.
        checked_first = not (
            self.get_parameter('approach_ignores_octomap').value
            and self.get_parameter('descend_ignores_octomap').value)

        # The same budget cartesian_move enforces when the leg is really
        # flown. Checking it here is the point: a candidate whose descent
        # solves 100% and costs 3 rad is not a candidate, and finding that
        # out after the arm has flown to TRANSIT costs the whole approach.
        budget = self.get_parameter('column_max_joint_travel').value

        # Candidate zero: no posture goal at all, just a straight line from
        # where the approach starts. Tried first because it is the motion
        # asked for -- the arm travelling in a line, as if the end-effector
        # arrow were being dragged -- and taken only if the descent still
        # flies from wherever that line leaves the arm.
        if self.get_parameter('linear_transit').value and self._approach_from:
            for candidate_quat in self.grasp_quat_options(quat):
                flown, landed, _ = self.probe_cartesian(
                    self._approach_from, approach, candidate_quat,
                    avoid_collisions=True)
                if flown is None or flown < wanted or landed is None:
                    continue
                legs = [('approach', flown)]
                reached, worst = landed, flown
                costly = None
                for leg_label, target in wanted_legs:
                    part, end, cost = self.probe_cartesian(
                        reached, target, candidate_quat,
                        avoid_collisions=checked_first)
                    if part is None:
                        break
                    if part < wanted and checked_first:
                        loose, loose_end, loose_cost = self.probe_cartesian(
                            reached, target, candidate_quat,
                            avoid_collisions=False)
                        if loose is not None and loose > part:
                            part, end, cost = loose, loose_end, loose_cost
                    legs.append((leg_label, part))
                    worst = min(worst, part)
                    # Solving is not the same as being flyable. The guard in
                    # cartesian_move will refuse this leg for exactly this
                    # number, so refuse it here where nothing has moved yet.
                    if budget > 0.0 and cost is not None and cost > budget:
                        costly = (leg_label, cost)
                        break
                    if end is not None:
                        reached = end
                    if part < wanted:
                        break
                if costly is not None:
                    self.get_logger().info(
                        f'{label}: the straight-line approach solves, but its '
                        f'{costly[0]} leg would cost {costly[1]:.2f} rad on '
                        f'one joint against a {budget:.2f} rad budget -- the '
                        f'tool would trace the line while the arm swings '
                        f'across the workspace. Trying a posture instead.')
                    continue
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
            costly = None
            for leg_label, target in wanted_legs:
                fraction, end, cost = self.probe_cartesian(
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
                    unchecked, end_unchecked, unchecked_cost = \
                        self.probe_cartesian(
                            reached, (target[0], target[1], target[2]),
                            entry['quat'], avoid_collisions=False)
                    if unchecked is not None and unchecked > fraction:
                        fraction, end = unchecked, end_unchecked
                        cost = unchecked_cost
                legs.append((leg_label, fraction))
                worst = min(worst, fraction)
                # Same budget the flown leg is held to, applied before
                # anything moves.
                if budget > 0.0 and cost is not None and cost > budget:
                    costly = (leg_label, cost)
                    worst = 0.0
                    break
                if end is not None:
                    reached = end
                if fraction < floor:
                    break
            entry['legs'] = legs
            entry['worst_leg'] = worst
            entry['costly'] = costly
            if costly is not None:
                self.get_logger().info(
                    f'{label}: candidate {index} solves but its {costly[0]} '
                    f'leg costs {costly[1]:.2f} rad on one joint against a '
                    f'{budget:.2f} rad budget; rejected before moving.')
            if best is None or worst > best['worst_leg']:
                best = entry
            detail = ', '.join(f'{name} {value * 100:.0f}%'
                               for name, value in legs)
            if worst >= wanted and not self.posture_is_gettable(
                    entry, label, index):
                # Proving the descent from a posture is only half the
                # question; the other half is whether the arm can get into it.
                # Measured, run 1788945589: the pre-flight passed a posture
                # candidate with 2.151 rad of travel, and TRANSIT -- the joint
                # goal that puts the arm in it -- came back -2. A pre-flight
                # that approves a posture the arm cannot reach has checked the
                # wrong thing.
                worst = 0.0
                entry['worst_leg'] = worst
                entry['unreachable_posture'] = True
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
        # Say what actually refused it. All the legs can read 100% and the
        # candidate still be rejected -- because the arm cannot be planned
        # into the posture that flies them. Reporting that as "no descent
        # exists" alongside "grasp 100%, lift 100%" is a contradiction the
        # reader has to untangle, and it points at the object's position when
        # the problem is the approach.
        stuck = sum(1 for c in candidates if c.get('unreachable_posture'))
        costly = sum(1 for c in candidates if c.get('costly'))
        if stuck == len(candidates) and stuck:
            blame = ('every posture that can fly the descent is one the arm '
                     'cannot be planned into')
            advice = ('The column is fine; getting into position for it is '
                      'not. Something is in the way between the staging pose '
                      'and the object -- check the octomap, and try clearing '
                      'it: ros2 service call /clear_octomap '
                      'std_srvs/srv/Empty')
            why = 'posture-unreachable'
        elif costly:
            blame = 'no descent worth flying exists'
            advice = (f'{costly} of {len(candidates)} solved the line but '
                      f'would have swung the arm across the workspace to do '
                      f'it. Bring the object closer to the base.')
            why = 'joint-travel'
        else:
            blame = 'no straight-line descent exists'
            advice = ('This is the arm running out of travel along the '
                      'descent, not an obstacle -- bring the object closer '
                      'to the base, or try the other arm.')
            why = 'no-line'
        self._preflight_reason = (
            f'{blame} from any of the {len(candidates)} postures that reach '
            f'({approach[0]:.3f}, {approach[1]:.3f}, {approach[2]:.3f}). '
            f'The best managed {detail}. Nothing moved. {advice}')
        self.log_motion(label, 'preflight', 'refused',
                        secs=round(self.get_clock().now().nanoseconds * 1e-9
                                   - preflight_from, 2),
                        target=[round(v, 5) for v in approach],
                        of=len(candidates), why=why,
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

    def _srv_place(self, _request, response):
        response.success, response.message = self._start_place()
        return response

    def _srv_place_left(self, _request, response):
        response.success, response.message = self._start_place('left')
        return response

    def _srv_place_right(self, _request, response):
        response.success, response.message = self._start_place('right')
        return response

    def _srv_stop_arm(self, _request, response):
        response.success, response.message = self._start_stop_arm()
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
            # A pick needs a free hand, not an empty robot. With one arm
            # full the other one takes the job -- choose_arm is told to
            # skip the occupied ones -- and only with both full is there
            # nowhere to put anything.
            if not self.free_arms():
                full = ', '.join(f'{arm} is holding {entry.get("object")}'
                                 for arm, entry in sorted(self._held.items()))
                return False, (f'both hands are full ({full}) -- press a '
                               f'Place button to put one down first')
            self._busy = True
        self._abort.clear()
        # Before the thread, and in the caller's own call, because the first
        # few seconds of a cycle are spent on service calls -- the states
        # file, the planning scene, the planner's tool frame, a plan-only
        # probe -- and until one of them finishes nothing publishes a state
        # at all. The panel sat on the previous cycle's DONE with its
        # buttons greyed, and the motion log's heartbeat recorded IDLE while
        # a cycle was demonstrably running.
        self._set_state('STARTING', 'checking the planner and the poses')
        threading.Thread(target=self._run_cycle, daemon=True).start()
        return True, 'cycle started'

    def _start_stop_arm(self):
        """Park the arm and let go of it. The deliberate way to end a session.

        Takes the motors off, which on direct-drive motors with no brakes
        means the arm is then held up by nothing -- so it goes home first
        and only lets go once it is there. Home is a folded posture resting
        on itself; that is the whole reason this parks before it releases
        rather than releasing where it stands.

        Interrupts whatever is running. A Stop that waited politely for a
        cycle to finish would be no use at the moment anyone reaches for
        it.
        """
        self._abort.set()
        threading.Thread(target=self._run_stop_arm, daemon=True).start()
        return True, 'stopping: parking the arm, then taking the motors off'

    def _run_stop_arm(self):
        # Let whatever is running unwind first. It sees the abort flag at
        # the next check and returns; this only has to outlast one move.
        deadline = time.time() + 30.0
        while time.time() < deadline:
            with self._lock:
                if not self._busy:
                    break
            time.sleep(0.2)
        with self._lock:
            if self._busy:
                self.get_logger().error(
                    'something is still running 30 s after the abort, so the '
                    'arm is not being parked from under it. Try again.')
                return
            self._busy = True
        # The abort flag is what every move checks before starting, so it
        # has to come down before this one can fly.
        self._abort.clear()
        try:
            self._set_state('STOPPING', 'parking the arm before letting go')
            states = self.load_states()
            if self._held:
                self.get_logger().warn(
                    f'still holding {", ".join(self.busy_arms())}. Whatever '
                    f'is in the jaws will be dropped wherever the arm ends '
                    f'up -- the motors cannot come off with the jaws driven.')
            parked = self._retreat_to_home(states, 'stopping')
            if not parked:
                self._set_state(
                    'STOPPED',
                    'could not park the arm, so the motors are staying on -- '
                    'letting go here would drop it')
                self.get_logger().error(
                    'not taking the motors off: the arm is not at a resting '
                    'posture and these motors have no brakes. Clear whatever '
                    'is blocking the way home and press Stop again, or drive '
                    'it out with record_states.py --play.')
                return
            self.disengage_motors('stopped by the operator, at home')
            self._set_state(
                'STOPPED',
                'parked at home and the motors are off. Restart the bringup '
                'to use it again')
        except Exception as exc:                     # noqa: BLE001 - reported
            self.get_logger().error(f'the stop failed: {exc}')
            self._set_state('STOPPED', f'the stop failed: {exc}')
        finally:
            with self._lock:
                self._busy = False
            self.publish_status()

    def _start_place(self, arm=None):
        """Put down what an arm is holding. Refused if it is not.

        `arm` names which hand. Left out, it means "the one that is
        holding something" -- which is unambiguous with one full hand and
        is refused with two, because a button that guesses which object to
        put down is worse than two buttons that say.

        Two questions, and both have to be yes. Does the robot think that
        arm picked something -- which is verify_grasp's answer, carried in
        _held -- and do the jaws agree *now*, which is a fresh reading and
        the one that catches an object dropped between the pick and the
        button.

        A place with nothing in the jaws is not harmless. It flies the whole
        approach to the drop surface, opens on nothing and reports a
        successful place, which is exactly the false success the grasp check
        exists to stop.
        """
        with self._lock:
            if self._busy:
                return False, 'a cycle is already running'
            full = self.busy_arms()
        if arm is None:
            if len(full) > 1:
                return False, (f'both hands are holding something '
                               f'({", ".join(full)}) -- say which to put '
                               f'down')
            if not full:
                return False, ('nothing has been picked yet -- Place puts '
                               'down what a gripper is holding')
            arm = full[0]
        if arm not in full:
            return False, (f'the {arm} hand is not holding anything'
                           + ('' if not full else
                              f' -- {" and ".join(full)} is'))

        # The gripper has to be asked about the arm being placed from, and
        # the readings are per arm, so configure it first. Nothing is
        # moving yet -- this only points the joint-state and gripper
        # plumbing at the right hand.
        if arm != self.arm:
            self.configure_arm(arm)
            self.await_joint_states()
        held = self.held_object(arm)
        # Outside the lock: finger_verdict takes it, and it is not reentrant.
        feel = self.finger_verdict()
        if feel['held'] is False:
            self.get_logger().error(
                f'refusing to place: {feel["why"]}. Whatever the {arm} hand '
                f'picked is not in its jaws now.')
            self._held.pop(arm, None)
            self.publish_status()
            return False, f'the {arm} gripper is empty -- {feel["why"]}'
        with self._lock:
            if self._busy:
                return False, 'a cycle is already running'
            self._busy = True
        self._abort.clear()
        self._set_state('STARTING', f'placing {held or "the object"} from '
                                    f'the {arm} hand')
        threading.Thread(target=self._run_place, daemon=True).start()
        return True, f'placing {held or "the object"} from the {arm} hand'

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
        # A planner that answered earlier in this run and has now vanished
        # is not going to come back inside a cycle, and every goal after it
        # costs the full wait. Measured, run 1789012345: three refuges, ten
        # seconds each, all of them hopeless.
        patience = 2.0 if self._planner_stalled else 10.0
        if not self.move_client.wait_for_server(timeout_sec=patience):
            if self._planner_answered:
                self.get_logger().error(
                    f'{label}: /move_action has stopped being served. '
                    f'move_group answered earlier in this run, so the node '
                    f'has jammed rather than never started -- it does that '
                    f'after rejecting a path as invalid. Nothing that needs '
                    f'a planner will work until it is restarted; the arm can '
                    f'still be driven out with record_states.py --play.')
            else:
                self.get_logger().error(
                    '/move_action unavailable -- is move_group running?')
            self._planner_stalled = self.get_clock().now().nanoseconds * 1e-9
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

        result, why = self.await_flight(handle, label)
        if result is None:
            self._planner_stalled = self.get_clock().now().nanoseconds * 1e-9
            if why == 'never-started':
                # Retryable, and accurately so: nothing was flown, the goal
                # has been cancelled, and resending cannot race with a
                # flight that never began.
                self.get_logger().error(
                    f'{label}: move_group took the goal and the arm never '
                    f'moved; cancelled it rather than waiting out the rest '
                    f'of motion_timeout')
                return MoveItErrorCodes.TIMED_OUT
            self.get_logger().error(f'{label}: timed out waiting for execution')
            # A goal that never came back may still be executing; resending
            # would race with it.
            return MoveItErrorCodes.FAILURE
        self._planner_answered = True
        return result.result.error_code.val

    def await_flight(self, handle, label):
        """Wait for an accepted goal to finish. (result, why).

        Two different waits, because there are two different failures and
        only one of them is worth a minute. A long flight is allowed
        motion_timeout. A goal that was accepted and then never moved the
        arm is not a long flight, and it is given motion_start_timeout
        before being cancelled -- see that parameter for the run this comes
        from.

        `why` is 'ok', 'timeout' or 'never-started'. Cancelling before
        returning is the part that makes 'never-started' safe to retry.
        """
        future = handle.get_result_async()
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())

        budget = self.get_parameter('motion_timeout').value
        patience = self.get_parameter('motion_start_timeout').value
        if patience > 0.0:
            # Never less than the planner is allowed to think for. cuMotion
            # plans *after* accepting the goal, so a goal that is still
            # being planned looks exactly like one that will never move,
            # and cancelling it would turn a slow plan into a failure.
            # Tied to planning_time rather than assumed, so raising one
            # cannot silently break the other.
            patience = max(patience,
                           self.get_parameter('planning_time').value + 5.0)
        began = self.joint_snapshot().get('joints') or {}
        start = time.monotonic()
        moved = False
        while True:
            if done.wait(0.1):
                return future.result(), 'ok'
            waited = time.monotonic() - start
            if waited > budget:
                return None, 'timeout'
            if moved or patience <= 0.0 or not began:
                continue
            here = self.joint_snapshot().get('joints') or {}
            # 0.01 rad is well above the resting jitter on these joints and
            # well below any move worth making.
            moved = any(abs(here[joint] - began[joint]) > 0.01
                        for joint in began if joint in here)
            if not moved and waited > patience:
                try:
                    handle.cancel_goal_async()
                except Exception:                    # noqa: BLE001
                    pass
                self.log_motion(label, 'goal', 'never-started',
                                waited_s=round(waited, 1))
                return None, 'never-started'

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
        """Is the arm measurably at the rest pose right now?

        With use_home off -- the default -- the rest pose *is* the staging
        pose, and that is what this asks about. Asking about HOME would
        fail for ever, since the arm is never sent there.

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
        # Which posture "the rest pose" means is the only thing use_home
        # changes here. Everything below -- the insistence on a reading
        # that post-dates the question above all -- applies either way:
        # at_state does *not* demand a fresh one, and going through it
        # instead compared the staging pose against the joint values from
        # before the move. 28 ms after the goal returned, the cached
        # reading was still the pose the arm had left, and a cycle that
        # had arrived correctly reported "joint7 is 1.092 rad off".
        wanted = self.observation_positions()
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

    def observation_positions(self):
        """The joint values of whatever the arm looks and rests at.

        The staging pose unless use_home puts HOME back, so a failure to
        get there is reported against the pose it was actually sent to
        rather than against one it was never going to.
        """
        if self.get_parameter('use_home').value:
            return self.home_positions()
        entry = (self.load_states() or {}).get(PRE_PICK_STATE) or {}
        joints = entry.get('joints')
        return list(joints) if joints else self.home_positions()

    def move_to_home(self):
        """The rest and observation pose -- which is the staging pose now.

        HOME is a folded posture on the far side of the workspace, and
        every trip to it is a long unplanned swing that buys nothing the
        staging pose does not: the map is captured from wherever the arms
        stand (gripper_never_in_map handles the gripper, robot_in_map
        throws away a map that caught anything else), the detector looks
        from staging, and a cycle starts and ends there.

        It also could not be reached. joint4's HOME value of 0.20 rad sits
        against its lower limit and the elbow stops short of it, so the
        arrival check had to be given its own slack just to pass -- and
        with the arm at the staging pose the miss reads as 1.979 rad,
        which is not a fault, it is simply "at pre_pick, not at home".

        use_home:=true restores it for a stack that needs the fold-away.
        """
        if self.get_parameter('use_home').value:
            return self._move_to_joints(self.home_positions(), 'HOME',
                                        skip_if_there=True)
        states = self.load_states()
        if not (states.get(PRE_PICK_STATE) or {}).get('joints'):
            # No staging pose to go to, so HOME it is after all. Falling
            # back rather than refusing, and falling back to exactly what
            # observation_positions() does: the two have to agree about
            # which posture "the rest pose" means, or the move goes to one
            # and the arrival check asks about the other.
            self.get_logger().warn(
                f'{PRE_PICK_STATE} is not recorded for the {self.arm} arm, '
                f'so there is no staging pose to rest at and this falls '
                f'back to HOME. Record one with "python3 record_states.py '
                f'{PRE_PICK_STATE}".')
            return self._move_to_joints(self.home_positions(), 'HOME',
                                        skip_if_there=True)
        self.get_logger().info(
            'going to the staging pose rather than HOME -- see move_to_home')
        return self.move_to_state(PRE_PICK_STATE, states,
                                  on_own_side=self.holding())

    def capture_octomap(self, anywhere=False, in_frame=False):
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
        if self._held:
            self.get_logger().info(
                'carrying the object: leaving the octomap alone so the payload '
                'is not captured as an obstacle')
            return False
        if not anywhere and not self.at_home_pose():
            self.get_logger().warn(
                'skipping the octomap update: the arm is not at home')
            return False
        # Whether the robot is standing in the picture for this capture.
        # at_home_pose() cannot answer that on its own: with use_home off
        # the rest pose *is* the staging pose, so it reads True while the
        # arm is out over the table and squarely in the camera's view --
        # which would skip the robot_in_map() check below at exactly the
        # moment it is needed. in_frame says so explicitly.
        arm_in_view = in_frame or not self.at_home_pose()
        if anywhere and arm_in_view:
            # Deliberate, and it depends on something being true: the
            # updater's self-filter renders the robot from the URDF at the
            # current TF and drops those pixels before they become voxels.
            # If that is not loaded, this captures the arm as an obstacle
            # sitting exactly where it is about to plan from -- which is
            # the failure the at-home rule was written for. The voxel count
            # below is what tells the two apart.
            self.get_logger().info(
                'capturing the map from where the arm stands. The arm is in '
                'the camera view here, so this is only sound if the '
                "updater's self-filter is loaded -- see sensors_3d.yaml.")
        if not self.refresh_octomap():
            return False
        # The updater integrates asynchronously; planning against a half-built
        # map is worse than planning against the previous one.
        self._abort.wait(self.get_parameter('octomap_settle_time').value)
        # And then check it actually landed. Every part of this cycle that
        # talks about voxels -- the gripper's exemption, the object being
        # in the map, a refuge being occupied -- is about a map that may
        # not exist.
        if anywhere and arm_in_view:
            # Did it capture the robot? This is the one failure that
            # poisons everything after it: an arm inside its own map has an
            # invalid start state, and then nothing plans -- not the pick,
            # not the retreat, not the trip home. Better to lose the map
            # than to keep one that stops the robot moving at all.
            caught = self.robot_in_map()
            if caught:
                self.get_logger().error(
                    f'the map captured the robot itself -- '
                    f'{", ".join(caught[:4])}. An arm inside its own map has '
                    f'an invalid start state and nothing will plan from it, '
                    f'so the map is being dropped rather than kept. The '
                    f'self-filter in sensors_3d.yaml is what should have '
                    f'removed it: raise padding_offset, or set '
                    f'boot_via_home so the map is taken from HOME with the '
                    f'arms out of frame.')
                self.clear_octomap()
                return False
        voxels = self.octomap_voxels()
        if voxels is None:
            self.get_logger().info(
                'the octomap was refreshed; the planning scene could not be '
                'read back to say how much is in it')
        elif voxels == 0:
            self.get_logger().error(
                'the octomap is EMPTY after a refresh, so the planner is '
                'checking against nothing the camera can see. Either no '
                'occupancy map updater loaded -- look for a pluginlib '
                'failure in move_group at startup -- or the self-filter is '
                'eating everything: padding_offset in sensors_3d.yaml is a '
                'margin around the whole robot, and a large one erases the '
                'table and the object along with the arm.')
        else:
            self.get_logger().info(
                f'octomap refreshed: {voxels} bytes of tree in the planning '
                f'scene')
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
        if self._held:
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

    def refresh_map_in_place(self):
        """Drop the old map and rebuild it from where the arm stands.

        What map_from_home() does, without the trip. A cycle that looks
        from the staging pose never passes through HOME, so before this
        existed it planned against whatever the boot walk captured --
        measured, run 1790054584: a map taken at 10:53 was still the one
        in the scene at 10:44 eleven minutes later, so anything put on or
        taken off the table in between simply was not there as far as the
        planner was concerned.

        The clear comes first and is not optional, for the same reason it
        is not optional in map_from_home(): voxels are only ever added by
        a refresh, never removed by one, so without a clear the scene
        accumulates every object that has ever been on the table --
        including the one just picked up and carried away.

        The cost is that the arm is in the picture here. That is what the
        self-filter in sensors_3d.yaml is for, and capture_octomap checks
        the result with robot_in_map() and throws the map away rather than
        keep one the arm cannot plan out of.
        """
        if self._held:
            self.get_logger().info(
                'carrying the object: not remapping, so the payload is not '
                'captured and the existing map is kept')
            return False
        self.clear_octomap()
        return self.capture_octomap(anywhere=True, in_frame=True)

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

    def move_to_state(self, name, states, on_own_side=False):
        """Replay a pose recorded by record_states.py.

        on_own_side measures the plan before flying it and keeps the tool
        out of the other arm's half on the way -- see fly_on_own_side. It
        costs one extra plan, so it is asked for on the leg that carries
        something back across the workspace rather than everywhere.
        """
        entry = states.get(name)
        if not entry or not entry.get('joints'):
            self.get_logger().error(
                f'{name} is not in {self.states_file}; record it with '
                f'"python3 record_states.py {name}"')
            return False
        joints = list(entry['joints'])
        if on_own_side and self.get_parameter('carry_guard').value:
            flown = self.fly_on_own_side(joints, name.upper())
            if flown is not None:
                return flown
        return self._move_to_joints(joints, name.upper(),
                                    skip_if_there=True)

    def own_side(self):
        """Which half this arm works in: +1.0 for +y, -1.0 for -y, None unknown.

        Measured rather than configured. The staging pose is recorded per
        arm by record_states.py, and running the arm's own kinematics over
        those seven numbers says where its tool stands when it is out of
        the way: measured, the left arm's at y=+0.175 and the right arm's
        at the mirror of it.

        None is a real answer and means the guard does not run. Guessing a
        direction here would be worse than not guarding at all -- guessed
        backwards, it would hold the arm to the half it is meant to keep
        out of.
        """
        with self._lock:
            side = self._own_side.get(self.arm)
        if side is not None:
            return side
        chain = self.kinematics()
        if chain is None:
            return None
        entry = (self.load_states() or {}).get(PRE_PICK_STATE) or {}
        joints = entry.get('joints')
        if not joints or len(joints) != len(self.arm_joints):
            return None
        y = float(chain.pose(chain.tool_link, list(joints))[1, 3])
        side = 1.0 if y >= 0.0 else -1.0
        with self._lock:
            self._own_side[self.arm] = side
        return side

    def tool_path(self, trajectory):
        """A planned trajectory as [(seconds, [x, y, z])], by forward kinematics.

        What the tool would do if the arm flew this plan. None when that
        cannot be worked out -- no kinematic chain, an empty trajectory, or
        one for joints that are not this arm's.
        """
        chain = self.kinematics()
        if chain is None or trajectory is None:
            return None
        points = trajectory.joint_trajectory.points
        if not points:
            return None
        order = {name: index for index, name in
                 enumerate(trajectory.joint_trajectory.joint_names)}
        if not all(joint in order for joint in self.arm_joints):
            return None
        path = []
        for point in points:
            values = [point.positions[order[joint]]
                      for joint in self.arm_joints]
            xyz = chain.pose(chain.tool_link, values)[:3, 3]
            path.append((round(point.time_from_start.sec
                               + point.time_from_start.nanosec * 1e-9, 3),
                         [float(v) for v in xyz]))
        return path

    def worst_crossing(self, trajectory, side, limit):
        """How far past `limit` this plan takes the tool, in metres.

        Returns (metres over, the worst point) -- (0.0, None) for a plan
        that stays put, (None, None) when the path cannot be measured at
        all, which is "do not know" rather than "clean".

        `limit` and the comparison are in the arm's own frame, side * y, so
        larger is always further into the arm's own half whichever arm it
        is and there is one inequality rather than two mirrored ones.
        """
        path = self.tool_path(trajectory)
        if path is None:
            return None, None
        worst = 0.0
        where = None
        for _, xyz in path:
            over = limit - side * xyz[1]
            if over > worst:
                worst, where = over, xyz
        return worst, where

    def plan_joint_path(self, positions, label):
        """Ask for a joint-goal plan without flying it. Trajectory, or None.

        The same request _move_to_joints sends, with plan_only set: nothing
        moves, and what comes back is the path the arm would take, which is
        the only way to judge a path when the planner takes no path
        constraints.
        """
        if not self.move_client.wait_for_server(timeout_sec=2.0):
            return None
        request = self._base_request()
        constraints = Constraints()
        for name, value in zip(self.arm_joints, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        request.goal_constraints = [constraints]
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = True          # nothing moves
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        handle = self._await(self.move_client.send_goal_async(goal), 15.0)
        if handle is None or not handle.accepted:
            self.get_logger().warn(
                f'{label}: the plan-only goal was not accepted')
            return None
        result = self._await(handle.get_result_async(),
                             self.get_parameter('motion_timeout').value)
        if result is None:
            self.get_logger().warn(f'{label}: the plan-only goal timed out')
            self._planner_stalled = self.get_clock().now().nanoseconds * 1e-9
            return None
        self._planner_answered = True
        code = result.result.error_code.val
        if code != MOVEIT_SUCCESS:
            self.get_logger().info(
                f'{label}: plan-only came back {code}')
            return None
        return result.result.planned_trajectory

    def fly_on_own_side(self, positions, label, tries=None):
        """Fly a joint goal by a path that stays out of the other arm's half.

        Why this exists: the goal is already on this arm's own side -- the
        staging pose is where it is -- so nothing about the *goal* can say
        which way round the arm gets there. A 7-DOF arm has a continuum of
        ways to reach one posture, and a free-space plan back from a pick
        is free to swing the tool across the middle and out again. The
        other arm is standing over there, in the air, and the octomap does
        not know it: the map is of the table, taken while both arms were
        out of frame.

        The planner takes no path constraints, so the path is asked for,
        measured with the arm's own kinematics, and flown only if it stays
        put. cuMotion's optimiser is stochastic, which is what makes
        re-asking worth anything: the same goal gives a different path.

        Returns True or False once something has been flown or refused, and
        None when the guard cannot run at all -- no kinematics, no
        plan-only answer -- so the caller sends the goal the ordinary way
        rather than standing still over a guard that cannot judge.
        """
        side = self.own_side()
        if side is None:
            return None
        if len(positions) == len(self.arm_joints):
            positions = self.clamp_to_limits(positions, label)
        worst = self.joint_error(positions, fresh=True)
        tolerance = self.get_parameter('at_goal_tolerance').value
        if worst is not None and worst <= tolerance:
            self.get_logger().info(
                f'{label}: already there ({worst:.4f} rad off), so there is '
                f'no path to keep on this side')
            return True
        here = self.fresh_tcp()
        slack = self.get_parameter('carry_side_slack').value
        # The line the tool may not go past. Not simply the centre-line:
        # a pick near the middle can legitimately leave the arm reaching
        # across it, and holding the way back to y >= 0 would refuse a move
        # that only ever comes *back*. So the rule is that the return does
        # not go further over than it already is.
        limit = -slack
        if here is not None:
            limit = min(0.0, side * here[1]) - slack
        attempts = max(1, int(self.get_parameter('carry_guard_tries').value
                              if tries is None else tries))
        best = None
        for attempt in range(1, attempts + 1):
            if self._abort.is_set():
                self.get_logger().warn(f'{label}: aborted')
                return False
            trajectory = self.plan_joint_path(positions, label)
            if trajectory is None:
                continue
            over, where = self.worst_crossing(trajectory, side, limit)
            if over is None:
                # Nothing to judge it with, and an unjudged plan is not
                # this guard's to refuse.
                return None
            if over <= 0.0:
                return self.fly_plan(trajectory, positions, label,
                                     attempt, 0.0)
            spot = (f'({where[0]:.3f}, {where[1]:.3f}, {where[2]:.3f})'
                    if where else 'somewhere on the way')
            self.get_logger().warn(
                f'{label}: plan {attempt} of {attempts} takes the tool '
                f'{over * 1000:.0f} mm into the other arm\'s half, at '
                f'{spot} -- asking for another one')
            if best is None or over < best[0]:
                best = (over, trajectory, attempt)
        if best is None:
            # The planner never answered. Not a crossing, and not something
            # this guard can fix; the ordinary path reports it properly.
            return None
        refuse = self.get_parameter('carry_side_refuse').value
        if best[0] > refuse:
            self.get_logger().error(
                f'{label}: every one of {attempts} plans crosses into the '
                f'other arm\'s half, the best by {best[0] * 1000:.0f} mm, '
                f'which is past the {refuse * 1000:.0f} mm this will fly. '
                f'That is a swing through where the other arm is parked, '
                f'not a clip of the middle. The arm stays where it is, '
                f'still holding. Park the other arm out of the way '
                f'(boot_other_arm:=home) or raise carry_side_refuse.')
            self.log_motion(label, 'joint', 'refused-crossing',
                            target=[round(v, 6) for v in positions],
                            attempt=best[2],
                            crossing_mm=round(best[0] * 1000, 1))
            return False
        # Flown, because the alternative is worse. A guard that leaves the
        # arm stranded over the table holding something has not made
        # anything safer -- see carry_side_refuse.
        self.get_logger().warn(
            f'{label}: none of the {attempts} plans stayed clear; flying the '
            f'best of them, which crosses {best[0] * 1000:.0f} mm into the '
            f'other arm\'s half. Under the {refuse * 1000:.0f} mm that '
            f'would be refused, and the alternative is stopping out here '
            f'with the object in the jaws.')
        return self.fly_plan(trajectory=best[1], positions=positions,
                             label=label, attempt=best[2], crossing=best[0])

    def fly_plan(self, trajectory, positions, label, attempt, crossing):
        """Execute a plan that has already been measured and accepted.

        Logged exactly as _send_move_goal logs its own moves, because the
        motion log is where a leg is explained afterwards and a leg that
        went through the guard is no less worth explaining.
        """
        before = self.joint_snapshot()
        finish = self.sample_motion()
        flown = self._execute(trajectory, label)
        path = finish()
        self.log_motion(label, 'joint', 'ok' if flown else 'failed',
                        target=[round(v, 6) for v in positions],
                        attempt=attempt, before=before, path=path,
                        crossing_mm=round(crossing * 1000, 1),
                        guarded=True)
        return flown

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
        self._forget_grasp()
        return self.command_gripper(
            self.get_parameter('gripper_open').value, label)

    def close_gripper(self, label='CLOSE'):
        """Shut the jaws, with no grasp check.

        Deliberately not close_gripper_to_cap: that one steps in, watches the
        torque and reports whether anything ended up between the fingers.
        Here there is nothing to grip -- the object has just been let go --
        so a 'closed on nothing' verdict would be both correct and useless.
        One command, and the arm leaves with a narrow profile instead of
        44 mm of open fingers.
        """
        self._forget_grasp()
        return self.command_gripper(0.0, label)

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
                if not self.command_gripper(position, 'HOLD', settle=settle):
                    return False
                self._latch_grasp('at the torque cap')
                return True
            position = advanced

        torque = abs(self.finger_effort() or 0.0)
        self.get_logger().info(
            f'closed to {position:.4f} m without reaching {cap:.3f} Nm '
            f'(torque {torque:.3f} Nm)')
        # Whether anything is between the fingers is answered by where the
        # fingers *are*, not by what they were last told.
        #
        # This compared `position` -- the last commanded step, which at the
        # end of a full close is always 0.0 -- against a 3 mm floor, so every
        # close that ran to the end was declared empty whatever the jaws were
        # actually doing. It threw away working grasps: measured across 14
        # real closes,
        #
        #   empty      measured finger 0.0025-0.0026 m, effort 1.14-1.24 Nm
        #   holding    measured finger 0.0039-0.0175 m, effort 1.84-2.46 Nm
        #
        # -- two bands with nothing between them, and grasp_finger_min at
        # 0.003 already sitting in the gap. Three runs in that table were
        # rejected as empty while gripping something ~4 mm thick at ~1.9 Nm;
        # one of them was a roll of tape the arm was visibly holding. The
        # parameter was right all along and was being compared with the wrong
        # number.
        #
        # The commanded value stays in the message, because a close that ran
        # all the way to 0.0 mm without the cap is still worth seeing.
        floor = self.get_parameter('grasp_finger_min').value
        measured = self.finger_position()
        if measured is None:
            self.get_logger().warn(
                'no finger feedback, so whether anything is held can only be '
                'guessed from the last command')
            measured = position
        if measured <= floor:
            self.get_logger().error(
                f'the gripper closed to {measured * 1000:.1f} mm (commanded '
                f'{position * 1000:.1f} mm) without the torque ever passing '
                f'{cap:.2f} Nm (peak {torque:.2f} Nm) -- there was nothing '
                f'between the fingers. Either the object is not where it was '
                f'detected, or the jaws are not where they were sent. Read '
                f'the CLOSE_GRIPPER line just above: it prints the measured '
                f'jaw position against the commanded grasp, and says '
                f'off-target when they disagree.')
            self.log_motion('CLOSE', 'gripper', 'closed-on-nothing',
                            target=round(float(position), 6),
                            finger=round(float(measured), 6),
                            torque=round(torque, 3), cap=cap)
            return False
        self.get_logger().info(
            f'the fingers stopped {measured * 1000:.1f} mm apart at '
            f'{torque:.2f} Nm without reaching the {cap:.2f} Nm cap -- thin, '
            f'but held. Below {floor * 1000:.1f} mm would be nothing.')
        self.log_motion('CLOSE', 'gripper', 'held-under-cap',
                        target=round(float(position), 6),
                        finger=round(float(measured), 6),
                        torque=round(torque, 3), cap=cap)
        self._latch_grasp('under the cap')
        return True

    def finger_position(self):
        with self._lock:
            return self._finger_position

    def _latch_grasp(self, how):
        """Remember the width and torque a good close ended at.

        Latched at the one moment the grip is known good -- a close that has
        just passed its own checks -- because everything asked later is a
        question about a change since then, and there was nothing to compare
        against. Recorded after the hold has settled, so it is the number
        verification will see if nothing moves, not the peak on the way in.
        """
        width = self.finger_position()
        torque = self.finger_effort()
        with self._lock:
            self._grasp_close = None if width is None else {
                'width': float(width),
                'torque': abs(float(torque or 0.0)),
                'how': how}
        if width is not None:
            self.get_logger().info(
                f'gripped {how}: {width * 1000:.1f} mm at '
                f'{abs(float(torque or 0.0)):.2f} Nm. That is what the '
                f'grasp check compares against.')

    def _forget_grasp(self):
        """Whatever was between the fingers is not any more."""
        with self._lock:
            self._grasp_close = None

    def grasp_close_reading(self):
        with self._lock:
            return None if self._grasp_close is None else dict(
                self._grasp_close)

    def finger_verdict(self):
        """What the gripper itself says about whether it is holding.

        Returns a dict: `held` True, False, or None where it cannot say;
        `source` naming the reading that decided; `why` a sentence with the
        numbers in it, because every one of these has been wrong at least
        once and the log is where that gets caught.

        Three readings, kept apart because they go blind in different
        places:

          width   where the jaws actually stopped. 4-18 mm holding against
                  1.1-2.5 mm empty, measured across every close on this
                  robot, with nothing in between. Blind to a jaw that
                  stopped inside the band on nothing at all -- friction,
                  backlash, a fingertip resting on an edge.
          torque  what they are pressing with. Blind to how wide they are,
                  and absent entirely on a build from before the driver read
                  gripper_motors[0].get_torque().
          drift   the width against the width the close ended at, which is
                  the one that catches a drop the band cannot: 6 mm is a
                  grip if the close ended at 6 mm and an empty jaw if it
                  ended at 14. Needs a latched close, so it says nothing
                  after a restart or an uncapped fallback close.

        None of them can say *what* is held -- a jaw shut on a cable reads
        like a jaw shut on the object -- which is why this is the tie-break
        in verify_grasp and not the test.
        """
        floor = self.get_parameter('grasp_finger_min').value
        ceiling = self.get_parameter('grasp_finger_max').value
        torque = self.finger_effort()
        torque = None if torque is None else abs(float(torque))
        width = self.finger_position()
        feel = {'held': None, 'source': None, 'why': '',
                'width': width, 'torque': torque}
        if floor < 0.0:
            feel['why'] = (
                'grasp_finger_min is negative, so the gripper check is off. '
                'That is the fake-hardware value and it leaves nothing to '
                'fall back on when the detector cannot see')
            return feel
        if width is None:
            feel['why'] = 'there is no finger feedback on /joint_states'
            return feel
        if not floor < width < ceiling:
            feel.update(held=False, source='width', why=(
                f'the jaws are {width * 1000:.1f} mm apart, outside the '
                f'{floor * 1000:.1f} to {ceiling * 1000:.1f} mm that means '
                f'holding'))
            return feel
        closed = self.grasp_close_reading()
        drop = self.get_parameter('grasp_width_drop').value
        if closed is not None and drop > 0.0 and width < closed['width'] - drop:
            feel.update(held=False, source='drift', why=(
                f'the jaws have travelled '
                f'{(closed["width"] - width) * 1000:.1f} mm further in since '
                f'they closed ({closed["width"] * 1000:.1f} to '
                f'{width * 1000:.1f} mm), which is what letting go looks '
                f'like'))
            return feel
        weak = self.get_parameter('grasp_hold_torque_min').value
        if torque is not None and weak > 0.0 and torque < weak:
            feel.update(held=False, source='torque', why=(
                f'the jaws are {width * 1000:.1f} mm apart but pressing with '
                f'only {torque:.2f} Nm, under the {weak:.2f} Nm a grip takes'
                + ('' if closed is None
                   else f' -- they closed at {closed["torque"]:.2f} Nm')))
            return feel
        feel.update(held=True, source='grip', why=(
            f'the jaws are {width * 1000:.1f} mm apart'
            + ('' if torque is None else f' at {torque:.2f} Nm')
            + ('' if closed is None
               else f', against {closed["width"] * 1000:.1f} mm and '
                    f'{closed["torque"]:.2f} Nm when they closed')))
        return feel

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
        self._held.pop(self.arm, None)
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

    def detect(self, min_count=1, prompt=None):
        """Wait for a detection message newer than now and matching a prompt.

        `prompt` overrides the cycle's own, for asking about a part of the
        object rather than the object -- see locate(). Both the request and
        the match use it, so an answer to the previous question cannot be
        mistaken for an answer to this one.

        Logged with how long the wait was, because a cycle's time is not
        where it looks: LOCATE spanned 10 s and 12.6 s of a measured 76 s run,
        but that window also holds the reachability probes and the pre-flight,
        and nothing in the log separated the detector's own latency from
        them.
        """
        asked = prompt or self.prompt
        self.prompt_pub.publish(String(data=asked))
        started = self.get_clock().now().nanoseconds * 1e-9
        timeout = self.get_parameter('detect_timeout').value
        deadline = started + timeout
        while not self._abort.is_set():
            with self._lock:
                payload = self._latest_detections
            fresh = (payload is not None
                     and payload.get('stamp', 0.0) > started
                     and payload.get('prompt') == asked)
            if fresh and len(payload.get('detections', [])) >= min_count:
                # 'seen', not 'ok': nothing was commanded and nothing
                # moved, so this record carries no `before` and no `path`,
                # and an outcome of 'ok' would put it among the motions that
                # do.
                self.log_motion(
                    'LOCATE', 'detect', 'seen',
                    secs=round(self.get_clock().now().nanoseconds * 1e-9
                               - started, 2),
                    asked=asked,
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

    def locate(self, min_count=1):
        """Find the object, asking for the part worth gripping first.

        The detector reports the centre of whatever it finds, and for some
        objects that is not a place to put the jaws: the centre of a roll
        of tape is the hole. So where a recipe exists, the part is asked
        for first -- "detect edge of the tape" -- and the plain object is
        the fallback.

        Always with that fallback, and never silently: a recipe that turned
        a findable object into an unfindable one would be worse than no
        recipe, so a part prompt that comes back empty costs one detection
        and a log line.

        Remembers which question produced the answer, so the grasp check
        later compares like with like -- an edge that has moved against the
        edge, not against the whole object's centre.
        """
        part = self.part_prompt()
        if part:
            # min_count=0 deliberately: this returns on the first fresh
            # answer whatever is in it, so a part the detector cannot find
            # costs one inference rather than the whole detect_timeout --
            # 15 s of waiting for a second answer that was never coming.
            payload = self.detect(min_count=0, prompt=part)
            if payload is not None and payload.get('detections'):
                self._pick_prompt = part
                return payload
            self.get_logger().info(
                f'"{part}" found nothing, so falling back to '
                f'"{self.prompt}". The recipe is a first try, not a rule.')
        self._pick_prompt = self.prompt
        return self.detect(min_count=min_count)

    def held_object_name(self):
        """What is in the jaws, in the words the operator typed.

        "detect tape" -> "tape". Not the part prompt: a recipe may have
        aimed the grasp at the edge of the tape, but what is being held is
        the tape, and that is what the panel should say.
        """
        words = grasp_recipes.object_words(self.prompt)
        return ' '.join(words) if words else None

    def part_prompt(self):
        """The part-of-the-object prompt for this cycle, or None.

        None when recipes are off, when there is no recipe for what was
        asked for, or when the recipe's own approach is one this cycle
        cannot fly -- a bottle's recipe says to come at the barrel from the
        side, and aiming a *top-down* grasp at the middle of a standing
        bottle is worse than aiming it at the object's own centre. The
        recipe is a pair, where and how, and half of it is not useful.
        """
        if not self.get_parameter('use_grasp_recipes').value:
            return None
        entry = grasp_recipes.recipe_for(self.prompt, self._recipes)
        if entry is None:
            return None
        if entry.get('approach') not in (None, 'top'):
            self.get_logger().info(
                f'the recipe for this object grips it from the '
                f'{entry.get("approach")} -- {entry.get("why", "")} -- and '
                f'the approach and retreat here are a vertical column, so '
                f'that cannot be flown. Asking for the object itself '
                f'instead and coming down on it.',
                throttle_duration_sec=60.0)
            return None
        prompt = grasp_recipes.part_prompt(self.prompt, self._recipes)
        if prompt:
            self.get_logger().info(
                f'{self.prompt} -> {prompt}: {entry.get("why", "")}')
        return prompt

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

    def tcp_orientation(self):
        """How the tool is currently held, as (x, y, z, w).

        For legs that should not change the tool's attitude -- rising
        straight up out of trouble, say, where re-deriving a top-down
        orientation would rotate the wrist at the same time.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as exc:                     # noqa: BLE001 - reported
            self.get_logger().warn(
                f'no {self.tcp_frame} orientation: {exc}')
            return None
        q = tf.transform.rotation
        return (q.x, q.y, q.z, q.w)

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

    def reachable(self, position, quat, arm=None, orientations=None,
                  cheap=False):
        """Can this arm put its tool there? True, False, or None for unknown.

        cheap stops after KDL. Its "yes" is conclusive and costs milliseconds;
        everything after it is there to second-guess a "no", and that is
        where the time goes -- the local solver runs 48 seeded
        damped-least-squares solves and takes 2.9 seconds whether it finds
        anything or not. Multiplied by three orientations, three heights and
        two arms, that was 52 seconds before the robot moved at all, measured
        as an 85-second gap between the first look and the second.

        For choosing *between* two arms an inconclusive answer is not a
        problem: it means "use the half of the frame the object is in", which
        is what arm_candidates already says. The pre-flight then settles
        whether the pick is possible, properly, for whichever arm was
        chosen -- so paying 52 seconds for a worse version of that answer
        beforehand buys nothing.

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

        if cheap:
            # KDL could not confirm it, and confirming is all this mode is
            # for. Unknown, not False.
            return None

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
            wanted = orientations if orientations is not None else \
                self.get_parameter('reach_orientations').value
            for candidate in self.grasp_quat_options(
                    quat, limit=max(1, int(wanted))):
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

        patience = self.get_parameter('planner_probe_timeout').value
        self.get_logger().info(
            f'checking the planner can plan: a plan_only goal to the arm\'s '
            f'own posture, up to {patience:.0f} s')
        handle = self._await(self.move_client.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            return None
        result = self._await(handle.get_result_async(), patience)
        if result is None:
            self.get_logger().error(
                f'move_group accepted a plan_only goal and then returned no '
                f'result within {patience:.0f} s. Nothing was asked to move, '
                f'so this is move_group itself rather than the planner or '
                f'the target -- check whether cuMotion logged a successful '
                f'plan for this query while move_group logged nothing, '
                f'which is what a wedged move_group looks like from here.')
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

    def robot_in_map(self):
        """Which of the robot's own links the octomap is colliding with.

        Empty when the map is clean, which is the answer the self-filter is
        supposed to produce. A non-empty list means the capture caught the
        robot: the arm's start state is then invalid and *nothing* plans
        from it, which presents as "the planner is not planning" on every
        goal until the map is cleared.

        Asked of both arms, because the boot walk parks both of them in
        frame and either one can poison the scene for the other.
        """
        if not self.validity_client.wait_for_service(timeout_sec=2.0):
            return []
        hits = set()
        for group in (f'{self.arm}_arm', f'{other_arm(self.arm)}_arm'):
            request = GetStateValidity.Request()
            request.group_name = group
            request.robot_state.is_diff = True
            result = self._await(
                self.validity_client.call_async(request), 5.0)
            if result is None or result.valid:
                continue
            for contact in result.contacts:
                pair = (contact.contact_body_1, contact.contact_body_2)
                if any('octomap' in name for name in pair):
                    hits.add('/'.join(pair))
        return sorted(hits)

    def octomap_voxels(self):
        """How much is in the planning scene's octomap, or None.

        Asked because "the map was refreshed" has never meant "the map has
        anything in it". The refresh call talks to octomap_gater, which
        lets depth frames through to move_group -- and if no octomap
        updater plugin is loaded there, the frames go nowhere and the scene
        the planner uses stays empty. That is not a hypothetical: the
        plugin sensors_3d.yaml names lives in moveit_ros_perception, which
        is a separate apt package, and a stack without it plans against
        nothing while every log line says the map was captured.
        """
        if not self.scene_query_client.wait_for_service(timeout_sec=2.0):
            return None
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.OCTOMAP
        result = self._await(self.scene_query_client.call_async(request), 5.0)
        if result is None:
            return None
        try:
            return len(result.scene.world.octomap.octomap.data)
        except AttributeError:
            return None

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

        With gripper_never_in_map the exemption is permanent and a request
        to withdraw it is ignored: the gripper stands in the camera's view
        at the staging pose, so it lands in every map captured from there,
        and the whole of it being an obstacle is what stops the robot
        planning at all. Only the three gripper links are exempt; the
        forearm and the upper arm are checked against the map exactly as
        before, so the arm still cannot sweep the table.
        """
        if not allow and self.get_parameter('gripper_never_in_map').value:
            return True
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

    def posture_is_gettable(self, entry, label, index):
        """Can the arm actually get into this posture? True unless told no.

        The pre-flight proves the column *from* a posture with an explicit
        start state, which says nothing about reaching the posture in the
        first place. TRANSIT is the joint goal that does that, and it is a
        free-space plan through the octomap -- a different question with a
        different answer. Measured, run 1788945589: a candidate passed the
        column check and TRANSIT then returned -2, after the arm had already
        driven to the staging pose.

        Asked with a plan-only goal, so nothing moves. Unknown counts as yes:
        a planner that will not answer is not evidence the posture is
        unreachable, and refusing on it would turn a stack that is merely
        busy into a workspace problem.
        """
        if not self.get_parameter('preflight_posture_reachable').value:
            return True
        joints = list(entry['joints'])
        if entry.get('tilt') is not None:
            joints[6] = entry['tilt']
        verdict = self.planner_can_hold(joints, f'{label} candidate {index + 1}')
        if verdict is False:
            self.get_logger().info(
                f'{label}: candidate {index + 1} proves the column, but the '
                f'arm cannot be planned into the posture it proves it from '
                f'-- which is what TRANSIT would have discovered after '
                f'driving to the staging pose. Trying the next.')
            self.log_motion(label, 'preflight', 'posture-unreachable',
                            target=[round(v, 5) for v in joints],
                            candidate=index + 1)
            return False
        return True

    def planner_can_hold(self, joints, label):
        """Plan-only joint goal: can the planner put the arm here? T/F/None.

        The same shape of question planner_is_planning asks, aimed at a
        posture the arm is not in rather than the one it is. Nothing moves.
        """
        if not self.move_client.wait_for_server(timeout_sec=2.0):
            return None
        request = self._base_request()
        constraints = Constraints()
        for name, value in zip(self.arm_joints, joints):
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
            return None
        code = result.result.error_code.val
        if code == MOVEIT_SUCCESS:
            return True
        self.get_logger().info(f'{label}: plan-only came back {code}')
        return False

    def probe_cartesian(self, start_joints, position, quat,
                        avoid_collisions=True):
        """Would a straight line to `position` solve from `start_joints`?

        Returns (fraction, end_joints, travel) -- or (None, None, None) if
        the service could not answer at all; note that a fraction of 0.0 is a
        real answer and None is not.

        travel is the worst per-joint distance the trajectory would cost,
        which is the other half of the question. A line can solve 100% and
        still be unflyable: /compute_cartesian_path constrains the *tool*, so
        it will happily return a path whose every waypoint is on the line
        while the shoulder and elbow sweep across the workspace. Measured,
        run 1788868354: the descent solved 100% checked and unchecked and
        cost 3.07 rad on one joint against a 1.5 rad budget, so the guard in
        cartesian_move refused it -- after the arm had already flown to
        TRANSIT for nothing, because the pre-flight had only ever looked at
        the fraction.

        This is the whole point of the pre-flight. Joint headroom says a
        posture is comfortable; it does not say a line out of it solves. The
        run that prompted this had joint1 at -1.356 with 0.040 rad left before
        its stop *before the descent started*, and got 18% of the line: the
        posture was fine, the direction it then had to travel was not. Asking
        the same question of /compute_cartesian_path with an explicit start
        state costs one service call and no motion at all.
        """
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            return None, None, None
        request = self._cartesian_request(position, quat, avoid_collisions,
                                          start_joints=start_joints)
        result = self._await(self.cartesian_client.call_async(request), 20.0)
        if result is None or result.error_code.val != MOVEIT_SUCCESS:
            return (0.0, None, None) if result is not None \
                else (None, None, None)
        points = result.solution.joint_trajectory.points
        names = list(result.solution.joint_trajectory.joint_names)
        end = None
        if points:
            index = {n: i for i, n in enumerate(names)}
            if all(j in index for j in self.arm_joints):
                end = [points[-1].positions[index[j]] for j in self.arm_joints]
        return (float(result.fraction), end,
                self.joint_travel(result.solution))

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
                self._column_refusal = (
                    f'the line to it solves completely, but flying it would '
                    f'cost {travel:.2f} rad on one joint against a '
                    f'{max_travel:.2f} rad budget -- the tool would trace the '
                    f'line while the arm swings across the workspace, which '
                    f'is how it hit the table before. This is geometry, not '
                    f'reach: no retry from the same posture improves it, and '
                    f'the pre-flight should have rejected the posture')
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
                        # What the re-timing did to it, so a leg that ran
                        # fast or slow can be attributed afterwards rather
                        # than guessed at. See _retime.
                        boost=self._last_boost,
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

    def trajectory_peaks(self, trajectory):
        """The fastest and hardest this trajectory asks any joint to work.

        Returns (rad/s, rad/s^2) from the commanded path itself -- its own
        positions and times, differenced twice. Not from the velocity and
        acceleration fields, which are not always filled in, and not from
        measured joint states, where differentiating 10 Hz position noise
        twice invents accelerations that were never commanded.

        (None, None) when there is not enough of a path to say.
        """
        points = trajectory.joint_trajectory.points
        if len(points) < 3:
            return None, None

        def when(point):
            return (point.time_from_start.sec
                    + point.time_from_start.nanosec * 1e-9)

        speeds = []
        for first, second in zip(points, points[1:]):
            span = when(second) - when(first)
            if span <= 0.0:
                continue
            speeds.append(([(b - a) / span
                            for a, b in zip(first.positions, second.positions)],
                           (when(first) + when(second)) / 2.0))
        if len(speeds) < 2:
            return None, None
        fastest = max(max(abs(v) for v in step[0]) for step in speeds)
        hardest = 0.0
        for (before, at), (after, then) in zip(speeds, speeds[1:]):
            span = then - at
            if span <= 0.0:
                continue
            hardest = max(hardest, max(abs(b - a) / span
                                       for a, b in zip(before, after)))
        return fastest, hardest

    def _retime(self, trajectory):
        """Scale a trajectory in time, and take the slack out of it.

        A pure time scaling: t/s, v*s, a*s^2.

        Two things are happening here, and they pull opposite ways.

        The first is velocity_scaling, the operator's speed, which only ever
        slows the path down -- and which the at_speed blocks turn right down
        for the legs that set something on a surface.

        The second is a correction, and it is why this can now also speed a
        path *up*. /compute_cartesian_path has no scaling field in this
        MoveIt build, so the timing it returns is whatever its own
        parameterisation produced -- and joint_limits.yaml declares
        `has_acceleration_limits: false` for every joint on this robot, so
        that parameterisation has no acceleration limit to work with and
        falls back to something very conservative. The result is measurable
        and lopsided: in run 1789640514 the four TRANSIT legs took 11.2 to
        11.6 s each and peaked at 0.8 rad/s^2, while the joint goals in the
        same run -- planned by cuMotion, which uses its own limits -- peaked
        at 3.4 to 5.4 rad/s^2 and finished in under 3.5 s. Same arm, same
        run, a fifth of the acceleration.

        So the path is sped up until it reaches a ceiling this file sets
        rather than one nobody chose, with the ceilings deliberately below
        what the joint goals already demonstrate on this hardware. The
        operator's scaling still applies on top, so a gentle leg stays
        gentle -- it is a gentle version of a properly timed path instead of
        a gentle version of a crawling one.
        """
        speed = max(1e-3, min(1.0, self.get_parameter('velocity_scaling').value))
        boost = 1.0
        ceiling = max(1.0, self.get_parameter('cartesian_boost_max').value)
        if ceiling > 1.0 and not self._gentle:
            fastest, hardest = self.trajectory_peaks(trajectory)
            room = []
            if fastest:
                room.append(self.get_parameter('cartesian_vel_max').value
                            / fastest)
            if hardest:
                room.append(math.sqrt(
                    self.get_parameter('cartesian_accel_max').value / hardest))
            if room:
                boost = max(1.0, min(ceiling, min(room)))
        scale = speed * boost
        for point in trajectory.joint_trajectory.points:
            seconds = (point.time_from_start.sec
                       + point.time_from_start.nanosec * 1e-9) / scale
            point.time_from_start.sec = int(seconds)
            point.time_from_start.nanosec = int((seconds % 1.0) * 1e9)
            point.velocities = [v * scale for v in point.velocities]
            point.accelerations = [a * scale * scale for a in point.accelerations]
        self._last_boost = round(boost, 3)
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

        # Watch the motors while it flies. Nothing else in the cycle notices
        # the arm leaning on something: the controller has no goal tolerance
        # and reports SUCCESS whatever it hit, and the collision world only
        # knows what the octomap saw.
        watch = self.watch_torque(label, handle, trajectory)
        try:
            result = self._await(handle.get_result_async(),
                                 self.get_parameter('motion_timeout').value)
        finally:
            hit = watch()
        if hit is not None:
            self.retreat_from_contact(label, hit)
            return False
        if result is None:
            self.get_logger().error(f'{label}: trajectory timed out')
            return False
        code = result.result.error_code.val
        if code != MOVEIT_SUCCESS:
            self.get_logger().error(f'{label}: execution failed, code {code}')
            return False
        return True

    def trajectory_plan(self, trajectory):
        """A trajectory as [(seconds, joints in this arm's order)].

        None when it is not this arm's trajectory, which drops the contact
        guard's motion test rather than comparing against the wrong joints.
        """
        if trajectory is None:
            return None
        names = list(trajectory.joint_trajectory.joint_names)
        try:
            order = [names.index(joint) for joint in self.arm_joints]
        except ValueError:
            return None
        plan = []
        for point in trajectory.joint_trajectory.points:
            if len(point.positions) < len(names):
                continue
            seconds = (point.time_from_start.sec
                       + point.time_from_start.nanosec * 1e-9)
            plan.append((seconds, [point.positions[i] for i in order]))
        return plan or None

    def watch_torque(self, label, handle, trajectory=None):
        """Stop the move if the arm starts leaning on something.

        Returns a stop function, which hands back what tripped -- including
        the postures the arm held over the last contact_rewind_seconds, so
        the caller can retrace them instead of planning a way out.

        The rule lives in ContactMonitor; this is the thread that feeds it
        and the cancel that acts on it. The cancel goes out from *here*, the
        moment the verdict comes back. It used to be sent by the stop
        function, which does not run until the wait for the result returns
        -- so the arm finished the move it was supposed to be stopped in the
        middle of, every time, and the first anyone knew of the "contact"
        was a completed motion being reported as a crash.

        On fake hardware there are no efforts at all, so this never fires.
        """
        margin = self.get_parameter('contact_torque_margin').value
        if margin <= 0.0:
            return lambda: None

        with self._lock:
            reported = {joint: self._arm_efforts.get(joint)
                        for joint in self.arm_joints}
        if not any(v is not None for v in reported.values()):
            return lambda: None            # nothing reports effort here

        plan = self.trajectory_plan(trajectory)
        if not plan:
            # Nothing to measure against, so nothing to measure. Said out
            # loud because a guard that is quietly not guarding is worse
            # than one that is not there.
            self.get_logger().warn(
                f'{label}: no trajectory to compare against, so the contact '
                f'guard is not watching this move',
                throttle_duration_sec=30.0)
            return lambda: None

        monitor = ContactMonitor(
            self.arm_joints,
            margin=margin,
            window=self.get_parameter('contact_torque_window').value,
            hold=self.get_parameter('contact_torque_hold').value,
            rewind=self.get_parameter('contact_rewind_seconds').value,
            lag_limit=self.get_parameter('contact_lag_rad').value,
            plan=plan)

        stop = threading.Event()
        tripped = {}

        def poll():
            while not stop.is_set():
                now = self.get_clock().now().nanoseconds * 1e-9
                with self._lock:
                    positions = [self._arm_positions.get(joint)
                                 for joint in self.arm_joints]
                    efforts = {joint: self._arm_efforts.get(joint)
                               for joint in self.arm_joints}
                verdict = monitor.add(now, positions, efforts)
                if verdict is not None:
                    tripped.update(verdict)
                    stop.set()
                    self.cancel_move(handle, label)
                    return
                stop.wait(0.02)

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()

        def finish():
            was_set = stop.is_set()
            stop.set()
            thread.join(timeout=2.0)
            if not (was_set and tripped):
                return None
            lag = tripped.get('lag')
            # Which of the two motion tests caught it, in words: they mean
            # different things. 'lag' is a long move stopped early, 'stall'
            # is an arm going nowhere while its plan carries on -- which is
            # what a descent onto a table looks like, where there is never
            # enough plan left to build up a lag.
            how = ('fell '
                   + ('behind its trajectory' if lag is None
                      else f'{lag:.3f} rad behind its trajectory')
                   if tripped.get('why') == 'lag' else
                   'stopped moving while its trajectory carried on')
            self.get_logger().error(
                f'{label}: {tripped["joint"]} went from '
                f'{tripped["baseline"]:.2f} to {tripped["effort"]:.2f} Nm '
                f'while the arm {how}. That is the arm pushing on something '
                f'rather than carrying itself, so the move was cancelled.')
            self.log_motion(label, 'joint', 'contact',
                            joint=tripped['joint'],
                            torque=round(tripped['effort'], 3),
                            baseline=round(tripped['baseline'], 3),
                            why=tripped.get('why'),
                            lag_rad=None if lag is None else round(lag, 4))
            return tripped

        return finish

    def cancel_move(self, handle, label):
        """Ask for a running trajectory to stop. Never raises."""
        try:
            handle.cancel_goal_async()
        except Exception as exc:                     # noqa: BLE001
            self.get_logger().warn(f'{label}: could not cancel the move: {exc}')

    def retreat_from_contact(self, label, hit):
        """Retrace the last contact_rewind_seconds, without planning.

        Driven straight at the joint trajectory controller. Two reasons it
        does not ask a planner:

          - the arm is leaning on something, so its own start state is very
            likely in collision and every plan from it comes back -2;
          - move_group is as often as not the thing being escaped from.

        Measured, run 1789012345: the back-off asked cuMotion, which planned
        it; move_group rejected the path for the gripper being inside the
        octomap; the resend was planned again and never answered. Two
        60-second timeouts later /move_action was no longer in the graph at
        all, /check_state_validity had stopped being served, and neither
        refuge could be checked or reached. The arm sat over the table for
        two minutes because the way out went through the thing that had
        jammed.

        The postures being retraced were measured on the way in, seconds
        ago, so this path is known good without asking anyone.

        _contact_fired is set here and cleared again below on a successful
        back-off, not left set for the rest of the cycle. It exists to tell
        safe_shutdown, at the very end, whether the arm is *still* resting
        on what it hit -- and once the back-off has worked that premise is
        gone. Measured, run at 17:45: TRANSIT tripped contact, backed off
        cleanly at 36.6 s, and the cycle went on to fly four more legs
        without incident -- yet a later, unrelated DESCEND failure still
        ran the full disengage/settle/re-engage because the flag from 18 s
        earlier was still set, and the arm sagged 32 cm in the 3 s it spent
        with no torque at all before either side could plan back.
        """
        self._contact_fired = True
        retrace = [list(p) for p in (hit.get('retrace') or [])]
        if not retrace:
            self.get_logger().error(
                f'{label}: stopped on contact, but no earlier posture was '
                f'recorded to back off to. The arm is holding where it '
                f'stopped.')
            self.log_motion(label, 'joint', 'stuck-on-contact')
            # No retrace, so nothing was done about the contact: this is
            # the one case where the arm may genuinely still be resting on
            # what it hit, and the flag stays set for safe_shutdown.
            return False

        self._set_state('CONTACT', f'{hit["joint"]} loaded up during {label}')
        rewind = self.get_parameter('contact_rewind_seconds').value
        self.get_logger().warn(
            f'backing off along the way it came, to where the arm was '
            f'{rewind:.1f} s before the contact')
        # Newest first: out of the contact, then back along the path.
        path = thin(list(reversed(retrace)), 12)
        ok = self.drive_joint_path(path, f'{label} back-off')
        if not ok:
            # The controller would not take it. A plan is a poor second --
            # see above -- but standing on the obstruction is worse.
            ok = self._move_to_joints(path[-1], f'{label} back-off',
                                      skip_if_there=True)
        self.log_motion(label, 'joint',
                        'backed-off' if ok else 'stuck-on-contact',
                        target=[round(v, 5) for v in path[-1]],
                        waypoints=len(path))
        if ok:
            # Off whatever it hit. safe_shutdown's disengage-on-contact
            # path is for an arm that is still leaning on something; this
            # one no longer is, and treating it as though it still were is
            # what turned one clean recovery into an unplanned free-fall.
            self._contact_fired = False
        return ok

    def drive_joint_path(self, path, label, speed=None):
        """Send a joint path straight to the controller. No planning.

        The one place in the cycle that moves the arm without a collision
        check, and deliberately so: it exists for the case where the planner
        cannot be reached or cannot be trusted, and it is only safe because
        every waypoint is a posture the arm measured itself in moments ago.
        Do not use it for anywhere the arm has not just been.
        """
        client = getattr(self, 'traj_client', None)
        if client is None or not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f'{label}: {self.arm}_joint_trajectory_controller is not '
                f'taking trajectories, so the arm cannot be driven back '
                f'without a planner')
            return False
        if speed is None:
            speed = self.get_parameter('contact_retreat_speed').value
        speed = max(0.05, speed)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.arm_joints)
        # Left at zero: the controller starts from wherever the arm is and
        # interpolates to the first waypoint over its time_from_start, which
        # is what makes it safe to send a path beginning next to the arm
        # rather than exactly on it.
        previous = self.measured_joints()
        elapsed = 0.0
        for target in path:
            step = (max(abs(a - b) for a, b in zip(target, previous))
                    if previous and len(previous) == len(target) else 0.5)
            elapsed += max(0.15, step / speed)
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in target]
            point.time_from_start.sec = int(elapsed)
            point.time_from_start.nanosec = int((elapsed % 1.0) * 1e9)
            goal.trajectory.points.append(point)
            previous = list(target)

        handle = self._await(client.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: the controller refused it')
            return False
        result = self._await(handle.get_result_async(), elapsed + 10.0)
        if result is None:
            self.get_logger().error(f'{label}: the controller never answered')
            return False
        code = result.result.error_code
        if code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info(
                f'{label}: retraced {len(path)} posture(s) in about '
                f'{elapsed:.1f} s')
            return True
        # A tolerance violation still moved the arm, and moved is the point.
        # What matters is where it ended up, not what the controller made of
        # the journey.
        worst = self.joint_error(path[-1], fresh=True)
        settled = self.get_parameter('at_goal_tolerance').value * 3.0
        if worst is not None and worst <= settled:
            self.get_logger().warn(
                f'{label}: the controller reported {code}, but the arm is '
                f'{worst:.4f} rad from where it was asked to go, so it went')
            return True
        self.get_logger().error(
            f'{label}: the controller reported {code} and the arm is '
            f'{"unknown" if worst is None else f"{worst:.4f} rad"} from the '
            f'posture it was sent to')
        return False

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
        self._column_refusal = None
        try:
            return self._descend_column(xy, from_z, to_z, quat, label,
                                        uncheck_collisions, linear_only)
        finally:
            # Withdraw it only once the arm is off the column.
            #
            # This used to withdraw per leg, and that is what turned the
            # gripper red in RViz. The descent leaves the jaws inside the
            # voxels of the object they came for; hand them back to collision
            # checking there and the *start state* is in collision, so every
            # plan from it is invalid. Measured, run 1788863209: CLEAR
            # succeeded -- it re-granted its own exemption -- and then
            # PRE_PICK_STATE and HOME both came back -2,
            # INVALID_MOTION_PLAN, with the arm stranded over the table.
            #
            # self._column is set for exactly the span that matters: from the
            # moment the descent starts to the moment the tool is clear
            # again. release_column() is what ends it.
            if self._octomap_exempt and self._column is None:
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

        # The robot looking at itself. Only possible because a cycle may
        # now look from the staging pose, which is over the work surface
        # and therefore in the frame -- and the detector always answers
        # something, so "the gripper" is exactly the kind of answer it
        # gives when the object it was asked for is behind the arm.
        # Reaching for the tool's own position would drive the gripper
        # into itself.
        near = self.get_parameter('self_detect_radius').value
        tool = self.tcp_position()
        if near > 0.0 and tool is not None and dist(point, tool) < near:
            return (f'the detection is {dist(point, tool) * 1000:.0f} mm '
                    f'from the tool, inside the {near * 1000:.0f} mm that '
                    f'means the camera is looking at the gripper rather '
                    f'than at anything on the table. Refusing it -- a '
                    f'later attempt goes home and looks from out of frame.')
        return None

    def object_top(self, detection):
        """The z a top-down grasp has to stop above, and why.

        `point` is the median depth over the inner half of the box, which
        is the right number for "where is it" and the wrong one for "how
        high is it": for anything that stands up it lands somewhere down
        the side of the object, and a grasp computed from it is aimed
        *inside* the object. The detector reports the nearest surface in
        the box separately for exactly this, so use it where it is there
        and say so where it is not.
        """
        top = detection.get('top')
        if top and len(top) == 3 and math.isfinite(top[2]):
            return float(top[2]), 'the measured top of the object'
        self.get_logger().warn(
            'the detector did not report an object top, so the grasp is '
            'computed from the box-centre depth instead. That is the median '
            'over the inner half of the box and it sits below the top of '
            'anything that stands up -- restart the detector to get one.',
            throttle_duration_sec=60.0)
        return float(detection['point'][2]), 'the box-centre depth'

    def grasp_from_detection(self, detection, strategy):
        point = detection['point']
        axis_yaw = detection.get('axis_yaw')
        if axis_yaw is None:
            axis_yaw = 0.0
            self.get_logger().warn('no object axis from the VLM; assuming yaw 0')
        yaw = axis_yaw + strategy['yaw_offset']

        top_z, top_from = self.object_top(detection)
        grasp_z = top_z + self.get_parameter('grasp_z_offset').value \
            + strategy['z_offset']
        tall = detection.get('height_m')
        if tall is not None:
            self.get_logger().info(
                f'the object stands {tall * 1000:.0f} mm above what it is on, '
                f'top at z={top_z:.3f} ({top_from})')
            limit = self.get_parameter('tall_object_height').value
            if limit > 0.0 and tall > limit:
                # Said rather than refused: a top grasp on a tall object is
                # a grasp on its cap or its shoulder, and whether that
                # holds depends on the object. What it is not is a grasp
                # around the barrel, which is what a bottle needs and what
                # a vertical approach cannot fly.
                self.get_logger().warn(
                    f'that is taller than tall_object_height '
                    f'({limit * 1000:.0f} mm). A top-down grasp closes on '
                    f'the top face of it, not around it -- for anything '
                    f'that wants gripping around the barrel the approach '
                    f'has to come in horizontally, and the approach is a '
                    f'vertical column. Expect the jaws to meet the top of '
                    f'it rather than its sides.')
        # How low the jaws may go, and what decides it.
        #
        # The table, where the detector measured one: it reports the depth
        # of the ring around the object, which is the surface right there
        # rather than the surface on average, and staying a few millimetres
        # above that is the whole of "do not press into the table". That is
        # a better floor than the object's own top face, which is what this
        # used and which meant the jaws could never reach down the side of
        # anything -- a grasp at the top face of a roll of tape is a grasp
        # on 10 mm of air, and the grip that results is the one that lets
        # go. A negative grasp_z_offset now reaches down the object.
        #
        # Without a surface reading it falls back to the old rule, which
        # cannot go below the top at all.
        surface = detection.get('surface')
        clearance = self.get_parameter('grasp_table_clearance').value
        if surface and len(surface) == 3 and math.isfinite(surface[2]):
            floor = surface[2] + clearance
            floor_from = (f'the surface at z={surface[2]:.3f} plus '
                          f'{clearance * 1000:.0f} mm')
        else:
            floor = top_z - self.get_parameter('grasp_max_depth').value
            floor_from = (f'the object top (no surface reading -- '
                          f'grasp_max_depth {self.get_parameter("grasp_max_depth").value:.3f})')
        if grasp_z < floor:
            self.get_logger().warn(
                f'grasp z {grasp_z:.3f} is {(floor - grasp_z) * 1000:.0f} mm '
                f'below what {floor_from} allows; clamping to {floor:.3f}.')
            grasp_z = floor
        else:
            self.get_logger().info(
                f'grasp z {grasp_z:.3f}, {(grasp_z - floor) * 1000:.0f} mm '
                f'above the floor set by {floor_from}')
        min_z = self.get_parameter('min_grasp_z').value
        if grasp_z < min_z:
            self.get_logger().warn(
                f'grasp z {grasp_z:.3f} below min_grasp_z {min_z:.3f}; clamping')
            grasp_z = min_z

        # The surface the object is standing on, kept for the place: the
        # height the tool ends up at above *this* is the height it has to
        # be above the paper for the object to be set down rather than
        # dropped. Provisional here, from the commanded grasp; _step_grasp
        # replaces it with where the tool actually was when the jaws
        # closed, which on this arm is 10-25 mm from what was commanded.
        if surface and len(surface) == 3 and math.isfinite(surface[2]):
            self._pick_surface_z = float(surface[2])
            self._pick_hold_offset = float(grasp_z - surface[2])
            self.get_logger().info(
                f'the object stands on z={surface[2]:.3f} and the grasp is '
                f'commanded {self._pick_hold_offset * 1000:.0f} mm '
                f'above it')
        else:
            self._pick_surface_z = None
            self._pick_hold_offset = None

        grasp = (point[0], point[1], grasp_z)
        pregrasp = (point[0], point[1],
                    grasp_z + self.get_parameter('approach_height').value)
        return grasp, pregrasp, top_down_quat(yaw), point

    # -- verification --------------------------------------------------------

    def verify_grasp(self, pick_point):
        """Did the object actually leave the spot it was picked from?

        Three sources, and each is only trusted where it can actually tell:

          seen at the pick point   it never left. Certain failure.
          seen away from it        it came along. Certain success.
          seen nowhere             says nothing either way, so the fingers
                                   decide.

        The middle case is the one this keeps getting wrong in both
        directions, so here is what each direction cost.

        Reading "seen nowhere" as success let run 1789014831 cycle 1
        "place" nothing and report DONE: the jaws had closed 15 mm off
        target and stalled at 1.08 mm with 0.50 Nm -- empty by any reading
        -- and the detector, which found nothing, was taken as agreeing.

        Reading it as failure then threw away a good pick. Run 1789022566:
        the jaws closed on a screwdriver handle at 14.0 mm and 2.28 Nm,
        the detector saw nothing twice, and the re-grasp opened them and
        dropped it. Of course it saw nothing -- the object was in the jaws,
        the tool was 38 mm above the point it came from, and the prompt was
        asking for the handle, which is the part inside the gripper.

        So an empty frame decides nothing and the gripper is asked
        instead -- see finger_verdict, which reads how far apart the jaws
        are, what they are pressing with, and whether either has changed
        since the close. What none of that can tell is *what* is held -- a
        jaw shut on a cable reads like a jaw shut on the object -- which is
        why it is the tie-break and not the test.

        Only the width ends the check here on its own. A jaw at 1.1 mm is
        empty and there is nothing to discuss; a jaw the right width but
        slack, or narrower than it closed, is a strong hint and not a
        proof, so it goes to the detector first and only decides if the
        detector cannot. That ordering is deliberate -- believing the
        fingers over a clear sighting is how a good pick got thrown away.

        The other thing that run showed: the position test only means
        something if the arm actually carried the object somewhere. Its
        lift was refused for joint travel and fell back to 38 mm, under the
        50 mm that separates "moved" from "did not", so the test could not
        have distinguished the two whatever the detector said. When that
        happens it is not run.
        """
        feel = self.finger_verdict()
        if feel['held'] is None:
            self.get_logger().error(f'the gripper cannot say: {feel["why"]}.')
        else:
            self.get_logger().info(
                f'the gripper says it is '
                f'{"holding something" if feel["held"] else "empty"}: '
                f'{feel["why"]}.')
        if feel['held'] is False and feel['source'] == 'width':
            return False

        # Has the arm carried it far enough for "did it move" to mean
        # anything? Below object_moved_eps, an object in the jaws and an
        # object still on the table are the same point.
        eps = self.get_parameter('object_moved_eps').value
        tcp = self.tcp_position()
        if tcp is not None and dist(tcp, pick_point) <= eps:
            return self.grasp_by_feel(feel, (
                f'the tool is only {dist(tcp, pick_point) * 1000:.0f} mm '
                f'from where it picked, and {eps * 1000:.0f} mm is what '
                f'separates "moved" from "did not" -- usually a lift that '
                f'was refused and fell back short'))

        # The object, not the part. Where a recipe aimed the grasp at a
        # part, that part is precisely the bit now inside the jaws.
        tries = max(1, int(self.get_parameter('verify_detect_tries').value))
        detections, answered = [], False
        for attempt in range(1, tries + 1):
            payload = self.detect(min_count=0, prompt=self.prompt)
            if payload is None:
                self.get_logger().warn(
                    f'the detector did not answer the grasp check '
                    f'({attempt} of {tries})')
                continue
            answered = True
            detections = payload.get('detections') or []
            if detections:
                break
            self.get_logger().warn(
                f'the detector sees nothing at all ({attempt} of {tries})')

        left_behind = [d for d in detections
                       if dist(d['point'], pick_point) < eps]
        if left_behind:
            self.get_logger().warn(
                f'object still at the pick point '
                f'({dist(left_behind[0]["point"], pick_point):.3f} m) '
                f'-- grasp failed')
            return False

        if not detections:
            return self.grasp_by_feel(feel, (
                'the detector ' + ('sees the object nowhere' if answered
                                   else 'never answered')
                + f' after {tries} attempt(s)'))

        nearest = None
        if tcp is not None:
            radius = self.get_parameter('object_hold_radius').value
            nearest = min(dist(d['point'], tcp) for d in detections)
            if nearest < radius:
                self.get_logger().info(
                    f'object {nearest:.3f} m from the tool: held')
                if feel['held'] is False:
                    # Seeing it up at the tool after the lift is the stronger
                    # evidence and it wins, but the two readings cannot both
                    # be right and the one being overruled is worth having in
                    # the log: either the object is lying on the table under
                    # the gripper, or the gripper is reading its own grip
                    # wrong and grasp_hold_torque_min wants moving.
                    self.get_logger().warn(
                        f'-- the gripper disagrees: {feel["why"]}. If this '
                        f'cycle places nothing, that is the reading that '
                        f'was right')
                return True
        self.get_logger().info(
            f'the nearest of {len(detections)} detection(s) is '
            + ('unknown' if nearest is None else f'{nearest:.3f} m')
            + f' from the tool and none is within {eps:.3f} m of the pick '
            f'point, so the object is not where it was: held')
        return True

    def grasp_by_feel(self, feel, why):
        """Decide on the gripper, when looking cannot decide.

        `feel` is a finger_verdict. Says out loud which reading it went on
        every time: a grasp confirmed this way has not been seen to have
        moved, it is held on the evidence that the jaws are the right
        distance apart and pressing, and that is weaker than watching the
        object leave.
        """
        if feel['held']:
            self.get_logger().warn(
                f'{why}, so this grasp is confirmed on the gripper alone -- '
                f'{feel["why"]}. Not seen to have moved, which is weaker '
                f'than it sounds: the jaws cannot tell the object from '
                f'anything else of that size.')
            return True
        self.get_logger().error(
            f'{why}, and the gripper says '
            f'{feel["why"] or "nothing either"}, so there is nothing left '
            f'to confirm this grasp with. Treating it as failed.')
        return False

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
                self.safe_shutdown(self.load_states(),
                                   'after the cycle crashed')
            except Exception as lift_exc:
                self.get_logger().error(
                    f'could not park the arm after the crash: {lift_exc}')
        finally:
            with self._lock:
                self._busy = False
            # Say so. Every status the UI has is published from _set_state,
            # and the terminal state -- DONE, FAILED, ABORTED -- is set
            # *before* this runs, so the last thing the browser heard was
            # busy=true and its Pick button stayed disabled for ever. One
            # more publish, after the flag is actually down.
            self.publish_status()

    def _cycle(self):
        self._failures = []
        self.release_column()
        # Only this cycle's own working values. What the *other* arm is
        # holding is not this cycle's to forget.
        self._pick_hold_offset = None
        self._pick_surface_z = None
        self._contact_fired = False
        # The model's candidates belong to the cycle that asked for them.
        # Dropping them here is what makes grasp_model_max_age:=0 -- no
        # limit -- a safe default: the bound is this cycle rather than the
        # clock, and nothing from the last object can be used on this one.
        with self._lock:
            self._grasps = None
        self._forget_grasp()
        self._last_detection = None
        self._last_payload = None
        self._pick_prompt = None
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
        view = self.take_up_the_view(states)
        if view is NOTHING_SEEN:
            self._set_state(
                'FAILED',
                f'nothing matched "{self.prompt}" -- check /vlm/debug_image, '
                f'and that the detector has finished loading')
            return
        if view is NO_DETECTOR:
            # Not a pose problem, and it used to be reported as one: the
            # panel said "could not reach home -- joint4 is 1.979 rad off"
            # when the arm was standing perfectly well at the staging pose
            # and the actual trouble was that nothing was publishing
            # /vlm/detections. 1.979 rad is simply how far pre_pick is from
            # home; it was never going there.
            # "Start it" is only half the advice, and often the wrong
            # half: pick_place.launch.py starts the detector on every
            # bringup, so a silent topic usually means it started and
            # died. Measured at 12:41 -- it had been exiting within
            # seconds of every launch on a 401 from HuggingFace, because
            # paligemma-3b-pt-224 is a gated model and the local cache had
            # been emptied. Four runs were spent looking at the robot.
            self._set_state(
                'FAILED',
                'the detector is not answering -- nothing is publishing '
                '/vlm/detections. The bringup starts it, so it has most '
                'likely started and exited: check its output for a model '
                'load failure (paligemma-3b-pt-224 is gated and needs a '
                'HuggingFace login). Otherwise start it with '
                'VLM/run_vlm_detector.sh.')
            return
        if not view:
            self._set_state(
                'FAILED',
                f'could not reach the pose to look from -- '
                f'{self._worst_joint(self.observation_positions())}')
            return

        if self.arm_selection == 'by_side':
            chosen_from = self.arm
            states = self.choose_arm(states)
            if states is None or states is OUT_OF_REACH:
                return
            # Only if it actually switched, and only if the arm that has
            # just been handed the job is not already standing where the
            # approach starts.
            #
            # Both arms wait at the staging pose, so after a switch the new
            # arm is usually already there -- and a fold down to HOME and
            # back from there is the round trip this cycle exists to avoid.
            # Re-homing regardless would also clear and rebuild the octomap
            # a second time every cycle for nothing.
            if self.arm != chosen_from and not self.at_state(
                    PRE_PICK_STATE, states):
                if not self.arrive_at_home():
                    self._set_state(
                        'FAILED',
                        f'could not reach the {self.arm} home -- '
                        f'{self._worst_joint(self.home_positions())}')
                    return
            elif self.arm != chosen_from:
                self.get_logger().info(
                    f'the {self.arm} arm is already at its staging pose, so '
                    f'the switch costs no motion at all')

        # Again, because the arm may have changed since the first check and
        # cuMotion accepts Cartesian goals for one link per bringup.
        if not self.check_planner_tool_frame():
            self._set_state('FAILED', 'the planner is unusable for this arm')
            return

        picked_point = None
        # The (yaw, height) pairs whose jaws already shut on nothing. A grip
        # that missed is a geometry problem, and a rung offering the same yaw
        # at the same height will miss the same way -- for the ~60 s a full
        # approach costs. Measured, run 1788869179 cycle 2: 'nominal' closed
        # 11 mm above a screwdriver and 'retry' is nominal again, so it would
        # have repeated it exactly.
        #
        # Only exact repeats are skipped. A different yaw is a different grasp
        # on a screwdriver, and a rung that re-detects gets a fresh height
        # estimate, so both are still worth their turn.
        missed = set()
        for attempt, strategy in enumerate(STRATEGIES):
            if self._abort.is_set():
                self._set_state('ABORTED')
                return
            shape = (round(strategy['yaw_offset'], 6),
                     round(strategy['z_offset'], 6))
            if shape in missed and not strategy['redetect']:
                self.get_logger().info(
                    f'skipping {strategy["name"]}: the jaws already shut on '
                    f'nothing at this yaw and height, and this rung offers '
                    f'neither a new one nor a fresh look')
                continue
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
            if picked_point is GRASP_MISSED:
                missed.add(shape)
                # The motion all worked and the jaws found nothing, which is
                # the one failure the ladder is actually for: its lower-8mm
                # rungs exist because the commanded height can be wrong for
                # an object even when every move is right. So carry on down
                # the ladder rather than stopping, whatever
                # retry_after_preflight says -- that guard is about descents
                # that will not fly, and this descent flew.
                picked_point = None
                self.get_logger().info(
                    'the grip missed but the motion was sound, so the next '
                    'rung is worth trying: this is what the lower-8mm '
                    'strategies are for')
                continue
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
                self.safe_shutdown(states, 'after a failed pick')
                return
        else:
            self._set_state(
                'FAILED',
                f'every pick strategy was exhausted -- {self._failure_summary()}')
            self.safe_shutdown(states, 'after a failed pick')
            return

        # Picked. Where the cycle goes from here is the one decision
        # place_after_pick makes.
        if not self.get_parameter('place_after_pick').value:
            return self._hold_at_staging(states)
        return self.run_place_half(states, 'cycle complete')

    def take_up_the_view(self, states):
        """Get somewhere the object can be looked for from.

        Usually HOME: it is out of the camera's frame, so the map can be
        rebuilt with the arm not in it and the detector sees the table
        rather than the robot.

        The exception is an arm already standing at the staging pose,
        which is where the boot walk leaves it and where a Pick is most
        often pressed from. Folding down to HOME and coming straight back
        out costs about ten seconds each way for a look that, from the
        staging pose, usually works anyway -- so with locate_from_staging
        the cycle looks from where it stands.

        What that costs is said out loud, because it has bitten before:
        the arm is in the picture here, so a detection can land on the
        gripper. That is caught rather than tolerated --
        implausible_detection refuses a point inside self_detect_radius of
        the tool, and the ladder's remap-from-home rung goes and looks
        properly.

        It used to give up the map refresh too, which was worse and
        quieter: the cycle planned against whatever the boot walk had
        captured, however long ago. map_before_pick drops that map and
        retakes it from here, with the same in-frame caveat and the same
        robot_in_map guard capture_octomap applies anywhere else.
        """
        if (self.get_parameter('locate_from_staging').value
                and self.at_state(PRE_PICK_STATE, states)):
            if self.get_parameter('map_before_pick').value:
                self.get_logger().info(
                    'the arm is already at the staging pose, so this cycle '
                    'looks from there rather than folding down to HOME and '
                    'coming back. The map is dropped and retaken from here '
                    'first, so the plan is made against the table as it is '
                    'now. locate_from_staging:=false to go home first '
                    'instead, map_before_pick:=false to keep the old map.')
                self.refresh_map_in_place()
            else:
                self.get_logger().info(
                    'the arm is already at the staging pose, so this cycle '
                    'looks from there rather than folding down to HOME and '
                    'coming back. The map is not refreshed -- it is the one '
                    'taken the last time the arm was out of the camera '
                    'frame, which the boot walk takes on its way through. '
                    'locate_from_staging:=false to always go home first.')
            self._set_state('LOCATE', f'{self.prompt} (from the staging pose)')
            payload = self.locate(min_count=1)
            seen = (payload or {}).get('detections') or []
            if seen:
                problem = self.implausible_detection(seen[0])
                if problem is None:
                    # Cached, so the attempt that follows reuses this answer
                    # rather than paying for a second inference.
                    self._last_detection = seen[0]
                    self._last_payload = payload
                    return True
                # Something was found and it was the robot: the view is
                # blocked rather than the object absent.
                if not self.get_parameter('use_home').value:
                    # And with HOME switched off there is nowhere out of
                    # frame to look from, so there is no trip to take. Say
                    # that, rather than driving to the pose the arm is
                    # already standing in and looking at the same gripper
                    # a second time.
                    self.get_logger().error(
                        f'{problem} The arm is at the staging pose and '
                        f'HOME is off (use_home:=false), so there is no '
                        f'other pose to look from. Move the object clear '
                        f'of the gripper, or set use_home:=true to fold '
                        f'the arm out of frame and look again.')
                    return NOTHING_SEEN
                self.get_logger().warn(
                    f'{problem} Going to HOME and looking from out of '
                    f'frame, which is what the trip is for.')
                return self.arrive_at_home()
            if payload is None:
                # No answer at all, which is not the same as an empty
                # frame and must not be reported as one. Measured, run at
                # 11:12: the log said "no fresh /vlm/detections -- is
                # VLM/run_vlm_detector.sh running?" and then, in the very
                # next line, "the detector answered, and it answered that
                # the frame is empty". It had not answered; there was no
                # publisher on the topic at all. Anyone reading that goes
                # looking for the object instead of starting the detector.
                self.get_logger().error(
                    f'no answer from the detector, so nothing can be said '
                    f'about whether "{self.prompt}" is in the frame. This '
                    f'is not an empty table -- it is an empty topic. Start '
                    f'it with VLM/run_vlm_detector.sh and press Pick again.')
                return NO_DETECTOR
            # An empty frame is an answer. The detector looked and there was
            # nothing there, and a second look from HOME finds the same
            # nothing thirty seconds later -- two 15 s waits and a fold down
            # and back, to report what the first look already knew.
            self.get_logger().error(
                f'nothing matched "{self.prompt}" from the staging pose. Not '
                f'driving home to look again: the detector answered, and it '
                f'answered that the frame is empty. If the arm was in the '
                f'way it would have found the gripper, not nothing.')
            return NOTHING_SEEN
        return self.arrive_at_home()

    def _hold_at_staging(self, states):
        """End the pick standing at the staging pose, still holding.

        Pick and place are two operator actions, and this is the pose
        between them: above the work surface, reachable from both ends, and
        the pose the place starts from anyway. Ending anywhere else would
        mean the place had to begin by getting out of wherever the pick
        finished -- which is exactly the move that sweeps across the table.

        Not fatal if it cannot be reached. The object is held either way,
        and saying where the arm actually is beats reporting a pose it is
        not in.
        """
        ctx = dict(self._ctx or {}, states=states, why='holding')
        self._step_pre_pick(ctx)
        held = self.held_object() or 'the object'
        if self.at_state(PRE_PICK_STATE, states):
            self._set_state(
                'HOLDING', f'{held} in the gripper, at the staging pose -- '
                           f'press Place to put it down')
        else:
            self._set_state(
                'HOLDING', f'{held} in the gripper, but short of the staging '
                           f'pose -- check the way is clear before placing')
        return None

    def _run_place(self):
        """The Place button's thread. Mirrors _run_cycle, including the
        crash path: a place that raises leaves the arm out over the drop
        exactly as a failed one does."""
        try:
            states = self.load_states()
            if not self.check_states(states):
                self._set_state('FAILED', 'recorded states are missing')
                return
            self.run_place_half(states, 'place complete')
        except Exception as exc:                     # noqa: BLE001 - reported
            self.get_logger().error(f'place crashed: {exc}')
            self._set_state('FAILED', str(exc))
            try:
                self.safe_shutdown(self.load_states(),
                                   'after the place crashed')
            except Exception as park_exc:            # noqa: BLE001 - reported
                self.get_logger().error(
                    f'could not park the arm after the crash: {park_exc}')
        finally:
            with self._lock:
                self._busy = False
            self.publish_status()

    def run_place_half(self, states, why):
        """Everything after the grasp is secured: carry, drop, come back.

        Shared by the Place button and by a place_after_pick cycle, so the
        two cannot drift apart -- in particular the failure handling below,
        which is the part that took three runs to get right.

        No detour home in the middle of it: capture_octomap refuses while
        holding -- the payload would be mapped as an obstacle that then
        travels with the tool -- so a trip home would clear the map, fail to
        replace it, and leave the drop planning against nothing.
        attach_object() has already put the payload in the planning scene,
        so these moves are planned with the object on the tool rather than
        with an invisible thing swinging through the octomap.
        """
        _pick_steps, place_steps = self.sequence_split()
        if not place_steps:
            # An edited sequence with nothing after the grasp. Reporting DONE
            # for a place that did not happen is the false success this
            # whole path is built to avoid.
            self.get_logger().error(
                'the sequence has no steps after the grasp, so there is '
                'nothing for a place to run. Add drop and release on the '
                'Setup tab.')
            self._set_state('FAILED', 'nothing in the sequence places')
            return
        ctx = dict(self._ctx or {}, states=states, why=why)
        if self.run_sequence(place_steps, ctx) is None:
            # Which half failed decides what to say and what to do.
            #
            # A place that never happened leaves the arm somewhere unplanned
            # and usually still holding the object: safe_shutdown checks a
            # refuge against the collision world, parks it, and takes the
            # motors off.
            #
            # A place that *did* happen and then could not be walked back from
            # is a different thing entirely. The object is where it was asked
            # to go and the job is done; only the trip home is unfinished.
            # Calling that "place failed" and disengaging the motors -- which
            # is what this did, on a cycle that had just picked and placed a
            # roll of tape -- is a worse answer than the truth.
            done = ctx.get('done') or []
            if 'release' in done or ctx.get('released'):
                self.get_logger().error(
                    'the object was placed, but the arm could not be walked '
                    'back afterwards. The cycle did its job; what is left is '
                    'an arm short of home. Parking it where it stands.')
                self.safe_shutdown(states, 'after placing, before parking')
                self._set_state(
                    'DONE',
                    'placed, but the arm is parked short of home -- clear '
                    'whatever is blocking the way back before the next cycle')
                return
            self.safe_shutdown(states, 'after a failed place')
            self._set_state('FAILED', 'place failed')
            return
        self._set_state('DONE')

    # -- getting out of trouble ----------------------------------------------

    def posture_is_clear(self, joints, label):
        """Would the arm be in collision standing here? True / False / None.

        Asks /check_state_validity, which answers against the *current*
        planning scene -- octomap included -- without planning anything. That
        is the difference between checking a refuge and discovering it by
        arriving: a joint goal into an occupied posture comes back
        INVALID_MOTION_PLAN with nothing said about what is in the way, and
        that is what a failed cycle was doing twice in a row before giving up
        and driving home anyway.

        None means nothing could answer, which is deliberately not False: a
        service that is down is not a reason to call a posture occupied.
        """
        if not self.validity_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                '/check_state_validity is not being served, so a refuge '
                'cannot be checked before moving to it')
            return None
        request = GetStateValidity.Request()
        request.group_name = self.group
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(self.arm_joints)
        request.robot_state.joint_state.position = [float(v) for v in joints]
        result = self._await(self.validity_client.call_async(request), 5.0)
        if result is None:
            self.get_logger().warn(f'{label}: the validity check did not answer')
            return None
        if result.valid:
            return True
        contacts = sorted({f'{c.contact_body_1}/{c.contact_body_2}'
                           for c in result.contacts})
        self.get_logger().warn(
            f'{label} is not a refuge: the arm would be in collision there'
            + (f' -- {", ".join(contacts[:4])}' if contacts else '')
            + '. Not driving into it.')
        self.log_motion(label, 'validity', 'occupied',
                        target=[round(v, 5) for v in joints],
                        contacts=contacts[:8])
        return False

    def disengage_motors(self, why):
        """Take the motors off, by deactivating the hardware components.

        openarm_hardware's on_deactivate calls disable_all(), so this is what
        "disengage" means on this robot: the joints stop being driven.

        **The arm is then held up by nothing.** These are direct-drive motors
        with no brakes, so a limp arm sags, and an extended one falls. That is
        why this is the last thing the failure path does and only once a
        refuge has been reached -- see safe_shutdown. Bringing it back needs
        the components reactivated, which in practice means restarting the
        bringup.
        """
        if not self.hardware_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                'the controller manager is not offering '
                'set_hardware_component_state, so the motors cannot be taken '
                'off from here. Stop the bringup instead.')
            return False
        done = []
        for component in ('openarm_left_hardware_interface',
                          'openarm_right_hardware_interface'):
            request = SetHardwareComponentState.Request()
            request.name = component
            request.target_state.id = LifecycleState.PRIMARY_STATE_INACTIVE
            request.target_state.label = 'inactive'
            result = self._await(
                self.hardware_client.call_async(request), 10.0)
            if result is None or not result.ok:
                self.get_logger().error(
                    f'could not deactivate {component}; its motors are still '
                    f'driven')
                continue
            done.append(component)
        if done:
            self.get_logger().warn(
                f'motors disengaged {why}: {", ".join(done)}. The arm is not '
                f'being held up any more -- support it before moving it, and '
                f'restart the bringup to drive it again.')
            self.log_motion('DISENGAGE', 'hardware', 'off', components=done)
        return bool(done)

    def engage_motors(self, why):
        """Put the motors back on, holding whatever posture the arm is in.

        The counterpart of disengage_motors, and only ever used by the
        recovery after a torque trip -- there is deliberately no button for
        it. Re-activating is safe in the one way that matters:
        openarm_hardware's on_activate seeds its command buffers from the
        *measured* state, so the arm holds where it has come to rest rather
        than being driven anywhere.

        What it cannot undo is the falling that happened while the motors
        were off. Whatever the arm slumped onto, it is still on.
        """
        if not self.hardware_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(
                'the controller manager is not offering '
                'set_hardware_component_state, so the motors cannot be put '
                'back on from here. Restart the bringup.')
            return False
        done = []
        for component in ('openarm_left_hardware_interface',
                          'openarm_right_hardware_interface'):
            request = SetHardwareComponentState.Request()
            request.name = component
            request.target_state.id = LifecycleState.PRIMARY_STATE_ACTIVE
            request.target_state.label = 'active'
            result = self._await(
                self.hardware_client.call_async(request), 10.0)
            if result is None or not result.ok:
                self.get_logger().error(
                    f'could not activate {component}; its motors are still '
                    f'off and the arm is held up by nothing')
                continue
            done.append(component)
        if done:
            self.get_logger().warn(
                f'motors re-engaged {why}: {", ".join(done)}. They are '
                f'holding the posture the arm came to rest in.')
            self.log_motion('ENGAGE', 'hardware', 'on', components=done)
        return len(done) == 2

    def recover_from_contact(self, why):
        """Let go, let it settle, take hold again, and glide back.

        What a torque trip means is that the arm is pressing on something
        and the motor is holding it there, so the first thing is to stop
        driving: that is the letting go. Then a pause, because the point of
        it is to let the arm come off whatever it is leaning on rather than
        to spring away from it. Then the motors back on -- they hold
        wherever it has settled -- and a slow walk to the staging pose, at
        boot speed, as a joint goal to a recorded posture.

        Starting over from there is the whole intent: after this the arm is
        where a cycle begins, with the motors on and nothing held.
        """
        self._set_state('RECOVER', f'letting go after {why}')
        if not self.disengage_motors(f'after {why}'):
            self.get_logger().error(
                'could not take the motors off after a torque trip, so the '
                'arm is still pressing on whatever it hit. Stop the bringup.')
            return False
        settle = self.get_parameter('contact_relax_seconds').value
        self.get_logger().warn(
            f'motors off for {settle:.0f} s to let the arm come off what it '
            f'hit. It is held up by nothing for that time -- if it was '
            f'extended over the table it is settling onto it now.')
        self._set_state('RECOVER', 'letting the arm settle')
        self._abort.wait(settle)
        if not self.engage_motors('to glide back to the staging pose'):
            self._set_state(
                'STOPPED',
                'the motors could not be put back on after the contact -- '
                'restart the bringup')
            return False
        # Slowly, and to a recorded posture. Whatever the arm settled into
        # while it was limp, this is a joint goal out of it.
        self._set_state('RECOVER', 'gliding back to the staging pose')
        detail = self._walk_to_boot_pose()
        self._set_state('IDLE', f'recovered after {why}: {detail}')
        return True

    def safe_shutdown(self, states, why):
        """Park the arm somewhere checked, then take the motors off.

        The failure path, and deliberately not _retreat_to_home. A cycle that
        failed has an arm somewhere unplanned -- often still holding the
        object, often with its start state already in collision -- and the
        old answer was a joint goal at HOME, which is a folded posture on the
        far side of the workspace. Measured, run 1788933180: PRE_PICK and DROP
        both came back -2 with the gripper in the octomap, and home was then
        driven to anyway, carrying the object.

        So: up off the surface, then the nearest refuge whose posture the
        collision world says is actually free, and only then the motors.

        A refuge that cannot be reached is not replaced by driving somewhere
        else and hoping. If none can be reached the arm stays where it is,
        still driven, and says so -- an arm stranded over the table is bad,
        an arm going limp while stranded over the table falls onto it.
        """
        self._set_state('RECOVER', why)
        self.clear_the_surface(why)

        # Hold the gripper out of the octomap for the whole escape, not for
        # one leg of it.
        #
        # Getting stuck is what happens otherwise, and it is the map that
        # does it. Measured, run 1788948709: the descent was refused, so the
        # tool was still at transit height with the jaws open right above the
        # object; clear_the_surface saw it as already clear and released the
        # column, which handed the gripper back to collision checking against
        # the very voxels it was standing in. Both refuges then came back -2
        # in under a second -- not because they were unreachable, but because
        # the arm's own start state had just been declared invalid.
        #
        # The exemption is for the gripper alone and the refuges are still
        # checked against everything else, so this does not blind the escape;
        # it stops the escape being blocked by the thing it is escaping.
        exempt = False
        if self.get_parameter('gripper_octomap_exemption').value:
            exempt = self.allow_gripper_in_octomap(True)
            self._octomap_exempt = exempt
        try:
            reached = self._reach_a_refuge(states, why)
        finally:
            if exempt:
                self.allow_gripper_in_octomap(False)
                self._octomap_exempt = False

        if reached is None:
            # Which of the two it is matters: one is cleared with a service
            # call and the other needs the stack restarted, and a single
            # line about refuges tells the operator neither.
            jammed = self._planner_stalled is not None and self._planner_answered
            self._set_state(
                'STOPPED',
                'move_group has stopped answering, so nothing can be planned '
                'and the arm is parked where it stands with the motors on'
                if jammed else
                'no refuge could be reached, so the arm is parked where it '
                'stands and the motors are left on')
            if jammed:
                self.get_logger().error(
                    'the motors are staying on because there is no planner '
                    'left to plan the way out: move_group answered earlier in '
                    'this run and has since stopped serving /move_action. '
                    'Restart the stack. Until then the arm can be driven out '
                    'with record_states.py --play, which talks to the '
                    'controller directly.')
            else:
                self.get_logger().error(
                    'nothing safe could be reached, so the motors are staying '
                    'on: a limp arm holds itself up with nothing, and letting '
                    'go here would drop it onto whatever it is over. Clear the '
                    'obstruction (the octomap usually: ros2 service call '
                    '/clear_octomap std_srvs/srv/Empty) and drive it out with '
                    'record_states.py --play.')
            self.log_motion('RECOVER', 'joint', 'no-refuge',
                            planner='jammed' if jammed else 'answering')
            return False

        if self.get_parameter('disengage_on_failure').value:
            self.disengage_motors(f'at {reached}, {why}')
        elif self._contact_fired and self.get_parameter(
                'disengage_on_contact').value:
            # _contact_fired means the arm is *still* resting on what it
            # hit -- retreat_from_contact clears it again the moment the
            # back-off actually works, which is the ordinary case. This
            # branch is for the back-off itself having failed, so the arm
            # really may still be leaning on the obstruction, and letting
            # go is genuinely the safer of the two options here.
            self.get_logger().warn(
                'the contact guard fired and the back-off away from it did '
                'not work, so the arm may still be resting on what it hit. '
                'Taking the motors off now that it is parked. Holding '
                'position against something it has run into is what cooks '
                'a motor.')
            if self.get_parameter('recover_after_contact').value:
                # Off, settle, back on, and glide to the staging pose --
                # which is where a cycle starts, so the next one is just a
                # press away.
                return self.recover_from_contact('a torque trip')
            self.disengage_motors(f'at {reached}, after a contact')
        else:
            self.get_logger().info(
                f'parked at {reached} with the motors on. Nothing here '
                f'needs them off -- press Stop to park and release, or set '
                f'disengage_on_failure if every failure should.')
        return True

    def _reach_a_refuge(self, states, why):
        """The nearest posture the collision world calls free, or None."""
        reached = None
        for name in (PRE_PICK_STATE, None):
            if name is None:
                target = self.home_positions()
                label = 'HOME'
            else:
                entry = (states or {}).get(name) or {}
                target = entry.get('joints')
                label = name.upper()
                if not target:
                    continue
            if self.posture_is_clear(target, label) is False:
                continue
            self._set_state(label, f'refuge, {why}')
            if name is None:
                if self.move_to_home():
                    reached = label
                    break
            elif self.move_to_state(name, states):
                reached = label
                break
            self.get_logger().warn(
                f'{label} is clear but could not be reached; trying the next '
                f'refuge')

        return reached

    def _retreat_to_home(self, states, why):
        """Back the way it came: the staging pose, then home.

        PRE_PICK is a posture the arm is known to be able to reach from both
        the drop and from home, which is what makes it a safe waypoint out --
        and it is above the work surface, so the trip home does not sweep
        across it.

        Used after a failed place as well as a successful one. Leaving the arm
        stopped where it failed, extended over the table and often still
        holding the object, is the worst outcome available.

        But going home *anyway* when pre_pick could not be reached is worse
        still, and that is what this used to do. Measured, run 1788869179
        cycle 2: CLEAR got the tool up to z=0.4286, PRE_PICK_STATE came back
        exhausted, and HOME was then commanded from there as a joint goal --
        the arm swept out to x=0.44 and down to z=0.358 on its way to folded,
        across the object it had just failed to pick. HOME is only safe from
        the staging pose. If pre_pick will not go, the arm stops where it is
        and says so.
        """
        # Before the staging pose, which is a joint goal: if the arm is still
        # down at the object, that goal is what sweeps it across the surface.
        self.clear_the_surface(why)
        self._set_state('PRE_PICK', f'on the way back, {why}')
        carrying = self.holding()
        if self.move_to_state(PRE_PICK_STATE, states, on_own_side=carrying):
            self._set_state('HOME', why)
            return self.move_to_home()

        # One more try, from higher up: the usual reason pre_pick refuses from
        # here is that the gripper is still inside the octomap of the thing it
        # was reaching for, and a few more centimetres of clearance is the
        # whole fix.
        lifted = self.lift_to_clearance('PRE_PICK retry')
        if lifted and self.move_to_state(PRE_PICK_STATE, states,
                                         on_own_side=carrying):
            self._set_state('HOME', why)
            return self.move_to_home()

        if not self.get_parameter('home_requires_pre_pick').value:
            self.get_logger().warn(
                'pre_pick will not go, and home_requires_pre_pick is off, so '
                'HOME is being commanded from here anyway -- this is a joint '
                'goal from wherever the arm is standing and it can sweep '
                'across the work surface')
            self._set_state('HOME', why)
            return self.move_to_home()

        self._set_state(
            'STOPPED',
            'cannot reach pre_pick, and HOME from here would sweep the arm '
            'across the work surface')
        self.get_logger().error(
            'the arm is parked where it stands. pre_pick could not be '
            'reached on the way back, and HOME is a joint goal -- commanding '
            'it from a low, extended posture is what drags the arm across '
            'the table, so it is not being sent. Clear whatever is blocking '
            'the plan (the octomap usually: ros2 service call '
            '/clear_octomap std_srvs/srv/Empty) and drive it out with '
            'record_states.py --play, or set home_requires_pre_pick:=false '
            'to accept the sweep.')
        self.log_motion('HOME', 'joint', 'refused-no-pre-pick')
        return False

    def _arm_that_reaches(self, candidates, targets, quat, where, states):
        """KDL alone: the first arm it confirms can reach every target.

        Returns (decided, states). decided is False only when nothing was
        confirmed, which means "ask again properly" rather than "no arm can"
        -- KDL's yes is conclusive and its no is not. A switch that finds no
        recorded states is decided *and* fatal, so the two are reported
        separately: collapsing them would send a fatal case round the slow
        path and overwrite the FAILED state on the way.
        """
        orientations = int(
            self.get_parameter('arm_choice_orientations').value)
        for arm in candidates:
            verdicts = [self.reachable(point, quat, arm=arm,
                                       orientations=orientations, cheap=True)
                        for _, point in targets]
            if not all(v is True for v in verdicts):
                continue
            if arm == self.arm:
                self.get_logger().info(f'the {arm} arm can reach it')
                return True, states
            self._set_state('SWITCH_ARM', f'{self.arm} -> {arm} ({where})')
            self.configure_arm(arm)
            switched = self.load_states()
            if not self.check_states(switched):
                self._set_state(
                    'FAILED', f'no recorded states for the {arm} arm')
                return True, None
            return True, switched
        return False, states

    def arm_candidates(self, detection, payload):
        """The arms to consider, best first.

        camera_half puts the half the object is in first and the other arm
        second. The orders only disagree when *both* arms can reach, and then
        the near arm wins -- a cross-body reach is slower, more likely to clip
        the other arm, and nearer its joint limits.
        """
        order = self.get_parameter('arm_order').value
        if order == 'right_then_left':
            arms = ['right', 'left']
        elif order == 'left_then_right':
            arms = ['left', 'right']
        else:
            preferred = self.arm_for(detection, payload)
            arms = [preferred, other_arm(preferred)]
        # An arm with something already in its jaws is not a candidate: it
        # would open them on the way down and drop what it is carrying. The
        # point of two arms is that the other one takes the job.
        free = [arm for arm in arms if not self.holding(arm)]
        if len(free) != len(arms):
            self.get_logger().info(
                f'{" and ".join(self.busy_arms())} already holding '
                f'something, so only {" and ".join(free) or "no arm"} can '
                f'take this one')
        return free

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

        An arm with something already in its jaws is not a candidate: it
        would open them on the way down and drop what it is carrying. With
        one hand full the other one takes the job, which is the whole point
        of having two.
        """
        self._set_state('LOCATE', f'{self.prompt} (choosing an arm)')
        payload = self.locate(min_count=1)
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

        # by_side: the half of the frame the object is in, and get on with
        # it. The camera half gets this right nearly every time -- an object
        # on the left is picked by the left arm -- and asking the solvers
        # instead cost 87 seconds between the first look and the second,
        # measured, for the same answer.
        #
        # Nothing is given up by not asking. The pre-flight settles whether
        # the pick is actually possible, properly, for the arm chosen, and it
        # runs before anything moves -- so a wrong guess costs a pre-flight,
        # not a motion. by_reach restores the old behaviour for the cases
        # where the camera half really is not the right signal.
        if self.arm_selection != 'by_reach':
            arm = candidates[0]
            if arm == self.arm:
                self.get_logger().info(
                    f'{where}, so the {arm} arm takes it')
                return states
            self._set_state('SWITCH_ARM', f'{self.arm} -> {arm} ({where})')
            self.configure_arm(arm)
            switched = self.load_states()
            if not self.check_states(switched):
                self._set_state(
                    'FAILED', f'no recorded states for the {arm} arm')
                return None
            return switched

        grasp, pregrasp, quat, _ = self.grasp_from_detection(
            detection, STRATEGIES[0])
        transit = (grasp[0], grasp[1],
                   grasp[2] + self.get_parameter('transit_height').value)
        targets = [('grasp', grasp), ('pre-grasp', pregrasp),
                   ('transit height', transit)]

        # Cheap first, thorough only if that settles nothing.
        #
        # KDL confirms the common case in milliseconds -- the object is in
        # front of an arm that can plainly reach it -- and the solvers behind
        # it cost 2.9 seconds per orientation per height per arm whether they
        # find anything or not, which was 52 seconds before the robot moved.
        #
        # But cheap alone loses the case this method exists for: an object
        # only the *far* arm can reach, where KDL's "cannot tell" would leave
        # the near arm selected and the pick refused. So when no arm is
        # confirmed cheaply, the expensive question gets asked after all. The
        # cost lands on the hard case, which is where it is worth paying.
        if self.get_parameter('arm_choice_cheap').value:
            decided, chosen = self._arm_that_reaches(
                candidates, targets, quat, where, states)
            if decided:
                return chosen
            self.get_logger().info(
                'no arm could be confirmed the quick way, so the object is '
                'somewhere awkward; asking the slow solvers')

        unreachable = []
        for arm in candidates:
            verdicts = [
                self.reachable(point, quat, arm=arm,
                               orientations=int(self.get_parameter(
                                   'arm_choice_orientations').value))
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
        # Same reasoning as the per-attempt check: this samples a few
        # orientations and asks whether a posture exists, while the pre-flight
        # samples the lot and asks whether the line actually flies from the
        # staging pose. When the thorough one is going to run anyway, the
        # cheap one does not get to veto it -- it was wrong about a roll of
        # tape the left arm had picked eleven minutes earlier.
        #
        # Which arm, then? The one arm_candidates preferred, which is the half
        # of the frame the object is in. The pre-flight will refuse properly
        # if it really cannot be done, and OUT_OF_REACH will mean it.
        if self.get_parameter('preflight_descent').value:
            self.get_logger().warn(
                f'no arm passed the reach check, but the pre-flight asks a '
                f'better question and nothing has moved yet, so the '
                f'{candidates[0]} arm gets its turn -- {message}')
            self._note_failure('REACH', f'(advisory) {message}')
            if candidates[0] != self.arm:
                self._set_state(
                    'SWITCH_ARM',
                    f'{self.arm} -> {candidates[0]} ({where}), on the '
                    f'pre-flight rather than the reach check')
                self.configure_arm(candidates[0])
                states = self.load_states()
                if not self.check_states(states):
                    self._set_state(
                        'FAILED',
                        f'no recorded states for the {candidates[0]} arm')
                    return None
            return states
        self._note_failure('REACH', message)
        self._set_state('OUT_OF_REACH', message)
        return OUT_OF_REACH

    def _on_grasp_candidates(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            self.get_logger().warn('unreadable /grasp/candidates payload')
            return
        with self._lock:
            self._grasps = payload
        found = payload.get('grasps') or []
        self.get_logger().info(
            f'grasp server proposed {len(found)} candidate(s)'
            + (f', best score {found[0]["score"]:.2f} at '
               f'{found[0]["width"] * 1000:.0f} mm' if found else ''))

    def model_proposals(self, detection, strategy):
        """The model's grasps for this object, best first, or [].

        Each comes back in the same shape the synthesised one has --
        (grasp, pregrasp, quat) -- so the pre-flight cannot tell where a
        proposal came from and does not need to.

        Filtered on age and on distance from the object actually being
        picked: the server publishes for whatever it was last asked about,
        and using yesterday's tape to pick today's screwdriver would be worse
        than not using it at all.
        """
        if not self.get_parameter('use_grasp_model').value:
            return []
        with self._lock:
            payload = self._grasps
        if not payload:
            return []
        age = self.get_clock().now().nanoseconds * 1e-9 - payload.get(
            'stamp', 0.0)
        limit = self.get_parameter('grasp_model_max_age').value
        if limit > 0.0 and age > limit:
            self.get_logger().info(
                f'the grasp server\'s candidates are {age:.0f} s old, so '
                f'they are about some earlier look; synthesising instead')
            return []
        point = detection.get('point') or []
        about = payload.get('about') or []
        if len(about) == 3 and len(point) == 3:
            offset = math.dist(about, point)
            if offset > self.get_parameter('grasp_model_max_offset').value:
                self.get_logger().info(
                    f'the candidates are about a point {offset * 1000:.0f} mm '
                    f'from this object, so they are for something else')
                return []

        approach = self.get_parameter('approach_height').value
        drop = strategy['z_offset']
        limit = math.degrees(self.get_parameter('grasp_model_max_tilt').value)
        out, too_tilted = [], []
        for entry in payload.get('grasps') or []:
            position = entry.get('position')
            quat = entry.get('quat')
            if not position or not quat or len(position) != 3:
                continue
            tilt = entry.get('tilt_deg')
            if tilt is not None and limit > 0.0 and tilt > limit:
                too_tilted.append(tilt)
                continue
            grasp = (position[0], position[1], position[2] + drop)
            out.append({
                'grasp': grasp,
                # Straight up, because that is the way back down: the
                # descent is a vertical column whatever the tool's
                # orientation. It is also why the tilt is bounded above.
                'pregrasp': (grasp[0], grasp[1], grasp[2] + approach),
                'quat': tuple(quat),
                'source': 'model',
                'score': entry.get('score'),
                'width': entry.get('width'),
                'tilt_deg': tilt,
            })
        if too_tilted:
            self.get_logger().info(
                f'{len(too_tilted)} of the model\'s grasps come in at '
                f'{min(too_tilted):.0f}-{max(too_tilted):.0f} deg off '
                f'vertical, past the {limit:.0f} deg this cycle can fly: the '
                f'approach and retreat are a vertical column, so a steeper '
                f'grasp would be driven down across the object rather than '
                f'onto it. Skipped rather than mis-flown.')
        return out

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
            payload = self.locate(min_count=1)
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
        # How far the tool will be above the bottom of the object once it
        # has it. The place needs exactly this number and cannot measure it
        # itself: the object is in the jaws by then and the surface it came
        # off is somewhere else.

        # Before anything moves. Out of reach is not a planning failure and no
        # strategy on the ladder recovers it, so it ends the cycle instead of
        # burning six identical attempts. Both ends of the column are checked:
        # a grasp the arm can touch but not approach from above is no use.
        transit = (grasp[0], grasp[1],
                   grasp[2] + self.get_parameter('transit_height').value)
        # Skipped entirely when the pre-flight is going to run, and that is
        # not an optimisation so much as removing work nobody reads.
        #
        # This check samples orientations and asks whether a *posture*
        # exists; the pre-flight samples the same set and asks whether the
        # actual straight line flies from the actual staging pose. The second
        # is strictly more informative, so the first was made advisory -- and
        # an advisory answer still cost a planning request per orientation
        # per target. Measured after reach_orientations went 3 -> 10: LOCATE
        # to PREFLIGHT took 162 seconds, most of it spent producing a verdict
        # that was then logged and ignored.
        if (self.get_parameter('check_reach').value
                and not self.get_parameter('preflight_descent').value):
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

        # What to try, best first: whatever the grasp model proposed, then
        # the synthesised top-down pose as the last resort. The pre-flight
        # decides between them -- it is the only thing here that knows
        # whether a grasp can actually be flown to -- and it takes the first
        # whose whole column works.
        #
        # The synthesised one stays on the end rather than being replaced.
        # GraspNet was trained for a 100 mm gripper against this one's 44 mm,
        # so on a wide object every proposal it makes can be one the robot
        # cannot close on, and the fallback is then the only way to pick at
        # all. Measured on a roll of tape: three candidates, needing 55, 66
        # and 98 mm.
        proposals = self.model_proposals(detection, strategy)
        proposals.append({'grasp': grasp, 'pregrasp': pregrasp, 'quat': quat,
                          'source': 'top-down', 'score': None, 'width': None})
        if len(proposals) > 1:
            self.get_logger().info(
                f'{len(proposals) - 1} grasp(s) from the model to try before '
                f'falling back to the synthesised top-down pose')

        self._preflight_choice = None
        for index, proposal in enumerate(proposals):
            if self._abort.is_set():
                self._set_state('ABORTED')
                return None
            grasp = tuple(proposal['grasp'])
            pregrasp = tuple(proposal['pregrasp'])
            quat = tuple(proposal['quat'])
            approach_point = ((grasp[0], grasp[1], grasp[2] + transit_height)
                              if staged else tuple(pregrasp))
            if proposal['source'] == 'model':
                self._set_state(
                    'PREFLIGHT',
                    f'model grasp {index + 1}, score {proposal["score"]}, '
                    f'{proposal["width"] * 1000:.0f} mm')
            self._preflight_choice = self.preflight_column(
                grasp, pregrasp, approach_point, quat, 'PREFLIGHT')
            if self._preflight_choice is not None:
                if proposal['source'] == 'model':
                    tilt = proposal.get('tilt_deg')
                    self.get_logger().info(
                        f'flying the model\'s grasp {index + 1} of '
                        f'{len(proposals) - 1}: score {proposal["score"]}, '
                        f'{proposal["width"] * 1000:.0f} mm opening'
                        + ('' if tilt is None else f', {tilt:.0f} deg off '
                           f'vertical')
                        + f' at {[round(v, 4) for v in grasp]}')
                self.log_motion('PREFLIGHT', 'proposal', 'chosen',
                                source=proposal['source'],
                                candidate=index + 1, of=len(proposals),
                                score=proposal['score'],
                                width=proposal['width'],
                                tilt_deg=proposal.get('tilt_deg'),
                                target=[round(v, 5) for v in grasp])
                break
            if proposal['source'] == 'model':
                self.get_logger().info(
                    f'the model\'s grasp {index + 1} does not fly from here; '
                    f'trying the next')

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

        # From here the cycle is data. pick_place_sequence.DEFAULT_SEQUENCE
        # is the order, the web UI edits it, and the steps below are what the
        # names resolve to. The approach above is deliberately not in it: the
        # pre-flight proves the whole column from the staging posture as one
        # unit, so reordering what it depends on would not give a different
        # cycle, it would give an unchecked one.
        ctx = {
            'grasp': grasp, 'pregrasp': pregrasp, 'quat': quat,
            'point': point, 'states': states, 'from_z': from_z,
            'strategy': strategy,
        }
        self._ctx = ctx
        pick_steps, _place_steps = self.sequence_split()
        return self.run_sequence(pick_steps, ctx, default=point)

    # -- the cycle, as data --------------------------------------------------

    def sequence_split(self):
        """(pick steps, place steps) -- where the retry ladder stops.

        The ladder retries the pick: locate, descend, grasp. It does not retry
        the place, because by then the object is held and starting over would
        mean putting it back. The boundary is the point the grasp is secured
        -- after verify_grasp if the sequence has one, after the grasp itself
        if it does not -- and the UI draws it, because "which of these get
        retried" is a fair thing to want to know before reordering them.
        """
        sequence = list(self._sequence)
        for anchor in ('verify_grasp', 'grasp'):
            if anchor in sequence:
                cut = sequence.index(anchor) + 1
                return sequence[:cut], sequence[cut:]
        return sequence, []

    def run_sequence(self, steps, ctx, default=True):
        """Run named steps in order. Returns `default`, None, or GRASP_MISSED.

        A step returns True to carry on, False to end the cycle, or
        GRASP_MISSED to send the ladder to its next rung. Anything a step
        raises ends the cycle rather than the process: the sequence is edited
        by hand from a browser, and a bad edit must not be able to take the
        orchestrator down with it.
        """
        for name in steps:
            if self._abort.is_set():
                self._set_state('ABORTED')
                return None
            handler = self.STEP_HANDLERS.get(name)
            if handler is None:
                self.get_logger().error(
                    f'{name!r} is in the sequence but nothing implements it; '
                    f'skipping. This should have been caught when the '
                    f'sequence was loaded.')
                continue
            self._step = name
            try:
                outcome = handler(self, ctx)
            except Exception as exc:                 # noqa: BLE001 - reported
                self.get_logger().error(f'step {name!r} raised: {exc}')
                self._note_failure(name.upper(), f'the step raised: {exc}')
                return None
            if outcome is GRASP_MISSED:
                return GRASP_MISSED
            if not outcome:
                return None
            # What has actually happened, so a caller can tell a cycle that
            # failed before doing its job from one that failed after.
            if isinstance(ctx, dict):
                ctx.setdefault('done', []).append(name)
        return default

    # -- the steps -------------------------------------------------------
    #
    # Each is the body that used to sit inline in _attempt_pick or _place, in
    # the same order and with the same reasoning; what changed is that the
    # order now comes from a file. Every one takes the cycle context and
    # returns True to carry on.

    def _step_open_gripper(self, ctx):
        """Before the descent, not between two of them: with the tool above
        the object at transit height there is nothing for open fingers to
        catch on, and a gripper that fails to open then costs no motion."""
        self._set_state('OPEN_GRIPPER')
        return self.open_gripper()

    def _step_descend(self, ctx):
        grasp, quat = ctx['grasp'], ctx['quat']
        # What the pre-flight promised was a line from the posture it chose.
        # The arm does not always hold that posture: measured at TRANSIT,
        # joint1 settled 70 mrad from its commanded value, ~35 mm at the tool,
        # and from there the same descent solved 48.6% instead of 100%. Let it
        # finish arriving, then report the gap, so a descent that fails after
        # a pre-flight that passed is not a mystery.
        entry = self._preflight_choice
        if entry and not entry.get('linear'):
            settling = list(entry['joints'])
            if entry.get('tilt') is not None:
                settling[6] = entry['tilt']
            self.settle_joints(settling, 'DESCEND')
        self._report_posture_drift('DESCEND')

        # Ask the question again, from where the arm actually is.
        #
        # The pre-flight proved this column from the posture it *predicted*
        # the approach would end in. The arm does not land there: measured,
        # run 1788948709, TRANSIT arrived 32.6 mm off and the descent the
        # pre-flight had just passed then cost 6.42 rad against a 1.5 rad
        # budget and was refused -- after the arm had flown the whole
        # approach. A check made from a posture the arm is not in has checked
        # a descent that is not the one about to be flown.
        better = self.descend_orientation(grasp, quat)
        if better is not None and better is not quat:
            self.get_logger().info(
                'the descent the pre-flight proved does not fly from where '
                'the arm actually landed; using an orientation that does')
            quat = better
            ctx['quat'] = quat

        # On the column before the descent, not during it.
        #
        # /compute_cartesian_path interpolates from where the tool *is* to
        # the waypoint, so a tool that is off the column does not descend
        # -- it slants, and closes the lateral gap on the way down, through
        # whatever is standing there. Measured, run 1789111587: TRANSIT
        # landed 21.4 mm short in x, the descent ended 22.8 mm past it, so
        # the tool travelled 44 mm sideways while dropping 43 mm. That is a
        # 45-degree line through the object, and the jaws then closed on
        # nothing 22 mm off target.
        #
        # The correction is horizontal and it happens up here, at transit
        # height, where there is nothing to sweep. Doing it low -- a
        # centimetre above the object, say -- is the very move that knocks
        # the object over.
        self.align_over_column(grasp, quat, 'DESCEND')

        # From here until the tool is clear again the arm is on a vertical
        # line over the object, and every way out has to go back up it first.
        self._column = (list(grasp), list(ctx['pregrasp']), list(quat))
        self._set_state('DESCEND')
        if not self.descend_column(
                grasp, ctx['from_z'], grasp[2], quat, 'DESCEND',
                uncheck_collisions=self.get_parameter(
                    'approach_ignores_octomap').value,
                linear_only=self.get_parameter('descend_linear_only').value):
            unchecked = (self.get_parameter('approach_ignores_octomap').value
                         and self.get_parameter(
                             'descend_ignores_octomap').value)
            # Say what actually refused it. A refusal for joint travel and a
            # refusal for reach look identical from here and have opposite
            # fixes.
            blame = self._column_refusal or (
                'the collision world may hold the object itself, or the '
                'grasp is below the work surface'
                if not unchecked else
                'collision checking was already off for this leg, so this is '
                'reach rather than an obstacle: the arm runs out of travel '
                'along the line. Most often it is not standing where the '
                'pre-flight assumed -- compare the commanded and measured '
                'joints in the motion log at TRANSIT')
            self._note_failure('DESCEND', (
                f'arrived above the object but could not descend to '
                f'z={grasp[2]:.3f} -- {blame}'))
            self.clear_the_surface('after a failed descent')
            return False
        self.close_descent_gap(grasp, quat)
        return True

    def _step_grasp(self, ctx):
        grasp = ctx['grasp']
        # Where the jaws actually are, before they close on whatever is
        # there. The number that decides whether the grip works is this one:
        # the plan ended exactly at the commanded point and the arm stopped
        # 29 mm above it, so the jaws closed on air over the object and the
        # only symptom was 'closed-on-nothing'. Say it plainly instead.
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
                    f'{short * 1000:+.0f} mm in height and '
                    f'{sideways * 1000:.0f} mm sideways. The descent plan '
                    f'ended on the point; the arm did not follow it there. '
                    f'If this grip fails it failed here, not at the fingers.')
                self.log_motion('CLOSE_GRIPPER', 'gripper', 'off-target',
                                target=list(grasp), tcp=list(at_jaws),
                                error_mm=round(miss * 1000.0, 1))
        # The number the place needs, measured rather than commanded.
        #
        # "Do not drop it from a height" is answered by one figure: how far
        # the tool was above the surface the object was standing on at the
        # moment the jaws closed. Put the tool that far above the paper and
        # the object is resting on the paper, whatever its shape. The
        # commanded grasp is the wrong figure -- this arm lands 10-25 mm
        # from it -- so this is taken from where the tool actually is.
        if at_jaws is not None and self._pick_surface_z is not None:
            self._pick_hold_offset = float(
                at_jaws[2] - self._pick_surface_z)
            self.get_logger().info(
                f'picked up from {self._pick_hold_offset * 1000:.0f} mm above the '
                f'surface (tool z={at_jaws[2]:.3f}, surface '
                f'z={self._pick_surface_z:.3f}). The place puts the tool that '
                f'far above the sheet, so it is set down rather than dropped.')

        if not self.close_gripper_to_cap() and not self.close_lower(ctx):
            landed = at_jaws[2] if at_jaws else None
            self._note_failure('CLOSE_GRIPPER', (
                'the descent arrived'
                + (f' within {abs(landed - grasp[2]) * 1000:.0f} mm of the '
                   f'commanded z={grasp[2]:.3f}' if landed is not None else '')
                + ', and the jaws then shut on nothing. The height is wrong '
                  'for this object rather than the motion: lower '
                  'grasp_z_offset, or let the ladder try 8 mm lower'))
            # Up first, jaws left closed. The grip failing does not make the
            # arm any less extended over the table, and closed jaws are the
            # slimmest thing to lift back through whatever is down there.
            self.clear_the_surface('the grip closed on nothing')
            return GRASP_MISSED
        return True

    def close_lower(self, ctx):
        """The jaws shut on nothing: step down and try again, right here.

        The cheapest thing that can possibly work, and the first thing to
        try. The tool is already on the column with the object somewhere
        just below it, so a few millimetres down and another close costs
        one short line and one grip -- against a ladder rung, which flies
        the whole approach again for about a minute.

        It is what the failure usually is. The grasp height comes from the
        detected top of the object plus grasp_z_offset, and for anything
        thin -- a screwdriver handle, a ruler, the edge of a roll -- ten
        millimetres above the top is above the object entirely. Measured,
        run at 15:24: the descent landed within 7 mm of its commanded z and
        the jaws closed on air.

        Bounded below by the surface the object is standing on, measured by
        the detector for this object rather than configured for the table,
        so stepping down cannot press into it.
        """
        tries = max(0, int(self.get_parameter('regrasp_lower_tries').value))
        step = self.get_parameter('regrasp_lower_step').value
        if tries == 0 or step <= 0.0:
            return False
        if self._pick_surface_z is None:
            self.get_logger().info(
                'the jaws shut on nothing and there is no measured surface '
                'to step down towards safely, so this goes to the ladder')
            return False
        floor = self._pick_surface_z + self.get_parameter(
            'grasp_table_clearance').value
        quat = ctx['quat']
        for attempt in range(1, tries + 1):
            here = self.fresh_tcp()
            if here is None:
                return False
            goal_z = here[2] - step
            room = (here[2] - floor) * 1000.0
            if goal_z < floor:
                self.get_logger().warn(
                    f'the jaws shut on nothing and {floor:.3f} is as low as '
                    f'this object\'s own surface allows -- {room:.0f} mm '
                    f'below the tool, less than the {step * 1000:.0f} mm '
                    f'step. Not pressing into the table; going to the '
                    f'ladder.')
                return False
            self._set_state(
                'CLOSE_GRIPPER',
                f'shut on nothing -- {step * 1000:.0f} mm lower, try '
                f'{attempt} of {tries}')
            gap = self.get_parameter('grasp_table_clearance').value * 1000.0
            self.get_logger().info(
                f'stepping the tool down from z={here[2]:.4f} to '
                f'{goal_z:.4f} and closing again. The floor is {floor:.4f}, '
                f'which is {gap:.0f} mm above the surface the object stands '
                f'on.')
            if not self.open_gripper('OPEN_TO_RETRY'):
                return False
            if not self.descend_column(
                    (here[0], here[1]), here[2], goal_z, quat,
                    'REGRASP_LOWER',
                    uncheck_collisions=self.get_parameter(
                        'approach_ignores_octomap').value):
                self.get_logger().warn(
                    'could not step the tool down; going to the ladder')
                return False
            if self.close_gripper_to_cap():
                landed = self.fresh_tcp()
                self.get_logger().info(
                    f'gripped {attempt * step * 1000:.0f} mm below the '
                    f'commanded grasp'
                    + ('' if landed is None else f' at z={landed[2]:.4f}')
                    + '. grasp_z_offset is that much too high for this '
                      'object.')
                return True
        return False

    def _step_lift(self, ctx):
        """Straight back up, and well clear -- not just to the pre-grasp.

        The next move carries the object to the drop as a free-space plan, and
        starting that 5 cm off the surface is what dragged the gripper across
        the table.
        """
        self._set_state('LIFT')
        if not self.lift_column(ctx['grasp'], ctx['pregrasp'], ctx['quat'],
                                'LIFT'):
            self._note_failure('LIFT', (
                'the object was gripped but the arm could not lift it '
                'straight up, at any height. It is still holding it, at the '
                'grasp.'))
            return False
        # Off the column and clear -- but only if it actually is. Handing
        # the gripper back to the octomap is what makes the arm's own
        # start state collide when it is still down among the voxels of
        # the object it has just picked up, and then nothing plans. Run
        # 1789103548: the lift settled 50 mm above the grasp, this
        # released anyway, and PRE_PICK came back INVALID_MOTION_PLAN
        # three times with the tape in the jaws.
        #
        # When the tool is short, the column and the exemption are kept.
        # The staging step flies the column first and releases there.
        clear_at = (ctx['grasp'][2]
                    + self.get_parameter('retreat_height').value)
        here = self.fresh_tcp()
        if here is not None and here[2] < clear_at - 0.01:
            self.get_logger().warn(
                f'the lift reached {here[2]:.3f}, '
                f'{(clear_at - here[2]) * 1000:.0f} mm short of clear, so '
                f'the gripper keeps its octomap exemption until the '
                f'staging pose. It is inside the voxels of the object it '
                f'just picked up, and handing it back here is what makes '
                f'every plan out of it invalid.')
            return True
        self.release_column()
        return True

    def _step_verify_grasp(self, ctx):
        """Ask the detector whether the object actually left, and try again
        if it did not.

        The fingers alone cannot tell a grip from a fingertip resting on an
        edge, so the object being *gone from where it was* is the check that
        matters. When it is still sitting there, nothing was picked -- and
        the useful answer is not to abandon the cycle but to have another go
        at the same grasp: open, come back down the column, close, lift.

        Bounded, because the same grasp failing the same way three times is
        not going to work on the fourth, and each go costs a descent. After
        that it gives up cleanly and the retreat takes the arm to pre_pick
        and home.
        """
        tries = max(1, int(self.get_parameter('regrasp_attempts').value))
        for attempt in range(1, tries + 1):
            self._set_state(
                'VERIFY_GRASP',
                '' if attempt == 1 else f'try {attempt} of {tries}')
            if self.verify_grasp(ctx['point']):
                self.attach_object()
                self._held[self.arm] = {
                    'object': self.held_object_name(),
                    'offset': self._pick_hold_offset,
                }
                self._set_state('VERIFY_GRASP', 'confirmed')
                return True
            if attempt >= tries:
                break
            self._set_state(
                'REGRASP',
                f'the object has not moved, so nothing was picked '
                f'({attempt} of {tries})')
            self.get_logger().warn(
                f'the detector still sees the object where it was, so the '
                f'grasp did not take. Going back down for another try '
                f'({attempt} of {tries}).')
            if not self._regrasp(ctx):
                break

        self._note_failure('VERIFY_GRASP', (
            f'unable to pick it in {tries} tries -- the object is still '
            f'where it was after each one. The jaws are closing somewhere '
            f'the object is not: check the grasp height against this object, '
            f'and grasp_finger_min against how thin it is'))
        self._set_state('VERIFY_GRASP', f'unable to pick after {tries} tries')
        self.open_gripper()
        return False

    def _regrasp(self, ctx):
        """Open, back down the column, close, lift. True if it got that far.

        The same column the descent already proved, so this is a repeat of a
        motion known to fly rather than a fresh plan -- and the arm is at the
        top of it, which is exactly where a retry should start from.
        """
        grasp = tuple(ctx['grasp'])
        pregrasp = tuple(ctx['pregrasp'])
        quat = tuple(ctx['quat'])
        # Back on the column, so the gripper is exempt from the octomap for
        # the descent and every way out goes back up it.
        self._column = (list(grasp), list(pregrasp), list(quat))

        self._set_state('OPEN_GRIPPER', 'before another try')
        if not self.open_gripper():
            return False
        here = self.fresh_tcp()
        from_z = here[2] if here else pregrasp[2]
        if not self.descend_column(
                grasp, from_z, grasp[2], quat, 'REGRASP_DESCEND',
                uncheck_collisions=self.get_parameter(
                    'approach_ignores_octomap').value,
                linear_only=self.get_parameter('descend_linear_only').value):
            self.get_logger().warn(
                'could not get back down the column for another try')
            self.clear_the_surface('after a failed re-descent')
            return False
        self._set_state('CLOSE_GRIPPER', 'another try')
        if not self.close_gripper_to_cap():
            self.clear_the_surface('the retry also closed on nothing')
            return False
        self._set_state('LIFT', 'after another try')
        if not self.lift_column(grasp, pregrasp, quat, 'REGRASP_LIFT'):
            self.get_logger().warn('picked it up but could not lift it again')
            return False
        self.release_column()
        return True

    def request_grasps(self, point, why):
        """Ask the grasp server what it makes of the object from here.

        From the staging pose, not from home. The arm is where the descent
        will start, so the point cloud is the one the descent has to work in,
        and anything the server proposes can be checked from the posture the
        pre-flight measures from.

        Fire and forget: the server answers on /grasp/candidates when it has
        something, and nothing in the cycle waits on it.
        """
        if not self.get_parameter('request_grasps').value or not point:
            return
        self.grasp_request_pub.publish(String(data=json.dumps({
            'point': [round(float(v), 5) for v in point],
            'prompt': self.prompt,
            'why': why,
        })))

    def _step_pre_pick(self, ctx):
        """The staging posture. Above the work surface and reachable from
        both ends, which is what makes it the safe waypoint -- a direct joint
        goal from the lift to the drop came back INVALID_MOTION_PLAN three
        times, which is what a swing through the octomap looks like.

        Not fatal on its own: the step reports and the cycle carries on, and
        the retreat is where an unreachable pre_pick becomes a refusal to fly
        home. See _retreat_to_home.
        """
        self._set_state('PRE_PICK', 'staging')
        # Clear of the surface first, if the arm is still on a column: a joint
        # goal from down there is what sweeps it across the table.
        self.clear_the_surface('before the staging pose')
        # Carrying something back across the workspace is the one leg where
        # the path matters as much as the goal: the other arm is parked on
        # its own side and the map does not know it. See fly_on_own_side.
        carrying = self.holding()
        staged = self.move_to_state(PRE_PICK_STATE, ctx['states'],
                                    on_own_side=carrying)
        if not staged:
            # The same recovery the retreat has, for the same reason and
            # after the same measured failure: the usual cause of a
            # refused staging goal is the gripper still being inside the
            # octomap of the thing it was reaching for, and a few more
            # centimetres of clearance is the whole fix. Run 1789103548
            # ended a good pick 150 mm low because this path, unlike the
            # retreat, gave up after one try.
            self.get_logger().warn(
                'pre_pick was refused; lifting further and trying once '
                'more -- the usual cause is the gripper still being inside '
                'the octomap of what it just picked up')
            if self.lift_to_clearance('PRE_PICK retry'):
                staged = self.move_to_state(PRE_PICK_STATE, ctx['states'],
                                            on_own_side=carrying)
        if staged:
            # Off the column at last, wherever the lift got to.
            self.release_column()
        else:
            self.get_logger().warn(
                'could not reach pre_pick; carrying on without staging, '
                'which may sweep low across the work surface')
        # Standing where the descent starts: a good moment to ask what the
        # object looks like from here. Only on the way in -- the pre_pick on
        # the way out has the object in the gripper and nothing to grasp.
        if not self.holding():
            self.request_grasps(ctx.get('point'), 'at the staging pose')
        return True

    def why_nothing(self, payload):
        """What the detector did with the boxes it found, in one sentence.

        PaliGemma always answers something, so an empty frame is nearly
        always "boxes were found and thrown away" -- and which reason it
        was decides what to do. A box covering half the view is the model
        shrugging; a box with no usable depth is the object out of the
        depth range or in a dropout; no boxes at all is the prompt.
        """
        rejected = payload.get('rejected')
        if rejected is None:
            return ('The detector did not say what it found -- restart it to '
                    'get the reasons.')
        if not rejected:
            return (f'The model returned {payload.get("raw", 0)} box(es) and '
                    f'none of them was thrown away, so it genuinely saw '
                    f'nothing matching. Try a different word for it.')
        why = {}
        for entry in rejected:
            why[entry.get('why', 'unknown')] = why.get(
                entry.get('why', 'unknown'), 0) + 1
        parts = ', '.join(f'{count}x {reason}'
                          for reason, count in sorted(why.items()))
        return (f'{len(rejected)} box(es) were found and thrown away: '
                f'{parts}. A box that covers most of the view is the model '
                f'shrugging; one with no usable depth is out of the depth '
                f'range, in a dropout, or behind the arm.')

    def place_target(self):
        """Where to let go, from a fresh look. (detection, why).

        The drop surface is detected rather than recorded because the whole
        point of a sheet of paper is that it can be moved: put it where the
        object should end up and the robot finds it. A recorded drop pose
        cannot be moved without re-recording it, and a fixed place_position
        cannot be moved at all.

        `detection` is None when there is nothing to place onto, and `why`
        says which of the three things went wrong -- no prompt, no answer,
        or an answer with nothing in it. All three are refusals: a place
        with no target would drop the object wherever the arm happened to
        be.
        """
        asked = to_detection_prompt(self.get_parameter('place_prompt').value)
        if not asked:
            return None, 'place_prompt is empty, so there is nothing to look for'
        payload = self.detect(min_count=1, prompt=asked)
        if payload is None:
            return None, (f'the detector never answered "{asked}" -- is '
                          f'VLM/run_vlm_detector.sh running?')
        seen = payload.get('detections') or []
        if not seen:
            return None, (f'nothing matching "{asked}" is in view, so there '
                          f'is nowhere to put it down. '
                          + self.why_nothing(payload))
        if len(seen) > 1:
            self.get_logger().info(
                f'{len(seen)} matches for "{asked}"; using the nearest')
        return seen[0], ''

    def _step_drop(self, ctx):
        """Where the object is released -- whichever place_mode says."""
        mode = self.place_mode
        if mode == 'detected':
            return self._drop_on_detected(ctx)
        if mode == 'home':
            self._set_state('HOME', 'carrying the object back')
            if not self.move_to_home():
                self._note_failure('DROP', 'could not carry the object home')
                return False
        elif mode == 'position':
            place = list(self.get_parameter('place_position').value)
            quat = top_down_quat(self.get_parameter('place_yaw').value)
            above = (place[0], place[1],
                     place[2] + self.get_parameter(
                         'place_approach_height').value)
            self._set_state('OVER_BOX')
            if not self.move_to_pose(above, quat, 'OVER_BOX'):
                return False
            if not self.move_to_pose(tuple(place), quat, 'PLACE'):
                return False
        else:
            self._set_state('DROP', 'moving to the recorded drop pose')
            if not self.move_to_state(DROP_STATE, ctx['states']):
                self._note_failure('DROP', 'could not reach the drop pose')
                return False
        ctx['release_point'] = self.tcp_position()
        return True

    def move_above_drop(self, above, quat, label, tolerance=None):
        """Get the tool above the drop point without needing a pose goal.

        cuMotion takes Cartesian goals for one ee_link per bringup, so a
        pose goal can only be sent for one of the two arms -- and which arm
        a cycle uses is decided per cycle, by which half of the frame the
        object is in. A *joint* goal from IK is accepted for either, which
        is the same reason approach_above uses one to get above the object.

        Measured, run at 11:29: the object was in the left half, the cycle
        correctly switched right -> left, and then refused to run at all
        because the place was going to need a pose goal for an arm cuMotion
        does not serve. Nothing about the place actually needs one: the
        descent onto the sheet and the retreat off it both go through
        /compute_cartesian_path, which takes a link name and serves either
        arm.
        """
        if tolerance is None:
            tolerance = self.get_parameter('place_above_tolerance').value
        # Where the tool is before any of this, so the check afterwards can
        # tell an arm that went most of the way from one that never left.
        start = self.fresh_tcp()
        chosen = self.choose_approach_posture(above, quat, label)
        if chosen is not None:
            joints = list(chosen[0])
            if chosen[1] is not None:
                joints[6] = chosen[1]
            if self._move_to_joints(joints, label) and self.arrived(
                    above, label, tolerance=tolerance, start=start):
                return True
            self.get_logger().warn(
                f'{label}: the posture chosen for the drop did not put the '
                f'tool above the sheet; falling back')
        if self._pose_goals_ok is False:
            self.get_logger().error(
                f'{label}: no arm posture could be solved for the drop, and '
                f'a pose goal cannot be sent for the {self.arm} arm -- '
                f'cuMotion serves Cartesian goals for one link per bringup '
                f'and it is not this one. Relaunch with '
                f'tool_frame:={self.tcp_frame}, or move the sheet somewhere '
                f'the other arm can reach.')
            return False
        return self.move_to_pose(above, quat, label)

    def arrived(self, position, label, tolerance=0.05, start=None):
        """Did the move take the tool to -- or at least towards -- there?

        A joint goal that MoveGroup calls SUCCESS has satisfied the
        *controller*, which is not the same as the tool being where the
        caller meant. And a posture solved by IK is only as good as the IK:
        measured at 13:5x, OVER_DROP reported ok and left the tool exactly
        where it started, at the staging pose, after which every leg below
        it was computed from a place the arm was not -- an 8.2 rad
        "correction" and a descent refused for joint travel, none of which
        named the real problem.

        `start` is where the tool was before the move, and it is what
        separates that failure from an ordinary one. This robot does not
        land on a joint goal: measured, run at 14:24, OVER_DROP settled
        0.115 rad off in joint1 and put the tool 102 mm from the point it
        was aimed at -- three times, on three otherwise sound places, all
        of which a flat 50 mm check threw away. The same check exists
        because of a 300 mm miss. The difference between them is not the
        distance left but the distance closed: 198 mm in one, none at all
        in the other. So with a start position this asks whether the arm
        went where it was sent, and the leg below -- which measures where
        the tool actually is and closes the rest horizontally, at height,
        before anything descends -- finishes the job.
        """
        here = self.fresh_tcp()
        if here is None:
            self.get_logger().warn(
                f'{label}: no tool transform, so there is no telling whether '
                f'the arm arrived')
            return True
        off = dist(here, position)
        if off <= tolerance:
            return True
        if start is not None:
            had = dist(start, position)
            gained = had - off
            need = self.get_parameter('arrival_progress').value
            if gained >= need:
                self.get_logger().warn(
                    f'{label}: the tool is {off * 1000:.0f} mm from where it '
                    f'was sent, having closed {gained * 1000:.0f} mm of the '
                    f'{had * 1000:.0f} mm it had to cover. Going on from '
                    f'there: the leg below measures where the tool actually '
                    f'is rather than assuming this one landed.')
                self.log_motion(label, 'arrival', 'short',
                                target=[round(v, 5) for v in position],
                                landed=[round(v, 5) for v in here],
                                error_mm=round(off * 1000, 1),
                                closed_mm=round(gained * 1000, 1))
                return True
        self.get_logger().error(
            f'{label}: the goal succeeded but the tool is {off * 1000:.0f} mm '
            f'from where it was sent -- at '
            f'({here[0]:.3f}, {here[1]:.3f}, {here[2]:.3f}), wanted '
            f'({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), '
            f'against a tolerance of {tolerance * 1000:.0f} mm. '
            f'Everything below this would be computed from a place the arm '
            f'is not in, so it is not being flown.')
        # In the log as well as on the console. The leg above this one
        # records outcome "ok" -- it planned and the controller finished --
        # so without a row of its own a cycle that stopped here reads as a
        # cycle that stopped for no reason at all. Measured, run at 14:24:
        # three places ended at OVER_DROP with nothing after it in the log.
        self.log_motion(label, 'arrival', 'off-target',
                        target=[round(v, 5) for v in position],
                        landed=[round(v, 5) for v in here],
                        error_mm=round(off * 1000, 1),
                        tolerance_mm=round(tolerance * 1000, 1))
        return False

    def sheet_height(self, target):
        """The top of the drop surface itself.

        `point` is the median depth over the inner half of the box, which
        for a sheet of paper lying flat is the sheet. `top` is the nearest
        surface in the box and is the better number where the detector
        reports one -- a sheet with something already on it has a top that
        is not the sheet, and stacking onto that is the right answer
        anyway.
        """
        top = target.get('top')
        if top and len(top) == 3 and math.isfinite(top[2]):
            return float(top[2])
        return float(target['point'][2])

    def on_the_sheet(self, target, point):
        """Is this x-y over the detected sheet, with a margin to spare?

        Anywhere on the paper will do, which is the whole reason to ask:
        the tool lands tens of millimetres from where a joint goal aimed
        it, and sliding sideways to the exact middle costs a move and
        risks dragging the held object. If it is already over the paper,
        straight down is both simpler and safer.

        False when the detector reported no footprint, which keeps the old
        behaviour of aiming at the middle.
        """
        box = target.get('footprint')
        if not box or len(box) != 4:
            return False
        margin = self.get_parameter('place_edge_margin').value
        x1, x2 = sorted((float(box[0]), float(box[2])))
        y1, y2 = sorted((float(box[1]), float(box[3])))
        return (x1 + margin <= point[0] <= x2 - margin
                and y1 + margin <= point[1] <= y2 - margin)

    def drop_point(self, target, release_z, quat):
        """Somewhere on the sheet this arm can actually reach, or None.

        The centre first, because it is the middle of the sheet and the
        most forgiving place to put something. But the centre of a sheet
        near the edge of the envelope can be out of reach when most of the
        sheet is not, and anywhere on the paper will do -- so the corners
        of its footprint, pulled in by place_edge_margin, are tried after
        it.

        Returns None only if none of them can be reached, which is a
        refusal and not a guess: a drop aimed at a point the arm cannot
        reach either fails after a 60 s planner timeout or lands
        somewhere else.
        """
        candidates = self.sheet_candidates(target)
        if not self.get_parameter('check_reach').value:
            return (candidates[0][0], candidates[0][1], release_z)
        for index, (x, y) in enumerate(candidates):
            # The release point, not the point above it. Getting the tool
            # *above* a spot proves nothing about getting it down: measured
            # at 13:31, the approach solved, the descent then solved 58% of
            # a 92 mm vertical line and the last 36 mm had no solution at
            # all -- checked and unchecked alike, so kinematics rather than
            # collision. That spot on the sheet was at the edge of this
            # arm's envelope, and the sheet is wide enough to have others.
            above = (x, y, release_z)
            if self._pose_goals_ok is False:
                # cuMotion takes Cartesian goals for one ee_link per
                # bringup, and this arm is not it -- so what comes back is
                # about the link name, not about the point, and reading it
                # as "out of reach" refuses a drop the arm could make.
                # Measured, run at 12:16: the pick worked on the left arm,
                # the sheet was found, and the place was refused before it
                # moved because the planner had been asked a question it
                # cannot answer for that arm.
                #
                # So ask the kinematics instead. Cheap mode answers True or
                # unknown and never False, so it can confirm a point but
                # never refuse one -- which is the right bias here: the
                # move that follows is a joint goal, which is refusable on
                # its own terms.
                verdict = self.reachable(above, quat, self.arm, cheap=True)
            else:
                # cuMotion is the authority where it can be asked, because
                # cuMotion is what executes, and asking costs a plan_only
                # round trip with nothing moving.
                verdict = self.planner_can_reach(above, quat, self.arm)
            if verdict is None:
                # Could not be asked. Not a reason to refuse: a planner
                # that is merely down must not look like a sheet out of
                # reach.
                self.get_logger().warn(
                    'the planner could not be asked whether the sheet is '
                    'reachable; aiming at the middle of it and finding out')
                return (candidates[0][0], candidates[0][1], release_z)
            if verdict:
                if index:
                    centre = target['point']
                    off = math.hypot(x - centre[0], y - centre[1])
                    self.get_logger().info(
                        f'the middle of the sheet is out of reach, so '
                        f'putting it down at ({x:.3f}, {y:.3f}) instead -- '
                        f'{off * 1000:.0f} mm from the middle and still well '
                        f'inside the paper. Tried {index + 1} of '
                        f'{len(candidates)} spots.')
                return (x, y, release_z)
        self.get_logger().error(
            f'none of the {len(candidates)} spots tried across the sheet can '
            f'be reached at z={release_z:.3f} by the {self.arm} arm. The '
            f'sheet is within reach *above* but not down at it -- move it '
            f'toward this arm, or use the other one.')
        return None

    def sheet_candidates(self, target):
        """Where on the sheet to try, best first.

        A grid across the footprint rather than a handful of corners,
        ordered by distance from the middle: the first one that works is
        then the most central spot the arm can actually get down to, which
        is the best place to put something on a piece of paper.

        Inset by place_edge_margin, because the object has width and has
        to land *on* the paper. Capped at place_grid squared so a large
        sheet cannot turn into a minute of planner probes.

        Falls back to the middle alone when the detector reported no
        footprint, which is the old behaviour.
        """
        centre = list(target['point'])
        box = target.get('footprint')
        margin = self.get_parameter('place_edge_margin').value
        steps = max(1, int(self.get_parameter('place_grid').value))
        if not box or len(box) != 4:
            return [(centre[0], centre[1])]
        x1, x2 = sorted((float(box[0]), float(box[2])))
        y1, y2 = sorted((float(box[1]), float(box[3])))
        x1, x2 = x1 + margin, x2 - margin
        y1, y2 = y1 + margin, y2 - margin
        if x1 >= x2 or y1 >= y2:
            # Margins have eaten the sheet. It is too small to be choosy
            # about, so aim at the middle and let the descent decide.
            return [(centre[0], centre[1])]
        points = []
        for i in range(steps):
            for j in range(steps):
                fx = 0.5 if steps == 1 else i / (steps - 1.0)
                fy = 0.5 if steps == 1 else j / (steps - 1.0)
                points.append((x1 + (x2 - x1) * fx, y1 + (y2 - y1) * fy))
        # The detected middle first, then outwards. A sheet whose middle
        # the arm can reach should have the object put there.
        points.append((centre[0], centre[1]))
        points.sort(key=lambda p: math.hypot(p[0] - centre[0],
                                             p[1] - centre[1]))
        # Drop near-duplicates, which a grid plus the centre always has.
        unique = []
        for point in points:
            if all(math.hypot(point[0] - kept[0], point[1] - kept[1]) > 0.01
                   for kept in unique):
                unique.append(point)
        return unique

    def _drop_on_detected(self, ctx):
        """Down onto whatever the place prompt found, gently.

        The same shape as the pick's own approach and for the same reasons:
        a free-space move to a point above the target, then a *straight
        vertical line* down to it. A single pose goal for the drop is what
        swings a held object across whatever is between here and there.

        Slower than the rest of the cycle, because this is the leg where
        something held is lowered onto something else. Nothing is gained by
        arriving fast and the object is already in the jaws.
        """
        target, why = self.place_target()
        if target is None and self.get_parameter(
                'place_look_from_home').value:
            # The staging pose is above the work surface, so the arm
            # standing in it is a plausible thing to be between the camera
            # and the sheet. HOME is out of frame by definition -- it is
            # where the object was located from -- and it is a joint goal
            # from the staging pose, which is the one safe way to get
            # there.
            self.get_logger().warn(
                f'{why}. Looking again from HOME, which is out of the '
                f'camera frame.')
            self._set_state('HOME', 'looking for the drop target')
            if self.move_to_home():
                target, why = self.place_target()
                if target is not None:
                    # Back to the staging pose before approaching anything:
                    # a free-space plan from the folded HOME posture out
                    # over the table is the move that sweeps across it.
                    self._set_state('PRE_PICK', 'staging for the place')
                    if not self.move_to_state(PRE_PICK_STATE,
                                              ctx.get('states') or {},
                                              on_own_side=self.holding()):
                        self.get_logger().warn(
                            'could not get back to the staging pose; '
                            'approaching the drop from where the arm is')
        if target is None:
            self._note_failure('DROP', why)
            self.get_logger().error(f'not placing: {why}')
            return False
        quat = top_down_quat(self.get_parameter('place_yaw').value)
        clearance = self.get_parameter('place_clearance').value
        sheet_z = self.sheet_height(target)
        # Where the tool has to be for the object to be sitting a
        # centimetre above the paper. The pick measured how far the tool is
        # above the bottom of what it is holding; without that, the
        # clearance is for the tool, which is what it used to mean.
        offset = self.hold_offset()
        if offset is not None:
            release_z = sheet_z + offset + clearance
            how = (f'the object hangs {offset * 1000:.0f} mm '
                   f'below the tool, so the tool goes '
                   f'{(offset + clearance) * 1000:.0f} mm above '
                   f'the sheet to leave {clearance * 1000:.0f} mm under it')
        else:
            release_z = sheet_z + clearance
            how = (f'no hold height was measured at the pick, so this is '
                   f'{clearance * 1000:.0f} mm of clearance for the tool '
                   f'rather than for the object')
        floor = self.get_parameter('min_grasp_z').value
        if release_z < floor:
            self.get_logger().warn(
                f'the release height {release_z:.3f} is below min_grasp_z '
                f'{floor:.3f}; clamping. The drop surface was detected at '
                f'z={sheet_z:.3f} -- if that is wrong, the depth at the '
                f'sheet is wrong.')
            release_z = floor
        point = self.drop_point(target, release_z, quat)
        if point is None:
            self._note_failure('DROP', (
                'no part of the sheet is within reach of the '
                f'{self.arm} arm. Move it closer, or use the other arm.'))
            return False
        above = (point[0], point[1],
                 release_z + self.get_parameter('place_approach_height').value)
        self.get_logger().info(
            f'placing on "{self.get_parameter("place_prompt").value}" at '
            f'({point[0]:.3f}, {point[1]:.3f}), sheet at z={sheet_z:.3f}, '
            f'releasing at z={release_z:.3f} -- {how}')

        # Already reach-checked: drop_point only returns somewhere the arm
        # can put its tool. That check is what stands between this and a
        # 60 s planner timeout -- an unreachable pose goal does not come
        # back "unreachable", it does not come back at all. Measured, run
        # at 15:11: OVER_DROP never answered and move_group was wedged
        # afterwards.
        self._set_state('OVER_DROP', 'above the place target')
        if not self.move_above_drop(above, quat, 'OVER_DROP'):
            self._note_failure('DROP', (
                'could not get above the place target -- either no posture '
                'reaches it, or the arm settled too far from the one it was '
                'given. motion_log.jsonl says which: an "off-target" row '
                'is the second.'))
            return False

        # Which vertical line to come down, given where the tool actually
        # ended up.
        #
        # A joint goal lands where it lands: measured, OVER_DROP put the
        # tool 47 mm from the point it was aimed at and 63 mm low. Coming
        # down to the commanded point from there is not a descent, it is a
        # diagonal -- 14% of the line solved on one attempt and the other
        # was refused for 3.285 rad of joint travel.
        #
        # Anywhere on the paper will do, so the first question is whether
        # the tool is already over it. If it is, the line is straight down
        # from here and no sideways move happens at all. If it is not, the
        # gap is closed horizontally, up here, before descending.
        here = self.fresh_tcp()
        column = [point[0], point[1]]
        # 0 turns the correction off, and with it the check on whether the
        # correction worked -- otherwise switching off a best-effort move
        # would turn a diagonal descent into a refusal, which is not what
        # switching it off asks for.
        align_tolerance = self.get_parameter('descend_align_tolerance').value
        if here is not None:
            drift = math.hypot(here[0] - point[0], here[1] - point[1])
            if self.on_the_sheet(target, here):
                column = [here[0], here[1]]
                self.get_logger().info(
                    f'the tool landed {drift * 1000:.0f} mm from the middle '
                    f'of the sheet and is still over it, so it goes straight '
                    f'down from here rather than sliding across first')
            elif align_tolerance > 0.0 and drift > align_tolerance:
                self.get_logger().info(
                    f'the tool landed {drift * 1000:.0f} mm off the sheet\'s '
                    f'middle and outside its footprint; closing that at '
                    f'height before descending')
                self.align_over_column(
                    (point[0], point[1], release_z), quat, 'PLACE',
                    floor=release_z + self.get_parameter(
                        'place_approach_height').value * 0.3)
                here = self.fresh_tcp()
                # align_over_column is best effort, which is the right
                # answer on the way *to* an object and the wrong one on
                # the way down with one in the jaws: a correction that did
                # not fly leaves the "descent" to the commanded column a
                # diagonal across whatever is between here and there.
                # So it is measured rather than assumed.
                if here is not None:
                    left = math.hypot(here[0] - point[0],
                                      here[1] - point[1])
                    if left > align_tolerance:
                        if self.on_the_sheet(target, here):
                            column = [here[0], here[1]]
                            self.get_logger().info(
                                f'the correction left the tool '
                                f'{left * 1000:.0f} mm off the middle but '
                                f'over the sheet, so it goes straight down '
                                f'from there')
                        else:
                            self._note_failure('DROP', (
                                f'the tool could not be brought over the '
                                f'drop point: {left * 1000:.0f} mm off it '
                                f'and not over the sheet. Coming down from '
                                f'there would be a diagonal across the '
                                f'work surface with the object in the '
                                f'jaws, so nothing is being lowered. The '
                                f'sheet is at the edge of this arm\'s '
                                f'envelope -- move it closer.'))
                            return False

        from_z = here[2] if here is not None else above[2]
        # On the column from here: if the release fails, clear_the_surface
        # knows which way is up and how far, exactly as it does for a pick.
        self._column = ([column[0], column[1], release_z], list(above),
                        list(quat))
        self._set_state('PLACE', 'lowering onto the target')
        with self.at_speed(self.get_parameter('place_speed').value,
                           'lowering the object onto the drop surface'):
            lowered = self.descend_column(
                (column[0], column[1]), from_z, release_z, quat, 'PLACE',
                uncheck_collisions=self.get_parameter(
                    'place_ignores_octomap').value)
        # How far the object's own base is above the sheet, which is the
        # only number that matters here: the tool is hold_offset above the
        # base, and release_z was chosen to leave `clearance` under it. So
        # anything the descent fell short by adds directly to the drop.
        cap = self.get_parameter('place_max_drop').value

        def base_above_sheet():
            landed = self.fresh_tcp()
            if landed is None:
                return None, None
            return landed, (landed[2] - release_z) + clearance

        landed, gap = base_above_sheet()
        if gap is not None and gap > cap and landed is not None:
            # One more short line at the remainder. The descent gives up
            # when a *segment* will not solve, and the leftover is often a
            # few millimetres that a single shorter hop manages -- the same
            # reason the pick closes its own gap at the end.
            self.get_logger().info(
                f'the object is {gap * 1000:.0f} mm above the sheet, over '
                f'the {cap * 1000:.0f} mm it may be dropped from. Closing '
                f'the last {(landed[2] - release_z) * 1000:.0f} mm.')
            with self.at_speed(self.get_parameter('place_speed').value,
                               'closing the last of the descent'):
                self.descend_column(
                    (column[0], column[1]), landed[2], release_z, quat,
                    'PLACE_GAP',
                    uncheck_collisions=self.get_parameter(
                        'place_ignores_octomap').value)
            landed, gap = base_above_sheet()

        if gap is None:
            self.get_logger().warn(
                'no tool transform at the drop, so how high the object is '
                'above the sheet cannot be checked. Letting go anyway -- '
                'the descent is the only evidence there is.')
        elif gap > cap:
            self._note_failure('DROP', (
                f'the object is {gap * 1000:.0f} mm above the sheet and only '
                f'{cap * 1000:.0f} mm may be dropped (place_max_drop). The '
                f'descent to z={release_z:.3f} would not go the whole way. '
                f'Raise place_max_drop to accept the drop, or move the sheet '
                f'somewhere this arm can reach further down.'))
            return False
        else:
            self.get_logger().info(
                f'the object is {gap * 1000:.0f} mm above the sheet, inside '
                f'the {cap * 1000:.0f} mm cap -- setting it down')
        ctx['release_point'] = self.tcp_position()
        # The line it actually came down, so the retreat goes back up the
        # same one. Using the commanded point instead makes the way out a
        # diagonal across the sheet with the jaws open around the object
        # that has just been set on it.
        ctx['place_above'] = [column[0], column[1], above[2]]
        ctx['place_quat'] = list(quat)
        return True

    def _step_release(self, ctx):
        self._set_state('RELEASE', f'at the {self.place_mode} drop')
        if not self.open_gripper():
            return False
        self.detach_object()
        # The object is down from this line on, whatever happens next. The
        # retreat below can still fail -- and a failed retreat is an arm
        # short of home, not a place that did not happen. Reporting the
        # second when it is the first is what once took the motors off a
        # cycle that had just placed a roll of tape.
        ctx['released'] = True
        if self.place_mode == 'detected':
            # Straight back up the line it came down, before anything else
            # plans: the jaws are open around an object standing on the
            # surface, and any move that is not vertical takes it with them.
            above = ctx.get('place_above')
            quat = ctx.get('place_quat')
            here = self.tcp_position()
            self._set_state('RETREAT', 'off the drop surface')
            if above is None or quat is None or here is None:
                self.clear_the_surface('after the release')
                return True
            with self.at_speed(
                    self.get_parameter('place_retreat_speed').value,
                    'lifting away from the placed object'):
                if not self.descend_column((above[0], above[1]), here[2],
                                           above[2], tuple(quat), 'RETREAT',
                                           uncheck_collisions=True):
                    self.get_logger().warn(
                        'could not retreat straight up from the release; '
                        'the object is down and the arm is still over it')
                    return False
            self.release_column()
            return True
        if self.place_mode == 'position':
            place = list(self.get_parameter('place_position').value)
            quat = top_down_quat(self.get_parameter('place_yaw').value)
            above = (place[0], place[1],
                     place[2] + self.get_parameter(
                         'place_approach_height').value)
            self._set_state('RETREAT')
            if not self.move_to_pose(above, quat, 'RETREAT'):
                return False
        return True

    def _step_verify_place(self, ctx):
        """Advisory: the box usually occludes the object, so a negative here
        is not a reason to call the cycle failed."""
        self._set_state('VERIFY_PLACE')
        release_point = ctx.get('release_point')
        if release_point is None:
            self.get_logger().warn(
                f'no {self.tcp_frame} transform at release; skipping the '
                f'check')
        else:
            self.verify_place(release_point)
        return True

    def _step_shut_jaws(self, ctx):
        """The trip back is made with the jaws shut rather than with 44 mm of
        open fingers hunting for something to catch on."""
        self._set_state('CLOSE_GRIPPER', 'shutting the jaws')
        if not self.close_gripper('CLOSE_AT_DROP'):
            # Not fatal. The object is placed and the cycle has done its job;
            # an open gripper on the way home is untidy, not unsafe.
            self.get_logger().warn(
                'could not shut the jaws; going on with the gripper open')
        return True

    def _step_home(self, ctx):
        """Home, and only ever from the staging pose.

        A joint goal from a low, extended posture is the one move that drags
        the arm across the table, so HOME is reached from pre_pick or not at
        all -- see _retreat_to_home.

        But when the sequence has *just* run its own pre_pick step, going
        through the retreat asks for it a second time, and the second ask is
        a fresh planning problem that can fail on its own. Measured, run at
        14:07: a completed pick and place, then PRE_PICK_STATE twice with -2,
        the cycle declared failed and the motors taken off -- and
        safe_shutdown then reached pre_pick on its very next attempt. The
        planner is stochastic here; the codebase carries plan_attempts for
        exactly that. Asking twice as often is asking to be unlucky twice.
        """
        if self.at_state(PRE_PICK_STATE, ctx['states']):
            self._set_state('HOME', ctx.get('why', 'cycle complete'))
            return self.move_to_home()
        return self._retreat_to_home(ctx['states'], ctx.get('why', 'the '
                                                            'sequence says so'))

    def at_state(self, name, states):
        """Is the arm already standing at a recorded posture?"""
        entry = (states or {}).get(name) or {}
        wanted = entry.get('joints')
        if not wanted:
            return False
        error = self.joint_error(list(wanted))
        if error is None:
            return False
        return error <= self.get_parameter('at_goal_tolerance').value

    STEP_HANDLERS = {
        'open_gripper': _step_open_gripper,
        'descend': _step_descend,
        'grasp': _step_grasp,
        'lift': _step_lift,
        'verify_grasp': _step_verify_grasp,
        'pre_pick': _step_pre_pick,
        'drop': _step_drop,
        'release': _step_release,
        'verify_place': _step_verify_place,
        'shut_jaws': _step_shut_jaws,
        'home': _step_home,
    }

    def align_over_column(self, grasp, quat, label, floor=None):
        """Move the tool onto the vertical line above the grasp, at height.

        A purely horizontal leg: same z, same orientation, x and y to the
        column. Above the work surface by the whole transit height, so it
        cannot reach anything standing on it -- which is the point, because
        the same correction made *during* the descent is a slant across the
        object, and the same correction made low is a swipe at it.

        Best effort. A correction that will not fly leaves the descent no
        worse than it was, and it is logged either way: the residual is the
        number that explains a grasp closing off-target.
        """
        tolerance = self.get_parameter('descend_align_tolerance').value
        if tolerance <= 0.0:
            return True
        here = self.fresh_tcp()
        if here is None:
            return True
        # High enough that a horizontal leg cannot reach anything standing
        # on the surface. This is the whole safety argument for correcting
        # here rather than lower down, so it is checked rather than
        # assumed: a sideways move a centimetre above the object is the
        # move that knocks it over.
        #
        # The transit height is the bar, because that is what the transit
        # height *is*: the height the approach ends at, chosen to be well
        # clear of anything on the surface. Using the retreat height here
        # -- which is higher -- meant the correction never ran at all: the
        # descent starts 150 mm above the grasp and the guard wanted 200,
        # so every descent went back to being a slant with a note in the
        # log explaining why it was safe not to fix it.
        if floor is None:
            floor = grasp[2] + min(
                self.get_parameter('transit_height').value,
                self.get_parameter('retreat_height').value)
        if here[2] < floor - 1e-6:
            self.get_logger().info(
                f'{label}: not correcting sideways at z={here[2]:.3f} -- '
                f'that is below {floor:.3f} and a horizontal leg down there '
                f'sweeps whatever is on the surface. Descending from where '
                f'the tool is.')
            return True
        off = math.hypot(here[0] - grasp[0], here[1] - grasp[1])
        if off <= tolerance:
            self.get_logger().info(
                f'{label}: on the column to {off * 1000:.1f} mm, descending '
                f'straight down')
            return True
        self.get_logger().info(
            f'{label}: the tool is {off * 1000:.1f} mm off the column '
            f'(dx {(grasp[0] - here[0]) * 1000:+.1f}, '
            f'dy {(grasp[1] - here[1]) * 1000:+.1f} mm). Closing that at '
            f'this height first -- carried into the descent it becomes a '
            f'slant through the object.')
        target = (grasp[0], grasp[1], here[2])
        # Collision-checked, unlike the column legs below it. Those are
        # exempt because their *target* is inside the octomap -- the object
        # being picked up is in the map and the gripper has to enter its
        # voxels. This one ends in clear air a retreat height above the
        # surface, so there is nothing to excuse and the map is worth
        # consulting.
        flown = self.converge_to(
            target, quat, f'{label}_ALIGN', avoid_collisions=True,
            max_travel=self.get_parameter('column_max_joint_travel').value)
        landed = self.fresh_tcp()
        if landed is not None:
            left = math.hypot(landed[0] - grasp[0], landed[1] - grasp[1])
            self.get_logger().info(
                f'{label}: {left * 1000:.1f} mm off the column after the '
                f'correction' + ('' if flown else ' (which did not fly)'))
        return bool(flown)

    def descend_orientation(self, grasp, quat, label='DESCEND'):
        """The orientation whose descent flies from here, or None.

        Probed from the arm's *measured* posture with an explicit start
        state, so it answers about the descent that is actually about to
        happen rather than the one the pre-flight modelled from the staging
        pose. Costs one service call per orientation and no motion.

        Returns the pre-flight's own choice untouched when that still works,
        which is the common case -- this is a correction for the landing
        error, not a second opinion on the grasp.
        """
        if not self.get_parameter('descend_recheck').value:
            return quat
        if not self.await_joint_states():
            return quat
        with self._lock:
            here = [self._arm_positions.get(j) for j in self.arm_joints]
        if any(v is None for v in here):
            return quat

        budget = self.get_parameter('column_max_joint_travel').value
        wanted = self.get_parameter('cartesian_min_fraction').value
        may_uncheck = self.get_parameter('descend_ignores_octomap').value
        options = [quat] + [q for q in self.grasp_quat_options(quat)
                            if tuple(q) != tuple(quat)]

        def flies(candidate):
            """Would the descent take this orientation, asked its way?

            Checked first and unchecked only as a fallback, because that is
            exactly what _descend_column does -- a probe that asks a
            different question from the leg it precedes is not a prediction
            of anything. The object being reached for is often in the
            octomap, which is why the unchecked retry exists at all.
            """
            for checked in ((True, False) if may_uncheck else (True,)):
                fraction, _end, cost = self.probe_cartesian(
                    here, grasp, candidate, avoid_collisions=checked)
                if fraction is None:
                    return None        # nothing could answer
                if fraction >= wanted and (budget <= 0.0 or cost is None
                                           or cost <= budget):
                    return (fraction, cost)
            return False

        for index, candidate in enumerate(options):
            verdict = flies(candidate)
            if verdict is None:
                return quat            # nothing could answer; carry on as was
            if verdict:
                fraction, cost = verdict
                if index:
                    self.log_motion(label, 'recheck', 'reoriented',
                                    candidate=index + 1, of=len(options),
                                    fraction=round(fraction, 4),
                                    travel_rad=(None if cost is None
                                                else round(cost, 3)))
                return candidate
        self.get_logger().warn(
            f'{label}: no orientation descends from where the arm actually '
            f'landed -- {len(options)} tried. The descent below will fail, '
            f'and the reason is the landing error rather than the grasp: '
            f'compare the commanded and measured joints at TRANSIT.')
        self.log_motion(label, 'recheck', 'none-fly', of=len(options))
        return quat

    def close_descent_gap(self, grasp, quat):
        """Fly whatever height the descent did not.

        Reads where the tool actually is and, if it is sitting above the
        commanded grasp, continues straight down the same column to it. See
        descend_close_gap for why this is not a second descent and why
        grasp_z_offset cannot do the job instead.

        Advisory: a gap that will not fly leaves the jaws where they are and
        the close reports off-target, which is the same outcome as before
        this existed.
        """
        if not self.get_parameter('descend_close_gap').value:
            return False
        here = self.fresh_tcp()
        if here is None:
            self.get_logger().warn(
                'no fresh tool transform after the descent, so the height it '
                'actually reached is unknown; closing where it stands')
            return False
        gap = here[2] - grasp[2]
        if gap <= self.get_parameter('descend_gap_tolerance').value:
            return False
        limit = self.get_parameter('descend_gap_max').value
        if gap > limit:
            self.get_logger().error(
                f'the descent stopped {gap * 1000:.0f} mm above the grasp, '
                f'more than the {limit * 1000:.0f} mm descend_gap_max allows '
                f'for tracking error. That is not the controller falling '
                f'short -- check the commanded grasp height against the '
                f'object, and the DESCEND record\'s fraction and '
                f'plan_error_mm. Closing where it stands.')
            self.log_motion('DESCEND_GAP', 'cartesian', 'refused-too-far',
                            target=[round(v, 5) for v in grasp],
                            tcp=[round(v, 5) for v in here],
                            error_mm=round(gap * 1000, 1))
            return False
        self.get_logger().info(
            f'the descent stopped {gap * 1000:.0f} mm above the grasp with '
            f'its plan ending on the point, so the controller did not finish '
            f'following it; flying the remainder straight down.')
        return bool(self.descend_column(
            here, here[2], grasp[2], quat, 'DESCEND_GAP',
            uncheck_collisions=self.get_parameter(
                'approach_ignores_octomap').value,
            linear_only=self.get_parameter('descend_linear_only').value))

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
        top = start[2] + self.get_parameter('retreat_height').value
        heights = [top]
        if pregrasp is not None:
            heights.append(pregrasp[2])
        for index, retreat_z in enumerate(heights):
            if retreat_z <= start[2] + 1e-6:
                continue
            if index:
                self.get_logger().warn(
                    f'{label}: no line to {heights[0] - start[2]:.2f} m up; '
                    f'retreating to {retreat_z - start[2]:.2f} m instead, '
                    f'then climbing what is left in steps.')
            if self.descend_column(
                    start, start[2], retreat_z, quat, label,
                    uncheck_collisions=self.get_parameter(
                        'approach_ignores_octomap').value,
                    linear_only=self.get_parameter(
                        'descend_linear_only').value):
                if retreat_z < top - 1e-6:
                    self.climb_in_steps(top, quat, label)
                return True
        return False

    def climb_in_steps(self, top, quat, label):
        """Keep rising towards `top`, one short line at a time.

        The joint-travel budget is per line, and that is the whole reason
        this works where the single long line did not. Measured, run
        1789103548: 200 mm straight up from the grasp was refused at
        3.582 rad against a 1.5 rad budget, so the lift settled for the
        pre-grasp height -- 50 mm -- and left the gripper inside the
        octomap voxels of the roll it had just picked up. Every plan out of
        there was then INVALID_MOTION_PLAN, three times, and the arm
        finished the pick 150 mm lower than it was supposed to.

        Four 50 mm lines cost about a quarter of the joint travel each, and
        a line that will not go simply ends the climb where it is: this is
        a best-effort improvement on a height already reached, never a
        reason to fail a lift that has otherwise worked.
        """
        step = max(self.get_parameter('descend_step').value, 0.01)
        while not self._abort.is_set():
            here = self.fresh_tcp()
            if here is None or here[2] >= top - 1e-3:
                return
            goal_z = min(here[2] + step, top)
            if not self.descend_column(
                    here, here[2], goal_z, quat, f'{label}_STEP',
                    uncheck_collisions=self.get_parameter(
                        'approach_ignores_octomap').value,
                    linear_only=self.get_parameter(
                        'descend_linear_only').value):
                self.get_logger().warn(
                    f'{label}: stopped climbing at {here[2]:.3f}, '
                    f'{(top - here[2]) * 1000:.0f} mm short of clear. '
                    f'Whatever moves next starts lower, so watch the '
                    f'clearance.')
                return

    def release_column(self):
        """The arm is off the column: forget it, and re-arm the octomap.

        The gripper's octomap exemption is held for the whole time the arm is
        down among the voxels -- the descent, the close, and the way back up
        -- because handing the jaws back to collision checking while they are
        still inside the map makes the arm's own start state invalid and
        every subsequent plan fails on it. This is the one place that ends
        that, so the two cannot get out of step.
        """
        self._column = None
        if self._octomap_exempt:
            self.allow_gripper_in_octomap(False)
            self._octomap_exempt = False

    def lift_to_clearance(self, label):
        """Get the tool higher, straight up from wherever it is.

        The recovery when a joint goal is refused from down near the work
        surface. The usual cause is the gripper still being inside the
        octomap of the object it was reaching for, which makes the arm's own
        start state invalid and every plan out of it fail -- and a few more
        centimetres of clearance is the whole fix.

        Distinct from clear_the_surface, which flies the recorded column and
        is a no-op once the tool is above the pre-grasp height. This one has
        no column to work from: it goes up from here, with the gripper exempt
        from the octomap for the leg, because the map is exactly what is in
        the way.
        """
        here = self.fresh_tcp()
        if here is None:
            self.get_logger().warn(
                f'{label}: no tool transform, so there is no telling which '
                f'way is up')
            return False
        # The attitude it is already in. Re-deriving a top-down quaternion
        # here would rotate the wrist while rising, which is not what "get
        # out of the way" means.
        quat = self.tcp_orientation() or top_down_quat(0.0)
        exempt = False
        if self.get_parameter('gripper_octomap_exemption').value:
            exempt = self.allow_gripper_in_octomap(True)
            self._octomap_exempt = exempt
        try:
            return bool(self.lift_column(list(here), None, quat, label))
        finally:
            if exempt:
                self.allow_gripper_in_octomap(False)
                self._octomap_exempt = False

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
        # The higher of the two, and the retreat height is the one that
        # matters. Settling for the pre-grasp -- 50 mm above the grasp --
        # is what handed the gripper back to the octomap while it was
        # still inside the voxels of the object it had just picked up, and
        # every plan from there came back INVALID_MOTION_PLAN.
        clear_at = grasp[2] + self.get_parameter('retreat_height').value
        if pregrasp is not None:
            clear_at = max(clear_at, pregrasp[2])
        # Where the arm actually is, not where it was told to be: a descent
        # that failed stopped somewhere between the two, and one that worked
        # still lands tens of millimetres out.
        here = self.tcp_position()
        if here is not None and here[2] >= clear_at - 0.01:
            self.release_column()
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
        self.release_column()
        return lifted


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
