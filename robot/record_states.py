#!/usr/bin/env python3
"""Record named arm poses into a YAML file for the pick-and-place cycle.

Jog the arm where you want it -- the RViz MotionPlanning panel, drag the
interactive marker or set the sliders on the Joints tab, then Plan & Execute --
and this snapshots where it actually ended up. Reading joint values back off the
robot beats typing them in: the Joints tab only shows whole degrees, and an
executed plan lands near the goal rather than exactly on it.

    python3 record_states.py --arm right              # walk through all three
    python3 record_states.py --arm left --missing     # only what is not on file
    python3 record_states.py --arm left drop_state    # re-record just one
    python3 record_states.py --arm left --list        # show what is on file
    python3 record_states.py --arm left --mirror      # derive from the right arm
    python3 record_states.py --arm left --play        # drive them, then come back

Each arm gets its own file, because the arms are mirrored: the same joint values
are a different posture on the other arm. Three poses drive a cycle:

    home_state       where the arm rests, observes from, and returns to. The
                     object is located from here and it is the only pose the
                     octomap is captured from, so the arm has to be out of the
                     camera's view of the table. Needed per-arm only when
                     arm_selection:=by_side can pick either arm; otherwise the
                     home_joint_positions parameter covers it. (Recordings that
                     predate the HOME/READY merge call this ready_state, and
                     that name is still read.)
    pre_pick_state   staging pose between the observation pose and the object.
                     The arm goes here after the object has been located, so the
                     approach to the object starts from a known posture instead
                     of from wherever the observation pose left the elbow.
    drop_state       where the object is released at the end of the cycle.

Each is stored as joint positions, which is what the orchestrator replays: a
joint goal is reproducible, and it is the one goal type cuMotion accepts for
either arm regardless of which one its ee_link points at. The tool position is
recorded alongside it for reference only -- it is what tells you how far the
object will fall from drop_state.

--mirror derives one arm's poses from the other's instead of jogging to them.
The arms are mirror images, and the reflection negates every joint except
joint4 -- see MIRRORED_JOINTS for how that was established, and for the two
near-miss rules it rules out. Every mirrored value is checked against the target
arm's own limits, which are not the mirror of the source arm's.

--play drives the arm through the poses on file and returns it to wherever it
started, so a recording can be seen rather than read. It is the only thing here
that commands motion, it asks before doing so, and it plans through move_group
like everything else -- so the octomap and self-collision still apply.

Needs the robot up (native/run_launch_everything.sh) so that /joint_states and
TF are live. --list works with the robot down.
"""

import argparse
import io
import os
import sys
import threading
from datetime import datetime

import rclpy
import yaml
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

WS = os.path.dirname(os.path.realpath(__file__))
LEGACY_FILE = os.path.join(WS, 'pick_place_states.yaml')

# Which joints change sign when a pose is reflected to the other arm, as
# zero-based indices into joint1..joint7: every joint except joint4.
#
# Measured, not reasoned about. All 128 sign patterns were scored by forward
# kinematics off the generated URDF against 15 postures -- the three recorded
# poses plus twelve random ones -- asking which pattern puts the other arm's
# tool at the y-mirror of this arm's tool with the whole rotation mirrored too.
# Flipping all but joint4 is exact (0.00 mm, 0.0000 rotation) and unique; the
# nearest rival is 344 mm out. joint4 is the elbow, the one axis the reflection
# maps to itself with the same sense.
#
# Two plausible-looking rules are wrong, and both fail only on poses that use
# the middle of the arm. Negating joint1 alone is 138 mm out and tips the tool
# 45 degrees the wrong way. Negating joint1/3/5/7 -- the joints that turn about
# z, plus joint7 whose axis is the one thing the description itself mirrors --
# is exact to 0.1 mm on home_state and pre_pick_state, because both hold
# joint2 and joint6 within 0.0003 rad of zero, and then misses drop_state by
# 107 mm, where joint2 is 0.185 and joint6 is 0.536. Hence the random postures
# in the sweep: three similar recordings cannot tell these rules apart.
MIRRORED_JOINTS = (0, 1, 2, 4, 5, 6)


# See RETRYABLE_MOVEIT_CODES in pick_place_orchestrator.py for the measurement
# behind this: cuMotion's optimiser returned TRAJOPT_FAIL on 20 of 138 joint
# goals, on poses that planned fine on other tries, and a resend recovers it.
RETRYABLE_MOVEIT_CODES = (-1, -2, -6)   # PLANNING_FAILED, INVALID_MOTION_PLAN, TIMED_OUT


def mirror_joints(values):
    """The same posture on the other arm."""
    return [(-v if i in MIRRORED_JOINTS else v) for i, v in enumerate(values)]


def other_arm(arm):
    return 'left' if arm == 'right' else 'right'


def default_file(arm):
    """One file per arm.

    The arms are mirrored, so a pose recorded on one is a different posture on
    the other -- they cannot share a file, and the orchestrator refuses to
    replay a recording made for the other arm.
    """
    return os.path.join(WS, f'pick_place_states_{arm}.yaml')


# Ordered: this is the sequence the walkthrough asks for them in, and the order
# they are used in during a cycle.
DEFAULT_STATES = ['home_state', 'pre_pick_state', 'drop_state']

DESCRIPTIONS = {
    'home_state': 'rest and observation pose -- the arm must be clear of the '
                  'camera here, and this is the only pose the octomap is '
                  'captured from',
    'pre_pick_state': 'staging pose the arm passes through on its way to the object',
    'drop_state': 'where the object is released -- mind the drop height under it',
}


class StateRecorder(Node):

    def __init__(self, arm):
        super().__init__('record_states')
        self.arm = arm
        self.tcp_frame = f'openarm_{arm}_hand_tcp'
        self.arm_joints = [f'openarm_{arm}_joint{i}' for i in range(1, 8)]

        self._lock = threading.Lock()
        self._positions = {}
        self._urdf = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        # Latched by robot_state_publisher, so this arrives even though we
        # subscribe long after it was published.
        self.create_subscription(
            String, '/robot_description', self._on_urdf,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        # Only ever asked whether it has a server; goals go through move_group.
        self.cumotion_client = ActionClient(self, MoveGroup,
                                            '/cumotion/move_group')

    def _on_urdf(self, msg):
        with self._lock:
            self._urdf = msg.data

    def _on_joint_states(self, msg):
        with self._lock:
            for joint in self.arm_joints:
                if joint in msg.name:
                    self._positions[joint] = msg.position[msg.name.index(joint)]

    def wait_for_joint_states(self, timeout=15.0):
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        while self.get_clock().now().nanoseconds * 1e-9 < deadline:
            with self._lock:
                if all(j in self._positions for j in self.arm_joints):
                    return True
            threading.Event().wait(0.1)
        return False

    def joint_limits(self, timeout=3.0):
        """This arm's position limits, {joint: (lower, upper)}, from the URDF.

        Read rather than assumed, because the two arms' limits are not mirror
        images of each other: the left joint2 runs [-3.316, +0.175] against the
        right's [-0.175, +3.316]. A pose that is legal on one arm can be out of
        range once reflected, which is exactly what --mirror has to catch.

        Prefers /robot_description, so what is checked is what the running
        robot was actually launched with. Falls back to generating the URDF
        from xacro, so mirroring works before the robot is up. Returns {} if
        neither worked, and callers carry on with a warning.
        """
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        urdf = None
        while self.get_clock().now().nanoseconds * 1e-9 < deadline:
            with self._lock:
                urdf = self._urdf
            if urdf:
                break
            threading.Event().wait(0.1)
        if not urdf:
            urdf = self._urdf_from_xacro()
        if not urdf:
            return {}

        import xml.etree.ElementTree as ET
        limits = {}
        try:
            for joint in ET.fromstring(urdf).findall('joint'):
                if joint.get('name') not in self.arm_joints:
                    continue
                limit = joint.find('limit')
                if limit is None:
                    continue
                limits[joint.get('name')] = (float(limit.get('lower')),
                                             float(limit.get('upper')))
        except ET.ParseError as exc:
            self.get_logger().warn(f'could not parse the robot description: {exc}')
            return {}
        return limits

    def _urdf_from_xacro(self):
        """Generate the description locally, for when the robot is not up."""
        import subprocess
        xacro_file = os.path.join(
            WS, 'src/openarm_description/urdf/robot/v10.urdf.xacro')
        if not os.path.exists(xacro_file):
            return None
        try:
            done = subprocess.run(['xacro', xacro_file, 'bimanual:=true'],
                                  capture_output=True, text=True, timeout=120)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if done.returncode != 0:
            return None
        print('note: no /robot_description, so limits were checked against '
              'the URDF generated from xacro')
        return done.stdout

    def _await(self, future, timeout):
        """Block the main thread on a future while the executor spins."""
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return None
        return future.result()

    def move_to_joints(self, values, label, pipeline, speed, timeout=90.0,
                       attempts=3):
        """Plan and execute a joint goal, resending it if planning wobbles.

        Deliberately the same path the orchestrator uses -- a joint goal on this
        arm's group -- so what --play shows is what a cycle will do, octomap and
        self-collision checks included. A joint goal also works on either arm
        whichever one cuMotion's ee_link points at, which a Cartesian goal does
        not: the planner turns a joint goal into plan_single_js on the merged
        full-body state, and only a *pose* goal has to name its single ee_link.

        The retry is not optional polish. cuMotion fails to converge on about
        one goal in seven, so a three-pose playback sent once each would fail
        about a third of the time for no reason at all.
        """
        if not self.move_client.wait_for_server(timeout_sec=10.0):
            print('error: /move_action unavailable -- is move_group running?',
                  file=sys.stderr)
            return False

        for attempt in range(1, max(1, attempts) + 1):
            code = self._move_once(values, label, pipeline, speed, timeout)
            if code == MoveItErrorCodes.SUCCESS:
                if attempt > 1:
                    print(f'   (planned on attempt {attempt} of {attempts})')
                return True
            if code not in RETRYABLE_MOVEIT_CODES:
                print(f'error: {label}: move_group error code {code} '
                      f'({describe_code(code)}), not retryable',
                      file=sys.stderr)
                return False
            if attempt < attempts:
                print(f'   {label}: error code {code} '
                      f'({describe_code(code)}), resending')
        print(f'error: {label}: still failing after {attempts} attempts '
              f'(last code {code}, {describe_code(code)})', file=sys.stderr)
        if code == MoveItErrorCodes.TIMED_OUT:
            print('       a run of timeouts usually means the cuMotion node '
                  'is gone. Check with:\n'
                  '           ros2 action info /cumotion/move_group\n'
                  '       "Action servers: 0" means it died -- it has hit '
                  'SIGFPE before -- and the fix is to relaunch the robot.',
                  file=sys.stderr)
        return False

    def _move_once(self, values, label, pipeline, speed, timeout):
        """One plan-and-execute attempt. Returns a MoveIt error code."""
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = f'{self.arm}_arm'
        request.pipeline_id = pipeline
        request.num_planning_attempts = 1
        request.allowed_planning_time = 10.0
        request.max_velocity_scaling_factor = speed
        request.max_acceleration_scaling_factor = speed
        request.start_state.is_diff = True
        request.workspace_parameters.header.frame_id = 'world'
        for corner, sign in ((request.workspace_parameters.min_corner, -1.0),
                             (request.workspace_parameters.max_corner, 1.0)):
            corner.x = corner.y = corner.z = sign * 1.5

        constraints = Constraints()
        for name, value in zip(self.arm_joints, values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        request.goal_constraints = [constraints]

        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        handle = self._await(self.move_client.send_goal_async(goal), 15.0)
        if handle is None or not handle.accepted:
            print(f'error: {label}: goal rejected by move_group', file=sys.stderr)
            return MoveItErrorCodes.FAILURE
        result = self._await(handle.get_result_async(), timeout)
        if result is None:
            # It may still be executing, so do not resend and race it.
            print(f'error: {label}: no result after {timeout:.0f} s',
                  file=sys.stderr)
            return MoveItErrorCodes.FAILURE
        return result.result.error_code.val

    def planner_is_serving(self, pipeline, timeout=3.0):
        """Is the pipeline's planner actually there?

        cuMotion plans in a separate node that move_group only holds a client
        for, so when it dies move_group stays up and every goal comes back
        PLANNING_FAILED or TIMED_OUT -- which reads as "that pose is bad" when
        the truth is that nothing is planning at all. It has died mid-run here,
        with SIGFPE, after planning five goals perfectly well.

        Asked through the action client, not the topic list: the feedback topic
        exists as soon as move_group subscribes to it, server or no server.
        """
        if pipeline != 'cumotion':
            return True
        return self.cumotion_client.wait_for_server(timeout_sec=timeout)

    def snapshot(self):
        """Current joint positions, plus the tool position if TF can supply it."""
        with self._lock:
            missing = [j for j in self.arm_joints if j not in self._positions]
            joints = [self._positions.get(j) for j in self.arm_joints]
        if missing:
            raise RuntimeError(f'no /joint_states for {missing}')

        tcp = None
        try:
            tf = self.tf_buffer.lookup_transform(
                'world', self.tcp_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
            t = tf.transform.translation
            tcp = [round(t.x, 4), round(t.y, 4), round(t.z, 4)]
        except Exception as exc:                       # noqa: BLE001 - advisory only
            self.get_logger().warn(f'no {self.tcp_frame} transform: {exc}')

        return {
            'joints': [round(v, 6) for v in joints],
            'joint_names': list(self.arm_joints),
            'tcp_xyz': tcp,
            'recorded': datetime.now().isoformat(timespec='seconds'),
        }


def load_states(path):
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    return data.get('states', {}) or {}


def write_states(path, arm, states):
    """Rewrite the file with `states`, preserving what is not being re-recorded."""
    document = {
        'arm': arm,
        'states': states,
    }
    header = (
        '# Arm poses for the VLM pick-and-place cycle.\n'
        '# Written by record_states.py -- re-record with:\n'
        '#     python3 record_states.py <state_name>\n'
        '# joints are openarm_<arm>_joint1..joint7 in radians. tcp_xyz is where\n'
        '# that pose puts the tool, in the world frame, and is informational.\n'
    )
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as handle:
        handle.write(header)
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=None)
    os.replace(tmp, path)                    # atomic: never leave a half-written file


def fmt_joints(values):
    return '[' + ', '.join(f'{v:+.3f}' for v in values) + ']'


def describe(name, entry):
    tcp = entry.get('tcp_xyz')
    where = f'  tool at {tcp}' if tcp else '  tool position not recorded'
    source = entry.get('mirrored_from')
    origin = f'  mirrored from {source}' if source else ''
    lines = [name, f'  joints {fmt_joints(entry["joints"])}', where]
    if origin:
        lines.append(origin)
    lines.append(f'  recorded {entry.get("recorded")}')
    return '\n'.join(lines)


MOVEIT_CODE_NAMES = {
    1: 'SUCCESS', -1: 'PLANNING_FAILED', -2: 'INVALID_MOTION_PLAN',
    -3: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE', -4: 'CONTROL_FAILED',
    -5: 'UNABLE_TO_AQUIRE_SENSOR_DATA', -6: 'TIMED_OUT', -7: 'PREEMPTED',
    -10: 'START_STATE_IN_COLLISION', -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
    -12: 'GOAL_IN_COLLISION', -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
    -14: 'GOAL_CONSTRAINTS_VIOLATED', -15: 'INVALID_GROUP_NAME',
    -16: 'INVALID_GOAL_CONSTRAINTS', -17: 'INVALID_ROBOT_STATE',
    -18: 'INVALID_LINK_NAME', -19: 'INVALID_OBJECT_NAME',
    -21: 'FRAME_TRANSFORM_FAILURE', -31: 'NO_IK_SOLUTION',
    99999: 'FAILURE',
}


def describe_code(code):
    """A bare MoveIt error code says nothing to anyone; name it."""
    return MOVEIT_CODE_NAMES.get(code, 'unknown')


def mirror_states(node, arm, wanted, existing):
    """Derive this arm's poses from the other arm's recording.

    Returns (states, ok). Out-of-range poses are reported and dropped rather
    than written: a joint goal past a limit is rejected by move_group anyway,
    and a file that silently held one would fail at pick time instead.
    """
    source_arm = other_arm(arm)
    source_path = default_file(source_arm)
    source = load_states(source_path)
    if not source:
        print(f'error: nothing to mirror -- {source_path} holds no states. '
              f'Record the {source_arm} arm first.', file=sys.stderr)
        return existing, False

    limits = node.joint_limits()
    if not limits:
        print('warning: no /robot_description, so joint limits were NOT '
              'checked. Plan before you execute.')

    states = dict(existing)
    ok = True
    for name in wanted:
        entry = source.get(name)
        if not entry or not entry.get('joints'):
            print(f'  {name}: not recorded on the {source_arm} arm, skipped')
            continue
        joints = mirror_joints(entry['joints'])

        outside = []
        for joint, value in zip(node.arm_joints, joints):
            low, high = limits.get(joint, (None, None))
            if low is not None and not low <= value <= high:
                outside.append(f'{joint} {value:+.4f} outside [{low:+.3f}, '
                               f'{high:+.3f}]')
        if outside:
            ok = False
            print(f'  {name}: OUT OF RANGE on the {arm} arm, not written')
            for line in outside:
                print(f'      {line}')
            continue

        tcp = entry.get('tcp_xyz')
        states[name] = {
            'joints': [round(v, 6) for v in joints],
            'joint_names': list(node.arm_joints),
            # The tool reflects with the pose: same x and z, y negated. Marked
            # computed because it was not read off TF like a recorded one.
            'tcp_xyz': ([tcp[0], -tcp[1], tcp[2]] if tcp else None),
            'recorded': datetime.now().isoformat(timespec='seconds'),
            'mirrored_from': f'{source_arm}:{name}',
        }
        tightest = min(
            (min(value - limits[joint][0], limits[joint][1] - value), joint)
            for joint, value in zip(node.arm_joints, joints)
            if joint in limits) if limits else None
        margin = (f', tightest margin {tightest[0]:.3f} rad on '
                  f'{tightest[1].split("_")[-1]}' if tightest else '')
        print(f'  {name}: mirrored{margin}')
    return states, ok


def play_states(node, args, states, wanted):
    """Drive the arm through the poses on file, then put it back.

    The starting posture is captured before anything moves and replayed last,
    so running this leaves the arm where it was found -- including when it was
    somewhere neither pose describes.
    """
    order = [n for n in wanted if (states.get(n) or {}).get('joints')]
    absent = [n for n in wanted if n not in order]
    if absent:
        print(f'not on file, skipping: {", ".join(absent)}')
    if not order:
        print(f'error: nothing to play from {args.file}', file=sys.stderr)
        return 1

    if not node.planner_is_serving(args.pipeline):
        print(f'error: the {args.pipeline} planner has no action server on '
              '/cumotion/move_group.\n'
              '       move_group is up and will accept goals, but nothing is '
              'planning them, so\n'
              '       every one comes back PLANNING_FAILED or TIMED_OUT. '
              'Relaunch the robot.', file=sys.stderr)
        return 1

    start = node.snapshot()
    print(f'\n{args.arm} arm, {args.pipeline} pipeline, '
          f'speed {args.speed:.2f}, up to {args.plan_attempts} plan attempts '
          f'per pose')
    print(f'starting posture  {fmt_joints(start["joints"])}')
    if start.get('tcp_xyz'):
        print(f'                  tool at {start["tcp_xyz"]}')
    print(f'\nwill visit: {" -> ".join(order)} -> back to the starting '
          f'posture\n')
    print('The arm will MOVE. Clear the workspace and keep the e-stop in '
          'reach.')
    if not args.yes:
        try:
            if input('type "go" to start, anything else to cancel: ').strip() \
                    != 'go':
                print('cancelled; nothing moved')
                return 1
        except (EOFError, KeyboardInterrupt):
            print('\ncancelled; nothing moved')
            return 1

    visited = []
    try:
        for name in order:
            target = states[name]['joints']
            print(f'\n-> {name}  {fmt_joints(target)}')
            if not node.move_to_joints(target, name, args.pipeline,
                                       args.speed,
                                       attempts=args.plan_attempts):
                print(f'{name} failed; returning to the starting posture')
                break
            visited.append(name)
            landed = node.snapshot()
            worst = max(abs(a - b) for a, b in zip(landed['joints'], target))
            print(f'   landed  {fmt_joints(landed["joints"])}')
            print(f'   tool at {landed.get("tcp_xyz")}, worst joint '
                  f'{worst:.4f} rad off the recording')
            if args.dwell:
                threading.Event().wait(args.dwell)
    except KeyboardInterrupt:
        # Do not drive home on a Ctrl-C: the reason for one is usually that the
        # arm is going somewhere it should not.
        print('\ninterrupted -- the arm is left where it is, NOT returned. '
              'Move it yourself in RViz.')
        return 1

    print(f'\n-> back to the starting posture  {fmt_joints(start["joints"])}')
    if not node.move_to_joints(start['joints'], 'starting posture',
                               args.pipeline, args.speed,
                               attempts=args.plan_attempts):
        where = visited[-1] if visited else 'where it stopped'
        print(f'error: could not return to the starting posture; the arm is '
              f'left at {where}', file=sys.stderr)
        if not node.planner_is_serving(args.pipeline):
            print('       the planner has died since this run started -- it '
                  'planned earlier goals\n'
                  '       and now has no action server. That, not the pose, '
                  'is why this failed.\n'
                  '       Relaunch the robot, then move the arm in RViz.',
                  file=sys.stderr)
        return 1
    print(f'\nplayed {len(visited)} of {len(order)}: '
          f'{", ".join(visited) or "none"}')
    return 0 if len(visited) == len(order) else 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('states', nargs='*', default=None,
                        help=f'states to record (default: {" ".join(DEFAULT_STATES)})')
    parser.add_argument('--arm', default='right', choices=('left', 'right'))
    parser.add_argument('--file', default=None,
                        help='YAML to write (default: pick_place_states_<arm>.yaml)')
    parser.add_argument('--list', action='store_true',
                        help='print the recorded states and exit')
    parser.add_argument('--missing', action='store_true',
                        help='record only the states not already on file')
    parser.add_argument('--mirror', action='store_true',
                        help='derive the poses from the other arm instead of '
                             'jogging to them')
    parser.add_argument('--play', action='store_true',
                        help='drive the arm through the poses on file, then '
                             'return it to where it started')
    parser.add_argument('--pipeline', default='cumotion',
                        help='--play planning pipeline (default: cumotion)')
    parser.add_argument('--speed', type=float, default=0.15,
                        help='--play velocity and acceleration scaling '
                             '(default: 0.15, deliberately slower than a cycle)')
    parser.add_argument('--dwell', type=float, default=2.0,
                        help='--play seconds to hold at each pose (default: 2)')
    parser.add_argument('--plan-attempts', type=int, default=3, dest='plan_attempts',
                        help='--play tries per pose when the planner fails to '
                             'converge (default: 3)')
    parser.add_argument('--yes', action='store_true',
                        help='--play without asking first')
    args = parser.parse_args()
    if args.play and args.mirror:
        parser.error('--play and --mirror do different things; run them one '
                     'at a time so you can look at what --mirror wrote')
    if not 0.0 < args.speed <= 1.0:
        parser.error('--speed must be in (0, 1]')
    if args.file is None:
        args.file = default_file(args.arm)
        # Poses recorded before the file became per-arm.
        if not os.path.exists(args.file) and os.path.exists(LEGACY_FILE):
            existing = yaml.safe_load(io.open(LEGACY_FILE)) or {}
            if existing.get('arm') == args.arm:
                print(f'note: reading the pre-split {LEGACY_FILE}; this will '
                      f'write {args.file}')
                args.file = LEGACY_FILE if args.list else args.file

    if args.list:
        existing = load_states(args.file)
        if not existing:
            print(f'no states recorded in {args.file}')
            return 0
        print(f'{args.file}:\n')
        for name, entry in existing.items():
            print(describe(name, entry), '\n')
        return 0

    wanted = args.states or DEFAULT_STATES

    rclpy.init()
    node = StateRecorder(args.arm)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    try:
        states = load_states(args.file)

        if args.mirror:
            if args.missing:
                wanted = [n for n in wanted
                          if not (states.get(n) or {}).get('joints')]
            print(f'mirroring {other_arm(args.arm)} -> {args.arm} into '
                  f'{args.file}')
            states, ok = mirror_states(node, args.arm, wanted, states)
            write_states(args.file, args.arm, states)
            print(f'\n{args.file} now holds: {", ".join(states)}')
            print('These were computed, not measured. Plan before you '
                  'execute, then check them with:')
            print(f'    python3 record_states.py --arm {args.arm} --play')
            return 0 if ok else 1

        # Everything past here reads the arm, so it needs the robot up.
        if not node.wait_for_joint_states():
            print('error: no /joint_states for the '
                  f'{args.arm} arm. Is the robot up? '
                  '(native/run_launch_everything.sh)', file=sys.stderr)
            return 1

        if args.play:
            return play_states(node, args, states, wanted)

        if args.missing:
            wanted = [n for n in wanted
                      if not (states.get(n) or {}).get('joints')]
            if not wanted:
                print(f'{args.file} already holds '
                      f'{", ".join(states)} -- nothing to record')
                return 0
        print(f'recording for the {args.arm} arm into {args.file}')
        print('jog the arm in RViz, then press Enter to capture. Ctrl-C to stop.\n')

        for name in wanted:
            note = DESCRIPTIONS.get(name, 'custom state')
            if name in states:
                print(f'{name} is already recorded:')
                print(describe(name, states[name]))
            try:
                input(f'\n-> move the arm to {name} ({note}), then press Enter ')
            except (EOFError, KeyboardInterrupt):
                print('\nstopped; nothing further recorded')
                break
            entry = node.snapshot()
            states[name] = entry
            # Written after every capture, not once at the end: a Ctrl-C halfway
            # through then still keeps the state already recorded.
            write_states(args.file, args.arm, states)
            print(f'\nrecorded {name}')
            print(describe(name, entry))

        print(f'\n{args.file} now holds: {", ".join(states)}')
        absent = [n for n in DEFAULT_STATES
                  if not (states.get(n) or {}).get('joints')]
        if absent:
            print(f'still missing: {", ".join(absent)}. Finish with:')
            print(f'    python3 record_states.py --arm {args.arm} '
                  f'{" ".join(absent)}')
            if absent == ['home_state']:
                print('(only needed when arm_selection:=by_side lets either arm '
                      'be chosen)')
        return 0
    finally:
        # shutdown() first so spin() returns, then join before the node is
        # destroyed: destroying it under a spinning executor aborts in the C
        # layer and drops a core file in the workspace.
        executor.shutdown()
        spin.join(timeout=5.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
