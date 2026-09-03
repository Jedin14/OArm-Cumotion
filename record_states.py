#!/usr/bin/env python3
"""Record named arm poses into a YAML file for the pick-and-place cycle.

Jog the arm where you want it -- the RViz MotionPlanning panel, drag the
interactive marker or set the sliders on the Joints tab, then Plan & Execute --
and this snapshots where it actually ended up. Reading joint values back off the
robot beats typing them in: the Joints tab only shows whole degrees, and an
executed plan lands near the goal rather than exactly on it.

    python3 record_states.py                 # walks through both states
    python3 record_states.py drop_state      # re-record just one
    python3 record_states.py --list          # show what is on file

Two states drive the cycle:

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

Needs the robot up (native/run_launch_everything.sh) so that /joint_states and
TF are live. It never commands motion; it only reads.
"""

import argparse
import os
import sys
import threading
from datetime import datetime

import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

WS = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FILE = os.path.join(WS, 'pick_place_states.yaml')

# Ordered: this is the sequence the walkthrough asks for them in, and the order
# they are used in during a cycle.
DEFAULT_STATES = ['pre_pick_state', 'drop_state']

DESCRIPTIONS = {
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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)

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


def describe(name, entry):
    tcp = entry.get('tcp_xyz')
    where = f'  tool at {tcp}' if tcp else '  tool position not recorded'
    joints = ', '.join(f'{v:+.3f}' for v in entry['joints'])
    return f'{name}\n  joints [{joints}]\n{where}\n  recorded {entry.get("recorded")}'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('states', nargs='*', default=None,
                        help=f'states to record (default: {" ".join(DEFAULT_STATES)})')
    parser.add_argument('--arm', default='right', choices=('left', 'right'))
    parser.add_argument('--file', default=DEFAULT_FILE,
                        help=f'YAML to write (default: {DEFAULT_FILE})')
    parser.add_argument('--list', action='store_true',
                        help='print the recorded states and exit')
    args = parser.parse_args()

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
        if not node.wait_for_joint_states():
            print('error: no /joint_states for the '
                  f'{args.arm} arm. Is the robot up? '
                  '(native/run_launch_everything.sh)', file=sys.stderr)
            return 1

        states = load_states(args.file)
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
