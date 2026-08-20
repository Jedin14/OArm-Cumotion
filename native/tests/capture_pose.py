#!/usr/bin/env python3
"""Capture the arm's current joint values and tool pose, ready to paste.

    source native/setup.bash
    python3 native/tests/capture_pose.py                 # one snapshot
    python3 native/tests/capture_pose.py --watch         # live, until Ctrl-C
    python3 native/tests/capture_pose.py --arm right

Jog the arm wherever you like in RViz (Plan then Execute), then run this. It
prints the joint values, the tool pose, how close each joint is to its URDF
limit, and the `home_joint_positions:=` string to hand straight to
pick_place.launch.py.

Two jobs:

* Fixing a ready pose. Zero is a poor place to start a task from -- the tool
  hangs at z=0.082, under the table. Jog to a pose that sees the work area,
  capture it, and pass it as home_joint_positions so every cycle begins and ends
  there.

* Settling where the tool really is. Jog until the gripper touches the object,
  capture, and compare the tool position with what
  VLM/pixel_to_world.py reports for that same object. If they disagree by more
  than a centimetre or so, the camera mount in cam_org.txt is off -- and every
  autonomous grasp would miss by that much, in the same direction.
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NEAR_LIMIT_RAD = 0.09          # ~5 degrees


def urdf_limits(arm):
    """Joint limits straight from the URDF cuMotion plans against."""
    root = ET.parse(os.path.join(WS, 'openarm.urdf')).getroot()
    limits = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        limit = joint.find('limit')
        if limit is not None and f'openarm_{arm}_' in name:
            limits[name] = (float(limit.get('lower')), float(limit.get('upper')))
    return limits


class PoseCapture(Node):

    def __init__(self, arm):
        super().__init__('capture_pose')
        self.arm = arm
        self.joint_names = [f'openarm_{arm}_joint{i}' for i in range(1, 8)]
        self.finger_joint = f'openarm_{arm}_finger_joint1'
        self.tcp_frame = f'openarm_{arm}_hand_tcp'
        self.limits = urdf_limits(arm)
        self.state = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, '/joint_states', self._on_state, 10)

    def _on_state(self, msg):
        self.state = dict(zip(msg.name, msg.position))

    def wait(self, timeout=10.0):
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            have_state = self.state and all(n in self.state for n in self.joint_names)
            have_tf = self.tf_buffer.can_transform('world', self.tcp_frame,
                                                   rclpy.time.Time())
            if have_state and have_tf:
                return True
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                missing = 'joint states' if not have_state else 'tf'
                self.get_logger().error(f'timed out waiting for {missing}')
                return False

    def tool_pose(self):
        tf = self.tf_buffer.lookup_transform(
            'world', self.tcp_frame, rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0))
        t, q = tf.transform.translation, tf.transform.rotation
        return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)

    def snapshot(self):
        positions = [self.state[n] for n in self.joint_names]
        (x, y, z), quat = self.tool_pose()

        print(f'\n{self.tcp_frame} in world')
        print(f'  position    x={x:+.4f}  y={y:+.4f}  z={z:+.4f}')
        print(f'  orientation x={quat[0]:+.4f} y={quat[1]:+.4f} '
              f'z={quat[2]:+.4f} w={quat[3]:+.4f}')

        # Tool Z axis: straight down is (0, 0, -1), which is what a top-down
        # grasp needs. Anything else tells you the wrist angle you actually used.
        qx, qy, qz, qw = quat
        approach = (2 * (qx * qz + qy * qw),
                    2 * (qy * qz - qx * qw),
                    1 - 2 * (qx * qx + qy * qy))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -approach[2]))))
        print(f'  approach axis ({approach[0]:+.3f}, {approach[1]:+.3f}, '
              f'{approach[2]:+.3f})  = {tilt:.1f} deg off straight-down')

        print('\njoints')
        flagged = []
        for name, value in zip(self.joint_names, positions):
            lo, up = self.limits.get(name, (float('-inf'), float('inf')))
            room = min(value - lo, up - value)
            mark = ''
            if value < lo or value > up:
                mark = '  OUTSIDE URDF LIMIT'
                flagged.append(name)
            elif room < NEAR_LIMIT_RAD:
                mark = f'  at the limit ({math.degrees(room):.1f} deg of room)'
                flagged.append(name)
            print(f'  {name:28} {value:+.4f} rad  {math.degrees(value):+7.1f} deg'
                  f'   [{lo:+.3f}, {up:+.3f}]{mark}')

        finger = self.state.get(self.finger_joint)
        if finger is not None:
            print(f'  {self.finger_joint:28} {finger:+.4f}')

        print('\npaste into pick_place.launch.py:')
        print('  home_joint_positions:="[' +
              ', '.join(f'{v:.4f}' for v in positions) + ']"')

        if flagged:
            print(f'\nnote: {", ".join(flagged)} at or past the URDF limit. If the '
                  'real hardware went further than the URDF allows, the limits in '
                  'openarm_description are wrong and that is worth fixing -- '
                  'cuMotion plans against them.')
        return positions

    def watch(self):
        print('live tool pose, Ctrl-C to stop\n')
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
                if not (self.state and all(n in self.state for n in self.joint_names)):
                    continue
                try:
                    (x, y, z), _ = self.tool_pose()
                except Exception:
                    continue
                joints = ' '.join(f'{math.degrees(self.state[n]):+6.1f}'
                                  for n in self.joint_names)
                print(f'\r tcp {x:+.3f} {y:+.3f} {z:+.3f}   deg {joints}',
                      end='', flush=True)
        except KeyboardInterrupt:
            print()
            self.snapshot()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', default='left', choices=('left', 'right'))
    parser.add_argument('--watch', action='store_true',
                        help='print continuously, snapshot on Ctrl-C')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = PoseCapture(args.arm)
    try:
        if not node.wait():
            return 1
        node.watch() if args.watch else node.snapshot()
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
