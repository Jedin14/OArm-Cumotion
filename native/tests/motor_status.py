#!/usr/bin/env python3
"""Every motor's status, position, torque and tracking error, once.

Reads three topics for a few seconds and prints a table. Commands nothing
and moves nothing.

    source native/setup.bash && python3 native/tests/motor_status.py
"""

import sys
import time

import rclpy
from control_msgs.msg import DynamicJointState, JointTrajectoryControllerState
from rclpy.node import Node
from sensor_msgs.msg import JointState

# What the status nibble in a Damiao feedback frame means. openarm_can
# discarded that byte until recently, so -1 here on a running arm means the
# driver predates the fix rather than that the motor is silent.
MEANS = {
    -1: 'not reported',
    0: 'NOT ENABLED',
    1: 'enabled',
    8: 'FAULT over-voltage',
    9: 'FAULT under-voltage',
    10: 'FAULT over-current',
    11: 'FAULT driver over-temp',
    12: 'FAULT motor over-temp',
    13: 'FAULT lost comms',
    14: 'FAULT overloaded',
}

JOINTS = ([f'openarm_{arm}_joint{i}' for arm in ('left', 'right')
           for i in range(1, 8)]
          + [f'openarm_{arm}_finger_joint1' for arm in ('left', 'right')])


class Listen(Node):
    def __init__(self):
        super().__init__('motor_status')
        self.position = {}
        self.effort = {}
        self.status = {}
        self.error = {}
        self.create_subscription(JointState, '/joint_states', self._joints, 10)
        self.create_subscription(DynamicJointState, '/dynamic_joint_states',
                                 self._dynamic, 10)
        for arm in ('left', 'right'):
            self.create_subscription(
                JointTrajectoryControllerState,
                f'/{arm}_joint_trajectory_controller/controller_state',
                self._tracking, 10)

    def _joints(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.position[name] = msg.position[i]
            if i < len(msg.effort):
                self.effort[name] = msg.effort[i]

    def _dynamic(self, msg):
        for i, name in enumerate(msg.joint_names):
            if i >= len(msg.interface_values):
                continue
            entry = msg.interface_values[i]
            for which, value in zip(entry.interface_names, entry.values):
                if which == 'status':
                    self.status[name] = int(round(value))

    def _tracking(self, msg):
        for i, name in enumerate(msg.joint_names):
            if i < len(msg.error.positions):
                self.error[name] = abs(msg.error.positions[i])


def main():
    rclpy.init()
    node = Listen()
    end = time.time() + 4.0
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not node.position:
        print('nothing on /joint_states -- is the bringup running?')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print('%-28s %-22s %9s %9s %10s'
          % ('joint', 'the motor says', 'rad', 'Nm', 'off by mrad'))
    for name in JOINTS:
        if name not in node.position:
            continue
        code = node.status.get(name, -1)
        off = node.error.get(name)
        print('%-28s %-22s %9.4f %9.3f %10s'
              % (name,
                 MEANS.get(code, f'unknown code {code}'),
                 node.position.get(name, float('nan')),
                 node.effort.get(name, float('nan')),
                 '-' if off is None else '%.2f' % (off * 1000)))
    if not node.status:
        print('\nNo status from any motor: the running driver is the build '
              'that discards it.\nRebuild and restart the bringup:  '
              'native/build_ws.sh --packages-select openarm_can '
              'openarm_hardware')
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
