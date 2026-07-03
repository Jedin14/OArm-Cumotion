#!/usr/bin/env python3
"""
ROS 2 to LeLab Bridge Node

Subscribes to:
- /joint_states  →  observation.state:
      joint_position  [16]  (left joint1-7 + finger, right joint1-7 + finger)
      joint_velocity  [16]  (same joints, velocities)
      ee_pose         [14]  (left xyz+quat[7], right xyz+quat[7])  via TF
      gripper_state   [2]   (left finger_joint1, right finger_joint1)
- /left_arm/joint_command  (left arm action)
- /right_arm/joint_command (right arm action)

Broadcasts the latest state via UDP to LeLab (port 19092) at 100 Hz.
"""

import json
import socket
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy

import tf2_ros
from tf2_ros import TransformListener, Buffer


# ---------------------------------------------------------------------------
# Ordered joint names for left and right arms (7 DOF + gripper each = 8 each)
# ---------------------------------------------------------------------------
LEFT_ARM_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_left_finger_joint1",
]
RIGHT_ARM_JOINTS = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
    "openarm_right_finger_joint1",
]
ALL_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS  # 16 total

# TF frames for end-effector poses (wrist / hand frame)
LEFT_EE_FRAME  = "openarm_left_hand"
RIGHT_EE_FRAME = "openarm_right_hand"
BASE_FRAME     = "world"


class LeLabBridgeNode(Node):
    def __init__(self):
        super().__init__('lelab_bridge_node')

        self.get_logger().info('LeLab UDP Bridge Node started.')

        # ------------------------------------------------------------------ #
        # State storage
        # ------------------------------------------------------------------ #
        # Per-joint storage keyed by joint name
        self._pos: dict[str, float] = {j: 0.0 for j in ALL_JOINTS}
        self._vel: dict[str, float] = {j: 0.0 for j in ALL_JOINTS}

        self.latest_action   = {}
        self.latest_buttons  = []
        self.lock            = threading.Lock()

        # ------------------------------------------------------------------ #
        # TF2 buffer / listener
        # ------------------------------------------------------------------ #
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------ #
        # UDP setup
        # ------------------------------------------------------------------ #
        self.udp_ip   = "127.0.0.1"
        self.udp_port = 19092
        self.sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ------------------------------------------------------------------ #
        # ROS subscriptions
        # ------------------------------------------------------------------ #
        self.create_subscription(JointState, '/joint_states',
                                 self.obs_callback, 10)
        self.create_subscription(JointState, '/left_arm/joint_command',
                                 self.left_action_callback, 10)
        self.create_subscription(JointState, '/right_arm/joint_command',
                                 self.right_action_callback, 10)
        self.create_subscription(Joy, '/exo/gamepad_keys',
                                 self.joy_callback, 10)

        # Broadcast timer @ 100 Hz
        self.create_timer(0.01, self.broadcast_loop)

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #

    def obs_callback(self, msg: JointState):
        """Store position & velocity for every recognised openarm joint."""
        with self.lock:
            for i, name in enumerate(msg.name):
                if name in self._pos:
                    self._pos[name] = float(msg.position[i]) if i < len(msg.position) else 0.0
                    self._vel[name] = float(msg.velocity[i]) if i < len(msg.velocity) else 0.0

    def _extract_action(self, side: str, msg: JointState):
        with self.lock:
            for i in range(min(7, len(msg.position))):
                self.latest_action[f"openarm_{side}_joint{i+1}"] = float(msg.position[i])

            if len(msg.position) >= 8:
                self.latest_action[f"openarm_{side}_finger_joint1"] = float(msg.position[7])
            elif f'{side}_gripper_joint' in msg.name:
                try:
                    idx = msg.name.index(f'{side}_gripper_joint')
                    self.latest_action[f"openarm_{side}_finger_joint1"] = float(msg.position[idx])
                except ValueError:
                    pass

    def left_action_callback(self, msg: JointState):
        self._extract_action('left', msg)

    def right_action_callback(self, msg: JointState):
        self._extract_action('right', msg)

    def joy_callback(self, msg: Joy):
        with self.lock:
            self.latest_buttons = list(msg.buttons)

    # ---------------------------------------------------------------------- #
    # TF helpers
    # ---------------------------------------------------------------------- #

    def _lookup_ee_pose(self, ee_frame: str) -> list[float] | None:
        """
        Return [x, y, z, qx, qy, qz, qw] for *ee_frame* relative to BASE_FRAME,
        or None if the transform is not yet available.
        """
        try:
            if not self.tf_buffer.can_transform(
                    BASE_FRAME, ee_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0)):
                return None
            t = self.tf_buffer.lookup_transform(
                BASE_FRAME, ee_frame, rclpy.time.Time())
            tr = t.transform.translation
            ro = t.transform.rotation
            return [tr.x, tr.y, tr.z, ro.x, ro.y, ro.z, ro.w]
        except Exception:
            return None

    # ---------------------------------------------------------------------- #
    # Broadcast loop
    # ---------------------------------------------------------------------- #

    def broadcast_loop(self):
        with self.lock:
            # --- joint_position / joint_velocity [16] ---
            joint_position = [self._pos[j] for j in ALL_JOINTS]
            joint_velocity = [self._vel[j] for j in ALL_JOINTS]

            # --- gripper_state [2] ---
            gripper_state = [
                self._pos["openarm_left_finger_joint1"],
                self._pos["openarm_right_finger_joint1"],
            ]

            action_snapshot  = self.latest_action.copy()
            buttons_snapshot = self.latest_buttons.copy()

        # --- ee_pose [14] (TF lookup outside the main lock to avoid blocking) ---
        left_ee  = self._lookup_ee_pose(LEFT_EE_FRAME)  or [0.0] * 7
        right_ee = self._lookup_ee_pose(RIGHT_EE_FRAME) or [0.0] * 7
        ee_pose  = left_ee + right_ee  # 14 values

        observation = {
            "joint_position": joint_position,   # [16]
            "joint_velocity": joint_velocity,   # [16]
            "ee_pose":        ee_pose,           # [14]
            "gripper_state":  gripper_state,     # [2]
        }

        payload = {
            "observation": observation,
            "action":      action_snapshot,
            "buttons":     buttons_snapshot,
            "timestamp":   time.time(),
        }

        # Only send once we have non-zero joint data
        if any(v != 0.0 for v in joint_position) or action_snapshot:
            try:
                data = json.dumps(payload).encode('utf-8')
                self.sock.sendto(data, (self.udp_ip, self.udp_port))
            except Exception as e:
                self.get_logger().error(f"UDP send error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LeLabBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
