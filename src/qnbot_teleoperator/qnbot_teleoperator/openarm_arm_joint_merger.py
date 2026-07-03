#!/usr/bin/env python3
"""
OpenArm手臂关节合并节点
订阅/left_arm/joint_command和/right_arm/joint_command话题
将手臂关节状态合并到完整的joint_state消息中发布

该节点会：
1. 订阅左右手臂的关节命令话题
2. 从joint_state_publisher_gui订阅其他关节的状态（如果有）
3. 合并所有关节状态并发布到/joint_states话题（含手指关节）

Fix: 之前只发布7个手臂关节，忽略了夹爪关节，导致robot_state_publisher
     缺少 openarm_left_finger_joint1 / openarm_right_finger_joint1，
     RViz 中机器人不更新。现在正确提取并发布夹爪关节。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy


class OpenArmArmJointMerger(Node):
    def __init__(self):
        super().__init__('openarm_arm_joint_merger')

        # OpenArm手臂关节名称（7自由度，不含夹爪）
        self.left_arm_joints = [
            'openarm_left_joint1', 'openarm_left_joint2', 'openarm_left_joint3',
            'openarm_left_joint4', 'openarm_left_joint5', 'openarm_left_joint6',
            'openarm_left_joint7'
        ]

        self.right_arm_joints = [
            'openarm_right_joint1', 'openarm_right_joint2', 'openarm_right_joint3',
            'openarm_right_joint4', 'openarm_right_joint5', 'openarm_right_joint6',
            'openarm_right_joint7'
        ]

        # Gripper joint names as published to /joint_states (URDF names)
        self.left_finger_joint  = 'openarm_left_finger_joint1'
        self.right_finger_joint = 'openarm_right_finger_joint1'

        # Names sent by the retargeting node inside /left_arm/joint_command
        # and /right_arm/joint_command for the gripper value
        self.left_gripper_cmd_name  = 'left_gripper_joint'
        self.right_gripper_cmd_name = 'right_gripper_joint'

        # State cache
        self.left_arm_positions  = [0.0] * len(self.left_arm_joints)
        self.right_arm_positions = [0.0] * len(self.right_arm_joints)
        self.left_gripper_position  = 0.0
        self.right_gripper_position = 0.0
        self.other_joint_positions  = {}

        self.lock = threading.Lock()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10
        )

        # Subscribers
        self.left_arm_sub = self.create_subscription(
            JointState, '/left_arm/joint_command', self.left_arm_callback, qos_profile)

        self.right_arm_sub = self.create_subscription(
            JointState, '/right_arm/joint_command', self.right_arm_callback, qos_profile)

        # GUI fallback (non-arm joints like body links, etc.)
        self.gui_joint_sub = self.create_subscription(
            JointState, '/joint_states_gui', self.gui_joint_callback, qos_profile)

        # Publisher
        self.joint_state_pub = self.create_publisher(
            JointState, '/joint_states', qos_profile)

        # 50 Hz publish timer
        self.timer = self.create_timer(0.02, self.publish_merged_joint_state)

        self.get_logger().info('✅ OpenArm手臂关节合并节点已启动')
        self.get_logger().info('   - 订阅左臂话题: /left_arm/joint_command')
        self.get_logger().info('   - 订阅右臂话题: /right_arm/joint_command')
        self.get_logger().info('   - 订阅GUI话题: /joint_states_gui')
        self.get_logger().info('   - 发布合并话题: /joint_states')
        self.get_logger().info(
            f'   - 夹爪关节映射: '
            f'{self.left_gripper_cmd_name} → {self.left_finger_joint}, '
            f'{self.right_gripper_cmd_name} → {self.right_finger_joint}')

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _extract_arm_and_gripper(self, msg: JointState, arm_joints: list, gripper_cmd_name: str):
        """
        Extract 7 arm joint positions and the gripper position from a JointState msg.

        The retargeting node publishes 8 entries:
          [joint1..joint7, gripper_joint]
        where gripper_joint name is e.g. 'left_gripper_joint'.

        Returns (arm_positions[7], gripper_position).
        """
        # Build a name→position mapping so order doesn't matter
        name_to_pos = {name: float(pos)
                       for name, pos in zip(msg.name, msg.position)}

        arm_positions = [name_to_pos.get(j, 0.0) for j in arm_joints]
        gripper_pos   = name_to_pos.get(gripper_cmd_name, 0.0)

        return arm_positions, gripper_pos

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #

    def left_arm_callback(self, msg: JointState):
        with self.lock:
            arm_pos, grip_pos = self._extract_arm_and_gripper(
                msg, self.left_arm_joints, self.left_gripper_cmd_name)
            self.left_arm_positions   = arm_pos
            self.left_gripper_position = grip_pos

    def right_arm_callback(self, msg: JointState):
        with self.lock:
            arm_pos, grip_pos = self._extract_arm_and_gripper(
                msg, self.right_arm_joints, self.right_gripper_cmd_name)
            self.right_arm_positions   = arm_pos
            self.right_gripper_position = grip_pos

    def gui_joint_callback(self, msg: JointState):
        """Keep any non-arm joints from the GUI publisher (e.g. body joints)."""
        arm_joint_set = set(self.left_arm_joints) | set(self.right_arm_joints) | {
            self.left_finger_joint, self.right_finger_joint}
        with self.lock:
            for i, joint_name in enumerate(msg.name):
                if joint_name not in arm_joint_set and i < len(msg.position):
                    self.other_joint_positions[joint_name] = msg.position[i]

    # ---------------------------------------------------------------------- #
    # Publish merged /joint_states
    # ---------------------------------------------------------------------- #

    def publish_merged_joint_state(self):
        with self.lock:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = ''

            # Left arm joints (7)
            for i, name in enumerate(self.left_arm_joints):
                msg.name.append(name)
                msg.position.append(self.left_arm_positions[i])

            # Left finger joint (1) — mapped from left_gripper_joint
            msg.name.append(self.left_finger_joint)
            msg.position.append(self.left_gripper_position)

            # Right arm joints (7)
            for i, name in enumerate(self.right_arm_joints):
                msg.name.append(name)
                msg.position.append(self.right_arm_positions[i])

            # Right finger joint (1) — mapped from right_gripper_joint
            msg.name.append(self.right_finger_joint)
            msg.position.append(self.right_gripper_position)

            # Any other joints from GUI (body links, etc.)
            for name, pos in self.other_joint_positions.items():
                msg.name.append(name)
                msg.position.append(pos)

            self.joint_state_pub.publish(msg)

        # Periodic stats
        if not hasattr(self, '_publish_count'):
            self._publish_count = 0
        self._publish_count += 1
        if self._publish_count % 500 == 0:
            self.get_logger().info(
                f'📊 关节状态发布统计: 总计{self._publish_count}次, '
                f'左臂有运动关节: {len([p for p in self.left_arm_positions if abs(p) > 0.001])}/7, '
                f'右臂有运动关节: {len([p for p in self.right_arm_positions if abs(p) > 0.001])}/7'
            )


def main(args=None):
    rclpy.init(args=args)
    node = OpenArmArmJointMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
