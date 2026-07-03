#!/usr/bin/env python3
"""
外骨骼-机械臂桥接节点（平滑版）

功能：
1. 订阅 /left_arm/joint_command 和 /right_arm/joint_command
2. 将命令平滑后发布到 forward_position_controller
3. 将夹爪命令平滑后发送到 GripperActionController
4. 启动时先从当前姿态平滑回零（home），再进入跟随模式
"""

import numpy as np
import rclpy
import yaml
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class ExoskeletonBridgeNode(Node):
    def __init__(self):
        super().__init__('exoskeleton_bridge_node')

        # Existing parameters
        self.declare_parameter('gripper_threshold', 0.005)
        self.declare_parameter('gripper_scaling_factor', 0.02)
        self.declare_parameter('left_joint_multipliers', [1.0] * 7)
        self.declare_parameter('right_joint_multipliers', [1.0] * 7)
        self.declare_parameter('left_gripper_reverse', False)
        self.declare_parameter('right_gripper_reverse', False)

        # New smoothing / homing parameters
        self.declare_parameter('control_rate_hz', 100.0)
        self.declare_parameter('joint_smoothing_alpha', 0.45)
        self.declare_parameter('joint_max_delta_per_sec', 1.8)
        self.declare_parameter('gripper_smoothing_alpha', 0.75)
        self.declare_parameter('gripper_max_delta_per_sec', 0.180)
        self.declare_parameter('gripper_action_min_period_sec', 0.01)
        self.declare_parameter('gripper_min_position_m', 0.0)
        self.declare_parameter('gripper_max_position_m', 0.050)
        self.declare_parameter('gripper_close_extra_m', 0.0)
        self.declare_parameter('enable_boot_homing', True)
        self.declare_parameter('boot_homing_duration_sec', 3.0)
        self.declare_parameter('boot_homing_arm_target', [0.0] * 7)
        self.declare_parameter('boot_homing_gripper_target', 0.044)

        self.gripper_threshold = float(self.get_parameter('gripper_threshold').value)
        self.gripper_scale = float(self.get_parameter('gripper_scaling_factor').value)
        self.left_joint_multipliers = self._parse_joint_multipliers(
            self.get_parameter('left_joint_multipliers').value, 'left'
        )
        self.right_joint_multipliers = self._parse_joint_multipliers(
            self.get_parameter('right_joint_multipliers').value, 'right'
        )
        self.left_gripper_reverse = bool(self.get_parameter('left_gripper_reverse').value)
        self.right_gripper_reverse = bool(self.get_parameter('right_gripper_reverse').value)

        self.control_rate_hz = max(10.0, float(self.get_parameter('control_rate_hz').value))
        self.joint_smoothing_alpha = float(np.clip(self.get_parameter('joint_smoothing_alpha').value, 0.01, 1.0))
        self.joint_max_delta_per_sec = max(0.05, float(self.get_parameter('joint_max_delta_per_sec').value))
        self.gripper_smoothing_alpha = float(np.clip(self.get_parameter('gripper_smoothing_alpha').value, 0.01, 1.0))
        self.gripper_max_delta_per_sec = max(0.002, float(self.get_parameter('gripper_max_delta_per_sec').value))
        self.gripper_action_min_period_sec = max(
            0.01, float(self.get_parameter('gripper_action_min_period_sec').value)
        )
        self.gripper_min_position_m = float(self.get_parameter('gripper_min_position_m').value)
        self.gripper_max_position_m = float(self.get_parameter('gripper_max_position_m').value)
        if self.gripper_max_position_m <= self.gripper_min_position_m:
            self.get_logger().warn('Invalid gripper range, fallback to [0.0, 0.044]')
            self.gripper_min_position_m = 0.0
            self.gripper_max_position_m = 0.044
        self.gripper_close_extra_m = max(0.0, float(self.get_parameter('gripper_close_extra_m').value))
        self.enable_boot_homing = bool(self.get_parameter('enable_boot_homing').value)
        self.boot_homing_duration_sec = max(0.2, float(self.get_parameter('boot_homing_duration_sec').value))
        self.boot_homing_arm_target = self._parse_joint_multipliers(
            self.get_parameter('boot_homing_arm_target').value, 'boot_homing_arm_target'
        )
        self.boot_homing_gripper_target = float(self.get_parameter('boot_homing_gripper_target').value)

        # IO
        self.left_arm_sub = self.create_subscription(JointState, '/left_arm/joint_command', self.left_arm_callback, 10)
        self.right_arm_sub = self.create_subscription(JointState, '/right_arm/joint_command', self.right_arm_callback, 10)
        self.joint_states_sub = self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)

        self.left_arm_pub = self.create_publisher(Float64MultiArray, '/left_forward_position_controller/commands', 10)
        self.right_arm_pub = self.create_publisher(Float64MultiArray, '/right_forward_position_controller/commands', 10)

        self.left_gripper_client = ActionClient(self, GripperCommand, '/left_gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_gripper_controller/gripper_cmd')

        # Desired inputs from retargeting
        self.input_arm = {'left': np.zeros(7, dtype=float), 'right': np.zeros(7, dtype=float)}
        self.input_gripper = {'left': 0.0, 'right': 0.0}
        self.have_input = {'left': False, 'right': False}

        # Current measured robot state
        self.current_arm = {'left': None, 'right': None}
        self.current_gripper = {'left': None, 'right': None}

        # Commanded smoothed state
        self.cmd_arm = {'left': np.zeros(7, dtype=float), 'right': np.zeros(7, dtype=float)}
        self.cmd_gripper = {'left': 0.0, 'right': 0.0}

        # Boot homing state machine
        self.boot_phase = 'wait_state' if self.enable_boot_homing else 'follow'
        self.boot_start_time = None
        self.boot_start_arm = {'left': np.zeros(7, dtype=float), 'right': np.zeros(7, dtype=float)}
        self.boot_start_gripper = {'left': 0.0, 'right': 0.0}

        # UI command states for fixed home lock
        self.left_fixed_home = False
        self.right_fixed_home = False
        self.lock_state = {'left': False, 'right': False}
        self.transition_start_time = {'left': 0.0, 'right': 0.0}
        self.transition_start_arm = {'left': np.zeros(7, dtype=float), 'right': np.zeros(7, dtype=float)}
        self.transition_start_gripper = {'left': 0.0, 'right': 0.0}
        from std_msgs.msg import String
        import json
        self.ui_command_sub = self.create_subscription(String, '/exo/ui_command', self.ui_command_callback, 10)

        self.last_control_time = None
        self.last_gripper_sent = {'left': 0.0, 'right': 0.0}
        self.last_gripper_sent_time = {'left': 0.0, 'right': 0.0}

        self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)

        self.get_logger().info(
            'Exoskeleton Bridge Node 已启动\n'
            f'  control_rate_hz: {self.control_rate_hz}\n'
            f'  joint_smoothing_alpha: {self.joint_smoothing_alpha}, joint_max_delta_per_sec: {self.joint_max_delta_per_sec}\n'
            f'  gripper_smoothing_alpha: {self.gripper_smoothing_alpha}, gripper_max_delta_per_sec: {self.gripper_max_delta_per_sec}\n'
            f'  gripper_threshold: {self.gripper_threshold}, gripper_action_min_period_sec: {self.gripper_action_min_period_sec}\n'
            f'  gripper_range_m: [{self.gripper_min_position_m}, {self.gripper_max_position_m}], gripper_close_extra_m: {self.gripper_close_extra_m}\n'
            f'  enable_boot_homing: {self.enable_boot_homing}, boot_homing_duration_sec: {self.boot_homing_duration_sec}\n'
            f'  left_joint_multipliers: {self.left_joint_multipliers}\n'
            f'  right_joint_multipliers: {self.right_joint_multipliers}\n'
            f'  left_gripper_reverse: {self.left_gripper_reverse}, right_gripper_reverse: {self.right_gripper_reverse}'
        )

    def ui_command_callback(self, msg):
        import json
        try:
            data = json.loads(msg.data)
            action = data.get('action')
            if action == 'toggle_left_home':
                self.left_fixed_home = bool(data.get('value', False))
                self.get_logger().info(f"UI Command: Left arm home fixed = {self.left_fixed_home}")
            elif action == 'toggle_right_home':
                self.right_fixed_home = bool(data.get('value', False))
                self.get_logger().info(f"UI Command: Right arm home fixed = {self.right_fixed_home}")
        except Exception as e:
            self.get_logger().error(f"Error parsing ui_command: {e}")

    def _parse_joint_multipliers(self, raw_value, name):
        default = [1.0] * 7
        parsed = raw_value
        if isinstance(raw_value, str):
            try:
                parsed = yaml.safe_load(raw_value)
            except Exception:
                self.get_logger().warn(f'[{name}] parse failed: {raw_value}, fallback to {default}')
                return default
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 7:
            self.get_logger().warn(f'[{name}] invalid: {parsed}, fallback to {default}')
            return default
        try:
            return [float(x) for x in parsed]
        except Exception:
            self.get_logger().warn(f'[{name}] non-numeric: {parsed}, fallback to {default}')
            return default

    def left_arm_callback(self, msg: JointState):
        self._update_input_from_arm_msg('left', msg)

    def right_arm_callback(self, msg: JointState):
        self._update_input_from_arm_msg('right', msg)

    def _update_input_from_arm_msg(self, side, msg: JointState):
        if len(msg.position) < 7:
            return

        multipliers = self.left_joint_multipliers if side == 'left' else self.right_joint_multipliers
        arm_positions = np.array(msg.position[:7], dtype=float)
        self.input_arm[side] = arm_positions * np.array(multipliers, dtype=float)
        self.have_input[side] = True

        gripper_norm = 0.0
        has_gripper_data = False
        if len(msg.position) >= 8:
            gripper_norm = float(msg.position[7])
            has_gripper_data = True
        elif f'{side}_gripper_joint' in msg.name:
            try:
                gripper_idx = msg.name.index(f'{side}_gripper_joint')
                if gripper_idx < len(msg.position):
                    gripper_norm = float(msg.position[gripper_idx])
                    has_gripper_data = True
            except ValueError:
                pass

        if has_gripper_data:
            gripper_norm = float(np.clip(gripper_norm, 0.0, 1.0))
            if (side == 'left' and self.left_gripper_reverse) or (side == 'right' and self.right_gripper_reverse):
                gripper_norm = 1.0 - gripper_norm
            self.input_gripper[side] = float(
                np.clip(
                    gripper_norm * self.gripper_scale,
                    self.gripper_min_position_m,
                    self.gripper_max_position_m
                )
            )

    def joint_states_callback(self, msg: JointState):
        index_by_name = {name: idx for idx, name in enumerate(msg.name)}
        for side in ('left', 'right'):
            arm_joint_names = [f'openarm_{side}_joint{i}' for i in range(1, 8)]
            if all(name in index_by_name for name in arm_joint_names):
                self.current_arm[side] = np.array(
                    [msg.position[index_by_name[name]] for name in arm_joint_names], dtype=float
                )

            gripper_name = f'openarm_{side}_finger_joint1'
            if gripper_name in index_by_name:
                self.current_gripper[side] = float(msg.position[index_by_name[gripper_name]])

    def _smooth_vector(self, current, target, alpha, max_delta):
        blended = current + alpha * (target - current)
        delta = np.clip(blended - current, -max_delta, max_delta)
        return current + delta

    def _smooth_scalar(self, current, target, alpha, max_delta):
        blended = current + alpha * (target - current)
        delta = float(np.clip(blended - current, -max_delta, max_delta))
        return current + delta

    def _publish_arm(self, side, arm_values):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in arm_values]
        if side == 'left':
            self.left_arm_pub.publish(msg)
        else:
            self.right_arm_pub.publish(msg)

    def _send_gripper_action(self, side, position, now_sec):
        client = self.left_gripper_client if side == 'left' else self.right_gripper_client
        if not client.server_is_ready():
            return

        if now_sec - self.last_gripper_sent_time[side] < self.gripper_action_min_period_sec:
            return

        if abs(position - self.last_gripper_sent[side]) < self.gripper_threshold:
            return

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = float(position)
        goal_msg.command.max_effort = 10.0
        client.send_goal_async(goal_msg)

        self.last_gripper_sent[side] = float(position)
        self.last_gripper_sent_time[side] = now_sec

    def _initialize_from_current_state(self, now_sec):
        for side in ('left', 'right'):
            if self.current_arm[side] is None or self.current_gripper[side] is None:
                return False

        for side in ('left', 'right'):
            self.cmd_arm[side] = self.current_arm[side].copy()
            current_gripper = float(
                np.clip(self.current_gripper[side], self.gripper_min_position_m, self.gripper_max_position_m)
            )
            self.cmd_gripper[side] = current_gripper
            self.last_gripper_sent[side] = current_gripper
            self.last_gripper_sent_time[side] = now_sec
            self.boot_start_arm[side] = self.current_arm[side].copy()
            self.boot_start_gripper[side] = current_gripper

        self.boot_start_time = now_sec
        if self.enable_boot_homing:
            self.boot_phase = 'homing'
            self.get_logger().info('Boot homing started: smooth move to home target.')
        else:
            self.boot_phase = 'follow'
        return True

    def control_loop(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_control_time is None:
            self.last_control_time = now_sec
            return

        dt = now_sec - self.last_control_time
        self.last_control_time = now_sec
        if dt <= 0.0:
            return

        joint_max_delta = self.joint_max_delta_per_sec * dt
        gripper_max_delta = self.gripper_max_delta_per_sec * dt

        if self.boot_phase == 'wait_state':
            if not self._initialize_from_current_state(now_sec):
                return

        if self.boot_phase == 'homing':
            progress = min(1.0, (now_sec - self.boot_start_time) / self.boot_homing_duration_sec)
            target_arm = np.array(self.boot_homing_arm_target, dtype=float)
            target_gripper = float(
                np.clip(self.boot_homing_gripper_target, self.gripper_min_position_m, self.gripper_max_position_m)
            )

            for side in ('left', 'right'):
                desired_arm = (1.0 - progress) * self.boot_start_arm[side] + progress * target_arm
                desired_gripper = (1.0 - progress) * self.boot_start_gripper[side] + progress * target_gripper
                desired_gripper = float(
                    np.clip(desired_gripper, self.gripper_min_position_m, self.gripper_max_position_m)
                )

                self.cmd_arm[side] = self._smooth_vector(
                    self.cmd_arm[side], desired_arm, self.joint_smoothing_alpha, joint_max_delta
                )
                self.cmd_gripper[side] = self._smooth_scalar(
                    self.cmd_gripper[side], desired_gripper, self.gripper_smoothing_alpha, gripper_max_delta
                )

                self._publish_arm(side, self.cmd_arm[side])
                self._send_gripper_action(side, self.cmd_gripper[side], now_sec)

            if progress >= 1.0:
                self.boot_phase = 'follow'
                self.get_logger().info('Boot homing completed. Follow mode enabled.')
            return

        # Follow mode
        for side in ('left', 'right'):
            is_locked = (side == 'left' and self.left_fixed_home) or (side == 'right' and self.right_fixed_home)
            
            # Detect state change
            if is_locked != self.lock_state[side]:
                self.lock_state[side] = is_locked
                self.transition_start_time[side] = now_sec
                self.transition_start_arm[side] = self.cmd_arm[side].copy()
                self.transition_start_gripper[side] = self.cmd_gripper[side]
                action_str = "Locking to home" if is_locked else "Unlocking to exoskeleton"
                self.get_logger().info(f"{side.capitalize()} arm: {action_str} with {self.boot_homing_duration_sec}s slow transition.")

            # Calculate raw target based on lock state
            if is_locked:
                target_arm = np.array(self.boot_homing_arm_target, dtype=float)
                target_gripper = float(
                    np.clip(self.boot_homing_gripper_target, self.gripper_min_position_m, self.gripper_max_position_m)
                )
            else:
                target_arm = self.input_arm[side] if self.have_input[side] else self.cmd_arm[side]
                target_gripper = self.input_gripper[side] if self.have_input[side] else self.cmd_gripper[side]

            # Apply slow transition if within the 3 second window
            time_since_transition = now_sec - self.transition_start_time[side]
            if time_since_transition < self.boot_homing_duration_sec:
                progress = time_since_transition / self.boot_homing_duration_sec
                desired_arm = (1.0 - progress) * self.transition_start_arm[side] + progress * target_arm
                desired_gripper = (1.0 - progress) * self.transition_start_gripper[side] + progress * target_gripper
            else:
                desired_arm = target_arm
                desired_gripper = target_gripper

            # No negative-position overshoot: the hardware linear formula maps
            # negative joint values to the wrong (opening) motor direction.
            # High GRIPPER_DEFAULT_KP in the hardware holds firmly at 0.0.
            desired_gripper = float(
                np.clip(desired_gripper, self.gripper_min_position_m, self.gripper_max_position_m)
            )

            self.cmd_arm[side] = self._smooth_vector(
                self.cmd_arm[side], desired_arm, self.joint_smoothing_alpha, joint_max_delta
            )
            self.cmd_gripper[side] = self._smooth_scalar(
                self.cmd_gripper[side], desired_gripper, self.gripper_smoothing_alpha, gripper_max_delta
            )

            self._publish_arm(side, self.cmd_arm[side])
            self._send_gripper_action(side, self.cmd_gripper[side], now_sec)


def main(args=None):
    rclpy.init(args=args)
    node = ExoskeletonBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
