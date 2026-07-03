import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 声明启动参数
    gripper_threshold_arg = DeclareLaunchArgument(
        'gripper_threshold',
        default_value='0.0008',
        description='夹爪位置变化的最小阈值 (m)，用于减少Action请求频率'
    )
    
    gripper_scaling_factor_arg = DeclareLaunchArgument(
        'gripper_scaling_factor',
        default_value='0.02',
        description='夹爪数值缩放因子：外骨骼归一化值(0-1) -> 机械臂物理值(米)。默认0.02表示外骨骼1.0对应机械臂2cm'
    )

    left_joint_multipliers_arg = DeclareLaunchArgument(
        'left_joint_multipliers',
        default_value='[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]',
        description='左臂7个关节倍率（列表字符串），用于单侧关节方向修正'
    )

    right_joint_multipliers_arg = DeclareLaunchArgument(
        'right_joint_multipliers',
        default_value='[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]',
        description='右臂7个关节倍率（列表字符串），用于单侧关节方向修正'
    )

    left_gripper_reverse_arg = DeclareLaunchArgument(
        'left_gripper_reverse',
        default_value='false',
        description='是否反向左夹爪归一化输入（x -> 1-x）'
    )

    right_gripper_reverse_arg = DeclareLaunchArgument(
        'right_gripper_reverse',
        default_value='false',
        description='是否反向右夹爪归一化输入（x -> 1-x）'
    )

    control_rate_hz_arg = DeclareLaunchArgument(
        'control_rate_hz',
        default_value='100.0',
        description='桥接控制循环频率（Hz）'
    )

    joint_smoothing_alpha_arg = DeclareLaunchArgument(
        'joint_smoothing_alpha',
        default_value='0.45',
        description='关节平滑系数（0~1，越大越快）'
    )

    joint_max_delta_per_sec_arg = DeclareLaunchArgument(
        'joint_max_delta_per_sec',
        default_value='1.8',
        description='关节每秒最大变化（rad/s），用于抑制突跳'
    )

    gripper_smoothing_alpha_arg = DeclareLaunchArgument(
        'gripper_smoothing_alpha',
        default_value='0.55',
        description='夹爪平滑系数（0~1，越大越快）'
    )

    gripper_max_delta_per_sec_arg = DeclareLaunchArgument(
        'gripper_max_delta_per_sec',
        default_value='0.120',
        description='夹爪每秒最大变化（m/s），用于软开合'
    )

    gripper_action_min_period_sec_arg = DeclareLaunchArgument(
        'gripper_action_min_period_sec',
        default_value='0.01',
        description='夹爪Action最小发送周期（秒）'
    )

    gripper_min_position_m_arg = DeclareLaunchArgument(
        'gripper_min_position_m',
        default_value='0.0',
        description='夹爪最小位置（m）'
    )

    gripper_max_position_m_arg = DeclareLaunchArgument(
        'gripper_max_position_m',
        default_value='0.044',
        description='夹爪最大位置（m）'
    )

    gripper_close_extra_m_arg = DeclareLaunchArgument(
        'gripper_close_extra_m',
        default_value='0.0',
        description='已废弃: 负值过冲会导致硬件向反方向运动. 通过提高硬件 KP 实现闭合力.'
    )

    enable_boot_homing_arg = DeclareLaunchArgument(
        'enable_boot_homing',
        default_value='true',
        description='启动时是否平滑回零'
    )

    boot_homing_duration_sec_arg = DeclareLaunchArgument(
        'boot_homing_duration_sec',
        default_value='3.0',
        description='启动平滑回零时长（秒）'
    )

    boot_homing_arm_target_arg = DeclareLaunchArgument(
        'boot_homing_arm_target',
        default_value='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
        description='启动回零目标（7关节）'
    )

    boot_homing_gripper_target_arg = DeclareLaunchArgument(
        'boot_homing_gripper_target',
        default_value='0.044',
        description='启动回零目标夹爪位置（m）'
    )

    # 桥接节点
    bridge_node = Node(
        package='qnbot_teleoperator',
        executable='exoskeleton_bridge_node',
        name='exoskeleton_bridge_node',
        output='screen',
        parameters=[{
            'gripper_threshold': LaunchConfiguration('gripper_threshold'),
            'gripper_scaling_factor': LaunchConfiguration('gripper_scaling_factor'),
            'left_joint_multipliers': LaunchConfiguration('left_joint_multipliers'),
            'right_joint_multipliers': LaunchConfiguration('right_joint_multipliers'),
            'left_gripper_reverse': LaunchConfiguration('left_gripper_reverse'),
            'right_gripper_reverse': LaunchConfiguration('right_gripper_reverse'),
            'control_rate_hz': LaunchConfiguration('control_rate_hz'),
            'joint_smoothing_alpha': LaunchConfiguration('joint_smoothing_alpha'),
            'joint_max_delta_per_sec': LaunchConfiguration('joint_max_delta_per_sec'),
            'gripper_smoothing_alpha': LaunchConfiguration('gripper_smoothing_alpha'),
            'gripper_max_delta_per_sec': LaunchConfiguration('gripper_max_delta_per_sec'),
            'gripper_action_min_period_sec': LaunchConfiguration('gripper_action_min_period_sec'),
            'gripper_min_position_m': LaunchConfiguration('gripper_min_position_m'),
            'gripper_max_position_m': LaunchConfiguration('gripper_max_position_m'),
            'gripper_close_extra_m': LaunchConfiguration('gripper_close_extra_m'),
            'enable_boot_homing': LaunchConfiguration('enable_boot_homing'),
            'boot_homing_duration_sec': LaunchConfiguration('boot_homing_duration_sec'),
            'boot_homing_arm_target': LaunchConfiguration('boot_homing_arm_target'),
            'boot_homing_gripper_target': LaunchConfiguration('boot_homing_gripper_target')
        }]
    )

    return LaunchDescription([
        gripper_threshold_arg,
        gripper_scaling_factor_arg,
        left_joint_multipliers_arg,
        right_joint_multipliers_arg,
        left_gripper_reverse_arg,
        right_gripper_reverse_arg,
        control_rate_hz_arg,
        joint_smoothing_alpha_arg,
        joint_max_delta_per_sec_arg,
        gripper_smoothing_alpha_arg,
        gripper_max_delta_per_sec_arg,
        gripper_action_min_period_sec_arg,
        gripper_min_position_m_arg,
        gripper_max_position_m_arg,
        gripper_close_extra_m_arg,
        enable_boot_homing_arg,
        boot_homing_duration_sec_arg,
        boot_homing_arm_target_arg,
        boot_homing_gripper_target_arg,
        bridge_node
    ])
