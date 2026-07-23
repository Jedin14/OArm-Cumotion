from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Bridge 1: Links 'world' to 'camera_init'
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_camera_init_bridge',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'world',
                '--child-frame-id', 'camera_init'
            ]
        ),

        # Bridge 2: Links 'camera_init' to 'unilidar_imu_initial'
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_init_to_unilidar_imu_bridge',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'camera_init',
                '--child-frame-id', 'unilidar_imu_initial'
            ]
        ),

        # Bridge 3: Links 'base_link' of the arm to the 'unilidar_lidar' frame (25 cm in front)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_unilidar_lidar_bridge',
            output='screen',
            arguments=[
                '--x', '0.25', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'unilidar_lidar'
            ]
        )
    ])
