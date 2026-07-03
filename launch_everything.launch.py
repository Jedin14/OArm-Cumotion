import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

# Inject override directory for patched cumotion planner
os.environ['PYTHONPATH'] = '/workspaces/isaac_ros-dev/src/isaac_ros_cumotion_override:' + os.environ.get('PYTHONPATH', '')

def generate_launch_description():
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    right_can = LaunchConfiguration('right_can_interface')
    left_can = LaunchConfiguration('left_can_interface')
    
    declare_fake_hardware = DeclareLaunchArgument(
        'use_fake_hardware',
        default_value='false',
        description='Use fake hardware'
    )
    declare_right_can = DeclareLaunchArgument(
        'right_can_interface',
        default_value='can0',
    )
    declare_left_can = DeclareLaunchArgument(
        'left_can_interface',
        default_value='can1',
    )
    
    # 1. Include the MoveIt demo launch file
    demo_launch_file = os.path.join(
        get_package_share_directory('openarm_bimanual_moveit_config'),
        'launch',
        'demo.launch.py'
    )
    
    demo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(demo_launch_file),
        launch_arguments={
            'use_fake_hardware': use_fake_hardware,
            'right_can_interface': right_can,
            'left_can_interface': left_can,
        }.items()
    )
    
    # 2. Add the cuMotion GPU Action Server Node
    cumotion_node = Node(
        package='isaac_ros_cumotion',
        executable='cumotion_goal_set_planner_node',
        name='cumotion_planner',
        parameters=[{
            'robot': '/workspaces/isaac_ros-dev/openarm.yml',
            'robot_file': '/workspaces/isaac_ros-dev/openarm.yml',
            'robot_filepath': '/workspaces/isaac_ros-dev/openarm.yml',
        }],
        output='screen'
    )
    
    return LaunchDescription([
        declare_fake_hardware,
        declare_right_can,
        declare_left_can,
        demo_launch,
        cumotion_node
    ])
