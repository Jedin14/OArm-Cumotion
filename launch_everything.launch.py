import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition

# Inject override directory for patched cumotion planner
os.environ['PYTHONPATH'] = '/workspaces/isaac_ros-dev/src/isaac_ros_cumotion_override:' + os.environ.get('PYTHONPATH', '')

def generate_launch_description():
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    right_can = LaunchConfiguration('right_can_interface')
    left_can = LaunchConfiguration('left_can_interface')
    octomap_mode = LaunchConfiguration('octomap')
    enable_4d = LaunchConfiguration('4d')
    
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
    declare_octomap = DeclareLaunchArgument(
        'octomap',
        default_value='live',
        description='Octomap mode: live or static'
    )
    declare_4d = DeclareLaunchArgument(
        '4d',
        default_value='false',
        description='Enable 4D lidar TF bridge'
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
    
    is_static = PythonExpression(["'", octomap_mode, "' == 'static'"])
    is_live = PythonExpression(["'", octomap_mode, "' != 'static'"])

    launch_nodes = [
        declare_fake_hardware,
        declare_right_can,
        declare_left_can,
        declare_octomap,
        declare_4d,
        demo_launch,
        cumotion_node
    ]

    is_4d_enabled = PythonExpression(["'", enable_4d, "' == 'true'"])
    try:
        tf_bridge_launch_file = os.path.join(
            get_package_share_directory('robot_tf_bringup'),
            'launch',
            'tf_bridge.launch.py'
        )
        tf_bridge_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(tf_bridge_launch_file),
            condition=IfCondition(is_4d_enabled)
        )
        launch_nodes.append(tf_bridge_node)
    except Exception as e:
        print(f"\n[WARNING]: robot_tf_bringup package not found!\n")


    # 3. Add the RealSense Camera Node
    try:
        realsense_launch_file = os.path.join(
            get_package_share_directory('realsense2_camera'),
            'launch',
            'rs_launch.py'
        )
        
        realsense_node_live = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file),
            launch_arguments={
                'depth_module.depth_profile': '848x480x15',
                'pointcloud.enable': 'false',
                'align_depth.enable': 'false',
                'enable_color': 'false',
            }.items(),
            condition=IfCondition(is_live)
        )
        
        realsense_node_static = GroupAction([
            SetRemap(src='/camera/camera/depth/image_rect_raw', dst='/camera/camera/depth/image_rect_raw_in'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    'depth_module.depth_profile': '848x480x15',
                    'pointcloud.enable': 'false',
                    'align_depth.enable': 'false',
                    'enable_color': 'false',
                }.items()
            )
        ], condition=IfCondition(is_static))
        
        octomap_gater = ExecuteProcess(
            cmd=['python3', '/workspaces/isaac_ros-dev/octomap_gater.py'],
            output='screen',
            condition=IfCondition(is_static)
        )
        
        launch_nodes.extend([realsense_node_live, realsense_node_static, octomap_gater])
        
    except Exception as e:
        # If realsense2_camera is not installed, fallback to just the robot without crashing
        print(f"\n[WARNING]: realsense2_camera package not found! Please run 'sudo apt-get install ros-humble-realsense2-camera ros-humble-realsense2-description -y' inside the container.\n")
    
    return LaunchDescription(launch_nodes)
