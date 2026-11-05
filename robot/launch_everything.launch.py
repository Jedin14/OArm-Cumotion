import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition


def realsense_imu_readable():
    """True when the D455's motion sensor (its accelerometer: /dev/iio:device*
    named accel_3d, on the camera's USB) can be read without root. It needs
    the RealSense udev rules (librealsense's 99-realsense-libusb.rules); without
    them the camera node hung at start with enable_accel on (2026-10-29)."""
    import glob
    found = False
    for d in glob.glob('/sys/bus/iio/devices/iio:device*'):
        try:
            name = open(os.path.join(d, 'name')).read().strip()
        except OSError:
            continue
        if name != 'accel_3d':
            continue
        found = True
        dev = os.path.join('/dev', os.path.basename(d))
        if not os.access(dev, os.R_OK):
            return False
    return found

# Inject override directory for patched cumotion planner
os.environ['PYTHONPATH'] = '/workspaces/isaac_ros-dev/src/isaac_ros_cumotion_override:' + os.environ.get('PYTHONPATH', '')

def generate_launch_description():
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    right_can = LaunchConfiguration('right_can_interface')
    left_can = LaunchConfiguration('left_can_interface')
    octomap_mode = LaunchConfiguration('octomap')
    enable_4d = LaunchConfiguration('4d')
    tool_frame = LaunchConfiguration('tool_frame')
    collision_distance = LaunchConfiguration('collision_activation_distance')
    
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
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Start RViz. false on a headless or VNC session: without '
                    'hardware GL it hangs at start-up with its window never '
                    'mapped, and nothing else needs it.'
    )
    declare_4d = DeclareLaunchArgument(
        '4d',
        default_value='false',
        description='Enable 4D lidar TF bridge'
    )
    # cuMotion has a single ee_link and rejects any pose goal aimed at a
    # different link with INVALID_LINK_NAME, so this is what decides which arm
    # can be given Cartesian goals -- from RViz or from pick_place_orchestrator.
    # The other arm still takes joint-space goals. Both arms stay in the
    # kinematic chain either way; see the note in openarm.yml.
    # cuRobo's own default is 0.01 m. 0.03 keeps the arm 2 cm further off
    # everything in the octomap, which is the knob to reach for when it clips
    # obstacles it planned around -- it inflates every voxel, so raising it too
    # far will make tight approaches unplannable.
    declare_collision_distance = DeclareLaunchArgument(
        'collision_activation_distance',
        default_value='0.03',
        description='Clearance cuMotion keeps from every obstacle, metres.'
    )
    # The D455's minimum depth range shrinks with the depth resolution: about
    # 0.4-0.5 m at 848x480, about 0.2 m at 424x240. With the V3 mount
    # (cam_org.txt) the camera looks flat at surfaces 0.3-0.5 m away, inside
    # the 848x480 blind zone, so click_to_move read wrong depths there. The
    # old high, tilted mount saw the table from further off, which is why
    # 848x480 was fine for the VLM. Go back to 848x480x15 only for work that
    # is all beyond ~0.5 m: it is more precise at range.
    # 640x480 (2026-10-29): three times the depth points of 424x240, for the box
    # picker -- a box top seen at a shallow angle got too few points at 424x240
    # and lost to the box's front -- still measuring from ~0.34 m. Use
    # depth_profile:=424x240x15 for work nearer than that.
    # auto (2026-11-04): on when the D455's motion sensor can be read -- the
    # RealSense udev rules. Without them the camera node hung at start with it on
    # (2026-10-29): its /dev/iio devices were root's only. Off, the box picker
    # uses the accelerometer's last saved reading (calibration/camera_tilt.yaml),
    # else the TF's tilt, which is not gravity's.
    imu_ok = realsense_imu_readable()
    if not imu_ok:
        print('[launch_everything] the D455 accelerometer is off: its motion sensor is not '
              'readable (install the RealSense udev rules, then replug the camera; see '
              'docs/cmds.md). accel:=true forces it on.')
    declare_accel = DeclareLaunchArgument(
        'accel', default_value='true' if imu_ok else 'false',
        description='the D455 accelerometer (the box picker\'s up direction). Default: on when '
                    'its motion sensor is readable (the RealSense udev rules), else off')
    declare_depth_profile = DeclareLaunchArgument(
        'depth_profile',
        default_value='640x480x15',
        description='D455 depth stream WxHxFPS. Lower resolution = shorter '
                    'minimum range (424x240: ~0.2 m, 640x480: ~0.34 m, 848x480: ~0.45 m).'
    )
    declare_boot_pre_pick = DeclareLaunchArgument(
        'boot_pre_pick',
        default_value='true',
        description='Once move_group is up, walk both arms to their recorded '
                    'navigation_state (boot_pre_pick.py; click_to_move switches '
                    'to pre_pick when picking starts).'
    )
    # The vacuum relay (2026-11-05): app/vacuum_bridge.py writes the pick stack's
    # /beast_arm/vacuum to the ESP32 ('1' / '0' over USB serial). Here, with the
    # rest of the hardware, so the GUI's restarts leave it alone: opening the port
    # restarts the ESP32, and a box on the cup would drop.
    declare_vacuum = DeclareLaunchArgument(
        'vacuum',
        default_value='true',
        description='Start the vacuum relay bridge (app/vacuum_bridge.py: '
                    '/beast_arm/vacuum -> the ESP32 over USB serial).'
    )
    declare_vacuum_port = DeclareLaunchArgument(
        'vacuum_port',
        default_value='/dev/serial/by-id/'
                      'usb-Espressif_USB_JTAG_serial_debug_unit_34:CD:B0:16:1D:30-if00',
        description="The ESP32's serial port (default: its own USB port by id, "
                    "/dev/ttyACM0 now; auto = the only USB serial device there is)."
    )
    declare_tool_frame = DeclareLaunchArgument(
        'tool_frame',
        default_value='openarm_right_hand_tcp',
        description='Link cuMotion plans Cartesian goals for. Use '
                    'openarm_left_hand_tcp for left-arm pose goals.'
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
            'use_rviz': LaunchConfiguration('use_rviz'),
        }.items()
    )
    
    # 2. Add the cuMotion GPU Action Server Node
    cumotion_node = Node(
        package='isaac_ros_cumotion',
        executable='cumotion_goal_set_planner_node',
        name='cumotion_planner',
        parameters=[{
            'robot': '/workspaces/isaac_ros-dev/robot/openarm.yml',
            'robot_file': '/workspaces/isaac_ros-dev/robot/openarm.yml',
            'robot_filepath': '/workspaces/isaac_ros-dev/robot/openarm.yml',
            'tool_frame': tool_frame,
            'collision_activation_distance': collision_distance,
        }],
        output='screen',
        # cuMotion is a Python node and launch pipes its stdout, so without this
        # its output block-buffers and is lost when it dies. It has crashed here
        # with SIGFPE (exit code -8) leaving nothing but launch's own "process
        # has died" line, which says nothing about why.
        additional_env={'PYTHONUNBUFFERED': '1'},
        # Bring it back when it dies, because it does. SIGFPE (exit code -8) in
        # most launches on 2026-09-07, and the whole cell is useless without it:
        # move_group's cumotion pipeline forwards every goal to this node, so a
        # dead planner turns even a joint goal to a recorded posture into
        # TIMED_OUT.
        #
        # The delay is for the CUDA kernels it loads on startup -- it logs
        # nothing at all while it does, and respawning faster than it can come
        # up would just churn. The orchestrator's readiness probe covers the
        # window where it is back but not yet planning.
        respawn=True,
        respawn_delay=4.0,
    )
    
    is_static = PythonExpression(["'", octomap_mode, "' == 'static'"])
    is_live = PythonExpression(["'", octomap_mode, "' != 'static'"])

    launch_nodes = [
        declare_fake_hardware,
        declare_right_can,
        declare_left_can,
        declare_octomap,
        declare_use_rviz,
        declare_4d,
        declare_tool_frame,
        declare_collision_distance,
        declare_boot_pre_pick,
        declare_depth_profile,
        declare_accel,
        declare_vacuum,
        declare_vacuum_port,
        demo_launch,
        cumotion_node,
        ExecuteProcess(
            cmd=['python3', '/workspaces/isaac_ros-dev/robot/boot_pre_pick.py'],
            output='screen',
            additional_env={'PYTHONUNBUFFERED': '1'},
            condition=IfCondition(LaunchConfiguration('boot_pre_pick')),
        ),
        # It catches every serial error itself (unplugged: it reopens the port);
        # respawned only if it dies anyway, else the vacuum would stay as it was.
        ExecuteProcess(
            cmd=['python3', '/workspaces/isaac_ros-dev/app/vacuum_bridge.py', '--ros-args',
                 '-p', ['port:=', LaunchConfiguration('vacuum_port')]],
            output='screen',
            additional_env={'PYTHONUNBUFFERED': '1'},
            name='vacuum_bridge',
            respawn=True,
            respawn_delay=2.0,
            condition=IfCondition(LaunchConfiguration('vacuum')),
        ),
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
    #
    # Colour and depth-to-colour alignment are on because the box picker (via
    # box_picker/src/ros_bridge.py) and the pick stack need them: they cannot open
    # the D455 themselves while this driver holds it, so they subscribe to
    # /camera/camera/color/image_raw and
    # /camera/camera/aligned_depth_to_color/image_raw instead. The aligned depth
    # follows the colour resolution, so /camera/camera/color/camera_info is the
    # one set of intrinsics that describes both.
    #
    # Colour is 640x480x15: this D455's RGB sensor offers only 1280x720,
    # 1280x800x8, 640x480 and 424x240 (ros2 param describe /camera/camera
    # rgb_camera.color_profile), so asking for depth's 848x480 silently fell
    # back to 640x480 anyway.
    #
    # Note the aligned topic is *not* affected by the octomap gater's remap below,
    # so the VLM keeps seeing live frames even with octomap:=static.
    try:
        realsense_launch_file = os.path.join(
            get_package_share_directory('realsense2_camera'),
            'launch',
            'rs_launch.py'
        )

        realsense_args = {
            'depth_module.depth_profile': LaunchConfiguration('depth_profile'),
            'rgb_camera.color_profile': '640x480x15',
            'pointcloud.enable': 'false',
            'align_depth.enable': 'true',
            'enable_color': 'true',
            # the D455's accelerometer: the true "up" for the box picker (box_model_node).
            # From the TF the picker was told the camera looked 4.5 deg up; the scene
            # showed ~4.7 deg down, the boxes' tops lost to their fronts (2026-10-29)
            'enable_accel': LaunchConfiguration('accel'),
        }

        realsense_node_live = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file),
            launch_arguments=realsense_args.items(),
            condition=IfCondition(is_live)
        )

        realsense_node_static = GroupAction([
            SetRemap(src='/camera/camera/depth/image_rect_raw', dst='/camera/camera/depth/image_rect_raw_in'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments=realsense_args.items()
            )
        ], condition=IfCondition(is_static))
        
        octomap_gater = ExecuteProcess(
            cmd=['python3', '/workspaces/isaac_ros-dev/robot/octomap_gater.py'],
            output='screen',
            condition=IfCondition(is_static)
        )
        
        launch_nodes.extend([realsense_node_live, realsense_node_static, octomap_gater])
        
    except Exception as e:
        # If realsense2_camera is not installed, fallback to just the robot without crashing
        print(f"\n[WARNING]: realsense2_camera package not found! Please run 'sudo apt-get install ros-humble-realsense2-camera ros-humble-realsense2-description -y' inside the container.\n")
    
    return LaunchDescription(launch_nodes)
