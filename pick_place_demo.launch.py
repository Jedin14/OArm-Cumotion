"""Everything, in one command: robot, planners, detector, orchestrator, panel.

    native/run_pick_place_demo.sh

That wrapper is the intended entry point -- it checks the things that fail
confusingly rather than clearly (no DISPLAY, no /workspaces symlink, CAN down)
before any of this starts. To run the launch file directly:

    ros2 launch pick_place_demo.launch.py

What it starts, and why in this order:

    launch_everything.launch.py   MoveIt, RViz, controllers, cuMotion, the
                                  camera, the octomap gater. tool_frame is
                                  forced to match `arm`, because cuMotion takes
                                  Cartesian goals for exactly one link and a
                                  mismatch makes every pose goal fail.
    pick_place.launch.py          the PaliGemma detector and the orchestrator.
    pick_place_ui.py              the panel you type into.

The detector loads several GB of PaliGemma weights, which takes tens of seconds;
it starts alongside the robot rather than after it so that load overlaps with
bringup. Until it prints "model loaded" a pick will sit in LOCATE. This is the
reason the two launch files are normally kept separate -- you do not want that
load on every robot bringup -- so use pick_place.launch.py on its own when the
robot is already up.

Before the first run, record the two poses the cycle needs:

    python3 record_states.py
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

WS = os.path.dirname(os.path.realpath(__file__))

# Passed straight through to pick_place.launch.py, which declares the rest.
FORWARDED = ['arm', 'prompt', 'states_file', 'place_mode', 'approach_height',
             'velocity_scaling', 'use_table_collision', 'table_z']


def generate_launch_description():
    arm = LaunchConfiguration('arm')

    declarations = [
        DeclareLaunchArgument(
            'arm', default_value='right',
            description='Which arm picks. Also decides cuMotion\'s tool_frame.'),
        DeclareLaunchArgument(
            'prompt', default_value='detect screwdriver',
            description='Initial detector prompt. The panel overrides it.'),
        DeclareLaunchArgument(
            'states_file', default_value=os.path.join(WS, 'pick_place_states.yaml'),
            description='Poses recorded by record_states.py.'),
        DeclareLaunchArgument(
            'place_mode', default_value='state',
            description='Where the object is released: state, ready or position.'),
        DeclareLaunchArgument(
            'approach_height', default_value='0.05',
            description='Pre-grasp height above the object, metres.'),
        DeclareLaunchArgument(
            'velocity_scaling', default_value='0.15',
            description='Fraction of joint velocity limits. Start low.'),
        DeclareLaunchArgument(
            'use_table_collision', default_value='false',
            description='Explicit collision box for the work surface.'),
        DeclareLaunchArgument(
            'table_z', default_value='0.0',
            description='Work surface height, world frame.'),
        DeclareLaunchArgument(
            'octomap', default_value='static',
            description='live or static. static plus the gater is the tested path.'),
        DeclareLaunchArgument(
            'use_fake_hardware', default_value='false',
            description='Rehearse without the arms. Grasp verification cannot '
                        'pass on fake hardware -- see the README.'),
        DeclareLaunchArgument(
            'right_can_interface', default_value='can0',
            description='CAN interface for the right arm.'),
        DeclareLaunchArgument(
            'left_can_interface', default_value='can1',
            description='CAN interface for the left arm.'),
        DeclareLaunchArgument(
            'ui', default_value='true',
            description='Start the panel. false leaves the service interface.'),
    ]

    # One link, one arm: openarm_<arm>_hand_tcp. Deriving it here rather than
    # exposing it means the two cannot disagree.
    tool_frame = PythonExpression(["'openarm_' + '", arm, "' + '_hand_tcp'"])

    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(WS, 'launch_everything.launch.py')),
        launch_arguments={
            'tool_frame': tool_frame,
            'octomap': LaunchConfiguration('octomap'),
            'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
            'right_can_interface': LaunchConfiguration('right_can_interface'),
            'left_can_interface': LaunchConfiguration('left_can_interface'),
        }.items(),
    )

    pick_place = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(WS, 'pick_place.launch.py')),
        launch_arguments={
            name: LaunchConfiguration(name) for name in FORWARDED
        }.items(),
    )

    ui = ExecuteProcess(
        cmd=['python3', os.path.join(WS, 'pick_place_ui.py')],
        name='pick_place_ui',
        output='screen',
        additional_env={
            'PICK_PLACE_STATES_FILE': LaunchConfiguration('states_file'),
            'PYTHONUNBUFFERED': '1',
        },
        condition=IfCondition(LaunchConfiguration('ui')),
    )

    return LaunchDescription(declarations + [robot, pick_place, ui])
