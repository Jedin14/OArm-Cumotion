"""VLM-guided pick and place, launched on top of a running launch_everything.

Bring the robot up first (native/run_launch_everything.sh), then start this in a
second terminal:

    source native/setup.bash
    ros2 launch pick_place.launch.py prompt:="detect screwdriver" \
        place_position:="[0.35, 0.30, 0.25]"

It is deliberately separate from launch_everything.launch.py for two reasons:
PaliGemma takes tens of seconds and several GB of VRAM to load, which you do not
want on every robot bringup; and the two run in different Python stacks --
vlm_detector_node.py needs VLM/.venv (torch 2.14+cu130) while everything else
needs native/venv (torch 2.7+cu128, the ABI cuRobo's kernels are built against).
run_vlm_detector.sh is what keeps those apart, so the detector is started through
that wrapper rather than as a launch_ros Node.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

WS = os.path.dirname(os.path.realpath(__file__))

# Orchestrator parameters that are worth exposing as launch arguments. Everything
# else is declared in pick_place_orchestrator.py and can be overridden with
# --ros-args -p on a manual run.
ORCHESTRATOR_ARGS = [
    ('arm', 'left',
     "Which arm to use. cuMotion's openarm.yml sets ee_link to the left "
     "hand_tcp, so 'left' is the well-supported side."),
    ('place_position', '[0.35, 0.30, 0.25]',
     'Where to drop the object, xyz in the world frame. Measure this in RViz.'),
    ('place_yaw', '0.0', 'Tool yaw when releasing, radians.'),
    ('approach_height', '0.12', 'Pre-grasp height above the grasp, metres.'),
    ('grasp_z_offset', '-0.005',
     'Added to the detected surface height to get the grasp height.'),
    ('min_grasp_z', '0.01', 'Hard floor on grasp height, metres.'),
    ('use_table_collision', 'false',
     'Add an explicit collision box for the work surface instead of relying on '
     'the octomap alone. Worth enabling with octomap:=static.'),
    ('table_z', '0.0',
     'Height of the work surface top face, world frame. Only used when '
     'use_table_collision is true.'),
    ('velocity_scaling', '0.15', 'Fraction of joint velocity limits.'),
    ('auto_start', 'false',
     'Start a cycle 3 s after launch instead of waiting for '
     '/pick_place/start.'),
]


def generate_launch_description():
    prompt = LaunchConfiguration('prompt')
    model_id = LaunchConfiguration('model_id')

    declarations = [
        DeclareLaunchArgument(
            'prompt', default_value='detect screwdriver',
            description='PaliGemma detection prompt for the object to pick.'),
        DeclareLaunchArgument(
            'model_id', default_value='google/paligemma-3b-pt-224',
            description='HuggingFace model id for the detector.'),
    ]
    declarations += [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in ORCHESTRATOR_ARGS
    ]

    detector = ExecuteProcess(
        cmd=[
            os.path.join(WS, 'VLM', 'run_vlm_detector.sh'),
            '--ros-args',
            '-p', ['prompt:=', prompt],
            '-p', ['model_id:=', model_id],
        ],
        name='vlm_detector',
        output='screen',
    )

    orchestrator_cmd = [
        'python3', os.path.join(WS, 'pick_place_orchestrator.py'),
        '--ros-args',
        '-p', ['prompt:=', prompt],
    ]
    for name, _default, _description in ORCHESTRATOR_ARGS:
        orchestrator_cmd += ['-p', [f'{name}:=', LaunchConfiguration(name)]]

    orchestrator = ExecuteProcess(
        cmd=orchestrator_cmd,
        name='pick_place_orchestrator',
        output='screen',
    )

    return LaunchDescription(declarations + [detector, orchestrator])
