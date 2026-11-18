"""The pick stack: box_model_node.py (the model) and pick_stack.py (the arms).

    source native/setup.bash
    ros2 launch launch/pick_stack.launch.py                 # window if there is a display
    ros2 launch launch/pick_stack.launch.py gui:=off approach:=top
    ros2 launch launch/pick_stack.launch.py model:=false    # another node serves the model
    ros2 launch launch/pick_stack.launch.py order:=v2       # the leftmost free box first

The robot itself (arms, camera, MoveIt, cuMotion) comes up first, as always:
native/run_launch_everything.sh. Topics (all under /beast_arm): see
pick_stack.py's docstring.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # app/: the scripts, on PYTHONPATH
WS = os.path.dirname(APP)                          # the workspace: the working directory


def generate_launch_description():
    gui, approach, model, tool, order, score, edges, ahead = (
        LaunchConfiguration(n) for n in ('gui', 'approach', 'model', 'pick_tool', 'order',
                                         'score', 'walls_edges', 'walls_ahead'))
    env = {'PYTHONPATH': APP + os.pathsep + os.environ.get('PYTHONPATH', ''),
           'PYTHONUNBUFFERED': '1'}
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='auto',
                              description='auto (window only with a display), on, off'),
        DeclareLaunchArgument('approach', default_value='auto',
                              description='auto (the model decides), top, front'),
        DeclareLaunchArgument('pick_tool', default_value='vacuum',
                              description='vacuum (left arm only) or finger (right arm only)'),
        DeclareLaunchArgument('order', default_value='v1',
                              description='which box first: v1 (stack tops, highest '
                                          'first, then nearest, then leftmost) or v2 '
                                          '(the leftmost free box)'),
        DeclareLaunchArgument('score', default_value='0.4',
                              description="SAM's confidence needed to count as a box"),
        DeclareLaunchArgument('walls_edges', default_value='depth',
                              description="the green walls' edges: depth (where the depth "
                                          "ends) or picture (the colour picture's: more "
                                          "reach at the sides, less safe)"),
        DeclareLaunchArgument('walls_ahead', default_value='0.11',
                              description='the walls start this far (m) in front of the '
                                          'nearest thing seen (0.05 before 2026-11-05)'),
        DeclareLaunchArgument('model', default_value='true',
                              description='start box_model_node.py (false: another node '
                                          'serves /beast_arm/box_model/detect)'),
        ExecuteProcess(
            cmd=['python3', os.path.join(APP, 'box_model_node.py'), '--ros-args',
                 '-p', ['approach:=', approach], '-p', ['tool:=', tool],
                 '-p', ['order:=', order], '-p', ['score:=', score]],
            cwd=WS, additional_env=env, output='screen', name='box_model',
            condition=IfCondition(model)),
        ExecuteProcess(
            cmd=['python3', os.path.join(APP, 'pick_stack.py'), '--gui', gui,
                 '--approach', approach, '--pick-tool', tool,
                 '--walls-edges', edges, '--walls-ahead', ahead],
            cwd=WS, additional_env=env, output='screen', name='pick_stack'),
    ])
