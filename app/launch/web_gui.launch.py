"""The web GUI (web_gui.py: every arm tool + the pick stack's topics) and
the box model (box_model_node.py).

    source native/setup.bash
    ros2 launch launch/web_gui.launch.py                    # http://beast:8080
    ros2 launch launch/web_gui.launch.py port:=8090 host:=127.0.0.1 model:=false
    ros2 launch launch/web_gui.launch.py click:=true        # click-only: no box model
    ros2 launch launch/web_gui.launch.py order:=v2          # the box model takes the leftmost free box
    ros2 launch launch/web_gui.launch.py drop_velocity:=0.5 drop_hold:=1.5   # gentler drop

The robot itself comes up first: native/run_launch_everything.sh.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # app/: the scripts, on PYTHONPATH
WS = os.path.dirname(APP)                          # the workspace: the working directory


def generate_launch_description():
    host, port, approach, model, click, tool, order, score, edges, ahead = (
        LaunchConfiguration(n) for n in
        ('host', 'port', 'approach', 'model', 'click', 'pick_tool', 'order', 'score',
         'walls_edges', 'walls_ahead'))
    drop_velocity, correct_speed, drop_hold, top_faces = (
        LaunchConfiguration(n) for n in ('drop_velocity', 'correct_speed', 'drop_hold', 'top_faces'))
    gui = ['python3', os.path.join(APP, 'web_gui.py'), '--host', host, '--port', port,
           '--approach', approach, '--pick-tool', tool,
           '--walls-edges', edges, '--walls-ahead', ahead,
           '--drop-velocity', drop_velocity, '--correct-speed', correct_speed,
           '--drop-hold', drop_hold]
    env = {'PYTHONPATH': APP + os.pathsep + os.environ.get('PYTHONPATH', ''),
           'PYTHONUNBUFFERED': '1'}
    return LaunchDescription([
        DeclareLaunchArgument('host', default_value='0.0.0.0',
                              description='where the page listens (127.0.0.1: this machine only)'),
        DeclareLaunchArgument('port', default_value='8080'),
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
        DeclareLaunchArgument('drop_velocity', default_value='0.7',
                              description='speed scaling (0..1) of the move from pre_pick to the '
                                          'drop pose with the box on the cup (0.4 before '
                                          '2026-11-05; the way out beside the boxes stays 0.4)'),
        DeclareLaunchArgument('correct_speed', default_value='0.04',
                              description='tool speed of the correction moves, m/s (0.02 before '
                                          '2026-11-05)'),
        DeclareLaunchArgument('drop_hold', default_value='1.0',
                              description='seconds at the drop pose with the vacuum off, for the '
                                          'box to let go'),
        DeclareLaunchArgument('top_faces', default_value='false',
                              description="the box picker's face rule (box_picker/src/pick_rules.py): "
                                          'false = a box on a box by its front (a top only when '
                                          'no front is found); true = the larger of front and top '
                                          'anywhere, as before 2026-11-05'),
        DeclareLaunchArgument('model', default_value='true',
                              description='start box_model_node.py'),
        DeclareLaunchArgument('click', default_value='false',
                              description='click-only, like click_to_move: no box model'),
        ExecuteProcess(
            cmd=['python3', os.path.join(APP, 'box_model_node.py'), '--ros-args',
                 '-p', ['approach:=', approach], '-p', ['tool:=', tool],
                 '-p', ['order:=', order], '-p', ['score:=', score],
                 '-p', ['top_faces:=', top_faces]],
            cwd=WS, additional_env=env, output='screen', name='box_model',
            condition=IfCondition(PythonExpression(
                ["'", model, "' == 'true' and '", click, "' != 'true'"]))),
        ExecuteProcess(cmd=gui, cwd=WS, additional_env=env, output='screen', name='web_gui',
                       condition=UnlessCondition(click)),
        ExecuteProcess(cmd=gui + ['--click'], cwd=WS, additional_env=env, output='screen',
                       name='web_gui', condition=IfCondition(click)),
    ])
