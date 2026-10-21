"""The web GUI (web_gui.py: every arm tool + the pick stack's topics) and
the box model (box_model_node.py).

    source native/setup.bash
    ros2 launch launch/web_gui.launch.py                    # http://beast:8080
    ros2 launch launch/web_gui.launch.py port:=8090 host:=127.0.0.1 model:=false
    ros2 launch launch/web_gui.launch.py click:=true        # click-only: no box model

The robot itself comes up first: native/run_launch_everything.sh.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_launch_description():
    host, port, approach, model, click = (LaunchConfiguration(n) for n in
                                          ('host', 'port', 'approach', 'model', 'click'))
    gui = ['python3', os.path.join(WS, 'web_gui.py'), '--host', host, '--port', port,
           '--approach', approach]
    env = {'PYTHONPATH': WS + os.pathsep + os.environ.get('PYTHONPATH', ''),
           'PYTHONUNBUFFERED': '1'}
    return LaunchDescription([
        DeclareLaunchArgument('host', default_value='0.0.0.0',
                              description='where the page listens (127.0.0.1: this machine only)'),
        DeclareLaunchArgument('port', default_value='8080'),
        DeclareLaunchArgument('approach', default_value='auto',
                              description='auto (the model decides), top, front'),
        DeclareLaunchArgument('model', default_value='true',
                              description='start box_model_node.py'),
        DeclareLaunchArgument('click', default_value='false',
                              description='click-only, like click_to_move: no box model'),
        ExecuteProcess(
            cmd=['python3', os.path.join(WS, 'box_model_node.py'), '--ros-args',
                 '-p', ['approach:=', approach]],
            cwd=WS, additional_env=env, output='screen', name='box_model',
            condition=IfCondition(PythonExpression(
                ["'", model, "' == 'true' and '", click, "' != 'true'"]))),
        ExecuteProcess(cmd=gui, cwd=WS, additional_env=env, output='screen', name='web_gui',
                       condition=UnlessCondition(click)),
        ExecuteProcess(cmd=gui + ['--click'], cwd=WS, additional_env=env, output='screen',
                       name='web_gui', condition=IfCondition(click)),
    ])
