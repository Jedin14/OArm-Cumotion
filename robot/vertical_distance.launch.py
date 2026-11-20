"""The stand-in /c60000004/vertical_distance publisher (robot/vertical_distance_pub.py).

    ros2 launch robot/vertical_distance.launch.py                       slider window, 45 cm, 10 Hz
    ros2 launch robot/vertical_distance.launch.py value_cm:=30
    ros2 launch robot/vertical_distance.launch.py mode:=sweep min_cm:=20 max_cm:=60 period_s:=20
    ros2 launch robot/vertical_distance.launch.py type:=string        "45.0 cm" (the old way)
    ros2 launch robot/vertical_distance.launch.py gui:=off            no window

Not with the real device publishing the same topic.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

SCRIPT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vertical_distance_pub.py')
ARGS = [
    ('topic', '/c60000004/vertical_distance', 'Topic to publish'),
    ('type', 'range', 'range (sensor_msgs/Range, metres), string ("45.0 cm"), float32 or float64 (cm)'),
    ('value_cm', '45.0', 'The distance sent in mode fixed, cm'),
    ('mode', 'fixed', 'fixed, sweep (min_cm..max_cm and back over period_s) or off'),
    ('min_cm', '20.0', 'Sweep low end, cm'),
    ('max_cm', '60.0', 'Sweep high end, cm'),
    ('period_s', '20.0', 'Sweep period, s'),
    ('rate_hz', '10.0', 'Messages per second'),
    ('gui', 'auto', 'The slider window: auto (with a DISPLAY), on, off'),
    ('slider_min_cm', '0.0', 'Slider bottom, cm (0: the sensor right above the box)'),
    ('slider_max_cm', '152.0', 'Slider top, cm'),
]


def generate_launch_description():
    cmd = ['python3', SCRIPT, '--ros-args']
    for name, _default, _ in ARGS:
        cmd += ['-p', [name, ':=', LaunchConfiguration(name)]]   # (the node takes 30 or 30.0)
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=t) for n, d, t in ARGS]
        + [ExecuteProcess(cmd=cmd, output='screen', name='vertical_distance_pub',
                          additional_env={'PYTHONUNBUFFERED': '1'})])
