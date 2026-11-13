"""Stationary Unitree L2 pipeline.

  ros2 launch l2_perception l2.launch.py                 # connection from config/l2.yaml
  ros2 launch l2_perception l2.launch.py connection:=udp # override (udp | serial)
  ros2 launch l2_perception l2.launch.py sim:=true       # no hardware: simulator
  ros2 launch l2_perception l2.launch.py rviz:=false

Prefer scripts/l2.sh: it also spins the lidar down when the pipeline stops.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def _nodes(context):
    share = get_package_share_directory("l2_perception")
    arg = lambda n: context.launch_configurations[n]
    cfg = arg("config")
    sim = arg("sim").lower() == "true"
    data = os.path.expanduser(arg("data_dir"))
    prefix = "sim_" if sim else ""
    files = {"calibration_file": os.path.join(data, prefix + "mount.yaml")}
    bg = os.path.join(data, prefix + "background.npz")   # written by l2_obstacles, read by l2_dense
    driver_overrides = {"connection": arg("connection")} if arg("connection") else {}
    nodes = []
    if sim:
        nodes.append(Node(package="l2_perception", executable="l2_sim", name="l2_sim", output="screen"))
    else:
        nodes.append(Node(package="l2_driver", executable="l2_driver_node", name="l2_driver",
                          output="screen", parameters=[cfg, driver_overrides]))
    nodes += [
        Node(package="l2_perception", executable="l2_stabilizer", name="l2_stabilizer",
             output="screen", parameters=[cfg, files]),
        Node(package="l2_perception", executable="l2_obstacles", name="l2_obstacles",
             output="screen", parameters=[cfg, files, {"background.file": bg}]),
        Node(package="l2_perception", executable="l2_dense", name="l2_dense", output="screen",
             parameters=[cfg, files, {"background_file": bg}]),
    ]
    if arg("rviz").lower() == "true":
        nodes.append(Node(package="rviz2", executable="rviz2", name="rviz2", output="log",
                          arguments=["-d", os.path.join(share, "rviz", "l2.rviz")]))
    return nodes


def generate_launch_description():
    share = get_package_share_directory("l2_perception")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "l2.yaml")),
        DeclareLaunchArgument("sim", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("connection", default_value="",
                              description="udp | serial; empty = value from the config file"),
        DeclareLaunchArgument("data_dir", default_value="~/.ros/l2"),
        OpaqueFunction(function=_nodes),
    ])
