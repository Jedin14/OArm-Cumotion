#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$(readlink -f "$0")")"

source /opt/ros/humble/setup.bash
source install/setup.bash

set -u

exec /usr/bin/python3 -c '
from launch import LaunchService, LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

launch_service = LaunchService()
launch_service.include_launch_description(
    LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                "/workspaces/isaac_ros-dev/launch_everything.launch.py"
            ),
            launch_arguments={
                "octomap": "static",
                "4d": "false",
                "use_fake_hardware": "false",
                "right_can_interface": "can0",
                "left_can_interface": "can1",
            }.items()
        )
    ])
)
raise SystemExit(launch_service.run())
'
