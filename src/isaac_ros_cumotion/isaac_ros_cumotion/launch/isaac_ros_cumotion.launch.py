# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import yaml


def read_params(pkg_name, params_dir, params_file_name):
    params_file = os.path.join(
        get_package_share_directory(pkg_name), params_dir, params_file_name)
    return yaml.safe_load(open(params_file, 'r'))


def launch_args_from_params(pkg_name, params_dir, params_file_name, prefix: str = None):
    launch_args = []
    launch_configs = {}
    params = read_params(pkg_name, params_dir, params_file_name)

    for param, value in params['/**']['ros__parameters'].items():
        if value is not None:
            arg_name = param if prefix is None else f'{prefix}.{param}'
            launch_args.append(DeclareLaunchArgument(name=arg_name, default_value=str(value)))
            launch_configs[param] = LaunchConfiguration(arg_name)

    return launch_args, launch_configs


def generate_launch_description():
    """
    Launch cuMotion components.

    Brings up MoveGroup and MotionPlan action servers and IK action server.
    """
    launch_args, launch_configs = launch_args_from_params(
        'isaac_ros_cumotion',
        'params',
        'isaac_ros_cumotion_params.yaml',
        'cumotion_action_server',
    )

    env_variables = dict(os.environ)

    if launch_configs['enable_cuda_mps']:
        env_variables.update({
            'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE':
                launch_configs['cuda_mps_active_thread_percentage'],
            'CUDA_MPS_PIPE_DIRECTORY': launch_configs['cuda_mps_pipe_directory'],
            'CUDA_MPS_CLIENT_PRIORITY': launch_configs['cuda_mps_client_priority']
        })

    # Static planning scene server
    static_planning_scene_server = ComposableNode(
        name='static_planning_scene_server',
        package='isaac_ros_cumotion',
        plugin='nvidia::isaac_ros::cumotion::StaticPlanningSceneServer',
        parameters=[{
            'moveit_collision_objects_scene_file':
                LaunchConfiguration('cumotion_action_server.moveit_collision_objects_scene_file')
        }],
    )

    cumotion_planner_node = ComposableNode(
        name='cumotion_planner',
        package='isaac_ros_cumotion',
        plugin='nvidia::isaac_ros::cumotion::CumotionPlanner',
        parameters=[
            launch_configs
        ],
    )

    cumotion_container = ComposableNodeContainer(
        name='cumotion_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            static_planning_scene_server,
            cumotion_planner_node,
        ],
        output='screen',
        env=env_variables,
    )

    return launch.LaunchDescription(launch_args + [
        cumotion_container,
    ])
