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
from isaac_ros_launch_utils import GroupAction
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
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
            launch_args.append(DeclareLaunchArgument(
                name=arg_name, default_value=str(value)))
            launch_configs[param] = LaunchConfiguration(arg_name)

    return launch_args, launch_configs


def launch_setup(context, *args, **kwargs):
    """Launch setup function that resolves container_name at runtime."""
    launch_args, launch_configs = launch_args_from_params(
        'isaac_ros_cumotion_object_attachment',
        'params', 'object_attachment_params.yaml', 'object_attachment')

    # Get container_name from launch configuration
    container_name = str(context.perform_substitution(
        LaunchConfiguration('object_attachment.container_name')))

    # Object attachment node
    object_attachment_node = ComposableNode(
        name='object_attachment',
        package='isaac_ros_cumotion_object_attachment',
        plugin='nvidia::isaac_ros::cumotion::ObjectAttachmentNode',
        parameters=[launch_configs]
    )

    # If container_name is provided and not empty, load into existing container
    # Otherwise, create a new container
    if container_name and container_name.strip():
        load_composable_nodes = LoadComposableNodes(
            target_container=container_name,
            composable_node_descriptions=[object_attachment_node],
        )
        final_launch = GroupAction(
            actions=[load_composable_nodes],
        )
        return [final_launch]
    else:
        # Create a new container if no container_name specified
        object_attachment_container = ComposableNodeContainer(
            name='object_attachment_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[object_attachment_node],
            output='screen',
        )
        return [object_attachment_container]


def generate_launch_description():
    """Launch file to bring up object attachment node."""
    # Declare container_name argument with empty default (will create own container)
    container_name_arg = DeclareLaunchArgument(
        name='object_attachment.container_name',
        default_value='',
        description='Name of an existing container to load the node into. '
                    'If empty, a new container will be created.'
    )

    # Get other launch args from params file
    launch_args, _ = launch_args_from_params(
        'isaac_ros_cumotion_object_attachment',
        'params', 'object_attachment_params.yaml', 'object_attachment')

    return LaunchDescription(
        [container_name_arg] + launch_args + [OpaqueFunction(function=launch_setup)]
    )
