# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared test utilities for object attachment test files."""

import os

from isaac_ros_cumotion_interfaces.action import AttachObject
from isaac_ros_test import IsaacROSBaseTest
import rclpy
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker

GET_ROBOT_DESCRIPTION_SERVICE = IsaacROSBaseTest.generate_namespace(
    'get_robot_description'
)
SET_ROBOT_DESCRIPTION_SERVICE = IsaacROSBaseTest.generate_namespace(
    'set_robot_description'
)
GRASP_FRAME = 'grasp_frame'
ATTACH_OBJECT_ACTION = 'attach_object'
TEST_URDF = '<robot name="test_robot"></robot>'
TEST_XRDF = (
    'collision:\n'
    '  geometry: default_geom\n'
    'geometry:\n'
    '  default_geom:\n'
    '    spheres: {}\n'
)


class ObjectAttachmentTestBase(IsaacROSBaseTest):
    """
    Shared helpers for object attachment POL and ESDF test files.

    Subclasses must still define their own generate_test_description(),
    setUpClass(), setUp(), and tearDown() because node parameters and
    lifecycle differ between test files.

    Marker factory methods require a ref_frame argument -- the coordinate
    frame in which the object's pose (position + orientation) is expressed.
    """

    # ---- Mock service callbacks ----

    def _get_robot_description_callback(self, request, response):
        del request
        response.urdf = TEST_URDF
        response.xrdf = TEST_XRDF
        return response

    def _set_robot_description_callback(self, request, response):
        self._xrdf_updates.append(request.xrdf)
        response.success = True
        return response

    # ---- Marker factory methods ----

    def _create_sphere_marker(
        self, ref_frame=GRASP_FRAME,
        pos_x=0.0, pos_y=0.0, pos_z=0.0,
        qw=1.0, qx=0.0, qy=0.0, qz=0.0,
        scale_x=0.2, scale_y=0.2, scale_z=0.2
    ):
        marker = Marker()
        marker.header.frame_id = ref_frame
        marker.type = Marker.SPHERE
        marker.pose.position.x = float(pos_x)
        marker.pose.position.y = float(pos_y)
        marker.pose.position.z = float(pos_z)
        marker.pose.orientation.w = float(qw)
        marker.pose.orientation.x = float(qx)
        marker.pose.orientation.y = float(qy)
        marker.pose.orientation.z = float(qz)
        marker.scale.x = float(scale_x)
        marker.scale.y = float(scale_y)
        marker.scale.z = float(scale_z)
        return marker

    def _create_cube_marker(
        self, ref_frame=GRASP_FRAME,
        pos_x=0.0, pos_y=0.0, pos_z=0.0,
        qw=1.0, qx=0.0, qy=0.0, qz=0.0,
        scale_x=0.2, scale_y=0.2, scale_z=0.2
    ):
        marker = Marker()
        marker.header.frame_id = ref_frame
        marker.type = Marker.CUBE
        marker.pose.position.x = float(pos_x)
        marker.pose.position.y = float(pos_y)
        marker.pose.position.z = float(pos_z)
        marker.pose.orientation.w = float(qw)
        marker.pose.orientation.x = float(qx)
        marker.pose.orientation.y = float(qy)
        marker.pose.orientation.z = float(qz)
        marker.scale.x = float(scale_x)
        marker.scale.y = float(scale_y)
        marker.scale.z = float(scale_z)
        return marker

    def _create_mesh_marker(
        self, mesh_path, ref_frame=GRASP_FRAME,
        pos_x=0.0, pos_y=0.0, pos_z=0.0,
        qw=1.0, qx=0.0, qy=0.0, qz=0.0,
        scale_x=1.0, scale_y=1.0, scale_z=1.0
    ):
        marker = Marker()
        marker.header.frame_id = ref_frame
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = f'file://{mesh_path}'
        marker.pose.position.x = float(pos_x)
        marker.pose.position.y = float(pos_y)
        marker.pose.position.z = float(pos_z)
        marker.pose.orientation.w = float(qw)
        marker.pose.orientation.x = float(qx)
        marker.pose.orientation.y = float(qy)
        marker.pose.orientation.z = float(qz)
        marker.scale.x = float(scale_x)
        marker.scale.y = float(scale_y)
        marker.scale.z = float(scale_z)
        return marker

    def _create_temp_mesh_file(self, temp_dir):
        """Create a cube mesh with vertices from (0,0,0) to (0.2,0.2,0.2)."""
        mesh_path = os.path.join(temp_dir, 'test_cube.obj')
        with open(mesh_path, 'w', encoding='utf-8') as mesh_file:
            mesh_file.write(
                'v 0.0 0.0 0.0\n'
                'v 0.2 0.0 0.0\n'
                'v 0.2 0.2 0.0\n'
                'v 0.0 0.2 0.0\n'
                'v 0.0 0.0 0.2\n'
                'v 0.2 0.0 0.2\n'
                'v 0.2 0.2 0.2\n'
                'v 0.0 0.2 0.2\n'
                'f 1 2 3\n'
                'f 1 3 4\n'
                'f 5 6 7\n'
                'f 5 7 8\n'
                'f 1 2 6\n'
                'f 1 6 5\n'
                'f 2 3 7\n'
                'f 2 7 6\n'
                'f 3 4 8\n'
                'f 3 8 7\n'
                'f 4 1 5\n'
                'f 4 5 8\n'
            )
        return mesh_path

    # ---- Action helpers ----

    def _create_action_client(self):
        action_client = ActionClient(self.node, AttachObject, ATTACH_OBJECT_ACTION)
        self.assertTrue(
            action_client.wait_for_server(timeout_sec=10.0),
            f'{ATTACH_OBJECT_ACTION} action server did not become available.'
        )
        return action_client

    def _send_goal_and_wait(self, action_client, goal_msg, expect_accept=True):
        send_goal_future = action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(
            self.node, send_goal_future, timeout_sec=5.0)
        goal_handle = send_goal_future.result()
        self.assertIsNotNone(goal_handle)
        if not goal_handle.accepted:
            if expect_accept:
                self.fail('Goal was rejected by action server.')
            return None

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self.node, get_result_future, timeout_sec=15.0)
        result = get_result_future.result()
        self.assertIsNotNone(result)
        return result.result

    def _attach_object(self, marker, expect_accept=True):
        action_client = self._create_action_client()
        goal_msg = AttachObject.Goal()
        goal_msg.attach_object = True
        goal_msg.object_config = marker
        return self._send_goal_and_wait(
            action_client, goal_msg, expect_accept=expect_accept)

    def _detach_object(self, expect_accept=True):
        action_client = self._create_action_client()
        goal_msg = AttachObject.Goal()
        goal_msg.attach_object = False
        goal_msg.object_config = Marker()
        return self._send_goal_and_wait(
            action_client, goal_msg, expect_accept=expect_accept)

    def _cleanup_detach(self):
        try:
            self._detach_object(expect_accept=False)
        except Exception:
            pass
