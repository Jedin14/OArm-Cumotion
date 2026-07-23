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

import tempfile

from isaac_ros_cumotion_interfaces.srv import GetRobotDescription, SetRobotDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from object_attachment_test_base import (
    GET_ROBOT_DESCRIPTION_SERVICE,
    ObjectAttachmentTestBase,
    SET_ROBOT_DESCRIPTION_SERVICE,
    TEST_XRDF,
)
import pytest
import rclpy
from visualization_msgs.msg import Marker
import yaml

BASE_LINK_FRAME = 'base_link'


@pytest.mark.rostest
def generate_test_description():
    """Launch the object attachment node for POL testing."""
    object_attachment_node = ComposableNode(
        name='object_attachment',
        package='isaac_ros_cumotion_object_attachment',
        plugin='nvidia::isaac_ros::cumotion::ObjectAttachmentNode',
        namespace=IsaacROSObjectAttachmentPOLTest.generate_namespace(),
        parameters=[{
            'get_robot_description_service': GET_ROBOT_DESCRIPTION_SERVICE,
            'set_robot_description_service': SET_ROBOT_DESCRIPTION_SERVICE,
            'max_overshoot': 0.01,
            'clear_esdf_on_attach': False,
        }])

    container = ComposableNodeContainer(
        name='object_attachment_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[object_attachment_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )
    return IsaacROSObjectAttachmentPOLTest.generate_test_description(
        nodes=[container],
        node_startup_delay=5.0
    )


class IsaacROSObjectAttachmentPOLTest(ObjectAttachmentTestBase):
    """Proof-of-life test for object attachment action server."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._shared_node = rclpy.create_node(
            'isaac_ros_object_attachment_pol_test_node',
            namespace=cls.generate_namespace()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._shared_node.destroy_node()
        super().tearDownClass()

    def setUp(self) -> None:
        self.node = self._shared_node
        self._xrdf_updates = []
        self._get_service = self.node.create_service(
            GetRobotDescription,
            GET_ROBOT_DESCRIPTION_SERVICE,
            self._get_robot_description_callback
        )
        self._set_service = self.node.create_service(
            SetRobotDescription,
            SET_ROBOT_DESCRIPTION_SERVICE,
            self._set_robot_description_callback
        )

    def tearDown(self) -> None:
        self._cleanup_detach()
        self.node.destroy_service(self._get_service)
        self.node.destroy_service(self._set_service)

    def test_attach_sphere_uniform_scale(self):
        """Attach a sphere with uniform scale and verify spheres are generated."""
        marker = self._create_sphere_marker(scale_x=0.2, scale_y=0.2, scale_z=0.2)

        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())
        self.assertGreaterEqual(len(self._xrdf_updates), 1)
        self.assertIn('attached_object', self._xrdf_updates[-1])

        # Verify sphere was created in XRDF
        xrdf_yaml = yaml.safe_load(self._xrdf_updates[-1])
        geometry_name = xrdf_yaml['collision']['geometry']
        spheres = xrdf_yaml['geometry'][geometry_name]['spheres']['attached_object']
        self.assertGreater(len(spheres), 0)
        self.assertIn('center', spheres[0])
        self.assertIn('radius', spheres[0])

    def test_attach_sphere_non_uniform_scale(self):
        """Attach a sphere with non-uniform scale (uses max dimension)."""
        marker = self._create_sphere_marker(scale_x=0.1, scale_y=0.3, scale_z=0.2)

        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())

        # Verify sphere was created with max dimension as radius
        xrdf_yaml = yaml.safe_load(self._xrdf_updates[-1])
        geometry_name = xrdf_yaml['collision']['geometry']
        spheres = xrdf_yaml['geometry'][geometry_name]['spheres']['attached_object']
        self.assertGreater(len(spheres), 0)
        # Radius should be 0.3/2 = 0.15
        self.assertAlmostEqual(spheres[0]['radius'], 0.15, places=3)

    def test_attach_cube_uniform_scale(self):
        """Attach a cube with uniform scale and verify spheres are generated."""
        marker = self._create_cube_marker(scale_x=0.2, scale_y=0.2, scale_z=0.2)

        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())
        self.assertIn('attached_object', self._xrdf_updates[-1])

        # Verify collision spheres were generated by cuMotion
        xrdf_yaml = yaml.safe_load(self._xrdf_updates[-1])
        geometry_name = xrdf_yaml['collision']['geometry']
        spheres = xrdf_yaml['geometry'][geometry_name]['spheres']['attached_object']
        self.assertGreater(len(spheres), 0)

    def test_attach_cube_non_uniform_scale(self):
        """Attach a cube with non-uniform scale."""
        marker = self._create_cube_marker(scale_x=0.1, scale_y=0.3, scale_z=0.2)

        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())

        # Verify collision spheres were generated
        xrdf_yaml = yaml.safe_load(self._xrdf_updates[-1])
        geometry_name = xrdf_yaml['collision']['geometry']
        spheres = xrdf_yaml['geometry'][geometry_name]['spheres']['attached_object']
        self.assertGreater(len(spheres), 0)

    def test_detach_object_action(self):
        """Detach after attaching and verify robot description reset."""
        marker = self._create_sphere_marker()

        # Detach immediately if the object is already attached; otherwise attach first.
        detach_result = self._detach_object(expect_accept=False)
        if detach_result is None:
            attach_result = self._attach_object(marker)
            self.assertIn('successfully', attach_result.outcome.lower())
            self.assertIn('attached_object', self._xrdf_updates[-1])
            detach_result = self._detach_object()

        self.assertIn('detached', detach_result.outcome.lower())
        self.assertEqual(self._xrdf_updates[-1], TEST_XRDF)

    def test_attach_mesh_uniform_scale(self):
        """Attach a mesh with uniform scale and verify spheres are generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = self._create_temp_mesh_file(temp_dir)
            marker = self._create_mesh_marker(
                mesh_path, scale_x=1.0, scale_y=1.0, scale_z=1.0)

            attach_result = self._attach_object(marker)
            self.assertIn('successfully', attach_result.outcome.lower())
            self.assertIn('attached_object', self._xrdf_updates[-1])

            xrdf_yaml = yaml.safe_load(self._xrdf_updates[-1])
            geometry_name = xrdf_yaml['collision']['geometry']
            spheres = xrdf_yaml['geometry'][geometry_name]['spheres']['attached_object']
            self.assertGreater(len(spheres), 0)
            self.assertIn('center', spheres[0])
            self.assertIn('radius', spheres[0])

    def test_attach_mesh_non_uniform_scale(self):
        """Attach a mesh with non-uniform scale."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = self._create_temp_mesh_file(temp_dir)
            marker = self._create_mesh_marker(
                mesh_path, scale_x=2.0, scale_y=1.5, scale_z=1.0)

            attach_result = self._attach_object(marker)
            self.assertIn('successfully', attach_result.outcome.lower())

            xrdf_yaml = yaml.safe_load(self._xrdf_updates[-1])
            geometry_name = xrdf_yaml['collision']['geometry']
            spheres = xrdf_yaml['geometry'][geometry_name]['spheres']['attached_object']
            self.assertGreater(len(spheres), 0)

    def test_mesh_attach_detach_action(self):
        """Attach and detach using a mesh and verify robot description updates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = self._create_temp_mesh_file(temp_dir)
            marker = self._create_mesh_marker(mesh_path)

            attach_result = self._attach_object(marker)
            self.assertIn('successfully', attach_result.outcome.lower())
            self.assertIn('attached_object', self._xrdf_updates[-1])

            detach_result = self._detach_object()
            self.assertIn('detached', detach_result.outcome.lower())
            self.assertEqual(self._xrdf_updates[-1], TEST_XRDF)

    def test_reject_double_attach(self):
        """Verify that attaching when an object is already attached is rejected."""
        marker1 = self._create_sphere_marker()
        marker2 = self._create_cube_marker()

        # Attach first object
        result1 = self._attach_object(marker1)
        self.assertIn('successfully', result1.outcome.lower())

        # Try to attach second object (should be rejected)
        result2 = self._attach_object(marker2, expect_accept=False)
        self.assertIsNone(result2)

    def test_reject_detach_when_nothing_attached(self):
        """Verify that detaching when nothing is attached is rejected."""
        # Ensure nothing is attached
        self._cleanup_detach()

        # Try to detach (should be rejected)
        result = self._detach_object(expect_accept=False)
        self.assertIsNone(result)

    def test_unsupported_marker_type(self):
        """Verify that unsupported marker types fail gracefully."""
        marker = Marker()
        marker.header.frame_id = BASE_LINK_FRAME
        marker.type = Marker.ARROW  # Unsupported type
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        result = self._attach_object(marker, expect_accept=True)
        # Should be accepted but fail during execution
        self.assertIn('fail', result.outcome.lower())

    def test_mesh_empty_resource_path(self):
        """Verify that empty mesh resource path fails gracefully."""
        marker = Marker()
        marker.header.frame_id = BASE_LINK_FRAME
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = ''  # Empty path
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        result = self._attach_object(marker, expect_accept=False)
        self.assertIsNone(result)

    def test_mesh_nonexistent_file(self):
        """Verify that non-existent mesh file fails gracefully."""
        marker = self._create_mesh_marker('/tmp/nonexistent_file.obj')

        result = self._attach_object(marker, expect_accept=True)
        self.assertIn('fail', result.outcome.lower())

    def test_mesh_package_uri_not_supported(self):
        """Verify that package:// URIs fail with appropriate error."""
        marker = Marker()
        marker.header.frame_id = BASE_LINK_FRAME
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = 'package://some_package/meshes/test.obj'
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        result = self._attach_object(marker, expect_accept=True)
        self.assertIn('fail', result.outcome.lower())

    def test_reject_zero_scale(self):
        """Verify that a marker with zero scale is rejected."""
        marker = self._create_sphere_marker(scale_x=0.0, scale_y=0.2, scale_z=0.2)
        result = self._attach_object(marker, expect_accept=False)
        self.assertIsNone(result)

    def test_reject_negative_scale(self):
        """Verify that a marker with negative scale is rejected."""
        marker = self._create_cube_marker(scale_x=-0.2, scale_y=0.2, scale_z=0.2)
        result = self._attach_object(marker, expect_accept=False)
        self.assertIsNone(result)

    def test_reject_non_unit_quaternion(self):
        """Verify that a marker with non-unit quaternion is rejected."""
        marker = self._create_sphere_marker()
        marker.pose.orientation.w = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        result = self._attach_object(marker, expect_accept=False)
        self.assertIsNone(result)

    def test_sphere_with_translation(self):
        """Verify sphere can be attached with non-zero position."""
        marker = self._create_sphere_marker()
        marker.pose.position.x = 0.5
        marker.pose.position.y = 0.3
        marker.pose.position.z = 0.1

        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())

        # Verify sphere center is at specified position
        xrdf_yaml = yaml.safe_load(self._xrdf_updates[-1])
        geometry_name = xrdf_yaml['collision']['geometry']
        spheres = xrdf_yaml['geometry'][geometry_name]['spheres']['attached_object']
        self.assertGreater(len(spheres), 0)
        self.assertAlmostEqual(spheres[0]['center'][0], 0.5, places=3)
        self.assertAlmostEqual(spheres[0]['center'][1], 0.3, places=3)
        self.assertAlmostEqual(spheres[0]['center'][2], 0.1, places=3)

    def test_cube_with_rotation(self):
        """Verify cube can be attached with rotation."""
        import math
        marker = self._create_cube_marker()
        # 45 degree rotation around Z axis
        marker.pose.orientation.w = math.cos(math.pi / 8)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(math.pi / 8)

        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())

        # Verify collision spheres were generated
        xrdf_yaml = yaml.safe_load(self._xrdf_updates[-1])
        geometry_name = xrdf_yaml['collision']['geometry']
        spheres = xrdf_yaml['geometry'][geometry_name]['spheres']['attached_object']
        self.assertGreater(len(spheres), 0)

    def test_sequential_attach_detach_cycles(self):
        """Verify multiple attach/detach cycles work correctly."""
        marker1 = self._create_sphere_marker(scale_x=0.1, scale_y=0.1, scale_z=0.1)
        marker2 = self._create_cube_marker(scale_x=0.2, scale_y=0.2, scale_z=0.2)

        # First cycle: sphere
        result = self._attach_object(marker1)
        self.assertIn('successfully', result.outcome.lower())
        result = self._detach_object()
        self.assertIn('detached', result.outcome.lower())

        # Second cycle: cube
        result = self._attach_object(marker2)
        self.assertIn('successfully', result.outcome.lower())
        result = self._detach_object()
        self.assertIn('detached', result.outcome.lower())

        # Third cycle: sphere again
        result = self._attach_object(marker1)
        self.assertIn('successfully', result.outcome.lower())
