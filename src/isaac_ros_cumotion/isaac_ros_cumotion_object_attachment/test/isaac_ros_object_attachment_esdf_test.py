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
import time
from typing import ClassVar

from geometry_msgs.msg import TransformStamped
from isaac_ros_cumotion_interfaces.srv import GetRobotDescription, SetRobotDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from nvblox_msgs.srv import EsdfAndGradients
from object_attachment_test_base import (
    GET_ROBOT_DESCRIPTION_SERVICE,
    GRASP_FRAME,
    ObjectAttachmentTestBase,
    SET_ROBOT_DESCRIPTION_SERVICE,
    TEST_URDF,
    TEST_XRDF,
)
import pytest
import rclpy
from scipy.spatial.transform import Rotation
import tf2_ros

WORLD_FRAME = 'world'
ESDF_SERVICE_NAME = 'nvblox_node/get_esdf_and_gradient'

MAX_OVERSHOOT = 0.01  # 1 cm
NODE_STARTUP_DELAY_SEC = 5.0
ASSERT_DECIMAL_PLACES = 3


@pytest.mark.rostest
def generate_test_description():
    """Launch the object attachment node with ESDF clearing enabled."""
    object_attachment_node = ComposableNode(
        name='object_attachment',
        package='isaac_ros_cumotion_object_attachment',
        plugin='nvidia::isaac_ros::cumotion::ObjectAttachmentNode',
        namespace=IsaacROSObjectAttachmentEsdfTest.generate_namespace(),
        parameters=[{
            'get_robot_description_service': GET_ROBOT_DESCRIPTION_SERVICE,
            'set_robot_description_service': SET_ROBOT_DESCRIPTION_SERVICE,
            'max_overshoot': MAX_OVERSHOOT,
            'clear_esdf_on_attach': True,
            'object_esdf_clearing_padding': [0.0, 0.0, 0.0],
            'esdf_reference_frame': WORLD_FRAME,
        }])

    container = ComposableNodeContainer(
        name='object_attachment_esdf_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[object_attachment_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )

    return IsaacROSObjectAttachmentEsdfTest.generate_test_description(
        nodes=[container],
        node_startup_delay=NODE_STARTUP_DELAY_SEC
    )


class IsaacROSObjectAttachmentEsdfTest(ObjectAttachmentTestBase):
    """ESDF clearing tests for object attachment."""

    _esdf_requests: ClassVar[list] = []
    _xrdf_updates: ClassVar[list] = []

    @classmethod
    def setUpClass(cls) -> None:
        """Create the shared ROS node and mock services."""
        super().setUpClass()
        cls.node = rclpy.create_node(
            'esdf_test_node',
            namespace=cls.generate_namespace()
        )
        # All mock services must be discoverable before the first test sends
        # a goal, since the node under test calls wait_for_service at goal time.
        cls._esdf_service = cls.node.create_service(
            EsdfAndGradients,
            ESDF_SERVICE_NAME,
            cls._esdf_callback
        )
        cls._get_service = cls.node.create_service(
            GetRobotDescription,
            GET_ROBOT_DESCRIPTION_SERVICE,
            cls._get_robot_description_callback_cls
        )
        cls._set_service = cls.node.create_service(
            SetRobotDescription,
            SET_ROBOT_DESCRIPTION_SERVICE,
            cls._set_robot_description_callback_cls
        )

    @classmethod
    def _esdf_callback(cls, request, response):
        """Record incoming ESDF clearing requests for later assertion."""
        cls._esdf_requests.append(request)
        response.success = True
        return response

    @classmethod
    def _get_robot_description_callback_cls(cls, request, response):
        del request
        response.urdf = TEST_URDF
        response.xrdf = TEST_XRDF
        return response

    @classmethod
    def _set_robot_description_callback_cls(cls, request, response):
        cls._xrdf_updates.append(request.xrdf)
        response.success = True
        return response

    @classmethod
    def tearDownClass(cls) -> None:
        """Destroy mock services and shared ROS node."""
        cls.node.destroy_service(cls._esdf_service)
        cls.node.destroy_service(cls._get_service)
        cls.node.destroy_service(cls._set_service)
        cls.node.destroy_node()
        super().tearDownClass()

    def setUp(self) -> None:
        """Reset per-test state."""
        self._esdf_requests.clear()
        self._xrdf_updates.clear()

    def tearDown(self) -> None:
        """Detach any attached object."""
        self._cleanup_detach()

    def _publish_world_to_grasp_tf(self, tx, ty, tz, qx, qy, qz, qw):
        """Publish a static TF from WORLD_FRAME to GRASP_FRAME and wait for propagation."""
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = WORLD_FRAME
        t.child_frame_id = GRASP_FRAME
        t.transform.translation.x = float(tx)
        t.transform.translation.y = float(ty)
        t.transform.translation.z = float(tz)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        # A fresh broadcaster instance is required each call: StaticTransformBroadcaster
        # tracks child_frame_ids internally and silently ignores sendTransform calls for
        # already-registered frames on the same instance.
        broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)
        broadcaster.sendTransform(t)
        # The composable container runs its own executor, so a wall-clock sleep is
        # needed to let it receive and process the TF update.
        time.sleep(0.5)

    # ---- ESDF Clearing Tests ----

    def test_sphere_clearing_identity(self):
        """
        Sphere at (1,2,3), uniform scale 0.4, TF=identity.

        Tests that sphere position passes through unchanged with identity TF.
        grasp_frame == world_frame, so marker at (1,2,3) stays at (1,2,3).
        radius = max(0.4, 0.4, 0.4) / 2 = 0.2.
        Expected: center=(1, 2, 3), radius=0.2, no AABBs.
        """
        # grasp_frame coincides with world_frame (identity TF).
        self._publish_world_to_grasp_tf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        marker = self._create_sphere_marker(
            ref_frame=GRASP_FRAME,
            pos_x=1.0, pos_y=2.0, pos_z=3.0,
            qw=1.0, qx=0.0, qy=0.0, qz=0.0,
            scale_x=0.4, scale_y=0.4, scale_z=0.4
        )
        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())

        # One attach should produce exactly one ESDF clearing request.
        self.assertEqual(len(self._esdf_requests), 1)
        req = self._esdf_requests[0]

        self.assertEqual(len(req.spheres_to_clear_center_m), 1)
        self.assertEqual(len(req.spheres_to_clear_radius_m), 1)
        self.assertEqual(len(req.aabbs_to_clear_min_m), 0)

        center = req.spheres_to_clear_center_m[0]
        self.assertAlmostEqual(center.x, 1.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(center.y, 2.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(center.z, 3.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(
            req.spheres_to_clear_radius_m[0], 0.2, places=ASSERT_DECIMAL_PLACES)

    def test_sphere_clearing_non_uniform_scale(self):
        """
        Sphere at origin, non-uniform scale (0.2, 0.4, 0.6), TF=identity.

        Math: radius = max(0.2, 0.4, 0.6) / 2 = 0.3.
        Tests that the max(scale) logic picks the largest dimension.
        Expected: center=(0, 0, 0), radius=0.3, no AABBs.
        """
        # grasp_frame coincides with world_frame (identity TF).
        self._publish_world_to_grasp_tf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        marker = self._create_sphere_marker(
            ref_frame=GRASP_FRAME,
            pos_x=0.0, pos_y=0.0, pos_z=0.0,
            qw=1.0, qx=0.0, qy=0.0, qz=0.0,
            scale_x=0.2, scale_y=0.4, scale_z=0.6
        )
        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())

        # One attach should produce exactly one ESDF clearing request.
        self.assertEqual(len(self._esdf_requests), 1)
        req = self._esdf_requests[0]

        self.assertEqual(len(req.spheres_to_clear_center_m), 1)
        center = req.spheres_to_clear_center_m[0]
        self.assertAlmostEqual(center.x, 0.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(center.y, 0.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(center.z, 0.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(
            req.spheres_to_clear_radius_m[0], 0.3, places=ASSERT_DECIMAL_PLACES)

    def test_cube_clearing_identity(self):
        """
        Cube at origin, scale (2, 4, 6), no rotation, TF=identity.

        Tests that cuboid AABB matches the scale dimensions with no rotation.
        Math: half-extents = (1, 2, 3). 8 vertices at +/-(1, 2, 3).
        AABB min = (-1, -2, -3), max = (1, 2, 3), size = (2, 4, 6).
        Expected: aabb_min=(-1, -2, -3), aabb_size=(2, 4, 6), no spheres.
        """
        # grasp_frame coincides with world_frame (identity TF).
        self._publish_world_to_grasp_tf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        marker = self._create_cube_marker(
            ref_frame=GRASP_FRAME,
            pos_x=0.0, pos_y=0.0, pos_z=0.0,
            qw=1.0, qx=0.0, qy=0.0, qz=0.0,
            scale_x=2.0, scale_y=4.0, scale_z=6.0
        )
        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())

        # One attach should produce exactly one ESDF clearing request.
        self.assertEqual(len(self._esdf_requests), 1)
        req = self._esdf_requests[0]

        self.assertEqual(len(req.aabbs_to_clear_min_m), 1)
        self.assertEqual(len(req.aabbs_to_clear_size_m), 1)
        self.assertEqual(len(req.spheres_to_clear_center_m), 0)

        aabb_min = req.aabbs_to_clear_min_m[0]
        aabb_size = req.aabbs_to_clear_size_m[0]
        self.assertAlmostEqual(aabb_min.x, -1.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_min.y, -2.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_min.z, -3.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_size.x, 2.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_size.y, 4.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_size.z, 6.0, places=ASSERT_DECIMAL_PLACES)

    def test_cube_clearing_with_rotation(self):
        """
        Cube at origin, scale (2, 4, 6), 90-deg Z rotation, TF=identity.

        Tests that rotation correctly swaps AABB dimensions.
        Math: 90-deg Z rotation swaps X and Y for each vertex.
        Vertex (1,2,3) -> (-2,1,3). After all 8 vertices:
        AABB min = (-2, -1, -3), size = (4, 2, 6).
        Expected: X and Y dimensions swap compared to identity case.
        """
        # grasp_frame coincides with world_frame (identity TF).
        self._publish_world_to_grasp_tf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        qx, qy, qz, qw = Rotation.from_euler('z', 90, degrees=True).as_quat()
        marker = self._create_cube_marker(
            ref_frame=GRASP_FRAME,
            pos_x=0.0, pos_y=0.0, pos_z=0.0,
            qw=qw, qx=qx, qy=qy, qz=qz,
            scale_x=2.0, scale_y=4.0, scale_z=6.0
        )
        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())

        # One attach should produce exactly one ESDF clearing request.
        self.assertEqual(len(self._esdf_requests), 1)
        req = self._esdf_requests[0]

        aabb_min = req.aabbs_to_clear_min_m[0]
        aabb_size = req.aabbs_to_clear_size_m[0]
        self.assertAlmostEqual(aabb_min.x, -2.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_min.y, -1.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_min.z, -3.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_size.x, 4.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_size.y, 2.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(aabb_size.z, 6.0, places=ASSERT_DECIMAL_PLACES)

    def test_sphere_clearing_with_tf_offset(self):
        """
        Sphere at (1,2,3) in grasp_frame, TF translation=(10,20,30).

        Tests that TF translation offsets the sphere center in world frame.
        grasp_frame is at (10,20,30) in world, marker is at (1,2,3) in
        grasp_frame, so world position = (11, 22, 33). radius = 0.4/2 = 0.2.
        Expected: center=(11, 22, 33), radius=0.2, no AABBs.
        """
        # Offset grasp_frame by (10, 20, 30) from world_frame.
        self._publish_world_to_grasp_tf(10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0)

        marker = self._create_sphere_marker(
            ref_frame=GRASP_FRAME,
            pos_x=1.0, pos_y=2.0, pos_z=3.0,
            qw=1.0, qx=0.0, qy=0.0, qz=0.0,
            scale_x=0.4, scale_y=0.4, scale_z=0.4
        )
        result = self._attach_object(marker)
        self.assertIn('successfully', result.outcome.lower())

        # One attach should produce exactly one ESDF clearing request.
        self.assertEqual(len(self._esdf_requests), 1)
        req = self._esdf_requests[0]

        center = req.spheres_to_clear_center_m[0]
        self.assertAlmostEqual(center.x, 11.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(center.y, 22.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(center.z, 33.0, places=ASSERT_DECIMAL_PLACES)
        self.assertAlmostEqual(
            req.spheres_to_clear_radius_m[0], 0.2, places=ASSERT_DECIMAL_PLACES)

    def test_mesh_clearing_with_scale(self):
        """
        Mesh (vertices 0-0.2), scale (5,5,5), TF=identity.

        Tests that mesh vertices are scaled correctly to produce the AABB.
        Math: Affine3d scales each vertex by 5: 0.2*5=1.0.
        Transformed vertices range (0,0,0) to (1,1,1).
        Expected: aabb_min=(0, 0, 0), aabb_size=(1, 1, 1), no spheres.
        """
        # grasp_frame coincides with world_frame (identity TF).
        self._publish_world_to_grasp_tf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = self._create_temp_mesh_file(temp_dir)
            marker = self._create_mesh_marker(
                mesh_path, ref_frame=GRASP_FRAME,
                pos_x=0.0, pos_y=0.0, pos_z=0.0,
                qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                scale_x=5.0, scale_y=5.0, scale_z=5.0
            )
            result = self._attach_object(marker)
            self.assertIn('successfully', result.outcome.lower())

            # One attach should produce exactly one ESDF clearing request.
            self.assertEqual(len(self._esdf_requests), 1)
            req = self._esdf_requests[0]

            self.assertEqual(len(req.aabbs_to_clear_min_m), 1)
            aabb_min = req.aabbs_to_clear_min_m[0]
            aabb_size = req.aabbs_to_clear_size_m[0]
            self.assertAlmostEqual(
                aabb_min.x, 0.0, places=ASSERT_DECIMAL_PLACES)
            self.assertAlmostEqual(
                aabb_min.y, 0.0, places=ASSERT_DECIMAL_PLACES)
            self.assertAlmostEqual(
                aabb_min.z, 0.0, places=ASSERT_DECIMAL_PLACES)
            self.assertAlmostEqual(
                aabb_size.x, 1.0, places=ASSERT_DECIMAL_PLACES)
            self.assertAlmostEqual(
                aabb_size.y, 1.0, places=ASSERT_DECIMAL_PLACES)
            self.assertAlmostEqual(
                aabb_size.z, 1.0, places=ASSERT_DECIMAL_PLACES)

    def test_esdf_tf_failure_causes_attach_failure(self):
        """
        When TF lookup fails, ESDF clearing fails and attachment is aborted.

        Tests that an invalid TF frame causes attachment to abort gracefully.
        With clear_esdf_on_attach=True, a TF failure in ClearObjectVoxelsFromWorld
        causes the entire attach to fail with a proper error response (no crash).
        The node remains healthy for subsequent requests.
        """
        marker = self._create_sphere_marker(
            ref_frame='nonexistent_frame',
            pos_x=0.0, pos_y=0.0, pos_z=0.0,
            qw=1.0, qx=0.0, qy=0.0, qz=0.0,
            scale_x=0.2, scale_y=0.2, scale_z=0.2
        )

        result = self._attach_object(marker)
        self.assertIn('fail', result.outcome.lower())
        # TF failure should prevent any ESDF clearing request from being sent.
        self.assertEqual(len(self._esdf_requests), 0)
