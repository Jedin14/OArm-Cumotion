# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from copy import deepcopy
from os import path

import os
import threading
import time

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid
from curobo.geom.types import Cylinder
from curobo.geom.types import Mesh
from curobo.geom.types import Sphere
from curobo.geom.types import VoxelGrid as CuVoxelGrid
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState as CuJointState
from curobo.util.logger import setup_curobo_logger
from curobo.wrap.reacher.motion_gen import MotionGen
from curobo.wrap.reacher.motion_gen import MotionGenConfig
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from curobo.wrap.reacher.motion_gen import MotionGenStatus
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from isaac_ros_cumotion.update_kinematics import get_robot_config
from isaac_ros_cumotion.update_kinematics import UpdateLinkSpheresServer
from isaac_ros_cumotion_python_utils.utils import \
    get_grid_center, get_grid_min_corner, get_grid_size, is_grid_valid, \
    load_grid_corners_from_workspace_file
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import RobotTrajectory
import numpy as np
import yaml
from isaac_ros_cumotion import scene_world
from nvblox_msgs.srv import EsdfAndGradients
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
import torch
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker


class CumotionActionServer(Node):

    def __init__(self):
        super().__init__('cumotion_action_server')
        self.tensor_args = TensorDeviceType()
        self.declare_parameter('robot', 'ur5e.yml')
        self.declare_parameter('urdf_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('yml_file_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('time_dilation_factor', 0.5)
        self.declare_parameter('max_attempts', 10)
        # Failed attempts before cuRobo's graph planner is tried; < 0 never.
        # Off by default: with openarm.yml's collision spheres it reports
        # "Start or End state in collision" for states the rest of cuRobo and
        # MoveIt both call valid, and a failed graph attempt ends the whole
        # plan -- measured 2026-09-30, 0/9 plans with it, 9/9 without.
        self.declare_parameter('enable_graph_attempt', -1)
        self.declare_parameter('num_graph_seeds', 6)
        self.declare_parameter('num_trajopt_seeds', 6)
        self.declare_parameter('include_trajopt_retract_seed', True)
        self.declare_parameter('num_trajopt_time_steps', 32)
        self.declare_parameter('trajopt_finetune_iters', 400)
        self.declare_parameter('interpolation_dt', 0.025)
        self.declare_parameter('collision_cache_mesh', 20)
        self.declare_parameter('collision_cache_cuboid', 20)
        self.declare_parameter('voxel_size', 0.05)
        # Clearance the planner keeps from everything in the world, in metres.
        # cuRobo's own default is 0.01; raising it inflates every obstacle,
        # including octomap voxels, so trajectories are pushed further away.
        self.declare_parameter('collision_activation_distance', 0.01)
        self.declare_parameter('read_esdf_world', False)
        self.declare_parameter('publish_curobo_world_as_voxels', False)
        self.declare_parameter('add_ground_plane', False)
        self.declare_parameter('publish_voxel_size', 0.05)
        self.declare_parameter('max_publish_voxels', 500000)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('tool_frame', rclpy.Parameter.Type.STRING)

        # The grid_center_m and grid_size_m parameters are loaded from the workspace file
        # if the workspace_file_path is set and valid.
        self.declare_parameter('workspace_file_path', '')
        self.declare_parameter('grid_center_m', [0.0, 0.0, 0.0])
        self.declare_parameter('grid_size_m', [2.0, 2.0, 2.0])
        self.declare_parameter('update_esdf_on_request', True)
        self.declare_parameter('use_aabb_on_request', True)

        self.declare_parameter('esdf_service_name', '/nvblox_node/get_esdf_and_gradient')
        self.declare_parameter('enable_curobo_debug_mode', False)
        self.declare_parameter('override_moveit_scaling_factors', False)
        self.declare_parameter('update_link_sphere_server',
                               'planner_attach_object')
        debug_mode = (
            self.get_parameter('enable_curobo_debug_mode').get_parameter_value().bool_value
        )
        if debug_mode:
            setup_curobo_logger('info')
        else:
            setup_curobo_logger('warning')

        self.__voxel_pub = self.create_publisher(Marker, '/curobo/voxels', 10)
        self.planner_busy = False
        self.lock = threading.Lock()

        self.__robot_file = self.get_parameter('robot').get_parameter_value().string_value

        try:
            self.__urdf_path = self.get_parameter('urdf_path')
            self.__urdf_path = self.__urdf_path.get_parameter_value().string_value
            if self.__urdf_path == '':
                self.__urdf_path = None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__urdf_path = None

        try:
            self.__yml_path = self.get_parameter('yml_file_path')
            self.__yml_path = self.__yml_path.get_parameter_value().string_value
            if self.__yml_path == '':
                self.__yml_path = None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__yml_path = None

        # If a YAML path is provided, override other XRDF/YAML file name
        if self.__yml_path is not None:
            self.__robot_file = self.__yml_path
        try:
            self.__tool_frame = self.get_parameter('tool_frame')
            self.__tool_frame = self.__tool_frame.get_parameter_value().string_value
            if self.__tool_frame == '':
                self.__tool_frame = None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__tool_frame = None

        self.__collision_activation_distance = (
            self.get_parameter('collision_activation_distance')
            .get_parameter_value().double_value
        )

        self.__joint_states_topic = (
            self.get_parameter('joint_states_topic').get_parameter_value().string_value
        )
        self.__add_ground_plane = (
            self.get_parameter('add_ground_plane').get_parameter_value().bool_value
        )
        self.__override_moveit_scaling_factors = (
            self.get_parameter('override_moveit_scaling_factors').get_parameter_value().bool_value
        )

        # Motion generation parameters

        self.__max_attempts = (
            self.get_parameter('max_attempts').get_parameter_value().integer_value
        )
        graph = self.get_parameter('enable_graph_attempt').get_parameter_value().integer_value
        self.__enable_graph_attempt = None if graph < 0 else graph
        self.__num_graph_seeds = (
            self.get_parameter('num_graph_seeds').get_parameter_value().integer_value
        )
        self.__num_trajopt_seeds = (
            self.get_parameter('num_trajopt_seeds').get_parameter_value().integer_value
        )
        self.__num_trajopt_time_steps = (
            self.get_parameter('num_trajopt_time_steps').get_parameter_value().integer_value
        )
        self.__trajopt_finetune_iters = (
            self.get_parameter('trajopt_finetune_iters').get_parameter_value().integer_value
        )
        self.__interpolation_dt = (
            self.get_parameter('interpolation_dt').get_parameter_value().double_value
        )

        include_trajopt_retract_seed = (
            self.get_parameter('include_trajopt_retract_seed').get_parameter_value().bool_value
        )
        if include_trajopt_retract_seed:
            self.__num_trajopt_noisy_seeds = 1
            self.__trajopt_seed_ratio = {'linear': 1.0}
        else:
            self.__num_trajopt_noisy_seeds = 2
            self.__trajopt_seed_ratio = {'linear': 0.5, 'bias': 0.5}

        collision_cache_cuboid = (
            self.get_parameter('collision_cache_cuboid').get_parameter_value().integer_value
        )
        collision_cache_mesh = (
            self.get_parameter('collision_cache_mesh').get_parameter_value().integer_value
        )
        self.__collision_cache = {
            'obb': collision_cache_cuboid,
            'mesh': collision_cache_mesh
        }

        # ESDF service

        self.__read_esdf_grid = (
            self.get_parameter('read_esdf_world').get_parameter_value().bool_value
        )
        self.__publish_curobo_world_as_voxels = (
            self.get_parameter('publish_curobo_world_as_voxels').get_parameter_value().bool_value
        )
        self.__grid_center_m = (
            self.get_parameter('grid_center_m').get_parameter_value().double_array_value
        )
        self.__max_publish_voxels = (
            self.get_parameter('max_publish_voxels').get_parameter_value().integer_value
        )
        self.__workspace_file_path = (
            self.get_parameter('workspace_file_path').get_parameter_value().string_value
        )
        self.__grid_size_m = (
            self.get_parameter('grid_size_m').get_parameter_value().double_array_value
        )
        self.__update_esdf_on_request = (
            self.get_parameter('update_esdf_on_request').get_parameter_value().bool_value
        )
        self.__use_aabb_on_request = (
            self.get_parameter('use_aabb_on_request').get_parameter_value().bool_value
        )
        self.__publish_voxel_size = (
            self.get_parameter('publish_voxel_size').get_parameter_value().double_value
        )
        self.__voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self._update_link_sphere_server = (
            self.get_parameter(
                'update_link_sphere_server').get_parameter_value().string_value
        )
        self.__esdf_client = None
        self.__esdf_req = None

        # Setup the grid position and dimension.
        if path.exists(self.__workspace_file_path):
            self.get_logger().info(
                f'Loading grid center and dims from workspace file: {self.__workspace_file_path}.')
            min_corner, max_corner = load_grid_corners_from_workspace_file(
                self.__workspace_file_path)
            self.__grid_size_m = get_grid_size(min_corner, max_corner, self.__voxel_size)
            self.__grid_center_m = get_grid_center(min_corner, self.__grid_size_m)

            self.get_logger().info(
                f'Loaded grid dims: {self.__grid_size_m}, ' + f'voxel size: {self.__voxel_size}')
        else:
            self.get_logger().info(
                'Loading grid position and dims from grid_center_m and grid_size_m parameters.')

        if is_grid_valid(self.__grid_size_m, self.__voxel_size):
            self.get_logger().fatal('Number of voxels should be at least 1 in every dimension.')
            raise SystemExit

        if self.__read_esdf_grid:
            esdf_service_name = (
                self.get_parameter('esdf_service_name').get_parameter_value().string_value
            )

            esdf_service_cb_group = MutuallyExclusiveCallbackGroup()
            self.__esdf_client = self.create_client(
                EsdfAndGradients, esdf_service_name, callback_group=esdf_service_cb_group
            )
            while not self.__esdf_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    f'Service({esdf_service_name}) not available, waiting again...'
                )
            self.__esdf_req = EsdfAndGradients.Request()

        self.load_motion_gen()
        self.warmup()
        self.__query_count = 0
        self.__tensor_args = self.motion_gen.tensor_args
        self.subscription = self.create_subscription(
            JointState, self.__joint_states_topic, self.js_callback, 10
        )
        self.__js_buffer = None
        # What move_group checks paths against, for cuMotion to plan around
        # (scene_world.py): tf for the camera pose, the stand column, and a
        # cache for the decoded octomap.
        import tf2_ros
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        try:
            with open(self.__robot_file) as handle:
                urdf_path = yaml.safe_load(handle)['robot_cfg']['kinematics']['urdf_path']
        except Exception:                            # noqa: BLE001
            urdf_path = '/workspaces/isaac_ros-dev/openarm.urdf'
        self._static = scene_world.StaticObstacles(urdf_path, self.get_logger())
        self._octomap = scene_world.OctomapCache()
        self._shadowed = scene_world.ShadowedMap()

        # Call on_timer every 0.01 seconds
        self.timer = self.create_timer(0.01, self.on_timer)

        self.__update_link_spheres_server = UpdateLinkSpheresServer(
            server_node=self,
            action_name=self._update_link_sphere_server,
            robot_kinematics=self.motion_gen.kinematics,
            robot_base_frame=self.__robot_base_frame
        )
        self._action_server = ActionServer(
            self, MoveGroup, 'cumotion/move_group', self.execute_callback
        )

    def js_callback(self, msg):
        self.__js_buffer = {
            'joint_names': msg.name,
            'position': msg.position,
            'velocity': msg.velocity,
        }

    def load_motion_gen(self):
        tensor_args = self.tensor_args
        world_file = WorldConfig.from_dict(
            {
                'cuboid': {
                    'table': {
                        'pose': [0, 0, -0.05, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
                        'dims': [2.0, 2.0, 0.1],
                    }
                },
                'voxel': {
                    'world_voxel': {
                        'dims': self.__grid_size_m,
                        'pose': [0, 0, 0, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
                        'voxel_size': self.__voxel_size,
                        'feature_dtype': torch.bfloat16,
                    },
                },
            }
        )

        robot_config = get_robot_config(
            robot_file=self.__robot_file,
            urdf_file_path=self.__urdf_path,
            logger=self.get_logger()
        )

        robot_dict = robot_config['robot_cfg']
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_dict,
            world_file,
            tensor_args,
            num_graph_seeds=self.__num_graph_seeds,
            num_trajopt_seeds=self.__num_trajopt_seeds,
            num_trajopt_noisy_seeds=self.__num_trajopt_noisy_seeds,
            trajopt_tsteps=self.__num_trajopt_time_steps,
            trajopt_seed_ratio=self.__trajopt_seed_ratio,
            interpolation_dt=self.__interpolation_dt,
            collision_cache=self.__collision_cache,
            collision_checker_type=CollisionCheckerType.VOXEL,
            collision_activation_distance=self.__collision_activation_distance,
            ee_link_name=self.__tool_frame,
            finetune_trajopt_iters=self.__trajopt_finetune_iters,
        )

        motion_gen = MotionGen(motion_gen_config)
        self.motion_gen = motion_gen
        self.__robot_base_frame = self.motion_gen.kinematics.base_link

        self.get_logger().info(
            'collision activation distance: '
            f'{self.__collision_activation_distance:.3f} m')

        self.__world_collision = self.motion_gen.world_coll_checker
        if not self.__add_ground_plane:
            self.motion_gen.clear_world_cache()
        self.__cumotion_grid_shape = self.__world_collision.get_voxel_grid(
            'world_voxel').get_grid_shape()[0]

    def warmup(self):
        self.get_logger().info('warming up cuMotion, wait until ready')
        self.motion_gen.warmup(enable_graph=True)
        self.get_logger().info('cuMotion is ready for planning queries!')

    def on_timer(self):
        with self.lock:
            if self.__js_buffer is None:
                return

            js = np.copy(self.__js_buffer['position'])
            j_names = deepcopy(self.__js_buffer['joint_names'])

        self.__update_link_spheres_server.publish_all_active_spheres(
            robot_joint_states=js,
            robot_joint_names=j_names,
            tensor_args=self.__tensor_args,
            rgb=[0.0, 1.0, 1.0, 1.0]
        )

    def update_voxel_grid(self):
        self.get_logger().info('Calling ESDF service')

        # Get the AABB
        min_corner = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)
        aabb_min = Point()
        aabb_min.x = min_corner[0]
        aabb_min.y = min_corner[1]
        aabb_min.z = min_corner[2]
        aabb_size = Vector3()
        aabb_size.x = self.__grid_size_m[0]
        aabb_size.y = self.__grid_size_m[1]
        aabb_size.z = self.__grid_size_m[2]

        # Request the esdf grid
        esdf_future = self.send_request(aabb_min, aabb_size)
        while not esdf_future.done():
            time.sleep(0.001)
        response = esdf_future.result()
        if not response.success:
            self.get_logger().info('ESDF request failed, try again after few seconds.')
            return False
        esdf_grid = self.get_esdf_voxel_grid(response)
        if torch.max(esdf_grid.feature_tensor) <= (-1000.0 + 0.5 * self.__voxel_size + 1e-5):
            self.get_logger().error('ESDF data is empty, try again after few seconds.')
            return False
        self.__world_collision.update_voxel_data(esdf_grid)
        self.get_logger().info('Updated ESDF grid')
        return True

    def send_request(self, aabb_min_m, aabb_size_m):
        self.__esdf_req.visualize_esdf = True
        self.__esdf_req.update_esdf = self.__update_esdf_on_request
        self.__esdf_req.use_aabb = self.__use_aabb_on_request
        self.__esdf_req.frame_id = self.__robot_base_frame
        self.__esdf_req.aabb_min_m = aabb_min_m
        self.__esdf_req.aabb_size_m = aabb_size_m
        self.get_logger().info(
            f'ESDF  req = {self.__esdf_req.aabb_min_m}, {self.__esdf_req.aabb_size_m}'
        )
        esdf_future = self.__esdf_client.call_async(self.__esdf_req)

        return esdf_future

    def get_esdf_voxel_grid(self, esdf_data):
        esdf_voxel_size = esdf_data.voxel_size_m
        if abs(esdf_voxel_size - self.__voxel_size) > 1e-4:
            self.get_logger().fatal(
                'Voxel size of esdf array is not equal to requested voxel_size, '
                f'{esdf_voxel_size} vs. {self.__voxel_size}')
            raise SystemExit

        # Get the esdf and gradient data
        esdf_array = esdf_data.esdf_and_gradients
        array_shape = [
            esdf_array.layout.dim[0].size,
            esdf_array.layout.dim[1].size,
            esdf_array.layout.dim[2].size,
        ]
        array_data = np.array(esdf_array.data, dtype=np.float32)
        if (array_data.shape[0] <= 0):
            self.get_logger().fatal(
                'array shape is zero: ' + str(array_data.shape)
            )
            raise SystemExit
        array_data = torch.as_tensor(array_data)

        # Verify the grid shape
        if array_shape != self.__cumotion_grid_shape:
            self.get_logger().fatal(
                'Shape of received esdf voxel grid does not match the cumotion grid shape, '
                f'{array_shape} vs. {self.__cumotion_grid_shape}')
            raise SystemExit

        # Get the origin of the grid
        grid_origin = [
            esdf_data.origin_m.x,
            esdf_data.origin_m.y,
            esdf_data.origin_m.z,
        ]
        # The grid position is defined as the center point of the grid.
        grid_center_m = get_grid_center(grid_origin, self.__grid_size_m)

        # Array data is reshaped to x y z channels
        array_data = array_data.view(array_shape[0], array_shape[1], array_shape[2]).contiguous()

        # Array is squeezed to 1 dimension
        array_data = array_data.reshape(-1, 1)

        # nvblox assigns a value of -1000.0 for unobserved voxels, making it positive
        array_data[array_data < -999.9] = 1000.0

        # nvblox uses negative distance inside obstacles, cuRobo needs the opposite:
        array_data = -1.0 * array_data

        # nvblox treats surface voxels as distance = 0.0, while cuRobo treats
        # distance = 0.0 as not in collision. Adding an offset.
        array_data += 0.5 * self.__voxel_size

        esdf_grid = CuVoxelGrid(
            name='world_voxel',
            dims=self.__grid_size_m,
            pose=grid_center_m + [1, 0.0, 0.0, 0.0],  # x, y, z, qw, qx, qy, qz
            voxel_size=self.__voxel_size,
            feature_dtype=torch.float32,
            feature_tensor=array_data,
        )

        return esdf_grid

    def get_cumotion_collision_object(self, mv_object: CollisionObject):
        objs = []
        pose = mv_object.pose

        world_pose = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ]
        world_pose = Pose.from_list(world_pose)
        supported_objects = True
        if len(mv_object.primitives) > 0:
            for k in range(len(mv_object.primitives)):
                pose = mv_object.primitive_poses[k]
                primitive_pose = [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ]
                object_pose = world_pose.multiply(Pose.from_list(primitive_pose)).tolist()

                if mv_object.primitives[k].type == SolidPrimitive.BOX:
                    # cuboid:
                    dims = mv_object.primitives[k].dimensions
                    obj = Cuboid(
                        name=str(mv_object.id) + '_' + str(k) + '_cuboid',
                        pose=object_pose,
                        dims=dims,
                    )
                    objs.append(obj)
                elif mv_object.primitives[k].type == SolidPrimitive.SPHERE:
                    # sphere:
                    radius = mv_object.primitives[k].dimensions[
                        mv_object.primitives[k].SPHERE_RADIUS
                    ]
                    obj = Sphere(
                        name=str(mv_object.id) + '_' + str(k) + '_sphere',
                        pose=object_pose,
                        radius=radius,
                    )
                    objs.append(obj)
                elif mv_object.primitives[k].type == SolidPrimitive.CYLINDER:
                    # cylinder:
                    cyl_height = mv_object.primitives[k].dimensions[
                        mv_object.primitives[k].CYLINDER_HEIGHT
                    ]
                    cyl_radius = mv_object.primitives[k].dimensions[
                        mv_object.primitives[k].CYLINDER_RADIUS
                    ]
                    obj = Cylinder(
                        name=str(mv_object.id) + '_' + str(k) + '_cylinder',
                        pose=object_pose,
                        height=cyl_height,
                        radius=cyl_radius,
                    )
                    objs.append(obj)
                elif mv_object.primitives[k].type == SolidPrimitive.CONE:
                    self.get_logger().error('Cone primitive is not supported')
                    supported_objects = False
                else:
                    self.get_logger().error('Unknown primitive type')
                    supported_objects = False
        if len(mv_object.meshes) > 0:
            for k in range(len(mv_object.meshes)):
                pose = mv_object.mesh_poses[k]
                mesh_pose = [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ]
                object_pose = world_pose.multiply(Pose.from_list(mesh_pose)).tolist()
                verts = mv_object.meshes[k].vertices
                verts = [[v.x, v.y, v.z] for v in verts]
                tris = [
                    [v.vertex_indices[0], v.vertex_indices[1], v.vertex_indices[2]]
                    for v in mv_object.meshes[k].triangles
                ]

                obj = Mesh(
                    name=str(mv_object.id) + '_' + str(len(objs)) + '_mesh',
                    pose=object_pose,
                    vertices=verts,
                    faces=tris,
                )
                objs.append(obj)
        return objs, supported_objects

    def get_joint_trajectory(self, js: CuJointState, dt: float, time_scale: float = 1.0):
        """`time_scale` < 1 slows the trajectory down: t/s, v*s, a*s^2."""
        traj = RobotTrajectory()
        cmd_traj = JointTrajectory()
        s = min(1.0, max(1e-3, float(time_scale or 1.0)))
        dt = dt / s
        q_traj = js.position.cpu().view(-1, js.position.shape[-1]).numpy()
        vel = js.velocity.cpu().view(-1, js.position.shape[-1]).numpy() * s
        acc = js.acceleration.view(-1, js.position.shape[-1]).cpu().numpy() * (s * s)
        for i in range(len(q_traj)):
            traj_pt = JointTrajectoryPoint()
            traj_pt.positions = q_traj[i].tolist()
            if js is not None and i < len(vel):
                traj_pt.velocities = vel[i].tolist()
            if js is not None and i < len(acc):
                traj_pt.accelerations = acc[i].tolist()
            time_d = rclpy.time.Duration(seconds=i * dt).to_msg()
            traj_pt.time_from_start = time_d
            cmd_traj.points.append(traj_pt)
        cmd_traj.joint_names = js.joint_names
        cmd_traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_trajectory = cmd_traj
        return traj

    def update_world_objects(self, moveit_objects):
        world_update_status = True
        if len(moveit_objects) > 0:
            cuboid_list = []
            sphere_list = []
            cylinder_list = []
            mesh_list = []
            for i, obj in enumerate(moveit_objects):
                cumotion_objects, world_update_status = self.get_cumotion_collision_object(obj)
                for cumotion_object in cumotion_objects:
                    if isinstance(cumotion_object, Cuboid):
                        cuboid_list.append(cumotion_object)
                    elif isinstance(cumotion_object, Cylinder):
                        cylinder_list.append(cumotion_object)
                    elif isinstance(cumotion_object, Sphere):
                        sphere_list.append(cumotion_object)
                    elif isinstance(cumotion_object, Mesh):
                        mesh_list.append(cumotion_object)
            if len(cuboid_list) > self.__collision_cache['obb'] - 2:
                # more boxes than the cuboid cache holds: one mesh (see _scene_world)
                v, f = scene_world.cuboids_mesh(cuboid_list)
                mesh_list.append(Mesh(name='scene_boxes', pose=[0, 0, 0, 1, 0, 0, 0],
                                      vertices=v.tolist(), faces=f.tolist()))
                cuboid_list = []

            world_model = WorldConfig(
                cuboid=cuboid_list,
                cylinder=cylinder_list,
                sphere=sphere_list,
                mesh=mesh_list,
            ).get_collision_check_world()
            self.motion_gen.update_world(world_model)
        if self.__read_esdf_grid:
            world_update_status = self.update_voxel_grid()
        if self.__publish_curobo_world_as_voxels:
            if self.__voxel_pub.get_subscription_count() > 0:
                # Calculate occupancy and publish only when subscribed.
                voxels = self.__world_collision.get_occupancy_in_bounding_box(
                    Cuboid(
                        name='test',
                        pose=[0.0, 0.0, 0.0, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
                        dims=self.__grid_size_m,
                    ),
                    voxel_size=self.__publish_voxel_size,
                )
                xyzr_tensor = voxels.xyzr_tensor.clone()
                xyzr_tensor[..., 3] = voxels.feature_tensor
                self.publish_voxels(xyzr_tensor)
        return world_update_status

    def _scene_world(self, scene, keep_clear, with_stand=True, with_octomap=True,
                     arm_spheres=None):
        """Give cuMotion move_group's obstacles: scene collision objects, the
        camera box, the stand column and the octomap (cropped to the arms'
        reach, and cleared only within 3 cm of `keep_clear`, the tool points
        at the start and goal: 10 cm used to delete objects right next to the
        approach point, so a plan could sweep through them -- 2026-10-09).
        The contact itself is a move_group Cartesian move."""
        cuboids, meshes, spheres, cylinders = [], [], [], []
        for obj in scene.world.collision_objects:
            for cu_obj in self.get_cumotion_collision_object(obj)[0]:
                {Cuboid: cuboids, Mesh: meshes, Sphere: spheres,
                 Cylinder: cylinders}.get(type(cu_obj), meshes).append(cu_obj)
        notes = []
        if len(cuboids) > self.__collision_cache['obb'] - 2:
            # More boxes than cuRobo has room for (it then fails every plan:
            # "number of OBB is larger than collision cache"): one mesh instead.
            v, f = scene_world.cuboids_mesh(cuboids)
            meshes.append(Mesh(name='scene_boxes', pose=[0, 0, 0, 1, 0, 0, 0],
                               vertices=v.tolist(), faces=f.tolist()))
            notes.append(f'{len(cuboids)} scene boxes as one mesh')
            cuboids = []
        cam = self._static.camera(self._tf_buffer, self.__robot_base_frame)
        if cam is not None:
            cuboids.append(Cuboid(name='camera', pose=cam,
                                  dims=list(scene_world.StaticObstacles.CAMERA_BOX)))
            notes.append('camera')
        if with_stand and self._static.stand is not None:
            if not getattr(self, '_body_spheres_off', False):
                # The stand is modelled exactly as a world mesh from here on,
                # so the robot's own body spheres go: they overlap that mesh
                # (and the calibrated camera box inside it), which made every
                # start pose "colliding with world". Arm-vs-stand is then a
                # world check against the real shape instead of a sphere fit.
                self.motion_gen.kinematics.kinematics_config.disable_link_spheres(
                    'openarm_body_link0')
                self._body_spheres_off = True
            v, f = self._static.stand
            meshes.append(Mesh(name='stand', pose=[0, 0, 0, 1, 0, 0, 0],
                               vertices=v.tolist(), faces=f.tolist()))
            notes.append('stand')
        if with_octomap and scene.world.octomap.octomap.data:
            octomap = scene.world.octomap.octomap
            centres, sizes = self._octomap.get(octomap)
            o = scene.world.octomap.origin.position
            centres = centres + [o.x, o.y, o.z]

            def crop(c):
                return ((np.abs(c[:, 0]) < 1.0) & (np.abs(c[:, 1]) < 1.0)
                        & (c[:, 2] > -0.1) & (c[:, 2] < 1.4))

            eye = self._static.eye(self._tf_buffer, self.__robot_base_frame)
            (v, f), n = self._shadowed.mesh(centres, sizes, float(octomap.resolution) or 0.02,
                                            eye, crop, keep_clear, spheres=arm_spheres)
            if n:
                meshes.append(Mesh(name='octomap', pose=[0, 0, 0, 1, 0, 0, 0],
                                   vertices=v.tolist(), faces=f.tolist()))
                surface, shadowed = self._shadowed.counts
                notes.append(f'octomap {surface} cells + {shadowed - surface} hidden behind '
                             f'them, {n} after clearing the grippers ({len(f)} triangles)')
        world = WorldConfig(cuboid=cuboids, mesh=meshes, sphere=spheres,
                            cylinder=cylinders).get_collision_check_world()
        # cuRobo caches meshes by name: without this a new octomap under the
        # same name is silently replaced by the old one.
        self.motion_gen.clear_world_cache()
        self.motion_gen.update_world(world)
        self.get_logger().info('cuMotion world: ' + (', '.join(notes) or 'empty'))

    def _robot_spheres(self, js):
        """(N, 4) world collision spheres of the robot at a joint state."""
        try:
            state = self.motion_gen.kinematics.get_state(js.position.view(1, -1))
            return state.link_spheres_tensor.view(-1, 4).cpu().numpy()
        except Exception:                            # noqa: BLE001
            return None

    def _tcp_positions(self, js):
        """World positions of both hand_tcp links for a joint state."""
        try:
            poses = self.motion_gen.kinematics.get_state(js.position.view(1, -1)).link_poses
            return [poses[name].position.view(-1).cpu().numpy()
                    for name in poses if name.endswith('hand_tcp')]
        except Exception:                            # noqa: BLE001
            return []

    def _self_collision_costs(self):
        """Every SelfCollisionCost inside motion_gen, found once."""
        if getattr(self, '_sc_costs', None) is None:
            from curobo.rollout.cost.self_collision_cost import SelfCollisionCost
            found, seen, stack = [], set(), [(self.motion_gen, 0)]
            while stack:
                obj, depth = stack.pop()
                if id(obj) in seen or depth > 8:
                    continue
                seen.add(id(obj))
                if isinstance(obj, SelfCollisionCost):
                    found.append(obj)
                elif isinstance(obj, (list, tuple)):
                    stack += [(x, depth + 1) for x in obj]
                elif isinstance(obj, dict):
                    stack += [(x, depth + 1) for x in obj.values()]
                elif (hasattr(obj, '__dict__') and not isinstance(obj, torch.Tensor)
                      and type(obj).__module__.startswith('curobo')):
                    stack += [(x, depth + 1) for x in vars(obj).values()]
            self._sc_costs = found
            self.get_logger().info(f'{len(found)} self-collision cost buffers will be '
                                   f'cleared before every plan')
        return self._sc_costs

    def clear_self_collision_buffers(self):
        """Zero cuRobo's self-collision output buffers before a plan.

        The self-collision kernel writes a distance only where it finds a
        collision and never clears the others, and these buffers are reused
        across plans. So once any trajectory collided at, say, interpolation
        step 53, step 53 read as colliding for *every* later trajectory: the
        final check rejected paths that were fine, and after a few bad goals
        cuMotion answered TRAJOPT_FAIL to everything -- goals it had planned
        twenty times -- until it was restarted. That is the arm that "didn't
        move" (2026-09-30). Measured offline on this robot: 46/150 plans
        without this, pre_pick and home failing every time after plan ~60;
        115/150 with it, pre_pick and home 100/100.
        """
        for cost in self._self_collision_costs():
            for name in ('_out_distance', '_out_vec', '_sparse_sphere_idx'):
                buf = getattr(cost, name, None)
                if isinstance(buf, torch.Tensor):
                    buf.zero_()

    def _still_healthy(self, start_state):
        """Can the planner still do something trivial? Run after repeated
        failures: a 0.05 rad move from where the arm is. If *that* fails, the
        solver state is broken (see _within_limits) and every later plan would
        fail too, so the node exits and launch respawns it -- ~30 s, instead
        of an arm that never moves again."""
        try:
            goal = start_state.clone()
            names = self.motion_gen.kinematics.kinematics_config.joint_names
            limits = self.motion_gen.kinematics.kinematics_config.joint_limits.position
            for i, name in enumerate(goal.joint_names):
                if name in names:
                    j = names.index(name)
                    mid = 0.5 * float(limits[0][j] + limits[1][j])
                    step = 0.05 if float(goal.position[0, i]) < mid else -0.05
                    goal.position[0, i] += step
                    break
            self.motion_gen.reset(reset_seed=False)
            self.clear_self_collision_buffers()
            test = self.motion_gen.plan_single_js(
                start_state, goal, MotionGenPlanConfig(max_attempts=4, enable_graph=False,
                                                       enable_finetune_trajopt=False))
            return bool(test.success.item()) or not test.valid_query
        except Exception as exc:                     # noqa: BLE001
            self.get_logger().warn(f'planner self-check could not run: {exc}')
            return True

    def _within_limits(self, js, commanded=(), label='state'):
        """Clamp a joint state into cuRobo's joint limits, in place.

        An out-of-limits joint *goal* poisons cuRobo: plan_single_js does not
        check it, trajopt runs on it, and from then on every plan fails with
        TRAJOPT_FAIL -- even goals that succeeded 20 times before -- until the
        node is restarted (reproduced 2026-09-30, click_to_move's arm
        "didn't move"). The hardware sags a little past its limits (right
        joint3 read -1.604 against -1.571), and a 7-joint goal is completed
        from the *current* reading of the other arm, so ordinary goals carried
        such values. Joints nobody commanded, and the start state, are pulled
        just inside (MoveIt allows the same 0.1 rad at the start). A commanded
        joint more than 0.01 rad outside is refused: that goal is wrong.
        Returns the name of a refused joint, or None.
        """
        limits = self.motion_gen.kinematics.kinematics_config.joint_limits.position
        names = self.motion_gen.kinematics.kinematics_config.joint_names
        margin = 1e-4
        for i, name in enumerate(js.joint_names):
            if name not in names:
                continue
            j = names.index(name)
            lo, hi = float(limits[0][j]), float(limits[1][j])
            value = float(js.position[0, i])
            if lo + margin <= value <= hi - margin:
                continue
            if name in commanded and (value < lo - 0.01 or value > hi + 0.01):
                return name
            self.get_logger().info(
                f'{label}: {name} = {value:.4f} is outside [{lo:.4f}, {hi:.4f}]; '
                f'clamped')
            js.position[0, i] = min(max(value, lo + margin), hi - margin)
        return None

    def execute_callback(self, goal_handle):
        if self.planner_busy:
            self.get_logger().error('Planner is busy')
            goal_handle.abort()
            result = MoveGroup.Result()
            result.error_code.val = MoveItErrorCodes.FAILURE
            return result

        plan_req = goal_handle.request.request
        if 'gripper' in plan_req.group_name.lower():
            self.get_logger().error(f"cuMotion does not support planning for {plan_req.group_name}. Please switch to 'ompl' pipeline in RViz.")
            goal_handle.abort()
            result = MoveGroup.Result()
            result.error_code.val = MoveItErrorCodes.INVALID_GROUP_NAME
            return result

        self.get_logger().info('Executing goal...')

        # check moveit scaling factors:
        min_scaling_factor = min(goal_handle.request.request.max_velocity_scaling_factor,
                                 goal_handle.request.request.max_acceleration_scaling_factor)
        time_dilation_factor = min(1.0, min_scaling_factor)

        if time_dilation_factor <= 0.0 or self.__override_moveit_scaling_factors:
            time_dilation_factor = self.get_parameter(
                'time_dilation_factor').get_parameter_value().double_value
        self.get_logger().info('Planning with time_dilation_factor: ' +
                               str(time_dilation_factor))
        plan_req = goal_handle.request.request

        goal_handle.succeed()

        scene = goal_handle.request.planning_options.planning_scene_diff

        world_objects = scene.world.collision_objects
        world_update_status = self.update_world_objects(world_objects)
        result = MoveGroup.Result()

        if not world_update_status:
            result.error_code.val = MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE
            self.get_logger().error('World update failed.')
            return result
        start_state = None
        if len(plan_req.start_state.joint_state.position) > 0:
            # Merge partial start state with current full state
            current_full_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(self.__js_buffer['position']).unsqueeze(0),
                    joint_names=self.__js_buffer['joint_names'],
                )
            )
            
            start_jnames = plan_req.start_state.joint_state.name
            start_js_pos_tensor = self.tensor_args.to_device(
                plan_req.start_state.joint_state.position
            ).unsqueeze(0)
            
            for idx, j_name in enumerate(start_jnames):
                if j_name in current_full_state.joint_names:
                    full_idx = current_full_state.joint_names.index(j_name)
                    current_full_state.position[0, full_idx] = start_js_pos_tensor[0, idx]
                    
            start_state = current_full_state
        else:
            self.get_logger().info(
                'PlanRequest start state was empty, reading current joint state'
            )
        if start_state is None or plan_req.start_state.is_diff:
            if self.__js_buffer is None:
                self.get_logger().error(
                    'joint_state was not received from ' + self.__joint_states_topic
                )
                return result

            # read joint state:
            state = CuJointState.from_position(
                position=self.tensor_args.to_device(self.__js_buffer['position']).unsqueeze(0),
                joint_names=self.__js_buffer['joint_names'],
            )
            state.velocity = self.tensor_args.to_device(self.__js_buffer['velocity']).unsqueeze(0)
            if state.velocity.shape != state.position.shape:
                self.get_logger().error(
                    'start joint position shape is  ' + str(state.position.shape) +
                    ' start velocity shape is ' + str(state.velocity.shape) +
                    ', both should match. JointState was read from ' + self.__joint_states_topic
                )
                return result
            current_joint_state = self.motion_gen.get_active_js(state)
            if start_state is not None and plan_req.start_state.is_diff:
                start_state.position += current_joint_state.position
                start_state.velocity += current_joint_state.velocity
            else:
                start_state = current_joint_state

        if len(plan_req.goal_constraints[0].joint_constraints) > 0:
            self.get_logger().info('Calculating goal pose from Joint target')
            goal_config = [
                plan_req.goal_constraints[0].joint_constraints[x].position
                for x in range(len(plan_req.goal_constraints[0].joint_constraints))
            ]
            goal_jnames = [
                plan_req.goal_constraints[0].joint_constraints[x].joint_name
                for x in range(len(plan_req.goal_constraints[0].joint_constraints))
            ]

            # Merge partial goal with current full state
            current_full_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(self.__js_buffer['position']).unsqueeze(0),
                    joint_names=self.__js_buffer['joint_names'],
                )
            )
            goal_js_pos_tensor = self.tensor_args.to_device(goal_config).view(1, -1)
            
            for idx, j_name in enumerate(goal_jnames):
                if j_name in current_full_state.joint_names:
                    full_idx = current_full_state.joint_names.index(j_name)
                    current_full_state.position[0, full_idx] = goal_js_pos_tensor[0, idx]

            goal_state = current_full_state
            refused = self._within_limits(goal_state, set(goal_jnames), 'goal')
            if refused is not None:
                self.get_logger().error(
                    f'Refusing a joint goal outside the joint limits ({refused}); '
                    f'planning to it would break every plan after it')
                result.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
                return result
            goal_pose = self.motion_gen.compute_kinematics(goal_state).ee_pose.clone()
        elif (
            len(plan_req.goal_constraints[0].position_constraints) > 0
            and len(plan_req.goal_constraints[0].orientation_constraints) > 0
        ):
            self.get_logger().info('Using goal from Pose')

            position = (
                plan_req.goal_constraints[0]
                .position_constraints[0]
                .constraint_region.primitive_poses[0]
                .position
            )
            position = [position.x, position.y, position.z]
            orientation = plan_req.goal_constraints[0].orientation_constraints[0].orientation
            orientation = [orientation.w, orientation.x, orientation.y, orientation.z]
            pose_list = position + orientation
            goal_pose = Pose.from_list(pose_list, tensor_args=self.tensor_args)

            # Check if link names match:
            position_link_name = plan_req.goal_constraints[0].position_constraints[0].link_name
            orientation_link_name = (
                plan_req.goal_constraints[0].orientation_constraints[0].link_name
            )
            plan_link_name = self.motion_gen.kinematics.ee_link
            if position_link_name != orientation_link_name:
                self.get_logger().error(
                    'Link name for Target Position "'
                    + position_link_name
                    + '" and Target Orientation "'
                    + orientation_link_name
                    + '" do not match'
                )
                result.error_code.val = MoveItErrorCodes.INVALID_LINK_NAME
                return result
            if position_link_name != plan_link_name:
                self.get_logger().error(
                    'Link name for Target Pose "'
                    + position_link_name
                    + '" and Planning frame "'
                    + plan_link_name
                    + '" do not match, relaunch node with tool_frame = '
                    + position_link_name
                )
                result.error_code.val = MoveItErrorCodes.INVALID_LINK_NAME
                return result
        else:
            self.get_logger().error('Goal constraints not supported')
        if start_state is not None:
            self._within_limits(start_state, (), 'start')
        with self.lock:
            self.planner_busy = True

        keep_clear = self._tcp_positions(start_state) if start_state is not None else []
        if len(plan_req.goal_constraints[0].joint_constraints) > 0:
            keep_clear += self._tcp_positions(goal_state)
        else:
            keep_clear.append(goal_pose.position.view(-1).cpu().numpy())
        self._scene_world(scene, keep_clear)

        self.motion_gen.reset(reset_seed=False)
        self.clear_self_collision_buffers()
        # Planned at full speed and slowed down afterwards (get_joint_trajectory),
        # not planned slow. cuRobo's time_dilation_factor makes the optimiser
        # fit a slow trajectory into its fixed number of steps, and with the
        # robot's own collision spheres in the model -- so that the path has to
        # go round the torso -- that fails outright: measured 0/8 at 0.15
        # against 4/4 undilated for the same home -> pre_pick move. A pure time
        # scaling afterwards gives the same slow motion along a path that
        # exists.
        # No finetune pass. It only shortens the trajectory's timing, which
        # is thrown away anyway (the trajectory is slowed afterwards), and on
        # this robot it failed every approach goal click_to_move asked for --
        # 0/10 with it, 10/10 without, all 10 valid in move_group
        # (2026-10-06, FINETUNE_TRAJOPT_FAIL in the log).
        plan_config = MotionGenPlanConfig(
            max_attempts=self.__max_attempts,
            enable_graph=False,
            enable_graph_attempt=self.__enable_graph_attempt,
            enable_finetune_trajopt=False,
        )
        
        if len(plan_req.goal_constraints[0].joint_constraints) > 0:
            motion_gen_result = self.motion_gen.plan_single_js(
                start_state,
                goal_state,
                plan_config,
            )
        else:
            motion_gen_result = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                plan_config,
            )
        if motion_gen_result.status in (MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,):
            # The stand column is a coarse hull; an arm parked close beside it
            # can read as touching. Plan once more without it.
            self.get_logger().warn('start state touches the modelled stand/obstacles; '
                                   'replanning without the stand column')
            self._scene_world(scene, keep_clear, with_stand=False)
            self.motion_gen.reset(reset_seed=False)
            self.clear_self_collision_buffers()
            if len(plan_req.goal_constraints[0].joint_constraints) > 0:
                motion_gen_result = self.motion_gen.plan_single_js(start_state, goal_state,
                                                                   plan_config)
            else:
                motion_gen_result = self.motion_gen.plan_single(start_state, goal_pose,
                                                                plan_config)
        if (motion_gen_result.status == MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION
                and start_state is not None):
            # Still colliding: the octomap claims space the arm is standing in
            # -- usually cells filled in as hidden behind a surface. The arm
            # is physically there, so that space is free; clear the map
            # around the arm as it stands and plan away from it.
            spheres = self._robot_spheres(start_state)
            if spheres is not None:
                self.get_logger().warn('start state inside the octomap; clearing the map '
                                       'around the arm where it stands')
                self._scene_world(scene, keep_clear, with_stand=False, arm_spheres=spheres)
                self.motion_gen.reset(reset_seed=False)
                self.clear_self_collision_buffers()
                if len(plan_req.goal_constraints[0].joint_constraints) > 0:
                    motion_gen_result = self.motion_gen.plan_single_js(start_state, goal_state,
                                                                       plan_config)
                else:
                    motion_gen_result = self.motion_gen.plan_single(start_state, goal_pose,
                                                                    plan_config)

        failed = (not motion_gen_result.success.item()) and motion_gen_result.valid_query
        self.__failures_in_a_row = (getattr(self, '_CumotionActionServer__failures_in_a_row', 0) + 1
                                    if failed else 0)
        if self.__failures_in_a_row >= 3 and not self._still_healthy(start_state):
            self.get_logger().fatal(
                'cuMotion can no longer plan even a 0.05 rad move: its solver state '
                'is broken. Exiting so launch respawns a fresh planner.')
            threading.Timer(0.5, lambda: os._exit(3)).start()
        elif self.__failures_in_a_row >= 3:
            self.__failures_in_a_row = 0
        with self.lock:
            self.planner_busy = False
        result = MoveGroup.Result()
        if motion_gen_result.success.item():
            result.error_code.val = MoveItErrorCodes.SUCCESS
            result.trajectory_start = plan_req.start_state
            traj = self.get_joint_trajectory(
                motion_gen_result.optimized_plan, motion_gen_result.optimized_dt.item(),
                time_scale=time_dilation_factor
            )
            # Never allow invalid GPU output to reach MoveIt or a hardware
            # controller.  NaN/Inf joint values can make FCL crash while it
            # updates collision-object transforms.
            points = traj.joint_trajectory.points
            arrays = [
                value
                for point in points
                for values in (point.positions, point.velocities,
                               point.accelerations)
                for value in values
            ]
            if not arrays or not np.isfinite(np.asarray(arrays, dtype=np.float64)).all():
                self.get_logger().error(
                    'Rejected cuMotion trajectory: it contains non-finite values.'
                )
                result.error_code.val = MoveItErrorCodes.PLANNING_FAILED
                result.planned_trajectory = RobotTrajectory()
                return result
            result.planning_time = motion_gen_result.total_time
            result.planned_trajectory = traj
        elif not motion_gen_result.valid_query:
            self.get_logger().error(
                f'Invalid planning query: {motion_gen_result.status}'
            )
            if motion_gen_result.status == MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS:
                result.error_code.val = MoveItErrorCodes.START_STATE_INVALID
            if motion_gen_result.status in [
                    MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,
                    MotionGenStatus.INVALID_START_STATE_SELF_COLLISION,
            ]:

                result.error_code.val = MoveItErrorCodes.START_STATE_IN_COLLISION
        else:
            self.get_logger().error(
                f'Motion planning failed wih status: {motion_gen_result.status}'
            )
            if motion_gen_result.status == MotionGenStatus.IK_FAIL:
                result.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
            else:
                # TRAJOPT_FAIL, GRAPH_FAIL, ...: left unset (0) this reached
                # MoveIt as a generic FAILURE (-1), which callers cannot tell
                # from a real fault. It is a plan that was not found.
                result.error_code.val = MoveItErrorCodes.PLANNING_FAILED

        self.get_logger().info(
            'returned planning result (query, success, failure_status): '
            + str(self.__query_count)
            + ' '
            + str(motion_gen_result.success.item())
            + ' '
            + str(motion_gen_result.status)
        )
        self.__query_count += 1
        return result

    def publish_voxels(self, voxels):
        vox_size = self.__publish_voxel_size

        # create marker:
        marker = Marker()
        marker.header.frame_id = self.__robot_base_frame
        marker.id = 0
        marker.type = 6  # cube list
        marker.ns = 'curobo_world'
        marker.action = 0
        marker.pose.orientation.w = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()
        marker.frame_locked = False
        marker.scale.x = vox_size
        marker.scale.y = vox_size
        marker.scale.z = vox_size
        marker.points = []

        # get only voxels that are inside surfaces:
        voxels = voxels[voxels[:, 3] > 0.0]
        vox = voxels.view(-1, 4).cpu().numpy()
        number_of_voxels_to_publish = len(vox)
        if len(vox) > self.__max_publish_voxels:
            self.get_logger().warn(
                f'Number of voxels to publish bigger than max_publish_voxels, '
                f'{len(vox)} > {self.__max_publish_voxels}'
            )
            number_of_voxels_to_publish = self.__max_publish_voxels
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        vox = vox.astype(np.float64)
        for i in range(number_of_voxels_to_publish):
            # Publish the markers at the center of the voxels:
            pt = Point()
            pt.x = vox[i, 0]
            pt.y = vox[i, 1]
            pt.z = vox[i, 2]
            marker.points.append(pt)

        # publish voxels:
        marker.header.stamp = self.get_clock().now().to_msg()

        self.__voxel_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    cumotion_action_server = CumotionActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(cumotion_action_server)
    try:
        executor.spin()
    except KeyboardInterrupt:
        cumotion_action_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    cumotion_action_server.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
