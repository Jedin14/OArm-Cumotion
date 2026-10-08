"""The ROS side of click_to_move: camera/joint inputs, tf, MoveIt services
and the motion primitives (plans, straight lines, guarded contact,
retrace) that the touch is built from."""

import collections
import math
import os
import re
import threading
import time
import warnings

import numpy as np
import yaml
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AllowedCollisionEntry, AllowedCollisionMatrix, Constraints,
                             JointConstraint, MoveItErrorCodes, PlanningScene,
                             PlanningSceneComponents, RobotTrajectory)
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath, GetPlanningScene,
                             GetPositionFK, GetPositionIK)
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Empty, Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

import robot_camera_calibration as rcc
from ctm.common import (BASE_FRAME, COLOR_TOPIC, DEPTH_TOPIC, HOME_JOINTS, INFO_TOPIC,
                        MoveGroupDown, OCTOMAP_NAME, RETRYABLE, SCREW_FRAME, WS, arm_joints,
                        decode_image, gripper_links, matrix_from_quat, pixel_ray,
                        quat_from_matrix, rot_about, tool_frame_along)


class ClickToMove(Node):
    def __init__(self, args):
        super().__init__('click_to_move')
        self.args = args
        cb = ReentrantCallbackGroup()

        self.lock = threading.Lock()
        self.color = None
        self.color_count = 0
        self.depth = None
        self.depths = collections.deque(maxlen=15)
        self.info = None
        self.joints = {}
        self.efforts = {}

        self.create_subscription(Image, COLOR_TOPIC, self._on_color, 2, callback_group=cb)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 2, callback_group=cb)
        self.create_subscription(CameraInfo, INFO_TOPIC, self._on_info, 2, callback_group=cb)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10,
                                 callback_group=cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, '/click_to_move/marker', latched)
        self.target_pub = self.create_publisher(PoseStamped, '/click_to_move/target', 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.move_client = ActionClient(self, MoveGroup, '/move_action', callback_group=cb)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik', callback_group=cb)
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk', callback_group=cb)
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path',
                                                   callback_group=cb)
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory',
                                           callback_group=cb)
        self.gripper_clients = {
            arm: ActionClient(self, GripperCommand, f'/{arm}_gripper_controller/gripper_cmd',
                              callback_group=cb) for arm in ('left', 'right')}
        self.refresh_client = self.create_client(Trigger, '/octomap_gater/refresh',
                                                 callback_group=cb)
        self.clear_client = self.create_client(Empty, '/clear_octomap', callback_group=cb)
        self.scene_query = self.create_client(GetPlanningScene, '/get_planning_scene',
                                              callback_group=cb)
        self.scene_apply = self.create_client(ApplyPlanningScene, '/apply_planning_scene',
                                              callback_group=cb)
        self.rsp_get = self.create_client(
            GetParameters, '/robot_state_publisher/get_parameters', callback_group=cb)
        self.rsp_set = self.create_client(
            SetParameters, '/robot_state_publisher/set_parameters', callback_group=cb)
        self.retract = self._retract_configs()

    # -- inputs ---------------------------------------------------------------

    def _on_color(self, msg):
        with self.lock:
            self.color = msg
            self.color_count += 1

    def _on_depth(self, msg):
        with self.lock:
            self.depth = msg
            self.depths.append(msg)

    def _on_info(self, msg):
        with self.lock:
            self.info = msg

    def _on_joints(self, msg):
        with self.lock:
            self.joints.update(zip(msg.name, msg.position))
            if msg.effort:
                self.efforts.update(zip(msg.name, msg.effort))

    def latest(self):
        with self.lock:
            return self.color, self.depth, self.info

    def fresh_color(self, timeout=3.0):
        """The next colour frame to arrive after this call, or None."""
        with self.lock:
            seen = self.color_count
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.color_count > seen + 1:          # one whole frame later
                    return self.color
            time.sleep(0.02)
        return None

    def averaged_depth(self, count, timeout=3.0):
        """Per-pixel median of the next `count` aligned depth frames, metres.

        One D455 frame is noisy by several mm at these ranges and has
        speckle dropouts; the median over a few frames of a still scene is
        both steadier and more complete. Zeros (no reading) are ignored, so a
        pixel is valid if most frames saw it.
        """
        with self.lock:
            self.depths.clear()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if len(self.depths) >= count:
                    frames = list(self.depths)[:count]
                    break
            time.sleep(0.03)
        else:
            with self.lock:
                frames = list(self.depths)
        if not frames:
            return None, None
        stack = np.stack([decode_image(m) for m in frames]).astype(np.float32)
        valid = stack > 0
        stack[~valid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN pixels
            depth = np.nanmedian(stack, axis=0)
        depth[valid.sum(axis=0) < (len(frames) + 1) // 2] = 0.0
        return np.nan_to_num(depth, nan=0.0), frames[-1]

    def arm_efforts(self, arm):
        with self.lock:
            names = arm_joints(arm)
            if not all(n in self.efforts for n in names):
                return None
            return [self.efforts[n] for n in names]

    def arm_positions(self, arm):
        with self.lock:
            names = arm_joints(arm)
            if not all(n in self.joints for n in names):
                return None
            return [self.joints[n] for n in names]

    def _retract_configs(self):
        """cuRobo's retract_config from openarm.yml, per arm: a good IK seed."""
        try:
            with open(os.path.join(WS, 'openarm.yml')) as handle:
                cspace = yaml.safe_load(handle)['robot_cfg']['kinematics']['cspace']
            table = dict(zip(cspace['joint_names'], cspace['retract_config']))
            return {arm: [table[n] for n in arm_joints(arm)] for arm in ('left', 'right')}
        except (OSError, KeyError, TypeError, yaml.YAMLError):
            return {}

    # -- geometry -------------------------------------------------------------

    def lookup(self, target, source):
        """4x4 transform taking points in `source` to `target`, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        except Exception:                            # noqa: BLE001
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        m = np.eye(4)
        m[:3, :3] = matrix_from_quat(q.x, q.y, q.z, q.w)
        m[:3, 3] = (t.x, t.y, t.z)
        return m

    def robot_surface(self):
        """The arms' visual meshes as surface samples, for CALIBRATE and
        OVERLAY. Built once from the live robot_description (about a minute
        the very first time, then from native/cache)."""
        with self.lock:
            if getattr(self, '_surface', None) is not None:
                return self._surface
        if not self.rsp_get.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('robot_state_publisher is not answering')
        got = self.wait(self.rsp_get.call_async(
            GetParameters.Request(names=['robot_description'])), 5.0)
        if got is None or not got.values or not got.values[0].string_value:
            raise RuntimeError('could not read robot_description')
        surface = rcc.RobotSurface(got.values[0].string_value)
        with self.lock:
            self._surface = surface
        return surface

    def fk(self, arm, joints, links):
        """{link: 4x4 world<-link} for this arm at `joints`, from move_group."""
        if not self.fk_client.wait_for_service(timeout_sec=2.0):
            return None
        req = GetPositionFK.Request()
        req.header.frame_id = BASE_FRAME
        req.fk_link_names = list(links)
        req.robot_state.is_diff = True
        req.robot_state.joint_state.name = arm_joints(arm)
        req.robot_state.joint_state.position = [float(v) for v in joints]
        res = self.wait(self.fk_client.call_async(req), 3.0)
        if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        out = {}
        for name, ps in zip(res.fk_link_names, res.pose_stamped):
            q, t = ps.pose.orientation, ps.pose.position
            m = np.eye(4)
            m[:3, :3] = matrix_from_quat(q.x, q.y, q.z, q.w)
            m[:3, 3] = (t.x, t.y, t.z)
            out[name] = m
        return out

    def link_poses(self, arms=('left', 'right')):
        """{link: 4x4 world<-link} from tf -- i.e. from the joint encoders."""
        return {f'openarm_{arm}_{short}': self.lookup(BASE_FRAME, f'openarm_{arm}_{short}')
                for arm in arms for short in rcc.RobotSurface.LINKS}

    def fingertip(self, arm):
        """Where the closed fingertips are now, in world -- from the encoders."""
        m = self.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
        if m is None:
            return None
        return m[:3, 3] + m[:3, 2] * self.args.tip_offset

    def pixel_to_world(self, u, v, depth_img, info, frame_id):
        """(point, camera origin, depth) in world, or raises ValueError."""
        h, w = depth_img.shape
        r = self.args.patch
        patch = depth_img[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
        valid = patch[(patch > 0.1) & (patch < 4.0) & np.isfinite(patch)]
        if valid.size < max(3, patch.size // 5):
            raise ValueError('no valid depth at that pixel -- pick a textured, '
                             'nearer surface')
        z = float(np.median(valid))
        # The colour image is not rectified and the D455 colour lens has real
        # distortion (k1 about -0.06): undone before deprojecting with K.
        p_cam = pixel_ray(u, v, info) * z
        m = self.lookup(BASE_FRAME, frame_id)
        if m is None:
            raise ValueError(f'no tf {BASE_FRAME} -> {frame_id}')
        return m[:3, :3] @ p_cam + m[:3, 3], m[:3, 3], z

    def candidates(self, point, cam_origin):
        """[(label, approach axis, quat, ring)] to try, in rings of preference.

        The axis is the tool's +Z, the direction the tip travels in. It is
        always horizontal -- the end effector stays parallel to the ground,
        never tilted down (2026-10-08) -- and from the side the camera sees:
        the camera's line of sight to the point, flattened, then turned 15 and
        30 deg either way. Any roll about the axis is allowed. The far side and
        top of an object are not in the octomap, so it is never approached
        from there. preplan() stops at the first ring that yields a plan.
        """
        ray = np.array(point, dtype=float) - cam_origin
        ray[2] = 0.0                                 # horizontal
        ray /= max(np.linalg.norm(ray), 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        out = []
        for ring, yaw in ((0, 0), (1, 15), (1, -15), (2, 30), (2, -30)):
            axis = rot_about(up, math.radians(yaw)) @ ray
            axis[2] = 0.0
            axis /= np.linalg.norm(axis)
            for deg in (0, 90, -90, 180):
                label = (f'level roll {deg:+d}' if ring == 0 else
                         f'level turn {yaw:+d} roll {deg:+d}')
                out.append((label, axis,
                            quat_from_matrix(tool_frame_along(axis, math.radians(deg))), ring))
        return out

    def tcp_for_tip(self, tip, axis):
        """hand_tcp position that puts the fingertips at `tip`."""
        return np.asarray(tip) - axis * self.args.tip_offset

    # -- ROS calls from worker threads ----------------------------------------

    @staticmethod
    def wait(future, timeout):
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return None
        return future.result()

    def wait_or_dead(self, future, timeout):
        """wait(), but checking every 5 s that move_group is still there, so
        a crash mid-goal is reported in seconds rather than after `timeout`."""
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        deadline = time.monotonic() + timeout
        while not done.wait(5.0):
            if time.monotonic() > deadline:
                return None
            self.require_move_group()
        return future.result()

    def publish_marker(self, point):
        x, y, z = (float(v) for v in point)
        now = self.get_clock().now().to_msg()
        sphere = Marker()
        sphere.header.frame_id = BASE_FRAME
        sphere.header.stamp = now
        sphere.ns, sphere.id = 'click_to_move', 0
        sphere.type, sphere.action = Marker.SPHERE, Marker.ADD
        sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = x, y, z
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.025
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1.0, 0.1, 0.1, 1.0
        label = Marker()
        label.header = sphere.header
        label.ns, label.id = 'click_to_move', 1
        label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y, z + 0.05
        label.pose.orientation.w = 1.0
        label.scale.z = 0.03
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f'x={x:+.3f} y={y:+.3f} z={z:+.3f}'
        self.marker_pub.publish(MarkerArray(markers=[sphere, label]))

    def publish_target(self, position, quat):
        target = PoseStamped()
        target.header.frame_id = BASE_FRAME
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x, target.pose.position.y, target.pose.position.z = \
            map(float, position)
        (target.pose.orientation.x, target.pose.orientation.y,
         target.pose.orientation.z, target.pose.orientation.w) = map(float, quat)
        self.target_pub.publish(target)

    def rebuild_octomap(self):
        """Clear the map and let the gater pass fresh depth frames into it."""
        if self.clear_client.wait_for_service(timeout_sec=2.0):
            self.wait(self.clear_client.call_async(Empty.Request()), 5.0)
        if not self.refresh_client.wait_for_service(timeout_sec=2.0):
            return 'octomap cleared; no /octomap_gater (octomap:=live refills it by itself)'
        result = self.wait(self.refresh_client.call_async(Trigger.Request()), 5.0)
        # The gater passes its frames at camera rate; give move_group a moment
        # to integrate them before anything is checked against the map.
        time.sleep(self.args.octomap_settle)
        return f'octomap rebuilt ({result.message})' if result else 'octomap refresh timed out'

    def allow_grippers_in_octomap(self):
        """Hand + finger links of both arms may touch <octomap>; nothing else.

        The matrix is read, extended and sent back whole: a diff carrying an
        ACM replaces it wholesale, and a partial one would throw away every
        disable_collisions entry from the SRDF (see pick_place_orchestrator's
        current_acm).
        """
        if not (self.scene_query.wait_for_service(timeout_sec=5.0)
                and self.scene_apply.wait_for_service(timeout_sec=5.0)):
            return False, 'planning scene services unavailable; gripper not exempted'
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        result = self.wait(self.scene_query.call_async(request), 5.0)
        if result is None:
            return False, 'move_group did not answer (busy?); gripper not exempted yet'
        if not result.scene.allowed_collision_matrix.entry_names:
            return False, 'empty collision matrix; gripper not exempted'
        matrix = result.scene.allowed_collision_matrix
        names = list(matrix.entry_names)
        rows = [list(entry.enabled) for entry in matrix.entry_values]
        if OCTOMAP_NAME not in names:
            names.append(OCTOMAP_NAME)
            for row in rows:
                row.append(False)
            rows.append([False] * len(names))
        index = {name: i for i, name in enumerate(names)}
        octomap = index[OCTOMAP_NAME]
        # Plus the links that never move: the stand, the camera and the arm
        # bases. Octomap voxels touching them mean nothing -- the stand's
        # collision mesh reaches out to x = +95 mm at the base plate and the
        # brace, so table, floor or cable voxels there turned it red in RViz
        # and made every start state "in collision" (2026-10-06).
        static = ['openarm_body_link0', 'camera_link', 'openarm_left_link0',
                  'openarm_right_link0']
        for link in gripper_links('left') + gripper_links('right') + static:
            if link in index:
                rows[index[link]][octomap] = rows[octomap][index[link]] = True
        updated = AllowedCollisionMatrix(entry_names=names)
        updated.entry_values = [AllowedCollisionEntry(enabled=row) for row in rows]
        scene = PlanningScene(is_diff=True, allowed_collision_matrix=updated)
        applied = self.wait(self.scene_apply.call_async(
            ApplyPlanningScene.Request(scene=scene)), 5.0)
        if applied is None:
            return False, 'move_group did not answer (busy?); gripper not exempted yet'
        if not applied.success:
            return False, 'move_group refused the gripper/octomap exemption'
        return True, ('grippers, stand, camera and arm bases may touch the octomap; the '
                      'moving arm links are still checked')

    def alive(self, timeout=3.0):
        """Does move_group answer at all? Its names linger in DDS after it
        dies, so wait_for_service passing proves nothing; a reply does."""
        if not self.scene_query.wait_for_service(timeout_sec=1.0):
            return False
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.SCENE_SETTINGS
        return self.wait(self.scene_query.call_async(request), timeout) is not None

    def require_move_group(self):
        if not self.alive():
            raise MoveGroupDown('move_group is not answering')

    def ensure_grippers_in_octomap(self, patience=20.0):
        """allow_grippers_in_octomap, retried: move_group answers slowly while
        it is executing (the boot walk), and a respawned one has forgotten it."""
        deadline = time.monotonic() + patience
        while True:
            ok, message = self.allow_grippers_in_octomap()
            if ok or time.monotonic() > deadline:
                return ok, message
            time.sleep(2.0)

    def close_gripper(self, arm):
        """Fingers together, so the tips are one point. Result status ignored:
        allow_stalling is false, so a stalled close reports ABORTED."""
        client = self.gripper_clients[arm]
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(f'/{arm}_gripper_controller/gripper_cmd unavailable')
            return False
        goal = GripperCommand.Goal()
        goal.command.position = 0.0
        goal.command.max_effort = 20.0
        handle = self.wait(client.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            return False
        self.wait(handle.get_result_async(), 5.0)
        # And wait for the fingers to stop before anything is planned. Both
        # move_group crashes on 2026-09-30 (SIGSEGV) came 0.6 and 1.5 s after
        # a gripper command, with IK and a plan in flight while the fingers
        # were still moving; the gripper-free runs never crashed.
        name = f'openarm_{arm}_finger_joint1'
        last, still = None, 0
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and still < 10:
            with self.lock:
                here = self.joints.get(name)
            still = still + 1 if (here is not None and last is not None
                                  and abs(here - last) < 1e-4) else 0
            last = here
            time.sleep(0.05)
        return True

    def live_update_camera(self, xyz, rpy):
        """Give robot_state_publisher the new camera origin, so tf -- and with
        it every clicked point, the octomap and the VLM -- uses it at once.
        (move_group's own robot model keeps the old camera box until the next
        restart; it only matters for collisions with the camera itself.)"""
        if not (self.rsp_get.wait_for_service(timeout_sec=2.0)
                and self.rsp_set.wait_for_service(timeout_sec=2.0)):
            return False, 'robot_state_publisher parameters unavailable'
        got = self.wait(self.rsp_get.call_async(
            GetParameters.Request(names=['robot_description'])), 5.0)
        if got is None or not got.values or not got.values[0].string_value:
            return False, 'could not read robot_description'
        (x, y, z), (r, p, yw) = xyz, rpy
        origin = f'<origin xyz="{x:.5f} {y:.5f} {z:.5f}" rpy="{r:.5f} {p:.5f} {yw:.5f}"/>'
        urdf, n = re.subn(r'(<joint\s+name="camera_joint"[^>]*>\s*)<origin\b[^>]*/>',
                          lambda m: m.group(1) + origin, got.values[0].string_value, count=1)
        if n != 1:
            return False, 'camera_joint not found in robot_description'
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=urdf)
        done = self.wait(self.rsp_set.call_async(SetParameters.Request(
            parameters=[Parameter(name='robot_description', value=value)])), 5.0)
        if done is None or not done.results or not done.results[0].successful:
            return False, 'robot_state_publisher refused the new robot_description'
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            m = self.lookup(BASE_FRAME, SCREW_FRAME)
            if m is not None and np.linalg.norm(m[:3, 3] - np.asarray(xyz)) < 0.0005:
                return True, 'live'
            time.sleep(0.1)
        return False, 'robot_state_publisher took the URDF but tf did not change'

    def settle(self, arm, expected, timeout=None):
        """Wait for the arm to reach `expected` joints, or to stop; return the
        worst joint error, rad.

        The controller reports SUCCESS when the trajectory ends, not when the
        arm arrives, and the hardware integral takes seconds to pull out the
        last of the sag. Done when every joint is within 2 mrad, or when the
        arm has *stopped* -- no joint moving more than 0.5 mrad over 0.5 s --
        or at the timeout. The integral term has a ~5 s time constant: with
        a 3 s timeout the arm was often still creeping when measured, and
        correcting on top of the creep made the corrections diverge (12.9 ->
        7.3 -> 10.0 -> 15.7 mm, 2026-10-06).
        """
        timeout = self.args.settle_time if timeout is None else timeout
        deadline = time.monotonic() + timeout
        worst = None
        history = collections.deque()
        while True:
            now = time.monotonic()
            here = self.arm_positions(arm)
            if here is not None and expected is not None:
                worst = max(abs(a - b) for a, b in zip(here, expected))
                if worst < 0.002:
                    break
            if here is not None:
                history.append((now, here))
                while len(history) > 1 and now - history[1][0] >= 0.5:
                    history.popleft()
                if now - history[0][0] >= 0.5 and max(
                        abs(a - b) for a, b in zip(here, history[0][1])) < 0.0005:
                    break
            if now > deadline:
                break
            time.sleep(0.05)
        time.sleep(0.15)                 # robot_state_publisher's tf catches up
        return worst

    def solve_ik(self, arm, position, quat, seed):
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = f'{arm}_arm'
        ik.ik_link_name = f'openarm_{arm}_hand_tcp'
        ik.avoid_collisions = True
        ik.timeout.nanosec = int(0.2 * 1e9)
        ik.robot_state.joint_state.name = arm_joints(arm)
        ik.robot_state.joint_state.position = [float(v) for v in seed]
        pose = PoseStamped()
        pose.header.frame_id = BASE_FRAME
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, position)
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = map(float, quat)
        ik.pose_stamped = pose
        result = self.wait(self.ik_client.call_async(req), 3.0)
        if result is None:
            self.require_move_group()
            return None
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        found = dict(zip(result.solution.joint_state.name,
                         result.solution.joint_state.position))
        if not all(n in found for n in arm_joints(arm)):
            return None
        return [found[n] for n in arm_joints(arm)]

    def seeds(self, arm, start):
        return [start, HOME_JOINTS] + ([self.retract[arm]] if arm in self.retract else [])

    def cartesian_plan(self, arm, position, quat, start_joints=None, check=True):
        """Straight line of hand_tcp to `position`, planned, not moved.

        From where the arm is, or from `start_joints` (so a whole touch can
        be checked before anything moves). Returns (fraction, trajectory,
        final joints) -- trajectory None if it did not solve.
        """
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('/compute_cartesian_path unavailable')
            return 0.0, None, None
        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.group_name = f'{arm}_arm'
        req.link_name = f'openarm_{arm}_hand_tcp'
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = check
        if start_joints is None:
            req.start_state.is_diff = True
        else:
            req.start_state.is_diff = True
            req.start_state.joint_state.name = arm_joints(arm)
            req.start_state.joint_state.position = [float(v) for v in start_joints]
        target = PoseStamped().pose
        target.position.x, target.position.y, target.position.z = map(float, position)
        (target.orientation.x, target.orientation.y,
         target.orientation.z, target.orientation.w) = map(float, quat)
        req.waypoints = [target]
        result = self.wait(self.cartesian_client.call_async(req), 10.0)
        if result is None:
            self.require_move_group()
            return 0.0, None, None
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return 0.0, None, None
        trajectory = result.solution
        points = trajectory.joint_trajectory.points
        names = list(trajectory.joint_trajectory.joint_names)
        final = None
        if points:
            end = dict(zip(names, points[-1].positions))
            final = [end.get(n) for n in arm_joints(arm)]
            final = None if None in final else final
        return float(result.fraction), trajectory, final

    @staticmethod
    def retime(trajectory, length, speed):
        """Scale a trajectory in time so the tool moves `length` m at `speed` m/s."""
        points = trajectory.joint_trajectory.points
        seconds = lambda p: p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
        duration = max(seconds(points[-1]), 1e-3) if points else 1e-3
        wanted = max(0.3, length / max(1e-3, speed))
        scale = duration / wanted
        for point in points:
            t = seconds(point) / scale
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1.0) * 1e9)
            point.velocities = [v * scale for v in point.velocities]
            point.accelerations = [a * scale * scale for a in point.accelerations]
        return wanted

    def retrace(self, arm, trajectory, label):
        """Fly a planned trajectory backwards from where the arm is now: a
        short lead-in onto its last point, then the same path in reverse.
        The way back is then the way in -- already checked, and nothing to
        plan (cuMotion's optimiser can fail the reverse of a move it just
        planned, near the edge of reach: 3/3 in simulation, 2026-10-07).
        False if the arm is not near the path's end, or it did not run."""
        jt = trajectory.joint_trajectory
        if not jt.points:
            return False
        names = list(jt.joint_names)
        with self.lock:
            now = [self.joints.get(n) for n in names]
        if any(v is None for v in now):
            return False
        seconds = lambda p: p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
        end = np.array(jt.points[-1].positions)
        gap = float(np.max(np.abs(np.array(now) - end)))
        if gap > 0.4:                # TEACH nudges can move it ~0.3 rad off the path
            return False
        lead = max(0.3, gap / 0.3) if gap > 0.002 else 0.0     # <= 0.3 rad/s onto the path
        total = seconds(jt.points[-1])
        out = JointTrajectory()
        out.joint_names = names
        if lead:
            out.points.append(JointTrajectoryPoint(positions=[float(v) for v in now],
                                                   velocities=[0.0] * len(names)))
        for k, p in enumerate(reversed(jt.points)):
            t = lead + total - seconds(p)
            q = JointTrajectoryPoint()
            q.positions = [float(v) for v in (now if (k == 0 and not lead) else p.positions)]
            q.velocities = [-float(v) for v in p.velocities] if p.velocities else []
            q.accelerations = [float(v) for v in p.accelerations] if p.accelerations else []
            q.time_from_start.sec = int(t)
            q.time_from_start.nanosec = int((t % 1.0) * 1e9)
            out.points.append(q)
        robot = RobotTrajectory()
        robot.joint_trajectory = out
        ok, _ = self.execute(robot, label, timeout=total + lead + 20.0)
        return ok

    def execute(self, trajectory, label, timeout=60.0, watch=None):
        """Execute a planned trajectory through move_group. (ok, stopped)

        `watch()` is polled every 10 ms while it runs; when it returns True
        the goal is cancelled -- the controller then holds where the arm is --
        and (True, True) is returned. That is how a guarded move stops on
        contact.
        """
        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('/execute_trajectory unavailable')
            return False, False
        handle = self.wait(self.execute_client.send_goal_async(
            ExecuteTrajectory.Goal(trajectory=trajectory)), 10.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: trajectory rejected')
            return False, False
        future = handle.get_result_async()
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        deadline = time.monotonic() + timeout
        while not done.wait(0.01):
            if watch is not None and watch():
                self.wait(handle.cancel_goal_async(), 3.0)
                done.wait(2.0)
                return True, True
            if time.monotonic() > deadline:
                handle.cancel_goal_async()
                self.require_move_group()
                self.get_logger().error(f'{label}: timed out; cancelled')
                return False, False
        code = future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'{label}: execution failed, code {code}')
            return False, False
        return True, False

    def line_move(self, arm, position, quat, label, speed=None, check=True):
        """Straight line of hand_tcp from where it is to `position`.

        Returns the joints the line ends on (what the arm should settle to),
        or None. /compute_cartesian_path interpolates the line and solves IK
        every 5 mm, so the path is straight by construction; two joint goals
        would only fix the ends. It is collision-checked like the rest, with
        the grippers exempt from the octomap. Its timing is not speed scaled,
        so it is re-timed here to a constant speed (`--line-speed` unless
        given).
        """
        fraction, trajectory, final = self.cartesian_plan(arm, position, quat, check=check)
        if trajectory is None or fraction < 0.98:
            self.get_logger().warn(f'{label}: only {100 * fraction:.0f}% of the straight line '
                                   f'solves')
            return None
        if len(trajectory.joint_trajectory.points) < 2:
            return self.arm_positions(arm)           # already there
        m = self.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
        length = (float(np.linalg.norm(np.asarray(position) - m[:3, 3]))
                  if m is not None else self.args.standoff)
        wanted = self.retime(trajectory, length, speed or self.args.line_speed)
        ok, _ = self.execute(trajectory, label, timeout=wanted + 20.0)
        if not ok:
            return None
        return final if final is not None else self.arm_positions(arm)

    def guarded_line(self, arm, position, quat, label):
        """Straight line towards `position`, slow, stopping on contact.

        The joint torques of this arm are watched against their level just
        before the move; when any joint departs from it by more than
        --contact-torque (and by more than 6x its resting noise) for 30 ms,
        the move is cancelled and the arm holds there. Returns (joints,
        contact) -- contact False if it reached `position` without touching
        anything, or if the torques were too noisy to watch.
        """
        args = self.args
        names = arm_joints(arm)
        samples = []
        t_end = time.monotonic() + 0.25        # torque baseline (100 Hz: 25 samples)
        while time.monotonic() < t_end:
            e = self.arm_efforts(arm)
            if e is not None:
                samples.append(e)
            time.sleep(0.01)
        fraction, trajectory, final = self.cartesian_plan(arm, position, quat)
        if trajectory is None or fraction < 0.98:
            self.get_logger().warn(f'{label}: only {100 * fraction:.0f}% of the line solves')
            return None, False
        m = self.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
        length = (float(np.linalg.norm(np.asarray(position) - m[:3, 3]))
                  if m is not None else 0.02)
        wanted = self.retime(trajectory, length, args.contact_speed)
        watch = None
        if args.contact_torque > 0 and len(samples) >= 10:
            base = np.array(samples)
            mean, noise = base.mean(0), base.std(0)
            if noise.max() > 0.15:
                self.get_logger().warn(
                    f'{label}: joint torques too noisy to feel contact (worst '
                    f'{noise.max():.2f} Nm on {names[int(noise.argmax())]}); not watching')
            else:
                limit = np.maximum(args.contact_torque, 6.0 * noise)
                over = {'n': 0}

                def watch():
                    e = self.arm_efforts(arm)
                    if e is None:
                        return False
                    hit = np.any(np.abs(np.array(e) - mean) > limit)
                    over['n'] = over['n'] + 1 if hit else 0
                    if over['n'] >= 3:
                        j = int(np.argmax(np.abs(np.array(e) - mean) / limit))
                        self.get_logger().info(
                            f'{label}: contact ({names[j]} {e[j] - mean[j]:+.2f} Nm)')
                        return True
                    return False
        ok, stopped = self.execute(trajectory, label, timeout=wanted + 20.0, watch=watch)
        if not ok:
            return None, False
        return (self.arm_positions(arm) if stopped else final), stopped

    def plan_joints(self, arm, positions, velocity=None):
        """cuMotion plan to `positions`, not executed: (code, trajectory)."""
        goal = self._joint_goal(arm, positions, velocity)
        goal.planning_options.plan_only = True
        handle = self.wait(self.move_client.send_goal_async(goal), 15.0)
        if handle is None:
            self.require_move_group()
        if handle is None or not handle.accepted:
            return MoveItErrorCodes.FAILURE, None
        result = self.wait_or_dead(handle.get_result_async(), self.args.motion_timeout)
        if result is None:
            handle.cancel_goal_async()
            return MoveItErrorCodes.TIMED_OUT, None
        code = result.result.error_code.val
        return code, (result.result.planned_trajectory if code == MoveItErrorCodes.SUCCESS
                      else None)

    def move_joints(self, arm, positions, label, velocity=None, via_home=True):
        """cuMotion joint goal, going via HOME_JOINTS if there is no direct path.

        cuRobo's optimiser only finds detours so long; from an arm hanging by
        the torso the direct path to pre_pick needs a longer one and fails
        every time, while via home both legs plan (measured 2026-09-30).
        """
        code = self._move_joints_once(arm, positions, label, velocity)
        if code == MoveItErrorCodes.SUCCESS:
            return True
        if not via_home:
            return False
        if code not in (MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.FAILURE) or \
                max(abs(a - b) for a, b in zip(positions, HOME_JOINTS)) < 0.02:
            return False
        self.get_logger().info(f'{label}: no direct path; going via home')
        if self._move_joints_once(arm, HOME_JOINTS, f'{label} via home',
                                  velocity) != MoveItErrorCodes.SUCCESS:
            return False
        return self._move_joints_once(arm, positions, label, velocity) == \
            MoveItErrorCodes.SUCCESS

    def _move_joints_once(self, arm, positions, label, velocity=None):
        """cuMotion joint goal through move_group. Returns the MoveIt code."""
        goal = self._joint_goal(arm, positions, velocity)
        goal.planning_options.plan_only = False
        return self._send_until_planned(goal, label)

    def _joint_goal(self, arm, positions, velocity=None):
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            raise MoveGroupDown('/move_action is not served')
        velocity = self.args.velocity if velocity is None else velocity
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = f'{arm}_arm'
        req.pipeline_id = 'cumotion'
        req.num_planning_attempts = 1
        req.allowed_planning_time = self.args.planning_time
        req.max_velocity_scaling_factor = velocity
        req.max_acceleration_scaling_factor = velocity
        req.start_state.is_diff = True
        req.workspace_parameters.header.frame_id = BASE_FRAME
        for corner, sign in ((req.workspace_parameters.min_corner, -1.0),
                             (req.workspace_parameters.max_corner, 1.0)):
            corner.x = corner.y = corner.z = sign * 1.5
        constraints = Constraints()
        for name, value in zip(arm_joints(arm), positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints = [constraints]
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def _send_until_planned(self, goal, label):
        for attempt in range(1, self.args.attempts + 1):
            handle = self.wait(self.move_client.send_goal_async(goal), 15.0)
            if handle is None:
                self.require_move_group()
            if handle is None or not handle.accepted:
                self.get_logger().error(f'{label}: goal rejected by move_group')
                return MoveItErrorCodes.FAILURE
            result = self.wait_or_dead(handle.get_result_async(), self.args.motion_timeout)
            if result is None:
                handle.cancel_goal_async()
                self.get_logger().error(f'{label}: timed out; goal cancelled')
                return MoveItErrorCodes.TIMED_OUT
            code = result.result.error_code.val
            if code == MoveItErrorCodes.SUCCESS:
                return code
            self.get_logger().warn(f'{label}: MoveIt error {code} (attempt {attempt})')
            if code not in RETRYABLE:
                return code
            if code == MoveItErrorCodes.CONTROL_FAILED:
                time.sleep(0.6)              # let the arm finish moving; replan from there
        return code
