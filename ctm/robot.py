"""The ROS side of click_to_move: camera/joint inputs, tf, MoveIt services
and the motion primitives (plans, straight lines, guarded contact,
retrace) that the touch is built from."""

import collections
import contextlib
import math
import os
import re
import threading
import time
import warnings

import numpy as np
import yaml
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AllowedCollisionEntry, AllowedCollisionMatrix,
                             AttachedCollisionObject, CollisionObject,
                             Constraints,
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
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Empty, Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

import robot_camera_calibration as rcc
from ctm import walls
from ctm.common import (BASE_FRAME, COLOR_TOPIC, DEPTH_TOPIC, FINGERTIP_BEYOND_TCP,
                        HOME_JOINTS, INFO_TOPIC,
                        MoveGroupDown, OCTOMAP_NAME, RETRYABLE, SCREW_FRAME, WS, arm_joints,
                        decode_image, gripper_links, matrix_from_quat, pixel_ray,
                        quat_from_matrix, rot_about, tool_frame_along)


JUMP = 0.15                     # rad: the most any joint may move between 5 mm line steps
# The calibration pieces folded round each gripper's rail stick out past the
# gripper the robot model knows -- ~4 cm beyond the rail's ends and 4 cm
# towards the fingers (2026-10-15). Only calibrate.py has them on: a box per
# hand for MoveIt, in the hand frame (y along the rail, z towards the
# fingers), and the same as 14 spheres on cuMotion's openarm_<arm>_pieces
# link (reserved, off, in openarm.yml).
PIECES_BOX_CENTRE = (0.0, 0.0, 0.027)
PIECES_BOX_SIZE = (0.06, 0.20, 0.06)        # 22 cm end to end across the rail
PIECES_SPHERES = [(0.0, float(y), z, 0.025) for y in np.linspace(-0.085, 0.085, 6)
                  for z in (0.012, 0.042)]  # the outermost reach +-11 cm
PIECES_SLOTS = 14                           # spheres reserved per pieces link in openarm.yml (unused: off)

# The tool past the fingertips (tools.yaml: the left arm's vacuum cup), in the
# planners as a cylinder 2.5 cm across from the fingertips to its end -- before
# 2026-10-16 the planners knew only the hand and fingers, so the last 2 cm
# were never checked. MoveIt: a cylinder attached to the hand; cuMotion:
# spheres on the reserved openarm_<arm>_tool link (openarm.yml), off with no tool.
HAND_TCP_Z = 0.08                           # hand -> hand_tcp along z (openarm.urdf)
TOOL_DIAMETER = 0.025
TOOL_SLOTS = 4                              # spheres reserved per tool link in openarm.yml


def tool_spheres(length):
    """cuMotion's spheres for a tool `length` m past the fingertips, hand
    frame: radius half the diameter, from the fingertips to the end, none
    past the end (it must still reach the surface)."""
    r = TOOL_DIAMETER / 2
    start = HAND_TCP_Z + FINGERTIP_BEYOND_TCP
    last = start + max(length - r, 0.0)
    n = int(min(TOOL_SLOTS, max(1, np.ceil((last - start) / 0.01) + 1)))
    return [(0.0, 0.0, float(z), r) for z in np.linspace(start, last, n)]


def tool_id(arm):
    return f'tool_{arm}'


class ClickToMove(Node):
    def __init__(self, args):
        super().__init__('click_to_move')
        self.args = args
        self.pieces_on = False          # calibrate.py: the calibration pieces in the planners
        # STOP (web GUI, pick stack): cancels the goals running now and refuses
        # new ones until resume() -- the arms hold where they are
        self.halt = threading.Event()
        self.active_goals = set()
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
        got = None
        for _ in range(3):                           # a big string; it was slow once (sim)
            got = self.wait(self.rsp_get.call_async(
                GetParameters.Request(names=['robot_description'])), 10.0)
            if got is not None and got.values and got.values[0].string_value:
                break
        if got is None or not got.values or not got.values[0].string_value:
            raise RuntimeError('could not read robot_description')
        surface = rcc.RobotSurface(got.values[0].string_value)
        with self.lock:
            self._surface = surface
        return surface

    def joint_limits(self, arm):
        """[(lower, upper)] for the arm's 7 joints, from the URDF."""
        if not hasattr(self, '_limits'):
            self._limits = {}
            try:
                with open(os.path.join(WS, 'openarm.urdf')) as handle:
                    urdf = handle.read()
                for a in ('left', 'right'):
                    lim = []
                    for name in arm_joints(a):
                        m = re.search(r'<joint name="%s".*?lower="([-\d.e]+)" upper="([-\d.e]+)"'
                                      % name, urdf, re.S)
                        lim.append((float(m.group(1)), float(m.group(2))))
                    self._limits[a] = lim
            except (OSError, AttributeError):
                pass
        return self._limits.get(arm)

    def limit_room(self, arm, joints):
        """Smallest distance of any joint from its limit, rad (inf if unknown)."""
        lim = self.joint_limits(arm)
        if lim is None:
            return float('inf')
        return min(min(q - lo, hi - q) for q, (lo, hi) in zip(joints, lim))

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
        return m[:3, 3] + m[:3, 2] * self.args.tip_offsets[arm]

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

    def candidates(self, point, cam_origin, top_down=False, arm=None):
        """[(label, approach axis, quat, ring)] to try, in rings of preference.

        The axis is the tool's +Z, the direction the tip travels in. It is
        horizontal -- the end effector parallel to the ground (2026-10-08) --
        unless no level approach reaches: then tilted down 10, then 20 deg
        (rings 3 and 4; a box at the left arm's full stretch, 2026-10-14,
        where the sag correction straightened the elbow onto its limit just
        short of the face). And from the side the camera sees:
        the camera's line of sight to the point, flattened, then turned 15 and
        30 deg either way. Any roll about the axis is allowed. The far side and
        top of an object are not in the octomap, so it is never approached
        from there. preplan() stops at the first ring that yields a plan.
        """
        if top_down:
            # TOP DOWN: the tool straight down onto the point (an object lying
            # flat), turned about the vertical in 45 deg steps. Where straight
            # down cannot make the 10 cm line (this arm's wrist runs out of
            # range ~44 cm out: 2026-10-08), the tool may lean 10, then 20 deg,
            # wrist back towards the robot -- still coming down onto the top.
            out = []
            shoulder = self.lookup(BASE_FRAME, f'openarm_{arm}_link1') if arm else None
            outward = np.array(point, dtype=float) - (shoulder[:3, 3] if shoulder is not None
                                                      else np.zeros(3))
            outward[2] = 0.0
            outward /= max(np.linalg.norm(outward), 1e-9)
            # A suction cup seals only (nearly) flat on the top: with a tool on
            # (the vacuum), at most --vacuum-lean -- a 20 deg lean touched the
            # top 4 mm from the click and picked nothing; straight down is
            # beyond the wrist (joint 6, +-45 deg) past ~40 cm, where the
            # camera cannot see the table anyway (2026-10-14).
            # (A bellows cup seals tilted: no cap by default -- the planner
            # takes the smallest lean that reaches, 2026-10-14.)
            tool_on = bool(getattr(self.args, 'tools', {}).get(arm, {}).get('on'))
            limit = self.args.vacuum_lean if tool_on else 30
            leans = tuple((k, lean) for k, lean in enumerate((0, 10, 15, 20, 25, 30))
                          if lean <= limit)
            for ring, lean in leans:
                axis = np.array([0.0, 0.0, -1.0]) * math.cos(math.radians(lean)) \
                    + outward * math.sin(math.radians(lean))
                for deg in (0, 90, -90, 180, 45, -45, 135, -135):
                    label = (f'top-down yaw {deg:+d}' if lean == 0 else
                             f'top-down lean {lean} yaw {deg:+d}')
                    out.append((label, axis,
                                quat_from_matrix(tool_frame_along(axis, math.radians(deg))),
                                ring))
            return out
        ray = np.array(point, dtype=float) - cam_origin
        ray[2] = 0.0                                 # horizontal
        ray /= max(np.linalg.norm(ray), 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        out = []
        for tilt, base in ((0, 0), (10, 3), (20, 4)):
            for ring, yaw in ((0, 0), (1, 15), (1, -15), (2, 30), (2, -30)):
                axis = rot_about(up, math.radians(yaw)) @ ray
                axis[2] = 0.0
                axis /= np.linalg.norm(axis)
                axis = axis * math.cos(math.radians(tilt)) - up * math.sin(math.radians(tilt))
                for deg in (0, 90, -90, 180):
                    head = 'level' if tilt == 0 else f'tilted {tilt}'
                    label = (f'{head} roll {deg:+d}' if yaw == 0 else
                             f'{head} turn {yaw:+d} roll {deg:+d}')
                    out.append((label, axis,
                                quat_from_matrix(tool_frame_along(axis, math.radians(deg))),
                                ring if tilt == 0 else base))
        return out

    def tcp_for_tip(self, tip, axis, arm):
        """hand_tcp position that puts this arm's working tip at `tip`."""
        return np.asarray(tip) - axis * self.args.tip_offsets[arm]

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

    def allow_grippers_in_octomap(self, touching=None):
        """What may touch <octomap>: the links that never move (stand, camera,
        arm bases) always; and the two fingers of the arm `touching`, only
        while it makes contact (fingers_touching()). Nothing else -- the
        hand and fingers used to be exempt all the time, so the checks let
        the gripper pass through OTHER objects on the way to the one clicked
        (2026-10-09).

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
        fingers = ([f'openarm_{touching}_left_finger', f'openarm_{touching}_right_finger',
                    tool_id(touching)] if touching else [])
        # the tools too: checked like the gripper, free only at the contact
        for name in (tool_id('left'), tool_id('right')):
            if name not in index:
                names.append(name)
                for row in rows:
                    row.append(False)
                rows.append([False] * len(names))
                index[name] = len(names) - 1
        for link in gripper_links('left') + gripper_links('right') + static + \
                [tool_id('left'), tool_id('right')]:
            if link in index:
                allowed = link in static or link in fingers
                rows[index[link]][octomap] = rows[octomap][index[link]] = allowed
        updated = AllowedCollisionMatrix(entry_names=names)
        updated.entry_values = [AllowedCollisionEntry(enabled=row) for row in rows]
        scene = PlanningScene(is_diff=True, allowed_collision_matrix=updated)
        applied = self.wait(self.scene_apply.call_async(
            ApplyPlanningScene.Request(scene=scene)), 5.0)
        if applied is None:
            return False, 'move_group did not answer (busy?); gripper not exempted yet'
        if not applied.success:
            return False, 'move_group refused the gripper/octomap exemption'
        return True, ('stand, camera and arm bases may touch the octomap'
                      + (f'; the {touching} fingers too, for the contact' if touching else
                         '; the arms and grippers are all checked'))

    @contextlib.contextmanager
    def fingers_touching(self, arm):
        """While inside: this arm's two fingers may touch the octomap (the last
        few cm onto the clicked surface). Revoked on the way out, always."""
        ok, message = self.allow_grippers_in_octomap(touching=arm)
        if not ok:
            self.get_logger().warn(f'{arm} fingers: {message}')
        try:
            yield ok
        finally:
            self.allow_grippers_in_octomap(touching=None)

    def refresh_walls(self, optical):
        """After a capture: walls in the planning scene where the camera cannot
        see -- above and below its picture, from just in front of the nearest
        thing it saw (ctm/walls.py). Returns how many boxes, or None."""
        _c, _d, info = self.latest()
        t_wo = self.lookup(BASE_FRAME, optical)
        if info is None or t_wo is None or not self.scene_query.wait_for_service(timeout_sec=5.0):
            return None
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.OCTOMAP
        result = self.wait(self.scene_query.call_async(request), 5.0)
        nearest = None
        if result is not None:
            o = result.scene.world.octomap
            nearest = walls.nearest_seen(o.octomap, np.array(
                [o.origin.position.x, o.origin.position.y, o.origin.position.z]))
        start = (nearest - walls.AHEAD) if nearest is not None else walls.X_MIN
        obj = CollisionObject()
        obj.header.frame_id = BASE_FRAME
        obj.id = walls.WALL_ID
        obj.operation = CollisionObject.ADD
        for centre, size in walls.boxes(t_wo, info, start):
            box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in size])
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, centre)
            pose.orientation.w = 1.0
            obj.primitives.append(box)
            obj.primitive_poses.append(pose)
        if not obj.primitives:
            obj.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = [obj]
        applied = self.wait(self.scene_apply.call_async(
            ApplyPlanningScene.Request(scene=scene)), 5.0)
        if applied is None or not applied.success:
            self.get_logger().warn('could not put the unseen-space walls into the scene')
            return None
        return len(obj.primitives)

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
            if ok:
                self.set_pieces(self.pieces_on)
                self.set_tool()
            if ok or time.monotonic() > deadline:
                return ok, message
            time.sleep(2.0)

    def set_pieces(self, on):
        """The calibration pieces on both hands in the planners, or not:
        boxes attached to the hands for MoveIt (IK, OMPL, straight lines),
        spheres on cuMotion's pieces links. Called with the gripper exemption,
        before every touch -- a respawned planner has forgotten them; turning
        them off is done once (a calibration run may have left them on)."""
        if not on and getattr(self, '_pieces_off_done', False):
            return True
        # each on its own: MoveIt's box even while cuMotion is still starting
        # (with "and" a not-yet-ready cuMotion skipped the box too, and a whole
        # CAL ARMS ran without the pieces -- they hit the wall, 2026-10-15)
        ok_moveit = self._pieces_moveit(on)
        ok_cumotion = self._pieces_cumotion(on)
        ok = ok_moveit and ok_cumotion
        if not on and ok:
            self._pieces_off_done = True
        return ok

    def set_tool(self):
        """Each arm's tool (args.tools) in both planners, as it is set now:
        after a tool change, and with the gripper exemption before every
        touch (a respawned planner has forgotten it)."""
        ok = True
        for arm in ('left', 'right'):
            t = getattr(self.args, 'tools', {}).get(arm, {})
            length = float(t.get('length', 0.0)) if t.get('on') else 0.0
            ok &= self._tool_moveit(arm, length)
            ok &= self._tool_cumotion(arm, length)
        return ok

    def _tool_moveit(self, arm, length):
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        att = AttachedCollisionObject()
        att.link_name = f'openarm_{arm}_hand'
        att.touch_links = gripper_links(arm) + [f'openarm_{arm}_hand_tcp']
        obj = att.object
        obj.header.frame_id = att.link_name
        obj.id = tool_id(arm)
        if length <= 0.0:
            obj.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(CollisionObject(
                id=obj.id, operation=CollisionObject.REMOVE))
        else:
            obj.operation = CollisionObject.ADD
            obj.primitives.append(SolidPrimitive(type=SolidPrimitive.CYLINDER,
                                                 dimensions=[float(length),
                                                             float(TOOL_DIAMETER / 2)]))
            pose = Pose()
            pose.position.z = float(HAND_TCP_Z + FINGERTIP_BEYOND_TCP + length / 2)
            pose.orientation.w = 1.0
            obj.primitive_poses.append(pose)
        scene.robot_state.attached_collision_objects.append(att)
        if not self.scene_apply.wait_for_service(timeout_sec=5.0):
            return False
        applied = self.wait(self.scene_apply.call_async(ApplyPlanningScene.Request(scene=scene)),
                            5.0)
        if applied is None or not applied.success:
            self.get_logger().warn(f'could not set the {arm} tool in the planning scene')
            return False
        return True

    def _tool_cumotion(self, arm, length):
        from isaac_ros_cumotion_interfaces.action import UpdateLinkSpheres
        if not hasattr(self, '_spheres_client'):
            self._spheres_client = ActionClient(self, UpdateLinkSpheres, 'planner_attach_object')
        if not self._spheres_client.wait_for_server(timeout_sec=60.0 if length > 0 else 3.0):
            self.get_logger().warn(f'cuMotion planner_attach_object unavailable: {arm} tool not set')
            return False
        spheres = tool_spheres(length) if length > 0 else []
        goal = UpdateLinkSpheres.Goal(
            attach_object=bool(spheres), object_link_name=f'openarm_{arm}_tool',
            flattened_sphere_arr=[float(v) for sph in
                                  (spheres + [(0.0, 0.0, 0.0, -10.0)]
                                   * (TOOL_SLOTS - len(spheres)))
                                  for v in sph] if spheres else [])
        handle = self.wait(self._spheres_client.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            self.get_logger().warn(f'cuMotion refused the {arm} tool spheres')
            return False
        self.wait(handle.get_result_async(), 5.0)
        return True

    def _pieces_cumotion(self, on):
        from isaac_ros_cumotion_interfaces.action import UpdateLinkSpheres
        if not hasattr(self, '_spheres_client'):
            self._spheres_client = ActionClient(self, UpdateLinkSpheres, 'planner_attach_object')
        if not self._spheres_client.wait_for_server(timeout_sec=60.0 if on else 3.0):
            self.get_logger().warn('cuMotion planner_attach_object unavailable: pieces not set')
            return False
        for arm in ('left', 'right'):
            goal = UpdateLinkSpheres.Goal(
                attach_object=bool(on), object_link_name=f'openarm_{arm}_pieces',
                flattened_sphere_arr=[float(v) for sph in
                                      (PIECES_SPHERES + [(0.0, 0.0, 0.0, -10.0)]
                                       * (PIECES_SLOTS - len(PIECES_SPHERES)))
                                      for v in sph] if on else [])
            handle = self.wait(self._spheres_client.send_goal_async(goal), 5.0)
            if handle is None or not handle.accepted:
                self.get_logger().warn(f'cuMotion refused the {arm} pieces spheres')
                return False
            self.wait(handle.get_result_async(), 5.0)
        return True

    def _pieces_moveit(self, on):
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        for arm in ('left', 'right'):
            link = f'openarm_{arm}_hand'
            att = AttachedCollisionObject()
            att.link_name = link
            # (the wrist too, as in openarm.yml for cuMotion: fixed to the
            # gripper, the pieces cannot reach links 5-7 -- without these
            # MoveIt refused the head-on wall-marker touches, 2026-10-16)
            att.touch_links = [link, f'openarm_{arm}_left_finger', f'openarm_{arm}_right_finger',
                               f'openarm_{arm}_link5', f'openarm_{arm}_link6',
                               f'openarm_{arm}_link7', f'openarm_{arm}_hand_tcp']
            obj = att.object
            obj.header.frame_id = link
            obj.id = f'calibration_pieces_{arm}'
            obj.operation = CollisionObject.ADD if on else CollisionObject.REMOVE
            if not on:
                scene.robot_state.attached_collision_objects.append(att)
                scene.world.collision_objects.append(CollisionObject(
                    id=obj.id, operation=CollisionObject.REMOVE))
                continue
            obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX,
                                                 dimensions=[float(v) for v in PIECES_BOX_SIZE]))
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, PIECES_BOX_CENTRE)
            pose.orientation.w = 1.0
            obj.primitive_poses.append(pose)
            scene.robot_state.attached_collision_objects.append(att)
        applied = self.wait(self.scene_apply.call_async(ApplyPlanningScene.Request(scene=scene)),
                            5.0)
        if applied is None or not applied.success:
            self.get_logger().warn('could not set the calibration pieces in the planning scene')
            return False
        return True

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
        ik.timeout.nanosec = int(0.05 * 1e9)       # solvable poses solve in a few ms; failures cost the whole timeout
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

    def seeds(self, arm, start, tcp=None, quat=None):
        """Starting postures for IK. For a tool pointing straight down, the
        three stored top-down postures (ctm/topdown_seeds.npz) nearest `tcp`
        at the same turn come first: from the usual seeds KDL never finds one."""
        out = [start, HOME_JOINTS] + ([self.retract[arm]] if arm in self.retract else [])
        if tcp is None or quat is None:
            return out
        r = matrix_from_quat(*quat)
        if r[2, 2] > -0.85:                          # not (near) top-down
            return out
        table = self._topdown_seeds()
        if table is None or f'{arm}_q' not in table:
            return out
        turn = math.atan2(r[1, 0], r[0, 0])
        same = np.abs(np.angle(np.exp(1j * (table[f'{arm}_turn'] - turn)))) < math.radians(25)
        if not same.any():
            return out
        idx = np.flatnonzero(same)
        near = idx[np.argsort(np.linalg.norm(table[f'{arm}_tcp'][idx] - tcp, axis=1))[:3]]
        seeds = [table[f'{arm}_q'][i] for i in near]
        lim = self.joint_limits(arm)
        if lim is not None:
            # cuRobo often parks a joint right on its limit; start KDL inside
            # them so it settles on a posture with room to move if there is one.
            lo, hi = np.array(lim).T
            seeds = [np.clip(q, lo + 0.15, hi - 0.15) for q in seeds]
        return [q.tolist() for q in seeds] + out

    def fast_ik(self, arm, tcp, quat, start=None, want=3):
        """Joint solutions (plain URDF, within limits, no collision check) for
        hand_tcp at (tcp, quat), from 48 spread-out starting postures at once
        (kinematics.ik_batch, ~20 ms) -- [] if the arm cannot get there at
        all. A failed collision-aware IK costs its whole timeout; asking this
        first took TOP DOWN planning from ~100 s to seconds (2026-10-14)."""
        from ctm.kinematics import Arm, ik_batch
        if not hasattr(self, '_kin'):
            self._kin = {}
        if arm not in self._kin:
            self._kin[arm] = Arm(arm)
        lim = self.joint_limits(arm)
        if lim is None:
            return []
        lo, hi = np.array(lim).T
        if not hasattr(self, '_ik_starts'):
            self._ik_starts = {}
        if arm not in self._ik_starts:
            rng = np.random.default_rng(7)           # the same starts every time
            self._ik_starts[arm] = rng.uniform(lo + 0.05, hi - 0.05, (48, 7))
        seeds = self._ik_starts[arm]
        if start is not None:
            seeds = np.vstack([np.clip(start, lo, hi), seeds])
        target = np.eye(4)
        target[:3, :3] = matrix_from_quat(*quat)
        target[:3, 3] = tcp
        sols = ik_batch(self._kin[arm], target, seeds, list(zip(lo, hi)))
        return [np.clip(q, lo + 0.01, hi - 0.01).tolist() for q in sols[:want]]

    def _topdown_seeds(self):
        if not hasattr(self, '_td'):
            self._td = None
            try:
                raw = dict(np.load(os.path.join(WS, 'ctm', 'topdown_seeds.npz')))
                for arm in ('left', 'right'):
                    if f'{arm}_yaw' in raw:          # the tool x axis's heading per entry
                        raw[f'{arm}_turn'] = np.array([
                            math.atan2(*tool_frame_along(np.array([0.0, 0.0, -1.0]),
                                                         math.radians(y))[[1, 0], 0])
                            for y in raw[f'{arm}_yaw']])
                self._td = raw
            except (OSError, ValueError) as exc:
                self.get_logger().warn(f'no top-down seeds ({exc}); run ctm/make_topdown_seeds.py')
        return self._td

    def cartesian_plan(self, arm, position, quat, start_joints=None, check=True, via=None):
        """Straight line of hand_tcp to `position`, planned, not moved.

        From where the arm is, or from `start_joints` (so a whole touch can
        be checked before anything moves). Returns (fraction, trajectory,
        final joints) -- trajectory None if it did not solve. via: [(position,
        quat)] waypoints before the end (a turn of the wrist in steps).
        """
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('/compute_cartesian_path unavailable')
            return 0.0, None, None
        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.group_name = f'{arm}_arm'
        req.link_name = f'openarm_{arm}_hand_tcp'
        req.max_step = 0.005
        # No joint may move more than JUMP between two 5 mm steps (a real step
        # needs < 0.05 rad). With jump detection off, a top-down line once
        # flipped IK branch mid-path: the arm would have swung 0.5 m off the
        # line and back (simulated, 2026-10-08).
        req.jump_threshold = 0.0
        req.revolute_jump_threshold = JUMP
        req.avoid_collisions = check
        if start_joints is None:
            req.start_state.is_diff = True
        else:
            req.start_state.is_diff = True
            req.start_state.joint_state.name = arm_joints(arm)
            req.start_state.joint_state.position = [float(v) for v in start_joints]
        req.waypoints = []
        for pos, q in list(via or []) + [(position, quat)]:
            target = PoseStamped().pose
            target.position.x, target.position.y, target.position.z = map(float, pos)
            (target.orientation.x, target.orientation.y,
             target.orientation.z, target.orientation.w) = map(float, q)
            req.waypoints.append(target)
        result = self.wait(self.cartesian_client.call_async(req), 10.0)
        if result is None:
            self.require_move_group()
            return 0.0, None, None
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return 0.0, None, None
        trajectory = result.solution
        points = trajectory.joint_trajectory.points
        names = list(trajectory.joint_trajectory.joint_names)
        fraction = float(result.fraction)
        # And check it here too: cut the path at any jump.
        n = len(points)
        for k in range(1, n):
            if np.max(np.abs(np.subtract(points[k].positions, points[k - 1].positions))) > JUMP:
                self.get_logger().warn(f'straight line to {np.round(position, 3)}: joint jump '
                                       f'at step {k}/{n}; cut there')
                del points[k:]
                fraction *= (k - 1) / max(1, n - 1)      # the share of the line before it
                break
        final = None
        if points:
            end = dict(zip(names, points[-1].positions))
            final = [end.get(n) for n in arm_joints(arm)]
            final = None if None in final else final
        return fraction, trajectory, final

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

    def stop_motion(self):
        """STOP: cancel every running goal; nothing moves until resume()."""
        self.halt.set()
        for handle in list(self.active_goals):
            try:
                handle.cancel_goal_async()
            except Exception:                        # noqa: BLE001 - finished meanwhile
                pass
        self.get_logger().error('STOP: motion cancelled; the arms hold where they are')

    def resume(self):
        if self.halt.is_set():
            self.halt.clear()
            self.get_logger().info('STOP cleared: moves allowed again')

    def execute(self, trajectory, label, timeout=60.0, watch=None):
        """Execute a planned trajectory through move_group. (ok, stopped)

        `watch()` is polled every 10 ms while it runs; when it returns True
        the goal is cancelled -- the controller then holds where the arm is --
        and (True, True) is returned. That is how a guarded move stops on
        contact.
        """
        if self.halt.is_set():
            self.get_logger().error(f'{label}: STOP is on - not moving')
            return False, False
        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('/execute_trajectory unavailable')
            return False, False
        handle = self.wait(self.execute_client.send_goal_async(
            ExecuteTrajectory.Goal(trajectory=trajectory)), 10.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: trajectory rejected')
            return False, False
        self.active_goals.add(handle)
        try:
            return self._execute_wait(handle, label, timeout, watch)
        finally:
            self.active_goals.discard(handle)

    def _execute_wait(self, handle, label, timeout, watch):
        future = handle.get_result_async()
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        deadline = time.monotonic() + timeout
        while not done.wait(0.01):
            if self.halt.is_set():
                self.wait(handle.cancel_goal_async(), 3.0)
                done.wait(2.0)
                self.get_logger().error(f'{label}: STOP')
                return False, False
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

    def line_move(self, arm, position, quat, label, speed=None, check=True, via=None,
                  length=None):
        """Straight line of hand_tcp from where it is to `position`.

        Returns the joints the line ends on (what the arm should settle to),
        or None. /compute_cartesian_path interpolates the line and solves IK
        every 5 mm, so the path is straight by construction; two joint goals
        would only fix the ends. It is collision-checked like the rest, with
        the grippers exempt from the octomap. Its timing is not speed scaled,
        so it is re-timed here to a constant speed (`--line-speed` unless
        given).
        """
        fraction, trajectory, final = self.cartesian_plan(arm, position, quat, check=check,
                                                          via=via)
        if trajectory is None or fraction < 0.98:
            self.get_logger().warn(f'{label}: only {100 * fraction:.0f}% of the straight line '
                                   f'solves')
            return None
        if len(trajectory.joint_trajectory.points) < 2:
            return self.arm_positions(arm)           # already there
        if length is None:
            m = self.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
            length = (float(np.linalg.norm(np.asarray(position) - m[:3, 3]))
                      if m is not None else self.args.standoff)
        wanted = self.retime(trajectory, length, speed or self.args.line_speed)
        ok, _ = self.execute(trajectory, label, timeout=wanted + 20.0)
        if not ok:
            return None
        return final if final is not None else self.arm_positions(arm)

    @staticmethod
    def turn_steps(tcp_a, quat_a, tcp_b, quat_b, step_deg=10.0):
        """[(tcp, quat)] from pose a to pose b, the tool turned in steps of at
        most step_deg (the end, b, not included): a turn as a straight-line
        path that never asks a joint to jump."""
        from scipy.spatial.transform import Rotation, Slerp
        rots = Rotation.from_quat([quat_a, quat_b])
        angle = math.degrees((rots[0].inv() * rots[1]).magnitude())
        n = max(1, int(math.ceil(angle / step_deg)))
        slerp = Slerp([0.0, 1.0], rots)
        out = []
        for k in range(1, n):
            f = k / n
            out.append((np.asarray(tcp_a) + (np.asarray(tcp_b) - np.asarray(tcp_a)) * f,
                        tuple(slerp([f]).as_quat()[0])))
        return out

    def guarded_line(self, arm, position, quat, label, needed=None):
        """Straight line towards `position`, slow, stopping on contact.

        Contact is a joint torque rising by more than --contact-torque (and
        6x its resting noise) against half a second before, for 30 ms, AND
        the tool stalled: under 0.5 mm of progress in 0.12 s while the
        command still moves on. Whatever happens, a joint torque rising by
        --max-touch-torque against half a second before (after the first
        0.8 s) stops it. Torque alone
        was not enough: in a TOP DOWN posture the left
        arm lags its command by ~60 mrad, its controller pushes harder to
        catch up, and that read as contact 1-2 cm above the object
        (2026-10-09) -- a lagging arm still moves, a blocked one does not.
        The move is then cancelled and the arm holds there. Returns (joints,
        contact) -- contact False if it reached `position` without touching
        anything, or if the torques were too noisy to watch.
        """
        args = self.args
        names = arm_joints(arm)
        # let the arm come to rest first: a lagging arm still settling made the
        # baseline "too noisy" and contact went unwatched (simulated, 2026-10-12)
        t_end, prev = time.monotonic() + 0.8, self.arm_positions(arm)
        while time.monotonic() < t_end:
            time.sleep(0.05)
            now = self.arm_positions(arm)
            if prev is not None and now is not None and \
                    np.max(np.abs(np.subtract(now, prev))) < 0.0005:
                break
            prev = now
        samples = []
        t_end = time.monotonic() + 0.25        # torque baseline (100 Hz: 25 samples)
        while time.monotonic() < t_end:
            e = self.arm_efforts(arm)
            if e is not None:
                samples.append(e)
            time.sleep(0.01)
        fraction, trajectory, final = self.cartesian_plan(arm, position, quat)
        m = self.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
        length = (float(np.linalg.norm(np.asarray(position) - m[:3, 3]))
                  if m is not None else 0.02)
        if trajectory is None or fraction < 0.98:
            # At the edge of reach the line past the surface may not solve to
            # its end; as far as it does is enough if that reaches `needed`
            # (the surface): it stops on contact anyway. (A box at 55 cm: the
            # last 12% of the line past it did not solve and the touch was
            # skipped, 2026-10-13.)
            if trajectory is None or needed is None or fraction * length < needed:
                self.get_logger().warn(f'{label}: only {100 * fraction:.0f}% of the line solves')
                return None, False
            self.get_logger().info(f'{label}: {100 * fraction:.0f}% of the line solves - '
                                   f'{1000 * fraction * length:.0f} mm, past the surface')
            length = fraction * length
        # A tool on (the rubber vacuum extension) squashes: the tip keeps
        # creeping on contact, so "stopped advancing" shows late and the push
        # builds. With a tool: slower, a lower limit, and contact on the
        # torque rise alone.
        soft = bool(getattr(args, 'tools', {}).get(arm, {}).get('on'))
        max_torque = min(args.max_touch_torque, args.soft_tool_torque) if soft \
            else args.max_touch_torque
        wanted = self.retime(trajectory, length,
                             min(args.contact_speed, 0.01) if soft else args.contact_speed)
        watch = None
        if args.contact_torque > 0 and len(samples) >= 10:
            base = np.array(samples)
            noise = base.std(0)
            # Torques are compared with their own recent past, not with the
            # level before the move: reaching out changes gravity's pull on the
            # shoulder and the lag the controller fights by over 2 Nm in free
            # air -- against the start level that stopped touches short of the
            # wall at "2 Nm" (2026-10-13). A contact is fast, those are slow.
            limit = np.minimum(np.maximum(args.contact_torque, 6.0 * min(noise.max(), 0.15)),
                               0.8 * max_torque)
            history = collections.deque([(time.monotonic(), e) for e in samples])
            p0 = m[:3, 3] if m is not None else None
            way = ((np.asarray(position) - p0) / max(length, 1e-6)) if p0 is not None else None
            track = collections.deque()                 # (time, progress along the line)
            t0 = time.monotonic()
            count = {'contact': 0, 'cap': 0}

            def stalled():
                if p0 is None:
                    return True                         # cannot tell: torque decides
                now = time.monotonic()
                tcp = self.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
                if tcp is None:
                    return True
                track.append((now, float((tcp[:3, 3] - p0) @ way)))
                # 0.12 s, not 0.2: every 0.1 s of waiting for the stall to show
                # is ~0.4 Nm more push on the surface (simulated, 2026-10-13)
                # 0.08 s: wall contacts show on the elbow (joint 4) at 0.33-0.46 Nm
                # by then (robot logs, 2026-10-15); each 0.04 s waited is ~0.6 mm
                # more push at contact speed
                while track and now - track[0][0] > 0.08:
                    track.popleft()
                if now - t0 < 0.3 or len(track) < 3:
                    return False                        # still getting going
                commanded = min(1.0, (now - t0) / wanted) * length
                moved = track[-1][1] - track[0][1]
                return moved < 0.0005 and commanded - track[-1][1] > 0.0015

            def past(seconds):
                """The torques `seconds` ago (the oldest kept if not that old)."""
                now = time.monotonic()
                for t, e in history:
                    if now - t <= seconds:
                        return np.array(e)
                return np.array(history[-1][1])

            def watch():
                e = self.arm_efforts(arm)
                is_stalled = stalled()
                if e is None:
                    return False
                now = time.monotonic()
                history.append((now, e))
                while history and now - history[0][0] > 1.2:
                    history.popleft()
                e = np.array(e)
                if now - t0 < 0.4:
                    return False            # starting: the arm takes up its lag (3 cm out)
                # the hard limit: --max-touch-torque on any joint, against 0.5 s ago --
                # not in the first 0.8 s: the arm taking up its lag as it starts
                # read as 1.1 Nm on the shoulder and stopped a TOP DOWN pick 2.4 cm
                # above the box (2026-10-14); at contact speed it is still >2 cm
                # out then
                big = np.abs(e - past(0.5)) if now - t0 >= 0.8 else np.zeros_like(e)
                count['cap'] = count['cap'] + 1 if np.any(big > max_torque) else 0
                if count['cap'] >= 2:
                    j = int(np.argmax(big))
                    self.get_logger().warn(
                        f'{label}: stopped at the torque limit ({names[j]} '
                        f'{big[j]:.2f} Nm, max {max_torque:.1f})')
                    return True
                # contact: blocked (no progress) and a torque rising against 0.3 s ago
                rise = np.abs(e - past(0.3))
                count['contact'] = count['contact'] + 1 if np.any(rise > limit) else 0
                if (count['contact'] >= 2 or noise.max() > 1.0) and (is_stalled or (
                        soft and count['contact'] >= 2 and now - t0 >= 0.8)):
                    j = int(np.argmax(rise / limit))
                    self.get_logger().info(f'{label}: contact ({names[j]} {rise[j]:+.2f} Nm '
                                           f'in 0.3 s, stopped advancing)')
                    return True
                return False
        ok, stopped = self.execute(trajectory, label, timeout=wanted + 20.0, watch=watch)
        if not ok:
            return None, False
        return (self.arm_positions(arm) if stopped else final), stopped

    def plan_joints(self, arm, positions, velocity=None, tries=2):
        """A plan to `positions`, not executed: (code, trajectory).

        cuMotion, retried once (its optimiser starts from random seeds). A
        big turn of the tool -- level at pre_pick to straight down for TOP
        DOWN -- still defeats it now and then; then MoveIt's sampling planner
        (OMPL, checked against the same octomap) finds the way."""
        for _ in range(tries):
            code, trajectory = self._plan_joints_once(arm, positions, velocity)
            if code != MoveItErrorCodes.PLANNING_FAILED:
                return code, trajectory
        code_o, trajectory_o = self._plan_joints_once(arm, positions, velocity, 'ompl')
        if code_o == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f'{arm}: cuMotion found no path; OMPL did')
            return code_o, trajectory_o
        return code, trajectory

    def _plan_joints_once(self, arm, positions, velocity=None, pipeline='cumotion'):
        goal = self._joint_goal(arm, positions, velocity, pipeline)
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

    def move_joints(self, arm, positions, label, velocity=None, via_home=True, attempts=None):
        """cuMotion joint goal, going via HOME_JOINTS if there is no direct path.

        cuRobo's optimiser only finds detours so long; from an arm hanging by
        the torso the direct path to pre_pick needs a longer one and fails
        every time, while via home both legs plan (measured 2026-09-30).
        """
        code = self._move_joints_once(arm, positions, label, velocity, attempts)
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

    def _move_joints_once(self, arm, positions, label, velocity=None, attempts=None):
        """Joint goal through move_group: cuMotion, and if it finds no path,
        MoveIt's sampling planner (OMPL) -- cuMotion's optimiser misses some
        longer repositioning moves (simulated CALIBRATE ARMS, 2026-10-12: a
        third of them). Returns the MoveIt code."""
        goal = self._joint_goal(arm, positions, velocity)
        goal.planning_options.plan_only = False
        code = self._send_until_planned(goal, label, attempts)
        # CONTROL_FAILED too: a cuMotion plan from a stale start state is
        # refused by the controller, every retry alike; OMPL starts from
        # move_group's own, current state (2026-10-13)
        if code in (MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.FAILURE,
                    MoveItErrorCodes.CONTROL_FAILED):
            goal = self._joint_goal(arm, positions, velocity, 'ompl')
            goal.planning_options.plan_only = False
            ompl = self._send_until_planned(goal, f'{label} (OMPL)')
            if ompl == MoveItErrorCodes.SUCCESS:
                self.get_logger().info(f'{label}: cuMotion found no path; OMPL did')
                return ompl
        return code

    def _joint_goal(self, arm, positions, velocity=None, pipeline='cumotion'):
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            raise MoveGroupDown('/move_action is not served')
        velocity = self.args.velocity if velocity is None else velocity
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = f'{arm}_arm'
        req.pipeline_id = pipeline
        req.num_planning_attempts = 1 if pipeline == 'cumotion' else 4
        req.allowed_planning_time = self.args.planning_time if pipeline == 'cumotion' else 3.0
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

    def _send_until_planned(self, goal, label, attempts=None):
        for attempt in range(1, (attempts or self.args.attempts) + 1):
            if self.halt.is_set():
                self.get_logger().error(f'{label}: STOP is on - not moving')
                return MoveItErrorCodes.FAILURE
            handle = self.wait(self.move_client.send_goal_async(goal), 15.0)
            if handle is None:
                self.require_move_group()
            if handle is None or not handle.accepted:
                self.get_logger().error(f'{label}: goal rejected by move_group')
                return MoveItErrorCodes.FAILURE
            self.active_goals.add(handle)
            try:
                result = self.wait_or_dead(handle.get_result_async(), self.args.motion_timeout)
            finally:
                self.active_goals.discard(handle)
            if self.halt.is_set():
                return MoveItErrorCodes.FAILURE
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
