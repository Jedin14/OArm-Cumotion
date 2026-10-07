#!/usr/bin/env python3
"""Take a frame, click a point on it, and an arm's fingertips touch it. cuMotion only.

No VLM, no grasp model. Bring the robot up exactly as for pick and place --
MoveIt, cuMotion, the D455 and the octomap gater all come from
launch_everything.launch.py -- then run this on top:

    native/run_launch_everything.sh              # terminal 1, unchanged
    native/run_click_to_move.sh                  # terminal 2

The window shows a captured frame, not live video.

    left click      select that pixel: marker + "x y z" in RViz and in the window
    MOVE / m        the arm on that side touches the selected point
    RECAPTURE / r   take a new frame *and* rebuild the octomap from it
    CALIBRATE / c   measure where the camera really is with the arm, and fix it
    OVERLAY / o     draw the robot model on the image: green = depth agrees
    APPLY           (only with --no-auto-apply) write the calibration
    q               quit

The x y z shown is the clicked surface in `world`, metres: origin at the base
of the stand, +x forward, +y the robot's left, +z up. The shoulders are at
(0.0126, -0.0448/+0.0572, 0.7344) since V7 (cam_org.txt).

Why a touch lands where it does -- the three errors, and what handles each
--------------------------------------------------------------------------

1. What touches. hand_tcp is 80 mm out from the hand, but the fingertips
   reach 95.4 mm (finger joint at z=0.015, finger mesh 80.4 mm beyond it; see
   openarm_hand.xacro). Aiming hand_tcp at the point drove the fingertips
   15 mm into the surface, and a stiff arm pressed into a surface slides off
   the spot. So the goal is the fingertips, `--tip-offset` beyond hand_tcp
   along the tool axis. The gripper is not commanded by default, so the point
   is aimed midway between the fingertips; `--close-gripper` shuts them
   first so they meet at one point and actually touch it.

2. The arm not arriving. The joint trajectory controller has no goal
   tolerance, so MoveIt reports SUCCESS the moment a trajectory ends; the
   hardware's integral term (KI 20 against KP 100, a 5 s time constant) is
   still pulling the last of the gravity sag out. pick_place_orchestrator
   measured tools arriving 15-36 mm short with a perfect plan. So after the
   approach the arm is left to settle, the fingertip is *measured* (tf from
   /joint_states, i.e. the encoders), and any error is flown out with a small
   straight correction; the same offset is carried into the touch. The window
   reports the residual.

3. The camera being somewhere else. The mount in the URDF is a tape
   measurement (cam_org.txt), and a centimetre or a degree there moves every
   clicked point. Nothing downstream can see that error -- the arm goes
   exactly where the camera said -- so it has to be measured. CALIBRATE does
   that by itself (robot_camera_calibration.py): it moves each arm to six
   poses in view, and at each one takes the arm as the depth camera sees it
   (thousands of points: what is now nearer than a frame taken before) and
   the robot's own model of it (the URDF visual meshes, posed by the joint
   encoders through tf). The camera pose is the rigid transform that puts
   the one onto the other, by robust point-to-plane ICP over all poses,
   started from several guesses around the tape measurement. It is a 3D fit,
   so it cannot trade camera height against tilt, and it calibrates exactly
   what a click uses. A plausible result is applied at once: written as the
   next version into cam_org.txt, v10.urdf.xacro and VLM/vlm_detect.py,
   openarm_description rebuilt, robot_state_publisher given the new URDF
   live. The window reports how far off clicks were and the model/depth
   agreement before and after. OVERLAY draws the model on the image (green:
   depth agrees, red: it does not) for checking by eye whenever an arm is
   in view. --manual-calibration clicks fingertips and uses solvePnP.

A touch (MOVE)
--------------

1. The pixel is deprojected with the aligned depth -- a per-pixel median over
   several frames, then over a small patch, undistorted -- and taken into
   `world` through tf. It goes on /click_to_move/marker.
2. The arm is chosen by the half of the frame: left half -> left arm.
3. (Gripper closed, with --close-gripper.) cuMotion joint goal to 5 cm (`--standoff`) from the point
   along the approach axis; settle, measure, correct; straight line in along
   the tool axis to the point (/compute_cartesian_path, `--line-speed`); hold
   `--hold` s; straight line back out; cuMotion back to the starting posture.
   Tool orientations tried: aiming along the camera ray at four rolls, then
   top-down. /compute_ik solves them for that arm's group only.

Touching needs two exemptions, as in pick_place_orchestrator: the hand and
fingers of both arms may touch <octomap> (the clicked surface is in the map);
the forearm and upper arm are still checked. And joint goals rather than
cuMotion pose goals, which serve one tool_frame only and plan all 14 joints.
"""

import argparse
import collections
import datetime
import math
import os
import re
import subprocess
import threading
import time
import warnings

import cv2
import numpy as np
import rclpy
import robot_camera_calibration as rcc
import gripper_markers as gm
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
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
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Empty, Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'
BASE_FRAME = 'world'
TCP_IN_HAND = 0.08     # hand -> hand_tcp along the hand's Z (openarm_hand_arguments.xacro)
# Farthest hand_tcp gets from openarm_<arm>_link1 (fixed): 0.668 / 0.673 m over
# 300 random joint sets each, from move_group's FK (2026-10-07).
ARM_REACH = 0.675
SCREW_FRAME = 'camera_bottom_screw_frame'      # what the URDF origin positions
WINDOW = 'click to move'
WS = os.path.dirname(os.path.abspath(__file__))
URDF_XACRO = os.path.join(WS, 'src/openarm_description/urdf/robot/v10.urdf.xacro')
CAM_ORG = os.path.join(WS, 'cam_org.txt')
VLM_DETECT = os.path.join(WS, 'VLM/vlm_detect.py')
CALIBRATION_FILE = os.path.join(WS, 'camera_calibration.yaml')
# Taught touch corrections (TEACH): per arm, the nudge a touch needed at a point.
CORRECTIONS_FILE = os.path.join(WS, 'touch_corrections.yaml')
CORRECTION_RADIUS = 0.15        # samples further than this from a point are ignored, m
NUDGE = 0.002                   # one TEACH nudge, m
OCTOMAP_NAME = '<octomap>'

# Closed fingertips, beyond hand_tcp along its +Z. finger_joint origin z=0.015
# in the hand frame, finger mesh spans 0.6585..0.7534 at an offset of -0.673,
# so the tip is at 0.015 + 0.0804 = 0.0954 m; hand_tcp is at 0.080.
FINGERTIP_BEYOND_TCP = 0.0154

# Same fallback as pick_place_orchestrator's home_joint_positions: symmetric,
# so a valid folded-down posture for either arm. Only an IK seed here.
HOME_JOINTS = [0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0]
# cuMotion's optimiser is stochastic; the same goal resent usually succeeds.
RETRYABLE = {MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.TIMED_OUT}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def quat_from_matrix(m):
    """3x3 rotation -> (x, y, z, w)."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    return tuple(q)


def matrix_from_quat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy_from_matrix(r):
    """URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return (math.atan2(r[2, 1], r[2, 2]),
            math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])),
            math.atan2(r[1, 0], r[0, 0]))


def rot_about(axis, angle):
    """Rotation matrix of `angle` radians about unit `axis`."""
    x, y, z = axis
    c, s_, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[t * x * x + c, t * x * y - s_ * z, t * x * z + s_ * y],
                     [t * x * y + s_ * z, t * y * y + c, t * y * z - s_ * x],
                     [t * x * z - s_ * y, t * y * z + s_ * x, t * z * z + c]])


def tool_frame_along(z_axis, roll):
    """Rotation whose +Z is `z_axis`, turned `roll` radians about it."""
    z = z_axis / np.linalg.norm(z_axis)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(z @ ref) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(roll), math.sin(roll)
    return np.column_stack((c * x + s * y, -s * x + c * y, z))


def decode_image(msg):
    """sensor_msgs/Image -> numpy, without cv_bridge (see vlm_detector_node)."""
    if msg.encoding in ('rgb8', 'bgr8'):
        img = np.frombuffer(msg.data, np.uint8).reshape(
            msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img.copy()
    if msg.encoding in ('16UC1', 'mono16'):
        img = np.frombuffer(msg.data, np.uint16).reshape(
            msg.height, msg.step // 2)[:, :msg.width]
        return img.astype(np.float32) / 1000.0          # mm -> m
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, np.float32).reshape(
            msg.height, msg.step // 4)[:, :msg.width].copy()
    raise ValueError(f'unsupported image encoding {msg.encoding}')


def intrinsics(info):
    k = np.array(info.k, dtype=np.float64).reshape(3, 3)
    d = np.array(info.d, dtype=np.float64) if len(info.d) else np.zeros(5)
    return k, d


def pixel_ray(u, v, info):
    """Unit-depth ray (x, y, 1) in the optical frame, lens distortion undone."""
    k, d = intrinsics(info)
    if np.any(np.abs(d) > 1e-6):
        xn, yn = cv2.undistortPoints(np.array([[[float(u), float(v)]]]), k, d)[0, 0]
    else:
        xn, yn = (u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1]
    return np.array([xn, yn, 1.0])


def project(points_world, t_world_optical, info):
    """World points -> pixels through the camera pose `t_world_optical`."""
    k, d = intrinsics(info)
    t_ow = np.linalg.inv(t_world_optical)
    rvec, _ = cv2.Rodrigues(t_ow[:3, :3])
    px, _ = cv2.projectPoints(np.asarray(points_world, dtype=np.float64).reshape(-1, 1, 3),
                              rvec, t_ow[:3, 3], k, d)
    return px.reshape(-1, 2)


class MoveGroupDown(RuntimeError):
    """move_group stopped answering. It has died before (SIGSEGV, no message);
    demo.launch.py respawns it, so the fix is to stop, say so, and let the
    operator carry on once it is back -- not to wait out every timeout."""


def load_corrections():
    try:
        with open(CORRECTIONS_FILE) as handle:
            return (yaml.safe_load(handle) or {}).get('samples', []) or []
    except (OSError, yaml.YAMLError):
        return []


def save_correction(arm, point, correction):
    samples = load_corrections()
    samples.append(plain({'arm': arm, 'point': [round(float(v), 4) for v in point],
                          'correction': [round(float(v), 4) for v in correction],
                          'when': datetime.datetime.now().isoformat(timespec='seconds')}))
    text = yaml.safe_dump({'samples': samples}, sort_keys=False)
    with open(CORRECTIONS_FILE, 'w') as handle:
        handle.write('# Touch corrections taught with click_to_move.py TEACH: where the fingertip\n'
                     '# had to be aimed, relative to the clicked point, to land on it.\n' + text)
    return len(samples)


def learned_correction(arm, point):
    """(correction (3,), samples nearby, samples in all) for this arm at this point.

    Two layers, so a correction carries to where nothing was taught yet:

    * a smooth trend through all of the arm's taught touches, c(p) = b + A p,
      ridge-regularised so that with few or bunched-up samples it stays near
      their average instead of extrapolating wildly. The arm's error changes
      with reach -- touches taught at 0.35 m did not hold at 0.46 m
      (2026-10-07) -- and the trend is what follows that, once touches have
      been taught at more than one distance;
    * a Gaussian-weighted (sigma 6 cm) blend of what the trend still misses
      at the samples within CORRECTION_RADIUS, for the local detail.
    """
    samples = [s_ for s_ in load_corrections() if s_.get('arm') == arm]
    if not samples:
        return np.zeros(3), 0, 0
    pts = np.array([s_['point'] for s_ in samples], dtype=float)
    cor = np.array([s_['correction'] for s_ in samples], dtype=float)
    centre = pts.mean(0)
    x = np.hstack([np.ones((len(pts), 1)), pts - centre])        # [1, dp]
    # Ridge on the slope only: expect ~10 mm per 10 cm at most, so a slope
    # needs real evidence spread over space before it moves off zero.
    ridge = np.diag([1e-9, 1.0, 1.0, 1.0]) * (0.005 / 0.1) ** 2
    coef = np.linalg.solve(x.T @ x + ridge, x.T @ cor)          # (4, 3)

    def trend(p):
        return np.hstack([1.0, np.asarray(p) - centre]) @ coef

    residual = cor - x @ coef
    local, weight, nearby = np.zeros(3), 0.0, 0
    for p_, r_ in zip(pts, residual):
        dist = float(np.linalg.norm(p_ - point))
        if dist > CORRECTION_RADIUS:
            continue
        w = math.exp(-0.5 * (dist / 0.06) ** 2)
        local += w * r_
        weight += w
        nearby += 1
    correction = trend(point) + (local / weight * min(1.0, weight) if weight > 0 else 0.0)
    norm = float(np.linalg.norm(correction))
    if norm > 0.06:                                  # never more than 6 cm
        correction *= 0.06 / norm
    return correction, nearby, len(samples)


def click_shift(t_old, t_new, info, depth=0.35):
    """How far a click moves between two camera poses: mean and max over a
    grid of pixels at `depth`, mm. I.e. how far off clicks were."""
    shifts = []
    for u in np.linspace(40, 600, 8):
        for v in np.linspace(40, 440, 6):
            ray = pixel_ray(u, v, info) * depth
            a = t_old[:3, :3] @ ray + t_old[:3, 3]
            b = t_new[:3, :3] @ ray + t_new[:3, 3]
            shifts.append(np.linalg.norm(a - b))
    return 1000.0 * float(np.mean(shifts)), 1000.0 * float(np.max(shifts))


def plain(value):
    """numpy scalars/arrays inside dicts and lists -> plain Python, for YAML."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def arm_joints(arm):
    return [f'openarm_{arm}_joint{i}' for i in range(1, 8)]


def gripper_links(arm):
    return [f'openarm_{arm}_hand', f'openarm_{arm}_left_finger',
            f'openarm_{arm}_right_finger']


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

        The axis is the tool's +Z, the direction the fingertips travel in.
        Default ("front"): along the camera's line of sight to the point, then
        tilted 15 deg, then 30 deg -- always from the side the camera sees.
        Only that side is mapped: the octomap knows nothing about the top or
        back of an object, so an approach from above (which the old "any"
        fell back to, and which won whenever it cost the joints less) went
        through space nobody had checked. touch() stops at the first ring
        that yields a plan. --orientation down is the old top-down approach.
        """
        ray = point - cam_origin
        ray /= np.linalg.norm(ray)
        out = []
        if self.args.orientation in ('front', 'point', 'any'):
            up = np.array([0.0, 0.0, 1.0])
            side = np.cross(up, ray)
            side /= max(np.linalg.norm(side), 1e-9)
            tilts = [(0, 0.0, 0.0)] + [(1, yaw, pitch) for yaw, pitch in
                                       ((15, 0), (-15, 0), (0, 15), (0, -15))]
            tilts += [(2, yaw, 0.0) for yaw in (30, -30)]
            for ring, yaw, pitch in tilts:
                axis = (rot_about(up, math.radians(yaw)) @
                        rot_about(side, math.radians(pitch)) @ ray)
                axis /= np.linalg.norm(axis)
                if axis[2] < -math.sin(math.radians(25)):
                    continue                         # steeply down = from above
                for deg in (0, 90, -90, 180):
                    label = (f'front roll {deg:+d}' if ring == 0 else
                             f'front tilt {yaw:+.0f}/{pitch:+.0f} roll {deg:+d}')
                    out.append((label, axis,
                                quat_from_matrix(tool_frame_along(axis, math.radians(deg))),
                                ring))
        if self.args.orientation in ('down', 'any'):
            down = np.array([0.0, 0.0, -1.0])
            for deg in (0, 90):
                out.append((f'down yaw {deg}', down,
                            quat_from_matrix(tool_frame_along(down, math.radians(deg))), 9))
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

    def cartesian_plan(self, arm, position, quat, start_joints=None):
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
        req.avoid_collisions = True
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
        if gap > 0.08:
            return False
        lead = max(0.4, gap / 0.1) if gap > 0.002 else 0.0     # <= 0.1 rad/s onto the path
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

    def line_move(self, arm, position, quat, label, speed=None):
        """Straight line of hand_tcp from where it is to `position`.

        Returns the joints the line ends on (what the arm should settle to),
        or None. /compute_cartesian_path interpolates the line and solves IK
        every 5 mm, so the path is straight by construction; two joint goals
        would only fix the ends. It is collision-checked like the rest, with
        the grippers exempt from the octomap. Its timing is not speed scaled,
        so it is re-timed here to a constant speed (`--line-speed` unless
        given).
        """
        fraction, trajectory, final = self.cartesian_plan(arm, position, quat)
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
        t_end = time.monotonic() + 0.4
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
        return code


def roll_across_view(axis, ray):
    """Roll that spreads the fingers across the view, not one behind the other.

    The fingers open along the tool's +Y. Seen end-on, one hides the other and
    perspective pulls the visible one well off the midpoint (~18 px measured
    in simulation at 0.3 m); side by side, the pair's centre is the midpoint.
    """
    x0, y0, z = tool_frame_along(axis, 0.0).T
    want = np.cross(z, ray)
    if np.linalg.norm(want) < 1e-6:
        return 0.0
    want /= np.linalg.norm(want)
    return math.atan2(-(want @ x0), want @ y0)


def _pnp(obj, img, seed, info):
    """solvePnP seeded with the camera pose `seed`; returns t_world_optical or None."""
    k, d = intrinsics(info)
    t_ow = np.linalg.inv(seed)
    rvec, _ = cv2.Rodrigues(t_ow[:3, :3])
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, d, rvec.copy(),
                                  t_ow[:3, 3].reshape(3, 1).copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    t_ow = np.eye(4)
    t_ow[:3, :3], t_ow[:3, 3] = rot, tvec.ravel()
    return np.linalg.inv(t_ow)


def solve_camera(samples, t_world_optical, info, max_drops=2):
    """solvePnP over (point in world, pixel) pairs, robust to bad points.

    Seeded with the current camera pose, so it refines rather than searches.
    Bad points -- a misclick, or a detection that caught something other
    than the fingertips -- are removed first by RANSAC (consensus at 8 px,
    which copes with several at once) and then by leave-one-out: each point
    is predicted by a camera solved *without* it, which one bad point cannot
    drag towards itself the way it drags an all-points fit. At most
    `max_drops` go that second way, and never below 6 points.
    Returns (new t_world_optical, errors before (px), errors after (px), used).
    """
    k, d = intrinsics(info)
    obj = np.array([s[0] for s in samples], dtype=np.float64)
    img = np.array([s[1] for s in samples], dtype=np.float64)
    before = np.linalg.norm(project(obj, t_world_optical, info) - img, axis=1)
    used = np.ones(len(samples), bool)
    if len(samples) >= 8:
        t_ow = np.linalg.inv(t_world_optical)
        rvec, _ = cv2.Rodrigues(t_ow[:3, :3])
        ok, _r, _t, inliers = cv2.solvePnPRansac(
            obj, img, k, d, rvec.copy(), t_ow[:3, 3].reshape(3, 1).copy(),
            useExtrinsicGuess=True, iterationsCount=500, reprojectionError=8.0,
            confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok and inliers is not None and len(inliers) >= 6:
            used[:] = False
            used[inliers.ravel()] = True
    for _round in range(max_drops):
        if used.sum() <= 6:
            break
        held_out = np.zeros(len(samples))
        for i in np.flatnonzero(used):
            keep = used.copy()
            keep[i] = False
            fit = _pnp(obj[keep], img[keep], t_world_optical, info)
            held_out[i] = (np.inf if fit is None else
                           np.linalg.norm(project(obj[i:i + 1], fit, info)[0] - img[i]))
        worst = int(np.argmax(np.where(used, held_out, -1.0)))
        if held_out[worst] > max(10.0, 3.0 * np.median(held_out[used])):
            used[worst] = False
        else:
            break
    solved = _pnp(obj[used], img[used], t_world_optical, info)
    if solved is None:
        return None, before, None, used
    after = np.linalg.norm(project(obj, solved, info) - img, axis=1)
    return solved, before, after, used


class App:
    """The OpenCV window, and the worker threads that capture and move."""

    def __init__(self, node):
        self.node = node
        self.frame = None             # (bgr, depth, info, frame_id)
        self.pick = None              # (u, v, world point, arm, camera origin)
        self.check = None             # the reach check of the current pick (dict)
        self.unreachable = None       # (H, W) bool: surface beyond either arm's reach
        self.plan_lock = threading.Lock()
        self.buttons = {}             # name -> (x0, y0, x1, y1) in window pixels
        self.status = 'capturing...'
        self.detail = ''
        self.busy = False
        # Calibration: the view being clicked, the answer, and a result to apply.
        self.calib_view = None        # (bgr, predicted (u, v) or None, index, total)
        self.calib_answer = None
        self.calib_event = threading.Event()
        self.calibration = None
        self.calibrating = False
        self.overlay = None           # (pixels (N,2), agree (N,) bool): robot model on the image
        self.markers = gm.MarkerCalibration.load()   # None until CALIBRATE MARKERS has run
        self.teach = False            # TEACH mode: hold each touch for nudges
        self.teaching = False         # a touch is being taught right now
        self.abort = threading.Event()

    def say(self, text, error=False):
        self.status = text
        # Two call sites on purpose: rclpy raises "Logger severity cannot be
        # changed between calls" when one line logs at two severities, which
        # turned the first error message of every run into a crash.
        if error:
            self.node.get_logger().error(text)
        else:
            self.node.get_logger().info(text)

    def in_background(self, work):
        if self.busy:
            return
        self.busy = True

        def run():
            try:
                work()
            except MoveGroupDown:
                self.say('move_group stopped answering -- it crashed (grep "process has died" '
                         '~/.ros/log/latest/launch.log). It respawns in a few seconds; then '
                         'RECAPTURE (its octomap is gone) and try again.', True)
            except Exception as exc:                 # noqa: BLE001 - shown in the window
                self.say(f'error: {exc}', True)
            finally:
                self.busy = False
                self.calib_view = None
        threading.Thread(target=run, daemon=True).start()

    def capture(self):
        """Take a frame and rebuild the octomap from the same moment."""
        deadline = time.monotonic() + 15.0
        while True:
            color, depth, info = self.node.latest()
            if color is not None and depth is not None and info is not None:
                break
            if time.monotonic() > deadline:
                self.say('no camera frames -- is the D455 streaming?', True)
                return
            self.status = 'waiting for camera topics...'
            time.sleep(0.2)
        self.say(f'frame captured; averaging {self.node.args.depth_frames} depth frames...')
        averaged, last = self.node.averaged_depth(self.node.args.depth_frames)
        if averaged is None:
            self.say('no depth frames arrived', True)
            return
        color, _d, info = self.node.latest()
        self.frame = (decode_image(color), averaged, info,
                      last.header.frame_id or info.header.frame_id)
        self.pick = None
        self.check = None
        self.overlay = None
        try:
            self.unreachable = self.reach_map()
        except Exception as exc:                     # noqa: BLE001 - only a hint
            self.unreachable = None
            self.node.get_logger().warn(f'reach map: {exc}')
        self.say('frame captured; rebuilding the octomap...')
        self.say(f'{self.node.rebuild_octomap()}. Click a point, then MOVE.')

    # -- input ----------------------------------------------------------------

    def answer(self, what, where=None):
        if what == 'abort':
            self.abort.set()
        self.calib_answer = (what, where)
        self.calib_event.set()

    def on_mouse(self, event, u, v, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 <= u <= x1 and y0 <= v <= y1:
                self.press(name)
                return
        if self.calib_view is not None:
            if v < self.calib_view[0].shape[0]:
                self.answer('click', (u, v))
            return
        if self.busy or self.frame is None:
            return
        bgr, depth, info, frame_id = self.frame
        if v >= bgr.shape[0]:
            return                                   # the button bar, off a button
        dh, dw = depth.shape
        bh, bw = bgr.shape[:2]
        try:
            point, origin, z = self.node.pixel_to_world(
                int(u * dw / bw), int(v * dh / bh), depth, info, frame_id)
        except Exception as exc:                     # noqa: BLE001
            self.pick = None
            self.say(f'pixel ({u}, {v}): {exc}', True)
            return
        if self.node.args.arm != 'auto':
            arm = self.node.args.arm
        else:
            arm = 'left' if u < bw / 2.0 else 'right'
        self.pick = (u, v, point, arm, origin)
        self.node.publish_marker(point)
        where = f'x={point[0]:+.3f} y={point[1]:+.3f} z={point[2]:+.3f} (depth {z:.2f} m)'
        self.say(f'{where}, {arm} arm - checking it can be touched...')
        check = {'pick': self.pick, 'where': where, 'ok': None, 'plan': None}
        self.check = check
        threading.Thread(target=self.check_pick, args=(check,), daemon=True).start()

    def press(self, name):
        if self.teaching and name in ('left', 'right', 'up', 'down', 'in', 'out', 'save',
                                      'skip'):
            self.answer(name)
        elif name == 'teach':
            self.teach = not self.teach
            self.say('TEACH on: after each touch, nudge the fingertip onto the spot and SAVE'
                     if self.teach else 'TEACH off: touches use what was taught')
        elif name == 'skip':
            self.answer('skip')
        elif name == 'abort':
            self.answer('abort')
        elif name == 'move':
            self.move()
        elif name == 'recapture':
            self.in_background(self.capture)
        elif name == 'calibrate':
            self.in_background(self.calibrate)
        elif name == 'apply':
            self.in_background(self.apply_calibration)
        elif name == 'markers':
            self.in_background(self.calibrate_markers)
        elif name == 'overlay':
            if self.overlay is not None:
                self.overlay = None
            else:
                self.in_background(self.show_overlay)

    def show_overlay(self):
        """Draw the robot model, posed by the encoders, onto a fresh frame:
        green where the camera's depth agrees with it (within 15 mm), red
        where it does not. If the calibration is right the model sits on
        the arms in the image; this is the check, before and after."""
        node = self.node
        self.say('loading the robot model (slow only the first time)...')
        surface = node.robot_surface()
        depth, last = node.averaged_depth(6)
        color, _d, info = node.latest()
        if depth is None or color is None or info is None:
            self.say('no camera frames', True)
            return
        optical = last.header.frame_id or info.header.frame_id
        t_wo = node.lookup(BASE_FRAME, optical)
        if t_wo is None:
            self.say(f'no tf for {optical}', True)
            return
        self.frame = (decode_image(color), depth, info, optical)
        self.pick = None
        mp, mn = surface.posed(node.link_poses())
        px, agree, seen = rcc.depth_agreement(depth, mp, mn, t_wo, info)
        if not seen.any():
            self.say('neither arm is in view', True)
            return
        self.overlay = (px[seen][::2], agree[seen][::2])
        self.say(f'model vs depth: {100 * agree[seen].mean():.0f}% of the visible arm agrees '
                 f'within 15 mm (green). Red = camera and robot disagree. o = hide')

    def reach_map(self):
        """Pixels of the captured frame whose surface is beyond the reach of
        the arm that would go there (left half: left arm), as (H, W) bool.
        Only a hint drawn on the image; the check after a click is the real one."""
        node, args = self.node, self.node.args
        bgr, depth, info, optical = self.frame
        t_wo = node.lookup(BASE_FRAME, optical)
        shoulders = {a: node.lookup(BASE_FRAME, f'openarm_{a}_link1') for a in ('left', 'right')}
        if t_wo is None or any(m is None for m in shoulders.values()):
            return None
        h, w = depth.shape
        vs, us = np.mgrid[0:h:4, 0:w:4]
        sx, sy = info.width / float(w), info.height / float(h)
        px = np.column_stack([us.ravel() * sx, vs.ravel() * sy]).reshape(-1, 1, 2)
        k, d = rcc.intrinsics(info)
        xy = cv2.undistortPoints(px.astype(np.float64), k, d).reshape(-1, 2)
        z = depth[vs, us].ravel().astype(np.float64)
        world = np.column_stack([xy * z[:, None], z]) @ t_wo[:3, :3].T + t_wo[:3, 3]
        left = (us.ravel() * sx < info.width / 2.0) if args.arm == 'auto' else \
            np.full(len(z), args.arm == 'left')
        shoulder = np.where(left[:, None], shoulders['left'][:3, 3], shoulders['right'][:3, 3])
        beyond = np.linalg.norm(world - shoulder, axis=1) - args.tip_offset - ARM_REACH
        bad = ((z > 0.1) & (beyond > 0)).reshape(vs.shape).astype(np.uint8)
        return cv2.resize(bad, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    def aim_for(self, arm, point, p_cam):
        """(aim, learned, taught, markers_on): where to send the fingertip.
        The TEACH corrections always apply -- they are the best first guess.
        With calibrated flags the camera then measures the fingertip against
        the click itself and corrects whatever is left; if no flag is in view
        the taught correction still stands (it used to be dropped as soon as
        any flag was calibrated, which left the right arm with no correction
        at all when its one flag was hidden, 2026-10-07)."""
        learned, _used, taught = learned_correction(arm, point)
        markers_on = self.markers is not None and self.markers.has_arm(arm) and p_cam is not None
        return point + learned, learned, taught, markers_on

    def check_pick(self, check):
        """Right after a click: can its arm touch that point -- reach, a
        front approach, a collision-free path and the straight line in, all
        planned, nothing moved. The plan is kept for MOVE if nothing changes."""
        node = self.node
        _u, _v, point, arm, origin = check['pick']
        try:
            with self.plan_lock:
                if self.check is not check or self.busy:
                    return
                node.require_move_group()
                node.ensure_grippers_in_octomap()
                scene = self.frame
                t_wo = node.lookup(BASE_FRAME, scene[3])
                p_cam = (np.linalg.inv(t_wo)[:3, :3] @ point + np.linalg.inv(t_wo)[:3, 3]
                         if t_wo is not None else None)
                aim = self.aim_for(arm, point, p_cam)[0]
                start = node.arm_positions(arm)
                plan, why = self.preplan(arm, aim, point, origin, start)
        except Exception as exc:                     # noqa: BLE001 - only a preview
            plan, why, aim, start = None, f'could not check ({exc})', None, None
        check.update(ok=plan is not None, plan=plan, aim=aim, start=start,
                     time=time.monotonic())
        if self.check is not check or self.busy:
            return
        if plan is not None:
            self.say(f'{check["where"]}: the {arm} arm can touch it [{plan["label"]}]'
                     + (' but it is at the edge of reach' if plan['edge'] else '')
                     + ' - press MOVE')
        else:
            self.say(f'{check["where"]}: the {arm} arm cannot touch it -- {why}', True)

    def move(self):
        """The MOVE button: touch the selected point with its arm."""
        if self.pick is None:
            self.say('click a point first, then press MOVE', True)
            return
        _u, _v, point, arm, origin = self.pick
        scene = self.frame                           # the frame clicked on: no arm in it
        self.in_background(lambda: self.touch(arm, point, origin, scene))

    # -- the touch ------------------------------------------------------------

    def correct_to(self, arm, tcp_goal, axis, quat, label, expected, offset=None):
        """Settle on `expected` joints, measure the fingertip, and fly out the
        error. Returns the offset that had to be added to the command (world,
        m) and the error left over (m).

        Plain corrections (gain 1), measured only once the arm has really
        stopped (settle()). Scaling corrections up to make up for apparent
        undershoot was tried and diverged: the "undershoot" was the arm still
        moving. If a correction makes things worse, the best offset so far is
        restored and the loop stops.
        """
        node, args = self.node, self.node.args
        wanted = tcp_goal + axis * args.tip_offset
        offset = np.zeros(3) if offset is None else np.array(offset, dtype=float)
        best = None                                  # (residual, offset, joints)
        for attempt in range(args.corrections + 1):
            t0 = time.monotonic()
            joint_err = node.settle(arm, expected)
            waited = time.monotonic() - t0
            tip = node.fingertip(arm)
            if tip is None:
                break
            error = wanted - tip
            residual = float(np.linalg.norm(error))
            self.node.get_logger().info(
                f'{label}: settled in {waited:.1f} s, joints {1000 * (joint_err or 0):.1f} mrad '
                f'from command, fingertip {1000 * residual:.1f} mm from target')
            if best is not None and residual > best[0]:
                self.say(f'{arm} arm: correction made it worse ({1000 * best[0]:.1f} -> '
                         f'{1000 * residual:.1f} mm); going back')
                node.line_move(arm, tcp_goal + best[1], quat, f'{arm} back')
                node.settle(arm, best[2])
                return best[1], best[0]
            best = (residual, offset.copy(), expected)
            if residual <= args.tolerance or attempt == args.corrections:
                break
            if residual > 0.04:
                self.say(f'{arm} arm: {1000 * residual:.0f} mm off at {label} -- too far '
                         f'for a correction; not correcting', True)
                break
            offset += error
            self.say(f'{arm} arm: {1000 * residual:.1f} mm off at {label}; correcting')
            expected = node.line_move(arm, tcp_goal + offset, quat, f'{arm} correct')
            if expected is None:
                break
        return (best[1], best[0]) if best else (offset, None)

    def camera_check(self, arm, scene, label):
        """Where the camera sees the gripper, against where the model puts it.

        The arm is what is now nearer than `scene` (the frame that was clicked
        on, taken with the arm out of view); the hand, fingers and wrist links
        of the model, posed by the encoders, are fitted to it by a
        translation (robot_camera_calibration.fit_translation). Returns the
        offset d -- the real gripper is at model + d -- or None if the camera
        does not see enough of it.

        This is what makes a touch land where the click was: the target and
        the arm are measured by the same camera in the same frame, so the
        camera's calibration, tilt and depth bias, and arm sag the encoders
        cannot see, all cancel. Measured: the encoders put the fingertip on
        target while the real touch was ~30 mm low (2026-10-06).
        """
        node = self.node
        if not node.args.camera_check or scene is None:
            return None
        _bgr, background, info, optical = scene
        t_wo = node.lookup(BASE_FRAME, optical)
        depth, _last = node.averaged_depth(6)
        if t_wo is None or depth is None:
            return None
        near = ('link6', 'link7', 'hand', 'left_finger', 'right_finger')
        poses = {k: v for k, v in node.link_poses((arm,)).items()
                 if k.rsplit(f'{arm}_', 1)[-1] in near}
        mp, mn = node.robot_surface().posed(poses)
        obs, _mask = rcc.arm_points(depth, background, info)
        obs = rcc.near_model(obs, t_wo, mp, radius=0.06)
        if len(obs) < 150:
            self.node.get_logger().info(f'camera check at {label}: only {len(obs)} gripper '
                                        f'points visible; skipped')
            return None
        d, matched, rms = rcc.fit_translation(obs, t_wo, mp, mn)
        if d is None or rms > 5.0 or np.linalg.norm(d) > 0.06:
            self.node.get_logger().info(f'camera check at {label}: no reliable fit '
                                        f'({matched} points, rms {rms:.1f} mm); skipped')
            return None
        self.node.get_logger().info(
            f'camera check at {label}: gripper is ({1000 * d[0]:+.1f}, {1000 * d[1]:+.1f}, '
            f'{1000 * d[2]:+.1f}) mm from where the model puts it ({matched} points, rms '
            f'{rms:.1f} mm)')
        return d

    # -- gripper flags ----------------------------------------------------------

    def flag_patches(self, arm):
        """{id or name: 4x4 hand<-marker} -- calibrated flags, else the two
        nominal rail-end positions."""
        if self.markers is not None and self.markers.has_arm(arm):
            return {i: self.markers.hand_marker(i) for i in self.markers.ids(arm)}
        out = {}
        for name, y in (('end+', 0.109), ('end-', -0.109)):
            m = np.eye(4)
            m[:3, :3] = np.diag([1.0, -1.0, -1.0])           # face looking back (-Z)
            m[:3, 3] = (0.0, y, 0.004)
            out[name] = m
        return out

    def flag_visible(self, arm, joints):
        """How many of this arm's gripper flags will be in the camera's view
        with the arm at `joints`? A z-buffer of both arms' meshes (this one by
        FK at `joints`, the other where it is), against each flag's patch.
        (Unknown -> 1.)"""
        node = self.node
        if self.frame is None:
            return 1
        info, optical = self.frame[2], self.frame[3]
        t_wo = node.lookup(BASE_FRAME, optical)
        links = [f'openarm_{arm}_{s_}' for s_ in rcc.RobotSurface.LINKS]
        poses = node.fk(arm, joints, links)
        if t_wo is None or poses is None:
            return 1
        other = 'left' if arm == 'right' else 'right'
        poses.update(node.link_poses((other,)))
        pts, _n = node.robot_surface().posed(poses)
        t_ow = np.linalg.inv(t_wo)
        pc = pts @ t_ow[:3, :3].T + t_ow[:3, 3]
        pc = pc[(pc[:, 2] > 0.05) & (np.abs(pc[:, 0]) < pc[:, 2]) & (np.abs(pc[:, 1]) < pc[:, 2])]
        px, z = rcc.project(pc, info)
        u, v = px[:, 0].astype(int), px[:, 1].astype(int)
        ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
        zb = np.full((480, 640), np.inf, np.float32)
        np.minimum.at(zb, (v[ok], u[ok]), z[ok].astype(np.float32))
        zb = cv2.erode(zb, np.ones((5, 5), np.uint8))
        hand = poses[f'openarm_{arm}_hand']
        g = np.stack(np.meshgrid(np.linspace(-0.02, 0.02, 5), np.linspace(-0.02, 0.02, 5)),
                     -1).reshape(-1, 2)
        local = np.column_stack([g, np.zeros(len(g)), np.ones(len(g))])
        count = 0
        for mk in self.flag_patches(arm).values():
            w = (hand @ mk @ local.T).T[:, :3]
            cam = w @ t_ow[:3, :3].T + t_ow[:3, 3]
            normal = t_ow[:3, :3] @ (hand[:3, :3] @ mk[:3, 2])
            facing = -float(normal @ (cam.mean(0) / np.linalg.norm(cam.mean(0))))
            pp, zz = rcc.project(cam, info)
            uu, vv = pp[:, 0].astype(int), pp[:, 1].astype(int)
            inside = (uu >= 0) & (uu < 640) & (vv >= 0) & (vv < 480)
            seen = np.zeros(len(cam), bool)
            seen[inside] = zz[inside] <= zb[vv[inside], uu[inside]] + 0.004
            if facing > 0.4 and seen.mean() > 0.85:
                count += 1
        return count

    def marker_offset(self, arm, p_cam, aim, axis, tcp_goal, label):
        """Where the real fingertip is, from the gripper flags, against where
        it should be at this stage. Returns d (world, m): the real fingertip
        is d away from where it should be -- or None if no flag is seen.

        Everything is compared in the camera's own frame: the target (from
        the click) and the fingertip (from the flags) are measured by the
        same camera, so its calibration, and arm flex the encoders cannot
        see, cancel out.
        """
        node = self.node
        _c, _d, info = node.latest()
        optical = self.frame[3] if self.frame is not None else 'camera_color_optical_frame'
        t_wo = node.lookup(BASE_FRAME, optical)
        hand = node.lookup(BASE_FRAME, f'openarm_{arm}_hand')
        if info is None or t_wo is None or hand is None:
            return None
        frames = []
        for _ in range(node.args.marker_frames):
            msg = node.fresh_color(timeout=1.0)
            if msg is None:
                break
            frames.append(decode_image(msg))
        r_ow = t_wo[:3, :3].T
        ids = self.markers.ids(arm)
        expected = {i: r_ow @ hand[:3, :3] @ self.markers.hand_marker(i)[:3, 2] for i in ids}
        seen = gm.observe(frames, info, ids, expected)
        if not seen:
            node.get_logger().info(f'flags at {label}: none of {ids} in view')
            return None
        tip_in_hand = np.array([0.0, 0.0, TCP_IN_HAND + node.args.tip_offset])
        hand_cam, used = gm.hand_from_flags(
            seen, {i: self.markers.hand_marker(i) for i in ids}, r_ow @ hand[:3, :3])
        tip_cam = hand_cam[:3, :3] @ tip_in_hand + hand_cam[:3, 3]
        # each flag on its own, for the agreement check
        tips = [m_cam[:3, 3] + hand_cam[:3, :3] @ (tip_in_hand - self.markers.hand_marker(i)[:3, 3])
                for i, (m_cam, _n, _s) in seen.items()]
        should = tcp_goal + axis * node.args.tip_offset          # world, model
        desired_cam = p_cam + r_ow @ (should - aim)
        d = t_wo[:3, :3] @ (tip_cam - desired_cam)
        node.get_logger().info(
            f'flags at {label}: {sorted(seen)} seen; fingertip is ({1000 * d[0]:+.1f}, '
            f'{1000 * d[1]:+.1f}, {1000 * d[2]:+.1f}) mm from where it should be'
            + (f' (flags disagree by {1000 * np.ptp(np.array(tips), axis=0).max():.1f} mm)'
               if len(tips) > 1 else ''))
        if np.linalg.norm(d) > 0.025:
            node.get_logger().warn(f'flags at {label}: {1000 * np.linalg.norm(d):.0f} mm is '
                                   f'implausible; ignored')
            return None
        return d

    def marker_targets(self, t_wo):
        """Fingertip spots for CALIBRATE MARKERS: a grid over each half of
        the image, 0.40 / 0.46 m out -- far enough that the whole gripper is
        beyond the D455's minimum range -- skipping any spot with something
        less than 6 cm behind it in the captured depth (the gripper would be
        lost against it). Interleaved so each arm's spots spread out."""
        bgr, depth, info, _frame = self.frame
        h, w = depth.shape
        out = {'left': [], 'right': []}
        for k, fv in enumerate((0.30, 0.50, 0.70)):
            for j, fu in enumerate((0.12, 0.28, 0.42, 0.58, 0.72, 0.88)):
                for wanted in ((0.40, 0.46) if (j + k) % 2 else (0.46, 0.40)):
                    u, v = fu * bgr.shape[1], fv * bgr.shape[0]
                    du, dv = int(u * w / bgr.shape[1]), int(v * h / bgr.shape[0])
                    patch = depth[max(0, dv - 8):dv + 9, max(0, du - 8):du + 9]
                    seen = patch[patch > 0.1]
                    if seen.size and np.percentile(seen, 10) < wanted + 0.06:
                        continue
                    p = t_wo[:3, :3] @ (pixel_ray(u, v, info) * wanted) + t_wo[:3, 3]
                    out['left' if fu < 0.5 else 'right'].append(('left' if fu < 0.5 else 'right', p))
        return out['right'] + out['left']

    def calibrate_markers(self):
        """CALIBRATE MARKERS (FLAGS): measure where each flag sits on its gripper.

        At a few spots 0.40-0.46 m out, the hand turns about its own axis
        (fingertip held still, -60..+60 deg in 30 deg steps) and the camera
        follows the flags round their circles (gripper_markers.flags_from_roll).
        No depth fitting: a fit of the gripper's shape to depth was tried and
        placed the flags 1-7 mm wrong in simulation (2026-10-07); the circles
        place them to ~0.5 mm. About 20 s per spot.
        """
        node, args = self.node, self.node.args
        self.say('flag calibration: taking a clean frame first...')
        self.capture()
        if self.frame is None:
            return
        _bgr, _depth, info, optical = self.frame
        t_wo = node.lookup(BASE_FRAME, optical)
        if t_wo is None:
            self.say(f'no tf for {optical}', True)
            return
        t_ow = np.linalg.inv(t_wo)
        cam = t_wo[:3, 3]
        targets = self.marker_targets(t_wo)
        if not targets:
            self.say('flag calibration needs clear space 40-50 cm in front of the camera on '
                     'both sides -- move things back and try again', True)
            return
        arms = sorted({a for a, _p in targets})
        starts = {a: node.arm_positions(a) for a in arms}
        if any(v is None for v in starts.values()):
            self.say('no /joint_states yet', True)
            return
        node.require_move_group()
        node.ensure_grippers_in_octomap()
        samples, quality, per_arm = {}, [], {}
        current = None
        self.abort.clear()
        self.calibrating = True
        try:
            for k, (arm, tip) in enumerate(targets, start=1):
                if self.abort.is_set():
                    raise KeyboardInterrupt
                if per_arm.get(arm, 0) >= args.marker_poses_per_arm:
                    continue
                node.require_move_group()
                if current and current != arm:
                    node.move_joints(current, starts[current], f'{current} return')
                current = arm
                axis = (tip - cam) / np.linalg.norm(tip - cam)
                tcp = node.tcp_for_tip(tip, axis)
                # a base roll from which the whole sweep solves, both flags in view
                sweep, best = None, -1
                for roll0 in (0, 90, -90, 180):
                    seed, sols = node.arm_positions(arm), []
                    for phi in (0, -30, -60, 30, 60):
                        quat = quat_from_matrix(tool_frame_along(axis, math.radians(roll0 + phi)))
                        sol = node.solve_ik(arm, tcp, quat, seed if phi else node.arm_positions(arm))
                        if sol is None:
                            break
                        sols.append((phi, sol))
                        seed = sols[0][1]
                    if len(sols) < 5:
                        continue
                    n = self.flag_visible(arm, sols[0][1])
                    if n > best:
                        sweep, best = sorted(sols), n
                    if best >= 2:
                        break
                if sweep is None:
                    self.say(f'flags {k}: the {arm} arm cannot turn there; skipping')
                    continue
                rows = []
                for phi, joints in sweep:
                    if self.abort.is_set():
                        raise KeyboardInterrupt
                    self.say(f'flags {k}: {arm} hand turned {phi:+d} deg')
                    if node._move_joints_once(arm, joints, f'flags {k} {phi:+d}') != \
                            MoveItErrorCodes.SUCCESS:
                        continue
                    node.settle(arm, joints, timeout=3.0)
                    frames = []
                    for _ in range(args.marker_frames):
                        msg = node.fresh_color(timeout=1.0)
                        if msg is None:
                            break
                        frames.append(decode_image(msg))
                    hand = node.lookup(BASE_FRAME, f'openarm_{arm}_hand')
                    if hand is None or not frames:
                        continue
                    hand_cam = t_ow @ hand
                    expected = {i: hand_cam[:3, :3] @ gm.NOMINAL_NORMAL for i in gm.IDS[arm]}
                    rows.append((hand_cam, gm.observe(frames, info, gm.IDS[arm], expected)))
                    if frames:
                        self.calib_view = (frames[-1], None, None, None)
                found, rms = gm.flags_from_roll(rows)
                for i, mats in found.items():
                    samples.setdefault(i, []).extend(mats)
                if found:
                    per_arm[arm] = per_arm.get(arm, 0) + 1
                    quality += list(rms.values())
                self.say(f'flags {k}: ' + (', '.join(f'ID {i} circle fit {r:.1f} mm'
                                                     for i, r in sorted(rms.items()))
                                           or 'no flag followed round its circle'))
        except KeyboardInterrupt:
            self.say('flag calibration aborted')
            return
        finally:
            self.calib_view = None
            self.calibrating = False
            if current and node.alive():
                node.move_joints(current, starts[current], f'{current} return')

        result, report = {}, []
        for i, mats in samples.items():
            if len(mats) < 4:
                report.append(f'ID {i}: only {len(mats)} views, not saved')
                continue
            mats = np.array(mats)
            t = np.median(mats[:, :3, 3], axis=0)
            dt = np.linalg.norm(mats[:, :3, 3] - t, axis=1)
            keep = dt < max(0.003, 3.0 * np.median(dt))
            mats = mats[keep] if keep.sum() >= 4 else mats
            m = np.eye(4)
            m[:3, :3] = gm.average_rotation(mats[:, :3, :3])
            m[:3, 3] = np.mean(mats[:, :3, 3], axis=0)
            lateral = np.linalg.norm(mats[:, :2, 3] - m[:2, 3], axis=1)
            spread = float(np.median(lateral) * 1000)
            if spread > 3.0:
                report.append(f'ID {i}: inconsistent by {spread:.0f} mm, NOT saved (loose, '
                              f'or seen too rarely)')
                continue
            result[i] = {'arm': gm.ARM_OF.get(i, '?'), 'hand_marker': m.tolist(),
                         'views': int(len(mats)), 'spread_mm': round(spread, 2)}
            report.append(f'ID {i}: {len(mats)} views, consistent to {spread:.1f} mm')
        never = [i for a in arms for i in gm.IDS[a] if i not in samples]
        if never:
            report.append(f'never seen: {never}')
        if not result:
            self.detail = '; '.join(report)
            self.say('flag calibration failed: ' + '; '.join(report) + ' -- are the flags '
                     'facing back towards the wrist, flat, and clear of the motor?', True)
            return
        merged = dict(self.markers.data) if self.markers is not None else {}
        merged.update(result)
        cal = gm.MarkerCalibration(merged)
        cal.save(note=datetime.datetime.now().isoformat(timespec='seconds'))
        self.markers = cal
        self.detail = '; '.join(report)
        self.say(f'flags calibrated ({", ".join(str(i) for i in sorted(result))}): touches now '
                 f'correct themselves with them')

    def teach_touch(self, arm, point, learned, axis, quat, command, scene):
        """TEACH: hold the touch while the operator nudges the fingertip onto
        the spot, then save the total correction for this arm and point."""
        node = self.node
        info, optical = (scene[2], scene[3]) if scene else (None, None)
        t_wo = node.lookup(BASE_FRAME, optical) if optical else None
        right = t_wo[:3, 0] if t_wo is not None else np.array([0.0, -1.0, 0.0])
        right = right - right[2] * np.array([0.0, 0.0, 1.0])     # level: image right
        right /= max(np.linalg.norm(right), 1e-9)
        steps = {'left': -right, 'right': right, 'up': np.array([0.0, 0.0, 1.0]),
                 'down': np.array([0.0, 0.0, -1.0]), 'in': axis, 'out': -axis}
        nudge = np.zeros(3)
        self.teaching = True
        try:
            while True:
                self.say(f'TEACH: nudge onto the spot ({1000 * np.linalg.norm(nudge):.0f} mm '
                         f'so far), then SAVE')
                self.calib_event.clear()
                if not self.calib_event.wait(120.0):
                    self.say('TEACH: no answer for 2 minutes; not saved', True)
                    return
                what = self.calib_answer[0]
                if what in steps:
                    nudge = nudge + steps[what] * NUDGE
                    node.line_move(arm, command + nudge, quat, f'{arm} nudge')
                elif what == 'save':
                    total = learned + nudge
                    n = save_correction(arm, point, total)
                    self.say(f'TEACH: saved ({1000 * total[0]:+.0f}, {1000 * total[1]:+.0f}, '
                             f'{1000 * total[2]:+.0f}) mm for the {arm} arm here '
                             f'({n} corrections stored)')
                    time.sleep(0.8)
                    return
                elif what in ('skip', 'abort'):
                    self.say('TEACH: not saved')
                    return
        finally:
            self.teaching = False

    def _flags_in_view_first(self, arm, solved):
        """`solved` in the same order, but postures where both gripper flags
        will be in view first, then one, then none -- checked lazily (~70 ms
        each). Both matters: with one flag the gripper's tilt comes from the
        joints alone, and their flex then shows at the fingertip (~3 mm per
        degree in simulation); two flags 218 mm apart measure it."""
        fewer = {1: [], 0: []}
        for s_ in solved:
            n = self.flag_visible(arm, s_[6])
            if n >= 2:
                yield s_
            else:
                fewer[n].append(s_)
        yield from fewer[1]
        yield from fewer[0]

    def _room_to_correct(self, arm, tcp, axis, quat, seed, room=0.01):
        """Can the arm still reach `tcp` moved `room` sideways, every way?
        The flags' corrections move the line in by the arm's flex (5-10 mm);
        at the edge of reach that can leave the corrected line unreachable."""
        side = np.cross(axis, [0.0, 0.0, 1.0])
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(axis, [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        up = np.cross(side, axis)
        return all(self.node.solve_ik(arm, tcp + room * d, quat, seed) is not None
                   for d in (side, -side, up, -up))

    def preplan(self, arm, aim, point, origin, start):
        """Plan the whole touch before anything moves. (plan, reason)

        For each candidate approach (front rings, then rolls by joint travel):
        IK for the approach and contact poses, a cuMotion plan from where the
        arm is to the approach pose (around the octomap), and the straight
        line from the approach pose to just past the contact. The first
        candidate that passes all of them is returned, with its planned
        trajectory, so execution flies exactly what was checked. None, with
        the reason, if nothing passes -- and then nothing moves.
        """
        node, args = self.node, self.node.args
        shoulder = node.lookup(BASE_FRAME, f'openarm_{arm}_link1')
        if shoulder is not None:
            far = float(np.linalg.norm(aim - shoulder[:3, 3])) - args.tip_offset - ARM_REACH
            if far > 0:
                return None, (f'out of reach: {100 * far:.0f} cm beyond what the {arm} arm '
                              f'can reach')
        tally = {'ik': 0, 'path': 0, 'line': 0}
        edge = []                    # plans with no room for corrections, if that is all
        rings = {}
        for cand in node.candidates(point, origin):
            rings.setdefault(cand[3], []).append(cand)
        for ring in sorted(rings):
            solved = []
            for label, axis, quat, _ring in rings[ring]:
                approach = node.tcp_for_tip(aim - axis * args.standoff, axis)
                contact = node.tcp_for_tip(aim - axis * args.touch_offset, axis)
                for seed in node.seeds(arm, start):
                    above = node.solve_ik(arm, approach, quat, seed)
                    if above is None:
                        continue
                    if node.solve_ik(arm, contact, quat, above) is not None:
                        cost = sum(abs(x - y) for x, y in zip(above, start))
                        solved.append((cost, label, axis, quat, approach, contact, above))
                        break
                else:
                    tally['ik'] += 1
            solved.sort(key=lambda s_: s_[0])
            if self.markers is not None and self.markers.has_arm(arm):
                solved = self._flags_in_view_first(arm, solved)
            for _cost, label, axis, quat, approach, contact, above in solved:
                # Cheap checks first (~50 ms line, ~100 ms room); a cuMotion
                # plan costs 0.5 s, a failed one ~5 s.
                # The contact move aims past the surface; at the edge of reach
                # 3 mm past is enough (it stops on contact anyway).
                for beyond in (args.past_contact, min(args.past_contact, 0.003)):
                    fraction, _t, _f = node.cartesian_plan(arm, contact + axis * beyond, quat,
                                                           start_joints=above)
                    if fraction >= 0.98:
                        break
                if fraction < 0.98:
                    tally['line'] += 1
                    continue
                plan = {'label': label, 'axis': axis, 'quat': quat, 'approach': approach,
                        'contact': contact, 'above': above, 'beyond': beyond, 'edge': False}
                if not self._room_to_correct(arm, contact + axis * beyond, axis, quat, above):
                    edge.append(dict(plan, edge=True))       # only if nothing better
                    continue
                _code, plan['trajectory'] = node.plan_joints(arm, above)
                if plan['trajectory'] is not None:
                    return plan, None
                tally['path'] += 1
        for plan in edge:
            _code, plan['trajectory'] = node.plan_joints(arm, plan['above'])
            if plan['trajectory'] is not None:
                return plan, None
            tally['path'] += 1
        if tally['path'] == 0 and tally['line'] == 0:
            return None, 'out of reach from the front (no arm posture gets there)'
        if tally['line'] and not tally['path']:
            return None, 'the straight line in is blocked or out of reach'
        return None, ('no collision-free path to the approach point (something is in '
                      'the way)' if tally['path'] else 'not reachable')

    def touch(self, arm, point, origin, scene=None):
        node, args = self.node, self.node.args
        start = node.arm_positions(arm)
        if start is None:
            self.say(f'no /joint_states for the {arm} arm yet', True)
            return
        node.require_move_group()
        # Every time: a respawned move_group has forgotten it, and without it
        # the clicked surface counts as a collision and nothing solves.
        ok, message = node.ensure_grippers_in_octomap()
        if not ok:
            self.say(f'{message} -- not moving', True)
            return
        if args.close_gripper:
            self.say(f'{arm} arm: closing the gripper so the fingertips meet')
            node.close_gripper(arm)

        # The target in the camera's own frame, for the marker check: it and
        # the gripper flags are then compared without going through tf.
        p_cam = None
        if scene is not None:
            t_wo = node.lookup(BASE_FRAME, scene[3])
            if t_wo is not None:
                p_cam = np.linalg.inv(t_wo)[:3, :3] @ point + np.linalg.inv(t_wo)[:3, 3]
        aim, learned, taught, markers_on = self.aim_for(arm, point, p_cam)

        t0 = time.monotonic()
        with self.plan_lock:                         # lets the click's own check finish
            check = self.check
            if (check is not None and check.get('plan') is not None
                    and check['pick'][2] is point and np.allclose(check['aim'], aim)
                    and time.monotonic() - check['time'] < 60.0
                    and np.max(np.abs(np.array(check['start']) - np.array(start))) < 0.003):
                plan, why = check['plan'], None          # checked on the click; nothing moved
            else:
                self.say(f'{arm} arm: checking the whole touch before moving...')
                plan, why = self.preplan(arm, aim, point, origin, start)
        if plan is None:
            self.say(f'{arm} arm: not moving -- {why}.', True)
            return
        axis, quat = plan['axis'], plan['quat']
        approach, contact, above = plan['approach'], plan['contact'], plan['above']
        node.publish_target(contact, quat)
        self.say(f'{arm} arm: planned in {time.monotonic() - t0:.1f} s [{plan["label"]}]'
                 + (f'; taught correction from {taught} touches' if taught else '')
                 + ('; flags will correct it' if markers_on else '')
                 + ('; AT THE EDGE OF REACH - a correction may not fit' if plan['edge']
                    else ''))

        ok, _ = node.execute(plan['trajectory'], f'{arm} approach')
        if not ok:
            self.say(f'{arm} arm: the approach did not complete', True)
            return
        offset = np.zeros(3)
        vis = np.zeros(3)            # flag measurement: real fingertip = model + vis
        off_at_approach = None
        d = None
        if markers_on:
            node.settle(arm, above, timeout=1.5)
            d = self.marker_offset(arm, p_cam, aim, axis, approach, 'the approach')
            if d is not None:
                vis = vis + d
                off_at_approach = float(np.linalg.norm(d - (d @ axis) * axis))
        if d is None:
            # No flag in view: at least get the joints onto their targets.
            offset, off_at_approach = self.correct_to(arm, approach, axis, quat,
                                                      'the approach', above)

        short = contact - axis * min(0.01, args.standoff / 2)
        near = node.line_move(arm, short - vis + offset, quat, f'{arm} in',
                              speed=args.fast_line_speed)
        ends, touched = None, False
        if near is not None:
            if markers_on:
                node.settle(arm, near, timeout=1.5)
                for _ in range(2):           # measure, correct, measure again
                    d = self.marker_offset(arm, p_cam, aim, axis, short, '1 cm short')
                    if d is None or np.linalg.norm(d) <= 0.001:
                        break
                    vis = vis + d
                    moved = node.line_move(arm, short - vis + offset, quat, f'{arm} flag fix')
                    if moved is None:
                        break
                    node.settle(arm, moved, timeout=1.5)
            else:
                offset, _off_near = self.correct_to(arm, short - vis, axis, quat, '1 cm short',
                                                    near, offset)
            # The last centimetre: slow, and stops when it feels the surface,
            # so the depth comes from the object itself, not from the camera.
            ends, touched = node.guarded_line(
                arm, contact + axis * plan['beyond'] - vis + offset, quat, f'{arm} contact')
        if ends is None:
            self.say(f'{arm} arm: no straight line in; could not touch', True)
        else:
            held = time.monotonic()
            node.settle(arm, ends, timeout=1.0)
            how = 'felt the surface' if touched else 'no contact felt'
            self.say(f'{arm} arm: touching ({how}); holding {args.hold:.0f} s')
            tip = node.fingertip(arm)
            seen_by = 'joints'
            if tip is not None:
                if markers_on:
                    # Where the camera sees the fingertip now, against the click.
                    here = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
                    d = (self.marker_offset(arm, p_cam, aim, axis, here[:3, 3], 'the touch')
                         if here is not None else None)
                    tip = tip + (d if d is not None else vis)
                    seen_by = 'camera' if d is not None else 'joints + flags'
                miss = aim - tip
                along = float(miss @ axis)
                sideways = float(np.linalg.norm(miss - along * axis))
                self.detail = (f'last touch: {how}; tip {1000 * sideways:.1f} mm from the click '
                               f'sideways ({seen_by})'
                               + (f', flags moved it {1000 * np.linalg.norm(vis):.1f} mm'
                                  if markers_on else '')
                               + (f', {1000 * off_at_approach:.1f} mm off at 5 cm'
                                  if off_at_approach is not None else '')
                               + f'; {held - t0:.1f} s to touch')
                node.get_logger().info(self.detail)
            if self.teach:
                here = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
                self.teach_touch(arm, point, learned, axis, quat,
                                 here[:3, 3] if here is not None else contact - vis + offset,
                                 scene)
            else:
                time.sleep(max(0.0, args.hold - (time.monotonic() - held)))
        self.say(f'{arm} arm: backing straight out')
        if node.line_move(arm, approach - vis + offset, quat, f'{arm} out',
                          speed=args.fast_line_speed) is None:
            node.move_joints(arm, above, f'{arm} retreat', velocity=args.touch_velocity,
                             via_home=False)
        self.say(f'{arm} arm: returning the way it came')
        back = node.retrace(arm, plan['trajectory'], f'{arm} return')
        if not back:
            back = node.move_joints(arm, start, f'{arm} return', via_home=False)
        if ends is not None and back:
            self.say(f'{arm} arm: done in {time.monotonic() - t0:.1f} s. Click the next point '
                     f'and press MOVE.')
        elif not back:
            self.say(f'{arm} arm: could not return to the start posture', True)

    # -- camera calibration ----------------------------------------------------

    def calibration_targets(self, t_world_optical):
        """Candidate fingertip positions: a 3 x 4 grid over the image, each
        cell at both depths (24), at least 7 cm in front of whatever the
        captured depth sees there.

        calibrate() stops once it has enough points, so a spot the arm cannot
        reach costs a candidate rather than a point -- with 12 candidates a
        run on 2026-10-01 lost half of them and was left with the bare
        minimum of 6. Per arm, every cell is tried once before any cell is
        tried at its other depth; the right arm's spots come first, then the
        left's, so the other arm is never parked in view.
        """
        bgr, depth, info, _frame = self.frame
        h, w = depth.shape
        near, far = self.node.args.calib_depths
        passes = ([], [])
        for row, fv in enumerate((0.25, 0.50, 0.75)):
            for col, fu in enumerate((0.15, 0.38, 0.62, 0.85)):
                order = (near, far) if (row + col) % 2 == 0 else (far, near)
                for targets, wanted in zip(passes, order):
                    u, v = fu * bgr.shape[1], fv * bgr.shape[0]
                    du, dv = int(u * w / bgr.shape[1]), int(v * h / bgr.shape[0])
                    patch = depth[max(0, dv - 5):dv + 6, max(0, du - 5):du + 6]
                    seen = patch[patch > 0.1]
                    if seen.size:
                        wanted = min(wanted, float(np.median(seen)) - 0.07)
                    if wanted < 0.22:
                        continue
                    p = t_world_optical[:3, :3] @ (pixel_ray(u, v, info) * wanted) \
                        + t_world_optical[:3, 3]
                    targets.append(('left' if fu < 0.5 else 'right', p))
        every = passes[0] + passes[1]
        return ([t for t in every if t[0] == 'right'] +
                [t for t in every if t[0] == 'left'])

    def calibration_axes(self, arm, tip, camera):
        """Tool axes to try at a calibration spot, best first.

        The fingertip has to be *seen*: the hand sits 95 mm behind it along
        the axis, so an axis within 60 deg of the camera ray puts the hand in
        front of the tip. Top-down first, then across the image, then the
        diagonals between -- whichever the arm can reach.
        """
        ray = tip - camera
        ray /= np.linalg.norm(ray)
        inward = np.array([0.0, 1.0 if arm == 'right' else -1.0, 0.0])
        down = np.array([0.0, 0.0, -1.0])
        back = -np.array([ray[0], ray[1], 0.0]) / max(1e-9, math.hypot(ray[0], ray[1]))
        preferred = [down, inward, down + inward, down + back, inward + back, back,
                     -inward, down - inward, np.array([0.0, 0.0, 1.0]), inward - down]
        axes = []
        for axis in preferred:
            axis = axis / np.linalg.norm(axis)
            if math.degrees(math.acos(np.clip(axis @ ray, -1.0, 1.0))) >= 60.0:
                axes.append(axis)
        return axes

    def calibration_pose(self, arm, tip, start, camera):
        """(joints, tool axis) for a pose that shows the fingertip to the
        camera, or (None, None)."""
        node = self.node
        seeds = [start] + ([node.retract[arm]] if arm in node.retract else [])
        ray = (tip - camera) / np.linalg.norm(tip - camera)
        for axis in self.calibration_axes(arm, tip, camera):
            best = roll_across_view(axis, ray)
            for roll in (best, best + math.pi):
                quat = quat_from_matrix(tool_frame_along(axis, roll))
                tcp = node.tcp_for_tip(tip, axis)
                for seed in seeds:
                    joints = node.solve_ik(arm, tcp, quat, seed)
                    if joints is not None:
                        return joints, axis
        return None, None

    def locate_tip(self, i, total, arm, measured, axis, t_wo, info, background):
        """--manual-calibration: the fingertip's pixel, clicked. None to skip;
        raises on abort."""
        node = self.node
        color = node.fresh_color()
        if color is None:
            return None
        predicted = project([measured], t_wo, info)[0]
        self.calib_event.clear()
        self.calib_view = (decode_image(color), predicted, None, None)
        where = ('the tip of the closed fingers' if node.args.close_gripper
                 else 'midway between the two fingertips')
        self.say(f'{i}/{total}: click {where} ({arm} arm); SKIP if hidden')
        self.calib_event.wait()
        what, clicked = self.calib_answer
        self.calib_view = None
        if what == 'abort':
            raise KeyboardInterrupt
        return None if what != 'click' else np.array(clicked, dtype=float)

    def observe_arm(self, i, total, arm, info, background, t_wo):
        """One pose's evidence for the 3D fit: the arm as the depth camera sees
        it, and the robot model where the encoders say it is."""
        node = self.node
        depth, _last = node.averaged_depth(6)
        color = node.fresh_color()
        if depth is None or color is None:
            return None
        surface = node.robot_surface()
        mp, mn = surface.posed(node.link_poses((arm,)))
        obs, mask = rcc.arm_points(depth, background, info)
        obs = rcc.near_model(obs, t_wo, mp)
        px, agree, seen = rcc.depth_agreement(depth, mp, mn, t_wo, info)
        self.overlay = (px[seen][::3], agree[seen][::3])
        self.calib_view = (decode_image(color), None, None, mask)
        if len(obs) < 300:
            self.say(f'{i}/{total}: only {len(obs)} arm points seen; skipping')
            time.sleep(0.5)
            return None
        self.last_pose_view = (decode_image(color), depth, mp, mn)
        self.say(f'{i}/{total}: {len(obs)} arm points; model/depth agreement '
                 f'{100 * agree[seen].mean() if seen.any() else 0:.0f}% before calibrating')
        time.sleep(0.5)
        return obs, mp, mn, depth

    def calibrate(self):
        node, args = self.node, self.node.args
        if not args.manual_calibration:
            self.say('loading the robot model (slow only the first time)...')
            node.robot_surface()
        self.say('calibration: taking a clean frame and map first...')
        self.capture()                               # background = the scene without the arm
        if self.frame is None:
            return
        background, info, optical = self.frame[1], self.frame[2], self.frame[3]
        t_wo = node.lookup(BASE_FRAME, optical)
        t_os = node.lookup(optical, SCREW_FRAME)
        if t_wo is None or t_os is None:
            self.say(f'no tf for {optical} / {SCREW_FRAME}', True)
            return
        targets = self.calibration_targets(t_wo)
        if len(targets) < 6:
            self.say('not enough free space in front of the camera to calibrate -- clear '
                     'the view to ~0.45 m and try again', True)
            return
        arms = sorted({arm for arm, _p in targets})
        starts = {arm: node.arm_positions(arm) for arm in arms}
        if any(s is None for s in starts.values()):
            self.say('no /joint_states yet', True)
            return
        node.require_move_group()
        node.ensure_grippers_in_octomap()
        if args.close_gripper:
            for arm in arms:
                node.close_gripper(arm)

        samples, observations, current = [], [], None
        self.abort.clear()
        self.calibrating = True
        try:
            per_arm = {}
            for i, (arm, tip) in enumerate(targets, start=1):
                if self.abort.is_set():
                    raise KeyboardInterrupt
                if per_arm.get(arm, 0) >= args.calib_points_per_arm:
                    continue                         # this arm has enough
                node.require_move_group()
                if current and current != arm:
                    self.say(f'{current} arm: back to where it started')
                    node.move_joints(current, starts[current], f'{current} return')
                current = arm
                self.say(f'calibration {i}/{len(targets)}: {arm} arm moving')
                joints, axis = self.calibration_pose(arm, tip, node.arm_positions(arm),
                                                     t_wo[:3, 3])
                # One direct attempt: a spot with no direct path is skipped,
                # not reached via home -- there are spare candidates.
                if joints is None or node._move_joints_once(
                        arm, joints, f'calib {i}') != MoveItErrorCodes.SUCCESS:
                    self.say(f'calibration {i}: {arm} arm cannot get there; skipping')
                    continue
                node.settle(arm, joints)
                if not args.manual_calibration:
                    seen = self.observe_arm(i, len(targets), arm, info, background, t_wo)
                    if seen is not None:
                        observations.append(seen)        # (obs, model pts, normals, depth)
                        per_arm[arm] = per_arm.get(arm, 0) + 1
                    continue
                measured = node.fingertip(arm)
                if measured is None:
                    continue
                pixel = self.locate_tip(i, len(targets), arm, measured, axis, t_wo, info,
                                        background)
                if pixel is not None:
                    samples.append((measured.tolist(), [float(pixel[0]), float(pixel[1])]))
                    per_arm[arm] = per_arm.get(arm, 0) + 1
        except KeyboardInterrupt:
            self.say('calibration aborted')
            return
        finally:
            self.calib_view = None
            self.calibrating = False
            if current and node.alive():
                self.say(f'{current} arm: back to where it started')
                node.move_joints(current, starts[current], f'{current} return')

        if args.manual_calibration:
            self.conclude_clicked(samples, t_wo, t_os, info)
        else:
            self.conclude_fitted(observations, t_wo, t_os, info)

    def conclude_fitted(self, captured, t_wo, t_os, info):
        """The 3D fit: camera pose from the arm's surfaces over every pose."""
        observations = [(obs, mp, mn) for obs, mp, mn, _depth in captured]
        arms_seen = len(observations)
        if arms_seen < 4:
            self.say(f'only {arms_seen} usable poses; 4 are needed. Clear the view in front '
                     f'of the camera and try again', True)
            return
        self.say(f'fitting the camera to {sum(len(o[0]) for o in observations)} arm points '
                 f'from {arms_seen} poses...')
        shape = self.frame[1].shape
        saved = rcc.save_run(captured, t_wo, t_os, info, shape)
        self.node.get_logger().info(f'calibration data saved to {saved}')
        solved, stats = rcc.fit_camera_multistart(observations, t_wo, info, shape)
        if getattr(self, 'last_pose_view', None) is not None:
            # The result, on the last pose's frame: the model as the new
            # camera pose places it. Judge it by eye before APPLY.
            img, depth, mp, mn = self.last_pose_view
            px, agree, seen = rcc.depth_agreement(depth, mp, mn, solved, info)
            self.frame = (img, depth, info, self.frame[3])
            self.pick = None
            self.overlay = (px[seen][::2], agree[seen][::2])
        before, within_before = rcc.alignment_error(observations, t_wo)
        after, within_after = rcc.alignment_error(observations, solved)

        def agreement(t):
            good = seen = 0
            for _obs, mp, mn, depth in captured:
                _px, agree, observed = rcc.depth_agreement(depth, mp, mn, t, info)
                good += int(agree[observed].sum())
                seen += int(observed.sum())
            return 100.0 * good / max(1, seen)
        agree_before, agree_after = agreement(t_wo), agreement(solved)
        shift = click_shift(t_wo, solved, info)
        evidence = (f'{arms_seen} arm poses, {sum(len(o[0]) for o in observations)} depth '
                    f'points fitted to the URDF meshes')
        ok = (stats['plane_rms_mm'] < 3.0 and stats['inlier_fraction'] > 0.7
              and within_after > 0.6)
        why = (f'fit rms {stats["plane_rms_mm"]:.1f} mm, {100 * within_after:.0f}% of points '
               f'within 5 mm')
        why += (f'; model/depth agreement {agree_before:.0f}% -> {agree_after:.0f}%; clicks '
                f'were {shift[0]:.0f} mm off on average, {shift[1]:.0f} mm at worst')
        self.finish_calibration(solved, t_wo, t_os, 'arm point cloud fitted to the robot model',
                                evidence, before, after, ok, why,
                                {'fit': stats, 'within_5mm_before': within_before,
                                 'within_5mm_after': within_after, 'poses': arms_seen,
                                 'agreement_before_pct': agree_before,
                                 'agreement_after_pct': agree_after,
                                 'click_shift_mm_avg': shift[0], 'click_shift_mm_max': shift[1]})

    def conclude_clicked(self, samples, t_wo, t_os, info):
        """--manual-calibration: solvePnP over clicked fingertips."""
        if len(samples) < 6:
            self.say(f'only {len(samples)} fingertips clicked; 6 are needed', True)
            return
        solved, before, after, used = solve_camera(samples, t_wo, info, max_drops=2)
        if solved is None:
            self.say('solvePnP failed -- try again', True)
            return
        k, _d = intrinsics(info)
        depth = float(np.mean([np.linalg.norm(np.array(s[0]) - t_wo[:3, 3]) for s in samples]))
        mm = lambda px: float(px * depth / k[0, 0] * 1000.0)
        rms_before = float(np.sqrt(np.mean(before[used] ** 2)))
        rms_after = float(np.sqrt(np.mean(after[used] ** 2)))
        self.finish_calibration(
            solved, t_wo, t_os, 'clicked fingertips (solvePnP)',
            f'{int(used.sum())} clicked fingertips', mm(rms_before), mm(rms_after),
            rms_after < 6.0, f'{rms_after:.1f} px left',
            {'samples': [{'fingertip_world': s[0], 'pixel': s[1], 'used': bool(u)}
                         for s, u in zip(samples, used)]})

    def finish_calibration(self, solved, t_wo, t_os, method, evidence, before_mm, after_mm,
                           ok, why, extra):
        """Record, sanity-check and (unless --no-auto-apply) apply a result."""
        old = t_wo @ t_os
        new = solved @ t_os
        moved = float(np.linalg.norm(new[:3, 3] - old[:3, 3]))
        turned = math.degrees(math.acos(np.clip(
            (np.trace(old[:3, :3].T @ new[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        record = plain({
            'when': datetime.datetime.now().isoformat(timespec='seconds'),
            'method': method, 'evidence': evidence,
            'frame': SCREW_FRAME, 'parent': BASE_FRAME,
            'xyz': [round(float(v), 5) for v in new[:3, 3]],
            'rpy': [round(float(v), 5) for v in rpy_from_matrix(new[:3, :3])],
            'previous_xyz': [round(float(v), 5) for v in old[:3, 3]],
            'previous_rpy': [round(float(v), 5) for v in rpy_from_matrix(old[:3, :3])],
            'error_before_mm': round(before_mm, 1), 'error_after_mm': round(after_mm, 1),
            'moved_mm': round(1000 * moved, 1), 'turned_deg': round(turned, 2),
            **extra,
        })
        # Plain Python types only: yaml.safe_dump refuses numpy scalars, and
        # that exception used to end the run right after the solve, with an
        # empty file and nothing applied (2026-10-01).
        text = yaml.safe_dump(record, sort_keys=False)
        with open(CALIBRATION_FILE, 'w') as handle:
            handle.write(text)
        self.detail = (f'camera moves {1000 * moved:.0f} mm, turns {turned:.1f} deg; {why}')
        plausible = ok and moved < 0.12 and turned < 15.0
        if not plausible:
            if moved < 0.12 and turned < 15.0:
                # Borderline: offer it, with the overlay on screen to judge by.
                self.calibration = record
                self.say(f'not sure of this one ({why}). The model is drawn where the new '
                         f'camera pose puts it: if it sits on the arm, press APPLY.', True)
            else:
                self.calibration = None
                self.say(f'result not trusted ({why}; moves {1000 * moved:.0f} mm / '
                         f'{turned:.1f} deg); nothing changed. See camera_calibration.yaml.',
                         True)
            return
        if moved < 0.002 and turned < 0.2:
            self.say(f'camera is already right (within {1000 * moved:.1f} mm / '
                     f'{turned:.2f} deg); nothing to change')
            return
        self.calibration = record
        if self.node.args.no_auto_apply:
            self.say('calibrated. APPLY to use it.')
            return
        self.apply_calibration()

    def apply_calibration(self):
        record = self.calibration
        if record is None:
            self.say('nothing to apply -- run CALIBRATE first', True)
            return
        (x, y, z), (r, p, yw) = record['xyz'], record['rpy']
        origin = f'<origin xyz="{x:.5f} {y:.5f} {z:.5f}" rpy="{r:.5f} {p:.5f} {yw:.5f}"/>'
        # How the user measures it: from the arm origin (link0). V7 moved the
        # arm mount to (0.0126, +/-, 0.7344); the centre between the arms is y=0.0062.
        from_arm = (f'{1000 * (x - 0.0126):.2f} mm in front of and {1000 * (0.7344 - z):.2f} mm '
                    f'below the arm origin, {1000 * (y - 0.0062):+.2f} mm to the left')

        with open(CAM_ORG) as handle:
            cam_org = handle.read()
        # "V4" alone or "V4  (note...)": both are version headings.
        version = 1 + max([int(n) for n in re.findall(r'^V(\d+)\b', cam_org, re.M)] or [3])
        tag = f'V{version}'

        # URDF: the whole sensor_d455 body is ours, so replace it outright.
        with open(URDF_XACRO) as handle:
            urdf = handle.read()
        body = (f'\n    <!-- {tag}: calibrated {record["when"]} with click_to_move.py '
                f'CALIBRATE ({record["method"]},\n'
                f'         {record["error_after_mm"]} mm left). History in cam_org.txt. -->\n'
                f'    {origin}\n  ')
        urdf, n = re.subn(r'(<xacro:sensor_d455[^>]*>).*?(</xacro:sensor_d455>)',
                          lambda m: m.group(1) + body + m.group(2), urdf, flags=re.S)
        if n != 1:
            self.say('could not find xacro:sensor_d455 in v10.urdf.xacro; nothing written',
                     True)
            return

        summary = (f'current ({tag}):    {origin}\n\n'
                   f'{from_arm}\n'
                   f'X = {1000 * x:.2f} mm, Y = {1000 * y:.2f} mm, Z = {1000 * z:.2f} mm\n'
                   f'roll = {math.degrees(r):.2f} deg, pitch = {math.degrees(p):.2f} deg, '
                   f'yaw = {math.degrees(yw):.2f} deg\n'
                   f'Measured with the arm (click_to_move.py CALIBRATE), not a tape.\n')
        cam_org = re.sub(r'current \(V\d+\):.*?(?=\nhistory:)', summary.rstrip('\n') + '\n',
                         cam_org, count=1, flags=re.S)
        block = (f'\n\n{tag}\n  {origin}\n  {from_arm}\n'
                 f'  calibrated {record["when"]}: {record["evidence"]}\n'
                 f'  arm vs model {record["error_before_mm"]} mm -> {record["error_after_mm"]} mm; '
                 f'moved {record["moved_mm"]} mm, turned {record["turned_deg"]} deg '
                 f'from the previous version\n')
        cam_org = cam_org.rstrip('\n') + block

        with open(VLM_DETECT) as handle:
            vlm = handle.read()
        vlm = re.sub(r'^URDF_MOUNT_XYZ = \(.*\)$', f'URDF_MOUNT_XYZ = ({x:.5f}, {y:.5f}, {z:.5f})',
                     vlm, count=1, flags=re.M)
        vlm = re.sub(r'^URDF_MOUNT_RPY = \(.*\)$', f'URDF_MOUNT_RPY = ({r:.5f}, {p:.5f}, {yw:.5f})',
                     vlm, count=1, flags=re.M)

        for path, text in ((URDF_XACRO, urdf), (CAM_ORG, cam_org), (VLM_DETECT, vlm)):
            with open(path, 'w') as handle:
                handle.write(text)
        self.calibration = None
        self.say(f'{tag} written; rebuilding openarm_description...')
        build = subprocess.run(
            [os.path.join(WS, 'native/build_ws.sh'), '--packages-select', 'openarm_description'],
            capture_output=True, text=True)
        built = build.returncode == 0
        live, why = self.node.live_update_camera((x, y, z), (r, p, yw))
        if live:
            # The map was built through the old camera pose.
            self.capture()
        notes = [] if built else ['rebuild failed: run native/build_ws.sh --packages-select '
                                  'openarm_description']
        if live:
            self.say(f'{tag} applied and live: camera is {from_arm}. '
                     + ' '.join(notes))
        else:
            self.say(f'{tag} written ({why}); restart launch_everything to use it. '
                     + ' '.join(notes), True)

    # -- drawing --------------------------------------------------------------

    def draw(self):
        if self.calib_view is not None:
            img, predicted, found, mask = self.calib_view
            img = img.copy()
            if mask is not None:
                img[mask] = (0.5 * img[mask] + 0.5 * np.array([255, 120, 0])).astype(np.uint8)
            if predicted is not None and np.all(np.isfinite(predicted)):
                pu, pv = int(round(predicted[0])), int(round(predicted[1]))
                cv2.circle(img, (pu, pv), 9, (0, 255, 0), 2)
                label = 'tf says the tip is here'
                lw = cv2.getTextSize(label, FONT, 0.45, 1)[0][0]
                lx = pu + 12 if pu + 12 + lw < img.shape[1] else pu - 12 - lw
                cv2.putText(img, label, (lx, pv + 4), FONT, 0.45, (0, 255, 0), 1)
            if found is not None:
                fu, fv = int(round(found[0])), int(round(found[1]))
                cv2.drawMarker(img, (fu, fv), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        elif self.frame is None:
            img = np.zeros((480, 640, 3), np.uint8)
        else:
            img = self.frame[0].copy()
            h, w = img.shape[:2]
            if self.unreachable is not None and self.unreachable.shape == (h, w):
                bad = self.unreachable
                img[bad] = (0.55 * img[bad] + 0.45 * np.array([40, 40, 200])).astype(np.uint8)
                cv2.putText(img, 'red tint: out of reach', (w // 2 - 80, h - 10), FONT, 0.45,
                            (120, 120, 255), 1)
            cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
            cv2.putText(img, 'LEFT ARM', (8, h - 10), FONT, 0.5, (255, 255, 255), 1)
            cv2.putText(img, 'RIGHT ARM', (w - 90, h - 10), FONT, 0.5, (255, 255, 255), 1)
            if self.pick:
                u, v, p, _arm, _origin = self.pick
                cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.circle(img, (u, v), 10, (0, 255, 255), 1)
                text = f'{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}'
                (tw, _th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
                tx = u + 12 if u + 12 + tw < img.shape[1] else u - 12 - tw
                cv2.putText(img, text, (tx, v - 12), FONT, 0.5, (0, 0, 0), 3)
                cv2.putText(img, text, (tx, v - 12), FONT, 0.5, (0, 255, 255), 1)
        if self.overlay is not None:
            pts, agree = self.overlay
            for (u, v), good in zip(pts.astype(int), agree):
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    img[v, u] = (0, 255, 0) if good else (0, 0, 255)
        if self.teaching:
            head = 'arrows: nudge   +/-: in/out   Enter: SAVE   Esc: SKIP'
        elif self.calibrating and self.node.args.manual_calibration:
            head = 'click the fingertip   SKIP / s   ABORT / x'
        elif self.calibrating:
            head = 'calibrating by itself - keep the view clear   ABORT / x'
        else:
            head = 'click=select  MOVE/m  RECAPTURE/r  CALIBRATE/c  OVERLAY/o  TEACH/t'
        for i, text in enumerate((head + ('  [busy]' if self.busy else ''), self.status)):
            y = 22 + 22 * i
            cv2.putText(img, text, (8, y), FONT, 0.5, (0, 0, 0), 3)
            cv2.putText(img, text, (8, y), FONT, 0.5, (255, 255, 255), 1)
        return self.with_buttons(img)

    def with_buttons(self, img):
        """The frame with a button bar under it (OpenCV has no widgets)."""
        h, w = img.shape[:2]
        bar = np.full((84, w, 3), 40, np.uint8)
        grey = (90, 90, 90)
        if self.teaching:
            nudge = (120, 90, 0)
            specs = [(n, n.upper(), nudge) for n in ('left', 'right', 'up', 'down', 'in', 'out')]
            specs += [('save', 'SAVE', (0, 150, 0)), ('skip', 'SKIP', (90, 90, 90))]
        elif self.calibrating:
            specs = [('abort', 'ABORT', (0, 0, 170))]
            if self.node.args.manual_calibration and self.calib_view is not None:
                specs.insert(0, ('skip', 'SKIP', (150, 90, 0)))
        else:
            idle = not self.busy
            ok = self.check['ok'] if self.check is not None else None
            move_colour = (grey if not (idle and self.pick) else (0, 150, 0) if ok
                           else (0, 120, 200) if ok is None else (0, 0, 150))
            specs = [('move', 'MOVE', move_colour),
                     ('recapture', 'RECAPTURE', (150, 90, 0) if idle else grey),
                     ('calibrate', 'CALIBRATE', (140, 0, 140) if idle else grey),
                     ('overlay', 'OVERLAY', (0, 140, 140) if idle or self.overlay else grey),
                     ('teach', 'TEACH ON' if self.teach else 'TEACH',
                      (0, 100, 220) if self.teach else (70, 70, 120)),
                     ('markers', 'FLAGS', (0, 140, 0) if self.markers is not None and idle
                      else (100, 100, 40) if idle else grey)]
            if self.calibration is not None:
                specs.append(('apply', 'APPLY', (0, 120, 200) if idle else grey))
        self.buttons = {}
        gap = 6
        width = min(146, (w - 16 - gap * (len(specs) - 1)) // len(specs))
        x = 8
        for name, text, colour in specs:
            x1 = x + width
            cv2.rectangle(bar, (x, 6), (x1, 44), colour, -1)
            cv2.rectangle(bar, (x, 6), (x1, 44), (255, 255, 255), 1)
            scale = 0.65
            while cv2.getTextSize(text, FONT, scale, 2)[0][0] > width - 10 and scale > 0.35:
                scale -= 0.05
            (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
            cv2.putText(bar, text, (x + (width - tw) // 2, 25 + th // 2), FONT, scale,
                        (255, 255, 255), 2)
            self.buttons[name] = (x, h + 6, x1, h + 44)
            x = x1 + gap
        if self.pick and not self.calibrating:
            _u, _v, p, arm, _o = self.pick
            info = f'{arm} arm   x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f} m (world)'
            cv2.putText(bar, info, (8, 62), FONT, 0.5, (0, 255, 255), 1)
        if self.detail:
            text = self.detail
            while cv2.getTextSize(text, FONT, 0.4, 1)[0][0] > w - 16 and len(text) > 10:
                text = text[:-4] + '...'
            cv2.putText(bar, text, (8, 78), FONT, 0.4, (200, 200, 200), 1)
        return np.vstack((img, bar))

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        def startup():
            self.say('waiting for move_group to take the gripper/octomap exemption...')
            _ok, message = self.node.ensure_grippers_in_octomap(patience=60.0)
            self.say(message)
            self.capture()
        self.in_background(startup)
        while rclpy.ok():
            cv2.imshow(WINDOW, self.draw())
            key = cv2.waitKey(30) & 0xFF
            # AUTOSIZE, not VISIBLE: GTK reports VISIBLE as 0 until the window
            # is mapped, which closed this on its first frame. AUTOSIZE is -1
            # only once the window has actually been destroyed.
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                if self.calibrating:
                    self.answer('abort')
                break
            if self.teaching:
                keys = {81: 'left', 83: 'right', 82: 'up', 84: 'down', ord('+'): 'in',
                        ord('='): 'in', ord('-'): 'out', 13: 'save', 10: 'save', 27: 'skip'}
                if key in keys:
                    self.answer(keys[key])
            elif self.calibrating:
                if key in (ord('s'), ord('S')):
                    self.answer('skip')
                elif key in (ord('x'), ord('X'), 27):
                    self.answer('abort')
            elif key in (ord('r'), ord('R')):
                self.in_background(self.capture)
            elif key in (ord('m'), ord('M'), 13, 10):
                self.move()
            elif key in (ord('c'), ord('C')):
                self.in_background(self.calibrate)
            elif key in (ord('o'), ord('O')):
                self.press('overlay')
            elif key in (ord('t'), ord('T')):
                self.press('teach')
            elif key in (ord('f'), ord('F')):
                self.press('markers')
        cv2.destroyAllWindows()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm', choices=['auto', 'left', 'right'], default='auto',
                        help='auto: left half of the frame -> left arm, right half -> right')
    parser.add_argument('--standoff', type=float, default=0.05,
                        help='the fingertips first stop this far from the point, then move '
                             'straight in, m')
    parser.add_argument('--line-speed', type=float, default=0.02,
                        help='fingertip speed for the straight move in and out, m/s')
    parser.add_argument('--touch-offset', type=float, default=0.0,
                        help='stop the fingertips this far short of the measured surface, m. '
                             'The arm is compliant (~6 N per cm of push), so 0 is safe')
    parser.add_argument('--tip-offset', type=float, default=FINGERTIP_BEYOND_TCP,
                        help='closed fingertips beyond hand_tcp along the tool axis, m')
    parser.add_argument('--close-gripper', action='store_true',
                        help='close the gripper before touching and calibrating, so the '
                             'fingertips meet at one point. Off by default: the gripper is '
                             'never commanded, and the point is aimed midway between the '
                             'fingertips')
    parser.add_argument('--hold', type=float, default=2.0, help='seconds to stay touching')
    parser.add_argument('--fast-line-speed', type=float, default=0.06,
                        help='tool speed for the straight moves in (to 1 cm short) and out, m/s')
    parser.add_argument('--contact-speed', type=float, default=0.01,
                        help='tool speed for the last, guarded centimetre, m/s')
    parser.add_argument('--past-contact', type=float, default=0.01,
                        help='how far past the measured surface the guarded move may go '
                             'before giving up on feeling it, m')
    parser.add_argument('--contact-torque', type=float, default=0.3,
                        help='joint torque change that counts as touching, Nm (0 = do not '
                             'watch: stop at the measured surface instead)')
    parser.add_argument('--orientation', choices=['front', 'down', 'any'], default='front',
                        help='front (default): approach from the side the camera sees, along '
                             'its line of sight or tilted up to 30 deg; down: top-down; any: '
                             'front, then top-down')
    parser.add_argument('--camera-check', action='store_true',
                        help='also correct the touch with what the camera sees of the gripper. '
                             'Off by default: in a front approach the hand is too close to the '
                             'camera to be measured, and the one fit it did get (at 0.46 m) '
                             'was 49 mm wrong and fought the taught corrections')
    parser.add_argument('--settle-time', type=float, default=8.0,
                        help='how long to let the arm converge before measuring it, s')
    parser.add_argument('--tolerance', type=float, default=0.002,
                        help='fingertip error at the approach worth correcting, m')
    parser.add_argument('--corrections', type=int, default=2,
                        help='correction moves at the approach and again 1 cm short, at most')
    parser.add_argument('--manual-calibration', action='store_true',
                        help='CALIBRATE asks you to click each fingertip instead of finding '
                             'it in the depth image')
    parser.add_argument('--no-auto-apply', action='store_true',
                        help='CALIBRATE stops at the result; APPLY writes it')
    parser.add_argument('--calib-points-per-arm', type=int, default=6,
                        help='fingertips CALIBRATE collects per arm before moving on '
                             '(out of 12 candidate spots each)')
    parser.add_argument('--marker-frames', type=int, default=8,
                        help='colour frames averaged for each flag measurement')
    parser.add_argument('--marker-poses-per-arm', type=int, default=3,
                        help='poses per arm for CALIBRATE MARKERS (FLAGS)')
    parser.add_argument('--calib-depths', type=float, nargs=2, default=[0.30, 0.40],
                        metavar=('NEAR', 'FAR'),
                        help='camera distances for the calibration fingertip grid, m')
    parser.add_argument('--velocity', type=float, default=0.6,
                        help='velocity and acceleration scaling for the long moves, 0..1')
    parser.add_argument('--touch-velocity', type=float, default=0.1,
                        help='scaling for the retreat when the straight line out fails')
    parser.add_argument('--planning-time', type=float, default=5.0)
    parser.add_argument('--attempts', type=int, default=3,
                        help='cuMotion resends per goal on PLANNING_FAILED')
    parser.add_argument('--motion-timeout', type=float, default=60.0)
    parser.add_argument('--patch', type=int, default=3,
                        help='depth median half-width around the click, pixels')
    parser.add_argument('--depth-frames', type=int, default=8,
                        help='aligned depth frames median-averaged per capture')
    parser.add_argument('--octomap-settle', type=float, default=1.5,
                        help='seconds to let move_group integrate a refreshed map')
    return parser


def main():
    args = make_parser().parse_args()

    rclpy.init()
    node = ClickToMove(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        App(node).run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
