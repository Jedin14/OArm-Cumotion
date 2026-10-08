"""Shared constants and small geometry/image helpers for click_to_move."""

import math
import os

import cv2
import numpy as np
from moveit_msgs.msg import MoveItErrorCodes


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


WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


URDF_XACRO = os.path.join(WS, 'src/openarm_description/urdf/robot/v10.urdf.xacro')


CAM_ORG = os.path.join(WS, 'cam_org.txt')




CALIBRATION_FILE = os.path.join(WS, 'camera_calibration.yaml')


NUDGE = 0.002                   # one TEACH nudge, m


OCTOMAP_NAME = '<octomap>'


# Closed fingertips, beyond hand_tcp along its +Z. finger_joint origin z=0.015
# in the hand frame, finger mesh spans 0.6585..0.7534 at an offset of -0.673,
# so the tip is at 0.015 + 0.0804 = 0.0954 m; hand_tcp is at 0.080.
FINGERTIP_BEYOND_TCP = 0.0154
# The vacuum extension fitted past the fingertips (2026-10-08): the working
# tip is this much further out along the tool axis. --tool-extension.
TOOL_EXTENSION = 0.020


# Same fallback as pick_place_orchestrator's home_joint_positions: symmetric,
# so a valid folded-down posture for either arm. Only an IK seed here.
HOME_JOINTS = [0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0]


# cuMotion's optimiser is stochastic; the same goal resent usually succeeds.
# CONTROL_FAILED: usually the arm was still moving when the plan started and
# MoveIt refused it (start point > 0.05 rad from the arm); worth one more try.
RETRYABLE = {MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.TIMED_OUT,
             MoveItErrorCodes.CONTROL_FAILED}


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
    """sensor_msgs/Image -> numpy, without cv_bridge."""
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
