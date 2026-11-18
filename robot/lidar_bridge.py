#!/usr/bin/env python3
"""The Unitree L2 4D LiDAR on the robot (2026-11-11): its mount, its cloud in the
robot's frames, the cloud merged with the D455's, and whether it is working.

    l2_driver (the user's 4D_lidar)  ->  /l2/cloud (frame l2_lidar), /l2/imu (l2_imu)
            this node        ->  TF openarm_body_link0 -> l2_lidar (the mount, static)
                                 /lidar/cloud         the same points (RViz)
                                 /lidar/cloud_work    only the work area in front (the octomap:
                                                      robot/octomap_gater.py passes it on a
                                                      refresh; octomap_live: every frame)
                                 /merged_cloud        the LiDAR (last accumulate_s) + the D455's
                                                      depth, coloured, frame openarm_body_link0
                                 /lidar/status        JSON, latched, 1 Hz: ok, rate, points, the
                                                      tilt -- the pick stack drops the
                                                      unseen-space walls while it is ok, and puts
                                                      them back if not
                                 the housing          a collision box in the planning scene, so
                                                      no plan swings an arm into the LiDAR --
                                                      where the L2 physically is (housing_xyz),
                                                      not where an alignment puts its mount

The driver hangs l2_imu on l2_lidar itself; this node hangs l2_lidar on the robot.
(Not the user's l2_stabilizer: it hangs l2_lidar on a 'world' of its own, the floor
below the LiDAR -- a second parent, and not the robot's world.)

The mount (the user, 2026-11-11): 111.3 mm above the centre of the two arms'
bases, 2.8 mm in front of it, tilted 60 deg down. The arms' bases (openarm.urdf)
are at x 0.0126, y +0.0572 / -0.0448, z 0.7344 in openarm_body_link0: their centre
is (0.0126, 0.0062, 0.7344), so the LiDAR is at (0.0154, 0.0062, 0.8457).

Its tilt: mount_rpy_deg as measured (pitched +60 deg, its X forward) by default.
tilt_from_imu takes it from its accelerometer instead (averaged over a still
imu_average_s: which way is up in its frame; its dome, +z, taken to lean straight
ahead). Not the default: on 2026-11-12 this L2's accelerometer read 1.9 g at rest
(x 15.8 m/s^2, z unchanged at 9.83) -- its x channel is not to be trusted. The IMU's
say goes into the status either way (imu_g, imu_lean_deg).

mount_spin_deg turns it about its own Z (RViz's blue axis, the dome) after the roll /
pitch / yaw: which side of it faces where. It and mount_xyz / mount_rpy_deg can be
changed while running (ros2 param set /lidar_bridge mount_spin_deg 135): the TF, the
housing and the work clouds follow at once.

Whether the mount is right: the D455 (calibrated) checks it. The LiDAR's work
points that fall in the camera's picture must sit at the camera's depth there
(camera_tol) -- min_agree of them, of at least min_in_view. A wrong mount (tilted
otherwise, turned, or still upright: 2026-11-12, 0 of 338 agreed) fails it, so the
LiDAR is not 'ok' and the walls stay.

Only true points are passed on (/lidar/cloud, /lidar/cloud_work, /merged_cloud, the
octomap):
- none while the LiDAR is not 'ok' -- then the camera alone is shown and used;
- none while the L2 reports its window dirty (/diagnostics, dirty_percentage above
  max_dirty_pct, 15): a dirty dome scatters its light into a shell of false points in
  front of the real surfaces (2026-11-12: 18%, the box 0.8 m away read 0.45-0.75 m);
- never a point the camera sees through (drop_seen_through): the camera has a surface
  farther along that view line, so that space is empty.
The driver's raw cloud stays on /l2/cloud.

2026-11-13, measured against the camera with the box close (0.9 m): two thirds of the
L2's points there were false -- short of the real surface by anything up to 0.6 m,
filling the empty space like fog -- and the true ones scattered +-3 cm. So:
- the robot is never an object (self_filter): points on the arms, the hands and the
  body (the URDF's collision shapes at the live TF, body_pad around them) are taken out
  of every cloud here, the camera's in /merged_cloud too, and so is anything on the way
  from the LiDAR to them (a ray that reached the arm met nothing before it);
- the camera comes first (camera_first): where it has a depth, it is millimetres
  right and the LiDAR is not, so there the LiDAR's points are dropped -- they stay only
  where the camera cannot see (outside its picture, in its holes, behind what it sees);
- no fog (steady_filter): a point stays only if, along its own view line, the last
  steady_window seconds put steady_min points at its range and no more than half as
  many beyond it (light that got farther proves the nearer point false).
- its range follows the camera (range_trim): the L2's ranges are off by a few cm, the
  same at 0.6 and at 1.0 m but not the same from hour to hour (2026-11-13: +1.2 cm with
  its window 10% dirty, -2.7 cm at 17%). Where both see the same surface the camera is
  right, so once a second the median difference along the LiDAR's rays is taken off ALL
  its ranges -- also where the camera does not see (at most range_trim_max, 5 cm).
"""

import json
import math
import os
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Pose, TransformStamped
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

HOUSING_ID = 'l2_lidar_housing'


def rpy_matrix(roll, pitch, yaw):
    """3x3 rotation, fixed-axis roll about x, then pitch about y, then yaw about z
    (the URDF / tf2 convention)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_from_matrix(m):
    """(x, y, z, w) of a 3x3 rotation."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return tuple(q)


def matrix_from_quat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mount_from_up(up, heading_deg=0.0):
    """The rotation (sensor -> parent) of a LiDAR whose accelerometer says `up` (in its
    own frame), its dome (+z) leaning toward heading_deg (left of straight ahead) --
    or, nearly upright (leaning < 10 deg) or hanging straight down, its X toward
    heading_deg. Returns (R, lean_deg): lean = its +z from straight up."""
    u = np.asarray(up, float)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    lean = math.degrees(math.acos(float(np.clip(u @ z, -1.0, 1.0))))
    f = z - (z @ u) * u                     # the dome's lean, horizontal
    if np.linalg.norm(f) < math.sin(math.radians(10.0)):
        x = np.array([1.0, 0.0, 0.0])
        f = x - (x @ u) * u
    f = f / np.linalg.norm(f)
    left = np.cross(u, f)
    h = math.radians(heading_deg)
    body = np.array([[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0],
                     [0.0, 0.0, 1.0]])
    return body @ np.stack([f, left, u], axis=1).T, lean


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_of(m):
    """(roll, pitch, yaw) deg of a 3x3 rotation (the rpy_matrix convention)."""
    return (math.degrees(math.atan2(m[2, 1], m[2, 2])),
            math.degrees(math.asin(float(np.clip(-m[2, 0], -1.0, 1.0)))),
            math.degrees(math.atan2(m[1, 0], m[0, 0])))


def camera_agreement(pts, depth, K, R, t, tol=0.04):
    """How many of the (N, 3) points (mount_parent's frame) fall in a depth image
    (metres, H x W; intrinsics K) and how many of those sit at its depth (within tol).
    R, t take mount_parent points into the camera's optical frame. -> (in_view, agree)"""
    if not len(pts):
        return 0, 0
    pc = np.asarray(pts, float) @ R.T + t
    z = pc[:, 2]
    front = z > 0.15
    pc, z = pc[front], z[front]
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = depth[v[inside].astype(int), u[inside].astype(int)]
    zl = z[inside]
    seen = d > 0.15                         # (no depth there: no say)
    return int(seen.sum()), int((np.abs(d[seen] - zl[seen]) < tol).sum())


def camera_sees_through(pts, depth, K, R, t, tol=0.04):
    """Bool mask (N) of the points (mount_parent's frame) that the depth camera sees
    THROUGH: they fall in its picture, it has a depth there, and that surface is farther
    than the point (by more than tol). The camera proves that space empty, so such a
    point is false (a dirty LiDAR window's scatter, a ghost). Points at the camera's
    surface, behind it (hidden from the camera) or outside its picture are not judged.
    R, t take mount_parent points into the camera's optical frame."""
    out = np.zeros(len(pts), bool)
    if not len(pts):
        return out
    pc = np.asarray(pts, float) @ R.T + t
    z = pc[:, 2]
    idx = np.where(np.isfinite(pc).all(axis=1) & (z > 0.15))[0]
    if not len(idx):
        return out
    u = K[0, 0] * pc[idx, 0] / z[idx] + K[0, 2]
    v = K[1, 1] * pc[idx, 1] / z[idx] + K[1, 2]
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = idx[inside]
    d = depth[v[inside].astype(int), u[inside].astype(int)]
    out[idx] = (d > 0.15) & (z[idx] < d - tol)
    return out


def camera_covers(pts, depth, K, R, t, tol=0.04, behind=None, edge_px=0, front=None):
    """Bool mask (N) of the points the depth camera speaks for: in its picture, with a
    depth there, and not behind that surface (by more than tol) -- in front of it (seen
    through: false) or at it (the camera has that surface, and better).
    behind (m, default tol): how far behind the camera's surface it still speaks for
    them (2026-11-17: the L2 sat 4-8 cm behind the camera's box faces, so with 4 cm its
    faces stayed in the map as a second, wrong surface). edge_px: points up to this
    many pixels outside the picture are judged by its edge pixel (a box at the edge of
    the picture had LiDAR-only voxels in front of it).
    A depth the camera did NOT measure there -- borrowed from a neighbour (fill_holes
    marks it negative) or an edge pixel standing in for a point outside the picture --
    speaks only for points AT that surface: from `front` (default tol) in front of it to
    `behind` behind it. The camera never saw through there, so a LiDAR point further in
    front is the only word on something there (a thin, dark or shiny object, one too
    near for the camera) and stays (2026-11-17 review: a pole 25 cm in front of the
    stack, in a 10 px camera hole, was taken out of the map)."""
    out = np.zeros(len(pts), bool)
    if not len(pts):
        return out
    pc = np.asarray(pts, float) @ R.T + t
    z = pc[:, 2]
    idx = np.where(np.isfinite(pc).all(axis=1) & (z > 0.15))[0]
    if not len(idx):
        return out
    u = K[0, 0] * pc[idx, 0] / z[idx] + K[0, 2]
    v = K[1, 1] * pc[idx, 1] / z[idx] + K[1, 2]
    h, w = depth.shape
    e = max(0, int(edge_px))
    inside = (u >= -e) & (u < w + e) & (v >= -e) & (v < h + e)
    idx = idx[inside]
    uu, vv = u[inside], v[inside]
    ui = np.clip(uu.astype(int), 0, w - 1)
    vi = np.clip(vv.astype(int), 0, h - 1)
    raw = depth[vi, ui]
    d = np.abs(raw)
    borrowed = (raw < 0) | (uu < 0) | (uu >= w) | (vv < 0) | (vv >= h)
    back = tol if behind is None else behind
    ahead = tol if front is None else front
    zi = z[idx]
    out[idx] = (d > 0.15) & (zi < d + back) & (~borrowed | (zi >= d - ahead))
    return out


def fill_holes(depth, px):
    """The depth picture with each hole (0) given the NEAREST depth within px pixels
    around it (still 0 where there is none), NEGATIVE: borrowed, not measured there --
    camera_covers lets it speak only for points at that surface. For camera_first
    only: the D455 has no depth along box edges, and the LiDAR's points that fell there
    were never judged -- their +-3-5 cm scatter put voxels 3-4 cm in front of a box
    face and blocked every way in (2026-11-17 17:46, box at x 0.408)."""
    if px <= 0 or not (depth <= 0).any():
        return depth
    from scipy.ndimage import minimum_filter
    big = np.where(depth > 0.15, depth, np.inf).astype(np.float32)
    near = minimum_filter(big, size=2 * int(px) + 1, mode='nearest')
    out = np.where(depth > 0.15, depth, np.where(np.isfinite(near), -near, 0.0))
    return out.astype(np.float32)


def camera_range_error(pts, dirs, depth, K, R, t, window=0.08):
    """How much too long (+) or short (-) the LiDAR's ranges are, per point, judged by the
    depth camera: for the points (N, 3; mount_parent's frame; dirs the unit view lines of
    the LiDAR to them) that fall in its picture within `window` of its surface, the step
    along the LiDAR's own ray that would put them on it. -> 1-D array (those points only)."""
    if not len(pts):
        return np.zeros(0)
    pc = np.asarray(pts, float) @ R.T + t
    along = (np.asarray(dirs, float) @ R.T)[:, 2]       # the ray's share along the camera's view
    z = pc[:, 2]
    idx = np.where(np.isfinite(pc).all(axis=1) & (z > 0.15) & (along > 0.5))[0]
    if not len(idx):
        return np.zeros(0)
    u = K[0, 0] * pc[idx, 0] / z[idx] + K[0, 2]
    v = K[1, 1] * pc[idx, 1] / z[idx] + K[1, 2]
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = idx[inside]
    d = depth[v[inside].astype(int), u[inside].astype(int)]
    ok = (d > 0.15) & (np.abs(z[idx] - d) < window)
    return (z[idx][ok] - d[ok]) / along[idx][ok]


def with_xyz(msg, xyz):
    """A copy of `msg` with its points' x, y, z replaced (float32 fields; else unchanged)."""
    off = {f.name: (f.offset, f.datatype) for f in msg.fields}
    n = msg.width * msg.height
    if n == 0 or any(off.get(k, (0, 0))[1] != PointField.FLOAT32 for k in 'xyz') or msg.is_bigendian:
        return msg
    raw = np.frombuffer(bytes(msg.data), np.uint8, count=n * msg.point_step).reshape(
        n, msg.point_step).copy()
    for i, k in enumerate('xyz'):
        raw[:, off[k][0]:off[k][0] + 4] = np.ascontiguousarray(xyz[:, i], '<f4').view(np.uint8).reshape(n, 4)
    out = restamp(msg, msg.header.frame_id)
    out.data = raw.tobytes()
    return out


def view_cells(xyz, rng, cell_deg):
    """An integer per point: which cell of the sensor's sky (cell_deg x cell_deg, equal
    area) its view line falls in. xyz in the sensor's own frame, rng its length (> 0)."""
    th = np.degrees(np.arccos(np.clip(xyz[:, 2] / rng, -1.0, 1.0)))
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    ti = (th / cell_deg).astype(np.int64)
    naz = np.maximum(1, np.round(360.0 * np.sin(np.radians((ti + 0.5) * cell_deg)) / cell_deg))
    return ti * 4096 + ((az + 180.0) / 360.0 * naz).astype(np.int64) % naz.astype(np.int64)


class SteadyFilter:
    """Which points of a scan are steady: along the point's own view line (its cell), the
    last `window` seconds hold at least min_n points at its range (within tol, growing 2
    cm a metre) and at most half as many beyond it. A dirty or dazzled LiDAR's false
    points are scattered along the line, short of the real surface; the surface itself
    repeats."""

    def __init__(self, cell_deg=2.0, window=2.0, tol=0.03, min_n=4):
        self.cell_deg, self.window, self.tol, self.min_n = cell_deg, window, tol, min_n
        self.hist = []                      # (monotonic, key * 100 + range)

    def __call__(self, xyz, rng, valid, now):
        """-> (bool mask of the steady points, the cells' keys; -1 where not valid)."""
        keys = np.full(len(xyz), -1, np.int64)
        keep = np.zeros(len(xyz), bool)
        idx = np.where(valid)[0]
        if not len(idx):
            return keep, keys
        r = np.minimum(rng[idx].astype(np.float64), 50.0)
        keys[idx] = view_cells(xyz[idx].astype(np.float64), np.maximum(r, 1e-6), self.cell_deg)
        c = keys[idx] * 100.0 + r
        self.hist = [(a, v) for a, v in self.hist if now - a < self.window] + [(now, c)]
        allc = np.sort(np.concatenate([v for _a, v in self.hist]))
        tol = self.tol + 0.02 * r
        lo = np.searchsorted(allc, c - tol, 'left')
        hi = np.searchsorted(allc, c + tol, 'right')
        end = np.searchsorted(allc, keys[idx] * 100.0 + 99.0, 'right')
        near, far = hi - lo, end - hi
        keep[idx] = (near >= self.min_n) & (near >= 2 * far)
        return keep, keys


class RobotBody:
    """The robot's own surfaces -- the URDF's collision shapes, as points a few mm apart
    in each link's frame -- to tell which points of a cloud are the robot itself."""

    def __init__(self, urdf_path, spacing=0.008):
        import trimesh                      # (not there: the caller reports, no self-filter)
        from scipy.spatial import cKDTree
        self.links = {}                     # link -> (tree, lowest corner, highest corner)
        rng = np.random.default_rng(0)
        for link in ET.parse(urdf_path).getroot().findall('link'):
            pts = []
            for col in link.findall('collision'):
                geo = col.find('geometry')
                shape = geo[0] if geo is not None and len(geo) else None
                if shape is None:
                    continue
                if shape.tag == 'mesh':
                    path = self.resolve(shape.attrib['filename'], urdf_path)
                    if path is None:
                        continue
                    mesh = trimesh.load(path, force='mesh')
                    scale = [float(v) for v in shape.attrib.get('scale', '1 1 1').split()]
                    verts = np.asarray(mesh.vertices, float) * scale
                    mesh = trimesh.Trimesh(verts, mesh.faces, process=False)
                elif shape.tag == 'cylinder':
                    mesh = trimesh.creation.cylinder(radius=float(shape.attrib['radius']),
                                                     height=float(shape.attrib['length']))
                elif shape.tag == 'box':
                    mesh = trimesh.creation.box([float(v) for v in shape.attrib['size'].split()])
                elif shape.tag == 'sphere':
                    mesh = trimesh.creation.icosphere(radius=float(shape.attrib['radius']))
                else:
                    continue
                n = int(np.clip(abs(mesh.area) / spacing ** 2, 300, 8000))
                tri = np.asarray(mesh.triangles, float)
                area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
                if area.sum() <= 0:
                    continue
                pick = rng.choice(len(tri), n, p=area / area.sum())
                a, b = rng.random(n), rng.random(n)
                flip = a + b > 1
                a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
                s = (tri[pick, 0] + a[:, None] * (tri[pick, 1] - tri[pick, 0])
                     + b[:, None] * (tri[pick, 2] - tri[pick, 0]))
                s = np.concatenate([s, np.asarray(mesh.vertices, float)])
                org = col.find('origin')
                if org is not None:
                    rpy = [float(v) for v in org.attrib.get('rpy', '0 0 0').split()]
                    xyz = [float(v) for v in org.attrib.get('xyz', '0 0 0').split()]
                    s = s @ rpy_matrix(*rpy).T + np.array(xyz)
                pts.append(s)
            if pts:
                pts = np.concatenate(pts)
                self.links[link.attrib['name']] = (cKDTree(pts), pts.min(axis=0), pts.max(axis=0))

    @staticmethod
    def resolve(filename, urdf_path):
        """The file of a URDF mesh name (package://pkg/..., file://..., a path), or None."""
        if filename.startswith('package://'):
            pkg, _, rest = filename[len('package://'):].partition('/')
            roots = []
            try:
                from ament_index_python.packages import get_package_share_directory
                roots.append(get_package_share_directory(pkg))
            except Exception:               # noqa: BLE001 - not installed: the source tree
                pass
            ws = os.path.dirname(os.path.dirname(os.path.abspath(urdf_path)))
            roots.append(os.path.join(ws, 'src', pkg))
            for root in roots:
                if os.path.isfile(os.path.join(root, rest)):
                    return os.path.join(root, rest)
            return None
        path = filename[len('file://'):] if filename.startswith('file://') else filename
        return path if os.path.isfile(path) else None

    def mask(self, pts, link_tf, pad):
        """Bool mask (N) of the points within pad of the robot's surfaces. link_tf(name)
        -> (R, t) taking that link's points into the points' frame, or None (not known:
        that link is not judged). Also returns how many links had no tf."""
        out = np.zeros(len(pts), bool)
        missing = 0
        if not len(pts):
            return out, missing
        pts = np.asarray(pts, float)
        for name, (tree, lo, hi) in self.links.items():
            tf = link_tf(name)
            if tf is None:
                missing += 1
                continue
            q = (pts - tf[1]) @ tf[0]
            cand = np.where(~out & (q >= lo - pad).all(axis=1) & (q <= hi + pad).all(axis=1))[0]
            if len(cand):
                d, _ = tree.query(q[cand], k=1, distance_upper_bound=pad)
                out[cand[np.isfinite(d)]] = True
        return out, missing


def in_box(p, box):
    """Bool mask of the (N, 3) points p inside box ((x0, x1), (y0, y1), (z0, z1))."""
    return ((p[:, 0] >= box[0, 0]) & (p[:, 0] <= box[0, 1]) & (p[:, 1] >= box[1, 0])
            & (p[:, 1] <= box[1, 1]) & (p[:, 2] >= box[2, 0]) & (p[:, 2] <= box[2, 1]))


def cloud_xyz(msg, extra=()):
    """(N, 3) float32 x, y, z of a PointCloud2 -- read by the field offsets, so the
    driver's padding does not matter -- and a dict of the `extra` fields."""
    offsets = {f.name: f.offset for f in msg.fields}
    kinds = {PointField.FLOAT32: 'f4', PointField.FLOAT64: 'f8', PointField.UINT16: 'u2',
             PointField.UINT8: 'u1', PointField.INT32: 'i4', PointField.UINT32: 'u4'}
    names = ['x', 'y', 'z'] + [n for n in extra if n in offsets]
    fmt = {f.name: kinds.get(f.datatype, 'f4') for f in msg.fields}
    n = msg.width * msg.height
    if n == 0 or not all(k in offsets for k in 'xyz'):
        return np.zeros((0, 3), np.float32), {}
    dtype = np.dtype({'names': names, 'formats': [('>' if msg.is_bigendian else '<') + fmt[k]
                                                  for k in names],
                      'offsets': [offsets[k] for k in names], 'itemsize': msg.point_step})
    arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)
    return xyz, {k: arr[k] for k in names[3:]}


def make_cloud(xyz, frame, stamp, rgb=None):
    """A PointCloud2 of x, y, z (and rgb, packed as the float RViz reads)."""
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
              for i, n in enumerate('xyz')]
    if rgb is not None:
        rgb = np.asarray(rgb, np.uint8).reshape(-1, 3)
        packed = ((rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8)
                  | rgb[:, 2].astype(np.uint32)).view(np.float32)
        data = np.concatenate([xyz, packed[:, None]], axis=1).astype(np.float32)
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
    else:
        data = xyz
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.header.stamp = stamp
    msg.height, msg.width = 1, int(len(data))
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 4 * data.shape[1] if len(data) else 4 * (4 if rgb is not None else 3)
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


def restamp(msg, frame):
    """The same cloud, said to be in `frame` (no copy of the points)."""
    out = PointCloud2()
    out.header.stamp = msg.header.stamp
    out.header.frame_id = frame
    out.height, out.width = msg.height, msg.width
    out.fields, out.is_bigendian = msg.fields, msg.is_bigendian
    out.point_step, out.row_step = msg.point_step, msg.row_step
    out.data, out.is_dense = msg.data, msg.is_dense
    return out


def select_points(msg, keep):
    """A copy of `msg` with only the points where keep (bool, N) is True -- every
    field kept as it is."""
    n = msg.width * msg.height
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8, count=n * msg.point_step)
    rows = raw.reshape(n, msg.point_step)[np.asarray(keep, bool)]
    out = restamp(msg, msg.header.frame_id)
    out.height, out.width = 1, int(len(rows))
    out.row_step = msg.point_step * out.width
    out.data = rows.tobytes()
    return out


class LidarBridge(Node):

    def __init__(self):
        super().__init__('lidar_bridge')
        p = self.declare_parameter
        p('in_topic', '/l2/cloud')
        p('imu_topic', '/l2/imu')
        p('tilt_from_imu', False)           # its tilt from its accelerometer (see the top)
        p('imu_average_s', 2.0)
        p('frame', 'l2_lidar')
        p('mount_parent', 'openarm_body_link0')
        p('mount_xyz', [0.0154, 0.0062, 0.8457])
        p('mount_rpy_deg', [0.0, 60.0, 0.0])
        p('mount_spin_deg', 0.0)            # then about its own Z (blue, the dome axis)
        # the work area in front (mount_parent's frame, m): what goes to the octomap.
        # Behind x_min is the robot itself and the drop: the arm going there (out of the
        # camera's view, not of the LiDAR's) never becomes voxels
        p('work_box', [0.15, 1.5, -0.9, 0.9, -0.5, 1.6])
        p('range_min', 0.15)                # nearer the sensor: its own mount
        p('accumulate_s', 3.0)              # /merged_cloud keeps this much LiDAR (denser: its
                                            # scan never repeats, so 3 s is ~2x the points of 1.5)
        p('merged_rate', 2.0)
        p('camera_stride', 4)               # every 4th pixel of the D455 (160 x 120)
        p('octomap_live', False)            # true with octomap:=live: every work cloud
        p('ok_rate', 2.0)                   # clouds a second, and ...
        p('ok_points', 200)                 # ... points in the reach box, to count as working
        # where the arms work, below the LiDAR (mount_parent's frame, m): the LiDAR is
        # only 'ok' (the walls off) when it sees in there. Standing upright it sees a
        # hemisphere above itself and nothing in here
        p('reach_box', [0.15, 1.0, -0.7, 0.7, -0.5, 0.7])
        p('min_lean_deg', 20.0)             # less: 'it looks up' (the status' why)
        # the camera's check of the mount (see the top)
        p('check_camera', True)
        p('camera_tol', 0.04)               # m, along the camera's view
        p('min_agree', 0.3)                 # of the LiDAR's points in the picture
        p('min_in_view', 50)
        # only true points: none while the LiDAR fails a check; never one the camera sees through
        p('drop_seen_through', True)
        # the L2's own 'cover dirty' share. 2026-11-12: at 18.5 its near field was a shell of
        # false points (3% agreed with the camera); wiped, it sits at 8-11 and 55% agree
        p('max_dirty_pct', 15.0)
        # once ok it stays ok until clearly worse (no flicker at a limit): dirty above
        # max_dirty_pct + dirty_slack, agreement below min_agree - agree_slack
        p('dirty_slack', 2.0)
        p('agree_slack', 0.10)
        # the robot is never an object: its own points out of every cloud (see the top)
        p('self_filter', True)
        p('urdf_path', '/workspaces/isaac_ros-dev/robot/openarm.urdf')
        p('body_pad', 0.08)                 # m around the robot's surfaces, the LiDAR's points (its
                                            # points on the arm: 91% within 6 cm, 95% within 8)
        p('body_pad_camera', 0.03)          # ... the camera's (it is the more exact)
        # where the camera has a depth, it speaks and the LiDAR does not
        p('camera_first', True)
        p('camera_first_behind', 0.12)      # m: LiDAR points this far behind a camera surface
                                            # are dropped too (2026-11-17: the L2 sat 4-8 cm
                                            # behind the camera's faces; 0.04 = as before)
        p('camera_first_fill_px', 25)       # px: camera holes this near a depth count as that
                                            # depth for camera_first (0: off, as before)
        p('camera_first_fill_front', 0.05)  # m: ... but only for LiDAR points at most this far
                                            # in front of it (and camera_first_behind behind):
                                            # the camera never saw through a hole, or past the
                                            # picture's edge (2026-11-17 review)
        # no fog: only points that repeat along their view line
        p('steady_filter', True)
        p('steady_cell_deg', 2.0)
        p('steady_window', 2.0)
        p('steady_tol', 0.03)
        p('steady_min', 4)
        p('housing_size', [0.09, 0.09, 0.08])
        # where the L2 physically IS, for its housing box -- not the mount above, which is a
        # calibration and moves with every alignment: 2026-11-16 one by hand put it 6.4 cm left
        # (y 0.067) and the box into the left arm's base (openarm_left_link0 painted red,
        # MoveIt: in collision). The 11-13 body fit: ~26 mm clear of both arm bases and the body.
        # housing_xyz [] = the box follows the mount, as before.
        p('housing_xyz', [-0.028, 0.003, 0.8829])
        p('housing_rpy_deg', [2.2, 113.4, 0.8])
        p('housing_spin_deg', 161.3)
        # its ranges follow the camera's where both see the same surface (see the top)
        p('range_trim', True)
        p('range_trim_max', 0.05)           # m
        p('range_trim_gain', 0.3)           # of the measured difference, a second
        p('range_trim_min_points', 300)
        # no specks in the air (2026-11-16: single voxels floating in front of the boxes
        # blocked the arm's straight lines in): a point goes on only with at least
        # density_min others within density_radius among the last density_window s
        p('density_filter', True)
        p('density_radius', 0.03)
        p('density_min', 8)                 # (replayed 2026-11-16: specks 134 -> 52, 10% of points)
        p('density_window', 2.0)
        self.dense_hist = []                # (monotonic, the passed points of a scan)
        self.trim = 0.0                     # m, added to every range
        self.trim_hist = []                 # (monotonic, range errors of the points on the camera's surface)
        self.frame = self.get_parameter('frame').value
        self.parent = self.get_parameter('mount_parent').value
        xyz = [float(v) for v in self.get_parameter('mount_xyz').value]
        self.R = self.measured_R()
        self.t = np.array(xyz)
        self.box = np.array([float(v) for v in self.get_parameter('work_box').value]).reshape(3, 2)
        self.reach = np.array([float(v) for v in self.get_parameter('reach_box').value]).reshape(3, 2)
        self.heading = float(self.get_parameter('mount_rpy_deg').value[2])
        self.mount_from = 'measured'        # or 'imu'
        self.up = None                      # the accelerometer's up (its frame), once known
        self.lean = None
        self.imu_samples = []
        self.imu_say = None                 # (|a| m/s^2, lean deg) of the last still window
        self.cam_check = None               # (in_view, agree) of the last status
        gp = lambda k: self.get_parameter(k).value
        self.steady = SteadyFilter(float(gp('steady_cell_deg')), float(gp('steady_window')),
                                   float(gp('steady_tol')), int(gp('steady_min')))
        self.body = None                    # RobotBody, or None: no self-filter
        self.body_why = 'off (self_filter:=false)'
        if gp('self_filter'):
            try:
                self.body = RobotBody(gp('urdf_path'))
                self.body_why = ''
                if not self.body.links:
                    self.body, self.body_why = None, f"no collision shapes found in {gp('urdf_path')}"
            except Exception as exc:        # noqa: BLE001 - no trimesh / scipy / urdf
                self.body_why = f'{type(exc).__name__}: {exc}'
            if self.body is None:
                self.get_logger().error('NO SELF-FILTER: the arms will show in /lidar/cloud and '
                                        f'/merged_cloud ({self.body_why})')
        self.filter_hist = []               # (monotonic, points, not steady, on the robot)
        self.body_missing = 0

        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_cloud = self.create_publisher(PointCloud2, '/lidar/cloud', 5)
        self.pub_work = self.create_publisher(PointCloud2, '/lidar/cloud_work', 5)
        self.pub_octo = self.create_publisher(PointCloud2, '/lidar/cloud_octomap', 5)
        self.pub_merged = self.create_publisher(PointCloud2, '/merged_cloud', 2)
        # for fitting the mount (sensor_align.py): steady, not the robot, but NOT judged by
        # the camera -- /lidar/cloud drops what the camera covers, so a fit on it only saw
        # the points that disagree with the camera (2026-11-17). Also while not trusted:
        # a wrong mount is exactly when it is needed.
        self.pub_fit = self.create_publisher(PointCloud2, '/lidar/cloud_fit', 2)
        self.pub_status = self.create_publisher(String, '/lidar/status', latched)
        topic = self.get_parameter('in_topic').value
        # best effort takes a reliable publisher's messages too
        self.create_subscription(PointCloud2, topic, self.on_cloud, sensor, callback_group=cb)
        self.create_subscription(Imu, self.get_parameter('imu_topic').value, self.on_imu,
                                 QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT),
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw',
                                 lambda m: self._keep('depth', m), sensor, callback_group=cb)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 lambda m: self._keep('color', m), sensor, callback_group=cb)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 lambda m: self._keep('info', m), sensor, callback_group=cb)
        self.create_subscription(DiagnosticArray, '/diagnostics', self.on_diagnostics, 10,
                                 callback_group=cb)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static = StaticTransformBroadcaster(self)
        self.static.sendTransform(self.mount_tf())
        self.add_on_set_parameters_callback(self.on_params)

        self.lock = threading.Lock()
        self.latest = {}
        self.history = []                   # (monotonic, (N, 3) work points in the parent frame)
        self.check_hist = []                # the same before the camera's filter: what its check judges
        self.drop_hist = []                 # (monotonic, points in the camera's view, of them seen through)
        self.trusted = False                # the last status was ok: its points are passed on
        self.dirty = None                   # (the L2's dirty-window %, monotonic)
        self._depth_m = (None, None)        # (the depth message, it in metres)
        self.arrivals = []                  # monotonic times of the last clouds
        self.last_work_n = 0
        self.reach_hist = []                # (monotonic, points in the reach box)
        self.scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene',
                                        callback_group=cb)
        self.create_timer(1.0, self.publish_status, callback_group=cb)
        self.create_timer(1.0 / max(0.2, float(self.get_parameter('merged_rate').value)),
                          self.publish_merged, callback_group=cb)
        self.create_timer(30.0, self.add_housing, callback_group=cb)
        self.create_timer(3.0, self._first_housing, callback_group=cb)
        self.get_logger().info(
            f'{topic} -> /lidar/cloud ({self.frame}); mount {self.parent} -> {self.frame} at '
            f'xyz {np.round(self.t, 4).tolist()} rpy {self.get_parameter("mount_rpy_deg").value} '
            f'deg, spin {self.get_parameter("mount_spin_deg").value} deg about its own Z (= rpy '
            f'{[round(v, 1) for v in rpy_of(self.R)]})'
            + (' until its IMU gives its tilt' if self.get_parameter('tilt_from_imu').value
                      else '') + f'; the work area {self.box.tolist()}'
            + (f'; self-filter: {len(self.body.links)} links of the robot' if self.body else ''))

    # -- the mount ---------------------------------------------------------------------

    def mount_tf(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.parent
        tf.child_frame_id = self.frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = \
            (float(v) for v in self.t)
        q = quat_from_matrix(self.R)
        (tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z,
         tf.transform.rotation.w) = (float(v) for v in q)
        return tf

    def to_parent(self, xyz):
        return xyz @ self.R.T + self.t

    def measured_R(self, rpy=None, spin=None):
        """The mount's rotation as set: roll / pitch / yaw, then mount_spin_deg about its
        own Z."""
        rpy = self.get_parameter('mount_rpy_deg').value if rpy is None else rpy
        spin = self.get_parameter('mount_spin_deg').value if spin is None else spin
        return rpy_matrix(*[math.radians(float(v)) for v in rpy]) @ rot_z(float(spin))

    def on_params(self, params):
        """mount_xyz / mount_rpy_deg / mount_spin_deg changed while running: the new mount
        at once (the TF, the housing; the clouds kept so far were in the old one)."""
        new = {q.name: q.value for q in params}
        if not {'mount_xyz', 'mount_rpy_deg', 'mount_spin_deg'} & set(new):
            return SetParametersResult(successful=True)
        try:
            xyz = np.array([float(v) for v in new.get('mount_xyz',
                                                      self.get_parameter('mount_xyz').value)])
            rpy = [float(v) for v in new.get('mount_rpy_deg',
                                             self.get_parameter('mount_rpy_deg').value)]
            spin = float(new.get('mount_spin_deg', self.get_parameter('mount_spin_deg').value))
            assert len(xyz) == 3 and len(rpy) == 3
        except (TypeError, ValueError, AssertionError):
            return SetParametersResult(successful=False,
                                       reason='mount_xyz / mount_rpy_deg: 3 numbers; spin: 1')
        R = self.measured_R(rpy, spin)
        with self.lock:
            self.t = xyz
            if self.mount_from != 'imu':
                self.R = R
            self.history, self.reach_hist, self.check_hist = [], [], []
            self.trim, self.trim_hist = 0.0, []
            if self.mount_from == 'imu':
                self.up = None              # (the next still window re-applies, new heading)
        self.heading = rpy[2]
        self.static.sendTransform(self.mount_tf())
        self._housing_done = False
        self.add_housing()
        eq = [round(v, 1) for v in rpy_of(R)]
        self.get_logger().info(
            f'mount set: xyz {np.round(xyz, 4).tolist()}, rpy {rpy} deg, then {spin:g} deg about its '
            f'own Z (blue) -- as one roll / pitch / yaw: {eq} deg'
            + (' (the tilt itself stays from the IMU)' if self.mount_from == 'imu' else ''))
        return SetParametersResult(successful=True)

    def on_imu(self, msg):
        """Average the accelerometer over a still imu_average_s: up in the LiDAR's frame,
        so its tilt. Set the mount from it, and again when it changes by > 3 deg."""
        a = msg.linear_acceleration
        now = time.monotonic()
        self.imu_samples.append((now, a.x, a.y, a.z))
        if now - self.imu_samples[0][0] < float(self.get_parameter('imu_average_s').value):
            return
        acc = np.array([v[1:] for v in self.imu_samples])
        self.imu_samples = []
        up = acc.mean(axis=0)
        g = float(np.linalg.norm(up))
        if len(acc) < 16 or g < 1e-3:
            return
        self.imu_say = (g, math.degrees(math.acos(float(np.clip(up[2] / g, -1, 1)))))
        if not self.get_parameter('tilt_from_imu').value or not 8.8 < g < 10.8:
            return                          # (not applied / not gravity alone: no tilt from it)
        up /= g
        # still? the means of 8 slices of the window all point the same way (within 2 deg).
        # Not the raw samples' spread: the head's spin shakes them, and averages out
        parts = np.array([c.mean(axis=0) for c in np.array_split(acc, 8)])
        parts /= np.linalg.norm(parts, axis=1, keepdims=True)
        if math.degrees(math.acos(float(np.clip((parts @ up).min(), -1, 1)))) > 2.0:
            return                          # turning (re-tilted, the base, a jolt): not this window
        if self.up is not None and math.degrees(math.acos(float(np.clip(up @ self.up, -1, 1)))) < 3.0:
            return
        R, lean = mount_from_up(up, self.heading)
        measured = self.measured_R()
        off = math.degrees(math.acos(float(np.clip((np.trace(measured.T @ R) - 1) / 2, -1, 1))))
        first = self.up is None
        with self.lock:
            self.R, self.up, self.lean, self.mount_from = R, up, lean, 'imu'
            self.history, self.reach_hist, self.check_hist = [], [], []      # (in the old mount)
        self.static.sendTransform(self.mount_tf())
        self._housing_done = False
        self.add_housing()
        r, pch, y = rpy_of(R)
        text = (f'{"the" if first else "re-tilted: the"} L2\'s tilt from its IMU: its dome {lean:.1f} '
                f'deg from straight up; mount rpy ({r:.1f}, {pch:.1f}, {y:.1f}) deg'
                + (f' -- {off:.0f} deg from the measured mount '
                   f'{list(self.get_parameter("mount_rpy_deg").value)}' if off > 10.0 else '')
                + (' -- it looks UP: it sees nothing below itself (the walls stay) until it is '
                   'tilted down' if lean < float(self.get_parameter('min_lean_deg').value) else ''))
        # (rclpy: one call site, one severity)
        if off > 10.0:
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)

    # -- the LiDAR's cloud -------------------------------------------------------------

    def on_diagnostics(self, msg):
        """The L2 driver's report of its window: 'dirty_percentage'."""
        for st in msg.status:
            if 'l2_driver' not in st.name:
                continue
            for kv in st.values:
                if kv.key == 'dirty_percentage':
                    try:
                        self.dirty = (float(kv.value), time.monotonic())
                    except ValueError:
                        pass

    def filled_depth(self, depth):
        """fill_holes(depth, camera_first_fill_px), once per depth picture."""
        px = int(self.get_parameter('camera_first_fill_px').value)
        cached = getattr(self, '_filled', (None, None, None))
        if cached[0] is depth and cached[1] == px:
            return cached[2]
        try:
            out = fill_holes(depth, px)
        except Exception as exc:            # noqa: BLE001 - no scipy: as before
            self.get_logger().warn(f'camera hole fill off ({exc})', throttle_duration_sec=60.0)
            out = depth
        self._filled = (depth, px, out)
        return out

    def depth_view(self):
        """(depth in metres, K, R, t) of the D455's latest depth picture -- R, t take
        mount_parent points into its optical frame -- or None: none fresh, no tf."""
        with self.lock:
            depth, info = self.latest.get('depth'), self.latest.get('info')
            fresh = time.monotonic() - self.latest.get('depth_t', -1e9) < 2.0
            cached = self._depth_m
        if depth is None or info is None or not fresh:
            return None
        frame = depth.header.frame_id or info.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(frame, self.parent, Time())
        except Exception:                   # noqa: BLE001 - no camera tf yet
            return None
        if cached[0] is depth:
            d = cached[1]
        else:
            dt = np.uint16 if depth.encoding in ('16UC1', 'mono16') else np.float32
            d = np.frombuffer(bytes(depth.data), dtype=dt).reshape(depth.height, depth.width)
            d = d.astype(np.float32) * (0.001 if dt == np.uint16 else 1.0)
            with self.lock:
                self._depth_m = (depth, d)
        tr, q = tf.transform.translation, tf.transform.rotation
        return (d, np.asarray(info.k, float).reshape(3, 3),
                matrix_from_quat(q.x, q.y, q.z, q.w), np.array([tr.x, tr.y, tr.z]))

    def link_tf(self, name):
        """(R, t) taking a robot link's points into mount_parent's frame, now; or None."""
        try:
            tf = self.tf_buffer.lookup_transform(self.parent, name, Time())
        except Exception:                   # noqa: BLE001 - not (yet) in the tf tree
            return None
        tr, q = tf.transform.translation, tf.transform.rotation
        return matrix_from_quat(q.x, q.y, q.z, q.w), np.array([tr.x, tr.y, tr.z])

    def on_cloud(self, msg):
        xyz, _ = cloud_xyz(msg)
        rng = np.linalg.norm(xyz, axis=1)
        trim = self.trim if self.get_parameter('range_trim').value else 0.0
        if trim:                            # its ranges, as the camera corrects them
            xyz = (xyz * (1.0 + trim / np.maximum(rng, 1e-3))[:, None]).astype(np.float32)
            rng = rng + trim
            msg = with_xyz(msg, xyz)
        out = restamp(msg, self.frame)
        R, t = self.R, self.t               # one mount for the whole cloud
        p = xyz @ R.T + t
        near_ok = np.isfinite(p).all(axis=1) & (rng >= float(self.get_parameter('range_min').value))
        now = time.monotonic()
        # fog: the points that do not repeat along their view line
        steady, keys = self.steady(xyz, rng, near_ok, now)
        if not self.get_parameter('steady_filter').value:
            steady = near_ok
        # the robot itself, and anything on the way from the LiDAR to it
        own = np.zeros(len(p), bool)
        if self.body is not None:
            idx = np.where(near_ok)[0]
            on_body, self.body_missing = self.body.mask(p[idx], self.link_tf,
                                                        float(self.get_parameter('body_pad').value))
            own[idx[on_body]] = True
            if on_body.any():
                cells, inv = np.unique(keys[idx[on_body]], return_inverse=True)
                reach = np.zeros(len(cells))
                np.maximum.at(reach, inv, rng[idx[on_body]])
                at = np.searchsorted(cells, keys[idx])
                at[at >= len(cells)] = 0
                same = cells[at] == keys[idx]
                own[idx[same & (rng[idx] < reach[at])]] = True
        real = near_ok & steady & ~own
        # the camera: what it sees through is false; what it sees at all, it shows better
        through = np.zeros(len(p), bool)
        hide = through
        seen_n = 0
        view = self.depth_view() if (self.get_parameter('drop_seen_through').value
                                     or self.get_parameter('camera_first').value) else None
        if view is not None:
            tol = float(self.get_parameter('camera_tol').value)
            through = camera_sees_through(p, view[0], view[1], view[2], view[3], tol)
            seen_n = camera_agreement(p[near_ok], view[0], view[1], view[2], view[3], tol)[0]
            if self.get_parameter('camera_first').value:
                hide = camera_covers(p, self.filled_depth(view[0]), view[1], view[2], view[3],
                                     tol, max(tol, float(self.get_parameter('camera_first_behind').value)),
                                     int(self.get_parameter('camera_first_fill_px').value),
                                     front=float(self.get_parameter('camera_first_fill_front').value))
            else:
                hide = through
            if not self.get_parameter('drop_seen_through').value:
                through = np.zeros(len(p), bool)
                hide = hide if self.get_parameter('camera_first').value else through
        in_work = near_ok & in_box(p, self.box)
        errs = np.zeros(0)
        if view is not None and self.get_parameter('range_trim').value:
            sel = real & in_work
            errs = camera_range_error(p[sel], (xyz[sel] / np.maximum(rng[sel], 1e-3)[:, None]) @ R.T,
                                      view[0], view[1], view[2], view[3])
        keep = real & in_work & ~hide
        # specks: too few neighbours (this scan and the last density_window s)
        if self.get_parameter('density_filter').value and keep.any():
            from scipy.spatial import cKDTree
            span = float(self.get_parameter('density_window').value)
            with self.lock:
                self.dense_hist = [(a, q) for a, q in self.dense_hist if now - a < span]
                past = [q for _a, q in self.dense_hist]
                self.dense_hist.append((now, p[real & ~hide]))
            idx = np.where(real & ~hide)[0]
            pool = np.concatenate(past + [p[idx]]) if past else p[idx]
            counts = np.array([len(c) for c in cKDTree(pool).query_ball_point(
                p[idx], float(self.get_parameter('density_radius').value))]) - 1
            lone = np.zeros(len(p), bool)
            lone[idx[counts < int(self.get_parameter('density_min').value)]] = True
            real = real & ~lone
            keep = keep & ~lone
        reach_n = int((real & ~through & in_box(p, self.reach)).sum())
        # passed on only while the LiDAR is trusted (the last status ok): otherwise nothing
        show = self.trusted
        nothing = np.zeros(len(p), bool)
        self.pub_cloud.publish(select_points(out, real & ~hide if show else nothing))
        if self.pub_fit.get_subscription_count():
            self.pub_fit.publish(select_points(out, real))
        work = select_points(out, keep if show else nothing)
        self.pub_work.publish(work)
        if self.get_parameter('octomap_live').value:
            self.pub_octo.publish(work)
        with self.lock:
            if R is not self.R:
                return                      # the mount changed meanwhile
            self.arrivals = [a for a in self.arrivals if now - a < 5.0] + [now]
            self.last_work_n = int(keep.sum())
            span = float(self.get_parameter('accumulate_s').value)
            self.history = [(a, pts) for a, pts in self.history if now - a < span]
            self.history.append((now, p[keep]))
            # the camera's check judges the LiDAR as it is: unfiltered, but not the robot
            self.check_hist = [(a, pts) for a, pts in self.check_hist if now - a < 1.5]
            self.check_hist.append((now, p[in_work & ~own]))
            self.drop_hist = [x for x in self.drop_hist if now - x[0] < 1.5]
            self.drop_hist.append((now, seen_n, int((near_ok & through).sum())))
            self.filter_hist = [x for x in self.filter_hist if now - x[0] < 1.5]
            self.filter_hist.append((now, int(near_ok.sum()), int((near_ok & ~steady).sum()),
                                     int(own.sum())))
            self.reach_hist = [(a, n) for a, n in self.reach_hist if now - a < 1.5] + [(now, reach_n)]
            self.trim_hist = [x for x in self.trim_hist if now - x[0] < 1.5] + [(now, errs)]

    def status(self):
        now = time.monotonic()
        with self.lock:
            recent = [a for a in self.arrivals if now - a < 3.0]
            rate = len(recent) / 3.0
            age = (now - self.arrivals[-1]) if self.arrivals else None
            work_n = sum(len(pts) for _a, pts in self.history)
            reach_n = sum(n for _a, n in self.reach_hist)
            lean, source, R = self.lean, self.mount_from, self.R
            seen_n = sum(x[1] for x in self.drop_hist)
            through_n = sum(x[2] for x in self.drop_hist)
            dirty = self.dirty
            was_ok = self.trusted
            all_n = sum(x[1] for x in self.filter_hist)
            fog_n = sum(x[2] for x in self.filter_hist)
            own_n = self.filter_hist[-1][3] if self.filter_hist else 0
            errs = np.concatenate([x[1] for x in self.trim_hist]) if self.trim_hist else np.zeros(0)
        gp = lambda k: self.get_parameter(k).value
        # the ranges follow the camera: a share of what is left of the difference, each second
        left = float(np.median(errs)) if len(errs) >= int(gp('range_trim_min_points')) else None
        if gp('range_trim') and left is not None:
            top = float(gp('range_trim_max'))
            self.trim = float(np.clip(self.trim - float(gp('range_trim_gain')) * left, -top, top))
        elif not gp('range_trim'):
            self.trim = 0.0
        rpy = [round(v, 1) for v in rpy_of(R)]
        cam = self.check_against_camera() if gp('check_camera') else None
        self.cam_check = cam
        mount_bad = ''
        if gp('check_camera'):
            if cam is None:
                mount_bad = 'no camera frames to check the LiDAR against'
            elif cam[0] < int(gp('min_in_view')):
                mount_bad = (f'the LiDAR sees almost nothing of what the camera sees ({cam[0]} '
                             f'points): it is not mounted as set (rpy {rpy} deg) -- upright, or '
                             'tilted / turned otherwise')
            elif cam[1] < (float(gp('min_agree')) - (float(gp('agree_slack')) if was_ok else 0.0)) * cam[0]:
                mount_bad = (f"the LiDAR doesn't match the camera ({100 * cam[1] // cam[0]}% of "
                             f'{cam[0]} points agree): it is not mounted as set (rpy {rpy} deg) '
                             '-- upright, or tilted / turned otherwise')
        dirty_pct = dirty[0] if dirty is not None and now - dirty[1] < 30.0 else None
        dirty_bad = ''
        if dirty_pct is not None and dirty_pct > (float(gp('max_dirty_pct'))
                                                   + (float(gp('dirty_slack')) if was_ok else 0.0)):
            dirty_bad = (f"the L2's window is dirty ({dirty_pct:.0f}% of it, by its own report; "
                         f"the limit is {float(gp('max_dirty_pct')):.0f}%): it scatters its light "
                         'into false points in front of the real surfaces -- clean the dome '
                         '(no film, no smears, nothing touching it)')
        ok = (rate >= float(gp('ok_rate')) and not dirty_bad and not mount_bad
              and reach_n >= int(gp('ok_points')))
        with self.lock:
            self.trusted = bool(ok)         # only then are its points passed on
        looks_up = lean is not None and lean < float(gp('min_lean_deg'))
        why = ('' if ok else
               'no clouds from the L2 driver (it takes ~25 s to spin up)' if not recent else
               'clouds too slow' if rate < float(gp('ok_rate')) else
               f'the L2 looks up (its dome {lean:.0f} deg from vertical): it sees nothing below '
               'itself -- tilt it down' if looks_up else
               dirty_bad if dirty_bad else
               mount_bad if mount_bad else
               'too few points below it, where the arms work')
        imu = self.imu_say
        return {'ok': bool(ok), 'rate_hz': round(rate, 1),
                'age_s': None if age is None else round(age, 1),
                'work_points': int(work_n), 'reach_points': int(reach_n), 'frame': self.frame,
                'lean_deg': None if lean is None else round(lean, 1),
                'mount_rpy_deg': rpy, 'mount_from': source,
                'mount_spin_deg': float(self.get_parameter('mount_spin_deg').value),
                'camera_in_view': None if cam is None else cam[0],
                'camera_agree_pct': None if not cam or not cam[0] else round(100.0 * cam[1] / cam[0]),
                'imu_g': None if imu is None else round(imu[0], 2),
                'imu_lean_deg': None if imu is None else round(imu[1], 1),
                'dirty_pct': None if dirty_pct is None else round(dirty_pct, 1),
                'seen_through_pct': None if not seen_n else round(100.0 * through_n / seen_n),
                'shown': bool(ok),
                'unsteady_pct': None if not all_n else round(100.0 * fog_n / all_n),
                'robot_points': int(own_n),
                'range_trim_cm': round(100.0 * self.trim, 1),
                'range_left_cm': None if left is None else round(100.0 * left, 1),
                'self_filter': ('on' if self.body is not None and not self.body_missing else
                                f'no tf for {self.body_missing} of the robot\'s links' if self.body
                                is not None else f'OFF: {self.body_why}'),
                'why': why, 'time': time.time()}

    def publish_status(self):
        self.pub_status.publish(String(data=json.dumps(self.status())))

    # -- the merged cloud --------------------------------------------------------------

    def _keep(self, kind, msg):
        with self.lock:
            self.latest[kind] = msg
            self.latest[kind + '_t'] = time.monotonic()

    def check_against_camera(self):
        """(in_view, agree) of the LiDAR's recent work points (before the camera's
        filter) against the D455's latest depth (see camera_agreement), or None: no
        fresh frame, or no camera tf."""
        view = self.depth_view()
        if view is None:
            return None
        with self.lock:
            lidar = [pts for _a, pts in self.check_hist]
        pts = np.concatenate(lidar) if lidar else np.zeros((0, 3))
        return camera_agreement(pts, view[0], view[1], view[2], view[3],
                                float(self.get_parameter('camera_tol').value))

    def camera_points(self):
        """(N, 3) points of the D455 (its depth aligned to colour, every
        camera_stride-th pixel) in the parent frame, and their colours; or None."""
        with self.lock:
            depth, color, info = (self.latest.get(k) for k in ('depth', 'color', 'info'))
        if depth is None or info is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(self.parent, depth.header.frame_id or
                                                 info.header.frame_id, Time())
        except Exception:                   # noqa: BLE001 - no camera tf yet
            return None
        s = max(1, int(self.get_parameter('camera_stride').value))
        dt = np.uint16 if depth.encoding in ('16UC1', 'mono16') else np.float32
        d = np.frombuffer(bytes(depth.data), dtype=dt).reshape(depth.height, depth.width)
        d = d[::s, ::s].astype(np.float32) * (0.001 if dt == np.uint16 else 1.0)
        k = np.asarray(info.k, float).reshape(3, 3)
        vs, us = np.mgrid[0:depth.height:s, 0:depth.width:s]
        ok = (d > 0.1) & (d < 2.0)
        z = d[ok]
        x = (us[ok] - k[0, 2]) * z / k[0, 0]
        y = (vs[ok] - k[1, 2]) * z / k[1, 1]
        cam = np.stack([x, y, z], axis=1)
        tr, q = tf.transform.translation, tf.transform.rotation
        R = matrix_from_quat(q.x, q.y, q.z, q.w)
        pts = cam @ R.T + np.array([tr.x, tr.y, tr.z])
        mine = (self.body.mask(pts, self.link_tf, float(self.get_parameter('body_pad_camera').value))[0]
                if self.body is not None else np.zeros(len(pts), bool))
        rgb = None
        if color is not None and color.height == depth.height and color.width == depth.width:
            ch = 3 if color.encoding in ('rgb8', 'bgr8') else None
            if ch:
                img = np.frombuffer(bytes(color.data), np.uint8).reshape(color.height, color.width, 3)
                img = img[::s, ::s][ok][~mine]
                rgb = img[:, ::-1] if color.encoding == 'bgr8' else img
        pts = pts[~mine]
        if rgb is None:
            rgb = np.full((len(pts), 3), 200, np.uint8)
        return pts.astype(np.float32), rgb

    def publish_merged(self):
        if self.pub_merged.get_subscription_count() == 0:
            return                          # (nobody looking: no work)
        with self.lock:
            lidar = [p for _t, p in self.history] if self.trusted else []
        parts, colours = [], []
        pts = np.concatenate(lidar).astype(np.float32) if lidar else np.zeros((0, 3), np.float32)
        if len(pts):                        # (every point may have been the camera's to show)
            parts.append(pts)
            # the LiDAR's points coloured by height: blue low, yellow high
            h = np.clip((pts[:, 2] - pts[:, 2].min()) / max(1e-3, np.ptp(pts[:, 2])), 0, 1)
            colours.append(np.stack([255 * h, 160 + 95 * h, 255 * (1 - h)], axis=1).astype(np.uint8))
        cam = self.camera_points()
        if cam is not None:
            parts.append(cam[0])
            colours.append(cam[1])
        if not parts:
            return
        self.pub_merged.publish(make_cloud(np.concatenate(parts), self.parent,
                                           self.get_clock().now().to_msg(),
                                           np.concatenate(colours)))

    # -- the housing, in the planning scene ------------------------------------------

    def _first_housing(self):
        if not getattr(self, '_housing_done', False):
            self.add_housing()

    def housing_pose(self):
        """(t, R) of the L2 itself for its box: housing_xyz / housing_rpy_deg /
        housing_spin_deg when set (where it physically is), else the mount."""
        try:
            xyz = [float(v) for v in self.get_parameter('housing_xyz').value]
            rpy = [float(v) for v in self.get_parameter('housing_rpy_deg').value]
            spin = float(self.get_parameter('housing_spin_deg').value)
        except (TypeError, ValueError):
            xyz = []
        if len(xyz) != 3 or len(rpy) != 3:
            return self.t, self.R
        return (np.array(xyz), rpy_matrix(*[math.radians(v) for v in rpy]) @ rot_z(spin))

    def add_housing(self):
        """A box around the LiDAR in the planning scene (MoveIt; cuMotion takes the
        scene's objects with every plan), so no move swings an arm into it. Again
        every 30 s: a respawned move_group has forgotten it."""
        if not self.scene.service_is_ready():
            return
        size = [float(v) for v in self.get_parameter('housing_size').value]
        t, R = self.housing_pose()
        obj = CollisionObject()
        obj.header.frame_id = self.parent
        obj.id = HOUSING_ID
        obj.operation = CollisionObject.ADD
        obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=size))
        pose = Pose()
        # the box's centre: half its height up the LiDAR's own Z from its origin
        c = t + R @ np.array([0.0, 0.0, size[2] / 2.0])
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in c)
        q = quat_from_matrix(R)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = \
            (float(v) for v in q)
        obj.primitive_poses.append(pose)
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = [obj]
        future = self.scene.call_async(ApplyPlanningScene.Request(scene=scene))
        future.add_done_callback(lambda f: setattr(self, '_housing_done', bool(
            f.result() is not None and f.result().success)))


def main(args=None):
    rclpy.init(args=args)
    node = LidarBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
