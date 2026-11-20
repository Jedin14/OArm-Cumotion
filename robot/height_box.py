#!/usr/bin/env python3
"""A box under the arms that rides up and down with /c60000004/vertical_distance
(2026-11-19, the user): "a variable obstacle box that goes up and down based on the data
from topic /c60000004/vertical_distance where it returns a value like x cm, the box is
actual 152-(x+10) cm to the bottom of the arm ... 60 cm x 20 cm x 28 cm where 28 cm is
the depth and 20 cm is 10 cm to the front and 10 cm to the back, similarly 60 cm is 30 to
left and 30 to right ... if no value is being read then use default value of 100 cm from
arm centre to the box".

    gap (arm centre -> the box's top)  = total_cm - (x + offset_cm)       (152 - (x + 10))
                                       = default_gap_cm (100) while no value comes in
    the box: size [x 0.20, y 0.60, z 0.28] m, its top `gap` below arm_centre, centred under
    it (10 cm in front / behind, 30 cm left / right), 28 cm deep downward.

arm_centre = the middle of the two arms' bases (v10.urdf.xacro: x 0.0126, y +0.0572 /
-0.0448, z 0.7344 in openarm_body_link0) -> (0.0126, 0.0062, 0.7344).

The stand is not in the way (the user, 2026-11-19: "ignore the aluminium extrusion and
all, just arms, camera and lidar collision must be checked"): the box goes up round the
stand's column -- it is sent as up to four pieces round a hole where the stand is (its
URDF collision shape over the box's height, + stand_pad), so MoveIt never finds the stand
touching it, and the arms still do. (Not by allowing the pair in MoveIt's collision
matrix: the app rewrites that matrix itself, app/ctm/robot.py. cuMotion needs nothing:
its override turns the stand's spheres off and models the stand as a world mesh, and
world objects are not checked against each other.)

What holds it down: the camera (camera_link's body box), the LiDAR (l2_lidar_body, the
Unitree L2's 75 x 75 x 65 mm) and the two arm bases -- each one standing over the box's
footprint keeps its top at least protect_margin below it (the camera, the lowest: top at
most ~0.55 m, a gap of >= ~18.5 cm, x <= ~123.5 cm). A value asking for more is held there
(clamped). Their places come from TF; until TF has them, max_top_z (-0.01, under the
stand's foot) as before.

An arm IN the box is not moved for: MoveIt / cuMotion then see a start state in collision
and will not plan -- the status names the links (arm_in_box), and it is warned once.

The topic's type is whatever is published there (looked up when it appears): a number
(std_msgs Float32/Float64/Int*; taken as cm), a string like "37.5 cm" / "375 mm" /
"0.375 m" (no unit: cm), or sensor_msgs/Range (m; the device's type, 2026-11-19 -- +-inf or
outside min_range..max_range is no reading). No value for timeout_s: the default.

    -> the planning scene (MoveIt; cuMotion takes its objects with every plan): object
       `height_box` (its pieces), see-through (rgba), sent when its top moves by >=
       move_tol, again every 30 s (a respawned move_group has forgotten it)
    -> /height_box/status  JSON, latched, 1 Hz: source live/default, x_cm, gap_cm, top_z,
       clamped, held_by, limit_top_z, pieces, arm_in_box, age_s
"""
import json
import math
import os
import re
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformListener

OBJECT_ID = 'height_box'
_UNIT = {'mm': 0.1, 'cm': 1.0, 'm': 100.0}
STAND = 'openarm_body_link0'
# what keeps the box down: frame -> (lowest corner, highest corner) of its body, in that frame
PROTECT = {
    'camera_link': ((-0.02145, -0.1095, -0.0145), (0.00455, 0.0145, 0.0145)),   # URDF box
    'l2_lidar_body': ((-0.0375, -0.0375, 0.0), (0.0375, 0.0375, 0.065)),        # L2 manual
}
PROTECT_MESH = ['openarm_left_link0', 'openarm_right_link0']   # the arm bases (URDF shapes)
ARM_LINKS = [f'openarm_{s}_{n}' for s in ('left', 'right')
             for n in ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand',
                       'left_finger', 'right_finger']]


def value_cm(msg):
    """The distance a message carries, in cm, or None."""
    if hasattr(msg, 'range') and isinstance(getattr(msg, 'range'), float):   # sensor_msgs/Range
        r = float(msg.range)
        lo, hi = float(getattr(msg, 'min_range', 0.0)), float(getattr(msg, 'max_range', 0.0))
        if not math.isfinite(r) or (hi > lo and not lo <= r <= hi):
            return None                     # REP 117: +-inf / outside its limits = no reading
        return r * 100.0
    v = getattr(msg, 'data', None)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return parse_cm(v)
    return None


def parse_cm(text):
    """'37.5 cm' / '375mm' / '0.375 m' / '37.5' (cm) -> 37.5, else None."""
    m = re.search(r'([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m)?\b', text.strip().lower())
    if not m:
        return None
    return float(m.group(1)) * _UNIT[m.group(2) or 'cm']


def box_pose(gap_m, centre, size, max_top_z):
    """(centre xyz of the box, its top z, clamped?) for a top `gap_m` below `centre`."""
    top = float(centre[2] - gap_m)
    clamped = bool(top > max_top_z)
    if clamped:
        top = float(max_top_z)
    return (centre[0], centre[1], top - size[2] / 2.0), top, clamped


def split_round(lo, hi, hole_lo, hole_hi, min_t=0.002):
    """The box lo..hi (3) less the vertical hole hole_lo..hole_hi (2: x, y; the box's full
    height): [(lo, hi)] of up to four pieces -- in front of / behind the hole (the full
    width), then left / right of it. No hole (None, or outside the box): the box whole."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if hole_lo is None:
        return [(lo, hi)]
    hx0, hy0 = max(hole_lo[0], lo[0]), max(hole_lo[1], lo[1])
    hx1, hy1 = min(hole_hi[0], hi[0]), min(hole_hi[1], hi[1])
    if hx1 <= hx0 or hy1 <= hy0:
        return [(lo, hi)]
    out = []
    for a, b in (((hx1, lo[1]), (hi[0], hi[1])),        # in front of the hole
                 ((lo[0], lo[1]), (hx0, hi[1])),        # behind it
                 ((hx0, hy1), (hx1, hi[1])),            # left of it
                 ((hx0, lo[1]), (hx1, hy0))):           # right of it
        if b[0] - a[0] >= min_t and b[1] - a[1] >= min_t:
            out.append((np.array([a[0], a[1], lo[2]]), np.array([b[0], b[1], hi[2]])))
    return out


def stand_hole(z0, z1, mesh_pts, spheres, pad):
    """(xy lowest, xy highest) of the stand between heights z0..z1 -- its mesh points
    (N, 3) and its collision spheres [(centre, r)] that reach into that band -- grown by
    pad, or (None, None) when nothing of it is there."""
    xs = []
    if mesh_pts is not None and len(mesh_pts):
        m = mesh_pts[(mesh_pts[:, 2] >= z0 - pad) & (mesh_pts[:, 2] <= z1 + pad)]
        if len(m):
            xs += [m[:, :2].min(0), m[:, :2].max(0)]
    for c, r in spheres:
        if c[2] + r >= z0 - pad and c[2] - r <= z1 + pad:
            xs += [np.array(c[:2]) - r, np.array(c[:2]) + r]
    if not xs:
        return None, None
    xs = np.array(xs)
    return xs.min(0) - pad, xs.max(0) + pad


def stand_layers(z0, z1, mesh_pts, pad, step=0.01, tol=0.02):
    """The band z0..z1 cut where the stand's cross-section changes (1 cm slices, merged
    while their outline stays within tol): [(za, zb, hole_lo, hole_hi)] -- so the hole is
    the stand's foot only where the foot is, its column above it (tol 0.02: ~28 pieces at
    the foot, 91% of the box kept)."""
    edges = list(np.arange(z0, z1, step)) + [z1]
    layers = []
    for za, zb in zip(edges[:-1], edges[1:]):
        if zb - za < 1e-6:
            continue
        lo, hi = stand_hole(za, zb, mesh_pts, [], pad)
        if layers:
            pa, pb, plo, phi = layers[-1]
            same = (lo is None and plo is None) or (
                lo is not None and plo is not None
                and np.abs(lo - plo).max() <= tol and np.abs(hi - phi).max() <= tol)
            if same:
                layers[-1] = (pa, zb, plo if lo is None else np.minimum(plo, lo),
                              phi if hi is None else np.maximum(phi, hi))
                continue
        layers.append((za, zb, lo, hi))
    return layers


def limit_top(parts, fp_lo, fp_hi, margin):
    """The highest the box's top may go: below the lowest of `parts` {name: (lo, hi)} (3,
    world AABBs) that stand over its footprint fp_lo..fp_hi (2, grown by margin).
    -> (z or inf, the part)."""
    best, who = math.inf, None
    for name, (lo, hi) in parts.items():
        over = (lo[0] <= fp_hi[0] + margin and hi[0] >= fp_lo[0] - margin
                and lo[1] <= fp_hi[1] + margin and hi[1] >= fp_lo[1] - margin)
        if over and lo[2] - margin < best:
            best, who = float(lo[2] - margin), name
    return best, who


def corners_aabb(R, t, lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    p = c @ np.asarray(R).T + np.asarray(t)
    return p.min(0), p.max(0)


def tf_matrix(tf):
    q, v = tf.rotation, tf.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R, np.array([v.x, v.y, v.z])


class HeightBox(Node):
    def __init__(self):
        super().__init__('height_box')
        p = self.declare_parameter
        p('topic', '/c60000004/vertical_distance')
        p('total_cm', 152.0)
        p('offset_cm', 10.0)
        p('default_gap_cm', 100.0)       # arm centre -> the box's top, while no value comes in
        p('timeout_s', 3.0)              # no value this long: the default
        p('size', [0.20, 0.60, 0.28])    # m: x front-back, y left-right, z its depth
        p('arm_centre', [0.0126, 0.0062, 0.7344])
        p('frame', 'openarm_body_link0')
        p('max_top_z', -0.01)            # m: the limit until TF has the camera / LiDAR / bases
        p('protect_margin', 0.015)       # m: the box's top this far below them
        p('stand_pad', 0.005)            # m: round the stand, the hole this much bigger
        p('urdf_path', '/workspaces/isaac_ros-dev/robot/openarm.urdf')
        p('move_tol', 0.005)             # m: re-send when its top moves this much
        p('rgba', [0.2, 0.55, 1.0, 0.35])
        self.x_cm = None
        self.t_value = 0.0
        self.sub = None
        self.sub_type = None
        self.sent_key = None
        self.sent_t = 0.0
        self.warned = None
        self.warned_type = None
        self.warned_arm = None
        self.last = None
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.body = self.load_robot()
        self.scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_status = self.create_publisher(String, '/height_box/status', latched)
        self.create_timer(2.0, self.find_topic)
        self.create_timer(0.2, self.update)
        self.create_timer(1.0, self.publish_status)
        self.find_topic()
        self.get_logger().info(
            f"box {self.get_parameter('size').value} m under the arms, gap = "
            f"{self.get_parameter('total_cm').value:g} - (x + {self.get_parameter('offset_cm').value:g}) cm "
            f"from {self.get_parameter('topic').value} (default {self.get_parameter('default_gap_cm').value:g} cm); "
            'round the stand, held under the camera / LiDAR / arm bases')

    def load_robot(self):
        """The URDF's collision shapes (lidar_bridge.RobotBody), or None -- then no hole and
        the old limit (under the stand's foot)."""
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from lidar_bridge import RobotBody
            return RobotBody(self.get_parameter('urdf_path').value, spacing=0.004)
        except Exception as e:              # noqa: BLE001 - no trimesh / URDF
            self.get_logger().error(f'robot shapes not loaded ({e}): the box stays under the '
                                    'stand\'s foot')
            return None

    # -- the value -------------------------------------------------------------------

    def find_topic(self):
        """Subscribe once the topic exists, with the type it is published as -- and again
        when that type changes (the stand-in went from text to sensor_msgs/Range, 11-19)."""
        name = self.get_parameter('topic').value
        # the publishers' types (not the topic's: our own subscription would keep the old one)
        types = []
        for info in self.get_publishers_info_by_topic(name):
            if info.topic_type not in types:
                types.append(info.topic_type)
        if self.sub is not None:
            if not types or self.sub_type in types:
                return
            self.get_logger().info(f'{name}: now {types[0]} (was {self.sub_type}): resubscribing')
            self.destroy_subscription(self.sub)
            self.sub = None
        if not types:
            return
        from rosidl_runtime_py.utilities import get_message
        try:
            cls = get_message(types[0])
        except (AttributeError, ModuleNotFoundError, ValueError) as e:
            if self.warned_type != types[0]:
                self.warned_type = types[0]
                self.get_logger().error(f'{name}: type {types[0]} not available here ({e}); default used')
            return
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)  # takes either kind
        self.sub = self.create_subscription(cls, name, self.on_value, qos)
        self.sub_type = types[0]
        self.get_logger().info(f'{name}: {types[0]}')

    def on_value(self, msg):
        v = value_cm(msg)
        if v is None or not math.isfinite(v):
            return
        self.x_cm = v
        self.t_value = time.monotonic()

    def gap(self):
        """(gap m, source, x cm or None)."""
        g = self.get_parameter
        fresh = self.x_cm is not None and time.monotonic() - self.t_value <= float(g('timeout_s').value)
        if not fresh:
            return float(g('default_gap_cm').value) / 100.0, 'default', None
        return (float(g('total_cm').value) - (self.x_cm + float(g('offset_cm').value))) / 100.0, \
            'live', self.x_cm

    # -- the robot, from TF ------------------------------------------------------------

    def pose_of(self, frame):
        try:
            return tf_matrix(self.tf.lookup_transform(self.get_parameter('frame').value, frame,
                                                      Time()).transform)
        except Exception:                   # noqa: BLE001 - not on TF (yet)
            return None

    def link_points(self, name):
        if self.body is None or name not in self.body.links:
            return None
        pose = self.pose_of(name)
        if pose is None:
            return None
        R, t = pose
        return np.asarray(self.body.links[name][0].data) @ R.T + t

    def protected(self):
        """{name: (lo, hi)} world AABBs of the camera, the LiDAR, the arm bases (those on TF)."""
        out = {}
        for frame, (lo, hi) in PROTECT.items():
            pose = self.pose_of(frame)
            if pose is not None:
                out[frame] = corners_aabb(*pose, lo, hi)
        for name in PROTECT_MESH:
            pts = self.link_points(name)
            if pts is not None and len(pts):
                out[name] = (pts.min(0), pts.max(0))
        return out

    # -- the box -----------------------------------------------------------------------

    def current(self):
        g = self.get_parameter
        gap, source, x = self.gap()
        size = np.array([float(v) for v in g('size').value])
        centre = np.array([float(v) for v in g('arm_centre').value])
        fp_lo, fp_hi = centre[:2] - size[:2] / 2, centre[:2] + size[:2] / 2
        parts = self.protected()
        stand = self.link_points(STAND) if self.body is not None else None
        if 'camera_link' in parts and stand is not None:
            limit, held_by = limit_top(parts, fp_lo, fp_hi, float(g('protect_margin').value))
        else:                               # TF not there yet: under the stand's foot
            limit, held_by = float(g('max_top_z').value), 'stand (no TF yet)'
            stand = None
        xyz, top, clamped = box_pose(gap, centre, size, limit)
        lo = np.array([fp_lo[0], fp_lo[1], top - size[2]])
        hi = np.array([fp_hi[0], fp_hi[1], top])
        if stand is None:
            layers = [(lo[2], hi[2], None, None)]
        else:
            layers = stand_layers(lo[2], hi[2], stand, float(g('stand_pad').value))
        pieces, holes = [], []
        for za, zb, hlo, hhi in layers:
            pieces += split_round(np.r_[lo[:2], za], np.r_[hi[:2], zb], hlo, hhi)
            if hlo is not None:
                holes.append([round(float(za), 3), round(float(zb), 3)]
                             + np.round(np.r_[hlo, hhi], 3).tolist())
        return dict(gap=gap, source=source, x=x, top=top, clamped=clamped, limit=limit,
                    held_by=held_by if clamped else None, pieces=pieces, hole=holes or None)

    def arms_in(self, pieces):
        """The arm links with a point inside the box now."""
        hits = []
        for name in ARM_LINKS:
            pts = self.link_points(name)
            if pts is None or not len(pts):
                continue
            for lo, hi in pieces:
                if ((pts >= lo) & (pts <= hi)).all(axis=1).any():
                    hits.append(name)
                    break
        return hits

    def update(self):
        if not self.scene.service_is_ready():
            return
        c = self.current()
        self.last = c
        now = time.monotonic()
        key = (round(c['top'] / float(self.get_parameter('move_tol').value)), len(c['pieces']),
               str(c['hole']))
        if key == self.sent_key and now - self.sent_t < 30.0:
            return
        if c['clamped'] and self.warned != c['held_by']:
            self.warned = c['held_by']
            self.get_logger().warning(
                f"gap {100 * c['gap']:.1f} cm would put the box into the {c['held_by']}: held "
                f"with its top at z {c['top']:.3f}")
        elif not c['clamped']:
            self.warned = None
        obj = CollisionObject()
        obj.header.frame_id = self.get_parameter('frame').value
        obj.id = OBJECT_ID
        obj.operation = CollisionObject.ADD
        for lo, hi in c['pieces']:
            obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX,
                                                 dimensions=[float(v) for v in hi - lo]))
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in (lo + hi) / 2)
            pose.orientation.w = 1.0
            obj.primitive_poses.append(pose)
        scene = PlanningScene(is_diff=True)
        # (a re-ADD of an existing id replaces its shapes: the pieces change with the hole)
        scene.world.collision_objects = [obj]
        rgba = [float(v) for v in self.get_parameter('rgba').value]
        if len(rgba) == 4:
            scene.object_colors = [ObjectColor(id=OBJECT_ID, color=ColorRGBA(
                r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3]))]
        self.sent_key, self.sent_t = key, now
        self.scene.call_async(ApplyPlanningScene.Request(scene=scene))

    def publish_status(self):
        c = self.last or self.current()
        arms = self.arms_in(c['pieces'])
        if arms and self.warned_arm != arms:
            self.get_logger().warning(f'arm links IN the box (no plan will start from here): {arms}')
        self.warned_arm = arms
        self.pub_status.publish(String(data=json.dumps({
            'source': c['source'], 'x_cm': None if c['x'] is None else round(c['x'], 2),
            'gap_cm': round(100 * c['gap'], 2), 'top_z': round(c['top'], 4),
            'clamped': c['clamped'], 'held_by': c['held_by'],
            'limit_top_z': None if not math.isfinite(c['limit']) else round(c['limit'], 4),
            'pieces': len(c['pieces']), 'stand_holes': c['hole'],   # [z lo, z hi, x lo, y lo, x hi, y hi]
            'arm_in_box': arms,
            'subscribed': self.sub is not None,
            'age_s': None if self.x_cm is None else round(time.monotonic() - self.t_value, 2)})))


def main(args=None):
    rclpy.init(args=args)
    node = HeightBox()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
