#!/usr/bin/env python3
"""A see-through guard round the D455 (2026-11-19, the user: "similar to how it is for the
4d lidar create a translucent box around the 3d depth camera such that it doesn't affect
the working but keeps a safety barrier so the arm doesn't hit or scratch the camera").

Why plates, not one box: the camera is half sunk into the body column in the URDF (the
column's front is level with the camera's face, the chest overhangs 15 mm above it), and
the camera itself is a robot link (camera_link) to MoveIt. A box over it would sit IN the
robot -- every start state "in collision", nothing would move. So the guard is a shell of
plates a few mm off the camera, only where the camera is exposed:

    front   20 mm deep, 7 mm off its face, 16 mm past each end
    left / right   15 mm, 4 mm off its ends
    top     over its front lip (the chest is above the rest)
    bottom  under its two ends, which stand out past the column

(plates: [x min, x max, y min, y max, z min, z max] m per plate, in the frame of the
camera's body box -- camera_link moved by box_offset, the URDF's collision box centre:
x out of the lens, y to its left, z up.)

Checked offline (scratchpad cam_guard_eval.py): >= 5.4 mm clear of the body, 67 mm of the
arm bases, 4 mm of the camera. Of the 455 poses the arms really went to it blocks 12 --
all left-arm approaches / ways out that passed within 8-27 mm of the camera (the forearm
coming down past its left end): the scratching this is for.

It follows the camera (TF openarm_body_link0 -> camera_link, read every 2 s: sensor_align
can move it), and before each send checks every plate against the body and the arm bases
(the URDF's collision shapes): a plate within body_clear of them is left out and said, so
the guard can never put the robot in collision with itself.

    -> the planning scene (MoveIt; cuMotion takes its objects with every plan): object
       `camera_guard`, see-through (rgba); again every 30 s (a respawned move_group)
    -> /camera_guard/status  JSON, latched: plates sent / left out, camera position
"""
import json
import math
import os
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_bridge import RobotBody, quat_from_matrix  # noqa: E402

OBJECT_ID = 'camera_guard'
PLATE_NAMES = ['front', 'left', 'right', 'top', 'bottom left', 'bottom right']
PLATES = [
    [0.020, 0.040, -0.081, 0.081, -0.0285, 0.026],      # front
    [-0.013, 0.040, 0.066, 0.081, -0.0285, 0.026],      # left end
    [-0.013, 0.040, -0.081, -0.066, -0.0285, 0.026],    # right end
    [0.016, 0.040, -0.081, 0.081, 0.0185, 0.026],       # top, over the front lip
    [-0.013, 0.040, 0.016, 0.081, -0.0285, -0.0185],    # under the left end
    [-0.013, 0.040, -0.081, -0.057, -0.0285, -0.0185],  # under the right end
]
FIXED_LINKS = ['openarm_body_link0', 'openarm_left_link0', 'openarm_right_link0']


def tf_matrix(tf):
    """(R, t) of a geometry_msgs Transform."""
    q, v = tf.rotation, tf.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R, np.array([v.x, v.y, v.z])


def box_distance(pts, lo, hi):
    """Smallest distance of pts (N, 3; the box's frame) to the box lo..hi (< 0: inside)."""
    if not len(pts):
        return math.inf
    c, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    q = np.abs(pts - c) - half
    out = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = (q <= 0).all(axis=1)
    return float(np.where(inside, np.max(q, axis=1), out).min())


def plates_clear(plates, R, t, fixed_pts, clear):
    """[(index, distance)] of the plates (box frame R, t in the parent) and whether each is
    at least `clear` from fixed_pts (M, 3; parent frame)."""
    local = (np.asarray(fixed_pts, float) - t) @ R if len(fixed_pts) else np.zeros((0, 3))
    out = []
    for i, p in enumerate(plates):
        lo, hi = np.array(p[0::2], float), np.array(p[1::2], float)
        d = box_distance(local, lo, hi)
        out.append((i, d, d >= clear))
    return out


class CameraGuard(Node):
    def __init__(self):
        super().__init__('camera_guard')
        p = self.declare_parameter
        p('plates', [v for plate in PLATES for v in plate])
        p('camera_frame', 'camera_link')
        p('box_offset', [-0.00845, -0.0475, 0.0])   # camera_link -> its collision box centre
        p('frame', 'openarm_body_link0')
        p('body_clear', 0.003)                      # m: a plate nearer the robot is left out
        p('urdf_path', '/workspaces/isaac_ros-dev/robot/openarm.urdf')
        p('rgba', [1.0, 0.55, 0.0, 0.3])
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_status = self.create_publisher(String, '/camera_guard/status', latched)
        self.body = None
        try:
            self.body = RobotBody(self.get_parameter('urdf_path').value, spacing=0.004)
        except Exception as e:              # noqa: BLE001 - no URDF / trimesh: no guard
            self.get_logger().error(f'robot shapes not loaded ({e}): no guard (it could not be '
                                    'checked against the robot)')
        self.sent = None                    # (camera t, R, plates kept) last sent
        self.sent_t = 0.0
        self.added = False
        self.said = None
        self.create_timer(2.0, self.update)

    def plates(self):
        v = [float(x) for x in self.get_parameter('plates').value]
        out = [v[i:i + 6] for i in range(0, len(v) - len(v) % 6, 6)]
        return [p for p in out if p[1] > p[0] and p[3] > p[2] and p[5] > p[4]]

    def fixed_points(self):
        """The body's and the arm bases' collision surface points, in `frame`, or None."""
        frame = self.get_parameter('frame').value
        pts = []
        for name in FIXED_LINKS:
            if name not in self.body.links:
                continue
            try:
                R, t = tf_matrix(self.tf.lookup_transform(frame, name, Time()).transform)
            except Exception:               # noqa: BLE001 - not on TF yet
                return None
            pts.append(np.asarray(self.body.links[name][0].data) @ R.T + t)
        return np.concatenate(pts) if pts else None

    def update(self):
        if self.body is None or not self.scene.service_is_ready():
            return
        g = self.get_parameter
        frame = g('frame').value
        try:
            Rc, tc = tf_matrix(self.tf.lookup_transform(frame, g('camera_frame').value,
                                                        Time()).transform)
        except Exception:                   # noqa: BLE001 - no camera on TF yet
            return
        fixed = self.fixed_points()
        if fixed is None:
            return
        t = tc + Rc @ np.array([float(v) for v in g('box_offset').value])
        plates = self.plates()
        checks = plates_clear(plates, Rc, t, fixed, float(g('body_clear').value))
        kept = [plates[i] for i, _d, ok in checks if ok]
        dropped = [(PLATE_NAMES[i] if i < len(PLATE_NAMES) else str(i), round(1000 * d, 1))
                   for i, d, ok in checks if not ok]
        now = time.monotonic()
        same = (self.sent is not None and np.linalg.norm(self.sent[0] - t) < 0.001
                and np.abs(self.sent[1] - Rc).max() < 0.002 and self.sent[2] == kept)
        if same and now - self.sent_t < 30.0:
            return
        if dropped and self.said != dropped:
            self.get_logger().warning(f'left out (too near the robot itself, mm): {dropped} -- '
                                      'has the camera moved?')
        self.said = dropped
        obj = CollisionObject()
        obj.header.frame_id = frame
        obj.id = OBJECT_ID
        scene = PlanningScene(is_diff=True)
        if kept:
            obj.operation = CollisionObject.ADD
            q = quat_from_matrix(Rc)
            for p in kept:
                lo, hi = np.array(p[0::2]), np.array(p[1::2])
                obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX,
                                                     dimensions=[float(v) for v in hi - lo]))
                pose = Pose()
                c = t + Rc @ ((lo + hi) / 2.0)
                pose.position.x, pose.position.y, pose.position.z = (float(v) for v in c)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = \
                    (float(v) for v in q)
                obj.primitive_poses.append(pose)
            rgba = [float(v) for v in g('rgba').value]
            if len(rgba) == 4:
                scene.object_colors = [ObjectColor(id=OBJECT_ID, color=ColorRGBA(
                    r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3]))]
            self.added = True
        elif self.added:
            obj.operation = CollisionObject.REMOVE
            self.added = False
        else:
            obj = None
        if obj is not None:
            scene.world.collision_objects = [obj]
            self.scene.call_async(ApplyPlanningScene.Request(scene=scene))
        self.sent, self.sent_t = (t, Rc, kept), now
        self.pub_status.publish(String(data=json.dumps({
            'plates_sent': len(kept), 'left_out': dropped,
            'nearest_robot_mm': round(1000 * min((d for _i, d, _ok in checks), default=math.inf), 1)
            if checks else None,
            'camera_xyz': [round(float(v), 4) for v in tc]})))


def main(args=None):
    rclpy.init(args=args)
    node = CameraGuard()
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
