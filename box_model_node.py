#!/usr/bin/env python3
"""BOX MODEL NODE: the trained box detector as a ROS 2 node.

On request it takes a frame from the head camera, finds the box to pick
(box_target_detector/: the topmost, leftmost of the top row) and says where
to touch it and how, as a pick_interfaces/PickTarget:

    position   the point the tool touches, world frame, m
    approach   "top"   - straight down, 2 cm in from the top's near edge
               "front" - level, from the camera side, at the box's centre
    arm        "" (the pick stack chooses), u v bbox confidence num_boxes

    service   /box_model/detect   pick_interfaces/srv/DetectTarget
              (request.approach: "auto" | "top" | "front"; "" = the parameter)
    publishes /pick_target        pick_interfaces/msg/PickTarget, each detection
              /pick_target/image  sensor_msgs/Image, the frame with the boxes drawn

    parameters  approach (auto), confidence (0.5), depth_frames (5),
                top_share (0.3: auto -> top when this share of the box's
                points is its top surface), world_frame (world)

pick_stack.py calls the service on ready_to_pick; any other model can take
its place by serving the same service.

    source native/setup.bash; python3 box_model_node.py
"""

import os
import sys
import threading
import time
import warnings
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros

from pick_interfaces.msg import PickTarget
from pick_interfaces.srv import DetectTarget

from ctm.common import (BASE_FRAME, COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC, decode_image,
                        matrix_from_quat, pixel_ray)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'box_target_detector'))


class BoxModelNode(Node):
    def __init__(self):
        super().__init__('box_model')
        self.declare_parameter('approach', 'auto')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('depth_frames', 5)
        self.declare_parameter('top_share', 0.3)
        self.declare_parameter('world_frame', BASE_FRAME)
        cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.color = self.info = None
        self.depths = deque(maxlen=10)
        self.create_subscription(Image, COLOR_TOPIC, self._on_color, 2, callback_group=cb)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 2, callback_group=cb)
        self.create_subscription(CameraInfo, INFO_TOPIC, self._on_info, 2, callback_group=cb)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_pub = self.create_publisher(PickTarget, '/pick_target', 10)
        self.image_pub = self.create_publisher(Image, '/pick_target/image', 1)
        self.detector = None
        self.detect_lock = threading.Lock()
        self.create_service(DetectTarget, '/box_model/detect', self._on_detect,
                            callback_group=cb)
        threading.Thread(target=self._load, daemon=True).start()

    # -- inputs ---------------------------------------------------------------

    def _on_color(self, msg):
        with self.lock:
            self.color = msg

    def _on_depth(self, msg):
        with self.lock:
            self.depths.append(msg)

    def _on_info(self, msg):
        with self.lock:
            self.info = msg

    def _load(self):
        """The model, once (~2 s, then ~5 ms a frame on the GPU)."""
        try:
            from box_detector import BoxTargetDetector
            t0 = time.monotonic()
            detector = BoxTargetDetector(
                conf=float(self.get_parameter('confidence').value))
            self.detector = detector
            self.get_logger().info(f'box model loaded in {time.monotonic() - t0:.1f} s; '
                                   f'serving /box_model/detect')
        except Exception as exc:                     # noqa: BLE001 - reported per request
            self.load_error = str(exc)
            self.get_logger().error(f'box model failed to load: {exc}')

    def lookup(self, target, source):
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        except Exception:                            # noqa: BLE001
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        m = np.eye(4)
        m[:3, :3] = matrix_from_quat(q.x, q.y, q.z, q.w)
        m[:3, 3] = (t.x, t.y, t.z)
        return m

    def frame(self, count, timeout=3.0):
        """(bgr, depth m (median of the next `count` frames), info, depth msg)."""
        with self.lock:
            self.depths.clear()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if len(self.depths) >= count and self.color is not None and self.info:
                    break
            time.sleep(0.03)
        with self.lock:
            frames, color, info = list(self.depths)[:count], self.color, self.info
        if not frames or color is None or info is None:
            return None
        stack = np.stack([decode_image(m) for m in frames]).astype(np.float32)
        valid = stack > 0
        stack[~valid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            depth = np.nanmedian(stack, axis=0)
        depth[valid.sum(axis=0) < (len(frames) + 1) // 2] = 0.0
        return decode_image(color), np.nan_to_num(depth, nan=0.0), info, frames[-1]

    # -- the request ----------------------------------------------------------

    def _on_detect(self, request, response):
        with self.detect_lock:
            try:
                found, message, target = self.detect(request.approach or
                                                     self.get_parameter('approach').value)
            except Exception as exc:                 # noqa: BLE001 - answered, not raised
                found, message, target = False, f'detection failed: {exc}', PickTarget()
        response.found, response.message, response.target = found, message, target
        self.get_logger().info(message if found else f'no target: {message}')
        return response

    def detect(self, approach):
        if approach not in ('auto', 'top', 'front'):
            return False, f'approach must be auto, top or front, not "{approach}"', PickTarget()
        if self.detector is None:
            return False, getattr(self, 'load_error', 'the box model is still loading'), \
                PickTarget()
        got = self.frame(int(self.get_parameter('depth_frames').value))
        if got is None:
            return False, 'no camera frames - is the D455 streaming?', PickTarget()
        bgr, depth, info, depth_msg = got
        world = self.get_parameter('world_frame').value
        t_wo = self.lookup(world, depth_msg.header.frame_id or info.header.frame_id)
        if t_wo is None:
            return False, f'no tf {world} -> the camera', PickTarget()
        t0 = time.monotonic()
        result = self.detector.detect(bgr)
        took = (time.monotonic() - t0) * 1000
        self._publish_image(bgr, result, depth_msg.header)
        if result is None:
            return False, f'no box in the frame ({took:.0f} ms)', PickTarget()
        points = self.box_points(result, bgr, depth, info, t_wo)
        if points is None:
            return False, 'box found, but too little depth on it (too near or far?)', \
                PickTarget()
        top_z = np.percentile(points[:, 2], 95)
        top = points[points[:, 2] > top_z - 0.01]
        share = len(top) / float(len(points))
        if approach == 'auto':
            approach = 'top' if share >= float(self.get_parameter('top_share').value) \
                else 'front'
        if approach == 'top':
            if len(top) < 15:
                return False, 'TOP: could not find the box\'s top in the depth', PickTarget()
            # 2 cm in from the top's near edge, never past its middle (the far
            # part needs more wrist lean than the arm has, auto_box 2026-10-14)
            position = np.median(top, axis=0)
            position[0] = min(position[0], np.percentile(top[:, 0], 5) + 0.02)
        else:
            position = self.centre_point(result, bgr, depth, info, t_wo)
            if position is None:
                return False, 'FRONT: no depth at the box\'s centre', PickTarget()
        target = PickTarget()
        target.header.frame_id = world
        target.header.stamp = depth_msg.header.stamp
        target.position.x, target.position.y, target.position.z = (float(v) for v in position)
        target.approach = approach
        target.arm = ''
        target.confidence = float(result['confidence'])
        target.u, target.v = float(result['x']), float(result['y'])
        target.bbox = [float(v) for v in result['bbox']]
        target.image_width, target.image_height = int(bgr.shape[1]), int(bgr.shape[0])
        target.num_boxes = int(result['num_boxes'])
        self.target_pub.publish(target)
        return True, (f'{approach} at x={position[0]:+.3f} y={position[1]:+.3f} '
                      f'z={position[2]:+.3f} (box {result["x"]:.0f},{result["y"]:.0f} px, '
                      f'confidence {result["confidence"]:.2f}, top {100 * share:.0f}% of '
                      f'its points, {result["num_boxes"]} boxes, {took:.0f} ms)'), target

    @staticmethod
    def box_points(result, bgr, depth, info, t_wo):
        """World points inside the box's outline (10 % in from it), or None."""
        dh, dw = depth.shape
        sx, sy = dw / float(bgr.shape[1]), dh / float(bgr.shape[0])
        x1, y1, x2, y2 = result['bbox']
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        us = np.arange(int((x1 + mx) * sx), int((x2 - mx) * sx), 2)
        vs = np.arange(int((y1 + my) * sy), int((y2 - my) * sy), 2)
        if not len(us) or not len(vs):
            return None
        uu, vv = np.meshgrid(us, vs)
        z = depth[vv, uu].ravel()
        keep = (z > 0.1) & (z < 4.0) & np.isfinite(z)
        if keep.sum() < 30:
            return None
        cam = np.array([pixel_ray(u / sx, v / sy, info) for u, v in
                        zip(uu.ravel()[keep], vv.ravel()[keep])]) * z[keep][:, None]
        return cam @ t_wo[:3, :3].T + t_wo[:3, 3]

    @staticmethod
    def centre_point(result, bgr, depth, info, t_wo, r=3):
        """World point at the box's centre pixel (median depth around it)."""
        dh, dw = depth.shape
        u = int(result['x'] * dw / float(bgr.shape[1]))
        v = int(result['y'] * dh / float(bgr.shape[0]))
        patch = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        valid = patch[(patch > 0.1) & (patch < 4.0) & np.isfinite(patch)]
        if valid.size < max(3, patch.size // 5):
            return None
        p = pixel_ray(result['x'], result['y'], info) * float(np.median(valid))
        return t_wo[:3, :3] @ p + t_wo[:3, 3]

    def _publish_image(self, bgr, result, header):
        if self.image_pub.get_subscription_count() == 0:
            return
        img = bgr.copy()
        for b in (result or {}).get('boxes', []):
            x1, y1, x2, y2 = (int(round(c)) for c in b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 160), 2)
        if result is not None:
            x1, y1, x2, y2 = (int(round(c)) for c in result['bbox'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.drawMarker(img, (int(result['x']), int(result['y'])), (0, 200, 0),
                           cv2.MARKER_CROSS, 18, 2)
        msg = Image(header=header, height=img.shape[0], width=img.shape[1], encoding='bgr8',
                    is_bigendian=0, step=img.shape[1] * 3, data=img.tobytes())
        self.image_pub.publish(msg)


def main():
    rclpy.init()
    node = BoxModelNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        except BaseException:                        # noqa: BLE001 - a second Ctrl-C
            pass
        rclpy.try_shutdown()
        os._exit(0)                                  # (no waiting on worker threads)


if __name__ == '__main__':
    main()
