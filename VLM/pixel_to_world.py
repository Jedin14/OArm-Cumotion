#!/usr/bin/env python3
"""Read the world coordinate of any point the camera can see.

    VLM/run_in_vlm_env.sh pixel_to_world.py                 # click on the image
    VLM/run_in_vlm_env.sh pixel_to_world.py --pixel 445 245 # one-shot, no window

This exists because place_position is a fixed coordinate, not something that has
to be detected. PaliGemma refuses to find plain boxes and bins reliably -- it is
a pretrained checkpoint, not an open-vocabulary detector -- but the drop point
does not care: point at where you want the object released and read the number.

It is also the quickest way to measure table_z: click bare table next to the
object and take the z.

The maths is imported from vlm_detector_node rather than copied, so what you
measure here is exactly what the detector will produce. That import pulls in
torch, which costs a few seconds at startup but never loads the model.
"""

import argparse
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from vlm_detector_node import VlmDetectorNode, colorize_depth, decode_image, quat_to_rot

PATCH = 5          # half-width of the median depth patch, pixels


class PixelToWorld(Node):

    def __init__(self, target_frame='world', depth_scale=0.001):
        super().__init__('pixel_to_world')
        self.target_frame = target_frame
        self.depth_scale = depth_scale
        self.min_depth, self.max_depth = 0.15, 3.0
        self.color = self.depth = self.info = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._on_color, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw',
            self._on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self._on_info, qos_profile_sensor_data)

    def _on_color(self, msg):
        self.color = msg

    def _on_depth(self, msg):
        self.depth = msg

    def _on_info(self, msg):
        self.info = msg

    def ready(self):
        return None not in (self.color, self.depth, self.info)

    def wait_for_frames(self, timeout=10.0):
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        while rclpy.ok() and not self.ready():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                return False
        return self.wait_for_transform()

    def wait_for_transform(self, timeout=10.0):
        """Spin until the camera->world chain exists.

        /tf_static is latched, but the listener still has to be spun before the
        transform is in the buffer -- and images arrive first, so a naive
        "frames are ready" check reports success a beat too early.
        """
        deadline = self.get_clock().now().nanoseconds * 1e-9 + timeout
        source = self.color.header.frame_id
        while rclpy.ok():
            if self.tf_buffer.can_transform(self.target_frame, source,
                                            rclpy.time.Time()):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                self.get_logger().error(
                    f'no {source} -> {self.target_frame} transform after '
                    f'{timeout:.0f}s; is robot_state_publisher running?')
                return False

    def solve(self, u, v):
        """Pixel -> (world point, camera point, depth). None if depth is missing."""
        depth = decode_image(self.depth)
        h, w = depth.shape
        if not (0 <= u < w and 0 <= v < h):
            return None, None, None

        patch = depth[max(0, v - PATCH):v + PATCH + 1,
                      max(0, u - PATCH):u + PATCH + 1].astype(np.float32)
        patch *= self.depth_scale
        valid = patch[(patch > self.min_depth) & (patch < self.max_depth)]
        if valid.size == 0:
            return None, None, None
        z = float(np.median(valid))

        point_cam = VlmDetectorNode._deproject(self, self.info, u, v, z)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self.color.header.frame_id,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as exc:
            self.get_logger().error(f'no transform to {self.target_frame}: {exc}')
            return None, point_cam, z
        t, q = tf.transform.translation, tf.transform.rotation
        rot = quat_to_rot(q.x, q.y, q.z, q.w)
        return rot @ point_cam + np.array([t.x, t.y, t.z]), point_cam, z


def report(node, u, v):
    world, cam, z = node.solve(u, v)
    if z is None:
        print(f'({u}, {v}): no valid depth in a {2 * PATCH + 1}px patch')
        return None
    if world is None:
        print(f'({u}, {v}): depth {z:.3f} m, camera {np.round(cam, 4)}, no transform')
        return None
    print(f'({u:>4}, {v:>4})  depth {z:.3f} m   world  '
          f'x={world[0]:+.4f}  y={world[1]:+.4f}  z={world[2]:+.4f}')
    return world


def interactive(node):
    print('\nClick a point to read its world coordinate.')
    print('  d = toggle depth view   s = save frame   q / Esc = quit\n')
    clicks = []
    state = {'show_depth': False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            world = report(node, x, y)
            clicks.append((x, y, world))

    window = 'pixel_to_world  (click a point)'
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)
        if not node.ready():
            continue
        colour = decode_image(node.color)
        canvas = (colorize_depth(decode_image(node.depth), node.depth_scale, 0.3, 1.5)
                  if state['show_depth'] else colour.copy())
        for x, y, world in clicks[-6:]:
            cv2.drawMarker(canvas, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
            if world is not None:
                cv2.putText(canvas,
                            f'{world[0]:+.3f} {world[1]:+.3f} {world[2]:+.3f}',
                            (max(2, x - 70), max(12, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                            cv2.LINE_AA)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('d'):
            state['show_depth'] = not state['show_depth']
        if key == ord('s'):
            cv2.imwrite('pixel_to_world_frame.png', canvas)
            print('saved pixel_to_world_frame.png')

    cv2.destroyAllWindows()
    if clicks:
        print('\nclicked points:')
        for x, y, world in clicks:
            if world is not None:
                print(f'  ({x}, {y}) -> [{world[0]:.4f}, {world[1]:.4f}, {world[2]:.4f}]')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pixel', nargs=2, type=int, metavar=('U', 'V'),
                        help='report one pixel and exit')
    parser.add_argument('--target-frame', default='world')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = PixelToWorld(target_frame=args.target_frame)
    try:
        if not node.wait_for_frames():
            print('no camera frames -- is launch_everything running with colour '
                  'and align_depth enabled?', file=sys.stderr)
            return 1
        if args.pixel:
            report(node, args.pixel[0], args.pixel[1])
        else:
            interactive(node)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
