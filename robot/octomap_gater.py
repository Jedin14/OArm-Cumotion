#!/usr/bin/env python3
"""Gate depth frames into the MoveIt octomap, on demand.

In `octomap:=static` mode launch_everything.launch.py remaps the RealSense depth
topic to .../image_rect_raw_in and puts this node in the middle, so the octomap
only updates when something asks it to. That used to be a Tk button only, which
made it unusable from an autonomous loop and fatal on a headless box (tkinter
raises on a missing DISPLAY before rclpy ever spins).

Now there are three ways in, and the button is still one of them:

    ros2 service call /octomap_gater/refresh std_srvs/srv/Trigger
    ros2 topic pub --once /octomap_gater/allow_frames std_msgs/Int32 "{data: 5}"
    the Refresh button, when a display is available

The Unitree L2 LiDAR (2026-11-11): each refresh also collects its work-area
clouds (/lidar/cloud_work, robot/lidar_bridge.py) for lidar_window seconds after
the call -- only clouds from after it: an older one could still have the arm in
it -- and passes them on as one cloud to /lidar/cloud_octomap, which MoveIt's
point-cloud updater (sensors_3d.yaml, l2_lidar) turns into voxels. No LiDAR: no
clouds, nothing passed, the camera's frames as before.

No specks (2026-11-16): each depth frame passed on has its small islands of depth
removed first (cv2.filterSpeckles: regions under speckle_px pixels whose neighbours
differ by more than speckle_diff_mm) -- the flying pixels at the boxes' edges, which
became single voxels in the air and blocked the arm's straight lines in.

Map version (2026-11-17): /octomap_gater/version (std_msgs/UInt32, latched) counts
the map changes this node makes -- up when a refresh starts, and again once its last
depth frame and its LiDAR cloud have been passed on. A plan checked against one map
and flown after a refresh (sensor_align.py refreshes too) is then known to be stale:
the 17:35 pick of 2026-11-16 flew a 17:32 plan after three rebuilds. Only "same or
not" means anything; the count restarts at 0 with this node.
"""

import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int32, UInt32
from std_srvs.srv import Trigger


class OctomapGater(Node):

    def __init__(self):
        super().__init__('octomap_gater')
        self.declare_parameter('default_frames', 3)
        self.declare_parameter('lidar_window', 0.8)
        self.declare_parameter('speckle_px', 200)       # 0: off
        self.declare_parameter('speckle_diff_mm', 25)

        cb = ReentrantCallbackGroup()
        self.subscription = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw_in', self.image_callback, 10,
            callback_group=cb)
        self.publisher = self.create_publisher(
            Image, '/camera/camera/depth/image_rect_raw', 10)

        self.create_service(Trigger, '/octomap_gater/refresh', self.refresh_callback,
                            callback_group=cb)
        self.create_subscription(Int32, '/octomap_gater/allow_frames',
                                 self.allow_frames_callback, 10, callback_group=cb)

        self.frames_to_pass = 0
        self.lock = threading.Lock()
        # the LiDAR: its work-area clouds collected during a refresh, then one cloud
        self.create_subscription(
            PointCloud2, '/lidar/cloud_work', self.lidar_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT), callback_group=cb)
        self.lidar_publisher = self.create_publisher(PointCloud2, '/lidar/cloud_octomap', 5)
        self.lidar_until = None
        self.lidar_parts = []
        self.create_timer(0.05, self.lidar_flush, callback_group=cb)
        # the map version (see the top), latched: a client started later gets it too
        self.version = 0
        self.version_lock = threading.Lock()        # (bumped from several callbacks)
        self.version_publisher = self.create_publisher(
            UInt32, '/octomap_gater/version',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.version_publisher.publish(UInt32(data=self.version))

        self.get_logger().info(
            'octomap gater started; call /octomap_gater/refresh to update the map')

    def image_callback(self, msg):
        with self.lock:
            if self.frames_to_pass > 0:
                self.publisher.publish(self.despeckle(msg))
                self.frames_to_pass -= 1
                if self.frames_to_pass == 0:
                    self.bump_version()             # the camera's part of the map is in
                    self.get_logger().info('frames sent; octomap is static again')

    def despeckle(self, msg):
        """The depth frame without its small islands (see the top). 16-bit (mm) only;
        anything else, or speckle_px 0, passes as it is."""
        size = int(self.get_parameter('speckle_px').value)
        if size <= 0 or msg.encoding not in ('16UC1', 'mono16') or msg.is_bigendian:
            return msg
        d = np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        s = np.minimum(d, 32767).astype(np.int16)
        cv2.filterSpeckles(s, 0, size, int(self.get_parameter('speckle_diff_mm').value))
        out = np.where(s == 0, 0, d).astype(np.uint16)
        msg.data = out.tobytes()
        msg.step = 2 * msg.width
        return msg

    def lidar_callback(self, msg):
        with self.lock:
            if self.lidar_until is not None and time.monotonic() < self.lidar_until:
                self.lidar_parts.append(msg)

    def lidar_flush(self):
        """Once the window is over: the clouds of it, as one, to the octomap."""
        with self.lock:
            if self.lidar_until is None or time.monotonic() < self.lidar_until:
                return
            parts, self.lidar_parts, self.lidar_until = self.lidar_parts, [], None
        merged = merge_clouds(parts)
        if merged is not None:
            self.lidar_publisher.publish(merged)
            self.bump_version()                     # the LiDAR's part of the map is in
            self.get_logger().info(f'LiDAR: {len(parts)} clouds, {merged.width} points passed '
                                   f'to the octomap')

    def bump_version(self):
        """The map changes: /octomap_gater/version up by one."""
        with self.version_lock:
            self.version = (self.version + 1) % (1 << 32)
            self.version_publisher.publish(UInt32(data=self.version))

    def allow_frames(self, num_frames=3):
        with self.lock:
            self.frames_to_pass = max(1, int(num_frames))
            count = self.frames_to_pass
            self.lidar_until = time.monotonic() + float(self.get_parameter('lidar_window').value)
            self.lidar_parts = []
            self.bump_version()                     # a new map is on its way
        self.get_logger().info(f'passing the next {count} frames to the octomap')
        return count

    def refresh_callback(self, _request, response):
        count = self.allow_frames(self.get_parameter('default_frames').value)
        response.success = True
        response.message = f'passing {count} frames'
        return response

    def allow_frames_callback(self, msg):
        self.allow_frames(msg.data)


def merge_clouds(parts):
    """One PointCloud2 of several of the same layout and frame (the LiDAR's, the
    robot standing still); the newest one's stamp. None if there are none."""
    parts = [m for m in parts if m.width * m.height > 0]
    if not parts:
        return None
    first = parts[-1]
    same = [m for m in parts if m.point_step == first.point_step
            and m.header.frame_id == first.header.frame_id
            and [f.name for f in m.fields] == [f.name for f in first.fields]]
    out = PointCloud2()
    out.header = first.header
    out.fields, out.is_bigendian = first.fields, first.is_bigendian
    out.point_step = first.point_step
    out.height = 1
    out.width = sum(m.width * m.height for m in same)
    out.row_step = out.point_step * out.width
    out.is_dense = all(m.is_dense for m in same)
    out.data = b''.join(bytes(m.data) for m in same)
    return out


def run_gui(node):
    """Tk control panel. Skipped when there is no display."""
    import tkinter as tk

    root = tk.Tk()
    root.title('Octomap Controller')
    root.geometry('300x150')

    tk.Label(root, text='Octomap Mode: STATIC', font=('Helvetica', 14)).pack(pady=10)
    btn = tk.Button(root, text='Refresh Octomap',
                    command=lambda: node.allow_frames(
                        node.get_parameter('default_frames').value),
                    font=('Helvetica', 16), bg='lightblue')
    btn.pack(expand=True, fill='both', padx=20, pady=10)

    root.protocol('WM_DELETE_WINDOW', rclpy.shutdown)
    root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = OctomapGater()

    gui_thread = None
    if os.environ.get('DISPLAY'):
        try:
            gui_thread = threading.Thread(target=run_gui, args=(node,), daemon=True)
            gui_thread.start()
        except Exception as exc:
            node.get_logger().warn(f'GUI unavailable ({exc}); service only')
    else:
        node.get_logger().info('no DISPLAY: running without the Tk panel')

    code = 0
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:                        # noqa: BLE001 - said, and a non-zero exit
        if type(exc).__name__ == 'ExternalShutdownException':
            pass                                    # (the Tk window closed: rclpy.shutdown)
        else:
            # a callback's bug: loud, with its traceback, and exit 1 -- so launch says
            # the gater died (it was silent, exit 0, 2026-11-17 review)
            traceback.print_exc()
            code = 1
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except BaseException:                       # a second Ctrl-C mid-cleanup
            pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException:                       # noqa: BLE001
            pass
        # leave now: a normal exit waits on the Tk thread's Tcl interpreter and
        # could outlive Ctrl-C and SIGTERM, leaving `ros2 launch` waiting (11-17)
        os._exit(code)


if __name__ == '__main__':
    main()
