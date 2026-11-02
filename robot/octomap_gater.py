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
"""

import os
import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from std_srvs.srv import Trigger


class OctomapGater(Node):

    def __init__(self):
        super().__init__('octomap_gater')
        self.declare_parameter('default_frames', 3)

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

        self.get_logger().info(
            'octomap gater started; call /octomap_gater/refresh to update the map')

    def image_callback(self, msg):
        with self.lock:
            if self.frames_to_pass > 0:
                self.publisher.publish(msg)
                self.frames_to_pass -= 1
                if self.frames_to_pass == 0:
                    self.get_logger().info('frames sent; octomap is static again')

    def allow_frames(self, num_frames=3):
        with self.lock:
            self.frames_to_pass = max(1, int(num_frames))
            count = self.frames_to_pass
        self.get_logger().info(f'passing the next {count} frames to the octomap')
        return count

    def refresh_callback(self, _request, response):
        count = self.allow_frames(self.get_parameter('default_frames').value)
        response.success = True
        response.message = f'passing {count} frames'
        return response

    def allow_frames_callback(self, msg):
        self.allow_frames(msg.data)


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

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
