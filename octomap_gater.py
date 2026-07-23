#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import tkinter as tk
import threading

class OctomapGater(Node):
    def __init__(self):
        super().__init__('octomap_gater')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw_in',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(
            Image, 
            '/camera/camera/depth/image_rect_raw', 
            10)
        
        self.frames_to_pass = 0
        self.lock = threading.Lock()
        
        self.get_logger().info('Octomap Gater started. Waiting for button press.')

    def image_callback(self, msg):
        with self.lock:
            if self.frames_to_pass > 0:
                self.publisher.publish(msg)
                self.frames_to_pass -= 1
                if self.frames_to_pass == 0:
                    self.get_logger().info('Finished sending frames. Octomap is static again.')

    def allow_frames(self, num_frames=3):
        with self.lock:
            self.frames_to_pass = num_frames
            self.get_logger().info(f'Allowing next {num_frames} frames through to update octomap...')

def run_gui(node):
    root = tk.Tk()
    root.title("Octomap Controller")
    root.geometry("300x150")
    
    label = tk.Label(root, text="Octomap Mode: STATIC", font=("Helvetica", 14))
    label.pack(pady=10)
    
    def on_click():
        node.allow_frames(3)
        
    btn = tk.Button(root, text="Refresh Octomap", command=on_click, font=("Helvetica", 16), bg="lightblue")
    btn.pack(expand=True, fill='both', padx=20, pady=10)
    
    root.protocol("WM_DELETE_WINDOW", lambda: rclpy.shutdown())
    root.mainloop()

def main(args=None):
    rclpy.init(args=args)
    node = OctomapGater()
    
    gui_thread = threading.Thread(target=run_gui, args=(node,))
    gui_thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        gui_thread.join()

if __name__ == '__main__':
    main()
