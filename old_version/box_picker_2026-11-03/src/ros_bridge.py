"""Bridge: the D455 over ROS 2 -> the pick pipeline, when the camera is on another machine.

ROS 2 Humble's Python is 3.10; the pipeline's .venv is 3.11 and can't load rclpy. So this
small node runs with the system Python: it subscribes to the colour image, the depth image
aligned to it and the camera info, pairs colour and depth by time stamp, and hands the newest
pair to the pipeline over a local socket. In the pipeline, rgbd.RosCamera reads from it (the
--ros option of live_pick.py, pick_frame.py, pick_pipeline.py and record_rgbd.py).

Run it in its own terminal (system Python, not the .venv), and leave it running:
    source /opt/ros/humble/setup.bash
    python3 ros_bridge.py
    python3 ros_bridge.py --qos reliable                 # if colour frames get lost (see its 10 s report)
    python3 ros_bridge.py --color /camera/camera/color/image_raw \
        --depth /camera/camera/aligned_depth_to_color/image_raw --info /camera/camera/color/camera_info

Protocol (127.0.0.1:--port): the client sends one byte per frame it wants; the bridge
answers with the next pair newer than that request: a 4-byte length, a JSON header (width,
height, fx, fy, cx, cy, stamp) and the raw colour (BGR uint8) then depth (uint16, mm) bytes.
"""
import argparse
import json
import socket
import struct
import threading
import time

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

PORT = 50555


def to_numpy(msg):
    """sensor_msgs/Image -> numpy (colour: BGR uint8 HxWx3; depth: uint16 HxW, mm)."""
    if msg.encoding in ("rgb8", "bgr8"):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        a = a.reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(a[:, :, ::-1] if msg.encoding == "rgb8" else a)
    if msg.encoding in ("16UC1", "mono16"):
        a = np.frombuffer(msg.data, np.uint16 if not msg.is_bigendian else ">u2")
        return np.ascontiguousarray(a.reshape(msg.height, msg.step // 2)[:, :msg.width].astype(np.uint16))
    if msg.encoding == "32FC1":                      # metres -> mm
        a = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width]
        return np.nan_to_num(a * 1000).clip(0, 65535).astype(np.uint16)
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


class Bridge(Node):
    def __init__(self, args):
        super().__init__("box_picker_bridge")
        self.lock = threading.Condition()
        self.pair, self.intr = None, None
        # best_effort connects to any publisher, but a lost piece of a big image drops the image;
        # reliable resends lost pieces, but only connects to a publisher that is reliable too.
        qos = qos_profile_sensor_data if args.qos == "best_effort" else QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(CameraInfo, args.info, self.on_info, qos)
        color = Subscriber(self, Image, args.color, qos_profile=qos)
        depth = Subscriber(self, Image, args.depth, qos_profile=qos)
        self.counts = {"colour": 0, "depth": 0, "pairs": 0}
        color.registerCallback(lambda m: self.count("colour"))
        depth.registerCallback(lambda m: self.count("depth"))
        self.create_timer(10.0, self.report)
        self.sync = ApproximateTimeSynchronizer([color, depth], queue_size=10, slop=args.slop)
        self.sync.registerCallback(self.on_pair)
        self.get_logger().info(f"waiting for {args.color}, {args.depth}, {args.info}")

    def count(self, what):
        self.counts[what] += 1

    def report(self):
        """Every 10 s: messages per second, so a slow or lossy link shows."""
        c = self.counts
        self.get_logger().info(f"last 10 s: colour {c['colour'] / 10:.1f}/s, depth {c['depth'] / 10:.1f}/s, "
                               f"matched pairs {c['pairs'] / 10:.1f}/s")
        self.counts = dict.fromkeys(c, 0)

    def on_info(self, msg):
        k = msg.k
        with self.lock:
            if self.intr is None:
                self.get_logger().info(f"camera info: {msg.width}x{msg.height} fx {k[0]:.1f} fy {k[4]:.1f} cx {k[2]:.1f} cy {k[5]:.1f}")
            self.intr = (k[0], k[4], k[2], k[5])

    def on_pair(self, color, depth):
        c, d = to_numpy(color), to_numpy(depth)
        if c.shape[:2] != d.shape:
            self.get_logger().warn(f"colour {c.shape[:2]} and depth {d.shape} differ: is depth aligned to colour?")
            return
        stamp = color.header.stamp.sec + color.header.stamp.nanosec * 1e-9
        self.counts["pairs"] += 1
        with self.lock:
            first = self.pair is None
            self.pair = (c, d, stamp, time.time())
            self.lock.notify_all()
        if first:
            self.get_logger().info(f"first colour + depth pair: {c.shape[1]}x{c.shape[0]}")

    def next_pair(self, after, timeout=5.0):
        """The first pair received after time `after` (so each read is a new frame)."""
        with self.lock:
            ok = self.lock.wait_for(lambda: self.pair is not None and self.intr is not None and self.pair[3] > after,
                                    timeout)
            return (self.pair, self.intr) if ok else (None, self.intr)


def listen(port):
    """The local socket the pipeline connects to; stops here if another bridge already has it."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("127.0.0.1", port))
    except OSError as e:
        raise SystemExit(f"port {port} is taken ({e}): is another ros_bridge.py running? Stop it, or use --port")
    srv.listen(4)
    return srv


def serve(bridge, srv):
    bridge.get_logger().info(f"serving frames on 127.0.0.1:{srv.getsockname()[1]}")
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=client, args=(bridge, conn), daemon=True).start()


def client(bridge, conn):
    with conn:
        while conn.recv(1):
            pair, intr = bridge.next_pair(time.time())
            if pair is None:
                header = json.dumps({"error": "no colour + depth pair in 5 s (camera info received: %s)" % (intr is not None)})
                conn.sendall(struct.pack("!I", len(header)) + header.encode())
                continue
            c, d, stamp, _ = pair
            fx, fy, cx, cy = intr
            header = json.dumps({"width": c.shape[1], "height": c.shape[0], "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                                 "stamp": stamp}).encode()
            conn.sendall(struct.pack("!I", len(header)) + header + c.tobytes() + d.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--color", default="/camera/camera/color/image_raw")
    ap.add_argument("--depth", default="/camera/camera/aligned_depth_to_color/image_raw")
    ap.add_argument("--info", default="/camera/camera/color/camera_info")
    ap.add_argument("--slop", type=float, default=0.05, help="max time between a colour and a depth frame, s")
    ap.add_argument("--qos", choices=["best_effort", "reliable"], default="best_effort",
                    help="reliable resends lost pieces of big images, but needs a reliable publisher")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    srv = listen(args.port)
    rclpy.init()
    bridge = Bridge(args)
    threading.Thread(target=serve, args=(bridge, srv), daemon=True).start()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
