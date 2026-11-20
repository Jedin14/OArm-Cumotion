"""Aligned colour + depth from the D455, saved or loaded as a set of three files.

A capture is <name>_color.png, <name>_depth_mm.png (16-bit, depth aligned to the
colour image, 0 = no reading) and <name>_meta.json (colour intrinsics, depth
scale and the accelerometer reading, which gives the up direction).

Usage:
    python rgbd.py captures/desk                # save one capture after auto exposure settles
    python rgbd.py captures/desk --count 5 --interval 2
    python rgbd.py captures/desk --ros          # camera on another machine, over ROS 2 (ros_bridge.py)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from faces import Intrinsics


class Camera:
    """D455 colour + depth aligned to colour, plus the accelerometer."""

    def __init__(self, width=1280, height=720, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.accel = self._read_accel()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)
        for attempt in range(2):
            self.pipe = rs.pipeline()
            profile = self.pipe.start(cfg)
            try:
                for _ in range(30):              # let auto exposure settle
                    self.pipe.wait_for_frames()
                break
            except RuntimeError:                 # "Frame didn't arrive": the camera is stuck
                self.pipe.stop()
                if attempt:
                    raise
                print("no frames from the camera: resetting it (like unplugging it) and trying again", flush=True)
                for dev in rs.context().query_devices():
                    dev.hardware_reset()
                time.sleep(8)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.intr = Intrinsics.from_rs(profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
        self.align = rs.align(rs.stream.color)

    def _read_accel(self, samples=50):
        """Mean accelerometer reading (m/s^2) over `samples` frames, or None if there is no IMU."""
        rs = self.rs
        try:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.accel)
            pipe = rs.pipeline()
            pipe.start(cfg)
            vals = []
            try:
                while len(vals) < samples:
                    m = pipe.wait_for_frames().first_or_default(rs.stream.accel)
                    if m:
                        d = m.as_motion_frame().get_motion_data()
                        vals.append((d.x, d.y, d.z))
            finally:
                pipe.stop()
            return np.mean(vals, axis=0).tolist()
        except RuntimeError as e:
            print(f"no accelerometer reading ({e}); the up direction will be assumed")
            return None

    def read(self):
        """(colour BGR uint8, depth in metres float32), aligned pixel for pixel."""
        while True:
            frames = self.align.process(self.pipe.wait_for_frames())
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if color and depth:
                return (np.asanyarray(color.get_data()).copy(),
                        np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale)

    def meta(self):
        i = self.intr
        return {"fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy, "depth_scale": self.depth_scale, "accel": self.accel}

    def close(self):
        self.pipe.stop()


ROS_SETUP = "/opt/ros/humble/setup.bash"
MSG_SETUP = Path(__file__).parent / "ros" / "install" / "setup.bash"   # the robot's box size message (README)
BRIDGE_LOG = Path(__file__).parent / "outputs" / "ros_bridge.log"


def start_bridge(port):
    """Start ros_bridge.py with the system Python and ROS 2 (rclpy can't load in the .venv).
    Its 10 s reports go to BRIDGE_LOG, not to stdout, which pick_loop uses for its events."""
    import os, subprocess
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")}
    env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":") if ".venv" not in p)
    BRIDGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    bridge = Path(__file__).with_name("ros_bridge.py")
    setup = f'source {ROS_SETUP}' + (f' && source "{MSG_SETUP}"' if MSG_SETUP.exists() else '')
    return subprocess.Popen(["bash", "-c", f'{setup} && exec /usr/bin/python3 "{bridge}" --port {port}'],
                            env=env, stdout=open(BRIDGE_LOG, "w"), stderr=subprocess.STDOUT)


class RosCamera:
    """The same as Camera, but frames come over ROS 2 through ros_bridge.py. For when the D455 is
    plugged into another machine. Uses a bridge that is already running, or starts its own."""

    def __init__(self, port=50555, host="127.0.0.1", wait_s=30):
        import atexit, json, socket, struct
        self._json, self._struct = json, struct
        self.bridge = None
        try:
            self.sock = socket.create_connection((host, port), timeout=10)
        except OSError:
            print(f"starting ros_bridge.py (log: {BRIDGE_LOG})", flush=True)
            self.bridge = start_bridge(port)
            atexit.register(self.close)
            end = time.time() + wait_s
            while True:
                try:
                    self.sock = socket.create_connection((host, port), timeout=10)
                    break
                except OSError:
                    if self.bridge.poll() is not None or time.time() > end:
                        raise SystemExit(f"ros_bridge.py didn't start; see {BRIDGE_LOG}")
                    time.sleep(0.5)
        self.accel = None                        # from each frame's header, when the driver sends it
        self.box_size = None                     # [height, width, depth] mm: the robot's, from the bridge
        self.depth_scale = 0.001
        end = time.time() + wait_s               # a new bridge needs a few seconds to find the camera
        while True:
            try:
                color, depth = self.read()       # first frame: also gives the intrinsics
                break
            except RuntimeError:
                if self.bridge is None or time.time() > end:
                    raise
        print(f"ROS camera: {color.shape[1]}x{color.shape[0]}, fx {self.intr.fx:.1f} fy {self.intr.fy:.1f}", flush=True)

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("ros_bridge.py closed the connection")
            buf += chunk
        return bytes(buf)

    def read(self):
        """(colour BGR uint8, depth in metres float32), aligned pixel for pixel; a new frame each call."""
        self.sock.sendall(b"f")
        (n,) = self._struct.unpack("!I", self._recv(4))
        h = self._json.loads(self._recv(n))
        if "error" in h:
            raise RuntimeError(f"ros_bridge.py: {h['error']}")
        w, ht = h["width"], h["height"]
        color = np.frombuffer(self._recv(w * ht * 3), np.uint8).reshape(ht, w, 3).copy()
        depth = np.frombuffer(self._recv(w * ht * 2), np.uint16).reshape(ht, w).astype(np.float32) / 1000
        self.intr = Intrinsics(h["fx"], h["fy"], h["cx"], h["cy"])
        self.accel = h.get("accel")              # mean of the last 0.5 s, None while moving
        self.box_size = h.get("box_size")
        return color, depth

    def meta(self):
        i = self.intr
        return {"fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy, "depth_scale": self.depth_scale, "accel": self.accel,
                "box_size": self.box_size, "source": "ros"}

    def close(self):
        if getattr(self, "sock", None):
            self.sock.close()
            self.sock = None
        if self.bridge and self.bridge.poll() is None:   # only the bridge this started
            self.bridge.terminate()
            self.bridge.wait(5)


class ReplayCamera:
    """A recording (record_rgbd.py / rgbd.py frames in one folder) played as if it were the camera, to
    try the pick loop without one: frame k is shown from k / fps seconds on, at the camera's 30 a
    second; after the last frame it starts again."""

    def __init__(self, folder, fps=1.0, start=0):
        self.prefixes = sorted(str(p)[:-len("_color.png")] for p in Path(folder).glob("*_color.png"))
        self.prefixes = [p for p in self.prefixes if int(Path(p).name.split("_")[1]) >= start] or self.prefixes
        if not self.prefixes:
            raise SystemExit(f"no frames (*_color.png) in {folder}")
        self.fps, self.t0, self.k, self.frame = fps, time.time(), None, None
        _, _, self.intr, self.accel = load(self.prefixes[0])
        self.box_size = json.loads(Path(f"{self.prefixes[0]}_meta.json").read_text()).get("box_size")
        print(f"replaying {len(self.prefixes)} frames of {folder}, {fps:g} a second", flush=True)

    def read(self):
        time.sleep(1 / 30)
        k = int((time.time() - self.t0) * self.fps) % len(self.prefixes)
        if k != self.k:
            color, depth, self.intr, accel = load(self.prefixes[k])
            self.accel = accel or self.accel
            self.k, self.frame = k, (color, depth)
        return self.frame[0].copy(), self.frame[1].copy()

    def meta(self):
        i = self.intr
        return {"fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy, "depth_scale": 0.001, "accel": self.accel,
                "box_size": self.box_size, "replay": self.prefixes[self.k or 0]}

    def close(self):
        pass


def open_camera(ros=False, **kw):
    """The D455 over USB (Camera), or over ROS 2 through ros_bridge.py (RosCamera)."""
    return RosCamera() if ros else Camera(**kw)


def save(prefix, color, depth_m, meta):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{prefix}_color.png", color)
    cv2.imwrite(f"{prefix}_depth_mm.png", np.round(depth_m * 1000).clip(0, 65535).astype(np.uint16))
    Path(f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2))


def load(prefix):
    """(colour, depth in metres, Intrinsics, accel or None) from a saved capture."""
    meta = json.loads(Path(f"{prefix}_meta.json").read_text())
    color = cv2.imread(f"{prefix}_color.png")
    depth = cv2.imread(f"{prefix}_depth_mm.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    return color, depth, Intrinsics(meta["fx"], meta["fy"], meta["cx"], meta["cy"]), meta.get("accel")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="output path prefix, e.g. captures/desk")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between captures")
    ap.add_argument("--ros", action="store_true", help="frames over ROS 2, through ros_bridge.py")
    args = ap.parse_args()
    cam = open_camera(args.ros)
    try:
        for k in range(args.count):
            color, depth = cam.read()
            prefix = args.prefix if args.count == 1 else f"{args.prefix}_{k:02d}"
            save(prefix, color, depth, cam.meta())
            print(f"saved {prefix}_*  (valid depth {np.mean(depth > 0):.0%}, accel {cam.accel})")
            if k + 1 < args.count:
                time.sleep(args.interval)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
