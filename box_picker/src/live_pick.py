"""Run the pick pipeline live on the D455, in a window.

The camera feed is shown at full rate. The pipeline (SAM 3.1 + depth, well under
a second a frame) runs over and over in the background on the newest frame, and its
latest result is drawn over the live feed, with how old it is. With --frames N each
answer comes from N new frames, as the robot asks after it stops (detect_frames:
the target most frames agree on, their median). Each new target
is also printed as JSON, and with --log appended to a file (one JSON line each,
with the time; null when no box is found).

Keys: Q / Esc quit, S save the current frame (colour + depth + meta) and overlay.

Usage (from src/):
    .venv/bin/python live_pick.py
    .venv/bin/python live_pick.py --pitch 20 --out outputs/live_pick
    .venv/bin/python live_pick.py --log outputs/live_pick/targets.jsonl
    .venv/bin/python live_pick.py --frames 5                  # one answer per 5 frames, like the robot
"""
import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from faces import DEFAULT_DEPTH_TOL, up_from_accel, up_from_pitch
from pick_frame import PickDetector
from pick_pipeline import draw
from rgbd import open_camera, save


class Worker(threading.Thread):
    """Runs the detector on the newest frame (or the next n frames), again and again."""

    def __init__(self, detector, intr, up, log=None, n=1):
        super().__init__(daemon=True)
        self.detector, self.intr, self.up, self.log, self.n = detector, intr, up, log, n
        self.lock = threading.Lock()
        self.frame = None                  # newest (colour, depth) from the camera
        self.result = None                 # (target dict, points, boxes, target index, seconds taken, time)
        self.running = True

    def submit(self, color, depth):
        with self.lock:
            self.frame = color, depth

    def next_frame(self):
        """Wait for a frame from the camera newer than the last one taken."""
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is not None:
                return frame
            time.sleep(0.005)

    def run(self):
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.time()
            if self.n == 1:
                target = self.detector.detect(frame[0], frame[1], self.intr, self.up)
            else:
                frames = [frame] + [self.next_frame() for _ in range(self.n - 1)]
                if not self.running:
                    break
                target = self.detector.detect_frames(frames, self.intr, self.up)
            points, boxes, idx = self.detector.last or ([], [], None)
            with self.lock:
                self.result = target, points, boxes, idx, time.time() - t0, time.time()
            print(json.dumps(target), flush=True)
            if self.log:
                with open(self.log, "a") as f:
                    f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "target": target}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pitch", type=float, help="camera tilt down in degrees (default: accelerometer, else 0)")
    ap.add_argument("--depth-tol", type=float, default=DEFAULT_DEPTH_TOL, help="front-layer thickness, metres")
    ap.add_argument("--out", default="outputs/live_pick", help="folder for frames saved with S")
    ap.add_argument("--log", metavar="FILE.jsonl", help="also append every result to this file")
    ap.add_argument("--ros", action="store_true", help="camera over ROS 2, through ros_bridge.py")
    ap.add_argument("--frames", type=int, default=1,
                    help="one answer per this many new frames (as the robot asks after it stops); 1 = each frame")
    ap.add_argument("--smooth", type=int, default=5,
                    help="average the last N results while the target stays on the same box (1 = off)")
    args = ap.parse_args()

    print("loading SAM 3.1 ...", flush=True)
    detector = PickDetector(depth_tol=args.depth_tol, smooth=args.smooth)
    cam = open_camera(args.ros)
    up = up_from_accel(cam.accel) if cam.accel and args.pitch is None else up_from_pitch(args.pitch or 0)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    worker = Worker(detector, cam.intr, up, args.log, n=args.frames)
    worker.start()
    window = "pick pipeline (Q quit, S save)"
    try:
        while True:
            color, depth = cam.read()
            worker.submit(color, depth)
            with worker.lock:
                res = worker.result
            if res is None:
                view = color.copy()
                cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(view, "first result in a few seconds ...", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                target, points, boxes, idx, took, at = res
                view = draw(color, points, boxes, idx)
                h = view.shape[0]
                cv2.rectangle(view, (0, h - 30), (view.shape[1], h), (0, 0, 0), -1)
                agree = f", {target['agreeing']}/{target['frames']} frames agree" if target and "frames" in target else ""
                cv2.putText(view, f"result {time.time() - at:.1f} s old, {took:.1f} s per update, {len(boxes)} boxes{agree}",
                            (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = time.strftime("live_%Y%m%d_%H%M%S")
                save(Path(args.out) / name, color, depth, cam.meta())
                cv2.imwrite(str(Path(args.out) / f"{name}.jpg"), view)
                print(f"saved {args.out}/{name}_*", flush=True)
    finally:
        worker.running = False
        worker.join(timeout=15)            # let a running detection finish before the GPU goes away
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
