"""Record colour + depth from the D455, so the pick pipeline can be run later without the camera.

Every saved frame is a capture in the rgbd.py format: <out>/frame_NNNNN_color.png,
_depth_mm.png (16-bit millimetres, aligned to the colour image pixel for pixel,
0 = no reading) and _meta.json (colour intrinsics, depth scale, accelerometer, time).
A colour-only preview video, <out>/color.mp4, is written next to them.

The window shows the colour image and the depth beside it. R starts / pauses
recording, Q or Esc stops. With --seconds it records straight away, no keys needed.

Usage (from src/):
    .venv/bin/python record_rgbd.py recordings/test2                 # R to record, Q to stop
    .venv/bin/python record_rgbd.py recordings/test2 --every 1       # all 30 frames a second
    .venv/bin/python record_rgbd.py recordings/test2 --seconds 60    # record 60 s, then stop

Then, without the camera:
    .venv/bin/python pick_pipeline.py --capture "recordings/test2/frame_*" --out outputs/test2
    .venv/bin/python pick_frame.py recordings/test2/frame_00030
"""
import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from rgbd import open_camera, save

FPS = 30


class Saver(threading.Thread):
    """Writes captures in the background, so the preview doesn't stall on PNG encoding."""

    def __init__(self):
        super().__init__(daemon=True)
        self.jobs = queue.Queue(maxsize=120)

    def run(self):
        while (job := self.jobs.get()) is not None:
            save(*job)


def depth_view(depth_m, max_m=1.5):
    """Depth as a colour picture: near = red, far = blue, no reading = black."""
    d = np.clip(depth_m / max_m * 255, 0, 255).astype(np.uint8)
    view = cv2.applyColorMap(255 - d, cv2.COLORMAP_JET)
    view[depth_m <= 0] = 0
    return view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output folder")
    ap.add_argument("--every", type=int, default=3, help="save every Nth camera frame (3 = 10 frames a second)")
    ap.add_argument("--seconds", type=float, help="record this long straight away, then stop (no keys)")
    ap.add_argument("--ros", action="store_true", help="camera over ROS 2, through ros_bridge.py")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = open_camera(args.ros, fps=FPS)
    print(f"camera ready; accelerometer {cam.accel}", flush=True)
    saver = Saver()
    saver.start()
    video = None
    recording, start, seen, saved = args.seconds is not None, time.time(), 0, 0
    try:
        while True:
            color, depth = cam.read()
            if recording:
                if seen % args.every == 0:
                    meta = {**cam.meta(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "t_s": round(time.time() - start, 3)}
                    saver.jobs.put((out / f"frame_{saved:05d}", color, depth, meta))
                    if video is None:
                        video = cv2.VideoWriter(str(out / "color.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                FPS / args.every, (color.shape[1], color.shape[0]))
                    video.write(color)
                    saved += 1
                seen += 1
                if args.seconds is not None and time.time() - start >= args.seconds:
                    break

            if args.seconds is None:
                view = np.hstack([color, depth_view(depth)])
                status = (f"REC {saved} frames, {saved * args.every / FPS:.1f}s   depth {np.mean(depth > 0):.0%}"
                          if recording else "PAUSED - press R to record, Q to stop")
                cv2.putText(view, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255) if recording else (0, 255, 255), 2)
                cv2.imshow("D455 colour + depth recorder (R record/pause, Q stop)",
                           cv2.resize(view, None, fx=0.5, fy=0.5))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    recording = not recording
                    start = time.time() if recording and saved == 0 else start
                elif key in (ord("q"), 27):
                    break
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if video is not None:
            video.release()
        print("writing the last frames ...", flush=True)
        saver.jobs.put(None)
        saver.join()
    print(f"saved {saved} frames (colour + depth + meta) to {out}/  ({saved * args.every / FPS:.1f}s of recording)")


if __name__ == "__main__":
    main()
