"""Box picker as a long-running process: one JSON answer per request, for the robot software.

Loads SAM 3.1 and opens the camera once, then waits on stdin. Each line is a request:
    pick    take --frames new frames (after dropping --skip, so none are from before the
            robot stopped) and print one JSON line: the answer of PickDetector.detect_frames,
            or {"target": null} when no box can be picked
    quit    close the camera and exit
Every reply is one line on stdout; anything else this program prints goes to stderr. The
first stdout line is {"ready": true, ...} once the model and camera are up.

Any language can drive it (a ROS 2 node in Python 3.10, C++, a shell): start it once, write
"pick\\n", read one line.

Usage:
    .venv/bin/python pick_server.py --pitch 0
    .venv/bin/python pick_server.py --pitch 0 --ros                # camera over ROS 2 (ros_bridge.py)
    .venv/bin/python pick_server.py --pitch 0 --save outputs/picks # keep each request's frames + overlay
    printf 'pick\\nquit\\n' | .venv/bin/python pick_server.py --pitch 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=5, help="frames per answer")
    ap.add_argument("--skip", type=int, default=5, help="frames dropped first, so all are from after the request")
    ap.add_argument("--pitch", type=float, help="camera tilt down in degrees (default: accelerometer, else 0)")
    ap.add_argument("--ros", action="store_true", help="camera over ROS 2, through ros_bridge.py")
    ap.add_argument("--save", metavar="DIR", help="save each request's frames (colour + depth + meta) and overlay here")
    args = ap.parse_args()

    # Everything the libraries print goes to stderr, so stdout carries only the JSON replies.
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    from faces import up_from_accel, up_from_pitch
    from pick_frame import PickDetector
    from rgbd import open_camera, save
    detector = PickDetector()
    cam = open_camera(args.ros)
    up = up_from_accel(cam.accel) if cam.accel and args.pitch is None else up_from_pitch(args.pitch or 0)
    sys.stdout = real_stdout
    reply({"ready": True, "frames": args.frames, "up": [round(float(v), 3) for v in up]})

    try:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in ("quit", "exit"):
                break
            if cmd != "pick":
                reply({"error": f"unknown request {cmd!r}: use pick or quit"})
                continue
            t0 = time.time()
            for _ in range(args.skip):
                cam.read()
            frames = [cam.read() for _ in range(args.frames)]
            result = detector.detect_frames(frames, cam.intr, up)
            out = {"target": result, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 2)}
            if args.save:
                name = time.strftime("pick_%Y%m%d_%H%M%S")
                for k, (c, d) in enumerate(frames):
                    save(Path(args.save) / f"{name}_{k:02d}", c, d, cam.meta())
                if detector.last is not None:
                    cv2.imwrite(str(Path(args.save) / f"{name}.jpg"), detector.annotate(frames[detector.best_frame or 0][0]))
                out["saved"] = str(Path(args.save) / name)
            reply(out)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
