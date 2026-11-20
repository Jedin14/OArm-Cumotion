"""Replay a pick check from the robot: the folder box_picker_server.py wrote for one pick
(<out>/<run>/checks/NNN_HHMMSS/, recorded with --record-checks or "set record on") through
pick_watch.py again, frame by frame -- to see why it decided what it did, and to try a change
to pick_watch.py on real picks.

It reads face.npz (the target's pick face, its outline, the box size, the camera), ref_*_depth.png
(the pick's frames: the depth before the pick) and frames/ (every frame watched, with its time).
The robot's "check" (its arm out of the view) comes when it came then (the first check's
timeline), or at --arm-out S. As on the robot, a missed / moved before it is not final (the
robot can press again): the watcher is reopened (--early-final: final, as pick_loop.py).

Usage (from box_picking/v1/):
    ../../vlm/molmo2-box-picking/.venv/bin/python replay_check.py <pick folder>
    ... replay_check.py <pick folder> --arm-out 12.5        # the robot's check at 12.5 s
    ... replay_check.py <pick folder> --one-ref --no-size   # as before 19 Nov 2026 (one frame, no box size)
    ... replay_check.py <pick folder> --trace               # a line for every frame
"""
import argparse
import glob
import json
import os
import re
from types import SimpleNamespace

import cv2
import numpy as np

import pick_watch as pw


def load_pick(folder):
    """-> (face, mask, intr, box_size or None, [reference depths, the target's frame first])."""
    z = np.load(os.path.join(folder, "face.npz"), allow_pickle=False)
    face = SimpleNamespace(pixels=z["pixels"], normal=z["normal"], centroid=z["centroid"], corners_px=z["corners_px"],
                           centroid_px=tuple(z["centroid_px"]), size_mm=tuple(z["size_mm"]), kind=str(z["kind"]))
    fx, fy, cx, cy = z["intr"]
    intr = SimpleNamespace(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
    box = tuple(round(float(v)) for v in z["box_size"]) if z["box_size"].size == 3 else None
    refs = [cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            for p in sorted(glob.glob(os.path.join(folder, "ref_*_depth.png")))]
    best = int(z["best"])
    refs = [refs[best]] + [d for i, d in enumerate(refs) if i != best]
    return face, z["mask"], intr, box, refs


def recorded_arm_out(folder):
    """When the robot's first "check" came, s from the pick (its timeline), else None."""
    for p in sorted(glob.glob(os.path.join(folder, "check1_*_verdict.json"))):
        for t, what in json.load(open(p)).get("timeline", []):
            if what.startswith("the robot says its arm is out"):
                return float(t)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="one pick's folder: <out>/<run>/checks/NNN_HHMMSS")
    ap.add_argument("--arm-out", type=float, help="the robot's check at this many s from the pick "
                                                  "(default: when it came; -1: never)")
    ap.add_argument("--one-ref", action="store_true", help="only the target's frame as the depth before the pick")
    ap.add_argument("--no-size", action="store_true", help="without the box size (gone = 2 cm back)")
    ap.add_argument("--trace", action="store_true", help="print every frame")
    ap.add_argument("--early-final", action="store_true",
                    help="a missed / moved before the robot's check is final (pick_loop.py), not reopened")
    args = ap.parse_args()

    face, mask, intr, box, refs = load_pick(args.folder)
    frames = sorted(glob.glob(os.path.join(args.folder, "frames", "*_depth.png")))
    if not frames:
        raise SystemExit(f"no frames/ in {args.folder}: record with --record-checks (or 'set record on')")
    arm_out = args.arm_out if args.arm_out is not None else recorded_arm_out(args.folder)
    if arm_out is not None and arm_out < 0:
        arm_out = None
    w = pw.PickWatcher(face, refs[0], mask, refs=[] if args.one_ref else refs[1:], box_mm=None if args.no_size else box)
    print(f"{args.folder}: {len(frames)} frames, {len(refs)} reference frames"
          f"{' (using 1)' if args.one_ref else ''}, box {box if not args.no_size else None} mm -> gone at "
          f"{w.gone_m * 1000:.0f} mm back; the robot's check at "
          f"{'never' if arm_out is None else f'{arm_out:.1f} s'}")
    result = None
    for p in frames:
        t = float(re.search(r"_(\d+\.\d+)_depth\.png$", p).group(1))
        depth = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        result = w.update(depth, intr, t, arm_out=arm_out is not None and t >= arm_out)
        if args.trace:
            print(json.dumps(w.trace[-1]))
        if result in ("missed", "moved") and not w.arm_out and not args.early_final:
            w.reopen(result)                 # as box_picker_server.py: the robot's check decides
            result = None
        if result:
            break
    print()
    for t, what in w.timeline:
        print(f"{t:8.2f} s  {what}")
    print(f"\nresult: {result or 'none yet (still waiting)'} after {w.n} frames ({w.fps} a second)")
    print(w.detail)


if __name__ == "__main__":
    main()
