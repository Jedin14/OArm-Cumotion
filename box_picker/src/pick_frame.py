"""Run the pick pipeline on one frame, or a few taken while the robot stands still, and
print the target, as box_detector.py does for v2.

A frame is a live D455 frame, a saved capture (rgbd.py), or any colour image with an
aligned 16-bit depth PNG in millimetres. Prints the target as JSON (null if no box is
found) and saves an overlay image.

With several frames of the same still scene, each is run and the target most of them
agree on is returned, as the median of those frames (one odd frame can't pull it off).

Usage (from src/):
    .venv/bin/python pick_frame.py --live                              # grab one frame now
    .venv/bin/python pick_frame.py --live --frames 5                   # 5 frames, one answer
    .venv/bin/python pick_frame.py captures/desk                       # a saved capture
    .venv/bin/python pick_frame.py "recordings/test3/frame_0001[0-4]"  # several saved frames, one answer
    .venv/bin/python pick_frame.py color.png --depth depth_mm.png      # colour + aligned depth
    .venv/bin/python pick_frame.py --live --annotate out.jpg --pitch 20

Python (robot side):
    from pick_frame import PickDetector
    detector = PickDetector()                          # loads SAM 3.1 once (~5 s)
    result = detector.detect(color_bgr, depth_m, intr) # one frame; depth in metres, aligned to colour
    result = detector.detect_frames([(color_bgr, depth_m), ...], intr)   # robot stopped: n frames, one answer

    up: the camera-frame up direction (default level). The camera is fixed on the robot and
    doesn't tilt, so it's the same for every frame: set it once (up_from_pitch(degrees down)).

Result:
    {"x_m", "y_m", "z_m"}  centroid of the pick face, camera coordinates (x right, y down, z forward)
    "approach"             "front" or "top"
    "u", "v"               the centroid in image pixels
    "size_mm"              the pick face, long x short edge
    "num_boxes"            boxes found in the frame
    "averaged_over"        how many results were averaged (PickDetector(smooth=N): the last N
                           while the target stays on the same box; 1 = this frame only)
    "frames", "agreeing"   detect_frames only: frames given, and how many found this target
                           (fewer than half: the frames disagree, worth taking new ones)
"""
import glob
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from faces import DEFAULT_DEPTH_TOL, Intrinsics, up_from_accel, up_from_pitch
from pick_pipeline import draw, run

D455_COLOR_1280x720 = Intrinsics(645.9, 644.8, 643.9, 369.4)   # this camera's colour intrinsics


SAME_TARGET_M = 0.02   # a new result this close to the running average is the same box


def overlap(a, b):
    """Intersection over union of two (x1, y1, x2, y2) rectangles."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class PickDetector:
    """smooth: average the last this many results while the target stays on the same box
    (same approach, centre within SAME_TARGET_M of the average); 1 = no averaging. Depth noise
    moves a still box's result by about a millimetre from frame to frame."""

    def __init__(self, tol=None, depth_tol=DEFAULT_DEPTH_TOL, smooth=1, prompt="box", score=0.5):
        from box_masks import BoxFinder
        self.finder = BoxFinder(prompt, score)
        self.tol, self.depth_tol, self.smooth = tol, depth_tol, smooth
        self.last = None   # (points, boxes, target) of the last call, for annotate()
        self.history = []  # recent results for the current target, newest last

    def detect(self, color, depth_m, intr, up=(0.0, -1.0, 0.0), avoid=()):
        """Target dict, or None if no box has a pickable face. up: camera-frame up direction.
        avoid: outlines (bool HxW) of boxes not to choose, same camera pose (faces.choose_target)."""
        points, boxes, target, timings, _ = run(self.finder, color, depth_m, intr, np.asarray(up, float),
                                                self.tol, self.depth_tol, avoid)
        self.last = points, boxes, target
        if target is None:
            self.history = []
            return None
        p = boxes[target].pick
        now = {"xyz": np.asarray(p.centroid, float), "uv": np.asarray(p.centroid_px, float),
               "size": np.asarray(p.size_mm, float), "approach": p.kind}
        if self.history:
            mean = np.mean([h["xyz"] for h in self.history], axis=0)
            if now["approach"] != self.history[-1]["approach"] or np.linalg.norm(now["xyz"] - mean) > SAME_TARGET_M:
                self.history = []                          # a different box: start again
        self.history = (self.history + [now])[-self.smooth:]
        avg = {k: np.mean([h[k] for h in self.history], axis=0) for k in ("xyz", "uv", "size")}
        x, y, z = (round(float(v), 4) for v in avg["xyz"])
        return {"x_m": x, "y_m": y, "z_m": z, "approach": p.kind,
                "u": round(float(avg["uv"][0]), 1), "v": round(float(avg["uv"][1]), 1),
                "size_mm": [round(float(v)) for v in avg["size"]], "num_boxes": len(boxes),
                "averaged_over": len(self.history), "timings": timings}

    def detect_frames(self, frames, intr, up=(0.0, -1.0, 0.0), avoid=()):
        """[(colour, depth_m), ...] of one still scene -> one target dict, or None.

        Each frame is run on its own; the results that agree (same approach, and centre within
        SAME_TARGET_M of each other or outline in the same place) are grouped, and the largest
        group's median is returned. For a top, the frame that found the most of it is returned
        instead: a top seen at a slant often has no depth on its far part, so some frames find
        only its front strip, and that strip's centre is too near the front edge.
        self.last / self.best_frame then refer to the frame used (the one nearest the median).
        """
        smooth, self.smooth = self.smooth, 1
        try:
            found = []                                     # (frame index, result, (points, boxes, target))
            for i, (color, depth_m) in enumerate(frames):
                self.history = []
                r = self.detect(color, depth_m, intr, up, avoid)
                if r is not None:
                    found.append((i, r, self.last))
        finally:
            self.smooth, self.history = smooth, []
        if not found:
            self.last, self.best_frame = None, None
            return None
        xyz = lambda r: np.array([r["x_m"], r["y_m"], r["z_m"]])
        target = lambda g: g[2][1][g[2][2]]                       # the target Box of a found frame
        same = lambda g, f: g[1]["approach"] == f[1]["approach"] and (
            np.linalg.norm(xyz(g[1]) - xyz(f[1])) <= SAME_TARGET_M or overlap(target(g).bbox, target(f).bbox) > 0.5)
        groups = [[g for g in found if same(g, f)] for f in found]
        group = max(groups, key=len)
        if group[0][1]["approach"] == "top":
            best = max(group, key=lambda g: target(g).pick.area_mm2)
            med = {k: best[1][k] for k in ("x_m", "y_m", "z_m", "u", "v")}
            size = np.asarray(best[1]["size_mm"], float)
        else:
            med = {k: float(np.median([g[1][k] for g in group])) for k in ("x_m", "y_m", "z_m", "u", "v")}
            size = np.median([g[1]["size_mm"] for g in group], axis=0)
            best = min(group, key=lambda g: np.linalg.norm(xyz(g[1]) - np.array([med["x_m"], med["y_m"], med["z_m"]])))
        self.best_frame, self.last = best[0], best[2]
        return {"x_m": round(med["x_m"], 4), "y_m": round(med["y_m"], 4), "z_m": round(med["z_m"], 4),
                "approach": group[0][1]["approach"], "u": round(med["u"], 1), "v": round(med["v"], 1),
                "size_mm": [round(float(v)) for v in size], "num_boxes": best[1]["num_boxes"],
                "frames": len(frames), "agreeing": len(group)}

    def annotate(self, color):
        """Overlay of the last detect() call."""
        return draw(color, *self.last)


def read_frames(args):
    """(name, [(colour, depth in metres), ...], Intrinsics, accel or None)."""
    if args.live:
        from rgbd import open_camera, save
        cam = open_camera(args.ros)
        try:
            name = time.strftime("live_%Y%m%d_%H%M%S")
            frames = [cam.read() for _ in range(args.frames)]
            for k, (color, depth) in enumerate(frames):              # keep the inputs so they can be re-run
                save(Path(args.out) / (name if args.frames == 1 else f"{name}_{k:02d}"), color, depth, cam.meta())
            return name, frames, cam.intr, cam.accel
        finally:
            cam.close()
    prefixes = sorted({p[:-len("_color.png")] for pat in args.frame for p in glob.glob(pat + "_color.png")})
    if len(prefixes) > 1:                                            # several saved captures of one scene
        from rgbd import load
        loaded = [load(p) for p in prefixes]
        return Path(prefixes[0]).name, [(c, d) for c, d, _, _ in loaded], loaded[0][2], loaded[0][3]
    args.frame = prefixes[0] if prefixes else args.frame[0]
    name, color, depth, intr, accel = read_frame(args)
    return name, [(color, depth)], intr, accel


def read_frame(args):
    """(name, colour, depth in metres, Intrinsics, accel or None) of one saved frame."""
    if Path(args.frame + "_color.png").exists():                     # a capture prefix
        if Path(args.frame + "_meta.json").exists():
            from rgbd import load
            color, depth, intr, accel = load(args.frame)
            return Path(args.frame).name, color, depth, intr, accel
        args.frame, args.depth = args.frame + "_color.png", args.frame + "_depth_mm.png"   # no meta: defaults
    if not args.depth:
        raise SystemExit("give --depth <aligned 16-bit depth PNG in mm> with a colour image, or use --live")
    color = cv2.imread(args.frame)
    raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if color is None or raw is None:
        raise SystemExit(f"could not read {args.frame} or {args.depth}")
    return Path(args.frame).stem, color, raw.astype(np.float32) / 1000, D455_COLOR_1280x720, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame", nargs="*", help="capture prefix(es) or glob(s) (rgbd.py), or a colour image path")
    ap.add_argument("--live", action="store_true", help="grab frames from the D455")
    ap.add_argument("--frames", type=int, default=1, help="--live: frames to take (one answer from all)")
    ap.add_argument("--ros", action="store_true", help="--live: camera over ROS 2, through ros_bridge.py")
    ap.add_argument("--depth", help="aligned depth PNG in millimetres (with a colour image path)")
    ap.add_argument("--annotate", metavar="OUT.jpg", help="overlay path (default: <out>/<frame name>.jpg)")
    ap.add_argument("--out", default="outputs/frame", help="folder for the overlay and live inputs")
    ap.add_argument("--pitch", type=float, help="camera tilt down in degrees (default: accelerometer, else 0)")
    ap.add_argument("--depth-tol", type=float, default=DEFAULT_DEPTH_TOL, help="front-layer thickness, metres")
    args = ap.parse_args()
    if not args.live and not args.frame:
        ap.error("give a frame or --live")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    name, frames, intr, accel = read_frames(args)
    up = up_from_accel(accel) if accel and args.pitch is None else up_from_pitch(args.pitch or 0)
    detector = PickDetector(depth_tol=args.depth_tol)
    if len(frames) == 1:
        print(json.dumps(detector.detect(frames[0][0], frames[0][1], intr, up)))
        shown = frames[0][0]
    else:
        print(json.dumps(detector.detect_frames(frames, intr, up)))
        shown = frames[detector.best_frame or 0][0]
    if detector.last is not None:
        path = args.annotate or str(Path(args.out) / f"{name}.jpg")
        cv2.imwrite(path, detector.annotate(shown))
        print(f"overlay: {path}")


if __name__ == "__main__":
    main()
