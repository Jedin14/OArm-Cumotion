"""Full pick pipeline: SAM 3.1 box outlines -> depth faces -> pick face -> target centroid.

Rules: docs/picking_rules.md. Input is a saved capture (rgbd.py) or the live D455.
For every frame it writes <name>.json (target + all boxes and faces),
<name>_masks.npz (the box outlines) and <name>.jpg (box outlines thin yellow;
faces green = front, blue = top, grey = side; each pick face's measured
rectangle and centroid in white, the target's in red).

The up direction comes from the D455 accelerometer when it can be read, else from
--pitch (camera tilted down by that many degrees; 0 = level).

Usage:
    python pick_pipeline.py --capture captures/desk
    python pick_pipeline.py --capture "captures/desk_*"            # several captures
    python pick_pipeline.py --live --count 5                       # 5 frames from the camera
    python pick_pipeline.py --capture "recordings/test2_d/frame_*" --frames 5   # one answer per 5 frames
"""
import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np

from faces import DEFAULT_DEPTH_TOL, Intrinsics, analyse, choose_target, split_doubles, up_from_accel, up_from_pitch

FACE_COLORS = {"front": (0, 200, 0), "top": (255, 120, 0), "side": (150, 150, 150)}   # BGR


def run(finder, color, depth, intr, up, tol, depth_tol, avoid=()):
    points, masks, timings, scores = finder(color)
    t0 = time.time()
    boxes = analyse(depth, masks, intr, up)
    # Two boxes pressed together in one outline: split, and measured again.
    halves = split_doubles(boxes)
    if any(len(h) == 2 for h in halves):
        centre = lambda m: (float(np.nonzero(m)[1].mean()), float(np.nonzero(m)[0].mean()))
        points = [q for p, h in zip(points, halves) for q in ([p] if len(h) == 1 else [centre(m) for m in h])]
        scores = [sc for sc, h in zip(scores, halves) for sc in [sc] * len(h)]
        masks = [m for h in halves for m in h]
        boxes = analyse(depth, masks, intr, up)
    target = choose_target(boxes, tol, depth_tol, avoid)
    timings["depth_s"] = round(time.time() - t0, 3)
    return points, boxes, target, timings, scores


def face_json(f):
    return {"kind": f.kind, "angle_deg": f.angle_deg, "size_mm": f.size_mm, "points": len(f.pixels),
            "centroid_m": [round(float(v), 4) for v in f.centroid],
            "centroid_px": [round(float(v), 1) for v in f.centroid_px]}


def result_json(name, points, boxes, target, timings, up, scores):
    t = None
    if target is not None:
        p = boxes[target].pick
        t = {"box": target, "approach": p.kind, **face_json(p)}
    return {
        "frame": name, "target": t, "up": [round(float(v), 3) for v in up], "timings": timings,
        "boxes": [{"point_px": [round(x, 1), round(y, 1)], "score": s, "bbox": b.bbox, "depth_m": round(b.depth, 4),
                   "blocked_by": b.blocked_by, "hidden_by": b.hidden_by,
                   "pick": b.faces.index(b.pick) if b.pick else None, "faces": [face_json(f) for f in b.faces]}
                  for (x, y), s, b in zip(points, scores, boxes)],
    }


def draw(color, points, boxes, target):
    img = color.copy()
    tint = img.copy()
    for b in boxes:
        for f in b.faces:
            tint[f.pixels[:, 1], f.pixels[:, 0]] = FACE_COLORS[f.kind]
    img = cv2.addWeighted(tint, 0.45, img, 0.55, 0)
    for i, ((x, y), b) in enumerate(zip(points, boxes)):
        is_t = i == target
        contours, _ = cv2.findContours(b.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 255), 1)       # whole box (SAM 3.1 outline)
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
        if b.pick:
            # The measured rectangle of the pick face: what the size and centroid come from.
            cv2.polylines(img, [np.round(b.pick.corners_px).astype(np.int32)], True,
                          (0, 0, 255) if is_t else (255, 255, 255), 3 if is_t else 1)
            c = tuple(int(round(v)) for v in b.pick.centroid_px)
            cv2.drawMarker(img, c, (0, 0, 255) if is_t else (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            label = f"{b.pick.kind} {b.pick.size_mm[0]:.0f}x{b.pick.size_mm[1]:.0f}" + (" (box on it)" if b.blocked_by else "") + (" (box in front)" if b.hidden_by else "")
            cv2.putText(img, label, (b.bbox[0], b.bbox[3] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if target is not None:
        p = boxes[target].pick
        x, y, z = p.centroid
        msg = f"TARGET box {target}: {p.kind}, centroid ({x:+.3f}, {y:+.3f}, {z:.3f}) m"
    else:
        msg = "no target"
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if target is not None else (255, 255, 255), 2)
    return img


def frames_from(args):
    """Yield (name, colour, depth, Intrinsics, up)."""
    def up_for(accel):
        return up_from_accel(accel) if accel and args.pitch is None else up_from_pitch(args.pitch or 0)

    if args.live:
        from rgbd import open_camera, save
        cam = open_camera(args.ros)
        try:
            for k in range(args.count):
                color, depth = cam.read()
                name = time.strftime("live_%Y%m%d_%H%M%S")
                save(Path(args.out) / name, color, depth, cam.meta())   # keep the input so it can be re-run
                yield name, color, depth, cam.intr, up_for(cam.accel)
        finally:
            cam.close()
        return
    from rgbd import load
    prefixes = sorted({p[: -len("_color.png")] for pat in args.capture for p in glob.glob(pat + "_color.png")})
    if not prefixes:
        raise SystemExit(f"no captures match {args.capture} (expected <prefix>_color.png)")
    for prefix in prefixes:
        if Path(prefix + "_meta.json").exists():
            color, depth, intr, accel = load(prefix)
        else:   # older samples without meta: depth png in mm + intrinsics from the command line
            color = cv2.imread(prefix + "_color.png")
            depth = cv2.imread(prefix + "_depth_mm.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            intr, accel = Intrinsics(args.fx, args.fy, args.cx, args.cy), None
        yield Path(prefix).name, color, depth, intr, up_for(accel)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", nargs="+", help="capture prefix(es) or glob(s), as written by rgbd.py")
    src.add_argument("--live", action="store_true", help="read frames from the D455")
    ap.add_argument("--count", type=int, default=1, help="--live: number of frames")
    ap.add_argument("--ros", action="store_true", help="--live: camera over ROS 2, through ros_bridge.py")
    ap.add_argument("--pitch", type=float, help="camera tilt down in degrees (default: accelerometer, else 0)")
    ap.add_argument("--depth-tol", type=float, default=DEFAULT_DEPTH_TOL, help="front-layer thickness, metres")
    ap.add_argument("--prompt", default="box", help="SAM 3.1 text prompt")
    ap.add_argument("--score", type=float, default=0.5, help="SAM 3.1 confidence threshold")
    ap.add_argument("--fx", type=float, default=645.9)
    ap.add_argument("--fy", type=float, default=644.8)
    ap.add_argument("--cx", type=float, default=643.9)
    ap.add_argument("--cy", type=float, default=369.4)
    ap.add_argument("--frames", type=int, default=1,
                    help="--capture: one answer per this many consecutive frames, as the robot asks "
                         "after it stops (PickDetector.detect_frames)")
    ap.add_argument("--out", default="outputs/pipeline")
    args = ap.parse_args()
    if args.frames > 1:
        if args.live:
            ap.error("--frames is for --capture; live, use pick_frame.py --live --frames N")
        return run_groups(args)

    from box_masks import BoxFinder
    finder = BoxFinder(args.prompt, args.score)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, color, depth, intr, up in frames_from(args):
        points, boxes, target, timings, scores = run(finder, color, depth, intr, up, None, args.depth_tol)
        res = result_json(name, points, boxes, target, timings, up, scores)
        (out / f"{name}.json").write_text(json.dumps(res, indent=2))
        cv2.imwrite(str(out / f"{name}.jpg"), draw(color, points, boxes, target))
        np.savez_compressed(out / f"{name}_masks.npz", masks=np.array([b.mask for b in boxes]))
        kinds = [[f.kind for f in b.faces] for b in boxes]
        print(f"[{name}] {len(boxes)} boxes {kinds}  target: {json.dumps(res['target'])}  {timings}")


def run_groups(args):
    """One answer per args.frames consecutive frames (in name order): <first frame>.json with the
    answer and the frames used, and <first frame>.jpg, the overlay of the frame nearest it."""
    from pick_frame import PickDetector
    detector = PickDetector(depth_tol=args.depth_tol, prompt=args.prompt, score=args.score)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    group = []
    for item in frames_from(args):
        group.append(item)
        if len(group) < args.frames:
            continue
        names = [g[0] for g in group]
        intr, up = group[0][3], group[0][4]
        t0 = time.time()
        result = detector.detect_frames([(g[1], g[2]) for g in group], intr, up)
        (out / f"{names[0]}.json").write_text(json.dumps({"frames": names, "target": result}, indent=2))
        if detector.last is not None:
            cv2.imwrite(str(out / f"{names[0]}.jpg"), detector.annotate(group[detector.best_frame][1]))
        print(f"[{names[0]} .. {names[-1]}] {json.dumps(result)}  {time.time() - t0:.1f}s", flush=True)
        group = []
    if group:
        print(f"last {len(group)} frame(s) left over (fewer than --frames {args.frames})")


if __name__ == "__main__":
    main()
