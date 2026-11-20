"""Find the boxes once in every labelled frame and save them, so the pick order can be
tried again and again without running SAM 3.1 (about 0.5 s a frame on the GPU).

For each frame in a recording's labels.json it runs the copied box-finding code:
SAM 3.1 outlines (box_masks.py), then each box's faces from depth (faces.analyse), and
two boxes pressed together in one outline split (faces.split_doubles), as pick_pipeline.run
does, and the cartons in view (carton.find_cartons, for v2). It saves cache/<recording>/<frame>.pkl.gz:
    {"stem", "shape", "intr": (fx, fy, cx, cy), "up", "points", "scores", "masks" (packed),
     "boxes": [faces.Box with mask=None and face pixels as int16], "sam_s",
     "cartons": {"masks" (packed), "scores"}}

Usage (from box_picking/v1/):
    ../../vlm/molmo2-box-picking/.venv/bin/python detect_cache.py ../../vlm/molmo2-box-picking/recordings/grid_part2
    ... detect_cache.py <recording> --score 0.4 --redo
    ... detect_cache.py <recording> --cartons      # add the cartons to frames saved without them
"""
import argparse
import gzip
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from faces import Intrinsics, analyse, split_doubles, up_from_accel, up_from_pitch

HERE = Path(__file__).parent


def load_frame(folder, stem):
    """colour BGR, depth in metres, Intrinsics, up (camera frame), from a recorded frame.
    A frame with no accelerometer reading (the camera over ROS 2 may send none) counts as level."""
    meta = json.loads((folder / f"{stem}_meta.json").read_text())
    color = cv2.imread(str(folder / f"{stem}_color.png"))
    depth = cv2.imread(str(folder / f"{stem}_depth_mm.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    intr = Intrinsics(meta["fx"], meta["fy"], meta["cx"], meta["cy"])
    return color, depth, intr, up_from_accel(meta["accel"]) if meta.get("accel") else up_from_pitch(0)


def find_boxes(finder, color, depth, intr, up):
    """pick_pipeline.run without choosing a target -> points, masks, scores, boxes, seconds in SAM."""
    points, masks, timings, scores = finder(color, depth)
    boxes = analyse(depth, masks, intr, up)
    halves = split_doubles(boxes)
    if any(len(h) == 2 for h in halves):
        centre = lambda m: (float(np.nonzero(m)[1].mean()), float(np.nonzero(m)[0].mean()))
        points = [q for p, h in zip(points, halves) for q in ([p] if len(h) == 1 else [centre(m) for m in h])]
        scores = [sc for sc, h in zip(scores, halves) for sc in [sc] * len(h)]
        masks = [m for h in halves for m in h]
        boxes = analyse(depth, masks, intr, up)
    return points, masks, scores, boxes, timings["sam_s"]


def slim(box):
    """A Box without its outline (saved packed beside it) and with small face pixel arrays."""
    faces = [replace(f, pixels=f.pixels.astype(np.int16)) for f in box.faces]
    pick = None
    if box.pick is not None:
        pick = next((g for f, g in zip(box.faces, faces) if f is box.pick), replace(box.pick, pixels=box.pick.pixels.astype(np.int16)))
    return replace(box, mask=None, faces=faces, pick=pick)


def load_cache(path):
    """-> the saved dict, with each Box's mask (bool HxW) unpacked again."""
    with gzip.open(path, "rb") as fh:
        d = pickle.load(fh)
    h, w = d["shape"]
    unpack = lambda packed: np.unpackbits(packed)[: h * w].reshape(h, w).astype(bool)
    for b, packed in zip(d["boxes"], d["masks"]):
        b.mask = unpack(packed)
    if "cartons" in d:
        d["cartons"] = [(unpack(m), s) for m, s in zip(d["cartons"]["masks"], d["cartons"]["scores"])]
    return d


def save_cache(path, d):
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=3) as fh:
        pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def pack_cartons(cartons):
    return {"masks": [np.packbits(m.ravel()) for m, _ in cartons], "scores": [s for _, s in cartons]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--score", type=float, default=0.4, help="SAM confidence cut-off (pick_loop uses 0.4)")
    ap.add_argument("--redo", action="store_true", help="run again on frames already saved")
    ap.add_argument("--cartons", action="store_true",
                    help="only add the cartons to frames saved without them (the boxes stay as saved)")
    args = ap.parse_args()
    labels = json.loads((args.recording / "labels.json").read_text())["frames"]
    out = HERE / "cache" / args.recording.name
    out.mkdir(parents=True, exist_ok=True)
    if args.cartons:
        todo = [s for s in sorted(labels) if (out / f"{s}.pkl.gz").exists()
                and (args.redo or "cartons" not in load_cache(out / f"{s}.pkl.gz"))]
    else:
        todo = [s for s in sorted(labels) if args.redo or not (out / f"{s}.pkl.gz").exists()]
    print(f"{args.recording.name}: {len(labels)} labelled frames, {len(todo)} to run", flush=True)
    if not todo:
        return
    from box_masks import BoxFinder
    from carton import find_cartons, keep_image
    finder = BoxFinder("box", args.score)
    keep_image(finder)
    t0 = time.time()
    for n, stem in enumerate(todo, 1):
        color, depth, intr, up = load_frame(args.recording, stem)
        if args.cartons:
            d = load_cache(out / f"{stem}.pkl.gz")
            for b in d["boxes"]:
                b.mask = None
            d["cartons"] = pack_cartons(find_cartons(finder, color))
            save_cache(out / f"{stem}.pkl.gz", d)
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {len(d['cartons']['scores'])} cartons in {stem}", flush=True)
            continue
        points, masks, scores, boxes, sam_s = find_boxes(finder, color, depth, intr, up)
        d = {"stem": stem, "shape": depth.shape, "intr": (intr.fx, intr.fy, intr.cx, intr.cy),
             "up": np.asarray(up, float), "points": points, "scores": scores,
             "masks": [np.packbits(m.ravel()) for m in masks], "boxes": [slim(b) for b in boxes], "sam_s": sam_s,
             "cartons": pack_cartons(find_cartons(finder))}
        save_cache(out / f"{stem}.pkl.gz", d)
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {len(boxes)} boxes in {stem}  {(time.time() - t0) / n:.2f} s/frame", flush=True)


if __name__ == "__main__":
    main()
