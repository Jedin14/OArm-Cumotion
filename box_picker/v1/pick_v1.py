"""Run v1 on recorded frames: the pick order, the face to grip each box by, and the box size.

Uses the boxes saved in cache/ (detect_cache.py) when they are there, else finds them with
SAM 3.1 (needs the GPU, ~4 GB, ~5 s to load). Prints one JSON line per frame and saves a
picture of it (outputs/pick_v1/<recording>_<frame>.jpg, or --out).

v2: only the boxes in the carton in front are used (carton.py); --no-carton uses every box.

Usage (from box_picking/v1/):
    PY=../../vlm/molmo2-box-picking/.venv/bin/python
    R=../../vlm/molmo2-box-picking/recordings
    $PY pick_v1.py $R/grid_part2 7 8 frame_00009
    $PY pick_v1.py $R/grid_part3 0-5 --out /tmp/v1
    $PY pick_v1.py $R/grid_part2 7 --sam            # find the boxes again, not from cache/

Output, one line per frame:
    {"frame", "box_size_mm": [w, h, d or null], "grid_boxes",
     "carton": null (not checked) | {"found": false} | {"found": true, "boxes_in", "left_out", "near_m"},
     "picks": [{"rank", "face": "front" | "top", "fallen", "u", "v", "x_m", "y_m", "z_m",
                "face_mm": [long, short], "cell": [layer, row, column] or null}, ...]}
    x right, y down, z away from the camera, in metres; u, v the same point in the picture.
    cell counts from 1: layer 1 = the front, row 1 = the highest row now in that layer,
    column 1 = the leftmost box seen. Fallen boxes have no cell.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import carton
import grid_order
from detect_cache import find_boxes, load_cache, load_frame

HERE = Path(__file__).parent


def stems(specs):
    """'7', '0-5', 'frame_00009' -> frame stems."""
    out = []
    for s in specs:
        s = s.removeprefix("frame_")
        if "-" in s:
            a, b = s.split("-")
            out += [f"frame_{n:05d}" for n in range(int(a), int(b) + 1)]
        else:
            out.append(f"frame_{int(s):05d}")
    return out


def draw(color, boxes, res):
    img = color.copy()
    for b in boxes:
        cs, _ = cv2.findContours(b.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (0, 220, 255), 1)
    found = res.get("carton")
    carton.draw(img, found, boxes)
    for k, p in reversed(list(enumerate(res["picks"]))):
        col = (0, 0, 255) if k == 0 else (0, 165, 255) if k < 4 else (160, 160, 160)
        face_col = (0, 200, 0) if p["face"] == "front" else (255, 120, 0)
        if k < 4:
            cv2.polylines(img, [np.int32(p["corners_px"])], True, face_col, 2)
        u, v = map(int, p["uv"])
        cv2.circle(img, (u, v), 6 if k < 4 else 3, col, 2)
        label = f"{k + 1}" + (" F" if p["fallen"] else "")
        cv2.putText(img, label, (u + 6, v - 6), 0, 0.55, (0, 0, 0), 4)
        cv2.putText(img, label, (u + 6, v - 6), 0, 0.55, col, 2)
    w, h, d = res["box_size_mm"] or (None, None, None)
    size = "?" if w is None else f"{w:.0f} x {h:.0f} x {'?' if d is None else f'{d:.0f}'} mm"
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, f"box {size}   {len(res['picks'])} to pick   red 1 = now, orange 2-4 next, F fallen;"
                     f" green face = front, blue = top", (6, 17), 0, 0.42, (255, 255, 255), 1)
    if found is not None:
        line = carton.message(res)[0] + (" (white; boxes outside it grey)" if found["found"] else "")
        cv2.rectangle(img, (0, img.shape[0] - 22), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, line, (6, img.shape[0] - 7), 0, 0.42, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("frames", nargs="+", help="frame numbers (7), ranges (0-5) or names (frame_00007)")
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "pick_v1")
    ap.add_argument("--sam", action="store_true", help="find the boxes with SAM 3.1 even if cache/ has them")
    ap.add_argument("--score", type=float, default=0.4, help="SAM confidence cut-off")
    ap.add_argument("--no-carton", action="store_true", help="use every box, not only those in the carton in front")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    finder = None
    for stem in stems(args.frames):
        color, depth, intr, up = load_frame(args.recording, stem)
        cached = HERE / "cache" / args.recording.name / f"{stem}.pkl.gz"
        if cached.exists() and not args.sam:
            d = load_cache(cached)
            boxes, cartons = d["boxes"], d.get("cartons")
            if cartons is None and not args.no_carton:
                print(f"{stem}: no cartons in cache/ (detect_cache.py --cartons): every box used", file=sys.stderr)
        else:
            if finder is None:
                from box_masks import BoxFinder
                finder = BoxFinder("box", args.score)
                carton.keep_image(finder)
            boxes = find_boxes(finder, color, depth, intr, up)[3]
            cartons = None if args.no_carton else carton.find_cartons(finder)
        res = grid_order.order({"boxes": boxes, "intr": intr, "up": up, "depth": depth,
                                "cartons": None if args.no_carton else cartons})
        size = res["box_size_mm"]
        print(json.dumps({
            "frame": stem,
            "box_size_mm": None if size is None else [None if v is None else round(v) for v in size],
            "grid_boxes": res["grid_boxes"],
            "carton": carton.summary(res["carton"]),
            "picks": [{"rank": k + 1, "face": p["face"], "fallen": p["fallen"],
                       "u": round(p["uv"][0], 1), "v": round(p["uv"][1], 1),
                       "x_m": round(p["xyz"][0], 4), "y_m": round(p["xyz"][1], 4), "z_m": round(p["xyz"][2], 4),
                       "face_mm": [round(v) for v in p["size_mm"]],
                       "cell": None if p["cell"] is None else [c + 1 for c in p["cell"]]}
                      for k, p in enumerate(res["picks"])]}), flush=True)
        cv2.imwrite(str(args.out / f"{args.recording.name}_{stem}.jpg"), draw(color, boxes, res))


if __name__ == "__main__":
    main()
