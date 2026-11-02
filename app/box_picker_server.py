"""The box picker for the pick stack: box_picker/src/pick_server.py, plus the
picture of what it found (2026-10-27).

Run by box_model_node.py in box_picker/src/.venv (Python 3.11), the working
directory box_picker/src; it imports the picker unchanged. Same protocol as
pick_server.py -- "pick" / "quit" on stdin, one JSON line per answer on
stdout -- and each answer also carries:

    overlay   the frame the answer is about, clean (nothing drawn: the GUI draws
              the boxes itself), written to the --overlay file (overwritten each time)
    boxes     each box: its outline's bounding box and polygon (pixels), the pick
              face's kind, size and corners (pixels), whether it is the target,
              and why not
    size      [width, height] of that frame
    face      the target face in 3D (camera frame): normal (out of the face),
              long (unit, along its long edge), size_mm (long, short)

and one more request, the picker's own pick check (pick_check.py, as
pick_loop.py uses it):

    pick ux uy uz        the same, with the up direction (colour camera frame) of now
    check [held|empty]   after the pick, with the arm out of the view: 3 new
              frames against the 5 of the last "pick", inside the target's
              outline -> {"check": verdict, "picked": true / false / null,
              "message", "title", "why", "outline" (the target's polygon),
              "overlay" (the newest frame, clean)} (null: can't tell)

    .venv/bin/python ../../box_picker_server.py --pitch -4.5 --ros --overlay /tmp/x.jpg
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def face_axes(face, intr):
    """The face's rectangle back in 3D: its corners (drawn into the image from
    the plane) put back on the plane -> normal, long edge's direction, size."""
    n = np.asarray(face.normal, float)
    c = np.asarray(face.centroid, float)
    pts = []
    for u, v in face.corners_px:
        ray = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
        pts.append(ray * (n @ c) / (n @ ray))
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    long = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    long = long - (long @ n) * n
    long /= np.linalg.norm(long)
    r = lambda a: [round(float(x), 4) for x in a]
    return {"normal": r(n), "long": r(long), "size_mm": [round(float(v)) for v in face.size_mm],
            "kind": face.kind}


def check(pc, picked_of, cam, before, words, args):
    """The "check" request: did the last pick take its box?"""
    if before is None:
        return {"check": None, "picked": None,
                "message": "nothing to check: no box was chosen by the last pick"}
    holding = {"held": True, "empty": False}.get(words[0]) if words else None
    for _ in range(args.skip):                        # frames queued while waiting
        cam.read()
    after = [cam.read() for _ in range(3)]
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        verdict = pc.check(before, after, holding)
        cv2.imwrite(args.overlay, after[-1][0])          # clean: the GUI draws
    finally:
        sys.stdout = real_stdout
    title, why, _colour = pc.RESULT[verdict["outcome"]]
    return {"check": verdict, "picked": picked_of.get(verdict["outcome"]),
            "message": f"{title}: {why}", "title": title, "why": why,
            "outline": outline(before.target), "overlay": args.overlay,
            "size": [int(after[-1][0].shape[1]), int(after[-1][0].shape[0])],
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def outline(mask):
    """A box's outline for the GUI, as the model gives it: the contour of its
    mask (largest part, pixels), unchanged in shape -- only straight runs
    merged (under 1 px), to keep the message small."""
    if mask is None or not mask.any():
        return None
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.8, True)
    return [[int(p[0][0]), int(p[0][1])] for p in c]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--ros", action="store_true")
    ap.add_argument("--overlay", required=True, help="where to write the drawing of each answer")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())                    # box_picker/src
    real_stdout, sys.stdout = sys.stdout, sys.stderr   # libraries' output -> stderr
    import pick_check as pc
    from faces import up_from_pitch
    from pick_and_check import PICKED
    from pick_frame import PickDetector
    from rgbd import open_camera
    detector = PickDetector()
    cam = open_camera(args.ros)
    up = up_from_pitch(args.pitch)
    sys.stdout = real_stdout
    reply({"ready": True, "frames": args.frames, "up": [round(float(v), 3) for v in up]})
    before = None                                      # the last pick, for "check"

    try:
        for line in sys.stdin:
            words = line.strip().lower().split()
            cmd = words[0] if words else ""
            if cmd in ("quit", "exit"):
                break
            if cmd == "check":
                reply(check(pc, PICKED, cam, before, words[1:], args))
                continue
            if cmd != "pick":
                reply({"error": f"unknown request {cmd!r}: use pick, check or quit"})
                continue
            # "pick ux uy uz": the up direction now (colour camera frame), from the
            # camera's accelerometer -- else the one from --pitch
            up_now = up
            if len(words) == 4:
                try:
                    v = np.array([float(x) for x in words[1:4]])
                    if np.linalg.norm(v) > 0.5:
                        up_now = v / np.linalg.norm(v)
                except ValueError:
                    pass
            t0 = time.time()
            for _ in range(args.skip):
                cam.read()
            frames = [cam.read() for _ in range(args.frames)]
            sys.stdout = sys.stderr
            try:
                result = detector.detect_frames(frames, cam.intr, up_now)
                before = pc.remember(detector, frames, result)
                boxes, overlay, face = [], None, None
                if detector.last is not None:
                    points, found, target = detector.last
                    for i, b in enumerate(found):
                        why = []
                        face_px = (b.pick.corners_px if b.pick is not None and
                                   b.pick.corners_px is not None else None)
                        if getattr(b, "blocked_by", None):
                            why.append("a box on it")
                        if getattr(b, "hidden_by", None):
                            why.append("a box in front")
                        if not b.pick:
                            why.append("no front or top face to grip")
                        boxes.append({"bbox": [int(v) for v in b.bbox],
                                      "poly": outline(b.mask),
                                      "face": ([[round(float(u), 1), round(float(v), 1)] for u, v in face_px]
                                               if face_px is not None else None),
                                      "kind": b.pick.kind if b.pick else None,
                                      "size_mm": [round(float(v)) for v in b.pick.size_mm] if b.pick else None,
                                      "target": i == target, "why_not": why})
                    if target is not None and found[target].pick is not None:
                        face = face_axes(found[target].pick, cam.intr)
                    img = frames[detector.best_frame or 0][0]   # clean: the GUI draws
                    cv2.imwrite(args.overlay, img)
                    overlay = args.overlay
                    # the frame the target came from, with its depth: to measure a
                    # miss afterwards (one file, overwritten; ~1 MB)
                    try:
                        bf = frames[detector.best_frame or 0]
                        np.savez_compressed(os.path.splitext(args.overlay)[0] + "_last.npz",
                                            color=bf[0], depth=bf[1].astype(np.float32),
                                            intr=np.array([cam.intr.fx, cam.intr.fy,
                                                           cam.intr.cx, cam.intr.cy]),
                                            result=json.dumps(result))
                    except Exception:                    # noqa: BLE001 - only a record
                        pass
                elif frames:
                    cv2.imwrite(args.overlay, frames[-1][0])
                    overlay = args.overlay
            finally:
                sys.stdout = real_stdout
            size = [int(frames[0][0].shape[1]), int(frames[0][0].shape[0])] if frames else None
            reply({"target": result, "boxes": boxes, "face": face, "overlay": overlay, "size": size,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 2)})
    finally:
        cam.close()


if __name__ == "__main__":
    main()
