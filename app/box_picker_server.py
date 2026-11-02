"""The box picker for the pick stack: box_picker/src/pick_server.py, plus the
picture of what it found (2026-10-27).

Run by box_model_node.py in box_picker/src/.venv (Python 3.11), the working
directory box_picker/src; it imports the picker unchanged. Same protocol as
pick_server.py -- "pick" / "quit" on stdin, one JSON line per answer on
stdout -- and each answer also carries:

    overlay   the picker's own drawing of its decision (every box outlined, the
              faces tinted by kind, the chosen face in red), written to the
              --overlay file (one file, overwritten each time)
    boxes     each box: its outline's bounding box (pixels), the pick face's
              kind and size, whether it is the target, and why not
    face      the target face in 3D (camera frame): normal (out of the face),
              long (unit, along its long edge), size_mm (long, short)

and one more request, the picker's own pick check (pick_check.py, as
pick_loop.py uses it):

    check [held|empty]   after the pick, with the arm out of the view: 3 new
              frames against the 5 of the last "pick", inside the target's
              outline -> {"check": verdict, "picked": true / false / null,
              "message", "overlay"} (null: can't tell -- blocked, camera moved)

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
        img = pc.big_result(pc.draw(before, after, verdict), verdict)
        cv2.imwrite(args.overlay, img)
    finally:
        sys.stdout = real_stdout
    title, why, _colour = pc.RESULT[verdict["outcome"]]
    return {"check": verdict, "picked": picked_of.get(verdict["outcome"]),
            "message": f"{title}: {why}", "overlay": args.overlay,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


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
            t0 = time.time()
            for _ in range(args.skip):
                cam.read()
            frames = [cam.read() for _ in range(args.frames)]
            sys.stdout = sys.stderr
            try:
                result = detector.detect_frames(frames, cam.intr, up)
                before = pc.remember(detector, frames, result)
                boxes, overlay, face = [], None, None
                if detector.last is not None:
                    points, found, target = detector.last
                    for i, b in enumerate(found):
                        why = []
                        if getattr(b, "blocked_by", None):
                            why.append("a box on it")
                        if getattr(b, "hidden_by", None):
                            why.append("a box in front")
                        if not b.pick:
                            why.append("no front or top face to grip")
                        boxes.append({"bbox": [int(v) for v in b.bbox],
                                      "kind": b.pick.kind if b.pick else None,
                                      "size_mm": [round(float(v)) for v in b.pick.size_mm] if b.pick else None,
                                      "target": i == target, "why_not": why})
                    if target is not None and found[target].pick is not None:
                        face = face_axes(found[target].pick, cam.intr)
                    img = detector.annotate(frames[detector.best_frame or 0][0])
                    cv2.imwrite(args.overlay, img)
                    overlay = args.overlay
                elif frames:
                    cv2.imwrite(args.overlay, frames[-1][0])
                    overlay = args.overlay
            finally:
                sys.stdout = real_stdout
            reply({"target": result, "boxes": boxes, "face": face, "overlay": overlay,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 2)})
    finally:
        cam.close()


if __name__ == "__main__":
    main()
