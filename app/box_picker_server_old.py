"""The box picker for the pick stack: box_picker/src/pick_server.py, plus the
picture of what it found (2026-10-27), and what pick_loop.py's window does
(2026-11-04).

Run by box_model_node.py in box_picker/src/.venv (Python 3.11), the working
directory box_picker/src; it imports the picker unchanged. Same protocol as
pick_server.py -- "pick" / "quit" on stdin, one JSON line per answer on
stdout -- and each answer also carries:

    overlay   the frame the answer is about, clean (nothing drawn: the GUI draws
              the boxes itself), written to the --overlay file (overwritten each time)
    picker_view  the picker's own drawing of it (every box, its faces, the grip
              points: pick_loop's V), <overlay>_picker.jpg
    close_up  the target box cut out, its pick face boxed in red, <overlay>_crop.jpg
    boxes     each box: its outline's bounding box and polygon (pixels), the pick
              face's kind, size and corners (pixels), whether it is the target,
              and why not
    size      [width, height] of that frame
    face      the target face in 3D (camera frame): normal (out of the face),
              long (unit, along its long edge), size_mm (long, short)
    took      {pictures, picker}: s taking the frames, s choosing the target
    settings  frames, check_frames, score, order, top_faces, skipped, passed

Requests, one line each; every answer carries "settings":

    pick ux uy uz [saved]  the same, with the up direction (colour camera frame) of now;
              "saved": the accelerometer's saved reading, so one sent with the frames
              (ros_bridge.py's) goes first; without any, the --pitch one (the TF)
    check [held|empty]   after the pick, with the arm out of the view: the
              result of the pick check (2026-11-11: the Pick Check team's new one,
              box_picker/src/pick_watch.py). From the "pick" on, a thread feeds it
              every camera frame: the target's pick face (arm in front of it, still
              there, gone), then once the box has left its spot the whole view, until
              it settles. "check" waits for its decision, up to CHECK_WAIT_S ->
              {"check": verdict, "picked": true / false / null, "message", "title",
              "why", "outline" (the target's polygon), "overlay" (the newest frame,
              clean), "picker_view" (the check's picture: the pick face in the
              result's colour, where a dropped box fell outlined), "frames"}.
              verdict["outcome"]: "success" (left its spot, then nothing new stayed
              in view), "missed" (the arm came and went, the box is still there),
              "moved" (knocked or dragged), "dropped" (left its spot, then a new
              surface stood still in the carton: it fell back in), "blocked" (no
              decision yet: null). held / empty, the gripper's word, wins over the
              camera: held -> "success", empty -> "dropped". A success forgets the
              passed boxes. check_frames is no longer used (kept for the GUI)
    set NAME VALUE       frames (1-15), check_frames (1-10), score (0.20-0.90),
              order (v1 / v2), top_faces (on / off: pick_rules.py -- off, a box
              on a box by its front, a top only as a last resort, "top_only";
              on, the larger of front and top anywhere, as before 5 Nov 2026):
              from the next pick / check on
    pass      the last target, and every box under it, is passed over until the
              next successful pick (pick_loop's T); with no other box left, the
              first again
    skip      the last target is never chosen again (until reset), but still
              blocks the boxes behind and under it
    reset     forget the skipped and passed boxes
    save [ux uy uz]  the newest colour + depth, as rgbd.save (pick_pipeline
              --capture reads it), in <out>/snapshots/ -> {"path"}
    flag      three photos of the last target for a person: the scene with it
              marked, the box close up, the last check -> {"photos"}
    settings  just the settings
    -> each: {"cmd", "ok", "message", "settings", ...}

As pick_loop.py (2026-11-03): SAM's score 0.4 (--score; thin boxes standing on
end score 0.45-0.6), --order v1 (stack tops, highest first, then nearest, then
leftmost) or v2 (the leftmost free box).

    .venv/bin/python ../../box_picker_server.py --pitch -4.5 --ros --overlay /tmp/x.jpg
"""
import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

LIMITS = {"frames": (1, 15, int), "check_frames": (1, 10, int), "score": (0.2, 0.9, float)}
RED = (0, 0, 255)                                       # BGR, the target, as the picker draws it
CHECK_WAIT_S = 15.0      # "check" waits this long for the pick check's decision (the node waits 20 s)
# The new pick check's results, in the old check's words (box_model_node / the GUI read "outcome"):
# outcome -> (picked, title, why, colour BGR)
RESULT = {"success": (True, "PICK SUCCESS", "the box left its spot and the camera's view", (0, 200, 0)),
          "missed": (False, "PICK FAILED: MISSED", "the arm came and went, the box is still there", (0, 0, 255)),
          "moved": (False, "PICK FAILED: BOX MOVED", "the box was knocked or dragged, not picked", (0, 0, 255)),
          "dropped": (False, "PICK FAILED: DROPPED", "the box was lifted but fell back into the carton", (0, 0, 255)),
          "blocked": (None, "CAN'T TELL YET", "the pick check has not decided: arm still in view, "
                                                "or the box has not left its spot", (0, 165, 255))}
OUTCOME = {"picked": "success", "missed": "missed", "moved": "moved", "dropped": "dropped"}


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


def contours(mask):
    return cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def close_up(img, mask, face_px=None, pad=30):
    """The target box cut out of img with a margin, its pick face boxed in red
    (loop_screen.close_up)."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    out = img.copy()
    rect = [np.round(np.asarray(face_px)).astype(np.int32)] if face_px is not None else contours(mask)
    cv2.drawContours(out, rect, -1, RED, 2, cv2.LINE_AA)
    return out[max(ys.min() - pad, 0):min(ys.max() + pad, h), max(xs.min() - pad, 0):min(xs.max() + pad, w)]


def check_view(colour, watcher, outcome, mask):
    """The check's picture: the target's outline and its pick face in the result's
    colour, and for "dropped" where the box fell (purple)."""
    col = RESULT[outcome][3]
    img = colour.copy()
    cv2.drawContours(img, contours(mask), -1, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.round(watcher.corners_px).astype(np.int32)], True, col, 4)
    if outcome == "dropped" and watcher.drop is not None:
        fell = cv2.resize(watcher.drop.astype(np.uint8), (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        cv2.drawContours(img, contours(fell), -1, (200, 0, 130), 3, cv2.LINE_AA)
    return img


class Target:
    """What is kept of the last pick's target."""
    def __init__(self, target, result):
        self.target, self.result = target, result        # the box's outline (bool HxW); the answer


class LockedCamera:
    """The camera, read by one thread at a time (the pick check's thread and the requests)."""
    def __init__(self, cam):
        self._cam, self._lock = cam, threading.Lock()

    def read(self):
        with self._lock:
            return self._cam.read()

    def __getattr__(self, name):                         # intr, accel, meta, close
        return getattr(self._cam, name)


class Watch(threading.Thread):
    """The pick check on every camera frame, from the "pick" until it has decided."""
    def __init__(self, cam, watcher):
        super().__init__(daemon=True)
        self.cam, self.watcher = cam, watcher
        self.result = self.error = self.last = None
        self.frames, self.stopped, self.done = 0, False, threading.Event()
        self.started = time.time()

    def run(self):
        try:
            while not self.stopped:
                colour, depth = self.cam.read()
                self.last, self.frames = (colour, depth), self.frames + 1
                self.result = self.watcher.update(depth, self.cam.intr)
                if self.result is not None:
                    break
        except Exception as exc:                         # noqa: BLE001 - told at the next "check"
            self.error = exc
        finally:
            self.done.set()

    def stop(self):
        self.stopped = True
        self.done.wait(5)


def outline(mask):
    """A box's outline for the GUI, as the model gives it: the contour of its
    mask (largest part, pixels), unchanged in shape -- only straight runs
    merged (under 1 px), to keep the message small."""
    if mask is None or not mask.any():
        return None
    found = contours(mask)
    if not found:
        return None
    c = cv2.approxPolyDP(max(found, key=cv2.contourArea), 0.8, True)
    return [[int(p[0][0]), int(p[0][1])] for p in c]


def up_of(words):
    """"ux uy uz" -> a unit vector, or None."""
    if len(words) != 3:
        return None
    try:
        v = np.array([float(x) for x in words])
    except ValueError:
        return None
    return v / np.linalg.norm(v) if np.linalg.norm(v) > 0.5 else None


class Server:
    def __init__(self, args, pw, detector, cam, up, save):
        self.args, self.pw = args, pw                    # pw: pick_watch, the pick check
        self.detector, self.cam, self.up, self.save = detector, cam, up, save
        self.watch = None                                # the pick check's thread for the last pick
        self.before, self.before_up = None, up          # the last pick, for "check"
        self.skipped, self.passed = [], []              # outlines never / not now chosen (avoid)
        self.picker_img = self.crop = self.check_img = None
        self.flags = 0
        self.up_from_accel = lambda a: np.asarray(a, float) / np.linalg.norm(a)   # (faces.up_from_accel)
        base = os.path.splitext(args.overlay)[0]
        self.paths = {k: f"{base}_{k}.jpg" for k in ("picker", "crop", "check")}
        self.out = os.path.join(os.path.abspath(args.out), time.strftime("%Y-%m-%d_%H%M%S"))

    def settings(self):
        a = self.args
        return {"frames": a.frames, "check_frames": a.check_frames, "score": round(a.score, 2),
                "order": self.detector.order, "top_faces": bool(getattr(self.detector, "top_faces", False)),
                "skipped": len(self.skipped), "passed": len(self.passed)}

    def write(self, key, img):
        if img is None:
            return None
        cv2.imwrite(self.paths[key], img)
        return self.paths[key]

    # -- pick -----------------------------------------------------------------------

    def camera_up(self):
        """Up from the accelerometer the bridge sends with each frame (the last 0.5 s,
        none while moving: ros_bridge.py 2026-11-04), or None."""
        accel = getattr(self.cam, "accel", None)
        return self.up_from_accel(accel) if accel else None

    def pick(self, words):
        """"pick ux uy uz": the up direction now (colour camera frame), from the
        camera's accelerometer (the node's) -- else the accelerometer sent with the
        frames, else the one from --pitch (the TF)."""
        args, detector, cam = self.args, self.detector, self.cam
        self.stop_watch()
        saved = words[3:4] == ["saved"]                  # the node's saved reading, not a live one
        given = up_of(words[:3]) if words else None
        t0 = time.time()
        for _ in range(args.skip):
            cam.read()
        frames = [cam.read() for _ in range(args.frames)]
        t1 = time.time()
        camera = self.camera_up() if given is None or saved else None
        if camera is not None:
            given = None                                 # live, with the frames: before a saved one
        up_now, up_source = ((given, "given") if given is not None else
                             (camera, "camera") if camera is not None else (self.up, "pitch"))
        result = detector.detect_frames(frames, cam.intr, up_now, avoid=self.skipped + self.passed)
        if result is None and self.passed:                # no box after the passed ones: the first again
            self.passed.clear()
            result = detector.detect_frames(frames, cam.intr, up_now, avoid=self.skipped)
        t2 = time.time()
        self.before, self.before_up = None, up_now
        if result is not None and detector.last is not None and detector.last[2] is not None:
            box = detector.last[1][detector.last[2]]
            self.before = Target(box.mask.copy(), result)
            # The pick check, from the depth of the frame the target came from, on every frame from now on.
            self.watch = Watch(cam, self.pw.PickWatcher(box.pick, frames[detector.best_frame or 0][1], box.mask))
            self.watch.start()
        self.picker_img = self.crop = None
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
            best = frames[detector.best_frame or 0]
            if target is not None and found[target].pick is not None:
                face = face_axes(found[target].pick, cam.intr)
            cv2.imwrite(args.overlay, best[0])              # clean: the GUI draws
            overlay = args.overlay
            try:                                            # the pictures are extras: never fail on them
                self.picker_img = detector.annotate(best[0])
                if self.before is not None:
                    pick_face = found[target].pick
                    self.crop = close_up(best[0], self.before.target,
                                         pick_face.corners_px if pick_face is not None else None)
            except Exception as exc:                        # noqa: BLE001
                print(f"picker view / close-up: {exc}", file=sys.stderr)
            # the frame the target came from, with its depth: to measure a
            # miss afterwards (one file, overwritten; ~1 MB)
            try:
                np.savez_compressed(os.path.splitext(args.overlay)[0] + "_last.npz",
                                    color=best[0], depth=best[1].astype(np.float32),
                                    intr=np.array([cam.intr.fx, cam.intr.fy, cam.intr.cx, cam.intr.cy]),
                                    result=json.dumps(result))
            except Exception:                               # noqa: BLE001 - only a record
                pass
        elif frames:
            cv2.imwrite(args.overlay, frames[-1][0])
            overlay = args.overlay
        size = [int(frames[0][0].shape[1]), int(frames[0][0].shape[0])] if frames else None
        return {"target": result, "boxes": boxes, "face": face, "overlay": overlay, "size": size,
                "picker_view": self.write("picker", self.picker_img), "close_up": self.write("crop", self.crop),
                "took": {"pictures": round(t1 - t0, 2), "picker": round(t2 - t1, 2)},
                "up": [round(float(v), 5) for v in up_now], "up_source": up_source,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 2)}

    # -- check ----------------------------------------------------------------------

    def stop_watch(self):
        if self.watch is not None:
            self.watch.stop()
            self.watch = None

    def check(self, words):
        """The "check" request: did the last pick take its box? The pick check's
        decision, waiting for it up to CHECK_WAIT_S; not decided by then: null
        (outcome "blocked"), and a later "check" asks again."""
        watch, before = self.watch, self.before
        if before is None or watch is None:
            return {"check": None, "picked": None,
                    "message": "nothing to check: no box was chosen by the last pick"}
        holding = {"held": True, "empty": False}.get(words[0]) if words else None
        watch.done.wait(CHECK_WAIT_S)
        if watch.error is not None:
            raise watch.error                            # the camera failed: the node restarts the picker
        w = watch.watcher
        outcome = OUTCOME[watch.result] if watch.result is not None else "blocked"
        if holding is True and outcome == "dropped":     # the gripper holds it: trust the gripper
            outcome = "success"
        elif holding is False and outcome == "success":  # gone from the view, but not in the gripper
            outcome = "dropped"
        picked, title, why, _ = RESULT[outcome]
        if watch.result is None:
            why += f" ({w.status})"
        verdict = {"outcome": outcome, "check": "pick_watch", "status": w.status, "holding": holding,
                   "seconds": None if watch.result is None else round(time.time() - watch.started, 1),
                   "shift_mm": None if w.shift is None else round(w.shift * 1000)}
        colour = watch.last[0] if watch.last is not None else self.cam.read()[0]
        cv2.imwrite(self.args.overlay, colour)           # clean: the GUI draws
        try:
            self.check_img = check_view(colour, w, outcome, before.target)
        except Exception as exc:                         # noqa: BLE001
            print(f"check view: {exc}", file=sys.stderr)
            self.check_img = None
        if picked is True:
            self.passed.clear()                          # pick_loop: until the next successful pick
        if watch.result is not None:
            self.watch = None                            # decided: a second "check" has nothing to check
        return {"check": verdict, "picked": picked,
                "message": f"{title}: {why}", "title": title, "why": why,
                "outline": outline(before.target), "overlay": self.args.overlay,
                "picker_view": self.write("check", self.check_img), "frames": watch.frames,
                "size": [int(colour.shape[1]), int(colour.shape[0])],
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    # -- the window's other keys ------------------------------------------------------

    def command(self, cmd, words):
        """set / pass / skip / reset / save / flag / settings -> (ok, message, extra)."""
        args, detector = self.args, self.detector
        if cmd == "settings":
            return True, "settings", {}
        if cmd == "set":
            if len(words) != 2:
                return False, "set NAME VALUE", {}
            name, value = words
            if name == "order":
                if value not in ("v1", "v2"):
                    return False, "order is v1 or v2", {}
                detector.order = value
                return True, f"order: {value}", {}
            if name == "top_faces":                      # pick_rules.py
                if value not in ("on", "off"):
                    return False, "top_faces is on or off", {}
                detector.top_faces = value == "on"
                return True, ("top faces: the larger of front and top, anywhere" if value == "on" else
                              "top faces off: a box on a box by its front (a top only if no front)"), {}
            if name not in LIMITS:
                return False, f"no setting {name!r} (frames, check_frames, score, order, top_faces)", {}
            lo, hi, kind = LIMITS[name]
            try:
                v = kind(float(value))
            except ValueError:
                return False, f"{name}: a number, please", {}
            v = round(min(max(v, lo), hi), 2)
            setattr(args, name, v)
            if name == "score":                          # box_masks runs SAM at LONE_SCORE and splits by score
                finder = getattr(detector, "finder", None)
                if hasattr(finder, "score_threshold"):
                    finder.score_threshold = v
                elif getattr(finder, "processor", None) is not None:
                    finder.processor.confidence_threshold = v
            return True, f"{name}: {v}", {}
        if cmd in ("pass", "skip"):
            if self.before is None:
                return False, f"no target to {cmd}", {}
            (self.passed if cmd == "pass" else self.skipped).append(self.before.target.copy())
            self.stop_watch()                            # this box isn't being picked now
            return True, ("passed over this box and the boxes under it, until the next successful pick"
                          if cmd == "pass" else "this box is skipped until reset"), {}
        if cmd == "reset":
            n = len(self.skipped) + len(self.passed)
            self.skipped, self.passed = [], []
            self.stop_watch()
            return True, f"forgot {n} skipped / passed box(es)", {}
        if cmd == "save":
            colour, depth = self.cam.read()
            meta = self.cam.meta()                       # (over ROS: with the bridge's accel, if any)
            up = up_of(words)                            # given: gravity's (the node's accelerometer)
            if up is not None:
                # as a reading at rest, so pick_pipeline.py --capture replays it with this tilt
                meta["accel"] = [round(9.80665 * float(v), 4) for v in up]
            elif meta.get("accel"):
                up = self.up_from_accel(meta["accel"])
            meta["tilt_source"] = "accelerometer" if up is not None else "tf (replay with --pitch)"
            up = self.up if up is None else up
            meta["up"] = [round(float(v), 5) for v in up]
            meta["pitch_deg"] = round(float(np.degrees(np.arcsin(np.clip(-up[2], -1, 1)))), 2)
            prefix = os.path.join(self.out, "snapshots", f"snap_{time.strftime('%H%M%S')}")
            self.save(prefix, colour, depth, meta)
            return True, f"saved {prefix}_*", {"path": prefix}
        if cmd == "flag":
            if self.before is None:
                return False, "no target to flag", {}
            self.flags += 1
            d = os.path.join(self.out, "flags")
            os.makedirs(d, exist_ok=True)
            photos = {}
            for kind, img in (("target", self.picker_img), ("crop", self.crop), ("check", self.check_img)):
                if img is not None:
                    photos[kind] = os.path.join(d, f"flag_{self.flags:02d}_{kind}.jpg")
                    cv2.imwrite(photos[kind], img)
            return True, f"flag {self.flags}: {len(photos)} photo(s) in {d}", {"photos": photos}
        return False, f"unknown request {cmd!r}", {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--check-frames", type=int, default=3)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--score", type=float, default=0.4, help="SAM's confidence needed to count as a box")
    ap.add_argument("--order", choices=["v1", "v2"], default="v1",
                    help="v1: stack tops (highest first), then nearest, then leftmost; v2: leftmost free box")
    ap.add_argument("--top-faces", action="store_true",
                    help="pick_rules.py: tops anywhere (the larger of front and top, also for a box on "
                         "a box), as before 5 Nov 2026; default: a box on a box by its front")
    ap.add_argument("--ros", action="store_true")
    ap.add_argument("--overlay", required=True, help="where to write the drawing of each answer")
    ap.add_argument("--out", default="outputs/gui", help="snapshots and flags, one folder per run")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())                    # box_picker/src
    real_stdout, sys.stdout = sys.stdout, sys.stderr   # libraries' output -> stderr
    import pick_watch as pw                            # the pick check (2026-11-11)
    from faces import up_from_accel, up_from_pitch
    from pick_frame import PickDetector
    from rgbd import open_camera, save
    detector = PickDetector(score=args.score, order=args.order, top_faces=args.top_faces)
    cam = LockedCamera(open_camera(args.ros))
    up = up_from_pitch(args.pitch)
    server = Server(args, pw, detector, cam, up, save)
    server.up_from_accel = up_from_accel
    sys.stdout = real_stdout
    reply({"ready": True, "frames": args.frames, "up": [round(float(v), 3) for v in up],
           "score": args.score, "order": args.order, "settings": server.settings()})

    try:
        for line in sys.stdin:
            words = line.strip().lower().split()
            cmd = words[0] if words else ""
            if cmd in ("quit", "exit"):
                break
            sys.stdout = sys.stderr                     # libraries' output -> stderr
            try:
                if cmd == "pick":
                    answer = server.pick(words[1:])
                elif cmd == "check":
                    answer = server.check(words[1:])
                else:
                    ok, message, extra = server.command(cmd, words[1:])
                    answer = {"cmd": cmd, "ok": ok, "message": message, **extra}
            except Exception as exc:                    # noqa: BLE001 - answered, not raised
                if cmd not in ("set", "pass", "skip", "reset", "save", "flag", "settings"):
                    raise                               # pick / check: the node restarts the picker
                answer = {"cmd": cmd, "ok": False, "message": f"{cmd} failed: {exc}"}
            finally:
                sys.stdout = real_stdout
            answer["settings"] = server.settings()
            reply(answer)
    finally:
        server.stop_watch()
        cam.close()


if __name__ == "__main__":
    main()
