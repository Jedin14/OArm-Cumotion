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
    pick ... box H W D   with the box size (height, width, depth mm: the node reads
              /beast_arm/held_object/fixed_size, the size entered in the robot's GUI).
              Order grid uses its depth for the layers (box_picker/v1/grid_order.py, 18 Nov
              2026) and doesn't pick without it: no target, "why": "box size not set ..."
    check [held|empty]   after the pick, with the arm out of the view: the
              result of the pick check (2026-11-11: the Pick Check team's new one,
              pick_watch.py: box_picker/v1's when there, else box_picker/src's). From the
              "pick" on, a thread feeds it every camera frame: the target's pick face (arm
              in front of it, still there, gone), then once the box has left its spot the
              whole view, until it settles. Its reference is all the pick's frames (pixels
              whose depth flickers between them are not compared), and with the robot's box
              size the face must read most of the box's thinnest side further back to be gone.
              "check" also tells it the robot's arm is out of the view (19 Nov 2026): it then
              decides on what it sees within a second -- still there: missed (also if the arm
              was never seen); pushed back or tipped: moved; gone: picked unless a new surface
              stands in the carton -- instead of waiting for the arm to be seen leaving. A
              missed / moved decided before the "check" is not final (the robot can press
              again): it keeps watching. "check" waits for the decision, up to CHECK_WAIT_S ->
              {"check": verdict, "picked": true / false / null, "message", "title",
              "why", "outline" (the target's polygon), "overlay" (the newest frame,
              clean), "picker_view" (the check's picture: the pick face in the
              result's colour, where a dropped box fell outlined), "frames"}.
              verdict["outcome"]: "success" (left its spot, then nothing new stayed
              in view), "missed" (the box is still there), "moved" (pushed, knocked or
              dragged), "dropped" (left its spot, then a new surface stood still in the
              carton: it fell back in), "camera_moved" (most of the view changed: null),
              "blocked" (no decision yet: null; "why" says what it waits for, e.g. something
              reaching in from the picture's edge). held / empty, the gripper's word, wins
              over the camera: held -> "success", empty -> "dropped"; and when the box left
              its spot but the view did not settle, it decides. verdict also has what the
              camera saw: "status", "detail" (the face's shares, where it reads, the change at
              the picture's edge), "timeline" (each change of status, s from the "pick"),
              "provisional", "fps", "gone_mm", "log" (the folder with it all: below). A
              success forgets the passed boxes. check_frames is no longer used (kept for the GUI)

              Every check is saved in <out>/<run>/checks/NNN_HHMMSS_<outcome>/ (the last
              KEEP_CHECKS): verdict.json, timeline.txt, trace.jsonl (every frame: the face's
              shares, the state, the change at the edge), a picture at each change of status
              (key_*.jpg, with its depth *_depth.png, mm), the pick's frames (ref_*_depth.png,
              ref_color.jpg) and face.npz; with --record-checks (or "set record on") also every
              frame watched (frames/), to replay it (replay_check.py in box_picker/v1, the repository's box_picking/v1)
    set NAME VALUE       record (on / off: every frame of each pick check saved, as
              --record-checks), frames (1-15), check_frames (1-10), score (0.20-0.90),
              order (grid / v1 / v2: see below), top_faces (on / off: pick_rules.py -- off, a box
              on a box by its front, a top only as a last resort, "top_only";
              on, the larger of front and top anywhere, as before 5 Nov 2026):
              from the next pick / check on
    pass      the last target is passed over until the next successful pick
              (pick_loop's T); with no other box left, the first again
    skip      the last target is never chosen again (until reset)
              Order grid (the default): only that box is left out, so the next
              target is the one that would be picked if it wasn't there (the box
              under it can be next). Orders v1 / v2 (the old picker): the boxes
              under and behind it are left out too
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

Order grid (2026-11-16, the default): box_picking's v1, in box_picker/v1/ (grid_order.py), for
boxes of one size stacked in a grid: fallen boxes first, each by its largest face (usually its
top); then the front layer, the top row first, each row left to right, by the front face; then
the next layer back. The same SAM 3.1 as the old picker (loaded once) and the same faces.py from
box_picker/src. Its target carries, besides the old fields: "fallen", "cell" ([layer, row,
column] from 1; null for a fallen box), "box_size_mm" ([w, h, d]) and "next" (the next 3 picks).
In "boxes", why_not says where a box is in the order, or "outside the carton in front" (v2: only
the carton in front is picked from). "set order v1" goes back to the old picker.

    .venv/bin/python ../../box_picker_server.py --pitch -4.5 --ros --overlay /tmp/x.jpg
"""
import argparse
import importlib.util
import json
import os
import queue
import shutil
import sys
import threading
import time

import cv2
import numpy as np

LIMITS = {"frames": (1, 15, int), "check_frames": (1, 10, int), "score": (0.2, 0.9, float)}
RED = (0, 0, 255)                                       # BGR, the target, as the picker draws it
CHECK_WAIT_S = 15.0      # "check" waits this long for the pick check's decision (the node waits 20 s)
KEEP_CHECKS = 30         # the last this many checks are kept in <out>/<run>/checks/
KEY_FRAMES = 12          # pictures kept per pick check: at each change of status (and the decision)
# The new pick check's results, in the old check's words (box_model_node / the GUI read "outcome"):
# outcome -> (picked, title, why, colour BGR)
RESULT = {"success": (True, "PICK SUCCESS", "the box left its spot and the camera's view", (0, 200, 0)),
          "missed": (False, "PICK FAILED: MISSED", "the box is still there", (0, 0, 255)),
          "moved": (False, "PICK FAILED: BOX MOVED", "the box was pushed, knocked or dragged, not picked", (0, 0, 255)),
          "dropped": (False, "PICK FAILED: DROPPED", "the box was lifted but fell back into the carton", (0, 0, 255)),
          "camera_moved": (None, "CAN'T TELL: CAMERA MOVED", "most of the view changed since before the pick: "
                                                               "the camera or the robot moved", (0, 165, 255)),
          "blocked": (None, "CAN'T TELL YET", "the pick check has not decided", (0, 165, 255))}
OUTCOME = {"picked": "success", "missed": "missed", "moved": "moved", "dropped": "dropped",
           "camera_moved": "camera_moved"}


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


class Recorder(threading.Thread):
    """Every frame a pick check watches, written to <folder>/frames/ by its own thread (colour
    .jpg, depth .png in mm; NNNNN_<s from the pick>), to replay it (replay_check.py). Frames that
    come while the disk is behind are left out (counted)."""
    def __init__(self, folder):
        super().__init__(daemon=True)
        self.folder = os.path.join(folder, "frames")
        os.makedirs(self.folder, exist_ok=True)
        self.q, self.k, self.left_out = queue.Queue(maxsize=90), 0, 0
        self.start()

    def put(self, t, colour, depth):
        self.k += 1
        try:
            self.q.put_nowait((self.k, t, colour, depth))
        except queue.Full:
            self.left_out += 1

    def close(self):
        self.q.put(None)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            k, t, colour, depth = item
            p = os.path.join(self.folder, f"{k:05d}_{t:08.3f}")
            cv2.imwrite(p + "_color.jpg", colour, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(p + "_depth.png", np.clip(depth * 1000, 0, 65535).astype(np.uint16))


class Watch(threading.Thread):
    """The pick check on every camera frame, from the "pick" until it has decided.
    arm_out (set by "check"): the robot's arm is out of the view -- the watcher then decides on
    what it sees. Before it, a missed / moved is not final (the robot can press again): the
    watcher is reopened and keeps watching. keyframes: (s, status, colour, depth) at each change
    of status. ctx: what the target was found with (for the check's log)."""
    def __init__(self, cam, watcher, ctx=None, record=None):
        super().__init__(daemon=True)
        self.cam, self.watcher, self.ctx, self.record = cam, watcher, ctx or {}, record
        self.result = self.error = self.last = None
        self.frames, self.stopped, self.done = 0, False, threading.Event()
        self.started = time.time()
        self.arm_out, self.arm_out_at, self.decided_at = False, None, None
        self.keyframes = []
        self.new = hasattr(watcher, "reopen")            # pick_watch.py of 19 Nov 2026 or newer

    def run(self):
        said = None
        try:
            while not self.stopped:
                colour, depth = self.cam.read()
                now = time.time()
                self.last, self.frames = (colour, depth), self.frames + 1
                if self.new:
                    result = self.watcher.update(depth, self.cam.intr, now, arm_out=self.arm_out)
                else:
                    result = self.watcher.update(depth, self.cam.intr)
                if self.record is not None:
                    self.record.put(now - self.started, colour, depth)
                status = self.watcher.status
                if status != said and len(self.keyframes) < KEY_FRAMES:
                    self.keyframes.append((round(now - self.started, 2), status, colour, depth))
                said = status
                if result is None:
                    continue
                if self.new and result in ("missed", "moved") and not self.arm_out:
                    self.watcher.reopen(result)          # not final: the robot's "check" decides
                    continue
                self.keyframes.append((round(now - self.started, 2), f"decided: {result}", colour, depth))
                self.result, self.decided_at = result, now
                break
        except Exception as exc:                         # noqa: BLE001 - told at the next "check"
            self.error = exc
        finally:
            if self.record is not None:
                self.record.close()
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


class Orders:
    """The two pickers behind one detector, with one SAM 3.1: the old one (box_picker/src/
    pick_frame.py, orders v1 / v2) and the grid one (box_picker/v1/grid_detector.py, order
    grid). The one that answered last gives last / best_frame / annotate."""

    def __init__(self, old, grid, order):
        self.old, self.grid, self.order = old, grid, order
        self.used = grid if order == "grid" else old

    @property
    def finder(self):
        return self.old.finder

    @property
    def top_faces(self):
        return self.old.top_faces

    @top_faces.setter
    def top_faces(self, value):
        self.old.top_faces = value

    @property
    def last(self):
        return self.used.last

    @property
    def best_frame(self):
        return self.used.best_frame

    def detect_frames(self, frames, intr, up, avoid=(), box_size=None):
        if self.order == "grid":
            self.used = self.grid
            return self.grid.detect_frames(frames, intr, up, avoid=avoid, box_size=box_size)
        self.old.order, self.used = self.order, self.old
        return self.used.detect_frames(frames, intr, up, avoid=avoid)

    def annotate(self, color):
        return self.used.annotate(color)

    def grid_picks(self):
        """Order grid: {box index: (place in the order from 0, the grid picker's pick)} of the
        last answer; other orders: None."""
        if self.used is not self.grid or self.grid is None or self.grid.res is None:
            return None
        return {p["box"]: (k, p) for k, p in enumerate(self.grid.res["picks"])}

    def left_out(self):
        """Order grid with v2's carton check: indices of the boxes outside the carton in front."""
        if self.used is not self.grid or self.grid is None or self.grid.res is None:
            return set()
        return set((self.grid.res.get("carton") or {}).get("left_out") or [])

    def carton(self):
        """Order grid: v2's carton check of the last answer, short and JSON-safe (carton.summary):
        {"found": true, "boxes_in", "left_out", "near_m"} / {"found": false}; else None."""
        if self.used is not self.grid or self.grid is None or self.grid.res is None:
            return None
        found = self.grid.res.get("carton")
        if found is None:
            return None
        if not found["found"]:
            return {"found": False}
        return {"found": True, "boxes_in": found["inside"], "left_out": len(found["left_out"]),
                "near_m": found["near_m"]}


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
        self.box_size = None                             # the robot's box size (height, width, depth mm), per pick
        self.picker_img = self.crop = self.check_img = None
        self.flags = 0
        self.up_from_accel = lambda a: np.asarray(a, float) / np.linalg.norm(a)   # (faces.up_from_accel)
        base = os.path.splitext(args.overlay)[0]
        self.paths = {k: f"{base}_{k}.jpg" for k in ("picker", "crop", "check")}
        self.out = os.path.join(os.path.abspath(args.out), time.strftime("%Y-%m-%d_%H%M%S"))
        self.checks = 0                                  # picks watched: their logs in <out>/checks/
        self.record = bool(getattr(args, "record_checks", False))

    def new_watcher(self, face, depth, mask, refs, box_mm):
        """The pick check for a target: with all the pick's frames and the box's size (pick_watch.py
        of 19 Nov 2026), or as before with an older pick_watch.py."""
        try:
            return self.pw.PickWatcher(face, depth, mask, refs=refs, box_mm=box_mm)
        except TypeError:
            print("pick_watch.py is older than 19 Nov 2026: the check uses one frame and no box size",
                  file=sys.stderr)
            return self.pw.PickWatcher(face, depth, mask)

    def pick_folder(self):
        """The folder for this pick's check log, <out>/<run>/checks/NNN_HHMMSS (made when first
        written); the oldest are removed so that only the last KEEP_CHECKS stay."""
        root = os.path.join(self.out, "checks")
        try:
            old = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
            for d in old[:max(0, len(old) - KEEP_CHECKS + 1)]:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        except OSError:
            pass
        return os.path.join(root, f"{self.checks:03d}_{time.strftime('%H%M%S')}")

    def settings(self):
        a = self.args
        return {"frames": a.frames, "check_frames": a.check_frames, "score": round(a.score, 2),
                "order": self.detector.order, "top_faces": bool(getattr(self.detector, "top_faces", False)),
                "skipped": len(self.skipped), "passed": len(self.passed),
                "box_size": getattr(self, "box_size", None), "record": getattr(self, "record", False)}

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
        self.box_size = None                             # "box H W D" at the end: the robot's box size
        if "box" in words:
            k = words.index("box")
            try:
                size = [float(v) for v in words[k + 1:k + 4]]
                self.box_size = size if len(size) == 3 and all(v > 0 for v in size) else None
            except ValueError:
                pass
            words = words[:k]
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
        if detector.order == "grid" and self.box_size is None:   # no box size: no pick (18 Nov 2026)
            self.before = None
            if frames:
                cv2.imwrite(args.overlay, frames[-1][0])
            return {"target": None, "boxes": [], "face": None, "overlay": args.overlay if frames else None,
                    "size": [int(frames[0][0].shape[1]), int(frames[0][0].shape[0])] if frames else None,
                    "carton": None, "picker_view": None, "close_up": None,
                    "why": "box size not set: enter it in the robot's GUI (/beast_arm/held_object/fixed_size)",
                    "took": {"pictures": round(t1 - t0, 2), "picker": 0.0},
                    "up": [round(float(v), 5) for v in up_now], "up_source": up_source,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 2)}
        size = self.box_size if detector.order == "grid" else None
        result = detector.detect_frames(frames, cam.intr, up_now, avoid=self.skipped + self.passed, box_size=size)
        if result is None and self.passed:                # no box after the passed ones: the first again
            self.passed.clear()
            result = detector.detect_frames(frames, cam.intr, up_now, avoid=self.skipped, box_size=size)
        t2 = time.time()
        self.before, self.before_up = None, up_now
        if result is not None and detector.last is not None and detector.last[2] is not None:
            box = detector.last[1][detector.last[2]]
            self.before = Target(box.mask.copy(), result)
            # The pick check, on every frame from now on: against the frame the target came from and the
            # pick's other frames (a pixel whose depth flickers between them is not compared), with the
            # robot's box size (how far back the face must read to be gone).
            best = detector.best_frame or 0
            refs = [f[1] for i, f in enumerate(frames) if i != best]
            self.checks += 1
            ctx = {"n": self.checks, "face": box.pick, "mask": box.mask, "frames": frames, "best": best,
                   "box_size": self.box_size, "intr": cam.intr, "target": result, "folder": self.pick_folder(),
                   "checks": 0}
            record = None
            if self.record:
                try:
                    record = Recorder(ctx["folder"])
                except OSError as exc:
                    print(f"record: {exc}", file=sys.stderr)
            self.watch = Watch(cam, self.new_watcher(box.pick, frames[best][1], box.mask, refs, self.box_size),
                               ctx, record)
            self.watch.start()
        self.picker_img = self.crop = None
        boxes, overlay, face = [], None, None
        if detector.last is not None:
            points, found, target = detector.last
            grid = detector.grid_picks() if hasattr(detector, "grid_picks") else None
            outside = detector.left_out() if hasattr(detector, "left_out") else set()
            for i, b in enumerate(found):
                why = []
                if grid is not None:                     # order grid: its face and place for each box
                    k, p = grid.get(i, (None, None))
                    face_px = p["corners_px"] if p is not None else None
                    kind = p["face"] if p is not None else None
                    size_mm = [round(float(v)) for v in p["size_mm"]] if p is not None else None
                    if p is None and i in outside:
                        why.append("outside the carton in front")
                    elif p is None:
                        why.append("not in the order: no face to grip, or nothing holds it up")
                    elif i != target:
                        why.append(f"later: number {k + 1} in the order" + (" (fallen box)" if p["fallen"] else ""))
                else:
                    face_px = (b.pick.corners_px if b.pick is not None and
                               b.pick.corners_px is not None else None)
                    kind = b.pick.kind if b.pick else None
                    size_mm = [round(float(v)) for v in b.pick.size_mm] if b.pick else None
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
                              "kind": kind, "size_mm": size_mm,
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
        carton = detector.carton() if hasattr(detector, "carton") else None
        return {"target": result, "boxes": boxes, "face": face, "overlay": overlay, "size": size, "carton": carton,
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
        asked = time.time()
        before_check = watch.result is not None          # decided while the robot was still at it
        # The robot asks once its arm is back out of the view (pre_pick): the watcher now decides on
        # what it sees, instead of waiting for the arm to be seen leaving.
        watch.arm_out = True
        if watch.arm_out_at is None:
            watch.arm_out_at = asked
        watch.done.wait(CHECK_WAIT_S)
        if watch.error is not None:
            raise watch.error                            # the camera failed: the node restarts the picker
        w = watch.watcher
        outcome = OUTCOME.get(watch.result, "blocked") if watch.result is not None else "blocked"
        by_gripper = False
        if outcome == "blocked" and holding is not None and getattr(w, "left", False) and w.drop is None:
            # the box left its spot, the view did not settle (something stays at the picture's edge):
            # the gripper's word decides
            outcome, by_gripper = ("success" if holding else "dropped"), True
        if holding is True and outcome == "dropped":     # the gripper holds it: trust the gripper
            outcome = "success"
        elif holding is False and outcome == "success":  # gone from the view, but not in the gripper
            outcome = "dropped"
        picked, title, why, _ = RESULT[outcome]
        detail = getattr(w, "detail", w.status)
        if watch.result is None and not by_gripper:
            why += f" in {CHECK_WAIT_S:.0f} s: {detail}"
        elif by_gripper:
            why += f" (the gripper says {'held' if holding else 'empty'}; the camera: {w.status})"
        ctx = watch.ctx
        verdict = {"outcome": outcome, "check": "pick_watch", "status": w.status, "detail": detail,
                   "holding": holding, "by_gripper": by_gripper,
                   "seconds": None if watch.result is None else round(watch.decided_at - watch.started, 1),
                   "decided_before_check": before_check,
                   "after_check_s": None if watch.decided_at is None or before_check
                   else round(watch.decided_at - asked, 2),
                   "shift_mm": None if w.shift is None else round(w.shift * 1000),
                   "provisional": [list(p) for p in getattr(w, "provisional", [])],
                   "fps": getattr(w, "fps", None), "frames": watch.frames,
                   "gone_mm": round(getattr(w, "gone_m", 0.01) * 1000), "box_size": ctx.get("box_size"),
                   "face_px": int(len(w.pixels)), "off_plane_px": getattr(w, "off_plane", None),
                   "timeline": [[t, s] for t, s in list(getattr(w, "timeline", []))[-30:]]}
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
        elif picked is not None:                         # the gripper's word decided: stop watching
            self.stop_watch()
        if watch.ctx.get("folder"):
            watch.ctx["checks"] = watch.ctx.get("checks", 0) + 1
            verdict["log"] = watch.ctx["folder"]
            threading.Thread(target=self.save_check, args=(watch, dict(verdict), self.check_img,
                                                           watch.ctx["checks"]), daemon=True).start()
        return {"check": verdict, "picked": picked,
                "message": f"{title}: {why}", "title": title, "why": why,
                "outline": outline(before.target), "overlay": self.args.overlay,
                "picker_view": self.write("check", self.check_img), "frames": watch.frames,
                "size": [int(colour.shape[1]), int(colour.shape[0])],
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save_check(self, watch, verdict, check_img, k):
        """The k-th check of a pick, written to its folder (see the module's notes); never fails
        the check. The first also writes what the target was found with, to replay it."""
        ctx, w = watch.ctx, watch.watcher
        folder = ctx["folder"]
        depth_png = lambda d: np.clip(d * 1000, 0, 65535).astype(np.uint16)
        try:
            os.makedirs(folder, exist_ok=True)
            tag = f"check{k}_{verdict['outcome']}"
            timeline = [[t, s] for t, s in getattr(w, "timeline", [])]
            with open(os.path.join(folder, f"{tag}_verdict.json"), "w") as f:
                json.dump({**verdict, "timeline": timeline, "target": ctx.get("target")}, f, indent=1, default=str)
            with open(os.path.join(folder, f"{tag}_timeline.txt"), "w") as f:
                f.write(f"pick {ctx['n']}, check {k}: {verdict['outcome']}  (s from the pick; "
                        f"{verdict['frames']} frames, {verdict['fps']} a second)\n\n")
                f.writelines(f"{t:8.2f} s  {s}\n" for t, s in timeline)
                f.write(f"\n{verdict['detail']}\n")
            with open(os.path.join(folder, f"{tag}_trace.jsonl"), "w") as f:
                f.writelines(json.dumps(e) + "\n" for e in list(getattr(w, "trace", [])))
            if check_img is not None:
                cv2.imwrite(os.path.join(folder, f"{tag}.jpg"), check_img)
            for i, (t, status, colour, depth) in enumerate(list(watch.keyframes)):
                img = colour.copy()
                cv2.polylines(img, [np.round(w.corners_px).astype(np.int32)], True, (255, 0, 255), 2)
                cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
                cv2.putText(img, f"{t:.2f} s  {status}"[:100], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(folder, f"key_{i:02d}_{t:07.2f}s.jpg"), img)
                cv2.imwrite(os.path.join(folder, f"key_{i:02d}_{t:07.2f}s_depth.png"), depth_png(depth))
            if k == 1:
                frames, face, intr = ctx["frames"], ctx["face"], ctx["intr"]
                cv2.imwrite(os.path.join(folder, "ref_color.jpg"), frames[ctx["best"]][0])
                for i, (_, d) in enumerate(frames):
                    cv2.imwrite(os.path.join(folder, f"ref_{i}_depth.png"), depth_png(d))
                np.savez_compressed(
                    os.path.join(folder, "face.npz"), pixels=face.pixels, normal=face.normal,
                    centroid=face.centroid, corners_px=np.asarray(face.corners_px, float),
                    centroid_px=np.asarray(getattr(face, "centroid_px", (0, 0)), float),
                    size_mm=np.asarray(face.size_mm, float), kind=str(face.kind), mask=ctx["mask"],
                    best=ctx["best"], box_size=np.asarray(ctx["box_size"] or [], float),
                    intr=np.array([intr.fx, intr.fy, intr.cx, intr.cy]))
        except Exception as exc:                         # noqa: BLE001 - only a record
            print(f"check log: {exc}", file=sys.stderr)

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
            if name == "record":                         # every frame of each pick check, to replay it
                if value not in ("on", "off"):
                    return False, "record is on or off", {}
                self.record = value == "on"
                return True, (f"recording every frame of each pick check (from the next pick) in "
                              f"{os.path.join(self.out, 'checks')}" if self.record else "not recording"), {}
            if name == "order":
                if value not in ("grid", "v1", "v2"):
                    return False, "order is grid, v1 or v2", {}
                if value == "grid" and getattr(detector, "grid", None) is None:
                    return False, "order grid is not installed (box_picker/v1)", {}
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
            return True, ("passed over this box until the next successful pick"
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


def load_pick_watch(grid_dir):
    """pick_watch.py: box_picker/v1's when it has the 19 Nov 2026 check (the box size add-on replaces
    v1/), else box_picker/src's."""
    path = os.path.join(os.path.abspath(grid_dir), "pick_watch.py")
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("pick_watch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod.PickWatcher, "reopen"):
            sys.modules["pick_watch"] = mod
            print(f"pick check: {path}", file=sys.stderr)
            return mod
    import pick_watch                                  # box_picker/src
    print(f"pick check: {pick_watch.__file__}" + ("" if hasattr(pick_watch.PickWatcher, "reopen") else
          " (older than 19 Nov 2026: one reference frame, no box size, no arm-out decision)"), file=sys.stderr)
    return pick_watch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--check-frames", type=int, default=3)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--score", type=float, default=0.4, help="SAM's confidence needed to count as a box")
    ap.add_argument("--order", choices=["grid", "v1", "v2"], default="grid",
                    help="grid: boxes in a grid (box_picker/v1): fallen first, then the front layer, top row, "
                         "left to right; v1: stack tops (highest first), then nearest, then leftmost; "
                         "v2: leftmost free box")
    ap.add_argument("--grid-dir", default="../v1",
                    help="order grid's folder (box_picker/v1), from the working directory box_picker/src")
    ap.add_argument("--top-faces", action="store_true",
                    help="pick_rules.py: tops anywhere (the larger of front and top, also for a box on "
                         "a box), as before 5 Nov 2026; default: a box on a box by its front")
    ap.add_argument("--ros", action="store_true")
    ap.add_argument("--overlay", required=True, help="where to write the drawing of each answer")
    ap.add_argument("--out", default="outputs/gui", help="snapshots, flags and pick check logs, one folder per run")
    ap.add_argument("--record-checks", action="store_true",
                    help="save every frame each pick check watches (<out>/<run>/checks/NNN_*/frames/), to replay "
                         "it with replay_check.py (box_picker/v1); ~4 MB a second")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())                    # box_picker/src
    # order grid's own files, after src: it uses src's faces.py and box_masks.py (the same box finding)
    sys.path.append(os.path.abspath(args.grid_dir))
    real_stdout, sys.stdout = sys.stdout, sys.stderr   # libraries' output -> stderr
    pw = load_pick_watch(args.grid_dir)                # the pick check (2026-11-11; 19 Nov 2026)
    from faces import up_from_accel, up_from_pitch
    from pick_frame import PickDetector
    from rgbd import open_camera, save
    old = PickDetector(score=args.score, order=args.order if args.order != "grid" else "v1",
                       top_faces=args.top_faces)
    try:
        from grid_detector import GridDetector
        grid = GridDetector(finder=old.finder)         # the same SAM 3.1, loaded once
    except ImportError as exc:
        if args.order == "grid":
            raise SystemExit(f"order grid needs box_picker/v1 (--grid-dir {args.grid_dir}): {exc}")
        grid = None
    detector = Orders(old, grid, args.order)
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
