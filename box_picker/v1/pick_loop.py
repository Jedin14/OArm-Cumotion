"""v1 pick loop. Pick, check, next: the box picker and the pick check in one loop, for the robot.
One program, one camera (only one program can open the D455).

Copied from vlm/molmo2-box-picking/pick_loop.py (2026-11-16); the only change is which box is picked:
v1's grid order (grid_order.py, through grid_detector.py): fallen boxes first (by their largest face,
usually the top), then the front layer, top row first, each row left to right (by the front face),
then the next layer back. The target event carries v1's extra fields: "fallen", "cell" [layer, row,
column], "box_size_mm" and the "next" picks; the old fields are unchanged, so the robot side is the same.
--replay plays a recording instead of the camera, to try the loop without one.

The camera has two jobs: before a pick, which box and where (the target); after it, whether it
worked (the box is gone from its place). When the pick happens is the robot's: it says "done".

    1. PICK    5 new frames -> PickDetector.detect_frames -> the target, printed as a JSON event.
               The pick check remembers the scene (pick_check.remember: the frames' median depth and
               the target's outline).
    2. WAIT    for the robot's "done" on stdin: the pick is over and the arm is out of view.
    3. CHECK   3 new frames (--check-frames) compared with the remembered scene (pick_check.check): was the pick
               successful?
               success          -> 1, the next box
               failed           -> 1 again: the picker chooses again (usually the same box, a retry).
                                   After 1 + --retries failed tries on the same box a FLAG is raised,
                                   once, with photos of the box for a person; the box is not skipped:
                                   it is tried again for as long as the picker chooses it.
               can't tell       -> 2 again (an arm or hand still in view: the robot clears the view
                                   and says "done" again). After CANT_TELL of them in a row (something
                                   new stays in view), back to 1: new frames, the target chosen again.
    No target (empty, or only boxes passed over with T): the picker tries again every few seconds.
    "reset" forgets the boxes passed over.

These steps are check v1 (--check v1, or W in the window; the default with --replay).

Check v2 (the default since 17 Nov 2026): steps 2 and 3 are done by the camera, as in the Pick
Check team's box_picking_integration (pick_watch.py, their pick_check.py renamed). No "done" is
needed: from the frame the target came from, every camera frame is compared with the target's
face (arm in view, box still there, box gone), and once the box has left its spot, the whole
view: picked once nothing new stays in view (the arm has carried the box out of the picture);
dropped if a new surface stands still in the carton (the box fell back in); missed (the arm came
and went, the box is still there) or moved (knocked or dragged). Anything but picked is a failed
try, retried as in v1 (also when the box chosen is where the dropped one fell). Needs every frame
at the camera's rate (30 fps). "picked" comes later than in v1: once the arm has left the view
(at most SETTLE_FRAMES, 10 s, if it stays in view).

Check v1's code (pick_check.py, pick_and_check.py) belongs to the Pick Check team and is used
unchanged: from this folder, else from ../../Pick Check/. Without it (the v2 robot package) only
check v2 is there.

stdin, one line at a time (the robot, or you):
    done | done held | done empty   the pick is over, the arm out of view: check it (held / empty: what
                                    the gripper says, so a box that left its place but isn't held is "dropped").
                                    Check v2: not needed, ignored
    pick                            choose a new target now
    reset                           forget the boxes passed over (T)
    quit
stdout carries only JSON events, one a line; everything else goes to stderr:
    {"event": "ready", ...}
    {"event": "target", "pick": n, "attempt": k, "target": {...}}     go and pick it (attempt k on this box)
    {"event": "result", "pick": n, "attempt": k, "picked": true / false / null, "verdict": {...}}
                                     check v2: "verdict": {"check": "v2", "outcome": "picked" / "missed" /
                                     "moved" / "dropped", "success": ..., "seconds": ..., "shift_mm": ...}
    {"event": "flag", "pick": n, "reason": "...", "outcomes": [...], "target": {...},
     "photo": ".../target.jpg", "crop": ".../crop.jpg", "check_photo": ".../check.jpg"}
                                     this box failed k tries; it is still tried (once per box). photo: the
                                     scene with the target marked; crop: the box close up; check_photo:
                                     why the last try failed
    {"event": "flag", "pick": n, "reason": "top face only", "target": {...}, "photo": ..., "crop": ...}
                                     a box on another box with no front found, only its top: the
                                     target is that top (pick_rules.py, a last resort). The robot
                                     still picks it; this tells a person. Once per box.
    {"event": "no_target", "pick": n}
    {"event": "no_target", "pick": n, "reason": "box size not set"}
                                     no box size (the robot's /beast_arm/held_object/fixed_size through
                                     the bridge, or --box-size H W D): no pick; asked again every 3 s

Keys in the window (for testing without the robot), also buttons on the panel's two tabs:
    PICK       C = "done" (check now), T next target (pass over this box until the next successful
               pick), A the next tab (PICK, ORDER, ADVANCED), Q / Esc quit
    ORDER      v1's order from the last look: the box size, fallen boxes, the next picks;
               R forget the boxes passed over
    ADVANCED   D depth picture, V the picker's view (v1's order numbered on all boxes), S save colour +
               depth (snapshots/), N new target now (new frames), R forget the boxes passed over,
               W check v1 (robot says done) / v2 (camera watches; looks again), − / + frames per target,
               SAM confidence (--score) and frames per pick success (from the next one on);
               BOX SIZE: B type this PC's box size (height width depth mm, then Enter; Esc cancels),
               G use the robot's (/beast_arm/held_object/fixed_size) or this PC's. From the next look on.
               No box size: no pick

Usage (from box_picking/v1/; the Python is the old folder's, which has SAM 3.1):
    PY=../../vlm/molmo2-box-picking/.venv/bin/python
    $PY pick_loop.py --ros                      # camera over ROS 2 (ros_bridge.py; tilt from its accel topic), window
    $PY pick_loop.py --ros --no-window          # for the robot: stdin / stdout only
    $PY pick_loop.py --ros --check v1           # the robot says "done", then the check (Pick Check's pick_check)
    $PY pick_loop.py                            # D455 by USB
    $PY pick_loop.py --replay ../../vlm/molmo2-box-picking/recordings/grid_part2 --replay-fps 1
                                                # no camera: a recording, one frame a second (check v1: C = done)
    -> outputs/pick_loop/<session>/pick_NN/{before_*, after_*, picker.jpg, check.jpg}, events.jsonl,
       flags/flag_NN_{target,crop,check}.jpg, flags/toponly_NN_{target,crop}.jpg
"""
import argparse
import collections
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Check v1 (Pick Check's pick_check.py): from this folder, else from ../../Pick Check/ in the repository.
# Without it (the v2 robot package) only check v2 is there.
sys.path.append(str(Path(__file__).resolve().parents[2] / "Pick Check"))
try:
    import pick_check as pc
    from pick_and_check import PICKED
except ImportError:
    pc, PICKED = None, {}
# pick_check.py puts vlm/molmo2-box-picking/ first on the import path; this folder's own copies
# (faces.py, rgbd.py, loop_screen.py, ...) come first again.
HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [HERE] + [p for p in sys.path if p != HERE]

import carton
import loop_screen as ui
import pick_watch as pw

RETRIES = 2              # a box that still isn't picked after this many retries is flagged (and still tried)
RESULT_S = 3.0           # a result stays on screen this long before the next target
# The ADVANCED tab's - / + settings: label, lowest, highest, step
FRAMES = {"frames": ("Frames per target", 1, 15, 1), "check_frames": ("Frames per pick success", 1, 10, 1),
          "score": ("SAM confidence", 0.2, 0.9, 0.05)}
CANT_TELL = 3            # this many "can't tell" in a row for one pick: choose the target again
# Check v2 (pick_watch.py, the camera watches the pick): title, why, colour (BGR) of each result
V2_RESULT = {"picked": ("PICK SUCCESS", "the box left its spot and the camera's view", (0, 200, 0)),
             "missed": ("PICK FAILED: MISSED", "the arm came and went, the box is still there", (0, 0, 255)),
             "moved": ("PICK FAILED: BOX MOVED", "the box was knocked or dragged, not picked", (0, 0, 255)),
             "dropped": ("PICK FAILED: DROPPED", "the box was lifted but fell back into the carton", (0, 0, 255)),
             "camera_moved": ("CAN'T TELL: CAMERA MOVED", "most of the view changed since before the pick",
                              (0, 165, 255))}
WINDOW = "pick loop"


def event(obj, log_file=None):
    line = json.dumps(obj)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Target:
    """What the loop keeps of a target (check v2; check v1 keeps pick_check.Before, which has these too)."""
    def __init__(self, colour, target, result):
        self.colour, self.target, self.result = colour, target, result   # BGR; bool HxW outline; the picker's answer


def remember(detector, frames, result):
    """Right after detector.detect_frames(frames, ...): the target to keep, or None (no target)."""
    if pc is not None:
        return pc.remember(detector, frames, result)
    if result is None or detector.last is None or detector.last[2] is None:
        return None
    _, boxes, t = detector.last
    return Target(frames[detector.best_frame][0].copy(), boxes[t].mask.copy(), result)


def same_target(previous, new):
    """The picker's next answer is the box it was just sent to (same approach, within SAME_TARGET_M):
    the last pick didn't take it."""
    if previous is None or new is None or previous["approach"] != new["approach"]:
        return False
    from grid_detector import SAME_TARGET_M
    xyz = lambda r: np.array([r["x_m"], r["y_m"], r["z_m"]])
    return bool(np.linalg.norm(xyz(previous) - xyz(new)) <= SAME_TARGET_M)


def where(result):
    """Where the target is, in the user's words: "fallen box" or "layer 1 · row 2 · column 3"."""
    if result.get("fallen"):
        return "fallen box"
    return "layer {} · row {} · column {}".format(*result["cell"]) if result.get("cell") else "box"


def carton_line(res):
    """(text,) or (text, colour): the carton in front, from detector.res (v2's carton check)."""
    text, warn = carton.message(res)
    return (text, ui.ORANGE) if warn else (text,)


def outline(img, before):
    """img with the target's outline and centre."""
    img = img.copy()
    contours, _ = cv2.findContours(before.target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
    cv2.drawMarker(img, (int(before.result["u"]), int(before.result["v"])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    return img


def raise_flag(session, n, picker_img, before, check_img=None, name="flag"):
    """Photos for a person, of a box given up on (or picked by its top only): the scene with the
    target marked (the picker's overlay), the box close up, and the last check (why it failed;
    none for a top-only flag). -> their paths."""
    d = session / "flags"
    d.mkdir(exist_ok=True)
    ys, xs = np.nonzero(before.target)
    pad = 40
    h, w = before.target.shape
    crop = outline(before.colour, before)[max(ys.min() - pad, 0):min(ys.max() + pad, h),
                                             max(xs.min() - pad, 0):min(xs.max() + pad, w)]
    paths = {"photo": d / f"{name}_{n:02d}_target.jpg", "crop": d / f"{name}_{n:02d}_crop.jpg"}
    if check_img is not None:
        paths["check_photo"] = d / f"{name}_{n:02d}_check.jpg"
    for key, img in (("photo", picker_img), ("crop", crop), ("check_photo", check_img)):
        if key in paths:
            cv2.imwrite(str(paths[key]), img)
    return {k: str(v) for k, v in paths.items()}


def result_text(verdict):
    """(title, why, colour) of a check's verdict, v1 (pick_check.check) or v2 (pick_watch)."""
    return V2_RESULT[verdict["outcome"]] if verdict.get("check") == "v2" else pc.RESULT[verdict["outcome"]]


def watch_image(colour, watcher, outcome):
    """Check v2's picture: the target's pick face, and for "dropped" where the box fell, in the
    result's colour, with the result in large letters at the bottom."""
    title, why, col = V2_RESULT[outcome]
    img = colour.copy()
    cv2.polylines(img, [np.round(watcher.corners_px).astype(np.int32)], True, col, 4)
    if outcome == "dropped" and watcher.drop is not None:
        fell = cv2.resize(watcher.drop.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.drawContours(img, cv2.findContours(fell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                         pw.RESULT_COLORS["dropped"], 3)
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 110), (img.shape[1], h), col, -1)
    cv2.putText(img, title, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    cv2.putText(img, why, (22, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def read_stdin(q):
    for line in sys.stdin:
        q.put(line.strip().lower())
    q.put("quit")                          # stdin closed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pitch", type=float, help="camera tilt down in degrees (default: accelerometer, else 0)")
    ap.add_argument("--ros", action="store_true", help="camera over ROS 2, through ros_bridge.py")
    ap.add_argument("--retries", type=int, default=RETRIES, help="retries on the same box before a flag is raised (it is still tried)")
    ap.add_argument("--frames", type=int, default=5, help="frames per picker answer")
    ap.add_argument("--check-frames", type=int, default=3, help="new frames the pick check compares with before")
    ap.add_argument("--score", type=float, default=0.4,
                    help="SAM's confidence needed to count as a box (thin standing boxes score 0.45-0.6)")
    ap.add_argument("--no-window", action="store_true", help="no window: stdin / stdout only")
    ap.add_argument("--replay", metavar="RECORDING", help="no camera: play a recording's frames instead (for testing)")
    ap.add_argument("--replay-fps", type=float, default=1.0, help="--replay: recorded frames per second (default 1)")
    ap.add_argument("--replay-from", type=int, default=0, help="--replay: the frame number to start at")
    ap.add_argument("--check", choices=["v1", "v2"],
                    help="v2 (default): the camera watches the pick on every frame and decides picked / missed / "
                         "moved / dropped itself, no done needed (pick_watch.py); v1: the robot says done, then new "
                         "frames are compared with before (Pick Check's pick_check; the default with --replay, whose "
                         "frames come too slowly to watch a pick)")
    ap.add_argument("--out", default="outputs/pick_loop", help="where each session is saved")
    ap.add_argument("--no-carton", action="store_true",
                    help="use every box in view, not only those in the carton in front (v2's carton check off)")
    ap.add_argument("--box-size", type=float, nargs=3, metavar=("H", "W", "D"),
                    help="the boxes' height, width and depth in mm, instead of the robot's "
                         "(/beast_arm/held_object/fixed_size, through the bridge): for the USB camera or --replay. "
                         "No box size: no pick")
    args = ap.parse_args()
    if args.check is None:
        args.check = "v1" if args.replay and pc else "v2"
    if args.check == "v1" and pc is None:
        ap.error("check v1 needs Pick Check's pick_check.py and pick_and_check.py (not in this package): use --check v2")

    real_stdout, sys.stdout = sys.stdout, sys.stderr  # library prints go to stderr
    from faces import up_from_accel, up_from_pitch
    from grid_detector import GridDetector
    from rgbd import ReplayCamera, open_camera, save
    detector = GridDetector(score=args.score, carton_check=not args.no_carton)
    cam = ReplayCamera(args.replay, args.replay_fps, args.replay_from) if args.replay else open_camera(args.ros)

    def tilt():
        """Up direction now: from the camera's accelerometer (over ROS: the last 0.5 s, so a new tilt
        every pick as the robot moves), else --pitch, else level."""
        return up_from_accel(cam.accel) if cam.accel and args.pitch is None else up_from_pitch(args.pitch or 0)
    if args.pitch is None and not cam.accel:
        print("no accelerometer reading: tilt 0 until one arrives (over ROS: driver enable_accel:=true)", file=sys.stderr)
    sys.stdout = real_stdout
    session = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True)
    events = session / "events.jsonl"
    commands = queue.Queue()
    threading.Thread(target=read_stdin, args=(commands,), daemon=True).start()
    window = not args.no_window
    clicks = collections.deque()              # keys of the buttons clicked in the window
    if window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, lambda e, x, y, *_: e == cv2.EVENT_LBUTTONUP and ui.clicked(x, y)
                             and clicks.append(ui.clicked(x, y)))
    event({"event": "ready", "session": str(session), "retries": args.retries}, events)

    sized = False

    stamps = collections.deque(maxlen=20)     # when the last frames came: the live pill's rate
    latest = [None]                           # the newest (colour, depth)
    tab, depth_on, picker_on, snap = "pick", False, False, ""
    # The box size set on this PC (ADVANCED tab: B types it, G switches robot / this PC), else the robot's.
    boxsize = {"pc": list(args.box_size) if args.box_size else None, "use_pc": bool(args.box_size), "typing": None}

    def order_tab():
        """The ORDER tab's panel: v1's order from the last look."""
        rows = [("label", "WHICH BOX FIRST (V1)"),
                ("small", "Fallen boxes first, by their top. Then layer 1 (the front): the top row first, "
                          "each row left to right, by the front. Then the next layer back.")]
        res = detector.res
        if res is None:
            return rows + [None, ("text", "No look yet.")]
        size, given = res["box_size_mm"], res.get("box_given_mm")
        fallen = sum(p["fallen"] for p in res["picks"])
        layers = len({p["cell"][0] for p in res["picks"] if p["cell"] is not None})
        rows += [None, ("label", "LAST LOOK"), ("text", *carton_line(res)),
                 ("text", "Box " + ("not set" if not given else
                                    f"{given[1]:.0f} × {given[0]:.0f} × {given[2]:.0f} mm (w × h × d, given)")),
                 ("small", "seen: " + ("?" if size is None or size[0] is None else f"{size[0]:.0f} × {size[1]:.0f} mm front")),
                 *([("text", "Check the box size: " + res["size_warning"], ui.ORANGE)] if res.get("size_warning") else []),
                 ("text", f"{len(res['picks']) - fallen} in the grid ({layers} layer{'s' if layers != 1 else ''}), "
                          f"{fallen} fallen")]
        for k, p in enumerate(res["picks"][:4]):
            where = "fallen box" if p["fallen"] else "layer {} · row {} · column {}".format(
                *(c + 1 for c in p["cell"]))
            line = f"{k + 1}.  {where}, {p['face']}"
            rows.append(("text", line, ui.ORANGE) if p["fallen"] else ("text", line))
        rows += [None, ("button", "R", "Forget passed boxes", f"{len(passed)} passed" if passed else "none")]
        return rows

    def advanced():
        """The ADVANCED tab's panel."""
        return [("label", "VIEW"),
                ("button", "D", "Depth picture", "on" if depth_on else "off", depth_on),
                ("button", "V", "Picker's view: all boxes", "on" if picker_on else "off", picker_on),
                ("button", "S", "Save colour + depth", snap),
                None, ("label", "PICKER"),
                ("button", "N", "Look again (new frames)", ""),
                ("setting", "frames", FRAMES["frames"][0], str(args.frames), False),
                ("setting", "score", FRAMES["score"][0], f"{args.score:.2f}", False),
                None, ("label", "PICK CHECK"),
                *([("button", "W", "Check", "v2: camera watches" if args.check == "v2" else "v1: robot says done",
                    args.check == "v2"),
                   ("setting", "check_frames", FRAMES["check_frames"][0], str(args.check_frames), False)]
                  if pc else [("text", "Check v2: the camera watches the pick")]),
                None, ("label", "BOX SIZE, H × W × D MM (NONE: NO PICK)"),
                ("button", "B", "Type it · G: robot/PC", box_value())]

    def box_value():
        """The box size row's value: what is being typed, or the size used and where from."""
        if boxsize["typing"] is not None:
            return boxsize["typing"] + "_"
        s, where = (boxsize["pc"], "PC") if boxsize["use_pc"] else (robot_size(), "robot")
        return f"{where} " + ("none" if not s else "{:.0f}×{:.0f}×{:.0f}".format(*s))

    def robot_size():
        """The robot's box size (height, width, depth mm) through the bridge, or None (none, or a side 0)."""
        s = getattr(cam, "box_size", None)
        return tuple(float(v) for v in s) if s and len(s) == 3 and all(v and v > 0 for v in s) else None

    def show(img, status, colour, panel, step, live=False, target=False):
        """Draw the window (loop_screen) and return the key pressed or clicked (lower case), or ''.
        live: img is the live colour picture (the depth picture instead when that is on); target:
        draw the target over it. A, D, V, O, F, W and S are handled here, in every step."""
        nonlocal sized, tab, depth_on, picker_on, snap
        if not window:
            return ""
        fps = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-3) if len(stamps) > 2 else None
        if stamps and time.time() - stamps[-1] > 2:
            fps = 0.0
        depth = depth_on and (live or target) and not (target and picker_on)
        view = ui.depth_view(latest[0][1]) if depth else img
        if target:
            view = picker_img if picker_on else ui.target_view(view, before)
        canvas = ui.render(view, status, colour,
                           advanced() if tab == "advanced" else order_tab() if tab == "order" else panel,
                           f"Picked {picked_n}     Failed tries {failed_n}     Flags {flag_n}",
                           step, fps, history if tab == "pick" else (), tab, depth)
        if not sized:
            cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
            sized = True
        cv2.imshow(WINDOW, canvas)
        k = cv2.waitKey(1) & 0xFF
        if boxsize["typing"] is not None and k != 255:    # typing the box size: digits, space, Backspace, Enter, Esc
            if k in (13, 10):
                try:
                    v = [float(x) for x in boxsize["typing"].replace(",", " ").split()]
                except ValueError:
                    v = []
                if len(v) == 3 and all(x > 0 for x in v):
                    boxsize["pc"], boxsize["use_pc"] = v, True
                    log(f"box size set on this PC: {v[0]:.0f} high x {v[1]:.0f} wide x {v[2]:.0f} deep mm")
                else:
                    log(f"box size not understood ({box['typing']!r}): type height width depth in mm, e.g. 125 125 54")
                boxsize["typing"] = None
            elif k == 27:
                boxsize["typing"] = None
            elif k in (8, 127):
                boxsize["typing"] = boxsize["typing"][:-1]
            elif chr(k) in "0123456789 .,":
                boxsize["typing"] += chr(k)
            return ""
        key = "q" if k == 27 else chr(k).lower() if 0 < k < 128 else clicks.popleft() if clicks else ""
        if key == "a":                                # the next tab
            names = [n for n, _ in ui.TABS]
            tab = names[(names.index(tab) + 1) % len(names)]
        elif key.startswith("tab:"):                  # a tab clicked
            tab = key[4:]
        elif key == "d":
            depth_on = not depth_on
        elif key == "v":
            picker_on = not picker_on
        elif key == "w" and pc is not None:
            args.check = "v2" if args.check == "v1" else "v1"
            log(f"pick check: {args.check} ({'the camera watches the pick' if args.check == 'v2' else 'the robot says done'})")
        elif key == "b":                              # type this PC's box size
            boxsize["typing"] = ""
            tab = "advanced"
        elif key == "g":                              # the box size from: the robot / this PC
            boxsize["use_pc"] = not boxsize["use_pc"]
            if boxsize["use_pc"] and not boxsize["pc"]:
                boxsize["pc"] = list(robot_size()) if robot_size() else None
                if boxsize["pc"] is None:
                    boxsize["typing"] = ""             # nothing to start from: type it
            log("box size from " + ("this PC" if boxsize["use_pc"] else "the robot"))
        elif key[2:] in FRAMES:                       # frames, SAM's confidence: from the next target / check on
            name = key[2:]
            _, lo, hi, step = FRAMES[name]
            setattr(args, name, round(min(max(getattr(args, name) + (step if key[0] == "+" else -step), lo), hi), 2))
            if name == "score":
                detector.finder.score_threshold = args.score   # SAM runs at LONE_SCORE; box_masks splits by score
            log(f"{FRAMES[name][0].lower()}: {getattr(args, name)}")
        elif key == "s" and latest[0] is not None:
            snap = f"snap_{time.strftime('%H%M%S')}"
            save(session / "snapshots" / snap, *latest[0], cam.meta())
            log(f"saved {session / 'snapshots' / snap}_*")
        return key

    def read():
        frame = cam.read()
        stamps.append(time.time())
        latest[0] = frame
        return frame

    def grab(n):
        for _ in range(3):                                   # frames queued while busy
            read()
        return [read() for _ in range(n)]

    def in_background(fn, status, panel):
        """fn() in a thread, while the window shows the live picture. -> fn's result, or None on Q."""
        out = {}

        def run():
            try:
                out["value"] = fn()
            except BaseException as e:
                out["error"] = e
        t = threading.Thread(target=run, daemon=True)
        t.start()
        while t.is_alive():
            c, _ = read()
            if show(c, ui.busy(status), ui.BLUE, panel, 0, live=True) == "q":
                return None
        if "error" in out:
            raise out["error"]
        return out["value"]

    def write(d, kind, frames):
        for k, (c, z) in enumerate(frames):
            save(d / f"{kind}_{k}", c, z, cam.meta())

    stage, n = "pick", 0
    before = d = ev = result = None
    passed = []                  # outlines of the boxes passed over with T, until the next pick succeeds
    failed, attempt = None, 0    # the last target that wasn't picked, and how many tries it has had
    size_missing, no_size = None, ""    # no box size given: no pick (see given_size); None: not looked yet
    size_missing_last = None             # the size (and where from) last used: logged when it changes

    def given_size():
        """((height, width, depth) mm, where from) or (None, None): this PC's (ADVANCED tab, or
        --box-size) when chosen there, else the robot's (cam.box_size, through the bridge); any side
        0 or missing counts as not set."""
        if boxsize["use_pc"]:
            s = boxsize["pc"]
            return (tuple(float(v) for v in s), "this PC") if s and all(v > 0 for v in s) else (None, None)
        s = robot_size()
        return (s, "robot") if s else (None, None)
    watcher, fell = None, None   # check v2: the pick being watched; where the last dropped box fell
    outcomes = []                # the check's outcome for each try on the current box
    cant_tell_run = 0            # "can't tell" results in a row for the current pick
    shown_until, result_img, check_img, picker_img, target_line = 0.0, None, None, None, ""
    picked_n, failed_n, flag_n = 0, 0, 0   # the tally
    flag_tries = 0               # the result screen: tries when a flag was just raised (0: none)
    history = []                 # the last picks for the window: (close-up, colour)
    try:
        while True:
            try:
                cmd = commands.get_nowait()
            except queue.Empty:
                cmd = ""
            words = cmd.split()
            if words[:1] == ["quit"]:
                break
            if words[:1] == ["reset"]:
                passed, failed = [], None
                log("boxes passed over forgotten")
            colour, depth = read()

            # ---- 1. PICK ------------------------------------------------------------------------
            if stage == "pick":
                size, size_from = given_size()
                if size is None:                  # no box size: no pick (the layers need its depth)
                    no_size = (("Box size not set: enter it in the robot's GUI" if args.ros and not boxsize["use_pc"]
                                else "Box size not set") +
                               (", or set it on this PC (ADVANCED: B)" if window else ": give --box-size H W D (mm)"))
                    if not size_missing:
                        event({"event": "no_target", "pick": n + 1, "reason": "box size not set"}, events)
                        log(f"no box size ({no_size}): not picking")
                    size_missing = True
                    stage, shown_until = "no_target", time.time() + RESULT_S
                    continue
                elif size_missing is None or (size, size_from) != size_missing_last:
                    log(f"box size {size[0]:.0f} high x {size[1]:.0f} wide x {size[2]:.0f} deep mm ({size_from})")
                size_missing_last = (size, size_from)
                size_missing, no_size = False, ""
                show(colour, f"PICK {n + 1}: TAKING PICTURES", ui.BLUE,
                     [("label", "PICKER"), ("text", f"Taking {args.frames} new frames...")], 0, live=True)
                flag_tries = 0
                t0 = time.time()
                frames = grab(args.frames)
                t1 = time.time()
                up = tilt()                                   # the tilt while these frames were taken

                def find():
                    r = detector.detect_frames(frames, cam.intr, up, avoid=passed, box_size=size)
                    if r is None and passed:      # no box after the passed ones: back to the first
                        passed.clear()
                        r = detector.detect_frames(frames, cam.intr, up, box_size=size)
                    return r, remember(detector, frames, r)
                found = in_background(find, f"PICK {n + 1}: FINDING THE TARGET",
                                      [("label", "PICKER"), ("text", f"SAM 3.1 on {args.frames} frames...")])
                if found is None:
                    break
                result, before = found
                took = (t1 - t0, time.time() - t1)            # taking the pictures, finding the target
                if before is not None:                        # the pick face's rectangle, for the window
                    before.face_px = detector.last[1][detector.last[2]].pick.corners_px
                n += 1
                d = session / f"pick_{n:02d}"
                d.mkdir(exist_ok=True)
                write(d, "before", frames)
                if before is None:
                    why = carton_line(detector.res)[0] if detector.res is not None else ""
                    event({"event": "no_target", "pick": n, "carton": carton.summary(
                        detector.res and detector.res.get("carton"))}, events)
                    log(f"pick {n}: no target" + (f" ({why})" if why else ""))
                    stage, shown_until = "no_target", time.time() + RESULT_S
                    continue
                # A retry when the picker chose the box that just failed again.
                # (check v2: or the box chosen is where the dropped one fell).
                box = detector.last[1][detector.last[2]]
                attempt = attempt + 1 if failed is not None and (same_target(failed, result)
                                                                 or pw.fell_there(fell, box.pick)) else 1
                outcomes = outcomes if attempt > 1 else []
                picker_img = detector.annotate(frames[detector.best_frame][0])
                cv2.imwrite(str(d / "picker.jpg"), picker_img)
                target_crop = ui.close_up(before.colour, before)
                event({"event": "target", "pick": n, "attempt": attempt, "target": result}, events)
                target_line = (f"Pick {n} (try {attempt}): {where(result)}, by its {result['approach']}"
                               f"  x {result['x_m']:+.3f}  y {result['y_m']:+.3f}  z {result['z_m']:.3f} m"
                               f"  (tilt {np.degrees(np.arctan2(-up[2], -up[1])):.1f} deg)")
                log(target_line)
                if result.get("top_only") and attempt == 1:
                    flag = raise_flag(session, n, picker_img, before, name="toponly")
                    event({"event": "flag", "pick": n, "reason": "top face only", "target": result, **flag}, events)
                    log(f"pick {n}: FLAG: top face only (a grid box with no front found); picking it by its top; "
                        f"photos in {Path(flag['photo']).parent}")
                # Check v2: from now on every frame is compared with the depth before the pick: all the frames
                # the target was found in (depth that flickers between them is not compared), with the box's
                # size (how far back its face must read to be gone).
                watcher = pw.PickWatcher(box.pick, frames[detector.best_frame][1], box.mask,
                                         refs=[f[1] for i, f in enumerate(frames) if i != detector.best_frame],
                                         box_mm=size) if args.check == "v2" else None
                watch_t0, arm_out, holding = time.time(), False, None
                stage, cant_tell_run = "wait", 0
                continue
            if stage == "no_target":
                panel = [("title", "No target"), ("text", "No box the robot can pick right now.")]
                if size_missing:
                    panel += [("text", no_size, ui.RED)]
                elif detector.res is not None and detector.res.get("carton") is not None:
                    panel += [("text", *carton_line(detector.res))]
                panel += [None, ("label", "NEXT"), ("text", f"Looking again in {max(shown_until - time.time(), 0):.0f} s")]
                key = show(colour, "NO TARGET", ui.GREY, panel, 0, live=True)
                if key == "q":
                    break
                if key == "r":
                    passed, failed = [], None
                if (time.time() >= shown_until or key in ("n", "w")
                        or words[:1] in (["pick"], ["reset"])):
                    stage = "pick"
                continue

            # ---- 2. WAIT for the robot's "done" ---------------------------------------------------
            if stage == "wait":
                if words[:1] == ["pick"]:
                    stage = "pick"
                    continue
                r, tries = result, args.retries + 1
                panel = [("label", "TARGET"), ("title", f"{r['approach'].capitalize()} face"),
                         ("text", where(r)[0].upper() + where(r)[1:], ui.ORANGE) if r["fallen"] else ("text", where(r)),
                         ("image", target_crop),
                         ("coords", [("X  RIGHT", f"{r['x_m'] * 100:+.1f} cm"), ("Y  DOWN", f"{r['y_m'] * 100:+.1f} cm"),
                                     ("Z  AWAY", f"{r['z_m'] * 100:.1f} cm")]),
                         None, ("dots", min(attempt, tries), tries, ui.ORANGE if attempt > 1 else ui.BLUE),
                         *([("text", f"Try {attempt}: flagged after {tries} tries, still trying", ui.RED)]
                           if attempt > tries else []), None,
                         *([("text", "FLAG: top face only. A grid box with no front found: "
                                     "picking it by its top.", ui.ORANGE), None] if r.get("top_only") else []),
                         *([("text", "Check the box size: " + detector.res["size_warning"], ui.ORANGE), None]
                           if detector.res is not None and detector.res.get("size_warning") else []),
                         ("small", f"{r['agreeing']} of {r['frames']} frames agree  \u00b7  {r['num_boxes']} boxes  "
                                   f"\u00b7  face {r['size_mm'][0]}\u00d7{r['size_mm'][1]} mm"),
                         ("small", f"found in {sum(took):.1f} s  \u00b7  pictures {took[0]:.1f} s  \u00b7  "
                                   f"picker {took[1]:.1f} s")]
                if watcher is not None:                  # check v2: what the camera sees of the pick
                    st = watcher.status
                    panel += [None, ("label", "PICK CHECK: CAMERA WATCHES"), ("text", st[0].upper() + st[1:])]
                key = show(colour, f"PICK {n}:  PICK IT" + (" (THE CAMERA WATCHES)" if watcher is not None
                                                            else ", THEN SAY DONE"), ui.BLUE, panel, 1, target=True)
                if key == "q":
                    break
                if key in ("n", "w"):              # look again, or the check changed
                    stage = "pick"
                    continue
                if key == "r":
                    passed, failed = [], None
                if key == "t":                    # pass over this box: the next one in the order
                    passed.append(before.target.copy())
                    failed, attempt = None, 0
                    log(f"pick {n}: passed over, next target")
                    stage, n = "pick", n - 1          # the same pick number for the next box
                    continue
                if watcher is not None:
                    # ---- 3. CHECK v2: every frame, until the camera has decided (no "done" needed) ----
                    # A "done" (robot_done.py: the arm is back) says the arm is out of the view: the camera
                    # then decides on what it sees (19 Nov 2026), and "held" / "empty" win over it.
                    if words[:1] == ["done"] or key == "c":
                        arm_out = True
                        holding = {"held": True, "empty": False}.get(words[1]) if len(words) > 1 else holding
                    outcome = watcher.update(depth, cam.intr, time.time(), arm_out=arm_out)
                    if outcome is None:
                        continue
                    if holding is True and outcome == "dropped":
                        outcome = "picked"
                    elif holding is False and outcome == "picked":
                        outcome = "dropped"
                    verdict = {"check": "v2", "outcome": outcome, "success": outcome == "picked",
                               "seconds": round(time.time() - watch_t0, 1),
                               "shift_mm": None if watcher.shift is None else round(watcher.shift * 1000),
                               "holding": holding, "arm_out": arm_out, "status": watcher.status,
                               "timeline": [list(x) for x in list(watcher.timeline)[-30:]]}
                    after = [(colour, depth)]
                    picked = {"picked": True, "camera_moved": None}.get(outcome, False)
                    result_img = check_img = watch_image(colour, watcher, outcome)
                    fell, watcher = (watcher.drop if outcome == "dropped" else None), None
                else:
                    done = words[:1] == ["done"] or key == "c"
                    if not done:
                        continue
                    holding = {"held": True, "empty": False}.get(words[1]) if words[:1] == ["done"] and len(words) > 1 else None

                    # ---- 3. CHECK v1: the robot said done; new frames against before the pick ----------
                    show(colour, f"PICK {n}: CHECKING THE PICK...", ui.BLUE,
                         [("label", "PICK CHECK"), ("text", f"Comparing {args.check_frames} new frames with before the pick...")], 2,
                         target=True)
                    after = grab(args.check_frames)
                    verdict = pc.check(before, after, holding)
                    result_img = pc.big_result(pc.draw(before, after, verdict), verdict)
                    check_img = ui.check_view(pc, before, after, verdict)
                    picked, fell = PICKED.get(verdict["outcome"]), None
                write(d, f"after_{time.strftime('%H%M%S')}", after)
                cv2.imwrite(str(d / "check.jpg"), result_img)
                ev = {"event": "result", "pick": n, "attempt": attempt, "picked": picked, "verdict": verdict,
                      "target": result}
                event(ev, events)
                log(f"pick {n} (try {attempt}): {result_text(verdict)[0]}")
                if picked is True:
                    failed, attempt, passed = None, 0, []
                    picked_n += 1
                elif picked is False:
                    failed = result
                    failed_n += 1
                    outcomes.append(verdict["outcome"])
                    if attempt == args.retries + 1:          # once per box; it is still tried
                        flag = raise_flag(session, n, picker_img, before, result_img)
                        event({"event": "flag", "pick": n,
                               "reason": f"not picked after {attempt} tries: still trying it",
                               "outcomes": outcomes, "target": result, **flag}, events)
                        log(f"pick {n}: FLAG: not picked after {attempt} tries ({', '.join(outcomes)}); still trying it; "
                            f"photos in {Path(flag['photo']).parent}")
                        flag_n, flag_tries = flag_n + 1, attempt
                history = (history + [(target_crop, {True: ui.GREEN, False: ui.RED}.get(picked, ui.ORANGE))])[-12:]
                cant_tell_run = cant_tell_run + 1 if picked is None else 0
                if cant_tell_run >= CANT_TELL:
                    log(f"pick {n}: can't tell {cant_tell_run} times in a row; choosing the target again")
                stage, shown_until = "result", time.time() + RESULT_S
                continue

            # ---- 4. NEXT ------------------------------------------------------------------------
            if stage == "result":
                # (check v2's can't tell -- camera moved -- chooses again: its watcher has decided)
                cant_tell = ev["picked"] is None and cant_tell_run < CANT_TELL and ev["verdict"].get("check") != "v2"
                nxt = "waiting for 'done' again" if cant_tell else "new pictures, choosing again" \
                    if ev["picked"] is None else "retry" if ev["picked"] is False else "next box"
                v = ev["verdict"]
                title, why, _ = result_text(v)
                col = {True: ui.GREEN, False: ui.RED}.get(ev["picked"], ui.ORANGE)
                if v.get("check") == "v2":
                    small = f"camera watched the pick  \u00b7  {v['seconds']:.1f} s" + \
                        (f"  \u00b7  face now {v['shift_mm']:+d} mm" if v["shift_mm"] is not None else "")
                else:
                    small = f"gone {v['gone']:.0%}  \u00b7  same {v['same']:.0%}  \u00b7  {v['by']}  \u00b7  {len(after)} frames" \
                        + (f"  \u00b7  {len(v['disturbed'])} other box(es) disturbed" if v["disturbed"] else "")
                panel = [("label", "RESULT"), ("title", title, col), ("text", why), ("small", small)]
                status = title
                if flag_tries:
                    panel += [None, ("title", "FLAG RAISED", ui.RED),
                              ("text", f"Not picked after {flag_tries} tries: a person is told (photos in flags/). "
                                       "It is still tried.")]
                    status = f"FLAG: PICK {n} NOT PICKED AFTER {flag_tries} TRIES"
                panel += [None, ("label", "NEXT"), ("text", nxt[0].upper() + nxt[1:]),
                          ("small", f"in {max(shown_until - time.time(), 0):.0f} s  \u00b7  N = now")]
                key = show(check_img, status, col, panel, 3)
                if key == "q":
                    break
                if key == "r":
                    passed, failed = [], None
                if cant_tell and words[:1] == ["done"]:
                    commands.put(cmd)                                 # the robot cleared the view: check again
                    stage = "wait"
                elif time.time() >= shown_until or key == "n" or words[:1] == ["pick"]:
                    # Can't tell (an arm or hand still in view): wait for the robot's next "done".
                    stage = "wait" if cant_tell and words[:1] != ["pick"] and key != "n" else "pick"
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if window:
            cv2.destroyAllWindows()
    log(f"saved: {session}")


if __name__ == "__main__":
    main()
