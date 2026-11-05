"""Pick, check, next: the box picker and the pick check in one loop, for the robot. One program,
one camera (only one program can open the D455).

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
                                   After 1 + --retries tries on the same box it is skipped: left out
                                   of the picker's choice from then on (it still blocks the boxes
                                   behind and under it), a FLAG is raised with photos of the box for
                                   a person, and the next box is taken.
               can't tell       -> 2 again (an arm or hand still in view: the robot clears the view
                                   and says "done" again). After CANT_TELL of them in a row (something
                                   new stays in view), back to 1: new frames, the target chosen again.
    No target (empty, or only skipped boxes and the ones they block): the picker tries again every
    few seconds. "reset" forgets the skipped boxes.

The pick check's code (pick_check.py, and pick_and_check.py for drawing) belongs to the Pick Check
team and is used
unchanged: here, from ../../Pick Check/ (or this folder, in the robot package).

stdin, one line at a time (the robot, or you):
    done | done held | done empty   the pick is over, the arm out of view: check it (held / empty: what
                                    the gripper says, so a box that left its place but isn't held is "dropped")
    pick                            choose a new target now
    reset                           forget the skipped boxes
    quit
stdout carries only JSON events, one a line; everything else goes to stderr:
    {"event": "ready", ...}
    {"event": "target", "pick": n, "attempt": k, "target": {...}}     go and pick it (attempt k on this box)
    {"event": "result", "pick": n, "attempt": k, "picked": true / false / null, "verdict": {...}}
    {"event": "flag", "pick": n, "reason": "...", "outcomes": [...], "target": {...},
     "photo": ".../target.jpg", "crop": ".../crop.jpg", "check_photo": ".../check.jpg"}
                                     given up on this box after k tries: it is skipped. photo: the scene
                                     with the target marked; crop: the box close up; check_photo: why
                                     the last try failed
    {"event": "no_target", "pick": n, "skipped": m}

Keys in the window (for testing without the robot), also buttons on the panel's two tabs:
    PICK       C = "done" (check now), T next target (pass over this box until the next successful
               pick), A the ADVANCED tab, Q / Esc quit
    ADVANCED   D depth picture, V the picker's view (all boxes and faces), S save colour + depth
               (snapshots/), O order v1 / v2 (looks again), N new target now (new frames),
               R forget the skipped boxes, − / + frames per target, SAM confidence (--score) and
               frames per pick success (from the next one on)

Usage (from this folder):
    .venv/bin/python pick_loop.py --pitch 0 --no-window      # for the robot: stdin / stdout only
    .venv/bin/python pick_loop.py --pitch 0                  # with a window (C stands in for the robot's "done")
    .venv/bin/python pick_loop.py --pitch 0 --ros            # camera over ROS 2 (ros_bridge.py)
    -> outputs/pick_loop/<session>/pick_NN/{before_*, after_*, picker.jpg, check.jpg}, events.jsonl,
       flags/flag_NN_{target,crop,check}.jpg
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

try:
    import pick_check as pc
except ImportError:                       # in the repository, the check lives in Pick Check/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Pick Check"))
    import pick_check as pc
from pick_and_check import PICKED, RESULT_S, pc_outline

import loop_screen as ui

RETRIES = 2              # a box that still isn't picked after this many retries is skipped
# The ADVANCED tab's - / + settings: label, lowest, highest, step
FRAMES = {"frames": ("Frames per target", 1, 15, 1), "check_frames": ("Frames per pick success", 1, 10, 1),
          "score": ("SAM confidence", 0.2, 0.9, 0.05)}
CANT_TELL = 3            # this many "can't tell" in a row for one pick: choose the target again
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


def raise_flag(session, n, picker_img, before, check_img):
    """Photos for a person, of a box given up on: the scene with the target marked (the picker's
    overlay), the box close up, and the last check (why it failed). -> their paths."""
    d = session / "flags"
    d.mkdir(exist_ok=True)
    ys, xs = np.nonzero(before.target)
    pad = 40
    h, w = before.target.shape
    crop = pc_outline(before.colour, before)[max(ys.min() - pad, 0):min(ys.max() + pad, h),
                                             max(xs.min() - pad, 0):min(xs.max() + pad, w)]
    paths = {"photo": d / f"flag_{n:02d}_target.jpg", "crop": d / f"flag_{n:02d}_crop.jpg",
             "check_photo": d / f"flag_{n:02d}_check.jpg"}
    for key, img in (("photo", picker_img), ("crop", crop), ("check_photo", check_img)):
        cv2.imwrite(str(paths[key]), img)
    return {k: str(v) for k, v in paths.items()}


def read_stdin(q):
    for line in sys.stdin:
        q.put(line.strip().lower())
    q.put("quit")                          # stdin closed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pitch", type=float, help="camera tilt down in degrees (default: accelerometer, else 0)")
    ap.add_argument("--ros", action="store_true", help="camera over ROS 2, through ros_bridge.py")
    ap.add_argument("--retries", type=int, default=RETRIES, help="retries on the same box before it is skipped")
    ap.add_argument("--frames", type=int, default=5, help="frames per picker answer")
    ap.add_argument("--check-frames", type=int, default=3, help="new frames the pick check compares with before")
    ap.add_argument("--score", type=float, default=0.4,
                    help="SAM's confidence needed to count as a box (thin standing boxes score 0.45-0.6)")
    ap.add_argument("--no-window", action="store_true", help="no window: stdin / stdout only")
    ap.add_argument("--order", choices=["v1", "v2"], default="v1",
                    help="v1: stack tops (highest first), then nearest, then leftmost; v2: leftmost free box")
    ap.add_argument("--out", default="outputs/pick_loop", help="where each session is saved")
    args = ap.parse_args()

    real_stdout, sys.stdout = sys.stdout, sys.stderr  # library prints go to stderr
    from faces import up_from_accel, up_from_pitch
    from pick_frame import PickDetector
    from rgbd import open_camera, save
    detector = PickDetector(score=args.score, order=args.order)
    cam = open_camera(args.ros)
    up = up_from_accel(cam.accel) if cam.accel and args.pitch is None else up_from_pitch(args.pitch or 0)
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

    def advanced():
        """The ADVANCED tab's panel."""
        return [("label", "VIEW"),
                ("button", "D", "Depth picture", "on" if depth_on else "off", depth_on),
                ("button", "V", "Picker's view: all boxes", "on" if picker_on else "off", picker_on),
                ("button", "S", "Save colour + depth", snap),
                None, ("label", "TARGET"),
                ("button", "O", "Order", "v2: leftmost" if detector.order == "v2" else "v1: stack tops",
                 detector.order == "v2"),
                ("button", "N", "Look again (new frames)", ""),
                ("button", "R", "Forget skipped boxes", f"{len(skipped)} skipped" if skipped else "none"),
                ("setting", "frames", FRAMES["frames"][0], str(args.frames), False),
                ("setting", "score", FRAMES["score"][0], f"{args.score:.2f}", False),
                None, ("label", "PICK CHECK"),
                ("setting", "check_frames", FRAMES["check_frames"][0], str(args.check_frames), False)]

    def show(img, status, colour, panel, step, live=False, target=False):
        """Draw the window (loop_screen) and return the key pressed or clicked (lower case), or ''.
        live: img is the live colour picture (the depth picture instead when that is on); target:
        draw the target over it. A, D, V, O and S are handled here, in every step."""
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
        canvas = ui.render(view, status, colour, advanced() if tab == "advanced" else panel,
                           f"Picked {picked_n}     Failed tries {failed_n}     Skipped {len(skipped)}",
                           step, fps, history if tab == "pick" else (), tab, depth)
        if not sized:
            cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
            sized = True
        cv2.imshow(WINDOW, canvas)
        k = cv2.waitKey(1) & 0xFF
        key = "q" if k == 27 else chr(k).lower() if 0 < k < 128 else clicks.popleft() if clicks else ""
        if key == "a":
            tab = "advanced" if tab == "pick" else "pick"
        elif key == "d":
            depth_on = not depth_on
        elif key == "v":
            picker_on = not picker_on
        elif key == "o":
            detector.order = "v2" if detector.order == "v1" else "v1"
            log(f"order: {detector.order}")
        elif key[2:] in FRAMES:                       # frames, SAM's confidence: from the next target / check on
            name = key[2:]
            _, lo, hi, step = FRAMES[name]
            setattr(args, name, round(min(max(getattr(args, name) + (step if key[0] == "+" else -step), lo), hi), 2))
            if name == "score":
                detector.finder.processor.confidence_threshold = args.score
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
    skipped = []                 # outlines of the boxes given up on (the picker's `avoid`)
    passed = []                  # outlines of the boxes passed over with T, until the next pick succeeds
    failed, attempt = None, 0    # the last target that wasn't picked, and how many tries it has had
    outcomes = []                # the check's outcome for each try on the current box
    cant_tell_run = 0            # "can't tell" results in a row for the current pick
    shown_until, result_img, check_img, picker_img, target_line = 0.0, None, None, None, ""
    picked_n, failed_n, flag_tries = 0, 0, 0   # the tally, and the tries before the last flag (0: none)
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
                skipped, passed, failed = [], [], None
                log("skipped boxes forgotten")
            colour, depth = read()

            # ---- 1. PICK ------------------------------------------------------------------------
            if stage == "pick":
                show(colour, f"PICK {n + 1}: TAKING PICTURES", ui.BLUE,
                     [("label", "PICKER"), ("text", f"Taking {args.frames} new frames...")], 0, live=True)
                flag_tries = 0
                t0 = time.time()
                frames = grab(args.frames)
                t1 = time.time()

                def find():
                    r = detector.detect_frames(frames, cam.intr, up, avoid=skipped + passed)
                    if r is None and passed:      # no box after the passed ones: back to the first
                        passed.clear()
                        r = detector.detect_frames(frames, cam.intr, up, avoid=skipped)
                    return r, pc.remember(detector, frames, r)
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
                    event({"event": "no_target", "pick": n, "skipped": len(skipped)}, events)
                    log(f"pick {n}: no target" + (f" ({len(skipped)} box(es) skipped)" if skipped else ""))
                    stage, shown_until = "no_target", time.time() + RESULT_S
                    continue
                # A retry when the picker chose the box that just failed again.
                attempt = attempt + 1 if failed is not None and pc.same_target(failed, result) else 1
                outcomes = outcomes if attempt > 1 else []
                picker_img = detector.annotate(frames[detector.best_frame][0])
                cv2.imwrite(str(d / "picker.jpg"), picker_img)
                target_crop = ui.close_up(before.colour, before)
                event({"event": "target", "pick": n, "attempt": attempt, "target": result}, events)
                target_line = (f"Pick {n} (try {attempt}): {result['approach']} box  x {result['x_m']:+.3f}"
                               f"  y {result['y_m']:+.3f}  z {result['z_m']:.3f} m")
                log(target_line)
                stage, cant_tell_run = "wait", 0
                continue
            if stage == "no_target":
                panel = [("title", "No target"), ("text", "No box the robot can pick right now.")]
                if skipped:
                    panel += [None, ("text", f"{len(skipped)} box(es) skipped after failed tries. "
                                             "Press R to try them again.", ui.ORANGE)]
                panel += [None, ("label", "NEXT"), ("text", f"Looking again in {max(shown_until - time.time(), 0):.0f} s")]
                key = show(colour, "NO TARGET", ui.GREY, panel, 0, live=True)
                if key == "q":
                    break
                if key == "r":
                    skipped, passed, failed = [], [], None
                if time.time() >= shown_until or key in ("n", "o") or words[:1] in (["pick"], ["reset"]):
                    stage = "pick"
                continue

            # ---- 2. WAIT for the robot's "done" ---------------------------------------------------
            if stage == "wait":
                if words[:1] == ["pick"]:
                    stage = "pick"
                    continue
                r, tries = result, args.retries + 1
                panel = [("label", "TARGET"), ("title", f"{r['approach'].capitalize()} face"), ("image", target_crop),
                         ("coords", [("X  RIGHT", f"{r['x_m'] * 100:+.1f} cm"), ("Y  DOWN", f"{r['y_m'] * 100:+.1f} cm"),
                                     ("Z  AWAY", f"{r['z_m'] * 100:.1f} cm")]),
                         None, ("dots", attempt, tries, ui.ORANGE if attempt > 1 else ui.BLUE), None,
                         ("small", f"{r['agreeing']} of {r['frames']} frames agree  \u00b7  {r['num_boxes']} boxes  "
                                   f"\u00b7  face {r['size_mm'][0]}\u00d7{r['size_mm'][1]} mm"),
                         ("small", f"found in {sum(took):.1f} s  \u00b7  pictures {took[0]:.1f} s  \u00b7  "
                                   f"picker {took[1]:.1f} s")]
                key = show(colour, f"PICK {n}:  PICK IT, THEN SAY DONE", ui.BLUE, panel, 1, target=True)
                if key == "q":
                    break
                if key in ("n", "o"):                # o: the order changed, choose again
                    stage = "pick"
                    continue
                if key == "r":
                    skipped, passed, failed = [], [], None
                if key == "t":                    # pass over this box: the next one in the order
                    passed.append(before.target.copy())
                    failed, attempt = None, 0
                    log(f"pick {n}: passed over, next target")
                    stage, n = "pick", n - 1          # the same pick number for the next box
                    continue
                done = words[:1] == ["done"] or key == "c"
                if not done:
                    continue
                holding = {"held": True, "empty": False}.get(words[1]) if words[:1] == ["done"] and len(words) > 1 else None

                # ---- 3. CHECK -------------------------------------------------------------------
                show(colour, f"PICK {n}: CHECKING THE PICK...", ui.BLUE,
                     [("label", "PICK CHECK"), ("text", f"Comparing {args.check_frames} new frames with before the pick...")], 2,
                     target=True)
                after = grab(args.check_frames)
                verdict = pc.check(before, after, holding)
                write(d, f"after_{time.strftime('%H%M%S')}", after)
                result_img = pc.big_result(pc.draw(before, after, verdict), verdict)
                cv2.imwrite(str(d / "check.jpg"), result_img)
                check_img = ui.check_view(pc, before, after, verdict)
                picked = PICKED.get(verdict["outcome"])
                ev = {"event": "result", "pick": n, "attempt": attempt, "picked": picked, "verdict": verdict,
                      "target": result}
                event(ev, events)
                log(f"pick {n} (try {attempt}): {pc.RESULT[verdict['outcome']][0]}")
                if picked is True:
                    failed, attempt, passed = None, 0, []
                    picked_n += 1
                elif picked is False:
                    failed = result
                    failed_n += 1
                    outcomes.append(verdict["outcome"])
                    if attempt > args.retries:
                        skipped.append(before.target.copy())
                        flag = raise_flag(session, n, picker_img, before, result_img)
                        event({"event": "flag", "pick": n,
                               "reason": f"not picked after {attempt} tries: skipped, taking the next box",
                               "outcomes": outcomes, "target": result, **flag}, events)
                        log(f"pick {n}: FLAG: gave up on this box after {attempt} tries ({', '.join(outcomes)}); "
                            f"photos in {Path(flag['photo']).parent}")
                        failed, attempt, flag_tries = None, 0, attempt
                history = (history + [(target_crop, {True: ui.GREEN, False: ui.RED}.get(picked, ui.ORANGE))])[-12:]
                cant_tell_run = cant_tell_run + 1 if picked is None else 0
                if cant_tell_run >= CANT_TELL:
                    log(f"pick {n}: can't tell {cant_tell_run} times in a row; choosing the target again")
                stage, shown_until = "result", time.time() + RESULT_S
                continue

            # ---- 4. NEXT ------------------------------------------------------------------------
            if stage == "result":
                cant_tell = ev["picked"] is None and cant_tell_run < CANT_TELL
                nxt = "waiting for 'done' again" if cant_tell else "new pictures, choosing again" \
                    if ev["picked"] is None else "skipping this box, next box" if failed is None and \
                    ev["picked"] is False else "retry" if ev["picked"] is False else "next box"
                v = ev["verdict"]
                title, why, _ = pc.RESULT[v["outcome"]]
                col = {True: ui.GREEN, False: ui.RED}.get(PICKED.get(v["outcome"]), ui.ORANGE)
                panel = [("label", "RESULT"), ("title", title, col),
                         ("text", why),
                         ("small", f"gone {v['gone']:.0%}  \u00b7  same {v['same']:.0%}  \u00b7  {v['by']}  \u00b7  {len(after)} frames"
                                   + (f"  \u00b7  {len(v['disturbed'])} other box(es) disturbed" if v["disturbed"] else ""))]
                status = title
                if flag_tries:
                    panel += [None, ("title", "FLAG RAISED", ui.RED),
                              ("text", f"Not picked after {flag_tries} tries: skipped. Photos saved in flags/.")]
                    status = f"FLAG: PICK {n} SKIPPED AFTER {flag_tries} TRIES"
                panel += [None, ("label", "NEXT"), ("text", nxt[0].upper() + nxt[1:]),
                          ("small", f"in {max(shown_until - time.time(), 0):.0f} s  \u00b7  N = now")]
                key = show(check_img, status, col, panel, 3)
                if key == "q":
                    break
                if key == "r":
                    skipped, passed, failed = [], [], None
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
