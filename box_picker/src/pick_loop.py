"""Pick, check, next: the box picker and the pick check in one loop, for the robot. One program,
one camera (only one program can open the D455).

The camera has two jobs: before a pick, which box and where (the target); after it, whether it
worked (the box is gone from its place). When the pick happens is the robot's: it says "done".

    1. PICK    5 new frames -> PickDetector.detect_frames -> the target, printed as a JSON event.
               The pick check remembers the scene (pick_check.remember: the frames' median depth and
               the target's outline).
    2. WAIT    for the robot's "done" on stdin: the pick is over and the arm is out of view.
    3. CHECK   3 new frames compared with the remembered scene (pick_check.check): was the pick
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
unchanged (in this folder).

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

Keys in the window (for testing without the robot): C = "done" (check now), N new target now,
R forget the skipped boxes, Q / Esc quit.

Usage (from src/):
    .venv/bin/python pick_loop.py --pitch 0 --no-window      # for the robot: stdin / stdout only
    .venv/bin/python pick_loop.py --pitch 0                  # with a window (C stands in for the robot's "done")
    .venv/bin/python pick_loop.py --pitch 0 --ros            # camera over ROS 2 (ros_bridge.py)
    -> outputs/pick_loop/<session>/pick_NN/{before_*, after_*, picker.jpg, check.jpg}, events.jsonl,
       flags/flag_NN_{target,crop,check}.jpg
"""
import argparse
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
from pick_and_check import PICKED, PICKER_SHOW_S, RESULT_S, banner, pc_outline

RETRIES = 2              # a box that still isn't picked after this many retries is skipped
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


def flag_banner(img, tries):
    """The check's picture with a FLAG line on top: shown before the next box."""
    img = img.copy()
    cv2.rectangle(img, (0, 40), (img.shape[1], 90), (0, 0, 200), -1)
    cv2.putText(img, f"FLAG: not picked after {tries} tries - skipped, photos saved", (20, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


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
    ap.add_argument("--no-window", action="store_true", help="no window: stdin / stdout only")
    ap.add_argument("--out", default="outputs/pick_loop", help="where each session is saved")
    args = ap.parse_args()

    real_stdout, sys.stdout = sys.stdout, sys.stderr  # library prints go to stderr
    from faces import up_from_accel, up_from_pitch
    from pick_frame import PickDetector
    from rgbd import open_camera, save
    detector = PickDetector()
    cam = open_camera(args.ros)
    up = up_from_accel(cam.accel) if cam.accel and args.pitch is None else up_from_pitch(args.pitch or 0)
    sys.stdout = real_stdout
    session = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True)
    events = session / "events.jsonl"
    commands = queue.Queue()
    threading.Thread(target=read_stdin, args=(commands,), daemon=True).start()
    window = not args.no_window
    if window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)
    event({"event": "ready", "session": str(session), "retries": args.retries}, events)

    def show(img, lines, colour=(255, 255, 255)):
        """Draw and return the key pressed (lower case), or ''."""
        if not window:
            return ""
        cv2.imshow(WINDOW, banner(img, lines, colour))
        k = cv2.waitKey(1) & 0xFF
        return "q" if k == 27 else chr(k).lower() if 0 < k < 128 else ""

    def grab(n):
        for _ in range(3):                                   # frames queued while busy
            cam.read()
        return [cam.read() for _ in range(n)]

    def write(d, kind, frames):
        for k, (c, z) in enumerate(frames):
            save(d / f"{kind}_{k}", c, z, cam.meta())

    stage, n = "pick", 0
    before = d = ev = result = None
    skipped = []                 # outlines of the boxes given up on (the picker's `avoid`)
    failed, attempt = None, 0    # the last target that wasn't picked, and how many tries it has had
    outcomes = []                # the check's outcome for each try on the current box
    cant_tell_run = 0            # "can't tell" results in a row for the current pick
    shown_until, result_img, picker_img, target_line = 0.0, None, None, ""
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
                skipped, failed = [], None
                log("skipped boxes forgotten")
            colour, depth = cam.read()

            # ---- 1. PICK ------------------------------------------------------------------------
            if stage == "pick":
                show(colour, ["Running the picker (SAM 3.1)..."], (0, 255, 255))
                frames = grab(args.frames)
                result = detector.detect_frames(frames, cam.intr, up, avoid=skipped)
                before = pc.remember(detector, frames, result)
                n += 1
                d = session / f"pick_{n:02d}"
                d.mkdir()
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
                event({"event": "target", "pick": n, "attempt": attempt, "target": result}, events)
                target_line = (f"Pick {n} (try {attempt}): {result['approach']} box  x {result['x_m']:+.3f}"
                               f"  y {result['y_m']:+.3f}  z {result['z_m']:.3f} m")
                log(target_line)
                stage, shown_until, cant_tell_run = "wait", time.time() + PICKER_SHOW_S, 0
                continue
            if stage == "no_target":
                key = show(colour, [f"Picker: NO TARGET{f' ({len(skipped)} skipped, R to forget them)' if skipped else ''}."
                                    "  Trying again...   Q quit"], (0, 0, 255))
                if key == "q":
                    break
                if key == "r":
                    skipped, failed = [], None
                if time.time() >= shown_until or key == "n" or words[:1] in (["pick"], ["reset"]):
                    stage = "pick"
                continue

            # ---- 2. WAIT for the robot's "done" ---------------------------------------------------
            if stage == "wait":
                if words[:1] == ["pick"]:
                    stage = "pick"
                    continue
                view = picker_img if time.time() < shown_until else pc_outline(colour, before)
                key = show(view, [target_line, "Robot: PICK it, then say \"done\".   (C = done, N new target, Q quit)"])
                if key == "q":
                    break
                if key == "n":
                    stage = "pick"
                    continue
                done = words[:1] == ["done"] or key == "c"
                if not done:
                    continue
                holding = {"held": True, "empty": False}.get(words[1]) if words[:1] == ["done"] and len(words) > 1 else None

                # ---- 3. CHECK -------------------------------------------------------------------
                after = grab(3)
                verdict = pc.check(before, after, holding)
                write(d, f"after_{time.strftime('%H%M%S')}", after)
                result_img = pc.big_result(pc.draw(before, after, verdict), verdict)
                cv2.imwrite(str(d / "check.jpg"), result_img)
                picked = PICKED.get(verdict["outcome"])
                ev = {"event": "result", "pick": n, "attempt": attempt, "picked": picked, "verdict": verdict,
                      "target": result}
                event(ev, events)
                log(f"pick {n} (try {attempt}): {pc.RESULT[verdict['outcome']][0]}")
                if picked is True:
                    failed, attempt = None, 0
                elif picked is False:
                    failed = result
                    outcomes.append(verdict["outcome"])
                    if attempt > args.retries:
                        skipped.append(before.target.copy())
                        flag = raise_flag(session, n, picker_img, before, result_img)
                        event({"event": "flag", "pick": n,
                               "reason": f"not picked after {attempt} tries: skipped, taking the next box",
                               "outcomes": outcomes, "target": result, **flag}, events)
                        log(f"pick {n}: FLAG: gave up on this box after {attempt} tries ({', '.join(outcomes)}); "
                            f"photos in {Path(flag['photo']).parent}")
                        result_img = flag_banner(result_img, attempt)
                        failed, attempt = None, 0
                cant_tell_run = cant_tell_run + 1 if picked is None else 0
                if cant_tell_run >= CANT_TELL:
                    log(f"pick {n}: can't tell {cant_tell_run} times in a row; choosing the target again")
                stage, shown_until = "result", time.time() + RESULT_S
                continue

            # ---- 4. NEXT ------------------------------------------------------------------------
            if stage == "result":
                cant_tell = ev["picked"] is None and cant_tell_run < CANT_TELL
                nxt = "waiting for 'done' again" if cant_tell else "new pictures, choosing again" \
                    if ev["picked"] is None else "FLAG raised, skipping it, next box" if failed is None and \
                    ev["picked"] is False else "retry" if ev["picked"] is False else "next box"
                key = show(result_img, [f"Next: {nxt} in {max(shown_until - time.time(), 0):.0f} s   (N now, Q quit)"])
                if key == "q":
                    break
                if key == "r":
                    skipped, failed = [], None
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
