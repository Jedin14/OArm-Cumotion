"""The box picker, then the pick check, live and in a loop: one program, one camera, one window.

    1. PICK    the picker (SAM 3.1, PickDetector.detect_frames on 5 new frames) chooses a target;
               it is printed as a JSON event for the robot, and the scene is remembered.
    2. WATCH   the live feed, watched only around the target (no SAM): waiting for the arm ->
               arm in view -> arm gone and the scene still for STILL_S. Then:
    3. CHECK   3 new frames compared with the remembered ones (pick_check.check) -> PICK SUCCESS /
               PICK FAILED / CAN'T TELL, shown in large letters and printed as a JSON event.
               CAN'T TELL (an arm or hand still in view) goes back to WATCH.
    4. NEXT    after RESULT_S, back to 1 for the next target (a missed box is usually chosen again).

The end of the pick is noticed from the camera, or told: the robot (or you) can write a line on
stdin at any time:
    done | done held | done empty   the arm is out of view: check now (held / empty: what the gripper says)
    pick                            choose a new target now
    quit
stdout carries only JSON events, one a line: {"event": "ready"}, {"event": "target", ...},
{"event": "result", "picked": true / false / null, "verdict": {...}}, {"event": "no_target"}.
Everything else goes to stderr.

Keys in the window: C check now, N new target now, SPACE pick (with --manual-pick), Q / Esc quit.

Usage (from Pick Check/, with the picker's venv):
    $PY pick_and_check.py                        # D455 on this machine, window
    $PY pick_and_check.py --ros                  # D455 over ROS 2 (ros_bridge.py)
    $PY pick_and_check.py --manual-pick          # take a new target only on SPACE / "pick" (e.g. the robot drives in between)
    $PY pick_and_check.py --no-window            # robot only: stdin / stdout, no window
    -> outputs/pick_and_check/<session>/pick_NN/{before_*, after_*, picker.jpg, check.jpg, events.jsonl}
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

import pick_check as pc

STILL_S = 1.0          # the scene around the target must stay still this long before the check
PAD_PX = 80            # watched area: the target's outline box, this much larger on each side
ACTIVE_SHARE = 0.05    # this share of the watched area changed (vs before): something happened there
NEAR_SHARE = 0.03      # this share 5 cm or more nearer than before: the arm (or a hand) is there
MOVING_SHARE = 0.01    # this share changing from one frame to the next: still moving
PICKER_SHOW_S = 2.0    # the picker's own overlay is shown this long after each new target
RESULT_S = 3.0         # a result stays on screen this long before the next target
PICKED = {"success": True, "missed": False, "moved": False, "dropped": False, "wrong_box": False}
WINDOW = "pick and check"


def event(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Watch:
    """Follows the area around the target from frame to frame, cheaply (a crop, no SAM)."""

    def __init__(self, before):
        ys, xs = np.nonzero(before.target)
        h, w = before.target.shape
        self.y0, self.y1 = max(ys.min() - PAD_PX, 0), min(ys.max() + PAD_PX + 1, h)
        self.x0, self.x1 = max(xs.min() - PAD_PX, 0), min(xs.max() + PAD_PX + 1, w)
        self.ref = before.depth[self.y0:self.y1, self.x0:self.x1]
        self.had = self.ref > 0
        self.prev = None
        self.active = False                # the arm (or a hand) has been seen around the target
        self.still_since = None

    def update(self, colour, depth):
        """-> "waiting" (nothing yet), "arm" (in view or moving), "settling", or "ready" (check now)."""
        d = depth[self.y0:self.y1, self.x0:self.x1]
        g = cv2.cvtColor(colour[self.y0:self.y1, self.x0:self.x1], cv2.COLOR_BGR2GRAY).astype(np.int16)
        valid = d > 0
        changed = self.had & (~valid | (np.abs(d - self.ref) > pc.FAR_M))
        nearer = self.had & valid & (d < self.ref - pc.BLOCKED_M)
        moving = 0.0
        if self.prev is not None:
            pd, pg = self.prev
            both = valid & (pd > 0)
            moving = max(float((both & (np.abs(d - pd) > pc.FAR_M)).mean()),
                         float((np.abs(g - pg) > 30).mean()))
        self.prev = d, g
        if changed.mean() >= ACTIVE_SHARE:
            self.active = True
        now = time.time()
        if nearer.mean() >= NEAR_SHARE or moving >= MOVING_SHARE:
            self.still_since = None
            return "arm"
        self.still_since = self.still_since or now
        if not self.active:
            return "waiting"
        return "ready" if now - self.still_since >= STILL_S else "settling"


def read_stdin(q):
    for line in sys.stdin:
        q.put(line.strip().lower())
    q.put("quit")                          # stdin closed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ros", action="store_true", help="camera over ROS 2, through ros_bridge.py")
    ap.add_argument("--pitch", type=float, help="camera tilt down in degrees (default: accelerometer, else 0)")
    ap.add_argument("--manual-pick", action="store_true", help="new target only on SPACE / 'pick', not after each result")
    ap.add_argument("--no-window", action="store_true", help="no window: stdin / stdout only")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs" / "pick_and_check"))
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
    commands = queue.Queue()
    threading.Thread(target=read_stdin, args=(commands,), daemon=True).start()
    window = not args.no_window
    if window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)
    event({"event": "ready", "session": str(session)})

    def show(img, lines, colour=(255, 255, 255)):
        """Draw and return the key pressed (lower case), or ''."""
        if not window:
            return ""
        cv2.imshow(WINDOW, banner(img, lines, colour))
        k = cv2.waitKey(1) & 0xFF
        return " " if k in (13, 32) else chr(k).lower() if 0 < k < 128 and k != 27 else ("q" if k == 27 else "")

    def grab(n):
        for _ in range(3):                                   # frames queued while busy
            cam.read()
        return [cam.read() for _ in range(n)]

    def write(d, kind, frames):
        for k, (c, z) in enumerate(frames):
            save(d / f"{kind}_{k}", c, z, cam.meta())

    stage = "idle" if args.manual_pick else "pick"
    n, before, watch, d, target_line = 0, None, None, None, ""
    shown_until, result_img, picker_img = 0.0, None, None
    try:
        while True:
            try:
                cmd = commands.get_nowait()
            except queue.Empty:
                cmd = ""
            words = cmd.split()
            if words[:1] == ["quit"]:
                break
            colour, depth = cam.read()

            # ---- 1. PICK ------------------------------------------------------------------------
            if stage == "idle":
                key = show(colour, ["Arm OUT OF VIEW?  SPACE = choose a target   Q = quit"])
                if key == "q":
                    break
                if key in (" ", "n") or words[:1] == ["pick"]:
                    stage = "pick"
                continue
            if stage == "pick":
                show(colour, ["Running the picker (SAM 3.1)..."], (0, 255, 255))
                frames = grab(5)
                result = detector.detect_frames(frames, cam.intr, up)
                before = pc.remember(detector, frames, result)
                n += 1
                d = session / f"pick_{n:02d}"
                d.mkdir()
                write(d, "before", frames)
                if before is None:
                    event({"event": "no_target", "pick": n})
                    log(f"pick {n}: no target")
                    stage, shown_until = ("idle" if args.manual_pick else "no_target"), time.time() + RESULT_S
                    continue
                picker_img = detector.annotate(frames[detector.best_frame][0])
                cv2.imwrite(str(d / "picker.jpg"), picker_img)
                ev = {"event": "target", "pick": n, "target": result}
                event(ev)
                with open(d / "events.jsonl", "a") as f:
                    f.write(json.dumps(ev) + "\n")
                target_line = (f"Pick {n}: {result['approach']} box  x {result['x_m']:+.3f}  y {result['y_m']:+.3f}"
                               f"  z {result['z_m']:.3f} m")
                log(target_line)
                watch, stage, shown_until = Watch(before), "watch", time.time() + PICKER_SHOW_S
                continue
            if stage == "no_target":
                key = show(colour, ["Picker: NO TARGET found.  Trying again...   Q = quit"], (0, 0, 255))
                if key == "q":
                    break
                if time.time() >= shown_until or key == "n" or words[:1] == ["pick"]:
                    stage = "pick"
                continue

            # ---- 2. WATCH -----------------------------------------------------------------------
            if stage == "watch":
                state = watch.update(colour, depth)
                holding = None
                if words[:1] == ["done"]:
                    state, holding = "ready", {"held": True, "empty": False}.get(words[1] if len(words) > 1 else "")
                if words[:1] == ["pick"]:
                    stage = "pick"
                    continue
                status = {"waiting": "waiting for the arm...", "arm": "arm in view",
                          "settling": f"arm gone, settling {time.time() - (watch.still_since or time.time()):.1f} s",
                          "ready": "checking..."}[state]
                view = picker_img if time.time() < shown_until else pc_outline(colour, before)
                key = show(view, [target_line, f"Robot: PICK it.   {status}   (C check now, N new target, Q quit)"])
                if key == "q":
                    break
                if key == "n":
                    stage = "pick"
                    continue
                if key == "c":
                    state = "ready"
                if state != "ready":
                    continue

                # ---- 3. CHECK -------------------------------------------------------------------
                after = grab(3)
                verdict = pc.check(before, after, holding)
                write(d, f"after_{time.strftime('%H%M%S')}", after)
                result_img = pc.big_result(pc.draw(before, after, verdict), verdict)
                cv2.imwrite(str(d / "check.jpg"), result_img)
                ev = {"event": "result", "pick": n, "picked": PICKED.get(verdict["outcome"]),
                      "verdict": verdict, "target": before.result}
                event(ev)
                with open(d / "events.jsonl", "a") as f:
                    f.write(json.dumps(ev) + "\n")
                log(f"pick {n}: {pc.RESULT[verdict['outcome']][0]}")
                stage, shown_until = "result", time.time() + RESULT_S
                continue

            # ---- 4. NEXT ------------------------------------------------------------------------
            if stage == "result":
                cant_tell = PICKED.get(ev["verdict"]["outcome"]) is None
                key = show(result_img, [f"Next: {'watching again' if cant_tell else 'new target'} in "
                                        f"{max(shown_until - time.time(), 0):.0f} s   (N now, Q quit)"])
                if key == "q":
                    break
                if time.time() >= shown_until or key == "n" or words[:1] in (["pick"], ["done"]):
                    if cant_tell and words[:1] != ["pick"] and key != "n":
                        watch.still_since, stage = None, "watch"     # an arm or hand was still there: wait again
                    else:
                        stage = "idle" if args.manual_pick and words[:1] != ["pick"] else "pick"
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if window:
            cv2.destroyAllWindows()
    log(f"saved: {session}")


def banner(img, lines, colour=(255, 255, 255)):
    """A copy of img with a black bar on top, one line of text per entry."""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30 * len(lines) + 10), (0, 0, 0), -1)
    for k, line in enumerate(lines):
        cv2.putText(img, line, (10, 28 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    return img


def pc_outline(img, before):
    """The live frame with the target's outline and centre."""
    img = img.copy()
    contours, _ = cv2.findContours(before.target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
    cv2.drawMarker(img, (int(before.result["u"]), int(before.result["v"])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    return img


if __name__ == "__main__":
    main()
