"""The robot's pick check end to end: box_picker_server.Server's "pick" then "check", on synthetic
depth (test_pick_watch.py's scenes) played by a fake camera at 15 frames a second, as on Beast.
No SAM, no GPU: the target is the synthetic scene's box.

Each case is one of the 18-19 Nov 2026 problems on the robot:
  - "can't tell" though the box was taken: the depth at the picture's left edge flickers (D455),
    which read as the arm still in view;
  - "can't tell" though the box was never touched: the watcher waited for the arm for ever;
  - "picked" though the box was only pushed back 2 cm;
plus a miss decided while the robot can still press again, the arm (with the box) staying in view,
and the gripper's word when the camera can't settle.

Run, in the repository (from beast_app/):
    ../vlm/molmo2-box-picking/.venv/bin/python test_check_server.py
on the robot (from box_picker_app/, the box picker in ../box_picker; else --picker DIR):
    ../box_picker/src/.venv/bin/python test_check_server.py
"""
import argparse
import os
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--picker", help="the box picker's folder (with src/ and v1/), on the robot box_picker")
_args = _ap.parse_args()
if _args.picker:
    _dirs = [os.path.join(_args.picker, "v1"), os.path.join(_args.picker, "src")]
elif os.path.isdir(os.path.join(HERE, "..", "box_picking", "v1")):           # the repository
    _dirs = [os.path.join(HERE, "..", "box_picking", "v1"), os.path.join(HERE, "..", "vlm", "molmo2-box-picking")]
else:                                                                           # the robot's workspace
    _dirs = [os.path.join(HERE, "..", "box_picker", "v1"), os.path.join(HERE, "..", "box_picker", "src")]
sys.path[1:1] = [os.path.abspath(d) for d in _dirs]   # pick_watch.py (19 Nov 2026) first; the synthetic scenes

import box_picker_server as bps                                             # noqa: E402
import pick_watch as pw                                                     # noqa: E402
from faces import analyse, choose_target, up_from_pitch                     # noqa: E402
from test_faces import INTR, PITCH, box_world, check, render               # noqa: E402
from test_pick_watch import ARM_FRONT, FRONT_TARGET, WALL, beside, carried, flicker   # noqa: E402

FPS = 15.0
REST = [beside(FRONT_TARGET), WALL]


def depth_of(boxes, seed):
    return render(boxes, 1.5, 0.02, seed=seed, floor=True)[0]


class FakeCamera:
    """Plays a list of depth frames at FPS (each flickering at the left edge, as the D455's), then
    holds the last one. `at` counts the frames read."""
    def __init__(self, script):
        self.script, self.at, self.intr, self.accel, self.box_size = script, 0, INTR, None, None
        self.colour = np.full((720, 1280, 3), 120, np.uint8)

    def read(self):
        time.sleep(1 / FPS)
        d = self.script[min(self.at, len(self.script) - 1)]
        self.at += 1
        return self.colour, flicker(d, self.at)

    def meta(self):
        return {}

    def close(self):
        pass


class FakeDetector:
    """The synthetic scene's target box, as the grid detector would give it."""
    order, top_faces, best_frame = "grid", False, 0

    def __init__(self):
        depth, masks = render([FRONT_TARGET] + REST, 1.5, 0.02, seed=0, floor=True)
        boxes = analyse(depth, masks, INTR, up_from_pitch(PITCH))
        t = choose_target(boxes)
        b = boxes[t]
        ys, xs = np.nonzero(b.mask)
        self.box = SimpleNamespace(mask=b.mask, pick=b.pick, bbox=[xs.min(), ys.min(), xs.max(), ys.max()])
        self.last = None

    def detect_frames(self, frames, intr, up, avoid=(), box_size=None):
        f = self.box.pick
        self.last = (None, [self.box], 0)
        return {"u": float(f.centroid_px[0]), "v": float(f.centroid_px[1]), "x_m": float(f.centroid[0]),
                "y_m": float(f.centroid[1]), "z_m": float(f.centroid[2]), "approach": "front"}

    def annotate(self, img):
        return img

    def grid_picks(self):
        f = self.box.pick
        return {0: (0, {"corners_px": f.corners_px, "face": f.kind, "size_mm": f.size_mm, "fallen": False})}

    def left_out(self):
        return set()

    def carton(self):
        return None


def server(tmp):
    args = SimpleNamespace(frames=5, skip=0, check_frames=3, score=0.3, overlay=os.path.join(tmp, "o.jpg"),
                           out=tmp, record_checks=False)
    s = bps.Server(args, pw, FakeDetector(), None, up_from_pitch(PITCH), None)
    return s


def run(name, script, arm_out_frame, holding=None, box_size=(80, 100, 40)):
    """'pick' with the box size (H W D mm), the fake camera playing `script` (the pick's 5 frames
    first); 'check' once frame arm_out_frame has been read (the robot back at pre_pick) ->
    (the check's answer, s from the 'check' to the answer)."""
    with tempfile.TemporaryDirectory() as tmp:
        s = server(tmp)
        s.cam = bps.LockedCamera(FakeCamera(script))
        answer = s.pick(["box", *[str(v) for v in box_size]])
        assert answer["target"] is not None, answer
        while s.cam._cam.at < arm_out_frame and not s.watch.done.is_set():
            time.sleep(0.02)
        if s.watch.error is not None:
            raise s.watch.error
        t0 = time.time()
        got = s.check([holding] if holding else [])
        took = time.time() - t0
        time.sleep(0.3)                                  # the log is written by a thread
        logged = os.path.isdir(got["check"].get("log", "")) and any(
            f.endswith("_timeline.txt") for f in os.listdir(got["check"]["log"]))
        s.stop_watch()
    v = got["check"]
    print(f"  {name}: {v['outcome']} in {took:.1f} s after the check ({got['why']})")
    print("     timeline: " + " | ".join(f"{t:.1f} s {w}" for t, w in v["timeline"]))
    return v, took, logged


def frames(*parts):
    out = []
    for boxes, n, seed in parts:
        d = depth_of(boxes, seed)
        out += [d] * n
    return out


THERE = [FRONT_TARGET] + REST
ARM = [FRONT_TARGET, ARM_FRONT] + REST
GONE = REST
PUSHED_2CM = [box_world(-0.06, 0.47, (0.10, 0.08, 0.04))] + REST


def main():
    ok = True
    print("taken; the left edge flickers (was: can't tell, 'arm still in view'):")
    v, took, logged = run("result", frames((THERE, 5, 1), (THERE, 10, 2), (ARM, 15, 3), (carried(FRONT_TARGET) + REST, 15, 4),
                                           (GONE, 200, 5)), 5 + 10 + 15 + 15 + 10)
    ok &= check("picked, within 2 s of the check", v["outcome"] == "success" and took < 2, f"{v['outcome']}, {took:.1f} s")
    ok &= check("check logged", logged, v.get("log"))

    print("never touched, the arm never seen on the face (was: can't tell, 'waiting for the arm'):")
    v, took, _ = run("result", frames((THERE, 5, 1), (THERE, 60, 2)), 40)
    ok &= check("missed, within 2 s", v["outcome"] == "missed" and took < 2, f"{v['outcome']}, {took:.1f} s")

    print("arm came and went, box still there: a miss before the check is not final:")
    v, took, _ = run("result", frames((THERE, 5, 1), (THERE, 10, 2), (ARM, 15, 3), (THERE, 80, 4)), 5 + 10 + 15 + 60)
    ok &= check("missed", v["outcome"] == "missed" and took < 2, f"{v['outcome']}, {took:.1f} s")
    ok &= check("the early miss was held until the check", len(v["provisional"]) >= 1, f"{v['provisional']}")

    print("missed, then the robot presses again and takes it:")
    v, took, _ = run("result", frames((THERE, 5, 1), (THERE, 10, 2), (ARM, 15, 3), (THERE, 30, 4), (ARM, 15, 5),
                                      (GONE, 100, 6)), 5 + 10 + 15 + 30 + 15 + 40)
    ok &= check("picked (not the first miss)", v["outcome"] == "success", f"{v['outcome']}, provisional {v['provisional']}")

    print("box pushed 2 cm back and left there, box 80 x 100 x 40 mm (was: picked):")
    v, took, _ = run("result", frames((THERE, 5, 1), (THERE, 10, 2), (ARM, 15, 3), (PUSHED_2CM, 80, 4)), 5 + 10 + 15 + 40)
    ok &= check("not picked: moved", v["outcome"] == "moved", f"{v['outcome']}, face {v['shift_mm']} mm")

    print("the arm stays in view holding the box (was: picked when the time ran out):")
    held_in_view = carried(FRONT_TARGET) + REST
    script = frames((THERE, 5, 1), (THERE, 10, 2), (ARM, 15, 3), (held_in_view, 300, 4))
    v, took, _ = run("no gripper word", script, 5 + 10 + 15 + 30)
    ok &= check("can't tell, and says why", v["outcome"] == "blocked" and "edge" in v["detail"], v["detail"])
    v, took, _ = run("the gripper says held", script, 5 + 10 + 15 + 30, holding="held")
    ok &= check("held: success by the gripper's word", v["outcome"] == "success" and v["by_gripper"], f"{v['outcome']}")

    print(f"\n{'all passed' if ok else 'FAILED'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
