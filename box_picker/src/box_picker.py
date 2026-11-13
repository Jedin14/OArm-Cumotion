"""One entry point for the robot code: find the target box, then check every pick.

Feed it every camera frame. While no pick is being watched it looks for a target
(SAM 3.1 + depth, pick_frame.PickDetector); once one is handed out, it only runs the
pick check (pick_watch.PickTracker, a few ms a frame) until the result is in.

    from box_picker import BoxPicker
    picker = BoxPicker()                                  # loads SAM 3.1 once (~6 s, 3.6 GB VRAM)
    while True:
        color, depth_m, intr, t = ...                     # BGR (H, W, 3); depth in metres, aligned to colour;
                                                          # faces.Intrinsics; time the frame was taken (s)
        for event in picker.step(color, depth_m, intr, t):
            send_to_robot(event)                          # JSON-serialisable dicts, see below

Events:
    {"event": "target", "pick": 3, "attempt": 1, "target": {"x_m": .., "y_m": .., "z_m": .., "approach": "front",
     "u": .., "v": .., "size_mm": [w, h], "num_boxes": 7, "timings": {..}}}
        -> go and pick the box at (x_m, y_m, z_m), camera frame, metres
    {"event": "pick_result", "pick": 3, "attempt": 1, "result": "picked" | "missed" | "moved" | "dropped",
     "success": true | false, "seconds": 4.2, "shift_mm": .., "target": {..}}
        -> success false: the box is still in the carton; the next target event for it
           (same place, or where it fell) comes with "attempt": 2, 3, ...

What is new against the earlier version:
  - the last box in the carton is found (box_masks.lone_boxes): SAM is much less sure
    of a box alone in an empty carton, so it fell under the score threshold;
  - "dropped": the box was lifted but fell back into the carton. "picked" now means the
    box has left the camera's view, so it is reported later than before (once the arm
    has carried it out of the picture).
The pick check counts frames (pick_watch.py: STEADY_FRAMES, DROP_FRAMES, CLEAR_FRAMES,
SETTLE_FRAMES), tuned for a 30 fps camera; feed it every frame, not a subsample.

Try it on a recording (prints the events):
    python box_picker.py recordings/test2_d
"""
import argparse
import glob
import json
import time

import numpy as np

from faces import DEFAULT_DEPTH_TOL, up_from_accel, up_from_pitch
from pick_watch import PickTracker, draw_pick
from pick_frame import PickDetector
from pick_pipeline import draw


class BoxPicker:
    def __init__(self, tol=0.5, depth_tol=DEFAULT_DEPTH_TOL, up=None, accel=None, pitch_deg=0.0):
        """tol: top-row leeway as a fraction of box height. depth_tol: front-layer thickness, metres.
        The up direction (camera frame): `up` if given, else from the D455 accelerometer
        reading `accel`, else from the camera's tilt `pitch_deg` (degrees down, 0 = level)."""
        self.detector = PickDetector(tol, depth_tol)
        self.tracker = PickTracker()
        if up is not None:
            self.up = np.asarray(up, float)
        else:
            self.up = up_from_accel(accel) if accel else up_from_pitch(pitch_deg)
        self.last_detection = None          # (points, boxes, target index) of the last search, for annotate()

    def step(self, color, depth_m, intr, t=None):
        """One camera frame -> list of events (usually empty). t: when the frame was taken,
        seconds, same clock for every frame (default: now)."""
        t = time.time() if t is None else t
        events = [self.tracker.update(depth_m, intr, t)]
        if self.tracker.watch is None and events[0] is None:
            target = self.detector.detect(color, depth_m, intr, self.up)
            self.last_detection = self.detector.last
            if target is not None:
                _, boxes, idx = self.detector.last
                events.append(self.tracker.offer(target, boxes[idx].pick, depth_m, t, boxes[idx].mask))
        return [e for e in events if e is not None]

    @property
    def busy(self):
        """True while a handed-out target is being watched (the robot is picking it)."""
        return self.tracker.watch is not None

    @property
    def status(self):
        return self.tracker.status

    def reset(self):
        """Forget the target being watched (after boxes were moved by hand, say)."""
        self.tracker.reset()

    def annotate(self, color, now=None):
        """The colour frame with the last search's boxes, the target being watched and the
        last result's banner drawn on it (for a debug window or saved pictures)."""
        img = color.copy()
        if self.last_detection is not None:
            img = draw(img, *self.last_detection)
        return draw_pick(img, self.tracker, time.time() if now is None else now)


def main():
    from rgbd import load
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="folder of frame_NNNNN_* captures (record_rgbd.py)")
    ap.add_argument("--start", type=int, default=0, help="index of the first frame to use")
    ap.add_argument("--count", type=int, help="number of frames")
    ap.add_argument("--pitch", type=float, help="camera tilt down in degrees (default: accelerometer, else 0)")
    args = ap.parse_args()
    prefixes = sorted(p[:-len("_color.png")] for p in glob.glob(f"{args.recording}/frame_*_color.png"))
    prefixes = prefixes[args.start:None if args.count is None else args.start + args.count]
    picker = None
    for p in prefixes:
        color, depth, intr, accel = load(p)
        t = json.load(open(p + "_meta.json"))["t_s"]
        if picker is None:
            picker = BoxPicker(accel=None if args.pitch is not None else accel, pitch_deg=args.pitch or 0.0)
        for e in picker.step(color, depth, intr, t):
            print(json.dumps({"frame": p.rsplit("_", 1)[-1], **{k: v for k, v in e.items() if k != "target"}}),
                  flush=True)


if __name__ == "__main__":
    main()
