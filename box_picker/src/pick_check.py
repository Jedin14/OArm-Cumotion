"""Did the pick work? Compares the depth inside the target box's outline before and after.

The box picker (SAM 3.1) runs once, before the pick, as usual. This check needs no SAM and
no GPU: it keeps the target's outline and the median depth of the "before" frames, and after
the pick compares a few new frames with them, pixel by pixel (numpy, a few milliseconds).

A box that was taken leaves a hole: the surface inside its outline is now further away (the
box behind or under it, the shelf, the back wall) or has no depth reading at all. A box that
is still there gives the same depth, to within the sensor noise (1-2 mm at 0.5 m). The other
boxes' outlines are checked the same way, to catch a neighbour that was knocked or taken too.

Needs the camera in the same pose before and after (the robot base doesn't move during the
pick), and the arm, gripper and held box out of view in the "after" frames.

Python (robot side):
    import sys; sys.path.insert(0, "/home/cmp/Applications/box detection2/vlm/molmo2-box-picking")
    from pick_check import remember, check

    result = detector.detect_frames(frames, intr, UP)  # the picker, as usual
    before = remember(detector, frames, result)        # right after it; before.save("x.npz") to keep it
    ...                                                # the arm picks, then leaves the view
    verdict = check(before, [cam.read() for _ in range(3)], holding=None)   # holding: gripper says so, if it can

Usage (with the picker's venv):
    PY="../vlm/molmo2-box-picking/.venv/bin/python"
    $PY pick_check.py --before "../vlm/molmo2-box-picking/recordings/test2_d/frame_0007[5-9]" \\
                      --after  "../vlm/molmo2-box-picking/recordings/test2_d/frame_0009[0-4]"
    $PY pick_check.py --before saved.npz --after "after/frame_*"        # no SAM: a saved remember()
    $PY pick_check.py --live                                            # D455: before, Enter, after

Verdict (a dict):
    "outcome"      success   the target is gone from its place
                   missed    the target is still there, unchanged
                   moved     part of it changed: shifted, tipped, or pulled part way
                   dropped   the target is gone but the gripper is empty (holding=False)
                   replaced  another box stands where the target was (same depth, another print) and a
                             neighbour moved: most likely taken, and a neighbour slid or leaned into the gap
                             ("picked": true; with holding=True it is "success"). check(..., detector=...)
                             settles it by counting boxes around the spot (SAM, once): one fewer -> "success",
                             as many (the boxes only slid) -> "moved"
                   wrong_box the target is still there but a neighbour is gone
                   blocked   something is in front of the boxes (arm, gripper, hand), or the target's
                             outline wasn't still while the after frames were taken: take new frames
                   camera_moved  the whole scene changed: the camera isn't where it was
                   unknown   too few depth readings in the outline to say
    "gone", "same", "closer"  share of the outline's pixels now further away (or no reading),
                              unchanged (depth), nearer
    "unsteady"     share of the outline whose depth moved between the after frames
    "light_changed"  the background changed colour too, so colour was left out (depth only)
    "disturbed"    other boxes that changed: [{"box", "changed", "gone", "u", "v"}]; empty if none
    "by"           "depth", or "colour" when the outline has too little depth
"""
import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PICKER = Path(__file__).resolve().parent.parent / "vlm" / "molmo2-box-picking"
sys.path.insert(0, str(PICKER))

FAR_M = 0.03           # surface this much further away (or no reading): the box isn't there any more
GONE_M = 0.01          # ... or this much, and the colour changed too (a box pushed back keeps its colour)
SAME_M = 0.005         # within this: unchanged (depth noise is 1-2 mm per frame, about 1 mm after a median)
BLOCKED_M = 0.05       # this much nearer than before, outside the target: something is in the way
ERODE_PX = 3           # stay off the outline's edges, where depth jumps between boxes
GONE_SHARE = 0.7       # the outline's pixels that must be gone for "success"
SAME_SHARE = 0.8       # ... unchanged for "missed"
NEIGHBOUR_SHARE = 0.3  # a neighbour with this much changed is "disturbed"
BLOCKED_SHARE = 0.02   # image share that's much nearer: "blocked"
STEADY_M = 0.03        # a pixel whose depth moves this much between the after frames isn't still
UNSTEADY_SHARE = 0.1   # the outline's share that isn't still: "blocked" (a hand or the arm passing)
CAMERA_SHARE = 0.25    # share of the background (no box) that changed: "camera_moved"
MIN_PIXELS = 50        # fewer depth readings in the outline: fall back to colour
COLOUR_STEP = 40       # change in any colour channel (0-255) that counts as a new colour, after:
COLOUR_BLUR = 3        # a blur (pixels), so a pixel's shift of the print on a box isn't a new colour
PATTERN_WIN = 15       # the print on the boxes is compared over windows this size (pixels) ...
PATTERN_TEX = 25       # ... that have print (grey-level variance; plain card has nothing to compare) ...
PATTERN_MATCH = 0.6    # ... and match above this correlation (brightness and white balance don't count)
PATTERN_DIFF = 0.5     # this share of the outline's print not matching ...
OTHER_SHARE = 0.3      # ... and this share at the target's depth with another print: another box stands there
TEXTURED_SHARE = 0.3   # the print is used only if this share of the outline has some
MAX_SHIFT_PX = 8       # a whole-image shift up to this is undone before comparing print (camera bumped)
NEAR_PX = 40           # a box within this of the target's outline is its neighbour (for "replaced")
COUNT_W, COUNT_H = 1.5, 0.5   # "replaced" is settled by counting boxes (SAM, once) whose centre is within
                              # this many target widths beside it / heights above and below it


@dataclass(eq=False)
class Before:
    """What the check needs from the picker's run: no SAM, no frames."""
    depth: np.ndarray      # median depth of the before frames, metres, 0 = no reading
    colour: np.ndarray     # mean colour image, BGR
    target: np.ndarray     # bool HxW, the target's outline
    others: np.ndarray     # int16 HxW, i + 1 where boxes[i] is (the target left out), 0 elsewhere
    result: dict           # the picker's answer

    def save(self, path):
        np.savez_compressed(path, depth=self.depth, colour=self.colour, target=self.target, others=self.others,
                            result=json.dumps(self.result))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["depth"], d["colour"], d["target"], d["others"], json.loads(str(d["result"])))


def median_frames(frames):
    """[(colour, depth_m), ...] -> (median depth, 0 where fewer than half the frames read; mean colour)."""
    depth = np.stack([d for _, d in frames]).astype(np.float32)
    valid = depth > 0
    depth[~valid] = np.inf                                   # sorts after every reading
    depth.sort(axis=0)
    count = valid.sum(0)
    med = np.take_along_axis(depth, np.maximum(count - 1, 0)[None] // 2, 0)[0]
    med[count * 2 < len(frames)] = 0
    colour = np.mean(np.stack([c for c, _ in frames]), axis=0, dtype=np.float32)   # blurred later anyway
    return med, colour.round().astype(np.uint8)


def remember(detector, frames, result):
    """Right after detector.detect_frames(frames, ...) returned `result`: what to compare with later.
    None when there was no target (nothing to check)."""
    if result is None or detector.last is None or detector.last[2] is None:
        return None
    _, boxes, t = detector.last
    others = np.zeros(boxes[t].mask.shape, np.int16)
    for i, b in enumerate(boxes):
        if i != t:
            others[b.mask & ~boxes[t].mask] = i + 1
    depth, colour = median_frames(frames)
    return Before(depth, colour, boxes[t].mask.copy(), others, result)


def _erode(mask):
    k = np.ones((2 * ERODE_PX + 1, 2 * ERODE_PX + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def _background(before, depth):
    """Where no box was, and there is depth before and after."""
    return (before.depth > 0) & (depth > 0) & ~before.target & (before.others == 0)


def _recoloured(before, depth, colour):
    """Per pixel: a new colour. All False if the background changed colour too (the light changed,
    or the exposure): depth only, then."""
    b = cv2.GaussianBlur(before.colour, (0, 0), COLOUR_BLUR).astype(np.float32)
    a = cv2.GaussianBlur(colour, (0, 0), COLOUR_BLUR).astype(np.float32)
    ok = (b > 20) & (b < 235) & (a > 20)                     # exposure / white balance: undo the overall gain
    a *= [np.median(b[..., k][ok[..., k]] / a[..., k][ok[..., k]]) if ok[..., k].any() else 1 for k in range(3)]
    recoloured = (np.abs(a - b) > COLOUR_STEP).any(axis=2)
    background = _background(before, depth)
    if background.any() and recoloured[background].mean() > CAMERA_SHARE:
        return np.zeros_like(recoloured), True
    return recoloured, False


def _pattern(before, colour):
    """Per pixel: (printed: there is print to compare, differs: printed and not the same print as
    before; the whole-image shift undone first, (dx, dy) in pixels). The correlation is taken over
    PATTERN_WIN windows after removing each window's mean and contrast, so brightness and white
    balance don't count (they made us drop plain colour from "same"), but a different box does."""
    g0 = cv2.cvtColor(before.colour, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(g0, g1)
    if np.hypot(dx, dy) > MAX_SHIFT_PX:
        dx = dy = 0.0
    g1 = cv2.warpAffine(g1, np.float32([[1, 0, -dx], [0, 1, -dy]]), (g1.shape[1], g1.shape[0]),
                        borderMode=cv2.BORDER_REPLICATE)
    k = (PATTERN_WIN, PATTERN_WIN)
    m0, m1 = cv2.blur(g0, k), cv2.blur(g1, k)
    v0, v1 = cv2.blur(g0 * g0, k) - m0 * m0, cv2.blur(g1 * g1, k) - m1 * m1
    corr = (cv2.blur(g0 * g1, k) - m0 * m1) / np.sqrt(np.maximum(v0 * v1, 1e-6))
    printed = (v0 > PATTERN_TEX) & (v1 > PATTERN_TEX)
    return printed, printed & (corr <= PATTERN_MATCH), (float(dx), float(dy))


def _changes(before, depth, colour):
    """Per pixel, where there was a depth reading before: gone (no reading now, or further away by
    FAR_M, or by GONE_M with a new colour), same (the same depth and, where there is print, the same
    print), closer, and other (the same depth but another print: another box stands there now, e.g. a
    neighbour that slid or leaned into the gap). A pixel in none of them moved a little. Plain colour
    is not used for "same": the colour of white boxes swings with white balance."""
    had = before.depth > 0
    missing = depth == 0
    dz = depth - before.depth
    recoloured, light = _recoloured(before, depth, colour)
    printed, differs, shift = _pattern(before, colour)
    gone = had & (missing | (dz > FAR_M) | ((dz > GONE_M) & recoloured))
    at_depth = had & ~missing & (np.abs(dz) <= SAME_M)
    same, other = at_depth & ~differs, at_depth & differs
    closer = had & ~missing & (dz < -GONE_M)
    return gone, same, closer, other, printed, recoloured, light, shift


def _unsteady(frames, mask):
    """Share of the mask's pixels whose depth moved by STEADY_M or more between the frames."""
    if not mask.any():
        return 0.0
    d = np.stack([f[1][mask] for f in frames])
    valid = d > 0
    hi = np.where(valid, d, -np.inf).max(0)
    lo = np.where(valid, d, np.inf).min(0)
    return float(((valid.sum(0) > 1) & (hi - lo >= STEADY_M)).mean())


def _count_area(mask):
    ys, xs = np.nonzero(mask)
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    return xs.min() - COUNT_W * w, xs.max() + COUNT_W * w, ys.min() - COUNT_H * h, ys.max() + COUNT_H * h


def _count(masks, area):
    x0, x1, y0, y1 = area
    n = 0
    for m in masks:
        ys, xs = np.nonzero(m)
        n += bool(len(xs)) and x0 <= xs.mean() <= x1 and y0 <= ys.mean() <= y1
    return n


def recount(before, after_frames, detector, intr, up):
    """Boxes whose centre is around the target's spot: (before, after). The picker's SAM 3.1 runs once,
    on the after frames' median. Used to settle "replaced": one box fewer -> the target was taken and
    a neighbour moved into its place; as many -> nothing was taken, the boxes only slid."""
    area = _count_area(before.target)
    n_before = 1 + _count([before.others == i for i in np.unique(before.others[before.others > 0])], area)
    depth, colour = median_frames(after_frames)
    detector.detect(colour, depth, intr, up)
    boxes = detector.last[1] if detector.last else []
    return n_before, _count([b.mask for b in boxes], area)


def check(before, after_frames, holding=None, detector=None, intr=None, up=None):
    """Compare frames taken after the pick with `before` (from remember()). holding: True / False if
    the gripper can tell whether it holds something, None if not. detector, intr, up: the picker's
    PickDetector (already loaded) and its camera intrinsics and up direction, to settle "replaced" by
    counting boxes (one SAM run, only then); without them "replaced" stays. Returns the verdict dict."""
    t0 = time.perf_counter()
    depth, colour = median_frames(after_frames)
    gone, same, closer, other, printed, recoloured, light, shift = _changes(before, depth, colour)
    both = (before.depth > 0) & (depth > 0)
    dz = depth - before.depth

    # Is the view still comparable? The background is where no box was.
    background = _background(before, depth)
    background_changed = float((np.abs(dz[background]) > GONE_M).mean()) if background.any() else 0.0
    blocked = float((both & ~before.target & (dz < -BLOCKED_M)).sum() / dz.size)

    m = _erode(before.target)
    unsteady = _unsteady(after_frames, m)
    n = int((m & (before.depth > 0)).sum())
    if n >= MIN_PIXELS and n >= 0.3 * m.sum():
        by, mm = "depth", m & (before.depth > 0)
        share = lambda a: float(a[mm].mean())
        g, s, c, o = share(gone), share(same), share(closer), share(other)
    else:                                                     # too few depth readings: colour instead
        by, n = "colour", int(m.sum())
        g = float(recoloured[m].mean()) if n else 0.0
        s, c, o = 1 - g, 0.0, 0.0
    # Another box where the target was: of the outline's printed pixels, the share whose print changed.
    tm = m & printed
    printed_share = float(tm.sum() / max(m.sum(), 1))
    print_changed = float((other | gone)[tm].mean()) if printed_share >= TEXTURED_SHARE else 0.0

    # The other boxes, all at once: each one's outline, eroded, where it had depth.
    k = np.ones((2 * ERODE_PX + 1, 2 * ERODE_PX + 1), np.uint8)
    lab = before.others
    inner = (lab > 0) & (cv2.erode(lab, k) == cv2.dilate(lab, k)) & (before.depth > 0)
    ids = lab[inner]
    total = np.bincount(ids, minlength=lab.max() + 1)
    n_gone = np.bincount(ids, gone[inner], minlength=lab.max() + 1)
    n_same = np.bincount(ids, (same | other)[inner], minlength=lab.max() + 1)   # depth unchanged
    n_other = np.bincount(ids, other[inner], minlength=lab.max() + 1)          # ... but another print
    near = set(np.unique(lab[cv2.dilate(before.target.astype(np.uint8), np.ones((2 * NEAR_PX + 1,) * 2, np.uint8))
                             .astype(bool) & (lab > 0)]).tolist())
    disturbed, shifted = [], []                               # depth changed / slid at the same depth
    for i in np.nonzero(total >= MIN_PIXELS)[0]:
        changed = 1 - n_same[i] / total[i]
        if changed >= NEIGHBOUR_SHARE:
            vs, us = np.nonzero(lab == i)
            disturbed.append({"box": int(i) - 1, "changed": round(float(changed), 2),
                              "gone": round(float(n_gone[i] / total[i]), 2), "next_to_target": int(i) in near,
                              "u": round(float(us.mean()), 1), "v": round(float(vs.mean()), 1)})
        elif n_other[i] / total[i] >= PATTERN_DIFF:
            shifted.append({"box": int(i) - 1, "next_to_target": int(i) in near})

    if background_changed > CAMERA_SHARE:
        outcome = "camera_moved"
    elif blocked > BLOCKED_SHARE or unsteady > UNSTEADY_SHARE:
        outcome = "blocked"
    elif n < MIN_PIXELS:
        outcome = "unknown"
    elif g >= GONE_SHARE:
        outcome = "dropped" if holding is False else "success"
    elif o >= OTHER_SHARE and print_changed >= PATTERN_DIFF:
        # At the target's depth, but another print: another box stands where the target was. With a
        # neighbour that moved too, most likely the target was taken and that neighbour slid or leaned
        # into the gap ("replaced"). Boxes printed alike can't tell that from the target itself sliding
        # a little, so only the gripper is sure: holding -> success, empty -> dropped. If it was not
        # taken, it is still there and the picker simply chooses it again.
        neighbour = any(d["next_to_target"] for d in disturbed + shifted)
        if holding is True:
            outcome = "success"
        elif holding is False:
            outcome = "dropped" if neighbour else "moved"
        else:
            outcome = "replaced" if neighbour else "moved"
    elif s >= SAME_SHARE:
        outcome = "wrong_box" if any(d["gone"] >= GONE_SHARE for d in disturbed) else "missed"
    else:
        outcome = "moved"
    counted = None
    if outcome == "replaced" and detector is not None:
        counted = [int(v) for v in recount(before, after_frames, detector, intr, up)]
        outcome = "success" if counted[1] < counted[0] else "moved"
    return {"outcome": outcome, "gone": round(g, 3), "same": round(s, 3), "closer": round(c, 3),
            "other_box": round(o, 3), "print_changed": round(print_changed, 3),
            "boxes_around": counted, "pixels": n, "by": by, "disturbed": disturbed, "shifted": shifted,
            "holding": holding,
            "light_changed": light,
            "shift_px": [round(shift[0], 1), round(shift[1], 1)],
            "background_changed": round(background_changed, 3), "blocked": round(blocked, 4),
            "unsteady": round(unsteady, 3),
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


def same_target(previous, new):
    """The picker's next answer is the box it was just sent to (same approach, within 2 cm): the
    last pick didn't take it. A free second check, as the picker runs before every pick anyway."""
    if previous is None or new is None or previous["approach"] != new["approach"]:
        return False
    xyz = lambda r: np.array([r["x_m"], r["y_m"], r["z_m"]])
    from pick_frame import SAME_TARGET_M
    return bool(np.linalg.norm(xyz(previous) - xyz(new)) <= SAME_TARGET_M)


def draw(before, after_frames, verdict):
    """The last after frame: the target's outline red where it's gone, green where unchanged, blue
    where nearer, purple where another box now stands; disturbed neighbours outlined in orange."""
    depth, colour = median_frames(after_frames)
    gone, same, closer, other = _changes(before, depth, colour)[:4]
    img = after_frames[-1][0].copy()
    tint = img.copy()
    tint[before.target & gone] = (0, 0, 255)
    tint[before.target & same] = (0, 200, 0)
    tint[before.target & closer] = (255, 0, 0)
    tint[before.target & other] = (255, 0, 200)
    img = cv2.addWeighted(tint, 0.5, img, 0.5, 0)
    outline = lambda mask, col, w: cv2.drawContours(
        img, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, col, w)
    outline(before.target, (0, 255, 255), 2)
    for d in verdict["disturbed"]:
        outline(before.others == d["box"] + 1, (0, 140, 255), 2)
    msg = f"{verdict['outcome'].upper()}  gone {verdict['gone']:.0%}  same {verdict['same']:.0%}  " \
          f"({verdict['by']}, {len(verdict['disturbed'])} disturbed)"
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


RESULT = {"success": ("PICK SUCCESS", "the box is gone from its place", (0, 200, 0)),
          "dropped": ("PICK FAILED: DROPPED", "the box left its place but the gripper is empty", (0, 0, 255)),
          "replaced": ("PICK SUCCESS (likely)", "another box slid or leaned into its place", (0, 160, 0)),
          "missed": ("PICK FAILED: MISSED", "the box is still there", (0, 0, 255)),
          "moved": ("PICK FAILED: BOX MOVED", "pushed, tipped or only pulled part way", (0, 0, 255)),
          "wrong_box": ("PICK FAILED: WRONG BOX", "the target is still there, a neighbour is gone", (0, 0, 255)),
          "blocked": ("CAN'T TELL: VIEW BLOCKED", "arm or hand in view, or not still: take new frames", (0, 165, 255)),
          "camera_moved": ("CAN'T TELL: CAMERA MOVED", "the scene doesn't line up with before", (0, 165, 255)),
          "unknown": ("CAN'T TELL", "too little depth on the target", (0, 165, 255))}


def big_result(overlay, verdict):
    """The check's overlay with the verdict in large letters at the bottom."""
    title, why, colour = RESULT[verdict["outcome"]]
    if verdict["outcome"] in ("success", "dropped") and verdict.get("other_box", 0) >= OTHER_SHARE:
        why += " (a neighbour slid or leaned into its place)"
    elif verdict["disturbed"] and verdict["outcome"] == "success":
        why += f" ({len(verdict['disturbed'])} other box(es) disturbed)"
    img = overlay.copy()
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 110), (img.shape[1], h), colour, -1)
    cv2.putText(img, title, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    cv2.putText(img, why, (22, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def load_frames(pattern):
    """Saved captures matching a glob (rgbd.py prefixes) -> ([(colour, depth_m), ...], Intrinsics, accel)."""
    from rgbd import load
    prefixes = sorted({p[:-len("_color.png")] for p in glob.glob(pattern + "_color.png")})
    if not prefixes:
        raise SystemExit(f"no captures match {pattern}")
    loaded = [load(p) for p in prefixes]
    return [(c, d) for c, d, _, _ in loaded], loaded[0][2], loaded[0][3]


def run_picker(frames, intr, accel, pitch):
    """SAM 3.1, as the robot would run it before the pick -> Before (None if no target)."""
    from faces import up_from_accel, up_from_pitch
    from pick_frame import PickDetector
    up = up_from_accel(accel) if accel and pitch is None else up_from_pitch(pitch or 0)
    detector = PickDetector()
    result = detector.detect_frames(frames, intr, up)
    print("picker:", json.dumps(result), flush=True)
    return remember(detector, frames, result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", help="glob of the before captures (runs the picker), or a saved .npz")
    ap.add_argument("--after", help="glob of the after captures")
    ap.add_argument("--live", action="store_true", help="D455: take the before frames, wait for Enter, take the after")
    ap.add_argument("--frames", type=int, default=5, help="--live: frames before (after: 3)")
    ap.add_argument("--holding", choices=["yes", "no"], help="what the gripper says, if it can tell")
    ap.add_argument("--save-before", metavar="X.npz", help="keep what remember() kept, to check again without SAM")
    ap.add_argument("--pitch", type=float, help="camera tilt down in degrees (default: accelerometer, else 0)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"), help="folder for the overlay")
    args = ap.parse_args()
    if not args.live and not (args.before and args.after):
        ap.error("give --before and --after, or --live")

    if args.live:
        from rgbd import Camera
        cam = Camera()
        try:
            before = run_picker([cam.read() for _ in range(args.frames)], cam.intr, cam.accel, args.pitch)
            if before is None:
                raise SystemExit("no target: nothing to check")
            input("pick the box (or not), get the arm out of view, then press Enter: ")
            after = [cam.read() for _ in range(3)]
        finally:
            cam.close()
        name = time.strftime("live_%Y%m%d_%H%M%S")
    else:
        if args.before.endswith(".npz"):
            before = Before.load(args.before)
        else:
            before = run_picker(*load_frames(args.before), args.pitch)
            if before is None:
                raise SystemExit("no target in the before frames: nothing to check")
        after, _, _ = load_frames(args.after)
        name = Path(sorted(glob.glob(args.after + "_color.png"))[0]).name[:-len("_color.png")]
    if args.save_before:
        before.save(args.save_before)

    verdict = check(before, after, {"yes": True, "no": False}.get(args.holding))
    print(json.dumps(verdict))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{name}_check.jpg"
    cv2.imwrite(str(path), draw(before, after, verdict))
    print(f"overlay: {path}")


if __name__ == "__main__":
    main()
