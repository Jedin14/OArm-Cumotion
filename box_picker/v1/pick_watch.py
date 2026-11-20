"""Did the arm pick the target box, or miss it? Checked from depth, on every camera frame.

When a target is chosen, its pick face is remembered: the plane it lies in and the
pixels it covers (those with depth in every frame it was found in, and on that plane).
In each later frame the depth at those pixels is compared with that plane, along the
face's normal:
  - in front of it (towards the camera): something is between the camera and the box,
    the arm or the suction cup;
  - on it: the box is still there;
  - well behind it: the camera now sees past where the box was, so the box is gone.
    "Well behind" is most of the box's thinnest side when its size is known (gone_depth():
    whatever a taken box uncovers is at least that far back), else 2 cm. Only a little
    behind it: the box was pushed back or tipped, not taken.
Anything between the camera and a visible face is on the camera's side of its plane,
so this works the same for a front face and a top face.

PickWatcher follows one pick: waiting for the arm -> arm in view -> arm gone again ->
missed (box still there), or moved (neither, for a second: the box was pushed, knocked
or dragged), or the box left its spot. Leaving its spot is not yet a pick: the box can
still be in the gripper above the carton and fall back in. So from then on the whole
view is compared with the depth from before the pick (where the box stood: with the
farthest depth seen since), until it settles:
  - change touching the picture's edge, in 2 frames in a row: the arm (or a hand) is
    still in view, wait;
  - nothing new in view: picked;
  - a new surface, standing still, not touching the edge: dropped, the box fell back
    into the carton (anywhere, also onto its own spot). A failed pick: try again.
  - after SETTLE_S: decide on what is seen, but not while a surface as big as a box stays
    at the picture's edge (an arm or a hand, maybe still holding the box).
A box that disappears without the arm being seen (taken by hand, or between two
frames) goes the same way.

The depth before the pick is the median of all the frames the target was found in (refs),
and a pixel whose depth comes and goes or varies between them is not compared: depth
flickers at the borders of no-depth areas (the D455's left strip, dark corners), and
compared with one frame, a still scene on the robot showed "arm in view" in a third of the
frames (18 Nov 2026), so "picked" waited for the time limit.

The robot can say when its arm is out of the view (update(..., arm_out=True): it asks for
the result after going back to pre_pick). From then on the watcher decides on what it sees,
within a few frames: still there -> missed (also when the arm was never seen); pushed or
tipped -> moved; something in front of the face that doesn't reach in from the picture's
edge -> moved; gone -> the view as above, where lost depth or a small change at the edge is
no longer taken for the arm; most of the view changed -> camera_moved (can't tell). It still
waits while something reaches in from the edge (the arm after all) or a surface as big as a
box stays at the edge: then the caller says it can't tell, and why (status, detail).

Every frame is kept in `trace` (shares on the face, the state, the change at the picture's
edge, ...) and every change of status in `timeline`, to see afterwards why a pick took
long or went the way it did.

PickTracker hands out one target at a time, holds it while the pick is watched, and
reports each step as an event for the next step (the robot, a log). A target is only
handed out once it has stayed put for half a second, so a box still being carried
away by the arm (or a hand) never becomes the next target. Events:
    {"event": "target", "pick": 3, "attempt": 1, "target": {...}}
    {"event": "pick_result", "pick": 3, "attempt": 1, "result": "picked", "seconds": 4.2, "shift_mm": -38, "target": {...}}
result: picked, or a failed pick: missed, moved or dropped ("success": result == "picked").
After a failed pick the next target is usually the same box again (where it was, or for
dropped, where it fell); its attempt number goes up.
shift_mm: where the surface now seen on the face's pixels is, mm from the face (minus =
further away). About one box thickness back: a box stands right behind the one picked,
or the box was pushed back that far and not picked; depth can't tell those apart, the
gripper's vacuum sensor can. Much further back: open space behind it.

draw_pick() shows it on a picture (live_watch.py, replay_picks.py, pick_loop.py --check v2): the target being
watched, the status, and for BANNER_S after each result a banner: green PICKED, red
MISSED, orange MOVED, purple DROPPED, with the old target's spot outlined in the same
colour (and for DROPPED, where the box fell).

No torch here, so it can be tested without a GPU (test_pick_watch.py).
"""
import collections

import cv2
import numpy as np

PLANE_TOL_M = 0.010    # depth within this of the face's plane is the face itself
BLOCKED_FRAC = 0.05    # this share of the face with something in front of it = the arm is there
                       # (the suction cup on a top face covers only about a tenth of it)
SHADOW_FRAC = 0.30     # ... or this share with no depth (the arm's depth shadow)
GONE_FRAC = 0.60       # this share well behind the plane = the box is gone
THERE_FRAC = 0.60      # this share on the plane = the box is still there
GONE_M = 0.020         # "well behind" when the box's size is not known (1 cm before 19 Nov 2026: a box
                       # pushed back 1-2 cm read as gone, and then as picked)
GONE_OF_THICK = 0.6    # with the box's size: this share of its thinnest side (on the robot, 18 Nov: what
                       # taken 54 mm deep boxes uncovered was 49-70 mm behind their faces; a miss, 0-2 mm)
GONE_MIN_M, GONE_MAX_M = 0.015, 0.08
STEADY_FRAMES = 5      # a state counts once seen in this many frames in a row (1/3 s at 15 fps)
NOMINAL_FPS = 30.0     # frame times when update() is given none: frame n at n / 30 s
MOVED_S = 1.0          # after the arm, this long neither there nor gone = the box was moved
MOVED_FRAMES = int(MOVED_S * NOMINAL_FPS)   # (the same in frames at 30 fps)
NEAR_M = 0.020         # once the box left its spot: depth this much nearer than before the pick = something new there
VIEW_STEP = 2          # the whole-view check looks at every 2nd pixel each way
MAJORITY_PX = 5        # ... by majority over windows this wide (in those pixels): drops speckle and flickering edges
REF_SPREAD_M = 0.010   # the depth before the pick: a pixel whose frames differ by more than this is not compared
HOLE_PX = 9            # ... (holes all over) a no-depth area this big (view pixels) has a flickering border
SPOT_GROW_PX = 10      # the box's spot: its outline (or pick face), grown by this many (view) pixels
DROP_AREA_FRAC = 0.2   # a new surface at least this share of the pick face's area ...
MIN_DROP_MM2 = 800     # ... and at least this big = a box (one fallen on its narrow side still shows more) ...
MOVED_AWAY_FRAC = 0.5  # ... unless this share of its area, within a box's diagonal of it and off the box's spot, is now
                       # further away: then boxes moved (a neighbour tipped into the gap, boxes moved by hand), none was added
EDGE_MIN_PX = 100      # change this big (view pixels) touching the picture's edge = the arm or a hand in view ...
EDGE_FRAMES = 2        # ... once seen in this many frames in a row (one noisy frame is not the arm)
EDGE_BAND_PX = 64      # no depth this close to the border (view pixels) is part of the edge (the D455's left strip
                       # is baseline x fx / z wide: about 120 px at 0.5 m, 60 view pixels)
DROP_FRAMES = 10       # a new surface standing still this many frames, the arm gone = the box fell back in
CLEAR_FRAMES = 10      # this many frames with nothing new in view, the arm gone = picked
SETTLE_S = 10.0        # after the box left its spot, stop waiting for the view to clear after this long (s) ...
SETTLE_FRAMES = int(SETTLE_S * NOMINAL_FPS)  # (the same in frames at 30 fps)
                       # ... but not while a surface as big as a box stays at the picture's edge (18 Nov 2026:
                       # a hand holding the box in view was read as picked when the time ran out)
CAMERA_MOVED_FRAC = 0.25  # (arm out) this share of the view nearer or further than before = the camera moved
TRACE_LEN = 3000       # frames kept in `trace` (over 3 minutes at 15 fps)
BANNER_S = 2.0         # seconds a result stays on the picture
RESULT_COLORS = {"picked": (0, 170, 0), "missed": (0, 0, 220), "moved": (0, 140, 255),
                 "dropped": (200, 0, 130), "camera_moved": (0, 165, 255)}   # BGR
SAME_BOX_M = 0.03      # a new target this close to a missed one is another try at the same box
STILL_M = 0.01         # a target is handed out once found this close to where it was found ...
STILL_S = 0.5          # ... at least this long before (a box still in the gripper or a hand moves)


def gone_depth(box_mm):
    """How far behind its face's plane the camera must see for the box to count as gone, m:
    GONE_OF_THICK of the box's thinnest side (box_mm: its 3 sides, mm, any order), else GONE_M.
    The thinnest side, not the one behind the face: a face measured too small could match the
    wrong sides, and a threshold deeper than what is behind the box would never see it gone."""
    try:
        sides = [float(v) for v in box_mm]
    except (TypeError, ValueError):
        return GONE_M
    if len(sides) != 3 or min(sides) <= 0:
        return GONE_M
    return float(np.clip(GONE_OF_THICK * min(sides) / 1000, GONE_MIN_M, GONE_MAX_M))


def plane_offsets(pixels, z, normal, centroid, intr):
    """Distance of each pixel's point (depth z, m) from the plane, m (+ towards the camera)."""
    u, v = pixels[:, 0], pixels[:, 1]
    pts = np.stack([(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z], axis=1)
    return (pts - centroid) @ normal


def shares_of(z, s, gone_m=GONE_M):
    """face_shares() from the pixels' depths z and their distances s from the plane."""
    has = z > 0
    n = max(len(z), 1)
    return {"front": np.sum(has & (s > PLANE_TOL_M)) / n, "on": np.sum(has & (np.abs(s) <= PLANE_TOL_M)) / n,
            "behind": np.sum(has & (s < -PLANE_TOL_M)) / n, "far": np.sum(has & (s < -gone_m)) / n,
            "none": np.sum(~has) / n, "shift": float(np.median(s[has])) if has.any() else None}


def face_shares(pixels, normal, centroid, depth_m, intr, gone_m=GONE_M):
    """Shares of the pixels whose depth is (in front of, on, behind) the plane, behind it by more
    than gone_m ("far"), and with no depth; shift: their median distance from the plane, metres
    (+ towards the camera, None without depth)."""
    z = depth_m[pixels[:, 1], pixels[:, 0]]
    return shares_of(z, plane_offsets(pixels, z, normal, centroid, intr), gone_m)


def frame_state(shares):
    """gone / blocked / there / unclear, from face_shares()."""
    if shares.get("far", shares["behind"]) >= GONE_FRAC:
        return "gone"
    if shares["front"] >= BLOCKED_FRAC or shares["none"] >= SHADOW_FRAC:
        return "blocked"
    if shares["on"] >= THERE_FRAC:
        return "there"
    return "unclear"


def out_state(shares):
    """frame_state() once the robot says its arm is out of the view: no depth on the face is then
    not the arm's shadow. With nothing in front and a fifth of the face with depth, gone or there
    by the share of the pixels that have depth."""
    seen = 1.0 - shares["none"]
    if shares["front"] < BLOCKED_FRAC and seen >= 0.2:
        if shares["far"] >= GONE_FRAC * seen:
            return "gone"
        if shares["on"] >= THERE_FRAC * seen:
            return "there"
    return frame_state(shares)


def majority(mask, valid, k=MAJORITY_PX):
    """Pixels where at least half of the valid pixels in the k x k window around them are in
    mask, and at least a third of the window is valid. Holes in the depth don't break up a
    surface, and a thin flickering edge or stray pixels go."""
    box = lambda m: cv2.boxFilter(m.astype(np.float32), -1, (k, k), borderType=cv2.BORDER_REPLICATE)
    share_valid = box(valid)
    return (box(mask) >= 0.5 * share_valid) & (share_valid >= 1 / 3)


def stable_reference(depths, step=VIEW_STEP):
    """The depth before the pick for the whole-view check (every step-th pixel), from one or more
    frames of the same view: their median where every frame has depth and they agree within
    REF_SPREAD_M, else 0 (not compared). Then only pixels whose whole MAJORITY_PX window has depth:
    at the borders of no-depth areas depth comes and goes from frame to frame (still robot frames,
    18 Nov 2026: "arm in view" in 32% of them against one frame as it is, 0.5% against this from 3).
    If that would leave out more than half (holes spread all over: a surface the camera reads
    poorly), only the borders of no-depth areas of HOLE_PX or more are left out."""
    stack = np.stack([d[::step, ::step] for d in depths]).astype(np.float32)
    has = np.all(stack > 0, axis=0)
    if len(stack) > 1:
        has &= (stack.max(axis=0) - stack.min(axis=0)) <= REF_SPREAD_M
    kernel = np.ones((MAJORITY_PX,) * 2, np.uint8)
    keep = cv2.erode(has.astype(np.uint8), kernel) > 0
    if keep.sum() < 0.5 * has.sum():
        n, lab, stats, _ = cv2.connectedComponentsWithStats((~has).astype(np.uint8), connectivity=4)
        big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= HOLE_PX)
        keep = has & ~(cv2.dilate(np.isin(lab, big[big > 0]).astype(np.uint8), kernel) > 0)
    return np.where(keep, np.median(stack, axis=0), 0).astype(np.float32)


def edge_zone(ref):
    """The picture's edge as depth sees it: the border, and the areas with no depth that reach
    it (the D455 has a strip with no depth down its left side), but only within EDGE_BAND_PX
    of the border, grown a little. An arm reaching in crosses it."""
    _, lab = cv2.connectedComponents((ref <= 0).astype(np.uint8), connectivity=8)
    rim = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    zone = np.isin(lab, rim[rim > 0])
    band = np.ones(ref.shape, bool)
    band[EDGE_BAND_PX:-EDGE_BAND_PX, EDGE_BAND_PX:-EDGE_BAND_PX] = False
    zone &= band
    zone[0], zone[-1], zone[:, 0], zone[:, -1] = True, True, True, True
    return cv2.dilate(zone.astype(np.uint8), np.ones((MAJORITY_PX,) * 2, np.uint8)) > 0


def view_change(ref, depth, intr, spot, min_m2, zone, reach_px, step=VIEW_STEP, info=None):
    """Compare a view (every step-th pixel) with the reference depth from before the pick.
    -> (edge, new): whether a change of at least EDGE_MIN_PX touches the picture's edge
    (`zone`, from edge_zone(): the arm or a hand); and where something was added (bool), else None: a connected
    change not touching the edge, with at least min_m2 square metres of it nearer than ref
    by NEAR_M, and not mostly balanced by surface within reach_px of it that is now further
    away (outside `spot`, where the box stood): that is boxes moved, not one added.
    Lost depth counts as change (the arm can be too close for depth), not as area.
    info (a dict), if given, gets: edge_px and edge_box (x, y, w, h, view pixels) of the biggest
    change touching the edge; edge_near_mm2, the most of one such change that is a surface nearer
    than before (not lost depth); edge_surface, where such changes have at least min_m2 of it
    (bool, else None: something as big as a box at the edge); changed_frac, the share of the
    compared view (off the spot) nearer or further than before (most of it: the camera moved)."""
    has = ref > 0
    both = has & (depth > 0)
    near = majority(both & (depth < ref - NEAR_M), both)
    far = majority(both & (depth > ref + NEAR_M), both) & ~spot
    changed = near | majority(has & (depth <= 0), has)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
    px = stats[:, cv2.CC_STAT_AREA]
    at_edge = np.bincount(labels[zone], minlength=n) > 0
    at_edge[0] = False
    z = np.where(has, ref, depth)
    pix_m2 = (step * z / intr.fx) * (step * z / intr.fy)           # what one view pixel covers there
    near_m2 = np.bincount(labels[near], weights=pix_m2[near], minlength=n)
    edge = bool(np.any(at_edge & (px >= EDGE_MIN_PX)))
    new = np.zeros(changed.shape, bool)
    grow = np.ones((2 * reach_px + 1,) * 2, np.uint8)
    for k in np.flatnonzero(~at_edge & (near_m2 >= min_m2)):
        blob = labels == k
        around = cv2.dilate(blob.astype(np.uint8), grow) > 0
        if np.sum(pix_m2[far & around]) < MOVED_AWAY_FRAC * near_m2[k]:
            new |= blob
    if info is not None:
        if at_edge.any():
            k = int(np.argmax(np.where(at_edge, px, -1)))
            info["edge_px"], info["edge_box"] = int(px[k]), [int(v) for v in stats[k, :4]]
            info["edge_near_mm2"] = round(float(near_m2[at_edge].max()) * 1e6)
        else:
            info["edge_px"], info["edge_box"], info["edge_near_mm2"] = 0, None, 0
        big = np.flatnonzero(at_edge & (near_m2 >= min_m2))
        info["edge_surface"] = np.isin(labels, big) if big.size else None
        off = both & ~spot
        info["changed_frac"] = float(np.sum((near | far) & off)) / max(int(off.sum()), 1)
    return edge, (new if new.any() else None)


class PickWatcher:
    """Follows one pick, from the target's face in the frame(s) it was found in."""

    def __init__(self, face, depth_m, mask=None, refs=(), box_mm=None):
        """face: the target's pick face; depth_m: the depth it was found in; mask: the box's
        whole outline, if known (where it stood, also its top and sides). refs: more depth frames
        of the same view from before the pick (the other frames the target was found in): the
        reference becomes their median, and pixels whose depth comes and goes or varies between
        them are not compared. box_mm: the box's sides (mm), if known: how far back the face must
        read to be gone (gone_depth())."""
        depths = [depth_m] + [d for d in (refs if refs is not None else ())
                              if d is not None and d is not depth_m and d.shape == depth_m.shape]
        px = face.pixels
        zs = np.stack([d[px[:, 1], px[:, 0]] for d in depths]).astype(np.float64)
        has = np.all(zs > 0, axis=0)                              # filled-in pixels have no depth to compare
        self.pixels = px[has]
        self.ref_z = np.median(zs[:, has], axis=0)                # the face's depth before the pick
        self.normal, self.centroid = face.normal, face.centroid
        self.corners_px = face.corners_px                         # for drawing
        self.gone_m = gone_depth(box_mm)
        self.base = None               # the face's shares before the pick, from ref_z (on the first frame)
        self.off_plane = 0             # face pixels left out: not on the face's plane before the pick
        self.arm_seen = False
        self.state, self.count = None, 0
        self.shift = None              # median distance of what is seen now from the face's plane, metres
        self.shares = None
        # For the whole-view check once the box has left its spot: the depth before the pick,
        # the box's spot in it, and how big a new surface must be to be a box.
        self.ref = stable_reference(depths)
        spot = np.zeros(self.ref.shape, np.uint8)
        spot[np.minimum(px[:, 1] // VIEW_STEP, spot.shape[0] - 1),
             np.minimum(px[:, 0] // VIEW_STEP, spot.shape[1] - 1)] = 1
        if mask is not None:
            spot |= mask[::VIEW_STEP, ::VIEW_STEP].astype(np.uint8)
        self.spot = cv2.dilate(spot, np.ones((2 * SPOT_GROW_PX + 1,) * 2, np.uint8)) > 0
        self.zone = edge_zone(self.ref)
        c = np.asarray(face.corners_px, float)
        self.reach_px = max(1, int(np.linalg.norm(c[0] - c[2]) / VIEW_STEP))   # the face's diagonal, view pixels
        self.min_drop_m2 = max(MIN_DROP_MM2, DROP_AREA_FRAC * face.size_mm[0] * face.size_mm[1]) / 1e6
        self.left = False              # the box has left its spot; picked once it has left the view too
        self.left_at = None            # when (s)
        self.clear, self.still = 0, 0
        self.edge_run = 0              # frames in a row with a change at the picture's edge
        self.view_edge = False         # the arm (or a hand) is in view
        self.drop = None               # where a new surface stands (bool, view pixels), while one is seen
        self.info = {}                 # the last whole-view comparison (view_change()'s info)
        self.arm_out = False           # the robot says its arm is out of the view
        self.view_moved = 0            # frames in a row with most of the view changed (arm out)
        self.n, self.t0, self.t, self.since = 0, None, 0.0, 0.0   # frames; time of the first, s; s since; state began
        self.provisional = []          # (s, result) decided but reopened: the robot was not done yet
        self.trace = collections.deque(maxlen=TRACE_LEN)
        self.timeline = collections.deque(maxlen=TRACE_LEN)   # (s, what): every change of status
        self._said = None

    # -- each frame ----------------------------------------------------------------------

    def update(self, depth_m, intr, now=None, arm_out=False):
        """Feed a newer frame: "picked", "missed", "moved" or "dropped" once decided (with arm_out
        also "camera_moved"), else None. now: the frame's time, s (else frame n at n / NOMINAL_FPS).
        arm_out: the robot says its arm is out of the view (it asks for the result): decide on what
        is seen now (see the module's notes)."""
        self.n += 1
        t = float(now) if now is not None else self.n / NOMINAL_FPS
        if self.t0 is None:
            self.t0 = t
        self.t = t - self.t0
        if self.base is None:
            self._baseline(intr)
        if arm_out and not self.arm_out:
            self.arm_out = True
            self.state, self.clear, self.still, self.view_moved = None, 0, 0, 0   # decide on frames from now on
            self.note("the robot says its arm is out of view: deciding on what is seen now")
        shares = face_shares(self.pixels, self.normal, self.centroid, depth_m, intr, self.gone_m)
        state = out_state(shares) if self.arm_out else frame_state(shares)
        self.shares, self.shift = shares, shares["shift"]
        if state != self.state:
            self.state, self.count, self.since = state, 0, self.t
        self.count += 1
        result = self._step(depth_m, intr)
        self._record(result)
        return result

    def _baseline(self, intr):
        """The face before the pick: its shares, and only its pixels on its plane from now on (a
        lip, a flap or a neighbour's edge in the outline would otherwise read as the arm for ever)."""
        s = plane_offsets(self.pixels, self.ref_z, self.normal, self.centroid, intr)
        self.base = shares_of(self.ref_z, s, self.gone_m)
        on = np.abs(s) <= PLANE_TOL_M
        if on.sum() >= max(20, 0.5 * len(on)):          # (else the plane doesn't fit its pixels: keep them all)
            self.off_plane = int(np.sum(~on))
            self.pixels = self.pixels[on]

    def _step(self, depth_m, intr):
        if self.left:
            return self._settle(depth_m, intr)
        if self.arm_out:
            return self._decide_out(depth_m, intr)
        if self.count < STEADY_FRAMES:
            return None
        if self.state == "gone":
            self._leave()
            return self._settle(depth_m, intr)
        if self.state == "blocked":
            self.arm_seen = True
        elif self.arm_seen and self.state == "there":
            return "missed"
        elif self.arm_seen and self.t - self.since >= MOVED_S - 1e-6:
            return "moved"
        return None

    def _leave(self):
        self.left, self.left_at = True, self.t
        self.clear = self.still = 0

    def _view(self, depth_m, intr):
        """The whole view against before the pick -> (edge seen EDGE_FRAMES in a row, drop)."""
        d = depth_m[::VIEW_STEP, ::VIEW_STEP]
        if self.left:      # where the box stood, compare with the farthest seen since: what was behind it
            self.ref[self.spot] = np.maximum(self.ref[self.spot], d[self.spot])
        self.info = {}
        edge, drop = view_change(self.ref, d, intr, self.spot, self.min_drop_m2, self.zone, self.reach_px,
                                 info=self.info)
        self.edge_run = self.edge_run + 1 if edge else 0
        return self.edge_run >= EDGE_FRAMES, drop

    def _camera_moved(self):
        """(arm out) Most of the view nearer or further than before, STEADY_FRAMES in a row."""
        self.view_moved = self.view_moved + 1 if self.info.get("changed_frac", 0) >= CAMERA_MOVED_FRAC else 0
        return self.view_moved >= STEADY_FRAMES

    def _decide_out(self, depth_m, intr):
        """The robot says its arm is out, the box has not left its spot (yet)."""
        edge, _ = self._view(depth_m, intr)
        self.view_edge = edge
        if self._camera_moved():
            return "camera_moved"
        if self.count < STEADY_FRAMES or self.view_moved:
            return None
        if self.state == "gone":
            self._leave()
            return None                # the whole view from the next frame on
        if self.state == "there":
            return "missed"
        if self.state == "unclear":
            return "moved"             # pushed back, tipped or shifted: still there, not taken
        # blocked: something in front of the face. Reaching in from the picture's edge: the arm
        # or a hand after all, wait. Else the box tipped or was pulled towards the camera.
        return None if edge else "moved"

    def _settle(self, depth_m, intr):
        """After the box left its spot: picked once the view is clear, dropped if a new
        surface stands still in it (the box fell back in), else None (still waiting)."""
        edge, drop = self._view(depth_m, intr)
        surface = self.info.get("edge_surface")
        if self.arm_out:
            if self._camera_moved():
                return "camera_moved"
            if surface is None:        # lost depth or a small change at the edge: not the arm, it is out
                edge = False
        self.view_edge = edge
        if edge:
            self.clear = self.still = 0
        elif drop is not None:
            prev = self.drop
            same = prev is not None and np.sum(prev & drop) >= 0.5 * np.sum(prev | drop)
            self.still = self.still + 1 if same else 1
            self.clear = 0
            if self.still >= DROP_FRAMES:
                self.drop = drop
                return "dropped"
        else:
            self.clear += 1
            self.still = 0
            if self.clear >= CLEAR_FRAMES:
                return "picked"
        self.drop = drop
        if not self.arm_out and self.t - self.left_at >= SETTLE_S - 1e-6:
            # waited long enough: decide on what is seen -- unless something as big as a box stays at
            # the picture's edge (an arm or a hand, maybe with the box): then keep waiting
            if self.drop is not None:
                return "dropped"
            if surface is None:
                return "picked"
        return None

    def reopen(self, result):
        """A result that is not final (the robot may not be done: it can press again): keep watching.
        The arm must be seen again before another miss."""
        self.provisional.append((round(self.t, 1), result))
        self.note(f"{result} for now; the robot isn't done yet: watching on")
        self.arm_seen = False
        self.state, self.count = None, 0

    # -- what it saw -----------------------------------------------------------------------

    def note(self, what):
        self.timeline.append((round(self.t, 2), what))

    def _record(self, result):
        sh, info = self.shares, self.info
        r3 = lambda x: round(float(x), 3)
        e = {"n": self.n, "t": round(self.t, 3), "state": self.state, "front": r3(sh["front"]), "on": r3(sh["on"]),
             "behind": r3(sh["behind"]), "far": r3(sh["far"]), "none": r3(sh["none"]),
             "shift_mm": None if sh["shift"] is None else round(sh["shift"] * 1000, 1),
             "arm_seen": self.arm_seen, "left": self.left, "arm_out": self.arm_out, "edge": self.view_edge}
        if info:
            e.update(edge_px=info["edge_px"], edge_near_mm2=info["edge_near_mm2"],
                     changed=round(info["changed_frac"], 3), drop_px=0 if self.drop is None else int(self.drop.sum()),
                     edge_box=None if info["edge_box"] is None else [v * VIEW_STEP for v in info["edge_box"]])
        if result:
            e["result"] = result
        self.trace.append(e)
        said = self.status
        if said != self._said:
            self.note(said)
            self._said = said
        if result:
            self.note(f"decided: {result}")

    @property
    def fps(self):
        """Frames a second fed so far."""
        return round((self.n - 1) / self.t, 1) if self.t > 0 else None

    @property
    def status(self):
        if self.left:
            if self.view_edge:
                return ("box left its spot; something as big as a box stays at the picture's edge (the arm?)"
                        if self.arm_out else "box left its spot, arm still in view")
            return "box left its spot, new surface in view (fell back?)" if self.drop is not None \
                else "box left its spot, checking the view"
        if self.arm_out:
            if self.view_moved:
                return "most of the view changed: the camera moved?"
            if self.state == "blocked" and self.count >= STEADY_FRAMES:
                return ("something in front of the box, reaching in from the picture's edge (the arm?)"
                        if self.view_edge else "something in front of the box")
            return "arm out (the robot says): checking the box"
        if self.state == "blocked" and self.count >= STEADY_FRAMES:
            return "arm in view"
        return "arm gone, checking" if self.arm_seen else "waiting for the arm"

    @property
    def detail(self):
        """The status, with what the camera sees: the face's shares, where the face reads,
        the change at the picture's edge (picture pixels)."""
        parts = [self.status]
        sh = self.shares
        if sh is not None:
            parts.append(f"face: {sh['front']:.0%} in front, {sh['on']:.0%} on it, {sh['far']:.0%} gone "
                         f"(over {self.gone_m * 1000:.0f} mm behind), {sh['none']:.0%} no depth")
        if self.shift is not None:
            parts.append(f"the face reads {self.shift * 1000:+.0f} mm")
        b = self.info.get("edge_box")
        if b is not None and self.info.get("edge_px", 0) >= EDGE_MIN_PX:
            x, y, w, h = (v * VIEW_STEP for v in b)
            parts.append(f"change at the picture's edge: x {x}-{x + w}, y {y}-{y + h} "
                         f"({self.info['edge_near_mm2']} mm2 of it a nearer surface)")
        return "; ".join(parts)


def fell_there(drop, face):
    """Is face's centre where a dropped box fell (drop: bool, view pixels, or None)?"""
    if drop is None:
        return False
    u, v = (int(c) // VIEW_STEP for c in face.centroid_px)
    grown = cv2.dilate(drop.astype(np.uint8), np.ones((2 * SPOT_GROW_PX + 1,) * 2, np.uint8))
    return 0 <= v < grown.shape[0] and 0 <= u < grown.shape[1] and bool(grown[v, u])


class PickTracker:
    """One target at a time: offer() each new pipeline result, update() each camera frame."""

    def __init__(self):
        self.watch = None
        self.pick, self.attempt = 0, 0
        self.target, self.started = None, 0.0
        self.last = None             # (result, face centre, where a dropped box fell (view pixels) or None)
                                     # of the last decided pick
        self.last_event, self.last_corners = None, None   # its pick_result event and face rectangle, for drawing
        self.decided_at = -np.inf
        self.seen = None             # (face centre, frame time) where the next target was first found

    def offer(self, target, face, depth_m, frame_time, mask=None, refs=(), box_mm=None):
        """A pipeline result (target dict, its pick face, the depth it came from, when that frame
        was taken, the box's outline if known; more depth frames from before the pick and the
        box's size, if known: see PickWatcher). It becomes the target only when no pick is being watched, its frame is
        newer than the last decision (an older frame still shows the box just picked), and
        the same box was found in the same place at least STILL_S before."""
        if self.watch or target is None or frame_time <= self.decided_at:
            return None
        if self.seen is None or np.linalg.norm(face.centroid - self.seen[0]) > STILL_M:
            self.seen = face.centroid, frame_time
        if frame_time - self.seen[1] < STILL_S:
            return None
        self.seen = None
        retry = (self.last is not None and self.last[0] != "picked"
                 and (np.linalg.norm(face.centroid - self.last[1]) < SAME_BOX_M or fell_there(self.last[2], face)))
        self.pick += 1
        self.attempt = self.attempt + 1 if retry else 1
        self.watch = PickWatcher(face, depth_m, mask, refs, box_mm)
        self.target, self.started = target, frame_time
        return {"event": "target", "pick": self.pick, "attempt": self.attempt, "target": target}

    def update(self, depth_m, intr, now):
        """A newer camera frame: the pick_result event once decided, else None."""
        if self.watch is None:
            return None
        result = self.watch.update(depth_m, intr, now)
        if result is None:
            return None
        shift = self.watch.shift
        drop = self.watch.drop if result == "dropped" else None
        self.last, self.last_corners = (result, self.watch.centroid, drop), self.watch.corners_px
        self.watch, self.decided_at = None, now
        self.last_event = {"event": "pick_result", "pick": self.pick, "attempt": self.attempt, "result": result,
                           "success": result == "picked", "seconds": round(now - self.started, 1),
                           "shift_mm": None if shift is None else round(shift * 1000), "target": self.target}
        return self.last_event

    def reset(self):
        """Stop watching the current target (the scene was changed by hand, say)."""
        self.watch, self.last, self.seen = None, None, None

    @property
    def status(self):
        if self.watch:
            return f"pick {self.pick} (try {self.attempt}): {self.watch.status}"
        return "looking for a target"


def draw_pick(img, tracker, now):
    """Draws the pick status on img (in place) and returns it: the target being watched
    (magenta), a status bar, and for BANNER_S after a result a big coloured banner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if tracker.watch:
        cv2.polylines(img, [np.round(tracker.watch.corners_px).astype(np.int32)], True, (255, 0, 255), 3)
        cv2.drawMarker(img, (int(tracker.target["u"]), int(tracker.target["v"])), (255, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 28, 3)
    e = tracker.last_event
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    last = f"   last: pick {e['pick']} {e['result']}" if e else ""
    cv2.putText(img, tracker.status + last, (10, 21), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if e and now - tracker.decided_at < BANNER_S:
        color = RESULT_COLORS[e["result"]]
        cv2.polylines(img, [np.round(tracker.last_corners).astype(np.int32)], True, color, 4)
        if tracker.last is not None and tracker.last[2] is not None:   # where the dropped box fell
            fell = cv2.resize(tracker.last[2].astype(np.uint8), (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(fell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 3)
        text = f"{e['result'].upper()}  pick {e['pick']}" + (f"  try {e['attempt']}" if e["attempt"] > 1 else "")
        (tw, th), _ = cv2.getTextSize(text, font, 1.8, 4)
        x0 = (img.shape[1] - tw) // 2 - 25
        cv2.rectangle(img, (x0, 45), (x0 + tw + 50, 45 + th + 36), color, -1)
        cv2.putText(img, text, (x0 + 25, 45 + th + 18), font, 1.8, (255, 255, 255), 4, cv2.LINE_AA)
    return img
