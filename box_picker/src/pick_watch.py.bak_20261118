"""Did the arm pick the target box, or miss it? Checked from depth, on every camera frame.

When a target is chosen, its pick face is remembered: the plane it lies in and the
pixels it covers (those with depth). In each later frame the depth at those pixels is
compared with that plane, along the face's normal:
  - in front of it (towards the camera): something is between the camera and the box,
    the arm or the suction cup;
  - on it: the box is still there;
  - behind it: the camera now sees past where the box was, so the box is gone.
Anything between the camera and a visible face is on the camera's side of its plane,
so this works the same for a front face and a top face.

PickWatcher follows one pick: waiting for the arm -> arm in view -> arm gone again ->
missed (box still there), or moved (neither, for a second: the box was knocked or
dragged), or the box left its spot. Leaving its spot is not yet a pick: the box can
still be in the gripper above the carton and fall back in. So from then on the whole
view is compared with the depth from before the pick (where the box stood: with the
farthest depth seen there since), until it settles:
  - change touching the picture's edge: the arm (or a hand) is still in view, wait;
  - nothing new in view: picked;
  - a new surface, standing still, not touching the edge: dropped, the box fell back
    into the carton (anywhere, also onto its own spot). A failed pick: try again.
A box that disappears without the arm being seen (taken by hand, or between two
frames) goes the same way.

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
import cv2
import numpy as np

PLANE_TOL_M = 0.010    # depth within this of the face's plane is the face itself
BLOCKED_FRAC = 0.05    # this share of the face with something in front of it = the arm is there
                       # (the suction cup on a top face covers only about a tenth of it)
SHADOW_FRAC = 0.30     # ... or this share with no depth (the arm's depth shadow)
GONE_FRAC = 0.60       # this share behind the plane = the box is gone
THERE_FRAC = 0.60      # this share on the plane = the box is still there
STEADY_FRAMES = 5      # a state counts once seen in this many frames in a row (1/6 s at 30 fps)
MOVED_FRAMES = 30      # after the arm, this many frames neither there nor gone = the box was moved
NEAR_M = 0.020         # once the box left its spot: depth this much nearer than before the pick = something new there
VIEW_STEP = 2          # the whole-view check looks at every 2nd pixel each way
MAJORITY_PX = 5        # ... by majority over windows this wide (in those pixels): drops speckle and flickering edges
SPOT_GROW_PX = 10      # the box's spot: its outline (or pick face), grown by this many (view) pixels
DROP_AREA_FRAC = 0.2   # a new surface at least this share of the pick face's area ...
MIN_DROP_MM2 = 800     # ... and at least this big = a box (one fallen on its narrow side still shows more) ...
MOVED_AWAY_FRAC = 0.5  # ... unless this share of its area, within a box's diagonal of it and off the box's spot, is now
                       # further away: then boxes moved (a neighbour tipped into the gap, boxes moved by hand), none was added
EDGE_MIN_PX = 100      # change this big (view pixels) touching the picture's edge = the arm or a hand in view
EDGE_BAND_PX = 64      # no depth this close to the border (view pixels) is part of the edge (the D455's left strip
                       # is baseline x fx / z wide: about 120 px at 0.5 m, 60 view pixels)
DROP_FRAMES = 10       # a new surface standing still this many frames, the arm gone = the box fell back in
CLEAR_FRAMES = 10      # this many frames with nothing new in view, the arm gone = picked
SETTLE_FRAMES = 300    # stop waiting for the view to settle after this many frames: decide on what is seen then
BANNER_S = 2.0         # seconds a result stays on the picture
RESULT_COLORS = {"picked": (0, 170, 0), "missed": (0, 0, 220), "moved": (0, 140, 255),
                 "dropped": (200, 0, 130)}   # BGR
SAME_BOX_M = 0.03      # a new target this close to a missed one is another try at the same box
STILL_M = 0.01         # a target is handed out once found this close to where it was found ...
STILL_S = 0.5          # ... at least this long before (a box still in the gripper or a hand moves)


def face_shares(pixels, normal, centroid, depth_m, intr):
    """Shares of the pixels whose depth is (in front of, on, behind) the plane, and with no depth;
    shift: their median distance from the plane, metres (+ towards the camera, None without depth)."""
    u, v = pixels[:, 0], pixels[:, 1]
    z = depth_m[v, u]
    has = z > 0
    pts = np.stack([(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z], axis=1)
    s = (pts - centroid) @ normal                      # metres towards the camera
    n = max(len(z), 1)
    return {"front": np.sum(has & (s > PLANE_TOL_M)) / n, "on": np.sum(has & (np.abs(s) <= PLANE_TOL_M)) / n,
            "behind": np.sum(has & (s < -PLANE_TOL_M)) / n, "none": np.sum(~has) / n,
            "shift": float(np.median(s[has])) if has.any() else None}


def frame_state(shares):
    """gone / blocked / there / unclear, from face_shares()."""
    if shares["behind"] >= GONE_FRAC:
        return "gone"
    if shares["front"] >= BLOCKED_FRAC or shares["none"] >= SHADOW_FRAC:
        return "blocked"
    if shares["on"] >= THERE_FRAC:
        return "there"
    return "unclear"


def majority(mask, valid, k=MAJORITY_PX):
    """Pixels where at least half of the valid pixels in the k x k window around them are in
    mask, and at least a third of the window is valid. Holes in the depth don't break up a
    surface, and a thin flickering edge or stray pixels go."""
    box = lambda m: cv2.boxFilter(m.astype(np.float32), -1, (k, k), borderType=cv2.BORDER_REPLICATE)
    share_valid = box(valid)
    return (box(mask) >= 0.5 * share_valid) & (share_valid >= 1 / 3)


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


def view_change(ref, depth, intr, spot, min_m2, zone, reach_px, step=VIEW_STEP):
    """Compare a view (every step-th pixel) with the reference depth from before the pick.
    -> (edge, new): whether a change of at least EDGE_MIN_PX touches the picture's edge
    (`zone`, from edge_zone(): the arm or a hand); and where something was added (bool), else None: a connected
    change not touching the edge, with at least min_m2 square metres of it nearer than ref
    by NEAR_M, and not mostly balanced by surface within reach_px of it that is now further
    away (outside `spot`, where the box stood): that is boxes moved, not one added.
    Lost depth counts as change (the arm can be too close for depth), not as area."""
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
    return edge, (new if new.any() else None)


class PickWatcher:
    """Follows one pick, from the target's face in the frame it was found in."""

    def __init__(self, face, depth_m, mask=None):
        """face: the target's pick face; depth_m: the depth it was found in; mask: the box's
        whole outline, if known (where it stood, also its top and sides)."""
        has = depth_m[face.pixels[:, 1], face.pixels[:, 0]] > 0     # filled-in pixels have no depth to compare
        self.pixels, self.normal, self.centroid = face.pixels[has], face.normal, face.centroid
        self.corners_px = face.corners_px                         # for drawing
        self.arm_seen = False
        self.state, self.count = None, 0
        self.shift = None              # median distance of what is seen now from the face's plane, metres
        # For the whole-view check once the box has left its spot: the depth before the pick,
        # the box's spot in it, and how big a new surface must be to be a box.
        self.ref = depth_m[::VIEW_STEP, ::VIEW_STEP].astype(np.float32)
        spot = np.zeros(self.ref.shape, np.uint8)
        spot[np.minimum(face.pixels[:, 1] // VIEW_STEP, spot.shape[0] - 1),
             np.minimum(face.pixels[:, 0] // VIEW_STEP, spot.shape[1] - 1)] = 1
        if mask is not None:
            spot |= mask[::VIEW_STEP, ::VIEW_STEP].astype(np.uint8)
        self.spot = cv2.dilate(spot, np.ones((2 * SPOT_GROW_PX + 1,) * 2, np.uint8)) > 0
        self.zone = edge_zone(self.ref)
        c = np.asarray(face.corners_px, float)
        self.reach_px = max(1, int(np.linalg.norm(c[0] - c[2]) / VIEW_STEP))   # the face's diagonal, view pixels
        self.min_drop_m2 = max(MIN_DROP_MM2, DROP_AREA_FRAC * face.size_mm[0] * face.size_mm[1]) / 1e6
        self.left = False              # the box has left its spot; picked once it has left the view too
        self.settle, self.clear, self.still = 0, 0, 0
        self.view_edge = False         # the arm (or a hand) is in view
        self.drop = None               # where a new surface stands (bool, view pixels), while one is seen

    def update(self, depth_m, intr):
        """Feed a newer frame: "picked", "missed", "moved" or "dropped" once decided, else None."""
        shares = face_shares(self.pixels, self.normal, self.centroid, depth_m, intr)
        state, self.shift = frame_state(shares), shares["shift"]
        self.count = self.count + 1 if state == self.state else 1
        self.state = state
        if self.left:
            return self._settle(depth_m, intr)
        if self.count < STEADY_FRAMES:
            return None
        if state == "gone":
            self.left = True
            return self._settle(depth_m, intr)
        if state == "blocked":
            self.arm_seen = True
        elif self.arm_seen and state == "there":
            return "missed"
        elif self.arm_seen and self.count >= MOVED_FRAMES:
            return "moved"
        return None

    def _settle(self, depth_m, intr):
        """After the box left its spot: picked once the view is clear, dropped if a new
        surface stands still in it (the box fell back in), else None (still waiting)."""
        d = depth_m[::VIEW_STEP, ::VIEW_STEP]
        # Where the box stood, compare with the farthest seen since: what was behind it.
        self.ref[self.spot] = np.maximum(self.ref[self.spot], d[self.spot])
        self.view_edge, drop = view_change(self.ref, d, intr, self.spot, self.min_drop_m2, self.zone, self.reach_px)
        self.settle += 1
        if self.view_edge:
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
        if self.settle >= SETTLE_FRAMES:           # the arm stays in view: decide on what is seen
            return "dropped" if self.drop is not None else "picked"
        return None

    @property
    def status(self):
        if self.left:
            if self.view_edge:
                return "box left its spot, arm still in view"
            return "box left its spot, new surface in view (fell back?)" if self.drop is not None \
                else "box left its spot, checking the view"
        if self.state == "blocked" and self.count >= STEADY_FRAMES:
            return "arm in view"
        return "arm gone, checking" if self.arm_seen else "waiting for the arm"


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

    def offer(self, target, face, depth_m, frame_time, mask=None):
        """A pipeline result (target dict, its pick face, the depth it came from, when that frame
        was taken, and the box's outline if known). It becomes the target only when no pick is being watched, its frame is
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
        self.watch = PickWatcher(face, depth_m, mask)
        self.target, self.started = target, frame_time
        return {"event": "target", "pick": self.pick, "attempt": self.attempt, "target": target}

    def update(self, depth_m, intr, now):
        """A newer camera frame: the pick_result event once decided, else None."""
        if self.watch is None:
            return None
        result = self.watch.update(depth_m, intr)
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
