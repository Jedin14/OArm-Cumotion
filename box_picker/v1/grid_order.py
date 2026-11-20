"""v1 pick order: the boxes of one carton are equal and stand in a grid (a lattice).

Use: order(frame) -> {"picks": [...every box to pick, in order...], "grid_boxes", "box_size_mm", "carton"};
frame and the fields of a pick are described in test_v1.py. pick_v1.py runs it on recorded frames.
Built 2026-11-13/16 from three tried designs (designs/); this one won on the hand labels, also
on grid_part3, which it was not tuned on.

Every tolerance is a share of the box measured in this frame: its width W and height H (from the
fronts), its thickness t = the smaller of the two, and its depth D (front to back). No set
distances, so the same rules hold for any box size, and nothing is tuned to one recording. The
layers use D only, never W, H or t: a box can be much shallower than its front is wide (the robot's
125 x 125 x 54 mm boxes, 18 Nov 2026), so nothing about its depth follows from its front.

0. THE CARTON (v2, carton.py)  With frame["cartons"] (SAM's carton outlines), only the boxes in the
   carton in front are used: the carton whose near edge is closest to the camera, and the boxes
   with at least half their outline inside the shape its outline goes round. Everything below
   then sees only those boxes, so clutter outside can't set the floor or count as fallen.
   "carton": None (no cartons given), or {"found": False} (none found: every box used), or
   {"found": True, "inside": n, "left_out": [box indices], "score", "near_m", "area": bool HxW}.
   An empty carton in front gives no picks, even if another carton has boxes.

1. THE GRID'S AXES
   up:      from the camera's accelerometer.
   forward: the typical direction the boxes' fronts face (area-weighted median of the faces turned
            to the camera), made level: from the camera into the carton.
   across:  forward x up: to the right as the camera sees it.
   Each face of each outline is then a front, top or side by the axis it faces most. An outline
   with neither front nor top (only a side: the side of a box beside it) is not a box. An outline
   with no faces at all (no depth on it) can't be gripped and is left out.

2. THE BOX  (W x H x D)
   W, H: the median width (across) and height (up) of the fronts in view.
   D: frame["box_depth"], the box size given from outside (live: the robot's
   /beast_arm/held_object/fixed_size, the size entered in its GUI; the pick loop doesn't pick
   without it). The layers below use it. Without it (offline runs), only reported: the step
   between the grid's layers (median), or the median depth of the tops as wide as a box and the
   sides as high as one (none shallower than t: those are slivers).

3. WHAT IS A BOX  Something must hold it up: its bottom is on the floor (the lowest front bottom
   in view) or on the top of a box that is held up and side by side with it, within t. A box seen
   only from above shows no bottom: it holds if the floor or a held box is under it within one box
   size. Pieces of the carton's rim float far above everything and are dropped.

4. FALLEN  A box that does not fit the lattice, any of:
   LOW     its top is less than half a box height above the floor (lying where boxes stand up);
   TALL    its front is more than one and a half box heights tall (standing on its side or end);
   FLAT    its front is less than half a box height tall but more than one and a half box widths
           wide (lying where the boxes stand up, on the floor or on the stack);
   TURNED  turned round so that one end of its front sticks out forward or back by more than t
           (a box seen only from above: one end of its top's long edge);
   APART   seen only from above, it lies in front of the box behind it with a gap of more than t
           (grid boxes touch the layer behind them).
   Leaning boxes (damaged ones) stay in the grid: leaning rolls a front sideways; it doesn't make
   it taller, lift it off the floor or turn it round.

5. THE LATTICE
   Layers: the boxes' fronts sorted by distance along forward; a step of more than half the box
   depth D starts a new layer (layers are whole box depths apart). Without D every box is in one
   layer: from one picture, a box one layer back can't be told from a box set back a little,
   without knowing how deep a box is. A box standing on a box of a nearer layer with its middle
   over it (its front less than half a box depth behind that box's back) joins that layer, so it
   is picked before the box under it. Layer 1 is the nearest.
   Rows: inside a layer, the boxes sorted by the height of their top; a drop of more than half a
   box height starts a new row (tops of one row lie within a third of H, rows a whole H apart).
   Row 1 is the highest.
   Columns: inside a row, left to right along across.

6. ORDER
   Fallen boxes first, each by its largest face (its front or its top), the highest first, then
   the nearest, then the leftmost.
   Then the grid by its front: layer 1 row by row from the top, each row left to right; then layer
   2 the same way, and so on. A grid box with no front in view (only its top) is gripped by its
   top. Hidden boxes are not listed: only outlines found in the picture.

grid_boxes: the grid's places seen from the front: each grid box (an outline as wide as k boxes
counts k), less those right behind a nearer grid box of the same row.
"""
import numpy as np

import carton

FRONT_DEG = 30   # faces within this of facing the camera set forward and the box size (faces.py's front)
# Shares of the box (W, H, t = min(W, H)); the comments give the margins seen on the dev recordings.
LOW = 0.5        # x H: a top lower than this above the floor is lying down   (grid >= 0.75, fallen <= 0.24)
TALL = 1.5       # x H: a front taller than this is on its side or end         (grid <= 1.3, fallen >= 3.7)
                 # x W, with a front lower than LOW x H: lying flat             (grid: none lower; wider <= 2.1)
TURN = 0.85      # x t: a front end sticking out more than this is turned      (grid <= 0.68, fallen >= 1.08)
APART = 1.0      # x t: a gap to the box behind wider than this: lying apart   (fallen >= 2.2)
LAYER = 0.5      # x D: fronts further back than this start a new layer (D: the box depth, from the picks)
DROP = 0.5       # x H: tops lower than this start a new row                   (in a row <= 0.33, rows >= 0.68)
HOLD = 1.0       # x t: a bottom this close to the floor / a box top rests on it


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rect3d(face, intr):
    """The face's fitted rectangle in 3D: its picture corners put back on the face's plane."""
    n, c = face.normal, face.centroid
    out = []
    for u, v in np.asarray(face.corners_px, float):
        r = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
        out.append(r * (n @ c) / (n @ r))
    return np.array(out)


def wmedian(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def frame_axes(boxes, up):
    """(up, forward, across): forward is level, away from the camera, square to the typical front."""
    up = unit(up)
    zl = unit(np.array([0.0, 0.0, 1.0]) - up[2] * up)
    xl = np.cross(zl, up)
    fr = [f for b in boxes if b.pick is not None for f in b.faces if f.kind == "front" and f.angle_deg <= FRONT_DEG]
    yaw = wmedian([np.arctan2(-f.normal @ xl, -f.normal @ zl) for f in fr], [f.area_mm2 for f in fr]) if fr else 0.0
    fwd = np.cos(yaw) * zl + np.sin(yaw) * xl
    return up, fwd, np.cross(fwd, up)


def edges(r, ax):
    """The rectangle's two edge vectors, the one more along ax first."""
    e0, e1 = r[1] - r[0], r[2] - r[1]
    return (e0, e1) if abs(unit(e0) @ ax) >= abs(unit(e1) @ ax) else (e1, e0)


class Piece:
    """One outline measured in the grid's axes (metres)."""

    def __init__(self, i, box, intr, up, fwd, acr):
        self.i, self.box = i, box
        by = {"front": [], "top": [], "side": []}
        for f in box.faces:
            n = f.normal
            k = int(np.argmax([-(n @ fwd), abs(n @ up), abs(n @ acr)]))
            by[("front", "top", "side")[k]].append(f)
        big = lambda fs: max(fs, key=lambda f: f.area_mm2) if fs else None
        self.front, self.top, self.side = big(by["front"]), big(by["top"]), big(by["side"])
        self.fw = self.fh = self.ta = self.tf = self.sf = self.sh = self.back = None
        self.stick = 0.0            # how far one end of its front (else its top) sticks out forward or back
        self.why, self.fallen, self.layer, self.row, self.col = "", False, None, None, None
        corners = []
        if self.front is not None:
            r = rect3d(self.front, intr)
            corners.append(r)
            eh, ev = edges(r, acr)
            self.fw, self.fh = float(np.linalg.norm(eh)), float(np.linalg.norm(ev))
            n = self.front.normal - (self.front.normal @ up) * up
            self.stick = self.fw * abs(np.sin(np.arctan2(abs(n @ acr), -(n @ fwd))))
        if self.top is not None:
            r = rect3d(self.top, intr)
            corners.append(r)
            ea, ef = edges(r, acr)
            self.ta, self.tf = float(np.linalg.norm(ea)), float(np.linalg.norm(ef))
            self.back = float(max(r @ fwd))
            if self.front is None:
                self.stick = min(abs(ea @ fwd), abs(ea @ acr))
        if self.side is not None:
            ef, ev = edges(rect3d(self.side, intr), fwd)
            self.sf, self.sh = float(np.linalg.norm(ef)), float(np.linalg.norm(ev))
        if self.front is not None:                 # where it stands: its front plane, across its middle
            self.f, self.a = float(self.front.centroid @ fwd), float(self.front.centroid @ acr)
        elif self.top is not None:                 # seen only from above: its top's near edge
            self.f, self.a = float(min(corners[0] @ fwd)), float(self.top.centroid @ acr)
        else:
            self.f = self.a = None
            return
        c = np.vstack(corners)
        self.left, self.right = float(min(c @ acr)), float(max(c @ acr))
        self.hi, self.lo = box.high, box.low


def front_size(ps):
    """W, H: the median width and height of the fronts (those turned to the camera, if any)."""
    fr = [p for p in ps if p.front is not None and p.front.angle_deg <= FRONT_DEG] or [p for p in ps if p.front is not None]
    if fr:
        return float(np.median([p.fw for p in fr])), float(np.median([p.fh for p in fr]))
    w = float(np.median([p.ta for p in ps]))
    return w, w


def supported(ps, W, H):
    """(the pieces something holds up, the floor height): see 3 in the docstring."""
    t = HOLD * min(W, H)
    floor = min(p.lo for p in ps if p.front is not None) if any(p.front is not None for p in ps) else min(p.lo for p in ps)
    held, todo = [], sorted(ps, key=lambda p: p.lo)
    changed = True
    while changed:
        changed = False
        for p in list(todo):
            beside = [q for q in held if q.left < p.right and p.left < q.right]
            if p.front is None:
                ok = p.hi - max(W, H) - t <= max([floor] + [q.hi for q in beside if q.hi <= p.hi + t])
            else:
                ok = p.lo <= floor + t or any(abs(p.lo - q.hi) <= t for q in beside)
            if ok:
                held.append(p)
                todo.remove(p)
                changed = True
    return held, floor


def find_fallen(ps, floor, W, H, D=None):
    """Set p.why ('' = fits the lattice) for each piece: see 4 in the docstring. D: the box depth,
    which tells what counts as behind (half a box depth back), else t as before."""
    t = min(W, H)
    back = LAYER * D if D else t
    for p in ps:
        if p.hi - floor < LOW * H:
            p.why = "low"
        elif p.fh is not None and p.fh > TALL * H:
            p.why = "tall"
        elif p.fh is not None and p.fh < LOW * H and p.fw > TALL * W:
            p.why = "flat"
        elif p.stick > TURN * t:
            p.why = "turned"
    for p in ps:
        if p.why or p.front is not None:
            continue
        behind = [q.f for q in ps if not q.why and q is not p and q.f > p.f + back
                  and q.left < p.right and p.left < q.right and q.lo < p.hi and p.lo < q.hi]
        if behind and min(behind) - p.back > APART * t:
            p.why = "apart"
    for p in ps:
        p.fallen = bool(p.why)


def runs(ps, key, gap):
    """ps sorted by key, cut where key jumps by more than gap."""
    ps = sorted(ps, key=key)
    out = [[ps[0]]] if ps else []
    for p in ps[1:]:
        if key(p) - key(out[-1][-1]) > gap:
            out.append([p])
        else:
            out[-1].append(p)
    return out


def rests_on(p, q, W, H, D):
    """p stands on q's top with its middle over q (p's middle, D/2 behind its front, is in front of
    q's back, D behind q's front): taking q first would drop p."""
    t = HOLD * min(W, H)
    return (abs(p.lo - q.hi) <= t and q.left < p.right and p.left < q.right
            and p.f + D / 2 < q.f + D)


def lattice(grid, W, H, D=None):
    """Set layer, row (0 = nearest / highest) and column of each grid piece; -> the layers.
    D: the box depth (frame["box_depth"]); None: one layer. A box resting on a box of a nearer
    layer (rests_on) joins that layer, so it comes before the box under it."""
    layers = runs(grid, lambda p: p.f, LAYER * D) if D else runs(grid, lambda p: p.f, np.inf)
    for k, layer in enumerate(layers):
        for p in layer:
            p.layer = k
    changed = D is not None
    while changed:
        changed = False
        for p in grid:
            under = [q for q in grid if q.layer < p.layer and rests_on(p, q, W, H, D)]
            if under:
                p.layer, changed = min(q.layer for q in under), True
    layers = [sorted([p for p in grid if p.layer == k], key=lambda p: p.f) for k in range(len(layers))]
    layers = [l for l in layers if l]
    a0 = min((p.a for p in grid), default=0.0)
    for k, layer in enumerate(layers):
        for r, row in enumerate(runs(layer, lambda p: -p.hi, DROP * H)):
            for p in row:
                p.layer, p.row, p.col = k, r, int(round((p.a - a0) / W))
    return layers


def depth(grid, layers, W, H, known=None):
    """D: see 2 in the docstring. -> (D or None, how it was measured)."""
    if known:
        return float(known), "given"
    steps = np.diff([np.median([p.f for p in l]) for l in layers if len(l) >= 2])
    if len(steps):
        return float(np.median(steps)), "layers"
    d = [p.tf for p in grid if p.ta is not None and abs(p.ta - W) < W / 4] + \
        [p.sf for p in grid if p.sh is not None and abs(p.sh - H) < H / 4]
    d = [x for x in d if x >= min(W, H)]
    return (float(np.median(d)), "faces") if d else (None, "none")


def seen_places(grid, W):
    """grid_boxes: see the end of the docstring."""
    n = 0
    for p in grid:
        behind = any(q.layer < p.layer and q.row is not None and abs(q.hi - p.hi) < (q.hi - q.lo) / 2
                     and min(q.right, p.right) - max(q.left, p.left) > min(q.right - q.left, p.right - p.left) / 2
                     for q in grid)
        if not behind:
            n += max(1, int(round(p.fw / W))) if p.fw is not None else 1
    return n


def pick(p, face):
    return {"box": p.i, "face": "front" if face is p.front else "top", "fallen": p.fallen,
            "uv": tuple(map(float, face.centroid_px)), "xyz": tuple(map(float, face.centroid)),
            "cell": None if p.fallen else (p.layer, p.row, p.col),
            "corners_px": np.asarray(face.corners_px, float).tolist(), "size_mm": tuple(map(float, face.size_mm)),
            "face_obj": face}       # the faces.Face itself (the live loop's pick check needs it); not JSON


def in_carton(frame):
    """v2 (carton.py): (indices of the boxes in the carton in front, what was found). With no
    "cartons" in the frame, or none found, every box is used."""
    boxes = frame["boxes"]
    if frame.get("cartons") is None:
        return list(range(len(boxes))), None
    front = carton.front_carton(frame["cartons"], frame["depth"], boxes)
    if front is None:
        return list(range(len(boxes))), {"found": False, "inside": len(boxes), "left_out": []}
    keep = carton.inside(boxes, front)
    return keep, {"found": True, "inside": len(keep), "left_out": [i for i in range(len(boxes)) if i not in keep],
                  "score": front["score"], "near_m": round(front["near_m"], 3), "area": front["area"]}


def order(frame):
    boxes, intr = frame["boxes"], frame["intr"]
    keep, found = in_carton(frame)
    up, fwd, acr = frame_axes([boxes[i] for i in keep], frame["up"])
    ps = [Piece(i, boxes[i], intr, up, fwd, acr) for i in keep if boxes[i].pick is not None]
    ps = [p for p in ps if p.f is not None]
    if not ps:
        return {"picks": [], "grid_boxes": 0, "box_size_mm": None, "carton": found}
    W, H = front_size(ps)
    ps, floor = supported(ps, W, H)
    known = frame.get("box_depth")
    find_fallen(ps, floor, W, H, known)
    grid = [p for p in ps if not p.fallen]
    fallen = [p for p in ps if p.fallen]
    layers = lattice(grid, W, H, known)
    D, how = depth(grid, layers, W, H, known)

    picks = [pick(p, max([f for f in (p.front, p.top) if f is not None], key=lambda f: f.area_mm2))
             for p in sorted(fallen, key=lambda p: (-p.hi, p.f, p.a))]
    picks += [pick(p, p.front if p.front is not None else p.top)
              for p in sorted(grid, key=lambda p: (p.layer, p.row, p.a))]
    out = {"picks": picks, "grid_boxes": seen_places(grid, W),
           "box_size_mm": (W * 1000, H * 1000, None if D is None else D * 1000), "box_depth_from": how,
           "forward": fwd, "carton": found}
    if frame.get("debug"):
        mm = lambda x: "-" if x is None else f"{x * 1000:.0f}"
        out["debug"] = [f"fwd {np.round(fwd, 3)} W {mm(W)} H {mm(H)} D {mm(D)} ({how}) floor {mm(floor)}"] + [
            f"{p.i:2d} f {mm(p.f)} a {mm(p.a)} hi {mm(p.hi - floor)} lo {mm(p.lo - floor)} front {mm(p.fw)}x{mm(p.fh)}"
            f" top {mm(p.ta)}x{mm(p.tf)} stick {mm(p.stick)} {p.why or ('L%d R%d C%d' % (p.layer, p.row, p.col))}"
            for p in sorted(ps, key=lambda p: p.i)]
    return out
