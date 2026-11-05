"""Depth part of the pick pipeline: box outline + aligned depth -> faces -> pick face -> centroid.

Follows docs/picking_rules.md:
  - each box is split into flat faces by fitting planes to its depth points;
  - a face pointing up is `top`, one pointing at the camera is `front`, anything
    else (left/right, tilted) is `side` and is never picked;
  - the pick face is the larger of the box's `front` and `top` faces;
  - target: nothing resting on it, nothing in front of it, then boxes standing on
    another box (the highest top first) before those on the floor, then the nearest,
    then leftmost.

Camera coordinates are the RealSense ones: x right, y down, z forward, metres.
No torch here, so it can be tested without a GPU (test_faces.py).
"""
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from pick_rules import pick_face

DEFAULT_DEPTH_TOL = 0.015  # metres: nearer by more than this counts as "in front"
FACE_ANGLE_DEG = 30      # how far a face may tilt and still count as front / top
MIN_FACE_POINTS = 150    # smaller planes are edges or noise
MAX_POINTS = 6000        # points per box used for plane fitting
SAME_FACE_DEG = 45      # faces of one box closer in direction than this are the same face
SAME_PLANE_M = 0.010    # same-direction planes closer than this are one face split by noise
FLAT_VIEW_DEG = 10       # a top whose front edge is seen at less than this angle loses to the box's front
SAME_WIDTH = 0.2         # a top must be this close (share) to its front's width, along the edge they share
REST_TOL_M = 0.03        # a box whose bottom is this close to another's top height rests on it
MAX_BOX_DEPTH_M = 0.30   # ... and can stand at most this far behind that box's front
SUPPORT_M = 0.02         # a box's top must reach this far behind another's front to hold it up
FAR_SMEAR_M = 0.015      # a top's depth reaching up to this past its outline's far edge is smeared past it
HIGHEST_TOL_M = 0.015    # boxes on a box whose tops are within this of the highest one's count as highest
                         # (under the thinnest box, 2.7 cm: a pile one box lower is lower)
NEAREST_TOL_M = 0.03     # free boxes whose grip point is within this of the nearest one's count as nearest
SKIP_OVERLAP = 0.5       # a box whose outline overlaps a skipped one's this much (of the smaller) is that box
EDGE_PX = 3              # a pick face this close to the picture's edge is cut off by it (at_edge)
SAME_LENGTH = 0.07       # fronts this close in length (share) are the same kind of box, for split_doubles
DOUBLE_TOL = 0.15        # a front this close (share) to twice the others' thickness is two boxes pressed together
DOUBLE_REFS = 3          # ... judged only with at least this many other fronts of that length in view
IN_WAY_M = 0.015         # a nearer box whose top reaches this far above a box's bottom is in front of it
CLEAR_M = 0.03           # a nearer box closer than this side to side is in the way (a gripper needs room)
EDGE_DENSITY = 0.15      # a 1 mm slice of a face with less than this share of a typical slice's points is past its edge
SQUARE_RATIO = 4.0      # spread ratio (length/width squared) under which a face is too square for a main direction
UPRIGHT_DEG = 30         # with a level top found, other planes must be within this of upright
LEVEL_SHARE = 0.25       # a level plane holding this share of a box's points is its top, found before the rest
LEVEL_DEG = 15           # a plane this close to level is a top, and is set exactly level (needs the right up)
TWIN_DEG = 10            # a new plane this close in direction to a found one, and within its band, is that one again
REMOVE_FRAC = 0.5       # after a plane is found, points within this share of the threshold leave the pool
FLAT_DEV_M = 0.003       # ... only where the depth there bends less than this from flat
DIR_PX = 7               # a pixel's surface direction is read from the points this far around it
ERODE_PX = 4             # shrink each outline so edge pixels (mixed depth) are dropped
MAX_FILL_PX = 40         # widest strip with no depth at a face's side that is placed on its plane
D455_BASELINE_M = 0.095  # distance between the D455's two depth sensors: sets how wide a depth shadow can be
SHADOW_SLACK_PX = 4      # allowance on that width (edge pixels, colour/depth alignment)
SHADOW_LOOK_PX = 60      # how far past the outline to look for the depth beside it (a shadow often carries on there)
MIN_RAY_COS = 0.5        # ... only where the camera ray meets the face at 30 degrees or more
NEARER_M = 0.01          # something beside a face this much nearer casts its shadow onto it


@dataclass(eq=False)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_rs(cls, i):
        return cls(i.fx, i.fy, i.ppx, i.ppy)

    def project(self, p):
        return (self.fx * p[0] / p[2] + self.cx, self.fy * p[1] / p[2] + self.cy)


@dataclass(eq=False)
class Face:
    kind: str                # front / top / side
    normal: np.ndarray       # unit vector, pointing towards the camera
    angle_deg: float         # angle to the ideal front / top direction (side: to the nearer of the two)
    size_mm: tuple           # (long, short) edge of the face, from a rectangle fitted in the plane
    centroid: np.ndarray     # centre of that rectangle, metres
    centroid_px: tuple
    pixels: np.ndarray       # (N, 2) u, v of the face's points, for drawing
    corners_px: np.ndarray   # (4, 2) the measured rectangle, projected into the image

    @property
    def area_mm2(self):
        return self.size_mm[0] * self.size_mm[1]


@dataclass(eq=False)
class Box:
    mask: np.ndarray
    bbox: tuple                              # x1, y1, x2, y2 of the outline
    depth: float                             # near edge of its front / top faces (10th percentile of z), 0 = unknown
    faces: list = field(default_factory=list)
    pick: Face = None                        # larger of front / top, None if neither was found
    low: float = 0.0                         # lowest and highest point of its faces, metres along up
    high: float = 0.0
    left: float = 0.0                        # leftmost and rightmost point of its faces, metres along x
    right: float = 0.0
    blocked_by: list = field(default_factory=list)   # indices of boxes resting on it
    top_only: bool = False                   # on a box, but only its top was found: picked by it as a last resort (flag it)
    hidden_by: list = field(default_factory=list)    # indices of boxes standing in front of it, hiding part of it


def deproject(depth_m, mask, intr, erode_px=ERODE_PX):
    """Pixels of `mask` with valid depth -> (points (N, 3) metres, pixels (N, 2) u, v)."""
    if erode_px:
        mask = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)) > 0
    v, u = np.nonzero(mask & (depth_m > 0))
    z = depth_m[v, u]
    pts = np.stack([(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z], axis=1)
    return pts, np.stack([u, v], axis=1)


def fit_plane(pts, thr, iters=200, rng=None):
    """RANSAC plane. Returns (unit normal, offset d with n.p + d = 0, inlier mask) or None."""
    rng = rng or np.random.default_rng(0)
    best = None
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inl = np.abs(pts @ n - n @ a) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        return None
    # Refine on the inliers: the normal is the direction of least spread.
    q = pts[best]
    centre = q.mean(axis=0)
    n = np.linalg.svd(q - centre, full_matrices=False)[2][2]
    inl = np.abs((pts - centre) @ n) < thr
    return n, -n @ centre, inl


def surface_dirs(depth_m, mask, intr, k=DIR_PX):
    """Surface direction (unit normal, towards the camera) at each of the outline's pixels,
    from the 3D points k pixels around it on smoothed depth, and whether it could be read.
    -> (normals (N, 3), ok (N,), pixels (N, 2)) for the outline's pixels."""
    v, u = np.nonzero(mask)
    pad = 3 * k
    y0, y1 = max(0, v.min() - pad), min(mask.shape[0], v.max() + pad + 1)
    x0, x1 = max(0, u.min() - pad), min(mask.shape[1], u.max() + pad + 1)
    z0 = cv2.medianBlur((depth_m[y0:y1, x0:x1] * 1000).astype(np.float32), 5) / 1000
    valid = (z0 > 0).astype(np.float32)
    z = cv2.GaussianBlur(z0, (0, 0), k / 2) / np.maximum(cv2.GaussianBlur(valid, (0, 0), k / 2), 1e-6) * (z0 > 0)
    uu, vv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    P = np.dstack([(uu - intr.cx) * z / intr.fx, (vv - intr.cy) * z / intr.fy, z])
    du, dv = np.zeros_like(P), np.zeros_like(P)
    du[:, k:-k] = P[:, 2 * k:] - P[:, :-2 * k]
    dv[k:-k] = P[2 * k:] - P[:-2 * k]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    n[(n * P).sum(axis=2) > 0] *= -1
    ok = z0 > 0
    for s_ in (-k, k):
        ok &= (np.roll(z0, s_, 0) > 0) & (np.roll(z0, s_, 1) > 0)
    # Only where the surface is flat: on a plane the middle depth sits halfway between the depths
    # k pixels either side; across a step (a box's edge over the floor) it doesn't, and the
    # smoothed depth there would look like a slope facing the camera.
    for ax in (0, 1):
        mid = (np.roll(z0, k, ax) + np.roll(z0, -k, ax)) / 2
        ok &= np.abs(z0 - mid) < FLAT_DEV_M
    return n[v - y0, u - x0], ok[v - y0, u - x0], np.stack([u, v], axis=1)


def classify(n, up, forward=np.array([0.0, 0.0, 1.0]), max_deg=FACE_ANGLE_DEG):
    """front / top / side, and the angle to the matching ideal direction."""
    toward_cam = -(forward - (forward @ up) * up)
    toward_cam /= np.linalg.norm(toward_cam)
    a_top = np.degrees(np.arccos(np.clip(n @ up, -1, 1)))
    a_front = np.degrees(np.arccos(np.clip(n @ toward_cam, -1, 1)))
    if a_top <= max_deg:
        return "top", a_top
    if a_front <= max_deg:
        return "front", a_front
    return "side", min(a_top, a_front)


def dense_span(x, bin_mm=1.0):
    """(lo, hi) of 1-D positions in mm: from the first to the last 1 mm slice holding at least
    EDGE_DENSITY of a typical slice's points, inside the 0.2-99.8 percentiles. Across a real
    face every slice holds about as many points; a bump or a few stray points hold far fewer."""
    lo, hi = np.percentile(x, [0.2, 99.8])
    if hi - lo < 3 * bin_mm:
        return lo, hi
    counts, edges = np.histogram(x, bins=np.arange(lo, hi + bin_mm, bin_mm))
    full = np.flatnonzero(counts >= EDGE_DENSITY * np.median(counts[counts > 0]))
    return edges[full[0]], edges[full[-1] + 1]


def describe(pts, pix, n, intr, up, edge=None):
    """Face from its inlier points: orientation, rectangle size and centre.

    edge: a 3D direction one rectangle edge must follow (the line a neighbouring face meets
    it along); otherwise the direction comes from how the points spread.
    """
    centre = pts.mean(axis=0)
    if n @ centre > 0:          # visible faces point back at the camera
        n = -n
    e1 = np.cross(n, up)
    if np.linalg.norm(e1) < 1e-6:                    # top face: any in-plane axis will do
        e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    q = ((pts - centre) @ np.stack([e1, e2], axis=1) * 1000).astype(np.float32)   # mm in the plane
    # Rectangle orientation: along the given edge, else the points' main direction (a bump on
    # the outline tilts the tightest-fit rectangle, and on a long face a few degrees adds
    # centimetres of width); nearly square faces, where that direction is weak, use the
    # tightest fit. Extent: where the points are dense (dense_span), so a bump or a few
    # noisy points beyond the edge don't stretch it. Once more on the points inside, so a
    # bump can't tilt the direction either.
    keep = np.ones(len(q), bool)
    for _ in range(2):
        if edge is not None and np.hypot(edge @ e1, edge @ e2) > 0.5:
            angle = np.arctan2(edge @ e2, edge @ e1)
        else:
            evals, evecs = np.linalg.eigh(np.cov(q[keep].T.astype(float)))
            if evals[1] > SQUARE_RATIO * evals[0]:
                angle = np.arctan2(evecs[1, 1], evecs[0, 1])
            else:
                angle = np.radians(cv2.minAreaRect(q[keep])[2])
        a1, a2 = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
        (lo1, hi1), (lo2, hi2) = dense_span(q[keep] @ a1), dense_span(q[keep] @ a2)
        keep = ((q @ a1 >= lo1) & (q @ a1 <= hi1) & (q @ a2 >= lo2) & (q @ a2 <= hi2)) | ~keep.any()
    w, h = hi1 - lo1, hi2 - lo2
    rx, ry = (lo1 + hi1) / 2 * a1 + (lo2 + hi2) / 2 * a2
    c3 = centre + (rx * e1 + ry * e2) / 1000
    d1, d2 = (a1[0] * e1 + a1[1] * e2) * w / 2000, (a2[0] * e1 + a2[1] * e2) * h / 2000
    corners = np.array([intr.project(c3 + s1 * d1 + s2 * d2) for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    kind, angle = classify(n, up)
    return Face(kind, n, round(float(angle), 1), (round(float(max(w, h)), 1), round(float(min(w, h)), 1)),
                c3, intr.project(c3), pix, corners)


def largest_region(pix):
    """Boolean mask of the pixels in the largest connected region (drops stray points)."""
    x0, y0 = pix.min(axis=0)
    img = np.zeros(tuple(pix.max(axis=0)[::-1] - (y0, x0) + 1), np.uint8)
    img[pix[:, 1] - y0, pix[:, 0] - x0] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # bridge depth holes
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels[pix[:, 1] - y0, pix[:, 0] - x0] == best


def inner_holes(depth_m, mask, intr, plane, pix, h_pts, h_pix):
    """No-depth pixels inside a face's area (the convex hull of its pixels, within the outline),
    placed on its plane and added to the strips fill_holes found. A top seen at a slant can
    have no depth on half of it."""
    hull = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pix.astype(np.int32)), 1)
    hole = (hull > 0) & mask & (depth_m <= 0)
    if len(h_pix):
        hole[h_pix[:, 1], h_pix[:, 0]] = False
    v, u = np.nonzero(hole)
    if not len(u):
        return h_pts, h_pix
    n, d = plane
    ray = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones(len(u))], axis=1)
    cos_n = ray @ n
    good = np.abs(cos_n) > 0.2                       # not a plane seen nearly edge-on
    pts = ray[good] * (-d / cos_n[good])[:, None]
    return np.concatenate([h_pts, pts]), np.concatenate([h_pix, np.stack([u[good], v[good]], axis=1)])


def fill_holes(depth_m, mask, intr, planes, face_pix, max_px=MAX_FILL_PX):
    """Strips with no depth at a face's left and right ends, placed on the face's plane.

    The D455 has no depth in the strip beside a box where only one of its two sensors
    can see what is behind (its depth "shadow"). Without this the face, its size and its
    centre stop short of that strip. Only runs of no-depth pixels that go from the face
    to the edge of the outline, along an image row, are filled, and only as wide as a
    shadow can be: sensor baseline x focal length x |1/z_face - 1/z_next|, z_face from the
    face's plane and z_next from the first depth past the outline (a nearer box beside it
    casts its shadow onto the face; a face in front of a wall has its edge lost the same
    way; no depth there at all = something far away, so a wide shadow is possible). A
    wider strip is something else (a side seen edge-on, a gap), so that row is left alone.
    Also only where the camera ray meets the plane at a useful angle (a face seen nearly
    edge-on would be stretched).
    -> [(points (M, 3), pixels (M, 2)) per face]
    """
    h, w = mask.shape
    hole = mask & (depth_m <= 0)
    ray = lambda u, v: np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
    out = []
    for (n, d), pix in zip(planes, face_pix):
        us, vs = [], []
        for v in np.unique(pix[:, 1]):
            row = pix[pix[:, 1] == v, 0]
            for start, step in ((row.min() - 1, -1), (row.max() + 1, 1)):
                u = start
                while 0 <= u < w and abs(u - start) < max_px and hole[v, u]:
                    u += step
                if u == start or (0 <= u < w and mask[v, u]):     # no run, or it doesn't reach the outline's edge
                    continue
                cos_n = ray(start - step, v) @ n
                if abs(cos_n) < 1e-6 or -d / cos_n <= 0:
                    continue
                z_face = -d / cos_n                                # rays have z = 1, so t is the depth
                past = np.arange(u, u + step * SHADOW_LOOK_PX, step)
                past = depth_m[v, past[(past >= 0) & (past < w)]]
                past = past[past > 0][:5]
                inv_next = 1 / float(np.median(past)) if len(past) else 0.0
                # The box's own side, seen at a steep angle, has no depth either. It can only show
                # on the edge the camera is beyond; there the strip is filled only when something
                # nearer stands beside it (then it's that thing's shadow, and it hides the side).
                p0 = ray(start - step, v) * z_face
                p1 = ray(start, v) * (-d / (ray(start, v) @ n))
                side_visible = -p0 @ (p1 - p0) > 0
                nearer_beside = len(past) and 1 / inv_next < z_face - NEARER_M
                if side_visible and not nearer_beside:
                    continue
                widest = D455_BASELINE_M * intr.fx * abs(1 / z_face - inv_next) + SHADOW_SLACK_PX
                if abs(u - start) <= widest:
                    run = np.arange(start, u, step)
                    us.append(run)
                    vs.append(np.full(len(run), v))
        need = [np.full(len(r), MIN_RAY_COS) for r in us]
        if not us:
            out.append((np.zeros((0, 3)), np.zeros((0, 2), int)))
            continue
        u, v, need = np.concatenate(us), np.concatenate(vs), np.concatenate(need)
        r = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones(len(u))], axis=1)
        cos = np.abs(r @ n) / np.linalg.norm(r, axis=1)
        t = -d / (r @ n)
        ok = (cos >= need) & (t > 0)
        out.append((r[ok] * t[ok, None], np.stack([u[ok], v[ok]], axis=1)))
    return out


def level_plane(pts, up, thr):
    """(normal, d) of a level plane holding at least LEVEL_SHARE of the points (within thr of
    one height, the most common), or None."""
    h = pts @ up
    bins = np.arange(h.min(), h.max() + thr, thr / 2)
    if len(bins) < 2:
        return None
    counts, edges = np.histogram(h, bins=bins)
    k = int(np.argmax(counts))
    centre = (edges[k] + edges[k + 1]) / 2
    # Two guesses at the height, the one holding more points kept: the points right at that
    # peak, and all near it (depth on a top seen at an angle comes out slightly bent, and its
    # busiest height is then its near part only; but further out may be the front's top rows).
    guesses = [float(np.median(h[np.abs(h - centre) < band * thr])) for band in (1, 2)]
    height = max(guesses, key=lambda g: (np.abs(h - g) < thr).sum())
    if (np.abs(h - height) < thr).sum() < max(MIN_FACE_POINTS, LEVEL_SHARE * len(pts)):
        return None
    return up, -height


def box_faces(depth_m, mask, intr, up, max_faces=3, rng=None):
    """Split one box's depth points into up to `max_faces` flat faces.

    Planes are fitted on the shrunken outline (edge pixels can have wrong depth), then
    each face is measured on all of the outline's points that lie on its plane, so
    shrinking the outline doesn't make faces look smaller.
    """
    fit_pts, _ = deproject(depth_m, mask, intr)
    all_pts, all_pix = deproject(depth_m, mask, intr, erode_px=0)
    if len(fit_pts) < MIN_FACE_POINTS:
        return [], all_pts
    rng = rng or np.random.default_rng(0)
    if len(fit_pts) > MAX_POINTS:
        fit_pts = fit_pts[rng.choice(len(fit_pts), MAX_POINTS, replace=False)]
    thr = max(0.004, 0.01 * float(np.median(fit_pts[:, 2])))   # D455 noise grows with distance
    planes, rest = [], np.ones(len(fit_pts), bool)
    # A top first, if there is one: its points all lie at one height, a front's spread over many.
    # Left to the random search, a box seen from afar with a thin front (a flat box) sometimes
    # got one slanted plane across its top and front, classed as a side, and so no pick face.
    level = level_plane(fit_pts, up, thr)
    if level is not None:
        n, d = level
        planes.append((n, d))
        rest[np.abs(fit_pts @ n + d) < REMOVE_FRAC * thr] = False
    for _ in range(max_faces + 2):
        if rest.sum() < MIN_FACE_POINTS or len(planes) == max_faces:
            break
        fit = fit_plane(fit_pts[rest], thr, rng=rng)
        if fit is None or fit[2].sum() < MIN_FACE_POINTS:
            break
        n, d = fit[:2]
        # The points left in the band around a found plane can make a second, nearly identical
        # plane. Kept, it would split the face's points in two (a striped, too-small face), so
        # it is skipped; the two extra rounds keep it from crowding out a real face.
        c = fit_pts[rest][fit[2]].mean(axis=0)
        # A top lies flat: one fitted within LEVEL_DEG of up is set level, only its height fitted.
        # Depth on a top seen at an angle can come out slightly bent, and its fitted tilt then
        # swings several degrees from frame to frame, moving its size and centre with it.
        if abs(n @ up) > np.cos(np.radians(LEVEL_DEG)):
            on = np.abs(fit_pts @ n + d) < thr
            n = up if n @ up > 0 else -up
            d = -float(np.median(fit_pts[on] @ n))
        # Once a level top is found, the box's other faces meet it at a right angle: a plane
        # tilted halfway (across its front edge) is not a face of it.
        tilt = np.degrees(np.arccos(min(1.0, abs(n @ up))))            # 0 = level, 90 = upright
        slanted = level is not None and LEVEL_DEG < tilt < 90 - UPRIGHT_DEG
        if not slanted and not any(n @ m > np.cos(np.radians(TWIN_DEG)) and abs(m @ c + e) < thr for m, e in planes):
            planes.append((n, d))
        # Take away only the points close to this plane: a full-width band also takes the next
        # face's points along their shared edge, and a narrow face (a box's 2 cm end) is then
        # left too thin to fit. Every point still goes to its nearest plane below.
        rest[np.flatnonzero(rest)[np.abs(fit_pts[rest] @ n + d) < REMOVE_FRAC * thr]] = False
    # A thin box's front (a flat box 1-2 cm high) lies within the level top's band, so the
    # top takes it and no front is found. The pixels' own directions still show it.
    kinds = [classify(n if d > 0 else -n, up)[0] for n, d in planes]
    if "top" in kinds and "front" not in kinds:
        dn, dok, dpix = surface_dirs(depth_m, mask, intr)
        fwd = np.array([0.0, 0.0, -1.0]); fwd -= (fwd @ up) * up; fwd /= np.linalg.norm(fwd)
        facing = dok & (dn @ fwd > np.cos(np.radians(FACE_ANGLE_DEG)))
        if facing.sum() >= MIN_FACE_POINTS:
            on = np.zeros(mask.shape, bool); on[dpix[facing, 1], dpix[facing, 0]] = True
            sel = on[all_pix[:, 1], all_pix[:, 0]]
            fit = fit_plane(all_pts[sel], thr, rng=rng) if sel.sum() >= MIN_FACE_POINTS else None
            if fit is not None and fit[2].sum() >= MIN_FACE_POINTS and classify(fit[0] if fit[1] > 0 else -fit[0], up)[0] == "front":
                planes.append(fit[:2])
    if not planes:
        return [], all_pts
    # Each point goes to the plane it's closest to, so faces sharing an edge split it fairly.
    resid = np.abs(np.stack([all_pts @ n + d for n, d in planes]))
    nearest = resid.argmin(axis=0)
    close = resid.min(axis=0) < thr
    # Front and top split where their planes meet, the box's edge: a point whose ray meets the
    # front's plane under the top's plane is front, else top. The top's own fitted plane, not the
    # one set level: with a wrong up, the level one cuts the edge too low, into the front.
    kinds = [classify(n if d > 0 else -n, up)[0] for n, d in planes]
    if "front" in kinds and "top" in kinds:
        # The higher top and the nearer front: a lower top is the floor the outline spilled onto.
        tops = [k for k, kd in enumerate(kinds) if kd == "top"]
        fronts = [k for k, kd in enumerate(kinds) if kd == "front"]
        height = lambda k: -planes[k][1] if planes[k][0] @ up > 0 else planes[k][1]
        kt = max(tops, key=height)
        kf = min(fronts, key=lambda k: float(np.median(all_pts[close & (nearest == k), 2])) if (close & (nearest == k)).any() else 9.0)
        on_t = close & (nearest == kt)
        own = fit_plane(all_pts[on_t], thr, rng=rng) if on_t.sum() >= MIN_FACE_POINTS else None
        if own is not None and abs(own[0] @ up) > np.cos(np.radians(FACE_ANGLE_DEG)):
            nt, et = (own[0], own[1]) if own[0] @ up > 0 else (-own[0], -own[1])
            nf, ef = planes[kf]
            ray = np.stack([(all_pix[:, 0] - intr.cx) / intr.fx, (all_pix[:, 1] - intr.cy) / intr.fy,
                            np.ones(len(all_pix))], axis=1)
            hit = ray * (-ef / (ray @ nf))[:, None]
            edge_kind = np.where(hit @ nt + et < 0, kf, kt)
            # Each pixel's own direction has the last word: one facing the camera is never moved
            # to the top, one pointing up never to the front (in a stack of thin boxes the level
            # "top" can be made of the front's own points). And one pointing the way of a face
            # joins it with its depth up to 3x the threshold off (glossy, noisy tops).
            dn, dok, dpix = surface_dirs(depth_m, mask, intr)
            fwd = np.array([0.0, 0.0, -1.0]); fwd -= (fwd @ up) * up; fwd /= np.linalg.norm(fwd)
            look = np.full(mask.shape, -1, int)                       # -1 unknown, kf front, kt top
            look[dpix[dok & (dn @ fwd > np.cos(np.radians(FACE_ANGLE_DEG))), 1], dpix[dok & (dn @ fwd > np.cos(np.radians(FACE_ANGLE_DEG))), 0]] = kf
            look[dpix[dok & (dn @ up > np.cos(np.radians(FACE_ANGLE_DEG))), 1], dpix[dok & (dn @ up > np.cos(np.radians(FACE_ANGLE_DEG))), 0]] = kt
            says = look[all_pix[:, 1], all_pix[:, 0]]
            ft = close & np.isin(nearest, [kf, kt])
            move = ft & ((says == -1) | (says == edge_kind))
            nearest[move] = edge_kind[move]
            for k in (kf, kt):
                join = ~close & (says == k) & (resid[k] < 3 * thr)
                nearest[join] = k
                close = close | join
    found = []
    for k in range(len(planes)):
        on = np.flatnonzero(close & (nearest == k))
        if len(on) >= MIN_FACE_POINTS:
            found.append((k, on[largest_region(all_pix[on])]))
    holes = fill_holes(depth_m, mask, intr, [planes[k] for k, _ in found], [all_pix[on] for _, on in found])
    holes = [inner_holes(depth_m, mask, intr, planes[k], all_pix[on], h_pts, h_pix)
             for (k, on), (h_pts, h_pix) in zip(found, holes)]
    faces, inputs = [], {}
    for (k, on), (hole_pts, hole_pix) in zip(found, holes):
        args = (np.concatenate([all_pts[on], hole_pts]), np.concatenate([all_pix[on], hole_pix]), planes[k][0])
        faces.append(describe(*args, intr, up))
        inputs[id(faces[-1])] = args
    # A box can't show two faces pointing the same way. Two such planes far apart are the box
    # and the floor / wall the outline spilled onto: keep the outer one (the higher top, the
    # nearer front). Close together they're one face split by noise: keep the bigger part.
    kept = []
    for f in sorted(faces, key=lambda f: -len(f.pixels)):
        same = [k for k in kept if f.normal @ k.normal > np.cos(np.radians(SAME_FACE_DEG))]
        if not same:
            kept.append(f)
        elif same[0].normal @ (f.centroid - same[0].centroid) > SAME_PLANE_M:
            kept[kept.index(same[0])] = f      # f is on the camera side of the kept face's plane
    # Box faces are rectangles that meet along an edge: the face seen with fewer points (often
    # a top seen nearly edge-on, with little depth) takes its rectangle's direction from the
    # line it shares with the other.
    front = [f for f in kept if f.kind == "front"]
    top = [f for f in kept if f.kind == "top"]
    if front and top:
        f, t = front[0], top[0]
        edge = np.cross(f.normal, t.normal)
        if np.linalg.norm(edge) > 0.5:
            edge /= np.linalg.norm(edge)
            if len(f.pixels) < len(t.pixels):
                kept[kept.index(f)] = describe(*inputs[id(f)], intr, up, edge=edge)
            if len(t.pixels) <= len(f.pixels):
                kept[kept.index(t)] = describe(*inputs[id(t)], intr, up, edge=edge)
    return kept, all_pts


def analyse(depth_m, masks, intr, up=np.array([0.0, -1.0, 0.0]), top_faces=False):
    """Faces and pick face for every box outline. top_faces: pick_rules.py's top faces mode."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    boxes = []
    for m in masks:
        v, u = np.nonzero(m)
        faces, _ = box_faces(depth_m, m, intr, up)
        pickable = [f for f in faces if f.kind in ("front", "top")]
        # No front or top within FACE_ANGLE_DEG (a box turned away): the face turned most towards
        # the camera is its front, however far it is turned.
        if not pickable and faces:
            k = min(range(len(faces)), key=lambda k: faces[k].angle_deg)
            faces[k] = replace(faces[k], kind="front")
            pickable = [faces[k]]
        # When the box has a front, a top counts only if it is this box's top and can really be
        # seen: as wide as the front along the edge they share (else it's a sliver of the box
        # underneath, seen in a gap), and its front edge seen at FLAT_VIEW_DEG or more (else
        # it has little depth on it and is measured badly).
        fronts = [f for f in pickable if f.kind == "front"]
        seen = faces
        if fronts:
            front = max(fronts, key=lambda f: f.area_mm2)
            flat = [f for f in pickable if f.kind == "top" and edge_view_angle(depth_m, f, intr, up) < FLAT_VIEW_DEG]
            pickable = [f for f in pickable if f.kind == "front"
                        or (same_width(depth_m, front, f, intr, up, m) and f not in flat)]
            # Seen that flat, a real top is a row or two of pixels: most of what was called top
            # is the front's upper part (shiny, so its depth is noisy). Back to the front.
            for t in flat:
                nf = rejoin_front(front, t, depth_m, intr, up)
                faces = [nf if f is front else f for f in faces]
                pickable = [nf if f is front else f for f in pickable]
                front = nf
            pickable = [top_from_outline(f, front, m, depth_m, intr, up) if f.kind == "top" else f for f in pickable]
            # A top seen too flat (or too narrow) to keep is dropped, not just left unpicked:
            # seen edge-on it is a few noisy rows, and it can't show where it ends (find_hidden).
            kept = next((g for g in pickable if g.kind == "top"), None)
            seen = [next((g for g in pickable if g.kind == f.kind), f) if f.kind == "top" else f for f in faces]
            faces = [kept if f.kind == "top" else f for f in faces if f.kind != "top" or kept is not None]
        # How near the box is: from the faces the robot can grip. A side face can hold the
        # neighbour's edge pixels and make a box deep in a gap look like the nearest one.
        near = pickable or faces
        z = np.concatenate([depth_m[f.pixels[:, 1], f.pixels[:, 0]] for f in near]) if near else np.zeros(0)
        z = z[z > 0]                                                   # filled-in pixels have no depth
        pts = face_points(depth_m, seen, intr)               # heights from every face seen, dropped tops too
        h, x = pts @ up, pts[:, 0]
        boxes.append(Box(
            mask=m, bbox=(int(u.min()), int(v.min()), int(u.max()), int(v.max())) if len(u) else (0, 0, 0, 0),
            depth=float(np.percentile(z, 10)) if len(z) else 0.0,
            faces=faces, pick=max(pickable, key=lambda f: f.area_mm2) if pickable else None,
            low=float(np.percentile(h, 2)) if len(h) else 0.0, high=float(np.percentile(h, 98)) if len(h) else 0.0,
            left=float(np.percentile(x, 2)) if len(x) else 0.0, right=float(np.percentile(x, 98)) if len(x) else 0.0,
        ))
    for i, a in enumerate(boxes):
        a.blocked_by = [j for j, b in enumerate(boxes) if j != i and rests_on(b, a)]
    find_hidden(boxes, depth_m, intr, up)
    # The pick face (pick_rules.py), now that it is known which boxes rest on others.
    for i, b in enumerate(boxes):
        if b.pick is not None:
            b.pick, b.top_only = pick_face(b.faces, any(i in a.blocked_by for a in boxes), top_faces)
    return boxes


def find_hidden(boxes, depth_m, intr=None, up=None):
    """Boxes standing behind a nearer box that hides their lower part.

    A box behind another shows only what's above that box, so its lowest visible point lines
    up with that box's top: just what a box resting on it looks like (rests_on). Box x touches
    box y's bottom edge in the picture and x's front is clearly nearer than y's (more than
    DEFAULT_DEPTH_TOL). Only when x's top is seen *beside* y and ends there (reaches less
    than SUPPORT_M behind y's front) does y stand behind x, so x goes first. Beside y, because
    a box resting on x hides x's top behind its own front, so above y x's top always seems
    to end at y. Otherwise, including when x's top can't be seen there (a level camera sees
    tops edge-on, or x is no wider than y), y counts as resting on x and x waits: nothing is
    pulled out from under a box.
    """
    for i, y in enumerate(boxes):
        if y.pick is None or not y.depth:
            continue
        for j, x in enumerate(boxes):
            if j == i or not x.depth or x.depth >= y.depth - DEFAULT_DEPTH_TOL or not touches_below(y.mask, x.mask):
                continue
            z, pct = top_beside(boxes, i, j, depth_m, intr, up)
            if len(z) < MIN_FACE_POINTS:
                # x's top is all under y (or other boxes): a twin's top tells instead, a box of the
                # same kind in the same row (same front size, front as near, top as high), seen
                # beside y. Same kind, same depth from front to back.
                more = [top_beside(boxes, i, k, depth_m, intr, up) for k in twins(boxes, j)]
                z, pct = np.concatenate([z] + [m[0] for m in more]), min([pct] + [m[1] for m in more])
            if len(z) < MIN_FACE_POINTS or np.percentile(z, pct) > y.depth + SUPPORT_M:
                continue                                   # top unseen beside y, or it runs on: y rests on x
            y.hidden_by.append(j)
            if i in x.blocked_by:
                x.blocked_by.remove(i)                     # y doesn't rest on x, it stands behind it


def twins(boxes, j):
    """Boxes of the same kind as box j in the same row: front the same size (within SAME_LENGTH),
    as near (within DEFAULT_DEPTH_TOL), top as high (within SAME_PLANE_M)."""
    x = boxes[j]
    if x.pick is None or x.pick.kind != "front":
        return []
    return [k for k, t in enumerate(boxes) if k != j and t.pick is not None and t.pick.kind == "front"
            and all(abs(a - b) <= SAME_LENGTH * b for a, b in zip(t.pick.size_mm, x.pick.size_mm))
            and abs(t.depth - x.depth) < DEFAULT_DEPTH_TOL and abs(t.high - x.high) < SAME_PLANE_M]


def top_beside(boxes, i, j, depth_m, intr, up):
    """Depths of box j's top in the columns beside box i (and beside every box resting on j),
    where it could be seen running on past i's front. -> (z values in metres, the percentile of
    them that is the top's far end: 98 for a top found as a face, 90 for one read straight from
    the depth, whose outline's edge pixels catch what's behind at the same height)."""
    x, y = boxes[j], boxes[i]
    top = [f for f in x.faces if f.kind == "top"]
    pix = top[0].pixels if top else np.zeros((0, 2), int)
    pct = 98 if top else 90
    # (Only when that top is seen at FLAT_VIEW_DEG or more: flatter, the depth can't show
    # where it ends. The camera is at height 0.)
    if not top and intr is not None and np.degrees(np.arctan2(-x.high, x.depth)) >= FLAT_VIEW_DEG:
        # A top too thin in the picture to be found as a face (a few rows): read it
        # straight from the depth, as the points in x's outline at its top's height.
        v, u = np.nonzero(x.mask & (depth_m > 0))
        zz = depth_m[v, u]
        q = np.stack([(u - intr.cx) * zz / intr.fx, (v - intr.cy) * zz / intr.fy, zz], axis=1)
        lev = (np.abs(q @ (up / np.linalg.norm(up)) - x.high) < 0.006) & (zz >= x.depth)
        pix = np.stack([u[lev], v[lev]], axis=1)
    # Beside y, and beside every other box resting on x: a box standing on x hides x's top
    # behind its front, so there it always seems to end.
    on_x = np.logical_or.reduce([boxes[k].mask for k in {i, *x.blocked_by} if boxes[k].mask is not None])
    pix = pix[~on_x.any(axis=0)[pix[:, 0]]]
    if intr is not None:
        # Only columns where x's top, if it ran on past y's front, could be seen doing so.
        # A nearer box looks wider: its top's sides draw in with distance, so in its
        # outermost columns the top always seems to end short of a box set back on it.
        far = y.depth + SUPPORT_M
        u_lo, u_hi = intr.cx + x.left * intr.fx / far, intr.cx + x.right * intr.fx / far
        pix = pix[(pix[:, 0] > u_lo) & (pix[:, 0] < u_hi)]
    z = depth_m[pix[:, 1], pix[:, 0]]
    return z[z > 0], pct


def touches_below(upper, lower, gap_px=8, min_cols=5):
    """Does outline `lower` start within gap_px below `upper`'s bottom edge, in at least
    min_cols columns?"""
    cols = np.flatnonzero(upper.any(axis=0))
    if not len(cols):
        return False
    bottom = upper.shape[0] - 1 - np.argmax(upper[::-1, cols], axis=0)      # last row of each column
    rows = bottom[:, None] + np.arange(1, gap_px + 1)[None, :]
    rows = np.clip(rows, 0, upper.shape[0] - 1)
    return int(lower[rows, cols[:, None]].any(axis=1).sum()) >= min_cols


def view_angle(face, point=None):
    """Degrees between the camera's line of sight to the face's centre (or to `point` on it)
    and the face itself."""
    p = face.centroid if point is None else point
    r = p / np.linalg.norm(p)
    return float(np.degrees(np.arcsin(min(1.0, abs(r @ face.normal)))))


def edge_view_angle(depth_m, top, intr, up):
    """view_angle of a top at its front edge, the edge nearest the camera. Not at its centre:
    the far part of a top seen at a slant often has no depth, so the centre of what was found
    moves back and forth between frames, and with it the angle. The front edge stays put."""
    pts = face_points(depth_m, [top], intr)
    if len(pts) < MIN_FACE_POINTS:
        return view_angle(top)
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]                  # forward, level
    fwd /= np.linalg.norm(fwd)
    near = np.percentile(pts @ fwd, 2)
    return view_angle(top, top.centroid + fwd * (near - top.centroid @ fwd))


def top_from_outline(top, front, mask, depth_m, intr, up):
    """The top re-measured from the box's front face and outline, its depth used only for width.

    A top seen at a slant has little depth on it: the camera reads some of it, leaves holes,
    or reads the far edge's distance over the rest, so the fitted top is sometimes whole,
    sometimes only its front strip, sometimes only a strip at the back. The front face is
    measured well, and gives the top's front edge (where the front is) and, with the top's own
    points, its height. The top is level, so each image row above that edge lies at one distance
    along the box, and the outline's top row (the colour picture's edge, sharp) is the far edge.
    Far edge: the outline's, or the depth's when that reaches further than FAR_SMEAR_M past it
    (the outline stops short where a box on the top hides its back); depth up to FAR_SMEAR_M
    past the outline is smeared past the edge (a 100 mm top read as 109). The outline isn't used when it spills onto what's behind the box (its
    top row isn't at the top's height) or makes the box deeper than MAX_BOX_DEPTH_M."""
    pts, fpts = face_points(depth_m, [top], intr), face_points(depth_m, [front], intr)
    if len(pts) < MIN_FACE_POINTS or len(fpts) < MIN_FACE_POINTS:
        return top
    edge = np.cross(up, front.normal)
    edge /= np.linalg.norm(edge)
    back = np.cross(edge, up)                                      # level, away from the camera
    if back @ front.normal > 0:
        back = -back
    # The top's height: the higher of the front's top edge and the top's own points. Each reads
    # low when it misses the edge (a front found cut short; only a strip at the back of the
    # top found, a little below it), and at this slant 1 cm of height moves the far edge ~4 cm.
    height = max(float(np.percentile(fpts @ up, 98)), float(np.median(pts @ up)))
    near = float(np.median(fpts @ back))                           # the front's plane
    far = float(np.percentile(pts @ back, 98))
    if far - near > MAX_BOX_DEPTH_M:
        return top
    cols = np.unique(top.pixels[:, 0])
    lo, hi = np.percentile(cols, 20), np.percentile(cols, 80)      # away from the corners
    cols = cols[(cols >= lo) & (cols <= hi) & mask.any(axis=0)[cols]]
    rows = np.argmax(mask[:, cols], axis=0)                        # the outline's top row, per column
    rays = np.stack([(cols - intr.cx) / intr.fx, (rows - intr.cy) / intr.fy, np.ones(len(cols))], axis=1)
    along = rays @ up
    ok = along * height > 0                                        # rays that meet the top's plane
    if ok.sum() >= 5:
        far_outline = float(np.median((rays[ok] * (height / along[ok])[:, None]) @ back))
        # Where depth was read just inside the outline's top row, it must lie at the top's
        # height: the far edge (or its distance smeared over the slanted part). Else the
        # outline has spilled onto what's behind the box, the floor or a wall.
        inside = np.minimum(rows + 2, mask.shape[0] - 1)
        z = depth_m[inside, cols]
        seen = z > 0
        p = rays.copy()
        p[:, 1] = (inside - intr.cy) / intr.fy
        spilled = seen.sum() >= 5 and np.median(np.abs((p[seen] * z[seen, None]) @ up - height)) > SUPPORT_M
        if not spilled and far_outline - near <= MAX_BOX_DEPTH_M:
            # Depth a little past the outline is smeared past the far edge: the outline's edge
            # is sharp. Far past it, the outline stops short (a box on the top hides its back).
            far = far_outline if 0 < far - far_outline <= FAR_SMEAR_M else max(far, far_outline)
    depth = far - near
    if depth <= 0:
        return top
    # The width was measured well (the top's strips span it): of the fitted rectangle's two
    # sides, the one closest to the points' spread along the front edge. Unless the top is wider
    # with the end the plane fit or the depth shadow left out (top_points): then that, centred
    # on it.
    spread = np.percentile(pts @ edge, 98) - np.percentile(pts @ edge, 2)
    width = min(top.size_mm, key=lambda s: abs(s / 1000 - spread)) / 1000
    mid = top.centroid @ edge
    x = top_points(depth_m, top, mask, intr, up) @ edge
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo > width:
        width, mid = hi - lo, (lo + hi) / 2
    # Nor narrower than its front: they share that edge, and a patchy top (shiny, seen at a
    # slant) loses its ends while the front, seen square on, keeps them.
    # (The front's points, not its fitted rectangle: that overshoots a few mm on some boxes.)
    fx = fpts @ edge
    lo, hi = np.percentile(fx, 2), np.percentile(fx, 98)
    if hi - lo > width:
        width, mid = hi - lo, (lo + hi) / 2
    centre = edge * mid + back * (near + far) / 2 + up * height
    corners = [centre + a * edge * width / 2 + b * back * depth / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return replace(top, size_mm=(1000 * max(width, depth), 1000 * min(width, depth)), centroid=centre,
                   centroid_px=intr.project(centre), corners_px=np.array([intr.project(c) for c in corners]))


def same_width(depth_m, front, top, intr, up, mask=None):
    """Are a box's front and top equally wide, along the edge they share (level, in the front's
    plane)? Measured along that edge, so a box turned at an angle passes too. mask: the box's
    outline; the top then also spans its no-depth pixels (top_points)."""
    edge = np.cross(up, front.normal)
    edge /= np.linalg.norm(edge)
    width = []
    for f in (front, top):
        pts = top_points(depth_m, f, mask, intr, up) if f is top and mask is not None else face_points(depth_m, [f], intr)
        x = pts @ edge
        if len(x) < MIN_FACE_POINTS:
            return True                                            # too few points to tell: keep it
        width.append(np.percentile(x, 98) - np.percentile(x, 2))
    return abs(width[1] - width[0]) <= SAME_WIDTH * width[0]


def top_points(depth_m, top, mask, intr, up):
    """A top's 3D points, plus the outline's pixels in the top's rows that touch the top and lie
    at its height (within SAME_PLANE_M) or have no depth (put on its level plane). Depth on a top
    seen at a slant comes out slightly bent, so the plane fit can leave out one end, and the depth
    shadow can cover an end (a white top with dark prints loses more): its points alone then make
    it look narrower than the box. Another box's top seen in a gap gains nothing: the pixels
    beside it are at other heights, and scattered holes in an outline spilling onto the floor
    don't touch it."""
    pts = face_points(depth_m, [top], intr)
    if len(pts) < MIN_FACE_POINTS:
        return pts
    height = float(np.median(pts @ up))
    v0, v1 = top.pixels[:, 1].min(), top.pixels[:, 1].max()
    vs, us = np.nonzero(mask[v0:v1 + 1])
    vs = vs + v0
    rays = np.stack([(us - intr.cx) / intr.fx, (vs - intr.cy) / intr.fy, np.ones(len(us))], axis=1)
    z = depth_m[vs, us]
    p = rays * z[:, None]
    hole = z == 0
    near = ~hole & (np.abs(p @ up - height) <= SAME_PLANE_M)
    along = rays @ up
    hole &= along * height > 0                                     # rays that meet the top's plane
    p[hole] = rays[hole] * (height / along[hole])[:, None]
    region = np.zeros((v1 - v0 + 1, mask.shape[1]), np.uint8)
    region[vs[near | hole] - v0, us[near | hole]] = 1
    own = np.zeros_like(region, bool)
    own[top.pixels[:, 1] - v0, top.pixels[:, 0]] = True
    _, lab = cv2.connectedComponents(region | own.astype(np.uint8), connectivity=4)
    touching = np.unique(lab[own])
    keep = (near | hole) & np.isin(lab[vs - v0, us], touching[touching > 0])
    return np.concatenate([pts, p[keep]])


def rejoin_front(front, top, depth_m, intr, up, max_m=0.01):
    """front with the pixels of a flat-seen top that lie on its plane (within max_m), or have
    no depth, added and re-measured."""
    n, c = front.normal, front.centroid
    def pts_of(pix):
        z = depth_m[pix[:, 1], pix[:, 0]]
        ray = np.stack([(pix[:, 0] - intr.cx) / intr.fx, (pix[:, 1] - intr.cy) / intr.fy, np.ones(len(pix))], axis=1)
        on_plane = ray * ((n @ c) / (ray @ n))[:, None]               # where the ray meets the front's plane
        return np.where((z > 0)[:, None], ray * z[:, None], on_plane), z
    tp, tz = pts_of(top.pixels)
    near = (tz == 0) | (np.abs((tp - c) @ n) < max_m)
    if not near.any():
        return front
    pix = np.concatenate([front.pixels, top.pixels[near]])
    pts, _ = pts_of(pix)
    return describe(pts, pix, n, intr, up)


def face_points(depth_m, faces, intr):
    """3D points (camera coordinates) of the faces' pixels that have depth."""
    if not faces:
        return np.zeros((0, 3))
    pix = np.concatenate([f.pixels for f in faces])
    z = depth_m[pix[:, 1], pix[:, 0]]
    pix, z = pix[z > 0], z[z > 0]
    return np.stack([(pix[:, 0] - intr.cx) * z / intr.fx, (pix[:, 1] - intr.cy) * z / intr.fy, z], axis=1)


def rests_on(b, a):
    """Is box b standing or lying on box a? Its bottom is at a's top height, it is above a in
    the picture and overlaps it side to side, and it is not in front of a nor further back
    than a box can be deep. A box lying flat may stick out in front of a: then the back of its
    top (its front plus the top's short side) must reach SUPPORT_M behind a's front. Its bottom
    must also be above a's middle, and not more than IN_WAY_M below a's top: a box lying flat
    can be thinner than REST_TOL_M, and a box standing on the floor beside it is not on it."""
    if b.pick is None or not a.depth or not b.depth:
        return False
    overlap = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
    narrower = min(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0])
    above_middle = a.high - a.low < 0.01 or b.low > (a.low + a.high) / 2   # a seen only from above: no middle
    # Its bottom can't be well below a's top either: a box seen only from above has no middle,
    # and one thinner than REST_TOL_M beside a box standing on the floor would pass.
    top = next((f for f in b.faces if f.kind == "top"), None)
    sticks_out = top is not None and b.depth + top.size_mm[1] / 1000 > a.depth + SUPPORT_M
    return (abs(b.low - a.high) < REST_TOL_M and b.low > a.high - IN_WAY_M
            and b.high > a.high + REST_TOL_M / 2 and above_middle
            and (b.bbox[1] + b.bbox[3]) < (a.bbox[1] + a.bbox[3])
            and overlap > 0.3 * max(narrower, 1)
            and (a.depth - REST_TOL_M <= b.depth or sticks_out) and b.depth <= a.depth + MAX_BOX_DEPTH_M)


def choose_target(boxes, tol=None, depth_tol=DEFAULT_DEPTH_TOL, avoid=(), order="v1"):
    """Index of the target box, or None. docs/picking_rules.md:
    free boxes (nothing resting on it, nothing in front of it), then the highest, on a box or
    on the floor (top within HIGHEST_TOL_M of the highest), then the nearest (within NEAREST_TOL_M of the nearest one's grip point), then
    leftmost. A box whose pick face runs off the picture's edge counts too: its grip point is
    the middle of the part in view. Never a box that nothing holds up (held_up: the AI takes
    pieces of the carton's rim at the top of the picture for boxes, some lying on others). avoid: outlines (bool HxW) of boxes
    not to choose (the robot gave up on them), from the same camera pose; they still count as in
    the way of the boxes behind and under them. order: "v1" as above, or "v2": the leftmost
    free box, whatever its height or distance. (tol: no longer used, kept so old calls work.)"""
    # Only outlines with a front or top face count: every box resting in the basket has one
    # (docs/picking_rules.md), so one without is a side split off a box, or a sliver.
    # A box with another box resting on it, or standing in front of it, waits: that one first.
    real = [i for i, b in enumerate(boxes) if b.pick is not None]
    left_out = {i for i in real if boxes[i].mask is not None and any(
        (boxes[i].mask & m).sum() > SKIP_OVERLAP * min(boxes[i].mask.sum(), m.sum()) for m in avoid)}
    floor = min(boxes[i].low for i in real) if real else 0.0
    cut = {i for i in real if not held_up(boxes, i, floor)}
    # Nor the boxes a left-out one rests on, stands in front of or hides, even as a last resort;
    # nor the ones those rest on, all the way down: the boxes under a pile wait for all of it.
    held = set(left_out)
    while True:
        more = {i for i in real if i not in held
                and (any(j in held for j in boxes[i].blocked_by + boxes[i].hidden_by)
                     or any(in_front(boxes[j], boxes[i], depth_tol) for j in held))}
        if not more:
            break
        held |= more
    cands = [i for i in real if i not in held and i not in cut]
    free = [i for i in cands if not boxes[i].blocked_by and not boxes[i].hidden_by
            and not any(in_front(boxes[j], boxes[i], depth_tol) for j in real if j != i)]
    free = free or [i for i in cands if not boxes[i].blocked_by] or cands
    if not free:
        return None
    if order == "v2":
        return min(free, key=lambda i: boxes[i].bbox[0] + boxes[i].bbox[2])
    # The highest top first, on a box or on the floor: stacks are cleared from the top down, so
    # the boxes under them (and their neighbours, which a pull can drag) are taken only when
    # nothing stands on them, and a tall box standing on the floor goes before a low stack.
    # By height, not by how many boxes are under it: a pile of thin boxes isn't higher than
    # one box on tall ones.
    top = max(boxes[i].high for i in free)
    free = [i for i in free if boxes[i].high >= top - HIGHEST_TOL_M]
    # Then the nearest: the arm reaches less deep among the boxes, and boxes behind come free.
    # Measured at the grip point (the pick face's centre), so a leaning box's near edge
    # doesn't set the distance for the rest.
    grip = {i: float(boxes[i].pick.centroid[2]) for i in free}
    nearest = min(grip.values())
    cand = [i for i in free if grip[i] <= nearest + NEAREST_TOL_M]
    return min(cand, key=lambda i: boxes[i].bbox[0] + boxes[i].bbox[2])



def held_up(boxes, i, floor, seen=frozenset()):
    """Is box i held up: its bottom down at the floor (the lowest box bottom in view, within
    REST_TOL_M), or on a box that is held up itself? On: resting on it (rests_on), or just its
    bottom at that box's top height (within REST_TOL_M) with the two overlapping side to side,
    whatever their depths (a box can stand a few cm in front of the pile it stands on). Only
    boxes with a pick face hold others up. A box seen only from above shows its top, not its
    bottom: it gets the benefit of the doubt."""
    b = boxes[i]
    if b.low <= floor + REST_TOL_M or (b.faces and all(f.kind == "top" for f in b.faces)):
        return True
    for j, a in enumerate(boxes):
        if j == i or j in seen or a.pick is None:
            continue
        overlap = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
        under = i in a.blocked_by or (abs(b.low - a.high) < REST_TOL_M and a.bbox[3] > b.bbox[3]
                                      and overlap > 0.3 * min(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0]))
        if under and held_up(boxes, j, floor, seen | {i}):
            return True
    return False


def at_edge(pixels, shape):
    """Do these (u, v) pixels reach the picture's edge (within EDGE_PX)? shape: (height, width)."""
    u, v = pixels[:, 0], pixels[:, 1]
    return bool(len(u) and (u.min() <= EDGE_PX or v.min() <= EDGE_PX
                            or u.max() >= shape[1] - 1 - EDGE_PX or v.max() >= shape[0] - 1 - EDGE_PX))


def split_doubles(boxes):
    """For each box, its outline, or the two halves of it when its front is two boxes pressed
    together: the AI draws one outline around two boxes standing (or lying) flush, with their
    fronts in one plane, so depth can't part them either. Told by size: a front as long as other
    fronts in view (SAME_LENGTH) and twice as thick as they are (DOUBLE_TOL), with at least
    DOUBLE_REFS of them to compare with. Split along the middle of the front's long side.
    -> [[mask] or [mask, mask], ...], one entry per box."""
    out = []
    for b in boxes:
        f = b.pick
        halves = [b.mask]
        if f is not None and f.kind == "front" and b.mask is not None:
            long_, short = f.size_mm
            refs = [o.pick.size_mm[1] for o in boxes if o is not b and o.pick is not None and o.pick.kind == "front"
                    and abs(o.pick.size_mm[0] - long_) <= SAME_LENGTH * long_]
            if len(refs) >= DOUBLE_REFS and abs(short - 2 * np.median(refs)) <= DOUBLE_TOL * 2 * np.median(refs):
                c = np.asarray(f.corners_px, float)                # a rectangle, corners in order
                if np.linalg.norm(c[1] - c[0]) < np.linalg.norm(c[2] - c[1]):
                    c = np.roll(c, -1, axis=0)                    # c0-c1 along the long side
                m1, m2 = (c[1] + c[2]) / 2, (c[3] + c[0]) / 2     # middles of the short sides
                v, u = np.nonzero(b.mask)
                side = (m2[0] - m1[0]) * (v - m1[1]) - (m2[1] - m1[1]) * (u - m1[0]) > 0
                a = np.zeros_like(b.mask)
                a[v[side], u[side]] = True
                halves = [a, b.mask & ~a]
        out.append(halves)
    return out


def in_front(b, a, depth_tol=DEFAULT_DEPTH_TOL):
    """Is box b in the way of box a? Nearer by more than depth_tol, sticking up in front of a
    (b's top more than IN_WAY_M above a's bottom, and b not wholly above a), and across from
    it side to side, with CLEAR_M to spare (a gripper can't use a narrow gap).

    Only b's top height counts, not its bottom: a box seen only from above (lying flat) shows
    just its top, so its lowest visible point is its top, not the floor it stands on. A stack
    below a box lying on top of it reaches only up to that box's bottom: not in its way."""
    if not a.depth or not b.depth or b.depth >= a.depth - depth_tol:
        return False
    sticks_up = min(a.high, b.high) - a.low > IN_WAY_M and b.low < a.high
    across = b.left - CLEAR_M < a.right and a.left < b.right + CLEAR_M
    return sticks_up and across


def up_from_pitch(pitch_deg):
    """Up direction in camera coordinates for a camera tilted down by pitch_deg (no roll)."""
    p = np.radians(pitch_deg)
    return np.array([0.0, -np.cos(p), -np.sin(p)])


def up_from_accel(accel):
    """Up direction in camera coordinates from a D455 accelerometer reading (m/s^2) at rest.

    At rest the accelerometer measures the reaction to gravity, so it points up; the
    D455 IMU axes match the depth camera's (x right, y down, z forward).
    """
    a = np.asarray(accel, float)
    return a / np.linalg.norm(a)
