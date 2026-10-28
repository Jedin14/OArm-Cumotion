"""Checks for faces.py on synthetic depth images with boxes of known size and pose.

Each scene is ray-cast into a 1280x720 depth image as seen by a D455-like camera
tilted down, optionally with noise and holes, and the pipeline's faces, pick face,
size and centroid are compared with the truth.

Run: python test_faces.py
"""
import numpy as np

from faces import Box, Intrinsics, analyse, choose_target, rests_on, split_doubles, up_from_pitch

W, H = 1280, 720
INTR = Intrinsics(640.0, 640.0, 640.0, 360.0)
PITCH = 25          # camera tilted down, degrees
CAM_HEIGHT = 0.30   # camera above the floor, metres


def cam_axes(pitch_deg=PITCH):
    """World (X right, Y up, Z forward) unit vectors of the camera's x, y, z axes."""
    p = np.radians(pitch_deg)
    return np.array([1.0, 0, 0]), np.array([0, -np.cos(p), -np.sin(p)]), np.array([0, -np.sin(p), np.cos(p)])


def to_cam(p_world):
    x, y, z = cam_axes()
    q = np.asarray(p_world, float) - [0, CAM_HEIGHT, 0]
    return np.array([q @ x, q @ y, q @ z])


def box_world(x, z, size, yaw_deg=0.0):
    """A box standing on the floor. size = (width along X, height, depth along Z) before yaw."""
    a = np.radians(yaw_deg)
    axes = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])   # rows: box x, y, z
    return {"centre": np.array([x, size[1] / 2, z]), "axes": axes, "half": np.array(size) / 2}


def render(boxes, noise_mm=0.0, holes=0.0, seed=0, floor=False, pitch=PITCH):
    """Depth image (metres) and one visible-pixel mask per box (floor pixels: in no mask)."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy, np.ones_like(u, float)], -1)
    cx, cy, cz = cam_axes(pitch)
    to_world = np.stack([cx, cy, cz], axis=1)          # camera vector -> world vector
    origin = np.array([0, CAM_HEIGHT, 0])
    depth = np.full((H, W), np.inf)
    owner = np.full((H, W), -1)
    for i, b in enumerate(boxes):
        o = b["axes"] @ (origin - b["centre"])
        d = (rays @ to_world.T) @ b["axes"].T          # ray directions in the box frame (z-scaled)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-b["half"] - o) / d, (b["half"] - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < depth)
        depth[hit], owner[hit] = tmin[hit], i          # t is the camera z, because rays have z = 1
    if floor:                                          # the plane Y = 0 under the boxes
        dy = (rays @ to_world.T)[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dy < 0, -CAM_HEIGHT / dy, np.inf)
        hit = t < depth
        depth[hit] = t[hit]
    rng = np.random.default_rng(seed)
    if noise_mm:
        depth = depth + rng.normal(0, noise_mm / 1000, depth.shape)
    depth[~np.isfinite(depth)] = 0
    if holes:
        depth[rng.random(depth.shape) < holes] = 0
    return depth, [owner == i for i in range(len(boxes))]


def face_centre_world(b, axis, sign):
    return b["centre"] + sign * b["half"][axis] * b["axes"][axis]


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def check(name, ok, detail):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def pick_checks(box, b, kind, size_mm, centre_world, tol_mm=5.0):
    p = box.pick
    if p is None:
        return check("pick face", False, "none found")
    err = np.linalg.norm(p.centroid - to_cam(centre_world)) * 1000
    ok = check("pick face", p.kind == kind, f"{p.kind} (expected {kind}), faces {[f.kind for f in box.faces]}")
    ok &= check("size", np.allclose(p.size_mm, size_mm, atol=tol_mm), f"{p.size_mm} mm (expected {size_mm})")
    ok &= check("centroid", err <= tol_mm, f"{err:.1f} mm from the true face centre")
    return ok


@case
def upright_box(noise=0.0, holes=0.0, tol=5.0):
    """Big face (100 x 80) at the camera, 40 deep: front wins over the 100 x 40 top."""
    b = box_world(0.0, 0.45, (0.10, 0.08, 0.04))
    depth, masks = render([b], noise, holes)
    box = analyse(depth, masks, INTR, up_from_pitch(PITCH))[0]
    return pick_checks(box, b, "front", (100, 80), face_centre_world(b, 2, -1), tol)


@case
def left_edge_shadow(noise=0.0, holes=0.0, tol=5.0):
    """As upright_box, on the floor, the left 25 px of the box without depth (the D455's shadow): full size anyway."""
    b = box_world(0.0, 0.45, (0.10, 0.08, 0.04))
    depth, masks = render([b], noise, holes, floor=True)
    for v in np.flatnonzero(masks[0].any(axis=1)):
        u0 = np.flatnonzero(masks[0][v])[0]
        depth[v, u0:u0 + 25] = 0
    box = analyse(depth, masks, INTR, up_from_pitch(PITCH))[0]
    return pick_checks(box, b, "front", (100, 80), face_centre_world(b, 2, -1), tol)


@case
def no_depth_strip_too_wide_for_a_shadow(noise=0.0, holes=0.0, tol=5.0):
    """Upright box, wall 3 cm behind it (8 cm behind its front): a shadow there is ~21 px, so a 30 px strip is not filled."""
    b = box_world(0.0, 0.45, (0.10, 0.08, 0.04))
    wall = box_world(0.0, 0.45 + 0.02 + 0.03 + 0.01, (0.40, 0.30, 0.02))
    depth, masks = render([b, wall], noise, holes, floor=True)
    for v in np.flatnonzero(masks[0].any(axis=1)):
        u0 = np.flatnonzero(masks[0][v])[0]
        depth[v, u0:u0 + 30] = 0
    box = analyse(depth, masks[:1], INTR, up_from_pitch(PITCH))[0]
    full = (100, 80)
    p = box.pick
    ok = check("pick face", p is not None and p.kind == "front", f"{p and p.kind}")
    return ok and check("not stretched", p.size_mm[0] < full[0] - 10,
                        f"{p.size_mm} mm: the strip is left out (full face {full})")


@case
def box_resting_on_a_box(noise=0.0, holes=0.0, tol=5.0):
    """A small box stands on the back of a flat box: the flat box is nearer, but the small one goes first."""
    flat = box_world(0.0, 0.45, (0.17, 0.03, 0.10))
    small = box_world(0.0, 0.45 + 0.03, (0.06, 0.06, 0.03))
    small["centre"] = small["centre"] + [0, 0.03, 0]          # on top of the flat box
    depth, masks = render([flat, small], noise, holes, floor=True)
    boxes = analyse(depth, masks, INTR, up_from_pitch(PITCH))
    ok = check("flat box", boxes[0].blocked_by == [1], f"blocked by {boxes[0].blocked_by} (expected [1])")
    return ok & check("target", choose_target(boxes) == 1, f"box {choose_target(boxes)} (expected 1, the box on top)")


@case
def box_behind_a_box(noise=0.0, holes=0.0, tol=5.0):
    """A tall box stands right behind a wider, short one, whose top shows beside it and ends there: the short one in
    front goes first (like frame 5280)."""
    short = box_world(0.0, 0.45, (0.16, 0.04, 0.04))
    tall = box_world(0.0, 0.45 + 0.04 + 0.005, (0.08, 0.10, 0.04))
    depth, masks = render([short, tall], noise, holes, floor=True)
    boxes = analyse(depth, masks, INTR, up_from_pitch(PITCH))
    ok = check("tall box", boxes[1].hidden_by == [0], f"hidden by {boxes[1].hidden_by} (expected [0])")
    ok &= check("short box", boxes[0].blocked_by == [], f"blocked by {boxes[0].blocked_by} (expected none)")
    return ok & check("target", choose_target(boxes) == 0, f"box {choose_target(boxes)} (expected 0, the box in front)")


@case
def box_lying_across_two(noise=0.0, holes=0.0, tol=5.0):
    """A box lies across two upright boxes, set back on their tops: it goes first, they wait."""
    left, right = box_world(-0.045, 0.45, (0.04, 0.10, 0.10)), box_world(0.045, 0.45, (0.04, 0.10, 0.10))
    across = box_world(0.0, 0.45 + 0.03, (0.17, 0.03, 0.06))
    across["centre"] = across["centre"] + [0, 0.10, 0]           # on top of the two
    depth, masks = render([left, right, across], noise, holes, floor=True)
    boxes = analyse(depth, masks, INTR, up_from_pitch(PITCH))
    ok = check("upright boxes", boxes[0].blocked_by == [2] and boxes[1].blocked_by == [2],
               f"blocked by {boxes[0].blocked_by}, {boxes[1].blocked_by} (expected [2], [2])")
    return ok & check("target", choose_target(boxes) == 2, f"box {choose_target(boxes)} (expected 2, the box across)")


@case
def row_on_row_one_sticking_out(noise=0.0, holes=0.0, tol=5.0):
    """Level camera; a row of 3 on a row of 3, the middle bottom box sticking out 2 cm: the box on it still rests on it."""
    size = (0.04, 0.10, 0.10)
    bottom = [box_world(x, 0.45 - (0.02 if x == 0 else 0), size) for x in (-0.045, 0.0, 0.045)]
    top = [box_world(x, 0.45, size) for x in (-0.045, 0.0, 0.045)]
    for b in top:
        b["centre"] = b["centre"] + [0, 0.10, 0]                 # standing on the bottom row
    depth, masks = render(bottom + top, noise, holes, floor=True, pitch=0)
    boxes = analyse(depth, masks, INTR, up_from_pitch(0))
    t = choose_target(boxes)
    ok = check("sticking-out box", 4 in boxes[1].blocked_by, f"blocked by {boxes[1].blocked_by} (expected to include 4)")
    return ok & check("target", t in (3, 4, 5), f"box {t} (expected a top-row box: 3, 4 or 5)")


@case
def level_camera_box_behind_unsure(noise=0.0, holes=0.0, tol=5.0):
    """Level camera, a tall box right behind a short one: the short box's top can't be seen, so 'behind' can't be
    told from 'on top'. The tall box counts as resting on it: the short one waits, nothing is pulled from under a box."""
    short = box_world(0.0, 0.45, (0.10, 0.04, 0.04))
    tall = box_world(0.0, 0.45 + 0.04 + 0.005, (0.10, 0.10, 0.04))
    for b in (short, tall):                                       # on a shelf: the short box's top 1 cm below the camera
        b["centre"] = b["centre"] + [0, CAM_HEIGHT - 0.05, 0]
    depth, masks = render([short, tall], noise, holes, pitch=0)
    boxes = analyse(depth, masks, INTR, up_from_pitch(0))
    ok = check("short box", boxes[0].blocked_by == [1], f"blocked by {boxes[0].blocked_by} (expected [1])")
    return ok & check("target", choose_target(boxes) == 1, f"box {choose_target(boxes)} (expected 1, the tall box)")


@case
def flat_box(noise=0.0, holes=0.0, tol=5.0):
    """Lying flat: top (100 x 80) wins over the 100 x 40 front."""
    b = box_world(0.0, 0.45, (0.10, 0.04, 0.08))
    depth, masks = render([b], noise, holes)
    box = analyse(depth, masks, INTR, up_from_pitch(PITCH))[0]
    return pick_checks(box, b, "top", (100, 80), face_centre_world(b, 1, +1), tol)


@case
def turned_box_top_wins(noise=0.0, holes=0.0, tol=5.0):
    """Narrow end (40 x 80) at the camera, top 100 x 40: top is bigger."""
    b = box_world(0.0, 0.45, (0.04, 0.08, 0.10))
    depth, masks = render([b], noise, holes)
    box = analyse(depth, masks, INTR, up_from_pitch(PITCH))[0]
    return pick_checks(box, b, "top", (100, 40), face_centre_world(b, 1, +1), tol)


@case
def turned_box_end_wins(noise=0.0, holes=0.0, tol=5.0):
    """Tall narrow end (40 x 140) at the camera, top 60 x 40: the end is bigger."""
    b = box_world(0.0, 0.45, (0.04, 0.14, 0.06))
    depth, masks = render([b], noise, holes)
    box = analyse(depth, masks, INTR, up_from_pitch(PITCH))[0]
    return pick_checks(box, b, "front", (140, 40), face_centre_world(b, 2, -1), tol)


@case
def box_at_45_deg(noise=0.0, holes=0.0, tol=5.0):
    """Turned 45 deg: both visible sides are sideways, so only the top can be picked."""
    b = box_world(0.0, 0.45, (0.10, 0.08, 0.06), yaw_deg=45)
    depth, masks = render([b], noise, holes)
    box = analyse(depth, masks, INTR, up_from_pitch(PITCH))[0]
    ok = check("sides", sum(f.kind == "side" for f in box.faces) == 2, f"faces {[f.kind for f in box.faces]}")
    return ok & pick_checks(box, b, "top", (100, 60), face_centre_world(b, 1, +1), tol)


@case
def outline_spills_onto_floor(noise=0.0, holes=0.0, tol=5.0):
    """Flat box on the floor, outline spilling 40 px onto the floor around it: the top, not the floor."""
    b = box_world(0.0, 0.45, (0.10, 0.04, 0.08))
    depth, masks = render([b], noise, holes, floor=True)
    import cv2
    spill = cv2.dilate(masks[0].astype(np.uint8), np.ones((81, 81), np.uint8)).astype(bool)
    box = analyse(depth, [spill], INTR, up_from_pitch(PITCH))[0]
    ok = pick_checks(box, b, "top", (100, 80), face_centre_world(b, 1, +1), tol)
    near = np.percentile(depth[masks[0] & (depth > 0)], 10)
    return ok & check("box depth", abs(box.depth - near) < 0.005, f"{box.depth:.3f} m (box's own {near:.3f} m)")


@case
def target_selection(noise=0.0, holes=0.0, tol=5.0):
    """Two boxes side by side at the front, one 6 cm behind and higher: the front-left one is the target."""
    size = (0.10, 0.08, 0.04)
    front_left, front_right = box_world(-0.12, 0.45, size), box_world(0.12, 0.45, size)
    behind = box_world(0.0, 0.51, (0.10, 0.12, 0.04))
    depth, masks = render([front_right, behind, front_left], noise, holes)
    boxes = analyse(depth, masks, INTR, up_from_pitch(PITCH))
    t = choose_target(boxes)
    return check("target", t == 2, f"box {t} (expected 2, the front-left box)")


@case
def highest_stack_top_first(noise=0.0, holes=0.0, tol=5.0):
    """Level camera; left a stack of 3 (the upper two set back 4 cm), right a stack of 2: the top of the
    higher stack goes first, though the right stack's top is 4 cm nearer (recording test2_d, frame 310)."""
    size = (0.16, 0.09, 0.10)
    stacks = [box_world(-0.10, 0.50, size), box_world(-0.10, 0.54, size), box_world(-0.10, 0.54, size),
              box_world(0.10, 0.50, size), box_world(0.10, 0.50, size)]
    for b, level in zip(stacks, (0, 1, 2, 0, 1)):                # on a shelf, the camera level with the 2nd row
        b["centre"] = b["centre"] + [0, 0.09 * level + CAM_HEIGHT - 0.13, 0]
    depth, masks = render(stacks, noise, holes, pitch=0)
    boxes = analyse(depth, masks, INTR, up_from_pitch(0))
    t = choose_target(boxes)
    ok = check("stacks", [b.blocked_by for b in boxes] == [[1], [2], [], [4], []],
               f"blocked by {[b.blocked_by for b in boxes]} (expected [[1], [2], [], [4], []])")
    return ok & check("target", t == 2, f"box {t} (expected 2, the top of the left stack)")


@case
def standing_box_beside_a_thin_box_seen_from_above(noise=0.0, holes=0.0, tol=5.0):
    """A box standing on the floor beside a 2.9 cm flat box seen only from above (no front, so no middle to compare
    with), its side overlapping the flat box in the picture: not resting on it, as its bottom is below the flat
    box's top. A box whose bottom is at that top still is. (Numbers from a live frame, pick loop, 26 Oct.)"""
    thin = Box(mask=None, bbox=(381, 690, 652, 760), depth=0.410, low=-0.201, high=-0.201)
    standing = Box(mask=None, bbox=(316, 520, 426, 775), depth=0.403, pick=object(), low=-0.230, high=-0.062)
    on_it = Box(mask=None, bbox=(420, 560, 600, 700), depth=0.415, pick=object(), low=-0.198, high=-0.120)
    ok = check("standing beside", not rests_on(standing, thin), "counted as resting on the thin box")
    return ok & check("lying on it", rests_on(on_it, thin), "not counted as resting on the thin box")


@case
def pile_one_thin_box_higher(noise=0.0, holes=0.0, tol=5.0):
    """Two piles of thin boxes lying flat (2.7 cm each), the right one a box higher: its top goes first, though the
    left pile's top is only 2.7 cm lower (live pick loop, 26 Oct, pick 19)."""
    size = (0.17, 0.027, 0.10)
    piles = [box_world(-0.10, 0.50, size) for _ in range(3)] + [box_world(0.10, 0.50, size) for _ in range(4)]
    for b, k in zip(piles, (0, 1, 2, 0, 1, 2, 3)):
        b["centre"] = b["centre"] + [0, 0.027 * k, 0]
    depth, masks = render(piles, noise, holes, floor=True, pitch=0)
    boxes = analyse(depth, masks, INTR, up_from_pitch(0))
    t = choose_target(boxes)
    return check("target", t == 6, f"box {t} (expected 6, the top of the higher pile)")


@case
def two_boxes_pressed_together(noise=0.0, holes=0.0, tol=5.0):
    """Two boxes standing flush, narrow ends at the camera in one plane, drawn as one outline (as the AI does), among
    single boxes: split in two, and the left one is the target, 27 mm wide (live pick loop, 26 Oct, picks 39 and 44)."""
    size = (0.027, 0.17, 0.10)
    xs = (-0.15, -0.123, -0.05, 0.0, 0.05, 0.10)                  # the first two touch
    row = [box_world(x, 0.50, size) for x in xs]
    for b in row:                                                 # the floor 23 cm below the camera
        b["centre"] = b["centre"] + [0, CAM_HEIGHT - 0.23, 0]
    depth, masks = render(row, noise, holes, pitch=0)
    masks = [masks[0] | masks[1]] + masks[2:]
    up = up_from_pitch(0)
    boxes = analyse(depth, masks, INTR, up)
    ok = check("merged", abs(boxes[0].pick.size_mm[1] - 54) < 8, f"front {boxes[0].pick.size_mm} (expected about 170 x 54)")
    halves = split_doubles(boxes)
    ok &= check("split", [len(h) for h in halves] == [2, 1, 1, 1, 1], f"{[len(h) for h in halves]} (expected [2, 1, 1, 1, 1])")
    boxes = analyse(depth, [m for h in halves for m in h], INTR, up)
    t = choose_target(boxes)
    left = row[0]["centre"][0]
    ok &= check("target", t is not None and abs(boxes[t].pick.size_mm[1] - 27) < 6
                and abs(float(boxes[t].pick.centroid[0]) - left) < 0.006,
                f"box {t}: front {boxes[t].pick.size_mm if t is not None else None}, "
                f"x {float(boxes[t].pick.centroid[0]) if t is not None else 0:.3f} (expected 27 mm wide, at x {left:.3f})")
    return ok


@case
def cut_by_the_picture_edge(noise=0.0, holes=0.0, tol=5.0):
    """A box cut off by the top of the picture (like pieces of the carton's rim the AI calls boxes) is never the
    target, even when it is the nearest: the box in full view is."""
    near_top = box_world(0.0, 0.40, (0.12, 0.05, 0.04))
    near_top["centre"] = near_top["centre"] + [0, CAM_HEIGHT + 0.20, 0]  # high up: cut by the top edge
    box = box_world(0.05, 0.50, (0.10, 0.08, 0.04))
    box["centre"] = box["centre"] + [0, CAM_HEIGHT - 0.23, 0]      # on a shelf, in full view
    depth, masks = render([near_top, box], noise, holes, pitch=0)
    boxes = analyse(depth, masks, INTR, up_from_pitch(0))
    ok = check("cut off", boxes[0].bbox[1] == 0, f"top box's outline starts at row {boxes[0].bbox[1]} (expected 0)")
    t = choose_target(boxes)
    return ok & check("target", t == 1, f"box {t} (expected 1, the box in full view)")


@case
def nothing_holds_it_up(noise=0.0, holes=0.0, tol=5.0):
    """Two "boxes", one lying on the other, floating 25 cm above the camera with nothing under them (as pieces of the
    carton's rim the AI calls boxes), nearer than the real box on the shelf below: the real box is the target."""
    rim = [box_world(-0.05, 0.42, (0.12, 0.02, 0.03)), box_world(-0.05, 0.42, (0.12, 0.02, 0.03))]
    for b, k in zip(rim, (0, 1)):
        b["centre"] = b["centre"] + [0, CAM_HEIGHT + 0.12 + 0.02 * k, 0]
    box = box_world(0.05, 0.50, (0.10, 0.08, 0.04))
    box["centre"] = box["centre"] + [0, CAM_HEIGHT - 0.23, 0]      # on a shelf, the floor 23 cm below the camera
    depth, masks = render(rim + [box], noise, holes, pitch=0)
    boxes = analyse(depth, masks, INTR, up_from_pitch(0))
    ok = check("on it", 1 in boxes[0].blocked_by, f"lower rim piece holds {boxes[0].blocked_by} (expected [1])")
    t = choose_target(boxes)
    return ok & check("target", t == 2, f"box {t} (expected 2, the box on the shelf)")


@case
def skipped_box(noise=0.0, holes=0.0, tol=5.0):
    """The robot gave up on a box (avoid): the next box is chosen, but never one the skipped box stands in front of."""
    size = (0.10, 0.08, 0.04)
    front_left, front_right = box_world(-0.12, 0.45, size), box_world(0.12, 0.45, size)
    behind = box_world(0.0, 0.51, (0.10, 0.12, 0.04))
    depth, masks = render([front_right, behind, front_left], noise, holes)
    boxes = analyse(depth, masks, INTR, up_from_pitch(PITCH))
    t = choose_target(boxes, avoid=[boxes[2].mask])
    ok = check("next box", t == 0, f"box {t} (expected 0, the front-right box)")
    short = box_world(0.0, 0.45, (0.16, 0.04, 0.04))
    tall = box_world(0.0, 0.45 + 0.04 + 0.005, (0.08, 0.10, 0.04))
    depth, masks = render([short, tall], noise, holes, floor=True)
    boxes = analyse(depth, masks, INTR, up_from_pitch(PITCH))
    t = choose_target(boxes, avoid=[boxes[0].mask])
    return ok & check("box behind it", t is None, f"box {t} (expected none: the box behind the skipped one is blocked)")


@case
def pile_of_thin_boxes_vs_box_on_tall_boxes(noise=0.0, holes=0.0, tol=5.0):
    """Right a pile of 4 thin boxes lying flat; left a flat box lying across two tall standing boxes: the flat box on
    the left goes first. Its top is higher, though fewer boxes are under it (recording test2.1, frame 345)."""
    pile = [box_world(0.12, 0.50, (0.17, 0.026, 0.10)) for _ in range(4)]
    for k, b in enumerate(pile):
        b["centre"] = b["centre"] + [0, 0.026 * k, 0]
    standing = [box_world(x, 0.50, (0.03, 0.17, 0.10)) for x in (-0.16, -0.04)]
    across = box_world(-0.10, 0.50, (0.17, 0.026, 0.10))
    across["centre"] = across["centre"] + [0, 0.17, 0]
    depth, masks = render(pile + standing + [across], noise, holes, floor=True, pitch=0)
    boxes = analyse(depth, masks, INTR, up_from_pitch(0))
    t = choose_target(boxes)
    return check("target", t == 6, f"box {t} (expected 6, the flat box on the standing boxes)")


@case
def standing_box_in_front_of_a_thin_box(noise=0.0, holes=0.0, tol=5.0):
    """A thin box lying flat (2.6 cm, less than REST_TOL_M) and a tall box standing on the floor just in front of
    its end: the tall box is not resting on the thin box (its bottom is down at the thin box's bottom)."""
    thin = box_world(0.0, 0.52, (0.17, 0.026, 0.10))
    tall = box_world(0.06, 0.465, (0.03, 0.17, 0.008))            # a thin card, so it fits in front of the box's end
    for b in (thin, tall):                                         # the floor 22.5 cm below the camera, as in frame 345
        b["centre"] = b["centre"] + [0, CAM_HEIGHT - 0.225, 0]
    depth, masks = render([thin, tall], noise, holes, pitch=0)
    boxes = analyse(depth, masks, INTR, up_from_pitch(0))
    return check("thin box", boxes[0].blocked_by == [], f"blocked by {boxes[0].blocked_by} (expected [])")


def main():
    results = {}
    for label, kw in [("clean", {}), ("noise 1.5 mm + 10% holes", {"noise": 1.5, "holes": 0.10, "tol": 8.0})]:
        print(f"\n== {label} ==")
        for fn in CASES:
            print(f"{fn.__name__}: {fn.__doc__}")
            results[(label, fn.__name__)] = fn(**kw)
    failed = [k for k, ok in results.items() if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed" + (f"; failed: {failed}" if failed else ""))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
