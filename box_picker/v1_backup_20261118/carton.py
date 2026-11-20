"""v2: pick only from the carton in front.

SAM 3.1 is asked for "open cardboard box" on the same picture the boxes were found in. It reuses
that picture's encoding, so it costs about 40 ms, not another 200 ms. Of the cartons it finds,
the one in front is the one whose near edge is closest to the camera. Its inside is the whole
shape its outline goes round: SAM leaves the boxes in it out of the carton's outline, as holes.
A box at least half inside that shape is in the carton; every other box (clutter, a box lying
outside, a box in another carton) is left out of the order.

    keep_image(finder)                       # once: the finder keeps its last picture's encoding
    points, masks, ... = finder(color, depth)  # (detect_cache.find_boxes) the boxes, as before
    cartons = find_cartons(finder)           # [(outline, score)] in that same picture
    front = front_carton(cartons, depth, boxes)   # the carton in front, or None
    keep = inside(boxes, front)              # indices of the boxes in it

grid_order.order() does the last two when its frame has "cartons".
"""
import cv2
import numpy as np

CARTON_PROMPT = "open cardboard box"
CARTON_SCORE = 0.5   # SAM's carton outlines score 0.95-0.98 on the recordings; other outlines < 0.5
INSIDE = 0.5         # a box with this share of its outline inside the carton is in it
NEAR_PCT = 5         # the carton's near edge: this percentile of the depth along its outline


def keep_image(finder):
    """Make a box_masks.BoxFinder keep the encoding of the last picture it was given
    (finder.processor.last_image), so find_cartons() can reuse it. Safe to call twice.
    -> False for a finder that has no SAM processor to do it with."""
    proc = getattr(finder, "processor", None)
    if proc is None or not hasattr(proc, "set_image") or not hasattr(proc, "set_text_prompt"):
        return False
    if getattr(proc, "keeps_image", False):
        return True
    set_image = proc.set_image

    def keeping(*a, **k):
        state = set_image(*a, **k)
        proc.last_image = dict(state)       # before the box prompt adds its own keys
        return state

    proc.set_image = keeping
    proc.keeps_image = True
    proc.last_image = None
    return True


def find_cartons(finder, color_bgr=None):
    """[(outline: bool HxW, score)] of the cartons SAM finds in the last picture the finder was
    given (keep_image first), or in color_bgr if given (encodes it again, ~200 ms)."""
    import torch
    proc = finder.processor
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if color_bgr is not None:
            from PIL import Image
            state = dict(proc.set_image(Image.fromarray(np.ascontiguousarray(color_bgr[:, :, ::-1]))))
        else:
            if getattr(proc, "last_image", None) is None:
                raise RuntimeError("find_cartons: call keep_image(finder) before the finder is run")
            state = dict(proc.last_image)
        state = proc.set_text_prompt(CARTON_PROMPT, state)
    masks, scores = state["masks"][:, 0], state["scores"].float()
    keep = torch.nonzero(scores >= CARTON_SCORE).flatten()
    return [(m, round(float(s), 3)) for m, s in zip(masks[keep].cpu().numpy(), scores[keep].cpu())]


def hull(pts, shape):
    area = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(area, cv2.convexHull(pts), 1)
    return area.astype(bool)


def filled(outline, boxes=()):
    """The whole shape a carton outline goes round: its convex hull. The boxes in the carton are
    holes in SAM's outline, and those at the opening's edge are not even closed in by it, so
    filling holes is not enough (SAM's outline also has gaps, e.g. a flap in shadow).

    With the camera close, the front row of boxes runs off the bottom of the picture and hides the
    carton's lower edge (grid_3x3x2_full frames 370-379: 11 front-row boxes were cut off). Then, on
    each side where the carton runs off the left or right of the picture, the shape is closed down
    to that bottom corner. Only then: otherwise a flap reaching the side of the picture would pull
    in the shelf below the carton (live, 16 Nov)."""
    pts = cv2.findNonZero(outline.astype(np.uint8))
    if pts is None:
        return np.zeros(outline.shape, bool)
    pts = pts.reshape(-1, 2)
    area = hull(pts, outline.shape)
    h, w = outline.shape
    if not any(b.mask[-2:].any() and (b.mask & area).any() for b in boxes):
        return area
    corners = [(x, h - 1) for x, col in ((0, outline[:, 0]), (w - 1, outline[:, -1])) if col.any()]
    return hull(np.vstack([pts, np.array(corners, pts.dtype)]), outline.shape) if corners else area


def near_edge(outline, depth_m):
    """Distance (m) from the camera to the carton's nearest part, or None with no depth on it."""
    d = depth_m[outline]
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.percentile(d, NEAR_PCT)) if len(d) else None


def front_carton(cartons, depth_m, boxes=()):
    """The carton in front: {"index", "score", "near_m", "outline", "area"} or None.

    cartons: [(outline, score)] from find_cartons. A carton whose shape could not hold one box
    (smaller than the median box outline) is ignored; of the rest, the one whose near edge is
    closest to the camera is in front."""
    one_box = np.median([b.mask.sum() for b in boxes]) if len(boxes) else 0
    best = None
    for i, (outline, score) in enumerate(cartons):
        area = filled(outline, boxes)
        near = near_edge(outline, depth_m)
        if near is None or area.sum() < one_box:
            continue
        if best is None or near < best["near_m"]:
            best = {"index": i, "score": score, "near_m": near, "outline": outline, "area": area}
    return best


def inside(boxes, front):
    """Indices of the boxes with at least INSIDE of their outline in the carton's shape."""
    return [i for i, b in enumerate(boxes) if (b.mask & front["area"]).sum() >= INSIDE * max(b.mask.sum(), 1)]


def message(res):
    """(text, warn) about the carton in front, for the window and the pictures, from grid_order's
    answer (its "carton" and "picks"). warn: worth a colour."""
    found = res.get("carton")
    if found is None:
        return "Carton check off: every box in view used", False
    if not found["found"]:
        return "No carton found: every box in view used", True
    out = len(found["left_out"])
    tail = f", {out} outside left out" if out else ""
    if found["inside"] == 0:
        return "The carton in front is empty" + tail, True
    if not res.get("picks"):
        return f"Carton in front: {found['inside']} boxes in it, none to pick now" + tail, True
    return f"Carton in front: {found['inside']} boxes in it" + tail, False


def summary(found):
    """grid_order's "carton" made short and JSON-safe, for the target message and the logs:
    None (no carton check), {"found": False}, or {"found": True, "boxes_in", "left_out", "near_m"}."""
    if found is None:
        return None
    if not found["found"]:
        return {"found": False}
    return {"found": True, "boxes_in": found["inside"], "left_out": len(found["left_out"]), "near_m": found["near_m"]}


def draw(img, found, boxes):
    """The carton in front's shape (white) and the boxes left out (grey, crossed) on img, in place."""
    if not found or not found["found"]:
        return
    for i in found["left_out"]:
        cs, _ = cv2.findContours(boxes[i].mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (130, 130, 130), 2)
        x1, y1, x2, y2 = map(int, boxes[i].bbox)
        cv2.line(img, (x1, y1), (x2, y2), (130, 130, 130), 1)
        cv2.line(img, (x1, y2), (x2, y1), (130, 130, 130), 1)
    cs, _ = cv2.findContours(found["area"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, (255, 255, 255), 2)
