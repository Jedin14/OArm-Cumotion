"""v1 for the live pick loop (pick_loop.py): the same calls as the old picker's PickDetector
(vlm/molmo2-box-picking/pick_frame.py), with v1's grid order (grid_order.py) choosing the target.

    detector = GridDetector(score=0.4)                     # loads SAM 3.1 once (~5 s)
    result = detector.detect(colour, depth_m, intr, up)    # one frame
    result = detector.detect_frames([(colour, depth_m), ...], intr, up)   # robot stopped: n frames, one answer

result: the old picker's fields, so the robot side needs no change
    {"x_m", "y_m", "z_m"}  centre of the face to grip, camera coordinates (x right, y down, z away), metres
    "approach"             "front" or "top"
    "u", "v"               the same point in the picture, pixels
    "size_mm"              that face, long x short edge
    "num_boxes"            boxes found
    "top_only"             a grid box with no front in view, gripped by its top (worth telling a person)
    "frames", "agreeing"   detect_frames only: frames given, and how many found this target
plus v1's:
    "fallen"               the target has fallen out of the grid
    "cell"                 [layer, row, column] from 1 (layer 1 = front, row 1 = the highest row now in
                           that layer, column 1 = the leftmost box seen); null for a fallen box
    "box_size_mm"          the box size measured, [w, h, d] (d null when it can't be measured)
    "next"                 the next picks after this one (up to 3): [{"face", "fallen", "u", "v", "cell"}]
    "carton"               v2: {"found": true, "boxes_in", "left_out", "near_m"}: only the boxes in the
                           carton in front were used (left_out: how many were outside it);
                           {"found": false}: no carton found, every box used; null: carton check off

avoid: outlines of boxes passed over (T in the window): they are left out of the order.
With no target (None), self.res["carton"] says why: the carton in front may be empty.

box_size: (height, width, depth) mm of the boxes, given (the robot's /beast_arm/held_object/fixed_size,
or --box-size): its depth sets the layers (grid_order.py, 5). The loop doesn't pick without it.
The front measured in the picture is compared with it: more than SIZE_WARN off (height or width)
-> self.res["size_warning"] (the size entered may be another product's; partly hidden fronts
can also measure short, so it only warns).
"""
import time

import numpy as np

import carton
import grid_order
from detect_cache import find_boxes
from pick_v1 import draw

SAME_TARGET_M = 0.02     # two answers this close (same face kind) are the same box
SKIP_OVERLAP = 0.5       # an outline overlapping a passed-over one this much (of the smaller) is that box
SIZE_WARN = 0.25         # the front measured this much off the given box size (height or width): warn


def overlap(a, b):
    """Intersection over union of two (x1, y1, x2, y2) rectangles."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def passed_over(mask, avoid):
    return any((mask & m).sum() > SKIP_OVERLAP * min(mask.sum(), m.sum()) for m in avoid)


def size_warning(measured, given):
    """None, or why the front measured in the picture doesn't fit the given box size."""
    if not measured or not given or measured[0] is None or measured[1] is None:
        return None
    w, h = measured[0], measured[1]
    gh, gw = given[0], given[1]
    if abs(w - gw) > SIZE_WARN * gw or abs(h - gh) > SIZE_WARN * gh:
        return (f"the boxes look {w:.0f} wide x {h:.0f} high, the box size given is {gw:.0f} x {gh:.0f}: "
                "check the box size entered")
    return None


def _keep_image(finder):
    if carton.keep_image(finder):
        return True
    print("carton check off: this box finder has no SAM processor to ask for the carton", flush=True)
    return False


def cell(p):
    return None if p["cell"] is None else [int(c) + 1 for c in p["cell"]]


class GridDetector:
    """finder: a box_masks.BoxFinder to share (e.g. the old picker's, so SAM 3.1 is loaded once);
    None: load one. carton_check: v2, use only the boxes in the carton in front (carton.py)."""

    def __init__(self, score=0.4, prompt="box", finder=None, carton_check=True):
        if finder is None:
            from box_masks import BoxFinder
            finder = BoxFinder(prompt, score)
        self.finder = finder
        self.carton_check = carton_check and _keep_image(finder)
        self.last = None        # (points, boxes, target index) of the frame used: the pick check reads it
        self.res = None         # grid_order's answer for that frame (picks without passed-over boxes)
        self.best_frame = None

    def detect(self, color, depth_m, intr, up=(0.0, -1.0, 0.0), avoid=(), box_size=None):
        """The target dict, or None if there is no box to pick. box_size: (height, width, depth) mm."""
        up = np.asarray(up, float)
        t0 = time.time()
        points, masks, scores, boxes, sam_s = find_boxes(self.finder, color, depth_m, intr, up)
        tc = time.time()
        cartons = carton.find_cartons(self.finder) if self.carton_check else None
        t1 = time.time()
        res = grid_order.order({"boxes": boxes, "intr": intr, "up": up, "depth": depth_m, "cartons": cartons,
                                "box_depth": None if not box_size else box_size[2] / 1000.0})
        picks = [p for p in res["picks"] if not passed_over(boxes[p["box"]].mask, avoid)]
        self.res = dict(res, picks=picks, size_warning=size_warning(res.get("box_size_mm"), box_size),
                        box_given_mm=None if not box_size else list(box_size))
        if not picks:
            self.last = (points, boxes, None)
            return None
        p = picks[0]
        t = p["box"]
        boxes[t].pick = p["face_obj"]          # the face v1 grips (faces.analyse may have chosen another)
        self.last = (points, boxes, t)
        size = res["box_size_mm"]
        return {"x_m": round(p["xyz"][0], 4), "y_m": round(p["xyz"][1], 4), "z_m": round(p["xyz"][2], 4),
                "approach": p["face"], "u": round(p["uv"][0], 1), "v": round(p["uv"][1], 1),
                "size_mm": [round(v) for v in p["size_mm"]], "num_boxes": len(boxes),
                "top_only": p["face"] == "top" and not p["fallen"], "fallen": p["fallen"], "cell": cell(p),
                "box_size_mm": None if size is None else [None if v is None else round(v) for v in size],
                "next": [{"face": q["face"], "fallen": q["fallen"], "u": round(q["uv"][0], 1),
                          "v": round(q["uv"][1], 1), "cell": cell(q)} for q in picks[1:4]],
                "carton": carton.summary(res["carton"]), "box_given_mm": None if not box_size else list(box_size),
                "timings": {"sam_s": round(sam_s, 3), "carton_s": round(t1 - tc, 3), "order_s": round(time.time() - t1, 3),
                            "total_s": round(time.time() - t0, 3)}}

    def detect_frames(self, frames, intr, up=(0.0, -1.0, 0.0), avoid=(), box_size=None):
        """[(colour, depth_m), ...] of one still scene -> one target dict, or None.

        Each frame is run on its own; the answers that agree (same face kind, and centre within
        SAME_TARGET_M or outline in the same place) are grouped, and the largest group's median is
        returned (for a top: the frame that found the most of it, as the old picker does).
        self.last / self.res / self.best_frame then refer to the frame used."""
        found = []                             # (frame index, result, last, res)
        for i, (color, depth_m) in enumerate(frames):
            r = self.detect(color, depth_m, intr, up, avoid, box_size)
            if r is not None:
                found.append((i, r, self.last, self.res))
        if not found:
            self.best_frame = None
            return None
        xyz = lambda r: np.array([r["x_m"], r["y_m"], r["z_m"]])
        target = lambda g: g[2][1][g[2][2]]
        same = lambda g, f: g[1]["approach"] == f[1]["approach"] and (
            np.linalg.norm(xyz(g[1]) - xyz(f[1])) <= SAME_TARGET_M or overlap(target(g).bbox, target(f).bbox) > 0.5)
        group = max(([g for g in found if same(g, f)] for f in found), key=len)
        if group[0][1]["approach"] == "top":
            best = max(group, key=lambda g: target(g).pick.area_mm2)
            med = {k: best[1][k] for k in ("x_m", "y_m", "z_m", "u", "v")}
            size = np.asarray(best[1]["size_mm"], float)
        else:
            med = {k: float(np.median([g[1][k] for g in group])) for k in ("x_m", "y_m", "z_m", "u", "v")}
            size = np.median([g[1]["size_mm"] for g in group], axis=0)
            best = min(group, key=lambda g: np.linalg.norm(xyz(g[1]) - np.array([med["x_m"], med["y_m"], med["z_m"]])))
        self.best_frame, self.last, self.res = best[0], best[2], best[3]
        out = dict(best[1])
        out.update({k: round(med[k], 4) for k in ("x_m", "y_m", "z_m")})
        out.update({"u": round(med["u"], 1), "v": round(med["v"], 1), "size_mm": [round(float(v)) for v in size],
                    "frames": len(frames), "agreeing": len(group)})
        out.pop("timings", None)
        return out

    def annotate(self, color):
        """v1's view of the frame used: outlines, the order numbered, fallen boxes marked F."""
        return draw(color, self.last[1], self.res)
