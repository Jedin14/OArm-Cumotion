"""Checks for box_masks.select_masks on made-up outlines (no SAM, no GPU needed), some with depth.

Run: python test_box_masks.py
"""
import numpy as np
import torch

from box_masks import lone_boxes, select_masks

H, W = 200, 400


def rect(x0, y0, x1, y1):
    m = torch.zeros(H, W, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def run(name, masks, scores, expect, depth=None):
    kept = sorted(select_masks(torch.stack(masks), torch.tensor(scores), depth))
    ok = kept == sorted(expect)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: kept {kept} (expected {sorted(expect)})")
    return ok


def run_lone(name, masks, scores, expect, depth):
    kept = sorted(lone_boxes(torch.stack(masks), torch.tensor(scores), depth))
    ok = kept == sorted(expect)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: kept {kept} (expected {sorted(expect)})")
    return ok


def plane_depth(z0, dzdv=0.0):
    """Depth of a plane, metres: z0 at the top row, changing by dzdv per row (a floor seen from above)."""
    return np.tile((z0 + dzdv * np.arange(H))[:, None], (1, W)).astype(np.float32)


def with_box(depth, m, z):
    """depth with the pixels of outline m at z (a box face), plus a little noise and some holes."""
    d = depth.copy()
    d[m.numpy()] = z
    rng = np.random.default_rng(1)
    d += rng.normal(0, 0.0015, d.shape).astype(np.float32)
    d[rng.random(d.shape) < 0.1] = 0
    return d


def main():
    box_a, box_b = rect(10, 10, 100, 100), rect(110, 10, 200, 100)
    # The last box left: SAM outlines the carton's wall (or floor) round it too, holding only that box.
    wall, last = rect(200, 40, 360, 160), rect(250, 80, 300, 150)
    before_wall = with_box(plane_depth(0.60), last, 0.50)
    on_floor = with_box(plane_depth(0.70, -0.0015), last, 0.50)    # floor nearer at the bottom of the picture
    sticker_depth = with_box(plane_depth(0.50), rect(40, 40, 60, 60), 0.499)
    results = [
        run("two boxes", [box_a, box_b], [0.9, 0.8], [0, 1]),
        run("a stack holding two boxes goes", [box_a, box_b, rect(5, 5, 205, 105)], [0.9, 0.8, 0.95], [0, 1]),
        run("sticker on a box: the box stays", [box_a, rect(40, 40, 60, 60)], [0.9, 0.7], [0]),
        run("sticker SAM is surer of than the box: the box still stays", [box_a, rect(40, 40, 60, 60)], [0.7, 0.9], [0]),
        run("same box twice: one stays", [box_b, rect(112, 12, 198, 98)], [0.8, 0.85], [0]),
        run("unsure outline over a sure box in front: the box stays", [rect(10, 10, 120, 150), rect(15, 60, 110, 100)],
            [0.53, 0.81], [1]),
        run("background covering most of the picture goes", [box_a, rect(0, 0, 400, 200)], [0.9, 0.7], [0]),
        run("last box before a wall, no depth: the wall wins (why depth is needed)", [wall, last], [0.8, 0.85], [0]),
        run("last box before a wall: the box stays", [wall, last], [0.8, 0.85], [1], before_wall),
        run("last box on the carton floor: the box stays", [wall, last], [0.8, 0.85], [1], on_floor),
        run("sticker on a box, with depth: the box stays", [box_a, rect(40, 40, 60, 60)], [0.9, 0.7], [0],
            sticker_depth),
        run("two boxes, with depth: both stay", [box_a, box_b], [0.9, 0.8], [0, 1], before_wall),
    ]
    # Nothing sure enough: the carton outline (SAM's largest) and weak outlines in it.
    carton, lone, stain = rect(0, 0, 400, 200), rect(250, 80, 300, 150), rect(60, 40, 120, 90)
    print("  no outline above the threshold (lone_boxes):")
    results += [
        run_lone("a weak box before the carton's back wall is kept", [carton, lone], [0.8, 0.4], [1],
                 with_box(plane_depth(0.60), lone, 0.50)),
        run_lone("a weak box on the carton floor is kept", [carton, lone], [0.8, 0.4], [1],
                 with_box(plane_depth(0.70, -0.0015), lone, 0.50)),
        run_lone("a weak flat stain on the wall goes", [carton, stain], [0.8, 0.35], [],
                 with_box(plane_depth(0.60), stain, 0.599)),
        run_lone("box kept, stain goes", [carton, lone, stain], [0.8, 0.4, 0.35], [1],
                 with_box(with_box(plane_depth(0.60), lone, 0.50), stain, 0.599)),
        run_lone("no carton outline: nothing to compare with, nothing kept", [lone], [0.4], [],
                 with_box(plane_depth(0.60), lone, 0.50)),
        run_lone("a weak outline without depth goes", [carton, lone], [0.8, 0.4], [], np.zeros((H, W), np.float32)),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
