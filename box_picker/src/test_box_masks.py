"""Checks for box_masks.select_masks on made-up outlines (no SAM, no GPU needed).

Run: python test_box_masks.py
"""
import torch

from box_masks import select_masks

H, W = 200, 400


def rect(x0, y0, x1, y1):
    m = torch.zeros(H, W, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def run(name, masks, scores, expect):
    kept = sorted(select_masks(torch.stack(masks), torch.tensor(scores)))
    ok = kept == sorted(expect)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: kept {kept} (expected {sorted(expect)})")
    return ok


def main():
    box_a, box_b = rect(10, 10, 100, 100), rect(110, 10, 200, 100)
    results = [
        run("two boxes", [box_a, box_b], [0.9, 0.8], [0, 1]),
        run("a stack holding two boxes goes", [box_a, box_b, rect(5, 5, 205, 105)], [0.9, 0.8, 0.95], [0, 1]),
        run("sticker on a box: the box stays", [box_a, rect(40, 40, 60, 60)], [0.9, 0.7], [0]),
        run("sticker SAM is surer of than the box: the box still stays", [box_a, rect(40, 40, 60, 60)], [0.7, 0.9], [0]),
        run("same box twice: one stays", [box_b, rect(112, 12, 198, 98)], [0.8, 0.85], [0]),
        run("unsure outline over a sure box in front: the box stays", [rect(10, 10, 120, 150), rect(15, 60, 110, 100)],
            [0.53, 0.81], [1]),
        run("background covering most of the picture goes", [box_a, rect(0, 0, 400, 200)], [0.9, 0.7], [0]),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
