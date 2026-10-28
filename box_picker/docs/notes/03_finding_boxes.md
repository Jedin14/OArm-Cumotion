# 3. Finding the boxes with SAM 3.1

**In short:** SAM 3.1 is an AI model from Meta. Given a picture and the word "box", it draws an outline (a **mask**) around every box and gives each a confidence **score**. We then throw away outlines that aren't one single box.

Code: [`box_masks.py`](../../src/box_masks.py).

## Detection, segmentation, masks

| Task | Output |
|---|---|
| Detection | a rectangle around each object (what an older YOLO model, not in this package, does) |
| Segmentation | the exact pixels of each object: a **mask**, true/false for every pixel |

We need masks, not rectangles: the depth step must use only the pixels of one box, not the background or the neighbour that a rectangle would also cover. In the overlay, a mask is drawn as the thin **yellow outline**.

## SAM 3.1 and text prompts

SAM means "Segment Anything Model". Version 3 added **text prompts**: you say what to find ("box") and it finds every instance, without any training on our boxes. That is why the project needs no labelled data. Version 3.1 (March 2026) is the same idea with improved weights.

SAM 3.1 ships as a video-tracking model. We only need its **detector** part for single frames, so `build_detector()` builds just that and runs it through Meta's `Sam3Processor`.

Each outline comes with a **score** from 0 to 1: how sure SAM is that it's a box. Outlines under 0.5 are dropped. Real boxes usually score 0.75–0.96.

The weights (3.5 GB) come from Meta's Hugging Face repo when you have access, otherwise from a public copy that is used only if its SHA-256 checksum matches Meta's file exactly.

## Cleaning up the outlines

SAM sometimes returns outlines that aren't one box. `select_masks()` drops:

| Outline | Why it's wrong | Rule |
|---|---|---|
| Covers more than 25% of the picture | the carton or the background | dropped |
| Holds 2 or more other outlines | a whole stack | dropped |
| Mostly inside another outline | the same box twice, or a sticker on a box | the **larger** one is kept |
| Unsure (score < 0.65) and covers a box SAM is much surer of (+0.15) | a box behind, drawn over the boxes in front of it | the **smaller, surer** one is kept |

Then `remove_overlaps()`: pixels claimed by two outlines belong to neither, because their depth could belong to either box.

Tests for these rules: [`test_box_masks.py`](../../src/test_box_masks.py) (no GPU needed).

## The yellow dot

Each box gets one point: the pixel deepest inside its outline (furthest from its edges, found with a **distance transform**). It just marks "here is a box"; it is **not** where the robot grips. That's the cross (note 4).

## What SAM can get wrong

- **Two boxes flush side by side**, with nothing visible between them, can get one outline. Depth can't split them either, since their fronts are in the same plane.
- **Anything box-shaped** counts, including things outside the carton.
- **A box in your hand** is still a box.
