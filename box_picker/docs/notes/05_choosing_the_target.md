# 5. Choosing the target

**In short:** of all the boxes found, the robot picks one. The rules are simple and fixed, written in [../picking_rules.md](../picking_rules.md); the code is `analyse` and `choose_target` in [`faces.py`](../../src/faces.py).

## Step 1: each box's pick face

For each box, look at its **front** and **top** faces (never a side) and take the **larger** one, by real size in mm². That decides how the robot approaches it: from the front or from above.

Exceptions, when the box has a front: a top whose front edge is seen at under 10° (almost edge-on) is measured badly, and a "top" narrower than the front is a sliver of another box. Then the front is used.

A box with neither a front nor a top isn't a whole box. It's usually a side split off a box, or a sliver. It is never picked, and it is never in another box's way.

## Step 2: which box

| Order | Rule | Why |
|---|---|---|
| 1 | **Nothing on top:** a box with another box resting on it waits | the box on top must go first, or it falls |
| 2 | **Nothing in front:** a box with another box in front of it waits | a box behind can't be reached yet |
| 3 | **On a box first, the highest top first:** boxes standing on another box before those on the floor; of those, keep the ones whose top is within 1.5 cm of the highest | stacks are cleared from the top down, so nothing under or beside a stack is pulled while it stands; by height, so a pile of thin boxes doesn't beat one box on tall ones; boxes on the floor are not ranked by height |
| 4 | **Nearest:** of those, keep the ones whose grip point is within 3 cm of the nearest one | the arm reaches less deep, and boxes behind come free |
| 5 | **Leftmost** of those | a fixed order, so the choice doesn't flicker |

![How the next box is chosen](../images/pick_logic_flow.png)

"Resting on" means: the box's bottom is within 3 cm of the other box's top height, above its middle and not below its top by more than 1.5 cm (so a box standing on the floor next to a thin flat box isn't "on" it), it is above it in the picture and overlaps it side to side, and it stands no more than 30 cm behind its front. In the overlay, such boxes say "(box on it)".

Never chosen, but still in the way of others: a box whose pick face runs off the edge of the picture, a box nothing holds up (not at the floor, not on a box that is held up: the AI calls pieces of the carton's rim "box"), and a box the robot gave up on. Two boxes pressed together, fronts level, look like one box twice as thick as the others: they are split in two first.

**In front** means either of:
- **In its way:** another box is nearer by more than 1.5 cm, at the same height, and across from it side to side with less than 3 cm to spare (a gripper can't use a narrow gap).
- **Hiding part of it:** a nearer box (by more than 1.5 cm) touches its bottom edge in the picture, and the camera can see that box's top **beside** it, ending there. Then it isn't standing on that box but behind it, and the box in front goes first.

  If that top can't be seen there (a level camera sees tops edge-on, or the box in front is no wider than the box above it), the camera can't tell "on top" from "behind". The box above then always counts as **resting on** it, so the box underneath waits: nothing is ever pulled out from under a box. The price: the robot may have to reach over a box for the one behind it.

A box lying on top of a stack is *not* blocked by the stack, even if it stands further back: the stack is below it.

How near a box is uses the depth of its front or top faces only (10th percentile): a side face can hold edge pixels of the neighbouring box and make a box deep in a gap look nearer.

## Step 3: the output

```json
{"x_m": -0.052, "y_m": 0.032, "z_m": 0.336, "approach": "front",
 "u": 543.0, "v": 431.0, "size_mm": [134, 41], "num_boxes": 27}
```

| Field | Meaning |
|---|---|
| `x_m`, `y_m`, `z_m` | centre of the pick face in metres (camera coordinates, note 2) |
| `approach` | `front` or `top` |
| `u`, `v` | the same point in pixels |
| `size_mm` | the pick face, long × short edge |
| `num_boxes` | boxes found in the frame |
| `averaged_over` | how many recent results were averaged (live: up to 5, while the target stays on the same box) |
| `frames`, `agreeing` | several frames of one still scene (`detect_frames`, `--frames N`): how many were given, and how many found this target; the result is their median (for a top: the frame that found the most of it) |

`null` means no box with a pickable face was found.

## Reading the overlay

| Mark | Meaning |
|---|---|
| thin yellow line | the box's outline from SAM 3.1 |
| yellow dot | "here is a box" (not the grip point) |
| green / blue fill | front face / top face |
| white rectangle + cross | the measured pick face and its centre |
| "(box on it)" | another box rests on this one |
| red rectangle + cross | the target |
