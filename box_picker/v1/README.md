# v1: the grid pick order

For boxes that are all the same size, stacked in a grid in a carton. It decides which box the robot picks next, and by which face:

1. **Fallen boxes first.** These are boxes that don't fit the grid: lying low, tipped on their side, turned round, or lying apart. Each is picked by its biggest face, usually its top.
2. **Then layer 1 (the front layer).** The top row first, each row left to right as the camera sees it, by the front face.
3. **Then the next layer back,** the same way.

It also measures the box size. Every tolerance is a share of that size, so no distances are fixed in cm.

**v2, the carton check (16 Nov):** it picks only from the carton in front.
- SAM 3.1 is also asked for "open cardboard box" in the same picture. That adds about 40 ms a frame.
- The carton whose nearest edge is closest to the camera is the one in front.
- Only boxes at least half inside its shape are used. Clutter, a box lying outside, and a box in another carton are left out.
- When the carton in front is empty, there is no pick and the robot app says "the carton in front is empty". It does not move on to another carton.
- When no carton is found, every box in view is used, as before.

On 723 hand-labelled frames (3 kinds of box, 2 cartons), it chose the right first box in all 723 and found all 67 fallen boxes. The old picker chose the right first box in 49%.

It uses the same box finding as `src/` (SAM 3.1, `src/faces.py`) and the same Python (`src/.venv`).

## Update from the v1 add-on (16 Nov, v2 carton check)

If the v1 add-on is already installed:

1. **Unzip in the workspace.** It overwrites `box_picker/v1/` (new file: `carton.py`):
   ```bash
   cd ~/workspaces/isaac_ros-dev
   unzip -o box_picker_v2_carton_addon.zip
   ```
2. **Replace the app's two files again** with the ones in `box_picker_app/`, as for v1:
   - `box_picker_server.py`: sends `carton` and says "outside the carton in front" for the boxes left out.
   - `box_model_node.py`: says "the carton in front is empty" when it is.

   Only the picking itself (step 1) is needed. Without step 2 the messages are less clear.
3. **Restart the robot app.** The launch files' `order` default stays `grid`.

## Install on the robot PC (first time)

This adds `box_picker/v1/` next to `src/`; `src/` is not changed.

1. **Unzip in the workspace:**
   ```bash
   cd ~/workspaces/isaac_ros-dev
   unzip box_picker_v1_addon.zip
   ```
2. **Replace the app's two files** with the ones in `box_picker_app/`: `box_picker_server.py` and `box_model_node.py`. They go where the current ones are; the node starts the server from its own folder. To find them:
   ```bash
   find ~/workspaces/isaac_ros-dev -name box_model_node.py -not -path "*/install/*" -not -path "*/build/*"
   ```
   If the app is built with colcon, build it again as usual.

   If your copies changed after 11 Nov, don't overwrite them. Make the changes by hand instead:
   - **`box_model_node.py`:** `declare_parameter('order', 'v1')` becomes `'grid'`, and `order if order in ('v1', 'v2') else 'v1'` becomes `order if order in ('grid', 'v1', 'v2') else 'grid'`.
   - **`box_picker_server.py`:** the new class `Orders`, the `--order grid` and `--grid-dir` options, the start-up lines in `main()`, `set order grid`, and `why_not` in `pick()`. Compare with `box_picker_server.py` here.
3. **Start the robot app as usual.** The picker now uses order `grid`.

To go back to the old order, set the ROS parameter `order:=v1`, or send `set order v1` to the picker. `set order grid` switches back.

## What the robot gets

The `target` message has all the old fields (`x_m`, `y_m`, `z_m`, `approach`, `u`, `v`, `size_mm`, ...), so nothing on the robot side has to change. It also has:

| Field | Meaning |
|---|---|
| `fallen` | true for a fallen box |
| `cell` | `[layer, row, column]`, counted from 1. Layer 1 is the front, row 1 is the highest row now in that layer, column 1 is the leftmost box seen. `null` for a fallen box |
| `box_size_mm` | `[w, h, d]`. `d` is `null` when it can't be measured |
| `next` | the next 3 picks after this one |
| `carton` | v2: `{"found": true, "boxes_in", "left_out", "near_m"}`, or `{"found": false}` when no carton was found |

In `boxes`, `why_not` says where each box is in the order, for example `later: number 3 in the order`, or `outside the carton in front`.

## Try it on the robot PC without the robot

The pick loop with a window (camera over ROS 2; the camera watches each pick):

```bash
cd ~/workspaces/isaac_ros-dev/box_picker/v1
../src/.venv/bin/python pick_loop.py --ros
../src/.venv/bin/python pick_loop.py --ros --no-carton     # every box in view (carton check off)
```

Take a box out by hand, as in `RUN_LIVE.md` step 3. The ORDER tab shows the box size and the next picks.

## Known limits

- **A thin box with no depth on it** (in the camera's depth shadow) can't be placed, so it is left out until it can be seen.
- **Two boxes in one outline** from SAM count as one box.
- **A box lying out of place on top of the stack** also counts as fallen and goes first, by its top. No recording had one.
- **The carton check is not yet tested on clips with clutter** (only on one snapshot):
  - The carton's shape includes its open flaps. Something on the shelf above the carton, or a box lying on the front flap, counts as inside. While the carton has boxes, the shelf things are not picked (nothing holds them up), but they could be once it is empty.
  - When a box runs off the bottom of the picture (camera close) and the carton runs off the left or right, its shape is closed down to that bottom corner, so things below the carton in that corner count as inside.
