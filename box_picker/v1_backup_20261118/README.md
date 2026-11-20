# Box picking

Picking boxes that are all the same size, stacked in a grid inside a carton (wide × high × deep).

| Version | What it does | State |
|---|---|---|
| v1 | the pick order: fallen boxes first (by their top), then the front layer, top row first, left to right (by the front face), then the next layer back. The layers come from the box size the robot gives (no box size: no pick) | done, tested on the hand labels and live (18 Nov) |
| v2 | carton check: pick only from the carton in front, leave out every box outside it | built; same scores on the hand labels, needs the new clips |
| v3 | how the robot moves in and out (room around the box) | later |
| v4 | choose the carton: the robot goes to a given carton and picks only inside it | later |

The pick check (was the pick a success?) is the Pick Check team's work (`Pick Check/`); v1's pick loop already uses it.

Words:
- **layer**: a slab of boxes from front to back (layer 1 = the front)
- **row**: a line across (row 1 = the top)
- **column**: a line up and down (column 1 = the left, as the camera sees it)

## v1

From one camera frame, v1 gives every box it can pick, in order. For each box:
- the face to grip it by (front or top);
- whether it has fallen;
- the grip point (x, y, z);
- its place in the grid (layer, row, column).

It also gives the box size (w × h × d mm).

**The box size comes from the robot** (since 18 Nov 2026): `/beast_arm/held_object/fixed_size`, the size entered in the robot's GUI (`pick_interfaces/ObjectSize`: height, width, depth in mm; latched). The PC only reads it, never publishes it. **No box size (or a side of 0): no pick.** The window then says "Box size not set: enter it in the robot's GUI, or set it on this PC (ADVANCED: B)". In the pick loop's ADVANCED tab, **B** types this PC's own box size (height width depth in mm, then Enter) and **G** switches between the robot's and this PC's; it counts from the next look. Without a window (the USB camera, `--replay`, `pick_v1.py`, `--no-window`) give it with `--box-size H W D`.

Why it's needed: a box one layer back can't be told from a box set back a little without knowing how deep a box is, and the depth doesn't follow from the front. The robot's 125 × 125 × 54 mm boxes are much shallower than their front is wide, so the old rule (a new layer more than the front's thin side further back) put a back-layer box 55 mm behind into layer 1.

Reading it needs the robot's message type on the PC. A copy of just that message is in `v1/ros/pick_interfaces`; build it once (the system Python, not the .venv; the bridge then finds it by itself):

```bash
cd "box_picking/v1/ros"
env PATH=/usr/bin:/bin bash -c "source /opt/ros/humble/setup.bash && colcon build --base-paths . --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3"
```

The camera's tilt matters too: with no accelerometer reading the picker takes the camera as level, and an 8.7° tilt (18 Nov) made the fronts of one layer look 16–20 mm apart. The bridge listens to `/camera/camera/accel/sample` and `/camera/camera/imu` (the robot's driver sends it on `imu`).

### The rules (`v1/grid_order.py`)

1. **Box size.**
   - Width and height: the typical front face in view (shown next to the size given; more than a quarter off: the window warns "check the box size entered").
   - Depth: the size given. It sets the layers (3).
2. **Fallen.** A box that doesn't fit the grid is fallen. That is a box that:
   - lies low where the boxes stand up;
   - is much taller than a box (standing on its side or end);
   - lies flat;
   - is turned round;
   - or lies apart from the boxes behind it.

   Leaning (damaged) boxes stay in the grid.
3. **Order.**
   - Fallen boxes first, each by its biggest face (usually the top): the highest first, then the nearest, then the leftmost.
   - Then layer 1, top row first, each row left to right, by the front face.
   - Then layer 2, the same way.

   Layers: a front more than half a box depth further back than the layer in front starts the next layer. A box standing on a box of a nearer layer with its middle over it (its front less than half a box depth behind that box's back) joins that layer, so it is picked before the box under it.

   Boxes that can't be seen are not listed.
4. **No fixed cm.** Every tolerance is a share of the box (measured in that frame, or the depth given), so the rules work for any box size. Each setting can move 30% either way without changing the score, except "too tall", which must stay well above one box height; the box depth given can be 20% off either way.

### Scores on the hand labels (738 frames)

| | old picker | v1 |
|---|---|---|
| pick 1 right | 49% | **100%** (723/723) |
| picks 1–4 all right | – | 99% |
| fallen boxes found | 0% | **100%** (67/67), none called fallen wrongly |
| pick 1 on `grid_part3`, which v1 was not tuned on | 42% | **100%** |

Box size, steady within each clip:

| Recording | Box size found |
|---|---|
| flat boxes | 172 × 27 mm (d about 105–115) |
| barcode boxes | 28 × 100 mm (d about 185) |
| narrow white boxes | 42 × 105 mm (d about 120) |

### Known limits

- **A thin box with no depth on it** (in the camera's depth shadow) can't be placed, so it is left out of that frame's order. This happened in 3 frames.
- **Two boxes in one outline** from SAM count as one box (`grid_part3` clip 2).
- **The box size must be right.** A wrong depth entered in the robot's GUI puts boxes in the wrong layer. The window warns when the front seen is more than a quarter off the size given, but the depth itself can't be checked from one picture.
- **A box resting on a box that can't be seen** (hidden behind a front box) can be left out until that box comes into view.
- **Boxes outside the carton:** v1 alone uses every box in view, so clutter can make it pick the wrong box. v2 (below) leaves them out.

## v2: the carton check (`v1/carton.py`)

v1 used every box in view. Clutter below or beside the carton could set the floor or count as a fallen box (snapshot `snap_133553`, 16 Nov). v2 picks only from the carton in front:

1. **Find the cartons.** SAM 3.1 is asked for "open cardboard box" in the same picture. It reuses the picture's encoding, so this costs about 40 ms a frame (0.2 s a pick of 5 frames).
2. **The carton in front** is the one whose nearest edge is closest to the camera. A shape too small to hold one box is ignored.
3. **Its inside** is the whole shape its outline goes round (convex hull). SAM leaves the boxes in it out of the carton's outline, as holes. When the camera is close, the front row of boxes runs off the bottom of the picture and hides the carton's lower edge. Only then, where the carton runs off the left or right of the picture, the shape is closed down to that bottom corner (otherwise a flap reaching the side of the picture would pull in the shelf below the carton, as seen live on 16 Nov).
4. **Only boxes at least half inside it are used.** v1's rules then run on those alone.
5. **The carton in front is empty:** no pick, and the window says so. It does not move on to a carton behind it.
6. **No carton found:** every box is used, as in v1, and the window says so.

On the 738 hand-labelled frames the carton was found in every frame, no labelled box was left out, and every score is the same as v1's. On all 748 saved frames (the bad ones too) the whole pick list is the same as v1's. Those clips show only the carton, though. On `snap_133553` it leaves out the 22 things below the carton and picks the box lying in it first, then the top row left to right.

Not yet tested on clips with clutter (the new clips `carton_clutter`, `carton_two`, `carton_box_outside`, `carton_last`, `carton_spots` are for that). Known weak spots:
- **The shape includes the open flaps.** Something on the shelf above the carton, or a box lying on the front flap, counts as inside. While the carton has boxes, the shelf things are not picked (nothing holds them up), but they could be once it is empty. Two other shapes were tried on the saved frames: filling only the holes in the carton's outline, and leaving out what can be reached from the top of the picture. Both lost boxes in the carton, because SAM's carton outline has gaps (a flap in shadow, `grid_part2` frame 14).
- **Closing to a bottom corner** (only when a box runs off the bottom of the picture) also takes in things below the carton in that corner.

The target message has `carton`: `{"found": true, "boxes_in", "left_out", "near_m"}`, or `{"found": false}`. `--no-carton` turns the check off (`pick_loop.py`, `pick_v1.py`, `test_v1.py`).

Later (v4): the robot goes to a given carton and picks only in it. For now it is always the carton in front.

### Live: the pick loop

`v1/pick_loop.py` is the old `pick_loop.py` with v1 choosing the box. Everything else works as before:
- the window and its keys;
- the robot's `done`;
- pick checks v1 and v2 (v2, the camera watches the pick, is the default; with `--replay`, v1);
- retries and flags.

The target event carries the old fields (so the robot side needs no change) plus `fallen`, `cell` (layer, row, column), `box_size_mm` and the `next` 3 picks. The ORDER tab shows v1's order from the last look.

```bash
cd "box_picking/v1"
PY=../../vlm/molmo2-box-picking/.venv/bin/python
$PY pick_loop.py --ros                  # camera over ROS 2, with the window; the camera watches each pick (check v2)
$PY pick_loop.py --ros --no-window      # for the robot: stdin / stdout only
$PY pick_loop.py --ros --check v1       # the robot says "done" (C in the window), then the check
$PY pick_loop.py --ros --no-carton      # every box in view (v2's carton check off)
(source /opt/ros/humble/setup.bash && /usr/bin/python3 robot_done.py) | $PY pick_loop.py --ros --check v1
                                        # check v1 on the robot's own picks: robot_done.py hears the robot's arm
                                        # come back (arm_action/status) and says "done"; nothing is sent to the robot
$PY pick_loop.py --box-size 125 125 54  # D455 by USB: the box size (height width depth, mm) must be given
$PY pick_loop.py --replay ../../vlm/molmo2-box-picking/recordings/grid_3x3x2_full --replay-from 152 --replay-fps 0.1 --box-size 27 172 110
                                        # no camera: plays a recording (here a fallen box), C = done
```

### Commands

The box finding is copied from `vlm/molmo2-box-picking/` (`box_masks.py`, `faces.py`, `pick_rules.py`); the old code there is not changed. v1 uses that folder's Python, which already has SAM 3.1:

```bash
cd "box_picking/v1"
PY=../../vlm/molmo2-box-picking/.venv/bin/python
R=../../vlm/molmo2-box-picking/recordings

$PY pick_v1.py $R/grid_part2 7 8 9      # v1 on frames: JSON per frame + a picture in outputs/pick_v1/
$PY pick_v1.py $R/grid_part2 7 --sam    # find the boxes again with SAM 3.1 (GPU), not from cache/
$PY pick_v1.py $R/grid_part2 7 --no-carton   # v1 alone: every box in view
$PY test_v1.py --clips --sheet          # score v1 against the hand labels, pictures of the misses
$PY test_v1.py --order order_old        # the old picker, for comparison
$PY detect_cache.py $R/grid_part2       # find the boxes and cartons once in every labelled frame (GPU, ~0.5 s a frame)
$PY detect_cache.py $R/grid_part2 --cartons   # add the cartons to frames saved before v2
```

### Files

| File | What it is |
|---|---|
| `grid_order.py` | the v1 order (with v2's carton step at the start) |
| `carton.py` | v2: finds the carton in front and the boxes in it |
| `pick_v1.py` | runs v1 on recorded frames |
| `test_v1.py` | scores an order against the hand labels |
| `detect_cache.py` | finds the boxes once and saves them in `cache/` |
| `pick_loop.py` | the live pick loop with v1 (copied from the old folder; `loop_screen.py`, `pick_watch.py`, `rgbd.py` and `ros_bridge.py` are copied with it) |
| `grid_detector.py` | v1 behind the old picker's calls, for the loop (takes the box size) |
| `ros/pick_interfaces` | a copy of the robot's box size message, only to read `/beast_arm/held_object/fixed_size` (build it once, above) |
| `robot_done.py` | listens to the robot's pick status (read only) and tells the loop when the arm is back, for check v1 on the robot's own picks (system Python with ROS 2) |
| `order_old.py` | the old picker's first pick |
| `designs/` | the three designs tried; `order_lattice.py` won and became `grid_order.py` |

The hand labels are in each recording's `labels.json`, made with `vlm/molmo2-box-picking/label_grid.py` (see `recordings/GRID_NOTES.md`).
