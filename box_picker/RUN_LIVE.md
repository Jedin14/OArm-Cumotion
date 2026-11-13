# Running the box picker live with the robot

This is the short version: how to install it, check it with the camera, and connect it to the robot. The full reference is [INTEGRATION.md](INTEGRATION.md).

## What it does

One program, `src/pick_loop.py`, owns the camera (Intel RealSense D455) and does two jobs:

1. **Before each pick:** chooses the next box and says where to grip it (x, y, z in metres, camera coordinates) and from which side (`front` or `top`).
2. **During each pick:** watches it on every camera frame and says whether it worked: picked, or missed, moved, or dropped (lifted, then fell back into the carton).

The robot does the picking. It doesn't have to report anything: the camera sees when the pick is over. The program never moves the robot.

```
 pick_loop                                   robot
 ─────────                                   ─────
 5 frames → target ── {"event": "target"} ──► pick the box, carry it out of the picture
 every frame: watch ─ {"event": "result"} ──► picked: true  → wait for the next target
                                              picked: false → wait for the next target (usually the same box: a retry)
 after 3 failed tries on one box ── {"event": "flag"} with photos ──► tell a person; the box is still tried
```

New in this version: the robot no longer sends `done`, and a box that falls back into the carton is caught (`dropped`).

## 1. Requirements

| | |
|---|---|
| OS | Linux (tested on Ubuntu 22.04) |
| GPU | NVIDIA, 6 GB or more (uses ~4.3 GB), driver 570 or newer (CUDA 12.8) |
| Camera | Intel RealSense D455 on a **USB 3** port. Only one program can use it at a time. Or on another machine, published over ROS 2 (section 3b) |
| Disk | ~10 GB |
| Internet | During setup only (packages + the 3.5 GB model). For a PC without internet, see INTEGRATION.md, "Offline" |

## 2. Install (once)

```bash
unzip box_picker_for_robot_v2.zip
cd box_picker/src
./setup.sh                      # makes src/.venv (Python 3.11) and downloads the model, 10-20 min
.venv/bin/python test_faces.py  # should end with "48/48 passed"
```

## 3. Check it with the camera (no robot yet)

Plug the camera in, then:

```bash
cd box_picker/src
.venv/bin/python pick_loop.py
```

A window opens. After ~10 s (the model loads, then the first 5 frames) it shows the face to pick on the chosen box framed in red, with a red cross on the grip point. The panel on the right shows a close-up, the position in cm and the try number.

Stand in for the robot with your hand. The panel shows what the check sees (`waiting for the arm`, `arm in view`, `box left its spot, arm still in view`...).

1. Reach in, take that box out of the carton, and take your hand (with the box) out of the picture.
2. About 1 s later the window shows **PICK SUCCESS**. A few seconds later the next box is chosen.
3. Reach in and touch the box **without** taking it, then take your hand out. The window shows **PICK FAILED: MISSED**, and the same box is chosen again ("try 2").
4. Lift the box and **drop it back** into the carton, then take your hand out. The window shows **PICK FAILED: DROPPED**, with where it fell outlined, and that box is chosen again.
5. After 3 failed tries a **FLAG** is shown (once), and the same box is tried again (try 4, 5, …).

Don't put anything else into the carton while a pick is watched: it would read as a dropped box.

Keys: **T** next target (pass over this box and every box under it), **S** save a snapshot, **A** the Advanced tab, **Q** quit.
The Advanced tab (buttons can be clicked too): **D** depth picture, **V** the picker's overlay of every box, **S** save a snapshot, **O** order v1 / v2, **F** top faces on / off, **N** choose again with new frames, **R** forget the boxes passed over with T, and − / + for **Frames per target** (5) and **SAM confidence** (0.40). On the command line: `--frames`, `--score`.

The Pick tab also shows how long the target took: `found in 2.9 s · pictures 0.4 s · picker 2.5 s`. The result screen shows how long the pick took.

**Camera tilt.** With the camera on USB, leave out `--pitch`: the camera's motion sensor gives its real tilt. That matters: 1° wrong moves a top's front-to-back size 5–6 mm (a 100 mm top read as 109). If the start-up says `no accelerometer reading`, install the RealSense USB rules once:

```bash
sudo wget -O /etc/udev/rules.d/99-realsense-libusb.rules https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger      # then unplug and replug the camera
```

Over ROS (below) the tilt comes from the driver's motion sensor topic. Without any motion sensor, give `--pitch` with the tilt in degrees down (0 = level).

### 3b. Camera on another machine (ROS 2)

If the D455 is plugged into another machine that publishes it with the RealSense ROS 2 driver, and this machine has ROS 2 Humble. On the **camera** machine, start the driver with colour, aligned depth and the motion sensor:

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true align_depth.enable:=true enable_accel:=true
```

Then on this machine:

```bash
.venv/bin/python pick_loop.py --ros
```

The tilt is read from `/camera/camera/accel/sample` anew for every pick, so it follows the robot as it moves; each target line in the log shows it (`tilt 1.9 deg`). If the start-up says `no accelerometer reading`, the driver wasn't started with `enable_accel:=true` (check: `ros2 topic hz /camera/camera/accel/sample`).

It starts the ROS bridge by itself and stops it on exit. If frames don't arrive or come very slowly, run this once on **this** machine (lost at reboot; INTEGRATION.md, "Camera over ROS 2", says how to keep it):

```bash
sudo sysctl -w net.core.rmem_max=2147483647 net.core.rmem_default=2147483647
sudo sysctl -w net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728
```

### 3c. Choosing the order, and when boxes are missed

- **Order:** `--order v2` (or **O** in the Advanced tab) takes the leftmost free box first. The default, v1, clears stacks from the top down first. v2 suits boxes standing apart; v1 suits piled or leaning boxes.
- **Top faces:** by default a box on another box is gripped by its front, and only a box on the base may be gripped from above (the bigger of its front and top). `--top-faces` (or **F** in the Advanced tab) allows tops anywhere. A box on a box with no front found is still picked by its top, and a `flag` with reason `top face only` tells a person.
- **Boxes not found:** SAM gives each box a confidence. Thin boxes standing on end score only 0.45–0.6, so the cut-off is 0.40 (`--score`). If boxes are missed, lower **SAM confidence** in the Advanced tab; if bits of the carton are taken for boxes, raise it.

## 4. Connect the robot

Run it without the window:

```bash
.venv/bin/python pick_loop.py --no-window          # over ROS: add --ros
```

- It reads commands from **stdin**, one per line.
- It writes events to **stdout**, one JSON line each.
- Messages for people go to stderr.

### Events it sends (stdout)

| Event | Means | Robot does |
|---|---|---|
| `{"event": "ready", ...}` | model and camera are up (~10 s after start) | wait for the first target |
| `{"event": "target", "pick": 1, "attempt": 1, "target": {...}}` | pick this box (`attempt` 2 or 3 = a retry on the same box) | convert to robot coordinates, pick it, carry it out of the picture |
| `{"event": "result", "picked": true, "verdict": {"outcome": "picked", ...}}` | the box left its spot and the arm has carried it out of view: the pick worked | wait for the next `target` |
| `{"event": "result", "picked": false, ...}` | `verdict.outcome`: `missed` (still there), `moved` (knocked or dragged) or `dropped` (fell back into the carton) | wait for the next `target` (usually the same box) |
| `{"event": "flag", "reason": "...", "photo": "...", "crop": "...", "check_photo": "..."}` | 3 tries on one box failed (sent once per box). The box is **not** skipped: it comes again as the next `target` | tell a person; the three photos show the box |
| `{"event": "no_target"}` | no box to pick: empty, or boxes it can't see well enough. It looks again every 3 s | stop, or wait |

The `target` dictionary:

```json
{"x_m": -0.275, "y_m": -0.026, "z_m": 0.506, "approach": "front",
 "u": 293.4, "v": 336.0, "size_mm": [170, 27], "num_boxes": 41, "frames": 5, "agreeing": 5}
```

| Field | Meaning |
|---|---|
| `x_m`, `y_m`, `z_m` | where to grip: the centre of the chosen face, metres, **camera coordinates** (x right, y down, z forward from the colour lens). Convert with the camera's pose on the robot (hand-eye calibration). Over ROS, the camera is the one in the topics' `camera_info` |
| `approach` | `front`: come at the face horizontally. `top`: come down from above |
| `size_mm` | that face's size, long × short. Worth checking against the boxes you expect |
| `agreeing` | how many of the 5 frames chose this box. Under 3 means the scene was moving |

### Commands it accepts (stdin)

| Line | Means |
|---|---|
| `pick` | choose a new box now (e.g. the robot can't reach this one; otherwise the loop keeps watching it) |
| `reset` | forget the boxes passed over with T |
| `quit` | close the camera and exit |

### Example: driving it from the robot's Python (any version, standard library only)

```python
import json, subprocess

SRC = "/path/to/box_picker/src"
loop = subprocess.Popen([f"{SRC}/.venv/bin/python", "pick_loop.py", "--no-window"],
                        cwd=SRC, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

def send(line):
    loop.stdin.write(line + "\n")
    loop.stdin.flush()

for line in loop.stdout:                      # one event per line
    e = json.loads(line)
    if e["event"] == "target":
        robot_pick(e["target"])               # yours: convert, pick, carry the box out of the picture
    elif e["event"] == "result":
        log(e["picked"], e["verdict"]["outcome"])           # picked / missed / moved / dropped; a retry comes as the next target
    elif e["event"] == "flag":
        notify_person(e["reason"], e["photo"], e["crop"])    # yours
    elif e["event"] == "no_target":
        break                                 # or wait: it looks again every 3 s
send("quit")
```

A ROS 2 node can run the same code: the program runs in its own Python 3.11, so the node's Python version doesn't matter.

## 5. Rules the robot side must keep

| Rule | Why |
|---|---|
| **The robot base doesn't move** from the `target` event until the `result` event | the check compares the depth pixel by pixel with the picture from before the pick |
| **After the pick, carry the box out of the picture** | `picked` comes once nothing new stays in view; an arm that parks in view delays it up to 10 s |
| **Nothing is put into the carton during a pick** (by hand, or a second box pulled along and let go) | it looks exactly like a dropped box. If the gripper's vacuum sensor says the box is held, trust the sensor |
| **Keep the arm out of view until the next `target` arrives** (about 3 s after `result`) | the next box is chosen from new frames, and a box in the gripper would be seen as a box |
| **`pick_loop.py` gets every camera frame** (don't run another program on the camera) | the check counts frames (5 for a steady state, 10 for a clear view or a drop) |
| **The camera doesn't move on the robot**, and its tilt is known (motion sensor over USB, else `--pitch`) | sets "up", which decides front vs top and "on top of" |
| **Boxes about 0.4–0.6 m from the camera** (what was tested) | closer than ~0.4 m the camera gives no depth; further away, more noise |

## 6. What it saves

Everything goes under `src/outputs/pick_loop/<date_time>/`:
- `pick_NN/`: the frames before each try and the frame the result was decided on, `picker.jpg` (the chosen box) and `check.jpg` (the verdict, drawn on the picture; for `dropped`, where the box fell)
- `flags/flag_NN_{target,crop,check}.jpg`: the photos of each flag
- `events.jsonl`: every event sent

## 7. Problems

| You see | Do |
|---|---|
| `Device or resource busy` / `Frame didn't arrive` | another program has the camera: close it. The program also resets the camera once by itself |
| `no accelerometer reading (... Permission denied)` | the camera is taken as level, so tops can be measured ~1 cm off: install the RealSense USB rules (section 3), or give `--pitch` |
| `CUDA out of memory` | another program is using the GPU |
| `dropped` though the box was taken | something new appeared in the carton during the pick (a hand, a box put back, a neighbour that fell over and stayed). Look at `check.jpg`: the outline shows where. The settings are at the top of `src/pick_watch.py` (INTEGRATION.md, "Tuning") |
| `picked` comes late, or only after 10 s | the arm (or a cable) stays at the picture's edge after the pick: move it out of view |
| no result at all after a `target` | the robot didn't reach into the picture (nothing was seen to happen). Send `pick` to choose again |
| `no colour + depth pair in 5 s` with `--ros` | frames lost on the network: the `sysctl` lines in section 3b |
| a box at the far left or right isn't picked | the camera gives no depth in a band at each side (black in the depth picture, D). Move the robot so the box is further in |
| `no_target` but a box is still there | A box standing with its **narrow end** (~2 cm) towards the camera is sometimes not found as pickable: turn it so a wider face shows, or send `pick` to look again. |
| the same box flagged, then tried again and again | look at the flag photos: a box the gripper can't hold, or two boxes pressed together seen as one (split automatically only when at least 3 other boxes of that size are in view) |

## Status

- The pick check (the camera watches): the Pick Check team's 34 tests on synthetic scenes pass (picked, missed, dropped elsewhere and onto its own spot, the arm staying in view, a neighbour sliding into the gap). A recording of hand picks run through `pick_loop.py`: 3 picked, 1 dropped (a person put boxes in during the pick), each followed by the next target. **Not yet run live: try section 3, including a deliberate drop, before the robot.**
- The earlier check (the robot said `done`) was run live with the D455 and a person in place of the robot: taken / missed / flag / moved, merged boxes split, the carton's rim ignored. The picking side is unchanged.
- Run live over ROS 2, the camera on another machine over Wi-Fi.
- The picker has been tested live with the D455. Sizes checked with ArUco markers and 170 × 100 × 25 boxes (3 Nov): fronts 169–172 × 98–107, tops 167–170 × 96–101 with the motion sensor's tilt.
- **The full loop has not yet been run with real robot picks.** Please keep the `outputs/` folder of the first runs: it shows what the camera saw and decided at each step.
