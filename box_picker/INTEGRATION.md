# Box picker: integration guide

**To run it live with the robot, start with [RUN_LIVE.md](RUN_LIVE.md)** (short). This file is the full reference.

The box picker looks at boxes with an Intel RealSense D455 (colour + depth). It tells the robot **which box to pick next**, **where the centre of the face to grip is** (x, y, z in metres, camera coordinates), and **whether to grip it from the front or from above**.

It never moves the robot. The camera has two jobs: **before** a pick, which box and where; **during** it, whether the pick worked (the **pick check**, made by the Pick Check team). The camera watches the pick itself, so the robot doesn't have to report anything.

```
1. PICK   5 frames ──► target {"x_m", "y_m", "z_m", "approach", ...} ──► robot picks it
2. WATCH  every camera frame: arm in view, box gone from its spot, the arm out of the picture again
          ──► picked: next box
          ──► missed / moved / dropped (fell back into the carton): pick again
              (a retry; after 2 retries: a FLAG with photos, once; the box is still tried)
```

**What changed since the last package:** the pick check is the Pick Check team's new one (`pick_watch.py`). The robot no longer sends `done`; a box lifted and then dropped back into the carton is now caught (`dropped`, a failed pick, retried); `picked` comes once the arm has carried the box out of the picture. The last box left alone in the carton is now found too (`box_masks.py`).

## What is in this folder

| Path | What it is |
|---|---|
| `src/pick_loop.py` | **Start here.** Pick, check, next, in one long-running process: prints each target as a JSON line on stdout, watches the pick, prints the result, then retries or takes the next box. Any language or ROS version can drive it |
| `src/pick_watch.py` | The pick check, by the Pick Check team (their `pick_check.py`, renamed): follows the pick from the depth on every camera frame, no GPU, a few ms a frame |
| `src/box_picker.py` | The Pick Check team's Python entry point: `BoxPicker.step(colour, depth, intr, t)` on every frame → the same target / result events (Option C) |
| `src/loop_screen.py` | `pick_loop.py`'s window: drawing only |
| `src/pick_server.py` | Pick only, no check: answers each `pick` line on stdin with one target |
| `src/pick_frame.py` | `PickDetector`: the Python API (`detect_frames`), if you call it from Python 3.11 directly. Also a command-line tool |
| `src/faces.py` | Depth → 3D faces of each box, and the rules that pick the target. Tunable constants are at the top |
| `src/pick_rules.py` | Which face of a box is gripped: front rules, top rules, the base rule, and the top faces mode, in one place |
| `src/box_masks.py` | SAM 3.1: finds each box's outline in the colour image from the text prompt "box" |
| `src/rgbd.py` | The D455 over USB (`Camera`) or over ROS 2 (`RosCamera`); save/load of captured frames |
| `src/ros_bridge.py` | Only when the camera is on another machine and published as ROS 2 topics; `--ros` starts it by itself (see "Camera over ROS 2") |
| `src/live_pick.py` | Live view with the target drawn on it, for checking by eye |
| `src/record_rgbd.py` | Records colour + depth frames, to test with later |
| `src/pick_pipeline.py` | Runs a recording through the picker and saves an overlay image + JSON per answer |
| `src/test_faces.py`, `src/test_box_masks.py`, `src/test_pick_watch.py` | Tests (no camera needed) |
| `src/setup.sh`, `src/requirements.txt` | Installation |
| `docs/robot_pick_logic.md` | The cycle, the answer, the picking rules, the assumptions. **Read this next** |
| `docs/picking_rules.md` | The rules in full, with the reasons |
| `docs/architecture.md`, `docs/notes/` | How it works inside, written for newcomers |
| `docs/images/pick_logic_flow.png` | Flowchart: how the next box is chosen |

## Requirements

| | |
|---|---|
| OS | Linux (tested on Ubuntu 22.04, kernel 6.8) |
| GPU | NVIDIA, **≥ 6 GB** memory (the model uses ~4.3 GB). Driver new enough for CUDA 12.8 (≥ 570). Tested on an RTX 5060 Ti (16 GB) |
| Python | 3.11, installed by `setup.sh` through `uv` (the system Python is not touched) |
| Camera | Intel RealSense D455 on a **USB 3** port. Only one program can open it at a time |
| Disk | ~10 GB (PyTorch + CUDA libraries ~6 GB, SAM 3.1 weights 3.5 GB) |
| Internet | During setup only (packages + weights). For a machine without internet, see "Offline" |

## Install

```bash
cd box_picker/src
./setup.sh                 # creates src/.venv and downloads the SAM 3.1 weights (no account needed)
```

The weights come from Meta's Hugging Face repo if this machine has access, else from a public copy. The copy is used only if its SHA-256 matches Meta's file. They are cached in `~/.cache/huggingface/`.

### Check the install

From `src/`:

```bash
.venv/bin/python test_faces.py          # no camera, no GPU: "48/48 passed" (24 scenes, clean and noisy)
.venv/bin/python test_box_masks.py      # "18/18 passed"
.venv/bin/python test_pick_watch.py     # the pick check: "34/34 passed"
.venv/bin/python pick_frame.py --live --frames 5              # one answer from the camera, as JSON
.venv/bin/python live_pick.py --frames 5                      # live window with the target drawn (Q quits)
```

### Offline (robot PC without internet)

1. Run `./setup.sh` on a machine with internet, or copy `src/.venv` from one (same OS, same Python path).
2. Copy the weights file, `sam3.1_multiplex.pt` (3.5 GB). After a setup it is at `~/.cache/huggingface/hub/models--*--sam3.1/snapshots/*/sam3.1_multiplex.pt`.
3. Point to it:

```bash
export SAM31_CHECKPOINT=/path/to/sam3.1_multiplex.pt
```

## How to call it

### Option A: `pick_loop.py`: pick, check, next (recommended)

Run it once at start-up and keep it running. It loops by itself:

1. **PICK:** takes 5 new frames, chooses the target, prints a `target` event. The pick check keeps the depth of the frame the target came from.
2. **WATCH:** every new camera frame is compared with it (a few ms each), until the pick is decided:
   - on the target's pick face: something in front of it (the arm), the face still there, or gone (the camera sees past it);
   - once the box has left its spot: the whole view, until it settles. Change touching the picture's edge is the arm (or a hand) still in view: wait.
3. **RESULT:** a `result` event:
   - `picked: true` (`picked`): the box left its spot and nothing new stays in view → back to 1 for the next box.
   - `picked: false` → back to 1: the picker usually chooses the same box again (a **retry**, `attempt` 2, 3), or for `dropped` the box where it fell. After 1 + 2 retries on the same box, a **`flag` event** is raised for a person, with three photos (the scene with the target marked, the box close up, the last check), once per box. The box is **not skipped**: it is tried again (`attempt` 4, 5, …) as long as the picker chooses it.

There is no "can't tell": if the arm stays in view, the check decides on what it sees after 300 frames (10 s at 30 fps). If the robot doesn't go for the box at all, the loop keeps watching: send `pick` to have it choose again.

```bash
cd box_picker/src
.venv/bin/python pick_loop.py --no-window           # for the robot
.venv/bin/python pick_loop.py                         # with a live window, for testing: pick the boxes by hand
.venv/bin/python pick_loop.py --ros                   # camera over ROS 2 instead of USB (see "Camera over ROS 2")
```

| Option | Default | What it does |
|---|---|---|
| `--pitch DEG` | accelerometer, else 0 | camera tilt down, degrees (0 = level). Over USB leave it out: the camera's accelerometer gives the real tilt (needs the RealSense udev rules, see "Assumptions"). Over ROS leave it out too when the driver sends the accelerometer (`enable_accel:=true`), else **set it** |
| `--ros` | off | frames from ROS 2 topics instead of the D455 on USB |
| `--order v1` / `v2` | v1 | which box goes first (see "Which box it chooses") |
| `--top-faces` | off | tops anywhere: every box by the larger of its front and top. Off: a box on another box by its front, a box on the base by the larger (see "Which box it chooses") |
| `--retries N` | 2 | retries on one box before a flag is raised (once; the box is still tried) |
| `--frames N` | 5 | frames per target |
| `--score X` | 0.4 | SAM's confidence needed to count as a box (thin boxes standing on end score 0.45–0.6) |
| `--no-window` | off | no window: stdin / stdout only |
| `--out DIR` | `outputs/pick_loop` | where each session is saved |

stdout, one JSON line per event:

```json
{"event": "ready", "session": "outputs/pick_loop/20261026_170000", "retries": 2}
{"event": "target", "pick": 1, "attempt": 1, "target": {"x_m": -0.275, "y_m": -0.026, "z_m": 0.506, "approach": "front", ...}}
{"event": "result", "pick": 1, "attempt": 1, "picked": true, "verdict": {"check": "v2", "outcome": "picked", "success": true, "seconds": 6.1, "shift_mm": -112}, "target": {...}}
{"event": "result", "pick": 2, "attempt": 1, "picked": false, "verdict": {"check": "v2", "outcome": "dropped", "success": false, "seconds": 8.4, "shift_mm": -30}, "target": {...}}
{"event": "flag", "pick": 7, "reason": "not picked after 3 tries: still trying it", "outcomes": ["missed", "missed", "dropped"], "target": {...},
 "photo": "outputs/pick_loop/<session>/flags/flag_07_target.jpg", "crop": ".../flag_07_crop.jpg", "check_photo": ".../flag_07_check.jpg"}
{"event": "flag", "pick": 8, "reason": "top face only", "target": {..., "approach": "top", "top_only": true},
 "photo": ".../flags/toponly_08_target.jpg", "crop": ".../flags/toponly_08_crop.jpg"}
{"event": "no_target", "pick": 9}
```

stdin, one line at a time:

| Line | Means |
|---|---|
| `pick` | choose a new target now (e.g. the robot can't reach this one) |
| `reset` | forget the boxes passed over with T |
| `quit` | exit |

The robot sends nothing for a normal pick: no `done`. (A `done` line is accepted and ignored, so old robot code still works.)

`verdict`:

| Field | Means |
|---|---|
| `outcome` | `picked` (`picked: true`): the box left its spot, then the view cleared. `missed`: the arm came and went, the box is still there. `moved`: knocked or dragged, neither there nor gone. `dropped`: it left its spot, then a new surface stood still in the carton: the box fell back in. All but `picked` are a failed try |
| `success` | the same as `picked` |
| `seconds` | from the target to the result |
| `shift_mm` | what the camera now sees on the face's pixels, mm from the face (minus = further away). About one box thickness back: a box stands right behind the one picked, or the box was pushed back that far; much further: open space behind it |

Each session is saved under `outputs/pick_loop/<session>/` (frames, overlays, `events.jsonl`).

#### The window (without `--no-window`)

For testing and commissioning. The picture shows the target's pick face framed in red, with a red cross on the grip point; the panel shows what the check sees (`waiting for the arm`, `arm in view`, `box left its spot, arm still in view`, ...). Pick the box by hand to try it. The panel has two tabs; each button can be clicked or pressed as a key.

| Tab | Key | Does |
|---|---|---|
| PICK | **T** | next target: passes over this box (and every box under it, all the way down) until the next successful pick; press again for the one after. With no other box left, back to the first |
| | **S** | save the current colour + depth to `<session>/snapshots/` |
| | **A** | the ADVANCED tab |
| | **Q** / Esc | quit |
| ADVANCED | **D** | depth picture instead of colour (red near, blue far, black: no reading), with the target drawn on it |
| | **V** | the picker's own overlay: every box found, its faces and grip points |
| | **S** | the same as on the PICK tab |
| | **O** | order v1 / v2, and choose again |
| | **F** | top faces on / off (as `--top-faces`), and choose again |
| | **N** | new frames, choose again |
| | **R** | forget the boxes passed over with T (the same as `reset`) |
| | − / + | Frames per target (1–15) and SAM confidence (0.20–0.90), the same as `--frames` / `--score`; used from the next target on |

The PICK tab shows how long the target took (`found in … s · pictures … s · picker … s`); the result screen shows how long the pick took.

**For the check to work:**
- The robot base must not move from the `target` event to the `result` (the depth is compared pixel by pixel).
- `pick_loop.py` must get **every camera frame** (it counts frames: 5 for a steady state, 10 for a clear view or a drop). Over USB that is 30 a second; over ROS 2 as many as arrive.
- The arm must leave the picture after the pick, carrying the box. An arm that parks in view delays `picked` up to 10 s.
- **Nothing may be put into the carton during a pick**: a box added (by hand, or a second box pulled along and let go) looks exactly like a dropped box. If the gripper's vacuum sensor says the box is held, trust the sensor.

No target (`no_target`): the picker tries again every 3 s.

Tested: the Pick Check team's 34 tests on synthetic scenes (picked, missed, dropped elsewhere and onto its own spot, a top pick dropped, the arm staying in view, a neighbour sliding into the gap), with noise and 10% depth holes; and a recording of hand picks run through `pick_loop.py` with the frames standing in for the camera: 3 `picked`, 1 `dropped` (a person put boxes into the carton during the pick), each followed by the next target. The earlier check (robot says `done`) was run live with the D455 over USB and over ROS 2. **This check has not yet been run live, nor with real robot picks: try a deliberate drop first.**

### Option A2: `pick_server.py`: pick only, no check

Run it once at start-up and keep it running. Loading takes ~5 s; after that each answer takes ~2–5 s (5 frames).

```bash
cd box_picker/src
.venv/bin/python pick_server.py
```

Protocol, one line each way:

| You write (stdin) | It answers (stdout, one JSON line) |
|---|---|
| (nothing, at start-up) | `{"ready": true, "frames": 5, "up": [0.0, -1.0, 0.0]}` once the model and camera are up |
| `pick` | `{"target": {...answer...} or null, "time": "...", "seconds": 2.4}` |
| `quit` | (exits) |
| anything else | `{"error": "unknown request ..."}` |

Log messages go to **stderr**, so stdout only carries the JSON lines.

Real output (from the D455, two boxes in view):

```json
{"ready": true, "frames": 5, "up": [0.0, -1.0, -0.0]}
{"target": {"x_m": -0.1254, "y_m": 0.1422, "z_m": 0.5874, "approach": "top", "u": 506.1, "v": 525.7, "size_mm": [149, 102], "num_boxes": 2, "frames": 5, "agreeing": 5}, "time": "2026-10-26 11:59:52", "seconds": 2.67}
```

Options:

| Option | Default | What it does |
|---|---|---|
| `--pitch DEG` | accelerometer, else 0 | How far the camera is tilted down, in degrees (0 = level). Leave it out to use the accelerometer: over USB (needs the RealSense udev rules) or over ROS (driver with `enable_accel:=true`, read anew every pick). Else set it |
| `--frames N` | 5 | Frames per answer. 3–5 is enough when the robot is still |
| `--skip N` | 5 | Frames thrown away first, so every frame used was taken after the request |
| `--save DIR` | off | Saves each request's frames + an overlay image of the target. Useful while commissioning |
| `--ros` | off | Camera over ROS 2 instead of USB (see below) |

Calling it from a ROS 2 node (Python 3.10, standard library only). The robot node starts the server as a child process:

```python
import json, subprocess

SRC = "/opt/box_picker/src"        # wherever the folder is
picker = subprocess.Popen([f"{SRC}/.venv/bin/python", "pick_server.py"],
                          cwd=SRC, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
assert json.loads(picker.stdout.readline())["ready"]    # waits ~5-10 s for the model and camera

def ask():
    """Call when the robot is stopped and the arm is out of view."""
    picker.stdin.write("pick\n")
    picker.stdin.flush()
    return json.loads(picker.stdout.readline())["target"]   # dict, or None

target = ask()
# ... at shutdown:
picker.stdin.write("quit\n"); picker.stdin.flush(); picker.wait()
```

From C++ or a shell, the same works: start the process, write `pick\n`, and read one line.

### Option B: the Python API (in Python 3.11, inside the `.venv`)

```python
import sys; sys.path.insert(0, "/opt/box_picker/src")
from pick_frame import PickDetector
from faces import Intrinsics, up_from_pitch
from rgbd import Camera

detector = PickDetector()                        # once: loads SAM 3.1
UP = up_from_pitch(0)                            # camera tilt down in degrees; fixed, so set once

cam = Camera()                                   # D455 over USB; or bring your own frames
frames = [cam.read() for _ in range(5)]          # each (colour BGR uint8 HxWx3, depth float32 metres HxW)
result = detector.detect_frames(frames, cam.intr, UP)    # dict or None
```

Frames from elsewhere (a ROS topic, files): colour BGR `uint8` and depth in **metres** `float32`, **aligned to colour** and the same size, and `Intrinsics(fx, fy, cx, cy)` of the colour camera. Depth 0 means no reading.

### Option C: `box_picker.py`: pick and check from your own Python 3.11 loop

The Pick Check team's entry point, if the robot program runs in the `.venv` and reads the camera itself. Feed it **every** camera frame:

```python
import sys, time; sys.path.insert(0, "/opt/box_picker/src")
from box_picker import BoxPicker
from rgbd import Camera

cam = Camera()
picker = BoxPicker(accel=cam.accel)              # loads SAM 3.1 once (~6 s); or pitch_deg=20
while True:
    colour, depth_m = cam.read()
    for event in picker.step(colour, depth_m, cam.intr, time.time()):
        handle(event)                            # {"event": "target", ...} / {"event": "pick_result", "result": ..., "success": ...}
```

It looks for a target (SAM, 0.3–1 s a frame) only while no pick is being watched, and hands one out once the same box was found in the same place 0.5 s apart. Its events are named `target` and `pick_result` (`result`: picked / missed / moved / dropped, `success`); a failed pick's next target carries `attempt` 2, 3, … with **no limit**, no flags and no skipping: that is left to the caller. `pick_loop.py` (Option A) has all of that, so use it unless you need your own loop.

## The answer

```json
{"x_m": -0.1254, "y_m": 0.1422, "z_m": 0.5874, "approach": "top",
 "u": 506.1, "v": 525.7, "size_mm": [149, 102], "num_boxes": 2, "top_only": false, "frames": 5, "agreeing": 5}
```

| Field | Meaning |
|---|---|
| `x_m`, `y_m`, `z_m` | **Where to grip**: the centre of the chosen face, metres, **camera coordinates** (see below) |
| `approach` | `"front"`: grip the face that points at the camera, coming in horizontally. `"top"`: grip from above |
| `size_mm` | That face's long × short edge, mm. Good for a sanity check against the boxes you expect |
| `u`, `v` | The same centre in image pixels (of the colour image: 1280 × 720 over USB; over ROS 2, the size the camera driver publishes, e.g. 640 × 480) |
| `num_boxes` | Boxes seen |
| `top_only` | `true`: a box on another box with no front found, only its top, so it is gripped from above as a last resort. Still pick it; the pick loop also raises a `flag` (reason `top face only`) for a person |
| `frames`, `agreeing` | Frames used, and how many of them chose this same box. The position is the median of the agreeing ones |

`null` / `None`: no box can be picked (empty, or nothing visible).

| Answer | What the robot should do |
|---|---|
| `null` | Nothing to pick: report done, or check the view |
| `agreeing` < half of `frames` | The frames disagree (something moved): ask again |
| otherwise | Convert to robot coordinates and pick from the `approach` side |

**After every pick, ask again.** A pick can move other boxes, so never reuse an old answer.

### Camera coordinates

```
            z (forward, away from the lens)
           /
 camera   o────── x (right)
          │
          y (down)
```

Metres, from the D455's **colour** camera optical centre (not the depth module's centre or the case). To use it on the robot you need the camera's pose in the robot frame (hand-eye calibration): `p_robot = T_robot_camera · p_camera`.

## Which box it chooses

**Order v1** (the default). In order (full detail and flowchart: `docs/robot_pick_logic.md`, `docs/images/pick_logic_flow.png`):

1. **Only front or top faces** (within 30° of facing the camera, or of level). A box turned further, with no such face, gets the face turned most towards the camera as its front. Which face (`src/pick_rules.py`):
   - a box **on another box**: its **front**. If no front was found, only its top: the top, as a last resort, with `top_only` and a flag;
   - a box **on the base** (no box under it): the **bigger** of front and top;
   - a top seen nearly edge-on (its front edge under 10°), or narrower than the front (a sliver of another box), never counts.

   With `--top-faces` (F in the window): the bigger of front and top for every box, as before 5 Nov 2026.
2. **Nothing resting on it.** A box with a box on top waits; the top one goes first.
3. **Nothing in front of it.** A box with a nearer box in its way waits.
4. **The highest top first**, on a box or on the floor (tops within 1.5 cm count as equal). Stacks are cleared from the top down, so nothing under a stack, or beside the boxes holding it up, is pulled while it stands; a tall box standing on the floor goes before a lower stack.
5. **Nearest** (grip point within 3 cm of the nearest), then **leftmost**.

When the camera can't tell whether a box stands *on* a nearer box or *behind* it (a level camera sees tops edge-on), it counts as *on*. Nothing is pulled from under a box, but the arm may need to reach over a box.

**Order v2** (`--order v2`, or O in the window): rules 1–3 as above (a free box: nothing on it, nothing in front of it), then simply the **leftmost**, whatever its height or distance. Predictable, and the arm moves less; but it can take a floor box standing right beside a stack before that stack is cleared, so use it where boxes stand apart or in neat rows, and v1 where they are piled or leaning.

Never the target, in either order: a box nothing holds up (pieces of the carton's rim), and boxes passed over with T, plus every box under them, all the way down.

## Assumptions the robot side must keep

| Assumption | If broken |
|---|---|
| The camera is fixed on the robot, and its tilt is known: from its motion sensor (leave out `--pitch`; USB, or ROS with `enable_accel:=true`; tested: 1.9° down), else `--pitch` | Wrong front/top choice, wrong "on top of" decisions, tops' depth off by 5–6 mm per degree |
| The robot is **still** while the frames are taken | `agreeing` drops; the position smears |
| **Arm and gripper out of view** | A box in the gripper is still a box and can become the target |
| Boxes ~0.35–0.6 m from the camera (what has been tested) | Depth noise grows with distance (at 0.5 m the centre moves ~1 mm frame to frame). Test before using other distances |
| Only the boxes to pick are in view | Anything box-shaped counts (a case on the table, a speaker) |

The accelerometer over USB needs the RealSense udev rules, once (the start-up says `no accelerometer reading (... Permission denied)` without them):

```bash
sudo wget -O /etc/udev/rules.d/99-realsense-libusb.rules https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger      # then unplug and replug the camera
```

## Camera over ROS 2 (instead of USB)

Use this when the D455 is plugged into another machine that publishes it as ROS 2 topics (`realsense2_camera` with `align_depth.enable:=true enable_accel:=true`). Needs ROS 2 Humble on this machine (`/opt/ros/humble`). Add `--ros`:

```bash
cd src && .venv/bin/python pick_loop.py --ros                 # also: pick_server.py, pick_frame.py --live, live_pick.py
```

ROS 2 Humble's Python is 3.10 and the `.venv` is 3.11, so `--ros` starts a small bridge, `ros_bridge.py`, with the system Python. It pairs colour and depth by time stamp and hands the newest pair over a local socket (127.0.0.1:50555). Its log is `outputs/ros_bridge.log`, and it stops when the program does. The default topics are `/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw` and `/camera/camera/color/camera_info`. To run the bridge by hand instead (e.g. other topic names): `source /opt/ros/humble/setup.bash && python3 ros_bridge.py --color ... --depth ... --info ...`, then `--ros` uses it.

The tilt comes from the driver's accelerometer topic `/camera/camera/accel/sample`: start the driver with `enable_accel:=true`. `pick_loop.py` and `pick_server.py` read it anew for every pick (the mean of the last 0.5 s), so it follows the robot as it moves. Without it, pass `--pitch`.

**If no frames arrive, or only a few a second:**

1. `ros2 topic list` here must show the camera topics. If not: the same `ROS_DOMAIN_ID` on both machines, `ROS_LOCALHOST_ONLY` unset, both on the same network with multicast allowed (`ros2 multicast send` on one, `ros2 multicast receive` on the other).
2. Topics listed but frames very slow or none (`no colour + depth pair in 5 s`): each image is split into many network packets, and Linux drops them once its reassembly space fills. This was the cause on our network (Wi-Fi). On **this** machine:

```bash
sudo sysctl -w net.core.rmem_max=2147483647 net.core.rmem_default=2147483647
sudo sysctl -w net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728
```

These are lost at reboot. To keep them, put the same four settings in `/etc/sysctl.d/60-ros-camera.conf` (`net.core.rmem_max = 2147483647`, ...) and run `sudo sysctl --system`. Nothing needs changing on the camera's machine. A wired network is better still.

Tested live from another machine over Wi-Fi: 640 × 480 colour + depth at ~10–15 frames a second, enough for the loop (it uses 5 frames per target, 3 per check).

## Tuning

Constants are at the top of `src/faces.py`, each with a comment. The ones most likely to matter:

| Constant | Default | Meaning |
|---|---|---|
| `HIGHEST_TOL_M` | 0.015 | Boxes whose tops are this close in height count as "equally high" (keep it under the thinnest box) |
| `NEAREST_TOL_M` | 0.03 | Boxes whose grip points are this close in depth count as "equally near"; then the leftmost |
| `IN_WAY_M` | 0.015 | A nearer box whose top is this much above another box's bottom is "in the way" |
| `FLAT_VIEW_DEG` | 10 | A top seen at a flatter angle than this is dropped (its pixels go to the front face) |
| `SKIP_OVERLAP` | 0.5 | A box whose outline overlaps a box passed over with T this much counts as that box (left out) |
| `SAME_LENGTH`, `DOUBLE_TOL`, `DOUBLE_REFS` | 0.07, 0.15, 3 | Two boxes pressed together: a front as long as others (7%), twice as thick (15%), with at least 3 others to compare |
| `MAX_BOX_DEPTH_M` | 0.30 | A box resting on another can stand at most this far behind the lower box's front (the deepest box expected) |

`pick_loop.py`'s retries: `--retries` (default 2). The pick check's settings are at the top of `src/pick_watch.py` (owned by the Pick Check team):

| Constant | Default | Meaning |
|---|---|---|
| `DROP_FRAMES` | 10 | frames a new surface must stand still to be a dropped box |
| `CLEAR_FRAMES` | 10 | frames with nothing new in view before `picked` |
| `SETTLE_FRAMES` | 300 | longest wait for the view to settle (10 s at 30 fps); then decided on what is seen |
| `NEAR_M` | 0.020 | how much nearer than before counts as something new |
| `DROP_AREA_FRAC`, `MIN_DROP_MM2` | 0.2, 800 | smallest new surface that counts as a box (share of the pick face, and mm²) |
| `EDGE_BAND_PX` | 64 | border (half-resolution pixels) where change means "arm in view"; covers the D455's left strip with no depth |

In `src/box_masks.py`: `LONE_SCORE` (0.3), the lowest SAM score looked at when no box is found, and `STANDS_OUT_M` (15 mm), how far such an outline must stand in front of the carton's surface.

The text prompt (`"box"`) and SAM's score cut-off are arguments of `PickDetector(prompt=..., score=...)` (0.5 by default there; `pick_loop.py` uses 0.4, `--score`).

After a change, run `test_faces.py`. To check it on real scenes, record frames and run them through the picker:

```bash
.venv/bin/python record_rgbd.py recordings/run1             # R starts/stops recording, Q quits
.venv/bin/python pick_pipeline.py --capture "recordings/run1/frame_*" --frames 5 --out outputs/run1   # the tilt saved with the frames
#   -> one overlay .jpg + .json per 5 frames in outputs/run1
```

## Known limits

- **Two boxes pressed together** (side by side, or one lying on the other) with their fronts level are drawn as one outline by the AI. When the doubled face is about twice as thick as other boxes of the same length in view (at least 3 of them), it is split in two and the left or top one is picked. Alone, or among boxes of other sizes, it can't be told: check `size_mm`. If a box really is twice as thick as its neighbours of the same length, it is split by mistake; the grip point then lands a quarter of the way across that box, still on it.
- **A box standing with its narrow end (~2 cm) towards the camera** is sometimes not found as pickable (its big side, seen at a slant, crowds out the narrow face when the faces are fitted). It is then left for later, or `no_target` if it is the last one: turn it, or look again (`pick`).
- **A box that can't be picked is tried again and again.** After 3 failed tries a flag tells a person, but the loop keeps choosing that box until it is gone or moved: a person must deal with it (or the robot sends `pick` to look again).
- **A box whose pick face runs off the edge of the picture** (e.g. the bottom row when the camera looks over a stack) is gripped at the middle of the part in view, not at its real centre: off by half the hidden part, but still on the box. Better: have the whole box in view.
- **Face rotation is not returned.** You get the centre and the approach side, but not how the face is turned (a box standing at an angle). It is measured inside (`detector.last`, each face's normal and corners), so it can be added to the answer if the gripper needs it.
- **Tops seen from a level camera** are measured less well: one pixel is ~2.5 mm front to back, and 1° of tilt error moves the depth 5–6 mm. With the motion sensor's tilt, 170 × 100 tops measured 167–170 × 96–101.
- **No depth at the picture's sides.** The D455 gives no depth in a band on the left (~40 px at 0.45 m, wider when nearer: part of the left camera's view the right one can't see) and on the right (the colour camera sees a little wider than the depth cameras). A box there is outlined but has no depth, so it isn't picked until it's fully inside. The depth picture (D in the window) shows the bands in black.
- **Shiny white tops** can have patchy depth (black in the depth picture). The top's width then comes from its outline (all of it at the top's height), so it stays right; a top that is mostly without depth can still come out small in one frame.
- **A box added to the carton during a pick reads as `dropped`** (see "For the check to work"). So does anything else that appears and stays put in the carton while the box is away.
- **The last box alone in the carton** is found when SAM gives it at least 0.3 and it stands 15 mm clear of the carton's surface. Not yet tried live with one box left: try it.
- **Speed**: ~0.3–1 s per frame on an RTX 5060 Ti; more boxes take longer.

## Versions (pinned)

Python 3.11, PyTorch 2.11 (CUDA 12.8), SAM 3 code at commit `7f473cf` with the SAM 3.1 weights (`sam3.1_multiplex.pt`, SHA-256 `0567debe…41cb6`), pyrealsense2 2.59.1, OpenCV 5.0, NumPy 2.4. See `src/requirements.txt`.
