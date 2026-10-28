# Running the box picker live with the robot

This is the short version: how to install it, check it with the camera, and connect it to the robot. The full reference is [INTEGRATION.md](INTEGRATION.md).

## What it does

One program, `src/pick_loop.py`, owns the camera (Intel RealSense D455) and does two jobs:

1. **Before each pick:** chooses the next box and says where to grip it (x, y, z in metres, camera coordinates) and from which side (`front` or `top`).
2. **After each pick:** checks whether the pick worked, meaning whether the box is gone from its place.

The robot does the picking and says when it is finished (`done`). The program never moves the robot.

```
 pick_loop                                   robot
 ─────────                                   ─────
 5 frames → target ── {"event": "target"} ──► pick the box, move the arm out of view
                    ◄──────── done ─────────
 3 frames → check ─── {"event": "result"} ──► picked: true  → wait for the next target
                                              picked: false → wait for the next target (usually the same box: a retry)
                                              picked: null  → clear the view, send done again (after 3 in a row: a new target)
 after 3 failed tries on one box ── {"event": "flag"} with photos ──► tell a person; the box is skipped
```

## 1. Requirements

| | |
|---|---|
| OS | Linux (tested on Ubuntu 22.04) |
| GPU | NVIDIA, 6 GB or more (uses ~4.3 GB), driver 570 or newer (CUDA 12.8) |
| Camera | Intel RealSense D455 on a **USB 3** port. Only one program can use it at a time |
| Disk | ~10 GB |
| Internet | During setup only (packages + the 3.5 GB model). For a PC without internet, see INTEGRATION.md, "Offline" |

## 2. Install (once)

```bash
unzip box_picker_for_robot.zip
cd box_picker/src
./setup.sh                      # makes src/.venv (Python 3.11) and downloads the model, 10-20 min
.venv/bin/python test_faces.py  # should end with "46/46 passed"
```

## 3. Check it with the camera (no robot yet)

Plug the camera in, then:

```bash
cd box_picker/src
.venv/bin/python pick_loop.py --pitch 0
```

A window opens. After ~10 s (the model loads, then the first 5 frames) it shows the chosen box in red.

1. Take that box away by hand, and keep your hand out of view.
2. Press **C**. This stands in for the robot's `done`.
3. The window shows **PICK SUCCESS**. A few seconds later the next box is chosen.
4. Press C **without** taking the box. The window shows **PICK FAILED: MISSED**, and the same box is chosen again ("try 2").
5. After 3 failed tries a **FLAG** is shown and the box is skipped.

Keys: **C** check now (= `done`), **N** choose a new box now, **R** forget the skipped boxes, **Q** quit.

`--pitch 0` means the camera is level. If it is mounted tilted down, give the angle in degrees.

## 4. Connect the robot

Run it without the window:

```bash
.venv/bin/python pick_loop.py --pitch 0 --no-window
```

- It reads commands from **stdin**, one per line.
- It writes events to **stdout**, one JSON line each.
- Messages for people go to stderr.

### Events it sends (stdout)

| Event | Means | Robot does |
|---|---|---|
| `{"event": "ready", ...}` | model and camera are up (~10 s after start) | wait for the first target |
| `{"event": "target", "pick": 1, "attempt": 1, "target": {...}}` | pick this box (`attempt` 2 or 3 = a retry on the same box) | convert to robot coordinates, pick it, move the arm out of view, then send `done` |
| `{"event": "result", "picked": true, "verdict": {...}}` | the box is gone: the pick worked | wait for the next `target` |
| `{"event": "result", "picked": false, ...}` | the box is still there, or was pushed or dropped (`verdict.outcome` says which) | wait for the next `target` |
| `{"event": "result", "picked": null, ...}` | can't tell (an arm or hand still in view, or something new appeared) | clear the view, send `done` again. After 3 in a row a new `target` comes instead |
| `{"event": "flag", "reason": "...", "photo": "...", "crop": "...", "check_photo": "..."}` | 3 tries on one box failed. The box is skipped, and the next box comes as a normal `target` | tell a person; the three photos show the box |
| `{"event": "no_target", "skipped": 0}` | no box to pick: empty, or only skipped boxes left (`skipped` > 0), or boxes it can't see well enough. It looks again every 3 s | stop, or wait. If `skipped` > 0 and a person has dealt with the flagged box, send `reset` |

The `target` dictionary:

```json
{"x_m": -0.275, "y_m": -0.026, "z_m": 0.506, "approach": "front",
 "u": 293.4, "v": 336.0, "size_mm": [170, 27], "num_boxes": 41, "frames": 5, "agreeing": 5}
```

| Field | Meaning |
|---|---|
| `x_m`, `y_m`, `z_m` | where to grip: the centre of the chosen face, metres, **camera coordinates** (x right, y down, z forward from the colour lens). Convert with the camera's pose on the robot (hand-eye calibration) |
| `approach` | `front`: come at the face horizontally. `top`: come down from above |
| `size_mm` | that face's size, long × short. Worth checking against the boxes you expect |
| `agreeing` | how many of the 5 frames chose this box. Under 3 means the scene was moving |

### Commands it accepts (stdin)

| Line | Means |
|---|---|
| `done` | the pick is over and the arm is out of view: check it now |
| `done held` / `done empty` | the same, plus what the gripper reports, if it can tell. A box that left its place but isn't in the gripper then reads as `dropped` |
| `pick` | choose a new box now |
| `reset` | forget the skipped boxes |
| `quit` | close the camera and exit |

### Example: driving it from the robot's Python (any version, standard library only)

```python
import json, subprocess

SRC = "/path/to/box_picker/src"
loop = subprocess.Popen([f"{SRC}/.venv/bin/python", "pick_loop.py", "--pitch", "0", "--no-window"],
                        cwd=SRC, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

def send(line):
    loop.stdin.write(line + "\n")
    loop.stdin.flush()

for line in loop.stdout:                      # one event per line
    e = json.loads(line)
    if e["event"] == "target":
        robot_pick(e["target"])               # yours: convert, pick, move the arm out of view
        send("done")
    elif e["event"] == "result" and e["picked"] is None:
        clear_view()                          # yours: something was in the way
        send("done")
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
| **The robot base doesn't move** from the `target` event until the `result` event | the check compares the picture before and after pixel by pixel |
| **Send `done` only when the arm and the held box are out of view** | anything in front of the boxes spoils the check (`picked: null`) |
| **Keep the arm out of view until the next `target` arrives** (about 3 s after `result`) | the next box is chosen from new frames, and a box in the gripper would be seen as a box |
| **The camera doesn't move on the robot**, and `--pitch` is its real tilt | sets "up", which decides front vs top and "on top of" |
| **Boxes about 0.4–0.6 m from the camera** (what was tested) | closer than ~0.4 m the camera gives no depth; further away, more noise |

## 6. What it saves

Everything goes under `src/outputs/pick_loop/<date_time>/`:
- `pick_NN/`: the frames before and after each try, `picker.jpg` (the chosen box) and `check.jpg` (the verdict, drawn on the picture)
- `flags/flag_NN_{target,crop,check}.jpg`: the photos of each flag
- `events.jsonl`: every event sent

## 7. Problems

| You see | Do |
|---|---|
| `Device or resource busy` / `Frame didn't arrive` | another program has the camera: close it. The program also resets the camera once by itself |
| `no accelerometer reading (... Permission denied)` | harmless: the camera is taken as level (`--pitch`) |
| `CUDA out of memory` | another program is using the GPU |
| `picked: null` again and again | something stays in view after `done` (arm, cable, hand, or a box put back), or the camera moved. After 3 in a row it chooses again |
| `no_target` but a box is still there | 1. `skipped` > 0: that box (or one in its place) was flagged earlier. Skipped boxes are remembered by their place in the picture until `reset` (R in the window) or a restart. 2. A box standing with its **narrow end** (~2 cm) towards the camera is sometimes not found as pickable: turn it so a wider face shows, or send `pick` to look again. 3. A box whose face runs off the edge of the picture is never picked: move the robot so it is fully in view |
| the same box flagged often | look at the flag photos: a box the gripper can't hold, or two boxes pressed together seen as one (split automatically only when at least 3 other boxes of that size are in view) |

## Status

- Tested on recordings, with saved frames standing in for the camera:
  - box taken → success, next box
  - box never taken → 3 tries, flag with photos, next box
  - can't tell → checked again on the next `done`
- Run live with the D455 and a person in place of the robot (C for `done`): taken / missed / flag / moved, merged boxes split, the carton's rim ignored.
- The picker has been tested live with the D455.
- **The full loop has not yet been run with real robot picks.** Please keep the `outputs/` folder of the first runs: it shows what the camera saw and decided at each step.
