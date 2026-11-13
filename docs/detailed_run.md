# Detailed run: robot stack + web GUI, start to finish

This is what happens when you run the OpenArm pick system on `beast`:

1. `native/run_launch_everything.sh` (the robot stack).
2. `native/run_web_gui.sh` (the web GUI and the pick stack).

It covers every step, every case (success, refusal, failure, retry, recovery), and for each one what goes out on ROS and what comes in.

Message fields, QoS and command syntax are in `ros_topic.md`. The everyday commands are in `cmds.md`. This file is about the run: what happens, in what order, and why.

Conventions:

- Topic names without a leading `/` are under `/beast_arm` (change it with `--ns`).
- "Latched" means transient-local, depth 1: a subscriber that joins late gets the last value at once.
- `log` means the human-readable line that appears in the GUI's Activity list. The same line is published on `/beast_arm/log` and written to `~/.ros/log/python3_<pid>_*.log`.
- Vacuum = left arm only, finger = right arm only. The vacuum is the default.

---

## Contents

1. [The processes at a glance](#1-the-processes-at-a-glance)
2. [Before you start](#2-before-you-start)
3. [Terminal 1: the robot stack](#3-terminal-1-the-robot-stack)
4. [Terminal 2: the web GUI and pick stack](#4-terminal-2-the-web-gui-and-pick-stack)
5. [The /beast_arm interface while running](#5-the-beast_arm-interface-while-running)
6. [The GUI page](#6-the-gui-page)
7. [Poses: navigation, pre_pick, return](#7-poses-navigation-pre_pick-return)
8. [Tool and box size](#8-tool-and-box-size)
9. [Detect box](#9-detect-box)
10. [Move: the pick itself, step by step](#10-move-the-pick-itself-step-by-step)
11. [After the pick: the camera check, retries, base_adjust](#11-after-the-pick-the-camera-check-retries-base_adjust)
12. [A full pick from ROS: ready_to_pick](#12-a-full-pick-from-ros-ready_to_pick)
13. [Advanced tab: typed camera coordinates](#13-advanced-tab-typed-camera-coordinates)
14. [STOP](#14-stop)
15. [Click-only mode (--click)](#15-click-only-mode---click)
16. [Calibrate tab](#16-calibrate-tab)
17. [Example timelines](#17-example-timelines)
18. [Every case in one table](#18-every-case-in-one-table)
19. [Shutting down](#19-shutting-down)

---

## 1. The processes at a glance

```
Terminal 1: native/run_launch_everything.sh
  ros2_control_node + controllers   CAN can0 (right), can1 (left); /joint_states; trajectory controllers
  robot_state_publisher             /tf, /tf_static (arms and the head camera, from the URDF)
  move_group (MoveIt)               /move_action, /execute_trajectory, IK/FK, cartesian paths, octomap
  cumotion_planner                  cuMotion (GPU) planner behind move_group; held-box / tool spheres
  realsense camera (D455)           colour 640x480x15, depth 640x480x15 aligned to colour
  octomap_gater                     lets depth frames into the octomap only when asked (static map)
  boot_pre_pick.py                  walks both arms to the navigation pose once, then exits
  rviz2                             3-D view (use_rviz:=false turns it off)

Terminal 2: native/run_web_gui.sh
  box_model_node.py                 /beast_arm/box_model/detect + check services
    +- ros_bridge.py                camera frames -> TCP 127.0.0.1:50555 (for the picker)
    +- box_picker_server.py         SAM 3.1 box picker (box_picker/src/.venv, Python 3.11)
  web_gui.py (ROS node "click_to_move")
                                    the browser page on :8080 and the whole /beast_arm interface
```

Run only one controller of the arms at a time: the web GUI, `run_pick_stack.sh` or one of the OpenCV tools. They all drive the same arms.

---

## 2. Before you start

| Need | Why | Check |
| --- | --- | --- |
| CAN up (run on beast with sudo): `ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on && ip link set can0 up` (same for `can1`) | The arms' motors talk CAN FD. The launch script does not configure CAN | `ip -details link show can0` shows `UP` |
| A display (`DISPLAY` set) on beast | Without one, the octomap updater's freeglut crashes move_group. The script refuses to start | `echo $DISPLAY` |
| The `/workspaces/isaac_ros-dev` symlink | The robot files refer to that path | `ls -l /workspaces/isaac_ros-dev` |
| Camera plugged in, no other program using it | The D455 hangs if another app holds it, or after a USB re-enumeration | `rs-enumerate-devices` |
| `source native/setup.bash` (the run scripts do it) | Sets `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_DOMAIN_ID=0`, `ROS_LOCALHOST_ONLY=0` | `printenv ROS_DOMAIN_ID` |

Network visibility: with `ROS_LOCALHOST_ONLY=0` and domain 0, every PC on the LAN with default ROS settings sees and can publish to `/beast_arm/*`. There is no login on the GUI or the topics, so anyone who can reach them can move the arms. Set `BEAST_ROS_LOCALHOST_ONLY=1` before the run scripts to keep ROS on beast only.

To see the topics from a terminal on beast or on another PC: Humble, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, and `ROS_DOMAIN_ID` unset or `0`. A terminal whose `~/.bashrc` sets another domain (beast's sets `42`) sees nothing until it runs `export ROS_DOMAIN_ID=0` or `source native/setup.bash`. If `ros2 topic list` still shows an old view, restart the daemon: `ros2 daemon stop`.

Other robots share domain 0. A mobile base on this LAN now keeps its topics under `/c40000003/`. Before that it published plain `/joint_states`, `/tf`, `/tf_static` and `/robot_description`, which move_group received: `Joint 'left_wheel_joint' not found in model 'openarm'`, repeated. If that message comes back, `ros2 topic info -v /joint_states` names the extra publisher.

---

## 3. Terminal 1: the robot stack

```bash
native/run_launch_everything.sh            # options: depth_profile:=424x240x15  accel:=true  use_rviz:=false ...
```

### 3.1 What the script does

1. Checks for `/workspaces/isaac_ros-dev` and `DISPLAY`. If either is missing, it prints why and exits.
2. Runs `native/clean_dds_shm.sh`. This removes FastDDS shared-memory files in `/dev/shm` that no running process maps, i.e. files left behind by crashed runs. A broken leftover port once made every new node deaf: the spawner, the planner, the GUI and the camera bridge all hung. It prints `FastDDS shared memory: N leftover file(s) removed`.
3. Runs `ros2 launch robot/launch_everything.launch.py octomap:=static 4d:=false use_fake_hardware:=false right_can_interface:=can0 left_can_interface:=can1`, plus your arguments.

### 3.2 Launch arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `depth_profile` | `640x480x15` | Depth resolution. The depth camera's minimum range grows with it: 424x240 ≈ 0.20 m, 640x480 ≈ 0.34 m, 848x480 ≈ 0.45 m |
| `accel` | auto | The camera's accelerometer (gravity's "up" for the box picker). On when its motion sensor can be read, which needs the RealSense udev rules (see `docs/cmds.md`); without them the camera node hung at start with it on. Off, the box picker uses the accelerometer's last saved reading (`calibration/camera_tilt.yaml`), else the TF's tilt, which is not gravity's. The GUI's Box picker card shows which (`TILT ... ACCEL` / `TF`) |
| `octomap` | `static` (from the script) | The 3-D map is filled only on request (Detect box, Recapture), not continuously |
| `use_rviz` | `true` | RViz window |
| `boot_pre_pick` | `true` | Walk the arms to the navigation pose at start |
| `collision_activation_distance` | `0.03` | cuMotion's collision margin, m |
| `use_fake_hardware` | `false` (from the script) | `true` = simulated arms |

### 3.3 Start-up, in order

| Step | Process | What it provides | Cases |
| --- | --- | --- | --- |
| 1 | `robot_state_publisher` | `/tf`, `/tf_static`. The head camera is in the URDF (`world` → `camera`), so its pose comes from here | - |
| 2 | `ros2_control_node` | Hardware interface on CAN; controllers at 100 Hz | CAN down: the controllers fail to start |
| 3 (after 1 s) | One controller spawner, in order: `joint_state_broadcaster`, `left_joint_trajectory_controller`, `right_joint_trajectory_controller`, `left_gripper_controller`, `right_gripper_controller` | `/joint_states` (positions and efforts); `/<arm>_joint_trajectory_controller/joint_trajectory` and `/follow_joint_trajectory`; `/<arm>_gripper_controller/gripper_cmd` | If it fails it is restarted after 2 s, twice more. After 3 failures: `controller spawner failed 3 times - the arms cannot move; check the CAN buses and restart` |
| 4 | `move_group` (respawns after 3 s if it dies) | `/move_action`, `/execute_trajectory`, `/compute_ik`, `/compute_fk`, `/compute_cartesian_path`, `/check_state_validity`, `/get_planning_scene`, `/apply_planning_scene`, `/clear_octomap`. Planners: cuMotion (default), OMPL, CHOMP, Pilz. Octomap from `/camera/camera/depth/image_rect_raw`, 2 cm voxels, 1.5 m range | - |
| 5 | `cumotion_planner` (respawns after 4 s) | GPU planning for move_group; the `planner_attach_object` action (spheres for the tool, the calibration pieces and a held box) | - |
| 6 | RealSense camera | `/camera/camera/color/image_raw`, `/camera/camera/color/camera_info`, `/camera/camera/aligned_depth_to_color/image_raw`, depth (renamed `..._in` for the gater); `/camera/camera/accel/sample` only with `accel:=true` | Hangs at start: replug the camera and close anything that uses it |
| 7 | `octomap_gater` | Passes depth frames from `.../image_rect_raw_in` into the octomap only when asked: service `/octomap_gater/refresh` (3 frames), topic `/octomap_gater/allow_frames` (Int32), or its Tk button | - |
| 8 | `boot_pre_pick.py` | Waits for `/move_action` and `/joint_states`, then moves both arms to `navigation_state` (right, then left, 40 % speed, up to 180 s), then exits | An arm already at the pose (within 0.02 rad) is left alone. An arm that starts in collision is first eased 15 % of the way without checks. If the fast OMPL path fails, it falls back to cuMotion per arm, retried every 3 s, via home after 2 failures |
| 9 | `rviz2` | 3-D view | - |

When terminal 1 settles, the arms are at navigation. Nothing under `/beast_arm` exists yet.

---

## 4. Terminal 2: the web GUI and pick stack

```bash
native/run_web_gui.sh                       # open http://beast:8080 (or http://beast.local:8080) from any PC
# options: port:=8090  host:=127.0.0.1  pick_tool:=vacuum|finger  approach:=auto|top|front  model:=false
#          order:=v1|v2 (which box first: v1 stack tops, then nearest, then leftmost; v2 the leftmost free box)
#          score:=0.4 (SAM's confidence to count as a box)
#          --click   (click-only: no box model)
```

### 4.1 What the script does

1. Sources `native/setup.bash`.
2. Runs `native/stop_pick_tools.sh`. This stops any earlier web GUI, pick stack, box model, picker, bridge or OpenCV tool, sending INT, then TERM, then KILL. It never touches the robot stack from terminal 1.
3. Runs `native/clean_dds_shm.sh` (see 3.1).
4. Runs `ros2 launch app/launch/web_gui.launch.py ...`, which starts two processes: the box model (unless `model:=false` or `--click`) and the web GUI.

### 4.2 The box model starting (`box_model_node.py`)

| Step | What happens | Log / case |
| --- | --- | --- |
| 1 | Stops leftover picker or bridge processes from a killed run (orphans whose parent is PID 1) | - |
| 2 | Subscribes to `/camera/camera/color/camera_info`, `/camera/camera/accel/sample` (only useful with `accel:=true`) and TF | - |
| 3 | Starts `ros_bridge.py` (Python 3.10). It subscribes to the colour image, aligned depth and camera info, and serves synchronised frame pairs on TCP `127.0.0.1:50555` | Log: `native/logs/box_picker_bridge.log` |
| 4 | Waits up to 30 s for camera info and the TF `world` → colour optical frame, and computes the camera's pitch | Fails: `no camera info / tf for the camera after 30 s` |
| 5 | Waits (up to 20 s) until the bridge takes connections, then starts `box_picker_server.py` in `box_picker/src/.venv` (Python 3.11) with `--score 0.4 --order v1` (the `score` / `order` launch options), which loads SAM 3.1. It waits up to 120 s for its first line, `{"ready": true, ...}` | Missing venv: `the box picker is not installed (...): cd box_picker/src && ./setup.sh`. Bridge not up: `ros_bridge.py not listening on 50555 after 20 s; see native/logs/box_picker_bridge.log`. Picker log: `native/logs/box_picker.log` |
| 6 | Logs `box picker ready (camera pitch X deg); serving /beast_arm/box_model/detect`. Now it serves `box_model/detect` and `box_model/check` | A Detect while it is still loading waits up to 90 s, then answers `the box picker is still starting - try again` |

It publishes `box_model/target`, `box_model/image` (latched), `box_model/scene` (latched JSON) and `box_model/check_result` (latched JSON), each only when a detection or check happens.

### 4.3 The web GUI starting (`web_gui.py`, ROS node `click_to_move`)

1. Creates the node and its clients:
   - Subscribes to the camera colour, aligned depth and camera info, and to `/joint_states` and TF.
   - Action clients: `/move_action`, `/execute_trajectory`, the gripper actions and cuMotion's `planner_attach_object`.
   - Service clients: IK, FK, cartesian path, planning scene, `/clear_octomap`, `/octomap_gater/refresh`.
   - Publishes to `/<arm>_joint_trajectory_controller/joint_trajectory`, used only to hold an arm still on STOP or contact.
2. Creates the `/beast_arm` interface (section 5).
3. Waits up to 10 s for `/joint_states`.
4. Publishes the first values:

   | Topic | First value |
   | --- | --- |
   | `arm_state` | `navigation` (the arms are where boot_pre_pick left them), `pre_pick`, or `unknown` |
   | `arm_action/phase` | `idle` |
   | `held_object/state` | `none` |
   | `status` | the JSON object, every 1 s and on every change |
   | `vacuum` | nothing until the first change |

5. Starts the web server on `0.0.0.0:8080`.

`stop_pick_tools.sh` stops only these GUI processes. Restarting the GUI never touches the robot stack.

---

## 5. The /beast_arm interface while running

What the GUI process (the same as `run_pick_stack.sh`) publishes and listens to:

| Topic / service | Dir | Type | QoS | When |
| --- | --- | --- | --- | --- |
| `arm_state` | out | String | latched | Every 1 s, and within 0.2 s of a change |
| `arm_action/status` | out | String | latched | At each step of a job (`detecting`, `target_found`, `picking`, `checking`, the result) |
| `arm_action/phase` | out | String | latched | At each motion step (`left approach`, `left contact`, ...) |
| `held_object/state` | out | String | latched | When an arm starts or stops holding a box in the planners (`left 170x100x25 mm` / `none`) |
| `status` | out | String (JSON) | latched | Everything at once, every 1 s and on change |
| `log` | out | String | depth 10 | Every message |
| `pick_target` | out | PickTarget | depth 10 | Once per detection |
| `pick_result` | out | String (JSON) | latched | Once per pick, and when the box is unreachable from here |
| `base_adjust` | out | BaseAdjust | depth 10, not latched | When the base should move (11.4) |
| `vacuum` | out | Bool | latched | On a change only: `true` just before each press, `false` at the drop pose and when the arm is back with nothing held. Left arm only. The vacuum bridge (`app/vacuum_bridge.py`, started by terminal 1) switches the ESP32 relay from it |
| `held_object/size` | in, and out from the GUI's Dummy size | ObjectSize | - | Size of the box to pick |
| `arm_state/cmd` | in | String | depth 10 | `navigation`, `pre_pick`, `return`, `status` |
| `arm_action` | in | String | depth 10 | `ready_to_pick [top\|front\|auto] [vacuum\|finger]`, `stop` |
| `stop` | in | Empty | depth 10 | Stop everything |
| `held_object/release` | in | Empty | depth 10 | Take the held box out of the planners |
| `arm_state/get` | service | Trigger | - | The current `arm_state` |
| `box_model/detect`, `box_model/check` | client | DetectTarget, CheckPick | - | Called by the GUI or pick stack |
| `box_model/image`, `box_model/scene` | in (GUI only) | Image, String | latched | The picker's picture and what to draw on it |

Also published for RViz: `/click_to_move/marker` (the chosen point) and `/click_to_move/target` (the tool pose aimed for at contact).

`arm_state` values: `navigation`, `pre_pick`, `moving_to_navigation`, `moving_to_pre_pick`, `detecting`, `picking`, `checking`, `returning`, `arm_out`, `unknown`.

- It is the running job if there is one. Otherwise it is `arm_out` if an arm is still away from its start, else the pose both arms are at (within 0.05 rad), else `unknown`.
- A change shorter than 0.2 s may not appear.

---

## 6. The GUI page

Open `http://beast:8080` from any PC on the network. The page gets its state from server-sent events (`/api/events`, 5 times a second). The picture is an MJPEG stream (`/stream.mjpg`, 15 fps). Buttons call `POST /api/action`. Add `?still` to the URL on a slow link (one still per second).

Over the picture:

- Chips: arm state, `LIVE` / `CAPTURED` / `CALIBRATION` (`· DEPTH` or `· PICKER'S VIEW` when on), `FRONT` / `TOP DOWN`, `STOPPED`, `CLICK ONLY`, the camera's rate (`15 FPS`; red under 5).
- The Live/Captured switch and **STOP** (Esc).
- The status line along the bottom.

Tabs:

| Tab | Controls |
| --- | --- |
| Pick | Navigation (N), Pre-pick (P); Gripper: Vacuum · left / Finger · right; the target card; the **Box picker** card (below); the result card (with the check's numbers and a FLAG line); **Move** (M; reads **Try again** after a failed pick), **Detect box** (Space; with **Bypass** on -- the default -- a box the reach check passes is picked at once, no Move), **Pick all** (every box in view, one after another: detect, pick, drop, the next -- one try per box: a box that cannot be picked, or that the camera says is still in its place, is skipped with the picker's skip (flagged first when the cup touched it; one dropped off its place is not skipped -- the picker skips by the place in the picture); while the arm takes a box to the drop, as soon as it and the box are out of the camera picture, the pick is checked and the next box detected and planned from pre_pick, skipping what cannot be picked; to clear all it can, when no box is left but some were skipped and others taken meanwhile it goes round again (up to 3 rounds; a box out of reach for good is skipped at once, without a reach check); no base_adjust; ends when a round takes none, on STOP, an arm out, or a box still on the cup; each box's time and the total in the Pick all card, the log and native/logs/pick_all/*.json + .csv, and a banner + chime when it ends), Recapture (R), Return arm (B, only when an arm is out); Held object: H/W/D, Dummy size, Release; tool length toggles and steppers. Approach, Teach and ArUco ext are shown only with `--click` |
| Calibrate | Live markers, the arm model table, Cal arms, Overlay, Anchor, Refit, Weigh tool, Apply weight, Auto calibrate (box), Depth capture/fit, Tip check, camera pose Calibrate/Apply |
| Advanced | **Vacuum** On / Off: by hand, at any time, also mid-pick (Off drops a box on the cup); the same `vacuum` topic a pick switches, with a warning when no ESP32 bridge listens. **Box picker**: Depth picture (D), Picker's view (V), order v1 / v2, − / + for frames per target, SAM confidence and frames per pick check, Look again, Forget skipped; Move to coordinates: X/Y/Z (camera frame, m), Front/Top, Left/Right, **Check target**, and a second **Move** |
| System | The ROS topic values and sample commands |

The Activity list (collapsible, newest first) shows the same lines as `/beast_arm/log`. A refused button press also shows as a toast at the top.

While a job runs, every button is disabled except STOP, Live/Captured, markers, Dummy size, Tip check, the depth and picker's views, Save picture, the picker's settings and Forget skipped.

### 6.1 The box picker's own window, in the page

Everything the picker's window (`box_picker/src/pick_loop.py`) shows and does, on the Pick and Advanced tabs. Nothing here moves an arm.

| Piece | Shows / does |
| --- | --- |
| Tilt badge | The camera tilt the picker uses and where it comes from: `ACCEL` (the accelerometer now, or its saved reading) or `TF` (orange: not gravity's; install the RealSense udev rules) |
| Tally | Picked, failed tries, skipped (and passed over) since the GUI started |
| Close-up | The target box cut out, its pick face boxed in red; the grip point in camera coordinates (X right, Y down, Z away, cm); the tries on this box as dots |
| Lines | `N of M frames agree · boxes · face A × B mm`; `found in … s · pictures … s · picker … s` |
| **Next target** (T) | Pass over this box (and every box under it) until the next successful pick, and detect again; press again for the one after |
| **Skip box** | Never choose this box again until Forget skipped, and detect again |
| **Check now** (H) | The pick check now, as if the robot said done: for a box taken by hand, or to look again. Counts in the tally |
| **Save picture** (S) | The colour + depth now, with the tilt, in `box_picker/src/outputs/gui/<run>/snapshots/`, for `pick_pipeline.py --capture` |
| History strip | The last 12 picks' close-ups, framed green (picked), red (not) or orange (can't tell) |
| Result card | The check's numbers: gone %, same %, another box %, depth or colour, frames used, disturbed neighbours, boxes counted around it; FLAG RAISED with the photos' folder when the stack gave up on a box |
| Depth picture (D) | The depth instead of the colour, live and captured: red near, blue far (0.30 to 0.65 m), black no reading (the bands at the sides have no depth). The target stays drawn on it |
| Picker's view (V) | In the captured view, the picker's own drawing: every box, its faces and grip points; after a pick, the check's picture (red gone, green unchanged, blue nearer, purple another box) |
| Order | v1: boxes on a box first, the highest first, then nearest, then leftmost (piles). v2: the leftmost free box (rows). Changing it detects again |
| − / + | Frames per target (1-15), SAM confidence (0.20-0.90), frames per pick check (1-10): from the next detection / check on |
| Forget skipped | The skipped and passed boxes can be chosen again |

---

## 7. Poses: navigation, pre_pick, return

From the GUI (Navigation / Pre-pick / Return arm) or ROS (`arm_state/cmd`):

```bash
ros2 topic pub --once /beast_arm/arm_state/cmd std_msgs/msg/String "{data: pre_pick}"
```

### 7.1 `pre_pick` / `navigation`

| Step | What happens | Published |
| --- | --- | --- |
| 1 | Refused if a job runs | `arm_action/status` = `busy` |
| 2 | Clears a STOP | - |
| 3 | If an arm was left out by a pick, it is brought back first (as `return`) | phases of the return |
| 4 | Both arms move: the right arm first, then the left; an arm already there is skipped. cuMotion plans; OMPL if cuMotion finds nothing | `arm_state` = `moving_to_pre_pick` / `moving_to_navigation`; `phase` = `both to_pre_pick` / `both to_navigation`; log `pick mode: both arms to pre_pick...` |
| 5 | Done | `arm_state` = `pre_pick` / `navigation`; `phase` = `idle`; log `pick mode: Waiting for ready_to_pick.` |

Cases:

- **move_group down:** `/get_planning_scene` does not answer within 3 s, so nothing moves and the log says why.
- **A planning failure:** the arm stays where it is and `arm_state` shows the pose it is actually at, or `unknown`.
- **Both arms always move,** whatever the tool. Only the pick itself uses one arm.

### 7.2 `return`

This brings back an arm a pick left out (`arm_state` = `arm_out`):

1. Straight out along the tool.
2. Level again (for a top pick).
3. Back along the way it came, or a planned path if that fails.

While it runs, `arm_state` = `returning`. If there is nothing to return: log `nothing to return from: no arm is out`.

### 7.3 `status` and `arm_state/get`

`status` republishes `arm_state` at once. The service `arm_state/get` returns the current state in `message`.

---

## 8. Tool and box size

### 8.1 Tool (gripper)

| Way | Effect |
| --- | --- |
| GUI Gripper switch | `vacuum` → left arm only, `finger` → right arm only. Log `<tool> picks: the <arm> arm only - ...`. Clears the detected box and any Try again |
| `ready_to_pick ... vacuum` / `finger` | Sets the tool for this and every later pick |
| `pick_tool:=` at launch | The starting tool (default `vacuum`) |

The other arm is never moved by a pick. A target that names the wrong arm is still picked with the tool's arm, and the log says so.

### 8.2 Box size (optional; this is what Dummy size sends)

| Way | Effect |
| --- | --- |
| GUI H/W/D fields | Set the dummy size (5-600 mm each; default 170 × 100 × 25) |
| GUI **Dummy size** on | Publishes `held_object/size` (ObjectSize, `frame_id: dummy`) every second |
| GUI **Dummy size** off | Publishes zeros once (= no object) |
| Another system | Publishes `held_object/size` itself |

With a size known:

- **Face matching:** the measured face is matched to the box's faces (each edge within 15 mm or 12 %; a top seen at a slant within 30 mm or 30 %). Log: `face seen 168 x 99 mm = the 170 x 100 face of the 170x100x25 mm box (2 mm off): 25 mm deep along the tool`. `pick_target.box_mm` is filled.
- **Front faces too narrow for the cup:** a front face narrower than 45 mm on a box at least 60 mm deep is picked from its **top** instead. Log: `the 28 mm front is too narrow for the cup ...: picking its 170 x 100 mm TOP instead`. Without a size, a flat box is gripped on its narrow front.
- **Holding in the planners:** at contact the box is attached to the arm's tool in MoveIt and in cuMotion, so the way back keeps it clear of everything. `held_object/state` = `left 170x100x25 mm`. It is released at the drop pose (`none`).
- **Holding blocks the next pick:** while an arm holds a box, a new pick with it is refused. Use **Release** in the GUI or `held_object/release`. Log `held box released from the planners`.

---

## 9. Detect box

GUI: **Detect box** (Space). It is hidden with `--click`.

### 9.1 Steps

| # | What happens | Published / received | Log |
| --- | --- | --- | --- |
| 1 | Refused if busy | toast `busy` | - |
| 2 | Clears a STOP and any Try again | - | - |
| 3 | Both arms to pre_pick if they are not there (7.1) | `arm_state` = `detecting` throughout; phase `both to_pre_pick` | - |
| 4 | Waits up to 3 s for the model service | - | none: `no box model (/beast_arm/box_model/detect) - is box_model_node.py running?` |
| 5 | **In parallel:** (a) the capture and (b) the model call | | `DETECT BOX: the box picker is looking (5 frames)...` |
| 5a | **Capture:** waits for camera frames, averages 8 depth frames, rebuilds the 3-D map, then puts the green walls (`unseen_walls`) where the camera cannot see: above, below and beside the part of the picture that has **depth** (about 45 px on the left, 60 px on the right and the top 9 rows of the colour picture have none, so an object there would be invisible), from 11 cm in front of the nearest thing seen but never nearer than x 0.30 m. Launch `walls_edges:=picture` / `walls_ahead:=0.05` for the walls as before 2026-11-05 (more reach at the sides, less safe) | calls `/clear_octomap`, then `/octomap_gater/refresh` (3 frames into the map), then `/apply_planning_scene` | `frame captured; averaging 8 depth frames...`, `frame captured; rebuilding the octomap...`, `octomap rebuilt (passing 3 frames); walls around the picture (138).` |
| 5b | **Model:** calls `box_model/detect` with approach `auto` and the tool, waiting up to 90 s | request `DetectTarget{approach: auto, tool: vacuum}` | - |
| 5b-i | The box model drops stale picker output, takes "up" (the accelerometer, else the reading the picker's bridge sends with the frames, else the saved reading, else the TF), sends `pick ux uy uz` to the picker | `box_model/settings` (the tilt and its source) | without the accelerometer, once: `no accelerometer readings ...: the tilt from its saved reading (...)` or `... from the TF, which is not gravity's` |
| 5b-ii | The picker takes 5 frames and finds every box's outline and faces. It chooses the box: nothing on it and nothing in front first, then the highest standing on another box, then the nearest, then the leftmost. It chooses the face: the largest visible front or top face. A top seen at under 10° loses to its front | - | - |
| 5b-iii | The box model turns the answer into a world point (with the depth correction if one is saved) and a PickTarget | publishes `box_model/target`, `box_model/image`, `box_model/scene` | `box: top, 5 of 5 frames agree, face 168 x 120 mm, 5 boxes seen, at x=+0.438 y=+0.013 z=+0.422 (2.15 s); vacuum -> left arm` |
| 6 | The GUI pairs the answer with the picker's picture (up to 2 s) and switches to the captured view with the outlines drawn | receives `box_model/image`, `box_model/scene` | - |
| 7 | `prepare_target`: tool and arm, face size, size matching, the top/front fixes (9.3) | publishes `pick_target` | `face seen ...`, `top seen ... aiming N mm nearer`, `... picking its ... TOP instead` |
| 8 | Marks the point | publishes `/click_to_move/marker` | `x=+0.438 y=+0.013 z=+0.422 (depth 0.39 m), left arm - checking it can be touched...` |
| 9 | **Reach check** (9.2) | - | `left reach check: 16.9 s (quick reach ..., IK ..., arm model ..., straight lines ..., path planning ...)` |
| 10a | Reachable: the target card says *reachable* (or *edge of reach*) and **Move** is enabled | - | `... the left arm can touch it [top-down lean 20 yaw +90, from 6.0 cm] - press MOVE` then `TOP pick of the box shown - press MOVE to pick it` |
| 10b | Not reachable from here: the target card says *not reachable*; Move is disabled | publishes `base_adjust` and a `pick_result` with `outcome: unreachable` | `... the left arm cannot touch it -- <why>`, `the arm cannot reach this box from here: base_adjust sent - move left 0.100 m; detect the box again once the base has moved` |

Detect does not publish `arm_action/status`; that topic is for the pick (Move) and `ready_to_pick`.

### 9.2 The reach check (nothing moves)

It looks for one plan that works all the way, checking against the 3-D map:

1. **Quick reach:** is the point within the arm's reach at all?
2. **IK:** for each tool direction (front: around the face normal; top: straight down or leaned up to 30°), is there an arm posture 3 cm short of the box?
3. **Approach points:** from 10 cm out (front) or 12 cm (top) down to 4 cm, in 1.5 cm steps, the furthest that works. The arm model corrects each pose for the arm's sag and play.
4. **Lines:** the straight line in to 3 cm short (every link checked) and the press line past the surface (only the tool may touch the map; 2 cm past, else 1 cm, else 3 mm). Also the room to correct: the line in still solves with its ends moved 2.5 cm sideways.
5. **Path planning:** a collision-free path from pre_pick to the approach point. For a top pick the tool arrives level and turns down up there.

It stops at the first plan that works. The budget is 30 s (`--plan-budget`).

| Result | Why / message |
| --- | --- |
| Reachable | `the left arm can touch it [<posture>, from <N> cm]` |
| Reachable at the edge | No room to correct sideways. Tried when the face's short side is at least 25 mm (`--edge-face-mm`); log `... but it is at the edge of reach` |
| Too near the camera | `only 0.29 m from the camera - its depth is not reliable under 0.35 m` (`--min-depth`) |
| Edge only, face too narrow | `at the edge of the arm's reach (no room there to correct the tool, and corrections diverge) - and the 20 mm face is too narrow to risk it` |
| Out of reach | `out of reach: N cm beyond what the left arm can reach`, or `out of reach from the front (no arm posture gets there)` |
| Wrist cannot point down | `the wrist cannot point the tool down there, even leaned up to 30 deg -- use the front approach` |
| Lines blocked | `the straight line in from the front is blocked, or runs the wrist out of range part-way, from every approach point 10 to 4 cm out` |
| No path | `no collision-free path to the approach point (something is in the way)` |
| Budget used | `no way to pick it from above found in 30 s (--plan-budget) - not from here` |
| The check itself failed (move_group down, an error) | `could not check (<error>)`. No `base_adjust`: it is not the robot's position |

Every "not from here" result sends `base_adjust` at once (11.4).

### 9.3 Target fixes before the reach check

| Case | Fix | Log |
| --- | --- | --- |
| Top face seen at a slant, longer along the view than the box (size known) | Aim nearer the camera by half the excess (3-40 mm). The far edge blurs; the near edge is sharp | `top seen N mm long along the view ...: aiming N mm nearer` |
| Front face under 45 mm, box at least 60 mm deep (size known) | Pick the top: front centre + half the height up + half the depth back | `... picking its L x D mm TOP instead (from the front face and the box size)` |
| The model named the other arm | The tool's arm is used | log line about the correction |

### 9.4 Detect cases

| Case | What you see |
| --- | --- |
| No box | `no box: no box to pick (2.1 s)`; nothing to pick, Move disabled |
| The frames disagree | `no box: the frames disagree (2 of 5: something moved) - ask again` |
| Picker still loading | `the box picker is still starting - try again` |
| Picker did not answer | `the box picker did not answer in 60 s` |
| Picker crashed | `the box picker stopped while looking: ... - it restarts on the next try` (restarted once at the next request) |
| No TF to the camera | `no tf world -> the camera` |
| Model service missing | `no box model (...) - is box_model_node.py running?` |
| Not reachable | 9.2; `base_adjust` sent |

---

## 10. Move: the pick itself, step by step

GUI: **Move** (M) after a Detect box that found a reachable box.

- If the point was clicked (`--click`) or typed (Advanced tab), Move only touches it. That path has no camera check, no `pick_result` and no retry.
- If the last pick failed, the button reads **Try again** (11.3).

### 10.1 Start

| Step | What happens | Published |
| --- | --- | --- |
| 1 | Refused if busy, if nothing was detected (`pick a point or DETECT BOX first`), or if the box is not reachable (`not reachable: <why>`) | toast |
| 2 | Clears a STOP. Both arms to pre_pick if needed | phase `both to_pre_pick` |
| 3 | Start | `arm_state` = `picking`; `arm_action/status` = `picking` |
| 4 | Last guards: an arm still out (`... still out ... send "return"`), a box still held (`still holds the box from the last pick - release it first`), no joint states, the wrong arm for the tool, STOP on | log; nothing moves |
| 5 | Uses the plan from the reach check if the arm has not moved since; otherwise checks again (`checking the whole touch before moving...`) | - |
| 6 | Sends the tool pose aimed for at contact | `/click_to_move/target` |
| 7 | Log `left arm: planned in 0.0 s [<posture>]` (`; AT THE EDGE OF REACH - a correction may not fit` for an edge plan) | - |

### 10.2 Out to the box

| Phase (`arm_action/phase`) | What happens | Log | Cases |
| --- | --- | --- | --- |
| `left approach` | Flies the planned path to the approach point: full speed, collision-checked | - | Fails: `the approach did not complete` → returns |
| `left turn_down` (top only) | Arrived level above the box; turns the tool down (slow line, else a joint move) | `above the box; turning the tool down` | Fails: `could not turn the tool down here - coming back` |
| `left correct` | Settles. The encoders say where the tip is against where it should be; it moves to cancel the sag (sideways only along the way in; 1-2 corrections, top 2) | `the approach: settled in 1.2 s, joints N mrad from command, fingertip 8.3 mm from target`, `8.3 mm off at the approach; correcting` | Worse after a correction: `correction made it worse (...); going back`. Over 4 cm (top 6 cm) off: `too far for a correction; not correcting` |
| (same) | Camera check of the tip if available; for top picks the gripper pieces if they are on; else the arm model | `pieces not seen above the box; using the arm model` | - |
| (same) | Re-aim with the arm model where the contact will happen | - | `re-aim: only N% of the straight line solves` (warning; it goes on from where it is) |
| `left in` | Straight line in to 5 cm short, every link checked, 8 cm/s; then corrections at 5 cm (up to 3, within 4 mm) | `5 cm short: settled ..., fingertip N mm from target`, `N mm off at 5 cm short; correcting` | Line does not solve: tries a step instead; else `could not touch -- the straight line in to 5 cm short did not solve` → returns |
| (same) | Go-in rule: if still off by more than `--max-miss` (10 mm), or more on a wide face (up to 25 mm), it does not go in | `still N mm off the target 5 cm short and it cannot be corrected (edge of reach?) - not going in; coming back` | → returns; `pick_result` outcome `no_contact` |
| (same) | Line on to 3 cm short; if it does not solve, the same point with the tool allowed near the box, slowly | `no straight line to 3 cm short - going to 3 cm in front of the box, the tool allowed near it, then in` | Still nothing: `could not touch -- no straight line in from 5 cm short, ...` |
| (same) | Last sideways correction at 3 cm short (2 tries) | `3 cm short: settled ...` | - |

### 10.3 The press (contact)

This loop runs up to 3 presses (1 + `--press-retries` 2).

| Phase | What happens | Published / log |
| --- | --- | --- |
| (still) | Holds still and records the joint torques (for the weighing) | - |
| `left contact` | **Vacuum on** | `vacuum` = `true`; log `vacuum: on (left arm)` |
| (same) | Works out the press line from where the arm is: to 2 cm past the measured surface with the corrections. If that does not solve: 1 cm past, then 3 mm, then with half and with none of the sideways correction | `the press line solves only 10 mm past the surface - pressing so` (when it falls back) |
| (same) | Slow guarded line (vacuum tool 1 cm/s), only this arm's tool allowed to touch the map. It watches the joint torques and the tip's progress every 10 ms | trace appended to `native/logs/guarded_lines.jsonl` |
| (same) | **Contact** is a torque rise on any joint (0.3 Nm or more against 0.3 s before) AND the tip stuck: under 1 mm of progress in 0.35 s while the command runs on. Or a torque limit (1 Nm near the box with the vacuum tool, 2 Nm anywhere), or blocked (never got going while the command ran 2 cm ahead). The arm is then held where it is | `left contact: contact (openarm_left_joint3 +0.33 Nm in 0.3 s, stuck 34 of 30 mm in)` / `left contact: stopped at the torque limit (openarm_left_joint1 1.09 Nm, max 1.0, 32 of 30 mm in)` / `left contact: contact (blocked: N mm commanded, M mm moved)` |
| `left holding` | Holds 0.8 s (+0.5 s per re-press) | `touching (felt the surface); holding 0.8 s`; `last touch: felt the surface; tip 6.9 mm from the click sideways (x +7 y +1 z +0 mm, arm model), ...; 18.7 s to touch` |
| (same) | Box size known and a real touch: the box is attached in the planners | `held_object/state` = `left 170x100x25 mm`; log `holding the box (170 x 100 x 25 mm, dummy) - the planners keep it clear until the drop` |
| `left off_surface` | Backs 1 cm past the 3 cm point at 3 cm/s, then onto it again at the same speed, so the joints' friction matches the "before" reading | - |
| (same) | **Weighing:** the torque difference before and after, at the same spot, in grams | `the box is on the cup (96 +- 4 g)` / `nothing on the cup (-60 +- 4 g)` / `the torques are too unsteady to weigh the box ... - the camera will tell` / `could not weigh what the cup holds - the camera will tell` |

What happens after each press:

| Case | Next |
| --- | --- |
| Touched within 1.5 cm of the measured surface, and the box weighs on (or can't tell) | Done pressing → back out (10.4) |
| Touched within 1.5 cm, nothing on the cup | Press again, 5 mm deeper and 0.5 s longer: `nothing on the cup - pressing again, 5 mm deeper and 0.5 s longer (1/2)`. The held box is taken out of the planners first |
| Stopped more than 1.5 cm before the measured surface | Not believed whatever the weighing says; press again: `stopped N cm before the measured surface`, `stopped N cm short of the box - pressing again, 5 mm deeper ...` |
| No contact felt, and the line ended more than 5 mm above the surface (the arm lags about 8 mm at the edge of reach) | Press again, as much deeper as it was short plus 5 mm: `the line ended 5 mm above the surface - pressing again, 10 mm deeper ...` |
| No contact felt, the line went past the surface | The weighing decides: nothing on the cup → press again |
| The cup pressed beside the face (off centre by more than half the face) | No more presses: `the cup pressed N mm from the centre of the N mm face - beside it; not pressing again` |
| Front pick, something pushes up on the cup (under -60 g) | No more presses: `something pushes up on the cup (-224 g) - not the box face; not pressing again` |
| 3 presses used | `still nothing on the cup after 3 presses` (or `still N cm above the measured surface after 3 presses`) → back out; the camera decides |
| The press line does not solve, even with the fallbacks | `could not touch -- the press line onto the box does not solve from where the arm is - ...` → back out; outcome `no_contact` |
| The press could not run again from there | `could not press again from here` |

### 10.4 Back home with the box (gentle)

From the first press until the drop, every move is gentle: straight lines at 3 cm/s (`--held-line-speed`), joint moves and the retrace at 40 % speed and 16 % acceleration (`--held-velocity 0.4`). At full speed the yank swung the box into its neighbours.

| Phase | What happens | Log | Cases |
| --- | --- | --- | --- |
| `left back_out` | Straight out along the tool to the approach point. A held box may overlap the old map here, where it still stands | `backing straight out` | The pose counts as colliding: `backing out along the way in (model says this pose touches something)` (unchecked line), else a joint move |
| `left box_clear` (size known) | Further out until the whole box has left its old spot (its length along the tool + 3 cm) | `lifting the box clear (N cm off the contact)` | `could not take the box further out along the tool; the planners may see it in its old spot` |
| (same) | Box held: the 3-D map is taken again without the box in its old spot | `box lifted - the camera map again, without the box in its old spot`; calls `/clear_octomap` and `/octomap_gater/refresh` | - |
| `left to_pre_pick_top` (top, after a press) | The tool stays pointing straight down, so the box stays level: to **pre_pick_top**, beside pre_pick (tool at 0.13, ±0.27, 0.46 m: the held box, any face up and turned any way, stays 3 cm behind where the green walls can start; `pre_pick_top_state` in `robot/pick_place_states_<arm>.yaml`). A straight line at 5 cm/s (`--top-carry-speed`), the tool turned only about the vertical, by the least that reaches; then straight on to `to_drop` (no `turn_level`, no `return`) | `to pre_pick_top in a straight line, the tool still pointing down (the box stays level)` | No straight line: `to pre_pick_top (planned), the tool pointing down` (joints found from where the arm is, 40 %); then `pre_pick_top_state`'s own joints; nothing reaches: `no way to pre_pick_top - turning the tool level ...` and the rows below as before |
| `left turn_level` (top, only without pre_pick_top) | Turns the tool level again, up there | `turning the tool level` | Fails with a box: `turning level with the box let through the camera map` (tried again with the box exempt) |
| `left return` (front; top without pre_pick_top) | Back to pre_pick along the way it came (replayed in reverse at 40 %), if that way is clear with the box on | `returning the way it came` | Too tight with the box: `the way in is too tight with the box - planning a way back` (cuMotion/OMPL). Nothing works with a box: `no planned way home with the box - retracing the way in` (box exempt) |
| `left to_drop` | To the drop pose (40 %), from pre_pick or pre_pick_top: the tool points down there too | `to the drop pose` | Unreachable: `could not reach the drop pose; back to pre_pick with the box` (vacuum stays on) |
| `left drop` | **Vacuum off**; holds 1 s (`--drop-hold`); the box leaves the planners | `vacuum` = `false`; `held_object/state` = `none`; logs `at the drop pose; holding 1 s`, `vacuum: off (left arm)`, `box dropped - out of the planners` | - |
| `left to_pre_pick` | Back to pre_pick at normal speed | `back to pre_pick` | - |
| `left done` | Back | `left arm: done in 45.1 s. Detect a box, then MOVE.` | Not back: `could not get back to the start posture -- press RETURN (b) to try again`; `arm_state` = `arm_out` |

If anything throws on the way: `touch failed (<error>); bringing it back`, and the return runs anyway. When the arm is back with nothing held, `vacuum` = `false` is published (if it was on).

---

## 11. After the pick: the camera check, retries, base_adjust

### 11.1 The camera check

| Step | What happens | Published / received |
| --- | --- | --- |
| 1 | Decide before the check, from the touch | see 11.2 |
| 2 | `arm_state` = `checking`; `arm_action/status` = `checking`; phase `left check` | - |
| 3 | Calls `box_model/check` with `holding` = `held` / `empty` / `""` from the weighing (25 s timeout; once more after 1.5 s if `blocked`) | request `CheckPick{holding: held}` |
| 4 | The picker takes 3 new frames and compares them with the 5 frames from the detection: depth and print inside the box's outline, its neighbours too. No SAM, milliseconds, unless another box now stands in the target's place: then SAM runs once to count the boxes around it | box model publishes `box_model/check_result`, `box_model/image`, `box_model/scene` (kind `check`) |
| 5 | Result | `pick_result` (JSON); `arm_action/status` = the result; log `pick check: PICK SUCCESS: the box is gone from its place` |
| 6 | GUI: shows the check picture with the verdict banner, captures a fresh frame, fills the result card | - |

Check outcomes, decided in this order:

| Outcome | Rule | `picked` | Status |
| --- | --- | --- | --- |
| `camera_moved` | Background changed over 25 % | `null` | `pick_unverified: camera_moved - ...` |
| `blocked` | Something in front, or the outline unsteady | `null` | `pick_unverified: blocked - ...` |
| `unknown` | Under 50 pixels of depth on the target | `null` | `pick_unverified: unknown - ...` |
| `success` | 70 % or more of the box gone | `true` | `pick_complete` |
| `replaced` → `success` / `moved` | Another print at the target's depth (30 % of it) and a neighbour moved: a neighbour slid or leaned into the gap. With `held` it is `success`, with `empty` `dropped`; otherwise SAM counts the boxes around the spot: one fewer → `success`, as many → `moved` | `true` / `false` | `pick_complete` / `pick_failed: moved - ...` |
| `dropped` | Gone, but the weighing said empty | `false` | `pick_failed: dropped - ...` |
| `wrong_box` | Target 80 % the same, a neighbour 70 % gone | `false` | `pick_failed: wrong_box - ...` |
| `missed` | Target 80 % the same | `false` | `pick_failed: missed - PICK FAILED: MISSED: the box is still there` |
| `moved` | Anything else (pushed, tipped, part way) | `false` | `pick_failed: moved - PICK FAILED: BOX MOVED: ...` |

### 11.2 Results decided without the camera

| Case | `outcome` | `picked` | Status | Counted? Retried? |
| --- | --- | --- | --- | --- |
| The arm did not get back | `arm_out` | `null` | `pick_failed: the left arm did not get back - send arm_state/cmd "return"` | No / no |
| Refused before the arm moved | `not_started` | `null` | `pick_failed: not started - <why>` | No / no |
| STOP before the arm moved | `stopped` | `null` | `stopped` | No / no |
| The arm never touched the box | `no_contact` | `false` | `pick_failed: the left arm did not reach the box (<why>)` | Yes / yes |
| Check not served (picker down) | `success` / `no_contact` / `empty` from contact and weight | `true`/`false` | `pick_complete` / `pick_failed: ...` | - |
| Check did not answer | `check_timeout` | `null` | `pick_unverified: ...` | No / no |
| Not reachable from here | `unreachable` | `false` | (Detect, or `pick_failed: <why>; base_adjust requested: ...`) | base_adjust at once |

`pick_result` JSON:

```json
{"picked": true, "outcome": "success", "message": "PICK SUCCESS: the box is gone from its place",
 "contact": true, "held": true, "held_g": 96.0, "tool": "vacuum", "arm": "left", "approach": "top",
 "xyz": [0.438, 0.013, 0.422], "box_mm": [170, 100, 25], "time": "2026-10-30 12:24:55",
 "verdict": {"outcome": "success", "gone": 0.94, "same": 0.03, "...": "..."},
 "base_adjust": {"direction": "left", "distance_m": 0.1, "left_m": 0.1, "failures": 2, "reason": "missed, moved", "goal": [0.45, 0.10]}}
```

`base_adjust` appears only when one was sent. `held` / `held_g` appear only when the arm weighed.

### 11.3 Retries: automatic, and the Try again button

After a failed pick (`picked: false`) on the same box (within 5 cm), with no `base_adjust` sent and the arm back:

1. **Automatically**, once:
   - Log `not picked (no_contact) - trying again by itself (2/2)`.
   - `arm_action/status` = `retrying: <outcome>` for the failed attempt (not a final result).
2. **Try again** (the same thing, also the Move button when it reads *Try again*):

   | Step | What happens | Log |
   | --- | --- | --- |
   | a | `arm_state` = `checking`: the camera looks at the box again (`box_model/check`) | `TRY AGAIN (last: missed): is the box as it was?` |
   | b | Gone now (after a can't-tell result) | `... it was picked`; `pick_complete` |
   | c | Unchanged (`missed`) | `the box is unchanged: picking it again` → reach check → pick (10) |
   | d | Changed | `the box changed (...) - detecting it again` → Detect (9) → the same box within 5 cm: `found it again (N mm ...)` → pick |
   | e | Changed, and the picker now chooses another box | `pick_failed: the picker now chooses another box - press MOVE for it` |
   | f | Nothing found any more | `no_target: ...` |

The failure count is kept across commands. It resets on a success, after a `base_adjust`, or when a pick is on another box. The log after each failure says `pick failed N time(s) on this box - at 2, the base is asked to move (base_adjust)`.

### 11.4 base_adjust

Sent:

- **at once** when the reach check says "not from here" (9.2), with `reason: unreachable` and `failures: 1`;
- **after 2 failed picks** in a row on the same box (`--adjust-after 2`), with `reason` = the outcomes, e.g. `no_contact, missed`.

| Field | Value |
| --- | --- |
| `header.frame_id` | `world` (the robot body: x forward, y left) |
| `direction` | `left` or `right`: move sideways only, never forward, back or a turn. `none`: the box is already lined up sideways (within 1 cm), so left or right cannot help (it is too near or too far) |
| `distance_m`, `left_m` | How far, m; `left_m` signed (+ left, − right) = the box's y minus the arm's sweet spot y (`--sweet-spot` 0.10 m for the left arm) |
| `target`, `goal` | The box now, and where the arm wants it |
| `arm`, `tool`, `approach`, `failures`, `reason` | As named |

- **After it is sent:** the arms do nothing more. Log `...: base_adjust sent - move left 0.100 m; detect the box again once the base has moved`, or `... - none: the box is already lined up sideways, and moving left or right cannot help (...)`.
- **Not latched:** a wheel node that starts later never acts on an old request.
- **After the base moves:** Detect box (or `ready_to_pick`) again.
- **The box is flagged** (after failed picks, not for `unreachable`): three photos for a person (the scene with it marked, close up, the last check) in `box_picker/src/outputs/gui/<run>/flags/`; the result card shows FLAG RAISED.
- **`none`: the box is skipped**, as the picker's own loop does: the next detection takes another box (Forget skipped brings it back). **`left` / `right`:** the skipped boxes are forgotten instead, because they are remembered by their place in the picture.

---

## 12. A full pick from ROS: ready_to_pick

```bash
ros2 topic pub --once /beast_arm/arm_action std_msgs/msg/String "{data: ready_to_pick}"
ros2 topic pub --once /beast_arm/arm_action std_msgs/msg/String "{data: ready_to_pick top vacuum}"
ros2 topic echo /beast_arm/arm_action/status
```

This does Detect box and Move in one command, with the same steps and messages as sections 9-11, and it retries by itself.

| # | Step | `arm_action/status` | `arm_state` |
| --- | --- | --- | --- |
| 1 | Parse: `auto`/`top`/`front` and `vacuum`/`finger` in any order. `left`/`right` only warn if they disagree with the tool. A bad word: `pick_failed: "<word>" - ready_to_pick [top\|front\|auto] [vacuum\|finger]` | - | - |
| 2 | Busy: refused, nothing queued | `busy` | unchanged |
| 3 | `--click` mode | `pick_failed: click-only mode (--click), no box model` | - |
| 4 | Guards: an arm out, a box still held, STOP | `pick_failed: an arm is still out ...` / `pick_failed: the arm still holds the last box ...` / `stopped` | - |
| 5 | To pre_pick if needed | - | `moving_to_pre_pick` |
| 6 | Model + capture in parallel (30 s for the model) | `detecting` | `detecting` |
| 7 | Nothing found | `no_target: <message>` (end) | pose |
| 8 | Target prepared | `target_found`; `pick_target` published | `detecting` |
| 9 | Reach check fails | `pick_failed: <why>[; base_adjust requested: move left/right N m]` (end) | pose |
| 10 | The pick (10.1-10.4) | `picking` | `picking` |
| 11 | The camera check | `checking` | `checking` |
| 12 | Result | `pick_complete` / `pick_failed: ...` / `pick_unverified: ...` | pose (or `arm_out`) |
| 13 | Failed and can try again: once more from step 6 (it looks again) | `retrying: <outcome>` → back to `detecting` ... | |

Every `ready_to_pick` ends in exactly one of `pick_complete`, `pick_unverified: ...`, `no_target: ...`, `pick_failed: ...` or `stopped`. `retrying: ...` is never the end.

Other ends:

| Case | Status |
| --- | --- |
| No model service | `pick_failed: no model (/beast_arm/box_model/detect not served - is box_model_node.py running?)` |
| Model too slow | `pick_failed: the model did not answer in 30 s` |
| No camera frame for the map | `pick_failed: no camera frame for the octomap` |
| No depth at the target | `pick_failed: no depth at the target` |
| Pre_pick not reached | `pick_failed: could not reach pre_pick` |
| Any error | `pick_failed: <error>` |

A typical sequence from another system:

1. `arm_state` = `navigation` → send `pre_pick` on `arm_state/cmd` and wait for `arm_state` = `pre_pick`.
2. (Optional) publish the box size on `held_object/size`.
3. Send `ready_to_pick`.
4. Follow `arm_action/phase`; wait for a final `arm_action/status`; read `pick_result`.
5. On `base_adjust`: move the base sideways (`direction`, `distance_m`), then go to 3.
6. To drive away: send `navigation`.

---

## 13. Advanced tab: typed camera coordinates

For a point you measured yourself, in the colour camera's optical frame (x right, y down, z forward, metres, as the box picker gives it).

1. Enter X, Y, Z. Choose Front (default) or Top, and Left or Right (the arm sets the tool: left = vacuum, right = finger).
2. **Check target**:
   - Checks: Z between 0.15 and 2 m, |X| and |Y| at most 1 m, not busy.
   - Both arms go to pre_pick, a fresh frame is captured and the map rebuilt.
   - The point goes into the world through the camera's TF.
   - Log: `typed target: camera (-0.028, +0.154, 0.393) m = world x=+0.426 y=+0.011 z=+0.424, TOP - checking the reach`. Off the picture it adds a warning that the typed point is outside the camera picture.
   - The reach check (9.2) runs; "not from here" sends `base_adjust`.
3. **Move** (either Move button) touches the point with the full press, weighing, back out, drop and return (10).
   - It does **not** run the camera check, publish `pick_result` or `arm_action/status`, or retry: a typed point has no "before" picture of a box.
   - During the motion `arm_state` reads `arm_out` (no job name is set on this path).

---

## 14. STOP

GUI **STOP** (Esc), `stop` (Empty), or `arm_action` = `stop`. It always works, even while busy.

| Step | What happens | Published |
| --- | --- | --- |
| 1 | The arms that are moving get one trajectory point at their current joints (50 ms), re-sent until the goal ends. This is what really stops them; cancelling a MoveIt goal alone does not | `/<arm>_joint_trajectory_controller/joint_trajectory` |
| 2 | Every running goal is cancelled | cancel on `/execute_trajectory`, `/move_action` |
| 3 | The job ends at its next step | `arm_action/status` = `stopped`; `status.halted` = `true`; log `STOP: motion cancelled; the arms hold where they are. Any move clears it` |
| 4 | Every new motion is refused until a command that moves is sent: Move, Detect box, a pose, Return, `ready_to_pick`, `arm_state/cmd` | - |

Cases:

- **Vacuum:** STOP does not switch the vacuum off (a held box is not dropped).
- **Arm left out:** an arm stopped half-way is `arm_out`; send `return` or press Return arm.
- **During calibration or TEACH:** STOP also aborts that step.

---

## 15. Click-only mode (`--click`)

```bash
native/run_web_gui.sh --click
```

- **No box model:** Detect box is hidden. `ready_to_pick` answers `pick_failed: click-only mode (--click), no box model`.
- **Click to choose:** click the picture to choose a point. In the live view the arms go to pre_pick and a frame is captured first. Then the reach check runs and the point is marked.
- **Extra controls:** Approach Front/Top down (D), Teach (T) and ArUco ext become available.
- **Move touches** the clicked point (10, without the camera check or `pick_result`).
- **TEACH:** after the touch it holds. Nudge with the arrows, In+/Out- (1 mm, or 5 mm), then Save or Skip. A saved correction is learned for nearby points.

---

## 16. Calibrate tab

Each of these runs as one job (`busy` refuses others). STOP aborts it.

| Button | What it does |
| --- | --- |
| Live markers | Shows every ArUco marker with its id over the live picture |
| Cal arms | Arm model calibration, both arms, with the folded marker pieces on the grippers and marker 11 on the wall 50 cm in front. Confirm the overlay when asked; about 35-40 min |
| Auto calibrate (box) | Left arm, with the 170 × 100 × 25 marker box on the cup (cup on the cross of face B), arm at pre-pick. 60 postures (30 level, 30 top-down), about 15 min. Fits the arm model to the box readings (each needs at least 3 markers). It saves only if it beats the old model on postures it did not fit; the old one is kept as `old_version/calibration_backups/arm_model.before_boxcal_*.yaml` |
| Depth: capture / fit | The A4 ChArUco board at 3-5 places. Fits a depth correction, saved only if it leaves under 4 mm |
| Overlay | The robot model drawn over the camera picture (a check of the camera pose) |
| Anchor / Refit | Anchor: touch a point, nudge the tip onto it, Save (3-4 anchors at different places). Refit: the arm model again from the last Cal arms recordings plus the anchors |
| Weigh tool / Apply weight | Weigh tool (nothing on the cup): 10-20 still postures, each joint moved a little each way, giving the real mass and centre of mass of everything past the left wrist. Apply weight writes it into the URDF, so the hardware's gravity compensation is right after terminal 1 is restarted |
| Tip check | Where the camera sees the tip now |
| Camera pose: Calibrate / Apply | The camera's pose measured from the arms (after the camera was moved). Apply makes it the camera pose the robot uses (enabled only when a calibration is ready) |

During Auto calibrate (box) the picture shows the live markers and a `box read` line (markers seen, fit in px).

---

## 17. Example timelines

### 17.1 A good GUI pick (top, vacuum)

From the run of 2026-10-30 12:23-12:24; the messages are as the current code prints them.

```
12:23:46  [Detect box]   arm_state=detecting
          log  DETECT BOX: the box picker is looking (5 frames)...
          svc  /clear_octomap, /octomap_gater/refresh          (capture, in parallel)
          svc  box_model/detect {approach: auto, tool: vacuum}
12:23:50  pub  box_model/target, box_model/image, box_model/scene
          log  box: top, 5 of 5 frames agree, face 168 x 120 mm, 5 boxes seen, at x=+0.438 y=+0.013 z=+0.422
          pub  pick_target {approach: top, tool: vacuum, arm: left, face_mm: [168, 120], ...}
          pub  /click_to_move/marker
12:24:07  log  left reach check: 16.9 s (...)
          log  ... the left arm can touch it [top-down lean 20 yaw +90, from 6.0 cm] but it is at the edge of reach - press MOVE
          arm_state=pre_pick
12:24:09  [Move]  arm_state=picking  arm_action/status=picking
          pub  /click_to_move/target
          phase  left approach -> left turn_down -> left correct -> left in
12:24:24  phase  left contact     pub vacuum=true
12:24:28  log  touching (no contact felt) ... the line ended above the surface
          phase  left off_surface   log nothing on the cup ... pressing again, deeper
12:24:36  log  left contact: contact (openarm_left_joint3 +0.33 Nm in 0.3 s, stuck 34 of 30 mm in)
          phase  left holding
12:24:40  log  the box is on the cup (36 +- 3 g)
          phase  left back_out -> left to_pre_pick_top -> left to_drop
12:24:50  phase  left drop        pub vacuum=false
          phase  left to_pre_pick -> left done
12:24:54  log  left arm: done in 45.1 s.
          arm_state=checking  arm_action/status=checking  phase=left check
          svc  box_model/check {holding: held}
          pub  box_model/check_result, box_model/image, box_model/scene (kind check)
12:24:55  pub  pick_result {"picked": true, "outcome": "success", ...}
          arm_action/status=pick_complete   arm_state=pre_pick   phase=idle
```

### 17.2 A failure, the automatic retry, then base_adjust

```
[Move]              arm_action/status=picking ... phases ...
                    log could not touch -- the slow line onto the box did not solve (edge of reach?)
                    pub pick_result {"picked": false, "outcome": "no_contact", ...}
                    arm_action/status=retrying: no_contact
                    log pick failed 1 time(s) on this box - at 2, the base is asked to move (base_adjust)
                    log not picked (no_contact) - trying again by itself (2/2)
[Try again]         arm_state=checking   svc box_model/check
                    log the box is unchanged: picking it again
                    reach check -> arm_action/status=picking ... phases ...
                    (fails again)
                    pub base_adjust {direction: right, distance_m: 0.05, left_m: -0.05, failures: 2, reason: "no_contact, no_contact"}
                    pub pick_result {... "base_adjust": {...}}
                    arm_action/status=pick_failed: ...; base_adjust requested: move right 0.050 m
```

### 17.3 Not reachable at detection

```
[Detect box]  ... box: front, 5 of 5 frames agree, face 173 x 28 mm, ... at x=+0.480 y=+0.200 z=+0.427
              log the left arm cannot touch it -- <why>
              pub base_adjust {direction: left, distance_m: 0.100, left_m: +0.100, failures: 1, reason: "unreachable"}
              pub pick_result {"picked": false, "outcome": "unreachable", ...}
              log ... base_adjust sent - move left 0.100 m; detect the box again once the base has moved
```

---

## 18. Every case in one table

| Situation | You see (log / status) | `pick_result.outcome` | What to do |
| --- | --- | --- | --- |
| Box picked | `PICK SUCCESS`; `pick_complete` | `success` | Next box |
| No box in view | `no box: no box to pick` / `no_target: ...` | - | Check the view, or move the base |
| Frames disagree | `the frames disagree ... - ask again` | - | Detect again; keep the scene still |
| Picker loading / down | `still starting - try again` / `did not answer` / `stopped while looking` | - | Wait; see `native/logs/box_picker.log` |
| Too near the camera | `only 0.29 m from the camera ...` + base_adjust | `unreachable` | base_adjust moves only sideways: if it says `none`, the robot has to be moved back some other way |
| Out of reach / no posture / no path / budget | 9.2 messages + base_adjust | `unreachable` | Base moves left/right as asked; `none` = sideways cannot help |
| Edge of reach, narrow face (< 25 mm) | `... too narrow to risk it` + base_adjust | `unreachable` | Base moves left/right, or set the box size so a flat box is picked from the top |
| Approach failed | `the approach did not complete` | `no_contact` | Retried automatically |
| Too far off to go in | `still N mm off ... not going in; coming back` | `no_contact` | Retried; recalibrate if it repeats |
| Press line does not solve | `could not touch -- ...` | `no_contact` | Retried; then base_adjust |
| Stopped short / line ended above | `stopped N cm before the measured surface` / `the line ended N mm above the surface` → pressed again | - | Automatic |
| Nothing on the cup | `nothing on the cup` → pressed again (up to 3 presses) | - | Automatic; check the vacuum is on |
| Cup beside the face | `... beside it; not pressing again` | camera decides | Recalibrate if it repeats |
| Something pushes up (front) | `something pushes up on the cup ...` | camera decides | - |
| Box taken, a neighbour slid into its place | `PICK SUCCESS` (settled by counting the boxes there) | `success` | Next box |
| Box still there | `PICK FAILED: MISSED` | `missed` | Retried; then base_adjust |
| Box pushed / tipped | `PICK FAILED: BOX MOVED` | `moved` | Retried with a new detection |
| Neighbour taken | `PICK FAILED: WRONG BOX` | `wrong_box` | Retried |
| Box gone, cup empty | `PICK FAILED: DROPPED` | `dropped` | Find the box |
| View blocked / camera moved / too little depth | `CAN'T TELL ...`; `pick_unverified: ...` | `blocked` / `camera_moved` / `unknown` | Press Try again (it looks again) |
| Check did not answer | `pick_unverified: ...` | `check_timeout` | Check the box model |
| Two failures on one box | base_adjust sent; FLAG photos saved | (the second outcome) | Move the base left/right as asked, Detect again; with `none` the box is skipped and the next detection takes another |
| Arm did not get back | `could not get back ... press RETURN`; `arm_out` | `arm_out` | Return arm / `arm_state/cmd return` |
| Box still held from before | `still holds the box from the last pick - release it first` | `not_started` | Take the box off, press Release |
| Command while busy | `busy` | - | Wait, or STOP |
| STOP | `stopped`; STOPPED chip | `stopped` (if before the arm moved) | Any move clears it |
| move_group down | `could not check (...)` / `MoveGroupDown` | - | Restart terminal 1 |
| Controllers failed | `controller spawner failed 3 times ...` (terminal 1) | - | Check CAN, restart terminal 1 |
| Camera hung | no frames; Detect fails | - | Replug the camera, close other camera apps, restart terminal 1 |
| New nodes hear nothing | topics missing, services never answer | - | Stop the GUI, run `native/clean_dds_shm.sh`, start again |

---

## 19. Shutting down

1. Send the arms to navigation (GUI **Navigation**, or `arm_state/cmd navigation`).
2. Terminal 2: Ctrl-C. Or run `native/stop_pick_tools.sh` from any shell; it stops only the GUI and pick tools. The box model sends `quit` to the picker and stops the bridge.
3. Terminal 1: Ctrl-C. This stops the controllers, so have the arms at a safe pose (navigation) first.
4. CAN down if needed (sudo on beast): `ip link set can0 down; ip link set can1 down`.
