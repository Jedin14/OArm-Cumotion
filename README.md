# 7DOF-OArm IK Environment

Runs the 7DOF-OArm bimanual exoskeleton with GPU-accelerated cuMotion kinematics,
MoveIt 2, and real hardware interfaces.

There are two ways to run it. **Use the native one** — see
[Why native](#why-native-and-not-the-container) below.

---

## 🚀 Quick Start (native — recommended)

One-time setup, if this machine has not been set up yet:

```bash
./install.sh
```

That builds both project environments and verifies they are isolated from the
system's Python — see [Installation](#-installation) for what it does and does
not touch. Then, the one step that needs root:

```bash
sudo mkdir -p /workspaces && sudo ln -s /home/mr/workspaces/isaac_ros-dev /workspaces/isaac_ros-dev
```

### 1. Bring up the CAN interfaces

```bash
sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can0 up
```

```bash
sudo ip link set can1 down && sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can1 up
```

Check they came up (`ERROR-ACTIVE` is the normal healthy state):

```bash
ip -details link show can0
```

### 2. Launch the robot and planners

From a terminal **on the machine's own desktop session** — RViz and the MoveIt
depth-image octomap updater both need a real `DISPLAY`, so plain ssh will not
work:

```bash
cd /home/mr/workspaces/isaac_ros-dev && source native/setup.bash
```

```bash
native/run_launch_everything.sh
```

That is the whole thing — no container, no separate shell. It uses the same
launch arguments as before (`octomap:=static`, `use_fake_hardware:=false`,
`right_can_interface:=can0`, `left_can_interface:=can1`). Override any of them by
appending, e.g.:

```bash
native/run_launch_everything.sh use_fake_hardware:=true octomap:=live
```

You should see cuMotion warm up in about 3 seconds and print
`cuMotion is ready for planning queries!`, and `move_group` report
`MoveGroup context using planning plugin isaac_ros_cumotion_moveit/CumotionPlanner`.

### 3. Rebuild after editing code

```bash
native/build_ws.sh
```

Or a single package:

```bash
native/build_ws.sh --packages-select openarm_description
```

### 4. Editing in VS Code

```bash
code ~/workspaces/isaac_ros-dev
```

---

## 📦 Installation

```bash
./install.sh                 # everything (native cuMotion stack + VLM)
./install.sh --native-only   # skip the VLM detector environment
./install.sh --vlm-only      # only the VLM detector environment
./install.sh --check         # verify an existing install, change nothing
```

It runs a preflight (Python 3.10, ROS Humble, an NVIDIA GPU, free disk), then
[`native/bootstrap.sh`](native/bootstrap.sh) and
[`VLM/bootstrap.sh`](VLM/bootstrap.sh), then the isolation check. Every step is
idempotent, so re-running after editing a `requirements.txt` is fine and cheap.

### Three Python stacks, none of them the system's

This workspace needs two mutually incompatible Python environments, and the host
already has a third. Mixing any two produces either a numpy ABI error on import
or a silently wrong torch:

| Stack | numpy | torch | Why it is pinned there |
| --- | --- | --- | --- |
| `native/venv` | 1.26.4 | 2.7.0+cu128 | the ABI cuRobo's prebuilt CUDA kernels are linked against |
| `VLM/.venv` | 2.2.6 | 2.14 nightly +cu130 | what PaliGemma / `transformers` needs |
| the host | 1.21.5 (apt), 2.2.6 (`~/.local`) | 2.12+cu130 | not used by this project at all |

The separation is enforced, not just documented:

- **Nothing is installed system-wide.** `install.sh` never runs `apt install` and
  never writes to `/usr`, `/opt` or `/usr/local`. The ROS packages the host is
  missing are downloaded as debs and unpacked into `native/root`, a private
  prefix — the host's `/opt/ros/humble` is used read-only as the underlay.
- **The host's `~/.local` is invisible.** `PYTHONNOUSERSITE=1` is exported by
  both runtime wrappers and set for every `pip`/`uv` call, in both directions.
- **The two venvs never share an interpreter.**
  [`VLM/run_in_vlm_env.sh`](VLM/run_in_vlm_env.sh) scrubs `PYTHONPATH` and
  `LD_LIBRARY_PATH` before launching the detector, because a `PYTHONPATH` set by
  `native/setup.bash` would otherwise beat the venv's own `site-packages` and
  hand PaliGemma the wrong numpy.

Verify all of that at any time:

```bash
./check_isolation.sh
```

It checks that each venv resolves `numpy` and `torch` to its own pinned copies
from inside the workspace, that neither can see `~/.local`, that the two carry
different versions, and that nothing of the project leaked into the system's
`dist-packages`. A failure here is an early warning for a crash that would
otherwise surface mid-launch.

The only host-level change the workspace needs is the
`/workspaces/isaac_ros-dev` symlink, which `install.sh` prints rather than
runs. (Bringing the CAN interfaces up also needs root, but that configures
kernel network devices and is per-boot, not part of installation.)

---

## 🤖 VLM-guided pick and place

A PaliGemma detector finds the object, cuMotion plans to it, and the same
detector checks whether each step actually worked — a failed grasp is retried
with an escalating strategy rather than repeated.

```
LOCATE ──► PREGRASP ──► DESCEND ──► CLOSE ──► VERIFY_GRASP ──► LIFT
   ▲                                             │ failed         │
   └─────────────────────────────────────────────┘                ▼
                                       HOME ──► OVER_BOX ──► RELEASE ──► VERIFY_PLACE
```

With the robot already up (`native/run_launch_everything.sh`), in a second
terminal:

```bash
cd /home/mr/workspaces/isaac_ros-dev && source native/setup.bash
```

```bash
ros2 launch pick_place.launch.py prompt:="detect screwdriver" place_position:="[0.35, 0.30, 0.25]"
```

Wait for `model loaded`, then start a cycle:

```bash
ros2 service call /pick_place/start std_srvs/srv/Trigger
```

`ros2 topic echo /pick_place/state` follows the state machine, and
`ros2 service call /pick_place/abort std_srvs/srv/Trigger` stops it after the
current motion. `ros2 launch pick_place.launch.py --show-args` lists every tunable.

### The pieces

| File | Runs in | Does |
| --- | --- | --- |
| [VLM/vlm_detector_node.py](VLM/vlm_detector_node.py) | `VLM/.venv` | PaliGemma → 3D points in `world` on `/vlm/detections` |
| [VLM/run_in_vlm_env.sh](VLM/run_in_vlm_env.sh) | — | environment isolation for the above |
| [VLM/bootstrap.sh](VLM/bootstrap.sh) | — | builds `VLM/.venv` from `requirements.txt` |
| [pick_place_orchestrator.py](pick_place_orchestrator.py) | `native/venv` | the state machine, MoveIt + gripper + planning scene |
| [pick_place.launch.py](pick_place.launch.py) | — | starts both, one set of arguments |
| [VLM/pixel_to_world.py](VLM/pixel_to_world.py) | `VLM/.venv` | click any point, get its world coordinate |
| [native/tests/check_reachability.py](native/tests/check_reachability.py) | `native/venv` | ask MoveIt if the arm can get there |
| [VLM/test_vlm_geometry.py](VLM/test_vlm_geometry.py) | `VLM/.venv` | pixel → world maths, no camera or model |
| [native/tests/test_grasp_geometry.py](native/tests/test_grasp_geometry.py) | `native/venv` | grasp pose maths, no robot |

### Interfaces

| Name | Type | |
| --- | --- | --- |
| `/vlm/detections` | `std_msgs/String` | JSON, authoritative — see the node's docstring for the schema |
| `/vlm/detection_poses` | `geometry_msgs/PoseArray` | same points, for RViz |
| `/vlm/debug_image` | `sensor_msgs/Image` | annotated view |
| `/vlm/prompt` | `std_msgs/String` | retarget the detector at runtime |
| `/pick_place/state` | `std_msgs/String` | current state machine step |
| `/pick_place/start` | `std_srvs/Trigger` | run one cycle |
| `/pick_place/abort` | `std_srvs/Trigger` | stop after the current motion |
| `/octomap_gater/refresh` | `std_srvs/Trigger` | let 3 depth frames through |

`vision_msgs` would be the idiomatic type for the detections, but it is not
installed on this box and adding it needs root, while every type above ships
with `ros-humble-desktop`.

### Parameters worth tuning

Detector — pass with `--ros-args -p name:=value`:

| Parameter | Default | |
| --- | --- | --- |
| `prompt` | `detect screwdriver` | any PaliGemma detection prompt |
| `axis_depth_tolerance` | `0.03` | depth band counted as "the object", metres |
| `inference_period` | `0.4` | seconds between inferences |
| `show_window` | `false` | OpenCV window instead of just the topic |
| `debug_dashboard` | `false` | colour + depth side by side |

Orchestrator — all exposed as launch arguments, `--show-args` lists the rest:

| Parameter | Default | |
| --- | --- | --- |
| `place_position` | `[0.35, 0.30, 0.25]` | **placeholder, measure yours** |
| `grasp_finger_min` | `0.003` | finger position above which the gripper counts as holding something — measure it on your object |
| `grasp_z_offset` | `-0.005` | applied to the object's detected *top* surface |
| `approach_height` | `0.12` | pre-grasp height above the grasp |
| `use_table_collision` / `table_z` | `false` / `0.0` | explicit work-surface box; worth enabling with `octomap:=static`, where the map can legitimately be empty |
| `velocity_scaling` | `0.15` | start lower on the first hardware run |

### Watching it live

`/vlm/debug_image` carries the annotated view: the detection box, the depth-derived
object axis in green, and the numbers the orchestrator is actually acting on —
distance, world xyz, grasp yaw, which segmentation path produced the axis, and how
many valid depth pixels backed it. A box drawn in **red** means the detection had no
usable depth, so it was skipped.

Three ways to see it, all showing the same frame:

```bash
ros2 run rqt_image_view rqt_image_view /vlm/debug_image
```

an `Image` display on `/vlm/debug_image` in RViz, or a plain OpenCV window
straight from the node:

```bash
VLM/run_vlm_detector.sh --ros-args -p show_window:=true
```

Add `-p debug_dashboard:=true` for the colour + depth side-by-side view, with the
boxes drawn on both panels — that is where you see *why* a detection had no depth.

The same numbers in text form:

```bash
ros2 topic echo /vlm/detections --once --full-length
```

`axis_source` is the field to watch: `depth` is the good path. `intensity` means
the object did not separate from the background in depth and the axis came from
image contrast instead, which is what produced a 90°-wrong grasp angle before —
raise `axis_depth_tolerance` if you see it. `bbox` means neither worked and the
axis is just the box's aspect ratio.

### Calibrate these two before trusting it

- **`place_position`** — the default `[0.35, 0.30, 0.25]` is a placeholder. Jog
  the arm over your box in RViz and read the `openarm_left_hand_tcp` position off
  TF.
- **The detection itself** — before any motion, add a `PoseArray` display on
  `/vlm/detection_poses` in RViz. The arrow must land *on* the real object. If it
  is offset, the camera mount measurement in `cam_org.txt` is wrong, not the code.
  `/vlm/debug_image` shows what the model sees and the world coordinates it
  produced.

Both nodes' geometry can be checked without hardware — no camera, no robot, no
model load. Expected values in these are worked out by hand rather than taken
from the code, so they catch the failures that still produce plausible-looking
numbers:

```bash
VLM/run_in_vlm_env.sh test_vlm_geometry.py
```

Pixel → world: image decoding, the `<locNNNN>` parsing, deprojection, the tf
composition, median depth sampling, and the axis segmentation including the
Otsu trap described below.

```bash
source native/setup.bash && python3 native/tests/test_grasp_geometry.py
```

Grasp poses: that the tool approach axis lands on world −Z at every yaw, that
the fingers close perpendicular to the detected object axis, and that the retry
ladder actually escalates.

### Reach: ask cuMotion, never `/compute_ik`

**`/compute_ik` lies on this robot.** `config/kinematics.yaml` configures
`kdl_kinematics_plugin` with a **5 ms** solver timeout, and KDL is a local
numerical solver seeded from the current state. On a redundant 7-DOF arm it
returns `-31 NO_IK_SOLUTION` for poses cuMotion plans without trouble — measured
here, it rejected the entire work area including poses that plan fine. OMPL
inherits the same problem, because it samples pose goals through that plugin.

So check reachability against the planner you will actually use:

```bash
python3 native/tests/check_reachability.py --point 0.38 0.15 0.35
```

That sends a `plan_only` goal through cuMotion and sweeps approach azimuth and
tilt, so a failure tells you whether the *position* is out or only some approach
angles are.

What the geometry actually is, verified three ways — cuMotion planning, a
position-only damped-least-squares IK over the URDF, and 300k random FK samples
(the FK was checked against TF at all-zeros: `0.000, 0.1735, 0.0819` vs TF's
`0.000, 0.173, 0.082`):

- shoulders at `(0, ±0.051, 0.698)`; link offsets sum to 0.804 m but the usable
  straight-line reach is **≈0.738 m**, less in awkward directions
- `(0.380, +0.150, 0.350)` — table height — **plans top-down** ✓
- `(0.541, -0.023, 0.350)` is **93 mm beyond reach**; `(0.496, -0.249, 0.375)`
  is 222 mm beyond ✗
- both reachable solutions sit exactly on `joint2`'s upper limit of +0.175 rad
  (10°). Relaxing that limit recovers only ~3 cm and then plateaus, so it is
  the arm's geometry that runs out, not that one limit.

Practically, for the left arm keep pick targets around **x ≤ 0.40, y ≥ +0.10** at
table height. Reach does improve with height, so a riser buys a few centimetres —
but moving the object into the near-left area is simpler and enough.

`VLM/pixel_to_world.py` tells you where something *is*; this tells you whether
the arm can get there. A pick that dies on an opaque MoveIt error code is nearly
always one of these two.

### Dry-running on fake hardware

With `use_fake_hardware:=true` the camera and the detector are real and only the
arm is simulated, so it is a genuine rehearsal of the motion sequence — but
**grasp verification is guaranteed to fail there**, and the ladder will burn all
six attempts. `mock_components` reports the finger exactly where it was
commanded, which is below the holding threshold, and the object never physically
moves so the re-detection always finds it back at the pick point. Neutralise
both checks for the rehearsal:

```bash
ros2 launch pick_place.launch.py grasp_finger_min:=-1.0 object_moved_eps:=0.0
```

Drop both arguments on real hardware — they are exactly what makes the retries
work.

### Why it is built this way

- **The detector subscribes to the camera, it does not open it.**
  `launch_everything.launch.py` already gives the D455 to `realsense2_camera` for
  the octomap, and the device only allows one owner. Colour and `align_depth` are
  enabled there for this reason. Colour runs at `640x480x15`: this D455's RGB
  sensor offers only 1280x720, 1280x800x8, 640x480 and 424x240 — no 848x480, so
  matching the depth profile silently fell back to 640x480 anyway. Check yours
  with `ros2 param describe /camera/camera rgb_camera.color_profile`. Since the
  aligned depth follows the colour resolution,
  `/camera/camera/color/camera_info` is the one set of intrinsics that describes
  both. Expect ~6–10 Hz on the topics with alignment and RViz running, which is
  ample against a 0.4 s inference period.
- **The grasp axis is segmented from depth, not from image contrast.**
  The earlier prototype ran Otsu on the colour crop. On a narrow crop of a
  screwdriver that locks onto the band where the handle meets the shaft and
  reports an axis ~80° off — measured on this rig: a 25×119 px vertical
  detection came back as `image_angle_deg: -10.3`, which would have closed the
  fingers *along* the tool instead of across it. Depth has no such texture, so
  the axis now comes from the pixels within `axis_depth_tolerance` (3 cm) of the
  object's own median depth, with intensity and then the box aspect ratio as
  fallbacks. `axis_source` in each detection says which one ran.
- **Two Python stacks, two processes, DDS in between.** `native/venv` is pinned
  to torch 2.7.0+cu128 / numpy 1.26.4 for cuRobo's prebuilt kernels; the VLM
  needs torch 2.14+cu130 / numpy 2.2.6. They cannot coexist in one process, so
  `run_in_vlm_env.sh` scrubs `PYTHONPATH`/`LD_LIBRARY_PATH` before starting the
  detector.
- **Extrinsics come from tf2.** The hardcoded `T_CAM_TO_ROBOT` matrix the early
  prototypes carried disagreed with the calibrated URDF mount; the node now
  looks up `camera_color_optical_frame → world`, so re-measuring the mount needs
  no code change.
- **The gripper bypasses MoveIt** — cuMotion rejects 1-DOF groups (see below), so
  it goes straight to `/left_gripper_controller/gripper_cmd`.
- **Grasp success is judged geometrically, not by asking the VLM.**
  `paligemma-3b-pt-224` is a pretrained checkpoint: `detect X` is well-formed,
  yes/no VQA is not. So verification re-detects and checks *where the object
  went* — still at the pick point means the grasp failed — combined with the
  finger position from `/joint_states`. Note the gripper controller has
  `allow_stalling` unset, so a **successful** grasp returns `ABORTED`; the action
  result is deliberately ignored.

---

## Why native and not the container

**cuMotion cannot run in the Isaac ROS container on this machine's RTX 5070 Ti.**
The container's CUDA toolkit is 12.2 (inherited from its `tritonserver:23.10-py3`
base). cuRobo has to JIT-compile its kernels on this GPU, and 12.2's `nvcc` does
not know `compute_120`:

```
nvcc fatal : Unsupported gpu architecture 'compute_120'
```

The host has CUDA 12.8, so the native path works. The native environment is fully
self-contained in `native/` and does not modify the host's ROS, Python or apt
state — see [native/README.md](native/README.md) for exactly what it touches, how
it is built, and the two GPU-specific bugs it had to work around.

### Falling back to the container

The container setup is untouched and still works for everything that does not
need cuMotion:

```bash
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh
```

Then inside the container:

```bash
source install/setup.bash && ros2 launch /workspaces/isaac_ros-dev/launch_everything.launch.py
```

Note the container writes to the root `build/` and `install/` trees, while the
native setup uses `native/build` and `native/install`, so the two never collide.

---

## 🛠️ Important Notes & Fixes Applied

- **cuMotion Planner (GPU IK)**: cuMotion is the default MoveIt planner for the
  7-DOF arms (`left_joint_trajectory_controller`,
  `right_joint_trajectory_controller`).
- **Gripper Planning**: cuMotion **does not support 1-DOF grippers**. Planning for
  them with cuMotion is safely rejected rather than crashing.
  **Always switch the Planning Pipeline to `ompl` in RViz** for `left_gripper` or
  `right_gripper`.
- **Gripper PID Tuning**: the DM4310 gripper motor gains (`Kp = 20.0`, `Kd = 0.5`)
  are tuned to remove high-frequency noise and vibration while keeping enough
  torque for accurate movement.
- **Blackwell register cap**: `native/setup.bash` sets
  `NVCC_APPEND_FLAGS=-maxrregcount=160`. Without it, every cuMotion *trajectory*
  optimisation fails on this GPU with `too many resources requested for launch`
  (IK alone still works). If you raise `num_steps` in
  `config/cumotion_planning.yaml` above 32, the cap must be recomputed —
  the formula is in [native/README.md](native/README.md).
- **Octomap gater**: in `octomap:=static`, `octomap_gater.py` sits between the
  camera and MoveIt. It now takes `/octomap_gater/refresh`
  (`std_srvs/srv/Trigger`) and `/octomap_gater/allow_frames`
  (`std_msgs/Int32`) as well as the Tk button, so an autonomous loop can update
  the map — the pick-and-place retry ladder calls it on its last attempt. It also
  no longer dies when there is no `DISPLAY`.
- **Camera mount**: the D455 pose lives in
  `src/openarm_description/urdf/robot/v10.urdf.xacro` as the `xacro:sensor_d455`
  origin, currently `xyz="0.10175 0.000 0.93272" rpy="0 1.0472 0"` — 101.75 mm
  forward, 932.72 mm high, tilted 60° forward. `cam_org.txt` tracks that
  measurement and its history. Note this positions
  `camera_bottom_screw_frame`; the D455's own geometry then puts `camera_link` at
  roughly `x=119.9, y=47.5, z=930.3` mm, which is expected.
- **Core Dump Cleanup**: `./scripts/run_dev.sh` deletes large `core.*` crash dumps
  from the workspace before the container starts, so the disk does not fill up.
  The native flow does not create them in the workspace root.

## Troubleshooting

If anything fails with a numpy ABI error, an unexpected torch version, or a
missing module, check the environment split first — it is the most common cause
and the check is instant:

```bash
./check_isolation.sh
```

Check the native environment itself (works headless):

```bash
source native/setup.bash && native/verify_overlay.sh
```

Test the planning stack without a display — skips the octomap updater that
requires OpenGL:

```bash
source native/setup.bash && python3 native/tests/test_move_group_planners.py
```

If `move_group` dies immediately with `freeglut failed to open display`, you are
running without a display; use the headless test above, or start from a desktop
session.
