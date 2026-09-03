# 7DOF-OArm IK Environment

Runs the 7DOF-OArm bimanual exoskeleton with GPU-accelerated cuMotion kinematics,
MoveIt 2, and real hardware interfaces.

There are two ways to run it. **Use the native one** — see
[Why native](#why-native-and-not-the-container) below.

| Document | |
| --- | --- |
| **README.md** (this file) | running the robot, the VLM pick-and-place, tuning, design rationale |
| [INSTALL.md](INSTALL.md) | setting up a machine: requirements, the Python-stack split, GPU/CUDA, portability |
| [native/README.md](native/README.md) | how the native environment is assembled and exactly what it touches |

---

## 🚀 Quick Start (native — recommended)

One-time setup, if this machine has not been set up yet:

```bash
./install.sh
```

That builds both project environments and verifies they are isolated from the
system's Python — see [INSTALL.md](INSTALL.md) for requirements and what it does
and does not touch. Then, the one step that needs root:

```bash
sudo mkdir -p /workspaces && sudo ln -s /home/mr/workspaces/isaac_ros-dev /workspaces/isaac_ros-dev
```

### 1. Bring up the CAN interfaces

```bash
sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can0 up
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

### 3. Or start everything, including the pick-and-place panel

```bash
native/run_pick_place_demo.sh
```

Same robot bringup, plus the VLM detector, the pick-and-place orchestrator and a
panel to type into — see [VLM-guided pick and place](#-vlm-guided-pick-and-place).
Use `run_launch_everything.sh` above when you only want the robot.

### 4. Rebuild after editing code

```bash
native/build_ws.sh
```

Or a single package:

```bash
native/build_ws.sh --packages-select openarm_description
```

### 5. Editing in VS Code

```bash
code ~/workspaces/isaac_ros-dev
```

---

## 📦 Installation

```bash
./install.sh          # everything; --native-only / --vlm-only / --check
./check_isolation.sh  # verify the venvs never overlap the system's Python
```

Requires Ubuntu 22.04, `ros-humble-desktop`, Python 3.10 and CUDA ≥ 12.8 already
on the host — it verifies the system, it does not provision it.

**Full details in [INSTALL.md](INSTALL.md)**: requirements, the three-Python-stack
split and how it is enforced, GPU/CUDA detection, what needs root, portability
limits, and install troubleshooting.

---

## 🤖 VLM-guided pick and place

A PaliGemma detector finds the object, cuMotion plans to it, and the same
detector checks whether each step actually worked — a failed grasp is retried
with an escalating strategy rather than repeated.

```
READY ──► LOCATE ──► PRE_PICK ──► PREGRASP ──► OPEN ──► DESCEND ──► CLOSE
   ▲                                                                  │
   └──────────────────────── failed ──────────────── VERIFY_GRASP ◄── LIFT
                                                          │
                    READY ──► DROP ──► RELEASE ──► VERIFY_PLACE
```

Three named postures, all replayed as joint goals:

| | Where it comes from | |
| --- | --- | --- |
| **READY** | `ready_joint_positions` | observation pose: the arm starts here and the object is located from here, so it has to leave the camera a clear view |
| **PRE_PICK** | `pre_pick_state` in the states file | staging pose entered once the object is located, so the descent starts from a known posture |
| **DROP** | `drop_state` in the states file | where the object is released |

**PREGRASP** is 5 cm (`approach_height`) above the object's detected top surface:
the arm stops there, opens the gripper, descends onto the object, closes, and
lifts back to the same 5 cm.

### Everything in one command

```bash
native/run_pick_place_demo.sh
```

That starts the robot, MoveIt, cuMotion, the camera, the detector, the
orchestrator and a small panel you type into — it sets cuMotion's `tool_frame`
to match the arm itself, so the two cannot disagree. Type what to pick and press
**Pick**:

```
┌─ OpenArm pick and place ──────────────────┐
│ What should the arm pick up?              │
│ ┌───────────────────────────────────────┐ │
│ │ pick up the screwdriver               │ │
│ └───────────────────────────────────────┘ │
│ detector prompt:  detect screwdriver      │
│  [ Pick ]  [ Abort ]                      │
│ PRE_PICK                                  │
└───────────────────────────────────────────┘
```

The panel strips the conversational wrapper before the prompt reaches the model —
`paligemma-3b-pt-224` is a pretrained checkpoint, so `detect screwdriver` is a
well-formed prompt for it and `pick up the screwdriver` is not. It shows the
translation live so there is no guessing which of the two the model saw.

PaliGemma takes tens of seconds and several GB of VRAM to load; it loads
alongside robot bringup, and until it is ready a pick sits in LOCATE. That load
is why the launch files are otherwise kept separate — when the robot is already
up, use `pick_place.launch.py` on its own instead.

**Before the first run**, record the two poses the cycle needs (below). Without
them a pick refuses to start rather than moving somewhere nobody chose.

### Recording pre_pick_state and drop_state

With the robot up, jog the arm in RViz to the pose you want — interactive marker
or the Joints tab, then Plan & Execute — and capture where it actually landed:

```bash
python3 record_states.py
```

It walks through both poses, waiting for Enter at each, and writes
`pick_place_states.yaml`. Re-record one without touching the other by naming it,
and check what is on file at any time:

```bash
python3 record_states.py drop_state
python3 record_states.py --list
```

Reading the values back off the robot beats typing them in: the Joints tab only
shows whole degrees, and an executed plan lands near its goal rather than exactly
on it. The orchestrator re-reads the file at the start of every cycle, so
re-recording a pose takes effect on the next pick — no restart, which matters
when a restart costs a PaliGemma load.

`drop_state` is where the object is *released*, so it falls from whatever height
that pose holds the tool at. `record_states.py --list` prints the tool position
of each pose, which is the number to look at before the first run.

### Driving it without the panel

The panel is a client of the ordinary interface, so anything it does can be done
from a terminal. With the robot already up:

```bash
cd /home/mr/workspaces/isaac_ros-dev && source native/setup.bash
ros2 launch pick_place.launch.py prompt:="detect screwdriver"
```

Wait for `model loaded`, then:

```bash
ros2 topic pub --once /pick_place/prompt std_msgs/String "{data: 'pick up the wrench'}"
ros2 service call /pick_place/start std_srvs/srv/Trigger
```

The prompt topic is normalised exactly as the panel does it, so publishing a
whole sentence works. `ros2 topic echo /pick_place/state` follows the state
machine and `ros2 service call /pick_place/abort std_srvs/srv/Trigger` stops it
after the current motion. `ros2 launch pick_place_demo.launch.py --show-args`
lists every tunable.

### Pick with the right arm, and why that needs a relaunch

**cuMotion accepts Cartesian goals for exactly one link.** It compares every
pose goal's `link_name` against its own `ee_link` and rejects a mismatch outright:

```
Link name for Target Pose "openarm_right_hand_tcp" and Planning frame
"openarm_left_hand_tcp" do not match, relaunch node with
tool_frame = openarm_right_hand_tcp
```

`openarm.yml` sets `ee_link` to the **left** hand, so with a default bringup every
right-arm pose goal fails — measured here, the whole right-hand work area came
back unreachable for that reason alone, not because the arm cannot get there. The
node reads `tool_frame` once at construction, so it cannot be fixed at runtime:

```bash
native/run_launch_everything.sh tool_frame:=openarm_right_hand_tcp
```

The orchestrator checks this before it moves anything — it reads the planner's
`tool_frame`, falls back to the `ee_link` in `openarm.yml` when that is unset, and
refuses to start on a mismatch rather than discovering it at PREGRASP with the
gripper already open. The *other* arm still takes joint-space goals normally; only
Cartesian goals are restricted to one link at a time. Both arms stay in cuMotion's
kinematic chain either way, which is why `openarm.yml` now lists both tool frames
in `link_names` — with only one listed, pointing `ee_link` at the right hand drops
the left arm to zero active joints, unmodelled and therefore not something the
right arm would plan around (measured: 14 active joints with both listed, 7 with
only the right).

### The ready pose

`ready_joint_positions` is `joint1..joint7` in radians. The default is the
elbow-up pose with the tool clear of the table, captured off the right arm rather
than read off the sliders. To use a different one, jog the arm there in RViz and
ask for the numbers:

```bash
ros2 service call /pick_place/capture_ready std_srvs/srv/Trigger
```

It returns a paste-ready list, which beats the Joints tab — that only shows whole
degrees.

**With the default `place_mode:=ready` the object is released at this pose**, so it
falls the distance between the tool and whatever is under it. Check what is
underneath before the first run. `place_mode:=position` restores the older
behaviour of moving over a separate `place_position` and releasing there.

### The pieces

| File | Runs in | Does |
| --- | --- | --- |
| [VLM/vlm_detector_node.py](VLM/vlm_detector_node.py) | `VLM/.venv` | PaliGemma → 3D points in `world` on `/vlm/detections` |
| [VLM/run_in_vlm_env.sh](VLM/run_in_vlm_env.sh) | — | environment isolation for the above |
| [VLM/bootstrap.sh](VLM/bootstrap.sh) | — | builds `VLM/.venv` from `requirements.txt` |
| [pick_place_orchestrator.py](pick_place_orchestrator.py) | `native/venv` | the state machine, MoveIt + gripper + planning scene |
| [pick_place_ui.py](pick_place_ui.py) | `native/venv` | the panel: type an object, Pick, Abort, live state |
| [record_states.py](record_states.py) | `native/venv` | capture `pre_pick_state` / `drop_state` into YAML |
| [vlm_prompt.py](vlm_prompt.py) | `native/venv` | "pick up the X" → "detect X", shared by the panel and the orchestrator |
| [pick_place_demo.launch.py](pick_place_demo.launch.py) | — | robot + detector + orchestrator + panel, one command |
| [native/run_pick_place_demo.sh](native/run_pick_place_demo.sh) | — | the above, with the preflight checks |
| [pick_place.launch.py](pick_place.launch.py) | — | detector + orchestrator only, for an already-running robot |
| [VLM/pixel_to_world.py](VLM/pixel_to_world.py) | `VLM/.venv` | click any point, get its world coordinate |
| [native/tests/check_reachability.py](native/tests/check_reachability.py) | `native/venv` | ask MoveIt if the arm can get there |
| [VLM/test_vlm_geometry.py](VLM/test_vlm_geometry.py) | `VLM/.venv` | pixel → world maths, no camera or model |
| [native/tests/test_grasp_geometry.py](native/tests/test_grasp_geometry.py) | `native/venv` | grasp pose maths, no robot |
| [native/tests/test_pick_cycle.py](native/tests/test_pick_cycle.py) | `native/venv` | a whole cycle against a fake robot, no hardware |

### Interfaces

| Name | Type | |
| --- | --- | --- |
| `/vlm/detections` | `std_msgs/String` | JSON, authoritative — see the node's docstring for the schema |
| `/vlm/detection_poses` | `geometry_msgs/PoseArray` | same points, for RViz |
| `/vlm/debug_image` | `sensor_msgs/Image` | annotated view |
| `/vlm/prompt` | `std_msgs/String` | retarget the detector at runtime |
| `/pick_place/prompt` | `std_msgs/String` | what to pick next; a whole sentence is fine |
| `/pick_place/state` | `std_msgs/String` | current state machine step |
| `/pick_place/start` | `std_srvs/Trigger` | run one cycle |
| `/pick_place/abort` | `std_srvs/Trigger` | stop after the current motion |
| `/pick_place/capture_ready` | `std_srvs/Trigger` | current arm joints as a `ready_joint_positions` list |
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
| `arm` | `right` | must match the `tool_frame` the robot was launched with |
| `ready_joint_positions` | elbow-up pose | the observation pose, `joint1..joint7` in radians |
| `states_file` | `pick_place_states.yaml` | the poses `record_states.py` writes; re-read every cycle |
| `place_mode` | `state` | `state` drops at `drop_state`; `ready` at the observation pose; `position` uses `place_position` |
| `place_position` | `[0.35, 0.30, 0.25]` | `place_mode:=position` only — **placeholder, measure yours** |
| `grasp_finger_min` | `0.003` | finger position above which the gripper counts as holding something — measure it on your object |
| `grasp_z_offset` | `-0.005` | applied to the object's detected *top* surface |
| `approach_height` | `0.05` | pre-grasp height above the grasp: where the gripper opens |
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

- **`pre_pick_state` and `drop_state`** — nothing ships with sensible defaults for
  these because they depend on your table. Record them with `record_states.py`,
  and check the tool height of `drop_state` with `--list` before the first run:
  that is how far the object falls.
- **The ready pose** — the default is a sensible elbow-up posture. Jog to the pose
  you want and capture it with `/pick_place/capture_ready`. Only under
  `place_mode:=position` does `place_position` matter, and its
  `[0.35, 0.30, 0.25]` default is a placeholder.
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

```bash
source native/setup.bash && python3 native/tests/test_pick_cycle.py
```

A whole cycle against a stubbed robot — `/move_action`, the gripper, TF,
`/joint_states`, the planning scene, cuMotion's parameters and the detector are
all faked, so the real orchestrator runs and every goal it sends is recorded.
It asserts the order of the eight goals, that PRE_PICK and DROP replay the
recorded joint values, that PREGRASP and LIFT sit `approach_height` above the
grasp, and that an unrecorded or wrong-arm states file makes the cycle refuse
without sending a single goal. This is the one that catches a reordered
sequence, which the geometry tests cannot see.

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

The right arm mirrors it. Measured by planning a top-down pose straight through
cuRobo from the ready pose, empty world, so this is reach and not collision
(`# ` = plans):

```
              x=0.25  0.30  0.35  0.40  0.45  0.50
z=0.42          #     #     #     #     .     .      y = -0.05 .. -0.30, all rows
z=0.35          #     #     #     #     .     .      y = -0.10 .. -0.25
z=0.25          #     #     .     .     .     .      y = -0.05 .. -0.30
```

So **x ≤ 0.40, y ≤ −0.10** at z ≥ 0.35, and only **x ≤ 0.30** if the object sits
as low as z = 0.25. Nothing at x ≥ 0.45 at any height, exactly as on the left.
Note the grasp itself is the *low* pose in a cycle, so a pick off a low table is
the case that runs out of reach first.

Reproduce either arm's figures with `check_reachability.py --arm right`, but
remember it goes through cuMotion: the robot has to have been launched with a
matching `tool_frame`, or every cell comes back unreachable for that reason
instead.

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
- **No grasp-pose network — the VLM's axis plus a top-down approach is enough.**
  AnyGrasp, Contact-GraspNet and friends earn their keep on cluttered bins of
  unknown objects, where the question "where *can* I grip this at all" is
  genuinely open. That is not this problem: single objects on a flat table, a
  parallel-jaw gripper, and an arm whose verified envelope is a top-down
  approach over a small work area. Once the approach is fixed to straight down,
  the only free parameter left is the wrist angle — one number, which the
  detector already produces from the depth segmentation, and which the retry
  ladder already escalates when it is wrong. A grasp network would decide the
  same number at the cost of a third Python stack (the two here already conflict
  over torch and numpy), a licence key in AnyGrasp's case, several more GB of
  VRAM alongside PaliGemma, and 6-DOF poses that mostly fall outside an envelope
  that only plans top-down anyway. Worth revisiting the day the objects arrive
  in a pile.
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
- **One Cartesian link at a time**: cuMotion rejects any pose goal whose
  `link_name` differs from its `ee_link` (`INVALID_LINK_NAME`), so the arm you can
  give pose goals to is fixed at bringup by
  `tool_frame:=openarm_{left,right}_hand_tcp`. `launch_everything.launch.py`
  defaults it to the right hand. Joint-space goals are unaffected for either arm.
- **Gripper Planning**: cuMotion **does not support 1-DOF grippers**. Planning for
  them with cuMotion is safely rejected rather than crashing.
  **Always switch the Planning Pipeline to `ompl` in RViz** for `left_gripper` or
  `right_gripper`.
- **Gripper PID Tuning**: the DM4310 gripper motor gains (`Kp = 20.0`, `Kd = 0.5`)
  are tuned to remove high-frequency noise and vibration while keeping enough
  torque for accurate movement.
- **Blackwell register cap**: on `sm_120` only, `native/setup.bash` sets
  `NVCC_APPEND_FLAGS=-maxrregcount=160`. Without it, every cuMotion *trajectory*
  optimisation fails on this GPU with `too many resources requested for launch`
  (IK alone still works). It is applied conditionally because on `sm_75/86/89`
  the prebuilt cubins are used and the kernel fits in 118 registers anyway, so
  the cap would only force needless spilling. If you raise `num_steps` in
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

Install-time problems — preflight failures, numpy ABI errors, an unexpected torch
version, the gated PaliGemma download — are tabulated in
[INSTALL.md](INSTALL.md#troubleshooting-the-install). Start there; the first check
is instant:

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

### "every pick strategy was exhausted"

All six attempts failed. The state now names the cause, so read the rest of the
line before anything else:

```bash
ros2 topic echo /pick_place/state --once
```

| It says | What to do |
| --- | --- |
| `the planner is unusable for this arm` | see below — usually cuMotion has died |
| `nothing matched "detect X"` | the detector saw no such object. Check `/vlm/debug_image`, and that it has finished loading — a pick started during the PaliGemma load fails this way six times over |
| `could not plan to the pre-grasp above (x, y, z)` | almost always **out of reach**. The line includes a ready-made `check_reachability.py` command; for the right arm keep objects around `x ≤ 0.40, y ≤ −0.10` |
| `could not plan to pre_pick_state` | the recorded staging pose is not reachable from the observation pose; re-record it |
| `reached the pre-grasp but could not descend` | the octomap probably contains the object itself, or the grasp is under the table |
| `the gripper closed but nothing was held` | `grasp_finger_min` or the grasp height is wrong for this object |

The out-of-reach case is the one that looks most like a software fault and is
not: the detector reports a perfectly good position, cuMotion simply cannot get
the tool there. `/vlm/detections` gives you the coordinate to check:

```bash
ros2 topic echo /vlm/detections --once --full-length
```

### cuMotion dies mid-session

It has crashed here with **SIGFPE**, six minutes into a session, leaving only
this in the launch output:

```
[ERROR] [cumotion_goal_set_planner_node-5]: process has died [pid ..., exit code -8, ...]
```

With no planner every pose goal fails, so a pick walks the whole retry ladder and
blames itself. The orchestrator now checks the planner is in the graph before it
moves anything and refuses with `the planner is unusable for this arm`, but if a
cycle is already running when it dies you still get the exhausted-ladder message.
Confirm with:

```bash
ros2 node list | grep cumotion || echo "cuMotion is not running"
```

The fix is to restart the robot; a dead planner cannot be revived in place, since
it reads `tool_frame` once at construction. Root cause unknown — the launch files
now run every Python node with `PYTHONUNBUFFERED=1` so that the next crash leaves
its traceback in `launch.log` instead of losing it in a stdout buffer, which is
what happened the first time.

### The detector never publishes, but the camera node is running

Check which streams are actually live — and note that **`ros2 topic echo` will
show nothing even on a healthy RealSense topic**, because it subscribes RELIABLE
while the camera publishes BEST_EFFORT:

```bash
ros2 topic hz /camera/camera/color/image_raw
```

If depth ticks and colour does not, look for a **second camera taking the video
device numbers**. librealsense pairs a colour stream with the adjacent
`/dev/videoN` for its metadata, so another UVC device landing in the middle of
the block breaks colour while leaving depth working:

```bash
for n in /sys/class/video4linux/video*; do
    echo "$(basename $n) $(cat $n/name) $(readlink -f $n/device | sed 's|.*/||')"
done
```

Every node of the RealSense colour interface (`:1.3` above) should be adjacent.
If they are split, unplug the other camera and **re-enumerate the RealSense** —
unplugging alone does not renumber what is already assigned:

```bash
sudo sh -c 'echo 0 > /sys/bus/usb/devices/2-4.1/authorized; sleep 2; echo 1 > /sys/bus/usb/devices/2-4.1/authorized'
```

### RViz never opens, but everything else starts

Look for this in the launch output — it scrolls past quickly, and every other
node comes up fine, so it reads like a launch-file problem when it is not:

```
rviz2: symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0:
undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
[ERROR] [rviz2-4]: process has died [pid ..., exit code 127, ...]
```

That is a terminal opened **inside a snap** — VS Code's integrated terminal is
the usual one here, since `code` is installed as a snap. The snap exports
`GTK_PATH` and friends pointing back into itself; rviz2 loads a GTK module from
`/snap/code`, whose RPATH pulls in core20's glibc 2.31 `libpthread` next to the
host's 2.35, and it aborts before drawing anything.

`native/setup.bash` now clears those variables when they point into a snap, and
says so when it does:

```
note: cleared snap-provided GTK_PATH GTK_EXE_PREFIX ... (they crash rviz2)
```

So sourcing it is the fix. If RViz still will not start, check nothing re-added
them after sourcing:

```bash
source native/setup.bash && env | grep -E '^(GTK_PATH|GIO_MODULE_DIR)=' ; rviz2 --help | head -1
```

That should print no `GTK_PATH` line and then rviz2's usage. Launching from a
plain GNOME terminal instead of the VS Code one also avoids it entirely.
