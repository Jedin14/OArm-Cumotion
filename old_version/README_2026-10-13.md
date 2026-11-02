# 7DOF-OArm IK Environment

Runs the 7DOF-OArm bimanual exoskeleton with GPU-accelerated cuMotion kinematics,
MoveIt 2, and real hardware interfaces.

There are two ways to run it. **Use the native one** — see
[Why native](#why-native-and-not-the-container) below.

| Document | |
| --- | --- |
| **README.md** (this file) | running the robot: click to move, CAL ARMS, the auto box pick |
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

Same robot bringup, plus the VLM detector, the pick-and-place orchestrator and
**the web UI on `http://<this machine>:8088`** — open it from wherever you are
sitting. See [The web UI](#the-web-ui) and
[VLM-guided pick and place](#-vlm-guided-pick-and-place). Use
`run_launch_everything.sh` above when you only want the robot.

It also brings `can0` and `can1` up at 1 Mbit/5 Mbit FD, which needs root:
expect one sudo prompt on a cold boot and none afterwards, because an
interface already up is left alone. `--no-can` skips it and prints the three
commands instead; `--fake` does not need a bus at all. Run it on the machine
the arms are wired to — over a remote mount the sudo authenticates on the
wrong host.

Add `--fake` to rehearse with no arms and no CAN bus: the motion is simulated
and watched in RViz, the camera and the detector are real. See
[Dry-running on fake hardware](#dry-running-on-fake-hardware).

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

## 🎯 Click to move (cuMotion only)

Take a frame, click a point on it, press MOVE, and the fingertips of the arm
on that side go to the point, hold for 2 s, and go back. There is no VLM and no
grasp model. The gripper is not commanded unless you pass `--close-gripper`. The robot bringup is the same as for pick and place,
so MoveIt, cuMotion, the D455 and the octomap (static, via the gater) work
exactly as they do there.

```bash
native/run_launch_everything.sh          # terminal 1 (arms boot to navigation_state)
native/run_click_to_move.sh              # terminal 2
```

The window shows a captured frame, not live video:

- **Left click** selects a point; its `x y z` shows in the window and as a
  marker in RViz. The `x y z` is the clicked surface in `world`, in metres:
  origin at the base of the stand, +x forward, +y the robot's left, +z up.
  The shoulders are at (0, ±0.051, 0.698).
- **MOVE** / `m` touches it with the left arm for the left half of the image
  and the right arm for the right half.
- **RECAPTURE** / `r` takes a new frame and rebuilds the octomap.
- **CALIBRATE** / `c` measures the camera pose, and **APPLY** writes it.
- `q` quits.

**Do CALIBRATE once after any camera move.** Accuracy depends on three
things, and the tool handles each:

1. **What touches.** `hand_tcp` is 80 mm from the hand, but the fingertips
   reach 95.4 mm, so the *fingertips* are aimed at the point
   (`--tip-offset 0.0154`). Aiming `hand_tcp` there used to push the fingers
   15 mm into the surface, where they slid off the spot. With the gripper
   left alone (the default), the point sits midway between the fingertips.
   `--close-gripper` shuts them first so they meet on the point and touch it.
2. **The arm arriving.** The trajectory controller reports success before
   the arm has converged; its gravity sag is only pulled out by the
   hardware's slow integral term. So at 5 cm (`--standoff`) the arm is left
   to settle, the fingertip is measured from the encoders, and any error
   over 2 mm is flown out. The same correction is carried into the straight
   move in. The last line in the window reports the result.
3. **The camera pose.** The URDF mount was a tape measurement, and nothing
   downstream can see its error; the arm goes exactly where the camera says.
   **CALIBRATE measures it and fixes it by itself**
   (`robot_camera_calibration.py`):
   - It takes a clean frame, then moves each arm to up to 6 poses in view
     (out of 12 candidate spots per arm; unreachable ones are skipped).
   - At each pose it takes the arm as the depth camera sees it (thousands
     of points: whatever is now nearer than the clean frame) and the
     robot's own model of it (the URDF visual meshes, posed by the joint
     encoders).
   - It finds the camera pose that puts the one onto the other: robust
     point-to-plane ICP over all poses, started from several guesses
     around the tape measurement.
   - Being a 3D fit, it can't confuse "camera lower" with "camera tilted",
     which is what sank the earlier fingertip-pixel method.
   - In simulation with the real meshes, depth noise, clutter, 1.5% depth
     scale error and starting 65 mm / 9° off, it recovered the camera to
     within a few mm.
   - A plausible result (fit rms < 3 mm, under 12 cm / 15° of change) is
     applied at once: written as the next version into `v10.urdf.xacro`,
     `cam_org.txt` (as mm from the arm origin) and `VLM/vlm_detect.py`,
     `openarm_description` rebuilt, and `robot_state_publisher` updated
     live.
   - The window reports how far off clicks *were* and the model/depth
     agreement before and after. `camera_calibration.yaml` keeps the
     numbers.

   **OVERLAY** (`o`) draws the robot model on a fresh frame (green where the
   depth camera agrees with it, red where not), so you can check the
   calibration by eye whenever an arm is in view. Keep the space in front of
   the camera clear to about 0.45 m while calibrating. `--manual-calibration`
   is the old click-the-fingertip method.

**TEACH** (`t`) removes what's left of the error: the arm bends under its own
weight in its gearboxes, after the motor encoders, so the robot can't see it,
and it changes with posture (5–14 mm rms between poses, up to ~30 mm). With
TEACH on, the arm holds each touch while you nudge the fingertip onto the spot
(arrow keys or LEFT/RIGHT/UP/DOWN, `+`/`-` = IN/OUT, 2 mm a press), then SAVE.
The correction is stored per arm in `touch_corrections.yaml`, and every later
touch applies it, whether TEACH is on or not: a smooth trend through all of that
arm's taught touches (so it carries to places not taught), refined by the ones
within 15 cm. **Teach at more than one distance from the camera:** the arm's
error changes with reach, and touches taught only at 0.35 m did not hold at
0.46 m. A few touches near, middle and far, left and right, per arm, are
enough; delete the file to start over. The camera check of the gripper is off
by default (`--camera-check`): in a front approach the hand is too close to the
camera to measure.

The approach is always from the front, the side the camera sees: along its line of
sight to the point, else tilted 15°, else 30°, and never steeply downward. The octomap
only knows the surfaces the camera sees, so coming from above or behind an object
would pass through space nobody checked. `--orientation down` is the old top-down
approach.

The touch itself: close the gripper (only with `--close-gripper`); cuMotion joint goal to 5 cm from the
point; settle, measure, correct; straight line in at 2 cm/s
(`/compute_cartesian_path`); hold `--hold` s; straight line out; cuMotion
back to the starting posture. The hand and finger links may touch the
octomap (the clicked surface is in it); the rest of the arm is still
collision-checked. That exemption is re-applied before every MOVE and
CALIBRATE, because a restarted move_group forgets it.

If move_group dies (it has segfaulted mid-goal, exit −11), the tool stops at
once and says so rather than waiting out its timeouts. Both crashes seen on
2026-09-30 came 0.6-1.5 s after a gripper command, with IK and a plan in
flight while the fingers were moving. So the gripper is not commanded unless
you ask with `--close-gripper`, and even then nothing is planned until the
fingers have stopped. `demo.launch.py` now
respawns move_group after 3 s; press RECAPTURE once it is back, because its
octomap starts empty.


## 📦 Auto box pick (box model)

`auto_box.py` picks boxes found by the trained detector in
[box_target_detector/](box_target_detector/README.md) (YOLO, one class `box`;
the target is the topmost box, leftmost of the top row). Planning and the
touch are exactly click_to_move's; only the point comes from the model.

```bash
native/run_launch_everything.sh          # terminal 1 (arms boot to navigation_state)
native/run_auto_box.sh                   # terminal 2
```

1. **PICK MODE** (`p`): both arms to `pre_pick_state`; the model loads (~2 s).
2. **DETECT BOX** (space): a fresh frame goes to the model. The target is drawn
   with its bounding box, centroid and world `x y z`; the whole pick is planned
   without moving: **green = pickable**, red = not (the reason is in the status line).
3. **PICK** (`m`): the arm picks it: approach, corrections, straight line in,
   stop on contact, hold, back out, `pre_pick` → drop → `pre_pick`.

Also NAVIGATION (`n`), TOP DOWN (`d`, a box lying flat), RETURN (`b`) and the
tool row (L/R TOOL). `--box-conf` sets the detection threshold (default 0.5).
The model needs `ultralytics` in the native venv (installed with `--no-deps`
so torch, numpy and OpenCV stay as they are).

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
  origin (it positions `camera_bottom_screw_frame`). `cam_org.txt` records the
  current version and every earlier one. V3 was a tape measurement; measure it
  properly with **CALIBRATE** in `click_to_move.py` (see Click to move), whose
  APPLY writes the next version into the URDF, `cam_org.txt` and
  `VLM/vlm_detect.py`.
- **cuMotion's collision model**: `openarm.yml` now carries the robot's own
  collision spheres (fitted by `native/make_collision_spheres.py`). Without
  them cuMotion planned the gripper through the torso, MoveIt rejected the path
  ("Computed path is not valid", finger vs `openarm_body_link0`), and every
  goal failed with PLANNING_FAILED (−2). The planner override also now plans at
  full speed and slows the trajectory down afterwards, because cuRobo's
  `time_dilation_factor` cannot fit a detour around the torso, and it leaves
  the graph planner off (`enable_graph_attempt: -1`), because that planner
  wrongly reports valid states as colliding. A move with no direct path goes
  via home.
- **move_group crashes and cuMotion "stops planning" (fixed 2026-09-30)**:
  three bugs, all reproduced on an isolated ROS domain before being fixed.
  1. `isaac_ros_cumotion_moveit` (move_group's cuMotion plugin) built each
     returned trajectory on an *uninitialised* `RobotState`, so the finger
     joints held garbage; path validation's self-collision check then
     segfaulted in FCL (exit −11). It also handled the result in two threads
     without a lock, and timed out after a hard-coded 5 s. Patched in
     `src/isaac_ros_cumotion/isaac_ros_cumotion_moveit` and built natively
     into `native/install` (it overrides the prebuilt 3.2.5 in `native/root`):
     `colcon build --base-paths src/isaac_ros_cumotion/isaac_ros_cumotion_moveit
     --build-base native/build --install-base native/install`.
  2. cuRobo's self-collision kernel only writes where it finds a collision and
     never clears its reused output buffer, so one colliding trajectory made
     the same interpolation steps "collide" in every later plan. After a few
     bad goals every plan failed with TRAJOPT_FAIL until a restart. The node
     (`cumotion_planner.py`) now zeroes those buffers before each plan.
  3. Out-of-limits joint goals are refused, and start states and joints that
     were not commanded (the other arm, sagging past a limit) are clamped
     inside the limits. A self-check restarts the node if it ever stops being
     able to plan a trivial move.
  Measured: 150 plan-only goals, 110 of the 111 planned succeeded, steady to
  the end, no crash. Before: crash at plan 58, or every plan failing after ~60.
- **Depth profile**: `launch_everything.launch.py depth_profile:=...` sets the
  D455 depth stream, default `424x240x15`. The V3 mount looks flat at surfaces
  0.3-0.5 m away, and the D455's minimum range is about 0.45 m at 848x480 but
  about 0.2 m at 424x240. Use `depth_profile:=848x480x15` for work that is all
  further away; it is more precise at range.
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
| *LOCATE, then HOME, then LOCATE again, then `nothing matched`* | fixed. An empty frame is an **answer**: the detector looked and there was nothing there, and a second look from HOME finds the same nothing thirty seconds later. It now reports not-found on the first look. The trip home is still made when the detector finds *the gripper* — that means the view is blocked, not that the object is absent |
| *the grip is weak, and the jaws close above the object* | the grasp is measured from the object's **top**, and `grasp_z_offset` (+10 mm) puts the jaws above that. The floor is now the **measured surface**, not the top, so a negative offset reaches down the side of the object — `grasp_z_offset:=-0.015` bites 15 mm down and `grasp_table_clearance` (5 mm) stops it pressing into the table |
| *the top reads centimetres high* | fixed. The top was the nearest tenth of the **whole box**, and a box is bigger than the thing in it: with the arm in frame at the staging pose the nearest tenth was the gripper, and the top came back 45 mm high. It is now the **inner half** of the box at the 20th percentile — where the object actually is, past the dropouts and the edge pixels |
| `OVER_DROP ... code 99999` *after 60 s* | an unreachable pose goal does not come back refused, it comes back **not at all**: cuMotion sits on it for the whole `motion_timeout` and move_group is wedged afterwards. The drop point is now reach-checked before it is commanded |
| `the detection is N mm from the tool` | the camera found the gripper rather than the object — possible only when a cycle looks from the staging pose, which is over the work surface. It is refused and the look is retaken from HOME. If it happens every time, the arm is sitting in front of whatever you are picking: `locate_from_staging:=false`, or re-record `pre_pick_state` further back |
| `nothing matched "detect X"` | the detector saw no such object. Check `/vlm/debug_image`, and that it has finished loading — a pick started during the PaliGemma load fails this way six times over |
| `could not plan to the pre-grasp above (x, y, z)` | almost always **out of reach**. The line includes a ready-made `check_reachability.py` command; for the right arm keep objects around `x ≤ 0.40, y ≤ −0.10` |
| `could not plan to pre_pick_state` | the recorded staging pose is not reachable from the observation pose; re-record it |
| `reached the pre-grasp but could not descend` | the octomap probably contains the object itself, or the grasp is under the table |
| *the gripper turns red in RViz, then every move fails with −2* | the jaws are inside the octomap and the arm's own **start state** is in collision, so no plan from it is valid. The descent leaves them among the voxels of the object they came for, so the gripper's octomap exemption is held for the whole time the arm is on the column — descent, close and the way back up — and `release_column()` is the one place that ends it. If this recurs, check that nothing clears `self._column` without going through it |
| `CONTACT: <joint> loaded up during TRANSIT` *on a move that was fine* | fixed. The guard compared each joint against what it was pulling **when the move started**, and gravity moves that a long way: measured over one transit, joint1 went from +3.42 Nm at the staging pose to −10.48 Nm at full stretch. It tripped 1.7 s in, on the shoulder *unloading*, and the cancel was then sent from the stop function — which does not run until the wait for the result returns, so the arm flew the whole move anyway and a successful transit was reported as a crash. Now the reference is taken from just before the arm **stopped following its trajectory**, contact needs both a torque climb and the arm failing to keep up, and the cancel goes out from the watching thread |
| `STOPPED: move_group has stopped answering` | move_group jammed. It does this after rejecting a computed path as invalid: measured, run `1789012345`, the second identical goal was planned by cuMotion and never answered, and `/move_action` and `/check_state_validity` both went out of the graph while the process stayed alive. Nothing that needs a planner will work until the stack is restarted. The contact back-off deliberately does not go through move_group for this reason; `record_states.py --play` talks to the controller directly and can drive the arm out |
| `PRE_PICK_STATE ... code=-2` *straight after a good grasp* | fixed. The lift is what went wrong, not the staging goal: 200 mm straight up was refused for **joint travel** (measured 3.582 rad against a 1.5 rad budget), the lift settled for the pre-grasp height — 50 mm — and the gripper was handed back to the octomap while still inside the voxels of the object it had just picked up, so the arm's own start state was in collision and every plan out of it was invalid. Three things changed: the lift now **climbs what is left in `descend_step` hops**, because the travel budget is per line; "clear" now means `retreat_height` rather than the pre-grasp; and the staging step retries from higher up the way the retreat always has |
| `BOOT: ... code 99999` *and the arms move a minute later* | fixed. 99999 is `FAILURE`, and on this path it means move_group did not **accept** the goal within 15 s — a cold bringup, with cuMotion still warming up. The goals were then accepted and executed long after the walk had reported failure. The walk now waits for a plan-only probe to come back (`boot_planner_wait`) before it sends anything, and moves nothing at all if the planner never becomes usable |
| `RECOVER` then `STOPPED: no refuge could be reached` | a failed cycle could not park the arm anywhere the collision world calls free, so it is left where it stands **with the motors still on** — a limp arm holds itself up with nothing, and letting go while stranded drops it. Clear the obstruction (`ros2 service call /clear_octomap std_srvs/srv/Empty`) and drive it out with `record_states.py --play` |
| `STOPPED: cannot reach pre_pick` | deliberate. The arm is parked clear of the surface because HOME from where it stands would sweep across the table. Usually the octomap is blocking the plan — `ros2 service call /clear_octomap std_srvs/srv/Empty`, then drive it out with `record_states.py --play`. `home_requires_pre_pick:=false` accepts the sweep instead |
| *the jaws shut on air after a descent that landed accurately* | the commanded height is wrong for **this object**, not the motion. Measured, run `1788869179` cycle 2: DESCEND landed 1.4 mm from its commanded z and the jaws closed 11 mm above a screwdriver. `grasp_z_offset` is measured from the detector's z, which is not reliably the graspable height — a tall roll of tape tolerates +10 mm, a screwdriver does not. The ladder's `lower-8mm` rungs are the automatic answer and now run on a missed grip; lower `grasp_z_offset` if it happens every time |
| *the arm presses down into the work surface* | the descent is aiming below the object. `grasp_z_offset` is added to the detected **top** of the object, so a negative value asks the tool to go below it; `grasp_max_depth` (`0.0`) is the floor that stops it, and `min_grasp_z` is **not** — that one is measured from the base. Note the two mask each other: the arm has been stopping tens of millimetres short of what it was told, so a command that is too deep can look fine until tracking improves |
| `closed-on-nothing` *while the arm is visibly holding the object* | fixed — the check compared the **commanded** finger position, which at the end of a full close is always 0.0, against `grasp_finger_min`. It now reads the measured position. Across 14 real closes the two cases separate cleanly: empty is 0.0025–0.0026 m at 1.14–1.24 Nm, holding is 0.0039–0.0175 m at 1.84–2.46 Nm, and `grasp_finger_min` (0.003) already sat in the gap. A thin object gripped below the torque cap now logs `held-under-cap` |
| *there is no Place button* | reload the page. It is in the Pick card, under Pick/Abort, and it turns blue when something is held. A browser can keep serving a copy of the panel it already has — aiohttp sends only `Last-Modified` and the browser invents its own freshness — so the page now goes out `no-cache` |
| *Place is greyed out* | nothing is held. Pick ends in `HOLDING` only when the grasp check passed; if it ended `FAILED`, there is nothing to put down. The pill next to the state says which object the robot thinks it has |
| `nothing has been picked yet` *from Place* | the same thing, said by the service. Place never flies the approach on an empty hand: it would open on nothing at the drop surface and report a successful place |
| `the gripper is empty -- the jaws are N mm apart...` *from Place* | the cycle thought it was holding something and the fingers disagree **now** — an object dropped between the pick and the press. The claim is dropped with it, so the panel stops saying "holding" |
| `nothing matching "detect piece of paper" is in view` | the drop target. It has already looked twice — once from the staging pose and once from HOME — so either the sheet is out of frame, or the phrasing is one the model does not answer to. Check with `VLM/run_in_vlm_env.sh vlm_detect.py "piece of paper" --once --source realsense`, and see [Where it gets put down](#where-it-gets-put-down) for the phrasings that were measured |
| `could not get above the place target` | the place stopped at `OVER_DROP`. `motion_log.jsonl` says which of the two it was: an `arrival` row with outcome `off-target` means a posture was found and the arm did not take it up — it closed less than `arrival_progress` of the distance, which in practice means it never left the staging pose. No such row means no posture reached the point at all, and the sheet is outside this arm's envelope. Measured at 14:24, three places in a row stopped here because the check was a flat 50 mm against an arm that settles 0.115 rad off in joint1 — about 100 mm at the tool — which is now judged on the distance closed instead |
| `the tool could not be brought over the drop point` | the horizontal correction above the sheet did not fly, and the tool is neither on the column nor over the paper. Coming down from there is a diagonal across the work surface with the object in the jaws, so nothing is lowered — the object stays held and the arm stays put. The sheet is at the edge of this arm's envelope: move it closer, or put it where the other arm can reach it |
| `every one of N plans crosses into the other arm's half` | the carry guard. The arm is holding something, the staging pose is the goal, and every plan the planner offered swung the tool across the middle where the other arm is parked. The arm stops where it is, still holding, rather than flying it. Press Place from there if the drop surface is in reach, or park the other arm out of the way — `boot_other_arm:=home` — and press Pick again. `carry_guard:=false` flies whatever the planner gives |
| *the object is placed too high, and falls* | it cannot be more than `place_max_drop` (10 mm) above the sheet — over that the place **refuses**. The height is measured: the pick records how far the tool was above the surface the object stood on at the moment the jaws closed, so the tool goes that much plus `place_clearance` above the paper. If the place starts refusing for this, the descent is not reaching — that spot is at the edge of the arm's envelope |
| `the gripper closed but nothing was held` | `grasp_finger_min` or the grasp height is wrong for this object — **read the `CLOSE_GRIPPER` line above it first**: if it says `off-target`, the jaws were not where they were sent and nothing about the detector or the fingers is wrong. Measured once at 29 mm high with the descent plan ending exactly on the point |

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
