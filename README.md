# OpenArm IK Environment

Runs the OpenArm bimanual exoskeleton with GPU-accelerated cuMotion kinematics,
MoveIt 2, and real hardware interfaces.

There are two ways to run it. **Use the native one** — see
[Why native](#why-native-and-not-the-container) below.

---

## 🚀 Quick Start (native — recommended)

One-time setup, if `native/` has not been built on this machine yet:

```bash
native/bootstrap.sh
```

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
