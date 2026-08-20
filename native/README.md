# Native (container-free) 7DOF-OArm + cuMotion environment

Runs the 7DOF-OArm bimanual stack with GPU cuMotion planning directly on the host,
with everything it needs kept inside this `native/` directory. The container
setup (`src/isaac_ros_common/scripts/run_dev.sh`, and the root `build/` +
`install/` trees) is untouched and still works, so you can switch back at any
time.

## Quick start

```bash
native/bootstrap.sh
```

This builds the native environment only. To set up the whole workspace — this
plus the VLM detector's separate venv, with a preflight and an isolation check —
use the top-level installer instead:

```bash
./install.sh
```

Then, once per machine (needs root, see [The one system change](#the-one-system-change)):

```bash
sudo mkdir -p /workspaces && sudo ln -s /home/mr/workspaces/isaac_ros-dev /workspaces/isaac_ros-dev
```

To use it:

```bash
source native/setup.bash
native/run_launch_everything.sh          # same launch args as the container flow
```

CAN bring-up is unchanged from the container workflow:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 up
```

## What this does and does not touch

**Kept inside `native/`:**

| Path | What it is |
|---|---|
| `debs/` | The 13 `.deb` files the environment is built from, kept for provenance |
| `root/` | Those debs unpacked into a workspace-local ament overlay (`root/opt/ros/humble`) |
| `venv/` | Python env: numpy 1.26.4, torch 2.7.0+cu128, warp-lang, cuRobo's deps |
| `build/`, `install/` | The colcon build, separate from the container's `build/` + `install/` |
| `cache/` | cuRobo/warp/triton JIT caches (kept here rather than `~/.cache`) |
| `src_moveit/` | MoveIt source checkout, for the one package built from source |
| `aptroot/` | Private apt state: own `sources.list`, own lists and cache |
| `logs/` | Build logs, per-deb file listings, the resolved download list |

**Never modified:** `/opt/ros/humble`, `/usr`, the dpkg database, `/etc/apt`, and
`~/.local` — including the host's own `torch 2.12.1+cu130` and `numpy 2.2.6`,
which this environment neither reads nor changes (`setup.bash` sets
`PYTHONNOUSERSITE=1`, so `~/.local` is invisible in both directions).

`fetch_debs.sh` resolves dependencies with apt pointed at a private root, reading
the real dpkg status read-only so host-installed packages count as satisfied. No
package is ever installed system-wide; `extract_debs.sh` unpacks them with
`dpkg -x` instead.

### The one system change

`/workspaces/isaac_ros-dev` must exist as a symlink to this workspace. 96 files
here hardcode the container's mount point — `openarm.yml`, `openarm.urdf` (mesh
paths), `launch_everything.launch.py`, `isaac_ros_cumotion_params.yaml` and more.
One symlink makes all of them work unchanged, and keeps the workspace usable from
inside the container too. It is a single symlink and one empty directory; nothing
is installed.

## Why the environment is put together this way

**cuRobo comes from NVIDIA's Isaac apt repo, not pip.** `ros-humble-curobo-core`
and the five `isaac-ros-cumotion-*` packages are the whole cuMotion dependency
closure — 8 packages, no GXF or NITROS needed. They unpack into the overlay.

**The Python pins are not optional.** cuRobo's prebuilt CUDA extensions are
linked against `torch 2.7.0+cu128`'s ABI and numpy 1.26. Installing that system-
wide would downgrade the host's torch and numpy, which is the whole reason for
the venv.

**Two source patches, copied from the container's `Dockerfile.user`**, applied by
`apply_patches.sh` to the local copies only:
- `torch/utils/cpp_extension.py`: `-std=c++20` → `-std=c++17`
- cuRobo `world_mesh.py`: `wp.torch.device_from_torch` → `wp.device_from_torch`
  (the API moved in warp-lang 1.15)

**cuRobo JIT-compiles its kernels here, and that is expected.** The shipped
`.so` files are built against a newer libtorch (`c10::cuda::SetDevice(signed
char, bool)`) and only carry sm_75/86/89 cubins, so cuRobo falls back to its JIT
path. `warm_kernels.py` pre-builds all five extensions into `cache/` so the first
planner start is not slow. The container behaves the same way — see below.

**`moveit_ros_perception` is built from source** (`build_moveit_perception.sh`)
rather than installed from apt. `sensors_3d.yaml` needs its
`DepthImageOctomapUpdater`, but the only build published on packages.ros.org is
newer than this host's MoveIt and links `libgeometric_shapes.so.2.3.4` while the
host has 2.3.2. Taking it from apt drags the newer `geometric_shapes` in, and
making the whole chain consistent that way would mean shadowing 141 host packages
(`rclcpp`, `rmw`, `rviz2` included). Building the one package against the host's
own MoveIt 2.5.9 keeps a single ABI. Everything else MoveIt-related comes from
the host's `ros-humble-desktop`.

**`isaac_ros_common` is not built from source** (`build_ws.sh` skips it). Its
`CMakeLists` requires VPI, an NVIDIA library the container pulled from the Jetson
OTA repo, and the only file using it is `vpi_utilities.cpp`, which nothing in the
7DOF-OArm / qnbot / realsense stack references. The prebuilt deb of the same
package supplies the package and its CMake extras.

**The container's FastDDS profile is off by default.**
`rtps_udp_profile.xml` sets `useBuiltinTransports=false` to keep DDS off the
shared-memory transport across the container boundary. Natively there is no
boundary, shared memory is faster for the depth topics, and the profile's
`maxInitialPeersRange=400` produces a steady stream of `sequence size exceeds
remaining buffer` warnings (76 in one planner run; zero without it). Set
`ISAAC_NATIVE_USE_UDP_PROFILE=1` before sourcing for container-identical DDS.

## Two real bugs this setup had to fix

### cuMotion trajectory planning was impossible on this GPU

cuRobo's LBFGS step kernel launches one thread per optimisation variable:
`action_horizon × dof` = 28 × 14 = **392 threads per block** for this bimanual
arm. Its `compile_m<27>` variant needs **118 registers** compiled for sm_89 (the
newest architecture NVIDIA ships cubins for) but **168** compiled for sm_120.
168 × 392 = 65,856 registers per block, against a hardware limit of 65,536 — so
every trajopt launch died with `CUDA error: too many resources requested for
launch`. IK worked (14 threads/block); trajectories never could.

`setup.bash` therefore sets `NVCC_APPEND_FLAGS=-maxrregcount=160`
(160 × 392 = 62,720, fits). Every other cuRobo kernel peaks at 143 registers, so
the cap binds only on the one kernel that is over budget. **If you raise
`num_steps` in `cumotion_planning.yaml`, recompute it:** the safe cap is
`floor(65536 / (action_horizon × dof))` rounded down to a multiple of 8, where
`action_horizon = num_steps - 4`.

### cuMotion cannot run in the container on this GPU at all

The container's CUDA toolkit is **12.2** (inherited from its
`tritonserver:23.10-py3` base). Since cuRobo has to JIT-compile here, and 12.2's
`nvcc` does not know `compute_120`, the build fails outright:

```
nvcc fatal : Unsupported gpu architecture 'compute_120'
RuntimeError: Error building extension 'kinematics_fused_cu'
```

Verified by running the same node inside `isaac_ros_dev-x86_64:latest` on this
machine. The host has CUDA 12.8, which is why the native path gets past this. So
switching off the container is not only viable here — for cuMotion on the 5070 Ti
it is currently the only thing that works.

## An X display is required

`sensors_3d.yaml` enables `occupancy_map_monitor/DepthImageOctomapUpdater`, whose
mesh self-filter uses freeglut/OpenGL. With no `DISPLAY` it prints
`freeglut failed to open display ''` and takes `move_group` down with SIGSEGV
before any planner loads. This is not native-specific — `run_dev.sh` forwards
`DISPLAY` and mounts `/tmp/.X11-unix` for the same reason — but it is easy to hit
over ssh. `run_launch_everything.sh` checks for it.

For a headless check of the planning stack:

```bash
python3 native/tests/test_move_group_planners.py
```

## Scripts

| Script | Purpose |
|---|---|
| `bootstrap.sh` | Runs everything below in order. Idempotent. |
| `fetch_debs.sh` | Resolve + download debs via the private apt root |
| `extract_debs.sh` | Unpack into `root/`, relocate paths, fix dangling symlinks |
| `apply_patches.sh` | The two container source patches |
| `build_ws.sh` | colcon build of `src/` into `build/` + `install/` |
| `build_moveit_perception.sh` | Source build of `moveit_ros_perception` |
| `warm_kernels.py` | Pre-compile cuRobo's five CUDA extensions |
| `verify_overlay.sh` | Check soname resolution + key Python imports |
| `run_launch_everything.sh` | Native equivalent of the root `run_launch_everything.sh` |

## Verified working

- `verify_overlay.sh`: 117 shared objects, all dependencies resolve; numpy
  1.26.4, torch 2.7.0+cu128, warp 1.15.0, cuRobo, rclpy, `isaac_ros_cumotion`
  all import; CUDA available on the RTX 5070 Ti at sm_120.
- cuRobo batch IK on the 7DOF-OArm URDF: 14 DOF, 10/10 solved in 40 ms, max
  position error 8 µm.
- cuMotion trajopt warmup completes (the LBFGS fix above).
- `cumotion_goal_set_planner_node`: *"cuMotion is ready for planning queries!"*
  after a 3.3 s warmup.
- `move_group` headless: loads ompl, chomp, pilz and cumotion pipelines, and
  reports `MoveGroup context using planning plugin
  isaac_ros_cumotion_moveit/CumotionPlanner`.
- 22 workspace packages build, plus `moveit_ros_perception`.

Not verified, because it needs the physical robot and a desktop session: the full
`launch_everything.launch.py` run with real CAN hardware, RViz, and the live
RealSense octomap.

## Disk

About 8.4 GB total:

| Path | Size |
|---|---|
| `venv/` | 7.2 GB (torch + the `nvidia-*-cu12` wheels dominate) |
| `aptroot/` | 426 MB (mostly the Ubuntu universe package indexes) |
| `root/` | 312 MB |
| `install/` | 241 MB |
| `build/` | 69 MB |
| `cache/` | 51 MB (cuRobo JIT output) |
| `debs/` | 49 MB |
| `src_moveit/` | 35 MB |

`aptroot/var/lib/apt/lists` is only needed to re-resolve dependencies; deleting it
costs nothing but an `apt-get update` on the next `fetch_debs.sh`. `build/` can be
removed once you are done building.

For comparison, `isaac_ros_dev-x86_64:latest` is 42 GB and
`ros2_humble-image:latest` another 41.8 GB (they share layers). This machine has
36 GB free, so if you settle on the native setup, dropping those images reclaims
far more than this costs:

```bash
docker image rm isaac_ros_dev-x86_64:latest ros2_humble-image:latest
```
