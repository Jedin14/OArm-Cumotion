# Installation

Setting up the 7DOF-OArm workspace on a machine. For what to *run* once it is
installed, see [README.md](../README.md) and [detailed_run.md](detailed_run.md).

- [Requirements](#requirements)
- [Install](#install)
- [Three Python stacks, none of them the system's](#three-python-stacks-none-of-them-the-systems)
- [GPU and CUDA](#gpu-and-cuda)
- [What needs root](#what-needs-root)
- [Portability](#portability)
- [Troubleshooting the install](#troubleshooting-the-install)

---

## Requirements

These must already be on the host. The installer checks all of them and refuses
to start if any are missing — it verifies the system, it does not provision it.

| | |
| --- | --- |
| **Ubuntu 22.04**, x86_64 | ROS Humble targets `jammy`, and the deb overlay pins those archives |
| **ROS Humble**, `ros-humble-desktop` | provides the MoveIt and ros2_control packages the overlay deliberately does not carry |
| **Python 3.10** | Humble's interpreter; the pinned wheels are built for it |
| **CUDA toolkit ≥ 12.8** | see [GPU and CUDA](#gpu-and-cuda) for why this exact floor |
| **An NVIDIA GPU** + driver | cuMotion is GPU-only |
| **~30 GB free disk** | the two venvs, the deb overlay and the JIT caches |

Not installed, and not installable by this repo: ROS itself, the CUDA toolkit,
the NVIDIA driver.

## Install

```bash
native/install.sh                  # the native cuMotion + MoveIt stack
native/install.sh --check          # verify an existing install, change nothing
cd box_picker/src && ./setup.sh    # the box picker's own environment (SAM 3.1), once
```

It runs the preflight above, then [`native/bootstrap.sh`](../native/bootstrap.sh)
(deb overlay, venv, workspace build, cuRobo kernel warm-up), then the isolation
check. The box picker's venv is made by its own
[`box_picker/src/setup.sh`](../box_picker/src/setup.sh) (uv, Python 3.11). Expect the native half to take a while — it downloads ~360 MB of debs and
builds the workspace from source.

Every step is idempotent, so re-running after editing a `requirements.txt` is
fine and cheap.

## Three Python stacks, none of them the system's

This workspace needs two mutually incompatible Python environments, and the host
already has a third. Mixing any two produces either a numpy ABI error on import
or a silently wrong torch:

| Stack | numpy | torch | Why it is pinned there |
| --- | --- | --- | --- |
| `native/venv` | 1.26.4 | 2.7.0+cu128 | the ABI cuRobo's prebuilt CUDA kernels are linked against |
| `box_picker/src/.venv` (Python 3.11) | 2.4.6 | 2.11.0+cu128 | what SAM 3.1 (the box picker) needs |
| the host | 1.21.5 (apt), 2.2.6 (`~/.local`) | 2.12+cu130 | not used by this project at all |

The separation is enforced, not just documented:

- **Nothing is installed system-wide.** `native/install.sh` never runs `apt install` and
  never writes to `/usr`, `/opt` or `/usr/local`. The ROS packages the host is
  missing are downloaded as debs and unpacked into `native/root`, a private
  prefix — the host's `/opt/ros/humble` is used read-only as the underlay.
- **The host's `~/.local` is invisible.** `PYTHONNOUSERSITE=1` is exported by
  both runtime wrappers and set for every `pip`/`uv` call, in both directions.
- **The two venvs never share an interpreter.** `box_model_node.py` starts the
  box picker with `PYTHONPATH`, `PYTHONHOME`, `LD_LIBRARY_PATH` and the ROS
  prefix variables removed, because a `PYTHONPATH` set by `native/setup.bash`
  would otherwise beat the venv's own `site-packages` and hand SAM the wrong
  numpy.

### How the two kinds of dependency are handled

`native/venv` is created with `--system-site-packages`, which is what lets both
cases work at once:

| Kind | Example | Resolves from |
| --- | --- | --- |
| Overlaps a system module | `numpy`, `torch` | `native/venv` — its `site-packages` precedes the system's on `sys.path`, so the pin wins |
| New, not on the system | `warp-lang`, `trimesh`, `yourdfpy` | `native/venv` |
| ROS core, deliberately shared | `rclpy`, generated message modules | the host's `/opt/ros/humble` |
| Not pip-installable | `curobo`, the cuMotion nodes, `librealsense2` | the deb overlay in `native/root` |

`box_picker/src/.venv` is built the opposite way — **no** system site-packages —
because its numpy 2.4 must not see anything else. It never imports ROS: the
camera frames reach it from `box_picker/src/ros_bridge.py` (run with the ROS
Python 3.10) over TCP `127.0.0.1:50555`, and it answers `box_model_node.py` on
its stdin/stdout.

### Verifying it

```bash
native/check_isolation.sh
```

Checks that each venv resolves `numpy` and `torch` to its own pinned copies from
inside the workspace, that neither can see `~/.local`, that the two carry
different versions, and that nothing of the project leaked into the system's
`dist-packages`. A failure here is an early warning for a crash that would
otherwise surface mid-launch.

## GPU and CUDA

The GPU architecture is **detected, not hardcoded**. `native/setup.bash` reads
the compute capability from `nvidia-smi` and sets `TORCH_CUDA_ARCH_LIST` from
it, so the JIT builds target the card actually present — building for the wrong
architecture produces cubins the GPU cannot execute, and that fails at *launch*,
not at build time. `source native/setup.bash` reports what it picked:

```
  gpu arch  : 12.0+PTX  (-maxrregcount=160)
  cuda      : /usr/local/cuda-12.8
```

Set `TORCH_CUDA_ARCH_LIST` yourself before sourcing to override the detection,
e.g. to build fat binaries for several cards. The `sm_120` register cap is
applied only when the detected arch actually needs it — see the comment in
`native/setup.bash` for the register arithmetic behind it.

**CUDA ≥ 12.8** is required for two independent reasons: the torch pin is
`cu128`, so extensions cannot be compiled against these wheels with an older
toolkit, and 12.8 is the first `nvcc` that knows `compute_120`. The newest
qualifying toolkit under `/usr/local/` is used, rather than one hardcoded path.

## What needs root

Two things, both outside the installer:

```bash
# once per machine -- 16 files reference this path
sudo mkdir -p /workspaces && sudo ln -s "$PWD" /workspaces/isaac_ros-dev
```

```bash
# once per boot -- configures kernel network devices
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
```

`native/install.sh` prints the symlink command rather than running it.

## Portability

Machine-independent: the venv isolation, the pinned Python stacks, the private
ROS overlay, and the GPU arch / CUDA selection.

Still specific to this rig:

| | |
| --- | --- |
| **x86_64 + Ubuntu 22.04 + ROS Humble** | `native/setup.bash` and `extract_debs.sh` hardcode `x86_64-linux-gnu`; `fetch_debs.sh` pins the `jammy` archives. No Jetson/ARM, no 24.04. |
| **The `/workspaces/isaac_ros-dev` path** | 16 files reference it; the symlink absorbs that, but the absolute paths in the docs are this machine's. |
| **The robot itself** | the 14-DOF bimanual URDF, the D455 mount measured in `calibration/cam_org.txt`, and `can0`/`can1`. That calibration does not transfer. |

So this installs unattended on another x86_64 Ubuntu 22.04 + `ros-humble-desktop`
machine with a CUDA ≥ 12.8 GPU. Other platforms need work beyond the installer —
there is **no Docker path**: the Isaac container ships CUDA 12.2, whose `nvcc`
cannot target `compute_120`, which is the reason `native/` exists at all.

## Troubleshooting the install

| Symptom | Cause |
| --- | --- |
| numpy ABI error, or an unexpected torch version | run `native/check_isolation.sh` first — it is the most common cause and the check is instant |
| preflight fails on `cuda toolkit` | no `nvcc` ≥ 12.8 under `/usr/local/`; the detected versions are listed |
| preflight fails on `gpu arch` | `nvidia-smi` cannot report `compute_cap` — usually a broken driver, check `nvidia-smi -L` |
| a MoveIt plugin silently does not load | `source native/setup.bash && native/verify_overlay.sh` — catches overlay libraries that install fine but fail to `dlopen` |
| `freeglut failed to open display` | no `DISPLAY`; use `python3 native/tests/test_move_group_planners.py` for a headless check |
| the box picker fails at model load | see `native/logs/box_picker.log`; the SAM 3.1 weights come from `$SAM31_CHECKPOINT`, else HuggingFace (`facebook/sam3.1`) |

For deeper detail on the native environment — exactly what it touches, how the
overlay is assembled, and the two GPU-specific bugs it works around — see
[native/README.md](../native/README.md).
