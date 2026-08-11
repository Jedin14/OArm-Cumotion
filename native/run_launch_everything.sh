#!/usr/bin/env bash
#
# Native equivalent of the workspace's run_launch_everything.sh -- same launch
# arguments, but running on the host instead of inside the container.
#
# Bring the CAN interfaces up first (unchanged from the container workflow, and
# still the one part that genuinely needs root, because it configures kernel
# network devices):
#
#   sudo ip link set can0 down
#   sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
#   sudo ip link set can0 up
#   sudo ip link set can1 down
#   sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
#   sudo ip link set can1 up
#
set -eo pipefail

NATIVE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null

if [[ ! -e /workspaces/isaac_ros-dev ]]; then
    echo "error: /workspaces/isaac_ros-dev does not exist." >&2
    echo "launch_everything.launch.py, openarm.yml and openarm.urdf all reference" >&2
    echo "that path. Create it once with:" >&2
    echo >&2
    echo "  sudo mkdir -p /workspaces && sudo ln -s '${WS_DIR}' /workspaces/isaac_ros-dev" >&2
    exit 1
fi

# The octomap path needs an X display. sensors_3d.yaml enables
# occupancy_map_monitor/DepthImageOctomapUpdater, whose mesh self-filter uses
# freeglut/OpenGL; with no DISPLAY it prints "freeglut failed to open display"
# and takes move_group down with it (SIGSEGV) before any planner loads. This is
# equally true inside the container -- run_dev.sh forwards DISPLAY and mounts
# /tmp/.X11-unix for exactly this reason -- so it is not specific to the native
# setup, but it is easy to trip over when starting from ssh or a bare tty.
if [[ -z "${DISPLAY:-}" ]]; then
    echo "error: DISPLAY is not set." >&2
    echo "The depth-image octomap updater needs an X display, and RViz needs one too." >&2
    echo "Run this from a desktop session, or for a headless planner-only check use:" >&2
    echo "  python3 native/tests/test_move_group_planners.py" >&2
    exit 1
fi

set -u

exec ros2 launch "${WS_DIR}/launch_everything.launch.py" \
    octomap:=static \
    4d:=false \
    use_fake_hardware:=false \
    right_can_interface:=can0 \
    left_can_interface:=can1 \
    "$@"
