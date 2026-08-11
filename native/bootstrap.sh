#!/usr/bin/env bash
#
# Build the whole native environment from scratch, in order.
#
#   native/bootstrap.sh
#
# Every step is idempotent, so re-running after a change is fine. Nothing is
# installed outside this directory; see README.md for exactly what touches the
# host and what does not.
#
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${NATIVE_DIR}"

step() { echo; echo "############ $* ############"; echo; }

step "1/7  download debs (private apt root, host dpkg untouched)"
./fetch_debs.sh

step "2/7  unpack into the workspace-local ament overlay"
./extract_debs.sh

step "3/7  create the isolated python venv"
if [[ ! -x venv/bin/python3 ]]; then
    python3 -m venv --system-site-packages venv
    PYTHONNOUSERSITE=1 ./venv/bin/python3 -m pip install --quiet --upgrade pip wheel
fi
PYTHONNOUSERSITE=1 ./venv/bin/python3 -m pip install -r requirements.txt

step "4/7  apply the container's two source patches locally"
./apply_patches.sh

step "5/7  build the workspace"
./build_ws.sh

step "6/7  build moveit_ros_perception from source (octomap updater)"
./build_moveit_perception.sh

step "7/7  pre-compile cuRobo CUDA kernels and verify"
set +u
# shellcheck disable=SC1091
source ./setup.bash > /dev/null
set -u
python3 warm_kernels.py
./verify_overlay.sh

echo
echo "############ native environment ready ############"
echo
echo "  source native/setup.bash"
echo
if [[ ! -e /workspaces/isaac_ros-dev ]]; then
    WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"
    echo "  One remaining step, needs root once (see README.md):"
    echo "    sudo mkdir -p /workspaces && sudo ln -s '${WS_DIR}' /workspaces/isaac_ros-dev"
    echo
fi
