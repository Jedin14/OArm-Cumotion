#!/usr/bin/env bash
# Run a Python script in VLM/.venv with ROS available and nothing else.
#
#   VLM/run_in_vlm_env.sh vlm_detector_node.py --ros-args -p prompt:="detect cup"
#
# Why the scrubbing matters: native/setup.bash exports a PYTHONPATH and
# LD_LIBRARY_PATH pointing at the cuRobo overlay, which carries numpy 1.26.4 and
# torch 2.7.0+cu128 (the ABI cuMotion's prebuilt kernels need). PYTHONPATH beats
# a venv's own site-packages, so launching a node from a native-env shell --
# which is exactly what pick_place.launch.py does -- would hand PaliGemma the
# wrong numpy and the wrong torch. Clearing those before sourcing ROS leaves
# /opt/ros/humble plus VLM/.venv (numpy 2.2.6, torch 2.14+cu130) and nothing
# else.
#
# The two stacks only ever meet over DDS, so the ROS networking variables are
# the one thing that must survive.
set -euo pipefail

VLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VLM_DIR}/.venv"
ROS_SETUP=/opt/ros/humble/setup.bash

if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") <script.py> [args...]" >&2
    exit 2
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "error: ${VENV} not found. Create it with:" >&2
    echo "  cd ${VLM_DIR} && uv venv" >&2
    echo "  uv pip install --index-url https://download.pytorch.org/whl/nightly/cu130 torch torchvision" >&2
    echo "  uv pip install -r requirements.txt" >&2
    exit 1
fi

SCRIPT="$1"
shift
[[ -f "${SCRIPT}" ]] || SCRIPT="${VLM_DIR}/${SCRIPT}"
if [[ ! -f "${SCRIPT}" ]]; then
    echo "error: no such script: ${SCRIPT}" >&2
    exit 1
fi

# Keep the DDS-relevant settings, drop everything that selects a Python stack.
_keep_domain="${ROS_DOMAIN_ID:-}"
_keep_rmw="${RMW_IMPLEMENTATION:-}"
_keep_localhost="${ROS_LOCALHOST_ONLY:-}"

unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH \
      VIRTUAL_ENV PYTHONHOME CPATH PKG_CONFIG_PATH \
      TORCH_CUDA_ARCH_LIST NVCC_APPEND_FLAGS

# set +u because ROS's setup.bash reads AMENT_TRACE_SETUP_FILES unguarded.
set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u

[[ -n "${_keep_domain}" ]] && export ROS_DOMAIN_ID="${_keep_domain}"
[[ -n "${_keep_rmw}" ]] && export RMW_IMPLEMENTATION="${_keep_rmw}"
[[ -n "${_keep_localhost}" ]] && export ROS_LOCALHOST_ONLY="${_keep_localhost}"

export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1      # hide ~/.local from both stacks

exec "${VENV}/bin/python" "${SCRIPT}" "$@"
