#!/usr/bin/env bash
# Start the GraspNet server in third_party/grasp_venv, with ROS available and
# nothing else.
#
#   grasp/run_grasp_server.sh
#   grasp/run_grasp_server.sh --ros-args -p object_radius:=0.10
#
# The scrubbing is the whole point, and it is the same reason
# VLM/run_in_vlm_env.sh does it: native/setup.bash exports a PYTHONPATH and
# LD_LIBRARY_PATH pointing at the cuRobo overlay, which carries torch
# 2.7.0+cu128. PYTHONPATH beats a venv's own site-packages, so launching from
# a native-env shell -- which is exactly what pick_place_demo.launch.py does
# -- would hand this node the wrong torch. That matters more here than
# anywhere else in the workspace: pointnet2 and knn are compiled C++
# extensions linked against the venv's libtorch, and importing them under a
# different one is an undefined-symbol crash at best.
#
# The stacks meet over DDS, so the ROS networking variables are the one thing
# that must survive.
#
# graspnet-baseline is SJTU's, academic/non-profit noncommercial research use
# only -- see third_party/graspnet-baseline/LICENSE. grasp/fetch_graspnet.sh
# fetches it; nothing here is committed.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS="$(cd "${HERE}/.." && pwd)"
VENV="${WS}/third_party/grasp_venv"
ROS_SETUP=/opt/ros/humble/setup.bash

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "error: ${VENV} is not there." >&2
    echo "Run grasp/fetch_graspnet.sh first -- it clones graspnet-baseline," >&2
    echo "patches it, builds its CUDA extensions and downloads the weights." >&2
    exit 1
fi
if [[ ! -d "${WS}/third_party/graspnet-baseline" ]]; then
    echo "error: third_party/graspnet-baseline is missing." >&2
    echo "Run grasp/fetch_graspnet.sh." >&2
    exit 1
fi

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
export PYTHONUNBUFFERED=1

exec "${VENV}/bin/python" "${HERE}/grasp_node.py" "$@"
