#!/usr/bin/env bash
#
# The two source fixes the Isaac ROS container applied in docker/Dockerfile.user.
# Both are applied to the workspace-local copies only -- the torch one to
# native/venv, the cuRobo one to the native/root overlay. Nothing under /usr,
# /opt or ~/.local is modified.
#
# Idempotent: re-running is a no-op.
#
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_SP="${NATIVE_DIR}/venv/lib/python3.10/site-packages"
OVERLAY_SP="${NATIVE_DIR}/root/opt/ros/humble/lib/python3.10/site-packages"

# 1. PyTorch's JIT extension builder defaults to -std=c++20, which does not
#    compile against the CUDA 12.8 headers used here. cuRobo needs this whenever
#    it JIT-builds a kernel (which it does on Blackwell: the shipped cuRobo
#    cubins only cover sm_75/86/89).
TORCH_EXT="${VENV_SP}/torch/utils/cpp_extension.py"
if [[ ! -f "${TORCH_EXT}" ]]; then
    echo "error: ${TORCH_EXT} not found -- create the venv first" >&2
    exit 1
fi
if grep -q 'std=c++20' "${TORCH_EXT}"; then
    cp -n "${TORCH_EXT}" "${TORCH_EXT}.orig"
    sed -i 's/-std=c++20/-std=c++17/g' "${TORCH_EXT}"
    echo "patched: torch cpp_extension.py  (-std=c++20 -> -std=c++17)"
else
    echo "already patched: torch cpp_extension.py"
fi

# 2. cuRobo 3.2.5 calls a warp API that moved: wp.torch.device_from_torch is
#    wp.device_from_torch in warp-lang 1.15.
CUROBO_MESH="${OVERLAY_SP}/curobo/geom/sdf/world_mesh.py"
if [[ ! -f "${CUROBO_MESH}" ]]; then
    echo "error: ${CUROBO_MESH} not found -- run ./extract_debs.sh first" >&2
    exit 1
fi
if grep -q 'wp\.torch\.device_from_torch' "${CUROBO_MESH}"; then
    cp -n "${CUROBO_MESH}" "${CUROBO_MESH}.orig"
    sed -i 's/wp\.torch\.device_from_torch/wp.device_from_torch/g' "${CUROBO_MESH}"
    # Drop the stale bytecode so the patched source is what gets imported.
    rm -f "${OVERLAY_SP}"/curobo/geom/sdf/__pycache__/world_mesh.*.pyc
    echo "patched: curobo world_mesh.py  (wp.torch.device_from_torch -> wp.device_from_torch)"
else
    echo "already patched: curobo world_mesh.py"
fi
