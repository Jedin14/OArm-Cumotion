#!/usr/bin/env bash
#
# Build the isolated Python environment for the VLM detector, in VLM/.venv.
#
#   VLM/bootstrap.sh
#
# Idempotent: re-running syncs the venv to requirements.txt without rebuilding
# it from scratch. Pass --recreate to start clean.
#
# Nothing is installed outside VLM/.venv. In particular the host's ~/.local
# stack is never written to and never read (PYTHONNOUSERSITE=1 everywhere), and
# no apt package is touched.
#
# Why this venv is separate from native/venv: this stack needs torch 2.14 nightly
# +cu130 and numpy 2.2.6 for PaliGemma, while native/venv is pinned to torch
# 2.7.0+cu128 / numpy 1.26.4 -- the ABI cuRobo's prebuilt CUDA kernels are
# linked against. The two can never share an interpreter. run_in_vlm_env.sh
# keeps them apart at runtime; this script keeps them apart at install time.
#
set -euo pipefail

VLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${VLM_DIR}"

VENV="${VLM_DIR}/.venv"

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

export PYTHONNOUSERSITE=1

step() { echo; echo "---- $* ----"; }

# --- 0. pick an installer ---------------------------------------------------
# uv is ~10x faster and resolves the nightly index more reliably, but a plain
# pip fallback keeps this script working on a machine without it.
if command -v uv > /dev/null 2>&1; then
    USE_UV=1
    echo "installer : uv $(uv --version | awk '{print $2}')"
else
    USE_UV=0
    echo "installer : pip (uv not found -- https://astral.sh/uv to speed this up)"
fi

# --- 1. create the venv -----------------------------------------------------
if [[ ${RECREATE} -eq 1 && -d "${VENV}" ]]; then
    step "removing existing ${VENV}"
    rm -rf "${VENV}"
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
    step "creating ${VENV}"
    # No --system-site-packages here, deliberately: this stack must not see the
    # host's numpy/torch at all. rclpy comes from /opt/ros/humble via PYTHONPATH
    # at runtime (see run_in_vlm_env.sh), not from the system site dir.
    if [[ ${USE_UV} -eq 1 ]]; then
        uv venv --python 3.10 "${VENV}"
    else
        python3 -m venv "${VENV}"
        "${VENV}/bin/python" -m pip install --quiet --upgrade pip wheel
    fi
else
    echo "venv      : ${VENV} (exists, reusing)"
fi

# --- 2. install the pinned stack --------------------------------------------
# One resolve, straight from requirements.txt. The nightly cu130 index is
# declared inside that file, so the pins there are authoritative -- notably the
# exact torch nightly (2.14.0.dev...+cu130) rather than "whatever is newest
# tonight", which would drift out from under the frozen transitive pins.
step "installing the pinned stack from requirements.txt"
if [[ ${USE_UV} -eq 1 ]]; then
    VIRTUAL_ENV="${VENV}" uv pip install -r requirements.txt
else
    # A venv created by `uv venv` has no pip in it, so an existing venv plus a
    # machine where uv has since gone missing would otherwise fail here.
    if ! "${VENV}/bin/python" -m pip --version > /dev/null 2>&1; then
        echo "   bootstrapping pip into the existing venv"
        "${VENV}/bin/python" -m ensurepip --upgrade
    fi
    "${VENV}/bin/python" -m pip install -r requirements.txt
fi

# --- 3. verify --------------------------------------------------------------
step "verifying the stack"
"${VENV}/bin/python" - <<'PY'
import sys

bad = 0
for name in ("numpy", "torch", "torchvision", "transformers", "cv2", "pyrealsense2"):
    try:
        mod = __import__(name)
        print(f"   ok   {name:16s} {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"   FAIL {name:16s} {type(exc).__name__}: {exc}")
        bad += 1

# The whole point of the split: this interpreter must NOT be able to see the
# host's ~/.local packages, and must be the venv's own python.
import site
if any("/.local/" in p for p in sys.path):
    print("   FAIL sys.path contains ~/.local -- isolation is broken")
    bad += 1
else:
    print("   ok   ~/.local is not on sys.path")

try:
    import torch
    print(f"   cuda available: {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        cap = "".join(map(str, torch.cuda.get_device_capability(0)))
        print(f"  {torch.cuda.get_device_name(0)} sm_{cap}")
    else:
        print("  (no GPU visible -- the detector will run on CPU, very slowly)")
except Exception:
    pass

raise SystemExit(1 if bad else 0)
PY

echo
echo "---- VLM environment ready ----"
echo
echo "  run the detector : VLM/run_vlm_detector.sh"
echo "  run any script   : VLM/run_in_vlm_env.sh <script.py> [args...]"
echo
