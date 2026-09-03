# shellcheck shell=bash
#
# Activate the native (container-free) 7DOF-OArm + cuMotion environment.
#
#   source native/setup.bash
#
# Everything this pulls in lives under native/: the ament overlay in
# native/root, the Python stack in native/venv, and the workspace build in
# native/build / native/install. The host's /opt/ros/humble is used read-only as
# the underlay; the host's ~/.local Python stack is deliberately made invisible.
#
# Safe to source more than once.

_NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_WS_DIR="$(cd "${_NATIVE_DIR}/.." && pwd)"
_OVERLAY_ROOT="${_NATIVE_DIR}/root"
_OVERLAY_PREFIX="${_OVERLAY_ROOT}/opt/ros/humble"
_VENV="${_NATIVE_DIR}/venv"

if [[ -n "${ISAAC_NATIVE_ENV:-}" ]]; then
    echo "native env already active (${ISAAC_NATIVE_ENV})"
    return 0 2>/dev/null || exit 0
fi

# --- sanity -----------------------------------------------------------------
for _needed in "${_OVERLAY_PREFIX}" "${_VENV}"; do
    if [[ ! -d "${_needed}" ]]; then
        echo "native env is not built yet: missing ${_needed}" >&2
        echo "run: native/bootstrap.sh" >&2
        return 1 2>/dev/null || exit 1
    fi
done

# --- 1. host ROS underlay ---------------------------------------------------
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

# --- 2. workspace-local ament overlay --------------------------------------
# Prepended so the overlay's packages (cuMotion, cuRobo, the MoveIt pieces the
# host lacks, librealsense2) win over the host prefix.
export AMENT_PREFIX_PATH="${_OVERLAY_PREFIX}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
# root/usr is a prefix too: the overlay carries a few ordinary Ubuntu -dev
# packages there (GLUT headers for the moveit_ros_perception source build, GLFW
# for librealsense) that the host does not have installed.
export CMAKE_PREFIX_PATH="${_OVERLAY_PREFIX}:${_OVERLAY_ROOT}/usr${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export PATH="${_OVERLAY_PREFIX}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${_OVERLAY_PREFIX}/lib:${_OVERLAY_PREFIX}/lib/x86_64-linux-gnu:${_OVERLAY_ROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Both python dirs matter: the debs split their payload between
# lib/python3.10/site-packages (cuRobo, the cuMotion nodes) and
# local/lib/python3.10/dist-packages (the generated message modules, e.g.
# isaac_ros_cumotion_interfaces). The container's PYTHONPATH carried both.
export PYTHONPATH="${_OVERLAY_PREFIX}/lib/python3.10/site-packages:${_OVERLAY_PREFIX}/local/lib/python3.10/dist-packages:${_OVERLAY_ROOT}/usr/lib/python3/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"
# Headers from the overlay (freeglut/glfw/librealsense) for source builds.
export CPATH="${_OVERLAY_PREFIX}/include:${_OVERLAY_ROOT}/usr/include${CPATH:+:${CPATH}}"
export PKG_CONFIG_PATH="${_OVERLAY_ROOT}/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"

# --- 3. isolated Python stack ----------------------------------------------
# The venv carries numpy 1.26.4 / torch 2.7.0+cu128, which is the ABI cuRobo's
# prebuilt CUDA extensions are linked against. PYTHONNOUSERSITE hides the host's
# ~/.local stack (torch 2.12+cu130, numpy 2.2.6) in both directions.
export VIRTUAL_ENV="${_VENV}"
export PATH="${_VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME

# --- 4. GPU / cuMotion runtime ---------------------------------------------
# The cuRobo debs ship cubins for sm_75/86/89 only, so on anything newer --
# Blackwell (sm_120) is the case this was built on -- kernels are JIT-compiled
# from the shipped PTX, and by torch's extension builder for anything cuRobo
# compiles itself. Both need an arch list, and the latter needs a CUDA toolkit
# matching the cu128 wheels.
#
# The arch is detected rather than hardcoded: building for the wrong one
# produces cubins the card cannot execute, and it fails at launch rather than at
# build time. Set TORCH_CUDA_ARCH_LIST yourself to override the detection (e.g.
# to build fat binaries for several cards).
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    # compute_cap is reported as e.g. "12.0", which is already the format
    # TORCH_CUDA_ARCH_LIST wants. With several GPUs installed, take the
    # distinct capabilities so the build covers all of them.
    # `|| true` matters: this file is sourced by bootstrap.sh and build_ws.sh,
    # which run under `set -o pipefail`, so a failing driver would abort the
    # build instead of falling through to the warning below.
    _CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
             | tr -d ' ' | sort -u | paste -sd';' - || true)"
    if [[ -n "${_CAPS}" ]]; then
        export TORCH_CUDA_ARCH_LIST="${_CAPS}+PTX"
    else
        echo "  WARNING: could not read compute capability from nvidia-smi;" >&2
        echo "  leaving TORCH_CUDA_ARCH_LIST unset (torch will guess)." >&2
    fi
fi

# Cap registers per thread when cuRobo JIT-compiles its kernels. sm_120 only --
# see below for why it must not be applied blindly.
#
# Why this is needed: cuRobo's LBFGS step kernel launches with one thread per
# optimisation variable, i.e. horizon x dof -- 28 x 14 = 392 threads/block for
# this bimanual arm. Its stable compile_m<27> variant needs 118 registers when
# built for sm_89 (the newest arch NVIDIA shipped cubins for) but 168 when built
# for sm_120. 168 x 392 = 65,856 registers per block, just over the hardware
# limit of 65,536, so every trajopt launch died with "too many resources
# requested for launch" -- meaning cuMotion could plan IK but never a trajectory.
#
# 160 x 392 = 62,720 fits. Every other cuRobo kernel peaks at 143 registers, so
# this cap only binds on the one kernel that is over budget, and costs it a
# small amount of spilling.
#
# On sm_75/86/89 the prebuilt cubins are used and the kernel fits in 118
# registers anyway, so the cap buys nothing and only forces needless spilling --
# hence it is applied only where it is actually needed.
#
# Raise the thread count and this needs revisiting: the safe cap is
# floor(65536 / (horizon * dof)) rounded down to a multiple of 8.
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *12.0* ]]; then
    export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+${NVCC_APPEND_FLAGS} }-maxrregcount=160"
fi

# CUDA toolkit for the JIT builds. The torch pin is cu128, so a toolkit older
# than 12.8 cannot compile extensions against these wheels, and 12.8 is also the
# first release whose nvcc knows compute_120. Pick the newest toolkit available
# rather than one hardcoded path.
if [[ -z "${CUDA_HOME:-}" ]]; then
    for _cuda in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda; do
        if [[ -x "${_cuda}/bin/nvcc" ]]; then
            _ver="$("${_cuda}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
            # 12.8 <= ver, compared as two integers rather than as a float
            if [[ -n "${_ver}" ]] \
               && (( ${_ver%%.*} > 12 || ( ${_ver%%.*} == 12 && ${_ver##*.} >= 8 ) )); then
                export CUDA_HOME="${_cuda}"
                export PATH="${CUDA_HOME}/bin:${PATH}"
                break
            fi
        fi
    done
fi
# Keep warp's and torch's JIT caches inside the workspace instead of ~/.cache.
export WARP_CACHE_PATH="${_NATIVE_DIR}/cache/warp"
export TORCH_EXTENSIONS_DIR="${_NATIVE_DIR}/cache/torch_extensions"
export TRITON_CACHE_DIR="${_NATIVE_DIR}/cache/triton"
mkdir -p "${WARP_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"

# --- 4b. snap contamination -------------------------------------------------
# A terminal opened inside a snap -- VS Code's, most often, since that is where
# this workspace gets edited -- exports GTK/GIO variables pointing back into the
# snap. rviz2 then loads a GTK module from /snap/code, whose RPATH pulls in
# core20's glibc 2.31 libpthread alongside the host's 2.35, and it dies at
# startup without drawing anything:
#
#   rviz2: symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/
#   libpthread.so.0: undefined symbol: __libc_pthread_init, GLIBC_PRIVATE
#
# The symptom is "RViz never opened" while every other node comes up fine, so it
# reads as a launch-file problem rather than an environment one. GTK_PATH is the
# variable that actually breaks it -- bisected -- but all of these point into
# the snap and would do the same to any other GUI node (rqt, the octomap gater's
# Tk window), so all of them go. Each is cleared only when its value really is
# inside a snap, which makes this a no-op in an ordinary terminal.
for _var in GTK_PATH GTK_EXE_PREFIX GIO_MODULE_DIR GDK_PIXBUF_MODULE_FILE \
            GDK_PIXBUF_MODULEDIR GSETTINGS_SCHEMA_DIR LOCPATH; do
    case "${!_var:-}" in
        /snap/*|"${HOME}"/snap/*)
            unset "${_var}"
            _snap_scrubbed="${_snap_scrubbed:+${_snap_scrubbed} }${_var}"
            ;;
    esac
done
if [ -n "${_snap_scrubbed:-}" ]; then
    echo "note: cleared snap-provided ${_snap_scrubbed} (they crash rviz2)"
fi

# --- 5. ROS middleware ------------------------------------------------------
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# The container exported FASTRTPS_DEFAULT_PROFILES_FILE=rtps_udp_profile.xml,
# which sets useBuiltinTransports=false and forces UDPv4 only. That profile is a
# container workaround: it exists so DDS does not try to use the shared-memory
# transport across the container boundary under --network host.
#
# Running natively there is no boundary, so shared memory is both available and
# faster -- which matters for the depth/point cloud topics feeding the octomap.
# The profile also sets maxInitialPeersRange=400, which on this host produces a
# steady stream of "sequence size exceeds remaining buffer" warnings from
# FastDDS (76 of them during one planner-node run; zero without the profile).
#
# So it is off by default. Set ISAAC_NATIVE_USE_UDP_PROFILE=1 before sourcing to
# get container-identical DDS behaviour instead.
if [[ "${ISAAC_NATIVE_USE_UDP_PROFILE:-0}" == "1" ]]; then
    _PROFILE="${_WS_DIR}/src/isaac_ros_common/docker/middleware_profiles/rtps_udp_profile.xml"
    if [[ -f "${_PROFILE}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${_PROFILE}"
    fi
fi

# --- 6. container path compatibility ---------------------------------------
# 96 files in this workspace hardcode the container's mount point, including
# openarm.yml, launch_everything.launch.py and the mesh paths inside
# openarm.urdf. A single symlink makes every one of them work unchanged, and
# keeps the workspace usable from inside the container as well. It is the only
# thing outside native/ that the native setup needs, and it needs root once.
if [[ ! -e /workspaces/isaac_ros-dev ]]; then
    echo
    echo "  NOTE: /workspaces/isaac_ros-dev does not exist yet."
    echo "  This workspace hardcodes that path in 96 files (openarm.yml,"
    echo "  launch_everything.launch.py, mesh paths in openarm.urdf, ...)."
    echo "  Create the symlink once:"
    echo
    echo "    sudo mkdir -p /workspaces && sudo ln -s '${_WS_DIR}' /workspaces/isaac_ros-dev"
    echo
fi

# --- 7. workspace overlay ---------------------------------------------------
export ISAAC_ROS_WS="${_WS_DIR}"
if [[ -f "${_NATIVE_DIR}/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "${_NATIVE_DIR}/install/setup.bash"
fi

# The patched cuMotion planner, same injection launch_everything.launch.py does.
export PYTHONPATH="${_WS_DIR}/src/isaac_ros_cumotion_override:${PYTHONPATH}"

export ISAAC_NATIVE_ENV="${_NATIVE_DIR}"

echo "native env active"
echo "  workspace : ${_WS_DIR}"
echo "  overlay   : ${_OVERLAY_PREFIX}"
echo "  python    : $(command -v python3)"
echo "  build     : ${_NATIVE_DIR}/build -> ${_NATIVE_DIR}/install"
echo "  gpu arch  : ${TORCH_CUDA_ARCH_LIST:-<unset>}${NVCC_APPEND_FLAGS:+  (${NVCC_APPEND_FLAGS})}"
echo "  cuda      : ${CUDA_HOME:-<none found>}"

unset _NATIVE_DIR _WS_DIR _OVERLAY_ROOT _OVERLAY_PREFIX _VENV _needed _PROFILE \
      _CAPS _cuda _ver _var _snap_scrubbed
