#!/usr/bin/env bash
#
# One-command install for the 7DOF-OArm workspace.
#
#   ./install.sh                 # everything (native cuMotion stack + VLM)
#   ./install.sh --native-only   # skip the VLM detector environment
#   ./install.sh --vlm-only      # only the VLM detector environment
#   ./install.sh --check         # verify an existing install, change nothing
#
# What this does NOT do, by design:
#
#   * it never runs apt install, and never writes to /usr, /opt or
#     /usr/local. The ROS packages the host is missing are downloaded as debs
#     and unpacked into native/root, a private prefix (see native/fetch_debs.sh).
#   * it never installs a Python package outside the two project venvs, and
#     never reads the host's ~/.local stack -- PYTHONNOUSERSITE=1 is set for
#     every pip/uv invocation and exported by both runtime wrappers.
#
# The only host-level change the workspace needs is a single symlink, which
# needs root once; this script prints the exact command rather than running it.
#
# Everything is idempotent. Re-run after changing a requirements file.
#
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WS_DIR}"

DO_NATIVE=1
DO_VLM=1
CHECK_ONLY=0

case "${1:-}" in
    --native-only) DO_VLM=0 ;;
    --vlm-only)    DO_NATIVE=0 ;;
    --check)       CHECK_ONLY=1 ;;
    "")            ;;
    -h|--help)     sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" \
                       | sed -e '$d' -e 's/^# \?//'; exit 0 ;;
    *)             echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
esac

banner() {
    echo
    echo "==================================================================="
    echo "  $*"
    echo "==================================================================="
    echo
}

fail=0
need() {
    local what="$1" hint="$2"
    if command -v "${what}" > /dev/null 2>&1; then
        printf '   ok      %-14s %s\n' "${what}" "$(command -v "${what}")"
    else
        printf '   MISSING %-14s %s\n' "${what}" "${hint}"
        fail=1
    fi
}

# --- preflight --------------------------------------------------------------
banner "preflight"

need python3 "apt install python3"
need git     "apt install git"
need curl    "apt install curl"
need dpkg    "part of any Debian/Ubuntu base system"

PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "${PY_VER}" == "3.10" ]]; then
    printf '   ok      %-14s %s\n' "python ver" "${PY_VER}"
else
    printf '   MISSING %-14s %s\n' "python ver" \
        "found ${PY_VER}, need 3.10 (ROS Humble's interpreter)"
    fail=1
fi

if python3 -c 'import venv' 2> /dev/null; then
    printf '   ok      %-14s %s\n' "venv module" "available"
else
    printf '   MISSING %-14s %s\n' "venv module" "apt install python3.10-venv"
    fail=1
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
    printf '   ok      %-14s %s\n' "ros humble" "/opt/ros/humble"
else
    printf '   MISSING %-14s %s\n' "ros humble" "/opt/ros/humble/setup.bash not found"
    fail=1
fi

# Every nvidia-smi call below is guarded with `|| true`: under `set -o pipefail`
# a failing driver would otherwise abort this script mid-preflight, which is the
# opposite of reporting the problem.
GPU_INFO=""
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
                2>/dev/null | head -1 || true)"
fi

if [[ -n "${GPU_INFO}" ]]; then
    printf '   ok      %-14s %s\n' "nvidia" "${GPU_INFO}"
elif command -v nvidia-smi > /dev/null 2>&1; then
    printf '   MISSING %-14s %s\n' "nvidia" \
        "nvidia-smi is installed but failed; check the driver (nvidia-smi -L)"
    fail=1
else
    printf '   MISSING %-14s %s\n' "nvidia" "no nvidia-smi; cuMotion needs a CUDA GPU"
    fail=1
fi

# --- GPU architecture -------------------------------------------------------
# native/setup.bash derives TORCH_CUDA_ARCH_LIST from this. Printing it here
# means the arch the build will target is visible before anything is compiled,
# rather than being discovered when a kernel fails to launch.
GPU_CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | sort -u | paste -sd',' - || true)"
if [[ -n "${GPU_CAPS}" ]]; then
    printf '   ok      %-14s %s\n' "gpu arch" \
        "compute ${GPU_CAPS} (sm_${GPU_CAPS//[.,]/})"
else
    printf '   MISSING %-14s %s\n' "gpu arch" \
        "nvidia-smi could not report compute_cap"
    fail=1
fi

# --- CUDA toolkit -----------------------------------------------------------
# The torch pin is cu128, so extensions cannot be compiled against these wheels
# with an older toolkit; 12.8 is also the first nvcc that knows compute_120.
# Without this check a pre-12.8 host passes preflight and then fails later, at
# the first cuRobo JIT compile.
CUDA_FOUND=""
CUDA_BEST=""
for _c in /usr/local/cuda-*/ /usr/local/cuda/; do
    [[ -x "${_c}bin/nvcc" ]] || continue
    _v="$("${_c}bin/nvcc" --version 2>/dev/null \
          | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' || true)"
    [[ -n "${_v}" ]] || continue
    CUDA_FOUND+="${_v} "
    if (( ${_v%%.*} > 12 || ( ${_v%%.*} == 12 && ${_v##*.} >= 8 ) )); then
        CUDA_BEST="${_v}"
    fi
done

if [[ -n "${CUDA_BEST}" ]]; then
    printf '   ok      %-14s %s\n' "cuda toolkit" "${CUDA_BEST} (>= 12.8)"
elif [[ -n "${CUDA_FOUND}" ]]; then
    printf '   MISSING %-14s %s\n' "cuda toolkit" \
        "found ${CUDA_FOUND% }, need >= 12.8 for the cu128 torch pin"
    fail=1
else
    printf '   MISSING %-14s %s\n' "cuda toolkit" \
        "no nvcc under /usr/local/cuda*; install CUDA >= 12.8"
    fail=1
fi

# sm_120 needs 12.8 as a hard floor; anything older cannot emit compute_120 at
# all, so call that out specifically rather than leaving it to the generic line.
if [[ "${GPU_CAPS}" == *12.0* && -z "${CUDA_BEST}" ]]; then
    echo "           this GPU is Blackwell (sm_120); nvcc < 12.8 cannot target it"
fi

if command -v uv > /dev/null 2>&1; then
    printf '   ok      %-14s %s\n' "uv" "$(uv --version | awk '{print $2}')"
else
    printf '   note    %-14s %s\n' "uv" "not found; pip will be used (slower)"
fi

AVAIL_GB="$(df -BG --output=avail "${WS_DIR}" | tail -1 | tr -dc '0-9')"
if [[ "${AVAIL_GB}" -ge 30 ]]; then
    printf '   ok      %-14s %s\n' "disk" "${AVAIL_GB} GB free"
else
    printf '   note    %-14s %s\n' "disk" \
        "${AVAIL_GB} GB free; a full install needs roughly 30 GB"
fi

if [[ ${fail} -ne 0 ]]; then
    echo
    echo "preflight failed -- resolve the MISSING entries above and re-run." >&2
    exit 1
fi

if [[ ${CHECK_ONLY} -eq 1 ]]; then
    exec "${WS_DIR}/check_isolation.sh"
fi

# --- native stack -----------------------------------------------------------
if [[ ${DO_NATIVE} -eq 1 ]]; then
    banner "native cuMotion + MoveIt environment  (native/, ~4.5 GB, slow)"
    ./native/bootstrap.sh
fi

# --- VLM stack --------------------------------------------------------------
if [[ ${DO_VLM} -eq 1 ]]; then
    banner "VLM detector environment  (VLM/.venv)"
    ./VLM/bootstrap.sh
fi

# --- isolation proof --------------------------------------------------------
banner "isolation check"
./check_isolation.sh || true

# --- what is left for the operator -----------------------------------------
banner "install complete"

if [[ ! -e /workspaces/isaac_ros-dev ]]; then
    cat <<EOF
  One step remains, and it needs root exactly once.

  This workspace hardcodes the container mount point in 96 files (openarm.yml,
  launch_everything.launch.py, the mesh paths inside openarm.urdf). A symlink
  makes all of them work unchanged:

    sudo mkdir -p /workspaces && sudo ln -s '${WS_DIR}' /workspaces/isaac_ros-dev

EOF
fi

cat <<'EOF'
  Then, to run the robot:

    source native/setup.bash
    native/run_launch_everything.sh

  Bring the CAN interfaces up first (needs root -- it configures kernel
  network devices):

    sudo ip link set can0 down
    sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
    sudo ip link set can0 up

  See README.md for the rest.

EOF
