#!/usr/bin/env bash
#
# Build moveit_ros_perception from source, against the host's MoveIt.
#
# openarm_bimanual_moveit_config/config/sensors_3d.yaml asks for
# occupancy_map_monitor/DepthImageOctomapUpdater, which lives in
# moveit_ros_perception -- a package ros-humble-desktop does not install.
#
# It cannot come from apt: the only build published on packages.ros.org is newer
# than this host's MoveIt and is linked against libgeometric_shapes.so.2.3.4
# while the host has 2.3.2. Installing it drags the newer geometric_shapes into
# the overlay, and the host's move_group -- built against 2.3.2 -- then
# segfaults. Building from source at the host's own MoveIt version keeps one
# consistent ABI.
#
# Source lands in native/src_moveit (git, so it is easy to inspect or bump) and
# is built into the same native/install tree as the rest of the workspace.
#
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${NATIVE_DIR}/src_moveit"

# Match the host's installed MoveIt exactly: 2.5.9-1jammy... -> tag 2.5.9.
HOST_MOVEIT_VER="$(dpkg-query -W -f='${Version}' ros-humble-moveit-core 2>/dev/null | cut -d- -f1)"
if [[ -z "${HOST_MOVEIT_VER}" ]]; then
    echo "error: ros-humble-moveit-core is not installed on the host" >&2
    exit 1
fi
echo "== host MoveIt is ${HOST_MOVEIT_VER}; building moveit_ros_perception from that tag =="

mkdir -p "${SRC_DIR}"
if [[ ! -d "${SRC_DIR}/moveit2/.git" ]]; then
    git clone --depth 1 --branch "${HOST_MOVEIT_VER}" \
        https://github.com/moveit/moveit2.git "${SRC_DIR}/moveit2"
else
    git -C "${SRC_DIR}/moveit2" fetch --depth 1 origin "tag" "${HOST_MOVEIT_VER}" || true
    git -C "${SRC_DIR}/moveit2" checkout "${HOST_MOVEIT_VER}"
fi

# Build only the perception package. COLCON_IGNORE everything else in the tree so
# colcon does not try to rebuild all of MoveIt over the host's copy.
find "${SRC_DIR}/moveit2" -mindepth 1 -maxdepth 1 -type d -not -name '.git' \
    -not -name 'moveit_ros' -exec touch {}/COLCON_IGNORE \;
find "${SRC_DIR}/moveit2/moveit_ros" -mindepth 1 -maxdepth 1 -type d \
    -not -name 'perception' -exec touch {}/COLCON_IGNORE \;

set +u
# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null
set -u

cd "${NATIVE_DIR}/.."
set -x
colcon build \
    --base-paths "${SRC_DIR}" \
    --build-base "${NATIVE_DIR}/build" \
    --install-base "${NATIVE_DIR}/install" \
    --packages-select moveit_ros_perception \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    "$@"
