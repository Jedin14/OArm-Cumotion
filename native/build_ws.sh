#!/usr/bin/env bash
#
# Build the workspace natively, into native/build and native/install.
#
# The container's own build/ and install/ trees at the workspace root are left
# alone, so the two setups can coexist and you can fall back to the container at
# any time.
#
# Any extra arguments are forwarded to colcon, e.g.
#   native/build_ws.sh --packages-select qnbot_teleoperator
#
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"

# The ROS setup scripts reference unset variables, so -u has to stand down while
# they are sourced.
set +u
# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null
set -u

# Packages deliberately not built natively:
#
# isaac_ros_common -- its CMakeLists does find_package(vpi REQUIRED). VPI is an
#   NVIDIA library the container pulled from the Jetson OTA repo, and the only
#   thing in the package that uses it is vpi_utilities.cpp, which nothing in the
#   7DOF-OArm / qnbot / realsense stack references. The prebuilt deb of the same
#   package (3.2.5, in the overlay) supplies the package and its CMake extras,
#   so skipping the source build costs nothing here.
SKIP=(isaac_ros_common)

cd "${WS_DIR}"
mkdir -p "${NATIVE_DIR}/logs"

set -x
colcon build \
    --base-paths src \
    --build-base "${NATIVE_DIR}/build" \
    --install-base "${NATIVE_DIR}/install" \
    --packages-skip "${SKIP[@]}" \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    "$@"
