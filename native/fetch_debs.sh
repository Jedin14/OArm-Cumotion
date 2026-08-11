#!/usr/bin/env bash
#
# Resolve and download every Debian package the native (container-free) setup
# needs, into native/debs.
#
# Nothing here touches the host: apt runs against a private root under
# native/aptroot, with its own sources.list and its own lists/cache. The host's
# real dpkg status file is read (read-only) so that already-installed packages
# are treated as satisfied and are never re-downloaded. No package is ever
# installed system-wide -- extract_debs.sh unpacks them into native/opt instead.
#
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APT_ROOT="${NATIVE_DIR}/aptroot"
DEB_DIR="${NATIVE_DIR}/debs"

# Packages we want present. Their dependency closure is resolved automatically;
# anything the host already provides drops out of the download list.
TARGETS=(
    # cuMotion / cuRobo (NVIDIA Isaac ROS apt repo). Pulls in curobo-core,
    # isaac_ros_cumotion, its interfaces, python-utils, robot-description,
    # isaac_ros_common and nvblox-msgs.
    ros-humble-isaac-ros-cumotion-moveit

    # librealsense2 SDK. realsense-ros itself is built from src/, exactly as the
    # container did (the image ships librealsense2 but not realsense2-camera).
    ros-humble-librealsense2

    # realsense2_camera needs this and the host does not have it.
    ros-humble-diagnostic-updater

    # Build and runtime dependency of moveit_ros_perception, which
    # build_moveit_perception.sh compiles from source. Not installed on the host.
    freeglut3-dev
)
#
# Deliberately NOT taken from apt, though the container has them:
#
#   ros-humble-moveit-ros-perception
#     Supplies occupancy_map_monitor/DepthImageOctomapUpdater, which
#     openarm_bimanual_moveit_config/config/sensors_3d.yaml does need. But the
#     only build of it on packages.ros.org is newer than this host's MoveIt: it
#     is linked against libgeometric_shapes.so.2.3.4 while the host has 2.3.2.
#     Pulling it in drags the newer geometric_shapes into the overlay, and then
#     the host's own move_group binary -- built against 2.3.2 -- segfaults
#     against it. Upgrading the whole chain instead means shadowing 141 host
#     packages (rclcpp, rmw, rviz2 included), i.e. reinstalling ROS in the
#     overlay. So this one package is built from source against the host's
#     MoveIt instead: see build_moveit_perception.sh.
#
#   ros-humble-moveit / -servo / -visual-tools / -setup-*, moveit-resources-*,
#   ur-description, ur-moveit-config, ros2-control, topic-based-ros2-control
#     Nothing in the OpenArm / qnbot / realsense stack references any of these
#     (checked against every package.xml, CMakeLists and launch file in src/).
#     The host's ros-humble-desktop already provides the MoveIt and ros2_control
#     packages that are actually used, at a build that matches its own ABI.

mkdir -p "${APT_ROOT}"/etc/apt/trusted.gpg.d \
         "${APT_ROOT}"/var/lib/apt/lists/partial \
         "${APT_ROOT}"/var/cache/apt/archives/partial \
         "${DEB_DIR}"

# --- repositories -----------------------------------------------------------
# The same two the Dockerfiles use. packages.ros.org is where the host's own
# ROS debs came from, so versions stay consistent.
# The Ubuntu archives are needed too: a few ROS debs pull ordinary universe
# libraries (libglfw3, freeglut3-dev) that this host does not have installed.
cat > "${APT_ROOT}/etc/apt/sources.list" <<'EOF'
deb http://packages.ros.org/ros2/ubuntu jammy main
deb https://isaac.download.nvidia.com/isaac-ros/release-3 jammy release-3.0
deb http://us.archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse
deb http://us.archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse
EOF

if [[ ! -f "${APT_ROOT}/etc/apt/trusted.gpg.d/ros.gpg" ]]; then
    cp /usr/share/keyrings/ros-archive-keyring.gpg \
       "${APT_ROOT}/etc/apt/trusted.gpg.d/ros.gpg"
fi
if [[ ! -f "${APT_ROOT}/etc/apt/trusted.gpg.d/ubuntu.gpg" ]]; then
    cp /usr/share/keyrings/ubuntu-archive-keyring.gpg \
       "${APT_ROOT}/etc/apt/trusted.gpg.d/ubuntu.gpg"
fi
if [[ ! -f "${APT_ROOT}/etc/apt/trusted.gpg.d/isaac.gpg" ]]; then
    curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
        | gpg --dearmor > "${APT_ROOT}/etc/apt/trusted.gpg.d/isaac.gpg"
fi

# --- private apt invocation -------------------------------------------------
apt_private() {
    apt-get \
        -o Dir::Etc::sourcelist="${APT_ROOT}/etc/apt/sources.list" \
        -o Dir::Etc::sourceparts=/dev/null \
        -o Dir::Etc::trusted=/dev/null \
        -o Dir::Etc::trustedparts="${APT_ROOT}/etc/apt/trusted.gpg.d" \
        -o Dir::State="${APT_ROOT}/var/lib/apt" \
        -o Dir::State::status=/var/lib/dpkg/status \
        -o Dir::Cache="${APT_ROOT}/var/cache/apt" \
        -o Debug::NoLocking=1 \
        -o APT::Sandbox::User=root \
        "$@"
}

echo "== updating private package lists =="
apt_private update

echo "== resolving dependency closure =="
# --print-uris resolves without installing. Only packages that are missing (or
# would be upgraded) on the host appear here.
apt_private install --no-install-recommends --print-uris -y "${TARGETS[@]}" \
    | grep -E "^'" > "${NATIVE_DIR}/logs/uris.raw"

# Decide, per package, whether the overlay needs to carry it.
#
#   - not installed on the host      -> download (a genuinely new package)
#   - installed, host version older  -> download (the overlay shadows the host
#                                       copy at runtime via AMENT_PREFIX_PATH /
#                                       LD_LIBRARY_PATH; the host's own install
#                                       is left in place and unmodified)
#   - installed, host version >= ours-> skip, the host copy is fine
#
# The middle case is not optional: the MoveIt 2.5.9 debs, for instance, are
# linked against libgeometric_shapes.so.2.3.4 while this host has 2.3.2, so
# without the newer soname in the overlay every octomap updater fails to load.
: > "${NATIVE_DIR}/logs/uris.txt"
: > "${NATIVE_DIR}/logs/shadowed-host-packages.txt"
: > "${NATIVE_DIR}/logs/skipped-host-is-current.txt"
while read -r line; do
    url="$(echo "${line}" | awk '{print $1}' | tr -d "'")"
    debname="$(basename "${url}")"
    pkg="${debname%%_*}"
    # Version is the middle field of name_version_arch.deb, %-decoded (apt
    # escapes ':' in epochs as %3a).
    cand_ver="$(echo "${debname}" | awk -F_ '{print $2}' | sed 's/%3a/:/g')"

    host_ver="$(dpkg-query -W -f='${Version}' "${pkg}" 2>/dev/null || true)"
    if [[ -z "${host_ver}" ]]; then
        echo "${url}" >> "${NATIVE_DIR}/logs/uris.txt"
    elif dpkg --compare-versions "${host_ver}" ge "${cand_ver}"; then
        echo "${pkg} (host ${host_ver} >= ${cand_ver})" \
            >> "${NATIVE_DIR}/logs/skipped-host-is-current.txt"
    else
        echo "${pkg}: host ${host_ver} -> overlay ${cand_ver}" \
            >> "${NATIVE_DIR}/logs/shadowed-host-packages.txt"
        echo "${url}" >> "${NATIVE_DIR}/logs/uris.txt"
    fi
done < "${NATIVE_DIR}/logs/uris.raw"

count=$(wc -l < "${NATIVE_DIR}/logs/uris.txt")
shadowed=$(wc -l < "${NATIVE_DIR}/logs/shadowed-host-packages.txt")
skipped=$(wc -l < "${NATIVE_DIR}/logs/skipped-host-is-current.txt")
echo "== ${count} packages to download (${shadowed} newer than the host copy, ${skipped} left to the host) =="
if [[ ${shadowed} -gt 0 ]]; then
    echo "   overlay carries a newer version of these; the host install is untouched:"
    sed 's/^/   - /' "${NATIVE_DIR}/logs/shadowed-host-packages.txt"
fi

echo "== downloading =="
cd "${DEB_DIR}"
xargs -a "${NATIVE_DIR}/logs/uris.txt" -r -n1 -P4 \
    curl -fsSL --remote-name --continue-at - --retry 3
echo "== done: $(ls -1 "${DEB_DIR}"/*.deb 2>/dev/null | wc -l) debs in ${DEB_DIR} =="
du -sh "${DEB_DIR}"
