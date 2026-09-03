#!/usr/bin/env bash
#
# The whole thing in one command: robot, MoveIt, cuMotion, camera, the PaliGemma
# detector, the pick-and-place orchestrator, and the panel you type into.
#
#   native/run_pick_place_demo.sh
#   native/run_pick_place_demo.sh use_fake_hardware:=true    # rehearsal
#   native/run_pick_place_demo.sh arm:=left
#
# Any pick_place_demo.launch.py argument can be appended; --show-args lists them.
#
# Bring the CAN interfaces up first -- still the one part that needs root,
# because it configures kernel network devices:
#
#   sudo ip link set can0 down
#   sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
#   sudo ip link set can0 up
#   sudo ip link set can1 down
#   sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
#   sudo ip link set can1 up
#
# Everything below is a check that turns a confusing downstream failure into one
# line of text. None of it starts anything.
set -eo pipefail

NATIVE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null

if [[ ! -e /workspaces/isaac_ros-dev ]]; then
    echo "error: /workspaces/isaac_ros-dev does not exist." >&2
    echo "launch_everything.launch.py, openarm.yml and openarm.urdf all reference" >&2
    echo "that path. Create it once with:" >&2
    echo >&2
    echo "  sudo mkdir -p /workspaces && sudo ln -s '${WS_DIR}' /workspaces/isaac_ros-dev" >&2
    exit 1
fi

# RViz, the depth-image octomap updater and the Tk panel all need a real
# display; without one move_group dies on "freeglut failed to open display".
if [[ -z "${DISPLAY:-}" ]]; then
    echo "error: DISPLAY is not set." >&2
    echo "Run this from the machine's own desktop session, not plain ssh." >&2
    exit 1
fi

# Skip the CAN check when the arms are simulated -- a rehearsal needs no bus.
FAKE_HARDWARE=false
for arg in "$@"; do
    [[ "${arg}" == "use_fake_hardware:=true" ]] && FAKE_HARDWARE=true
done

if [[ "${FAKE_HARDWARE}" == false ]]; then
    for iface in can0 can1; do
        if ! ip link show "${iface}" &> /dev/null; then
            echo "error: ${iface} does not exist." >&2
            echo "Plug the adapter in, or rehearse without the arms:" >&2
            echo "  native/run_pick_place_demo.sh use_fake_hardware:=true" >&2
            exit 1
        fi
        if ! ip link show "${iface}" | grep -q "state UP"; then
            echo "error: ${iface} is down. Bring it up with:" >&2
            echo >&2
            echo "  sudo ip link set ${iface} down" >&2
            echo "  sudo ip link set ${iface} type can bitrate 1000000 dbitrate 5000000 fd on" >&2
            echo "  sudo ip link set ${iface} up" >&2
            exit 1
        fi
    done
fi

# Advisory only, never fatal: record_states.py needs the robot running, so the
# very first bringup is legitimately the one where these do not exist yet.
STATES_FILE="${WS_DIR}/pick_place_states.yaml"
for arg in "$@"; do
    [[ "${arg}" == states_file:=* ]] && STATES_FILE="${arg#states_file:=}"
done
if [[ ! -f "${STATES_FILE}" ]]; then
    echo "note: ${STATES_FILE} does not exist yet, so a pick will refuse to start."
    echo "      Once this is up, record the two poses in another terminal:"
    echo
    echo "        cd ${WS_DIR} && source native/setup.bash"
    echo "        python3 record_states.py"
    echo
fi

set -u

# Defaults first, then "$@": a repeated launch argument takes its last value,
# so anything appended on the command line wins.
exec ros2 launch "${WS_DIR}/pick_place_demo.launch.py" \
    octomap:=static \
    use_fake_hardware:=false \
    right_can_interface:=can0 \
    left_can_interface:=can1 \
    "$@"
