#!/usr/bin/env bash
#
# The whole thing in one command: robot, MoveIt, cuMotion, camera, the PaliGemma
# detector, the pick-and-place orchestrator, and the panel you type into.
#
#   native/run_pick_place_demo.sh                # everything, incl. the web UI
#   native/run_pick_place_demo.sh --fake         # rehearsal, no arms, no CAN
#   native/run_pick_place_demo.sh --no-can       # do not touch the interfaces
#   native/run_pick_place_demo.sh arm:=left
#
# The web UI comes up on http://<this machine>:8088 -- open it from wherever
# you are sitting. web:=false leaves it out, ui:=true brings back the old Tk
# panel as well.
#
# grasp_model:=true also starts the GraspNet server, which publishes ranked
# 6-DoF grasps on /grasp/candidates. It only publishes; nothing acts on them
# unless use_grasp_model is on too. Fetch it first with
# grasp/fetch_graspnet.sh.
#
# Any pick_place_demo.launch.py argument can be appended; --show-args lists them.
#
# --fake is the rehearsal: the arms are simulated by mock_components and the
# whole sequence plays out in RViz, while the camera, the depth stream and the
# PaliGemma detector are all real. So the object really is found where it is,
# and only the moving is pretend. It needs no CAN bus and no arms plugged in.
#
# It also neutralises the two checks that cannot pass on simulated arms, which
# is the difference between watching a cycle and watching the retry ladder burn
# six attempts: mock_components reports the finger exactly where it was
# commanded, so the grip never registers, and the object never physically
# moves, so the place is never confirmed. --fake sets grasp_finger_min:=-1.0
# and object_moved_eps:=0.0 for that reason and no other -- never pass those on
# real hardware, they are exactly what makes the retries work.
#
# The CAN interfaces are brought up here, which needs root -- configuring a
# kernel network device does. Expect one sudo prompt on a cold boot and none
# afterwards, because an interface already up at the right bitrate is left
# alone. --no-can skips it, --fake does not need it.
#
# Run this on the machine the arms are wired to. Over a remote mount the sudo
# would authenticate on the wrong host and the CAN devices would not be there
# to configure.
#
# Everything else below is a check that turns a confusing downstream failure
# into one line of text. None of it starts anything.
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
#
# --fake expands here rather than being passed through, so that the CAN check
# below sees it too. Expanded arguments go first in ARGS, so anything the
# caller writes after --fake still wins: a repeated launch argument takes its
# last value.
FAKE_HARDWARE=false
BRING_CAN_UP=true
ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --no-can)
            BRING_CAN_UP=false
            ;;
        --fake|--sim)
            FAKE_HARDWARE=true
            ARGS+=(use_fake_hardware:=true
                   grasp_finger_min:=-1.0
                   object_moved_eps:=0.0)
            ;;
        use_fake_hardware:=true)
            FAKE_HARDWARE=true
            ARGS+=("${arg}")
            ;;
        *)
            ARGS+=("${arg}")
            ;;
    esac
done

# The bitrates the arms run at. Both interfaces, both arms, every time --
# these are a property of the hardware, not a choice.
CAN_BITRATE=1000000
CAN_DBITRATE=5000000

can_is_up() {
    ip link show "$1" 2>/dev/null | grep -q "state UP"
}

bring_can_up() {
    local iface="$1"
    echo "bringing ${iface} up at ${CAN_BITRATE}/${CAN_DBITRATE} (needs sudo)"
    # Down first: the bitrate of a running interface cannot be changed, and
    # setting it on one that is already up fails with a bare "Device or
    # resource busy". Down on an already-down interface is harmless.
    sudo ip link set "${iface}" down \
        && sudo ip link set "${iface}" type can \
               bitrate "${CAN_BITRATE}" dbitrate "${CAN_DBITRATE}" fd on \
        && sudo ip link set "${iface}" up
}

if [[ "${FAKE_HARDWARE}" == false ]]; then
    for iface in can0 can1; do
        if ! ip link show "${iface}" &> /dev/null; then
            echo "error: ${iface} does not exist." >&2
            echo "Plug the adapter in, or rehearse without the arms:" >&2
            echo "  native/run_pick_place_demo.sh --fake" >&2
            exit 1
        fi
        if can_is_up "${iface}"; then
            continue
        fi
        if [[ "${BRING_CAN_UP}" == false ]]; then
            echo "error: ${iface} is down and --no-can was given. Either drop" >&2
            echo "--no-can, or bring it up yourself with:" >&2
            echo >&2
            echo "  sudo ip link set ${iface} down" >&2
            echo "  sudo ip link set ${iface} type can bitrate ${CAN_BITRATE} dbitrate ${CAN_DBITRATE} fd on" >&2
            echo "  sudo ip link set ${iface} up" >&2
            exit 1
        fi
        if ! bring_can_up "${iface}"; then
            echo "error: could not bring ${iface} up." >&2
            echo "If sudo asked for a password and did not get one, run the" >&2
            echo "three commands by hand -- they are in the README -- or add" >&2
            echo "an /etc/sudoers.d rule for 'ip link set can*'." >&2
            exit 1
        fi
        if ! can_is_up "${iface}"; then
            echo "error: ${iface} still reports down after being brought up." >&2
            echo "Usually the adapter: check 'dmesg | tail' and the cabling." >&2
            exit 1
        fi
        echo "  ${iface} is up"
    done
fi

# Advisory only, never fatal: record_states.py needs the robot running, so the
# very first bringup is legitimately the one where these do not exist yet.
STATES_FILE="${WS_DIR}/pick_place_states.yaml"
for arg in "${ARGS[@]}"; do
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
    ${ARGS[@]+"${ARGS[@]}"}
