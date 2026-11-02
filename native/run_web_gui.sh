#!/usr/bin/env bash
#
# WEB GUI: every arm tool in one browser page -- click a point, detect and
# pick boxes, TEACH, tools, ARUCO EXT, the live ArUco markers, CAL ARMS,
# camera CALIBRATE, OVERLAY, ANCHOR + REFIT, STOP -- plus the pick stack's
# ROS topics (all under /beast_arm: arm_state, arm_action, held_object, ...),
# and the box model.
#
# Bring the robot up first, unchanged:
#
#   native/run_launch_everything.sh
#
# then:
#
#   native/run_web_gui.sh                 # open http://beast:8080 on the laptop
#   native/run_web_gui.sh port:=8090 host:=127.0.0.1 model:=false
#   native/run_web_gui.sh --click         # click-only, like click_to_move: no box model
set -eo pipefail

NATIVE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null

# every earlier instance of the pick tools stopped first (port 8080, GPU memory)
"${NATIVE_DIR}/stop_pick_tools.sh" || exit 1

# --click -> click:=true (the other arguments go to the launch file as they are)
ARGS=()
for a in "$@"; do
    if [[ "${a}" == "--click" ]]; then ARGS+=("click:=true"); else ARGS+=("${a}"); fi
done

# FastDDS shared-memory files left by ROS processes that are gone (a crash, a
# Ctrl-C escalated to SIGKILL): a broken leftover port makes every new node that
# draws it deaf (2026-10-29). Only files no live process uses are removed.
"${NATIVE_DIR}/clean_dds_shm.sh" || true

exec ros2 launch "${WS_DIR}/app/launch/web_gui.launch.py" "${ARGS[@]}"
