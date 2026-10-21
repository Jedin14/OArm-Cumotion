#!/usr/bin/env bash
#
# WEB GUI: every arm tool in one browser page -- click a point, detect and
# pick boxes, TEACH, tools, ARUCO EXT, the live ArUco markers, CAL ARMS,
# camera CALIBRATE, OVERLAY, ANCHOR + REFIT, STOP -- plus the pick stack's
# ROS topics (arm_state, arm_action, pick_target), and the box model.
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

# --click -> click:=true (the other arguments go to the launch file as they are)
ARGS=()
for a in "$@"; do
    if [[ "${a}" == "--click" ]]; then ARGS+=("click:=true"); else ARGS+=("${a}"); fi
done

exec ros2 launch "${WS_DIR}/launch/web_gui.launch.py" "${ARGS[@]}"
