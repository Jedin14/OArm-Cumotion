#!/usr/bin/env bash
#
# PICK STACK: the arms as a ROS 2 module (pick_stack.py) plus the box model
# (box_model_node.py). A window if there is a display, headless if not.
#
# Bring the robot up first, unchanged:
#
#   native/run_launch_everything.sh
#
# then:
#
#   native/run_pick_stack.sh                       # gui:=auto approach:=auto model:=true
#   native/run_pick_stack.sh gui:=off approach:=top
#
# Drive it with topics (see pick_stack.py), e.g.
#   ros2 topic pub --once /arm_action std_msgs/msg/String "{data: ready_to_pick}"
#   ros2 topic echo /arm_action/status
set -eo pipefail

NATIVE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null

# every earlier instance of the pick tools stopped first (port 8080, GPU memory)
"${NATIVE_DIR}/stop_pick_tools.sh" || exit 1

exec ros2 launch "${WS_DIR}/launch/pick_stack.launch.py" "$@"
