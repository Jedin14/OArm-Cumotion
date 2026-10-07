#!/usr/bin/env bash
#
# Click a point in the camera image and the TCP goes near it, planned by
# cuMotion around the octomap. No VLM, no grasp model.
#
# Bring the robot up first, unchanged:
#
#   native/run_launch_everything.sh
#
# then, in a second terminal on the same machine:
#
#   native/run_click_to_move.sh
#   native/run_click_to_move.sh --arm left --standoff 0.10
#
# Keys and options are in click_to_move.py (--help lists the options).
set -eo pipefail

NATIVE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null

if [[ -z "${DISPLAY:-}" ]]; then
    echo "error: DISPLAY is not set -- the click window needs a desktop session." >&2
    exit 1
fi

exec python3 "${WS_DIR}/click_to_move.py" "$@"
