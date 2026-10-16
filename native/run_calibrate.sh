#!/usr/bin/env bash
#
# CALIBRATE: the calibration tools (CAL ARMS, camera CALIBRATE, OVERLAY) with a
# live camera view of every ArUco marker, and click + MOVE to test a point.
#
# Bring the robot up first, unchanged:
#
#   native/run_launch_everything.sh
#
# then, in a second terminal on the same machine:
#
#   native/run_calibrate.sh
#
# Keys are in calibrate.py; options as click_to_move.py.
set -eo pipefail

NATIVE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null

if [[ -z "${DISPLAY:-}" ]]; then
    echo "error: DISPLAY is not set -- the window needs a desktop session." >&2
    exit 1
fi

exec python3 "${WS_DIR}/calibrate.py" "$@"
