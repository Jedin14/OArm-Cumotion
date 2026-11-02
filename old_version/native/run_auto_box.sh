#!/usr/bin/env bash
#
# AUTO BOX: the box model finds the box to pick; PICK sends the arm to it,
# with the same planning and touch sequence as click_to_move.
#
# Bring the robot up first, unchanged:
#
#   native/run_launch_everything.sh
#
# then, in a second terminal on the same machine:
#
#   native/run_auto_box.sh
#
# Keys are in auto_box.py; options as click_to_move.py, plus --box-conf.
set -eo pipefail

NATIVE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null

if [[ -z "${DISPLAY:-}" ]]; then
    echo "error: DISPLAY is not set -- the window needs a desktop session." >&2
    exit 1
fi

exec python3 "${WS_DIR}/auto_box.py" "$@"
