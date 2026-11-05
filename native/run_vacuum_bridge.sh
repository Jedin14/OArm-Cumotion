#!/usr/bin/env bash
#
# THE VACUUM RELAY on its own: /beast_arm/vacuum (std_msgs/Bool, from the web
# GUI / pick stack) -> the ESP32 over USB serial, '1' on / '0' off.
#
# Terminal 1 (run_launch_everything.sh) starts it already: use this only after
# run_launch_everything.sh vacuum:=false (two on one port: the second is refused).
# It keeps running when the GUI or the pick stack restart (stop_pick_tools.sh
# leaves it alone) and when the ESP32 is unplugged (it reopens the port and
# writes the state again).
#
#   native/run_vacuum_bridge.sh                         # our ESP32, by id (app/vacuum_bridge.py)
#   native/run_vacuum_bridge.sh port:=/dev/serial/by-id/usb-...   # or port:=auto
#   native/run_vacuum_bridge.sh baud:=115200 resend:=1.0
#
# (the parameters: app/vacuum_bridge.py). Restarting it while a box hangs on the
# cup can drop the box: many ESP32 boards restart when the port opens.
set -eo pipefail

NATIVE_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null

PARAMS=()
for a in "$@"; do
    if [[ "${a}" != *:=* ]]; then
        echo "usage: $0 [port:=/dev/ttyUSB0] [baud:=115200] [resend:=1.0] [topic:=/beast_arm/vacuum]" >&2
        exit 2
    fi
    PARAMS+=(-p "${a}")
done

# FastDDS leftovers from dead processes can make a new node deaf (2026-10-29):
# a deaf bridge would never switch the vacuum. Only unused files are removed.
"${NATIVE_DIR}/clean_dds_shm.sh" || true

exec python3 "${WS_DIR}/app/vacuum_bridge.py" --ros-args "${PARAMS[@]}"
