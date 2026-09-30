#!/usr/bin/env bash
#
# One report on what every motor is doing, for pasting into a bug report.
#
#   native/check_motors.sh            # to the terminal
#   native/check_motors.sh > out.txt  # to a file
#
# Run it with the robot up. It moves nothing and commands nothing: every
# line below is read from a topic or a log.
set -uo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${NATIVE_DIR}/.." && pwd)"
set +u
# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null 2>&1
set -u

echo "=== when ==="
date
echo
echo "=== is the driver the rebuilt one ==="
# If these predate the fix, the status below will all be -1 and the arm is
# still running the build that discards the motor status byte.
ls -l --time-style=+"%Y-%m-%d %H:%M:%S" \
    "${NATIVE_DIR}/install/openarm_can/lib/libopenarm_can.a" \
    "${NATIVE_DIR}/install/openarm_hardware/lib/libopenarm_hardware.so" \
    2>&1
echo
echo "=== what the motors say about themselves ==="
echo "1 enabled, 0 not enabled, 8 over-voltage, 9 under-voltage,"
echo "10 over-current, 11 driver over-temp, 12 motor over-temp,"
echo "13 lost comms, 14 overloaded, -1 not reported"
timeout 15 python3 "${NATIVE_DIR}/tests/motor_status.py" 2>&1
echo
echo "=== what each motor did or did not do at activation ==="
# verify_enabled() writes these. Nothing here means the arm was brought up
# before the fix, or the log has rotated.
grep -hiE "did NOT enable|did not enable|fault state|are enabled|Activating OpenArm|OpenArm V10 activated" \
    $(ls -t "${HOME}"/.ros/log/*/launch.log "${HOME}"/.ros/log/*.log 2>/dev/null | head -20) \
    2>/dev/null | tail -30
echo
echo "=== is the detector running ==="
timeout 10 ros2 topic info /vlm/detections 2>&1 | sed -n '1,3p'
echo "one detection, if it answers within 25 s:"
timeout 25 ros2 topic echo --once /vlm/detections 2>&1 | head -12
echo
echo "=== nodes up ==="
timeout 10 ros2 node list 2>&1 | sort
