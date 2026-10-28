#!/usr/bin/env bash
#
# Stop every running instance of the pick tools -- before a relaunch, so no
# old GUI holds port 8080 and no old box picker holds ~4 GB of GPU memory
# (one left over made the next run's first detection run out, 2026-10-27).
#
#   native/stop_pick_tools.sh           stop them
#   native/stop_pick_tools.sh --list    only show what would be stopped
#
# Stopped: the web GUI, the pick stack, the box model node and the box
# picker it runs (pick_server.py, ros_bridge.py), the OpenCV tools
# (click_to_move, auto_box, calibrate), and the "ros2 launch" of the GUI /
# pick stack. NOT touched: the robot (run_launch_everything: MoveIt, cuMotion,
# the camera, the motors).
#
# Ctrl-C first (they shut down cleanly: the picker gets "quit"), then TERM,
# then KILL for anything still there after a few seconds.

PATTERNS=(
    'launch/web_gui.launch.py'
    'launch/pick_stack.launch.py'
    'web_gui.py'
    'pick_stack.py'
    'box_model_node.py'
    'box_picker/src/.venv/bin/python pick_server.py'
    'pick_server.py --pitch'
    'box_picker_server.py'
    'ros_bridge.py'
    'click_to_move.py'
    'auto_box.py'
    'calibrate.py'
)

me=$$
pids_of() {
    local out=()
    for p in "${PATTERNS[@]}"; do
        while read -r pid; do
            [[ -n "$pid" && "$pid" != "$me" && "$pid" != "$PPID" ]] && out+=("$pid")
        done < <(pgrep -u "$(id -u)" -f -- "$p" 2>/dev/null)
    done
    # never this script, its shell, or the robot stack
    for pid in "${out[@]}"; do
        args=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
        [[ -z "$args" ]] && continue
        [[ "$args" == *stop_pick_tools* || "$args" == *launch_everything* ]] && continue
        exe=$(basename "${args%% *}")
        [[ "$exe" == python* ]] || continue          # the tools, the picker, ros2 launch: Python
        echo "$pid"
    done | sort -u
}

running=$(pids_of)
if [[ "${1:-}" == "--list" ]]; then                 # show, stop nothing
    for pid in $running; do
        printf '%s  %s\n' "$pid" "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-110)"
    done
    exit 0
fi
if [[ -z "$running" ]]; then
    echo "pick tools: nothing running"
    exit 0
fi
echo "pick tools: stopping earlier instances:"
for pid in $running; do
    printf '  %s  %s\n' "$pid" "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-110)"
done

for sig in INT TERM KILL; do
    left=$(pids_of)
    [[ -z "$left" ]] && break
    kill -s "$sig" $left 2>/dev/null
    for _ in $(seq 1 20); do                 # up to 4 s each step
        sleep 0.2
        [[ -z "$(pids_of)" ]] && break
    done
done

left=$(pids_of)
if [[ -n "$left" ]]; then
    echo "pick tools: could not stop: $left" >&2
    exit 1
fi
if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ':8080 '; then
    echo "pick tools: note: something else still listens on port 8080" >&2
fi
echo "pick tools: all stopped"
