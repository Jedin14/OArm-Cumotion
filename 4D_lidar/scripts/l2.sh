#!/usr/bin/env bash
# One command for the Unitree L2: start spins the lidar up and runs the whole pipeline;
# stopping it (Ctrl+C, `l2.sh stop`, systemd stop) always spins the lidar down.
#
#   scripts/l2.sh start [launch args]   spin up + run everything in this terminal; Ctrl+C = stop + spin down
#                                        e.g.  scripts/l2.sh start rviz:=false
#                                              scripts/l2.sh start sim:=true      (no hardware)
#   scripts/l2.sh stop                  stop a running "start" from another terminal (+ spin down)
#   scripts/l2.sh status                is the pipeline running / is the lidar spinning?
#   scripts/l2.sh spin-up | spin-down   motor only (pipeline must not be running: it owns the port)
#
# Uses the installed config (install/l2_perception/share/l2_perception/config/l2.yaml);
# set L2_CONFIG=/path/to/l2.yaml to use another one.

WS="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
RUN_DIR="${L2_RUN_DIR:-${XDG_RUNTIME_DIR:-/tmp}/unitree-l2}"
PIDFILE="$RUN_DIR/l2.pid"
mkdir -p "$RUN_DIR"

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash" || { echo "build the workspace first: cd \"$WS\" && colcon build"; exit 1; }
CFG="${L2_CONFIG:-$WS/install/l2_perception/share/l2_perception/config/l2.yaml}"
DRIVER="$WS/install/l2_driver/lib/l2_driver/l2_driver_node"
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}] {message}"

say() { echo "[l2.sh] $*"; }

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

lidar_cmd() {  # start | stop | status  (one-shot, exits when done)
  "$DRIVER" --ros-args --params-file "$CFG" -p command:="$1" -p wait_for_clock_sync:=0.0
}

cmd_start() {
  if running; then say "already running (pid $(cat "$PIDFILE")). Use: $0 stop"; exit 1; fi
  SIM=0
  for a in "$@"; do [[ "$a" == "sim:=true" ]] && SIM=1; done
  echo $$ > "$PIDFILE"
  LPID=""
  STOPPING=0
  shutdown() {
    [[ $STOPPING == 1 ]] && return
    STOPPING=1
    say "stopping pipeline ..."
    # Signal the launch's whole process group (launch + every node), so nothing is orphaned.
    if [[ -n "$LPID" ]] && kill -0 -- -"$LPID" 2>/dev/null; then
      kill -INT -- -"$LPID" 2>/dev/null
      for _ in $(seq 1 40); do kill -0 -- -"$LPID" 2>/dev/null || break; sleep 0.5; done
      if kill -0 -- -"$LPID" 2>/dev/null; then
        say "nodes did not exit in 20 s, terminating them"
        kill -TERM -- -"$LPID" 2>/dev/null; sleep 3; kill -KILL -- -"$LPID" 2>/dev/null
      fi
    fi
    if [[ $SIM == 0 ]]; then
      # The driver already sends stop-rotation on exit; this repeats it (idempotent) and
      # verifies, so the lidar also stops if the driver crashed or was killed.
      say "spinning the lidar down ..."
      lidar_cmd stop
    fi
    rm -f "$PIDFILE"
    say "stopped."
  }
  trap shutdown INT TERM
  trap shutdown EXIT
  say "starting (config: $CFG) - Ctrl+C to stop and spin the lidar down"
  set -m   # job control: the launch gets its own process group (and SIGINT is not ignored)
  ros2 launch l2_perception l2.launch.py "$@" &
  LPID=$!
  set +m
  wait "$LPID"
  shutdown
}

cmd_stop() {
  if running; then
    local pid; pid="$(cat "$PIDFILE")"
    say "stopping pipeline (pid $pid) ..."
    kill -TERM "$pid"   # handled like Ctrl+C (SIGINT may be ignored by background shells)
    for _ in $(seq 1 120); do kill -0 "$pid" 2>/dev/null || { say "stopped."; return 0; }; sleep 0.5; done
    say "still running after 60 s"; return 1
  fi
  say "pipeline not running; spinning the lidar down ..."
  lidar_cmd stop
}

cmd_status() {
  if running; then
    say "pipeline running (pid $(cat "$PIDFILE")); stable cloud rate:"
    timeout 5 ros2 topic hz /l2/points 2>/dev/null | grep -m1 "average rate" || say "no /l2/points messages"
  else
    say "pipeline not running; asking the lidar ..."
    lidar_cmd status
  fi
}

case "${1:-}" in
  start) shift; cmd_start "$@" ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  spin-up|spin-down)
    if running; then say "pipeline is running and owns the port; use: $0 stop"; exit 1; fi
    if [[ "$1" == spin-up ]]; then lidar_cmd start; else lidar_cmd stop; fi ;;
  *) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
