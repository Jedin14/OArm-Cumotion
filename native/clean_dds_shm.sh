#!/usr/bin/env bash
# Remove FastDDS shared-memory files that belong to processes that are gone.
#
# Every live ROS 2 process (FastDDS, shared-memory transport) maps its own
# segment and port files in /dev/shm (fastrtps_<id>, fastrtps_portNNNN) and those
# of the peers it sends to. A segment or port no process maps belongs to a
# process that is gone (a crash, a Ctrl-C escalated to SIGKILL). Such a leftover
# PORT can be broken: on 2026-10-29 fastrtps_port7427 (from a 09:26 launch) made
# every new Python node that drew it deaf -- the controller spawner,
# boot_pre_pick, the cuMotion planner, the web GUI and the box picker's camera
# bridge all hung -- and `fastdds shm clean` left it alone.
#
# Only the main files decide: their *_el lock files and sem.*_mutex semaphores
# are NOT mapped even while in use, so they go only with their main file (or
# when it is already gone).
#
#   native/clean_dds_shm.sh            remove the leftovers
#   native/clean_dds_shm.sh --dry-run  only list them
#
# Safe with ROS running: files of live processes are never touched.
shopt -s nullglob
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

in_use="$(cat /proc/[0-9]*/maps 2>/dev/null | grep -o '/dev/shm/fastrtps_[^ ]*' | sort -u)"
gone=()

for f in /dev/shm/fastrtps_*; do
    [[ "${f}" == *_el ]] && continue                 # goes with its main file
    grep -qxF "${f}" <<< "${in_use}" && continue      # a live process uses it
    gone+=("${f}" "${f}_el")
    name="${f#/dev/shm/fastrtps_}"
    [[ "${name}" == port* ]] && gone+=("/dev/shm/sem.fastrtps_${name}_mutex")
done
# lock files and mutexes whose main file is already gone
for f in /dev/shm/fastrtps_*_el; do
    [[ -e "${f%_el}" ]] || gone+=("${f}")
done
for f in /dev/shm/sem.fastrtps_port*_mutex; do
    port="${f#/dev/shm/sem.fastrtps_}"
    [[ -e "/dev/shm/fastrtps_${port%_mutex}" ]] || gone+=("${f}")
done

removed=0
for f in "${gone[@]}"; do
    [[ -e "${f}" ]] || continue
    if (( DRY )); then
        echo "leftover: ${f}"
    else
        rm -f "${f}" 2>/dev/null && removed=$((removed + 1))
    fi
done
(( DRY )) || echo "FastDDS shared memory: ${removed} leftover file(s) removed"
