#!/usr/bin/env bash
# Stop launch_everything (and the web GUI) when Ctrl-C does not: asks every node to
# stop, waits, then kills what is left, including a `ros2 launch` that hangs waiting
# for children that already exited (seen 2026-11-17). Run from any terminal on beast.
#
# Only the stack's own processes (2026-11-17 review: the old pattern matched any
# command line that merely NAMED a stack script -- `less app/web_gui.py`, an editor,
# `tail -f` on a log, a test -- and killed it):
#   - the program itself must be the stack's: the command line's FIRST word, or the
#     second after a python3 interpreter (no option in between), ends in one of the
#     names below -- so `less x/web_gui.py` or `python3 -c ... web_gui.py` never match;
#   - this user's processes only;
#   - on the robot's ROS domain only (ROS_DOMAIN_ID unset or 0, native/setup.bash):
#     a test stack on another domain (e.g. 77) is left alone.
PROGS='bin/ros2 launch .*(launch_everything|web_gui)[^ ]*\.launch\.py|lib/controller_manager/(ros2_control_node|spawner)|lib/moveit_ros_move_group/move_group|lib/isaac_ros_cumotion/cumotion_[a-z_]+|/robot/(octomap_gater|lidar_bridge|boot_pre_pick)\.py|/app/(vacuum_bridge|box_model_node|web_gui)\.py|l2_driver/l2_driver_node|lib/realsense2_camera/realsense2_camera_node|lib/rviz2/rviz2|lib/robot_state_publisher/robot_state_publisher|lib/tf2_ros/static_transform_publisher'
NODES="^((/usr/bin/)?python3(\.[0-9]+)? )?[^ ]*(${PROGS})( |\$)"
on_robot_domain() {        # ROS_DOMAIN_ID of process $1 unset or 0
    local dom
    dom=$(tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n 's/^ROS_DOMAIN_ID=//p' | tail -n 1)
    [ "${dom:-0}" = "0" ]
}
list() {
    local pid
    for pid in $(pgrep -u "$(id -u)" -f "$NODES"); do
        [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ] && continue
        on_robot_domain "$pid" && echo "$pid"
    done
}
[ -z "$(list)" ] && { echo "nothing running"; exit 0; }
echo "asking to stop: $(list | xargs -r ps -o comm= -p | sort | uniq -c | tr -s ' ' | tr '\n' ',')"
list | xargs -r kill -INT 2>/dev/null
for i in $(seq 1 20); do sleep 0.5; [ -z "$(list)" ] && { echo "all stopped"; exit 0; }; done
echo "still running after 10 s, killing: $(list | xargs -r ps -o comm= -p | tr '\n' ' ')"
list | xargs -r kill -9 2>/dev/null
sleep 1; [ -z "$(list)" ] && echo "all stopped" || { echo "LEFT:"; list | xargs -r ps -o pid,stat,cmd -p; }
