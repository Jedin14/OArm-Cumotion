#!/usr/bin/env bash
set -euo pipefail

# OpenArm bimanual motion test script
# Usage:
#   source /path/to/install/setup.bash
#   bash test_motion.sh [--use-fd] [--delay 2]
# Notes:
# - Assumes the robot is already launched. Example:
#     ros2 launch openarm_bringup openarm.bimanual.launch.py can_fd:=false
# - --use-fd has no effect here (left for future extension)

DELAY=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --delay)
      DELAY=${2:-2}
      shift 2
      ;;
    --use-fd)
      shift 1
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

info() { echo "[test_motion] $*"; }

# Left arm to pose
info "Left arm moving to pose in ${DELAY}s"
ros2 action send_goal /left_joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
"{trajectory: {joint_names: [openarm_left_joint1, openarm_left_joint2, openarm_left_joint3, openarm_left_joint4, openarm_left_joint5, openarm_left_joint6, openarm_left_joint7], points: [{positions: [0.5, -0.5, 0.5, 1.0, 0.3, -0.2, 0.0], time_from_start: {sec: ${DELAY}}}]}}"

sleep 1

# Right arm to pose
info "Right arm moving to pose in ${DELAY}s"
ros2 action send_goal /right_joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
"{trajectory: {joint_names: [openarm_right_joint1, openarm_right_joint2, openarm_right_joint3, openarm_right_joint4, openarm_right_joint5, openarm_right_joint6, openarm_right_joint7], points: [{positions: [0.5, 0.5, -0.3, 1.0, 0.0, 0.2, -0.1], time_from_start: {sec: ${DELAY}}}]}}"

sleep 1

# Gripper open/close
info "close left gripper"
ros2 action send_goal /left_gripper_controller/gripper_cmd control_msgs/action/GripperCommand \
"{command: {position: 0.0, max_effort: 10.0}}"

sleep 1


info "open left gripper"
ros2 action send_goal /left_gripper_controller/gripper_cmd control_msgs/action/GripperCommand \
"{command: {position: 0.03, max_effort: 10.0}}"


sleep 1

info "Close right gripper"
ros2 action send_goal /right_gripper_controller/gripper_cmd control_msgs/action/GripperCommand \
"{command: {position: 0.0, max_effort: 10.0}}"

sleep 1

# Return arms to zero
info "Return both arms to zero in ${DELAY}s"
ros2 action send_goal /left_joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
"{trajectory: {joint_names: [openarm_left_joint1, openarm_left_joint2, openarm_left_joint3, openarm_left_joint4, openarm_left_joint5, openarm_left_joint6, openarm_left_joint7], points: [{positions: [0, 0, 0, 0, 0, 0, 0], time_from_start: {sec: ${DELAY}}}]}}"

ros2 action send_goal /right_joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
"{trajectory: {joint_names: [openarm_right_joint1, openarm_right_joint2, openarm_right_joint3, openarm_right_joint4, openarm_right_joint5, openarm_right_joint6, openarm_right_joint7], points: [{positions: [0, 0, 0, 0, 0, 0, 0], time_from_start: {sec: ${DELAY}}}]}}"

sleep 1

# Return grippers to zero
info "Return both grippers to zero"
ros2 action send_goal /left_gripper_controller/gripper_cmd control_msgs/action/GripperCommand \
"{command: {position: 0.02, max_effort: 20.0}}"

ros2 action send_goal /right_gripper_controller/gripper_cmd control_msgs/action/GripperCommand \
"{command: {position: 0.02, max_effort: 20.0}}"

info "Test sequence completed." 