// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_cumotion_moveit/cumotion_interface.hpp"

#include <chrono>
#include <memory>

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "moveit/robot_state/conversions.h"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

namespace
{

constexpr unsigned kSleepIntervalInMs = 5;
// cuMotion's own attempts take ~4.2 s to fail here, and a successful plan can
// take longer than that; 5 s timed good plans out and left cuMotion planning
// for a request nobody was waiting for any more.
constexpr unsigned kTimeoutIntervalInSeconds = 30;

}  // namespace

void CumotionInterface::solve(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & request,
  planning_interface::MotionPlanDetailedResponse & response)
{
  RCLCPP_INFO(node_->get_logger(), "Planning trajectory");

  if (!planner_busy) {
    action_client_->updateGoal(planning_scene, request);
    action_client_->sendGoal();
    planner_busy = true;
  }

  rclcpp::Time start_time = node_->now();
  moveit_msgs::msg::MotionPlanDetailedResponse plan;
  bool ok = false;
  bool ready = false;
  while (
    !(ready = action_client_->takeResult(plan, ok)) &&
    node_->now().seconds() - start_time.seconds() < kTimeoutIntervalInSeconds)
  {
    action_client_->getGoal();
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepIntervalInMs));
  }

  if (!ready) {
    RCLCPP_ERROR(node_->get_logger(), "Timed out!");
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT;
    planner_busy = false;
    return;
  }
  RCLCPP_INFO(node_->get_logger(), "Received trajectory result");

  if (!ok || plan.trajectory.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No trajectory");
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    planner_busy = false;
    return;
  }
  RCLCPP_INFO(node_->get_logger(), "Trajectory success!");

  // A local copy, taken under the lock: nothing here reads state another
  // thread can be writing.
  response.error_code_ = plan.error_code;
  response.description_ = plan.description;
  auto result_traj = std::make_shared<robot_trajectory::RobotTrajectory>(
    planning_scene->getRobotModel(), request.group_name);
  // Start from the scene's real state, never a bare RobotState: in MoveIt
  // Humble that constructor leaves joint values *uninitialised*, and
  // cuMotion's trajectory only sets the 14 arm joints, so the finger joints of
  // every trajectory state were whatever was in memory. When that was NaN or
  // huge, the path validation's self-collision check segfaulted inside FCL
  // (DynamicAABBTreeCollisionManager::registerObjects) -- the move_group
  // crash (exit -11) seen right after successful plans, 2026-09-30.
  moveit::core::RobotState robot_state = planning_scene->getCurrentState();
  moveit::core::robotStateMsgToRobotState(plan.trajectory_start, robot_state);
  robot_state.update();
  result_traj->setRobotTrajectoryMsg(robot_state, plan.trajectory[0]);
  response.trajectory_.clear();
  response.trajectory_.push_back(result_traj);
  response.processing_time_ = plan.processing_time;

  planner_busy = false;
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia
