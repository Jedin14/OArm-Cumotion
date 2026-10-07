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

#ifndef ISAAC_ROS_CUMOTION_MOVEIT__CUMOTION_MOVE_GROUP_CLIENT_HPP_
#define ISAAC_ROS_CUMOTION_MOVEIT__CUMOTION_MOVE_GROUP_CLIENT_HPP_

#include <future>
#include <memory>
#include <mutex>

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "moveit_msgs/action/move_group.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

class CumotionMoveGroupClient
{
  using GoalHandle = rclcpp_action::ClientGoalHandle<moveit_msgs::action::MoveGroup>;

public:
  CumotionMoveGroupClient(const rclcpp::Node::SharedPtr & node, const std::string & group_name);

  bool sendGoal();

  void updateGoal(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & req);

  void getGoal();

  // Thread-safe: the action result arrives on the executor thread while
  // solve() polls from the planning thread. Returns false until a result is
  // in; then copies it out and says whether it is a trajectory.
  bool takeResult(moveit_msgs::msg::MotionPlanDetailedResponse & response, bool & success);

  bool result_ready;
  bool success;
  moveit_msgs::msg::MotionPlanDetailedResponse plan_response;

private:
  void goalResponseCallback(const GoalHandle::SharedPtr & future);

  void feedbackCallback(
    GoalHandle::SharedPtr,
    const std::shared_ptr<const moveit_msgs::action::MoveGroup::Feedback> feedback);

  void resultCallback(const GoalHandle::WrappedResult & result);

  // Guards result_ready, success and plan_response. They used to be written
  // by resultCallback (executor thread) and getGoal (planning thread) at
  // once, and read by solve() as soon as result_ready flipped -- which
  // resultCallback did *before* filling the trajectory. solve() copying a
  // vector another thread was resizing is what segfaulted move_group
  // (exit -11) right after cuMotion returned a successful plan.
  std::mutex result_mutex_;

  bool get_goal_handle_;
  bool get_result_handle_;
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::CallbackGroup::SharedPtr client_cb_group_;
  rclcpp_action::Client<moveit_msgs::action::MoveGroup>::SharedPtr client_;
  rclcpp_action::Client<moveit_msgs::action::MoveGroup>::SendGoalOptions send_goal_options_;
  std::shared_future<GoalHandle::SharedPtr> goal_h_;
  std::shared_future<GoalHandle::WrappedResult> result_future_;
  moveit_msgs::msg::PlanningScene planning_scene_;
  planning_interface::MotionPlanRequest planning_request_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_MOVEIT__CUMOTION_MOVE_GROUP_CLIENT_HPP_
