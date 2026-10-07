// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_cumotion_moveit/cumotion_move_group_client.hpp"

#include <chrono>
#include <future>
#include <memory>
#include <string>

#include "moveit_msgs/action/move_group.hpp"
#include "moveit_msgs/msg/planning_options.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

namespace
{

constexpr unsigned kGetGoalWaitIntervalInMs = 10;

}  // namespace

CumotionMoveGroupClient::CumotionMoveGroupClient(const rclcpp::Node::SharedPtr & node, const std::string & group_name)
: result_ready(false),
  success(false),
  get_goal_handle_(false),
  get_result_handle_(false),
  node_(node),
  client_cb_group_(node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive))
{
  // One server for every group, as the 3.2.5 plugin this replaces had it:
  // cumotion_planner.py serves "cumotion/move_group". The 4.x per-group name
  // ("cumotion_<group>/move_group") has no server here and would wait forever.
  (void)group_name;
  std::string action_name = "cumotion/move_group";
  client_ = rclcpp_action::create_client<moveit_msgs::action::MoveGroup>(
    node_,
    action_name,
    client_cb_group_);

  send_goal_options_ = rclcpp_action::Client<moveit_msgs::action::MoveGroup>::SendGoalOptions();

  send_goal_options_.goal_response_callback = std::bind(
    &CumotionMoveGroupClient::goalResponseCallback, this, std::placeholders::_1);
  send_goal_options_.feedback_callback = std::bind(
    &CumotionMoveGroupClient::feedbackCallback, this, std::placeholders::_1, std::placeholders::_2);
  send_goal_options_.result_callback = std::bind(
    &CumotionMoveGroupClient::resultCallback, this, std::placeholders::_1);
}

void CumotionMoveGroupClient::updateGoal(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & req)
{
  planning_request_ = req;
  planning_scene->getPlanningSceneMsg(planning_scene_);
}

bool CumotionMoveGroupClient::sendGoal()
{
  {
    std::lock_guard<std::mutex> lock(result_mutex_);
    result_ready = false;
    success = false;
    plan_response = moveit_msgs::msg::MotionPlanDetailedResponse();
  }

  moveit_msgs::msg::PlanningOptions plan_options;
  plan_options.planning_scene_diff = planning_scene_;

  if (!client_->wait_for_action_server()) {
    RCLCPP_ERROR(node_->get_logger(), "Action server not available after waiting");
    rclcpp::shutdown();
  }

  auto goal_msg = moveit_msgs::action::MoveGroup::Goal();

  goal_msg.planning_options = plan_options;
  goal_msg.request = planning_request_;
  RCLCPP_INFO(node_->get_logger(), "Sending goal");

  auto goal_handle_future = client_->async_send_goal(goal_msg, send_goal_options_);
  goal_h_ = goal_handle_future;
  get_result_handle_ = true;
  get_goal_handle_ = true;
  return true;
}

void CumotionMoveGroupClient::getGoal()
{
  // Only the goal handle is looked at here. The result is handled in one
  // place, resultCallback; this used to process it a second time,
  // concurrently, writing the same plan_response.
  if (get_goal_handle_) {
    if (goal_h_.wait_for(std::chrono::milliseconds(kGetGoalWaitIntervalInMs)) !=
      std::future_status::ready)
    {
      return;
    }
    get_goal_handle_ = false;
    if (!goal_h_.get()) {
      RCLCPP_ERROR(node_->get_logger(), "Goal was rejected by server");
      std::lock_guard<std::mutex> lock(result_mutex_);
      result_ready = true;
      success = false;
    }
  }
}

bool CumotionMoveGroupClient::takeResult(
  moveit_msgs::msg::MotionPlanDetailedResponse & response, bool & ok)
{
  std::lock_guard<std::mutex> lock(result_mutex_);
  if (!result_ready) {
    return false;
  }
  ok = success;
  response = plan_response;
  return true;
}

void CumotionMoveGroupClient::goalResponseCallback(const GoalHandle::SharedPtr & future)
{
  auto goal_handle = future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(node_->get_logger(), "Goal was rejected by server");
    std::lock_guard<std::mutex> lock(result_mutex_);
    result_ready = true;
    success = false;
  } else {
    RCLCPP_INFO(node_->get_logger(), "Goal accepted by server, waiting for result");
  }
}

void CumotionMoveGroupClient::feedbackCallback(
  GoalHandle::SharedPtr,
  const std::shared_ptr<const moveit_msgs::action::MoveGroup::Feedback> feedback)
{
  std::string status = feedback->state;
  RCLCPP_INFO(node_->get_logger(), "Checking status");
  RCLCPP_INFO(node_->get_logger(), status.c_str());
}

void CumotionMoveGroupClient::resultCallback(const GoalHandle::WrappedResult & result)
{
  RCLCPP_INFO(node_->get_logger(), "Received result");

  // Build the whole response first, then publish it under the lock with
  // result_ready set *last*, so solve() can never see a half-written one.
  moveit_msgs::msg::MotionPlanDetailedResponse response;
  bool ok = false;
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      response.error_code = result.result->error_code;
      if (response.error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        ok = true;
        response.trajectory_start = result.result->trajectory_start;
        response.group_name = planning_request_.group_name;
        response.trajectory = {result.result->planned_trajectory};
        response.processing_time = {result.result->planning_time};
      }
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(node_->get_logger(), "Goal was aborted");
      break;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_ERROR(node_->get_logger(), "Goal was canceled");
      break;
    default:
      RCLCPP_ERROR(node_->get_logger(), "Unknown result code");
      break;
  }

  std::lock_guard<std::mutex> lock(result_mutex_);
  plan_response = std::move(response);
  success = ok;
  result_ready = true;
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia
