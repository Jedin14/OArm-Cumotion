// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_cumotion/static_planning_scene.hpp"

#include <exception>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>

#include <rclcpp_components/register_node_macro.hpp>

#include "isaac_ros_cumotion/impl/utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

StaticPlanningSceneServer::StaticPlanningSceneServer(const rclcpp::NodeOptions & options)
: Node("static_planning_scene_server", options),
  moveit_collision_objects_scene_file_(
    declare_parameter<std::string>("moveit_collision_objects_scene_file", ""))
{
  srv_ = this->create_service<isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene>(
    "publish_static_planning_scene",
    std::bind(
      &StaticPlanningSceneServer::PublishPlanningSceneCallback, this, std::placeholders::_1,
      std::placeholders::_2));

  // Register a matched-event callback so that whenever a new subscriber connects,
  // the cached static scene is re-published automatically.
  rclcpp::PublisherOptions pub_options;
  pub_options.event_callbacks.matched_callback =
    std::bind(&StaticPlanningSceneServer::OnSubscriberMatched, this, std::placeholders::_1);

  planning_scene_pub_ =
    this->create_publisher<moveit_msgs::msg::PlanningScene>(
    "/planning_scene", rclcpp::QoS(10), pub_options);

  RCLCPP_INFO(this->get_logger(), "Static Planning Scene Server initialized");
}

void StaticPlanningSceneServer::PublishPlanningSceneCallback(
  const std::shared_ptr<isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene::Request>
  request,
  std::shared_ptr<isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene::Response>
  response)
{
  const std::string scene_file_path =
    !request->scene_file_path.empty() ? request->scene_file_path :
    moveit_collision_objects_scene_file_;

  if (scene_file_path.empty()) {
    response->success = false;
    response->message = "No static planning scene file path provided";
    response->status =
      isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene::Response::NO_SCENE_FILE;
    RCLCPP_INFO(this->get_logger(), "No static planning scene file path provided");
    return;
  }

  std::error_code ec;
  const bool exists = std::filesystem::exists(scene_file_path, ec);
  if (ec || !exists) {
    response->success = false;
    response->message = "Scene file not found: " + scene_file_path;
    response->status =
      isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene::Response::
      SCENE_FILE_NOT_FOUND;
    return;
  }

  RCLCPP_INFO(
    this->get_logger(), "Loading collision objects from scene file: %s", scene_file_path.c_str());

  try {
    moveit_msgs::msg::PlanningScene planning_scene =
      nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file_path);

    {
      std::lock_guard<std::mutex> lock(scene_mutex_);
      cached_scene_ = planning_scene;
    }

    planning_scene_pub_->publish(planning_scene);

    RCLCPP_INFO(
      this->get_logger(), "Published %zu collision objects to /planning_scene",
      planning_scene.world.collision_objects.size());

    response->planning_scene = std::move(planning_scene);
    response->success = true;
    response->message = "Planning scene published successfully.";
    response->status =
      isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene::Response::SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to publish planning scene: %s", e.what());
    response->success = false;
    response->message = std::string("Failed to publish planning scene: ") + e.what();
    response->status =
      isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene::Response::PARSING_ERROR;
  }
}

void StaticPlanningSceneServer::OnSubscriberMatched(rclcpp::MatchedInfo & info)
{
  if (info.current_count_change <= 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(scene_mutex_);
  if (cached_scene_.world.collision_objects.empty()) {
    return;
  }

  planning_scene_pub_->publish(cached_scene_);
  RCLCPP_INFO(
    this->get_logger(),
    "New subscriber on /planning_scene — re-published %zu static collision objects",
    cached_scene_.world.collision_objects.size());
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::cumotion::StaticPlanningSceneServer)
