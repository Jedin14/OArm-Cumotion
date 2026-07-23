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

#ifndef ISAAC_ROS_CUMOTION__STATIC_PLANNING_SCENE_HPP_
#define ISAAC_ROS_CUMOTION__STATIC_PLANNING_SCENE_HPP_

#include <memory>
#include <mutex>
#include <string>

#include <isaac_ros_cumotion_interfaces/srv/publish_static_planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

// Service node that publishes a static MoveIt planning scene from a collision-object file.
class StaticPlanningSceneServer final : public rclcpp::Node
{
public:
  /**
   * Constructs and initializes the static planning scene service node. `NodeOptions` is accepted so
   * this node can be configured when composed into a container.
   */
  explicit StaticPlanningSceneServer(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // Handles requests to publish the configured static planning scene.
  void PublishPlanningSceneCallback(
    const std::shared_ptr<isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene::Request>
    request,
    std::shared_ptr<isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene::Response>
    response);

  // Called by the DDS layer whenever a new subscriber matches our publisher.
  // Re-publishes the cached scene so the new subscriber receives it immediately.
  void OnSubscriberMatched(rclcpp::MatchedInfo & info);

  // File path to MoveIt collision objects used to build the static planning scene.
  std::string moveit_collision_objects_scene_file_;

  // Service server for static planning scene publication requests.
  rclcpp::Service<isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene>::SharedPtr srv_;
  // Publisher for MoveIt planning scene messages.
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_pub_;

  std::mutex scene_mutex_;
  moveit_msgs::msg::PlanningScene cached_scene_;
};

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION__STATIC_PLANNING_SCENE_HPP_
