// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_CUMOTION__IMPL__UTILS_HPP_
#define ISAAC_ROS_CUMOTION__IMPL__UTILS_HPP_

#include <cumotion/obstacle.h>
#include <cumotion/pose3.h>
#include <cumotion/trajectory_optimizer.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

// Alias for cuMotion library namespace to avoid collision with the nvidia::isaac_ros::cumotion
// namespace
namespace cumotion_lib = ::cumotion;

// Convert cuMotion TrajectoryOptimizer result status to a string.
std::string TrajectoryOptimizerStatusToString(
  cumotion_lib::TrajectoryOptimizer::Results::Status status);

// Convert ROS `geometry_msgs::Pose` to cuMotion `Pose3`.
cumotion_lib::Pose3 ToCuMotionPose(const geometry_msgs::msg::Pose & ros_pose);

// Convert cuMotion `Pose3` to ROS `geometry_msgs::Pose`.
geometry_msgs::msg::Pose ToROSPose(const cumotion_lib::Pose3 & cu_pose);

// Convert std::vector<double> to `Eigen::VectorXd`.
Eigen::VectorXd ToEigenVector(const std::vector<double> & data);

// Create `sensor_msgs::JointState` from Eigen position vector and joint names.
sensor_msgs::msg::JointState ToJointState(
  const Eigen::VectorXd & positions,
  const std::vector<std::string> & joint_names);

/**
 * Convert cuMotion trajectory to ROS `moveit_msgs::RobotTrajectory` with positions, velocities,
 * accelerations sampled at intervals of `dt`. Optionally set `frame_id` for the trajectory header.
 */
moveit_msgs::msg::RobotTrajectory ToROSTrajectory(
  const std::vector<Eigen::VectorXd> & trajectory,
  const std::vector<Eigen::VectorXd> & velocities,
  const std::vector<Eigen::VectorXd> & accelerations,
  const std::vector<std::string> & joint_names,
  double dt,
  const std::string & frame_id = "");

// Create RViz CUBE_LIST marker message from voxels array for visualizing the collision world.
visualization_msgs::msg::Marker CreateVoxelMarker(
  const std::vector<Eigen::Vector4f> & voxels,
  const std::string & frame_id,
  double voxel_size);

/**
 * Read file as a raw string.
 * Throws std::runtime_error if file cannot be opened.
 */
std::string ReadFileAsString(const std::string & file_path);

// Parse a MoveIt `.scene` file and return a `moveit_msgs::msg::PlanningScene`.
moveit_msgs::msg::PlanningScene ParseMoveItSceneFile(const std::string & scene_file_path);

// Calculate grid center from minimal corner and grid size.
Eigen::Vector3d GetGridCenter(
  const Eigen::Vector3d & min_corner,
  const Eigen::Vector3d & grid_size);

// Calculate grid minimal corner from center and grid size.
Eigen::Vector3d GetGridMinCorner(
  const Eigen::Vector3d & center,
  const Eigen::Vector3d & grid_size);

// Calculate grid size from minimal and maximal corners, rounded to nearest voxel size.
Eigen::Vector3d GetGridSize(
  const Eigen::Vector3d & min_corner,
  const Eigen::Vector3d & max_corner,
  double voxel_size);

// Check if grid dimensions are valid (at least one voxel in each dimension).
bool IsGridValid(const Eigen::Vector3d & grid_size, double voxel_size);

/**
 * Load grid minimal and maximal corners from workspace YAML file in nvblox format.
 * Returns a pair of [min_corner, max_corner] vectors.
 */
std::pair<Eigen::Vector3d, Eigen::Vector3d> LoadGridCornersFromWorkspaceFile(
  const std::string & workspace_file_path);

// =============================================================================
// YAML Utility Functions
// =============================================================================

/**
 * Read YAML file and return as YAML::Node.
 * Throws std::runtime_error if file cannot be loaded.
 */
YAML::Node ReadYamlFile(const std::string & file_path);

/**
 * Convert YAML::Node to string representation.
 * Returns a formatted YAML string with proper indentation.
 */
std::string YamlToString(const YAML::Node & yaml_node);

/**
 * Convert string to YAML::Node.
 * Throws YAML::ParserException if string is not valid YAML.
 */
YAML::Node StringToYaml(const std::string & yaml_string);

/**
 * Read YAML file and convert to string.
 * Convenience function that combines ReadYamlFile and YamlToString.
 * Throws std::runtime_error if file cannot be loaded.
 */
std::string ReadYamlFileAsString(const std::string & file_path);

/**
 * Write YAML string to file.
 * Returns true on success, false on failure.
 */
bool WriteYamlStringToFile(const std::string & yaml_string, const std::string & file_path);

/**
 * Write YAML::Node to file.
 * Returns true on success, false on failure.
 */
bool WriteYamlToFile(const YAML::Node & yaml_node, const std::string & file_path);

/**
 * Convert MoveIt `CollisionObject` to cuMotion obstacles. Supports BOX (cuboid) primitives.
 * SPHERE and CYLINDER are not currently supported by cuMotion and will generate warnings.
 * Returns a vector of obstacle-pose pairs.
 */
std::vector<std::pair<std::unique_ptr<cumotion_lib::Obstacle>, cumotion_lib::Pose3>>
ToCuMotionObstacles(const moveit_msgs::msg::CollisionObject & collision_object);

/**
 * Compose a base pose with an offset pose.
 *
 * If offset_in_base_frame is true, the offset is applied in the base pose frame (base * offset).
 * Otherwise, the offset is applied in the world frame (offset * base).
 */
cumotion_lib::Pose3 ComposePoseWithOffset(
  const cumotion_lib::Pose3 & base_pose,
  const cumotion_lib::Pose3 & offset,
  bool offset_in_base_frame);

// Buffer for storing joint state data received from ROS joint state topic.
struct JointStateBuffer
{
  std::vector<std::string> joint_names;
  std::vector<double> position;
  std::vector<double> velocity;

  bool IsValid() const
  {
    return !joint_names.empty() && !position.empty();
  }

  void Clear()
  {
    joint_names.clear();
    position.clear();
    velocity.clear();
  }
};

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION__IMPL__UTILS_HPP_
