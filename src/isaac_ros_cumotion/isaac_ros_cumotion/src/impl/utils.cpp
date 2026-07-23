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

#include "isaac_ros_cumotion/impl/utils.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/duration.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <visualization_msgs/msg/marker.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

std::string TrajectoryOptimizerStatusToString(
  cumotion_lib::TrajectoryOptimizer::Results::Status status)
{
  using Status = cumotion_lib::TrajectoryOptimizer::Results::Status;
  switch (status) {
    case Status::SUCCESS:
      return "SUCCESS";
    case Status::INVALID_INITIAL_CSPACE_POSITION:
      return "INVALID_INITIAL_CSPACE_POSITION";
    case Status::INVALID_TARGET_SPECIFICATION:
      return "INVALID_TARGET_SPECIFICATION";
    case Status::INVERSE_KINEMATICS_FAILURE:
      return "INVERSE_KINEMATICS_FAILURE";
    case Status::GEOMETRIC_PLANNING_FAILURE:
      return "GEOMETRIC_PLANNING_FAILURE";
    case Status::TRAJECTORY_OPTIMIZATION_FAILURE:
      return "TRAJECTORY_OPTIMIZATION_FAILURE";
    default:
      return "UNKNOWN_STATUS";
  }
}

std::string ReadFileAsString(const std::string & file_path)
{
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_path);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

namespace
{

/**
 * Trim whitespace from both ends of a string, and return the trimmed string.
 *
 * For example, in the `.scene` file we may read lines with leading/trailing spaces or a trailing
 * newline; trimming allows us to:
 * - Treat whitespace-only lines as empty (and skip them).
 * - Parse tokens consistently (e.g., "* Box_0" vs "  * Box_0  ").
 */
std::string Trim(std::string_view s)
{
  // Trim whitespace from the beginning of the string.
  const auto is_space = [](unsigned char c) {return std::isspace(c) != 0;};
  std::size_t start = 0;
  while (start < s.size() && is_space(static_cast<unsigned char>(s[start]))) {
    ++start;
  }
  // Trim whitespace from the end of the string.
  std::size_t end = s.size();
  while (end > start && is_space(static_cast<unsigned char>(s[end - 1]))) {
    --end;
  }
  // Return the trimmed string.
  return std::string{s.substr(start, end - start)};
}


/**
 * Extract the collision object id from a line that begins with `*`.
 *
 * In the `.scene` format, each object starts with a line like:
 *   "* Table"
 * We strip off the leading "* " prefix and trim whitespace. If the id is missing, this returns an
 * empty string so the caller can throw `Invalid object ID`.
 */
std::string ExtractObjectId(std::string_view star_line)
{
  if (star_line.size() <= 2) {
    return std::string{};
  }
  return Trim(star_line.substr(2));
}

/**
 * Read the next non-empty trimmed line from a stream.
 *
 * The `.scene` files often contain blank lines. This helper:
 * - Reads lines until a non-empty (after Trim) line is found.
 * - Writes the trimmed line into `out_line`.
 * - Increments `non_empty_line_index` (1-indexed) for error reporting that matches the parser's
 *   notion of "line number" (counting only non-empty trimmed lines).
 *
 * Returns false on EOF (no more non-empty lines).
 */
bool ReadNextNonEmptyTrimmedLine(
  std::istream & in, std::string & out_line, std::size_t & non_empty_line_index)
{
  std::string line;
  while (std::getline(in, line)) {
    out_line = Trim(line);
    if (!out_line.empty()) {
      ++non_empty_line_index;
      return true;
    }
  }
  return false;
}

/**
 * Parse exactly `expected_count` doubles from a whitespace-separated string.
 *
 * Used for fixed-size fields in the `.scene` format:
 * - Position: "x y z" (3 values)
 * - Orientation: "qx qy qz qw" (4 values)
 *
 * Throws `std::invalid_argument(error_msg)` if parsing fails or if there are extra tokens.
 */
std::vector<double> ParseDoublesExactCountOrThrow(
  const std::string & s, std::size_t expected_count, const std::string & error_msg)
{
  std::istringstream iss(s);
  std::vector<double> out;
  out.reserve(expected_count);

  for (std::size_t i = 0; i < expected_count; ++i) {
    double v = 0.0;
    if (!(iss >> v)) {
      throw std::invalid_argument(error_msg);
    }
    out.push_back(v);
  }

  std::string extra;
  if (iss >> extra) {
    throw std::invalid_argument(error_msg);
  }
  return out;
}

/**
 * Parse one or more doubles from a whitespace-separated string.
 *
 * Used for variable-length fields in the `.scene` format:
 * - Primitive dimensions (validated later based on primitive type)
 *
 * Throws `std::invalid_argument(error_msg)` if parsing fails or yields no values.
 */
std::vector<double> ParseDoublesAnyCountOrThrow(
  const std::string & s, const std::string & error_msg)
{
  std::istringstream iss(s);
  std::vector<double> out;
  double v = 0.0;
  while (iss >> v) {
    out.push_back(v);
  }
  if (!iss.eof() || out.empty()) {
    throw std::invalid_argument(error_msg);
  }
  return out;
}

/**
 * Convert a shape type + dimensions to a `shape_msgs::msg::SolidPrimitive`.
 *
 * The `.scene` format specifies a primitive on two lines:
 *   box|sphere|cylinder
 *   <dimensions...>
 *
 * Note: MoveIt expects cylinder dimensions as [height, radius], so we swap the input order
 * (which is stored in this format as [radius, height]).
 */
shape_msgs::msg::SolidPrimitive CreatePrimitive(
  const std::string & shape_type_lower,
  const std::vector<double> & dims)
{
  shape_msgs::msg::SolidPrimitive primitive;

  if (shape_type_lower == "box") {
    if (dims.size() != 3U) {
      throw std::invalid_argument("Invalid dimensions for box");
    }
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {dims[0], dims[1], dims[2]};
  } else if (shape_type_lower == "sphere") {
    if (dims.size() != 1U) {
      throw std::invalid_argument("Invalid dimensions for sphere");
    }
    primitive.type = shape_msgs::msg::SolidPrimitive::SPHERE;
    primitive.dimensions = {dims[0]};
  } else if (shape_type_lower == "cylinder") {
    if (dims.size() != 2U) {
      throw std::invalid_argument("Invalid dimensions for cylinder");
    }
    primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    // SolidPrimitive expects [height, radius].
    primitive.dimensions = {dims[1], dims[0]};
  } else {
    throw std::invalid_argument("Unsupported shape type");
  }

  return primitive;
}

/**
 * Create a MoveIt `CollisionObject` from the 6 defining lines of an object record.
 *
 * Expected layout (after trimming, counting only non-empty lines):
 * 0: "* <id>"
 * 1: "x y z"
 * 2: "qx qy qz qw"
 * 3: (used / unused, e.g., "0" or "1")
 * 4: "box" | "sphere" | "cylinder"
 * 5: dimensions (count validated by shape type)
 *
 * The resulting object:
 * - uses `header.frame_id = "world"`
 * - contains exactly one primitive + one primitive pose
 */
moveit_msgs::msg::CollisionObject CreateCollisionObject(const std::vector<std::string> & lines)
{
  moveit_msgs::msg::CollisionObject obj;
  obj.header.frame_id = "world";

  obj.id = ExtractObjectId(lines[0]);
  if (obj.id.empty()) {
    throw std::invalid_argument("Invalid object ID");
  }

  const std::vector<double> pos =
    ParseDoublesExactCountOrThrow(lines[1], 3, "Invalid position values");
  const std::vector<double> ori =
    ParseDoublesExactCountOrThrow(lines[2], 4, "Invalid orientation values");

  geometry_msgs::msg::Pose pose;
  pose.position.x = pos[0];
  pose.position.y = pos[1];
  pose.position.z = pos[2];
  pose.orientation.x = ori[0];
  pose.orientation.y = ori[1];
  pose.orientation.z = ori[2];
  pose.orientation.w = ori[3];

  std::string shape_type = lines[4];
  std::transform(
    shape_type.begin(), shape_type.end(), shape_type.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  const std::vector<double> dims =
    ParseDoublesAnyCountOrThrow(lines[5], "Invalid dimensions format");

  shape_msgs::msg::SolidPrimitive primitive = CreatePrimitive(shape_type, dims);
  obj.primitives.push_back(std::move(primitive));
  obj.primitive_poses.push_back(std::move(pose));
  return obj;
}

}  // namespace

moveit_msgs::msg::PlanningScene ParseMoveItSceneFile(const std::string & scene_file_path)
{
  std::ifstream in(scene_file_path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to read scene file: " + scene_file_path);
  }

  std::size_t non_empty_line_index = 0;  // 1-indexed, counts only non-empty trimmed lines.
  std::string header;
  if (!ReadNextNonEmptyTrimmedLine(in, header, non_empty_line_index)) {
    throw std::invalid_argument("Empty scene file");
  }
  if (header.empty() || header.back() != '+') {
    throw std::invalid_argument("Missing header");
  }

  moveit_msgs::msg::PlanningScene planning_scene;
  planning_scene.is_diff = true;

  std::unordered_set<std::string> object_ids;

  std::string line;
  while (ReadNextNonEmptyTrimmedLine(in, line, non_empty_line_index)) {
    if (line.empty() || line[0] != '*') {
      continue;
    }

    const std::size_t object_start_non_empty_line = non_empty_line_index;
    try {
      // Expect 9 additional non-empty lines after '*': 5 for object definition + 4 metadata lines.
      std::vector<std::string> tail;
      tail.reserve(9);
      for (std::size_t k = 0; k < 9; ++k) {
        std::string t;
        if (!ReadNextNonEmptyTrimmedLine(in, t, non_empty_line_index)) {
          throw std::invalid_argument("Missing required parameters");
        }
        tail.push_back(std::move(t));
      }

      const std::string obj_id = ExtractObjectId(line);
      if (object_ids.find(obj_id) != object_ids.end()) {
        throw std::invalid_argument("Duplicate object ID: " + obj_id);
      }
      object_ids.insert(obj_id);

      const std::vector<std::string> obj_lines =
      {line, tail[0], tail[1], tail[2], tail[3], tail[4]};
      planning_scene.world.collision_objects.push_back(CreateCollisionObject(obj_lines));
    } catch (const std::invalid_argument & e) {
      throw std::invalid_argument(
              "Error parsing object starting at line " +
              std::to_string(object_start_non_empty_line) + ": " + e.what());
    }
  }

  return planning_scene;
}

// =============================================================================
// Message Conversion Functions
// =============================================================================

cumotion_lib::Pose3 ToCuMotionPose(const geometry_msgs::msg::Pose & ros_pose)
{
  Eigen::Vector3d translation(ros_pose.position.x, ros_pose.position.y, ros_pose.position.z);
  Eigen::Quaterniond quat(
    ros_pose.orientation.w, ros_pose.orientation.x,
    ros_pose.orientation.y, ros_pose.orientation.z);
  cumotion_lib::Rotation3 rotation(quat);
  return cumotion_lib::Pose3(rotation, translation);
}

geometry_msgs::msg::Pose ToROSPose(const cumotion_lib::Pose3 & cu_pose)
{
  geometry_msgs::msg::Pose ros_pose;

  ros_pose.position.x = cu_pose.translation.x();
  ros_pose.position.y = cu_pose.translation.y();
  ros_pose.position.z = cu_pose.translation.z();

  Eigen::Quaterniond quat = cu_pose.rotation.quaternion();
  ros_pose.orientation.w = quat.w();
  ros_pose.orientation.x = quat.x();
  ros_pose.orientation.y = quat.y();
  ros_pose.orientation.z = quat.z();

  return ros_pose;
}

Eigen::VectorXd ToEigenVector(const std::vector<double> & data)
{
  Eigen::VectorXd result(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    result(i) = data[i];
  }
  return result;
}

sensor_msgs::msg::JointState ToJointState(
  const Eigen::VectorXd & positions,
  const std::vector<std::string> & joint_names)
{
  sensor_msgs::msg::JointState joint_state;
  joint_state.name = joint_names;
  joint_state.position.resize(positions.size());

  for (int i = 0; i < positions.size(); ++i) {
    joint_state.position[i] = positions(i);
  }

  return joint_state;
}

moveit_msgs::msg::RobotTrajectory ToROSTrajectory(
  const std::vector<Eigen::VectorXd> & trajectory,
  const std::vector<Eigen::VectorXd> & velocities,
  const std::vector<Eigen::VectorXd> & accelerations,
  const std::vector<std::string> & joint_names,
  double dt,
  const std::string & frame_id)
{
  moveit_msgs::msg::RobotTrajectory robot_traj;
  trajectory_msgs::msg::JointTrajectory & joint_traj = robot_traj.joint_trajectory;

  joint_traj.joint_names = joint_names;
  joint_traj.header.frame_id = frame_id;

  for (size_t i = 0; i < trajectory.size(); ++i) {
    trajectory_msgs::msg::JointTrajectoryPoint point;

    // Set positions
    point.positions.resize(trajectory[i].size());
    for (int j = 0; j < trajectory[i].size(); ++j) {
      point.positions[j] = trajectory[i](j);
    }

    // Set velocities if available
    if (!velocities.empty() && i < velocities.size()) {
      point.velocities.resize(velocities[i].size());
      for (int j = 0; j < velocities[i].size(); ++j) {
        point.velocities[j] = velocities[i](j);
      }
    }

    // Set accelerations if available
    if (!accelerations.empty() && i < accelerations.size()) {
      point.accelerations.resize(accelerations[i].size());
      for (int j = 0; j < accelerations[i].size(); ++j) {
        point.accelerations[j] = accelerations[i](j);
      }
    }

    // Set time from start
    point.time_from_start = rclcpp::Duration::from_seconds(i * dt);

    joint_traj.points.push_back(point);
  }

  return robot_traj;
}

/**
 * There are two ways this function can be called - one with `offset_in_base_frame` set to true
 * and one with it set to false. We will walk through both cases and demonstrate how this function
 * works.
 *
 * Case A: `offset_in_base_frame` is true.
 *
 * Here the offset O is defined in the base pose's local axes.
 * Take a point x in the offset/local frame.
 *
 * 1) Apply the offset in base coordinates:
 *      x_B = O * x = R_O * x + t_O
 *
 * 2) Then place that result into the parent/world frame using the base pose:
 *      x_W = B * x_B
 *          = R_B * (R_O * x + t_O) + t_B
 *
 * Expand:
 *      x_W = (R_B * R_O) * x + (R_B * t_O + t_B)
 *
 * So the composed pose is P = B * O.
 *
 * Case B: `offset_in_base_frame` is false.
 *
 * Here the offset O is defined in the parent/world axes (i.e., apply it "outside" B).
 * Take a point x in the base frame.
 *
 * 1) Put it into the parent/world frame with the base pose:
 *      x_W = B * x = R_B * x + t_B
 *
 * 2) Apply a parent/world-frame offset to that world point:
 *      x_W' = O * x_W
 *           = R_O * (R_B * x + t_B) + t_O
 *
 * Expand:
 *      x_W' = (R_O * R_B) * x + (R_O * t_B + t_O)
 *
 * So the composed pose is P = O * B.
 */
cumotion_lib::Pose3 ComposePoseWithOffset(
  const cumotion_lib::Pose3 & base_pose,
  const cumotion_lib::Pose3 & offset,
  bool offset_in_base_frame)
{
  if (offset_in_base_frame) {
    return base_pose * offset;
  }
  return offset * base_pose;
}

// =============================================================================
// Grid and Workspace Helper Functions
// =============================================================================

Eigen::Vector3d GetGridCenter(
  const Eigen::Vector3d & min_corner,
  const Eigen::Vector3d & grid_size)
{
  return min_corner + grid_size * 0.5;
}

Eigen::Vector3d GetGridMinCorner(
  const Eigen::Vector3d & center,
  const Eigen::Vector3d & grid_size)
{
  return center - grid_size * 0.5;
}

Eigen::Vector3d GetGridSize(
  const Eigen::Vector3d & min_corner,
  const Eigen::Vector3d & max_corner,
  double voxel_size)
{
  if (voxel_size <= 0.0) {
    throw std::invalid_argument("voxel_size must be positive");
  }
  Eigen::Vector3d size = max_corner - min_corner;
  // Round to nearest voxel size
  for (int i = 0; i < 3; ++i) {
    size(i) = std::round(size(i) / voxel_size) * voxel_size;
  }
  return size;
}

bool IsGridValid(const Eigen::Vector3d & grid_size, double voxel_size)
{
  if (voxel_size <= 0.0) {
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    if (grid_size(i) < voxel_size) {
      return false;
    }
  }
  return true;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> LoadGridCornersFromWorkspaceFile(
  const std::string & workspace_file_path)
{
  YAML::Node config = YAML::LoadFile(workspace_file_path);

  if (!config["/**"] || !config["/**"]["ros__parameters"] ||
    !config["/**"]["ros__parameters"]["static_mapper"])
  {
    throw std::runtime_error("Required YAML structure not found in workspace file");
  }

  auto static_mapper = config["/**"]["ros__parameters"]["static_mapper"];

  // Validate all required fields exist
  if (!static_mapper["workspace_bounds_min_corner_x_m"] ||
    !static_mapper["workspace_bounds_min_corner_y_m"] ||
    !static_mapper["workspace_bounds_min_height_m"] ||
    !static_mapper["workspace_bounds_max_corner_x_m"] ||
    !static_mapper["workspace_bounds_max_corner_y_m"] ||
    !static_mapper["workspace_bounds_max_height_m"])
  {
    throw std::runtime_error("Missing required workspace bounds fields in YAML");
  }

  Eigen::Vector3d min_corner, max_corner;

  // Load min corner
  min_corner(0) = static_mapper["workspace_bounds_min_corner_x_m"].as<double>();
  min_corner(1) = static_mapper["workspace_bounds_min_corner_y_m"].as<double>();
  min_corner(2) = static_mapper["workspace_bounds_min_height_m"].as<double>();

  // Load max corner
  max_corner(0) = static_mapper["workspace_bounds_max_corner_x_m"].as<double>();
  max_corner(1) = static_mapper["workspace_bounds_max_corner_y_m"].as<double>();
  max_corner(2) = static_mapper["workspace_bounds_max_height_m"].as<double>();

  // Validate that max > min
  for (int i = 0; i < 3; ++i) {
    if (max_corner(i) <= min_corner(i)) {
      throw std::runtime_error(
              "Invalid workspace bounds: max corner must be greater than min corner");
    }
  }

  return std::make_pair(min_corner, max_corner);
}

// =============================================================================
// YAML Utility Functions
// =============================================================================

YAML::Node ReadYamlFile(const std::string & file_path)
{
  try {
    return YAML::LoadFile(file_path);
  } catch (const YAML::BadFile & e) {
    throw std::runtime_error("Failed to load YAML file: " + file_path + " - " + e.what());
  } catch (const YAML::ParserException & e) {
    throw std::runtime_error("Failed to parse YAML file: " + file_path + " - " + e.what());
  }
}

std::string YamlToString(const YAML::Node & yaml_node)
{
  YAML::Emitter emitter;
  emitter << yaml_node;
  return emitter.c_str();
}

YAML::Node StringToYaml(const std::string & yaml_string)
{
  try {
    return YAML::Load(yaml_string);
  } catch (const YAML::ParserException & e) {
    throw YAML::ParserException(e.mark, "Failed to parse YAML string: " + std::string(e.what()));
  }
}

std::string ReadYamlFileAsString(const std::string & file_path)
{
  YAML::Node yaml_node = ReadYamlFile(file_path);
  return YamlToString(yaml_node);
}

bool WriteYamlStringToFile(const std::string & yaml_string, const std::string & file_path)
{
  try {
    std::ofstream file(file_path);
    if (!file.is_open()) {
      RCLCPP_ERROR(
        rclcpp::get_logger("isaac_ros_cumotion"),
        "Failed to open file for writing: %s", file_path.c_str());
      return false;
    }
    file << yaml_string;
    file.close();
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      rclcpp::get_logger("isaac_ros_cumotion"),
      "Failed to write YAML string to file %s: %s", file_path.c_str(), e.what());
    return false;
  }
}

bool WriteYamlToFile(const YAML::Node & yaml_node, const std::string & file_path)
{
  std::string yaml_string = YamlToString(yaml_node);
  return WriteYamlStringToFile(yaml_string, file_path);
}

// =============================================================================
// Collision Object Conversion Functions
// =============================================================================

std::vector<std::pair<std::unique_ptr<cumotion_lib::Obstacle>, cumotion_lib::Pose3>>
ToCuMotionObstacles(const moveit_msgs::msg::CollisionObject & collision_object)
{
  std::vector<std::pair<std::unique_ptr<cumotion_lib::Obstacle>, cumotion_lib::Pose3>> obstacles;

  // Get the base pose of the collision object
  cumotion_lib::Pose3 base_pose = ToCuMotionPose(collision_object.pose);

  // Convert primitives
  for (size_t i = 0; i < collision_object.primitives.size(); ++i) {
    const auto & primitive = collision_object.primitives[i];
    cumotion_lib::Pose3 primitive_pose = base_pose;

    if (i < collision_object.primitive_poses.size()) {
      cumotion_lib::Pose3 local_pose = ToCuMotionPose(collision_object.primitive_poses[i]);
      primitive_pose = base_pose * local_pose;
    }

    switch (primitive.type) {
      case shape_msgs::msg::SolidPrimitive::BOX: {
          if (primitive.dimensions.size() >= 3) {
            auto box = cumotion_lib::CreateObstacle(cumotion_lib::Obstacle::Type::CUBOID);
            box->setAttribute(
              cumotion_lib::Obstacle::Attribute::SIDE_LENGTHS,
              Eigen::Vector3d(
                primitive.dimensions[0],
                primitive.dimensions[1],
                primitive.dimensions[2]));
            obstacles.push_back({std::move(box), primitive_pose});
          }
          break;
        }

      case shape_msgs::msg::SolidPrimitive::SPHERE: {
          // Spherical obstacles are not currently supported by cuMotion.
          RCLCPP_WARN(
            rclcpp::get_logger("isaac_ros_cumotion"),
            "Spherical obstacles are not supported by cuMotion");
          break;
        }

      case shape_msgs::msg::SolidPrimitive::CYLINDER: {
          // Cylindrical obstacles are not currently supported by cuMotion.
          RCLCPP_WARN(
            rclcpp::get_logger("isaac_ros_cumotion"),
            "Cylindrical obstacles are not supported by cuMotion");
          break;
        }

      default:
        // Unsupported primitive type
        break;
    }
  }

  return obstacles;
}

visualization_msgs::msg::Marker CreateVoxelMarker(
  const std::vector<Eigen::Vector4f> & voxels,
  const std::string & frame_id,
  double voxel_size)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id;
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
  marker.ns = "cumotion_world";
  marker.action = visualization_msgs::msg::Marker::ADD;

  marker.pose.orientation.w = 1.0;
  marker.lifetime = rclcpp::Duration::from_seconds(0.0);
  marker.frame_locked = false;

  marker.scale.x = voxel_size;
  marker.scale.y = voxel_size;
  marker.scale.z = voxel_size;

  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;

  // Add voxel points
  for (const auto & voxel : voxels) {
    if (voxel(3) > 0.0f) {  // Only add occupied voxels
      geometry_msgs::msg::Point pt;
      pt.x = voxel(0);
      pt.y = voxel(1);
      pt.z = voxel(2);
      marker.points.push_back(pt);
    }
  }

  return marker;
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia
