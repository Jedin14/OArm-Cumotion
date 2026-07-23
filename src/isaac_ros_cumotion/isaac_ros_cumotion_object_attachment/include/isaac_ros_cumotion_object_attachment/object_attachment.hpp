// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_CUMOTION_OBJECT_ATTACHMENT__OBJECT_ATTACHMENT_HPP_
#define ISAAC_ROS_CUMOTION_OBJECT_ATTACHMENT__OBJECT_ATTACHMENT_HPP_

#include <assimp/scene.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <string>
#include <vector>

#include <assimp/Importer.hpp>
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nvblox_msgs/srv/esdf_and_gradients.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker_array.hpp"

#include "isaac_ros_cumotion_interfaces/action/attach_object.hpp"
#include "isaac_ros_cumotion_interfaces/srv/get_robot_description.hpp"
#include "isaac_ros_cumotion_interfaces/srv/set_robot_description.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

// Node for attaching and detaching objects to robot collision geometry. Provides action servers
// for attaching/detaching grasped objects to the robot's collision model using cuMotion kinematics.
class ObjectAttachmentNode : public rclcpp::Node
{
public:
  using AttachObjectAction = isaac_ros_cumotion_interfaces::action::AttachObject;
  using GoalHandleAttachObject = rclcpp_action::ServerGoalHandle<AttachObjectAction>;
  using GetRobotDescription = isaac_ros_cumotion_interfaces::srv::GetRobotDescription;
  using SetRobotDescription = isaac_ros_cumotion_interfaces::srv::SetRobotDescription;

  // Constructs and initializes an ObjectAttachmentNode with the given options.
  explicit ObjectAttachmentNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  ~ObjectAttachmentNode() = default;

  // Public accessors for pure-geometry helpers exposed solely for unit testing.

  // Returns the vertices of a cuboid with the given size.
  std::vector<Eigen::Vector3d> GetCuboidVertices(
    double size_x, double size_y, double size_z);

  // Returns the triangles of a unit cuboid.
  std::vector<Eigen::Vector3i> GetCuboidTriangles();

private:
  // Handle goal requests for object attachment. Returns goal response (accept or reject).
  rclcpp_action::GoalResponse HandleGoal(
    const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const AttachObjectAction::Goal> goal);

  // Handle cancel requests. Returns cancel response.
  rclcpp_action::CancelResponse HandleCancel(
    const std::shared_ptr<GoalHandleAttachObject> goal_handle);

  // Handle accepted goals.
  void HandleAccepted(const std::shared_ptr<GoalHandleAttachObject> goal_handle);

  // Execute the attachment/detachment action.
  void Execute(const std::shared_ptr<GoalHandleAttachObject> goal_handle);

  // Get robot description (URDF and XRDF) from service. Returns true if successful.
  bool GetRobotDescriptionFromService();

  // Collision sphere structure.
  struct CollisionSphere
  {
    Eigen::Vector3d center;
    double radius;
  };

  // Structure to hold objects that should be cleared from the ESDF. This includes both
  // axis-aligned bounding boxes (AABBs) and spheres that represent regions to be cleared from
  // the distance field when an object is attached to the robot.
  struct EsdfClearingObjects
  {
    // Minimum corners of AABBs to clear from ESDF.
    std::vector<geometry_msgs::msg::Point> aabbs_min;

    // Sizes of AABBs to clear from ESDF.
    std::vector<geometry_msgs::msg::Vector3> aabbs_size;

    // Centers of spheres to clear from ESDF.
    std::vector<geometry_msgs::msg::Point> spheres_center;

    // Radii of spheres to clear from ESDF.
    std::vector<float> spheres_radius;

    // Check if any clearing objects are present. Returns true if there are any AABBs or spheres.
    bool HasObjects() const
    {
      return !aabbs_min.empty() || !spheres_center.empty();
    }
  };

  // Update robot description with collision spheres and send to service.
  // Returns true if successful.
  bool UpdateRobotDescriptionWithSpheres(
    const std::vector<CollisionSphere> & spheres);

  // Attach object collision spheres to robot model using the object_config marker configuration.
  // Returns true if successful.
  bool AttachObject(
    const visualization_msgs::msg::Marker & object_config);

  // Detach object from robot model. Returns true if successful.
  bool DetachObject();

  // Generate the 8 vertices of a cuboid centered at the origin with the given full dimensions
  // (size_x, size_y, size_z) along each axis. Returns a vector of 8 vertices.
  std::vector<Eigen::Vector3d> GenerateCuboidVertices(
    double size_x, double size_y, double size_z);

  // Compute an axis-aligned bounding box from vertices in local (object) frame by transforming
  // them into world frame via world_transform and tracking component-wise min/max. The resulting
  // AABB is then grown by subtracting the full object_esdf_clearing_padding_ from the min corner
  // and adding it to the max corner along each axis, giving padding-per-side clearance. The
  // computed AABB minimum corner and dimensions are written to the output parameters aabb_min
  // and aabb_size respectively.
  //
  // Returns false if vertices is empty, true otherwise.
  bool ComputeAABBFromVertices(
    const std::vector<Eigen::Vector3d> & vertices,
    const Eigen::Affine3d & world_transform,
    geometry_msgs::msg::Point & aabb_min,
    geometry_msgs::msg::Vector3 & aabb_size);

  // Generate cuboid triangle indices (12 triangles, 2 per face).
  std::vector<Eigen::Vector3i> GenerateCuboidTriangles();

  // Load mesh from the given mesh_resource URI using Assimp (cached). Returns true if successful.
  bool LoadMesh(const std::string & mesh_resource);

  // Extract vertices and triangles from cached mesh. Populates cached_mesh_vertices_ and
  // cached_mesh_triangles_ from the loaded Assimp scene. Returns early if geometry is already
  // cached. Returns true if successful.
  bool ExtractMeshGeometry();

  // Generate collision spheres for a sphere-type object described by object_config. Populates the
  // spheres output vector with the computed collision spheres. Returns true if successful.
  bool GenerateSpheresForSphere(
    const visualization_msgs::msg::Marker & object_config,
    std::vector<CollisionSphere> & spheres);

  // Generate collision spheres for a cuboid-type object described by object_config. Populates the
  // spheres output vector with the computed collision spheres. Returns true if successful.
  bool GenerateSpheresForCuboid(
    const visualization_msgs::msg::Marker & object_config,
    std::vector<CollisionSphere> & spheres);

  // Generate collision spheres for a mesh resource object described by object_config. Populates
  // the spheres output vector with the computed collision spheres. Returns true if successful.
  bool GenerateSpheresForMesh(
    const visualization_msgs::msg::Marker & object_config,
    std::vector<CollisionSphere> & spheres);

  // Clear the attached object's voxels from the world representation using the object_config
  // marker configuration. Returns true if clearing succeeded, false on failure.
  //
  // This method updates the 3D voxel grid (ESDF) maintained by nvblox to reflect that the object
  // is no longer an independent obstacle in the world - it's now part of the robot. The workflow
  // includes looking up transforms from world frame to grasp frame, calculating AABBs or spheres
  // that bound the object, and sending a clearing request to nvblox to remove those voxels.
  //
  // When clear_esdf_on_attach is enabled, any failure in this method will cause the object
  // attachment to fail.
  bool ClearObjectVoxelsFromWorld(const visualization_msgs::msg::Marker & object_config);

  // Calculate clearing regions (AABBs or spheres) for an attached object given its
  // world_pose_object (pose in world frame) and object_config (marker configuration). Dispatches
  // to shape-specific methods based on the object's marker type (SPHERE, CUBE, or MESH_RESOURCE).
  //
  // object_config.scale defines the object's dimensions in its local frame, while
  // object_esdf_clearing_padding_ [x, y, z] adds extra per-side clearance around the
  // already-computed world-frame clearing region (full padding per side for AABBs, max component
  // added to radius for spheres). Because scale operates in the object frame before the world
  // transform, its effect on the world-frame clearing region depends on the object's pose.
  // Padding applies directly in world-frame axes regardless of the object's orientation.
  //
  // Returns EsdfClearingObjects containing AABBs and/or spheres to clear from ESDF.
  EsdfClearingObjects CalculateClearingRegions(
    const geometry_msgs::msg::Pose & world_pose_object,
    const visualization_msgs::msg::Marker & object_config);

  // Calculate clearing regions for a SPHERE-type object given its world_pose_object (pose in world
  // frame) and object_config (marker configuration). object_config.scale defines the sphere
  // dimensions in object frame; the max component of object_esdf_clearing_padding_ is added to the
  // clearing sphere radius. Populates clearing_objects. Returns true if successful.
  bool CalculateClearingRegionsForSphere(
    const geometry_msgs::msg::Pose & world_pose_object,
    const visualization_msgs::msg::Marker & object_config,
    EsdfClearingObjects & clearing_objects);

  // Calculate clearing regions for a CUBOID-type object given its world_pose_object (pose in world
  // frame) and object_config (marker configuration). object_config.scale defines the cuboid
  // dimensions in object frame; the full object_esdf_clearing_padding_ is added per side of the
  // resulting world-frame AABB. Populates clearing_objects. Returns true if successful.
  bool CalculateClearingRegionsForCuboid(
    const geometry_msgs::msg::Pose & world_pose_object,
    const visualization_msgs::msg::Marker & object_config,
    EsdfClearingObjects & clearing_objects);

  // Calculate clearing regions for a MESH_RESOURCE-type object given its world_pose_object (pose
  // in world frame) and object_config (marker configuration). object_config.scale factors size the
  // mesh in object frame; the full object_esdf_clearing_padding_ is added per side of the resulting
  // world-frame AABB. Populates clearing_objects. Returns true if successful.
  bool CalculateClearingRegionsForMesh(
    const geometry_msgs::msg::Pose & world_pose_object,
    const visualization_msgs::msg::Marker & object_config,
    EsdfClearingObjects & clearing_objects);

  // Send ESDF clearing request to nvblox service with the given clearing_objects containing AABBs
  // and spheres to clear. Returns true if request sent successfully.
  bool SendEsdfClearingRequest(
    const EsdfClearingObjects & clearing_objects);

  // Validate node parameters. This method validates parameter values that were declared and
  // initialized in the constructor's member initializer list. It checks constraints and resets
  // invalid values to defaults.
  void ValidateParameters();
  // Action server
  rclcpp_action::Server<AttachObjectAction>::SharedPtr action_server_;

  // Cached mesh for current operation (importer owns scene memory)
  std::unique_ptr<Assimp::Importer> cached_mesh_importer_;
  std::vector<Eigen::Vector3d> cached_mesh_vertices_;
  std::vector<Eigen::Vector3i> cached_mesh_triangles_;

  // Service clients for robot description management
  rclcpp::Client<GetRobotDescription>::SharedPtr get_robot_description_client_;
  rclcpp::Client<SetRobotDescription>::SharedPtr set_robot_description_client_;

  // ESDF service client for voxel clearing
  rclcpp::Client<nvblox_msgs::srv::EsdfAndGradients>::SharedPtr esdf_client_;

  // Cached robot description strings
  std::string xrdf_;
  std::string urdf_;
  bool robot_description_cached_;

  // Publishers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr sphere_markers_pub_;

  // State
  bool object_attached_;

  // Parameters

  // Maximum distance (in meters) that collision spheres may extend beyond the object's mesh
  // surface.  A larger value produces fewer, coarser spheres (faster collision checking but less
  // precise), while a smaller value produces more, tighter spheres (slower but more accurate).
  // Must be non-negative.
  double max_overshoot_;
  // Whether to clear the ESDF (Euclidean Signed Distance Field) voxels occupied by the object
  // when it is attached to the robot.  This should be enabled when nvblox is running and
  // providing an ESDF world representation (i.e., read_esdf_world is true).  When nvblox is not
  // active, set this to false to avoid waiting for the unavailable nvblox ESDF service.
  // The bringup launch file sets this automatically based on the read_esdf_world argument.
  bool clear_esdf_on_attach_;
  // Extra clearance [x, y, z] added around the world-frame clearing region, not the object itself.
  // For AABBs, the full padding is added per side; for spheres, the max component is added to the
  // radius. The value represents clearance distance per side, not total growth.
  std::vector<double> object_esdf_clearing_padding_;
  // Reference frame for ESDF (must match nvblox's global_frame parameter)
  std::string esdf_reference_frame_;
  bool esdf_visualize_when_clearing_;  // Whether to visualize ESDF when clearing
  std::string get_robot_description_service_;  // Service name for getting robot description
  std::string set_robot_description_service_;  // Service name for setting robot description

  // TF2 for coordinate transforms
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_OBJECT_ATTACHMENT__OBJECT_ATTACHMENT_HPP_
