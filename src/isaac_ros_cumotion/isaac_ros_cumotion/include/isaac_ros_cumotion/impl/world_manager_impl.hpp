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

#ifndef ISAAC_ROS_CUMOTION__IMPL__WORLD_MANAGER_IMPL_HPP_
#define ISAAC_ROS_CUMOTION__IMPL__WORLD_MANAGER_IMPL_HPP_

#include <cumotion/cumotion.h>
#include <cumotion/obstacle.h>
#include <cumotion/pose3.h>
#include <cumotion/world.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <assimp/Importer.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <nvblox_msgs/srv/esdf_and_gradients.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "isaac_ros_cumotion/impl/utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

/**
 * Structure to hold objects that should be cleared from the ESDF. This includes both axis-aligned
 * bounding boxes and spheres that represent regions to be cleared from the distance field.
 */
struct EsdfClearingObjects
{
  // Minimum corners of AABBs to clear from ESDF.
  std::vector<geometry_msgs::msg::Point> aabbs_min;

  // Sizes of AABBs to clear from ESDF.
  std::vector<geometry_msgs::msg::Vector3> aabbs_size;

  // Centers of spheres to clear from ESDF.
  std::vector<geometry_msgs::msg::Point> spheres_center;

  // Radii of spheres to clear from ESDF.
  std::vector<double> spheres_radius;

  /**
   * Check if any clearing objects are present. Returns true if there are any AABBs or spheres
   * defined for clearing, false otherwise.
   */
  bool HasObjects() const
  {
    return !aabbs_min.empty() || !spheres_center.empty();
  }
};

/**
 * The WorldManagerImpl class is responsible for creating, storing and updating the obstacles used
 * in the cuMotion world. It manages collision objects, ESDF grid integration, and visualization.
 */
class WorldManagerImpl
{
public:
  /**
   * Configuration parameters for different components of the WorldManagerImpl class. These parameters
   * control grid dimensions, ESDF integration, visualization, and world setup.
   */
  struct Config
  {
    /**
     * Grid geometry fields below are populated from the first ESDF service call.
     * Zero defaults indicate the grid has not yet been initialized.
     */

    // Grid shape: number of voxels in [x, y, z]. Physical size = grid_shape * voxel_size_m.
    Eigen::Vector3i grid_shape{0, 0, 0};

    // Grid origin: minimum corner position in meters.
    Eigen::Vector3d grid_origin_m{0.0, 0.0, 0.0};

    // Size of a voxel in meters.
    double voxel_size_m = 0.0;

    // Flag to enable reading ESDF world from nvblox.
    bool read_esdf_world = false;

    // Flag to update ESDF on each planning request.
    bool update_esdf_on_request = true;

    // Name of the ESDF service provided by nvblox.
    std::string esdf_service_name = "";

    // Flag to publish world as voxel visualization markers.
    bool publish_world_as_voxels = false;

    // Size of voxels for visualization purposes.
    double publish_voxel_size = 0.05;

    // Maximum number of voxels to publish for visualization.
    int max_publish_voxels = 50000;

    // Flag to add a ground plane to the world.
    bool add_ground_plane = false;

    // Ground plane configuration parameters (read from ROS params).
    double ground_plane_size_x;
    double ground_plane_size_y;
    double ground_plane_thickness;
    double ground_plane_z_offset;

    /**
     * Robot base frame name for coordinate transformations.
     * NOTE: We currently don't have a world frame in @cumotion_planner_node.cpp, but it might be
     * useful to have a world_frame there and then use that here.
     */
    std::string robot_base_frame = "";
  };

  // Construct a WorldManagerImpl with the given configuration and logger.
  WorldManagerImpl(const Config & config, const rclcpp::Logger & logger);

  // Update the world with MoveIt collision_objects from the planning scene.
  bool UpdateWorldObjects(
    const std::vector<moveit_msgs::msg::CollisionObject> & collision_objects);

  // Process ESDF response and update SDF obstacle. Returns true on success.
  bool ProcessEsdfResponse(const nvblox_msgs::srv::EsdfAndGradients::Response & response);

  /**
   * Get the cuMotion world view for use in trajectory optimization and collision checking. The
   * world view provides a snapshot of enabled obstacles for distance queries.
   */
  const cumotion_lib::WorldViewHandle & GetWorldView() const {return world_view_;}

  /**
   * Get the underlying cuMotion world for direct obstacle management operations such as adding,
   * removing, enabling, or disabling obstacles.
   */
  std::shared_ptr<cumotion_lib::World> GetWorld() const {return world_;}

  /**
   * Remove all dynamic obstacles from the world, preserving the ground plane and ESDF grid if
   * they were configured. If ground plane was previously removed, it will be re-added.
   */
  void ClearWorld();

  // Updates the robot base frame used for ESDF requests and visualization headers.
  void SetRobotBaseFrame(const std::string & robot_base_frame);

  // Check whether ESDF grid updates from nvblox are enabled in the configuration.
  bool IsEsdfEnabled() const {return config_.read_esdf_world;}

  // Get the cached grid shape specifying the number of voxels along the X, Y, and Z axes.
  Eigen::Vector3i GetGridShape() const;

  // Get the cached grid origin position (minimum corner) in meters.
  Eigen::Vector3d GetGridOrigin() const;

  // Get the cached voxel size in meters.
  double GetVoxelSize() const;

  // Calculate occupied voxel positions in the workspace for visualization.
  std::vector<Eigen::Vector4f> CalculateOccupancyForVisualization();

  /**
   * Calculate AABBs to clear from ESDF for an object at the given world_pose_object (pose in
   * world frame). Dispatches to shape-specific helper methods based on object_shape, which must
   * be "SPHERE", "CUBOID", or "CUSTOM_MESH". The mesh_resource path is used only for CUSTOM_MESH
   * shapes.
   *
   * The object_scale defines the object's dimensions in its local frame, while
   * object_esdf_clearing_padding [x, y, z] adds extra per-side clearance around the
   * already-computed world-frame clearing region (full padding per side for AABBs, max
   * component added to radius for spheres). Because scale operates in the object frame
   * before the world transform, its effect on the world-frame clearing region depends on
   * the object's pose. Padding applies directly in world-frame axes regardless of the
   * object's orientation.
   *
   * For CUSTOM_MESH shapes, uses the member Assimp importer to load and process the mesh.
   * The loaded scene is freed via FreeScene() before returning for CUSTOM_MESH shape.
   *
   * Returns a unique pointer to EsdfClearingObjects containing AABBs/spheres to clear.
   */
  std::unique_ptr<EsdfClearingObjects> CalculateAabbsToClear(
    const geometry_msgs::msg::Pose & world_pose_object,
    const std::string & mesh_resource,
    const std::vector<double> & object_esdf_clearing_padding,
    const std::string & object_shape,
    const geometry_msgs::msg::Vector3 & object_scale);

private:
  // Initialize the cuMotion world and add ground plane if configured.
  bool Initialize();

  /**
   * Create a cuMotion SDF obstacle structure with the specified grid_shape, grid_origin
   * (minimal corner), and voxel_size. Returns a pair of the SDF obstacle and its world pose.
   * Actual distance values are set separately via setSdfGridValuesFromHost.
   */
  std::pair<std::unique_ptr<cumotion_lib::Obstacle>, cumotion_lib::Pose3>
  CreateSdfObstacleStructure(
    const Eigen::Vector3i & grid_shape,
    const Eigen::Vector3d & grid_origin,
    double voxel_size_m);

  /**
   * Add a 2x2 meter ground plane table positioned 5cm below the robot base (z=-0.05) to the world
   * as a cuboid obstacle with 10cm thickness.
   */
  void AddGroundPlane();

  /**
   * Convert MoveIt collision objects (primitives and meshes) to cuMotion obstacle representations
   * and add them to the world.
   *
   * Returns true if all object types are supported and added, false otherwise.
   */
  bool AddCollisionObjects(const std::vector<moveit_msgs::msg::CollisionObject> & objects);

  /**
   * Load a mesh file into the member Assimp importer.
   * Supports "file://" URI prefix; "package://" is not yet supported.
   *
   * Returns true if the mesh was loaded successfully.
   */
  bool LoadMesh(const std::string & mesh_resource);

  /**
   * Compute an axis-aligned bounding box from vertices in local (object) frame by transforming
   * them into world frame via world_transform and tracking component-wise min/max. The resulting
   * AABB is then grown by subtracting the full padding from the min corner and adding it to the
   * max corner along each axis, giving padding-per-side clearance. The computed AABB minimum
   * corner and dimensions are written to the output parameters aabb_min and aabb_size
   * respectively.
   *
   * Returns false if vertices is empty, true otherwise.
   */
  bool ComputeAABBFromVertices(
    const std::vector<Eigen::Vector3d> & vertices,
    const Eigen::Affine3d & world_transform,
    const Eigen::Vector3d & padding,
    geometry_msgs::msg::Point & aabb_min,
    geometry_msgs::msg::Vector3 & aabb_size);

  // Generate the 8 vertices of a cuboid centered at the origin with the given full dimensions
  // (size_x, size_y, size_z) along each axis.
  static std::vector<Eigen::Vector3d> GenerateCuboidVertices(
    double size_x, double size_y, double size_z);

  // Calculate clearing regions for a SPHERE-type object at world_pose_object (pose in world frame).
  // The object_scale defines the sphere dimensions in object frame; the max component of padding
  // is added to the clearing sphere radius. Populates clearing_objects. Returns true if successful.
  bool CalculateAabbsForSphere(
    const geometry_msgs::msg::Pose & world_pose_object,
    const Eigen::Vector3d & padding,
    const geometry_msgs::msg::Vector3 & object_scale,
    EsdfClearingObjects * clearing_objects);

  // Calculate clearing regions for a CUBOID-type object at world_pose_object (pose in world frame).
  // The object_scale defines the cuboid dimensions in object frame; the full padding is added per
  // side of the resulting world-frame AABB. Populates clearing_objects. Returns true if successful.
  bool CalculateAabbsForCuboid(
    const geometry_msgs::msg::Pose & world_pose_object,
    const Eigen::Vector3d & padding,
    const geometry_msgs::msg::Vector3 & object_scale,
    EsdfClearingObjects * clearing_objects);

  // Calculate clearing regions for a CUSTOM_MESH-type object at world_pose_object (pose in world
  // frame) using the given mesh_resource file path or URI. The object_scale factors size the mesh
  // in object frame; the full padding is added per side of the resulting world-frame AABB.
  // Populates clearing_objects. Returns true if successful.
  bool CalculateAabbsForMesh(
    const geometry_msgs::msg::Pose & world_pose_object,
    const std::string & mesh_resource,
    const Eigen::Vector3d & padding,
    const geometry_msgs::msg::Vector3 & object_scale,
    EsdfClearingObjects * clearing_objects);

  // Logger for diagnostics (only ROS type permitted here).
  rclcpp::Logger logger_;

  // Configuration parameters loaded from ROS parameters.
  Config config_;

  // cuMotion world containing all obstacles including primitives, meshes, ground plane, and ESDF.
  std::shared_ptr<cumotion_lib::World> world_;

  // View into the world for collision checking and distance queries during trajectory optimization.
  cumotion_lib::WorldViewHandle world_view_;

  // Handle for the ESDF voxel grid obstacle received from nvblox.
  cumotion_lib::World::ObstacleHandle esdf_obstacle_handle_;

  // Flag indicating whether ESDF obstacle has been added to the world.
  bool has_esdf_obstacle_ = false;

  // Mutex protecting concurrent access to world data from multiple threads.
  mutable std::mutex world_mutex_;

  // Flag indicating whether ground plane has been added to the world.
  bool ground_plane_added_ = false;

  // Map from obstacle names to their handles for tracking and removal of dynamic obstacles.
  std::unordered_map<std::string, cumotion_lib::World::ObstacleHandle> obstacle_handles_;

  // Assimp importer used by LoadMesh / CalculateAabbsForMesh (owns the scene memory).
  std::unique_ptr<Assimp::Importer> mesh_importer_;
};

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION__IMPL__WORLD_MANAGER_IMPL_HPP_
