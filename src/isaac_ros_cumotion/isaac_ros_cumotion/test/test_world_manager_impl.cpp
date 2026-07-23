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

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_cumotion/impl/world_manager_impl.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

// Passed as mesh_resource when no mesh is involved (sphere/cuboid shapes).
constexpr const char * kNoMeshResource = "";

// Shape type strings matching visualization_msgs::msg::Marker constants.
constexpr const char * kShapeSphere = "SPHERE";
constexpr const char * kShapeCuboid = "CUBOID";
constexpr const char * kShapeCustomMesh = "CUSTOM_MESH";
constexpr const char * kShapeCylinder = "CYLINDER";

constexpr double kTolerance = 1e-6;

class WorldManagerImplEsdfClearingTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    test_node_ = std::make_shared<rclcpp::Node>("test_world_manager_esdf_clearing");

    WorldManagerImpl::Config config;
    config.read_esdf_world = false;
    config.add_ground_plane = false;

    world_manager_ = std::make_unique<WorldManagerImpl>(config, test_node_->get_logger());

    temp_dir_ = std::filesystem::temp_directory_path() / "world_manager_impl_test";
    std::filesystem::create_directories(temp_dir_);
    CreateTestMeshFile();
  }

  void TearDown() override
  {
    world_manager_.reset();
    test_node_.reset();
    std::filesystem::remove_all(temp_dir_);
  }

  static geometry_msgs::msg::Pose MakePose(
    double px, double py, double pz,
    double qw, double qx, double qy, double qz)
  {
    geometry_msgs::msg::Pose pose;
    pose.position.x = px;
    pose.position.y = py;
    pose.position.z = pz;
    pose.orientation.w = qw;
    pose.orientation.x = qx;
    pose.orientation.y = qy;
    pose.orientation.z = qz;
    return pose;
  }

  static geometry_msgs::msg::Vector3 MakeScale(double x, double y, double z)
  {
    geometry_msgs::msg::Vector3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
  }

  // Cube mesh with vertices from (0,0,0) to (0.2,0.2,0.2), matching the
  // Python integration test's _create_temp_mesh_file helper.
  void CreateTestMeshFile()
  {
    mesh_file_path_ = (temp_dir_ / "test_cube.obj").string();
    std::ofstream f(mesh_file_path_);
    f << "v 0.0 0.0 0.0\n"
      "v 0.2 0.0 0.0\n"
      "v 0.2 0.2 0.0\n"
      "v 0.0 0.2 0.0\n"
      "v 0.0 0.0 0.2\n"
      "v 0.2 0.0 0.2\n"
      "v 0.2 0.2 0.2\n"
      "v 0.0 0.2 0.2\n"
      "f 1 2 3\n"
      "f 1 3 4\n"
      "f 5 6 7\n"
      "f 5 7 8\n"
      "f 1 2 6\n"
      "f 1 6 5\n"
      "f 2 3 7\n"
      "f 2 7 6\n"
      "f 3 4 8\n"
      "f 3 8 7\n"
      "f 4 1 5\n"
      "f 4 5 8\n";
    f.close();
  }

  std::shared_ptr<rclcpp::Node> test_node_;
  std::unique_ptr<WorldManagerImpl> world_manager_;
  std::filesystem::path temp_dir_;
  std::string mesh_file_path_;
};

// ---- Sphere clearing tests ----

// Tests that a sphere's position and radius pass through correctly at identity pose.
TEST_F(WorldManagerImplEsdfClearingTest, SphereIdentityPoseUniformScale)
{
  // Sphere at (1,2,3), uniform scale 0.4, no padding.
  // radius = max(0.4, 0.4, 0.4) / 2 = 0.2
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {0.0, 0.0, 0.0},
    kShapeSphere,
    MakeScale(0.4, 0.4, 0.4));

  // Result should contain exactly one sphere clearing region, no AABBs.
  ASSERT_TRUE(result->HasObjects());
  ASSERT_EQ(result->spheres_center.size(), 1u);
  ASSERT_EQ(result->spheres_radius.size(), 1u);
  EXPECT_EQ(result->aabbs_min.size(), 0u);

  EXPECT_NEAR(result->spheres_center[0].x, 1.0, kTolerance);
  EXPECT_NEAR(result->spheres_center[0].y, 2.0, kTolerance);
  EXPECT_NEAR(result->spheres_center[0].z, 3.0, kTolerance);
  EXPECT_NEAR(result->spheres_radius[0], 0.2, kTolerance);
}

// Tests that non-uniform scale uses max(scale)/2 as the sphere radius.
TEST_F(WorldManagerImplEsdfClearingTest, SphereNonUniformScaleUsesMaxDimension)
{
  // Sphere at origin, scale (0.2, 0.4, 0.6).
  // radius = max(0.2, 0.4, 0.6) / 2 = 0.3
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {0.0, 0.0, 0.0},
    kShapeSphere,
    MakeScale(0.2, 0.4, 0.6));

  ASSERT_TRUE(result->HasObjects());
  EXPECT_NEAR(result->spheres_center[0].x, 0.0, kTolerance);
  EXPECT_NEAR(result->spheres_center[0].y, 0.0, kTolerance);
  EXPECT_NEAR(result->spheres_center[0].z, 0.0, kTolerance);
  EXPECT_NEAR(result->spheres_radius[0], 0.3, kTolerance);
}

// Tests that padding adds max(padding) to the sphere radius.
TEST_F(WorldManagerImplEsdfClearingTest, SpherePaddingAddsMaxCoeffToRadius)
{
  // Sphere at (1,2,3), scale 0.4, padding (0.1, 0.2, 0.3).
  // radius = 0.4/2 + max(0.1, 0.2, 0.3) = 0.2 + 0.3 = 0.5
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {0.1, 0.2, 0.3},
    kShapeSphere,
    MakeScale(0.4, 0.4, 0.4));

  ASSERT_TRUE(result->HasObjects());
  EXPECT_NEAR(result->spheres_radius[0], 0.5, kTolerance);
  EXPECT_NEAR(result->spheres_center[0].x, 1.0, kTolerance);
  EXPECT_NEAR(result->spheres_center[0].y, 2.0, kTolerance);
  EXPECT_NEAR(result->spheres_center[0].z, 3.0, kTolerance);
}

// ---- Cuboid clearing tests ----

// Tests that cuboid AABB matches scale dimensions at identity pose with no rotation.
TEST_F(WorldManagerImplEsdfClearingTest, CuboidIdentityPoseAlignedAabb)
{
  // Cuboid at origin, scale (2,4,6), no rotation, no padding.
  // Half-extents: (1,2,3). AABB min = (-1,-2,-3), size = (2,4,6).
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {0.0, 0.0, 0.0},
    kShapeCuboid,
    MakeScale(2.0, 4.0, 6.0));

  // Result should contain exactly one AABB clearing region, no spheres.
  ASSERT_TRUE(result->HasObjects());
  ASSERT_EQ(result->aabbs_min.size(), 1u);
  ASSERT_EQ(result->aabbs_size.size(), 1u);
  EXPECT_EQ(result->spheres_center.size(), 0u);

  EXPECT_NEAR(result->aabbs_min[0].x, -1.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].y, -2.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].z, -3.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].x, 2.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].y, 4.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].z, 6.0, kTolerance);
}

// Tests that 90-degree Z rotation swaps the X and Y AABB dimensions.
TEST_F(WorldManagerImplEsdfClearingTest, CuboidRotated90DegreesZSwapsXY)
{
  // 90-deg Z rotation swaps X and Y dimensions.
  // Vertex (1,2,3) -> (-2,1,3). AABB min = (-2,-1,-3), size = (4,2,6).
  const double angle = M_PI / 2.0;
  const double qw = std::cos(angle / 2.0);
  const double qz = std::sin(angle / 2.0);

  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, qw, 0.0, 0.0, qz),
    kNoMeshResource,
    {0.0, 0.0, 0.0},
    kShapeCuboid,
    MakeScale(2.0, 4.0, 6.0));

  ASSERT_TRUE(result->HasObjects());
  EXPECT_NEAR(result->aabbs_min[0].x, -2.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].y, -1.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].z, -3.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].x, 4.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].y, 2.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].z, 6.0, kTolerance);
}

// Tests that translation offsets the AABB min corner accordingly.
TEST_F(WorldManagerImplEsdfClearingTest, CuboidTranslatedOffsetsAabb)
{
  // Cuboid at (5,10,15), scale (2,4,6), no rotation.
  // AABB min = (5-1, 10-2, 15-3) = (4, 8, 12), size = (2, 4, 6).
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(5.0, 10.0, 15.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {0.0, 0.0, 0.0},
    kShapeCuboid,
    MakeScale(2.0, 4.0, 6.0));

  ASSERT_TRUE(result->HasObjects());
  EXPECT_NEAR(result->aabbs_min[0].x, 4.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].y, 8.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].z, 12.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].x, 2.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].y, 4.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].z, 6.0, kTolerance);
}

// Tests that padding expands the AABB symmetrically in each axis.
TEST_F(WorldManagerImplEsdfClearingTest, CuboidPaddingExpandsAabb)
{
  // Cuboid at origin, scale (2,4,6), padding (0.1, 0.2, 0.3).
  // Unpadded: min = (-1,-2,-3), max = (1,2,3).
  // Padded:   min -= padding, max += padding.
  // min = (-1.1, -2.2, -3.3), size = (2.2, 4.4, 6.6).
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {0.1, 0.2, 0.3},
    kShapeCuboid,
    MakeScale(2.0, 4.0, 6.0));

  ASSERT_TRUE(result->HasObjects());
  EXPECT_NEAR(result->aabbs_min[0].x, -1.1, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].y, -2.2, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].z, -3.3, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].x, 2.2, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].y, 4.4, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].z, 6.6, kTolerance);
}

// ---- Custom mesh clearing tests ----

// Tests that mesh vertices are scaled and produce the correct AABB at identity pose.
TEST_F(WorldManagerImplEsdfClearingTest, MeshScaledIdentityPose)
{
  // Mesh vertices (0,0,0)-(0.2,0.2,0.2), scale (5,5,5), identity pose.
  // Scaled range: (0,0,0)-(1,1,1). AABB min = (0,0,0), size = (1,1,1).
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "file://" + mesh_file_path_,
    {0.0, 0.0, 0.0},
    kShapeCustomMesh,
    MakeScale(5.0, 5.0, 5.0));

  // Result should contain exactly one AABB clearing region, no spheres.
  ASSERT_TRUE(result->HasObjects());
  ASSERT_EQ(result->aabbs_min.size(), 1u);
  EXPECT_EQ(result->spheres_center.size(), 0u);

  EXPECT_NEAR(result->aabbs_min[0].x, 0.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].y, 0.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].z, 0.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].x, 1.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].y, 1.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].z, 1.0, kTolerance);
}

// Tests that translation shifts the mesh AABB min corner.
TEST_F(WorldManagerImplEsdfClearingTest, MeshTranslatedOffsetsAabb)
{
  // Same mesh, translated to (10,20,30). Scaled vertices shift accordingly.
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0),
    "file://" + mesh_file_path_,
    {0.0, 0.0, 0.0},
    kShapeCustomMesh,
    MakeScale(5.0, 5.0, 5.0));

  ASSERT_TRUE(result->HasObjects());
  EXPECT_NEAR(result->aabbs_min[0].x, 10.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].y, 20.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].z, 30.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].x, 1.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].y, 1.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].z, 1.0, kTolerance);
}

// Tests that non-uniform scale produces different AABB sizes per axis.
TEST_F(WorldManagerImplEsdfClearingTest, MeshNonUniformScale)
{
  // Vertices (0-0.2) scaled by (5,10,15): (0,0,0) to (1,2,3).
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "file://" + mesh_file_path_,
    {0.0, 0.0, 0.0},
    kShapeCustomMesh,
    MakeScale(5.0, 10.0, 15.0));

  ASSERT_TRUE(result->HasObjects());
  EXPECT_NEAR(result->aabbs_min[0].x, 0.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].y, 0.0, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].z, 0.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].x, 1.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].y, 2.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].z, 3.0, kTolerance);
}

// Tests that padding expands the mesh AABB symmetrically.
TEST_F(WorldManagerImplEsdfClearingTest, MeshWithPaddingExpandsAabb)
{
  // Scaled mesh (0,0,0)-(1,1,1) with padding (0.1, 0.1, 0.1).
  // min = (-0.1,-0.1,-0.1), size = (1.2, 1.2, 1.2).
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "file://" + mesh_file_path_,
    {0.1, 0.1, 0.1},
    kShapeCustomMesh,
    MakeScale(5.0, 5.0, 5.0));

  ASSERT_TRUE(result->HasObjects());
  EXPECT_NEAR(result->aabbs_min[0].x, -0.1, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].y, -0.1, kTolerance);
  EXPECT_NEAR(result->aabbs_min[0].z, -0.1, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].x, 1.2, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].y, 1.2, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].z, 1.2, kTolerance);
}

// Tests that absolute path without file:// prefix also loads the mesh correctly.
TEST_F(WorldManagerImplEsdfClearingTest, MeshAbsolutePathWithoutFileUriPrefix)
{
  // Absolute path without file:// prefix exercises a different LoadMesh branch.
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    mesh_file_path_,
    {0.0, 0.0, 0.0},
    kShapeCustomMesh,
    MakeScale(5.0, 5.0, 5.0));

  ASSERT_TRUE(result->HasObjects());
  EXPECT_EQ(result->aabbs_min.size(), 1u);
  EXPECT_NEAR(result->aabbs_size[0].x, 1.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].y, 1.0, kTolerance);
  EXPECT_NEAR(result->aabbs_size[0].z, 1.0, kTolerance);
}

// ---- Error / edge-case tests ----

// Tests that padding with wrong number of elements returns no clearing objects.
TEST_F(WorldManagerImplEsdfClearingTest, InvalidPaddingSizeReturnsEmpty)
{
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {0.1, 0.2},  // Only 2 elements instead of 3.
    kShapeSphere,
    MakeScale(0.4, 0.4, 0.4));

  EXPECT_FALSE(result->HasObjects());
}

// Tests that empty padding vector returns no clearing objects.
TEST_F(WorldManagerImplEsdfClearingTest, EmptyPaddingReturnsEmpty)
{
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {},
    kShapeCuboid,
    MakeScale(2.0, 4.0, 6.0));

  EXPECT_FALSE(result->HasObjects());
}

// Tests that an unsupported shape type returns no clearing objects.
TEST_F(WorldManagerImplEsdfClearingTest, UnknownShapeReturnsEmpty)
{
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {0.0, 0.0, 0.0},
    kShapeCylinder,
    MakeScale(0.4, 0.4, 0.4));

  EXPECT_FALSE(result->HasObjects());
}

// Tests that CUSTOM_MESH with empty resource path returns no clearing objects.
TEST_F(WorldManagerImplEsdfClearingTest, MeshEmptyResourceReturnsEmpty)
{
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    kNoMeshResource,
    {0.0, 0.0, 0.0},
    kShapeCustomMesh,
    MakeScale(1.0, 1.0, 1.0));

  EXPECT_FALSE(result->HasObjects());
}

// Tests that a nonexistent mesh file path returns no clearing objects.
TEST_F(WorldManagerImplEsdfClearingTest, MeshNonexistentFileReturnsEmpty)
{
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "/nonexistent/path/mesh.obj",
    {0.0, 0.0, 0.0},
    kShapeCustomMesh,
    MakeScale(1.0, 1.0, 1.0));

  EXPECT_FALSE(result->HasObjects());
}

// Tests that package:// URIs are unsupported and return no clearing objects.
TEST_F(WorldManagerImplEsdfClearingTest, MeshPackageUriNotSupported)
{
  auto result = world_manager_->CalculateAabbsToClear(
    MakePose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "package://some_package/meshes/object.stl",
    {0.0, 0.0, 0.0},
    kShapeCustomMesh,
    MakeScale(1.0, 1.0, 1.0));

  EXPECT_FALSE(result->HasObjects());
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
