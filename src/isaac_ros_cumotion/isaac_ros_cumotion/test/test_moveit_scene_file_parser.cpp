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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <shape_msgs/msg/solid_primitive.hpp>

#include "isaac_ros_cumotion/impl/utils.hpp"

namespace
{

std::string CreateTempSceneFile(const std::string & content)
{
  const std::filesystem::path dir = std::filesystem::temp_directory_path();
  std::filesystem::path path;
  for (int i = 0; i < 100; ++i) {
    path = dir / ("isaac_ros_cumotion_scene_" + std::to_string(std::rand()) + ".scene");
    if (!std::filesystem::exists(path)) {
      break;
    }
  }

  std::ofstream out(path.string());
  if (!out.is_open()) {
    throw std::runtime_error("Failed to create temp scene file");
  }
  out << content;
  out.close();
  return path.string();
}

void AssertCollisionObject(
  const moveit_msgs::msg::CollisionObject & obj,
  const std::string & expected_id,
  const std::vector<double> & expected_pos,
  const std::vector<double> & expected_ori,
  const uint8_t expected_type,
  const std::vector<double> & expected_dims)
{
  ASSERT_EQ(obj.id, expected_id);
  ASSERT_EQ(obj.header.frame_id, "world");
  ASSERT_EQ(obj.primitives.size(), 1U);
  ASSERT_EQ(obj.primitive_poses.size(), 1U);

  const auto & pose = obj.primitive_poses[0];
  EXPECT_DOUBLE_EQ(pose.position.x, expected_pos[0]);
  EXPECT_DOUBLE_EQ(pose.position.y, expected_pos[1]);
  EXPECT_DOUBLE_EQ(pose.position.z, expected_pos[2]);
  EXPECT_DOUBLE_EQ(pose.orientation.x, expected_ori[0]);
  EXPECT_DOUBLE_EQ(pose.orientation.y, expected_ori[1]);
  EXPECT_DOUBLE_EQ(pose.orientation.z, expected_ori[2]);
  EXPECT_DOUBLE_EQ(pose.orientation.w, expected_ori[3]);

  const auto & primitive = obj.primitives[0];
  EXPECT_EQ(primitive.type, expected_type);
  ASSERT_EQ(primitive.dimensions.size(), expected_dims.size());
  for (std::size_t i = 0; i < expected_dims.size(); ++i) {
    EXPECT_DOUBLE_EQ(primitive.dimensions[i], expected_dims[i]);
  }
}

}  // namespace

class TestMoveItSceneFileParser : public ::testing::Test
{
protected:
  void SetUp() override
  {
    valid_header_ = "(noname)+\n";
    valid_box_ =
      "* Box_0\n"
      "-0.32 0.02 0\n"
      "0 0 0 1\n"
      "1\n"
      "box\n"
      "0.2 0.6 0.8\n"
      "0 0 0\n"
      "0 0 0 1\n"
      "0 0 0 0\n"
      "0\n";
    valid_sphere_ =
      "* Sphere_0\n"
      "0 0 0\n"
      "0 0 0 1\n"
      "1\n"
      "sphere\n"
      "0.1\n"
      "0 0 0\n"
      "0 0 0 1\n"
      "0 0 0 0\n"
      "0\n";
    valid_cylinder_ =
      "* Cylinder_0\n"
      "-0.8 0 0\n"
      "0 0 0 1\n"
      "1\n"
      "cylinder\n"
      "0.05 0.8\n"
      "0 0 0\n"
      "0 0 0 1\n"
      "0 0 0 0\n"
      "0\n";
  }

  std::string valid_header_;
  std::string valid_box_;
  std::string valid_sphere_;
  std::string valid_cylinder_;
};

TEST_F(TestMoveItSceneFileParser, ParseEmptyFile)
{
  const std::string scene_file = CreateTempSceneFile("");
  EXPECT_THROW(
    nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file),
    std::invalid_argument);
  std::filesystem::remove(scene_file);
}

TEST_F(TestMoveItSceneFileParser, ParseMissingHeader)
{
  const std::string scene_file = CreateTempSceneFile("invalid_header\n");
  EXPECT_THROW(
    nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file),
    std::invalid_argument);
  std::filesystem::remove(scene_file);
}

TEST_F(TestMoveItSceneFileParser, ParseInvalidObjectId)
{
  const std::string invalid_id =
    valid_header_ +
    "*\n"
    "-0.32 0.02 0\n"
    "0 0 0 1\n"
    "1\n"
    "box\n"
    "0.2 0.6 0.8\n"
    "0 0 0\n"
    "0 0 0 1\n"
    "0 0 0 0\n"
    "0\n";
  const std::string scene_file = CreateTempSceneFile(invalid_id);
  EXPECT_THROW(
    nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file),
    std::invalid_argument);
  std::filesystem::remove(scene_file);
}

TEST_F(TestMoveItSceneFileParser, ParseInvalidPosition)
{
  const std::string invalid_pos =
    valid_header_ +
    "* Box_0\n"
    "invalid\n"
    "0 0 0 1\n"
    "1\n"
    "box\n"
    "0.2 0.6 0.8\n"
    "0 0 0\n"
    "0 0 0 1\n"
    "0 0 0 0\n"
    "0\n";
  const std::string scene_file = CreateTempSceneFile(invalid_pos);
  EXPECT_THROW(
    nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file),
    std::invalid_argument);
  std::filesystem::remove(scene_file);
}

TEST_F(TestMoveItSceneFileParser, ParseInvalidOrientation)
{
  const std::string invalid_ori =
    valid_header_ +
    "* Box_0\n"
    "-0.32 0.02 0\n"
    "invalid\n"
    "1\n"
    "box\n"
    "0.2 0.6 0.8\n"
    "0 0 0\n"
    "0 0 0 1\n"
    "0 0 0 0\n"
    "0\n";
  const std::string scene_file = CreateTempSceneFile(invalid_ori);
  EXPECT_THROW(
    nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file),
    std::invalid_argument);
  std::filesystem::remove(scene_file);
}

TEST_F(TestMoveItSceneFileParser, ParseInvalidDimensions)
{
  const std::string invalid_dims =
    valid_header_ +
    "* Box_0\n"
    "-0.32 0.02 0\n"
    "0 0 0 1\n"
    "1\n"
    "box\n"
    "invalid\n"
    "0 0 0\n"
    "0 0 0 1\n"
    "0 0 0 0\n"
    "0\n";
  const std::string scene_file = CreateTempSceneFile(invalid_dims);
  EXPECT_THROW(
    nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file),
    std::invalid_argument);
  std::filesystem::remove(scene_file);
}

TEST_F(TestMoveItSceneFileParser, ParseUnsupportedShape)
{
  const std::string invalid_shape =
    valid_header_ +
    "* Box_0\n"
    "-0.32 0.02 0\n"
    "0 0 0 1\n"
    "1\n"
    "invalid\n"
    "0.2 0.6 0.8\n"
    "0 0 0\n"
    "0 0 0 1\n"
    "0 0 0 0\n"
    "0\n";
  const std::string scene_file = CreateTempSceneFile(invalid_shape);
  EXPECT_THROW(
    nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file),
    std::invalid_argument);
  std::filesystem::remove(scene_file);
}

TEST_F(TestMoveItSceneFileParser, ParseDuplicateObjectId)
{
  const std::string scene_file = CreateTempSceneFile(valid_header_ + valid_box_ + valid_box_);
  EXPECT_THROW(
    nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file),
    std::invalid_argument);
  std::filesystem::remove(scene_file);
}

TEST_F(TestMoveItSceneFileParser, ParseValidSceneFile)
{
  const std::string scene_file = CreateTempSceneFile(valid_header_ + valid_box_ + valid_sphere_);
  const auto scene = nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file);
  ASSERT_EQ(scene.world.collision_objects.size(), 2U);

  AssertCollisionObject(
    scene.world.collision_objects[0],
    "Box_0",
    {-0.32, 0.02, 0.0},
    {0.0, 0.0, 0.0, 1.0},
    shape_msgs::msg::SolidPrimitive::BOX,
    {0.2, 0.6, 0.8});

  AssertCollisionObject(
    scene.world.collision_objects[1],
    "Sphere_0",
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0},
    shape_msgs::msg::SolidPrimitive::SPHERE,
    {0.1});

  std::filesystem::remove(scene_file);
}

TEST_F(TestMoveItSceneFileParser, ParseComplexScene)
{
  const std::string scene_file =
    CreateTempSceneFile(valid_header_ + valid_box_ + valid_sphere_ + valid_cylinder_);
  const auto scene = nvidia::isaac_ros::cumotion::ParseMoveItSceneFile(scene_file);
  ASSERT_EQ(scene.world.collision_objects.size(), 3U);

  AssertCollisionObject(
    scene.world.collision_objects[0],
    "Box_0",
    {-0.32, 0.02, 0.0},
    {0.0, 0.0, 0.0, 1.0},
    shape_msgs::msg::SolidPrimitive::BOX,
    {0.2, 0.6, 0.8});

  AssertCollisionObject(
    scene.world.collision_objects[1],
    "Sphere_0",
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0},
    shape_msgs::msg::SolidPrimitive::SPHERE,
    {0.1});

  // Cylinder dimensions are [height, radius] (matching the historic parser behavior).
  AssertCollisionObject(
    scene.world.collision_objects[2],
    "Cylinder_0",
    {-0.8, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0},
    shape_msgs::msg::SolidPrimitive::CYLINDER,
    {0.8, 0.05});

  std::filesystem::remove(scene_file);
}
