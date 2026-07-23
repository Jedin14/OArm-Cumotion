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

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_cumotion_object_attachment/object_attachment.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

class ObjectAttachmentNodeTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    node_ = std::make_shared<ObjectAttachmentNode>();
  }

  void TearDown() override
  {
    node_.reset();
  }

  std::shared_ptr<ObjectAttachmentNode> node_;
};

// Tests that GenerateCuboidVertices returns exactly 8 vertices.
TEST_F(ObjectAttachmentNodeTest, GenerateCuboidVerticesReturnsEightVertices)
{
  auto vertices = node_->GetCuboidVertices(2.0, 4.0, 6.0);
  ASSERT_EQ(vertices.size(), 8u);
}

// Tests that every vertex lies within the half-extent bounds.
TEST_F(ObjectAttachmentNodeTest, GenerateCuboidVerticesHasCorrectBounds)
{
  const double size_x = 2.0;
  const double size_y = 4.0;
  const double size_z = 6.0;
  const double hx = size_x / 2.0;  // 1.0
  const double hy = size_y / 2.0;  // 2.0
  const double hz = size_z / 2.0;  // 3.0

  auto vertices = node_->GetCuboidVertices(size_x, size_y, size_z);

  for (size_t i = 0; i < vertices.size(); ++i) {
    EXPECT_GE(vertices[i].x(), -hx) << "vertex " << i << " x below -hx";
    EXPECT_LE(vertices[i].x(), hx) << "vertex " << i << " x above  hx";
    EXPECT_GE(vertices[i].y(), -hy) << "vertex " << i << " y below -hy";
    EXPECT_LE(vertices[i].y(), hy) << "vertex " << i << " y above  hy";
    EXPECT_GE(vertices[i].z(), -hz) << "vertex " << i << " z below -hz";
    EXPECT_LE(vertices[i].z(), hz) << "vertex " << i << " z above  hz";
  }
}

// Tests that GenerateCuboidTriangles returns exactly 12 triangles.
TEST_F(ObjectAttachmentNodeTest, GenerateCuboidTrianglesReturnsTwelveTriangles)
{
  auto triangles = node_->GetCuboidTriangles();
  ASSERT_EQ(triangles.size(), 12u);
}

// Tests that all triangle indices reference valid vertices [0..7].
TEST_F(ObjectAttachmentNodeTest, GenerateCuboidTrianglesValidIndices)
{
  auto triangles = node_->GetCuboidTriangles();

  for (size_t i = 0; i < triangles.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_GE(triangles[i][axis], 0) << "triangle " << i << " index " << axis << " < 0";
      EXPECT_LE(triangles[i][axis], 7) << "triangle " << i << " index " << axis << " > 7";
    }
  }
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
