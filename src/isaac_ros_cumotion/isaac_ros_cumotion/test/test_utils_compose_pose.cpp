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

#include "test/test_utils_compose_pose.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>

#include "isaac_ros_cumotion/impl/utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

namespace
{

void ExpectQuaternionNear(const Eigen::Quaterniond & a, const Eigen::Quaterniond & b, double tol)
{
  // Account for the q and -q equivalence.
  const double dot = std::abs(a.dot(b));
  EXPECT_NEAR(dot, 1.0, tol);
}

}  // namespace

TEST(UtilsComposePoseWithOffsetTest, AppliesOffsetInBaseFrame)
{
  // Base pose: rotate 90 degrees about Z, translate by (1, 0, 0).
  const Eigen::Quaterniond q_base(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));
  const cumotion_lib::Pose3 base_pose(cumotion_lib::Rotation3(q_base), Eigen::Vector3d(
      1.0, 0.0,
      0.0));

  // Offset pose: translate by (1, 0, 0) in the base frame.
  const Eigen::Quaterniond q_offset(1.0, 0.0, 0.0, 0.0);
  const cumotion_lib::Pose3 offset_pose(
    cumotion_lib::Rotation3(q_offset), Eigen::Vector3d(1.0, 0.0, 0.0));

  const cumotion_lib::Pose3 out = ComposePoseWithOffset(
    base_pose, offset_pose,
    /*offset_in_base_frame=*/ true);

  // Expected: t = t_base + R_base * t_offset = (1, 0, 0) + (0, 1, 0) = (1, 1, 0)
  EXPECT_NEAR(out.translation.x(), 1.0, 1e-9);
  EXPECT_NEAR(out.translation.y(), 1.0, 1e-9);
  EXPECT_NEAR(out.translation.z(), 0.0, 1e-9);

  const Eigen::Quaterniond q_out = out.rotation.quaternion();
  ExpectQuaternionNear(q_out, q_base, 1e-9);
}

TEST(UtilsComposePoseWithOffsetTest, AppliesOffsetInWorldFrame)
{
  // Base pose: rotate 90 degrees about Z, translate by (1, 0, 0).
  const Eigen::Quaterniond q_base(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));
  const cumotion_lib::Pose3 base_pose(cumotion_lib::Rotation3(q_base), Eigen::Vector3d(
      1.0, 0.0,
      0.0));

  // Offset pose: translate by (1, 0, 0) in the world frame.
  const Eigen::Quaterniond q_offset(1.0, 0.0, 0.0, 0.0);
  const cumotion_lib::Pose3 offset_pose(
    cumotion_lib::Rotation3(q_offset), Eigen::Vector3d(1.0, 0.0, 0.0));

  const cumotion_lib::Pose3 out = ComposePoseWithOffset(
    base_pose, offset_pose,
    /*offset_in_base_frame=*/ false);

  // Expected: t = t_offset + R_offset * t_base = (1, 0, 0) + (1, 0, 0) = (2, 0, 0)
  EXPECT_NEAR(out.translation.x(), 2.0, 1e-9);
  EXPECT_NEAR(out.translation.y(), 0.0, 1e-9);
  EXPECT_NEAR(out.translation.z(), 0.0, 1e-9);

  const Eigen::Quaterniond q_out = out.rotation.quaternion();
  ExpectQuaternionNear(q_out, q_base, 1e-9);
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia
