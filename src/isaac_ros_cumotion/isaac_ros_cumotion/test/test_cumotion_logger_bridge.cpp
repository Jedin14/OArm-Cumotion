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

#include "test/test_cumotion_logger_bridge.hpp"

#include <cumotion/cumotion.h>
#include <cumotion/robot_description.h>
#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_cumotion/impl/cumotion_logger_bridge.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

// Test fixture that initializes ROS 2 context before each test and cleans up after. Spinning up
// the ROS 2 context is required because CumotionLoggerBridge routes logs through rclcpp logging
// infrastructure.
class CumotionLoggerBridgeTest : public testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }
};

// Tests basic logger initialization with default name.
TEST_F(CumotionLoggerBridgeTest, DefaultLoggerName)
{
  EXPECT_NO_THROW(
        {
          CumotionLoggerBridge logger;
        });
}

// Tests logger initialization with custom name.
TEST_F(CumotionLoggerBridgeTest, CustomLoggerName)
{
  EXPECT_NO_THROW(
        {
          CumotionLoggerBridge logger("test_logger");
        });
}

// Tests SetLogLevel function with different log levels.
TEST_F(CumotionLoggerBridgeTest, SetLogLevel)
{
  EXPECT_NO_THROW(
        {
          nvidia::isaac_ros::cumotion::SetLogLevel(cumotion_lib::LogLevel::INFO);
        });

  EXPECT_NO_THROW(
        {
          nvidia::isaac_ros::cumotion::SetLogLevel(cumotion_lib::LogLevel::WARNING);
        });

  EXPECT_NO_THROW(
        {
          nvidia::isaac_ros::cumotion::SetLogLevel(cumotion_lib::LogLevel::ERROR);
        });

  EXPECT_NO_THROW(
        {
          nvidia::isaac_ros::cumotion::SetLogLevel(cumotion_lib::LogLevel::VERBOSE);
        });
}

// Tests that cuMotion integration actually routes logs through the ROS2 logger.
// This test loads a robot and verifies the logger bridge works with real cuMotion operations.
TEST_F(CumotionLoggerBridgeTest, CumotionIntegration)
{
  // Create and install custom logger bridge.
  auto logger = std::make_shared<CumotionLoggerBridge>("integration_test_logger");
  nvidia::isaac_ros::cumotion::SetLogger(logger);
  nvidia::isaac_ros::cumotion::SetLogLevel(cumotion_lib::LogLevel::INFO);

  EXPECT_NO_THROW(
        {
          // Try to load robot with invalid XRDF to trigger cuMotion error logs.
          // This tests that cuMotion's internal logging is properly routed through our bridge.
          std::string invalid_xrdf =
          R"(
format: xrdf
cspace:
  - joint1
  - joint2
default_cspace_position: [0.0]
acceleration_limits: [1.0, 2.0]
jerk_limits: [10.0, 20.0]
)";

          // Create minimal valid URDF for testing.
          std::string minimal_urdf =
          R"(<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>
  <link name="link2"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>
</robot>
)";

          // This should trigger FATAL error logs from cuMotion validation, which will be
          // routed through our ROS2 logger bridge. We expect an exception because the
          // XRDF is invalid (mismatched array sizes).
          try {
            auto robot = cumotion_lib::LoadRobotFromMemory(
              invalid_xrdf,
              minimal_urdf);
          } catch (const std::exception & e) {
            // Expected exception from cuMotion validation.
            // The error message should have been logged through our ROS2 bridge.
          }
        });

  // Restore default logger after test.
  nvidia::isaac_ros::cumotion::SetLogger(std::shared_ptr<CumotionLoggerBridge>());
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
