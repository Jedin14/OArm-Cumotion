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

#include "test/test_trajectory_optimizer_impl.hpp"

#include <cumotion/cumotion.h>
#include <cumotion/kinematics.h>
#include <cumotion/pose3.h>
#include <cumotion/robot_description.h>
#include <cumotion/world.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_cumotion/impl/trajectory_optimizer_impl.hpp"
#include "isaac_ros_cumotion/impl/utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

class TrajectoryOptimizerImplTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    test_logger_ = rclcpp::get_logger("test_trajectory_optimizer");

    // Locate robot description.
    LocateTestRobotFiles();

    // Load robot description.
    robot_description_ = cumotion_lib::LoadRobotFromFile(test_xrdf_path_, test_urdf_path_);
    ASSERT_NE(robot_description_, nullptr) << "Failed to load test robot";

    // Get tool frame.
    auto tool_frames = robot_description_->toolFrameNames();
    ASSERT_FALSE(tool_frames.empty()) << "No tool frames in test robot";
    tool_frame_ = tool_frames[0];

    // Create world and world view.
    world_ = cumotion_lib::CreateWorld();
    ASSERT_NE(world_, nullptr) << "Failed to create world";
    world_view_ = world_->addWorldView();
  }

  void TearDown() override
  {
    world_.reset();
    robot_description_.reset();
    rclcpp::shutdown();
  }

  // Locate URDF and XRDF files for UR robot.
  void LocateTestRobotFiles()
  {
    // Use the ROS 2 ament index to locate the installed package share directory.
    std::string share_dir;
    share_dir =
      ament_index_cpp::get_package_share_directory("isaac_ros_cumotion_robot_description");

    // Assume files are present. Not adding checking as these are golden paths.
    const std::string urdf_path = share_dir + "/urdf/ur5e_robotiq_2f_85.urdf";
    const std::string xrdf_path = share_dir + "/xrdf/ur5e_robotiq_2f_85.xrdf";

    test_urdf_path_ = urdf_path;
    test_xrdf_path_ = xrdf_path;
  }

  struct TestConfig
  {
    std::unique_ptr<cumotion_lib::TrajectoryOptimizerConfig> trajopt_config;
    double interpolation_dt{0.0};
  };

  // Helper to create a default config using cuMotion's TrajectoryOptimizerConfig.
  TestConfig CreateDefaultConfig()
  {
    TestConfig cfg;
    cfg.interpolation_dt = 0.05;

    cfg.trajopt_config =
      cumotion_lib::CreateDefaultTrajectoryOptimizerConfig(
      *robot_description_,
      tool_frame_,
      world_view_);
    return cfg;
  }

  rclcpp::Logger test_logger_{rclcpp::get_logger("test_trajectory_optimizer")};
  std::string test_urdf_path_;
  std::string test_xrdf_path_;
  std::shared_ptr<cumotion_lib::RobotDescription> robot_description_;
  std::string tool_frame_;
  std::shared_ptr<cumotion_lib::World> world_;
  cumotion_lib::WorldViewHandle world_view_;
};

// Tests basic construction of TrajectoryOptimizerImpl.
TEST_F(TrajectoryOptimizerImplTest, Construction)
{
  auto cfg = CreateDefaultConfig();

  EXPECT_NO_THROW(
        {
          TrajectoryOptimizerImpl optimizer(
            std::move(cfg.trajopt_config),
            cfg.interpolation_dt,
            test_logger_);
        });
}

// Tests construction fails with null robot description.
TEST_F(TrajectoryOptimizerImplTest, ConstructionNullRobotDescription)
{
  auto cfg = CreateDefaultConfig();
  std::unique_ptr<cumotion_lib::TrajectoryOptimizerConfig> null_config;

  EXPECT_THROW(
        {
          TrajectoryOptimizerImpl optimizer(
            std::move(null_config),
            cfg.interpolation_dt,
            test_logger_);
        }, std::runtime_error);
}

// Tests initialization succeeds.
TEST_F(TrajectoryOptimizerImplTest, InitializeSuccess)
{
  auto cfg = CreateDefaultConfig();
  TrajectoryOptimizerImpl optimizer(
    std::move(cfg.trajopt_config),
    cfg.interpolation_dt,
    test_logger_);

  EXPECT_NE(optimizer.GetTrajectoryOptimizer(), nullptr);
}

// Tests planning to C-space target.
TEST_F(TrajectoryOptimizerImplTest, PlanToCSpaceTarget)
{
  auto cfg = CreateDefaultConfig();
  TrajectoryOptimizerImpl optimizer(
    std::move(cfg.trajopt_config),
    cfg.interpolation_dt,
    test_logger_);

  // Define start and goal configurations.
  Eigen::VectorXd start_state = robot_description_->defaultCSpaceConfiguration();
  Eigen::VectorXd goal_config = start_state;
  // Make goal configuration slightly different from start configuration.
  if (goal_config.size() > 0) {
    goal_config[0] += 0.2;
  }
  if (goal_config.size() > 1) {
    goal_config[1] -= 0.2;
  }

  // Plan trajectory.
  std::vector<Eigen::VectorXd> trajectory;
  std::vector<Eigen::VectorXd> velocities;
  std::vector<Eigen::VectorXd> accelerations;
  double dt = 0.0;

  cumotion_lib::TrajectoryOptimizer::CSpaceTarget cspace_target(goal_config);

  bool success = optimizer.PlanToCSpaceTarget(
    start_state, cspace_target, 1.0, trajectory, velocities, accelerations, dt);

  EXPECT_TRUE(success);
  EXPECT_GT(trajectory.size(), 0);
  EXPECT_EQ(trajectory.size(), velocities.size());
  EXPECT_EQ(trajectory.size(), accelerations.size());
  EXPECT_GT(dt, 0.0);

  // Verify trajectory starts near start_state and ends near goal_config.
  EXPECT_NEAR(trajectory.front()[0], start_state[0], 0.1);
  EXPECT_NEAR(trajectory.back()[0], goal_config[0], 0.1);
}

// Tests planning to task space target (pose).
TEST_F(TrajectoryOptimizerImplTest, PlanToTaskSpaceTarget)
{
  auto cfg = CreateDefaultConfig();
  TrajectoryOptimizerImpl optimizer(
    std::move(cfg.trajopt_config),
    cfg.interpolation_dt,
    test_logger_);

  // Use the robot's default configuration for start state.
  Eigen::VectorXd start_state = robot_description_->defaultCSpaceConfiguration();

  // Build a reachable task-space target.
  auto kinematics = robot_description_->kinematics();
  auto tool_handle = kinematics->frame(tool_frame_);

  Eigen::VectorXd goal_config = start_state;
  if (goal_config.size() > 0) {
    goal_config[0] += 0.2;
  }
  if (goal_config.size() > 1) {
    goal_config[1] -= 0.2;
  }

  const Eigen::Vector3d target_position = kinematics->position(goal_config, tool_handle);
  const cumotion_lib::Rotation3 target_orientation =
    kinematics->orientation(goal_config, tool_handle);
  cumotion_lib::TrajectoryOptimizer::TranslationConstraint translation_constraint =
    cumotion_lib::TrajectoryOptimizer::TranslationConstraint::Target(target_position);
  cumotion_lib::TrajectoryOptimizer::OrientationConstraint orientation_constraint =
    cumotion_lib::TrajectoryOptimizer::OrientationConstraint::TerminalTarget(target_orientation);
  cumotion_lib::TrajectoryOptimizer::TaskSpaceTarget task_target(
    translation_constraint, orientation_constraint);

  // Plan trajectory.
  std::vector<Eigen::VectorXd> trajectory;
  std::vector<Eigen::VectorXd> velocities;
  std::vector<Eigen::VectorXd> accelerations;
  double dt = 0.0;

  bool success = optimizer.PlanToTaskSpaceTarget(
    start_state, task_target, 1.0, trajectory, velocities, accelerations, dt);

  EXPECT_TRUE(success);
  EXPECT_GT(trajectory.size(), 0);
  EXPECT_EQ(trajectory.size(), velocities.size());
  EXPECT_EQ(trajectory.size(), accelerations.size());
  EXPECT_GT(dt, 0.0);
}

// Tests planning with time dilation = 0.1 (slower trajectory).
TEST_F(TrajectoryOptimizerImplTest, PlanWithTimeDilationSlower)
{
  auto cfg = CreateDefaultConfig();
  const double interpolation_dt = cfg.interpolation_dt;
  TrajectoryOptimizerImpl optimizer(
    std::move(cfg.trajopt_config),
    interpolation_dt,
    test_logger_);

  Eigen::VectorXd start_state = robot_description_->defaultCSpaceConfiguration();
  Eigen::VectorXd goal_config = start_state;
  if (goal_config.size() > 0) {
    goal_config[0] += 0.2;
  }

  std::vector<Eigen::VectorXd> trajectory;
  std::vector<Eigen::VectorXd> velocities;
  std::vector<Eigen::VectorXd> accelerations;
  double dt = 0.0;

  // Plan with time dilation = 0.1.
  double time_dilation = 0.1;
  cumotion_lib::TrajectoryOptimizer::CSpaceTarget cspace_target(goal_config);
  bool success = optimizer.PlanToCSpaceTarget(
    start_state, cspace_target, time_dilation, trajectory, velocities,
    accelerations, dt);

  EXPECT_TRUE(success);
  EXPECT_GT(trajectory.size(), 0);

  // dt is computed as interpolation_dt / time_dilation.
  EXPECT_NEAR(dt, interpolation_dt / time_dilation, 1e-6);
}

// Tests planning with time dilation = 0.9 (faster trajectory).
TEST_F(TrajectoryOptimizerImplTest, PlanWithTimeDilationFaster)
{
  auto cfg = CreateDefaultConfig();
  const double interpolation_dt = cfg.interpolation_dt;
  TrajectoryOptimizerImpl optimizer(
    std::move(cfg.trajopt_config),
    interpolation_dt,
    test_logger_);

  Eigen::VectorXd start_state = robot_description_->defaultCSpaceConfiguration();
  Eigen::VectorXd goal_config = start_state;
  if (goal_config.size() > 0) {
    goal_config[0] += 0.2;
  }

  std::vector<Eigen::VectorXd> trajectory;
  std::vector<Eigen::VectorXd> velocities;
  std::vector<Eigen::VectorXd> accelerations;
  double dt = 0.0;

  // Plan with time dilation = 0.9 (faster).
  double time_dilation = 0.9;
  cumotion_lib::TrajectoryOptimizer::CSpaceTarget cspace_target(goal_config);
  bool success = optimizer.PlanToCSpaceTarget(
    start_state, cspace_target, time_dilation, trajectory, velocities,
    accelerations, dt);

  EXPECT_TRUE(success);
  EXPECT_GT(trajectory.size(), 0);

  // dt is computed as interpolation_dt / time_dilation.
  EXPECT_NEAR(dt, interpolation_dt / time_dilation, 1e-6);
}

// Tests extract trajectory points with valid trajectory.
TEST_F(TrajectoryOptimizerImplTest, ExtractTrajectoryPoints)
{
  auto cfg = CreateDefaultConfig();
  TrajectoryOptimizerImpl optimizer(
    std::move(cfg.trajopt_config),
    cfg.interpolation_dt,
    test_logger_);

  // First, plan to get a valid trajectory.
  Eigen::VectorXd start_state = robot_description_->defaultCSpaceConfiguration();
  Eigen::VectorXd goal_config = start_state;
  if (goal_config.size() > 0) {
    goal_config[0] += 0.2;
  }

  cumotion_lib::TrajectoryOptimizer::CSpaceTarget cspace_target(goal_config);
  auto trajopt_result = optimizer.GetTrajectoryOptimizer()->planToCSpaceTarget(
    start_state, cspace_target);

  ASSERT_NE(trajopt_result, nullptr);
  ASSERT_EQ(
    trajopt_result->status(),
    cumotion_lib::TrajectoryOptimizer::Results::Status::SUCCESS);

  auto traj = trajopt_result->trajectory();
  ASSERT_NE(traj, nullptr);

  // Extract trajectory points.
  std::vector<Eigen::VectorXd> trajectory;
  std::vector<Eigen::VectorXd> velocities;
  std::vector<Eigen::VectorXd> accelerations;

  EXPECT_NO_THROW(
        {
          optimizer.ExtractTrajectoryPoints(traj.get(), trajectory, velocities, accelerations);
        });

  EXPECT_GT(trajectory.size(), 0);
  EXPECT_EQ(trajectory.size(), velocities.size());
  EXPECT_EQ(trajectory.size(), accelerations.size());
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
