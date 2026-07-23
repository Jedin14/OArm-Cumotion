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

#include "isaac_ros_cumotion/impl/trajectory_optimizer_impl.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

// Constructor implementation.
TrajectoryOptimizerImpl::TrajectoryOptimizerImpl(
  std::unique_ptr<cumotion_lib::TrajectoryOptimizerConfig> trajopt_config,
  double interpolation_dt,
  const rclcpp::Logger & logger)
: trajopt_config_(std::move(trajopt_config)),
  interpolation_dt_(interpolation_dt),
  logger_(logger)
{
  if (interpolation_dt_ <= 0.0) {
    RCLCPP_FATAL(logger_, "interpolation_dt must be positive, got: %.6f", interpolation_dt_);
    throw std::invalid_argument("interpolation_dt must be positive");
  }
  RCLCPP_DEBUG(logger_, "TrajectoryOptimizerImpl created");
  Initialize();
}

// Initialize implementation.
void TrajectoryOptimizerImpl::Initialize()
{
  if (initialized_) {
    RCLCPP_WARN(logger_, "TrajectoryOptimizerImpl already initialized, skipping");
    return;
  }

  RCLCPP_INFO(logger_, "Initializing TrajectoryOptimizerImpl");
  if (!trajopt_config_) {
    RCLCPP_FATAL(logger_, "TrajectoryOptimizerConfig is null");
    throw std::runtime_error("TrajectoryOptimizerConfig is null");
  }

  // Create trajectory optimizer.
  trajectory_optimizer_ = std::shared_ptr<cumotion_lib::TrajectoryOptimizer>(
    cumotion_lib::CreateTrajectoryOptimizer(*trajopt_config_));

  if (!trajectory_optimizer_) {
    RCLCPP_FATAL(logger_, "Failed to create trajectory optimizer");
    throw std::runtime_error("Failed to create trajectory optimizer");
  }

  initialized_ = true;
  RCLCPP_INFO(logger_, "TrajectoryOptimizerImpl initialized successfully");
}

// Trajectory optimizer accessor.
std::shared_ptr<cumotion_lib::TrajectoryOptimizer> TrajectoryOptimizerImpl::GetTrajectoryOptimizer()
{
  return trajectory_optimizer_;
}

// Plan to task-space target implementation.
bool TrajectoryOptimizerImpl::PlanToTaskSpaceTarget(
  const Eigen::VectorXd & start_state,
  const cumotion_lib::TrajectoryOptimizer::TaskSpaceTarget & task_target,
  const double time_dilation,
  std::vector<Eigen::VectorXd> & trajectory,
  std::vector<Eigen::VectorXd> & velocities,
  std::vector<Eigen::VectorXd> & accelerations,
  double & dt)
{
  if (!initialized_) {
    RCLCPP_ERROR(logger_, "Cannot plan: TrajectoryOptimizerImpl not initialized");
    return false;
  }

  // Plan using trajectory optimizer.
  auto trajopt_result = trajectory_optimizer_->planToTaskSpaceTarget(start_state, task_target);

  if (!trajopt_result) {
    RCLCPP_ERROR(logger_, "Trajectory optimization to pose failed (trajopt: NO_RESULT)");
    return false;
  }

  if (trajopt_result->status() != cumotion_lib::TrajectoryOptimizer::Results::Status::SUCCESS) {
    const auto status_str = TrajectoryOptimizerStatusToString(trajopt_result->status());
    RCLCPP_ERROR(
      logger_, "Trajectory optimization to pose failed (trajopt: %s)",
      status_str.c_str());
    return false;
  }

  // Extract trajectory from result at the interpolation_dt_ sampling.
  auto traj = trajopt_result->trajectory();
  if (!traj) {
    RCLCPP_ERROR(logger_, "Failed to extract trajectory from optimization result");
    return false;
  }

  ExtractTrajectoryPoints(traj.get(), trajectory, velocities, accelerations);

  // Start from the `interpolation_dt_` and then apply `time_dilation` to the trajectory.
  dt = interpolation_dt_;
  ApplyTimeDilationToTrajectory(time_dilation, dt, velocities, accelerations);

  return !trajectory.empty();
}

// Plan to c-space target implementation.
bool TrajectoryOptimizerImpl::PlanToCSpaceTarget(
  const Eigen::VectorXd & start_state,
  const cumotion_lib::TrajectoryOptimizer::CSpaceTarget & cspace_target,
  const double time_dilation,
  std::vector<Eigen::VectorXd> & trajectory,
  std::vector<Eigen::VectorXd> & velocities,
  std::vector<Eigen::VectorXd> & accelerations,
  double & dt)
{
  if (!initialized_) {
    RCLCPP_ERROR(logger_, "Cannot plan: TrajectoryOptimizerImpl not initialized");
    return false;
  }

  // Plan using trajectory optimizer.
  auto trajopt_result = trajectory_optimizer_->planToCSpaceTarget(start_state, cspace_target);

  if (!trajopt_result) {
    RCLCPP_ERROR(logger_, "Trajectory optimization to joint config failed (trajopt: NO_RESULT)");
    return false;
  }

  if (trajopt_result->status() != cumotion_lib::TrajectoryOptimizer::Results::Status::SUCCESS) {
    const auto status_str = TrajectoryOptimizerStatusToString(trajopt_result->status());
    RCLCPP_ERROR(
      logger_, "Trajectory optimization to joint config failed (trajopt: %s)",
      status_str.c_str());
    return false;
  }

  // Extract trajectory from result at the `interpolation_dt_` sampling.
  auto traj = trajopt_result->trajectory();
  if (!traj) {
    RCLCPP_ERROR(logger_, "Failed to extract trajectory from optimization result");
    return false;
  }

  ExtractTrajectoryPoints(traj.get(), trajectory, velocities, accelerations);

  // Start from the `interpolation_dt_` and then apply `time_dilation` to the trajectory.
  dt = interpolation_dt_;
  ApplyTimeDilationToTrajectory(time_dilation, dt, velocities, accelerations);

  return !trajectory.empty();
}

bool TrajectoryOptimizerImpl::PlanToTaskSpaceGoalset(
  const Eigen::VectorXd & start_state,
  const cumotion_lib::TrajectoryOptimizer::TaskSpaceTargetGoalset & goalset,
  const double time_dilation,
  std::vector<Eigen::VectorXd> & trajectory,
  std::vector<Eigen::VectorXd> & velocities,
  std::vector<Eigen::VectorXd> & accelerations,
  double & dt,
  int & goal_index)
{
  if (!initialized_) {
    RCLCPP_ERROR(logger_, "Cannot plan: TrajectoryOptimizerImpl not initialized");
    return false;
  }

  auto result = trajectory_optimizer_->planToTaskSpaceGoalset(start_state, goalset);

  if (!result) {
    RCLCPP_ERROR(
      logger_, "Trajectory optimization to task-space goalset failed (trajopt: NO_RESULT)");
    return false;
  }

  if (result->status() != cumotion_lib::TrajectoryOptimizer::Results::Status::SUCCESS) {
    const auto status_str = TrajectoryOptimizerStatusToString(result->status());
    RCLCPP_ERROR(
      logger_, "Trajectory optimization to task-space goalset failed (trajopt: %s)",
      status_str.c_str());
    return false;
  }

  goal_index = result->targetIndex();

  auto traj = result->trajectory();
  if (!traj) {
    RCLCPP_ERROR(logger_, "Failed to extract trajectory from goalset optimization result");
    return false;
  }

  ExtractTrajectoryPoints(traj.get(), trajectory, velocities, accelerations);

  dt = interpolation_dt_;
  ApplyTimeDilationToTrajectory(time_dilation, dt, velocities, accelerations);

  return !trajectory.empty();
}

// Extract trajectory points implementation.
void TrajectoryOptimizerImpl::ExtractTrajectoryPoints(
  const cumotion_lib::Trajectory * traj,
  std::vector<Eigen::VectorXd> & trajectory,
  std::vector<Eigen::VectorXd> & velocities,
  std::vector<Eigen::VectorXd> & accelerations)
{
  if (!traj) {
    RCLCPP_WARN(logger_, "Cannot extract trajectory points: trajectory is null");
    return;
  }

  auto domain = traj->domain();
  double duration = domain.span();
  double dt = interpolation_dt_;

  int num_points = static_cast<int>(duration / dt) + 1;
  trajectory.resize(num_points);
  velocities.resize(num_points);
  accelerations.resize(num_points);

  for (int i = 0; i < num_points; ++i) {
    double t = domain.lower + (i * dt);
    t = std::min(t, domain.upper);

    trajectory[i].resize(traj->numCSpaceCoords());
    velocities[i].resize(traj->numCSpaceCoords());
    accelerations[i].resize(traj->numCSpaceCoords());

    traj->eval(t, &trajectory[i], &velocities[i], &accelerations[i]);
  }

  RCLCPP_DEBUG(
    logger_, "Extracted %d trajectory points (dt=%.4f)",
    num_points, dt);
}

void TrajectoryOptimizerImpl::ApplyTimeDilationToTrajectory(
  const double time_dilation_factor,
  double & dt,
  std::vector<Eigen::VectorXd> & velocities,
  std::vector<Eigen::VectorXd> & accelerations)
{
  if (time_dilation_factor <= 0.0 || time_dilation_factor > 1.0) {
    RCLCPP_WARN(logger_, "time_dilation_factor <= 0.0 or > 1.0; skipping time dilation");
    return;
  }

  const double old_dt = dt;
  const double new_dt = old_dt * (1.0 / time_dilation_factor);

  // Scale derivatives for the new dt; positions remain unchanged.
  const double vel_scale = old_dt / new_dt;
  const double acc_scale = vel_scale * vel_scale;

  for (auto & v : velocities) {
    v *= vel_scale;
  }
  for (auto & a : accelerations) {
    a *= acc_scale;
  }

  dt = new_dt;
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia
