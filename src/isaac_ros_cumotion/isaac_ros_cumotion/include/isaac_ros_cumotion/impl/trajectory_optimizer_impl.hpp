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

#ifndef ISAAC_ROS_CUMOTION__IMPL__TRAJECTORY_OPTIMIZER_IMPL_HPP_
#define ISAAC_ROS_CUMOTION__IMPL__TRAJECTORY_OPTIMIZER_IMPL_HPP_

#include <cumotion/pose3.h>
#include <cumotion/trajectory.h>
#include <cumotion/trajectory_optimizer.h>

#include <Eigen/Core>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include "isaac_ros_cumotion/impl/utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

/**
 * This class is responsible for exposing the trajectory optimization interface from the cuMotion
 * library. It handles trajectory optimizer creation and configuration, planning to c-space and
 * task-space targets (single targets and goalset targets), and extraction of trajectory points.
 */
class TrajectoryOptimizerImpl
{
public:
  /**
   * Constructs a trajectory optimizer implementation with a configured cuMotion trajectory
   * optimizer config, and logger.
   *
   * NOTE: The interpolation timestep used once cuMotion trajectory optimization is completed to
   * sample the trajectory.
   */
  TrajectoryOptimizerImpl(
    std::unique_ptr<cumotion_lib::TrajectoryOptimizerConfig> trajopt_config,
    double interpolation_dt,
    const rclcpp::Logger & logger);

  // Gets the trajectory optimizer object.
  std::shared_ptr<cumotion_lib::TrajectoryOptimizer> GetTrajectoryOptimizer();

  /**
   * Plans a trajectory from start_state to a task-space target.
   *
   * The returned dt is computed from the configured interpolation_dt_ and the time_dilation factor
   * (in that order).
   */
  bool PlanToTaskSpaceTarget(
    const Eigen::VectorXd & start_state,
    const cumotion_lib::TrajectoryOptimizer::TaskSpaceTarget & task_target,
    const double time_dilation,
    std::vector<Eigen::VectorXd> & trajectory,
    std::vector<Eigen::VectorXd> & velocities,
    std::vector<Eigen::VectorXd> & accelerations,
    double & dt);

  /**
   * Plans a trajectory from start_state to a CSpaceTarget.
   *
   * The returned dt is computed from the configured interpolation_dt_ and the time_dilation factor.
   */
  bool PlanToCSpaceTarget(
    const Eigen::VectorXd & start_state,
    const cumotion_lib::TrajectoryOptimizer::CSpaceTarget & cspace_target,
    const double time_dilation,
    std::vector<Eigen::VectorXd> & trajectory,
    std::vector<Eigen::VectorXd> & velocities,
    std::vector<Eigen::VectorXd> & accelerations,
    double & dt);

  /**
   * Plans a trajectory to a task-space goalset. The cuMotion optimizer internally selects the best
   * reachable goal from the set and returns a single optimized trajectory. On success, this method
   * extracts a discrete trajectory, velocities, accelerations and dt (after applying
   * `interpolation_dt_` and the provided `time_dilation` factor), and outputs the selected
   * goal index.
   */
  bool PlanToTaskSpaceGoalset(
    const Eigen::VectorXd & start_state,
    const cumotion_lib::TrajectoryOptimizer::TaskSpaceTargetGoalset & goalset,
    const double time_dilation,
    std::vector<Eigen::VectorXd> & trajectory,
    std::vector<Eigen::VectorXd> & velocities,
    std::vector<Eigen::VectorXd> & accelerations,
    double & dt,
    int & goal_index);

  /**
   * Extracts trajectory points from a cuMotion trajectory using the configured interpolation
   * timestep. Populates trajectory, velocities, and accelerations vectors with sampled points.
   */
  void ExtractTrajectoryPoints(
    const cumotion_lib::Trajectory * traj,
    std::vector<Eigen::VectorXd> & trajectory,
    std::vector<Eigen::VectorXd> & velocities,
    std::vector<Eigen::VectorXd> & accelerations);

private:
  /**
   * Initializes the trajectory optimizer by creating the cuMotion optimizer with the configured
   * parameters. Throws runtime_error if trajectory optimizer creation fails.
   */
  void Initialize();

  // cuMotion trajectory optimizer configuration.
  std::unique_ptr<cumotion_lib::TrajectoryOptimizerConfig> trajopt_config_;

  // Interpolation timestep used for trajectory resampling.
  double interpolation_dt_;

  // Logger for diagnostics.
  rclcpp::Logger logger_;

  // cuMotion trajectory optimizer.
  std::shared_ptr<cumotion_lib::TrajectoryOptimizer> trajectory_optimizer_;

  // Flag indicating if the optimizer has been initialized.
  bool initialized_{false};

  // Applies time dilation to an already-sampled trajectory.
  void ApplyTimeDilationToTrajectory(
    const double time_dilation_factor,
    double & dt,
    std::vector<Eigen::VectorXd> & velocities,
    std::vector<Eigen::VectorXd> & accelerations);
};

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION__IMPL__TRAJECTORY_OPTIMIZER_IMPL_HPP_
