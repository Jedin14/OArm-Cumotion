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

#ifndef ISAAC_ROS_CUMOTION__IMPL__IK_SOLVER_IMPL_HPP_
#define ISAAC_ROS_CUMOTION__IMPL__IK_SOLVER_IMPL_HPP_

#include <cumotion/collision_free_ik_solver.h>

#include <Eigen/Core>
#include <memory>
#include <string>
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
 * This class is responsible for exposing the collision-free inverse kinematics interface from the
 * cuMotion library. It handles solver creation and configuration and provides a method to solve
 * for c-space targets given a cuMotion CollisionFreeIkSolverConfig.
 */
class IkSolverImpl
{
public:
  /**
   * Constructs an IK solver implementation with a configured cuMotion collision-free IK solver
   * config and logger.
   */
  IkSolverImpl(
    std::unique_ptr<cumotion_lib::CollisionFreeIkSolverConfig> ik_config,
    const rclcpp::Logger & logger);

  /**
   * Solves collision-free inverse kinematics for the given task-space target using
   * seed_cspace_positions as optional warm-start seeds.
   *
   * NOTE: currently no interface to the goalset IK solver is provided. If needed, the current
   * interface can be extended to support goalset IK solving.
   */
  bool Solve(
    const cumotion_lib::CollisionFreeIkSolver::TaskSpaceTarget & task_target,
    const std::vector<Eigen::VectorXd> & seed_cspace_positions,
    std::vector<Eigen::VectorXd> & solutions);

private:
  /**
   * Initializes the IK solver by creating the cuMotion collision-free IK solver with the
   * configured parameters. Throws runtime_error if solver creation fails.
   */
  void Initialize();

  // cuMotion collision-free IK solver configuration.
  std::unique_ptr<cumotion_lib::CollisionFreeIkSolverConfig> ik_config_;

  // Logger for diagnostics.
  rclcpp::Logger logger_;

  // cuMotion collision-free IK solver.
  std::unique_ptr<cumotion_lib::CollisionFreeIkSolver> ik_solver_;
};

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION__IMPL__IK_SOLVER_IMPL_HPP_
