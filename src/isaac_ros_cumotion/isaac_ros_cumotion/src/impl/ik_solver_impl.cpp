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

#include "isaac_ros_cumotion/impl/ik_solver_impl.hpp"

#include <Eigen/Core>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

IkSolverImpl::IkSolverImpl(
  std::unique_ptr<cumotion_lib::CollisionFreeIkSolverConfig> ik_config,
  const rclcpp::Logger & logger)
: ik_config_(std::move(ik_config)),
  logger_(logger)
{
  RCLCPP_DEBUG(logger_, "IkSolverImpl created");
  Initialize();
}

void IkSolverImpl::Initialize()
{
  RCLCPP_INFO(logger_, "Initializing IkSolverImpl");

  if (!ik_config_) {
    RCLCPP_FATAL(logger_, "CollisionFreeIkSolverConfig is null");
    throw std::runtime_error("CollisionFreeIkSolverConfig is null");
  }

  // Create IK solver.
  ik_solver_ = cumotion_lib::CreateCollisionFreeIkSolver(*ik_config_);
  if (!ik_solver_) {
    RCLCPP_FATAL(logger_, "Failed to create CollisionFreeIkSolver");
    throw std::runtime_error("Failed to create CollisionFreeIkSolver");
  }

  RCLCPP_INFO(logger_, "IkSolverImpl initialized successfully");
}

bool IkSolverImpl::Solve(
  const cumotion_lib::CollisionFreeIkSolver::TaskSpaceTarget & task_target,
  const std::vector<Eigen::VectorXd> & seed_cspace_positions,
  std::vector<Eigen::VectorXd> & solutions)
{
  // Solve IK.
  auto ik_results = ik_solver_->solve(task_target, seed_cspace_positions);
  if (!ik_results ||
    ik_results->status() != cumotion_lib::CollisionFreeIkSolver::Results::Status::SUCCESS)
  {
    RCLCPP_WARN(logger_, "Collision-free IK failed to find a solution");
    return false;
  }

  // Collect up to requested number of solutions.
  const auto cspace_solutions = ik_results->cSpacePositions();
  if (cspace_solutions.empty()) {
    return false;
  }

  solutions.clear();
  solutions.reserve(cspace_solutions.size());
  solutions.insert(solutions.end(), cspace_solutions.begin(), cspace_solutions.end());

  return !solutions.empty();
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia
