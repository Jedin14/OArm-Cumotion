// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_cumotion_moveit/cumotion_planning_context.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

CumotionPlanningContext::CumotionPlanningContext(
  const std::string & name, const std::string & group, const rclcpp::Node::SharedPtr & node)
: planning_interface::PlanningContext(name, group),
  cumotion_interface_(std::make_shared<CumotionInterface>(node, group))
{
}

bool CumotionPlanningContext::solve(planning_interface::MotionPlanDetailedResponse & res)
{
  cumotion_interface_->solve(planning_scene_, request_, res);
  return res.error_code_.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
}

bool CumotionPlanningContext::solve(planning_interface::MotionPlanResponse & res)
{
  planning_interface::MotionPlanDetailedResponse res_detailed;
  const bool ok = solve(res_detailed);

  res.error_code_ = res_detailed.error_code_;

  if (ok && !res_detailed.trajectory_.empty()) {
    res.trajectory_ = res_detailed.trajectory_[0];
    res.planning_time_ = res_detailed.processing_time_.empty() ? 0.0 :
      res_detailed.processing_time_[0];
  }
  return ok && !res_detailed.trajectory_.empty();
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia
