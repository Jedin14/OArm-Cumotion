// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ISAAC_ROS_CUMOTION_CONTROLLERS__BIMANUAL_IK_CONTROLLER_HPP_
#define ISAAC_ROS_CUMOTION_CONTROLLERS__BIMANUAL_IK_CONTROLLER_HPP_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/header.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "cumotion/kinematics.h"
#include "cumotion/robot_description.h"
#include "cumotion/rmpflow.h"
#include "cumotion/world.h"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "realtime_tools/realtime_buffer.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion_controllers
{

using PoseData = std::pair<Eigen::Vector3d, Eigen::Quaterniond>;

struct PoseTargets {
  std::optional<PoseData> right{};
  std::optional<PoseData> left{};
};

/* Bimanual IK controller using cuMotion RMPflow for joint acceleration generation.
 * Subscribes to ~/reference_pose (PoseArray): poses[0] = right EE, poses[1] = left EE.
 * Writes position/velocity/effort commands to hardware interfaces. */
class BimanualIkController : public controller_interface::ControllerInterface
{
public:
  BimanualIkController();

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  // Parameters
  std::vector<std::string> joint_names_{};
  double kp_{};
  double kd_{};
  double drift_reset_threshold_{};
  std::string command_prefix_{};
  std::string command_suffix_{};

  // RMPflow (cuMotion)
  std::unique_ptr<cumotion::RmpFlow> rmpflow_{nullptr};
  std::unique_ptr<cumotion::Kinematics> kinematics_{nullptr};
  cumotion::Kinematics::FrameHandle left_ee_frame_handle_{};
  cumotion::Kinematics::FrameHandle right_ee_frame_handle_{};
  std::string left_ee_frame_name_{};
  std::string right_ee_frame_name_{};
  std::string left_ee_command_frame_name_{};
  std::string right_ee_command_frame_name_{};
  std::string base_frame_{};

  // Inverse dynamics (Pinocchio RNEA)
  struct PinMapping { int ctrl_idx; int pin_v_idx; };
  std::vector<PinMapping> pin_mappings_{};
  pinocchio::Model pin_model_{};
  pinocchio::Data pin_data_{};
  // Full-model state vectors — non-controlled joints held at zero (zeroed once on activation).
  Eigen::VectorXd q_full_{};
  Eigen::VectorXd v_full_{};
  Eigen::VectorXd a_full_{};
  // Per-controller-joint feedforward torque — zeroed on activation, overwritten each cycle.
  Eigen::VectorXd tau_ff_{};

  // Hardware interface index maps — built once in on_activate
  struct HardwareIndexMaps {
    std::vector<size_t> pos_state{};
    std::vector<size_t> vel_state{};
    std::vector<size_t> pos_cmd{};
    std::vector<size_t> vel_cmd{};
    std::vector<size_t> effort_cmd{};
    std::vector<size_t> kp_cmd{};
    std::vector<size_t> kd_cmd{};
  };
  HardwareIndexMaps hw_{};

  // Hardware state — updated each cycle from state interfaces.
  Eigen::VectorXd joint_position_{};
  Eigen::VectorXd joint_velocity_{};
  Eigen::VectorXd joint_accel_{};
  
  // Integrated state — open-loop integration of RMPflow output;
  // reset to hardware on activation only.
  Eigen::VectorXd joint_position_integrated_{};
  Eigen::VectorXd joint_velocity_integrated_{};

  // ROS communication
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pose_sub_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_{nullptr};
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  realtime_tools::RealtimeBuffer<PoseTargets> pose_targets_buffer_{};

  // Helpers
  std::string MakeCommandInterfaceName(
    const std::string & joint, const std::string & iface) const;
  std::optional<PoseData> ExtractAndTransformPose(
    const std_msgs::msg::Header & header,
    const geometry_msgs::msg::Pose & pose,
    const std::string & ee_command_frame,
    const std::string & ee_frame) const;
};

}  // namespace cumotion_controllers
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_CONTROLLERS__BIMANUAL_IK_CONTROLLER_HPP_
