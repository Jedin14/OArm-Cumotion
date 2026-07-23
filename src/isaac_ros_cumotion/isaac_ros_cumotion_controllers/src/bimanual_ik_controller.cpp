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

#include "isaac_ros_cumotion_controllers/bimanual_ik_controller.hpp"

#include "cumotion/cumotion.h"
#include "cumotion/rotation3.h"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace
{

std::optional<std::vector<size_t>> FindInterfaceIndices(
  const std::vector<std::string> & names,
  const auto & interfaces)
{
  std::vector<size_t> indices;
  indices.reserve(names.size());
  for (const auto & name : names) {
    bool found = false;
    for (size_t i = 0; i < interfaces.size(); ++i) {
      if (interfaces[i].get_name() == name) {
        indices.push_back(i);
        found = true;
        break;
      }
    }
    if (!found) return std::nullopt;
  }
  return indices;
}

}  // namespace

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion_controllers
{

BimanualIkController::BimanualIkController() = default;

std::string BimanualIkController::MakeCommandInterfaceName(
  const std::string & joint, const std::string & iface) const
{
  if (command_prefix_.empty()) return joint + "/" + iface;
  return command_prefix_ + "/" + joint + "/" + iface + command_suffix_;
}

controller_interface::CallbackReturn BimanualIkController::on_init()
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn BimanualIkController::on_configure(
  const rclcpp_lifecycle::State &)
{
  joint_names_ = auto_declare<std::vector<std::string>>("joints", {});
  const auto left_ee = auto_declare<std::string>("left_end_effector_frame", "left_hand_palm_link");
  const auto right_ee = auto_declare<std::string>("right_end_effector_frame", "right_hand_palm_link");
  left_ee_command_frame_name_  = auto_declare<std::string>("left_ee_command_frame", left_ee);
  right_ee_command_frame_name_ = auto_declare<std::string>("right_ee_command_frame", right_ee);
  const std::filesystem::path urdf_path = auto_declare<std::string>("urdf_path", "");
  const std::filesystem::path xrdf_path = auto_declare<std::string>("xrdf_path", "");
  const std::filesystem::path rmpflow_config = auto_declare<std::string>("rmpflow_config_path", "");
  kp_ = auto_declare<double>("kp", 20.0);
  kd_ = auto_declare<double>("kd", 1.0);
  command_prefix_ = auto_declare<std::string>("command_prefix", "");
  command_suffix_ = auto_declare<std::string>("command_suffix", "");
  const auto pose_topic = auto_declare<std::string>("pose_topic", "~/reference_pose");
  drift_reset_threshold_ = auto_declare<double>("drift_reset_threshold", 0.5);

  auto & n = *get_node();
  if (joint_names_.empty()) {
    RCLCPP_ERROR(n.get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (urdf_path.empty() || xrdf_path.empty() || rmpflow_config.empty()) {
    RCLCPP_ERROR(n.get_logger(), "urdf_path, xrdf_path, and rmpflow_config_path must all be set");
    return controller_interface::CallbackReturn::ERROR;
  }

  auto robot_description = cumotion::LoadRobotFromFile(xrdf_path, urdf_path);
  if (!robot_description) {
    RCLCPP_ERROR(n.get_logger(), "Failed to load robot description from '%s' / '%s'",
      xrdf_path.c_str(), urdf_path.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  if (static_cast<int>(joint_names_.size()) != robot_description->numCSpaceCoords()) {
    RCLCPP_ERROR(n.get_logger(),
      "Joint count mismatch: controller has %zu joints but robot model has %d DOF",
      joint_names_.size(), robot_description->numCSpaceCoords());
    return controller_interface::CallbackReturn::ERROR;
  }

  kinematics_            = robot_description->kinematics();
  left_ee_frame_handle_  = kinematics_->frame(left_ee);
  right_ee_frame_handle_ = kinematics_->frame(right_ee);
  left_ee_frame_name_    = kinematics_->frameName(left_ee_frame_handle_);
  right_ee_frame_name_   = kinematics_->frameName(right_ee_frame_handle_);
  base_frame_            = kinematics_->frameName(kinematics_->baseFrame());

  auto world = cumotion::CreateWorld();
  rmpflow_ = cumotion::CreateRmpFlow(*cumotion::CreateRmpFlowConfigFromFile(
    rmpflow_config, *robot_description, world->addWorldView()));
  rmpflow_->addTargetFrame(left_ee_frame_name_);
  rmpflow_->addTargetFrame(right_ee_frame_name_);

  try {
    pinocchio::urdf::buildModel(urdf_path, pin_model_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(n.get_logger(), "Failed to build Pinocchio model from '%s': %s",
      urdf_path.c_str(), e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  pin_data_ = pinocchio::Data(pin_model_);
  q_full_   = Eigen::VectorXd::Zero(pin_model_.nq);
  v_full_   = Eigen::VectorXd::Zero(pin_model_.nv);
  a_full_   = Eigen::VectorXd::Zero(pin_model_.nv);

  const auto n_dof = static_cast<int>(joint_names_.size());
  pin_mappings_.clear();
  for (int i = 0; i < n_dof; ++i) {
    if (!pin_model_.existJointName(joint_names_[i])) {
      RCLCPP_ERROR(n.get_logger(),
        "Joint '%s' not found in Pinocchio model — torque will be zero for this DOF",
        joint_names_[i].c_str());
      continue;
    }
    pin_mappings_.push_back({
      i,
      static_cast<int>(pin_model_.idx_vs[pin_model_.getJointId(joint_names_[i])])
    });
  }

  tf_buffer_   = std::make_unique<tf2_ros::Buffer>(n.get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  pose_sub_ = n.create_subscription<geometry_msgs::msg::PoseArray>(
    pose_topic, rclcpp::SensorDataQoS(),
    [this](geometry_msgs::msg::PoseArray::SharedPtr msg) {
      if (msg->poses.size() < 2) {
        RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
          "Expected at least 2 reference poses (right + left), got %zu — ignoring",
          msg->poses.size());
        return;
      }
      PoseTargets targets;
      targets.right = ExtractAndTransformPose(
        msg->header, msg->poses[0], right_ee_command_frame_name_, right_ee_frame_name_);
      targets.left  = ExtractAndTransformPose(
        msg->header, msg->poses[1], left_ee_command_frame_name_, left_ee_frame_name_);
      pose_targets_buffer_.writeFromNonRT(targets);
    });

  RCLCPP_INFO(n.get_logger(), "BimanualIkController configured: %zu joints, left EE '%s', right EE '%s'",
    joint_names_.size(), left_ee_frame_name_.c_str(), right_ee_frame_name_.c_str());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn BimanualIkController::on_activate(
  const rclcpp_lifecycle::State &)
{
  const auto n_dof = static_cast<int>(joint_names_.size());
  joint_accel_               = Eigen::VectorXd::Zero(n_dof);
  joint_position_integrated_ = Eigen::VectorXd::Zero(n_dof);
  joint_velocity_integrated_ = Eigen::VectorXd::Zero(n_dof);
  joint_position_            = Eigen::VectorXd::Zero(n_dof);
  joint_velocity_            = Eigen::VectorXd::Zero(n_dof);
  tau_ff_                    = Eigen::VectorXd::Zero(n_dof);

  auto make_names = [&](const std::string & iface, bool cmd) -> std::vector<std::string> {
    std::vector<std::string> out;
    out.reserve(joint_names_.size());
    for (const auto & j : joint_names_)
      out.push_back(cmd ? MakeCommandInterfaceName(j, iface) : j + "/" + iface);
    return out;
  };
  auto find_state = [&](const std::string & iface) -> std::optional<std::vector<size_t>> {
    return FindInterfaceIndices(make_names(iface, false), state_interfaces_);
  };
  auto find_cmd = [&](const std::string & iface) -> std::optional<std::vector<size_t>> {
    return FindInterfaceIndices(make_names(iface, true), command_interfaces_);
  };

  const auto pos_s = find_state(hardware_interface::HW_IF_POSITION);
  const auto vel_s = find_state(hardware_interface::HW_IF_VELOCITY);
  const auto pos_c = find_cmd(hardware_interface::HW_IF_POSITION);
  const auto vel_c = find_cmd(hardware_interface::HW_IF_VELOCITY);
  const auto eff_c = find_cmd(hardware_interface::HW_IF_EFFORT);
  const auto kp_c  = find_cmd("kp");
  const auto kd_c  = find_cmd("kd");
  if (!pos_s || !vel_s || !pos_c || !vel_c || !eff_c || !kp_c || !kd_c) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to find required interfaces for joints");
    return controller_interface::CallbackReturn::ERROR;
  }
  hw_.pos_state  = *pos_s; hw_.vel_state  = *vel_s;
  hw_.pos_cmd    = *pos_c; hw_.vel_cmd    = *vel_c;
  hw_.effort_cmd = *eff_c; hw_.kp_cmd     = *kp_c; hw_.kd_cmd = *kd_c;

  for (int i = 0; i < n_dof; ++i) {
    const auto pos = state_interfaces_[hw_.pos_state[i]].get_optional<double>();
    const auto vel = state_interfaces_[hw_.vel_state[i]].get_optional<double>();
    if (!pos || !vel) {
      RCLCPP_ERROR(get_node()->get_logger(),
        "Failed to read state for joint '%s' on activation — hardware not ready",
        joint_names_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    joint_position_integrated_(i) = *pos;
    joint_velocity_integrated_(i) = *vel;
  }
  joint_position_ = joint_position_integrated_;
  joint_velocity_ = joint_velocity_integrated_;

  q_full_.setZero(); v_full_.setZero(); a_full_.setZero();
  for (const auto & m : pin_mappings_) {
    q_full_(m.pin_v_idx) = joint_position_integrated_(m.ctrl_idx);
    v_full_(m.pin_v_idx) = joint_velocity_integrated_(m.ctrl_idx);
  }
  rmpflow_->setPoseTarget(left_ee_frame_name_,
    kinematics_->pose(joint_position_integrated_, left_ee_frame_handle_));
  rmpflow_->setPoseTarget(right_ee_frame_name_,
    kinematics_->pose(joint_position_integrated_, right_ee_frame_handle_));
  rmpflow_->setCSpaceAttractor(Eigen::VectorXd::Zero(n_dof));
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn BimanualIkController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  rmpflow_->clearPoseTarget(left_ee_frame_name_);
  rmpflow_->clearPoseTarget(right_ee_frame_name_);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
BimanualIkController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & j : joint_names_) {
    config.names.push_back(MakeCommandInterfaceName(j, hardware_interface::HW_IF_POSITION));
    config.names.push_back(MakeCommandInterfaceName(j, hardware_interface::HW_IF_VELOCITY));
    config.names.push_back(MakeCommandInterfaceName(j, hardware_interface::HW_IF_EFFORT));
    config.names.push_back(MakeCommandInterfaceName(j, "kp"));
    config.names.push_back(MakeCommandInterfaceName(j, "kd"));
  }
  return config;
}

controller_interface::InterfaceConfiguration
BimanualIkController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & j : joint_names_) {
    config.names.push_back(j + "/" + hardware_interface::HW_IF_POSITION);
    config.names.push_back(j + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return config;
}

controller_interface::return_type BimanualIkController::update(
  const rclcpp::Time &, const rclcpp::Duration & period)
{
  const auto n_dof = static_cast<int>(joint_names_.size());
  const double dt = period.seconds();

  // Read hardware state.
  for (int i = 0; i < n_dof; ++i) {
    joint_position_(i) =
      state_interfaces_[hw_.pos_state[i]].get_optional<double>().value_or(joint_position_(i));
    joint_velocity_(i) =
      state_interfaces_[hw_.vel_state[i]].get_optional<double>().value_or(joint_velocity_(i));
  }

  // Reset integrator if drift from hardware exceeds threshold.
  const double drift = (joint_position_integrated_ - joint_position_).norm();
  if (drift > drift_reset_threshold_) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "Integrator drift %.3f rad exceeds threshold %.3f — resetting to hardware state",
      drift, drift_reset_threshold_);
    joint_position_integrated_ = joint_position_;
    joint_velocity_integrated_ = joint_velocity_;
  }

  // Apply pose targets from subscription (RT-safe read).
  const auto targets = *pose_targets_buffer_.readFromRT();
  if (targets.right) {
    rmpflow_->setPoseTarget(right_ee_frame_name_,
      cumotion::Pose3(cumotion::Rotation3(targets.right->second), targets.right->first));
  }
  if (targets.left) {
    rmpflow_->setPoseTarget(left_ee_frame_name_,
      cumotion::Pose3(cumotion::Rotation3(targets.left->second), targets.left->first));
  }

  // Evaluate RMPflow on integrated state.
  rmpflow_->evalAccel(joint_position_integrated_, joint_velocity_integrated_, joint_accel_);

  // One-step reference from integrated state.
  const Eigen::VectorXd v_target = joint_velocity_integrated_ + dt * joint_accel_;
  const Eigen::VectorXd q_target = joint_position_integrated_ + dt * v_target;

  // Inverse dynamics: tau_ff = M(q)*a + C(q,v)*v + G(q).
  for (const auto & m : pin_mappings_) {
    q_full_(m.pin_v_idx) = joint_position_integrated_(m.ctrl_idx);
    v_full_(m.pin_v_idx) = joint_velocity_integrated_(m.ctrl_idx);
    a_full_(m.pin_v_idx) = joint_accel_(m.ctrl_idx);
  }
  pinocchio::rnea(pin_model_, pin_data_, q_full_, v_full_, a_full_);
  for (const auto & m : pin_mappings_) {
    tau_ff_(m.ctrl_idx) = pin_data_.tau(m.pin_v_idx);
  }

  // Write commands: RNEA feedforward + hardware PD on position/velocity targets.
  for (int i = 0; i < n_dof; ++i) {
    (void)command_interfaces_[hw_.pos_cmd[i]].set_value(q_target(i));
    (void)command_interfaces_[hw_.vel_cmd[i]].set_value(v_target(i));
    (void)command_interfaces_[hw_.effort_cmd[i]].set_value(tau_ff_(i));
    (void)command_interfaces_[hw_.kp_cmd[i]].set_value(kp_);
    (void)command_interfaces_[hw_.kd_cmd[i]].set_value(kd_);
  }

  // Advance integrator.
  joint_velocity_integrated_ += dt * joint_accel_;
  joint_position_integrated_ += dt * joint_velocity_integrated_;

  return controller_interface::return_type::OK;
}

std::optional<PoseData> BimanualIkController::ExtractAndTransformPose(
  const std_msgs::msg::Header & header, const geometry_msgs::msg::Pose & pose,
  const std::string & ee_command_frame, const std::string & ee_frame) const
{
  try {
    // cmd_parent_T_cmd:
    tf2::Transform cmd_parent_T_cmd;
    tf2::fromMsg(pose, cmd_parent_T_cmd);

    // base_T_cmd_parent:
    const auto base_T_cmd_parent_msg = tf_buffer_->lookupTransform(
      base_frame_, header.frame_id, tf2_ros::fromMsg(header.stamp),
      tf2::durationFromSec(0.1));
    tf2::Transform base_T_cmd_parent;
    tf2::fromMsg(base_T_cmd_parent_msg.transform, base_T_cmd_parent);

    // base_T_cmd: 
    const auto base_T_cmd = base_T_cmd_parent * cmd_parent_T_cmd;

    // cmd_T_ee:
    const auto cmd_T_ee_msg = tf_buffer_->lookupTransform(
        ee_command_frame, ee_frame, tf2::TimePointZero, tf2::durationFromSec(0.1));
    tf2::Transform cmd_T_ee;
    tf2::fromMsg(cmd_T_ee_msg.transform, cmd_T_ee);
    
    // base_T_ee:
    const auto base_T_ee = base_T_cmd * cmd_T_ee;

    const auto & p = base_T_ee.getOrigin();
    const auto & q = base_T_ee.getRotation();
    return std::make_pair(
      Eigen::Vector3d{p.x(), p.y(), p.z()},
      Eigen::Quaterniond{q.w(), q.x(), q.y(), q.z()});
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "Failed to transform pose from '%s' to '%s': %s",
      header.frame_id.c_str(), base_frame_.c_str(), ex.what());
    return std::nullopt;
  }
}


}  // namespace cumotion_controllers
}  // namespace isaac_ros
}  // namespace nvidia

PLUGINLIB_EXPORT_CLASS(
  nvidia::isaac_ros::cumotion_controllers::BimanualIkController,
  controller_interface::ControllerInterface)
