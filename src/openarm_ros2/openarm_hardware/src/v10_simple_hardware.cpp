// Copyright 2025 Enactic, Inc.
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

#include "openarm_hardware/v10_simple_hardware.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"

namespace openarm_hardware {

OpenArm_v10HW::OpenArm_v10HW() = default;

bool OpenArm_v10HW::parse_config(const hardware_interface::HardwareInfo& info) {
  // Parse CAN interface (default: can0)
  auto it = info.hardware_parameters.find("can_interface");
  can_interface_ = (it != info.hardware_parameters.end()) ? it->second : "can0";

  // Parse arm prefix (default: empty for single arm, "left_" or "right_" for
  // bimanual)
  it = info.hardware_parameters.find("arm_prefix");
  arm_prefix_ = (it != info.hardware_parameters.end()) ? it->second : "";

  // Parse gripper enable (default: true for V10)
  it = info.hardware_parameters.find("hand");
  if (it == info.hardware_parameters.end()) {
    hand_ = true;  // Default to true for V10
  } else {
    // Handle both "true"/"True" and "false"/"False"
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    hand_ = (value == "true");
  }

  // Parse CAN-FD enable (default: true for V10)
  it = info.hardware_parameters.find("can_fd");
  std::string raw_can_fd = (it != info.hardware_parameters.end()) ? it->second : std::string("<unset>");
  if (it == info.hardware_parameters.end()) {
    can_fd_ = true;  // Default to true for V10
  } else {
    // Handle both "true"/"True" and "false"/"False"
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    can_fd_ = (value == "true");
  }

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Raw can_fd param: %s", raw_can_fd.c_str());

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Configuration: CAN=%s, arm_prefix=%s, hand=%s, can_fd=%s",
              can_interface_.c_str(), arm_prefix_.c_str(),
              hand_ ? "enabled" : "disabled", can_fd_ ? "enabled" : "disabled");
  return true;
}

void OpenArm_v10HW::generate_joint_names() {
  joint_names_.clear();
  // TODO: read from urdf properly and sort in the future.
  // Currently, the joint names are hardcoded for order consistency to align
  // with hardware. Generate arm joint names: openarm_{arm_prefix}joint{N}
  for (size_t i = 1; i <= ARM_DOF; ++i) {
    std::string joint_name =
        "openarm_" + arm_prefix_ + "joint" + std::to_string(i);
    joint_names_.push_back(joint_name);
  }

  // Generate gripper joint name if enabled
  if (hand_) {
    std::string gripper_joint_name = "openarm_" + arm_prefix_ + "finger_joint1";
    joint_names_.push_back(gripper_joint_name);
    RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "Added gripper joint: %s",
                gripper_joint_name.c_str());
  } else {
    RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
                "Gripper joint NOT added because hand_=false");
  }

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Generated %zu joint names for arm prefix '%s'",
              joint_names_.size(), arm_prefix_.c_str());
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_init(
    const hardware_interface::HardwareInfo& info) {
  if (hardware_interface::SystemInterface::on_init(info) !=
      CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
  // Parse configuration
  if (!parse_config(info)) {
    return CallbackReturn::ERROR;
  }

  // Generate joint names based on arm prefix
  generate_joint_names();

  // Validate joint count (7 arm joints + optional gripper)
  size_t expected_joints = ARM_DOF + (hand_ ? 1 : 0);
  if (joint_names_.size() != expected_joints) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"),
                 "Generated %zu joint names, expected %zu", joint_names_.size(),
                 expected_joints);
    return CallbackReturn::ERROR;
  }

  // Initialize OpenArm with configurable CAN-FD setting
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Initializing OpenArm on %s with CAN-FD %s...",
              can_interface_.c_str(), can_fd_ ? "enabled" : "disabled");
  openarm_ =
      std::make_unique<openarm::can::socket::OpenArm>(can_interface_, can_fd_);

  // Initialize arm motors with V10 defaults
  openarm_->init_arm_motors(DEFAULT_MOTOR_TYPES, DEFAULT_SEND_CAN_IDS,
                            DEFAULT_RECV_CAN_IDS);

  // Initialize gripper if enabled
  if (hand_) {
    RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "Initializing gripper...");
    openarm_->init_gripper_motor(DEFAULT_GRIPPER_MOTOR_TYPE,
                                 DEFAULT_GRIPPER_SEND_CAN_ID,
                                 DEFAULT_GRIPPER_RECV_CAN_ID);
  }

  // Initialize state and command vectors based on generated joint count
  const size_t total_joints = joint_names_.size();
  pos_commands_.resize(total_joints, 0.0);
  vel_commands_.resize(total_joints, 0.0);
  tau_commands_.resize(total_joints, 0.0);
  pos_states_.resize(total_joints, 0.0);
  vel_states_.resize(total_joints, 0.0);
  tau_states_.resize(total_joints, 0.0);

  // Initialize KDL for gravity compensation
  KDL::Tree kdl_tree;
  if (!kdl_parser::treeFromString(info.original_xml, kdl_tree)) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), "Failed to parse KDL tree from URDF");
    return CallbackReturn::ERROR;
  }

  // Use the true root of the tree (e.g. 'world' or 'base_link') so that the
  // gravity vector (0, 0, -9.81) is correctly applied in the world frame,
  // accounting for any mounting rotations of the arms.
  std::string base_link = kdl_tree.getRootSegment()->first;
  // Tip at the hand, not link7. The hand and fingers weigh 0.4221 kg -- close
  // to link7's own 0.466 kg -- and hang at the longest moment arm on the arm,
  // so leaving them out of the gravity model is an uncompensated load exactly
  // where it costs most. Measured before this change: a repeatable elbow
  // (joint4) error of 14-28 mrad in the direction gravity pulls, holding
  // 4.4-5.0 Nm, giving a tool that arrived 13.7 mm low at z=0.405 and 33.5 mm
  // low at z=0.555 -- the sag grows with extension, and it is why the gripper
  // closed above the object.
  //
  // Safe for the joint-count check below: right_openarm_hand_joint is fixed,
  // so the chain still has exactly ARM_DOF movable joints. The fingers hang
  // off the hand on prismatic joints and stay out of the chain, leaving only
  // their 0.072 kg unmodelled.
  std::string tip_link = "openarm_" + arm_prefix_ + "hand";
  KDL::Chain kdl_chain;
  if (!kdl_tree.getChain(base_link, tip_link, kdl_chain)) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), "Failed to extract KDL chain from %s to %s", base_link.c_str(), tip_link.c_str());
    return CallbackReturn::ERROR;
  }

  // Verify that the chain has exactly ARM_DOF movable joints
  if (kdl_chain.getNrOfJoints() != ARM_DOF) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), 
                 "KDL Chain from %s to %s has %d joints, but expected %zu", 
                 base_link.c_str(), tip_link.c_str(), kdl_chain.getNrOfJoints(), ARM_DOF);
    return CallbackReturn::ERROR;
  }

  KDL::Vector gravity(0.0, 0.0, -9.81);
  kdl_dyn_param_ = std::make_unique<KDL::ChainDynParam>(kdl_chain, gravity);
  kdl_q_.resize(ARM_DOF);
  kdl_gravity_torque_.resize(ARM_DOF);
  integral_errors_.resize(total_joints, 0.0);

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "OpenArm V10 Simple HW initialized successfully");

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // Set callback mode to ignore during configuration
  openarm_->refresh_all();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  openarm_->recv_all();

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
OpenArm_v10HW::export_state_interfaces() {
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], hardware_interface::HW_IF_POSITION, &pos_states_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], hardware_interface::HW_IF_VELOCITY, &vel_states_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], hardware_interface::HW_IF_EFFORT, &tau_states_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
OpenArm_v10HW::export_command_interfaces() {
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  // TODO: consider exposing only needed interfaces to avoid undefined behavior.
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_names_[i], hardware_interface::HW_IF_POSITION,
        &pos_commands_[i]));
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_names_[i], hardware_interface::HW_IF_VELOCITY,
        &vel_commands_[i]));
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_names_[i], hardware_interface::HW_IF_EFFORT, &tau_commands_[i]));
  }

  return command_interfaces;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "Activating OpenArm V10...");
  openarm_->set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
  openarm_->enable_all();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  openarm_->recv_all();

  // Hold the arm where it is, rather than driving it anywhere.
  //
  // This used to call return_to_zero(), which interpolates every joint to
  // zero over 2.4 seconds -- unplanned, uncollision-checked, and from
  // wherever the arm happens to be standing. On a cold bringup that is a
  // surprise. After a disengage it is worse: these are direct-drive motors
  // with no brakes, so the arm has sagged under its own weight, and
  // activating it made the arm jump. Measured on the robot; it jumped.
  //
  // Seeding the command with the measured state means the very first write()
  // asks for the position the arm is already in, so enabling is only
  // enabling -- red light to green light and nothing else. Moving it
  // somewhere afterwards is then a planned motion like any other, which is
  // what /pick_place/engage does once this returns.
  hold_current_position();

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "OpenArm V10 activated");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Deactivating OpenArm V10...");

  // Disable all motors (like full_arm.cpp exit)
  openarm_->disable_all();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  openarm_->recv_all();

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "OpenArm V10 deactivated");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type OpenArm_v10HW::read(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  // Receive all motor states
  openarm_->refresh_all();
  openarm_->recv_all();

  // Read arm joint states
  const auto& arm_motors = openarm_->get_arm().get_motors();
  for (size_t i = 0; i < ARM_DOF && i < arm_motors.size(); ++i) {
    pos_states_[i] = arm_motors[i].get_position();
    vel_states_[i] = arm_motors[i].get_velocity();
    tau_states_[i] = arm_motors[i].get_torque();
  }

  // Read gripper state if enabled
  if (hand_ && joint_names_.size() > ARM_DOF) {
    const auto& gripper_motors = openarm_->get_gripper().get_motors();
    if (!gripper_motors.empty()) {
      // TODO the mappings are approximates
      // Convert motor position (radians) to joint value (0-0.044m)
      double motor_pos = gripper_motors[0].get_position();
      pos_states_[ARM_DOF] = motor_radians_to_joint(motor_pos);

      // Velocity mapping is still unimplemented. Torque is not: it is read
      // straight off the motor, in Nm, because a grip-force cap is impossible
      // without it -- the closing force is position error times the fixed
      // GRIPPER_DEFAULT_KP, so the only way to bound it is to stop advancing
      // the position command once measured torque says the cap is reached.
      // Read-only: this changes no command, it only populates the effort state
      // interface that is already exported for this joint, so the value shows
      // up in /joint_states for whoever enforces the cap.
      vel_states_[ARM_DOF] = 0;  // gripper_motors[0].get_velocity();
      tau_states_[ARM_DOF] = gripper_motors[0].get_torque();
    }
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type OpenArm_v10HW::write(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  
  // Calculate gravity compensation torques
  for (size_t i = 0; i < ARM_DOF; ++i) {
    kdl_q_(i) = pos_states_[i]; 
  }
  
  bool kdl_success = false;
  if (kdl_dyn_param_ && kdl_dyn_param_->JntToGravity(kdl_q_, kdl_gravity_torque_) >= 0) {
    kdl_success = true;
  }

  // Control arm motors with MIT control + Gravity + Integral Compensation
  std::vector<openarm::damiao_motor::MITParam> arm_params;
  double dt = period.seconds();
  if (dt <= 0.0) dt = 0.001; // fallback

  for (size_t i = 0; i < ARM_DOF; ++i) {
    // Integral control to eliminate steady state error
    double error = pos_commands_[i] - pos_states_[i];
    integral_errors_[i] += error * dt;
    
    double i_torque = DEFAULT_KI[i] * integral_errors_[i];
    // Anti-windup
    i_torque = std::clamp(i_torque, -MAX_I_TORQUE, MAX_I_TORQUE);
    // Clamp the integral error itself as well
    if (DEFAULT_KI[i] > 0.0) {
      integral_errors_[i] = std::clamp(integral_errors_[i], -MAX_I_TORQUE / DEFAULT_KI[i], MAX_I_TORQUE / DEFAULT_KI[i]);
    }

    double ff_torque = tau_commands_[i] + i_torque;
    if (kdl_success) {
      ff_torque += kdl_gravity_torque_(i);
    }
    arm_params.push_back({DEFAULT_KP[i], DEFAULT_KD[i], pos_commands_[i],
                          vel_commands_[i], ff_torque});
  }
  openarm_->get_arm().mit_control_all(arm_params);
  // Control gripper if enabled
  if (hand_ && joint_names_.size() > ARM_DOF) {
    // TODO the true mappings are unimplemented.
    double motor_command = joint_to_motor_radians(pos_commands_[ARM_DOF]);
    openarm_->get_gripper().mit_control_all(
        {{GRIPPER_DEFAULT_KP, GRIPPER_DEFAULT_KD, motor_command, 0, 0}});
  }
  openarm_->recv_all(1000);
  return hardware_interface::return_type::OK;
}

void OpenArm_v10HW::hold_current_position() {
  // Read where the motors actually are, and make that the command. Nothing
  // moves: the position error the controller sees is zero, so the MIT
  // command is the arm's own weight and nothing more.
  //
  // Both buffers, because a command left at its resize() default of 0.0 is
  // the jump this exists to prevent -- the first write() after activation
  // would ask for zero on every joint.
  openarm_->refresh_all();
  openarm_->recv_all();

  const auto& arm_motors = openarm_->get_arm().get_motors();
  for (size_t i = 0; i < ARM_DOF && i < arm_motors.size(); ++i) {
    pos_states_[i] = arm_motors[i].get_position();
    pos_commands_[i] = pos_states_[i];
  }

  if (hand_ && joint_names_.size() > ARM_DOF) {
    const auto& gripper_motors = openarm_->get_gripper().get_motors();
    if (!gripper_motors.empty()) {
      pos_states_[ARM_DOF] =
          motor_radians_to_joint(gripper_motors[0].get_position());
      pos_commands_[ARM_DOF] = pos_states_[ARM_DOF];
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Holding the posture the arm was already in");
}

// Kept, and deliberately not called from on_activate any more -- see there.
// Still the right thing for anything that genuinely wants the arm folded up
// and can guarantee the path is clear.
void OpenArm_v10HW::return_to_zero() {
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Returning to zero position...");

  // Smoothly interpolate from current position to home to avoid startup jumps.
  constexpr size_t kHomingSteps = 120;
  constexpr double kHomingDurationSec = 2.4;
  const auto step_sleep = std::chrono::microseconds(
      static_cast<int>(kHomingDurationSec * 1e6 / static_cast<double>(kHomingSteps)));

  openarm_->refresh_all();
  openarm_->recv_all();

  const auto& arm_motors = openarm_->get_arm().get_motors();
  std::vector<double> arm_start(ARM_DOF, 0.0);
  for (size_t i = 0; i < ARM_DOF && i < arm_motors.size(); ++i) {
    arm_start[i] = arm_motors[i].get_position();
  }

  double gripper_start_joint = GRIPPER_JOINT_0_POSITION;
  if (hand_) {
    const auto& gripper_motors = openarm_->get_gripper().get_motors();
    if (!gripper_motors.empty()) {
      gripper_start_joint = motor_radians_to_joint(gripper_motors[0].get_position());
    }
  }

  for (size_t step = 1; step <= kHomingSteps; ++step) {
    const double alpha = static_cast<double>(step) / static_cast<double>(kHomingSteps);

    std::vector<openarm::damiao_motor::MITParam> arm_params;
    arm_params.reserve(ARM_DOF);
    for (size_t i = 0; i < ARM_DOF; ++i) {
      const double target_pos = (1.0 - alpha) * arm_start[i] + alpha * 0.0;
      arm_params.push_back(
          {DEFAULT_KP[i], DEFAULT_KD[i], target_pos, 0.0, 0.0});
    }
    openarm_->get_arm().mit_control_all(arm_params);

    if (hand_) {
      const double gripper_target_joint =
          (1.0 - alpha) * gripper_start_joint + alpha * GRIPPER_JOINT_0_POSITION;
      const double gripper_target_motor = joint_to_motor_radians(gripper_target_joint);
      openarm_->get_gripper().mit_control_all({{GRIPPER_DEFAULT_KP, GRIPPER_DEFAULT_KD,
                                                gripper_target_motor, 0.0, 0.0}});
    }

    std::this_thread::sleep_for(step_sleep);
    openarm_->recv_all();
  }
}

// Gripper mapping helper functions
double OpenArm_v10HW::joint_to_motor_radians(double joint_value) {
  // Joint 0=closed -> motor 0 rad, Joint 0.044=open -> motor -1.0472 rad
  return (joint_value / GRIPPER_JOINT_0_POSITION) *
         GRIPPER_MOTOR_1_RADIANS;  // Scale from 0-0.044 to 0 to -1.0472
}

double OpenArm_v10HW::motor_radians_to_joint(double motor_radians) {
  // Motor 0 rad=closed -> joint 0, Motor -1.0472 rad=open -> joint 0.044
  return GRIPPER_JOINT_0_POSITION *
         (motor_radians /
          GRIPPER_MOTOR_1_RADIANS);  // Scale from 0 to -1.0472 to 0-0.044
}

}  // namespace openarm_hardware

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(openarm_hardware::OpenArm_v10HW,
                       hardware_interface::SystemInterface)
