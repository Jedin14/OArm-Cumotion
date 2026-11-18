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
#include <cmath>
#include <thread>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "openarm_hardware/gains_file.hpp"
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

  // The control-law file (2026-11-17), calibration/motor_gains_<arm>.yaml by
  // the ros2_control xacro's default. Unset or empty: none, the compiled law.
  it = info.hardware_parameters.find("gains_file");
  gains_file_ = (it != info.hardware_parameters.end()) ? it->second : "";

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
  // -1 until a frame has been read: "not known", not "not enabled".
  status_states_.resize(total_joints, -1.0);
  integral_torque_states_.resize(ARM_DOF, 0.0);
  prev_pos_commands_.resize(total_joints, 0.0);
  status_seen_.assign(total_joints, -1);
  status_shown_.assign(total_joints, -1);
  status_changes_.assign(total_joints, 0);
  status_logged_at_.assign(total_joints, std::chrono::steady_clock::now());

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

  // The control law: the compiled gains, then the gains file if there is one.
  load_control_law();

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "OpenArm V10 Simple HW initialized successfully");

  return CallbackReturn::SUCCESS;
}

void OpenArm_v10HW::load_control_law() {
  const auto logger = rclcpp::get_logger("OpenArm_v10HW");
  // Compiled: exactly the law write() ran before 2026-11-17 (gains, caps,
  // deadband), plus the two new switches on.
  laws_.assign(ARM_DOF, mit_law::JointLaw());
  for (size_t i = 0; i < ARM_DOF; ++i) {
    mit_law::JointLaw& law = laws_[i];
    law.kp = DEFAULT_KP[i];
    law.kd = DEFAULT_KD[i];
    law.ki = DEFAULT_KI[i];
    law.i_cap = std::min(MAX_I_TORQUE, MAX_I_TORQUE_JOINT[i]);
    law.deadband = INTEGRAL_DEADBAND;
    law.gate_in = INTEGRAL_DEADBAND;
    law.gate_out = INTEGRAL_DEADBAND;
    law.vff_max = VFF_MAX[i];
  }
  law_flags_ = mit_law::LawFlags();
  law_memory_.assign(ARM_DOF, mit_law::JointMemory());
  recover_.assign(ARM_DOF, recovery::State());

  // Velocity feed-forward and the integral freeze are ON without a file, or
  // with one that is missing or rejected: they are the fix for the lag and
  // the overshoot, and the file is how to turn them off. The gains profile
  // is only ever on from an accepted file that says so.
  std::string arm = arm_prefix_;
  if (!arm.empty() && arm.back() == '_') arm.pop_back();
  gains_file::Contents contents;
  std::string why;
  switch (gains_file::load(gains_file_, arm, laws_, contents, why)) {
    case gains_file::Load::kNoFile:
      gains_source_ = "no gains file configured: compiled law";
      RCLCPP_INFO(logger, "%s arm: %s", arm.c_str(), gains_source_.c_str());
      break;
    case gains_file::Load::kMissing:
      gains_source_ =
          "gains file " + gains_file_ + " not found: compiled law";
      RCLCPP_WARN(logger, "%s arm: %s", arm.c_str(), gains_source_.c_str());
      break;
    case gains_file::Load::kRejected:
      gains_source_ = "gains file " + gains_file_ + " REJECTED (" + why +
                      "): compiled law, none of the file used";
      RCLCPP_ERROR(logger, "%s arm: %s", arm.c_str(), gains_source_.c_str());
      break;
    case gains_file::Load::kLoaded:
      gains_file::apply(contents, laws_, law_flags_);
      gains_source_ = "gains file " + gains_file_ +
                      (contents.profile_enabled ? " (profile on)"
                                                : " (profile off)");
      RCLCPP_INFO(logger, "%s arm: %s", arm.c_str(), gains_source_.c_str());
      break;
  }
}

void OpenArm_v10HW::log_control_law() const {
  // What write() will actually run, so a log answers "which law was this?".
  const auto logger = rclcpp::get_logger("OpenArm_v10HW");
  RCLCPP_INFO(logger,
              "%s control law: velocity feed-forward %s, integral freeze %s "
              "(moving above %.3f rad/s, still for %.2f s), held-point "
              "velocity guard %s, motor re-enable %s (ease back %.2f rad/s); %s",
              arm_prefix_.c_str(),
              law_flags_.velocity_feedforward ? "on" : "off",
              law_flags_.freeze_while_moving ? "on" : "off",
              law_flags_.freeze_rate, law_flags_.freeze_hold,
              law_flags_.zero_velocity_when_held ? "on" : "off",
              law_flags_.auto_reenable ? "on" : "off", law_flags_.reenable_rate,
              gains_source_.c_str());
  for (size_t i = 0; i < ARM_DOF && i < laws_.size(); ++i) {
    const mit_law::JointLaw& law = laws_[i];
    RCLCPP_INFO(logger,
                "  %s: kp %.1f kd %.2f ki %.1f i_cap %.2f Nm, deadband %.2f "
                "mrad, gate %.2f/%.2f mrad, vff_max %.1f rad/s",
                joint_names_[i].c_str(), law.kp, law.kd, law.ki, law.i_cap,
                1000.0 * law.deadband, 1000.0 * law.gate_in,
                1000.0 * law.gate_out, law.vff_max);
  }
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
    // Not one of the standard hardware_interface names, deliberately:
    // there is no standard one for "the motor's own status", and
    // joint_state_broadcaster publishes whatever it is given on
    // /dynamic_joint_states, which is where anything wanting to know
    // whether a motor is driving can read it.
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], "status", &status_states_[i]));
    // Same deal (2026-11-17): the integral's torque, N m, arm joints only
    // (the gripper has no integral). Nothing claims it but the
    // joint_state_broadcaster, which publishes every state interface it is
    // given on /dynamic_joint_states (as it does "status"); the trajectory
    // controllers claim only "position", and /joint_states carries only
    // position, velocity and effort.
    if (i < ARM_DOF) {
      state_interfaces.emplace_back(hardware_interface::StateInterface(
          joint_names_[i], "integral_torque", &integral_torque_states_[i]));
    }
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
  // Then check that they actually came up, and do something about the ones
  // that did not.
  //
  // enable_all() sends seven commands back to back and reads nothing back.
  // A motor that misses its own, or that is sitting in a latched fault and
  // will not accept it, stays limp -- and since its encoder keeps reporting
  // perfectly, nothing downstream notices: the joint reads a plausible
  // position, near-zero torque, and moves freely by hand. Measured on this
  // robot: the left wrist roll did exactly that while every other joint on
  // the same bus, including the gripper past it, was driving.
  verify_enabled();

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

  // No velocity either (2026-11-17): the velocity command is now the motor's
  // velocity target, and the JTC reads the command interfaces back as its
  // starting state on activation (set_last_command_interface_value_as_state_
  // on_activation), so it must start at "standing still". And the integral's
  // "has the command moved?" starts from the held command, not from 0.
  std::fill(vel_commands_.begin(), vel_commands_.end(), 0.0);
  prev_pos_commands_ = pos_commands_;
  for (auto& m : law_memory_) {
    m.held_cycles = 0;
  }
  log_control_law();
  start_status_watch();

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "OpenArm V10 activated");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Deactivating OpenArm V10...");
  // Disabling the motors changes every status on purpose; not news.
  watch_status_ = false;

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
    status_states_[i] = static_cast<double>(arm_motors[i].get_status());
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
      status_states_[ARM_DOF] =
          static_cast<double>(gripper_motors[0].get_status());
    }
  }

  watch_status();
  return hardware_interface::return_type::OK;
}

void OpenArm_v10HW::start_status_watch() {
  // The baseline is what the motors say right after verify_enabled(), which
  // has already reported any that did not come up.
  const auto now = std::chrono::steady_clock::now();
  const auto& arm_motors = openarm_->get_arm().get_motors();
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    int s = -1;
    if (i < ARM_DOF && i < arm_motors.size()) {
      s = arm_motors[i].get_status();
    } else if (i == ARM_DOF && hand_) {
      const auto& gripper_motors = openarm_->get_gripper().get_motors();
      if (!gripper_motors.empty()) s = gripper_motors[0].get_status();
    }
    status_seen_[i] = s;
    status_shown_[i] = s;
    status_changes_[i] = 0;
    // Backdated, so the first change is logged at once.
    status_logged_at_[i] =
        now - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double>(STATUS_LOG_PERIOD));
  }
  watch_status_ = true;
}

void OpenArm_v10HW::watch_status() {
  // Logs only. Deciding anything from these codes waits until the Damiao
  // codes are confirmed for libopenarm_can 1.2.9; until then a change is
  // worth seeing, and what to do about it is for whoever reads "status" on
  // /dynamic_joint_states (the app).
  if (!watch_status_) return;
  const auto logger = rclcpp::get_logger("OpenArm_v10HW");
  const auto now = std::chrono::steady_clock::now();
  for (size_t i = 0; i < joint_names_.size() && i < status_states_.size();
       ++i) {
    const int s = static_cast<int>(status_states_[i]);
    if (s != status_seen_[i]) {
      status_seen_[i] = s;
      ++status_changes_[i];
    }
    if (status_changes_[i] == 0) continue;
    // A flapping status would otherwise log 100 times a second; this logs it
    // at most once per period, and always the latest value within a period.
    if (std::chrono::duration<double>(now - status_logged_at_[i]).count() <
        STATUS_LOG_PERIOD) {
      continue;
    }
    if (s == 1) {
      RCLCPP_WARN(logger,
                  "%s motor status %d -> %d (%d change(s)): enabled again "
                  "(1 enabled; 0 disabled, 8-14 faults)",
                  joint_names_[i].c_str(), status_shown_[i], s,
                  status_changes_[i]);
    } else {
      RCLCPP_ERROR(logger,
                   "%s motor status %d -> %d (%d change(s)): this motor may "
                   "not be driving -- it can hold nothing while its encoder "
                   "reads on (1 enabled; 0 disabled, 8-14 faults)",
                   joint_names_[i].c_str(), status_shown_[i], s,
                   status_changes_[i]);
    }
    status_shown_[i] = s;
    status_changes_[i] = 0;
    status_logged_at_[i] = now;
  }
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

  // The law itself is in mit_law.hpp (2026-11-17): the same integral, caps
  // and deadband as before, plus velocity feed-forward and the integral held
  // while the command moves -- each switchable in the gains file.
  const double now_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0_).count();
  for (size_t i = 0; i < ARM_DOF; ++i) {
    const double gravity = kdl_success ? kdl_gravity_torque_(i) : 0.0;
    // A motor that stopped driving (recovery.hpp, 2026-11-18): held at where it
    // is, re-enabled, eased back -- never a jump to a command it fell away from.
    if (i < recover_.size()) {
      const recovery::Out ro = recovery::step(
          recover_[i], watch_status_ && law_flags_.auto_reenable,
          static_cast<int>(status_states_[i]), pos_states_[i], pos_commands_[i],
          now_s, dt, law_flags_.reenable_rate);
      report_recovery(i, ro);
      if (ro.action == recovery::Action::kEnable) {
        openarm_->get_arm().enable_one(static_cast<int>(i));
      } else if (ro.action == recovery::Action::kClear) {
        openarm_->get_arm().clear_error_one(static_cast<int>(i));
      }
      if (ro.overriding) {
        law_memory_[i].integral = 0.0;       // nothing stored while it hung
        law_memory_[i].still_time = 0.0;
        prev_pos_commands_[i] = ro.q;
        integral_torque_states_[i] = 0.0;
        arm_params.push_back({laws_[i].kp, laws_[i].kd, ro.q, 0.0, gravity});
        continue;
      }
    }
    const mit_law::Command c = mit_law::step(
        laws_[i], law_flags_, pos_states_[i], pos_commands_[i],
        prev_pos_commands_[i], vel_commands_[i], tau_commands_[i], gravity,
        dt, law_memory_[i]);
    // The command actually used (a NaN command keeps the last finite one).
    prev_pos_commands_[i] = c.q;
    integral_torque_states_[i] = c.i_torque;
    arm_params.push_back({c.kp, c.kd, c.q, c.dq, c.tau});
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

void OpenArm_v10HW::report_recovery(size_t i, const recovery::Out& ro) {
  // Events only (a handful per drop-out), so logging here costs nothing per cycle.
  if (!(ro.went_down || ro.came_back || ro.done || ro.gave_up)) return;
  const auto logger = rclcpp::get_logger("OpenArm_v10HW");
  const char* name = joint_names_[i].c_str();
  const int s = static_cast<int>(status_states_[i]);
  if (ro.went_down) {
    RCLCPP_ERROR(logger,
                 "%s stopped driving (status %d): held at its measured %.3f rad, "
                 "re-enabling%s -- check its power and CAN cable",
                 name, s, pos_states_[i],
                 law_flags_.auto_reenable ? "" : " is OFF (auto_reenable false)");
  }
  if (ro.gave_up) {
    RCLCPP_ERROR(logger,
                 "%s: status %d is not re-enabled automatically (%s): it holds "
                 "nothing until the cause is fixed and terminal 1 restarted",
                 name, s, recovery::retried(s) ? "25 enables did not take"
                                               : "a fault: over-current / heat / voltage / load");
  }
  if (ro.came_back) {
    RCLCPP_WARN(logger, "%s is enabled again at %.3f rad; easing back to the command "
                "at %.2f rad/s", name, pos_states_[i], law_flags_.reenable_rate);
  }
  if (ro.done) {
    RCLCPP_WARN(logger, "%s is back on its command", name);
  }
}

void OpenArm_v10HW::verify_enabled() {
  // Up to three rounds: read what the motors say, retry the ones that are
  // not enabled, and clear a latched fault first where there is one, since
  // a faulted motor will not take an enable until it is cleared.
  const auto& logger = rclcpp::get_logger("OpenArm_v10HW");
  for (int round = 0; round < 3; ++round) {
    openarm_->refresh_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    openarm_->recv_all();

    std::vector<size_t> limp;
    const auto& motors = openarm_->get_arm().get_motors();
    for (size_t i = 0; i < ARM_DOF && i < motors.size(); ++i) {
      if (motors[i].get_status() != 1) limp.push_back(i);
    }
    if (limp.empty()) {
      if (round > 0) {
        RCLCPP_INFO(logger, "all %zu arm motors are enabled", ARM_DOF);
      }
      return;
    }
    if (round == 2) {
      for (size_t i : limp) {
        RCLCPP_ERROR(
            logger,
            "%s did NOT enable: the motor reports status %d. It will report "
            "position and torque like any other joint and hold nothing -- it "
            "moves freely by hand. Check its CAN link and its power.",
            joint_names_[i].c_str(), motors[i].get_status());
      }
      return;
    }
    for (size_t i : limp) {
      int status = motors[i].get_status();
      if (status >= 8) {
        // A latched fault. It has to be cleared before an enable will take.
        RCLCPP_WARN(logger, "%s is in fault state %d; clearing it",
                    joint_names_[i].c_str(), status);
        openarm_->get_arm().clear_error_one(static_cast<int>(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      } else {
        RCLCPP_WARN(logger, "%s did not enable (status %d); asking again",
                    joint_names_[i].c_str(), status);
      }
      openarm_->get_arm().enable_one(static_cast<int>(i));
      // Paced. Seven enables sent back to back is how one gets missed.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    openarm_->recv_all();
  }
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
