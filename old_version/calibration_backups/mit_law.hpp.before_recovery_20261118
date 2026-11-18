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

#pragma once

// The host side of one arm joint's control law, as v10_simple_hardware.cpp
// write() runs it every cycle (100 Hz). Pure: no ROS, no CAN, no allocation,
// so it can be unit-tested (test/test_mit_law.cpp) and mirrored in an offline
// simulation exactly.
//
// The motor runs its own MIT loop on what this returns:
//   tau = kp (q - q_meas) + kd (dq - qdot_meas) + tau_ff
// The host adds an integral, gravity (KDL) and the effort command to tau_ff.
//
// 2026-11-17, three changes, each with an off switch (calibration/
// motor_gains_<arm>.yaml), measured and modelled in the precision review of
// the 2026-11-16 17:21-17:40 run:
//
// 1. Velocity feed-forward. Until now dq was always 0 (vel_commands_ was never
//    written), so the kd term braked every move and the arm lagged the
//    trajectory by (kd/kp) x speed: about 190 mrad on joint 7 during the drop
//    move. The JTC now writes its velocity, and it goes to the motor as dq.
//    Free motion is driven by the same torque as before (kp x lag was about
//    kd x v), only without the lag. A blocked joint gets kd x v more only while
//    the trajectory still moves; once it ends v = 0 and the peak is kp x error
//    as before.
// 2. The integral holds while the command moves. It used to integrate the
//    lag during every move and release it afterwards as overshoot past the
//    target: about 10 mrad per rad moved on joints 1-2 and 100-150 mrad on the
//    wrist (joint 7 was 69 mrad past the drop pose 1 s after the move,
//    17:36:02, and the next plan's start check failed). It now integrates
//    only once the command has been still for freeze_hold. Never reset: it
//    carries the gravity-model trim from one move to the next.
// 3. A hysteresis gate on the integral (gate_in/gate_out), for a faster
//    integral without hunting against stiction: it stops at |e| <= gate_in
//    and starts again only at |e| >= gate_out. The defaults (gate_in =
//    gate_out = deadband = 0.5 mrad) reproduce the old deadband exactly.
//
// Plus a guard of my own (2026-11-17): a velocity that comes with a position
// command that stands exactly still is dropped. The JTC (humble 2.53) holds
// the last point of a finished trajectory velocity and all, when that velocity
// is not 0 (allow_nonzero_velocity_at_trajectory_end), and a held dq would push
// the joint off its target by kd x dq / kp for good, and keep the integral
// frozen. A real trajectory changes the position command every cycle.

#include <algorithm>
#include <cmath>

namespace openarm_hardware {
namespace mit_law {

// One joint's gains. Units: rad, s, N m.
struct JointLaw {
  double kp = 0.0;           // motor spring, N m/rad
  double kd = 0.0;           // motor damper, N m s/rad
  double ki = 0.0;           // host integral, N m/(rad s)
  double i_cap = 0.0;        // cap on |integral torque|, N m (and on the store)
  double deadband = 0.0005;  // no integration at |e| <= deadband, rad
  double gate_in = 0.0005;   // integration stops at |e| <= gate_in, rad
  double gate_out = 0.0005;  // ... and starts again at |e| >= gate_out, rad
  double vff_max = 0.0;      // |velocity feed-forward| cap, rad/s
};

// Flags shared by the joints of one arm.
struct LawFlags {
  bool velocity_feedforward = true;
  bool freeze_while_moving = true;
  double freeze_rate = 0.01;   // the command moves above this, rad/s
  double freeze_hold = 0.15;   // still this long before integrating again, s
  bool zero_velocity_when_held = true;
};

// A position command that repeats exactly this many cycles is a held point.
constexpr int kHeldCycles = 2;

// What one joint remembers from cycle to cycle.
struct JointMemory {
  double integral = 0.0;      // integral of e dt, rad s. Never reset.
  double still_time = 1e9;    // s the command has been still (starts "long")
  bool gated = false;         // inside the gate: not integrating
  int held_cycles = 0;        // cycles the position command repeated exactly
};

// What goes to the motor (the MITParam fields, in their order), plus what the
// driver exports or logs about it.
struct Command {
  double kp = 0.0;
  double kd = 0.0;
  double q = 0.0;
  double dq = 0.0;
  double tau = 0.0;
  double i_torque = 0.0;      // the integral's share of tau, N m
  bool integrating = false;
  bool moving = false;
};

// One cycle of one joint.
//   q           measured position
//   q_cmd       position command this cycle; q_cmd_prev the Command::q of
//               the cycle before
//   v_cmd       velocity command (NaN or anything non-finite counts as 0)
//   tau_cmd     effort command
//   gravity     gravity-model torque (0 when the model failed this cycle)
//   dt          cycle time, s (> 0)
inline Command step(const JointLaw& law, const LawFlags& flags, double q,
                    double q_cmd, double q_cmd_prev, double v_cmd,
                    double tau_cmd, double gravity, double dt,
                    JointMemory& m) {
  // Nothing non-finite reaches the motor: openarm_can's packer clamps with
  // std::min/std::max, which turn a NaN into the bottom of the range --
  // -12.5 rad, -vMax (45 rad/s on a DM8009), -tMax. A NaN position command
  // keeps the last finite one (the driver passes back Command::q as
  // q_cmd_prev), or the measured position if there never was one.
  if (!std::isfinite(q_cmd)) {
    q_cmd = std::isfinite(q_cmd_prev) ? q_cmd_prev : q;
  }
  if (!std::isfinite(tau_cmd)) tau_cmd = 0.0;
  if (!std::isfinite(gravity)) gravity = 0.0;

  // Velocity feed-forward, bounded. Off: 0, the old behaviour.
  double v = 0.0;
  if (flags.velocity_feedforward && std::isfinite(v_cmd)) {
    v = std::clamp(v_cmd, -law.vff_max, law.vff_max);
  }
  // A held point keeps no velocity (see the top).
  m.held_cycles = (q_cmd == q_cmd_prev) ? std::min(m.held_cycles + 1, 1 << 20)
                                        : 0;
  if (flags.zero_velocity_when_held && m.held_cycles >= kHeldCycles) {
    v = 0.0;
  }

  // Is the command moving? A NaN rate counts as moving (holds the integral).
  const double rate = std::abs(q_cmd - q_cmd_prev) / dt;
  const bool moving = !(rate <= flags.freeze_rate) ||
                      std::abs(v) > flags.freeze_rate;
  if (moving) {
    m.still_time = 0.0;
  } else {
    m.still_time += dt;
  }

  // Integral, gated, deadbanded, frozen while moving.
  const double e = q_cmd - q;
  const double ae = std::abs(e);
  if (ae <= law.gate_in) {
    m.gated = true;
  } else if (ae >= law.gate_out) {
    m.gated = false;
  }
  const bool may = !flags.freeze_while_moving ||
                   m.still_time >= flags.freeze_hold;
  const bool integrating = std::isfinite(e) && may && !m.gated &&
                           ae > law.deadband;
  if (integrating) {
    m.integral += e * dt;
  }
  // Clamped exactly as before: the torque, then the store, so it cannot wind
  // up past the cap.
  const double i_torque =
      std::clamp(law.ki * m.integral, -law.i_cap, law.i_cap);
  if (law.ki > 0.0) {
    m.integral = std::clamp(m.integral, -law.i_cap / law.ki,
                            law.i_cap / law.ki);
  }

  Command c;
  c.kp = law.kp;
  c.kd = law.kd;
  c.q = q_cmd;
  c.dq = v;
  c.tau = tau_cmd + i_torque + gravity;
  c.i_torque = i_torque;
  c.integrating = integrating;
  c.moving = moving;
  return c;
}

}  // namespace mit_law
}  // namespace openarm_hardware
