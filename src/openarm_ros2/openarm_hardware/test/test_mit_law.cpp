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

// The arm joints' control law (mit_law.hpp) and the gains-file checks
// (gains_file.hpp), 2026-11-17. No hardware, no ROS.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "openarm_hardware/gains_file.hpp"
#include "openarm_hardware/mit_law.hpp"

namespace ml = openarm_hardware::mit_law;
namespace gf = openarm_hardware::gains_file;

namespace {

// The compiled gains in v10_simple_hardware.hpp, joints 1-7.
const double KP[7] = {100, 100, 100, 100, 10, 10, 10};
const double KD[7] = {5.5, 5.0, 1.4, 0.8, 1.4, 1.2, 1.0};
const double KI[7] = {20, 20, 20, 20, 10, 10, 10};
const double CAP[7] = {8, 8, 6, 6, 1.5, 1.5, 1.5};
const double DEADBAND = 0.0005;
const double DT = 0.01;

std::vector<ml::JointLaw> compiled() {
  std::vector<ml::JointLaw> laws(7);
  for (int i = 0; i < 7; ++i) {
    laws[i].kp = KP[i];
    laws[i].kd = KD[i];
    laws[i].ki = KI[i];
    laws[i].i_cap = std::min(20.0, CAP[i]);
    laws[i].deadband = DEADBAND;
    laws[i].gate_in = DEADBAND;
    laws[i].gate_out = DEADBAND;
    laws[i].vff_max = 4.0;
  }
  return laws;
}

// write() as it was before 2026-11-17, verbatim in effect.
struct Old {
  double kp, kd, q, dq, tau;
};
Old old_step(int i, double q, double q_cmd, double v_cmd, double tau_cmd,
             double gravity, bool kdl_ok, double dt, double& integral) {
  double error = q_cmd - q;
  if (std::abs(error) > DEADBAND) integral += error * dt;
  const double i_cap = std::min(20.0, CAP[i]);
  double i_torque = KI[i] * integral;
  i_torque = std::clamp(i_torque, -i_cap, i_cap);
  if (KI[i] > 0.0) {
    integral = std::clamp(integral, -i_cap / KI[i], i_cap / KI[i]);
  }
  double ff = tau_cmd + i_torque;
  if (kdl_ok) ff += gravity;
  return {KP[i], KD[i], q_cmd, v_cmd, ff};
}

}  // namespace

// With both new switches off, the law is the old one, bit for bit.
TEST(MitLaw, SwitchesOffIsTheOldLawExactly) {
  const auto laws = compiled();
  ml::LawFlags off;
  off.velocity_feedforward = false;
  off.freeze_while_moving = false;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int i = 0; i < 7; ++i) {
    ml::JointMemory m;
    double integral = 0.0;
    double q_cmd_prev = 0.0;
    for (int k = 0; k < 20000; ++k) {
      // Errors from inside the deadband to well past the cap.
      const double scale = (k % 3 == 0) ? 0.0004 : (k % 3 == 1 ? 0.004 : 0.4);
      const double q_cmd = 0.3 * std::sin(k * 0.003) + 0.01 * u(rng);
      const double q = q_cmd + scale * u(rng);
      const double tau = 0.2 * u(rng);
      const double g = 3.0 * u(rng);
      const bool kdl_ok = (k % 17) != 0;
      const auto c = ml::step(laws[i], off, q, q_cmd, q_cmd_prev, 0.0, tau,
                              kdl_ok ? g : 0.0, DT, m);
      const auto o = old_step(i, q, q_cmd, 0.0, tau, g, kdl_ok, DT, integral);
      ASSERT_EQ(c.kp, o.kp);
      ASSERT_EQ(c.kd, o.kd);
      ASSERT_EQ(c.q, o.q);
      ASSERT_EQ(c.dq, o.dq);
      ASSERT_EQ(c.tau, o.tau) << "joint " << i + 1 << " step " << k;
      ASSERT_EQ(m.integral, integral);
      q_cmd_prev = c.q;
    }
  }
}

// With the defaults (both on) and a command that stands still, also the old
// law exactly: the switches only act while the command moves.
TEST(MitLaw, DefaultsWithAStillCommandIsTheOldLaw) {
  const auto laws = compiled();
  const ml::LawFlags on;
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int i = 0; i < 7; ++i) {
    ml::JointMemory m;
    double integral = 0.0;
    const double q_cmd = 0.25;
    for (int k = 0; k < 5000; ++k) {
      const double q = q_cmd + 0.01 * u(rng);
      const auto c = ml::step(laws[i], on, q, q_cmd, q_cmd, 0.0, 0.0, 1.0,
                              DT, m);
      const auto o = old_step(i, q, q_cmd, 0.0, 0.0, 1.0, true, DT, integral);
      ASSERT_EQ(c.tau, o.tau);
      ASSERT_EQ(c.dq, 0.0);
    }
  }
}

// A ramping command (a trajectory) holds the integral; 0.15 s after it stops
// the integral runs again.
TEST(MitLaw, RampFreezesStillIntegrates) {
  const auto laws = compiled();
  const ml::LawFlags on;
  ml::JointMemory m;
  m.integral = 0.01;  // a gravity trim carried over from before
  double q = 0.0, prev = 0.0;
  // 1.5 s ramp at 0.5 rad/s, the arm lagging 20 mrad behind.
  for (int k = 1; k <= 150; ++k) {
    const double cmd = 0.005 * k;
    q = cmd - 0.02;
    const auto c = ml::step(laws[6], on, q, cmd, prev, 0.5, 0.0, 0.0, DT, m);
    EXPECT_TRUE(c.moving);
    EXPECT_FALSE(c.integrating);
    EXPECT_NEAR(m.integral, 0.01, 1e-12) << "step " << k;
    prev = c.q;
  }
  // Stopped 20 mrad short: frozen for 0.15 s, then integrating.
  int first = -1;
  for (int k = 1; k <= 40; ++k) {
    const auto c = ml::step(laws[6], on, q, prev, prev, 0.0, 0.0, 0.0, DT, m);
    if (c.integrating && first < 0) first = k;
  }
  // still_time reaches 0.15 s on the 15th or 16th still cycle (float sums).
  EXPECT_GE(first, 15);
  EXPECT_LE(first, 16);
  EXPECT_GT(m.integral, 0.01);
}

// Slow commands (below freeze_rate) integrate as before.
TEST(MitLaw, SlowCommandIsNotMoving) {
  const auto laws = compiled();
  const ml::LawFlags on;
  ml::JointMemory m;
  double prev = 0.0;
  for (int k = 1; k <= 100; ++k) {
    const double cmd = 0.00005 * k;  // 0.005 rad/s
    const auto c = ml::step(laws[0], on, cmd - 0.003, cmd, prev, 0.005, 0.0,
                            0.0, DT, m);
    EXPECT_FALSE(c.moving);
    prev = c.q;
  }
  EXPECT_GT(m.integral, 0.0);
}

TEST(MitLaw, FreezeOffIntegratesWhileMoving) {
  const auto laws = compiled();
  ml::LawFlags f;
  f.freeze_while_moving = false;
  ml::JointMemory m;
  double prev = 0.0;
  for (int k = 1; k <= 50; ++k) {
    const double cmd = 0.005 * k;
    const auto c = ml::step(laws[0], f, cmd - 0.02, cmd, prev, 0.5, 0.0, 0.0,
                            DT, m);
    EXPECT_TRUE(c.integrating);
    prev = c.q;
  }
  EXPECT_NEAR(m.integral, 50 * 0.02 * DT, 1e-12);
}

TEST(MitLaw, VelocityFeedForwardNaNAndClamp) {
  const auto laws = compiled();
  const ml::LawFlags on;
  ml::JointMemory m;
  // passed through while the command moves
  auto c = ml::step(laws[0], on, 0.0, 0.01, 0.0, 1.0, 0.0, 0.0, DT, m);
  EXPECT_DOUBLE_EQ(c.dq, 1.0);
  // NaN and infinities count as 0
  c = ml::step(laws[0], on, 0.0, 0.02,
               0.01, std::numeric_limits<double>::quiet_NaN(), 0, 0, DT, m);
  EXPECT_EQ(c.dq, 0.0);
  c = ml::step(laws[0], on, 0.0, 0.03,
               0.02, std::numeric_limits<double>::infinity(), 0, 0, DT, m);
  EXPECT_EQ(c.dq, 0.0);
  // clamped to vff_max
  c = ml::step(laws[0], on, 0.0, 0.10, 0.03, 9.0, 0, 0, DT, m);
  EXPECT_DOUBLE_EQ(c.dq, 4.0);
  c = ml::step(laws[0], on, 0.0, 0.0, 0.10, -9.0, 0, 0, DT, m);
  EXPECT_DOUBLE_EQ(c.dq, -4.0);
  // off: always 0
  ml::LawFlags off = on;
  off.velocity_feedforward = false;
  c = ml::step(laws[0], off, 0.0, 0.01, 0.0, 1.0, 0, 0, DT, m);
  EXPECT_EQ(c.dq, 0.0);
}

// A finished trajectory held by the JTC with its last velocity: dropped after
// kHeldCycles, and the integral is then free to run.
TEST(MitLaw, HeldPointDropsItsVelocity) {
  const auto laws = compiled();
  const ml::LawFlags on;
  ml::JointMemory m;
  auto c = ml::step(laws[6], on, 0.49, 0.5, 0.495, 0.3, 0, 0, DT, m);
  EXPECT_DOUBLE_EQ(c.dq, 0.3);  // still moving: kept
  c = ml::step(laws[6], on, 0.49, 0.5, 0.5, 0.3, 0, 0, DT, m);
  EXPECT_DOUBLE_EQ(c.dq, 0.3);  // one repeat: kept
  c = ml::step(laws[6], on, 0.49, 0.5, 0.5, 0.3, 0, 0, DT, m);
  EXPECT_EQ(c.dq, 0.0);         // two: a held point
  int integrating = 0;
  for (int k = 0; k < 30; ++k) {
    c = ml::step(laws[6], on, 0.49, 0.5, 0.5, 0.3, 0, 0, DT, m);
    EXPECT_EQ(c.dq, 0.0);
    integrating += c.integrating ? 1 : 0;
  }
  EXPECT_GT(integrating, 10);
  // The guard off: the velocity stays, and keeps the integral frozen.
  ml::LawFlags noguard = on;
  noguard.zero_velocity_when_held = false;
  ml::JointMemory m2;
  for (int k = 0; k < 30; ++k) {
    c = ml::step(laws[6], noguard, 0.49, 0.5, 0.5, 0.3, 0, 0, DT, m2);
    EXPECT_DOUBLE_EQ(c.dq, 0.3);
    EXPECT_FALSE(c.integrating);
  }
}

TEST(MitLaw, NonFiniteCommandsNeverReachTheMotor) {
  const auto laws = compiled();
  const ml::LawFlags on;
  ml::JointMemory m;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto c = ml::step(laws[0], on, 0.1, nan, 0.2, 0.0, nan, nan, DT, m);
  EXPECT_DOUBLE_EQ(c.q, 0.2);  // the last finite command
  EXPECT_TRUE(std::isfinite(c.tau));
  EXPECT_TRUE(std::isfinite(m.integral));
  c = ml::step(laws[0], on, 0.1, nan, nan, 0.0, 0.0, 0.0, DT, m);
  EXPECT_DOUBLE_EQ(c.q, 0.1);  // none: where the joint is
}

TEST(MitLaw, IntegralClampedAsBefore) {
  const auto laws = compiled();
  const ml::LawFlags on;
  ml::JointMemory m;
  for (int k = 0; k < 2000; ++k) {
    const auto c = ml::step(laws[4], on, 0.0, 0.5, 0.5, 0, 0, 0, DT, m);
    EXPECT_LE(std::abs(c.i_torque), 1.5 + 1e-12);
  }
  EXPECT_NEAR(m.integral, 1.5 / 10.0, 1e-12);
  EXPECT_NEAR(ml::step(laws[4], on, 0.0, 0.5, 0.5, 0, 0, 0, DT, m).i_torque,
              1.5, 1e-12);
}

// gate_in 1 mrad / gate_out 2 mrad: stops at <= 1, starts again at >= 2,
// holds its state in between.
TEST(MitLaw, GateHysteresis) {
  auto law = compiled()[0];
  law.gate_in = 0.001;
  law.gate_out = 0.002;
  const ml::LawFlags on;
  ml::JointMemory m;
  auto at = [&](double e) {
    return ml::step(law, on, 0.0, e, e, 0, 0, 0, DT, m).integrating;
  };
  EXPECT_TRUE(at(0.003));    // outside: integrating
  EXPECT_TRUE(at(0.0015));   // between, coming from outside: still on
  EXPECT_FALSE(at(0.0009));  // inside gate_in: stops
  EXPECT_FALSE(at(0.0015));  // between, coming from inside: stays off
  EXPECT_FALSE(at(0.0019));
  EXPECT_TRUE(at(0.002));    // at gate_out: on again
  EXPECT_TRUE(at(-0.0015));  // symmetric in sign
  EXPECT_FALSE(at(-0.001));
}

// ---------------------------------------------------------------- gains file

namespace {

bool parse(const std::string& text, gf::Contents& c, std::string& why,
           const std::string& arm = "left") {
  return gf::parse(text, arm, compiled(), c, why);
}

const char* kGood = R"(
arm: left
velocity_feedforward: true
integral_freeze_while_moving: true
freeze_rate: 0.01
freeze_hold: 0.15
profile_enabled: false
joints:
  joint1: {ki: 60.0, gate_in: 0.001, gate_out: 0.002}
  joint2: {ki: 60.0, gate_in: 0.001, gate_out: 0.002}
  joint3: {ki: 60.0, gate_in: 0.001, gate_out: 0.002}
  joint4: {ki: 60.0, gate_in: 0.001, gate_out: 0.002}
  joint5: {kp: 10.0, ki: 20.0, gate_in: 0.001, gate_out: 0.003}
  joint6: {kp: 10.0, ki: 20.0, gate_in: 0.001, gate_out: 0.003}
  joint7: {kp: 10.0, ki: 20.0, gate_in: 0.001, gate_out: 0.003}
)";

std::string with(const std::string& from, const std::string& to) {
  std::string s = kGood;
  const auto at = s.find(from);
  EXPECT_NE(at, std::string::npos) << from;
  if (at != std::string::npos) s.replace(at, from.size(), to);
  return s;
}

}  // namespace

TEST(GainsFile, GoodFileFlagsAlwaysProfileOnlyWhenEnabled) {
  gf::Contents c;
  std::string why;
  ASSERT_TRUE(parse(kGood, c, why)) << why;
  // profile off: flags applied, gains untouched
  auto laws = compiled();
  ml::LawFlags flags;
  gf::apply(c, laws, flags);
  EXPECT_TRUE(flags.velocity_feedforward);
  EXPECT_DOUBLE_EQ(laws[0].ki, 20.0);
  EXPECT_DOUBLE_EQ(laws[0].gate_in, DEADBAND);
  // switches off in the file: applied even with the profile off
  ASSERT_TRUE(parse(with("velocity_feedforward: true",
                         "velocity_feedforward: false"), c, why)) << why;
  flags = ml::LawFlags();
  gf::apply(c, laws, flags);
  EXPECT_FALSE(flags.velocity_feedforward);
  EXPECT_TRUE(flags.freeze_while_moving);
  ASSERT_TRUE(parse(with("integral_freeze_while_moving: true",
                         "integral_freeze_while_moving: false"), c, why));
  flags = ml::LawFlags();
  gf::apply(c, laws, flags);
  EXPECT_FALSE(flags.freeze_while_moving);
  // profile on: the per-joint values replace the compiled ones, the rest stay
  ASSERT_TRUE(parse(with("profile_enabled: false", "profile_enabled: true"),
                    c, why)) << why;
  laws = compiled();
  flags = ml::LawFlags();
  gf::apply(c, laws, flags);
  EXPECT_DOUBLE_EQ(laws[0].ki, 60.0);
  EXPECT_DOUBLE_EQ(laws[0].kp, 100.0);  // not in the file: compiled
  EXPECT_DOUBLE_EQ(laws[0].kd, 5.5);    // not in the file: compiled
  EXPECT_DOUBLE_EQ(laws[0].gate_out, 0.002);
  EXPECT_DOUBLE_EQ(laws[6].ki, 20.0);
  EXPECT_DOUBLE_EQ(laws[6].gate_out, 0.003);
  EXPECT_DOUBLE_EQ(laws[6].i_cap, 1.5);
}

TEST(GainsFile, OutOfBoundsRejectsTheWholeFile) {
  const std::vector<std::pair<std::string, std::string>> bad = {
      {"joint1: {ki: 60.0", "joint1: {kp: 101.0, ki: 60.0"},   // KP j1-4 > 100
      {"joint5: {kp: 10.0", "joint5: {kp: 31.0"},              // wrist KP > 30
      {"joint5: {kp: 10.0", "joint5: {kp: 0.0"},               // no spring
      {"joint2: {ki: 60.0", "joint2: {ki: 60.0, kd: 5.1"},     // KD > 5
      {"joint3: {ki: 60.0", "joint3: {ki: 101.0"},             // KI > 100
      {"joint3: {ki: 60.0", "joint3: {ki: -1.0"},
      {"joint1: {ki: 60.0", "joint1: {ki: 60.0, i_cap: 8.5"},  // cap raised
      {"joint7: {kp: 10.0", "joint7: {kp: 10.0, i_cap: 1.6"},
      {"gate_in: 0.001, gate_out: 0.002}\n  joint2",
       "gate_in: 0.001, gate_out: 0.006}\n  joint2"},           // gate > 5 mrad
      {"gate_in: 0.001, gate_out: 0.002}\n  joint2",
       "gate_in: 0.003, gate_out: 0.002}\n  joint2"},           // out < in
      {"joint6: {kp: 10.0, ki: 20.0, gate_in: 0.001, gate_out: 0.003}",
       "joint6: {gate_in: 0.001}"},                // out (compiled 0.5) < in
      {"freeze_rate: 0.01", "freeze_rate: 0.001"},             // < 0.002
      {"freeze_hold: 0.15", "freeze_hold: 5.0"},
      {"freeze_rate: 0.01", "freeze_rate: fast"},              // not a number
      {"freeze_rate: 0.01", "freeze_rate: .nan"},
      {"velocity_feedforward: true", "velocity_feedforward: 3"},  // not bool
      {"velocity_feedforward: true", "velocity_feedfoward: false"},  // typo
      {"joint1: {ki", "joint1: {kii"},                          // typo
      {"joint7:", "joint8:"},
      {"arm: left", "arm: right"},                              // wrong arm
      {"joints:\n", "joints: 3\n#"},
  };
  // The profile is off in every one of these: still rejected, before anyone
  // turns it on.
  for (const auto& [from, to] : bad) {
    gf::Contents c;
    std::string why;
    EXPECT_FALSE(parse(with(from, to), c, why)) << "accepted: " << to;
    EXPECT_FALSE(why.empty()) << to;
  }
  gf::Contents c;
  std::string why;
  EXPECT_FALSE(parse("velocity_feedforward: [true", c, why));   // bad YAML
  EXPECT_FALSE(parse("- a\n- b\n", c, why));                     // not a map
  EXPECT_FALSE(parse(kGood, c, why, "right"));                   // other arm
  // The file's kd bound lies below the compiled j1 KD (5.5): the compiled
  // value stays when the file does not name kd, as checked above.
}

TEST(GainsFile, EmptyAndMinimalFiles) {
  gf::Contents c;
  std::string why;
  // an empty document is not a mapping
  EXPECT_FALSE(parse("", c, why));
  // only the switches, no arm key, no joints
  ASSERT_TRUE(parse("velocity_feedforward: false\n", c, why)) << why;
  auto laws = compiled();
  ml::LawFlags flags;
  gf::apply(c, laws, flags);
  EXPECT_FALSE(flags.velocity_feedforward);
  EXPECT_TRUE(flags.freeze_while_moving);
  EXPECT_TRUE(flags.zero_velocity_when_held);
}

TEST(GainsFile, LoadMissingEmptyAndUnreadable) {
  gf::Contents c;
  std::string why;
  EXPECT_EQ(gf::load("", "left", compiled(), c, why), gf::Load::kNoFile);
  EXPECT_EQ(gf::load("/nonexistent/motor_gains_left.yaml", "left",
                     compiled(), c, why),
            gf::Load::kMissing);
  EXPECT_EQ(gf::load("/tmp", "left", compiled(), c, why),
            gf::Load::kRejected);  // a directory
  char name[] = "/tmp/test_mit_law_XXXXXX";
  const int fd = mkstemp(name);
  ASSERT_GE(fd, 0);
  close(fd);
  {
    std::ofstream f(name);
    f << with("velocity_feedforward: true", "velocity_feedforward: false");
  }
  EXPECT_EQ(gf::load(name, "left", compiled(), c, why), gf::Load::kLoaded)
      << why;
  EXPECT_FALSE(*c.velocity_feedforward);
  {
    std::ofstream f(name);
    f << with("joint5: {kp: 10.0", "joint5: {kp: 31.0");
  }
  EXPECT_EQ(gf::load(name, "left", compiled(), c, why), gf::Load::kRejected);
  EXPECT_FALSE(c.velocity_feedforward.has_value());  // nothing kept
  std::remove(name);
}

// The files in calibration/ (whatever their switches say) are acceptable:
// `colcon test` then also checks an edited file before the robot reads it.
TEST(GainsFile, ShippedFilesAreValid) {
#ifdef OPENARM_CALIBRATION_DIR
  for (const std::string arm : {"left", "right"}) {
    const std::string path = std::string(OPENARM_CALIBRATION_DIR) +
                             "/motor_gains_" + arm + ".yaml";
    std::FILE* fp = std::fopen(path.c_str(), "r");
    if (fp == nullptr) {
      GTEST_SKIP() << "no " << path;
    }
    std::fclose(fp);
    gf::Contents c;
    std::string why;
    EXPECT_EQ(gf::load(path, arm, compiled(), c, why), gf::Load::kLoaded)
        << path << ": " << why;
  }
#else
  GTEST_SKIP() << "OPENARM_CALIBRATION_DIR not set";
#endif
}

// ---- recovery.hpp (2026-11-18): a motor that stops driving ------------------
#include "openarm_hardware/recovery.hpp"
namespace rc = openarm_hardware::recovery;

TEST(Recovery, DropHoldsMeasuredEnablesThenEasesBack) {
  rc::State s;
  // enabled and following: nothing
  rc::Out o = rc::step(s, true, 1, 0.50, 0.50, 0.0, 0.001, 0.25);
  EXPECT_FALSE(o.overriding);
  // drops out (status 0) and sags to 0.30 while the command stays 0.50
  o = rc::step(s, true, 0, 0.45, 0.50, 1.0, 0.001, 0.25);
  EXPECT_TRUE(o.went_down);
  EXPECT_TRUE(o.overriding);
  EXPECT_DOUBLE_EQ(o.q, 0.45);                      // the measured, never the command
  EXPECT_EQ(o.action, rc::Action::kEnable);
  o = rc::step(s, true, 0, 0.40, 0.50, 1.1, 0.001, 0.25);
  EXPECT_EQ(o.action, rc::Action::kNone);           // paced: kRetryS
  o = rc::step(s, true, 0, 0.30, 0.50, 1.21, 0.001, 0.25);
  EXPECT_EQ(o.action, rc::Action::kEnable);
  EXPECT_DOUBLE_EQ(o.q, 0.30);
  // enabled again at 0.30: starts there, eases up at <= 0.25 rad/s
  o = rc::step(s, true, 1, 0.30, 0.50, 1.3, 0.001, 0.25);
  EXPECT_TRUE(o.came_back);
  EXPECT_TRUE(o.overriding);
  EXPECT_NEAR(o.q, 0.30025, 1e-9);
  double t = 1.3, q = o.q, worst_step = 0.0;
  int cycles = 0;
  while (o.overriding && cycles < 100000) {
    t += 0.001;
    o = rc::step(s, true, 1, q, 0.50, t, 0.001, 0.25);
    if (o.overriding) {
      worst_step = std::max(worst_step, std::abs(o.q - q));
      q = o.q;
    }
    ++cycles;
  }
  EXPECT_TRUE(o.done);
  EXPECT_LE(worst_step, 0.25 * 0.001 + 1e-12);
  EXPECT_NEAR(cycles * 0.001, 0.2 / 0.25, 0.02);    // 0.2 rad at 0.25 rad/s: ~0.8 s
  o = rc::step(s, true, 1, 0.50, 0.50, t + 0.001, 0.001, 0.25);
  EXPECT_FALSE(o.overriding);
}

TEST(Recovery, FaultsNotRetriedUnderVoltageCleared) {
  rc::State s;
  rc::Out o = rc::step(s, true, 10, 0.1, 0.2, 0.0, 0.001, 0.25);   // over-current
  EXPECT_TRUE(o.overriding);
  EXPECT_EQ(o.action, rc::Action::kNone);
  EXPECT_TRUE(o.gave_up);
  o = rc::step(s, true, 10, 0.1, 0.2, 0.5, 0.001, 0.25);
  EXPECT_FALSE(o.gave_up);                                          // said once
  rc::State u;
  o = rc::step(u, true, 9, 0.1, 0.2, 0.0, 0.001, 0.25);            // under-voltage
  EXPECT_EQ(o.action, rc::Action::kClear);
  rc::State n;
  o = rc::step(n, true, 13, 0.1, 0.2, 0.0, 0.001, 0.25);           // CAN lost
  EXPECT_EQ(o.action, rc::Action::kClear);
}

TEST(Recovery, GivesUpAfterMaxTriesAndOffDoesNothing) {
  rc::State s;
  int enables = 0;
  bool gave = false;
  for (int k = 0; k < 400; ++k) {
    rc::Out o = rc::step(s, true, 0, 0.1, 0.2, k * 0.05, 0.05, 0.25);
    enables += o.action == rc::Action::kEnable;
    gave = gave || o.gave_up;
  }
  EXPECT_EQ(enables, rc::kMaxTries);
  EXPECT_TRUE(gave);
  rc::State off;
  rc::Out o = rc::step(off, false, 0, 0.1, 0.2, 0.0, 0.001, 0.25);
  EXPECT_FALSE(o.overriding);
  EXPECT_EQ(o.action, rc::Action::kNone);
  rc::State unknown;
  o = rc::step(unknown, true, -1, 0.1, 0.2, 0.0, 0.001, 0.25);
  EXPECT_FALSE(o.overriding);
  rc::State nan_cmd;
  rc::step(nan_cmd, true, 0, 0.1, 0.2, 0.0, 0.001, 0.25);
  o = rc::step(nan_cmd, true, 1, 0.1, std::nan(""), 0.1, 0.001, 0.25);
  EXPECT_TRUE(std::isfinite(o.q) || o.done);
}

TEST(GainsFile, ReenableKeys) {
  const char* name = "/tmp/claude-1000/test_gains_reenable.yaml";
  {
    std::FILE* f = std::fopen(name, "w");
    ASSERT_NE(f, nullptr);
    std::fputs("arm: left\nauto_reenable: false\nreenable_rate: 0.5\n", f);
    std::fclose(f);
  }
  gf::Contents c;
  std::string why;
  ASSERT_EQ(gf::load(name, "left", compiled(), c, why), gf::Load::kLoaded) << why;
  std::vector<openarm_hardware::mit_law::JointLaw> laws = compiled();
  openarm_hardware::mit_law::LawFlags flags;
  EXPECT_TRUE(flags.auto_reenable);
  gf::apply(c, laws, flags);
  EXPECT_FALSE(flags.auto_reenable);
  EXPECT_DOUBLE_EQ(flags.reenable_rate, 0.5);
  {
    std::FILE* f = std::fopen(name, "w");
    std::fputs("arm: left\nreenable_rate: 3.0\n", f);   // above the 1.0 bound
    std::fclose(f);
  }
  gf::Contents c2;
  EXPECT_EQ(gf::load(name, "left", compiled(), c2, why), gf::Load::kRejected);
  std::remove(name);
}
