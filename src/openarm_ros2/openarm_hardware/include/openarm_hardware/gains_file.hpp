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

// calibration/motor_gains_<arm>.yaml: the control-law switches and an opt-in
// gains profile, read once when the hardware starts (2026-11-17). No ROS here,
// so the checks are unit-tested (test/test_mit_law.cpp).
//
// The file is all or nothing. Anything wrong in it -- a key nobody knows (a
// typo would otherwise be ignored without a word), a value of the wrong type,
// a value outside the bounds below, the wrong arm -- and none of it is used:
// the driver logs an ERROR and runs the compiled law. The bounds are checked
// for the profile too when the profile is off, so a bad profile is reported
// before anyone turns it on.
//
// The bounds are a ceiling the file cannot lift: KP on joints 1-4 never above
// the compiled 100, the integral caps never above the compiled caps. What a
// file can do is make the wrist stiffer (KP up to 30), the integral faster
// (KI up to 100) and gate it against hunting.

#include <yaml-cpp/yaml.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "openarm_hardware/mit_law.hpp"

namespace openarm_hardware {
namespace gains_file {

constexpr size_t kJoints = 7;

// Bounds (rad, s, N m).
constexpr double kKpMaxJ1to4 = 100.0;   // the compiled value: never raised
constexpr double kKpMaxWrist = 30.0;    // joints 5-7
constexpr double kKdMax = 5.0;          // the motor packs kd into 0..5 anyway
constexpr double kKiMax = 100.0;
constexpr double kGateMax = 0.005;
constexpr double kFreezeRateMin = 0.002;
constexpr double kFreezeRateMax = 0.5;
constexpr double kFreezeHoldMax = 2.0;
constexpr double kReenableRateMin = 0.05;  // rad/s
constexpr double kReenableRateMax = 1.0;
constexpr size_t kMaxFileBytes = 64 * 1024;

// One joint's profile values; a missing key keeps the compiled value.
struct JointProfile {
  std::optional<double> kp, kd, ki, i_cap, gate_in, gate_out;
};

struct Contents {
  // Top level. A missing key keeps the compiled default.
  std::optional<bool> velocity_feedforward;
  std::optional<bool> integral_freeze_while_moving;
  std::optional<double> freeze_rate;
  std::optional<double> freeze_hold;
  std::optional<bool> zero_velocity_when_held;
  std::optional<bool> auto_reenable;
  std::optional<double> reenable_rate;
  bool profile_enabled = false;
  std::array<JointProfile, kJoints> joints;
};

namespace detail {

inline bool as_bool(const YAML::Node& n, const std::string& key, bool& out,
                    std::string& why) {
  try {
    if (!n.IsScalar()) throw YAML::Exception(YAML::Mark(), "not a scalar");
    out = n.as<bool>();
    return true;
  } catch (const std::exception&) {
    why = key + ": not true/false";
    return false;
  }
}

inline bool as_number(const YAML::Node& n, const std::string& key,
                      double& out, std::string& why) {
  try {
    if (!n.IsScalar()) throw YAML::Exception(YAML::Mark(), "not a scalar");
    out = n.as<double>();
  } catch (const std::exception&) {
    why = key + ": not a number";
    return false;
  }
  if (!std::isfinite(out)) {
    why = key + ": not a finite number";
    return false;
  }
  return true;
}

inline bool within(double v, double lo, double hi, const std::string& key,
                   std::string& why) {
  if (v < lo || v > hi) {
    std::ostringstream s;
    s << key << " = " << v << " is outside [" << lo << ", " << hi << "]";
    why = s.str();
    return false;
  }
  return true;
}

}  // namespace detail

// Parse and check a file's text. `arm` is "left" or "right" (an `arm:` key,
// when present, must match it). `compiled` are the compiled laws of joints
// 1-7: the integral caps are bounded by theirs, and a profile's gate pair is
// checked against them where the file gives only one side. True if the whole
// file is acceptable; otherwise `why` says the first thing wrong.
inline bool parse(const std::string& text, const std::string& arm,
                  const std::vector<mit_law::JointLaw>& compiled,
                  Contents& out, std::string& why) {
  out = Contents();
  if (compiled.size() < kJoints) {
    why = "internal: fewer than 7 compiled joint laws";
    return false;
  }
  YAML::Node root;
  try {
    root = YAML::Load(text);
  } catch (const std::exception& ex) {
    why = std::string("not valid YAML: ") + ex.what();
    return false;
  }
  if (!root.IsMap()) {
    why = "the top level is not a mapping";
    return false;
  }
  for (const auto& kv : root) {
    std::string key;
    try {
      key = kv.first.as<std::string>();
    } catch (const std::exception&) {
      why = "a top-level key is not a string";
      return false;
    }
    const YAML::Node& v = kv.second;
    if (key == "arm") {
      std::string a;
      try {
        a = v.as<std::string>();
      } catch (const std::exception&) {
        why = "arm: not a string";
        return false;
      }
      if (a != arm) {
        why = "arm: the file is for the '" + a + "' arm, this is the '" +
              arm + "' arm";
        return false;
      }
    } else if (key == "velocity_feedforward") {
      bool b;
      if (!detail::as_bool(v, key, b, why)) return false;
      out.velocity_feedforward = b;
    } else if (key == "integral_freeze_while_moving") {
      bool b;
      if (!detail::as_bool(v, key, b, why)) return false;
      out.integral_freeze_while_moving = b;
    } else if (key == "zero_velocity_when_held") {
      bool b;
      if (!detail::as_bool(v, key, b, why)) return false;
      out.zero_velocity_when_held = b;
    } else if (key == "auto_reenable") {
      bool b;
      if (!detail::as_bool(v, key, b, why)) return false;
      out.auto_reenable = b;
    } else if (key == "reenable_rate") {
      double d;
      if (!detail::as_number(v, key, d, why)) return false;
      if (!detail::within(d, kReenableRateMin, kReenableRateMax, key, why))
        return false;
      out.reenable_rate = d;
    } else if (key == "profile_enabled") {
      bool b;
      if (!detail::as_bool(v, key, b, why)) return false;
      out.profile_enabled = b;
    } else if (key == "freeze_rate") {
      double d;
      if (!detail::as_number(v, key, d, why)) return false;
      if (!detail::within(d, kFreezeRateMin, kFreezeRateMax, key, why))
        return false;
      out.freeze_rate = d;
    } else if (key == "freeze_hold") {
      double d;
      if (!detail::as_number(v, key, d, why)) return false;
      if (!detail::within(d, 0.0, kFreezeHoldMax, key, why)) return false;
      out.freeze_hold = d;
    } else if (key == "joints") {
      if (v.IsNull()) continue;
      if (!v.IsMap()) {
        why = "joints: not a mapping of joint1..joint7";
        return false;
      }
      for (const auto& jv : v) {
        std::string jname;
        try {
          jname = jv.first.as<std::string>();
        } catch (const std::exception&) {
          why = "joints: a key is not a string";
          return false;
        }
        size_t idx = kJoints;
        for (size_t i = 0; i < kJoints; ++i) {
          if (jname == "joint" + std::to_string(i + 1)) idx = i;
        }
        if (idx == kJoints) {
          why = "joints: unknown joint '" + jname + "' (joint1..joint7)";
          return false;
        }
        const YAML::Node& jn = jv.second;
        if (jn.IsNull()) continue;
        if (!jn.IsMap()) {
          why = "joints." + jname + ": not a mapping";
          return false;
        }
        JointProfile& p = out.joints[idx];
        const double cap = compiled[idx].i_cap;
        const double kp_max = idx < 4 ? kKpMaxJ1to4 : kKpMaxWrist;
        for (const auto& pv : jn) {
          std::string pk;
          try {
            pk = pv.first.as<std::string>();
          } catch (const std::exception&) {
            why = "joints." + jname + ": a key is not a string";
            return false;
          }
          const std::string full = "joints." + jname + "." + pk;
          double d;
          if (pk == "kp" || pk == "kd" || pk == "ki" || pk == "i_cap" ||
              pk == "gate_in" || pk == "gate_out") {
            if (!detail::as_number(pv.second, full, d, why)) return false;
          } else {
            why = full + ": unknown key (kp, kd, ki, i_cap, gate_in, gate_out)";
            return false;
          }
          if (pk == "kp") {
            // > 0: a joint with no spring holds nothing.
            if (d <= 0.0) {
              why = full + " must be > 0";
              return false;
            }
            if (!detail::within(d, 0.0, kp_max, full, why)) return false;
            p.kp = d;
          } else if (pk == "kd") {
            if (!detail::within(d, 0.0, kKdMax, full, why)) return false;
            p.kd = d;
          } else if (pk == "ki") {
            if (!detail::within(d, 0.0, kKiMax, full, why)) return false;
            p.ki = d;
          } else if (pk == "i_cap") {
            if (!detail::within(d, 0.0, cap, full, why)) return false;
            p.i_cap = d;
          } else if (pk == "gate_in") {
            if (!detail::within(d, 0.0, kGateMax, full, why)) return false;
            p.gate_in = d;
          } else {
            if (!detail::within(d, 0.0, kGateMax, full, why)) return false;
            p.gate_out = d;
          }
        }
        const double gi = p.gate_in.value_or(compiled[idx].gate_in);
        const double go = p.gate_out.value_or(compiled[idx].gate_out);
        if (go < gi) {
          std::ostringstream s;
          s << "joints." << jname << ": gate_out " << go << " < gate_in "
            << gi << " (the gate would never open)";
          why = s.str();
          return false;
        }
      }
    } else {
      why = "unknown key '" + key + "'";
      return false;
    }
  }
  return true;
}

enum class Load { kNoFile, kMissing, kRejected, kLoaded };

// Read and check `path`. kNoFile when the path is empty, kMissing when there
// is no such file, kRejected (with `why`) when it cannot be read or fails a
// check, kLoaded when it is accepted.
inline Load load(const std::string& path, const std::string& arm,
                 const std::vector<mit_law::JointLaw>& compiled,
                 Contents& out, std::string& why) {
  out = Contents();
  if (path.empty()) return Load::kNoFile;
  // fopen rather than ifstream: it tells "no such file" (the documented
  // fallback) from "there but unreadable" (an error).
  errno = 0;
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) {
    const int err = errno;
    if (err == ENOENT) {
      why = "no such file";
      return Load::kMissing;
    }
    why = std::string("cannot open it: ") + std::strerror(err);
    return Load::kRejected;
  }
  std::string text;
  char buf[4096];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0 &&
         text.size() <= kMaxFileBytes) {
    text.append(buf, n);
  }
  const bool bad = std::ferror(fp) != 0;
  std::fclose(fp);
  if (bad) {
    why = "cannot read it";  // a directory, an I/O error
    return Load::kRejected;
  }
  if (text.size() > kMaxFileBytes) {
    why = "larger than 64 KiB: not a gains file";
    return Load::kRejected;
  }
  if (!parse(text, arm, compiled, out, why)) {
    out = Contents();
    return Load::kRejected;
  }
  return Load::kLoaded;
}

// Apply accepted contents: the flags always, the profile only when enabled.
inline void apply(const Contents& c, std::vector<mit_law::JointLaw>& laws,
                  mit_law::LawFlags& flags) {
  if (c.velocity_feedforward) flags.velocity_feedforward = *c.velocity_feedforward;
  if (c.integral_freeze_while_moving)
    flags.freeze_while_moving = *c.integral_freeze_while_moving;
  if (c.freeze_rate) flags.freeze_rate = *c.freeze_rate;
  if (c.freeze_hold) flags.freeze_hold = *c.freeze_hold;
  if (c.zero_velocity_when_held)
    flags.zero_velocity_when_held = *c.zero_velocity_when_held;
  if (c.auto_reenable) flags.auto_reenable = *c.auto_reenable;
  if (c.reenable_rate) flags.reenable_rate = *c.reenable_rate;
  if (!c.profile_enabled) return;
  for (size_t i = 0; i < kJoints && i < laws.size(); ++i) {
    const JointProfile& p = c.joints[i];
    if (p.kp) laws[i].kp = *p.kp;
    if (p.kd) laws[i].kd = *p.kd;
    if (p.ki) laws[i].ki = *p.ki;
    if (p.i_cap) laws[i].i_cap = *p.i_cap;
    if (p.gate_in) laws[i].gate_in = *p.gate_in;
    if (p.gate_out) laws[i].gate_out = *p.gate_out;
  }
}

}  // namespace gains_file
}  // namespace openarm_hardware
