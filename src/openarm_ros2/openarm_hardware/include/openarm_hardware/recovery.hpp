// Copyright 2026 -- openarm_hardware
//
// A motor that stops driving mid-run, taken back without a jump (2026-11-18).
//
// On 2026-11-18 10:48:37 left joint2 reported status 1 -> 0 (disabled) in the
// middle of PICK ALL -- no fault code, the way a Damiao motor comes back from a
// brownout (the same minutes the USB hub dropped a keyboard for "EMI?"). The
// driver only logged it; the shoulder hung limp for the rest of the run, 206 to
// 391 mrad off its command. Now, per arm joint:
//
//   down     status 0 (disabled), 9 (under-voltage) or 13 (CAN lost): its
//            position command is its MEASURED position (so the moment it takes
//            an enable it holds where it is -- the integral cleared, nothing
//            stored while it hung), and an enable is sent every kRetryS s, at
//            most kMaxTries times (9 / 13 are cleared first). Any other code
//            (over-current, over-heat, over-voltage, over-load: 8, 10-12, 14) is
//            NOT retried -- the cause has to be looked at -- but the command
//            still follows the measured position so a manual re-enable is safe.
//   easing   enabled again: the command moves from where it took the enable to
//            the controller's command at no more than `rate` rad/s, then the
//            normal law takes over.
//
// Pure: no ROS, no CAN. The driver sends what `action` says and logs the events.
#ifndef OPENARM_HARDWARE__RECOVERY_HPP_
#define OPENARM_HARDWARE__RECOVERY_HPP_

#include <algorithm>
#include <cmath>

namespace openarm_hardware {
namespace recovery {

constexpr int kMaxTries = 25;      // 5 s of enables, one every kRetryS
constexpr double kRetryS = 0.2;
constexpr double kDoneRad = 0.001; // eased back to within this of the command

enum class Action { kNone, kEnable, kClear };

struct State {
  int phase = 0;        // 0 normal, 1 down, 2 easing back
  double q = 0.0;       // the command used while down / easing
  int tries = 0;
  double next_try = 0.0;
  bool gave_up_said = false;
};

struct Out {
  bool overriding = false;  // use q (velocity 0, no integral) instead of the law
  double q = 0.0;
  Action action = Action::kNone;
  bool went_down = false;   // events, for the log
  bool came_back = false;
  bool done = false;
  bool gave_up = false;     // tries used up, or a code that is not retried
};

inline bool retried(int status) { return status == 0 || status == 9 || status == 13; }

// status: the motor's code (1 enabled; -1 not read yet: ignored). now: seconds
// on any steady clock. rate: rad/s while easing back (> 0).
inline Out step(State& s, bool enabled, int status, double q_meas, double q_cmd,
                double now, double dt, double rate) {
  Out o;
  if (!enabled || status == -1 || !std::isfinite(q_meas)) {
    s = State{};
    return o;
  }
  if (status != 1) {
    if (s.phase != 1) {
      s.phase = 1;
      s.tries = 0;
      s.next_try = now;
      s.gave_up_said = false;
      o.went_down = true;
    }
    s.q = q_meas;
    o.overriding = true;
    o.q = q_meas;
    if (retried(status) && s.tries < kMaxTries) {
      if (now >= s.next_try) {
        o.action = status == 0 ? Action::kEnable : Action::kClear;
        ++s.tries;
        s.next_try = now + kRetryS;
      }
    } else if (!s.gave_up_said) {
      s.gave_up_said = true;
      o.gave_up = true;
    }
    return o;
  }
  if (s.phase == 1) {
    s.phase = 2;
    s.q = q_meas;
    o.came_back = true;
  }
  if (s.phase == 2) {
    const double target = std::isfinite(q_cmd) ? q_cmd : s.q;
    const double stepmax = std::max(rate, 1e-3) * std::max(dt, 0.0);
    s.q += std::clamp(target - s.q, -stepmax, stepmax);
    if (std::abs(target - s.q) <= kDoneRad) {
      s.phase = 0;
      o.done = true;
      return o;  // the law from here (its command is within kDoneRad)
    }
    o.overriding = true;
    o.q = s.q;
  }
  return o;
}

}  // namespace recovery
}  // namespace openarm_hardware

#endif  // OPENARM_HARDWARE__RECOVERY_HPP_
