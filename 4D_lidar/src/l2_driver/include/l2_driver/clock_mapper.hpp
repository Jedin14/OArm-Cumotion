// Copyright 2026. BSD-3-Clause.
//
// ClockMapper: maps timestamps of a device clock (the L2's internal clock, which
// may start at 1970, be in the future, or drift) onto the host's CLOCK_MONOTONIC.
//
// Every packet gives one observation   d = host_arrival - device_stamp
//                                        = true_offset + transport_latency.
// Latency is always >= its physical minimum, and bursty delivery (the L2 / SDK
// "1000 packets at once, then 100 ms of silence" behaviour) only ever makes it
// *larger*. So the offset is the LOWER ENVELOPE of d, not its mean. Bursts and
// stalls are therefore ignored by construction; they never reach the stamps.
//
//  * The estimate follows any sample that is lower (a faster packet was seen).
//  * Without new evidence it may only rise at max_drift_rate (oscillator drift).
//  * After lock, a sample implausibly far below the estimate is quarantined and
//    only accepted if outlier_confirm consecutive samples agree (corrupt stamp
//    protection).
//  * If every sample stays far above the estimate for lag_window, the latency
//    floor really moved up, so the estimate is re-anchored to the lowest of them.
//
// Feed it host CLOCK_MONOTONIC_RAW: it never steps and is not slewed by NTP
// (chrony / RTC-less 1970 boots only step or slew CLOCK_REALTIME), so the
// estimator only sees the true oscillator drift. Convert to wall time only at
// publish time.
//
// The raw envelope has a small sawtooth (it falls instantly, rises slowly). The
// value used for mapping is the output of a critically damped 2nd-order tracker
// on it (time constant smoothing_time): consecutive stamps keep their exact
// device spacing, and a linearly drifting offset is followed without lag (the
// tracker's rate state is the measured oscillator drift). Confirmed corrections
// (quarantine accepted, re-anchor) snap the tracker immediately.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace l2_driver
{

struct ClockMapperConfig
{
  double max_drift_rate = 2.0e-4;   // [s/s] max upward slope of the estimate without evidence
  double outlier_threshold = 0.02;  // [s] below-estimate distance that triggers quarantine
  int outlier_confirm = 25;         // consecutive quarantined samples needed to accept them
  double lag_threshold = 0.01;      // [s] above-estimate distance that counts as "far"
  double lag_window = 2.0;          // [s] continuously "far" for this long => re-anchor
  double lock_time = 1.0;           // [s] warm-up before the estimate is considered locked
  double smoothing_time = 5.0;      // [s] averaging of the envelope used for mapping
};

enum class MapperEvent
{
  kNone,
  kInitialized,
  kQuarantined,          // sample ignored for estimation (looked impossible)
  kAcceptedLowerOffset,  // quarantine confirmed, estimate moved down
  kReanchored,           // latency floor moved up, estimate moved up
};

class ClockMapper
{
public:
  explicit ClockMapper(const ClockMapperConfig & cfg = ClockMapperConfig())
  : cfg_(cfg) {}

  void reset() {initialized_ = false; q_count_ = 0; far_since_ = kNone;}

  bool initialized() const {return initialized_;}

  bool locked(int64_t host_ns) const
  {
    return initialized_ && (host_ns - init_host_ns_) >= sec_to_ns(cfg_.lock_time);
  }

  // Offset such that host_mono = device + offset (smoothed value used for mapping).
  int64_t offset_ns() const {return smooth_ns_;}
  // Raw lower-envelope value (diagnostics).
  int64_t envelope_ns() const {return offset_ns_;}
  // Measured relative drift of the two clocks [ppm] (diagnostics).
  double drift_ppm() const {return rate_ * 1e6;}

  // Latency of the last sample relative to the envelope (>= 0 unless quarantined).
  int64_t last_residual_ns() const {return last_residual_ns_;}

  int64_t to_host(int64_t device_ns) const {return device_ns + smooth_ns_;}

  MapperEvent update(int64_t device_ns, int64_t host_ns)
  {
    const int64_t d = host_ns - device_ns;
    if (!initialized_) {
      initialized_ = true;
      offset_ns_ = d;
      smooth_ns_ = d;
      frac_ = 0.0;
      rate_ = 0.0;
      init_host_ns_ = host_ns;
      last_host_ns_ = host_ns;
      last_dev_ns_ = device_ns;
      q_count_ = 0;
      far_since_ = kNone;
      last_residual_ns_ = 0;
      return MapperEvent::kInitialized;
    }

    const int64_t dt_host = std::max<int64_t>(0, host_ns - last_host_ns_);
    const int64_t dt_dev = std::max<int64_t>(0, device_ns - last_dev_ns_);
    last_host_ns_ = std::max(last_host_ns_, host_ns);
    last_dev_ns_ = std::max(last_dev_ns_, device_ns);
    last_residual_ns_ = d - offset_ns_;

    // Quarantine: a sample far *below* the envelope means negative latency,
    // which is impossible unless the stamp is corrupt or the clock changed.
    if (last_residual_ns_ < -sec_to_ns(cfg_.outlier_threshold) && locked(host_ns)) {
      q_min_ns_ = (q_count_ == 0) ? d : std::min(q_min_ns_, d);
      if (++q_count_ >= cfg_.outlier_confirm) {
        offset_ns_ = q_min_ns_;
        snap();
        q_count_ = 0;
        far_since_ = kNone;
        return MapperEvent::kAcceptedLowerOffset;
      }
      return MapperEvent::kQuarantined;
    }
    q_count_ = 0;

    // Lower-envelope tracking with bounded upward slope.
    const int64_t rise = static_cast<int64_t>(cfg_.max_drift_rate * static_cast<double>(dt_host));
    offset_ns_ = std::min(offset_ns_ + rise, d);
    if (!locked(host_ns)) {
      snap();  // converge fast during warm-up
    } else {
      // Critically damped tracker, integrated in device time.
      const double dt = static_cast<double>(std::min<int64_t>(dt_dev, sec_to_ns(0.1)));
      const double w = 1.0 / (cfg_.smoothing_time * 1e9);
      const double e = static_cast<double>(offset_ns_ - smooth_ns_) - frac_;
      frac_ += rate_ * dt + 2.0 * w * dt * e;
      rate_ += w * w * dt * e;
      const int64_t step = static_cast<int64_t>(std::llround(frac_));
      smooth_ns_ += step;
      frac_ -= static_cast<double>(step);
    }

    // Lag guard: latency floor moved up permanently (e.g. network path changed).
    if (d - offset_ns_ > sec_to_ns(cfg_.lag_threshold)) {
      if (far_since_ == kNone) {
        far_since_ = host_ns;
        far_min_ns_ = d;
      } else {
        far_min_ns_ = std::min(far_min_ns_, d);
      }
      if (host_ns - far_since_ >= sec_to_ns(cfg_.lag_window)) {
        offset_ns_ = far_min_ns_;
        snap();
        far_since_ = kNone;
        return MapperEvent::kReanchored;
      }
    } else {
      far_since_ = kNone;
    }
    return MapperEvent::kNone;
  }

  static int64_t sec_to_ns(double s) {return static_cast<int64_t>(s * 1e9);}

private:
  void snap() {smooth_ns_ = offset_ns_; frac_ = 0.0;}

  static constexpr int64_t kNone = std::numeric_limits<int64_t>::min();

  ClockMapperConfig cfg_;
  bool initialized_ = false;
  int64_t offset_ns_ = 0;
  int64_t smooth_ns_ = 0;
  double frac_ = 0.0;   // sub-ns remainder of the tracker
  double rate_ = 0.0;   // d(offset)/d(device time)
  int64_t last_dev_ns_ = 0;
  int64_t init_host_ns_ = 0;
  int64_t last_host_ns_ = 0;
  int64_t last_residual_ns_ = 0;
  int q_count_ = 0;
  int64_t q_min_ns_ = 0;
  int64_t far_since_ = kNone;
  int64_t far_min_ns_ = 0;
};

}  // namespace l2_driver
