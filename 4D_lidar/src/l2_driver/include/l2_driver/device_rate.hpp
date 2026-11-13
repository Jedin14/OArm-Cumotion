// Copyright 2026. BSD-3-Clause.
//
// DeviceRate: measures how fast the device clock runs compared with the host clock and
// converts device timestamps into "scaled" device time that advances at the host rate.
//
// Measured on a real Unitree L2 (firmware 2.8.8.1, serial): every hardware timestamp
// (IMU stamps, point-packet stamps and the per-point time_increment) advances at
// ~0.5x real time, while the scan itself runs at its nominal rate (scan_period, 65 k pts/s).
// The ClockMapper only tolerates ppm-level drift, so the rate must be removed first.
// Nothing is hard-coded: the rate is measured, so a firmware with a correct clock simply
// yields ~1.0.
//
// Each block (default 2 s of host time) contributes one anchor: its lowest-latency sample
// (minimum of host - rate * device). The rate is the slope between the first anchor and
// the latest one, so its accuracy grows with the baseline (latency jitter / baseline).
// to_scaled() is piecewise linear and continuous: refining the rate never makes stamps jump.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace l2_driver
{

struct DeviceRateConfig
{
  double block = 2.0;       // [s host] anchor block length
  int min_blocks = 3;       // blocks before the rate is trusted (baseline ~ (min_blocks-1) * block)
  double min_rate = 0.05;   // plausible host-seconds per device-second
  double max_rate = 20.0;
};

class DeviceRate
{
public:
  explicit DeviceRate(const DeviceRateConfig & cfg = DeviceRateConfig())
  : cfg_(cfg) {}

  void reset()
  {
    started_ = ready_ = have_first_ = false;
    rate_ = 1.0;
    blocks_ = 0;
    blk_.clear();
    first_blk_.clear();
  }

  // After a device clock step: keep the measured rate, restart the anchor baseline.
  void rebase()
  {
    have_first_ = false;
    started_ = false;
    blk_.clear();
    first_blk_.clear();
  }

  bool ready() const {return ready_;}
  double rate() const {return rate_;}   // host seconds per device second

  int64_t to_scaled(int64_t dev) const
  {
    return s_ref_ + static_cast<int64_t>(std::llround(rate_ * static_cast<double>(dev - d_ref_)));
  }

  // Feed (device stamp, host arrival). Returns true exactly once: when the rate first
  // becomes trustworthy (downstream mappings should then be re-initialised).
  bool update(int64_t dev, int64_t host)
  {
    if (!started_) {
      started_ = true;
      if (!ready_) {d_ref_ = dev; s_ref_ = dev;}
      else {s_ref_ = dev; d_ref_ = dev;}   // after rebase: start the scaled axis here
      block_start_ = host;
      dev0_ = dev;
      host0_ = host;
    }
    blk_.push_back({static_cast<double>(dev - dev0_), static_cast<double>(host - host0_)});
    if (static_cast<double>(host - block_start_) < cfg_.block * 1e9 || blk_.size() < 20) {
      return false;
    }
    block_start_ = host;
    ++blocks_;
    const double last_dev = blk_.back().d;

    if (!have_first_) {
      // Provisional slope from this block's end points; keep the samples for a joint refit.
      const double k = slope(blk_.front(), blk_.back());
      if (plausible(k)) {set_rate(k, last_dev);}
      first_blk_.swap(blk_);
      blk_.clear();
      have_first_ = true;
      first_ = low(first_blk_, rate_);
      return finish();
    }
    double k = rate_;
    Pt a = low(blk_, k);
    if (!first_blk_.empty()) {
      // Second block: refine both anchors together (the first was picked with a rough slope).
      for (int it = 0; it < 4; ++it) {
        first_ = low(first_blk_, k);
        a = low(blk_, k);
        const double kk = slope(first_, a);
        if (!plausible(kk)) {break;}
        k = kk;
      }
      first_blk_.clear();
    } else {
      const double kk = slope(first_, a);
      if (plausible(kk) && std::fabs(kk / rate_ - 1.0) < 0.02) {k = kk;}
    }
    if (plausible(k)) {set_rate(k, last_dev);}
    blk_.clear();
    return finish();
  }

private:
  struct Pt
  {
    double d, h;   // device / host time relative to (dev0_, host0_)
  };

  static double slope(const Pt & a, const Pt & b)
  {
    return (b.d > a.d) ? (b.h - a.h) / (b.d - a.d) : 0.0;
  }

  bool plausible(double k) const {return k >= cfg_.min_rate && k <= cfg_.max_rate;}

  static Pt low(const std::vector<Pt> & v, double k)
  {
    Pt best = v.front();
    double m = 1e300;
    for (const auto & p : v) {
      const double r = p.h - k * p.d;
      if (r < m) {m = r; best = p;}
    }
    return best;
  }

  // Change the rate at device time `at` (relative) without a discontinuity.
  void set_rate(double k, double at_rel)
  {
    const int64_t at = dev0_ + static_cast<int64_t>(at_rel);
    s_ref_ = to_scaled(at);
    d_ref_ = at;
    rate_ = k;
  }

  bool finish()
  {
    if (!ready_ && blocks_ >= cfg_.min_blocks) {
      ready_ = true;
      return true;
    }
    return false;
  }

  DeviceRateConfig cfg_;
  bool started_ = false, ready_ = false, have_first_ = false;
  double rate_ = 1.0;
  int blocks_ = 0;
  int64_t d_ref_ = 0, s_ref_ = 0;
  int64_t dev0_ = 0, host0_ = 0, block_start_ = 0;
  Pt first_{0.0, 0.0};
  std::vector<Pt> blk_, first_blk_;
};

}  // namespace l2_driver
