// Copyright 2026. BSD-3-Clause.
//
// SequenceTimeline: reconstructs evenly spaced IMU sample times from the packet
// sequence counter.
//
// The IMU is sampled by hardware at a fixed rate, so sample k happened at
//     t_k = t_0 + P * (seq_k - seq_0)        (P = sample period, device time)
// If the firmware stamps packets when it *sends* them instead of when it
// samples them, the stamps arrive bunched (many nearly identical stamps, then a
// gap). Such stamps are only ever late, never early, so:
//
//  1. P is measured between robust LOW anchors of 2 s blocks on an ever-growing
//     baseline (ppm accuracy after a minute, independent of the bunching);
//  2. a "sequence clock" advances by P per sample;
//  3. a ClockMapper (lower envelope + smoothing) maps that sequence clock onto
//     the device stamps, absorbing any residual period error as drift.
//
// With well-behaved stamps the output equals the raw stamps to microseconds;
// with bunched stamps it restores the true, regular sampling instants. Until a
// healthy estimate exists, update() returns the raw device stamp.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "l2_driver/clock_mapper.hpp"

namespace l2_driver
{

struct SequenceTimelineConfig
{
  double block = 2.0;               // [s] block length for period anchors
  double envelope_quantile = 0.005; // robust "minimum" (ignores the lowest 0.5 %)
  double max_residual = 0.25;       // [s] stamps later than the timeline by more are "bad"
  double early_tolerance = 0.005;   // [s] stamps earlier than the timeline by more are "bad"
  double max_bad_fraction = 0.05;   // per block, above this the timeline is not used
};

class SequenceTimeline
{
public:
  explicit SequenceTimeline(const SequenceTimelineConfig & cfg = SequenceTimelineConfig())
  : cfg_(cfg), map_(mapper_config()) {}

  void reset()
  {
    started_ = false;
    healthy_ = false;
    have_early_ = false;
    period_ns_ = 0.0;
    map_.reset();
    block_.clear();
    res_.clear();
  }

  bool healthy() const {return healthy_;}
  double period_s() const {return period_ns_ * 1e-9;}
  double residual_p99_s() const {return res_p99_ns_ * 1e-9;}
  double residual_max_s() const {return res_max_ns_ * 1e-9;}

  // useq: unwrapped sequence number (TimelineGuard::Item::useq), dev_ns: device stamp.
  int64_t update(int64_t useq, int64_t dev_ns)
  {
    if (!started_) {
      started_ = true;
      last_seq_ = useq;
      useq_ = 0;
      d0_ = dev_ns;
      block_start_d_ = 0.0;
      block_.clear();
      block_.push_back({0, 0.0});
      return dev_ns;
    }
    const int64_t dseq = useq - last_seq_;
    if (dseq <= 0) {
      return dev_ns;  // the TimelineGuard normally prevents this
    }
    last_seq_ = useq;
    useq_ += dseq;
    const double d = static_cast<double>(dev_ns - d0_);
    block_.push_back({useq_, d});

    if (d - block_start_d_ >= cfg_.block * 1e9 && block_.size() >= 20) {
      finish_block();
      block_start_d_ = d;
    }
    if (period_ns_ <= 0.0) {
      return dev_ns;
    }

    sclock_ += static_cast<double>(dseq) * period_ns_;
    const int64_t s = static_cast<int64_t>(std::llround(sclock_));
    map_.update(s, dev_ns - d0_);
    const int64_t out = d0_ + map_.to_host(s);
    res_.push_back(static_cast<double>(dev_ns - out));
    return healthy_ ? out : dev_ns;
  }

private:
  static ClockMapperConfig mapper_config()
  {
    ClockMapperConfig c;
    c.max_drift_rate = 5e-4;     // residual period error early on
    c.outlier_threshold = 0.02;  // stamps 20 ms earlier than the timeline: corrupt
    c.lag_threshold = 0.3;       // bunching makes most stamps late; only react to big shifts
    c.lag_window = 2.0;
    c.lock_time = 1.0;
    c.smoothing_time = 2.0;
    return c;
  }

  struct Entry
  {
    int64_t u;
    double d;
  };

  // Robust low point of entries [begin, end) w.r.t. slope b.
  std::pair<int64_t, double> low_anchor(size_t begin, size_t end, double b)
  {
    tmp_.clear();
    for (size_t i = begin; i < end; ++i) {
      tmp_.push_back({block_[i].d - b * static_cast<double>(block_[i].u), i});
    }
    const size_t k = static_cast<size_t>(cfg_.envelope_quantile * static_cast<double>(tmp_.size() - 1));
    std::nth_element(tmp_.begin(), tmp_.begin() + k, tmp_.end());
    const auto & e = block_[tmp_[k].second];
    return {e.u, e.d};
  }

  void finish_block()
  {
    const auto & f = block_.front();
    const auto & l = block_.back();
    double b = (period_ns_ > 0.0) ? period_ns_ :
      (l.d - f.d) / static_cast<double>(std::max<int64_t>(1, l.u - f.u));

    if (!have_early_) {
      // First block: slope from its two halves, anchor = its low point.
      const size_t mid = block_.size() / 2;
      for (int it = 0; it < 3; ++it) {
        const auto a1 = low_anchor(0, mid, b);
        const auto a2 = low_anchor(mid, block_.size(), b);
        if (a2.first > a1.first) {
          b = (a2.second - a1.second) / static_cast<double>(a2.first - a1.first);
        }
      }
      if (b > 0.0) {
        early_ = low_anchor(0, block_.size(), b);
        have_early_ = true;
        period_ns_ = b;
        sclock_ = static_cast<double>(useq_) * period_ns_;
      }
    } else {
      // Growing baseline from the first block's anchor to this block's anchor.
      const auto a = low_anchor(0, block_.size(), b);
      if (a.first > early_.first) {
        const double nb = (a.second - early_.second) / static_cast<double>(a.first - early_.first);
        if (nb > 0.0 && std::fabs(nb / b - 1.0) < 0.01) {period_ns_ = nb;}
      }
    }

    // Health of the timeline over this block.
    if (!res_.empty()) {
      size_t bad = 0;
      for (double r : res_) {
        if (r > cfg_.max_residual * 1e9 || r < -cfg_.early_tolerance * 1e9) {++bad;}
      }
      const size_t k = static_cast<size_t>(0.99 * static_cast<double>(res_.size() - 1));
      std::nth_element(res_.begin(), res_.begin() + k, res_.end());
      res_p99_ns_ = res_[k];
      res_max_ns_ = *std::max_element(res_.begin(), res_.end());
      healthy_ = map_.initialized() &&
        static_cast<double>(bad) <= cfg_.max_bad_fraction * static_cast<double>(res_.size());
      res_.clear();
    }
    const Entry last = block_.back();
    block_.clear();
    block_.push_back(last);
  }

  SequenceTimelineConfig cfg_;
  ClockMapper map_;
  bool started_ = false;
  bool healthy_ = false;
  int64_t last_seq_ = 0;
  int64_t useq_ = 0;
  int64_t d0_ = 0;
  double block_start_d_ = 0.0;
  double period_ns_ = 0.0;
  double sclock_ = 0.0;
  bool have_early_ = false;
  std::pair<int64_t, double> early_{0, 0.0};
  double res_p99_ns_ = 0.0;
  double res_max_ns_ = 0.0;
  std::vector<Entry> block_;
  std::vector<double> res_;
  std::vector<std::pair<double, size_t>> tmp_;
};

}  // namespace l2_driver
