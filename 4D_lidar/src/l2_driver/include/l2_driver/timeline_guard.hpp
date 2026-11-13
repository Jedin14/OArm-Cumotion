// Copyright 2026. BSD-3-Clause.
//
// TimelineGuard: sanity-checks the *device* timeline of one packet stream before
// its stamps are trusted. It uses the packet sequence counter as an independent
// clock: between two packets the device stamp must advance by about
// (seq difference) x (nominal period).
//
//  * duplicate packet               -> dropped
//  * single corrupted stamp         -> dropped (one-packet look-ahead decides)
//  * real device clock step / lidar reboot (seq restarts, clock re-synced)
//                                   -> released with timeline_reset = true so the
//                                      caller re-initialises its clock mapping
//
// The counter may be narrower than 32 bit (the L2, firmware 2.8.8, counts 0..1023): the
// modulus is learned from the first wrap that the device stamps confirm, and every item
// carries an unwrapped 64-bit sequence number (`useq`).
//
// A host-side transport stall can never be mistaken for a device clock step,
// because only device stamps and sequence numbers are compared (when the
// sequence counter is trusted). Until the counter has been verified, the guard
// falls back to comparing device-elapsed against host-elapsed time with a
// generous tolerance.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace l2_driver
{

struct TimelineGuardConfig
{
  double seq_tolerance = 0.25;    // [s] allowed |device_elapsed - dseq*period| (absolute part)
  double seq_rel_tolerance = 0.01;  // relative part (of the expected elapsed time)
  double host_tolerance = 0.5;    // [s] allowed |device_elapsed - host_elapsed| (untrusted seq)
  int learn_samples = 200;        // samples used to verify the sequence counter
};

struct GuardStats
{
  uint64_t accepted = 0;
  uint64_t duplicates = 0;
  uint64_t glitches = 0;    // corrupted / inconsistent stamps dropped
  uint64_t resets = 0;      // confirmed device timeline discontinuities
  uint64_t lost = 0;        // packets missing according to the sequence counter
  uint64_t unit_steps = 0;  // accepted packets whose sequence number advanced by exactly 1
};

template<typename Payload>
class TimelineGuard
{
public:
  struct Item
  {
    uint32_t seq;
    int64_t dev_ns;
    int64_t host_ns;
    Payload payload;
    bool timeline_reset;  // true for the first item of a new device timeline
    int64_t useq;         // unwrapped sequence number (restarts at 0 with each timeline)
  };

  explicit TimelineGuard(const TimelineGuardConfig & cfg = TimelineGuardConfig())
  : cfg_(cfg) {}

  // Feed one packet. Zero, one or two items are appended to `out`, in order.
  void push(uint32_t seq, int64_t dev_ns, int64_t host_ns, Payload payload, std::vector<Item> & out)
  {
    Sample s{seq, dev_ns, host_ns};
    if (!has_last_) {
      release(s, std::move(payload), true, out);
      return;
    }
    if (!has_pending_) {
      if (is_duplicate(last_, s)) {++stats_.duplicates; return;}
      if (consistent(last_, s)) {release(s, std::move(payload), false, out); return;}
      pending_ = s;
      pending_payload_ = std::move(payload);
      has_pending_ = true;
      return;
    }
    // A suspicious packet is pending: this packet decides what it was.
    if (is_duplicate(pending_, s)) {++stats_.duplicates; return;}
    if (consistent(last_, s)) {
      // Old timeline continues: the pending stamp was a glitch.
      ++stats_.glitches;
      has_pending_ = false;
      release(s, std::move(payload), false, out);
      return;
    }
    if (consistent(pending_, s)) {
      // Two packets agree on a new timeline: real discontinuity.
      ++stats_.resets;
      has_pending_ = false;
      release(pending_, std::move(pending_payload_), true, out);
      release(s, std::move(payload), false, out);
      return;
    }
    ++stats_.glitches;
    pending_ = s;
    pending_payload_ = std::move(payload);
  }

  const GuardStats & stats() const {return stats_;}
  bool seq_trusted() const {return seq_trusted_;}
  // Fraction of packets whose counter advanced by exactly one (1.0 for a per-stream counter
  // without loss; clearly lower if the counter is shared with other packet types).
  double unit_step_fraction() const
  {
    return stats_.accepted > 1 ?
           static_cast<double>(stats_.unit_steps) / static_cast<double>(stats_.accepted - 1) : 0.0;
  }
  double period_s() const {return period_ns_ * 1e-9;}
  uint64_t seq_modulus() const {return seq_mod_;}   // 0 = full 32-bit counter
  // Host seconds per device second; used only for the untrusted-counter consistency check.
  void set_rate(double host_per_device) {rate_ = host_per_device;}

private:
  struct Sample
  {
    uint32_t seq;
    int64_t dev_ns;
    int64_t host_ns;
  };

  static int64_t raw_diff(uint32_t a, uint32_t b)
  {
    return static_cast<int64_t>(static_cast<int32_t>(b - a));
  }

  // Width of a counter that wraps from `a` (near its top) back to `b` (small), or 0.
  uint64_t wrap_candidate(uint32_t a, uint32_t b) const
  {
    uint64_t m = 64;
    while (m <= max_seq_seen_) {m <<= 1;}
    if (m > (1ULL << 31) || b >= m || a < m - m / 8 || b >= m / 8) {return 0;}
    return m;
  }

  // Sequence step from a to b, modulo the (learned) counter width.
  int64_t seq_diff(uint32_t a, uint32_t b) const
  {
    if (seq_mod_ > 0) {
      return static_cast<int64_t>((static_cast<uint64_t>(b % seq_mod_) + seq_mod_ - a % seq_mod_) % seq_mod_);
    }
    const int64_t d = raw_diff(a, b);
    if (d < 0) {
      const uint64_t m = wrap_candidate(a, b);
      if (m) {return static_cast<int64_t>((static_cast<uint64_t>(b) + m - a) % m);}
    }
    return d;
  }

  static bool is_duplicate(const Sample & a, const Sample & b)
  {
    return a.seq == b.seq && a.dev_ns == b.dev_ns;
  }

  bool consistent(const Sample & a, const Sample & b) const
  {
    const int64_t dd = b.dev_ns - a.dev_ns;
    if (dd < 0) {return false;}  // equal stamps are legal: firmware may stamp a burst at send time
    if (seq_trusted_) {
      const int64_t dseq = seq_diff(a.seq, b.seq);
      if (dseq <= 0) {return false;}
      const double expected = static_cast<double>(dseq) * period_ns_;
      const double tol = cfg_.seq_tolerance * 1e9 + cfg_.seq_rel_tolerance * expected;
      return std::fabs(static_cast<double>(dd) - expected) <= tol;
    }
    const int64_t dh = b.host_ns - a.host_ns;
    return std::fabs(static_cast<double>(dd) * rate_ - static_cast<double>(dh)) <=
           cfg_.host_tolerance * 1e9;
  }

  void release(const Sample & s, Payload && p, bool reset, std::vector<Item> & out)
  {
    if (has_last_ && !reset) {
      const int64_t dseq = seq_diff(last_.seq, s.seq);
      if (seq_mod_ == 0 && raw_diff(last_.seq, s.seq) < 0 && dseq > 0) {
        seq_mod_ = wrap_candidate(last_.seq, s.seq);   // first confirmed wrap: learn the width
      }
      if (seq_trusted_ && dseq > 1) {stats_.lost += static_cast<uint64_t>(dseq - 1);}
      if (dseq == 1) {++stats_.unit_steps;}
      out_useq_ += std::max<int64_t>(dseq, 0);
      learn(s);
    } else {
      out_useq_ = 0;
      // New timeline: restart the long-baseline period estimate, keep trust.
      anchor_ = s;
      anchor_useq_ = 0;
      useq_ = 0;
      learn_.clear();
    }
    last_ = s;
    has_last_ = true;
    max_seq_seen_ = std::max<uint64_t>(max_seq_seen_, s.seq);
    ++stats_.accepted;
    out.push_back(Item{s.seq, s.dev_ns, s.host_ns, std::move(p), reset, out_useq_});
  }

  // Long-baseline period estimate and verification of the sequence counter.
  void learn(const Sample & s)
  {
    const int64_t dseq = seq_diff(last_.seq, s.seq);
    useq_ += dseq;
    const int64_t du = useq_ - anchor_useq_;
    const int64_t dd = s.dev_ns - anchor_.dev_ns;
    if (seq_trusted_) {
      // Refine on an ever-growing baseline (ppm-level accuracy, immune to stamp bunching).
      if (du >= kMinRefineBaseline && dd > 0) {
        period_ns_ = static_cast<double>(dd) / static_cast<double>(du);
      }
      return;
    }
    learn_.push_back({du, dd, dseq});
    if (static_cast<int>(learn_.size()) < cfg_.learn_samples) {return;}

    // Verify: counter strictly increasing and stamps follow seq*period within tolerance.
    const auto & back = learn_.back();
    if (back.u <= 0 || back.d <= 0) {learn_.clear(); return;}
    const double p = static_cast<double>(back.d) / static_cast<double>(back.u);
    int increasing = 0;
    double worst = 0.0;
    for (const auto & e : learn_) {
      if (e.dseq >= 1) {++increasing;}
      worst = std::max(worst, std::fabs(static_cast<double>(e.d) - p * static_cast<double>(e.u)));
    }
    const bool ok = increasing >= static_cast<int>(0.95 * learn_.size()) &&
      worst <= cfg_.seq_tolerance * 1e9;
    if (ok) {
      seq_trusted_ = true;
      period_ns_ = p;
    }
    learn_.clear();
    anchor_ = s;
    anchor_useq_ = useq_;
  }

  static constexpr int64_t kMinRefineBaseline = 2000;

  struct LearnEntry
  {
    int64_t u;
    int64_t d;
    int64_t dseq;
  };

  TimelineGuardConfig cfg_;
  GuardStats stats_;
  bool has_last_ = false;
  bool has_pending_ = false;
  Sample last_{};
  Sample pending_{};
  Payload pending_payload_{};
  bool seq_trusted_ = false;
  double period_ns_ = 0.0;
  Sample anchor_{};
  int64_t anchor_useq_ = 0;
  int64_t out_useq_ = 0;
  uint64_t seq_mod_ = 0;
  uint64_t max_seq_seen_ = 0;
  double rate_ = 1.0;
  int64_t useq_ = 0;
  std::vector<LearnEntry> learn_;
};

}  // namespace l2_driver
