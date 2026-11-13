// Copyright 2026. BSD-3-Clause.
// Unit tests for the L2 timing pipeline: TimelineGuard -> SequenceTimeline -> ClockMapper.
// Every test simulates a real failure mode seen with the Unitree L2.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "l2_driver/clock_mapper.hpp"
#include "l2_driver/device_rate.hpp"
#include "l2_driver/sequence_timeline.hpp"
#include "l2_driver/timeline_guard.hpp"

using l2_driver::ClockMapper;
using l2_driver::ClockMapperConfig;
using l2_driver::DeviceRate;
using l2_driver::MapperEvent;
using l2_driver::SequenceTimeline;
using l2_driver::TimelineGuard;

namespace
{

constexpr int64_t kMs = 1000000;
constexpr int64_t kSec = 1000000000;
constexpr int64_t kImuPeriod = 2500000;  // 400 Hz

struct Packet
{
  uint32_t seq;
  int64_t true_host;  // host-monotonic instant the sample was taken
  int64_t dev;        // stamp written by the device
  int64_t arrival;    // host-monotonic arrival time
};

// IMU stream generator.
//   device_epoch: device clock value at host-mono 0 (1970-ish, future, ...)
//   drift_ppm:    device clock rate error
//   bursty:       packets are withheld and released in bursts (L2 delivery bug)
//   bunched:      the device stamps packets when *sending* the burst
//   speed:        device clock speed (the real L2 runs at ~0.5)
//   seq_mod:      sequence counter width (the real L2 wraps at 1024), 0 = 32 bit
std::vector<Packet> make_stream(
  int n, int64_t host_start, int64_t device_epoch, double drift_ppm,
  bool bursty, bool bunched, uint32_t seed = 1, double speed = 1.0, uint32_t seq_mod = 0)
{
  std::mt19937 rng(seed);
  std::exponential_distribution<double> lat_exp(1.0 / 0.4e6);  // mean 0.4 ms extra
  std::uniform_int_distribution<int> burst_len_ms(20, 100);
  std::vector<Packet> out;
  out.reserve(n);
  int64_t next_release = host_start + burst_len_ms(rng) * kMs;
  for (int i = 0; i < n; ++i) {
    Packet p;
    p.seq = static_cast<uint32_t>(1000 + i);
    if (seq_mod) {p.seq %= seq_mod;}
    p.true_host = host_start + static_cast<int64_t>(i) * kImuPeriod;
    const double dev_elapsed = static_cast<double>(p.true_host) * speed * (1.0 + drift_ppm * 1e-6);
    p.dev = device_epoch + static_cast<int64_t>(dev_elapsed);
    const int64_t base_latency = 300000 + static_cast<int64_t>(lat_exp(rng));  // >= 0.3 ms
    if (bursty) {
      while (next_release < p.true_host) {next_release += burst_len_ms(rng) * kMs;}
      p.arrival = next_release + base_latency;
      if (bunched) {
        const double send_elapsed = static_cast<double>(next_release) * speed * (1.0 + drift_ppm * 1e-6);
        p.dev = device_epoch + static_cast<int64_t>(send_elapsed);
      }
    } else {
      p.arrival = p.true_host + base_latency;
    }
    out.push_back(p);
  }
  return out;
}

struct PipelineResult
{
  double max_err_ms = 0;       // |mapped - (true + min latency)| after warm-up
  double max_spacing_err_ms = 0;  // |(mapped_k - mapped_k-1) - true spacing|
  int resets = 0;
};

// Full driver timing path as used in l2_driver_node.cpp.
PipelineResult run_pipeline(const std::vector<Packet> & pk, double warmup_s, bool use_seq_timeline)
{
  TimelineGuard<int> guard;
  SequenceTimeline seqtl;
  DeviceRate rate;
  ClockMapper mapper;
  std::vector<TimelineGuard<int>::Item> items;
  PipelineResult r;
  bool have_prev = false;
  int64_t prev_mapped = 0, prev_true = 0;
  const int64_t t0 = pk.front().true_host;
  for (const auto & p : pk) {
    items.clear();
    guard.push(p.seq, p.dev, p.arrival, 0, items);
    for (const auto & it : items) {
      if (it.timeline_reset) {
        mapper.reset();
        seqtl.reset();
        rate.rebase();
        ++r.resets;
      }
      if (rate.update(it.dev_ns, it.host_ns)) {mapper.reset();}   // rate known: restart mapping
      guard.set_rate(rate.rate());
      const int64_t dev = use_seq_timeline ? seqtl.update(it.useq, it.dev_ns) : it.dev_ns;
      const int64_t sdev = rate.to_scaled(dev);
      mapper.update(sdev, it.host_ns);
      const int64_t mapped = mapper.to_host(sdev);
      if (!rate.ready()) {continue;}   // the driver does not publish before this
      // true time of this packet (no packets are dropped by make_stream)
      const int64_t true_t = pk[static_cast<size_t>(it.useq)].true_host;
      if (true_t - t0 > static_cast<int64_t>(warmup_s * 1e9)) {
        // expected: true sample time + min latency (0.3 ms), which the driver
        // removes with transport_latency.
        const double err = std::fabs(static_cast<double>(mapped - (true_t + 300000))) / 1e6;
        r.max_err_ms = std::max(r.max_err_ms, err);
        if (have_prev) {
          const double sp = std::fabs(
            static_cast<double>((mapped - prev_mapped) - (true_t - prev_true))) / 1e6;
          r.max_spacing_err_ms = std::max(r.max_spacing_err_ms, sp);
        }
        have_prev = true;
      }
      prev_mapped = mapped;
      prev_true = true_t;
    }
  }
  return r;
}

}  // namespace

// --------------------------------------------------------------------------
// ClockMapper
// --------------------------------------------------------------------------

TEST(ClockMapper, Device1970ClockWithBurstyDelivery)
{
  // Device clock near the 1970 epoch, host monotonic at 50 000 s, bursty arrivals.
  auto pk = make_stream(20000, 50000 * kSec, 12 * kSec, 0.0, true, false);
  auto r = run_pipeline(pk, 15.0, false);   // device rate baseline needs ~15 s for < 0.5 ms
  EXPECT_LT(r.max_err_ms, 0.5);
  EXPECT_LT(r.max_spacing_err_ms, 0.2);  // even spacing despite 100 ms bursts
  EXPECT_EQ(r.resets, 1);                // only the initial one
}

TEST(ClockMapper, DeviceFutureClock)
{
  // Device clock in 2040 (future timestamps).
  auto pk = make_stream(8000, 1000 * kSec, 2208988800LL * kSec, 0.0, true, false);
  auto r = run_pipeline(pk, 15.0, false);
  EXPECT_LT(r.max_err_ms, 0.5);
  EXPECT_LT(r.max_spacing_err_ms, 0.2);
}

TEST(ClockMapper, OscillatorDriftOneHour)
{
  for (double ppm : {+80.0, -80.0}) {
    auto pk = make_stream(400 * 3600, 100 * kSec, 5 * kSec, ppm, true, false, 7);
    auto r = run_pipeline(pk, 10.0, false);
    EXPECT_LT(r.max_err_ms, 1.0) << "ppm=" << ppm;   // absolute, over 1.44 M bursty samples
    EXPECT_LT(r.max_spacing_err_ms, 0.05) << "ppm=" << ppm;
    EXPECT_EQ(r.resets, 1) << "ppm=" << ppm;
  }
}

TEST(ClockMapper, CorruptEarlyStampIsQuarantined)
{
  ClockMapper m;
  int64_t host = 10 * kSec, dev = 0;
  for (int i = 0; i < 1000; ++i, host += kImuPeriod, dev += kImuPeriod) {
    m.update(dev, host + kMs);
  }
  const int64_t before = m.offset_ns();
  // stamp claims to be 1 s in the future => negative latency => impossible
  EXPECT_EQ(m.update(dev + kSec, host + kMs), MapperEvent::kQuarantined);
  EXPECT_EQ(m.offset_ns(), before);
}

TEST(ClockMapper, LatencyFloorStepIsReanchored)
{
  ClockMapper m;
  int64_t host = 10 * kSec, dev = 0;
  for (int i = 0; i < 2000; ++i, host += kImuPeriod, dev += kImuPeriod) {
    m.update(dev, host + kMs);
  }
  bool reanchored = false;
  for (int i = 0; i < 1000; ++i, host += kImuPeriod, dev += kImuPeriod) {  // 2.5 s
    if (m.update(dev, host + 41 * kMs) == MapperEvent::kReanchored) {reanchored = true;}
  }
  EXPECT_TRUE(reanchored);
  EXPECT_NEAR(static_cast<double>(m.offset_ns() - (10 * kSec + 41 * kMs)), 0.0, 1.0e5);
}

// --------------------------------------------------------------------------
// TimelineGuard
// --------------------------------------------------------------------------

namespace
{
struct GuardRun
{
  std::vector<TimelineGuard<int>::Item> items;
  TimelineGuard<int> guard;
  void push(uint32_t seq, int64_t dev, int64_t host)
  {
    guard.push(seq, dev, host, static_cast<int>(seq), items);
  }
};
}  // namespace

TEST(TimelineGuard, LearnsPeriodAndTrustsSequence)
{
  GuardRun g;
  for (int i = 0; i < 1000; ++i) {
    g.push(i, i * kImuPeriod, 5 * kSec + i * kImuPeriod + kMs);
  }
  EXPECT_TRUE(g.guard.seq_trusted());
  EXPECT_NEAR(g.guard.period_s(), 0.0025, 1e-7);
  EXPECT_EQ(g.items.size(), 1000u);
}

TEST(TimelineGuard, DropsDuplicatesAndGlitchesDetectsLoss)
{
  GuardRun g;
  int i = 0;
  for (; i < 500; ++i) {g.push(i, i * kImuPeriod, i * kImuPeriod);}
  g.push(i - 1, (i - 1) * kImuPeriod, (i - 1) * kImuPeriod);  // duplicate
  g.push(i, i * kImuPeriod + 5 * kSec, i * kImuPeriod);        // corrupt stamp
  ++i;
  for (; i < 600; ++i) {g.push(i, i * kImuPeriod, i * kImuPeriod);}
  i += 10;  // 10 packets lost
  for (; i < 700; ++i) {g.push(i, i * kImuPeriod, i * kImuPeriod);}
  EXPECT_EQ(g.guard.stats().duplicates, 1u);
  EXPECT_EQ(g.guard.stats().glitches, 1u);
  EXPECT_EQ(g.guard.stats().resets, 0u);
  EXPECT_EQ(g.guard.stats().lost, 11u);  // 10 lost + the dropped corrupt one
  for (const auto & it : g.items) {
    EXPECT_FALSE(it.timeline_reset && it.payload != 0);
  }
}

TEST(TimelineGuard, DetectsDeviceClockStep)
{
  GuardRun g;
  int i = 0;
  for (; i < 500; ++i) {g.push(i, i * kImuPeriod, i * kImuPeriod);}
  const int64_t jump = 100 * kSec;  // e.g. lidar clock re-synced
  for (; i < 600; ++i) {g.push(i, i * kImuPeriod + jump, i * kImuPeriod);}
  EXPECT_EQ(g.guard.stats().resets, 1u);
  int resets_seen = 0;
  for (const auto & it : g.items) {
    if (it.timeline_reset) {
      ++resets_seen;
      if (resets_seen == 2) {EXPECT_EQ(it.payload, 500);}
    }
  }
  EXPECT_EQ(resets_seen, 2);  // initial + the step
  EXPECT_EQ(g.items.size(), 600u);  // nothing lost
}

TEST(TimelineGuard, DetectsLidarRebootSeqRestart)
{
  GuardRun g;
  for (int i = 0; i < 500; ++i) {g.push(5000 + i, i * kImuPeriod, i * kImuPeriod);}
  // reboot: 3 s of silence, seq restarts at 0, device clock restarts near 0
  for (int i = 0; i < 100; ++i) {
    g.push(i, 1 * kSec + i * kImuPeriod, 500 * kImuPeriod + 3 * kSec + i * kImuPeriod);
  }
  EXPECT_EQ(g.guard.stats().resets, 1u);
}

TEST(TimelineGuard, HostStallIsNotAClockStep)
{
  GuardRun g;
  int i = 0;
  for (; i < 500; ++i) {g.push(i, i * kImuPeriod, i * kImuPeriod + kMs);}
  // host thread stalls 0.8 s, then the backlog arrives in one burst
  const int64_t release = i * kImuPeriod + 800 * kMs;
  for (; i < 820; ++i) {g.push(i, i * kImuPeriod, std::max<int64_t>(release, i * kImuPeriod) + kMs);}
  EXPECT_EQ(g.guard.stats().resets, 0u);
  EXPECT_EQ(g.guard.stats().glitches, 0u);
}

// --------------------------------------------------------------------------
// SequenceTimeline
// --------------------------------------------------------------------------

TEST(SequenceTimeline, RegularStampsPassThrough)
{
  SequenceTimeline s;
  std::mt19937 rng(3);
  std::normal_distribution<double> jit(0.0, 2000.0);  // 2 us
  double max_err = 0;
  for (int i = 0; i < 4000 * 3; ++i) {
    const int64_t t = 7 * kSec + static_cast<int64_t>(i) * kImuPeriod;
    const int64_t out = s.update(i, t + static_cast<int64_t>(std::fabs(jit(rng))));
    if (i > 2000) {max_err = std::max(max_err, std::fabs(static_cast<double>(out - t)));}
  }
  EXPECT_TRUE(s.healthy());
  EXPECT_LT(max_err, 20000.0);  // < 20 us
}

TEST(SequenceTimeline, BunchedStampsAreRegularised)
{
  // Firmware stamps at send time: whole bursts share one stamp.
  auto pk = make_stream(400 * 30, 100 * kSec, 3 * kSec, 30.0, true, true, 11);
  auto raw = run_pipeline(pk, 10.0, false);
  auto fixed = run_pipeline(pk, 10.0, true);
  EXPECT_GT(raw.max_spacing_err_ms, 15.0);   // raw stamps are useless for integration
  EXPECT_LT(fixed.max_spacing_err_ms, 0.1);  // regularised: even 2.5 ms spacing
  EXPECT_LT(fixed.max_err_ms, 1.0);
}

TEST(SequenceTimeline, SmallEarlyGlitchDoesNotBendTimeline)
{
  SequenceTimeline s;
  double max_err = 0;
  for (int i = 0; i < 400 * 20; ++i) {
    const int64_t t = static_cast<int64_t>(i) * kImuPeriod;
    int64_t stamp = t;
    if (i == 400 * 8) {stamp -= 50 * kMs;}  // 50 ms early: below the guard threshold
    const int64_t out = s.update(i, stamp);
    if (i > 400 * 3 && i != 400 * 8) {
      max_err = std::max(max_err, std::fabs(static_cast<double>(out - t)));
    }
  }
  EXPECT_LT(max_err, 20000.0);
}

// --------------------------------------------------------------------------
// End-to-end: the combination that breaks Point-LIO with the stock driver
// --------------------------------------------------------------------------

TEST(Pipeline, WorstCaseCombination)
{
  // 1970 device clock + 60 ppm drift + bursty delivery + bunched stamps, 5 minutes.
  auto pk = make_stream(400 * 300, 12345 * kSec, 2 * kSec, 60.0, true, true, 5);
  auto r = run_pipeline(pk, 10.0, true);
  EXPECT_LT(r.max_err_ms, 1.0);
  EXPECT_LT(r.max_spacing_err_ms, 0.1);
  EXPECT_EQ(r.resets, 1);
}

TEST(TimelineGuard, UnitStepFractionDistinguishesSharedCounter)
{
  GuardRun own, shared;
  for (int i = 0; i < 1000; ++i) {
    own.push(i, i * kImuPeriod, i * kImuPeriod);
    // counter shared with ~1 point packet per IMU packet: advances by 1 or 2 or 3
    const uint32_t seq = static_cast<uint32_t>(i * 2 + (i % 3 == 0 ? 1 : 0));
    shared.push(seq, i * kImuPeriod, i * kImuPeriod);
  }
  EXPECT_GT(own.guard.unit_step_fraction(), 0.99);
  EXPECT_LT(shared.guard.unit_step_fraction(), 0.9);
}


// --------------------------------------------------------------------------
// Behaviour measured on a real L2 (firmware 2.8.8.1, serial):
//   device clock at ~0.5x real time, 10-bit sequence counter (0..1023)
// --------------------------------------------------------------------------

TEST(TimelineGuard, SequenceCounterWrapsAt1024)
{
  GuardRun g;
  for (int i = 0; i < 5000; ++i) {
    g.push(static_cast<uint32_t>(i % 1024), i * 2 * kMs, i * 4 * kMs);
  }
  EXPECT_EQ(g.guard.stats().resets, 0u);
  EXPECT_EQ(g.guard.stats().glitches, 0u);
  EXPECT_EQ(g.guard.stats().lost, 0u);
  EXPECT_EQ(g.guard.seq_modulus(), 1024u);
  EXPECT_TRUE(g.guard.seq_trusted());
  EXPECT_GT(g.guard.unit_step_fraction(), 0.999);
  for (size_t k = 0; k < g.items.size(); ++k) {EXPECT_EQ(g.items[k].useq, static_cast<int64_t>(k));}
}

TEST(TimelineGuard, RebootStillDetectedWithNarrowCounter)
{
  GuardRun g;
  for (int i = 0; i < 3000; ++i) {g.push(static_cast<uint32_t>(i % 1024), i * kImuPeriod, i * kImuPeriod);}
  // reboot mid-cycle: counter restarts at 0 from 952, device clock restarts
  for (int i = 0; i < 200; ++i) {g.push(static_cast<uint32_t>(i), 5 * kSec + i * kImuPeriod, 3000 * kImuPeriod + 2 * kSec + i * kImuPeriod);}
  EXPECT_EQ(g.guard.stats().resets, 1u);
}

TEST(DeviceRate, MeasuresHalfSpeedClock)
{
  auto pk = make_stream(400 * 30, 100 * kSec, 7 * kSec, 0.0, true, false, 3, 0.5);
  DeviceRate rate;
  for (const auto & p : pk) {rate.update(p.dev, p.arrival);}
  EXPECT_TRUE(rate.ready());
  EXPECT_NEAR(rate.rate(), 2.0, 2e-4);   // ppm-level after 30 s despite bursts
  // continuity: scaled time advances at host rate between consecutive samples
  const int64_t a = rate.to_scaled(pk[1000].dev), b = rate.to_scaled(pk[1001].dev);
  EXPECT_NEAR(static_cast<double>(b - a), static_cast<double>(kImuPeriod), 2000.0);
}

TEST(Pipeline, RealL2HalfSpeedClockAndWrappingCounter)
{
  // what the real lidar does, plus bursty delivery and 30 ppm drift, 5 minutes
  auto pk = make_stream(400 * 300, 5000 * kSec, 1794312871LL * kSec, 30.0, true, false, 9, 0.5, 1024);
  auto r = run_pipeline(pk, 10.0, true);
  EXPECT_LT(r.max_err_ms, 1.0);
  EXPECT_LT(r.max_spacing_err_ms, 0.1);
  EXPECT_EQ(r.resets, 1);   // only the start: no false reboots at every counter wrap
}
