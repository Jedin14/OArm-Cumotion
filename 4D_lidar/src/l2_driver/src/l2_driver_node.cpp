// Copyright 2026. BSD-3-Clause.
//
// l2_driver_node: Unitree L2 driver with hardware-timestamp synchronisation.
//
// Differences to the stock unitree_lidar_ros2 node, each fixing a known failure:
//  * Packets are read on a dedicated thread in a tight loop, not one packet per
//    1 ms ROS timer tick (that backlog is what produces the "1000 IMU packets at
//    once, then 100 ms of silence" delivery).
//  * Stamps come from the lidar's own clock (use_system_timestamp = false), so a
//    burst of late packets keeps its true, even spacing. The device clock is
//    mapped onto host time by a lower-envelope estimator (ClockMapper) running on
//    CLOCK_MONOTONIC_RAW, so a device clock at 1970 / in the future / drifting,
//    and host NTP steps, cannot corrupt it.
//  * Device timeline glitches, duplicates, steps and lidar reboots are detected
//    with the packet sequence counter (TimelineGuard). IMU sample instants are
//    regularised from the sequence counter (SequenceTimeline) in case the
//    firmware stamps at send time.
//  * The L2's hardware clock runs at ~0.5x real time (measured on firmware 2.8.8.1);
//    DeviceRate measures the rate and every device time (stamps and per-point
//    time_increment) is rescaled to real seconds before mapping. The 10-bit packet
//    counter (wraps at 1024) is handled by TimelineGuard.
//  * Cloud and IMU share one clock mapping once verified, so per-point times and
//    IMU stamps are consistent with each other.
//  * Optionally waits for the host clock to be NTP/PTP synchronised before
//    publishing (RTC-less computers boot at 1970 and step later).
//  * The device orientation quaternion (which drifts in yaw and is published in
//    the wrong component order by the stock driver) is not exposed by default.

#include <pthread.h>
#include <sched.h>
#include <sys/timex.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

#include "unitree_lidar_sdk.h"

#include "l2_driver/clock_mapper.hpp"
#include "l2_driver/device_rate.hpp"
#include "l2_driver/sequence_timeline.hpp"
#include "l2_driver/timeline_guard.hpp"

namespace l2_driver
{

namespace sdk = unilidar_sdk2;

namespace
{

int64_t now_ns(clockid_t id)
{
  timespec ts{};
  clock_gettime(id, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

// CLOCK_REALTIME - CLOCK_MONOTONIC_RAW, sampled symmetrically.
int64_t wall_minus_raw()
{
  const int64_t r1 = now_ns(CLOCK_MONOTONIC_RAW);
  const int64_t w = now_ns(CLOCK_REALTIME);
  const int64_t r2 = now_ns(CLOCK_MONOTONIC_RAW);
  return w - (r1 + (r2 - r1) / 2);
}

// Same criterion as systemd's ntp_synced() (timedatectl "System clock synchronized"):
// the kernel's maximum error estimate is below 16 s. STA_UNSYNC is deliberately
// ignored because NTP daemons set it to stop the kernel from writing the RTC. On a
// boot without any time sync maxerror starts at 16 s, so an RTC-less 1970 boot
// reads as "not synchronised" until chrony/timesyncd/ptp4l disciplines the clock.
bool kernel_clock_synced()
{
  struct timex tx;
  std::memset(&tx, 0, sizeof(tx));
  if (adjtimex(&tx) < 0) {return false;}
  return tx.maxerror < 16000000L;
}

builtin_interfaces::msg::Time to_msg(int64_t ns)
{
  builtin_interfaces::msg::Time t;
  // floor division so negative values (host clock at 1970) still round-trip
  int64_t sec = ns / 1000000000LL;
  int64_t rem = ns % 1000000000LL;
  if (rem < 0) {rem += 1000000000LL; --sec;}
  t.sec = static_cast<int32_t>(sec);
  t.nanosec = static_cast<uint32_t>(rem);
  return t;
}

int64_t device_ns(const sdk::TimeStamp & s)
{
  return static_cast<int64_t>(s.sec) * 1000000000LL + static_cast<int64_t>(s.nsec);
}

diagnostic_msgs::msg::KeyValue kv(const std::string & k, const std::string & v)
{
  diagnostic_msgs::msg::KeyValue x;
  x.key = k;
  x.value = v;
  return x;
}

std::string fmt(double v, int prec = 3)
{
  char b[64];
  std::snprintf(b, sizeof(b), "%.*f", prec, v);
  return b;
}

// Rolling per-window statistics of one stream (reset at every diagnostics tick).
struct WindowStats
{
  uint64_t packets = 0;
  int64_t last_arrival = 0;
  double max_arrival_gap = 0.0;    // [s] longest silence between two arrivals
  uint64_t burst_arrivals = 0;     // arrivals < 25 % of the nominal period apart
  double max_residual = 0.0;       // [s] latency above the envelope (how late packets were)
  uint64_t published = 0;
  int64_t last_pub_stamp = 0;
  double pub_dt_min = 1e9, pub_dt_max = 0.0;  // [s] spacing of published stamps
  uint64_t non_monotonic = 0;

  void arrival(int64_t host_ns, double nominal_period)
  {
    if (last_arrival != 0) {
      const double gap = static_cast<double>(host_ns - last_arrival) * 1e-9;
      max_arrival_gap = std::max(max_arrival_gap, gap);
      if (nominal_period > 0.0 && gap < 0.25 * nominal_period) {++burst_arrivals;}
    }
    last_arrival = host_ns;
    ++packets;
  }

  void publish(int64_t stamp)
  {
    if (last_pub_stamp != 0) {
      const double dt = static_cast<double>(stamp - last_pub_stamp) * 1e-9;
      pub_dt_min = std::min(pub_dt_min, dt);
      pub_dt_max = std::max(pub_dt_max, dt);
    }
    last_pub_stamp = stamp;
    ++published;
  }

  void reset_window()
  {
    const int64_t la = last_arrival, lp = last_pub_stamp;
    *this = WindowStats();
    last_arrival = la;
    last_pub_stamp = lp;
  }
};

struct PointPacket
{
  std::vector<sdk::PointUnitree> points;
  int64_t duration_ns = 0;
  sdk::LidarInsideState state{};
};

}  // namespace

class L2DriverNode : public rclcpp::Node
{
public:
  explicit L2DriverNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node("l2_driver", options)
  {
    // ---- connection
    connection_ = declare_parameter<std::string>("connection", "udp");  // udp | serial
    serial_port_ = declare_parameter<std::string>("serial_port", "/dev/ttyACM0");
    baudrate_ = declare_parameter<int>("baudrate", 4000000);
    lidar_ip_ = declare_parameter<std::string>("lidar_ip", "192.168.1.62");
    lidar_port_ = declare_parameter<int>("lidar_port", 6101);
    local_ip_ = declare_parameter<std::string>("local_ip", "192.168.1.2");
    local_port_ = declare_parameter<int>("local_port", 6201);
    work_mode_ = declare_parameter<int>("work_mode", -1);  // -1: do not touch the lidar config
    start_rotation_ = declare_parameter<bool>("start_rotation", true);
    stop_rotation_on_exit_ = declare_parameter<bool>("stop_rotation_on_exit", true);
    // One-shot motor control instead of streaming: "start" | "stop" | "status" ("" = stream).
    command_ = declare_parameter<std::string>("command", "");
    sync_lidar_clock_ = declare_parameter<bool>("sync_lidar_clock_on_start", false);

    // ---- output
    frame_id_ = declare_parameter<std::string>("frame_id", "l2_lidar");
    imu_frame_id_ = declare_parameter<std::string>("imu_frame_id", "l2_imu");
    cloud_period_ = declare_parameter<double>("cloud_period", 0.1);
    range_min_ = declare_parameter<double>("range_min", 0.05);
    range_max_ = declare_parameter<double>("range_max", 60.0);
    publish_device_orientation_ = declare_parameter<bool>("publish_device_orientation", false);

    // ---- timing
    imu_timebase_ = declare_parameter<std::string>("imu_timebase", "sequence");  // sequence | device
    transport_latency_ = declare_parameter<double>("transport_latency", -1.0);  // -1: from SDK
    wait_for_clock_sync_ = declare_parameter<double>("wait_for_clock_sync", 30.0);
    shared_clock_tolerance_ = declare_parameter<double>("shared_clock_tolerance", 0.02);
    ClockMapperConfig mc;
    mc.max_drift_rate = declare_parameter<double>("clock.max_drift_ppm", 200.0) * 1e-6;
    mc.smoothing_time = declare_parameter<double>("clock.smoothing_time", 5.0);
    mc.lock_time = declare_parameter<double>("clock.lock_time", 1.0);
    imu_map_ = ClockMapper(mc);
    cloud_map_ = ClockMapper(mc);

    // ---- runtime
    idle_sleep_us_ = declare_parameter<int>("idle_sleep_us", 100);
    realtime_priority_ = declare_parameter<int>("realtime_priority", 0);
    log_period_ = declare_parameter<double>("log_period", 10.0);

    if (imu_timebase_ != "sequence" && imu_timebase_ != "device") {
      throw std::invalid_argument("imu_timebase must be 'sequence' or 'device'");
    }
    if (transport_latency_ >= 0.0) {
      latency_ns_ = static_cast<int64_t>(transport_latency_ * 1e9);
      latency_latched_ = true;
    }

    if (!command_.empty()) {
      open_lidar();
      one_shot_rc_ = run_command(command_);
      one_shot_ = true;
      return;
    }

    rclcpp::QoS imu_qos(rclcpp::KeepLast(400));
    rclcpp::QoS cloud_qos(rclcpp::KeepLast(10));
    pub_imu_ = create_publisher<sensor_msgs::msg::Imu>("l2/imu", imu_qos);
    pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("l2/cloud", cloud_qos);
    pub_diag_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);

    publish_static_tf();
    open_lidar();
    if (start_rotation_) {
      lsdk_->startLidarRotation();
      RCLCPP_INFO(get_logger(), "Start-rotation command sent.");
    }
    if (sync_lidar_clock_) {lsdk_->syncLidarTimeStamp();}
    streaming_ = true;

    start_wall_minus_raw_ = wall_minus_raw();
    last_wall_minus_raw_ = start_wall_minus_raw_;
    start_raw_ = now_ns(CLOCK_MONOTONIC_RAW);
    check_clock_gate(true);

    running_ = true;
    reader_ = std::thread([this] {reader_loop();});
    diag_timer_ = create_wall_timer(std::chrono::seconds(1), [this] {diagnostics_tick();});
  }

  ~L2DriverNode() override
  {
    running_ = false;
    if (reader_.joinable()) {reader_.join();}
    if (streaming_ && stop_rotation_on_exit_ && lsdk_) {
      // Reader thread is joined, so the SDK is ours alone again.
      send_twice(false);
      std::fprintf(stderr, "[l2_driver] Stop-rotation command sent (stop_rotation_on_exit).\n");
    }
    // Deliberately NOT calling lsdk_->closeUDP()/closeSerial(): in SDK 2.0.9
    // closeUDP() recurses infinitely in UDPHandler::~UDPHandler() and crashes.
    // The OS releases the socket / tty when the process exits.
  }

private:
  using ImuItem = TimelineGuard<sdk::LidarImuData>::Item;
  using PktItem = TimelineGuard<PointPacket>::Item;

  // ------------------------------------------------------------------ setup

  void publish_static_tf()
  {
    tf_static_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = now();
    t.header.frame_id = frame_id_;
    t.child_frame_id = imu_frame_id_;
    // IMU origin in the lidar frame, from the Unitree L2 documentation (axes parallel).
    t.transform.translation.x = -0.007698;
    t.transform.translation.y = -0.014655;
    t.transform.translation.z = 0.00667;
    t.transform.rotation.w = 1.0;
    tf_static_->sendTransform(t);
  }

  void open_lidar()
  {
    lsdk_ = sdk::createUnitreeLidarReader();
    int rc;
    // use_system_timestamp = false: keep the lidar's hardware timestamps.
    if (connection_ == "serial") {
      rc = lsdk_->initializeSerial(serial_port_, static_cast<uint32_t>(baudrate_), 18, false,
          static_cast<float>(range_min_), static_cast<float>(range_max_));
    } else if (connection_ == "udp") {
      rc = lsdk_->initializeUDP(static_cast<uint16_t>(lidar_port_), lidar_ip_,
          static_cast<uint16_t>(local_port_), local_ip_, 18, false,
          static_cast<float>(range_min_), static_cast<float>(range_max_));
    } else {
      throw std::invalid_argument("connection must be 'udp' or 'serial'");
    }
    if (rc != 0) {
      RCLCPP_FATAL(get_logger(), "Failed to open the L2 (%s). Check cabling / IP %s / port %s.",
        connection_.c_str(), local_ip_.c_str(), serial_port_.c_str());
      throw std::runtime_error("L2 initialisation failed");
    }
    RCLCPP_INFO(get_logger(), "L2 opened via %s (%s)", connection_.c_str(),
      connection_ == "udp" ?
      (lidar_ip_ + ":" + std::to_string(lidar_port_) + " -> " + local_ip_ + ":" +
      std::to_string(local_port_)).c_str() : serial_port_.c_str());

    if (work_mode_ >= 0) {
      RCLCPP_WARN(get_logger(), "Writing work_mode=%d to the lidar (persistent setting).", work_mode_);
      lsdk_->setLidarWorkMode(static_cast<uint32_t>(work_mode_));
    }
  }

  // Count packets for `seconds` (call only when the reader thread is not running).
  void count_packets(double seconds, int & imu, int & points)
  {
    imu = points = 0;
    const int64_t end = now_ns(CLOCK_MONOTONIC_RAW) + static_cast<int64_t>(seconds * 1e9);
    while (now_ns(CLOCK_MONOTONIC_RAW) < end) {
      const int r = lsdk_->runParse();
      if (r == LIDAR_IMU_DATA_PACKET_TYPE) {++imu;} else if (r == LIDAR_POINT_DATA_PACKET_TYPE) {
        ++points;
      } else if (r == 0) {std::this_thread::sleep_for(std::chrono::microseconds(200));}
    }
  }

  void send_twice(bool start)
  {
    for (int i = 0; i < 2; ++i) {  // idempotent; the repeat covers a lost first command
      if (start) {lsdk_->startLidarRotation();} else {lsdk_->stopLidarRotation();}
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }

  // Exit codes: 0 = done / spinning, 1 = command did not take effect, 3 = status: not spinning.
  int run_command(const std::string & cmd)
  {
    auto log = get_logger();
    int imu = 0, pts = 0;
    if (cmd == "status") {
      count_packets(2.0, imu, pts);
      if (pts > 0) {
        RCLCPP_INFO(log, "Lidar is spinning: %.0f point packets/s, %.0f IMU packets/s.", pts / 2.0, imu / 2.0);
        return 0;
      }
      RCLCPP_WARN(log, "Lidar is NOT streaming (%d IMU packets in 2 s): stopped/standby, not powered, "
        "or wrong port / work mode.", imu);
      return 3;
    }
    if (cmd == "start") {
      send_twice(true);
      for (int i = 0; i < 25; ++i) {  // spin-up from standby takes a while
        count_packets(1.0, imu, pts);
        if (pts > 0) {
          RCLCPP_INFO(log, "Lidar started: %d point packets/s.", pts);
          return 0;
        }
      }
      RCLCPP_ERROR(log, "Start command sent but no point data within 25 s (powered? serial mode?).");
      return 1;
    }
    if (cmd == "stop") {
      send_twice(false);
      count_packets(1.0, imu, pts);  // let the motor wind down, then verify
      count_packets(1.0, imu, pts);
      if (pts == 0) {
        RCLCPP_INFO(log, "Lidar stopped (standby, no point data).");
        return 0;
      }
      RCLCPP_ERROR(log, "Stop command sent but the lidar still sends point data (%d packets/s).", pts);
      return 1;
    }
    RCLCPP_ERROR(log, "Unknown command '%s' (use start | stop | status).", cmd.c_str());
    return 2;
  }

public:
  bool one_shot() const {return one_shot_;}
  int one_shot_rc() const {return one_shot_rc_;}

private:

  void set_thread_priority()
  {
    if (realtime_priority_ <= 0) {return;}
    sched_param sp{};
    sp.sched_priority = realtime_priority_;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
      RCLCPP_WARN(get_logger(), "Could not set SCHED_FIFO priority %d (needs rtprio limits / "
        "CAP_SYS_NICE). Continuing with normal priority.", realtime_priority_);
    } else {
      RCLCPP_INFO(get_logger(), "Reader thread running SCHED_FIFO priority %d", realtime_priority_);
    }
  }

  // ------------------------------------------------------------------ reader

  void reader_loop()
  {
    set_thread_priority();
    std::vector<ImuItem> imu_items;
    std::vector<PktItem> pkt_items;
    sdk::PointCloudUnitree parsed;
    int64_t last_poll = 0;
    while (running_ && rclcpp::ok()) {
      try {
        read_once(imu_items, pkt_items, parsed, last_poll);
      } catch (const std::exception & e) {
        if (!rclcpp::ok()) {break;}  // publishing during shutdown
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "reader: %s", e.what());
      }
    }
  }

  void read_once(std::vector<ImuItem> & imu_items, std::vector<PktItem> & pkt_items,
    sdk::PointCloudUnitree & parsed, int64_t & last_poll)
  {
    {
      const int r = lsdk_->runParse();
      if (r == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(idle_sleep_us_));
        return;
      }
      const int64_t host = now_ns(CLOCK_MONOTONIC_RAW);
      if (r == LIDAR_IMU_DATA_PACKET_TYPE) {
        const sdk::LidarImuData & d = lsdk_->getLidarImuDataPacket().data;
        std::lock_guard<std::mutex> lk(mtx_);
        imu_win_.arrival(host, imu_guard_.period_s() * rate_.rate());
        imu_items.clear();
        imu_guard_.push(d.info.seq, device_ns(d.info.stamp), host, d, imu_items);
        for (auto & it : imu_items) {handle_imu(it);}
      } else if (r == LIDAR_POINT_DATA_PACKET_TYPE) {
        const sdk::LidarPointDataPacket & pkt = lsdk_->getLidarPointDataPacket();
        sdk::parseFromPacketToPointCloud(parsed, pkt, false,
          static_cast<float>(range_min_), static_cast<float>(range_max_));
        PointPacket pp;
        pp.points = std::move(parsed.points);
        pp.duration_ns = static_cast<int64_t>(
          static_cast<double>(pkt.data.point_num) * pkt.data.time_increment * 1e9);
        pp.state = pkt.data.state;
        std::lock_guard<std::mutex> lk(mtx_);
        cloud_win_.arrival(host, pkt_guard_.period_s() * rate_.rate());
        pkt_items.clear();
        pkt_guard_.push(pkt.data.info.seq, device_ns(pkt.data.info.stamp), host, std::move(pp),
          pkt_items);
        for (auto & it : pkt_items) {handle_point_packet(it);}
      } else if (r == LIDAR_2D_POINT_DATA_PACKET_TYPE && !warned_2d_) {
        warned_2d_ = true;
        RCLCPP_WARN(get_logger(), "Lidar is in 2D mode; this driver publishes 3D data only.");
      }
      if (host - last_poll > 200000000LL) {  // 5 Hz: cheap SDK info queries
        last_poll = host;
        poll_sdk_info();
      }
    }
  }

  void poll_sdk_info()
  {
    if (!version_logged_) {
      std::string fw, hw, sdkv;
      if (lsdk_->getVersionOfLidarFirmware(fw)) {
        lsdk_->getVersionOfLidarHardware(hw);
        lsdk_->getVersionOfSDK(sdkv);
        RCLCPP_INFO(get_logger(), "L2 firmware %s, hardware %s, SDK %s", fw.c_str(), hw.c_str(),
          sdkv.c_str());
        std::lock_guard<std::mutex> lk(mtx_);
        firmware_ = fw;
        version_logged_ = true;
      }
    }
    double delay = 0.0;
    if (lsdk_->getTimeDelay(delay)) {
      std::lock_guard<std::mutex> lk(mtx_);
      sdk_delay_ = delay;
      if (!latency_latched_ && delay >= 0.0 && delay < 0.05) {
        latency_ns_ = static_cast<int64_t>(delay * 1e9);
        latency_latched_ = true;
        RCLCPP_INFO(get_logger(), "Transport latency compensation: %.3f ms (measured by SDK)",
          delay * 1e3);
      }
    }
    float dirty = 0.0f;
    if (lsdk_->getDirtyPercentage(dirty)) {
      std::lock_guard<std::mutex> lk(mtx_);
      dirty_ = dirty;
    }
  }

  // Wall-clock stamp for a host CLOCK_MONOTONIC_RAW instant (caller holds mtx_).
  int64_t to_wall(int64_t host_raw_ns)
  {
    const int64_t wmr = wall_minus_raw();
    if (std::llabs(wmr - last_wall_minus_raw_) > 1000000LL) {  // > 1 ms: CLOCK_REALTIME stepped
      ++wall_steps_;
      RCLCPP_WARN(get_logger(), "Host wall clock stepped by %+.3f s. Device->host mapping is "
        "unaffected; published stamps follow the new wall time.",
        static_cast<double>(wmr - last_wall_minus_raw_) * 1e-9);
    }
    last_wall_minus_raw_ = wmr;
    return host_raw_ns - latency_ns_ + wmr;
  }

  // ------------------------------------------------------------------ IMU

  void handle_imu(const ImuItem & it)
  {
    if (it.timeline_reset) {
      if (imu_map_.initialized()) {
        RCLCPP_WARN(get_logger(), "IMU device timeline discontinuity (lidar reboot or clock "
          "re-sync). Re-initialising the clock mapping.");
      }
      imu_map_.reset();
      imu_seq_.reset();
      rate_.rebase();
    }
    imu_last_host_ = it.host_ns;
    feed_rate(it.dev_ns, it.host_ns);
    // Sequence timebase only when the counter is verified to be a per-IMU-sample counter.
    const int64_t reg = imu_seq_.update(it.useq, it.dev_ns);
    const bool use_seq = imu_timebase_ == "sequence" && imu_guard_.seq_trusted() &&
      imu_guard_.unit_step_fraction() > 0.9;
    const int64_t dev = use_seq ? reg : it.dev_ns;
    imu_seq_in_use_ = use_seq && imu_seq_.healthy();
    const int64_t sdev = rate_.to_scaled(dev);   // device time rescaled to real seconds
    imu_map_.update(sdev, it.host_ns);
    imu_win_.max_residual = std::max(imu_win_.max_residual,
        static_cast<double>(imu_map_.last_residual_ns()) * 1e-9);

    if (!publishing_ || !rate_.ready() || !imu_map_.locked(it.host_ns)) {return;}
    int64_t stamp = to_wall(imu_map_.to_host(sdev));
    if (imu_win_.last_pub_stamp != 0 && stamp <= imu_win_.last_pub_stamp) {
      if (imu_win_.last_pub_stamp - stamp < 2000000LL) {
        stamp = imu_win_.last_pub_stamp + 1000;  // sub-2 ms wobble: keep strictly increasing
      } else {
        ++imu_win_.non_monotonic;
      }
    }
    imu_win_.publish(stamp);

    const sdk::LidarImuData & d = it.payload;
    sensor_msgs::msg::Imu m;
    m.header.stamp = to_msg(stamp);
    m.header.frame_id = imu_frame_id_;
    if (publish_device_orientation_) {
      // The L2 sends w, x, y, z (verified against the gravity vector).
      m.orientation.w = d.quaternion[0];
      m.orientation.x = d.quaternion[1];
      m.orientation.y = d.quaternion[2];
      m.orientation.z = d.quaternion[3];
    } else {
      m.orientation.w = 1.0;
      m.orientation_covariance[0] = -1.0;  // REP-145: orientation not provided
    }
    m.angular_velocity.x = d.angular_velocity[0];
    m.angular_velocity.y = d.angular_velocity[1];
    m.angular_velocity.z = d.angular_velocity[2];
    m.linear_acceleration.x = d.linear_acceleration[0];
    m.linear_acceleration.y = d.linear_acceleration[1];
    m.linear_acceleration.z = d.linear_acceleration[2];
    pub_imu_->publish(m);
  }

  // Learn the device clock rate (IMU stream; point packets only when there is no IMU).
  void feed_rate(int64_t dev, int64_t host)
  {
    if (rate_.update(dev, host)) {
      // Mappings learned with the provisional rate are discarded once the rate is known.
      imu_map_.reset();
      cloud_map_.reset();
      shared_clock_ = false;
      shared_votes_ = 0;
      const double k = rate_.rate();
      if (std::fabs(k - 1.0) > 0.01) {
        RCLCPP_WARN(get_logger(), "L2 hardware clock runs at %.4fx real time (firmware); all lidar "
          "timestamps and point times are rescaled by %.4f.", 1.0 / k, k);
      } else {
        RCLCPP_INFO(get_logger(), "L2 hardware clock rate %.6f (nominal).", 1.0 / k);
      }
    }
    imu_guard_.set_rate(rate_.rate());
    pkt_guard_.set_rate(rate_.rate());
  }

  // ------------------------------------------------------------------ cloud

  void handle_point_packet(PktItem & it)
  {
    if (it.timeline_reset) {
      flush_cloud();
      if (cloud_map_.initialized()) {
        RCLCPP_WARN(get_logger(), "Point packet device timeline discontinuity; re-initialising.");
      }
      cloud_map_.reset();
      if (it.host_ns - imu_last_host_ > 1000000000LL) {rate_.rebase();}
    }
    last_state_ = it.payload.state;
    if (it.host_ns - imu_last_host_ > 1000000000LL) {feed_rate(it.dev_ns, it.host_ns);}
    // Everything below in rescaled (real-time) device time.
    const double k = rate_.rate();
    const int64_t sdev = rate_.to_scaled(it.dev_ns);
    const int64_t end_s = sdev + static_cast<int64_t>(std::llround(k * static_cast<double>(it.payload.duration_ns)));
    // Observation for the cloud stream's own mapping: arrival vs end of the scan line.
    cloud_map_.update(end_s, it.host_ns);
    cloud_win_.max_residual = std::max(cloud_win_.max_residual,
        static_cast<double>(cloud_map_.last_residual_ns()) * 1e-9);

    const int64_t period_ns = static_cast<int64_t>(cloud_period_ * 1e9);
    if (cur_packets_ > 0 &&
      (sdev < cur_start_s_ || sdev - cur_last_s_ > period_ns))
    {
      flush_cloud();  // device-time gap (lost packets) or reordering: never stitch across it
    }
    if (cur_packets_ == 0) {
      cur_start_s_ = sdev;
      cur_points_.clear();
    }
    const double offset = static_cast<double>(sdev - cur_start_s_) * 1e-9;
    const uint16_t ring = static_cast<uint16_t>(cur_packets_);
    const float kf = static_cast<float>(k);   // per-point time_increment is in device time too
    for (const auto & p : it.payload.points) {
      cur_points_.push_back({p.x, p.y, p.z, p.intensity,
          static_cast<float>(offset) + kf * p.time, ring});
    }
    cur_last_s_ = sdev;
    ++cur_packets_;
    if (end_s - cur_start_s_ >= period_ns) {flush_cloud();}
  }

  void update_shared_clock(int64_t host_now)
  {
    if (!imu_map_.locked(host_now) || !cloud_map_.locked(host_now)) {
      shared_clock_ = false;
      shared_votes_ = 0;
      return;
    }
    const double diff = static_cast<double>(cloud_map_.offset_ns() - imu_map_.offset_ns()) * 1e-9;
    last_clock_diff_ = diff;
    const bool agree = std::fabs(diff) < shared_clock_tolerance_;
    if (agree == shared_clock_) {shared_votes_ = 0; return;}
    if (++shared_votes_ >= 10) {
      shared_clock_ = agree;
      shared_votes_ = 0;
      if (agree) {
        RCLCPP_INFO(get_logger(), "IMU and point packets share the device clock (offset "
          "difference %.3f ms): clouds use the IMU clock mapping.", diff * 1e3);
      } else {
        RCLCPP_WARN(get_logger(), "IMU and point packet clocks disagree by %.3f ms; mapping "
          "each stream independently.", diff * 1e3);
      }
    }
  }

  struct OutPoint
  {
    float x, y, z, intensity, time;
    uint16_t ring;
  };

  void flush_cloud()
  {
    const size_t n = cur_points_.size();
    const int packets = cur_packets_;
    cur_packets_ = 0;
    if (n == 0 || packets == 0) {return;}
    const int64_t host_now = now_ns(CLOCK_MONOTONIC_RAW);
    update_shared_clock(host_now);
    if (!publishing_ || !rate_.ready() || !cloud_map_.locked(host_now)) {return;}

    const ClockMapper & m = shared_clock_ ? imu_map_ : cloud_map_;
    const int64_t stamp = to_wall(m.to_host(cur_start_s_));
    cloud_win_.publish(stamp);
    cloud_points_ += n;

    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = to_msg(stamp);
    msg.header.frame_id = frame_id_;
    msg.height = 1;
    msg.width = static_cast<uint32_t>(n);
    msg.is_bigendian = false;
    msg.is_dense = true;
    msg.point_step = 24;
    msg.row_step = msg.point_step * msg.width;
    auto field = [&](const char * name, uint32_t off, uint8_t type) {
        sensor_msgs::msg::PointField f;
        f.name = name;
        f.offset = off;
        f.datatype = type;
        f.count = 1;
        msg.fields.push_back(f);
      };
    using PF = sensor_msgs::msg::PointField;
    field("x", 0, PF::FLOAT32);
    field("y", 4, PF::FLOAT32);
    field("z", 8, PF::FLOAT32);
    field("intensity", 12, PF::FLOAT32);
    field("time", 16, PF::FLOAT32);   // [s] relative to header.stamp
    field("ring", 20, PF::UINT16);    // scan-line index within this cloud
    msg.data.resize(msg.row_step);
    uint8_t * dst = msg.data.data();
    for (const auto & p : cur_points_) {
      std::memcpy(dst + 0, &p.x, 4);
      std::memcpy(dst + 4, &p.y, 4);
      std::memcpy(dst + 8, &p.z, 4);
      std::memcpy(dst + 12, &p.intensity, 4);
      std::memcpy(dst + 16, &p.time, 4);
      std::memcpy(dst + 20, &p.ring, 2);
      dst[22] = dst[23] = 0;
      dst += 24;
    }
    pub_cloud_->publish(std::move(msg));
  }

  // ------------------------------------------------------------------ host clock gate

  void check_clock_gate(bool first)
  {
    if (publishing_) {return;}
    const bool synced = kernel_clock_synced();
    const double waited = static_cast<double>(now_ns(CLOCK_MONOTONIC_RAW) - start_raw_) * 1e-9;
    if (wait_for_clock_sync_ <= 0.0 || synced) {
      publishing_ = true;
      if (!synced) {
        RCLCPP_WARN(get_logger(), "Host clock is NOT synchronised (no NTP/PTP). Publishing "
          "anyway (wait_for_clock_sync=0). Stamps will jump when it syncs.");
      }
      return;
    }
    if (waited >= wait_for_clock_sync_) {
      publishing_ = true;
      RCLCPP_WARN(get_logger(), "Host clock still not synchronised after %.0f s. Publishing; "
        "stamps will jump once NTP/PTP steps the clock (downstream nodes handle it). "
        "Install chrony to fix this permanently.", waited);
      return;
    }
    if (first) {
      RCLCPP_WARN(get_logger(), "Waiting up to %.0f s for the host clock to be synchronised "
        "(RTC-less computers start at 1970). Clock mapping is warming up meanwhile.",
        wait_for_clock_sync_);
    }
  }

  // ------------------------------------------------------------------ diagnostics

  void diagnostics_tick()
  {
    std::lock_guard<std::mutex> lk(mtx_);
    check_clock_gate(false);
    const int64_t host_now = now_ns(CLOCK_MONOTONIC_RAW);
    const double win = last_diag_raw_ == 0 ? 1.0 :
      static_cast<double>(host_now - last_diag_raw_) * 1e-9;
    last_diag_raw_ = host_now;
    const bool synced = kernel_clock_synced();

    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    using DS = diagnostic_msgs::msg::DiagnosticStatus;

    // IMU timing
    {
      DS s;
      s.name = "l2_driver: IMU timing";
      s.hardware_id = "unitree_l2";
      const auto & g = imu_guard_.stats();
      const double rate = static_cast<double>(imu_win_.packets) / win;
      s.level = DS::OK;
      s.message = "OK";
      if (imu_win_.packets == 0) {s.level = DS::ERROR; s.message = "no IMU packets";} else if (!rate_.ready() || !imu_map_.locked(host_now)) {
        s.level = DS::WARN; s.message = "clock rate / mapping warming up";
      } else if (g.resets > imu_resets_seen_) {s.level = DS::WARN; s.message = "device timeline reset";}
      imu_resets_seen_ = g.resets;
      s.values.push_back(kv("rate_hz", fmt(rate, 1)));
      s.values.push_back(kv("imu_period_ms", fmt(imu_guard_.period_s() * rate_.rate() * 1e3, 4)));
      s.values.push_back(kv("device_clock_speed", rate_.ready() ? fmt(1.0 / rate_.rate(), 5) :
        "measuring"));
      s.values.push_back(kv("seq_counter_modulus", imu_guard_.seq_modulus() ?
        std::to_string(imu_guard_.seq_modulus()) : "2^32"));
      s.values.push_back(kv("timebase", imu_seq_in_use_ ? "sequence (regularised)" :
        (imu_timebase_ == "sequence" ? "device stamps (sequence timebase not verified yet)" :
        "device stamps")));
      s.values.push_back(kv("seq_unit_step_fraction", fmt(imu_guard_.unit_step_fraction(), 4)));
      s.values.push_back(kv("device_stamp_lateness_p99_ms", fmt(imu_seq_.residual_p99_s() * 1e3, 3)));
      s.values.push_back(kv("arrival_max_gap_ms", fmt(imu_win_.max_arrival_gap * 1e3, 1)));
      s.values.push_back(kv("arrival_burst_fraction", fmt(imu_win_.packets ?
        static_cast<double>(imu_win_.burst_arrivals) / static_cast<double>(imu_win_.packets) : 0.0, 3)));
      s.values.push_back(kv("arrival_latency_max_ms", fmt(imu_win_.max_residual * 1e3, 2)));
      s.values.push_back(kv("published_dt_min_ms", fmt(imu_win_.published ? imu_win_.pub_dt_min * 1e3 : 0.0, 4)));
      s.values.push_back(kv("published_dt_max_ms", fmt(imu_win_.published ? imu_win_.pub_dt_max * 1e3 : 0.0, 4)));
      s.values.push_back(kv("clock_drift_ppm", fmt(imu_map_.drift_ppm(), 2)));
      s.values.push_back(kv("seq_trusted", imu_guard_.seq_trusted() ? "true" : "false"));
      s.values.push_back(kv("lost_total", std::to_string(g.lost)));
      s.values.push_back(kv("duplicates_total", std::to_string(g.duplicates)));
      s.values.push_back(kv("glitches_dropped_total", std::to_string(g.glitches)));
      s.values.push_back(kv("timeline_resets_total", std::to_string(g.resets)));
      s.values.push_back(kv("non_monotonic_total", std::to_string(imu_nonmono_ += imu_win_.non_monotonic)));
      arr.status.push_back(s);
    }
    // Cloud timing
    {
      DS s;
      s.name = "l2_driver: cloud timing";
      s.hardware_id = "unitree_l2";
      const auto & g = pkt_guard_.stats();
      const double rate = static_cast<double>(cloud_win_.published) / win;
      s.level = DS::OK;
      s.message = shared_clock_ ? "OK (shared clock with IMU)" : "OK";
      if (cloud_win_.packets == 0) {s.level = DS::ERROR; s.message = "no point packets";} else if (!shared_clock_ && imu_map_.locked(host_now)) {
        s.level = DS::WARN; s.message = "IMU / cloud clock not (yet) verified as shared";
      }
      s.values.push_back(kv("cloud_rate_hz", fmt(rate, 2)));
      s.values.push_back(kv("points_per_cloud", fmt(cloud_win_.published ?
        static_cast<double>(cloud_points_) / static_cast<double>(cloud_win_.published) : 0.0, 0)));
      s.values.push_back(kv("packet_rate_hz", fmt(static_cast<double>(cloud_win_.packets) / win, 1)));
      s.values.push_back(kv("imu_cloud_offset_diff_ms", fmt(last_clock_diff_ * 1e3, 3)));
      s.values.push_back(kv("arrival_max_gap_ms", fmt(cloud_win_.max_arrival_gap * 1e3, 1)));
      s.values.push_back(kv("lost_total", std::to_string(g.lost)));
      s.values.push_back(kv("glitches_dropped_total", std::to_string(g.glitches)));
      s.values.push_back(kv("timeline_resets_total", std::to_string(g.resets)));
      arr.status.push_back(s);
    }
    // Host clock + device health
    {
      DS s;
      s.name = "l2_driver: clock & device";
      s.hardware_id = "unitree_l2";
      s.level = synced ? DS::OK : DS::WARN;
      s.message = synced ? "host clock synchronised" : "host clock NOT synchronised (install chrony)";
      if (!publishing_) {s.message += "; waiting before publishing";}
      if (dirty_ > 10.0f) {s.level = std::max<uint8_t>(s.level, DS::WARN); s.message += "; lidar cover dirty";}
      s.values.push_back(kv("publishing", publishing_ ? "true" : "false"));
      s.values.push_back(kv("host_clock_synchronised", synced ? "true" : "false"));
      s.values.push_back(kv("host_wall_clock_steps", std::to_string(wall_steps_)));
      s.values.push_back(kv("transport_latency_comp_ms", fmt(static_cast<double>(latency_ns_) * 1e-6, 3)));
      s.values.push_back(kv("sdk_time_delay_ms", fmt(sdk_delay_ * 1e3, 3)));
      s.values.push_back(kv("device_clock_speed", rate_.ready() ? fmt(1.0 / rate_.rate(), 5) :
        "measuring"));
      s.values.push_back(kv("firmware", firmware_));
      s.values.push_back(kv("dirty_percentage", fmt(dirty_, 2)));
      s.values.push_back(kv("imu_temperature_c", fmt(last_state_.imu_temperature, 1)));
      s.values.push_back(kv("apd_temperature_c", fmt(last_state_.apd_temperature, 1)));
      s.values.push_back(kv("packet_lost_up", fmt(last_state_.packet_lost_up, 4)));
      s.values.push_back(kv("packet_lost_down", fmt(last_state_.packet_lost_down, 4)));
      arr.status.push_back(s);
    }
    pub_diag_->publish(arr);

    log_accum_ += win;
    if (log_period_ > 0.0 && log_accum_ >= log_period_) {
      log_accum_ = 0.0;
      RCLCPP_INFO(get_logger(),
        "IMU %.0f Hz (published dt %.3f..%.3f ms, arrival gaps up to %.0f ms, %s) | "
        "cloud %.1f Hz, %zu pts | drift %.1f ppm | lost imu %lu / pkt %lu | resets %lu",
        static_cast<double>(imu_win_.packets) / win,
        imu_win_.published ? imu_win_.pub_dt_min * 1e3 : 0.0,
        imu_win_.published ? imu_win_.pub_dt_max * 1e3 : 0.0,
        imu_win_.max_arrival_gap * 1e3,
        imu_seq_in_use_ ? "seq timebase" : "device timebase",
        static_cast<double>(cloud_win_.published) / win,
        cloud_win_.published ? static_cast<size_t>(cloud_points_ / cloud_win_.published) : 0u,
        imu_map_.drift_ppm(),
        static_cast<unsigned long>(imu_guard_.stats().lost),
        static_cast<unsigned long>(pkt_guard_.stats().lost),
        static_cast<unsigned long>(imu_guard_.stats().resets + pkt_guard_.stats().resets));
    }
    imu_win_.reset_window();
    cloud_win_.reset_window();
    cloud_points_ = 0;
  }

  // ------------------------------------------------------------------ members

  // parameters
  std::string connection_, serial_port_, lidar_ip_, local_ip_;
  int baudrate_{}, lidar_port_{}, local_port_{}, work_mode_{};
  bool start_rotation_{}, sync_lidar_clock_{}, stop_rotation_on_exit_{};
  std::string command_;
  bool one_shot_ = false, streaming_ = false;
  int one_shot_rc_ = 0;
  std::string frame_id_, imu_frame_id_, imu_timebase_;
  double cloud_period_{}, range_min_{}, range_max_{};
  bool publish_device_orientation_{};
  double transport_latency_{}, wait_for_clock_sync_{}, shared_clock_tolerance_{};
  int idle_sleep_us_{}, realtime_priority_{};
  double log_period_{};

  // ROS
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_diag_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  // SDK + thread
  sdk::UnitreeLidarReader * lsdk_ = nullptr;
  std::thread reader_;
  std::atomic<bool> running_{false};
  std::mutex mtx_;  // protects everything below

  // timing
  TimelineGuard<sdk::LidarImuData> imu_guard_;
  TimelineGuard<PointPacket> pkt_guard_;
  SequenceTimeline imu_seq_;
  DeviceRate rate_;
  int64_t imu_last_host_ = 0;
  bool imu_seq_in_use_ = false;
  ClockMapper imu_map_, cloud_map_;
  bool shared_clock_ = false;
  int shared_votes_ = 0;
  double last_clock_diff_ = 0.0;
  int64_t latency_ns_ = 0;
  bool latency_latched_ = false;
  double sdk_delay_ = 0.0;
  bool publishing_ = false;
  int64_t start_raw_ = 0, start_wall_minus_raw_ = 0, last_wall_minus_raw_ = 0;
  uint64_t wall_steps_ = 0;

  // cloud assembly
  std::vector<OutPoint> cur_points_;
  int cur_packets_ = 0;
  int64_t cur_start_s_ = 0, cur_last_s_ = 0;   // rescaled device time

  // stats
  WindowStats imu_win_, cloud_win_;
  uint64_t cloud_points_ = 0;
  uint64_t imu_resets_seen_ = 0, imu_nonmono_ = 0;
  int64_t last_diag_raw_ = 0;
  double log_accum_ = 0.0;
  sdk::LidarInsideState last_state_{};
  float dirty_ = 0.0f;
  std::string firmware_ = "unknown";
  bool version_logged_ = false;
  bool warned_2d_ = false;
};

}  // namespace l2_driver

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    auto node = std::make_shared<l2_driver::L2DriverNode>();
    if (node->one_shot()) {
      rc = node->one_shot_rc();
    } else {
      rclcpp::spin(node);
    }
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("l2_driver"), "%s", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
