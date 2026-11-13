"""l2_stabilizer: turns the raw L2 cloud into a drift-free, levelled, vibration-
compensated cloud in a fixed world frame.

    l2/cloud (lidar frame) + l2/imu  ->  l2/points (world frame)

* Pose = constant mount calibration (loaded from file, or estimated once from
  gravity + ground plane and saved). No odometry, no SLAM: nothing integrates,
  so nothing can drift.
* Vibration: per-point de-rotation with a leaky (bounded, self-centring) gyro
  integral, looked up at each point's own timestamp. Clouds wait (max_imu_wait)
  for IMU samples covering their last point, so the IMU/cloud stamps are
  matched in time, never by arrival order.
* Stamp guards: duplicate / backwards / jumping stamps (host clock step, driver
  restart, bag loop) reset the vibration state instead of corrupting it.
"""

import os
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from . import pointcloud as pc
from .geometry import matrix_to_quaternion
from .mount import MountCalibration, calibrate, tilt_change_deg
from .stamp_guard import StampGuard
from .vibration import VibrationCompensator, VibrationConfig


def stamp_to_sec(s) -> float:
    return s.sec + s.nanosec * 1e-9


def sec_to_stamp(t: float):
    from builtin_interfaces.msg import Time
    sec = int(np.floor(t))
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1000000000)


class Stabilizer(Node):
    def __init__(self):
        super().__init__("l2_stabilizer")
        P = self.declare_parameter
        self.world_frame = P("world_frame", "world").value
        self.lidar_frame = P("lidar_frame", "l2_lidar").value
        self.cal_file = os.path.expanduser(P("calibration_file", "~/.ros/l2/mount.yaml").value)
        self.auto_calibrate = P("auto_calibrate", True).value
        self.cal_duration = P("calibration.duration", 3.0).value
        self.cal_reference = P("calibration.level_reference", "ground").value
        self.cal_yaw = np.deg2rad(P("calibration.yaw_deg", 0.0).value)
        self.cal_x = P("calibration.x", 0.0).value
        self.cal_y = P("calibration.y", 0.0).value
        self.cal_mount_height = P("calibration.mount_height", -1.0).value
        self.cal_ground_distance = P("calibration.ground_distance", 0.04).value
        self.cal_ground_tilt = P("calibration.ground_max_tilt_deg", 10.0).value
        self.monitor_deg = P("mount_monitor.threshold_deg", 0.5).value

        self.range_min = P("filter.range_min", 0.3).value
        self.range_max = P("filter.range_max", 40.0).value
        self.crop_min = np.array(P("filter.crop_min", [-1e9, -1e9, -1e9]).value, dtype=float)
        self.crop_max = np.array(P("filter.crop_max", [1e9, 1e9, 1e9]).value, dtype=float)

        vc = VibrationConfig(
            enabled=P("vibration.enabled", False).value,   # see config/l2.yaml: off on the real L2
            highpass_time=P("vibration.highpass_time", 0.5).value,
            bias_time=P("vibration.bias_time", 30.0).value,
            bias_init_time=P("vibration.bias_init_time", 1.0).value,
            max_angle_deg=P("vibration.max_angle_deg", 2.0).value,
            max_gap=P("vibration.max_imu_gap", 0.05).value,
        )
        self.max_imu_wait = P("max_imu_wait", 0.06).value
        self.vib = VibrationCompensator(vc)

        self.cal = MountCalibration.load(self.cal_file)
        if self.cal is not None:
            self.get_logger().info(
                f"Loaded mount calibration {self.cal_file}: roll {np.rad2deg(self.cal.roll):.2f} "
                f"pitch {np.rad2deg(self.cal.pitch):.2f} yaw {np.rad2deg(self.cal.yaw):.2f} deg, "
                f"height {self.cal.z:.3f} m ({self.cal.source}, {self.cal.created})")
        self._calibrating = self.cal is None and self.auto_calibrate
        if self.cal is None and not self.auto_calibrate:
            self.get_logger().error(f"No calibration at {self.cal_file} and auto_calibrate=false; "
                                    "call ~/calibrate.")
        self._cal_acc, self._cal_pts, self._cal_t0 = [], [], None
        if self._calibrating:
            self.get_logger().warn(f"No mount calibration found: calibrating for {self.cal_duration:.0f} s. "
                                   "Keep the scene static and the floor visible.")

        self.imu_guard = StampGuard("imu", max_jump=1.0)
        self.cloud_guard = StampGuard("cloud", max_jump=2.0)
        self.queue = deque()
        self.acc_lp = None
        self.stats = dict(clouds_in=0, clouds_out=0, late_imu=0, no_imu=0, points_out=0,
                          resets=0, tilt=0.0, vib_deg=0.0, proc_ms=0.0)
        self.last_imu_wall = 0.0

        qos_imu = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=400,
                             reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_cloud = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, P("imu_topic", "l2/imu").value, self.on_imu, qos_imu)
        self.create_subscription(PointCloud2, P("cloud_topic", "l2/cloud").value, self.on_cloud,
                                 qos_cloud)
        self.pub = self.create_publisher(PointCloud2, P("output_topic", "l2/points").value, 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.tf = StaticTransformBroadcaster(self)
        self.create_service(Trigger, "~/calibrate", self.srv_calibrate)
        self.create_timer(0.02, self.flush)
        self.create_timer(1.0, self.diagnostics)
        if self.cal is not None:
            self.publish_tf()

    # ------------------------------------------------------------------ inputs
    def on_imu(self, m: Imu):
        t = stamp_to_sec(m.header.stamp)
        st = self.imu_guard.check(t)
        if st == "duplicate":
            return
        if st in ("backward", "jump"):
            self.get_logger().warn(f"IMU stamp {st} ({self.imu_guard.last_delta:+.3f} s): "
                                   "resetting vibration state")
            self.vib.reset(full=False)
            self.queue.clear()
            self.stats["resets"] += 1
        g = m.angular_velocity
        a = m.linear_acceleration
        acc = np.array([a.x, a.y, a.z])
        self.vib.add_imu(t, (g.x, g.y, g.z))
        self.acc_lp = acc if self.acc_lp is None else self.acc_lp + 0.002 * (acc - self.acc_lp)
        self.last_imu_wall = time.monotonic()
        if self._calibrating:
            self._cal_acc.append(acc)

    def on_cloud(self, m: PointCloud2):
        t = stamp_to_sec(m.header.stamp)
        st = self.cloud_guard.check(t)
        if st == "duplicate":
            return
        if st in ("backward", "jump"):
            self.get_logger().warn(f"cloud stamp {st} ({self.cloud_guard.last_delta:+.3f} s): resetting")
            self.queue.clear()
            self.stats["resets"] += 1
        arr = pc.to_struct(m)
        pts = pc.xyz(arr)
        times = pc.field_or(arr, "time", 0.0)
        inten = pc.field_or(arr, "intensity", 0.0)
        self.stats["clouds_in"] += 1
        if self._calibrating:
            self._accumulate_calibration(t, pts)
            return
        if self.cal is None:
            return
        self.queue.append((t, t + (float(times.max()) if times.size else 0.0), time.monotonic(),
                           m.header, pts, times, inten))
        self.flush()

    # ------------------------------------------------------------------ calibration
    def _accumulate_calibration(self, t, pts):
        r = np.linalg.norm(pts, axis=1)
        self._cal_pts.append(pts[(r > self.range_min) & (r < self.range_max)])
        if self._cal_t0 is None:
            self._cal_t0 = t
        if t - self._cal_t0 < self.cal_duration:
            return
        if len(self._cal_acc) < 50:
            self.get_logger().warn("Calibration: not enough IMU samples yet, extending.")
            return
        acc = np.mean(np.array(self._cal_acc), axis=0)
        cloud = np.concatenate(self._cal_pts) if self._cal_pts else np.zeros((0, 3))
        cal = calibrate(acc, cloud, level_reference=self.cal_reference, yaw=self.cal_yaw,
                        x=self.cal_x, y=self.cal_y, mount_height=self.cal_mount_height,
                        ground_distance=self.cal_ground_distance,
                        ground_max_tilt_deg=self.cal_ground_tilt,
                        log=self.get_logger().info)
        self.cal = cal
        self._calibrating = False
        self._cal_acc, self._cal_pts, self._cal_t0 = [], [], None
        try:
            cal.save(self.cal_file)
            saved = f"saved to {self.cal_file}"
        except OSError as e:
            saved = f"NOT saved ({e})"
        self.get_logger().info(
            f"Mount calibrated ({cal.source}): roll {np.rad2deg(cal.roll):.2f} pitch "
            f"{np.rad2deg(cal.pitch):.2f} deg, height {cal.z:.3f} m; {saved}")
        self.publish_tf()

    def srv_calibrate(self, req, res):
        self._calibrating = True
        self._cal_acc, self._cal_pts, self._cal_t0 = [], [], None
        res.success = True
        res.message = f"recalibrating for {self.cal_duration:.1f} s; result -> {self.cal_file}"
        self.get_logger().info(res.message)
        return res

    def publish_tf(self):
        tfm = TransformStamped()
        tfm.header.stamp = self.get_clock().now().to_msg()
        tfm.header.frame_id = self.world_frame
        tfm.child_frame_id = self.lidar_frame
        t = self.cal.translation()
        q = matrix_to_quaternion(self.cal.rotation())
        tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z = map(float, t)
        (tfm.transform.rotation.x, tfm.transform.rotation.y,
         tfm.transform.rotation.z, tfm.transform.rotation.w) = map(float, q)
        self.tf.sendTransform(tfm)

    # ------------------------------------------------------------------ processing
    def flush(self):
        now = time.monotonic()
        imu_alive = now - self.last_imu_wall < 0.5
        while self.queue:
            t0, t_end, recv, header, pts, times, inten = self.queue[0]
            cov = self.vib.coverage_end()
            covered = cov is not None and cov >= t_end
            waited = now - recv
            if self.vib.cfg.enabled and imu_alive and not covered and waited < self.max_imu_wait:
                return  # wait for IMU samples covering the whole cloud
            self.queue.popleft()
            if self.vib.cfg.enabled:
                if not imu_alive or not self.vib.ready:
                    self.stats["no_imu"] += 1
                elif not covered:
                    self.stats["late_imu"] += 1
            self.process(t0, header, pts, times, inten)

    def process(self, t0, header, pts, times, inten):
        tic = time.perf_counter()
        r = np.linalg.norm(pts, axis=1)
        keep = (r >= self.range_min) & (r <= self.range_max)
        pts, times, inten = pts[keep], times[keep], inten[keep]
        pts = self.vib.compensate(pts, t0 + times)
        w = self.cal.apply(pts)
        box = np.all((w >= self.crop_min) & (w <= self.crop_max), axis=1)
        w, times, inten = w[box], times[box], inten[box]
        h = Header(stamp=header.stamp, frame_id=self.world_frame)
        self.pub.publish(pc.make_cloud(h, {
            "x": w[:, 0].astype(np.float32), "y": w[:, 1].astype(np.float32),
            "z": w[:, 2].astype(np.float32), "intensity": inten.astype(np.float32),
            "time": times.astype(np.float32)}))
        self.stats["clouds_out"] += 1
        self.stats["points_out"] += int(w.shape[0])
        self.stats["proc_ms"] = max(self.stats["proc_ms"], (time.perf_counter() - tic) * 1e3)

    # ------------------------------------------------------------------ diagnostics
    def diagnostics(self):
        s = self.stats
        st = DiagnosticStatus(name="l2_stabilizer", hardware_id="unitree_l2")
        st.level = DiagnosticStatus.OK
        msgs = []
        if self._calibrating:
            st.level = DiagnosticStatus.WARN
            msgs.append("calibrating mount")
        elif self.cal is None:
            st.level = DiagnosticStatus.ERROR
            msgs.append("no mount calibration")
        if s["clouds_in"] == 0:
            st.level = DiagnosticStatus.ERROR
            msgs.append("no input cloud")
        if self.vib.cfg.enabled and s["no_imu"]:
            st.level = max(st.level, DiagnosticStatus.WARN)
            msgs.append("clouds without IMU (vibration not compensated)")
        tilt = 0.0
        if self.cal is not None and self.acc_lp is not None:
            tilt = tilt_change_deg(self.cal, self.acc_lp)
            if tilt > self.monitor_deg:
                st.level = max(st.level, DiagnosticStatus.WARN)
                msgs.append(f"mount tilted {tilt:.2f} deg since calibration - recalibrate")
        st.message = "; ".join(msgs) if msgs else "OK"
        vib_deg = self.vib.recent_amplitude_deg(1.0)
        kv = lambda k, v: KeyValue(key=k, value=str(v))
        st.values = [
            kv("clouds_in", s["clouds_in"]), kv("clouds_out", s["clouds_out"]),
            kv("points_per_cloud", s["points_out"] // max(1, s["clouds_out"])),
            kv("clouds_waited_past_imu_deadline", s["late_imu"]),
            kv("clouds_without_imu", s["no_imu"]),
            kv("vibration_rms_deg", f"{vib_deg:.4f}"),
            kv("gyro_bias_dps", np.array2string(np.rad2deg(self.vib.bias), precision=3)),
            kv("vibration_resets", self.vib.resets), kv("stamp_resets", s["resets"]),
            kv("mount_tilt_change_deg", f"{tilt:.3f}"),
            kv("max_processing_ms", f"{s['proc_ms']:.1f}"),
            kv("calibration", "none" if self.cal is None else
               f"{self.cal.source} roll {np.rad2deg(self.cal.roll):.2f} pitch "
               f"{np.rad2deg(self.cal.pitch):.2f} z {self.cal.z:.3f}"),
        ]
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [st]
        self.pub_diag.publish(arr)
        for k in ("clouds_in", "clouds_out", "late_imu", "no_imu", "points_out", "proc_ms"):
            s[k] = 0


def main():
    rclpy.init()
    node = Stabilizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise  # a real error while running
        # else: context already shut down by Ctrl+C/SIGTERM (rclpy Humble shutdown race)
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:  # second Ctrl-C while shutting down
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
