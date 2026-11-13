"""l2_sim: synthetic stationary Unitree L2 for testing without hardware.

Publishes exactly what l2_driver publishes (l2/cloud with per-point 'time',
l2/imu with hardware-correct stamps), for a lidar mounted on a vibrating
bracket in a room:

* non-repetitive scan (random directions over the 180 deg hemisphere of the lidar)
* arbitrary mount (default: hanging upside-down at 3 m, 2 deg mounting error)
* rotational vibration (sum of sinusoids), gyro bias + noise, accel bias + noise
* IMU messages delivered in bursts (stamps stay correct, like the real driver)
* a person walking a loop from `person_start` and a box dropped at `box_start`
* optional host-clock step (`clock_step_at`) to exercise downstream recovery
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from builtin_interfaces.msg import Time
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Header

from . import pointcloud as pc
from .geometry import rotate_by_rotvec, rpy_to_matrix

G = 9.80665


def to_stamp(t: float) -> Time:
    s = int(np.floor(t))
    return Time(sec=s, nanosec=min(999999999, int(round((t - s) * 1e9))))


class Scene:
    def __init__(self, half_x=12.0, half_y=8.0):
        self.hx, self.hy = half_x, half_y
        self.static_boxes = [
            (np.array([-6.0, -4.0, 0.0]), np.array([-5.4, -3.4, 4.0])),   # pillar
            (np.array([5.0, 3.0, 0.0]), np.array([5.6, 3.6, 4.0])),       # pillar
            (np.array([-11.0, 5.0, 0.0]), np.array([-9.0, 7.5, 1.2])),    # shelf
            (np.array([8.0, -7.5, 0.0]), np.array([11.5, -6.5, 0.9])),    # bench
        ]

    def dynamic_boxes(self, t, person_start, box_start):
        out = []
        if t >= person_start:
            a = (t - person_start) * 0.25                      # 1 m/s on a 4 m radius loop
            c = np.array([2.0 + 4.0 * np.cos(a), 0.5 + 4.0 * np.sin(a)])
            out.append((np.array([c[0] - 0.25, c[1] - 0.25, 0.0]),
                        np.array([c[0] + 0.25, c[1] + 0.25, 1.75])))
        if t >= box_start:
            out.append((np.array([-3.0, 3.0, 0.0]), np.array([-2.2, 3.6, 1.0])))
        return out

    def raycast(self, o, d, boxes):
        """o (3,), d (N,3) unit -> range (N,) (inf where nothing is hit)."""
        n = d.shape[0]
        r = np.full(n, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for axis, val in ((2, 0.0), (0, self.hx), (0, -self.hx), (1, self.hy), (1, -self.hy)):
                t = (val - o[axis]) / d[:, axis]
                r = np.where((t > 0) & (t < r), t, r)
            for lo, hi in self.static_boxes + boxes:
                t1 = (lo - o) / d
                t2 = (hi - o) / d
                tn = np.nanmax(np.minimum(t1, t2), axis=1)
                tf = np.nanmin(np.maximum(t1, t2), axis=1)
                hit = (tn <= tf) & (tn > 0)
                r = np.where(hit & (tn < r), tn, r)
        return r


class L2Sim(Node):
    def __init__(self):
        super().__init__("l2_sim")
        P = self.declare_parameter
        self.frame = P("frame_id", "l2_lidar").value
        self.imu_rate = P("imu_rate", 400.0).value
        self.cloud_period = P("cloud_period", 0.1).value
        self.pps = P("points_per_second", 43000.0).value
        self.mount_t = np.array(P("mount.xyz", [0.0, 0.0, 3.0]).value, dtype=float)
        rpy = np.deg2rad(P("mount.rpy_deg", [182.0, -3.0, 25.0]).value)
        self.R_mount = rpy_to_matrix(*rpy)
        self.vib_amp = np.deg2rad(np.array(P("vibration.amplitude_deg", [0.15, 0.10, 0.05]).value))
        self.vib_freq = np.array(P("vibration.frequency_hz", [7.0, 12.0, 3.0]).value)
        self.gyro_bias = np.deg2rad(np.array(P("imu.gyro_bias_dps", [0.3, -0.2, 0.1]).value))
        self.gyro_noise = P("imu.gyro_noise", 0.003).value
        self.acc_bias = np.array(P("imu.accel_bias", [0.10, -0.05, 0.08]).value)
        self.acc_noise = P("imu.accel_noise", 0.02).value
        self.range_noise = P("range_noise", 0.015).value
        self.bursty = P("burst_delivery", True).value
        self.person_start = P("person_start", 40.0).value
        self.box_start = P("box_start", 50.0).value
        self.step_at = P("clock_step_at", -1.0).value
        self.step = P("clock_step", 0.0).value
        self.rng = np.random.default_rng(P("seed", 1).value)
        self.phase = self.rng.uniform(0, 2 * np.pi, 3)
        self.scene = Scene()

        self.pub_cloud = self.create_publisher(PointCloud2, "l2/cloud", 10)
        self.pub_imu = self.create_publisher(Imu, "l2/imu", 400)
        self.t0 = self.get_clock().now().nanoseconds * 1e-9
        self.imu_next = 0.0
        self.cloud_next = 0.0
        self.pending_imu = []
        self.next_burst = 0.0
        self.create_timer(0.005, self.tick)
        self.get_logger().info("L2 simulator running (inverted mount, vibration, bursty IMU)")

    def vib(self, t):
        """Deflection rotation vector (N,3) and body rate (N,3) at sim times t (N,)."""
        arg = 2 * np.pi * self.vib_freq[None, :] * t[:, None] + self.phase[None, :]
        v = self.vib_amp[None, :] * np.sin(arg)
        w = self.vib_amp[None, :] * 2 * np.pi * self.vib_freq[None, :] * np.cos(arg)
        return v, w

    def stamp(self, t):
        return self.t0 + t + (self.step if 0 <= self.step_at <= t else 0.0)

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        # ---------------- IMU
        dt = 1.0 / self.imu_rate
        while self.imu_next <= now:
            t = self.imu_next
            v, w = self.vib(np.array([t]))
            gyro = w[0] + self.gyro_bias + self.rng.normal(0, self.gyro_noise, 3)
            R_wl = self.R_mount @ _rv(v[0])
            acc = R_wl.T @ np.array([0, 0, G]) + self.acc_bias + self.rng.normal(0, self.acc_noise, 3)
            m = Imu()
            m.header = Header(stamp=to_stamp(self.stamp(t)), frame_id="l2_imu")
            m.orientation.w = 1.0
            m.orientation_covariance[0] = -1.0
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, gyro)
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, acc)
            self.pending_imu.append(m)
            self.imu_next += dt
        if not self.bursty or now >= self.next_burst:
            for m in self.pending_imu:
                self.pub_imu.publish(m)
            self.pending_imu = []
            self.next_burst = now + self.rng.uniform(0.02, 0.1)
        # ---------------- cloud
        while self.cloud_next + self.cloud_period <= now:
            self.publish_cloud(self.cloud_next)
            self.cloud_next += self.cloud_period

    def publish_cloud(self, tc):
        n = int(self.pps * self.cloud_period)
        tt = np.sort(self.rng.uniform(0, self.cloud_period, n))
        # random directions on the lidar's +z hemisphere
        d = self.rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d[:, 2] = np.abs(d[:, 2])
        v, _ = self.vib(tc + tt)
        # actual ray direction in world at each point's instant: R_mount * Exp(v) * d
        d_w = rotate_by_rotvec(d, v) @ self.R_mount.T
        r = self.scene.raycast(self.mount_t, d_w, self.scene.dynamic_boxes(tc, self.person_start,
                                                                           self.box_start))
        ok = np.isfinite(r) & (r < 40.0) & (r > 0.05)
        r = r[ok] + self.rng.normal(0, self.range_noise, np.count_nonzero(ok))
        p = d[ok] * r[:, None]            # measured in the (vibrating) lidar frame
        tt = tt[ok]
        h = Header(stamp=to_stamp(self.stamp(tc)), frame_id=self.frame)
        self.pub_cloud.publish(pc.make_cloud(h, {
            "x": p[:, 0].astype(np.float32), "y": p[:, 1].astype(np.float32),
            "z": p[:, 2].astype(np.float32),
            "intensity": np.full(p.shape[0], 100.0, np.float32),
            "time": tt.astype(np.float32),
            "ring": np.zeros(p.shape[0], np.uint16)}))


def _rv(v):
    from .geometry import rotvec_to_matrix
    return rotvec_to_matrix(v)


def main():
    rclpy.init()
    node = L2Sim()
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
