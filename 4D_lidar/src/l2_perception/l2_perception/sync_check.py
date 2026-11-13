"""l2_sync_check: measure IMU / cloud timing quality of any L2 driver.

    ros2 run l2_perception l2_sync_check                       # our driver (l2/imu, l2/cloud)
    ros2 run l2_perception l2_sync_check --imu unilidar/imu --cloud unilidar/cloud   # stock driver

Reports, separately for *stamps* (what algorithms use) and *arrival* (what the
transport does):
  IMU   rate, stamp spacing jitter, non-monotonic stamps, arrival bursts / gaps
  cloud rate, stamp spacing, per-point time span, latency
  sync  whether IMU stamps cover every cloud's [first point, last point]
  clock stamp offset to host wall time (detects 1970 / future device clocks)
  vibration  gyro RMS and high-passed angle RMS (is vibration compensation worth it?)
"""

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, PointCloud2

from . import pointcloud as pc


def st(s):
    return s.sec + s.nanosec * 1e-9


class Check(Node):
    def __init__(self, imu_topic, cloud_topic):
        super().__init__("l2_sync_check")
        q = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1000,
                       reliability=ReliabilityPolicy.BEST_EFFORT)
        self.imu = []     # (stamp, arrival_wall, gx, gy, gz)
        self.cloud = []   # (stamp, arrival_wall, span, n)
        self.create_subscription(Imu, imu_topic, self.on_imu, q)
        self.create_subscription(PointCloud2, cloud_topic, self.on_cloud, q)

    def on_imu(self, m):
        g = m.angular_velocity
        self.imu.append((st(m.header.stamp), time.time(), g.x, g.y, g.z))

    def on_cloud(self, m):
        arr = pc.to_struct(m)
        tt = pc.field_or(arr, "time", 0.0)
        span = float(tt.max() - tt.min()) if tt.size else 0.0
        t0 = float(tt.min()) if tt.size else 0.0
        self.cloud.append((st(m.header.stamp) + t0, time.time(), span, arr.shape[0]))


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def report(imu, cloud, out=print):
    ok = True
    out("=" * 72)
    if len(imu) < 20:
        out("IMU: too few messages received"); return False
    I = np.array(imu)
    s, a = I[:, 0], I[:, 1]
    ds, da = np.diff(s), np.diff(a)
    med = float(np.median(ds))
    out(f"IMU  messages {len(I)}   rate (stamps) {1 / med:.1f} Hz")
    jitter = np.abs(ds - med)
    out(f"  stamp spacing   median {med * 1e3:.3f} ms   p99 |dev| {pct(jitter, 99) * 1e3:.3f} ms   "
        f"max |dev| {jitter.max() * 1e3:.3f} ms")
    back = int(np.count_nonzero(ds <= 0))
    out(f"  non-increasing stamps {back}")
    burst = float(np.mean(da < 0.25 * med))
    out(f"  arrival: max gap {da.max() * 1e3:.1f} ms, {100 * burst:.0f} % of messages arrive in bursts")
    bad_spacing = pct(jitter, 99) > 0.25 * med
    if back or bad_spacing:
        ok = False
        out("  FAIL  IMU stamps are not evenly spaced -> stamps reflect delivery, not sampling.\n"
            "        (stock driver with use_system_timestamp=true does this; use l2_driver)")
    else:
        out("  PASS  IMU stamps evenly spaced" + (" despite bursty arrival (that is fine)" if burst > 0.05 else ""))
    offset = float(np.median(a - s))
    out(f"  arrival - stamp: median {offset * 1e3:.2f} ms, p99 {pct(a - s, 99) * 1e3:.2f} ms")
    if abs(offset) > 1.0:
        ok = False
        out(f"  FAIL  stamps are {offset:+.1f} s away from host wall time "
            f"(device clock not mapped to host clock: 1970 / future timestamps)")

    # vibration estimate: leaky integral of the de-meaned gyro
    g = I[:, 2:5] - I[:, 2:5].mean(axis=0)
    th = np.zeros(3)
    ang = []
    for k in range(1, len(I)):
        dt = min(max(s[k] - s[k - 1], 0.0), 0.05)
        th = np.exp(-dt / 0.5) * th + g[k] * dt
        ang.append(np.linalg.norm(th))
    gr = np.rad2deg(np.sqrt(np.mean(np.sum(g ** 2, axis=1))))
    ar = np.rad2deg(np.sqrt(np.mean(np.square(ang[len(ang) // 4:])))) if ang else 0.0
    out(f"VIBRATION  gyro RMS {gr:.3f} deg/s, angular deflection RMS {ar:.4f} deg "
        f"(= {np.deg2rad(ar) * 10 * 100:.2f} cm at 10 m)\n"
        f"           note: on the L2 this includes a 4.3 Hz head-rotation signal that is not mount motion")

    if len(cloud) < 3:
        out("CLOUD: too few messages received"); return False
    C = np.array(cloud)
    cs, ca, span, n = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    dcs = np.diff(cs)
    out(f"CLOUD messages {len(C)}   rate {1 / np.median(dcs):.2f} Hz   points/cloud {np.median(n):.0f}   "
        f"point-time span {np.median(span) * 1e3:.1f} ms")
    if np.any(dcs <= 0):
        ok = False
        out(f"  FAIL  {int(np.count_nonzero(dcs <= 0))} non-increasing cloud stamps")
    lat = ca - (cs + span)
    out(f"  latency (arrival - last point): median {np.median(lat) * 1e3:.1f} ms, max {lat.max() * 1e3:.1f} ms")
    # sync: IMU coverage of each cloud
    covered = 0
    worst_gap = 0.0
    # only judge clouds inside the recorded IMU window (edges are recording artefacts)
    inside_rec = (cs >= s[0]) & (cs + span <= s[-1])
    edge = int(np.count_nonzero(~inside_rec))
    cs, span, ca = cs[inside_rec], span[inside_rec], ca[inside_rec]
    if cs.size == 0:
        out("SYNC  no cloud lies inside the recorded IMU time range -> IMU and cloud stamps do not overlap")
        out("RESULT: FAIL")
        return False
    for c0, sp in zip(cs, span):
        i0 = np.searchsorted(s, c0)
        i1 = np.searchsorted(s, c0 + sp)
        inside = s[max(0, i0 - 1):min(len(s), i1 + 1)]
        if inside.size >= 2 and inside[0] <= c0 and inside[-1] >= c0 + sp:
            covered += 1
            worst_gap = max(worst_gap, float(np.max(np.diff(inside))))
    frac = covered / len(cs)
    out(f"SYNC  {100 * frac:.0f} % of {len(cs)} clouds fully covered by IMU stamps "
        f"({edge} at the recording edges skipped), largest IMU gap inside a cloud {worst_gap * 1e3:.2f} ms")
    if abs(np.median(ca - cs) - np.median(a - s)) > 0.5:
        ok = False
        out("  FAIL  cloud and IMU stamps are on different clocks")
    elif frac < 0.9:
        ok = False
        out("  FAIL  IMU and cloud stamps do not overlap -> per-point IMU lookup impossible")
    else:
        out("  PASS  cloud points and IMU samples are on one consistent timeline")
    out("=" * 72)
    out("RESULT: " + ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imu", default="l2/imu")
    ap.add_argument("--cloud", default="l2/cloud")
    ap.add_argument("--duration", type=float, default=15.0)
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = Check(args.imu, args.cloud)
    print(f"Recording {args.imu} and {args.cloud} for {args.duration:.0f} s ...")
    end = time.time() + args.duration
    try:
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    ok = report(node.imu, node.cloud)
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
