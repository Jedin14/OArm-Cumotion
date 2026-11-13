"""l2_obstacles: object / obstacle detection for a stationary L2.

    l2/points (world frame, from l2_stabilizer)
      -> l2/obstacles           (l2_msgs/ObstacleArray)
      -> l2/obstacles/markers   (visualization_msgs/MarkerArray)
      -> l2/obstacles/points    (foreground PointCloud2)

mode "background" (default): learn the static scene once (or load it), report
    everything that is not part of it. Best for monitoring a fixed area.
mode "geometric": report every cluster above the ground inside the ROI, static
    or not. Best when the "obstacles" are permanent structures too.

The L2's non-repetitive scan is sparse per cloud, so clouds are accumulated
over a short window (accumulate_time). Because the sensor is static and the
stabilizer removes vibration, accumulation does not smear static geometry.
"""

import hashlib
import os
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, Vector3
from l2_msgs.msg import Obstacle, ObstacleArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from . import pointcloud as pc
from .background import VoxelBackground
from .detection import Tracker, cluster
from .mount import MountCalibration
from .stamp_guard import StampGuard


def stamp_to_sec(s) -> float:
    return s.sec + s.nanosec * 1e-9


def mount_signature(cal: MountCalibration) -> str:
    if cal is None:
        return ""
    v = np.round([cal.roll, cal.pitch, cal.yaw, cal.x, cal.y, cal.z], 4)
    return hashlib.sha1(v.tobytes()).hexdigest()[:16]


class ObstacleDetector(Node):
    def __init__(self):
        super().__init__("l2_obstacles")
        P = self.declare_parameter
        self.mode = P("mode", "background").value
        self.window = P("accumulate_time", 0.3).value
        self.roi_min = np.array(P("roi.min", [-30.0, -30.0, -1.0]).value, dtype=float)
        self.roi_max = np.array(P("roi.max", [30.0, 30.0, 3.5]).value, dtype=float)
        self.ground_z = P("ground.z_max", 0.10).value
        self.ground_filter = P("ground.enabled", True).value

        self.bg = VoxelBackground(
            voxel_size=P("background.voxel_size", 0.10).value,
            slice_time=P("background.slice_time", 1.0).value,
            min_slices=P("background.min_slices", 2).value,
            dilation=P("background.dilation", 1).value,
            min_span=P("background.min_span", 0.5).value)
        self.bg_file = os.path.expanduser(P("background.file", "~/.ros/l2/background.npz").value)
        self.bg_learn_time = P("background.learn_time", 30.0).value
        self.bg_load = P("background.load_on_start", True).value
        self.bg_auto_learn = P("background.auto_learn", True).value

        self.cl_voxel = P("cluster.voxel_size", 0.20).value
        self.cl_min_points = P("cluster.min_points", 8).value
        self.cl_max_extent = P("cluster.max_extent", 15.0).value
        self.tracker = Tracker(gate=P("tracker.gate", 1.0).value,
                               min_hits=P("tracker.min_hits", 3).value,
                               max_missed_time=P("tracker.max_missed_time", 0.5).value)
        self.publish_unconfirmed = P("tracker.publish_unconfirmed", False).value
        self.marker_labels = P("markers.labels", False).value   # text "#id range" above boxes
        self.cal_file = os.path.expanduser(P("calibration_file", "~/.ros/l2/mount.yaml").value)
        self.cal, self.cal_mtime = None, None
        self.sensor_origin = np.zeros(3)
        self._refresh_calibration(startup=True)

        if self.mode == "background":
            why = self.bg.load(self.bg_file, mount_signature(self.cal)) if self.bg_load else "disabled"
            if why == "":
                self.get_logger().info(f"Loaded background model {self.bg_file} "
                                       f"({self.bg.keys.size} voxels, {self.bg.meta.get('created')})")
            elif self.bg_auto_learn:
                self.get_logger().warn(f"Background not loaded ({why}). Learning for "
                                       f"{self.bg_learn_time:.0f} s - keep moving objects out of view.")
                self.bg.start_learning(self.bg_learn_time)
            else:
                self.get_logger().error(f"Background not loaded ({why}); call ~/learn_background")
        elif self.mode != "geometric":
            raise ValueError("mode must be 'background' or 'geometric'")

        self.frames = deque()
        self.guard = StampGuard("points", max_jump=2.0)
        self.stats = dict(frames=0, proc_ms=0.0, clusters=0, fg_points=0)

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PointCloud2, P("input_topic", "l2/points").value, self.on_points, qos)
        self.pub = self.create_publisher(ObstacleArray, "l2/obstacles", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "l2/obstacles/markers", 10)
        self.pub_fg = self.create_publisher(PointCloud2, "l2/obstacles/points", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.create_service(Trigger, "~/learn_background", self.srv_learn)
        self.create_service(Trigger, "~/save_background", self.srv_save)
        self.create_timer(1.0, self.diagnostics)

    # ------------------------------------------------------------------ calibration watch
    def _refresh_calibration(self, startup=False):
        """(Re)load the mount calibration written by l2_stabilizer when the file changes."""
        try:
            mtime = os.path.getmtime(self.cal_file)
        except OSError:
            return
        if mtime == self.cal_mtime:
            return
        old_sig = mount_signature(self.cal)
        self.cal = MountCalibration.load(self.cal_file)
        self.cal_mtime = mtime
        if self.cal is None:
            return
        self.sensor_origin = self.cal.translation()
        new_sig = mount_signature(self.cal)
        if not startup and old_sig and new_sig != old_sig and self.mode == "background":
            self.get_logger().warn("Mount calibration changed: the background model is invalid, "
                                   + ("re-learning." if self.bg_auto_learn else "call ~/learn_background."))
            self.tracker.reset()
            if self.bg_auto_learn:
                self.bg.start_learning(self.bg_learn_time)
            else:
                self.bg.keys = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------ services
    def srv_learn(self, req, res):
        self.bg.start_learning(self.bg_learn_time)
        self.tracker.reset()
        res.success = True
        res.message = f"learning background for {self.bg_learn_time:.0f} s"
        self.get_logger().info(res.message)
        return res

    def srv_save(self, req, res):
        if not self.bg.ready:
            res.success, res.message = False, "no background model"
            return res
        self.bg.save(self.bg_file, mount_signature(self.cal))
        res.success, res.message = True, f"saved {self.bg_file}"
        return res

    # ------------------------------------------------------------------ main loop
    def on_points(self, m: PointCloud2):
        tic = time.perf_counter()
        t = stamp_to_sec(m.header.stamp)
        st = self.guard.check(t)
        if st == "duplicate":
            return
        if st in ("backward", "jump"):
            self.get_logger().warn(f"point stamps {st} ({self.guard.last_delta:+.2f} s); "
                                   "clearing window and tracks")
            self.frames.clear()
            self.tracker.reset()
        arr = pc.to_struct(m)
        p = pc.xyz(arr)
        inside = np.all((p >= self.roi_min) & (p <= self.roi_max), axis=1)
        p = p[inside]

        if self.bg.learning:
            if self.bg.add_frame(p, t):
                self.get_logger().info(f"Background learned: {self.bg.meta}")
                try:
                    self.bg.save(self.bg_file, mount_signature(self.cal))
                    self.get_logger().info(f"Background saved to {self.bg_file}")
                except OSError as e:
                    self.get_logger().error(f"Could not save background: {e}")
            self.publish(m.header, [], np.zeros((0, 3)))
            return

        self.frames.append((t, p))
        while self.frames and t - self.frames[0][0] > self.window:
            self.frames.popleft()
        pts = np.concatenate([f[1] for f in self.frames]) if self.frames else np.zeros((0, 3))

        if self.ground_filter:
            pts = pts[pts[:, 2] > self.ground_z]
        if self.mode == "background":
            if not self.bg.ready:
                self.publish(m.header, [], np.zeros((0, 3)))
                return
            pts = pts[~self.bg.is_background(pts)]

        clusters = cluster(pts, self.sensor_origin, voxel_size=self.cl_voxel,
                           min_points=self.cl_min_points, max_extent=self.cl_max_extent)
        tracks = self.tracker.update(clusters, t)
        if not self.publish_unconfirmed:
            tracks = [tr for tr in tracks if tr.confirmed]
        self.publish(m.header, tracks, pts)
        self.stats["frames"] += 1
        self.stats["clusters"] = len(clusters)
        self.stats["fg_points"] = int(pts.shape[0])
        self.stats["proc_ms"] = max(self.stats["proc_ms"], (time.perf_counter() - tic) * 1e3)

    # ------------------------------------------------------------------ output
    def publish(self, header: Header, tracks, fg_points):
        out = ObstacleArray(header=header)
        markers = MarkerArray()
        markers.markers.append(Marker(header=header, action=Marker.DELETEALL))
        for tr in tracks:
            c = tr.cluster
            o = Obstacle()
            o.id = tr.id
            o.confirmed = tr.confirmed
            o.age = float(tr.last_t - tr.first_t)
            o.position = Point(x=float(c.centroid[0]), y=float(c.centroid[1]), z=float(c.centroid[2]))
            o.velocity = Vector3(x=float(tr.velocity[0]), y=float(tr.velocity[1]), z=float(tr.velocity[2]))
            o.center = Point(x=float(c.center[0]), y=float(c.center[1]), z=float(c.center[2]))
            o.size = Vector3(x=float(c.size[0]), y=float(c.size[1]), z=float(c.size[2]))
            o.yaw = float(c.yaw)
            o.closest_point = Point(x=float(c.closest[0]), y=float(c.closest[1]), z=float(c.closest[2]))
            o.range = float(c.range)
            o.num_points = int(c.num_points)
            out.obstacles.append(o)

            # wireframe (12 edges), so the box does not hide the shape of what is inside it
            box = Marker(header=header, ns="obstacles", id=tr.id, type=Marker.LINE_LIST, action=Marker.ADD)
            box.pose.position = o.center
            box.pose.orientation.z = float(np.sin(c.yaw / 2))
            box.pose.orientation.w = float(np.cos(c.yaw / 2))
            box.scale.x = 0.015                               # line width [m]
            half = np.maximum(0.05, c.size) / 2
            box.points = [Point(x=float(half[0] * (1 if i & 1 else -1)),
                                y=float(half[1] * (1 if i & 2 else -1)),
                                z=float(half[2] * (1 if i & 4 else -1)))
                          for a in range(8) for b in (1, 2, 4) if a < (a ^ b) for i in (a, a ^ b)]
            box.color = ColorRGBA(r=1.0, g=0.45, b=0.1, a=0.9) if tr.confirmed else \
                ColorRGBA(r=1.0, g=1.0, b=0.2, a=0.5)
            box.lifetime.nanosec = 300000000
            markers.markers.append(box)
            if not self.marker_labels:
                continue
            txt = Marker(header=header, ns="labels", id=tr.id, type=Marker.TEXT_VIEW_FACING,
                         action=Marker.ADD)
            txt.pose.position = Point(x=o.center.x, y=o.center.y, z=o.center.z + o.size.z / 2 + 0.3)
            txt.scale.z = 0.3
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            speed = float(np.linalg.norm(tr.velocity[:2]))
            txt.text = f"#{tr.id} {o.range:.1f} m" + (f" {speed:.1f} m/s" if speed > 0.2 else "")
            txt.lifetime.nanosec = 300000000
            markers.markers.append(txt)
        self.pub.publish(out)
        self.pub_markers.publish(markers)
        f = fg_points.astype(np.float32)
        self.pub_fg.publish(pc.make_cloud(header, {"x": f[:, 0], "y": f[:, 1], "z": f[:, 2]}))

    def diagnostics(self):
        self._refresh_calibration()
        st = DiagnosticStatus(name="l2_obstacles", hardware_id="unitree_l2", level=DiagnosticStatus.OK)
        if self.mode == "background" and self.bg.learning:
            st.level = DiagnosticStatus.WARN
            st.message = f"learning background ({100 * self.bg.progress():.0f} %)"
        elif self.mode == "background" and not self.bg.ready:
            st.level = DiagnosticStatus.ERROR
            st.message = "no background model"
        elif self.stats["frames"] == 0:
            st.level = DiagnosticStatus.ERROR
            st.message = "no input"
        else:
            st.message = "OK"
        st.values = [KeyValue(key=k, value=str(v)) for k, v in [
            ("mode", self.mode), ("frames_per_s", self.stats["frames"]),
            ("clusters", self.stats["clusters"]), ("foreground_points", self.stats["fg_points"]),
            ("tracks", len(self.tracker.tracks)), ("background_voxels", int(self.bg.keys.size)),
            ("max_processing_ms", f"{self.stats['proc_ms']:.1f}")]]
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [st]
        self.pub_diag.publish(arr)
        self.stats["frames"] = 0
        self.stats["proc_ms"] = 0.0


def main():
    rclpy.init()
    node = ObstacleDetector()
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
