"""l2_dense: the scene as a depth-camera-style point cloud, a live layer for people / moving
objects, and a depth image.

    l2/points (world frame, from l2_stabilizer)
      -> l2/points_image      the static scene as a depth camera would give it: one point per
                              pixel of a 0.2 deg grid around the lidar, nearest surface, range
                              noise averaged out, gaps between the L2's scan lines bridged where
                              both sides agree (range_image.py). Whole frame, 1 Hz, coloured by
                              height (low red .. yellow .. high green).
      -> l2/points_scene      raw static-scene points of the last 1/publish_rate s, same colours
                              (RViz can accumulate them with a long Decay Time)
      -> l2/points_live       people, moving / new objects (raw points, 10 Hz)
      -> l2/points_dense      the persistent scene map, one point per voxel (built only while
                              subscribed, for other consumers)
      -> l2/depth_image (rgb8), l2/depth_image/raw (32FC1, metres)   360 deg panorama
    Clouds carry x y z intensity range (+ rgb for the scene).

Measured on the real L2 over UART: ~39 k points/s, ~3700 points per steradian per second,
surface noise sigma ~3 mm, intensity saturated (255) beyond ~1 m. The scan puts a fixed number
of points into every direction, so point spacing grows with range: one fixed point size is
either too coarse close by or full of gaps far away. Every point is routed:

  static  inside the learned background (l2_obstacles' model, read from its file) and not in or
          under a tracked obstacle box -> the RangeImageMap (l2/points_image), l2/points_scene
          (raw) and the BandedVoxelMap (scene_map.py; depth image and l2/points_dense).
  live    everything else -> raw points, at most `live.hold` s old. Older points are shown only
          while (a) their `live.cell` cell keeps being hit - a cell that is still occupied is
          hit again within  tau(r) = confirm_hits * r^2 / (scan_density * cell^2)  (clamped) -
          and (b) they lie inside a box l2_obstacles tracks right now (moved forward with its
          velocity). A person standing still builds up `hold` seconds of detail; a walking
          person leaves only a short trail.

Without a background model (still learning, or mode geometric) only the obstacle boxes route
points to the live layer; without l2/obstacles only rule (a) applies.
"""

import array
import os
import time
from collections import deque

import numpy as np
import rclpy
from l2_msgs.msg import ObstacleArray
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header

from . import pointcloud as pc
from .mount import MountCalibration
from .range_image import RangeImageMap, fill_holes, grid_points, remove_speckles
from .scene_map import BandedVoxelMap
from .voxel import isin_sorted, voxel_keys


def turbo(t: np.ndarray) -> np.ndarray:
    """Google 'turbo' colormap (polynomial approximation), t in [0,1] -> uint8 RGB."""
    t = np.clip(t, 0.0, 1.0)[..., None]
    c = np.concatenate([
        0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))),
        0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))),
        0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))),
    ], axis=-1)
    return (np.clip(c, 0.0, 1.0) * 255).astype(np.uint8)


def depth_panorama(rel, rng, W, H, el_min, el_max, fill=2):
    """Equirectangular min-range image around the lidar (world-aligned axes).
    rel: (N,3) points relative to the lidar, rng: (N,) ranges. Returns (H,W) float32, inf = empty."""
    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.arctan2(rel[:, 2], np.hypot(rel[:, 0], rel[:, 1]))
    u = ((np.pi - az) / (2 * np.pi) * W).astype(np.int64) % W
    v = ((el_max - el) / (el_max - el_min) * H).astype(np.int64)
    ok = (v >= 0) & (v < H) & (rng > 0.05)
    depth = np.full(H * W, np.inf, dtype=np.float32)
    np.minimum.at(depth, v[ok] * W + u[ok], rng[ok].astype(np.float32))
    depth = depth.reshape(H, W)
    for _ in range(int(fill)):                 # fill isolated holes from neighbours (nearest wins)
        hole = ~np.isfinite(depth)
        if not hole.any():
            break
        pad = np.pad(depth, 1, constant_values=np.inf)
        nb = np.min(np.stack([pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                              for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
        depth = np.where(hole, nb, depth)
    return depth


def colorize_depth(depth, range_max=0.0):
    """near = red, far = blue (turbo section 0.1..0.95), empty = black."""
    valid = np.isfinite(depth)
    rmax = range_max if range_max > 0 else (float(np.percentile(depth[valid], 98)) if valid.any() else 1.0)
    rgb = turbo(0.1 + 0.85 * np.clip(1.0 - depth / max(rmax, 0.1), 0.0, 1.0))
    rgb[~valid] = 0
    return rgb


def auto_elevation(rel, current=None, step_deg=5.0):
    """Elevation range [rad] covering the data (0.5..99.5 %), rounded to 5 deg, with hysteresis."""
    el = np.rad2deg(np.arctan2(rel[:, 2], np.hypot(rel[:, 0], rel[:, 1])))
    if el.size < 100:
        return current
    lo = max(-90.0, np.floor((np.percentile(el, 0.5) - 2) / step_deg) * step_deg)
    hi = min(90.0, np.ceil((np.percentile(el, 99.5) + 2) / step_deg) * step_deg)
    if current is not None and abs(np.rad2deg(current[0]) - lo) <= step_deg and \
            abs(np.rad2deg(current[1]) - hi) <= step_deg:
        return current
    return (np.deg2rad(lo), np.deg2rad(hi))


def voxel_average(points, intensity, hits_id, voxel, min_hits):
    """Average points per voxel. hits_id: per-point cloud index (voxel 'hits' = distinct clouds)."""
    if points.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0)
    keys = voxel_keys(points, voxel)
    uniq, inv = np.unique(keys, return_inverse=True)
    inv = inv.ravel()
    n = uniq.size
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    out = np.stack([np.bincount(inv, points[:, i], n) for i in range(3)], axis=1) / cnt[:, None]
    inten = np.bincount(inv, intensity, n) / cnt
    if min_hits > 1:
        pair = np.unique(inv.astype(np.int64) * 1_000_003 + hits_id)   # distinct (voxel, cloud) pairs
        hits = np.bincount(pair // 1_000_003, minlength=n)
        keep = hits >= min_hits
        out, inten = out[keep], inten[keep]
    return out, inten


# ----------------------------------------------------------------------------- static / live routing
def in_boxes(points, boxes, margin=0.15, below=0.3, min_z=-np.inf):
    """Points inside any oriented box (center(3), size(3), yaw) grown by `margin` and extended
    `below` metres downwards (feet close to the floor sit in the dilated background model), but
    not below `min_z` (the floor itself)."""
    m = np.zeros(points.shape[0], dtype=bool)
    for c, s, yaw in boxes:
        d = points[:, :2] - c[:2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        lx = d[:, 0] * cy + d[:, 1] * sy
        ly = -d[:, 0] * sy + d[:, 1] * cy
        z = points[:, 2]
        m |= ((np.abs(lx) <= s[0] / 2 + margin) & (np.abs(ly) <= s[1] / 2 + margin)
              & (z >= max(c[2] - s[2] / 2 - below, min_z)) & (z <= c[2] + s[2] / 2 + margin))
    return m


def static_mask(points, bg_keys, bg_voxel, boxes, margin=0.15, below=0.3, min_z=0.05):
    """True = static scene (long exposure), False = live layer. Points of the background inside
    a tracked box (+ margin) are live too, except the floor (z < min_z, world z = 0 is the floor)."""
    if bg_keys is None:
        m = np.ones(points.shape[0], dtype=bool)
    else:
        m = isin_sorted(voxel_keys(points, bg_voxel), bg_keys)
    if boxes:
        m &= ~in_boxes(points, boxes, margin, below, min_z)
    return m


def confirm_time(r, cell, scan_density, confirm_hits=4.0, t_min=0.2, t_max=1.5):
    """Time within which an occupied `cell` at range r is hit again: the scan puts
    scan_density * (cell / r)^2 points per second into it, so confirm_hits = 4 means a still
    occupied cell is missed for that long only with ~2 % probability."""
    return np.clip(confirm_hits * r * r / (scan_density * cell * cell), t_min, t_max)


def live_select(points, t, now, origin, hold, cell, tau, gate=None, fresh=0.2):
    """Indices of the live points to show at time `now`: at most `hold` old, and their `cell`
    voxel was hit within tau(range) (callable) - i.e. whatever was there is still there.
    gate: optional mask, where older points may still be (inside a currently tracked box);
    points younger than `fresh` are always shown."""
    keep = np.flatnonzero(now - t <= hold)
    if keep.size == 0:
        return keep
    p = points[keep]
    _, inv = np.unique(voxel_keys(p, cell), return_inverse=True)
    inv = inv.ravel()
    last = np.full(inv.max() + 1, -np.inf)
    np.maximum.at(last, inv, t[keep])
    r = np.linalg.norm(p - origin, axis=1)
    ok = now - last[inv] <= tau(r)
    if gate is not None:
        ok &= gate[keep]
        ok |= now - t[keep] <= fresh
    return keep[ok]


def stamp_sec(s) -> float:
    return s.sec + s.nanosec * 1e-9


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """uint8 (N,3) -> float32 'rgb' field (PCL / RViz RGB8 convention 0x00RRGGBB)."""
    u = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2]
    return u.astype(np.uint32).view(np.float32)


def height_colors(z, z0, z1):
    """Low = red, then orange, yellow, high = green (uint8 RGB)."""
    t = np.clip((z - z0) / max(z1 - z0, 1e-3), 0.0, 1.0)
    rgb = np.stack([np.minimum(1.0, 2.0 * (1.0 - t)), np.minimum(1.0, 2.0 * t), np.zeros_like(t)], axis=1)
    return (rgb * 255).astype(np.uint8)


class Dense(Node):
    def __init__(self):
        super().__init__("l2_dense")
        loose = ParameterDescriptor(dynamic_typing=True)

        def P(name, default):
            """declare_parameter that also accepts 20 for 20.0 (and [1, 2] for [1.0, 2.0])."""
            if isinstance(default, (float, list)):
                v = self.declare_parameter(name, default, loose).value
                return [float(x) for x in v] if isinstance(default, list) else float(v)
            return self.declare_parameter(name, default)
        self.sizes = P("voxel_sizes", [0.005, 0.01, 0.02, 0.04, 0.08])
        self.fill_time = P("fill_time", 60.0)
        forget = P("forget_time", 0.0)               # 0 = 2 x fill_time
        self.min_points = P("min_points", 1).value
        self.density = P("scan_density", 3700.0)
        self.rate = P("publish_rate", 2.0)
        self.color_by = P("color_by", "height").value              # height | range
        self.z_min, self.z_max = P("color_z_min", 0.0), P("color_z_max", 0.0)   # equal = auto
        self.range_max = P("color_range_max", 0.0)   # 0 = auto (98th percentile)
        if self.color_by not in ("height", "range"):
            raise ValueError("color_by must be 'height' or 'range'")
        self.cal_file = os.path.expanduser(P("calibration_file", "~/.ros/l2/mount.yaml").value)
        self.bg_file = os.path.expanduser(P("background_file", "~/.ros/l2/background.npz").value)
        self.split = P("live.enabled", True).value
        self.hold = P("live.hold", 3.0)
        self.cell = P("live.cell", 0.15)
        self.confirm_hits = P("live.confirm_hits", 4.0)
        self.confirm_min = P("live.confirm_min", 0.2)
        self.confirm_max = P("live.confirm_max", 1.5)
        self.box_margin = P("live.box_margin", 0.15)
        self.box_below = P("live.box_below", 0.3)
        self.box_min_z = P("live.box_min_z", 0.05)
        self.box_gate = P("live.box_gate", True).value
        self.box_hold = P("live.box_hold", 0.25)
        self.box_lag = P("live.box_lag", 0.15)
        self.img_on = P("depth_image.enabled", True).value
        img_rate = P("depth_image.rate", 2.0)
        self.img_window = P("depth_image.window", 8.0)
        self.W = P("depth_image.width", 2048).value
        self.H = P("depth_image.height", 512).value
        self.el_min = np.deg2rad(P("depth_image.elevation_min_deg", -90.0))
        self.el_max = np.deg2rad(P("depth_image.elevation_max_deg", 90.0))
        self.auto_el = P("depth_image.auto_elevation", True).value
        self.el_range = None
        self.fill = P("depth_image.hole_fill", 2).value
        self.ri_on = P("image.enabled", True).value
        ri_rate = P("image.rate", 1.0)
        ri_res = P("image.resolution_deg", 0.2)
        self.ri_hits = P("image.min_hits", 2).value
        self.ri_gap = P("image.max_gap", 3).value
        self.ri_thin = P("image.thin_poles", True).value

        self.map = BandedVoxelMap(self.sizes, self.fill_time, self.density, 3.0, forget)
        self.rimg = RangeImageMap(ri_res, self.map.forget_time)
        self.pending = []                # static points not yet merged into the map
        self.recent = deque()            # (t, xyz) raw static points for the depth image
        self.live = deque()              # (t, xyz, intensity)
        self.live_now = (np.zeros((0, 3)), np.zeros(0))
        self.rmax = None                 # colour scales (auto), kept stable with hysteresis
        self.zr = None
        self.origin = np.zeros(3)
        self.cal_mtime = None
        self.bg_keys, self.bg_voxel, self.bg_mtime = None, 0.1, None
        self.box_hist = deque()          # (t, boxes) from l2/obstacles
        self.live_frac = 0.0
        self.frame_id = "world"
        self.last_stamp = None
        self.last_t = None
        q = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PointCloud2, P("input_topic", "l2/points").value, self.on_points, q)
        self.create_subscription(ObstacleArray, "l2/obstacles", self.on_obstacles, 10)
        self.pub_image = self.create_publisher(PointCloud2, "l2/points_image", 2)
        self.pub_scene = self.create_publisher(PointCloud2, "l2/points_scene", 10)
        self.pub = self.create_publisher(PointCloud2, "l2/points_dense", 2)
        self.pub_live = self.create_publisher(PointCloud2, "l2/points_live", 5)
        self.pub_img = self.create_publisher(Image, "l2/depth_image", 2)
        self.pub_raw = self.create_publisher(Image, "l2/depth_image/raw", 2)
        self.create_timer(1.0 / self.rate, self.publish)
        self.create_timer(1.0 / img_rate, self.publish_depth)
        self.create_timer(1.0 / ri_rate, self.publish_image)
        self.create_timer(2.0, self.refresh_files)
        self.refresh_files()

    # ------------------------------------------------------------------ inputs
    def refresh_files(self):
        """Lidar pose (ranges and the depth image are measured from it) and background model."""
        try:
            mtime = os.path.getmtime(self.cal_file)
            if mtime != self.cal_mtime:
                cal = MountCalibration.load(self.cal_file)
                if cal is not None:
                    if self.cal_mtime is not None and len(self.map):
                        self.get_logger().info("mount calibration changed: rebuilding the scene")
                        self._reset_scene()
                    self.origin = cal.translation()
                    self.cal_mtime = mtime
        except OSError:
            pass
        try:
            mtime = os.path.getmtime(self.bg_file)
        except OSError:
            if self.bg_keys is not None:
                self.get_logger().warn("background model file gone: only obstacle boxes are live now")
            self.bg_keys, self.bg_mtime = None, None
            return
        if mtime == self.bg_mtime:
            return
        try:
            with np.load(self.bg_file, allow_pickle=False) as z:
                keys = np.sort(np.asarray(z["keys"], dtype=np.int64))
                vs = float(z["voxel_size"])
        except Exception as e:  # noqa: BLE001 - half-written / foreign file: keep the old model
            self.get_logger().warn(f"cannot read {self.bg_file}: {e}", throttle_duration_sec=60.0)
            return
        self.bg_keys, self.bg_voxel, self.bg_mtime = keys, vs, mtime
        self._reset_scene()      # whatever moved while there was no model must not stay in it
        self.get_logger().info(f"static/live split uses {self.bg_file} ({keys.size} voxels)")

    def _reset_scene(self):
        self.map.clear()
        self.rimg.clear()
        self.pending = []
        self.recent.clear()

    def on_obstacles(self, m: ObstacleArray):
        t = stamp_sec(m.header.stamp)
        if self.box_hist and t < self.box_hist[-1][0]:
            self.box_hist.clear()
        self.box_hist.append((t, [(np.array([o.center.x, o.center.y, o.center.z]),
                                   np.array([o.size.x, o.size.y, o.size.z]), o.yaw,
                                   np.array([o.velocity.x, o.velocity.y, 0.0]))
                                  for o in m.obstacles]))
        while self.box_hist and t - self.box_hist[0][0] > self.box_hold:
            self.box_hist.popleft()

    def current_boxes(self, t):
        """Boxes tracked during the last `box_hold` s (bridges a missed detection), moved to time
        t with their velocity (+ box_lag: a box describes the middle of l2_obstacles' detection
        window), or None if l2/obstacles is not running."""
        if not self.box_hist or abs(t - self.box_hist[-1][0]) > 1.0:
            return None
        out = []
        for tb, boxes in self.box_hist:
            dt = float(np.clip(t - tb + self.box_lag, 0.0, 0.5))
            for c, size, yaw, v in boxes:
                speed = np.linalg.norm(v)
                v = v * min(1.0, 3.0 / speed) if speed > 0 else v     # never trust > 3 m/s
                out.append((c + v * dt, size, yaw))
        return out

    def on_points(self, m: PointCloud2):
        t = stamp_sec(m.header.stamp)
        if self.last_t is not None and (t < self.last_t or t - self.last_t > 2.0):
            self.pending = []                       # stamp jump (clock step / restart):
            self.recent.clear()                     # the geometry is still valid
            self.live.clear()
            self.map.touch(t)
            self.rimg.touch(t)
        self.last_t = t
        a = pc.to_struct(m)
        p, it = pc.xyz(a), pc.field_or(a, "intensity", 0.0)
        boxes = self.current_boxes(t)
        if self.split:
            s = static_mask(p, self.bg_keys, self.bg_voxel, boxes or [],
                            self.box_margin, self.box_below, self.box_min_z)
        else:
            s = np.ones(p.shape[0], dtype=bool)
        self.pending.append((t, p[s], it[s]))
        self.recent.append((t, p[s]))
        while self.recent and t - self.recent[0][0] > self.img_window:
            self.recent.popleft()
        if not s.all():
            self.live.append((t, p[~s], it[~s]))
        while self.live and t - self.live[0][0] > self.hold:
            self.live.popleft()
        self.frame_id = m.header.frame_id
        self.last_stamp = m.header.stamp
        self.publish_live(Header(stamp=m.header.stamp, frame_id=m.header.frame_id), t, boxes)

        if p.shape[0] and self.bg_keys is not None:     # background no longer matches the scene?
            self.live_frac = 0.98 * self.live_frac + 0.02 * float(1.0 - s.mean())
            if self.live_frac > 0.5:
                self.get_logger().warn(
                    f"{100 * self.live_frac:.0f} % of the points are not in the background model: "
                    "the scene or mount changed. Re-learn with an empty area: ros2 service call "
                    "/l2_obstacles/learn_background std_srvs/srv/Trigger", throttle_duration_sec=60.0)

    # ------------------------------------------------------------------ outputs
    def _tau(self, r):
        return confirm_time(r, self.cell, self.density, self.confirm_hits, self.confirm_min, self.confirm_max)

    def _cloud(self, h, p, it, rng, rgb=None):
        cols = {"x": p[:, 0].astype(np.float32), "y": p[:, 1].astype(np.float32),
                "z": p[:, 2].astype(np.float32), "intensity": it.astype(np.float32),
                "range": rng.astype(np.float32)}
        if rgb is not None:
            cols["rgb"] = rgb
        return pc.make_cloud(h, cols)

    def _update_scales(self, fallback):
        """Auto colour scales from the scene map (a stable, complete picture of the scene), with
        hysteresis: RViz keeps already received points for its Decay Time, so colours must not
        drift. fallback: points to use while the map is still nearly empty."""
        n = len(self.map)
        if n >= 2000:
            k = slice(None, None, max(1, n // 100000))
            pts = self.map.sums[k, :3] / self.map.cnt[k, None]
        else:
            pts = fallback
        if pts.shape[0] < 100:
            return
        lo, hi = np.percentile(pts[:, 2], [1.0, 99.0])
        span = max(hi - lo, 0.1)
        if self.zr is None or abs(lo - self.zr[0]) > 0.1 * span or abs(hi - self.zr[1]) > 0.1 * span:
            self.zr = (float(lo), float(lo + span))
        r98 = float(np.percentile(np.linalg.norm(pts - self.origin, axis=1), 98))
        if self.rmax is None or abs(r98 / self.rmax - 1.0) > 0.1:
            self.rmax = max(r98, 0.5)

    def _colors(self, p, rng):
        if self.color_by == "range":
            rmax = self.range_max if self.range_max > 0 else (self.rmax or 10.0)
            return pack_rgb(turbo(0.1 + 0.85 * np.clip(1.0 - rng / rmax, 0.0, 1.0)))
        z0, z1 = (self.z_min, self.z_max) if self.z_max > self.z_min else (self.zr or (0.0, 2.0))
        return pack_rgb(height_colors(p[:, 2], z0, z1))

    def publish_live(self, h, now, boxes):
        """Every cloud (10 Hz), also when empty so RViz clears what has left."""
        if self.live:
            pts = np.concatenate([f[1] for f in self.live])
            inten = np.concatenate([f[2] for f in self.live])
            ts = np.concatenate([np.full(f[1].shape[0], f[0]) for f in self.live])
            gate = in_boxes(pts, boxes, self.box_margin, self.box_below) \
                if (self.box_gate and boxes is not None) else None
            idx = live_select(pts, ts, now, self.origin, self.hold, self.cell, self._tau,
                              gate, self.confirm_min)
            self.live_now = (pts[idx], inten[idx])
        else:
            self.live_now = (np.zeros((0, 3)), np.zeros(0))
        p, it = self.live_now
        self.pub_live.publish(self._cloud(h, p, it, np.linalg.norm(p - self.origin, axis=1)))

    def publish(self):
        """Send the static points collected since the last call as l2/points_scene: RViz keeps
        them for its Decay Time, which builds the dense, fine-grained picture of the scene.
        Moving things never get here, so the long decay leaves no trails. The points are also
        merged into the scene map (depth image, l2/points_dense)."""
        if not self.pending or self.last_stamp is None:
            return
        tic = time.perf_counter()
        pts = np.concatenate([f[1] for f in self.pending])
        inten = np.concatenate([f[2] for f in self.pending])
        t_last = self.pending[-1][0]
        self.pending = []
        rng = np.linalg.norm(pts - self.origin, axis=1)
        self.map.insert(pts, inten, rng, t_last)
        self.map.forget(t_last)
        if self.ri_on:
            self.rimg.insert(pts - self.origin, t_last)
            self.rimg.forget(t_last)
        self._update_scales(pts)
        h = Header(stamp=self.last_stamp, frame_id=self.frame_id)
        self.pub_scene.publish(self._cloud(h, pts, inten, rng, self._colors(pts, rng)))
        if self.pub.get_subscription_count():
            c, it, _ = self.map.points(self.min_points)
            cr = np.linalg.norm(c - self.origin, axis=1)
            self.pub.publish(self._cloud(h, c, it, cr, self._colors(c, cr)))
        dt = (time.perf_counter() - tic) * 1e3
        if dt > 0.8e3 / self.rate:
            self.get_logger().warn(f"scene update took {dt:.0f} ms ({len(self.map)} voxels); lower "
                                   "publish_rate or use larger voxel_sizes", throttle_duration_sec=30.0)

    def publish_image(self):
        """l2/points_image: the scene as a depth camera would give it (see range_image.py).
        Each message is the whole picture (RViz: Decay Time 0)."""
        if not self.ri_on or self.last_stamp is None or not len(self.rimg) or \
                not self.pub_image.get_subscription_count():
            return
        tic = time.perf_counter()
        d, v0 = self.rimg.depth(self.ri_hits)
        d = remove_speckles(fill_holes(d, self.ri_gap))
        rel, rng, _, _ = grid_points(self.rimg, d, v0, self.ri_thin)
        p = rel + self.origin
        h = Header(stamp=self.last_stamp, frame_id=self.frame_id)
        self.pub_image.publish(pc.make_cloud(h, {
            "x": p[:, 0].astype(np.float32), "y": p[:, 1].astype(np.float32),
            "z": p[:, 2].astype(np.float32), "range": rng.astype(np.float32),
            "rgb": self._colors(p, rng)}))
        dt = time.perf_counter() - tic
        if dt > 0.5:
            self.get_logger().warn(f"image cloud took {1e3 * dt:.0f} ms ({p.shape[0]} points); lower "
                                   "image.rate or raise image.resolution_deg", throttle_duration_sec=30.0)

    def publish_depth(self):
        """Equirectangular panorama around the lidar, world-aligned (columns = azimuth), from the
        scene map, the raw static points of the last `depth_image.window` s and the live points."""
        if not self.img_on or self.last_stamp is None or not (
                self.pub_img.get_subscription_count() or self.pub_raw.get_subscription_count()):
            return
        parts = [self.map.points(self.min_points)[0]] + [f[1] for f in self.recent] + [self.live_now[0]]
        rel = np.concatenate(parts) - self.origin
        if rel.shape[0] == 0:
            return
        rng = np.linalg.norm(rel, axis=1)
        if self.auto_el:
            self.el_range = auto_elevation(rel, self.el_range)
        lo, hi = self.el_range if (self.auto_el and self.el_range) else (self.el_min, self.el_max)
        depth = depth_panorama(rel, rng, self.W, self.H, lo, hi, self.fill)
        h = Header(stamp=self.last_stamp, frame_id=self.frame_id)
        scale = self.range_max if self.range_max > 0 else (self.rmax or 0.0)
        self.pub_img.publish(self._image(h, "rgb8", colorize_depth(depth, scale), 3))
        raw = np.where(np.isfinite(depth), depth, np.nan).astype(np.float32)
        self.pub_raw.publish(self._image(h, "32FC1", raw, 4))

    def _image(self, h, enc, arr, bpp):
        img = Image(header=h, height=self.H, width=self.W, encoding=enc, is_bigendian=0,
                    step=self.W * bpp)
        buf = array.array("B")
        buf.frombytes(np.ascontiguousarray(arr).tobytes())
        img.data = buf
        return img


def main():
    rclpy.init()
    node = Dense()
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
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
