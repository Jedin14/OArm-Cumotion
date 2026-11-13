"""Mount calibration of a stationary lidar: a constant lidar -> world transform.

The world frame: origin on the ground directly below the lidar, +z up
(opposite to gravity / normal of the ground plane), +x = the lidar's x axis
projected onto the ground and rotated by `yaw` (user-chosen heading).

The calibration is estimated ONCE (gravity from the accelerometer, refined with
a RANSAC ground-plane fit) and persisted to YAML. Later runs load it, so the
result never depends on what the sensor happens to see during start-up.
"""

import datetime
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import yaml

from .geometry import matrix_to_rpy, rotation_aligning, rpy_to_matrix

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class MountCalibration:
    roll: float = 0.0     # [rad] lidar orientation in world, ZYX
    pitch: float = 0.0
    yaw: float = 0.0
    x: float = 0.0        # [m] lidar origin in world
    y: float = 0.0
    z: float = 0.0
    gravity_lidar: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])  # "up" seen by accel
    ground_rms: float = -1.0     # [m] RMS of ground inliers, -1 if no ground fit
    ground_inliers: int = 0
    source: str = "default"
    created: str = ""

    def rotation(self) -> np.ndarray:
        return rpy_to_matrix(self.roll, self.pitch, self.yaw)

    def translation(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def apply(self, points_lidar: np.ndarray) -> np.ndarray:
        return points_lidar @ self.rotation().T + self.translation()

    # ---------------------------------------------------------------- io
    def save(self, path: str) -> None:
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = asdict(self)
        d["roll_deg"] = float(np.rad2deg(self.roll))
        d["pitch_deg"] = float(np.rad2deg(self.pitch))
        d["yaw_deg"] = float(np.rad2deg(self.yaw))
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write("# Unitree L2 stationary mount calibration (lidar pose in the world frame).\n")
            f.write("# Edit yaw / x / y freely; roll, pitch, z come from calibration.\n")
            yaml.safe_dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                            for k, v in d.items()}, f, sort_keys=False)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> Optional["MountCalibration"]:
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        c = MountCalibration()
        for k in ("roll", "pitch", "yaw", "x", "y", "z", "ground_rms"):
            if k in d:
                setattr(c, k, float(d[k]))
        # degrees win if the user edited them
        for k in ("roll", "pitch", "yaw"):
            if f"{k}_deg" in d and abs(np.rad2deg(getattr(c, k)) - float(d[f"{k}_deg"])) > 1e-6:
                setattr(c, k, float(np.deg2rad(float(d[f"{k}_deg"]))))
        c.gravity_lidar = [float(v) for v in d.get("gravity_lidar", [0, 0, 1])]
        c.ground_inliers = int(d.get("ground_inliers", 0))
        c.source = str(d.get("source", "file"))
        c.created = str(d.get("created", ""))
        return c


# -------------------------------------------------------------------- estimation
def up_from_accel(acc_mean: np.ndarray) -> np.ndarray:
    """A static accelerometer measures the specific force, which points UP."""
    a = np.asarray(acc_mean, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-6:
        raise ValueError("accelerometer mean is zero")
    return a / n


def fit_ground_plane(points: np.ndarray, distance: float = 0.04, iterations: int = 300,
                     max_tilt_deg: float = 10.0, min_inliers: int = 500,
                     seed: int = 0):
    """RANSAC for the ground in an (approximately) levelled frame whose origin is the lidar.

    Returns (normal_up, height_of_lidar_above_plane, inlier_mask, rms) or None.
    Among near-horizontal planes below the sensor with a comparable number of
    inliers, the LOWEST one is the ground (rejects tables, shelves, ...).
    """
    pts = np.asarray(points, dtype=float)
    below = pts[:, 2] < -0.3
    cand = pts[below]
    if cand.shape[0] < min_inliers:
        return None
    rng = np.random.default_rng(seed)
    cos_tilt = np.cos(np.deg2rad(max_tilt_deg))
    sub = cand if cand.shape[0] <= 60000 else cand[rng.choice(cand.shape[0], 60000, replace=False)]
    planes = []
    for _ in range(iterations):
        i = rng.choice(sub.shape[0], 3, replace=False)
        p0, p1, p2 = sub[i]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n[2] < 0:
            n = -n
        if n[2] < cos_tilt:
            continue
        d = -float(n @ p0)            # n.p + d = 0 ; lidar (origin) height = d
        cnt = int(np.count_nonzero(np.abs(sub @ n + d) < distance))
        planes.append((cnt, d, n))
    if not planes:
        return None
    best = max(p[0] for p in planes)
    good = [p for p in planes if p[0] >= 0.5 * best]
    _, d, n = max(good, key=lambda p: p[1])  # largest d = lowest plane below the sensor
    # least-squares refinement on the inliers of the full candidate set
    for _ in range(2):
        inl = np.abs(cand @ n + d) < distance
        if np.count_nonzero(inl) < min_inliers:
            return None
        q = cand[inl]
        c = q.mean(axis=0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        n = vt[2]
        if n[2] < 0:
            n = -n
        d = -float(n @ c)
    inl_full = np.zeros(pts.shape[0], dtype=bool)
    inl_full[np.flatnonzero(below)[inl]] = True
    rms = float(np.sqrt(np.mean(np.square(q @ n + d))))
    return n, d, inl_full, rms


def calibrate(acc_mean: np.ndarray, points_lidar: Optional[np.ndarray], *,
              level_reference: str = "ground", yaw: float = 0.0, x: float = 0.0, y: float = 0.0,
              mount_height: float = -1.0, ground_distance: float = 0.04,
              ground_max_tilt_deg: float = 10.0, min_inliers: int = 500,
              log=print) -> MountCalibration:
    """Estimate the mount.

    level_reference:
      "ground"  - roll/pitch from the ground plane if one is found (most accurate
                  for obstacle heights; immune to accelerometer bias), else gravity
      "gravity" - roll/pitch from the accelerometer only
    mount_height: used when no ground plane is found (or > 0 to force it).
    """
    up_l = up_from_accel(acc_mean)
    # Rotation that makes the measured "up" the world +z, with zero yaw.
    R_g = rotation_aligning(up_l, UP)
    r, p, _ = matrix_to_rpy(R_g)
    R_g = rpy_to_matrix(r, p, 0.0)
    R = R_g
    source = "gravity"
    height = mount_height if mount_height > 0 else 0.0
    ground_rms, n_inl = -1.0, 0

    if points_lidar is not None and points_lidar.shape[0] > 0:
        lev = points_lidar @ R_g.T
        res = fit_ground_plane(lev, distance=ground_distance, max_tilt_deg=ground_max_tilt_deg,
                               min_inliers=min_inliers)
        if res is not None:
            n, d, inl, ground_rms = res
            n_inl = int(np.count_nonzero(inl))
            tilt = np.rad2deg(np.arccos(np.clip(n[2], -1, 1)))
            log(f"ground plane: {n_inl} inliers, rms {ground_rms * 100:.1f} cm, "
                f"{tilt:.2f} deg from gravity, lidar {d:.3f} m above it")
            if level_reference == "ground":
                R_n = rotation_aligning(n, UP)
                R = R_n @ R_g
                r, p, _ = matrix_to_rpy(R)
                R = rpy_to_matrix(r, p, 0.0)
                source = "ground"
            if mount_height <= 0:
                height = d
        else:
            log("no ground plane found; using gravity only"
                + ("" if mount_height > 0 else " and z = 0 (set mount_height!)"))

    r, p, _ = matrix_to_rpy(R)
    return MountCalibration(
        roll=r, pitch=p, yaw=float(yaw), x=float(x), y=float(y), z=float(height),
        gravity_lidar=[float(v) for v in up_l], ground_rms=float(ground_rms),
        ground_inliers=n_inl, source=source,
        created=datetime.datetime.now().isoformat(timespec="seconds"))


def tilt_change_deg(cal: MountCalibration, acc_mean: np.ndarray) -> float:
    """Angle between gravity now and at calibration time (accelerometer bias cancels)."""
    g0 = np.asarray(cal.gravity_lidar, dtype=float)
    g0 /= np.linalg.norm(g0)
    g1 = up_from_accel(acc_mean)
    return float(np.rad2deg(np.arccos(np.clip(g0 @ g1, -1.0, 1.0))))
