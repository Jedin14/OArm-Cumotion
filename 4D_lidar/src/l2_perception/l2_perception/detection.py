"""Clustering, box fitting and tracking of foreground points."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .voxel import connected_components


@dataclass
class Cluster:
    centroid: np.ndarray
    center: np.ndarray       # oriented box centre
    size: np.ndarray         # extents (length, width, height)
    yaw: float
    closest: np.ndarray      # point closest to the lidar
    range: float
    num_points: int
    points: np.ndarray = field(repr=False, default=None)


def _oriented_box(p: np.ndarray):
    xy = p[:, :2]
    c = xy.mean(axis=0)
    if p.shape[0] >= 3:
        cov = np.cov((xy - c).T)
        w, v = np.linalg.eigh(cov)
        major = v[:, np.argmax(w)]
        yaw = float(np.arctan2(major[1], major[0]))
    else:
        yaw = 0.0
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, -sy], [sy, cy]])
    loc = (xy - c) @ R                # into box axes
    lo, hi = loc.min(axis=0), loc.max(axis=0)
    mid = c + R @ ((lo + hi) / 2)
    zlo, zhi = p[:, 2].min(), p[:, 2].max()
    center = np.array([mid[0], mid[1], (zlo + zhi) / 2])
    size = np.array([hi[0] - lo[0], hi[1] - lo[1], zhi - zlo])
    return center, size, yaw


def cluster(points: np.ndarray, sensor_origin: np.ndarray, voxel_size: float = 0.2,
            min_points: int = 8, max_points: int = 200000,
            min_extent: float = 0.0, max_extent: float = 15.0) -> List[Cluster]:
    """Euclidean clustering via voxel connectivity (gap tolerance ~ 1-2 voxels)."""
    if points.shape[0] < min_points:
        return []
    labels, n = connected_components(points, voxel_size)
    counts = np.bincount(labels, minlength=n)
    keep = np.flatnonzero((counts >= min_points) & (counts <= max_points))
    out = []
    order = np.argsort(labels, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(counts)])
    for k in keep:
        p = points[order[bounds[k]:bounds[k + 1]]]
        center, size, yaw = _oriented_box(p)
        ext = float(size.max())
        if ext < min_extent or ext > max_extent:
            continue
        d = np.linalg.norm(p - sensor_origin, axis=1)
        i = int(np.argmin(d))
        out.append(Cluster(centroid=p.mean(axis=0), center=center, size=size, yaw=yaw,
                           closest=p[i].copy(), range=float(d[i]), num_points=int(p.shape[0]),
                           points=p))
    return out


@dataclass
class Track:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    first_t: float
    last_t: float
    hits: int = 1
    cluster: Cluster = None
    confirmed: bool = False


class Tracker:
    """Greedy nearest-neighbour tracker with constant-velocity prediction.

    Its job here is temporal consistency: stable ids and suppression of
    one-frame flicker caused by the sparse, non-repetitive scan pattern.
    """

    def __init__(self, gate: float = 1.0, min_hits: int = 3, max_missed_time: float = 0.5,
                 velocity_smoothing: float = 0.5):
        self.gate = gate
        self.min_hits = min_hits
        self.max_missed_time = max_missed_time
        self.alpha = velocity_smoothing
        self.tracks: List[Track] = []
        self._next_id = 1

    def reset(self):
        self.tracks = []

    def update(self, clusters: List[Cluster], t: float) -> List[Track]:
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.max_missed_time]
        pred = [tr.position + tr.velocity * (t - tr.last_t) for tr in self.tracks]
        pairs = []
        for i, tr in enumerate(self.tracks):
            for j, c in enumerate(clusters):
                d = float(np.linalg.norm((pred[i] - c.centroid)[:2]))
                if d <= self.gate:
                    pairs.append((d, i, j))
        pairs.sort()
        used_t, used_c = set(), set()
        updated = []
        for d, i, j in pairs:
            if i in used_t or j in used_c:
                continue
            used_t.add(i)
            used_c.add(j)
            tr = self.tracks[i]
            c = clusters[j]
            dt = t - tr.last_t
            if dt > 1e-6:
                v = (c.centroid - tr.position) / dt
                tr.velocity = self.alpha * tr.velocity + (1 - self.alpha) * v
            tr.position = c.centroid.copy()
            tr.last_t = t
            tr.hits += 1
            tr.cluster = c
            tr.confirmed = tr.confirmed or tr.hits >= self.min_hits
            updated.append(tr)
        for j, c in enumerate(clusters):
            if j in used_c:
                continue
            tr = Track(id=self._next_id, position=c.centroid.copy(), velocity=np.zeros(3),
                       first_t=t, last_t=t, cluster=c, confirmed=self.min_hits <= 1)
            self._next_id += 1
            self.tracks.append(tr)
            updated.append(tr)
        return updated
