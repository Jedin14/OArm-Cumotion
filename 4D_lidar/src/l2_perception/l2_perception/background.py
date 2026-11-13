"""Static-scene (background) model for a stationary lidar.

Because the lidar does not move, every surface that is part of the scene shows
up in the same voxels forever. During a learning period each voxel counts in
how many time slices (default 1 s) it was hit. Static surfaces are hit over and
over; a person walking through occupies a voxel for a fraction of a second.
Voxels hit in >= min_slices slices become background, then the set is dilated
by one voxel to absorb range noise and the sparse, non-repetitive scan pattern
of the L2 at long range.

A voxel is background when it was hit in >= min_slices slices AND those hits
span >= min_span (fraction) of the learning period. The span test is what
rejects things that were only *temporarily* there (a person standing around for
a few seconds), while sparse but persistent far-away surfaces still qualify.

At run time anything outside the background set is foreground (object /
obstacle). The model is saved to disk and reloaded, so it only has to be
learned once per installation (re-learn after changing the scene or mount).
"""

import os
import time
from typing import Optional

import numpy as np

from .voxel import dilate, isin_sorted, voxel_keys


class VoxelBackground:
    def __init__(self, voxel_size: float = 0.1, slice_time: float = 1.0,
                 min_slices: int = 2, dilation: int = 1, min_span: float = 0.5):
        self.voxel_size = float(voxel_size)
        self.slice_time = float(slice_time)
        self.min_slices = int(min_slices)
        self.dilation = int(dilation)
        self.min_span = float(min_span)
        self.keys = np.zeros(0, dtype=np.int64)   # sorted background voxels
        self.learning = False
        self.meta = {}
        self._slices = []
        self._cur = []
        self._slice_start = None
        self._learn_start = None
        self._learn_duration = 0.0
        self._last_t = 0.0
        self._learn_end = None

    @property
    def ready(self) -> bool:
        return self.keys.size > 0 and not self.learning

    # ------------------------------------------------------------- learning
    def start_learning(self, duration: float, t_now: Optional[float] = None) -> None:
        self.learning = True
        self._slices = []
        self._cur = []
        self._slice_start = None
        self._learn_duration = float(duration)
        self._learn_start = t_now
        self._last_t = t_now if t_now is not None else 0.0
        self._learn_end = None

    def progress(self) -> float:
        if not self.learning or self._learn_start is None:
            return 0.0
        return min(1.0, (self._last_t - self._learn_start) / self._learn_duration)

    def add_frame(self, points: np.ndarray, t: float) -> bool:
        """Feed one cloud while learning. Returns True when learning just finished."""
        if not self.learning:
            return False
        if self._learn_start is None:
            self._learn_start = t
        self._last_t = t
        if self._slice_start is None:
            self._slice_start = t
        if t - self._slice_start >= self.slice_time:
            self._close_slice()
            self._slice_start = t
        if points.shape[0]:
            self._cur.append(np.unique(voxel_keys(points, self.voxel_size)))
        if t - self._learn_start >= self._learn_duration:
            self._close_slice()
            self._finalize()
            return True
        return False

    def _close_slice(self):
        if self._cur:
            self._slices.append(np.unique(np.concatenate(self._cur)))
        self._cur = []

    def _finalize(self):
        self.learning = False
        if not self._slices:
            self.keys = np.zeros(0, dtype=np.int64)
            return
        ns = len(self._slices)
        allk = np.concatenate(self._slices)
        idx = np.concatenate([np.full(k.size, i, dtype=np.int64) for i, k in enumerate(self._slices)])
        order = np.argsort(allk, kind="stable")
        allk, idx = allk[order], idx[order]
        uniq, start, counts = np.unique(allk, return_index=True, return_counts=True)
        first = np.minimum.reduceat(idx, start)
        last = np.maximum.reduceat(idx, start)
        need = min(self.min_slices, ns)
        span_ok = (last - first + 1) >= self.min_span * ns
        core = uniq[(counts >= need) & span_ok]
        self.keys = dilate(core, self.dilation)
        self.meta = {
            "slices": ns,
            "core_voxels": int(core.size),
            "voxels": int(self.keys.size),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._slices = []

    # ------------------------------------------------------------- query
    def is_background(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        return isin_sorted(voxel_keys(points, self.voxel_size), self.keys)

    # ------------------------------------------------------------- io
    def save(self, path: str, mount_signature: str = "") -> None:
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, keys=self.keys, voxel_size=self.voxel_size,
                            dilation=self.dilation, mount_signature=mount_signature,
                            created=self.meta.get("created", ""))
        os.replace(tmp, path)

    def load(self, path: str, mount_signature: str = "") -> str:
        """Returns '' on success, otherwise the reason it was not loaded."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            return "no file"
        with np.load(path, allow_pickle=False) as z:
            if abs(float(z["voxel_size"]) - self.voxel_size) > 1e-9:
                return f"voxel size {float(z['voxel_size'])} != {self.voxel_size}"
            sig = str(z["mount_signature"])
            if mount_signature and sig and sig != mount_signature:
                return "mount calibration changed since it was learned"
            self.keys = np.asarray(z["keys"], dtype=np.int64)
            self.meta = {"voxels": int(self.keys.size), "created": str(z["created"])}
        self.learning = False
        return ""
