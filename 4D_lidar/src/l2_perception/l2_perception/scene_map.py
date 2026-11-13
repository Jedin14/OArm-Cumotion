"""Persistent static-scene map with range-banded voxels (constant angular resolution).

The L2 puts a fixed number of points per steradian per second into every direction, so the
point spacing on a surface grows linearly with range. A fixed voxel / point size is therefore
too coarse close by and leaves gaps far away. Here the voxel size grows with range instead:

    band b holds points with  edges[b-1] <= r < edges[b],  voxel size sizes[b],
    edges[b] = sizes[b] / k,  k = sqrt(hits / (scan_density * fill_time))

so a voxel at the far edge of its band receives `hits` (default 3, i.e. ~95 % of the voxels
are hit) points within `fill_time`, at every range - for a scan that covers every direction
evenly over time. The real L2 does not: over UART it repeats fixed elevation lines ~0.6 deg
apart, so voxels between two lines stay empty. The gap-free view is range_image.py, which
bridges those lines; this map serves the depth image and l2/points_dense.

The map is persistent: a voxel stays (and its average keeps improving) while it is hit at least
once every `forget_time` seconds. Only static-scene points are inserted, so nothing moving can
leave a trail in it.
"""

import numpy as np

_BITS = 20
_OFF = np.int64(1 << (_BITS - 1))     # +-524288 voxels per axis (2.6 km at 5 mm)
_MASK = np.int64((1 << _BITS) - 1)
_BAND_SHIFT = np.int64(3 * _BITS)      # band index in bits 60..62 (max 8 bands)


def band_edges(sizes, fill_time, scan_density, hits=3.0):
    """Upper range limit of each band but the last (which is open-ended)."""
    k = np.sqrt(hits / (scan_density * fill_time))
    return np.asarray(sizes[:-1], dtype=float) / k


def band_of(r, edges):
    return np.searchsorted(edges, r, side="right")


def banded_keys(points, band, sizes):
    s = np.asarray(sizes, dtype=float)[band][:, None]
    idx = np.floor(points / s).astype(np.int64) + _OFF
    return ((band.astype(np.int64) << _BAND_SHIFT) | (idx[:, 0] << (2 * _BITS))
            | (idx[:, 1] << _BITS) | idx[:, 2])


def band_average(points, values, r, edges, sizes):
    """One-shot banded voxel average (for the live layer). Returns centroids, mean values, band."""
    if points.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64)
    band = band_of(r, edges)
    u, inv = np.unique(banded_keys(points, band, sizes), return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv, minlength=u.size).astype(np.float64)
    c = np.stack([np.bincount(inv, points[:, i], u.size) for i in range(3)], axis=1) / cnt[:, None]
    v = np.bincount(inv, values, u.size) / cnt
    return c, v, (u >> _BAND_SHIFT).astype(np.int64)


class SortedAccumulator:
    """Sparse per-key sums: sorted int64 keys, value sums, hit counts and last-hit time.
    Keys not hit for `forget_time` seconds are dropped by forget()."""

    def __init__(self, n_values, forget_time):
        self.n_values = int(n_values)
        self.forget_time = float(forget_time)
        self.clear()

    def clear(self):
        self.keys = np.zeros(0, dtype=np.int64)        # sorted
        self.sums = np.zeros((0, self.n_values))
        self.cnt = np.zeros(0, dtype=np.int64)
        self.last = np.zeros(0)

    def __len__(self):
        return self.keys.size

    def add(self, keys, values, t):
        """keys (N,) int64, values (N, n_values), all stamped t."""
        if keys.size == 0:
            return
        u, inv = np.unique(keys, return_inverse=True)
        inv = inv.ravel()
        bs = np.stack([np.bincount(inv, values[:, i], u.size) for i in range(self.n_values)], axis=1)
        bc = np.bincount(inv, minlength=u.size)
        pos = np.searchsorted(self.keys, u)
        found = pos < self.keys.size
        found[found] = self.keys[pos[found]] == u[found]
        f = pos[found]
        self.sums[f] += bs[found]
        self.cnt[f] += bc[found]
        self.last[f] = np.maximum(self.last[f], t)
        new = ~found
        if new.any():
            at = pos[new]
            self.keys = np.insert(self.keys, at, u[new])
            self.sums = np.insert(self.sums, at, bs[new], axis=0)
            self.cnt = np.insert(self.cnt, at, bc[new])
            self.last = np.insert(self.last, at, t)

    def forget(self, now):
        """Drop keys not hit for forget_time (the scene changed, or the spot is now hidden)."""
        keep = now - self.last <= self.forget_time
        if not keep.all():
            self.keys, self.sums, self.cnt, self.last = \
                self.keys[keep], self.sums[keep], self.cnt[keep], self.last[keep]

    def touch(self, now):
        """After a stamp jump: the geometry is still valid, only the clock moved."""
        self.last[:] = now


class BandedVoxelMap(SortedAccumulator):
    def __init__(self, sizes, fill_time=60.0, scan_density=3700.0, hits=3.0, forget_time=None):
        self.sizes = np.asarray(sizes, dtype=float)
        if self.sizes.size < 1 or self.sizes.size > 8 or np.any(np.diff(self.sizes) <= 0):
            raise ValueError("voxel_sizes: 1..8 increasing values")
        self.edges = band_edges(self.sizes, fill_time, scan_density, hits)
        super().__init__(4, forget_time if forget_time and forget_time > 0 else 2.0 * fill_time)

    def insert(self, points, intensity, r, t):
        """Add a batch of static points (world frame) with ranges r, all stamped t."""
        if points.shape[0] == 0:
            return
        keys = banded_keys(points, band_of(r, self.edges), self.sizes)
        self.add(keys, np.column_stack([points, intensity]), t)

    def points(self, min_points=1):
        """Centroids (N,3), mean intensity (N,), band (N,) - sorted by band."""
        m = self.cnt >= min_points
        c = self.cnt[m].astype(np.float64)[:, None]
        s = self.sums[m] / c
        return s[:, :3], s[:, 3], (self.keys[m] >> _BAND_SHIFT).astype(np.int64)


def split_bands(band, n_bands):
    """Index ranges [start, end) per band for band-sorted arrays."""
    b = np.searchsorted(band, np.arange(n_bands + 1))
    return list(zip(b[:-1], b[1:]))
