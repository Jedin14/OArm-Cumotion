"""Depth-camera-style ("image") point cloud of the static scene.

A depth camera gives one point per pixel on a regular grid of viewing directions: dense, smooth
surfaces with sharp edges. A single L2 sweep is sparse instead (~39 k points/s spread over the
whole field of view), and raw points show the scan pattern (rings, lines) and the range noise.
Because the lidar does not move, the same pixel grid can be filled over time:

  RangeImageMap     persistent per-pixel range histogram on an equirectangular grid around the
                    lidar (pixel = res x res degrees), range bins ~1 % of the range wide
  .depth()          per pixel the nearest surface hit at least `min_hits` times (one-off hits:
                    dust, mixed "veil" points at edges, are skipped), averaged over all its hits
  fill_holes        bridges up to `max_gap` empty pixels between two pixels that agree on the
                    surface (up/down, left/right or diagonal) by interpolation - never across a
                    depth edge. Needed because the L2's pattern repeats: over UART it scans fixed
                    elevation lines ~0.6 deg apart, so the rows between them never fill by waiting
  remove_speckles   a pixel without `min_neighbors` consistent neighbours is dropped
  grid_points       pixel ray x range: one point per pixel, rows = elevation, columns = azimuth,
                    thinned towards the poles so the grid is uniform on the sphere

Everything is relative to the lidar origin, axes world-aligned (as l2/depth_image).
"""

import numpy as np

from .scene_map import SortedAccumulator

_RB = 12                         # range-bin bits in a key (pixel index above)
_RMASK = np.int64((1 << _RB) - 1)
_RSTEP = 0.01                    # bin = 1 % of (r + 0.5 m): 8 mm at 0.3 m, 1.5 cm at 1 m, 5.5 cm at 5 m
_ROFF = 100


def range_bin(r):
    return np.clip(np.floor(np.log(r + 0.5) / _RSTEP).astype(np.int64) + _ROFF, 0, _RMASK)


class RangeImageMap(SortedAccumulator):
    def __init__(self, res_deg=0.2, forget_time=120.0):
        super().__init__(1, forget_time)            # value: sum of ranges
        self.W = int(round(360.0 / res_deg))
        self.H = int(round(180.0 / res_deg))
        self.du = 2 * np.pi / self.W
        self.dv = np.pi / self.H

    def pixels(self, rel):
        """Pixel index (row-major, row 0 = straight up) and range of points relative to the lidar."""
        r = np.linalg.norm(rel, axis=1)
        az = np.arctan2(rel[:, 1], rel[:, 0])
        el = np.arcsin(np.clip(rel[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0))
        u = ((np.pi - az) / self.du).astype(np.int64) % self.W
        v = np.clip(((np.pi / 2 - el) / self.dv).astype(np.int64), 0, self.H - 1)
        return v * self.W + u, r

    def insert(self, rel, t):
        """Add static points (relative to the lidar), all stamped t."""
        pix, r = self.pixels(rel)
        ok = r > 0.05
        self.add((pix[ok] << _RB) | range_bin(r[ok]), r[ok, None], t)

    def depth(self, min_hits=2, margin=4):
        """Range of the nearest solid surface per pixel (float32, inf = nothing seen), cropped to
        the rows that hold data (+ margin): returns (depth (h, W), first row v0).
        A surface = run of neighbouring range bins (one empty bin allowed); it is solid with
        >= min_hits hits, or when the whole pixel has fewer hits than that + 1 (too sparse to tell)."""
        if self.keys.size == 0:
            return np.full((0, self.W), np.inf, dtype=np.float32), 0
        d = np.full(self.H * self.W, np.inf, dtype=np.float32)
        pix = self.keys >> _RB
        rb = self.keys & _RMASK
        newpix = np.r_[True, pix[1:] != pix[:-1]]
        brk = newpix | np.r_[True, np.diff(rb) > 2]                     # a new surface starts
        start = np.flatnonzero(brk)
        run = np.cumsum(brk) - 1
        rc = np.bincount(run, self.cnt.astype(np.float64), start.size)
        rs = np.bincount(run, self.sums[:, 0], start.size)
        group = np.cumsum(newpix) - 1                                    # pixel group per key
        total = np.bincount(group, self.cnt.astype(np.float64))
        solid = (rc >= min_hits) | (total[group[start]] < min_hits + 1)
        idx = np.flatnonzero(solid)
        rp = pix[start[idx]]
        front = idx[np.r_[True, rp[1:] != rp[:-1]]]                      # first solid run per pixel
        d[pix[start[front]]] = rs[front] / rc[front]
        v0 = max(int(pix[0] // self.W) - margin, 0)
        v1 = min(int(pix[-1] // self.W) + 1 + margin, self.H)
        return d.reshape(self.H, self.W)[v0:v1], v0

    def rays(self, v0=0, v1=None):
        """Unit view direction of each pixel centre, rows v0..v1-1: (v1-v0, W, 3)."""
        v1 = self.H if v1 is None else v1
        el = np.pi / 2 - (np.arange(v0, v1) + 0.5) * self.dv
        az = np.pi - (np.arange(self.W) + 0.5) * self.du
        ce = np.cos(el)[:, None]
        return np.stack([ce * np.cos(az), ce * np.sin(az),
                         np.broadcast_to(np.sin(el)[:, None], (v1 - v0, self.W))], axis=2)

    def pole_keep(self, v0=0, v1=None):
        """(rows, W) mask keeping a fraction cos(elevation) of each row's pixels, evenly spread:
        towards the poles the columns crowd together, this keeps the density uniform."""
        v1 = self.H if v1 is None else v1
        c = np.cos(np.pi / 2 - (np.arange(v0, v1) + 0.5) * self.dv)[:, None]
        u = np.arange(self.W)[None, :]
        return np.ceil((u + 1) * c) > np.ceil(u * c)


def _pad(depth, n=1):
    """Pad by n pixels: rows with inf (poles), columns wrapped (360 deg azimuth)."""
    p = np.pad(depth, ((n, n), (0, 0)), constant_values=np.inf)
    return np.concatenate([p[:, -n:], p, p[:, :n]], axis=1)


def _shift(P, n, H, W, dy, dx):
    """View of the n-padded array P: value of the pixel (v + dy, u + dx) for every (v, u)."""
    return P[n + dy:n + dy + H, n + dx:n + dx + W]


def fill_holes(depth, max_gap=3, rel_tol=0.03, abs_tol=0.02, passes=1):
    """Bridge runs of up to `max_gap` empty pixels between two valid pixels on a straight line -
    vertical (between the L2's scan lines), horizontal or diagonal - by linear interpolation,
    if both ends agree: |a - b| <= rel_tol * span * mean + abs_tol (span in pixels). A depth edge
    (foreground / background) is never bridged. The shortest, then straightest bridge wins."""
    H, W = depth.shape
    G = int(max_gap)
    with np.errstate(invalid="ignore"):          # inf - inf where both ends are empty
        for _ in range(int(passes)):
            hole = ~np.isfinite(depth)
            if not hole.any() or G < 1:
                break
            P = _pad(depth, G)
            best = np.full((H, W), np.inf, dtype=depth.dtype)
            score = np.full((H, W), np.inf)
            for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
                ends = []
                for sgn in (1, -1):                     # nearest valid pixel each way, its distance
                    val = np.full((H, W), np.inf, dtype=depth.dtype)
                    dist = np.zeros((H, W), dtype=np.int32)
                    for k in range(1, G + 1):
                        q = _shift(P, G, H, W, sgn * k * dy, sgn * k * dx)
                        new = ~np.isfinite(val) & np.isfinite(q)
                        val[new] = q[new]
                        dist[new] = k
                    ends.append((val, dist))
                (a, ia), (b, ib) = ends
                span = ia + ib
                ok = hole & np.isfinite(a) & np.isfinite(b) & (span <= G + 1)
                m = np.where(ok, (a + b) * 0.5, 0.0)
                diff = np.where(ok, np.abs(a - b), np.inf)
                ok &= diff <= rel_tol * span * m + abs_tol
                sc = np.where(ok, span + diff / np.maximum(m * span, 1e-6), np.inf)
                better = sc < score
                best = np.where(better, a + (b - a) * ia / np.maximum(span, 1), best)
                score = np.where(better, sc, score)
            filled = hole & np.isfinite(best)
            if not filled.any():
                break
            depth = np.where(filled, best, depth)
    return depth


def remove_speckles(depth, min_neighbors=2, rel_tol=0.03, abs_tol=0.02):
    """Drop pixels with fewer than `min_neighbors` of their 8 neighbours at about the same range."""
    H, W = depth.shape
    valid = np.isfinite(depth)
    P = _pad(depth)
    d0 = np.where(valid, depth, 0)
    n = np.zeros((H, W), dtype=np.int32)
    tol = rel_tol * d0 + abs_tol
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                q = _shift(P, 1, H, W, dy, dx)
                n += np.isfinite(q) & (np.abs(d0 - np.where(np.isfinite(q), q, 0)) <= tol)
    out = depth.copy()
    out[valid & (n < min_neighbors)] = np.inf
    return out


def grid_points(rmap, depth, v0=0, thin_poles=True):
    """One point per filled pixel of `depth` (rows v0.. of the map's grid): points relative to the
    lidar (N,3), ranges (N,), pixel rows and columns (N,) - in row-major (image) order."""
    ok = np.isfinite(depth)
    if thin_poles:
        ok &= rmap.pole_keep(v0, v0 + depth.shape[0])
    vv, uu = np.nonzero(ok)
    r = depth[vv, uu].astype(np.float64)
    return rmap.rays(v0, v0 + depth.shape[0])[vv, uu] * r[:, None], r, vv + v0, uu
