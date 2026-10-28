"""The depth check: the depth image made to agree with the colour image
(2026-10-26).

Points to touch come from the depth image; the arm's calibration from the
colour image (the markers). On 2026-10-16 the two disagreed by ~3 cm. The
ChArUco board (ctm/calib_box.py) shows both at once: the colour image gives
its exact pose, the depth image its surface. Over 3-5 placements a small
correction is fitted, per pixel:

    z_true = z * (1 + a) + b + c * x + e * y      (x, y: the pixel's normalised ray)

and applied to every depth frame the tools use (apply()).
"""

import os

import numpy as np
import yaml

from ctm.common import WS

FILE = os.path.join(WS, 'depth_correction.yaml')
_cache = {'mtime': None, 'coef': None}


def load():
    """(a, b, c, e) or None."""
    try:
        mtime = os.path.getmtime(FILE)
    except OSError:
        _cache.update(mtime=None, coef=None)
        return None
    if mtime != _cache['mtime']:
        try:
            with open(FILE) as handle:
                d = yaml.safe_load(handle) or {}
            _cache['coef'] = tuple(float(d[k]) for k in ('a', 'b', 'c', 'e'))
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError):
            _cache['coef'] = None
        _cache['mtime'] = mtime
    return _cache['coef']


def apply(depth, info):
    """The depth image (m) with the correction applied; unchanged without one."""
    coef = load()
    if coef is None or depth is None:
        return depth
    a, b, c, e = coef
    h, w = depth.shape
    k = np.array(info.k, float).reshape(3, 3)
    sx, sy = w / float(info.width or w), h / float(info.height or h)
    x = (np.arange(w) / sx - k[0, 2]) / k[0, 0]
    y = (np.arange(h) / sy - k[1, 2]) / k[1, 1]
    out = depth * (1.0 + a) + b + c * x[None, :] + e * y[:, None]
    return np.where(depth > 0, out, 0.0).astype(depth.dtype)


def samples_from(board_cam, depth, info, corners_px, shrink=0.15):
    """(x, y, z_measured, z_true) for the depth pixels on the board: inside
    the outline of its found corners (shrunk, away from the edges), with a
    reading. board_cam: the board's pose in the camera frame (4x4)."""
    import cv2
    h, w = depth.shape
    hull = cv2.convexHull(corners_px.astype(np.float32)).reshape(-1, 2)
    centre = hull.mean(axis=0)
    hull = centre + (hull - centre) * (1.0 - shrink)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(hull).astype(np.int32)], 1)
    vv, uu = np.nonzero((mask > 0) & (depth > 0.1))
    if len(uu) < 200:
        return None
    k = np.array(info.k, float).reshape(3, 3)
    x = (uu - k[0, 2]) / k[0, 0]
    y = (vv - k[1, 2]) / k[1, 1]
    z = depth[vv, uu].astype(np.float64)
    n = board_cam[:3, 2]                         # the board's normal
    d = float(n @ board_cam[:3, 3])
    rays = np.c_[x, y, np.ones_like(x)]
    z_true = d / (rays @ n)
    good = np.abs(z_true - z) < 0.08             # holes, other things in front
    return np.c_[x, y, z, z_true][good]


def fit(samples):
    """samples: [N x (x, y, z, z_true)] from several placements. Returns
    (coef, rms before, rms after) in m."""
    s = np.vstack(samples)
    x, y, z, zt = s.T
    A = np.c_[z, np.ones_like(z), x, y]
    sol, *_ = np.linalg.lstsq(A, zt - z, rcond=None)
    for _ in range(3):                           # drop the worst 2 %, again
        res = A @ sol - (zt - z)
        keep = np.abs(res) < np.percentile(np.abs(res), 98)
        sol, *_ = np.linalg.lstsq(A[keep], (zt - z)[keep], rcond=None)
    before = float(np.sqrt(np.mean((zt - z) ** 2)))
    after = float(np.sqrt(np.mean((A @ sol - (zt - z)) ** 2)))
    return tuple(float(v) for v in sol), before, after


def save(coef, before, after, placements, points):
    a, b, c, e = coef
    with open(FILE, 'w') as handle:
        handle.write('# Depth correction from the ChArUco board (ctm/depth_cal.py):\n'
                     '# z_true = z * (1 + a) + b + c * x + e * y\n'
                     + yaml.safe_dump({'a': a, 'b': b, 'c': c, 'e': e,
                                       'rms_before_mm': round(1000 * before, 2),
                                       'rms_after_mm': round(1000 * after, 2),
                                       'placements': placements, 'points': points},
                                      sort_keys=False))
