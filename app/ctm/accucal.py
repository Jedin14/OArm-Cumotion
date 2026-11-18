"""AUTO CALIBRATE: the arm model from the box readings -- fitted, chosen and
checked with nobody deciding anything (2026-11-16).

The run (ctm/boxcal_run.py calibrate_auto) moves the left arm, holding the
calibration box on its cup, through ~90 postures. At each it keeps the joint
angles and the box's marker corners as the camera saw them: a READING. This
file turns those readings into the arm model:

    the box on the cup   where the user put it, a little crooked, and the
                         bellows let it tilt. Its pose in hand_tcp is solved
                         along with the arm, one per mounting ("segment": the
                         run starts a new one when the box slipped), and it
                         is NEVER used for picks -- they keep the nominal tool
                         tip, hand_tcp + tip_offset along the tool axis.
    the arm              one of VARIANTS, simplest first:
        rigid            only the base: where the arm sits relative to the
                         camera (rotation about the shoulder + shift)
        zeros            + the zero offsets of joints 1-6. Joint 7's stays 0:
                         with the box's mount free, turning joint 7 is the
                         same as turning the box on the cup. And joint 1's
                         offset IS a base turn about joint 1's own axis (the
                         base turns about the shoulder, which is on it): the
                         base only tilts across that axis here, joint 1's
                         offset takes the turn.
        zeros+sag        + each joint's give under gravity (comply, 7)

Backlash, link errors (geom), a tip offset on the hand (tip_delta) and a
world shift (anchor_shift) are never fitted: on 10-30 they only cancelled
each other (j7 offset against its comply and backlash, a 12 deg base turn
against joint 1, the tip 29 mm off on the hand) and the model raised front
picks by 2 cm.

Every model -- and the plain URDF, and the model in use now -- is scored on
postures left out of its fit (cross-validation, each fold spread over the
whole workspace), in mm at the tool tip. The simplest one within a whisker
of the best wins: no correction is better than a needless one.

Pure computation (numpy, scipy, OpenCV, yaml through ctm.common): no ROS,
safe to run offline on saved readings.
"""

import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from ctm import calib_box as cb
from ctm.common import plain
from ctm.kinematics import Arm, frames_batch, rotvec, tcp_batch

VARIANTS = ('rigid', 'zeros', 'zeros+sag')   # simplest first

# Prior widths: how far each number may plausibly be from the URDF. Weak --
# one prior counts as much as one corner pixel off by one sigma, and a run
# has thousands of corners: they only settle what the pictures cannot.
SIGMA = {'base_rot': 0.03, 'base_pos': 0.03, 'offset': 0.06, 'comply': 0.01}
MOUNT_SIGMA = (0.5, 0.10)       # rad, m: the box on the cup -- free, only kept finite
F_SCALE = 1.5                   # px: corners further off than this count less (huber)
DROP_PX = 3.0                   # a reading off its markers by more than this (and 3x the median) is dropped
# A held-out reading's tip comes from its own markers alone: two small ones
# on one face leave the box's tilt loose (one such reading put the tip 76 mm
# off, 2026-10-30) -- a measuring error, not the model's. Such readings help
# the fit (the arm holds the box there) but are not scored.
EVAL_MARKERS = 3
MARGIN = (1.1, 0.3)             # within 1.1 x best + 0.3 mm counts as "as good"
DRIFT_MM = 5.0                  # a slip trend larger than this across a mounting is flagged
MIN_READINGS = 12
MAX_NFEV = 200
ORDER = ('urdf',) + VARIANTS    # simplest first, for the choice


# -- small geometry helpers ------------------------------------------------------

def _pose(rv, t):
    m = np.eye(4)
    m[:3, :3] = rotvec(rv)
    m[:3, 3] = t
    return m


def _mean_pose(poses):
    """The mean rotation (chordal, by SVD) and median position of 4x4 poses."""
    u, _s, vt = np.linalg.svd(np.sum([p[:3, :3] for p in poses], axis=0))
    m = np.eye(4)
    m[:3, :3] = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    m[:3, 3] = np.median([p[:3, 3] for p in poses], axis=0)
    return m


def _nudge(m0, v):
    """m0 turned by v[:3] (rotation vector, in the parent frame) and moved by v[3:]."""
    m = np.eye(4)
    m[:3, :3] = rotvec(v[:3]) @ m0[:3, :3]
    m[:3, 3] = m0[:3, 3] + v[3:]
    return m


def _base(kin, b):
    """4x4 of a base correction (rotation vector about the shoulder + shift),
    as kinematics.Arm.tcp applies it."""
    m = np.eye(4)
    m[:3, :3] = rotvec(b[:3])
    m[:3, 3] = kin.shoulder - m[:3, :3] @ kin.shoulder + np.asarray(b[3:], float)
    return m


def _swing_basis(kin):
    """Two unit vectors square to joint 1's axis: the base turns the zeros
    models may fit (joint 1's offset is the turn about the axis itself)."""
    w = kin.frames(np.zeros(7))[1][0][1]
    w = w / np.linalg.norm(w)
    _u, _s, vt = np.linalg.svd(w[None, :])
    return vt[1:]                                  # (2, 3)


def _zero_params():
    return {'base': np.zeros(6), 'offset': np.zeros(7), 'comply': np.zeros(7)}


def _as_params(raw):
    """A model as arrays (from arm_model.yaml or a dict of lists), or None if
    it lacks the basic pieces."""
    if not raw or any(k not in raw for k in ('base', 'offset', 'comply')):
        return None
    out = {}
    for k in ('base', 'offset', 'comply', 'backlash', 'geom', 'hysteresis', 'tip_delta',
              'anchor_shift'):
        if k in raw and raw[k] is not None:
            out[k] = np.asarray(raw[k], float)
    if 'geom' in out:
        out['geom'] = out['geom'].reshape(7, 6)
    return out


def tip_of(kin, q, params, tip_offset):
    """The model's tool tip (world, m): hand_tcp under params (None = the
    plain URDF), tip_offset out along the tool axis."""
    p = _as_params(params) if isinstance(params, dict) else None
    return (kin.tcp(np.asarray(q, float), p) @ np.array([0.0, 0.0, float(tip_offset), 1.0]))[:3]


def _tip_point(params, tip_offset):
    """The tip on the hand (homogeneous, hand_tcp frame) as a model has it:
    nominal, plus tip_delta if the model carries one (the old models)."""
    p = np.array([0.0, 0.0, float(tip_offset), 1.0])
    if params is not None and 'tip_delta' in params:
        p[:3] += params['tip_delta']
    return p


# -- the readings, ready to fit ----------------------------------------------------

def _geometry(readings, log):
    """The box as built (calib_box.refine_geometry over all the views), or
    None if refining does not fit the markers better."""
    views = [{int(i): np.asarray(c, float) for i, c in r['seen'].items()}
             for r in readings if r.get('seen')]
    if len(views) < 5:
        return None
    g, _before, _after = cb.refine_geometry(views, readings[0]['k'], readings[0]['d'], log=log)
    return g or None


def _rows(kin, readings, geometry):
    """Each usable reading as arrays: joints, camera, the box's pose in the
    camera (again with the refined geometry, if any), its corners in the box
    frame and in the picture, and where the URDF puts the hand."""
    pts = cb.corners_with(geometry)
    rows = []
    for n, r in enumerate(readings):
        seen = {int(i): np.asarray(c, float).reshape(4, 2)
                for i, c in (r.get('seen') or {}).items() if int(i) in cb.MARKERS}
        if not seen:
            continue
        k = np.asarray(r['k'], float).reshape(3, 3)
        d = np.asarray(r['d'], float) if len(r['d']) else np.zeros(5)
        cam_box = None
        if geometry:
            got = cb.box_pose_with(seen, k, d, geometry)
            cam_box = got[0] if got is not None else None
        if cam_box is None and r.get('cam_box') is not None:
            cam_box = np.asarray(r['cam_box'], float)
        if cam_box is None:
            got = cb.box_pose(seen, k, d)
            cam_box = got[0] if got is not None else None
        if cam_box is None:
            continue
        ids = sorted(seen)
        q = np.asarray(r['q'], float)
        dirs = r.get('dirs')
        rows.append({'q': q, 'dirs': np.zeros(7) if dirs is None else np.asarray(dirs, float),
                     'segment': int(r.get('segment') or 0),
                     'index': int(r['index']) if r.get('index') is not None else n,
                     'kind': r.get('kind') or 'front', 'n': len(ids), 'k': k, 'd': d,
                     't_wo': np.asarray(r['t_wo'], float), 'cam_box': cam_box,
                     'local': np.concatenate([pts[i] for i in ids]),
                     'obs': np.concatenate([seen[i] for i in ids]),
                     'fk0': kin.tcp(q)})
    return rows


def _light(rows):
    """What the hand poses need: joints, directions, URDF hand poses."""
    return {'q': np.array([r['q'] for r in rows]), 'dirs': np.array([r['dirs'] for r in rows]),
            'fk0': np.array([r['fk0'] for r in rows])}


def _stack(rows):
    """All corners of all readings as one batch (one reading at a time took
    minutes a fit, CAL ARMS 2026-10-13), and the cameras they were seen by."""
    s = _light(rows)
    keys, cams, cam_of = [], [], []
    for r in rows:
        key = np.r_[r['k'].ravel(), r['d'], r['t_wo'].ravel()]
        for c, kk in enumerate(keys):
            if len(kk) == len(key) and np.allclose(kk, key, atol=1e-9):
                cam_of.append(c)
                break
        else:
            keys.append(key)
            cams.append((r['k'], r['d'], np.linalg.inv(r['t_wo'])))
            cam_of.append(len(cams) - 1)
    s['post'] = np.concatenate([[j] * len(r['local']) for j, r in enumerate(rows)]).astype(int)
    s['local'] = np.concatenate([r['local'] for r in rows])
    s['obs'] = np.concatenate([r['obs'] for r in rows])
    s['cams'] = cams
    s['cam'] = np.asarray(cam_of)[s['post']]
    s['seg'] = np.array([r['segment'] for r in rows])
    return s


def _hands(kin, s, params):
    """(N, 4, 4) world <- hand_tcp under params (None: the plain URDF), with
    an old model's anchor_shift applied (its own semantics)."""
    if params is None:
        return s['fk0']
    offset, comply = params['offset'], params['comply']
    if not any(k in params for k in ('backlash', 'geom', 'hysteresis')) and not np.any(comply):
        # no gravity in it: the plain chain, cheaper
        t = s['fk0'] if not np.any(offset) else frames_batch(kin, s['q'] + offset)[0][-1]
        t = _base(kin, params['base']) @ t
    else:
        t = tcp_batch(kin, s['q'], params, s['dirs'] if 'hysteresis' in params else None)
    if 'anchor_shift' in params:
        t = t.copy()
        t[:, :3, 3] += params['anchor_shift']
    return t


def _project(s, world_box):
    """Every corner's pixel with the box at world_box (one 4x4 per reading)."""
    w = world_box[s['post']]
    world = np.einsum('mij,mj->mi', w[:, :3, :3], s['local']) + w[:, :3, 3]
    px = np.empty((len(world), 2))
    for c, (k, d, t_ow) in enumerate(s['cams']):
        m = s['cam'] == c
        cam = world[m] @ t_ow[:3, :3].T + t_ow[:3, 3]
        got, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d)
        px[m] = got.reshape(-1, 2)
    return px


# -- the models -----------------------------------------------------------------

def _sigma(variant):
    s = SIGMA
    if variant == 'rigid':
        return np.r_[[s['base_rot']] * 3, [s['base_pos']] * 3]
    out = np.r_[[s['base_rot']] * 2, [s['base_pos']] * 3, [s['offset']] * 6]
    return np.r_[out, [s['comply']] * 7] if variant == 'zeros+sag' else out


def _unpack(variant, x, swing):
    """The parameter vector of a variant -> a kinematics params dict."""
    p = _zero_params()
    if variant == 'rigid':
        p['base'] = np.array(x[:6], float)
        return p
    p['base'] = np.r_[x[:2] @ swing, x[2:5]]
    p['offset'][:6] = x[5:11]
    if variant == 'zeros+sag':
        p['comply'] = np.array(x[11:18], float)
    return p


def _fit(kin, rows, variant=None, fixed=None):
    """One fit: the variant's arm parameters (or, variant None, the fixed
    model `fixed` -- None: the plain URDF) and every segment's box mount,
    to every corner pixel. Returns {'params', 'mounts' {segment: 4x4},
    'rms' (each reading's px), 'fit_px'}."""
    s = _stack(rows)
    segs = sorted(set(s['seg'].tolist()))
    sidx = np.searchsorted(segs, s['seg'])
    hand0 = _hands(kin, s, fixed) if variant is None else s['fk0']
    # each mount first as the median of what the readings imply through the
    # URDF (or the fixed model)
    m0 = [_mean_pose([np.linalg.inv(hand0[j]) @ r['t_wo'] @ r['cam_box']
                      for j, r in enumerate(rows) if r['segment'] == g]) for g in segs]
    sig_a = _sigma(variant) if variant else np.zeros(0)
    n_a, n_m = len(sig_a), len(segs)
    sigma = np.r_[sig_a, np.tile([MOUNT_SIGMA[0]] * 3 + [MOUNT_SIGMA[1]] * 3, n_m)]
    x_scale = np.r_[sig_a, np.tile([0.02] * 3 + [0.005] * 3, n_m)]
    swing = _swing_basis(kin)

    def model(x):
        params = _unpack(variant, x[:n_a], swing) if variant else fixed
        hands = _hands(kin, s, params) if variant else hand0
        mounts = np.array([_nudge(m0[g], x[n_a + 6 * g:n_a + 6 * g + 6]) for g in range(n_m)])
        return params, hands, mounts

    def residuals(x):
        _p, hands, mounts = model(x)
        r = _project(s, hands @ mounts[sidx]) - s['obs']
        return np.r_[r.ravel(), x / sigma]

    sol = least_squares(residuals, np.zeros(n_a + 6 * n_m), loss='huber', f_scale=F_SCALE,
                        x_scale=x_scale, max_nfev=MAX_NFEV)
    params, _h, mounts = model(sol.x)
    sq = np.sum(sol.fun[:2 * len(s['obs'])].reshape(-1, 2) ** 2, axis=1)
    rms = np.sqrt(np.bincount(s['post'], sq) / np.bincount(s['post']))
    return {'params': params, 'mounts': {g: mounts[i] for i, g in enumerate(segs)},
            'rms': rms, 'fit_px': float(np.sqrt(np.mean(sq)))}


def _screen(kin, rows, variant=None, fixed=None):
    """Fit, drop the readings the model cannot explain (rms > max(DROP_PX, 3x
    median): a misread marker, an arm still swinging), fit again. Returns
    (kept rows, dropped count, the last fit)."""
    got = _fit(kin, rows, variant, fixed)
    bad = got['rms'] > max(DROP_PX, 3.0 * float(np.median(got['rms'])))
    if not bad.any():
        return rows, 0, got
    rows = [r for r, b in zip(rows, bad) if not b]
    return rows, int(bad.sum()), _fit(kin, rows, variant, fixed)


def _point_errors(kin, rows, params, mounts, tip_offset):
    """Model tip minus measured tip at each reading whose segment has a mount
    (and enough markers to be measured well): the measured tip is the hand
    the camera saw (box pose through the mount) with the model's tip on it."""
    hands = _hands(kin, _light(rows), params)
    tip = _tip_point(params, tip_offset)
    out = []
    for j, r in enumerate(rows):
        if r['segment'] not in mounts or r['n'] < EVAL_MARKERS:
            continue
        meas = (r['t_wo'] @ r['cam_box'] @ np.linalg.inv(mounts[r['segment']]) @ tip)[:3]
        err = (hands[j] @ tip)[:3] - meas
        out.append({'index': r['index'], 'kind': r['kind'], 'segment': r['segment'],
                    'tip_meas': [float(v) for v in meas],
                    'err_mm': [float(v) for v in 1000.0 * err],
                    'q': [float(v) for v in r['q']]})
    return out


def _summary(points):
    if not points:
        return {'mean_mm': None, 'p90_mm': None, 'max_mm': None, 'n': 0}
    e = np.linalg.norm([p['err_mm'] for p in points], axis=1)
    return {'mean_mm': float(np.mean(e)), 'p90_mm': float(np.percentile(e, 90)),
            'max_mm': float(np.max(e)), 'n': int(len(e))}


# -- cross-validation ---------------------------------------------------------------

def _space_order(points):
    """The points along a path that always steps to the nearest one not yet
    visited (from a corner of the cloud): neighbours on it are neighbours in
    space, so dealing them out in turn spreads each fold over all of it."""
    points = np.asarray(points, float)
    left = list(range(len(points)))
    here = int(np.argmin(points.sum(axis=1)))
    order = [here]
    left.remove(here)
    while left:
        here = left[int(np.argmin(np.linalg.norm(points[left] - points[here], axis=1)))]
        order.append(here)
        left.remove(here)
    return order


def _folds(rows, folds, tip_offset):
    """Each reading's fold, spread in space: front and top readings each
    along their own path, dealt out in turn."""
    tip = np.array([0.0, 0.0, tip_offset, 1.0])
    pts = np.array([(r['fk0'] @ tip)[:3] for r in rows])
    fold_of = np.zeros(len(rows), int)
    deal = 0
    for kind in sorted(set(r['kind'] for r in rows)):
        idx = [i for i, r in enumerate(rows) if r['kind'] == kind]
        for i in _space_order(pts[idx]):
            fold_of[idx[i]] = deal % folds
            deal += 1
    return fold_of


def _cv(kin, rows, fold_of, tip_offset, variant=None, fixed=None):
    """Each fold left out in turn, the model fitted to the rest (its box
    mounts too), its tip error at the left-out readings. A left-out reading
    whose segment has no training readings is skipped (its mount unknown).
    Returns (summary, per point)."""
    points = []
    for f in sorted(set(fold_of.tolist())):
        train = [r for r, g in zip(rows, fold_of) if g != f]
        held = [r for r, g in zip(rows, fold_of) if g == f]
        if not train or not held:
            continue
        got = _fit(kin, train, variant, fixed)
        points += _point_errors(kin, held, got['params'], got['mounts'], tip_offset)
    points.sort(key=lambda p: p['index'])
    return _summary(points), points


def _choose(cv):
    """The simplest model within MARGIN of the best (the plain URDF first)."""
    means = {k: v['mean_mm'] for k, v in cv.items() if v['mean_mm'] is not None}
    best = min(means.values())
    for name in ORDER:
        if name in means and means[name] <= MARGIN[0] * best + MARGIN[1]:
            return name
    return min(means, key=means.get)


# -- drift: did the box move on the cup during the run? ---------------------------

def _drift(kin, rows, params, mounts):
    """Each reading's mount as its own markers say (through the model) minus
    the fitted one (translation, hand frame, mm), against the measuring
    order. A trend within a mounting is the box slipping on the cup (on 10-30
    the camera-minus-URDF tip height crept 22 mm over the run)."""
    hands = _hands(kin, _light(rows), params)
    d, idx, seg = [], [], []
    for j, r in enumerate(rows):
        if r['n'] < EVAL_MARKERS or r['segment'] not in mounts:
            continue
        implied = np.linalg.inv(hands[j]) @ r['t_wo'] @ r['cam_box']
        d.append(1000.0 * (implied[:3, 3] - mounts[r['segment']][:3, 3]))
        idx.append(r['index'])
        seg.append(r['segment'])
    out = {'max_mm': 0.0, 'slope_mm_per_reading': 0.0, 'trend_mm': 0.0, 'slope_xyz': [0.0] * 3,
           'flag': False, 'n': len(d)}
    if len(d) < 4:
        out['text'] = f'drift: only {len(d)} readings - not judged'
        return out
    d, idx, seg = np.array(d), np.array(idx, float), np.array(seg)
    # within each mounting (a new one starts where a slip was seen): the
    # readings round their own means
    x, y, span = np.zeros_like(idx), np.zeros_like(d), 0.0
    for g in set(seg.tolist()):
        m = seg == g
        x[m] = idx[m] - idx[m].mean()
        y[m] = d[m] - d[m].mean(axis=0)
        span = max(span, float(idx[m].max() - idx[m].min()))
    slope = (x @ y) / max(float(x @ x), 1e-9)
    trend = float(np.linalg.norm(slope) * span)
    out.update(max_mm=float(np.max(np.linalg.norm(d, axis=1))),
               slope_mm_per_reading=float(np.linalg.norm(slope)), trend_mm=trend,
               slope_xyz=[float(v) for v in slope], flag=trend > DRIFT_MM)
    out['text'] = (f'drift: box {trend:.1f} mm over the run (hand x/y/z '
                   f'{", ".join(f"{v * span:+.1f}" for v in slope)} mm), largest {out["max_mm"]:.1f} '
                   f'mm' + (' - THE BOX MOVED ON THE CUP' if out['flag'] else ' - steady'))
    return out


# -- the public part ---------------------------------------------------------------

def _prepare(arm, readings, tip_offset, folds, log, urdf_path, variant=None, fixed=None):
    """Kinematics, box geometry, usable rows (screened under the given model)
    and spatial folds."""
    kin = Arm(arm, urdf_path)
    geometry = _geometry(readings, log)
    rows = _rows(kin, readings, geometry)
    if len(rows) < MIN_READINGS:
        raise ValueError(f'only {len(rows)} usable box readings (need {MIN_READINGS})')
    rows, dropped, _got = _screen(kin, rows, variant, fixed)
    if len(rows) < MIN_READINGS:
        raise ValueError(f'only {len(rows)} box readings left after dropping {dropped} '
                         f'(need {MIN_READINGS})')
    folds = int(max(2, min(folds, len(rows))))
    return kin, geometry, rows, dropped, _folds(rows, folds, tip_offset), folds


def evaluate(arm, readings, params, tip_offset, folds=5, log=print, urdf_path=None):
    """How far off a FIXED model (params dict, or None = the plain URDF) is
    on these readings: only the box mounts are fitted, cross-validated as in
    fit_auto. An old model's tip_delta / anchor_shift / hysteresis (with the
    readings' 'dirs') are honoured. Returns {'mean_mm', 'p90_mm', 'max_mm',
    'n', 'per_point', 'dropped'}."""
    fixed = _as_params(params) if params is not None else None
    kin, _g, rows, dropped, fold_of, _f = _prepare(arm, readings, tip_offset, folds, log,
                                                   urdf_path, fixed=fixed)
    summary, points = _cv(kin, rows, fold_of, tip_offset, fixed=fixed)
    return plain(dict(summary, per_point=points, dropped=dropped))


def fit_auto(arm, readings, tip_offset, current=None, folds=5, log=print, urdf_path=None):
    """The arm model from the box readings, chosen by what predicts left-out
    postures best (see the module's docstring). readings: the run's READING
    dicts ('q', 'seen', 'k', 'd', 't_wo', 'cam_box', 'rms', 'n', 'kind',
    'segment', 'index'). current: the model in use now (arm_model.yaml's
    entry, with its tip_delta / anchor_shift), scored the same way.
    Raises ValueError with too few usable readings."""
    t0 = time.monotonic()
    # drop misreads under the most flexible model: a reading only a simpler
    # model cannot explain is not a misread
    kin, geometry, rows, dropped, fold_of, folds = _prepare(
        arm, readings, tip_offset, folds, log, urdf_path, variant=VARIANTS[-1])
    log(f'autocal {arm}: {len(rows)} readings ({dropped} dropped), '
        f'{len(set(r["segment"] for r in rows))} box mounting(s), {folds} folds, box '
        + ('as built' if geometry else 'as drawn'))
    cv, per = {}, {}
    for name in ORDER:
        cv[name], per[name] = _cv(kin, rows, fold_of, tip_offset,
                                  variant=None if name == 'urdf' else name)
        log(f'autocal {arm}: {name:9s} {_text(cv[name])} on left-out postures')
    cur = _as_params(current) if current else None
    cur_cv = None
    if cur is not None:
        cur_cv, _p = _cv(kin, rows, fold_of, tip_offset, fixed=cur)
        log(f'autocal {arm}: {"current":9s} {_text(cur_cv)} on left-out postures')
    variant = _choose(cv)
    final = _fit(kin, rows, None if variant == 'urdf' else variant)
    params = _zero_params() if variant == 'urdf' else final['params']
    fit_pts = _point_errors(kin, rows, params, final['mounts'], tip_offset)
    drift = _drift(kin, rows, params, final['mounts'])
    val = dict(cv[variant], per_point=per[variant])
    log(f'autocal {arm}: chose {variant} - {_text(cv[variant])} on left-out postures, '
        f'fit {final["fit_px"]:.2f} px; {drift["text"]} ({time.monotonic() - t0:.0f} s)')
    segs = max(final['mounts']) + 1
    stats = {'model': variant, 'box_tips': len(rows), 'box_holdout_mm': val['mean_mm'],
             'box_holdout_p90_mm': val['p90_mm'], 'box_fit_mm': _summary(fit_pts)['mean_mm'],
             'cv': {k: {'mean_mm': v['mean_mm'], 'p90_mm': v['p90_mm']} for k, v in cv.items()},
             'dropped': dropped, 'segments': len(final['mounts']), 'fit_px': final['fit_px'],
             'tip_offset': float(tip_offset), 'calibration': 'autocal (ctm/accucal.py)'}
    return plain({
        'variant': variant,
        'params': {k: params[k].tolist() for k in ('base', 'offset', 'comply')},
        'mounts': [final['mounts'][g].tolist() if g in final['mounts'] else None
                   for g in range(segs)],
        'geometry': {f: m.tolist() for f, m in geometry.items()} if geometry else None,
        'cv': cv,
        'val': val,
        'baseline': {'urdf': {k: cv['urdf'][k] for k in ('mean_mm', 'p90_mm', 'max_mm')},
                     'current': ({k: cur_cv[k] for k in ('mean_mm', 'p90_mm', 'max_mm')}
                                 if cur_cv is not None else None)},
        'dropped': dropped,
        'drift': drift,
        'stats': stats,
    })


def _text(s):
    if s['mean_mm'] is None:
        return 'not scored'
    return (f'mean {s["mean_mm"]:.1f} mm, 90 % under {s["p90_mm"]:.1f}, max '
            f'{s["max_mm"]:.1f} ({s["n"]})')


def saveable(result):
    """fit_auto's model as armcal.save_model wants its params: the fitted
    base, offset and comply; backlash, link errors, tip_delta and
    anchor_shift explicitly zero (so nothing of an old model lingers)."""
    p = result['params']
    return {'base': np.asarray(p['base'], float), 'offset': np.asarray(p['offset'], float),
            'comply': np.asarray(p['comply'], float), 'backlash': np.zeros(7),
            'geom': np.zeros((7, 6)), 'tip_delta': np.zeros(3), 'anchor_shift': np.zeros(3)}
