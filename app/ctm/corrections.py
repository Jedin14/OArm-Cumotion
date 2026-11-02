"""Touch corrections: where to aim the fingertip, relative to a point, so it
lands on it -- kept in touch_corrections.yaml, and the correction they give
at a new point.

Samples come from TEACH (you nudge the tip onto the spot), on top of the
current arm model (CAL ARMS). Each records the tool's mode ("level" or "top-down"):
the arm sags differently reaching down, so a mode uses only its own samples.
"""

import datetime
import math
import os

import numpy as np
import yaml

from ctm.common import WS, plain


# Taught touch corrections (TEACH): per arm, the nudge a touch needed at a point.
CORRECTIONS_FILE = os.path.join(WS, 'touch_corrections.yaml')


CORRECTION_RADIUS = 0.15        # samples further than this from a point are ignored, m


def load_corrections():
    try:
        with open(CORRECTIONS_FILE) as handle:
            return (yaml.safe_load(handle) or {}).get('samples', []) or []
    except (OSError, yaml.YAMLError):
        return []


def _sample(arm, point, correction, source, mode, model=None):
    out = {'arm': arm, 'point': [round(float(v), 4) for v in point],
           'correction': [round(float(v), 4) for v in correction],
           'source': source, 'mode': mode,
           'when': datetime.datetime.now().isoformat(timespec='seconds')}
    if model:
        out['model'] = model            # taught on top of this arm model (CALIBRATE ARMS)
    return plain(out)


def _write(samples):
    text = yaml.safe_dump({'samples': samples}, sort_keys=False)
    with open(CORRECTIONS_FILE, 'w') as handle:
        handle.write('# Touch corrections (click_to_move.py TEACH): where the\n'
                     '# fingertip had to be aimed, relative to a point, to land on it.\n' + text)


def save_correction(arm, point, correction, mode='level', source='teach', model=None):
    samples = load_corrections()
    samples.append(_sample(arm, point, correction, source, mode, model))
    _write(samples)
    return len(samples)


def learned_correction(arm, point, mode='level', model=None):
    """(correction (3,), samples nearby, samples in all) for this arm at this point.

    Two layers, so a correction carries to where nothing was taught yet:

    * a smooth trend through all of the arm's taught touches, c(p) = b + A p,
      ridge-regularised so that with few or bunched-up samples it stays near
      their average instead of extrapolating wildly. The arm's error changes
      with reach -- touches taught at 0.35 m did not hold at 0.46 m
      (2026-10-07) -- and the trend is what follows that, once touches have
      been taught at more than one distance;
    * a Gaussian-weighted (sigma 6 cm) blend of what the trend still misses
      at the samples within CORRECTION_RADIUS, for the local detail.
    """
    # Only samples taught on top of the same arm model: one taught before a
    # CALIBRATE ARMS would correct again what the model already corrects.
    samples = [s_ for s_ in load_corrections()
               if s_.get('arm') == arm and s_.get('model') == model]
    # this mode's own only: a top-down touch's correction says nothing about
    # a front one (front was right with none, 2026-10-14)
    samples = [s_ for s_ in samples if s_.get('mode', 'level') == mode]
    if not samples:
        return np.zeros(3), 0, 0
    pts = np.array([s_['point'] for s_ in samples], dtype=float)
    cor = np.array([s_['correction'] for s_ in samples], dtype=float)
    centre = pts.mean(0)
    x = np.hstack([np.ones((len(pts), 1)), pts - centre])        # [1, dp]
    # Ridge on the slope only: expect ~10 mm per 10 cm at most, so a slope
    # needs real evidence spread over space before it moves off zero.
    ridge = np.diag([1e-9, 1.0, 1.0, 1.0]) * (0.005 / 0.1) ** 2
    coef = np.linalg.solve(x.T @ x + ridge, x.T @ cor)          # (4, 3)

    def trend(p):
        return np.hstack([1.0, np.asarray(p) - centre]) @ coef

    residual = cor - x @ coef
    local, weight, nearby = np.zeros(3), 0.0, 0
    for p_, r_ in zip(pts, residual):
        dist = float(np.linalg.norm(p_ - point))
        if dist > CORRECTION_RADIUS:
            continue
        w = math.exp(-0.5 * (dist / 0.06) ** 2)
        local += w * r_
        weight += w
        nearby += 1
    correction = trend(point) + (local / weight * min(1.0, weight) if weight > 0 else 0.0)
    norm = float(np.linalg.norm(correction))
    if norm > 0.06:                                  # never more than 6 cm
        correction *= 0.06 / norm
    return correction, nearby, len(samples)
