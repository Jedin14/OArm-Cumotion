#!/usr/bin/env python3
"""Replay the recorded guarded presses (native/logs/guarded_lines.jsonl: the last 50, each with
its progress and torque rise every 10 ms) through the torque-limit rule -- the one before
2026-11-19 (the tool's 1.0 Nm limit over the whole press) and the one now (at/past the measured
surface, or while the tip is not keeping up). Run it before changing any contact threshold.

    python3 app/tests/replay_presses.py
"""
import json
import os

import numpy as np

LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'native', 'logs', 'guarded_lines.jsonl')
HARD, SOFT, STUCK_S, KEEP_FRAC = 2.0, 1.0, 0.35, 0.5


def first_stop(rec, new):
    """mm in where the limit stops this press (two samples over it), or None."""
    soft = rec['label'].startswith('left')              # the left arm carries the vacuum tool
    v = 10.0 if soft else 15.0                          # mm/s commanded
    tr, surf = rec['trace'], rec['surface_mm']
    n = 0
    for i, (t, p, d) in enumerate(tr):
        big = np.abs(d) if t >= 0.8 else np.zeros(7)
        if not soft:
            cap = HARD
        elif not new:
            cap = SOFT if p >= surf - 30.0 else HARD
        else:
            j = i
            while j > 0 and t - tr[j - 1][0] <= STUCK_S:
                j -= 1
            span = t - tr[j][0]
            keeping = True if span < 0.6 * STUCK_S else (p - tr[j][1]) >= KEEP_FRAC * v * span
            cap = SOFT if p >= 3.0 and (p >= surf or not keeping) else HARD
        # (the recorded torques are rounded to 0.01 Nm)
        n = n + 1 if np.any(big >= cap - 0.005) else 0
        if n >= 2:
            return p
    return None


rows = [json.loads(line) for line in open(LOG)]
false_gone, real_changed = [], []
print(f'{"time":19s} {"surface":>7s} {"end":>6s}  old stop  new stop')
for r in rows:
    if not r['trace']:
        continue
    o, n = first_stop(r, False), first_stop(r, True)
    mark = ''
    if o != n:
        if o is not None and o < r['surface_mm']:
            false_gone.append(r['time'])
            mark = '  <- stopped short before, goes on now'
        else:
            real_changed.append(r['time'])
            mark = '  <- CHANGED AT/PAST THE SURFACE'
    print(f"{r['time']:19s} {r['surface_mm']:7.1f} {r['trace'][-1][1]:6.1f}  "
          f"{'-' if o is None else f'{o:6.1f}':>8s}  {'-' if n is None else f'{n:6.1f}':>8s}{mark}")
print(f'{len(rows)} presses: {len(false_gone)} short stops removed, '
      f'{len(real_changed)} changed at/past the surface (want 0)')
