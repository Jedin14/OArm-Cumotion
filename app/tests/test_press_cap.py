#!/usr/bin/env python3
"""The guarded press (app/ctm/robot.py guarded_line) on scripted physics -- tip progress and
joint torques over time, in real time; no ROS, no robot. 2026-11-19: the vacuum tool's soft
1.0 Nm limit tripped on the shoulder's stick-slip in free air and stopped presses short of the
box. Optionally against an older robot.py for comparison:

    source native/setup.bash; PYTHONNOUSERSITE=1 python3 app/tests/test_press_cap.py [old/robot.py]
"""
import os
import sys
import time
import types

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
import ctm.robot as new_mod        # noqa: E402


def cls_of(m):
    return next(c for c in vars(m).values() if isinstance(c, type) and hasattr(c, 'guarded_line')
                and c.__module__ == m.__name__)


V = 0.010                          # m/s: the tool's contact speed


def stickslip(t, amp, period=0.3):
    ph = (t % period) / period
    return amp * (2 * ph - 1) * (1 if int(t / period) % 2 else -1)


def scenario(block_at, squash_to=None, ramp=3.0, amp=1.6):
    """Free air with stick-slip (0.15 s stalled, 0.15 s at 2V) up to block_at (m); the cup
    squashing freely on to squash_to; then blocked, the torque on j2 ramping `ramp` Nm/s."""
    end = squash_to if squash_to is not None else block_at
    t_block = end / V

    def prog(t):
        if t >= t_block:
            return end
        k, r = divmod(t, 0.3)
        return min(k * 0.3 * V + (0.0 if r < 0.15 else (r - 0.15) * 2 * V), end)

    def torq(t):
        e = np.zeros(7)
        if t < t_block:
            e[0] = stickslip(t, amp) if prog(t) < block_at else 0.3 * stickslip(t, amp)
        else:
            e[1] = ramp * (t - t_block)
        return e + np.random.default_rng(int(t * 1000)).normal(0, 0.01, 7)
    return prog, torq


def run(cls, sc, tool=True, surface=0.030, length=0.050):
    prog, torq = sc
    logs = []
    p0, way = np.array([0.5, 0.1, 0.6]), np.array([1.0, 0.0, 0.0])
    clock = {'t0': None}

    def now():
        return 0.0 if clock['t0'] is None else time.monotonic() - clock['t0']

    def lookup(base, frame):
        m = np.eye(4)
        m[:3, 3] = p0 + way * prog(now())
        return m

    def execute(traj, label, timeout, watch):
        clock['t0'] = time.monotonic()
        while now() < length / V + 0.5:
            if watch():
                return True, True
            time.sleep(0.01)
        return True, False
    log = types.SimpleNamespace(info=logs.append, warn=logs.append, warning=logs.append,
                                error=logs.append)
    me = types.SimpleNamespace(
        args=types.SimpleNamespace(contact_torque=0.3, max_touch_torque=2.0, soft_tool_torque=1.0,
                                   contact_speed=0.015, past_contact=0.02,
                                   tools={'left': {'on': tool}, 'right': {'on': False}}),
        arm_positions=lambda arm: [0.0] * 7, arm_efforts=lambda arm: list(torq(now())),
        cartesian_plan=lambda arm, pos, quat: (1.0, object(), [0.0] * 7), lookup=lookup,
        retime=lambda traj, L, speed: L / speed, execute=execute, get_logger=lambda: log,
        _save_trace=lambda *a: None)
    cls.guarded_line(me, 'left' if tool else 'right', p0 + way * length, [0, 0, 0, 1],
                     'left contact', needed=surface + 0.003)
    return round(1000 * prog(now()), 1), [m for m in logs if 'contact' in m or 'limit' in m]


FAILS = []


def check(label, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + label, '' if ok else info)
    if not ok:
        FAILS.append(label)


NEW = cls_of(new_mod)
OLD = None
if len(sys.argv) > 1:
    from importlib.machinery import SourceFileLoader
    OLD = cls_of(SourceFileLoader('ctm.robot_old', sys.argv[1]).load_module())

mm, log = run(NEW, scenario(0.030, squash_to=0.031))
check('stick-slip in free air (swings 1.0-1.2 Nm as recorded): goes on to the box (>= 29.5 mm)',
      mm >= 29.5, (mm, log[-1:]))
if OLD is not None:
    mm, log = run(OLD, scenario(0.030, squash_to=0.031))
    print(f'     (the given old robot.py stopped at {mm} mm: {log[-1:]})')
mm, log = run(NEW, scenario(0.008, amp=0.3))
check('a box 22 mm nearer than measured: stopped there at the tool limit (1.0 Nm)',
      7.0 <= mm <= 9.0 and any('torque limit' in m and 'max 1.0' in m for m in log), (mm, log[-1:]))
mm, log = run(NEW, scenario(0.030, squash_to=0.044, amp=0.3))
check('the normal press: felt at the squashed cup (~44 mm)', mm >= 43.0, (mm, log[-1:]))
mm, log = run(NEW, scenario(0.030, squash_to=0.031), tool=False)
check('no tool (right arm): stopped at the box face', 29.5 <= mm <= 32.0, (mm, log[-1:]))
print('ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}')
sys.exit(1 if FAILS else 0)
