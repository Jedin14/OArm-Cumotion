#!/usr/bin/env python3
"""Checks on pick_place_orchestrator's grasp pose maths. No robot needed.

    source native/setup.bash
    python3 native/tests/test_grasp_geometry.py

top_down_quat is the one piece of geometry that will happily drive the wrist into
the table if a sign is wrong, and the failure looks like a planning problem
rather than a maths problem. The two properties asserted here are:

  * the tool approach axis (hand_tcp +Z, since hand_tcp sits 0.08 m along +Z of
    openarm_<arm>_hand) points at world -Z, i.e. straight down;

  * the finger closing direction (hand_tcp +/-Y, since finger_joint1's axis is
    0 -1 0 in the hand frame) sits 90 degrees off the requested yaw, so feeding
    in the object's long axis closes the fingers across it rather than along it.

Expected values are derived from the URDF and the joint axis, not from the code.
"""

import importlib.util
import math
import os
import sys

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
FAILURES = []


def load_orchestrator():
    # Loading by path does not put the workspace on sys.path, and the
    # orchestrator imports vlm_prompt from beside itself.
    if WS not in sys.path:
        sys.path.insert(0, WS)
    path = os.path.join(WS, 'pick_place_orchestrator.py')
    spec = importlib.util.spec_from_file_location('pick_place_orchestrator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def check(label, got, want, tol=1e-6):
    ok = np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float),
                     atol=tol)
    print(f'{"pass" if ok else "FAIL"}  {label}')
    if not ok:
        print(f'        got  {got}')
        print(f'        want {want}')
        FAILURES.append(label)


def test_quat_mul(m):
    check('identity * identity', m.quat_mul((0, 0, 0, 1), (0, 0, 0, 1)), (0, 0, 0, 1))
    # Two quarter turns about Z compose into a half turn about Z.
    quarter = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    check('Rz(90) * Rz(90) = Rz(180)', m.quat_mul(quarter, quarter), (0, 0, 1, 0))


def test_unit_quaternions(m):
    for yaw in (0.0, 0.4, math.pi / 2, -1.2, math.pi):
        norm = np.linalg.norm(m.top_down_quat(yaw))
        check(f'top_down_quat({yaw:.2f}) is unit', norm, 1.0)


def test_approach_axis_points_down(m):
    for yaw in (0.0, math.pi / 4, 1.0, -2.0, math.pi):
        rot = quat_to_rot(m.top_down_quat(yaw))
        check(f'tool +Z is world -Z at yaw {yaw:.2f}', rot[:, 2], [0.0, 0.0, -1.0])


def test_closing_direction_crosses_the_object(m):
    for yaw in (0.0, math.pi / 4, 1.0, -2.0):
        rot = quat_to_rot(m.top_down_quat(yaw))
        closing = rot[:, 1]
        check(f'closing direction is yaw+90 at yaw {yaw:.2f}',
              math.atan2(closing[1], closing[0]), yaw + math.pi / 2)
        # Perpendicular to the object axis is the whole point of the +90.
        axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        check(f'closing direction perpendicular to the axis at yaw {yaw:.2f}',
              float(np.dot(closing, axis)), 0.0)


def test_dist(m):
    check('dist 3d', m.dist((0.0, 0.0, 0.0), (1.0, 2.0, 2.0)), 3.0)
    check('dist 2d slice', m.dist((0.0, 0.0), (3.0, 4.0)), 5.0)


def test_retry_ladder_escalates(m):
    names = [s['name'] for s in m.STRATEGIES]
    print(f'pass  retry ladder: {" -> ".join(names)}')
    modifiers = {(s['yaw_offset'], s['z_offset'], s['refresh_octomap'])
                 for s in m.STRATEGIES}
    check('every strategy is a distinct escalation',
          len(modifiers), len(m.STRATEGIES) - 1)   # 'nominal' and 'redetect' share
    check('the ladder ends with an octomap refresh',
          [m.STRATEGIES[-1]['refresh_octomap']], [True])


def main():
    module = load_orchestrator()
    test_quat_mul(module)
    test_unit_quaternions(module)
    test_approach_axis_points_down(module)
    test_closing_direction_crosses_the_object(module)
    test_dist(module)
    test_retry_ladder_escalates(module)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} failure(s): {", ".join(FAILURES)}')
        return 1
    print('all grasp geometry checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
