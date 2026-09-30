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

import ast
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


def load_grasp_node_source():
    """The server's source, read rather than imported.

    Importing it needs torch, open3d and graspnet-baseline's compiled
    extensions, none of which belong in a geometry check -- and the venv
    they live in is not the one this runs under.
    """
    return open(os.path.join(WS, 'grasp', 'grasp_node.py')).read()


def test_span_uses_graspnets_own_jaw_volume():
    """The measured span has to be measured where the fingers actually are.

    The model's width is what GraspNet recommends the gripper open to,
    binned over a 100 mm gripper's range; these jaws span 44 mm, so the
    recommendation is coarse exactly where it has to be fine. Measuring the
    cloud is what makes a candidate usable -- but only if the volume
    measured is the one the fingers sweep, which is
    (depth - finger_length, depth) along the approach and |y| < width/2
    across it. Those bounds are graspnet-baseline's, so they are checked
    against its source rather than against a number typed twice.
    """
    node = load_grasp_node_source()
    span = node.split('def _span_between_jaws(')[1].split('\n    def ')[0]
    check('the span is taken along the closing direction',
          'local[between, 1]' in span, True)
    check('over the space between the fingers, not the whole cloud',
          'local[:, 0] > depth - FINGER_LENGTH' in span, True)
    check('and inside the gripper\'s own thickness',
          'local[:, 2]) < max(height' in span, True)
    check('noise is not an object',
          'MIN_SPAN_POINTS' in span, True)

    detector = os.path.join(WS, 'third_party', 'graspnet-baseline', 'utils',
                            'collision_detector.py')
    if os.path.exists(detector):
        theirs = open(detector).read()
        want = None
        for line in theirs.splitlines():
            if 'self.finger_length' in line and '=' in line and 'depths' not in line:
                want = float(line.split('=')[1].strip())
                break
        ours = float(node.split('FINGER_LENGTH = ')[1].split('\n')[0])
        check('FINGER_LENGTH matches graspnet-baseline\'s own',
              ours, want)
    else:
        print('pass  (graspnet-baseline not fetched; finger length unchecked)')

    served = node.split('def _serve(')[1].split('\n    def ')[0]
    check('a too-wide grasp is measured before it is refused',
          served.index('_span_between_jaws') < served.index('too_wide += 1'),
          True)
    check('and what it is then published as is the measured span',
          "entry['width'] = round(span + margin, 4)" in served, True)
    check('with the model\'s own number kept beside it',
          "entry['model_width']" in served, True)
    check('every filter says how many it took, not just the width one',
          all(part in served for part in
              ('below score', 'further than', 'these jaws span')), True)


def load_span_function():
    """_span_between_jaws, lifted out by source and run on its own.

    Importing grasp_node needs torch, open3d and graspnet-baseline's
    compiled extensions, which live in third_party/grasp_venv -- not the
    interpreter this runs under. The function itself touches nothing but
    numpy, so the module is parsed and just that method compiled.
    """
    node = load_grasp_node_source()
    tree = ast.parse(node)
    consts = [n for n in tree.body if isinstance(n, ast.Assign)
              and getattr(n.targets[0], 'id', '') in
              ('FINGER_LENGTH', 'MIN_SPAN_POINTS')]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body
              if getattr(n, 'name', '') == '_span_between_jaws')
    namespace = {'np': np}
    exec(compile(ast.Module(body=consts + [fn], type_ignores=[]),
                 'grasp_node', 'exec'), namespace)
    return namespace['_span_between_jaws']


def test_span_measures_what_fits(m):
    """The arithmetic, on clouds whose answer is known by construction."""
    span = load_span_function()

    def row(width, height, depth, rot=None, at=None):
        raw = np.zeros(17)
        raw[0], raw[1], raw[2], raw[3] = 0.9, width, height, depth
        raw[4:13] = (np.eye(3) if rot is None else rot).reshape(-1)
        raw[13:16] = np.zeros(3) if at is None else at
        return raw

    rng = np.random.default_rng(3)
    n, depth, width, height = 4000, 0.02, 0.08, 0.02

    def slab(half_across, x_lo=None, x_hi=None):
        return np.stack([
            rng.uniform(depth - 0.06 if x_lo is None else x_lo,
                        depth if x_hi is None else x_hi, n),
            rng.uniform(-half_across, half_across, n),
            rng.uniform(-0.005, 0.005, n)], axis=1)

    here = slab(0.010)
    check('a 20 mm slab measures 20 mm however wide the model asked for',
          span(None, row(width, height, depth), here), 0.020, tol=0.002)
    # The fingers sweep (depth - finger_length, depth). Material behind the
    # palm is not between the jaws and must not widen the answer -- this is
    # the case that would quietly refuse a good rim grasp on a deep object.
    behind = slab(0.039, depth - 0.20, depth - 0.09)
    check('material behind the palm does not count',
          span(None, row(width, height, depth), np.vstack([here, behind])),
          0.020, tol=0.002)
    check('and something genuinely too wide reads as too wide',
          span(None, row(width, height, depth), slab(0.035)) > 0.06, True)
    away = np.stack([rng.uniform(0.5, 0.6, n), rng.uniform(-0.01, 0.01, n),
                     rng.uniform(-0.005, 0.005, n)], axis=1)
    check('nothing between the jaws gives no answer rather than zero',
          span(None, row(width, height, depth), away) is None, True)

    # Measured across the grasp's own closing axis, not the world's.
    turn = math.radians(37.0)
    rot = np.array([[math.cos(turn), -math.sin(turn), 0.0],
                    [math.sin(turn), math.cos(turn), 0.0],
                    [0.0, 0.0, 1.0]])
    at = np.array([0.1, -0.2, 0.3])
    check('the same slab under a turned, offset grasp measures the same',
          span(None, row(width, height, depth, rot, at), here @ rot.T + at),
          span(None, row(width, height, depth), here), tol=1e-9)


def test_model_grasps_are_bounded_by_the_vertical_column(m):
    """A side grasp cannot be flown down a column that only moves in z.

    The cycle's approach and retreat interpolate in z and hold x and y, so
    a tilted *tool* is fine and a tilted *path* is not. The model proposes
    both; the steep ones have to be skipped rather than driven down across
    the object.
    """
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    proposals = source.split('def model_proposals(')[1].split('\n    def ')[0]
    check('the tilt is read off the candidate',
          "entry.get('tilt_deg')" in proposals, True)
    check('and compared against a limit',
          'grasp_model_max_tilt' in proposals, True)
    check('the ones skipped are named, not silently dropped',
          'Skipped rather than mis-flown' in proposals, True)
    check('the pre-grasp is straight up, which is why the limit exists',
          "grasp[2] + approach" in proposals, True)
    # The column itself, so this test fails if it ever learns to follow an
    # approach axis and the limit is then wrong rather than merely cautious.
    heights = source.split('def column_heights(')[1].split('\n    def ')[0]
    check('the column is still a list of heights',
          'to_z if i == count else from_z' in heights, True)


def main():
    module = load_orchestrator()
    test_quat_mul(module)
    test_unit_quaternions(module)
    test_approach_axis_points_down(module)
    test_closing_direction_crosses_the_object(module)
    test_dist(module)
    test_retry_ladder_escalates(module)
    test_span_uses_graspnets_own_jaw_volume()
    test_span_measures_what_fits(module)
    test_model_grasps_are_bounded_by_the_vertical_column(module)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} failure(s): {", ".join(FAILURES)}')
        return 1
    print('all grasp geometry checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
