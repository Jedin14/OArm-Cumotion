#!/usr/bin/env python3
"""Check that a pose mirrored to the other arm really is the same posture.

The claim under test is MIRRORED_JOINTS in record_states.py: reflecting a pose
to the other arm negates every joint except joint4. That is not obvious -- the
description mirrors the link origins in y but flips only joint7's axis -- so it
is checked here rather than argued, by forward kinematics off the generated
URDF.

A true mirror must put the other arm's tool at the y-mirror of this arm's tool
with the whole rotation mirrored to match. Anything else is a different posture
that merely looks symmetric in the joint numbers.

The postures are deliberately varied, including random ones. An earlier rule
that negated joint1/3/5/7 was exact on ready_state and pre_pick_state and 107
mm out on drop_state -- the first two hold joint2 and joint6 near zero, so
similar-looking recordings cannot tell the rules apart.

    source native/setup.bash && python3 native/tests/test_mirror_states.py

Needs xacro (from the ROS environment) to generate the URDF, but no robot and
no running stack.
"""

import itertools
import math
import os
import subprocess
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if WS not in sys.path:
    sys.path.insert(0, WS)

XACRO = os.path.join(WS, 'src/openarm_description/urdf/robot/v10.urdf.xacro')

failures = []


def check(label, got, want, tolerance=None):
    if tolerance is None:
        ok = got == want
    else:
        ok = abs(got - want) <= tolerance
    print(f'{"pass" if ok else "FAIL"}  {label}')
    if not ok:
        print(f'        got  {got}')
        print(f'        want {want}')
        failures.append(label)
    return ok


# -- forward kinematics straight off the URDF --------------------------------

def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def homogeneous(rotation, translation):
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def about_axis(axis, angle):
    unit = np.asarray(axis, float)
    unit = unit / np.linalg.norm(unit)
    skew = np.array([[0, -unit[2], unit[1]],
                     [unit[2], 0, -unit[0]],
                     [-unit[1], unit[0], 0]])
    return (np.eye(3) + math.sin(angle) * skew
            + (1 - math.cos(angle)) * (skew @ skew))


class Kinematics:
    """Just enough URDF to walk a chain and place a frame."""

    def __init__(self, urdf_text):
        self.joints = {}
        self.parent_joint = {}
        for joint in ET.fromstring(urdf_text).findall('joint'):
            origin = joint.find('origin')
            axis = joint.find('axis')
            child = joint.find('child').get('link')
            self.joints[joint.get('name')] = {
                'type': joint.get('type'),
                'xyz': np.array([float(v) for v in (
                    (origin.get('xyz') if origin is not None else None)
                    or '0 0 0').split()]),
                'rpy': [float(v) for v in (
                    (origin.get('rpy') if origin is not None else None)
                    or '0 0 0').split()],
                'axis': np.array([float(v) for v in (
                    (axis.get('xyz') if axis is not None else None)
                    or '0 0 1').split()]),
                'parent': joint.find('parent').get('link'),
            }
            self.parent_joint[child] = joint.get('name')

    def chain(self, link):
        names = []
        while link in self.parent_joint:
            name = self.parent_joint[link]
            names.append(name)
            link = self.joints[name]['parent']
        return list(reversed(names))

    def pose(self, link, values):
        frame = np.eye(4)
        for name in self.chain(link):
            joint = self.joints[name]
            frame = frame @ homogeneous(rpy_to_matrix(*joint['rpy']),
                                        joint['xyz'])
            angle = values.get(name, 0.0)
            if joint['type'] in ('revolute', 'continuous'):
                frame = frame @ homogeneous(
                    about_axis(joint['axis'], angle), np.zeros(3))
            elif joint['type'] == 'prismatic':
                frame = frame @ homogeneous(np.eye(3), joint['axis'] * angle)
        return frame


def build_urdf():
    """Generate the bimanual URDF, or None if xacro is unavailable."""
    if not os.path.exists(XACRO):
        print(f'skip: {XACRO} not found')
        return None
    try:
        out = subprocess.run(['xacro', XACRO, 'bimanual:=true'],
                             capture_output=True, text=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f'skip: could not run xacro ({exc}) -- source native/setup.bash')
        return None
    if out.returncode != 0:
        print(f'skip: xacro failed: {out.stderr.strip()[-300:]}')
        return None
    return out.stdout


def joint_values(arm, values):
    return {f'openarm_{arm}_joint{i + 1}': v for i, v in enumerate(values)}


class FakePlayNode:
    """Stands in for the arm, so --play's sequencing can be checked offline."""

    def __init__(self, start):
        self.start = list(start)
        self.current = list(start)
        self.visited = []
        self.fail_at = None
        self.die_at = None            # label at which the planner "crashes"
        self.planner_up = True

    def snapshot(self):
        return {'joints': list(self.current), 'tcp_xyz': [0.1, 0.2, 0.3]}

    def planner_is_serving(self, pipeline, timeout=3.0):
        return self.planner_up

    def move_to_joints(self, values, label, pipeline, speed, attempts=3):
        self.visited.append(label)
        if label == self.die_at:
            # What cuMotion actually did: planned several goals, then died, so
            # everything after it fails with the planner simply absent.
            self.planner_up = False
            return False
        if not self.planner_up or label == self.fail_at:
            return False
        self.current = list(values)
        return True


def play_args(arm='left'):
    return types.SimpleNamespace(arm=arm, pipeline='cumotion', speed=0.15,
                                 dwell=0.0, yes=True, plan_attempts=3,
                                 file='pick_place_states_left.yaml')


def check_play(rs):
    print('\n-- --play sequencing --')
    states = {
        'ready_state': {'joints': [0.1] * 7},
        'pre_pick_state': {'joints': [0.2] * 7},
        'drop_state': {'joints': [0.3] * 7},
    }
    order = ['ready_state', 'pre_pick_state', 'drop_state']
    start = [0.9] * 7

    node = FakePlayNode(start)
    code = play_states(rs, node, play_args(), states, order)
    check('a clean run reports success', code, 0)
    check('it visits every pose in order, then goes back',
          node.visited, order + ['starting posture'])
    check('and the arm ends where it started', node.current, start)

    # The point of capturing the start first: it is replayed even though it is
    # not one of the recorded poses.
    check('the returned-to pose is the captured start, not a recorded one',
          start not in [s['joints'] for s in states.values()], True)

    node = FakePlayNode(start)
    node.fail_at = 'pre_pick_state'
    code = play_states(rs, node, play_args(), states, order)
    check('a failure part-way is reported', code, 1)
    check('and it still returns to the starting posture',
          node.visited[-1], 'starting posture')
    check('leaving the arm where it started', node.current, start)
    check('and it does not carry on to later poses',
          'drop_state' in node.visited, False)

    node = FakePlayNode(start)
    code = play_states(rs, node, play_args(), {'drop_state': states['drop_state']},
                       order)
    check('poses not on file are skipped, not faked',
          node.visited, ['drop_state', 'starting posture'])

    node = FakePlayNode(start)
    code = play_states(rs, node, play_args(), {}, order)
    check('an empty file plays nothing at all', node.visited, [])
    check('and says so', code, 1)

    # A dead planner must be caught before the arm moves. Every goal comes back
    # PLANNING_FAILED or TIMED_OUT when the cuMotion node is gone, which reads
    # as "that pose is unreachable" unless it is named for what it is.
    node = FakePlayNode(start)
    node.planner_up = False
    code = play_states(rs, node, play_args(), states, order)
    check('a dead planner is caught before anything moves', node.visited, [])
    check('and reported rather than attempted', code, 1)

    # And when it dies part-way, the arm is left put and the run says why
    # instead of blaming the starting posture.
    node = FakePlayNode(start)
    node.die_at = 'pre_pick_state'
    code = play_states(rs, node, play_args(), states, order)
    check('a mid-run crash stops the run', code, 1)
    check('it got through the pose before the crash',
          node.visited[:2], ['ready_state', 'pre_pick_state'])
    check('it still tries to bring the arm home',
          'starting posture' in node.visited, True)
    check('and the arm is left where the crash left it',
          node.current, states['ready_state']['joints'])


def play_states(rs, node, args, states, wanted):
    """Call the real thing, with stdout kept quiet."""
    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        return rs.play_states(node, args, states, wanted)


def main():
    import record_states as rs

    # The pose the right arm actually uses, from the orchestrator's default.
    right_ready = [-0.828374, 0.000191, -0.000191, 2.324140,
                   -0.000191, -0.000191, -0.391966]

    print('-- the mirror rule itself --')
    check('every joint but joint4 flips',
          tuple(rs.MIRRORED_JOINTS), (0, 1, 2, 4, 5, 6))

    ones = [1.0] * 7
    check('and joint4 is the one left alone',
          rs.mirror_joints(ones), [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    check('mirroring twice is the original pose',
          rs.mirror_joints(rs.mirror_joints(right_ready)), right_ready)
    check('other_arm pairs them up',
          (rs.other_arm('right'), rs.other_arm('left')), ('left', 'right'))

    mirrored = rs.mirror_joints(right_ready)
    check('right READY mirrors to +47.5 deg at joint1',
          round(math.degrees(mirrored[0]), 1), 47.5)
    check('joint4 keeps its sign, +133.2 deg',
          round(math.degrees(mirrored[3]), 1), 133.2)
    # The one that is easy to get wrong: joint7 must flip, because its axis is
    # the single thing the description mirrors between the arms.
    check('joint7 flips to +22.5 deg, not -22.5',
          round(math.degrees(mirrored[6]), 1), 22.5)

    print("\n-- against the URDF's own kinematics --")
    urdf = build_urdf()
    if urdf is None:
        print('FK checks skipped (see above); the rule checks above still ran')
        return 1 if failures else 0

    kin = Kinematics(urdf)
    # Guard against a degenerate chain: an unknown tip link walks no joints and
    # returns the identity, and every comparison below would then compare zero
    # against zero and pass.
    for arm in ('right', 'left'):
        check(f'the {arm} chain reaches its tool',
              len(kin.chain(f'openarm_{arm}_hand_tcp')) >= 8, True)
    if failures:
        print('the URDF has no usable chain; the FK checks below are meaningless')
        return 1

    right = kin.pose('openarm_right_hand_tcp',
                     joint_values('right', right_ready))
    left = kin.pose('openarm_left_hand_tcp', joint_values('left', mirrored))
    check('and puts the tool somewhere other than the origin',
          float(np.linalg.norm(right[:3, 3])) > 0.1, True)

    target = np.array([right[0, 3], -right[1, 3], right[2, 3]])
    print(f'        right tool  [{right[0, 3]:+.4f} {right[1, 3]:+.4f} '
          f'{right[2, 3]:+.4f}]')
    print(f'        mirror wants[{target[0]:+.4f} {target[1]:+.4f} '
          f'{target[2]:+.4f}]')
    print(f'        left tool   [{left[0, 3]:+.4f} {left[1, 3]:+.4f} '
          f'{left[2, 3]:+.4f}]')
    error = float(np.linalg.norm(left[:3, 3] - target))
    check('the mirrored pose lands on the y-mirror of the tool (<1 mm)',
          error < 0.001, True)
    print(f'        position error {error * 1000:.2f} mm')

    # The tool axis has to mirror too: same x and z, y negated. A pose that
    # matches in position but not orientation is not the same grasp.
    axis_right = right[:3, 2]
    axis_left = left[:3, 2]
    axis_error = float(np.linalg.norm(
        axis_left - np.array([axis_right[0], -axis_right[1], axis_right[2]])))
    check('and the tool axis mirrors with it', axis_error < 0.002, True)
    print(f'        right tool-Z [{axis_right[0]:+.3f} {axis_right[1]:+.3f} '
          f'{axis_right[2]:+.3f}]  left tool-Z [{axis_left[0]:+.3f} '
          f'{axis_left[1]:+.3f} {axis_left[2]:+.3f}]')

    # And the mistakes this rule exists to prevent.
    naive = [-right_ready[0]] + right_ready[1:]
    naive_pose = kin.pose('openarm_left_hand_tcp', joint_values('left', naive))
    naive_error = float(np.linalg.norm(naive_pose[:3, 3] - target))
    check('negating joint1 alone is measurably wrong (>10 cm out)',
          naive_error > 0.10, True)
    print(f'        joint1-only error {naive_error * 1000:.1f} mm, tool-Z z '
          f'{naive_pose[2, 2]:+.3f} against {axis_right[2]:+.3f} '
          f'(tips the wrong way)')

    print('\n-- no other sign pattern does as well --')
    # Random postures on purpose: the recorded three all hold joint2 and joint6
    # near zero, and a rule that gets those two joints wrong still passes on
    # them. Seeded, so a failure is reproducible.
    rng = np.random.default_rng(7)
    postures = [right_ready] + [
        list(rng.uniform(-0.6, 0.6, 7) + np.array([0, 0, 0, 1.8, 0, 0, 0]))
        for _ in range(12)]

    def mirror_error(signs, values):
        source = kin.pose('openarm_right_hand_tcp',
                          joint_values('right', values))
        image = kin.pose(
            'openarm_left_hand_tcp',
            joint_values('left', [s * v for s, v in zip(signs, values)]))
        want = np.array([source[0, 3], -source[1, 3], source[2, 3]])
        reflect = np.diag([1.0, -1.0, 1.0])
        return (float(np.linalg.norm(image[:3, 3] - want))
                + float(np.linalg.norm(
                    image[:3, :3] - reflect @ source[:3, :3] @ reflect)))

    rule = [-1 if i in rs.MIRRORED_JOINTS else 1 for i in range(7)]
    rule_worst = max(mirror_error(rule, q) for q in postures)
    check('the rule is exact over 13 postures', rule_worst < 1e-3, True)
    print(f'        worst combined error {rule_worst:.6f}')

    rivals = []
    for signs in itertools.product([1, -1], repeat=7):
        if list(signs) == rule:
            continue
        rivals.append((max(mirror_error(signs, q) for q in postures), signs))
    rivals.sort()
    check('and every one of the other 127 patterns is worse',
          rivals[0][0] > 10 * max(rule_worst, 1e-9), True)
    flipped = [i + 1 for i, s in enumerate(rivals[0][1]) if s < 0]
    print(f'        best rival flips {flipped}, worst error {rivals[0][0]:.4f}')

    # The specific near-miss that shipped: exact on two poses, wrong on the third.
    near = [-1, 1, -1, 1, -1, 1, -1]
    drop = [-0.396162, 0.185206, -0.310712, 2.180323,
            0.141718, 0.536164, -0.663577]
    check('the joint1/3/5/7 rule is exact on ready_state',
          mirror_error(near, right_ready) < 1e-3, True)
    check('and wrong on drop_state, which is why it was caught',
          mirror_error(near, drop) > 0.05, True)
    print(f'        joint1/3/5/7 on drop_state: {mirror_error(near, drop):.4f}')

    # Every recorded right-arm pose must mirror to something the left arm can
    # actually hold, so check the real file if it is there.
    print('\n-- the recorded poses --')
    right_file = rs.default_file('right')
    recorded = rs.load_states(right_file)
    if not recorded:
        print(f'skip: {right_file} holds no states yet')
    else:
        limits = {}
        for joint in ET.fromstring(urdf).findall('joint'):
            limit = joint.find('limit')
            if limit is not None:
                limits[joint.get('name')] = (float(limit.get('lower')),
                                             float(limit.get('upper')))
        for name, entry in recorded.items():
            values = rs.mirror_joints(entry['joints'])
            worst = None
            for index, value in enumerate(values):
                joint = f'openarm_left_joint{index + 1}'
                low, high = limits[joint]
                margin = min(value - low, high - value)
                if worst is None or margin < worst[0]:
                    worst = (margin, joint)
            check(f"{name} mirrors inside the left arm's limits",
                  worst[0] > 0.0, True)
            print(f'        tightest margin {worst[0]:+.3f} rad on '
                  f'{worst[1].split("_")[-1]}')

            source = kin.pose('openarm_right_hand_tcp',
                              joint_values('right', entry['joints']))
            image = kin.pose('openarm_left_hand_tcp',
                             joint_values('left', values))
            want = np.array([source[0, 3], -source[1, 3], source[2, 3]])
            check(f'{name} mirrors to the mirrored tool position',
                  float(np.linalg.norm(image[:3, 3] - want)) < 0.001, True)

    check_play(rs)

    print()
    if failures:
        print(f'{len(failures)} check(s) failed: {", ".join(failures)}')
        return 1
    print('the mirror rule holds against the URDF, and --play sequences right')
    return 0


if __name__ == '__main__':
    sys.exit(main())
