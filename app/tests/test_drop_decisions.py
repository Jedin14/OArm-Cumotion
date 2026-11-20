#!/usr/bin/env python3
"""The way back after a touch (app/ctm/touch.py _return_rest): the drop or not, letting go at
pre_pick, the camera's check of the box's spot, the automatic tries home. Fakes only -- no ROS
graph, no robot. Run from anywhere:

    source native/setup.bash; PYTHONNOUSERSITE=1 python3 app/tests/test_drop_decisions.py
"""
import os
import sys
import time
import types

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
import ctm.touch as touch          # noqa: E402
import ctm.robot as robot          # noqa: E402

touch.time.sleep = lambda s: None
HOME, DROP = [0.1] * 7, [0.9] * 7
ROBOT = next(c for c in vars(robot).values() if isinstance(c, type) and hasattr(c, 'vertical_x'))


class FakeNode:
    def __init__(self, x=None, age=0.0, rng=(87.0, 152.0), held=True, contacts=None,
                 home_fails=0, drop_plan=True, after=None):
        self.args = types.SimpleNamespace(drop_hold=1.0, skip_empty_drop='on', spot_check='on',
                                          no_drop_x=list(rng) if rng else None,
                                          velocity=1.0, drop_velocity=0.7)
        self.held = {'left': object()} if held else {}
        self.log = []
        self.vertical = None if x is None else (x, time.monotonic() - age)
        self.halt = types.SimpleNamespace(is_set=lambda: False)
        self.contacts = contacts
        self.home_fails = home_fails
        self.drop_plan = drop_plan
        self.after = after                       # the depth image once back (spot check)

    vertical_x = ROBOT.vertical_x

    def state_contacts(self, arm, joints):
        return self.contacts

    def averaged_depth(self, n):
        return self.after, None

    def settle(self, *a, **k):
        pass

    def path_is_clear(self, *a):
        return True

    def retrace(self, arm, traj, label, scale=1.0):
        self.log.append(('retrace', label))
        if self.home_fails > 0:
            self.home_fails -= 1
            return False
        return True

    def move_joints(self, arm, goal, label, **k):
        self.log.append(('move', label))
        if self.home_fails > 0:
            self.home_fails -= 1
            return False
        return True

    def held_leaving(self, arm):
        import contextlib
        return contextlib.nullcontext()

    def detach_held(self, arm):
        self.log.append(('detach', arm))
        self.held.pop(arm, None)

    def get_logger(self):
        return types.SimpleNamespace(warn=lambda m: None, info=lambda m: None)


class T(touch.TouchMixin):
    def __init__(self, node, weighed=None):
        self.node = node
        self.last_touch = 'x'
        self.last_contact = {'held': weighed}

    def _carry(self, lt, holding):
        return 0.02, 0.5

    def on_returning(self, *a):
        pass

    def return_target(self, arm, lt):
        return HOME

    def pose(self, arm, name):
        return DROP if name == 'drop_state' else HOME

    def _at_goal(self, arm, pose):
        return True

    def _leg(self, arm, key, goal, label, velocity=None, start=None, start_name=None,
             fallback=True):
        if key == 'to_drop' and not self.node.drop_plan:
            self.node.log.append(('leg_failed', key))
            return False
        self.node.log.append(('leg', key, 'DROP' if goal is DROP else 'HOME'))
        return True

    def phase(self, p):
        pass

    def say(self, text, loud=False):
        self.node.log.append(('say', text))

    def vacuum(self, arm, on):
        at = 'at_drop' if any(e[:2] == ('leg', 'to_drop') for e in self.node.log) else 'at_pre_pick'
        self.node.log.append(('vacuum', on, at))

    def on_dropped(self, arm):
        self.node.log.append(('dropped',))

    def _commanded(self, arm):
        return HOME

    def _pose_name(self, arm, j):
        return 'pre_pick_state'


REF_DEPTH = np.full((480, 640), 0.555, np.float32)
REF_MASK = np.zeros((480, 640), bool)
REF_MASK[200:250, 300:350] = True
SPOT = {'mask': REF_MASK, 'depth': REF_DEPTH, 'n': int(REF_MASK.sum())}


def scene(change):
    """The depth once back: 'same', 'gone' (the box behind 6 cm further), 'front'."""
    d = REF_DEPTH.copy()
    if change == 'gone':
        d[REF_MASK] = 0.615
    elif change == 'front':
        d[REF_MASK] = 0.40
    return d


def run(lt_extra=None, weighed=None, top=False, **node_kw):
    n = FakeNode(**node_kw)
    t = T(n, weighed)
    if top:
        t._at_goal = lambda arm, pose: False
    lt = {'above': HOME, 'start': HOME, 'trajectory': [], 'held': bool(n.held)}
    lt.update(lt_extra or {})
    t._return_rest('left', lt, bool(n.held) or lt.get('pressed', False))
    went_drop = any(e[:2] == ('leg', 'to_drop') for e in n.log)
    vac = [e for e in n.log if e[0] == 'vacuum']
    says = [e[1] for e in n.log if e[0] == 'say']
    return went_drop, vac, says, n.log


FAILS = []


def check(label, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + label, '' if ok else info)
    if not ok:
        FAILS.append(label)


# -- x (the range sensor) and the drop pose ------------------------------------------------
for x, want_drop in ((50.0, True), (86.9, True), (87.0, False), (100.0, False), (152.0, False),
                     (152.1, True)):
    d, vac, says, log = run(x=x, contacts=[])
    where = 'at_drop' if want_drop else 'at_pre_pick'
    check(f'x {x}: {"the drop pose" if want_drop else "let go at pre_pick"}',
          d == want_drop and vac == [('vacuum', False, where)], log)
d, vac, says, log = run(x=100.0, age=5.0, contacts=[])
check('x 100 but 5 s old: the drop pose', d and vac == [('vacuum', False, 'at_drop')], log)
d, vac, says, log = run(x=100.0, rng=(0.0, 0.0), contacts=[])
check('--no-drop-x 0 0: the drop pose even at x 100', d, log)
d, vac, says, log = run(x=84.5, contacts=[('height_box', 'openarm_left_hand', (0, 0, 0))])
check('x 84.5, the drop pose in the height box: let go at pre_pick, says why',
      not d and vac == [('vacuum', False, 'at_pre_pick')] and any('height_box' in s for s in says),
      log)
d, vac, says, log = run(x=84.5, contacts=None)
check('MoveIt cannot say: the drop pose as usual', d, log)
d, vac, says, log = run(x=110.0, top=True, contacts=[])
order = [e for e in log if e[0] in ('leg', 'vacuum')]
check('top pick at x 110: to pre_pick first, then let go there',
      order[:2] == [('leg', 'to_rest', 'HOME'), ('vacuum', False, 'at_pre_pick')], order)

# -- the camera on the box's spot (2026-11-20) -----------------------------------------------
# the 17:36 case: the cup weighed nothing, the box was on it -> the spot is empty: to the drop
d, vac, says, log = run({'held': False, 'pressed': True, 'spot': SPOT}, weighed=False,
                        held=False, contacts=[], after=scene('gone'))
check('weighed nothing, but the box left its spot: TO THE DROP, vacuum off there',
      d and vac == [('vacuum', False, 'at_drop')] and any('left its spot' in s for s in says), log)
d, vac, says, log = run({'held': False, 'pressed': True, 'spot': SPOT}, weighed=False,
                        held=False, contacts=[], after=scene('same'))
check('weighed nothing, the box still in its spot: no drop trip',
      not d and not vac and any('still in its spot' in s for s in says), log)
d, vac, says, log = run({'held': True, 'pressed': True, 'spot': SPOT}, weighed=None,
                        held=True, contacts=[], after=scene('same'))
check('... the planners holding a box though it is still there: released from them, no drop',
      not d and ('detach', 'left') in log, log)
d, vac, says, log = run({'held': True, 'pressed': True, 'spot': SPOT}, weighed=True,
                        held=True, contacts=[], after=scene('same'))
check('weighed a box, the camera says still there (they disagree): the drop, to be safe', d, log)
d, vac, says, log = run({'held': False, 'pressed': True, 'spot': SPOT}, weighed=False,
                        held=False, contacts=[], after=scene('front'))
check('something in front of the spot (cannot tell): the drop, as before', d, log)
d, vac, says, log = run({'held': False, 'pressed': True, 'spot': None}, weighed=False,
                        held=False, contacts=[], after=None)
check('no spot taken before (no camera): pressed -> the drop, as before', d, log)
d, vac, says, log = run({'held': False, 'pressed': False}, weighed=None, held=False, contacts=[])
check('never pressed, nothing held: straight back, no drop (as before)', not d, log)

# -- never left hanging / waiting ------------------------------------------------------------
d, vac, says, log = run(contacts=[], drop_plan=False)
check('the drop leg does not plan: back at pre_pick and LET GO there (not left on the cup)',
      vac == [('vacuum', False, 'at_pre_pick')] and ('dropped',) in log
      and any('letting go at pre_pick' in s for s in says), log)
d, vac, says, log = run(contacts=[], home_fails=4)
check('the way home fails all 4 usual ways: tries again by itself, gets home, then the drop',
      any('trying again by itself (1/2)' in s for s in says) and d, log)
d, vac, says, log = run(contacts=[], home_fails=99)
check('the way home never works: 2 more tries, then it says so (no endless loop)',
      sum('trying again by itself' in s for s in says) == 2 and not d
      and any('could not get back' in s for s in says), log)

# -- the spot itself, on synthetic frames ----------------------------------------------------
info = types.SimpleNamespace(header=types.SimpleNamespace(frame_id='cam'), width=640, height=480,
                             k=[387.4, 0, 322.3, 0, 386.8, 245.7, 0, 0, 1], d=[0.0] * 5,
                             distortion_model='plumb_bob')
t_wo = np.eye(4)                                     # the camera at the origin, looking +z
frames = {'d': REF_DEPTH.copy()}
probe = types.SimpleNamespace(latest=lambda: (None, None, info), lookup=lambda b, f: t_wo,
                              averaged_depth=lambda n: (frames['d'], None), pick_face_mm=125.0)
tt = T(FakeNode())
tt.node = probe
ref = tt._spot_ref([0.0, 0.0, 0.555])
check('the spot: pixels on the face round the point', ref is not None and ref['n'] > 1500,
      None if ref is None else ref['n'])
for change, want in (('same', False), ('gone', True), ('front', None)):
    d1 = REF_DEPTH.copy()
    d1[ref['mask']] = {'same': 0.556, 'gone': 0.62, 'front': 0.40}[change]
    frames['d'] = d1
    got, text = tt._spot_check(ref)
    check(f'the spot check, box {change}: {want}', got is want, text)
frames['d'] = np.zeros_like(REF_DEPTH)
got, text = tt._spot_check(ref)
check('the spot with no reading at all: counted as emptied (to the drop: the safe side)',
      got is True, text)
probe.lookup = lambda b, f: None
check('no camera pose: no spot (the check is skipped)', tt._spot_ref([0.0, 0.0, 0.555]) is None)

print('ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}')
sys.exit(1 if FAILS else 0)
