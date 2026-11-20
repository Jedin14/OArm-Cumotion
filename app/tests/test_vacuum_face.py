#!/usr/bin/env python3
"""The vacuum on the top row (2026-11-20): no way in more than --vacuum-face-max off square to
the face, and a top-row box that has none sent to --front-zone-high by the base correction.
The reach part runs the real numeric reach check (reach_fast.rank + room_check) on the plain
URDF. No ROS graph, no robot:

    source native/setup.bash; PYTHONNOUSERSITE=1 python3 app/tests/test_vacuum_face.py
"""
import math
import os
import sys
import types

import numpy as np
import yaml

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(APP)
sys.path.insert(0, APP)
import ctm.robot as robot          # noqa: E402
import ctm.reach_fast as rf        # noqa: E402
import ctm.tools as tools          # noqa: E402

ROBOT = next(c for c in vars(robot).values() if isinstance(c, type) and hasattr(c, 'candidates'))
CAM = np.array([0.03, 0.067, 0.58])
FAILS = []


def check(label, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + label, '' if ok else info)
    if not ok:
        FAILS.append(label)


def make_node(tool_on=True, cap=20.0):
    args = types.SimpleNamespace(standoff=0.10, min_standoff=0.04, touch_offset=0.0,
                                 limit_margin=0.10, free_roll='on', past_contact=0.02,
                                 face_first=True, vacuum_lean=30, vacuum_face_max=cap,
                                 correction_room=0.025)
    tl = tools.load()
    tl['left']['on'] = tool_on
    tools.apply(args, tl)
    args.tools = tl
    node = types.SimpleNamespace(args=args, pick_normal=np.array([-1.0, 0.0, 0.0]), retract=None)
    node.joint_limits = types.MethodType(ROBOT.joint_limits, node)
    node.lookup = lambda a, b: None
    node.candidates = types.MethodType(ROBOT.candidates, node)
    return node


def off_face(axis):
    return math.degrees(math.acos(float(np.clip(np.dot(axis, [1.0, 0.0, 0.0]), -1, 1))))


TOP_BOX = np.array([0.493, 0.130, 0.711])       # 11-20 12:56-12:59: four tries at 23 deg, none held

# -- the cap on the candidates ---------------------------------------------------------------
c_on = make_node(True).candidates(TOP_BOX, CAM, arm='left')
c_off = make_node(False).candidates(TOP_BOX, CAM, arm='left')
worst_on = max(off_face(c[1]) for c in c_on)
worst_off = max(off_face(c[1]) for c in c_off)
check(f'vacuum on: every way in within 20 deg of square (worst {worst_on:.1f})', worst_on <= 20.0 + 1e-6)
check(f'tool off (fingers): no cap, as before (worst {worst_off:.1f} deg)', worst_off > 30.0)
check('square on is still first', c_on[0][0].startswith('square on'), c_on[0][0])
td_on = make_node(True).candidates(TOP_BOX, CAM, top_down=True, arm='left')
td_off = make_node(True, cap=90).candidates(TOP_BOX, CAM, top_down=True, arm='left')
check('top-down picks unchanged (their cap is --vacuum-lean)', len(td_on) == len(td_off))
check('--vacuum-face-max 90: all of them, as before',
      len(make_node(True, cap=90).candidates(TOP_BOX, CAM, arm='left')) == len(c_off))

# -- the reach on the top row -----------------------------------------------------------------
P = yaml.safe_load(open(os.path.join(WS, 'robot', 'pick_place_states_left.yaml')))['states']
START = np.array(P['pre_pick_state']['joints'])


def best_square(node, pt):
    """The most square way in with joint margin and the full correction room, deg (None)."""
    app = types.SimpleNamespace(node=node, pose=lambda arm, n: P[n]['joints'])
    entries, _info = rf.rank(app, 'left', pt, pt, CAM, START)
    cand = sorted([e for e in entries if e['margin_ok']], key=lambda e: e['oblique'])
    lim = np.array(node.joint_limits('left'), float)
    for i in range(0, min(len(cand), 80), 8):
        chunk = cand[i:i + 8]
        rf.room_check(app, 'left', chunk, lim)
        ok = [e for e in chunk if e.get('room') is True]
        if ok:
            return min(e['oblique'] for e in ok)
    return None


b_now = best_square(make_node(True, cap=90), TOP_BOX)
b_cap = best_square(make_node(True, cap=20), TOP_BOX)
check(f'the 12:56 top-row box (49 cm ahead): only oblique ways in ({b_now:.0f} deg) without the cap',
      b_now is not None and b_now > 20.0, b_now)
check('... none with the cap: it goes to a base move instead of a slipping cup', b_cap is None, b_cap)
back = TOP_BOX + np.array([0.08, 0.0, 0.0])
b_back = best_square(make_node(True, cap=20), back)
check(f'the same box 57 cm ahead (--front-zone-high): square to its face ({b_back} deg)',
      b_back is not None and b_back <= 5.0, b_back)

# -- the base correction for the top row ------------------------------------------------------
import web_gui                     # noqa: E402
args = web_gui.make_parser().parse_args([]) if hasattr(web_gui, 'make_parser') else None
if args is None:
    import argparse
    args = argparse.Namespace(top_zone=[0.38, 0.47, -0.02, 0.06], front_zone=[0.44, 0.50, 0.06, 0.20],
                              front_zone_high=[0.55, 0.60, 0.05, 0.30], high_row_z=0.65)
GUI = next(c for c in vars(web_gui).values() if isinstance(c, type) and hasattr(c, 'base_correction_for'))
me = types.SimpleNamespace(node=types.SimpleNamespace(args=args), ZONE_MARGIN_M=GUI.ZONE_MARGIN_M,
                           MOVE_MIN_M=GUI.MOVE_MIN_M)
front = web_gui.PickTarget.APPROACH_FRONT if hasattr(web_gui.PickTarget, 'APPROACH_FRONT') else 0
corr = GUI.base_correction_for(me, TOP_BOX, front, 'test')
check('a top-row box 49 cm ahead: the base moves BACK into 55-60 cm',
      [m['dir'] for m in corr['moves']] == ['back'] and 0.06 <= corr['moves'][0]['m'] <= 0.08, corr['moves'])
low = np.array([0.493, 0.130, 0.441])
corr = GUI.base_correction_for(me, low, front, 'test')
check('the same spot one row down: already in --front-zone (no move)', corr['inside'] and not corr['moves'],
      corr)

print('ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}')
sys.exit(1 if FAILS else 0)
