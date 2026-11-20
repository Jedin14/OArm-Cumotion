#!/usr/bin/env python3
"""Where can the left arm (vacuum) come in square to a FRONT face? The real numeric reach check
(reach_fast.rank + room_check, plain URDF, the tool from tools.yaml) on a grid of box positions
at one height: the most square way in with joint margin and the full correction room, deg off
the face ("-": none within 30). Used to set --front-zone / --front-zone-high (2026-11-20):

    source native/setup.bash; PYTHONNOUSERSITE=1 python3 app/tests/reach_map.py 0.71
    (z 0.44: square 40-50 cm ahead; 0.58: 40-60; 0.71: only 55-60 -- on 11-20)
"""
import os
import sys
import types

import numpy as np
import yaml

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
import ctm.robot as robot          # noqa: E402
import ctm.reach_fast as rf        # noqa: E402
import ctm.tools as tools          # noqa: E402

ROBOT = next(c for c in vars(robot).values() if isinstance(c, type) and hasattr(c, 'candidates'))
CAM = np.array([0.03, 0.067, 0.58])
z = float(sys.argv[1]) if len(sys.argv) > 1 else 0.71
args = types.SimpleNamespace(standoff=0.10, min_standoff=0.04, touch_offset=0.0, limit_margin=0.10,
                             free_roll='on', past_contact=0.02, face_first=True, vacuum_lean=30,
                             vacuum_face_max=90.0, correction_room=0.025)
tools.apply(args, tools.load())
args.tools = tools.load()
node = types.SimpleNamespace(args=args, pick_normal=np.array([-1.0, 0.0, 0.0]), retract=None)
node.joint_limits = types.MethodType(ROBOT.joint_limits, node)
node.lookup = lambda a, b: None
node.candidates = types.MethodType(ROBOT.candidates, node)
P = yaml.safe_load(open(os.path.join(os.path.dirname(APP), 'robot',
                                     'pick_place_states_left.yaml')))['states']
app = types.SimpleNamespace(node=node, pose=lambda arm, n: P[n]['joints'])
start = np.array(P['pre_pick_state']['joints'])
lim = np.array(node.joint_limits('left'), float)
xs = [0.40, 0.45, 0.50, 0.55, 0.60]
ys = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
print(f'z {z}: deg off the face of the most square way in with room ("-": none within 30)')
print('   x\\y ' + ''.join(f'{y:7.2f}' for y in ys))
for x in xs:
    row = []
    for y in ys:
        pt = np.array([x, y, z])
        entries, _info = rf.rank(app, 'left', pt, pt, CAM, start)
        cand = sorted([e for e in entries if e['margin_ok']], key=lambda e: e['oblique'])
        best = None
        for i in range(0, min(len(cand), 60), 8):
            chunk = cand[i:i + 8]
            rf.room_check(app, 'left', chunk, lim)
            ok = [e for e in chunk if e.get('room') is True]
            if ok:
                best = min(e['oblique'] for e in ok)
                break
        row.append('     -' if best is None or best > 30 else f'{best:6.0f}')
    print(f'  {x:.2f} ' + ' '.join(row), flush=True)
