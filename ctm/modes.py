"""Navigation and pick modes: the arms' named poses and switching between them.

The arms boot into navigation_state (boot_pre_pick.py). Picking works from
pre_pick_state: the first click (or PICK MODE) takes both arms there; the
NAVIGATION button takes them back. Poses come from pick_place_states_<arm>.yaml
(navigation_state, pre_pick_state, drop_state), the same files the boot walk
and record_states.py use.
"""

import os

import numpy as np
import yaml

from ctm.common import WS

POSE_FILE = os.path.join(WS, 'pick_place_states_{arm}.yaml')
MODE_POSE = {'navigation': 'navigation_state', 'pick': 'pre_pick_state'}
AT_POSE = 0.05                  # rad: an arm this close to a pose is "at" it


def load_poses():
    """{arm: {state name: [7 joints]}}"""
    out = {}
    for arm in ('left', 'right'):
        try:
            with open(POSE_FILE.format(arm=arm)) as handle:
                states = (yaml.safe_load(handle) or {}).get('states') or {}
        except (OSError, yaml.YAMLError):
            states = {}
        out[arm] = {name: [float(v) for v in s['joints']] for name, s in states.items()
                    if s and len(s.get('joints') or []) == 7}
    return out


class ModesMixin:
    """Part of click_to_move.App: self.mode is 'navigation', 'pick' or None
    (neither -- e.g. an arm left somewhere else)."""

    def pose(self, arm, name):
        return self.poses.get(arm, {}).get(name)

    def detect_mode(self):
        for mode, name in MODE_POSE.items():
            at = True
            for arm in ('left', 'right'):
                target, now = self.pose(arm, name), self.node.arm_positions(arm)
                if target is None or now is None or \
                        np.max(np.abs(np.array(now) - target)) > AT_POSE:
                    at = False
            if at:
                return mode
        return None

    def go_mode(self, mode):
        """Both arms to the mode's pose (right, then left). True when there."""
        node = self.node
        name = MODE_POSE[mode]
        self.say(f'{mode} mode: both arms to {name}...')
        node.require_move_group()
        ok = True
        for arm in ('right', 'left'):
            target, now = self.pose(arm, name), node.arm_positions(arm)
            if target is None:
                self.say(f'no {name} for the {arm} arm in {POSE_FILE.format(arm=arm)}', True)
                ok = False
                continue
            if now is not None and np.max(np.abs(np.array(now) - target)) <= AT_POSE:
                continue
            if not node.move_joints(arm, target, f'{arm} to {name}', via_home=False):
                self.say(f'{arm} arm: could not reach {name}', True)
                ok = False
        self.mode = mode if ok else self.detect_mode()
        if ok:
            self.say(f'{mode} mode' + (': click a point, then MOVE' if mode == 'pick' else ''))
        return ok

    def ensure_pick_mode(self):
        return self.mode == 'pick' or self.go_mode('pick')
