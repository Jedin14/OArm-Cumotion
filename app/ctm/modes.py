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

from ctm.common import ROBOT_DIR

POSE_FILE = os.path.join(ROBOT_DIR, 'pick_place_states_{arm}.yaml')
MODE_POSE = {'navigation': 'navigation_state', 'pick': 'pre_pick_state'}
AT_POSE = 0.05                  # rad: an arm this close to a pose is "at" it
# 2026-11-17: an arm last SENT to a pose (node.last_goal, ctm/robot.py) is at it
# while it is within GOAL_NEAR of it -- the soft wrist rests 50-70 mrad off its
# command (j7 0.063), and every PICK ALL round moved it to pre_pick again (3.2 s)
GOAL_SAME = 1e-3                # rad: the last goal IS the pose
GOAL_NEAR = 0.15                # rad: ... and the arm is this near it (more: moved by hand)


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

    def at_pose(self, arm, target, now):
        """Is the arm at the pose `target`: within AT_POSE of it, or (2026-11-17)
        last sent exactly there (node.last_goal[arm], when the node keeps it)
        and still within GOAL_NEAR."""
        if target is None or now is None:
            return False
        try:
            target = np.asarray(target, float)
            off = float(np.max(np.abs(np.asarray(now, float) - target)))
        except (TypeError, ValueError):
            return False
        if off <= AT_POSE:
            return True
        try:
            goal = (getattr(self.node, 'last_goal', None) or {}).get(arm)
            goal = None if goal is None else np.asarray(goal, float)
        except (AttributeError, TypeError, ValueError):
            goal = None
        return (goal is not None and goal.shape == target.shape and off <= GOAL_NEAR
                and float(np.max(np.abs(goal - target))) <= GOAL_SAME)

    def detect_mode(self):
        for mode, name in MODE_POSE.items():
            at = True
            for arm in ('left', 'right'):
                target, now = self.pose(arm, name), self.node.arm_positions(arm)
                if not self.at_pose(arm, target, now):
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
            self.say(f'{mode} mode' + (f': {self.next_hint}' if mode == 'pick' else ''))
        return ok

    def ensure_pick_mode(self):
        """Both arms to pre_pick (a pose move: both arms, whatever the pick
        tool -- only the pick itself is the tool's arm alone, ctm/touch.py)."""
        return self.mode == 'pick' or self.go_mode('pick')
