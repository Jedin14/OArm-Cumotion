"""A touch: the reach check after a click, the plan of the whole touch,
and the sequence itself (approach, correct out at the approach point, one straight
line in with guarded contact, hold, back out, retrace home); plus TEACH."""

import contextlib
import math
import time

import cv2
import numpy as np

import robot_camera_calibration as rcc
from ctm.kinematics import directions
from ctm.common import (ARM_REACH, BASE_FRAME, TOOL_ARM, NUDGE, decode_image, matrix_from_quat,
                        quat_from_matrix,
                        tool_frame_along)
from ctm import armcal
from ctm import box_match
from ctm import tip_vision
from ctm.corrections import learned_correction, save_correction

# The last NEAR m onto the surface is the only part where the gripper (its
# fingers, this arm only) may touch the octomap; everything before it is
# checked for every link.
NEAR = 0.03
# The re-aim and the corrections happen this far short of the surface; from
# there in it is one straight line (the user asked for 5 cm, 2026-10-13).
CORRECT_AT = 0.05
# The most the tip may still be off the target there before going in, m
# (--max-miss). 5 mm stopped good front touches 7 mm off, where the guarded
# contact finds the face anyway (2026-10-26); the case it guards against was
# 26 mm beside a box (2026-10-14).
MAX_LEFT = 0.010


class TouchMixin:
    """Part of click_to_move.App; uses self.node, self.frame, self.say..."""

    def reach_map(self):
        """Pixels of the captured frame whose surface is beyond the reach of
        the arm that would go there (left half: left arm), as (H, W) bool.
        Only a hint drawn on the image; the check after a click is the real one."""
        node, args = self.node, self.node.args
        bgr, depth, info, optical = self.frame
        t_wo = node.lookup(BASE_FRAME, optical)
        shoulders = {a: node.lookup(BASE_FRAME, f'openarm_{a}_link1') for a in ('left', 'right')}
        if t_wo is None or any(m is None for m in shoulders.values()):
            return None
        h, w = depth.shape
        vs, us = np.mgrid[0:h:4, 0:w:4]
        sx, sy = info.width / float(w), info.height / float(h)
        px = np.column_stack([us.ravel() * sx, vs.ravel() * sy]).reshape(-1, 1, 2)
        k, d = rcc.intrinsics(info)
        xy = cv2.undistortPoints(px.astype(np.float64), k, d).reshape(-1, 2)
        z = depth[vs, us].ravel().astype(np.float64)
        world = np.column_stack([xy * z[:, None], z]) @ t_wo[:3, :3].T + t_wo[:3, 3]
        left = (us.ravel() * sx < info.width / 2.0) if args.arm == 'auto' else \
            np.full(len(z), args.arm == 'left')
        shoulder = np.where(left[:, None], shoulders['left'][:3, 3], shoulders['right'][:3, 3])
        tip = np.where(left, args.tip_offsets['left'], args.tip_offsets['right'])
        beyond = np.linalg.norm(world - shoulder, axis=1) - tip - ARM_REACH
        bad = ((z > 0.1) & (beyond > 0)).reshape(vs.shape).astype(np.uint8)
        return cv2.resize(bad, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    def aim_for(self, arm, point):
        """(aim, learned, taught): where to send the fingertip -- the TEACH
        corrections for this arm, point and mode, taught on the current arm
        model."""
        learned, _used, taught = learned_correction(arm, point, self.mode_name(),
                                                    self.arm_models.tag(arm))
        return point + learned, learned, taught

    def mode_name(self):
        return 'top-down' if self.top_down else 'level'

    def check_pick(self, check, in_job=False):
        """Right after a click: can its arm touch that point -- reach, a
        front approach, a collision-free path and the straight line in, all
        planned, nothing moved. The plan is kept for MOVE if nothing changes."""
        node = self.node
        _u, _v, point, arm, origin = check['pick']
        try:
            with self.plan_lock:
                if self.check is not check or (self.busy and not in_job):
                    return
                node.require_move_group()
                node.ensure_grippers_in_octomap()
                aim = self.aim_for(arm, point)[0]
                start = node.arm_positions(arm)
                plan, why = self.preplan(arm, aim, point, origin, start)
        except Exception as exc:                     # noqa: BLE001 - only a preview
            plan, why, aim, start = None, f'could not check ({exc})', None, None
        check.update(ok=plan is not None, plan=plan, aim=aim, start=start, why=why,
                     time=time.monotonic())
        if self.check is not check or (self.busy and not in_job):
            return
        if plan is not None:
            self.say(f'{check["where"]}: the {arm} arm can touch it [{plan["label"]}]'
                     + (' but it is at the edge of reach' if plan['edge'] else '')
                     + ' - press MOVE')
        else:
            self.say(f'{check["where"]}: the {arm} arm cannot touch it -- {why}', True)

    def move(self):
        """The MOVE button: touch the selected point with its arm."""
        if self.pick is None:
            self.say('click a point first, then press MOVE', True)
            return
        _u, _v, point, arm, origin = self.pick
        scene = self.frame                           # the frame clicked on: no arm in it
        # Never from navigation: both arms to pre_pick first, then the touch.
        self.in_background(lambda: self.ensure_pick_mode() and
                           self.touch(arm, point, origin, scene))

    # -- the touch ------------------------------------------------------------

    def correct_to(self, arm, tcp_goal, axis, quat, label, expected, offset=None,
                   corrections=None, too_far=0.04, settle=None, gain=1.0):
        """Settle on `expected` joints, measure the fingertip, and fly out the
        error. Returns the offset that had to be added to the command (world,
        m) and the error left over (m).

        Plain corrections (gain 1), measured only once the arm has really
        stopped (settle()). Scaling corrections up to make up for apparent
        undershoot was tried and diverged: the "undershoot" was the arm still
        moving. If a correction makes things worse, the best offset so far is
        restored and the loop stops.
        """
        node, args = self.node, self.node.args
        wanted = tcp_goal + axis * args.tip_offsets[arm]
        offset = np.zeros(3) if offset is None else np.array(offset, dtype=float)
        best = None                                  # (residual, offset, joints)
        corrections = args.corrections if corrections is None else corrections
        for attempt in range(corrections + 1):
            t0 = time.monotonic()
            joint_err = node.settle(arm, expected, timeout=settle)
            waited = time.monotonic() - t0
            tip = node.fingertip(arm)
            if tip is None:
                break
            error = wanted - tip
            residual = float(np.linalg.norm(error))
            self.node.get_logger().info(
                f'{label}: settled in {waited:.1f} s, joints {1000 * (joint_err or 0):.1f} mrad '
                f'from command, fingertip {1000 * residual:.1f} mm from target')
            if best is not None and residual > best[0]:
                self.say(f'{arm} arm: correction made it worse ({1000 * best[0]:.1f} -> '
                         f'{1000 * residual:.1f} mm); going back')
                node.line_or_step(arm, tcp_goal + best[1], quat, f'{arm} back')
                node.settle(arm, best[2])
                return best[1], best[0]
            best = (residual, offset.copy(), expected)
            if residual <= args.tolerance or attempt == corrections:
                break
            if residual > too_far:
                self.say(f'{arm} arm: {1000 * residual:.0f} mm off at {label} -- too far '
                         f'for a correction; not correcting', True)
                break
            # the first step in full, later ones damped: the joints stop within
            # a few mm of where they are sent, and a full step on a 5 mm error
            # overshot as often as not ("4.8 -> 4.8 mm", 2026-10-26)
            offset += error * (1.0 if attempt == 0 else gain)
            self.say(f'{arm} arm: {1000 * residual:.1f} mm off at {label}; correcting')
            expected = node.line_or_step(arm, tcp_goal + offset, quat, f'{arm} correct')
            if expected is None:
                break
        return (best[1], best[0]) if best else (offset, None)

    def teach_touch(self, arm, point, learned, axis, quat, command, scene):
        """TEACH: hold the touch while the operator nudges the fingertip onto
        the spot, then save the total correction for this arm and point."""
        node = self.node
        info, optical = (scene[2], scene[3]) if scene else (None, None)
        t_wo = node.lookup(BASE_FRAME, optical) if optical else None
        right = t_wo[:3, 0] if t_wo is not None else np.array([0.0, -1.0, 0.0])
        right = right - right[2] * np.array([0.0, 0.0, 1.0])     # level: image right
        right /= max(np.linalg.norm(right), 1e-9)
        steps = {'left': -right, 'right': right, 'up': np.array([0.0, 0.0, 1.0]),
                 'down': np.array([0.0, 0.0, -1.0]), 'in': axis, 'out': -axis}
        nudge = np.zeros(3)
        self.teaching = True
        try:
            while True:
                self.say(f'TEACH: nudge onto the spot ({1000 * np.linalg.norm(nudge):.0f} mm '
                         f'so far), then SAVE')
                self.calib_event.clear()
                if not self.calib_event.wait(120.0):
                    self.say('TEACH: no answer for 2 minutes; not saved', True)
                    return
                what = self.calib_answer[0]
                times = 5 if what.endswith('5') and what[:-1] in steps else 1   # 'left5': 5 mm
                what = what[:-1] if times == 5 else what
                if what in steps:
                    nudge = nudge + steps[what] * NUDGE * times
                    node.line_move(arm, command + nudge, quat, f'{arm} nudge')
                elif what == 'save' and getattr(self, 'anchor_mode', False):
                    # ANCHOR (calibrate.py): the real tip is on the point now --
                    # ground truth for the arm model's fit
                    node.settle(arm, node.arm_positions(arm), timeout=1.5)
                    n = armcal.save_anchor(arm, node.arm_positions(arm), point,
                                           node.args.tip_offsets[arm])
                    self.say(f'ANCHOR saved for the {arm} arm ({n} so far; 3-4 at different '
                             f'spots, then REFIT)')
                    time.sleep(0.8)
                    return
                elif what == 'save':
                    total = learned + nudge
                    n = save_correction(arm, point, total, mode=self.mode_name(),
                                        model=self.arm_models.tag(arm))
                    self.say(f'TEACH: saved ({1000 * total[0]:+.0f}, {1000 * total[1]:+.0f}, '
                             f'{1000 * total[2]:+.0f}) mm for the {arm} arm here '
                             f'({n} corrections stored)')
                    time.sleep(0.8)
                    return
                elif what in ('skip', 'abort'):
                    self.say('TEACH: not saved')
                    return
        finally:
            self.teaching = False

    def _line_room(self, arm, approach, contact, way, quat, seed, near_d, room):
        """Does the straight line in (approach -> NEAR short) still solve, in
        full, with both ends moved `room` sideways -- each of 4 ways?"""
        node = self.node
        side = np.cross(way, [0.0, 0.0, 1.0])
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(way, [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        up = np.cross(side, way)
        for d in (side, -side, up, -up):
            q = node.solve_ik(arm, approach + room * d, quat, seed)
            if q is None:
                return False
            fraction, _t, _f = node.cartesian_plan(arm, contact + room * d - way * near_d, quat,
                                                   start_joints=q)
            if fraction < 0.98:
                return False
        return True

    def _room_to_correct(self, arm, tcp, axis, quat, seed, room=0.01):
        """Can the arm still reach `tcp` moved `room` sideways, every way?
        The corrections move the line in by the arm's flex (5-10 mm); at
        the edge of reach that can leave the corrected line unreachable."""
        side = np.cross(axis, [0.0, 0.0, 1.0])
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(axis, [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        up = np.cross(side, axis)
        return all(self.node.solve_ik(arm, tcp + room * d, quat, seed) is not None
                   for d in (side, -side, up, -up))

    def _standoffs(self):
        """Approach distances to try, far to near: --standoff (TOP DOWN:
        --top-standoff) down to --min-standoff in 1.5 cm steps."""
        args = self.node.args
        far = args.top_standoff if self.top_down else args.standoff
        near = min(args.min_standoff, far)
        out, d = [far], far
        while d - 0.015 > near + 1e-6:
            d = round(d - 0.015, 4)
            out.append(d)
        if out[-1] > near + 1e-6:
            out.append(near)
        return out

    def _solve_standoff(self, arm, aim, label, axis, quat, way, contact, standoff, start, tally):
        """Up to 3 arm postures that reach the approach point `standoff` short
        of the contact AND from which the straight line in (fully checked)
        and the contact line both solve. [] if none: counted in `tally`."""
        node, args = self.node, self.node.args
        approach = node.tcp_for_tip(aim - way * standoff, axis, arm)
        if self.top_down:
            # Can the arm point this way there at all? (fast, all postures;
            # KDL from a few seeds missed most of them -- 2026-10-14)
            seeds = node.fast_ik(arm, approach, quat, start)
        else:
            seeds = node.seeds(arm, start, approach, quat)
        near_d = min(NEAR, standoff)
        out, found = [], []
        for seed in seeds:
            above = node.solve_ik(arm, approach, quat, seed)
            if above is None or any(np.max(np.abs(np.subtract(above, f))) < 0.1 for f in found):
                continue
            # the posture NEAR short of the surface, fully checked: gripper
            # included (it may only touch the clicked surface)
            near_q = node.solve_ik(arm, contact - way * near_d, quat, above)
            if near_q is None:
                continue
            app_c, con_c, quat_c = approach, contact, quat
            if self.arm_models.has(arm):
                # CALIBRATE ARMS: the joints that, by the arm model, put the real
                # tool NEAR short of the surface -- and where the URDF puts those
                # joints, which is what the planner gets (sag out of the tilt
                # as well as the position). Re-taken at contact (_touch).
                cmd = self.arm_models.command(arm, contact - way * near_d,
                                              matrix_from_quat(*quat), near_q,
                                              node.joint_limits(arm), directions(above, near_q))
                if cmd is None:
                    continue
                delta = cmd[0] - (contact - way * near_d)
                quat_c = quat_from_matrix(cmd[1])
                app_c, con_c = approach + delta, contact + delta
                above = node.solve_ik(arm, app_c, quat_c, above)
                near_q = (node.solve_ik(arm, con_c - way * near_d, quat_c, above)
                          if above is not None else None)
                if above is None or near_q is None:
                    continue
            found.append(above)
            # The line in, fully checked -- every link, the gripper too -- to
            # NEAR short of the surface; then the contact, where only this
            # arm's tip may touch the map. It aims past the surface; at the
            # edge of reach 3 mm past is enough (it stops on contact anyway).
            fraction, _t, near_joints = node.cartesian_plan(arm, con_c - way * near_d, quat_c,
                                                            start_joints=above)
            if fraction < 0.98 or near_joints is None:
                continue
            with node.fingers_touching(arm):
                for beyond in (args.past_contact, min(args.past_contact, 0.003)):
                    fraction, _t, _f = node.cartesian_plan(arm, con_c + way * beyond, quat_c,
                                                           start_joints=near_joints)
                    if fraction >= 0.98:
                        break
            if fraction < 0.98:
                continue
            # Room for the corrections: the WHOLE line in must still solve from
            # an approach point moved --correction-room sideways, every way --
            # they moved it ~3 cm, and the checked line then failed part-way
            # at the edge of reach (2026-10-27). 1 cm only: an edge plan, used
            # if nothing better; not even that: not used at all.
            room = self._line_room(arm, app_c, con_c, way, quat_c, above, near_d,
                                   args.correction_room)
            room_min = room or self._line_room(arm, app_c, con_c, way, quat_c, above, near_d,
                                               0.01)
            if not room_min:
                continue
            # joint travel, plus a penalty for a joint near its limit (no room
            # there for a correction); a nearer approach point costs a little
            # (less room to correct out there)
            lim = min(node.limit_room(arm, above), node.limit_room(arm, near_q))
            cost = (sum(abs(x - y) for x, y in zip(above, start)) + 20.0 * max(0.0, 0.1 - lim)
                    + 2.0 * (args.standoff - standoff if not self.top_down
                             else args.top_standoff - standoff))
            if self.top_down and self.arm_models.pieces.get(arm) and \
                    self.strip_visible(arm, above) < 4:
                cost += 3.0         # the camera cannot check the tip at the hover point
            out.append({'cost': cost, 'label': label, 'axis': axis, 'quat': quat_c,
                        'quat0': quat, 'approach': app_c, 'contact': con_c, 'contact0': contact,
                        'above': above, 'raw': (contact, quat), 'way': way,
                        'standoff': standoff, 'beyond': beyond, 'room': room})
            if len(found) >= 3:
                break
        if not out:
            tally['line' if found else 'ik'] += 1
        return out

    def _finish_plan(self, arm, aim, origin, g, ring, tally, edge):
        """A posture that passed _solve_standoff: (TOP DOWN) the level way up
        there, then the planned path to it. The plan, or None (a plan with
        no room to correct goes to `edge`, used only if nothing better)."""
        node = self.node
        plan = {'label': g['label'] + f', from {100 * g["standoff"]:.1f} cm',
                'axis': g['axis'], 'way': g['way'], 'quat': g['quat'],
                'standoff': g['standoff'], 'approach': g['approach'],
                'contact': g['contact'], 'above': g['above'], 'beyond': g['beyond'],
                'edge': False, 'contact_raw': g['raw'], 'level': None}
        if self.top_down:
            # Fly there level and turn the wrist down only up there: flown in
            # already pointing down, the hanging hand hit the boxes (2026-10-14).
            plan['level'] = self._level_above(arm, aim, g['standoff'], origin, g['above'],
                                              g['approach'], g['quat'])
            if plan['level'] is None:
                tally['line'] += 1
                return None
        if not g['room'] and not (self.top_down and ring == 0):
            edge.append(dict(plan, edge=True))       # only if nothing better
            return None
        # (TOP DOWN: straight down at the edge of reach beats a lean -- the cup
        # seals best flat; if it cannot be corrected to MAX_LEFT, it does not go in)
        plan['edge'] = not g['room']
        _code, plan['trajectory'] = node.plan_joints(
            arm, plan['level']['joints'] if plan['level'] else g['above'])
        if plan['trajectory'] is not None:
            return plan
        tally['path'] += 1
        return None

    def preplan(self, arm, aim, point, origin, start):
        """Plan the whole touch before anything moves. (plan, reason)

        For each candidate approach (front rings, then rolls by joint travel):
        IK for the approach and contact poses, a cuMotion plan from where the
        arm is to the approach pose (around the octomap), and the straight
        line from the approach pose to just past the contact. The first
        candidate that passes all of them is returned, with its planned
        trajectory, so execution flies exactly what was checked. None, with
        the reason, if nothing passes -- and then nothing moves.
        """
        node, args = self.node, self.node.args
        shoulder = node.lookup(BASE_FRAME, f'openarm_{arm}_link1')
        if shoulder is not None:
            far = float(np.linalg.norm(aim - shoulder[:3, 3])) - args.tip_offsets[arm] - ARM_REACH
            if far > 0:
                return None, (f'out of reach: {100 * far:.0f} cm beyond what the {arm} arm '
                              f'can reach')
        tally = {'ik': 0, 'path': 0, 'line': 0}
        edge = []                    # plans with no room for corrections, if that is all
        rings = {}
        for cand in node.candidates(point, origin, top_down=self.top_down, arm=arm):
            rings.setdefault(cand[3], []).append(cand)
        standoffs = self._standoffs()
        for ring in sorted(rings):
            solved = []
            for label, axis, quat, _ring in rings[ring]:
                # The way the tip travels in: along the tool, or in TOP DOWN
                # straight down whatever the tool's lean -- coming in along a
                # leaned tool (20 deg: the wrist cannot point straight down
                # ~45 cm out) caught the box's near edge (2026-10-14).
                way = np.array([0.0, 0.0, -1.0]) if self.top_down else axis
                contact = node.tcp_for_tip(aim - way * args.touch_offset, axis, arm)
                # The approach point: as far out as works -- the arm gets there,
                # the whole straight line in AND the contact solve from there --
                # down to --min-standoff (room for the cup to seat). One fixed
                # distance was sometimes unreachable, and then the line in
                # failed part-way ("only 64 % of the straight line", 2026-10-16).
                for k, standoff in enumerate(standoffs):
                    got = self._solve_standoff(arm, aim, label, axis, quat, way, contact,
                                               standoff, start, tally)
                    if got:
                        for g in got:
                            g['nearer'] = standoffs[k + 1:]
                        solved.extend(got)
                        break
            solved.sort(key=lambda g: g['cost'])
            for g in solved:
                plan = self._finish_plan(arm, aim, origin, g, ring, tally, edge)
                if plan is not None:
                    return plan, None
                # the way to that approach point is blocked: one nearer try
                for standoff in g['nearer']:
                    more = self._solve_standoff(arm, aim, g['label'], g['axis'], g['quat0'],
                                                g['way'], g['contact0'], standoff, start, tally)
                    if more:
                        plan = self._finish_plan(arm, aim, origin, more[0], ring, tally, edge)
                        if plan is not None:
                            return plan, None
                        break
        for plan in edge:
            _code, plan['trajectory'] = node.plan_joints(arm, plan['above'])
            if plan['trajectory'] is not None:
                return plan, None
            tally['path'] += 1
        how = 'from above' if self.top_down else 'from the front'
        if self.top_down and args.tools[arm]['on'] and tally['path'] == 0:
            return None, (f'the wrist cannot point the tool down there, even leaned up to '
                          f'{min(args.vacuum_lean, 30):.0f} deg -- use the front approach')
        if tally['path'] == 0 and tally['line'] == 0:
            return None, f'out of reach {how} (no arm posture gets there)'
        if tally['line'] and not tally['path']:
            tried = self._standoffs()
            return None, (f'the straight line in {how} is blocked, or runs the wrist out of '
                          f'range part-way, from every approach point {100 * tried[0]:.0f} to '
                          f'{100 * tried[-1]:.0f} cm out' + (' -- try a point nearer the robot, or TOP DOWN '
                                               'off' if self.top_down else ''))
        return None, ('no collision-free path to the approach point (something is in '
                      'the way)' if tally['path'] else 'not reachable')

    def measure_tip_now(self, arm, way=None):
        """Where the camera sees this arm's tip now (ctm/tip_vision.py): the
        colour picture now against the one captured with the arms away (the
        tool's outline: exact across the view), and the depth readings on the
        tool for the other sideways direction. `way`: the line in (default
        the tool's axis). The result dict, kept in self.tip_seen for the
        picture; None without a captured frame. Every one is saved for replay."""
        node = self.node
        if self.frame is None:
            return None
        node.settle(arm, node.arm_positions(arm), timeout=1.0)
        depth, last = node.averaged_depth(6)
        msg = node.fresh_color(timeout=1.5)
        _c, _d, info = node.latest()
        if msg is None or info is None:
            return {'ok': False, 'why': 'no camera frames'}
        color = decode_image(msg)
        frame_id = (last.header.frame_id if last is not None else '') or info.header.frame_id
        t_wo = node.lookup(BASE_FRAME, frame_id)
        tcp = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
        if t_wo is None or tcp is None:
            return {'ok': False, 'why': 'no tf for the camera or the tool'}
        axis = tcp[:3, 2]
        way = axis if way is None else np.asarray(way, float)
        tip_pred = tcp[:3, 3] + axis * node.args.tip_offsets[arm]
        if self.arm_models.has(arm):
            tip_pred = tip_pred + self.arm_models.tip_shift(arm, node.arm_positions(arm),
                                                            node.args.tip_offsets[arm])
        bg_color = self.frame[0]
        bg_depth = self.frame[1] if depth is not None and self.frame[1].shape == depth.shape \
            else None
        result = tip_vision.measure_tip_rgb(color, bg_color, info, t_wo, tip_pred, axis, way,
                                            depth=depth, depth_bg=bg_depth)
        result['pred'] = tip_pred
        result['arm'] = arm
        result['when'] = time.monotonic()
        self.tip_seen = result
        enc = lambda img: np.frombuffer(cv2.imencode('.png', img)[1].tobytes(), np.uint8)
        tip_vision.save_debug(arm, color=enc(color), bg_color=enc(bg_color),
                              depth=None if depth is None else depth.astype(np.float32),
                              background=None if bg_depth is None else bg_depth.astype(np.float32),
                              k=np.array(info.k), d=np.array(info.d), t_wo=t_wo,
                              tip_pred=tip_pred, axis=axis, way=way,
                              tip=result.get('tip'), why=result.get('why', ''))
        return result

    def _tip_vision_fix(self, arm, want, way, command, quat, label):
        """The camera's check of the tip (measure_tip_now): the sideways error
        between where it sees the tip and the line it must travel in on
        (through `want`, along `way`), moved out -- up to 3 times. The total
        shift (zero if it was already on the line), or None if the camera
        could not see the tip (then the arm model, as before)."""
        node, args = self.node, self.node.args
        mode = getattr(args, 'tip_vision', 'log')
        if mode in (False, None, 'off') or self.frame is None:
            return None
        shift = np.zeros(3)
        self.vision_miss = None              # what is left after the last move, m
        if mode == 'log':
            # measure and keep (for tuning on the robot's own pictures), move nothing
            seen = self.measure_tip_now(arm, way)
            if seen is not None and seen.get('ok'):
                err = np.asarray(seen['tip']) - np.asarray(want)
                lat = err - (err @ way) * way
                self.say(f'{arm} arm: camera (log only) sees the tip {1000 * np.linalg.norm(lat):.1f}'
                         f' mm off the line at {label}: across {1000 * (lat @ seen["across"]):+.1f}'
                         f', other {1000 * (lat @ seen["other"]):+.1f} mm ({seen["other_from"]})')
            else:
                self.say(f'{arm} arm: camera (log only) at {label}: '
                         f'{(seen or {}).get("why", "no frame")}')
            return None
        for attempt in range(3):
            seen = self.measure_tip_now(arm, way)
            if seen is None or not seen.get('ok'):
                why = (seen or {}).get('why', 'no frame')
                if attempt == 0:
                    self.say(f'{arm} arm: camera check at {label}: {why} - using the arm model')
                    return None
                break
            err = np.asarray(seen['tip']) - np.asarray(want)
            lat = err - (err @ way) * way
            miss = float(np.linalg.norm(lat))
            self.vision_miss = miss
            if miss > args.tip_vision_max:
                self.say(f'{arm} arm: the camera puts the tip {1000 * miss:.0f} mm off the line '
                         f'at {label} - too far to trust; using the arm model', True)
                return None if attempt == 0 else shift
            self.say(f'{arm} arm: camera sees the tip {1000 * miss:.1f} mm off the line at '
                     f'{label}' + ('; moving it on' if miss > args.tolerance else ''))
            if miss <= args.tolerance:
                break
            shift = shift - lat
            if node.line_or_step(arm, command + shift, quat, f'{arm} onto the line',
                              speed=0.02) is None:
                shift = shift + lat
                break
            self.vision_miss = None              # moved: unknown until measured again
        if self.vision_miss is None:             # the last move was not measured after
            seen = self.measure_tip_now(arm)
            if seen is not None and seen.get('ok'):
                err = np.asarray(seen['tip']) - np.asarray(want)
                self.vision_miss = float(np.linalg.norm(err - (err @ way) * way))
        return shift

    def _top_vision_fix(self, arm, aim, way, standoff, approach, quat, offset):
        """TOP DOWN, at the hover point: the gripper pieces face up -- read
        them, and so where the real tip is, and move it over the point. No
        model there: measured, sag and play and all. (shift for the targets
        that follow, offset) -- or None if the pieces are not read (then the
        arm model, as before)."""
        node = self.node
        boards = self.arm_models.pieces.get(arm)
        if not boards or self.frame is None:
            return None
        t_wo = node.lookup(BASE_FRAME, self.frame[3])
        if t_wo is None:
            return None
        want = np.asarray(aim, float) - way * standoff        # the tip, over the point
        shift = np.zeros(3)
        for attempt in range(3):
            node.settle(arm, node.arm_positions(arm), timeout=1.0)
            frames = []
            for _ in range(6):
                msg = node.fresh_color(timeout=1.0)
                if msg is not None:
                    frames.append(decode_image(msg))
            real = self.tip_from_pieces(arm, frames, boards, t_wo, self.frame[2])
            if real is None:
                if attempt == 0:
                    self.say(f'{arm} arm: pieces not seen above the box; using the arm model')
                    return None
                break
            err = real - want
            if np.linalg.norm(err) > 0.05:
                self.say(f'{arm} arm: the camera puts the tip {1000 * np.linalg.norm(err):.0f} mm '
                         f'off - too far to trust; using the arm model', True)
                return None if attempt == 0 else (shift, offset)
            self.say(f'{arm} arm: camera sees the tip {1000 * np.linalg.norm(err):.1f} mm from '
                     f'over the point' + ('; moving it over' if np.linalg.norm(err) > 0.0015
                                          else ''))
            if np.linalg.norm(err) <= 0.0015:
                break
            shift = shift - err
            if node.line_move(arm, approach + shift + offset, quat, f'{arm} over the point',
                              speed=0.02) is None:
                shift = shift + err
                break
        return shift, offset

    def _level_above(self, arm, aim, standoff, origin, down_q, down_tcp, down_quat):
        """TOP DOWN: the pose to fly to -- the tip `standoff` above the point,
        the tool level (pointing away from the camera, like a front touch),
        from which the wrist turns down in place (in 10 deg steps, the path
        checked). {'joints', 'tcp', 'quat', 'via'} or None."""
        node = self.node
        ray = np.asarray(aim, float) - np.asarray(origin, float)
        ray[2] = 0.0
        ray /= max(np.linalg.norm(ray), 1e-9)
        tip = np.asarray(aim, float) + np.array([0.0, 0.0, standoff])
        tcp = node.tcp_for_tip(tip, ray, arm)
        for roll in (0, 90, -90, 180):
            quat = quat_from_matrix(tool_frame_along(ray, math.radians(roll)))
            q = node.solve_ik(arm, tcp, quat, down_q)
            if q is None:
                continue
            via = node.turn_steps(tcp, quat, down_tcp, down_quat)
            fraction, _t, _end = node.cartesian_plan(arm, down_tcp, down_quat, start_joints=q,
                                                     via=via)
            if fraction >= 0.98:
                return {'joints': q, 'tcp': tcp, 'quat': quat, 'via': via}
        return None

    def touch(self, arm, point, origin, scene=None):
        """Touch `point` with `arm`, then always try to bring the arm back --
        also when something failed part-way (an arm left out at the board,
        with every later move failing, was the "stuck" of 2026-10-07)."""
        want = TOOL_ARM.get(getattr(self, 'pick_tool', 'auto'))
        if want is not None and arm != want:         # the last guard: vacuum = left only
            self.say(f'{self.pick_tool} pick: only the {want} arm picks - the {arm} arm is '
                     f'not moved', True)
            self.last_contact = None
            return
        self.last_touch = None
        self.last_contact = None
        try:
            self._touch(arm, point, origin, scene)
        except Exception as exc:                     # noqa: BLE001 - shown in the window
            self.say(f'{arm} arm: touch failed ({exc}); bringing it back', True)
        finally:
            if self.last_touch is not None:
                self.return_to_start()

    def return_to_start(self):
        """Back out along the tool axis (if near the surface), let the joints
        settle, then retrace the approach path to the starting posture --
        planning a way back only if that is not possible. Also the RETURN
        button (b). True once the arm is back."""
        node, args = self.node, self.node.args
        lt = self.last_touch
        if lt is None:
            self.say('nothing to return from: no arm is out')
            return True
        arm = lt['arm']
        holding = bool(lt.get('held')) and arm in node.held
        if not lt['out']:
            self.phase(f'{arm} back_out')
            self.say(f'{arm} arm: backing straight out')
            if not lt.get('near_out') and 'out_short' in lt:
                with node.fingers_touching(arm):     # still at the surface
                    if node.line_move(arm, lt['out_short'], lt['quat'], f'{arm} off the surface',
                                      speed=args.fast_line_speed) is not None:
                        lt['near_out'] = True
            # a held box leaves the spot it came from straight along the tool,
            # where the octomap still has it: free of the map until it is out
            with (node.held_leaving(arm) if holding else contextlib.nullcontext()):
                self._back_out(arm, lt, holding)
            lt['out'] = True
        self._return_rest(arm, lt, holding)
        return self.last_touch is None

    def _back_out(self, arm, lt, holding):
        node, args = self.node, self.node.args
        if node.line_move(arm, lt['out_to'], lt['quat'], f'{arm} out',
                          speed=args.fast_line_speed) is None:
            # MoveIt may see this pose as colliding (TEACH nudges once put
            # the left forearm against the camera model): then every checked
            # move fails. The way out is the line it came in on, unchecked.
            self.say(f'{arm} arm: backing out along the way in (model says this pose '
                     f'touches something)')
            if node.line_move(arm, lt['out_to'], lt['quat'], f'{arm} out (unchecked)',
                              speed=args.fast_line_speed, check=False) is None:
                node.move_joints(arm, lt['above'], f'{arm} retreat', via_home=False)
        if holding:
            # all of the box out of its old spot: the tool backed off the
            # contact by at least the box's length along it, plus 3 cm
            along = node.held[arm]['extents'][2] + 0.03
            clear = lt['contact_tcp'] - lt['way'] * along
            here = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
            if here is None or (lt['contact_tcp'] - here[:3, 3]) @ lt['way'] < along - 0.005:
                self.phase(f'{arm} box_clear')
                self.say(f'{arm} arm: lifting the box clear ({100 * along:.0f} cm off the '
                         f'contact)')
                if node.line_move(arm, clear, lt['quat'], f'{arm} box clear',
                                  speed=args.fast_line_speed) is None:
                    self.say(f'{arm} arm: could not take the box further out along the '
                             f'tool; the planners may see it in its old spot', True)

    def _return_rest(self, arm, lt, holding):
        """From out at the approach: level again (TOP DOWN), back to pre_pick,
        the drop, pre_pick. Clears self.last_touch once back."""
        node, args = self.node, self.node.args
        node.settle(arm, lt['above'], timeout=1.5)   # plans must start from where it is
        lvl = lt.get('level')
        if lvl is not None and lt.get('turned'):
            # TOP DOWN: the tool level again, up there, before retracing the
            # (level) way it came
            self.phase(f'{arm} turn_level')
            self.say(f'{arm} arm: turning the tool level')
            if node.line_move(arm, lvl['tcp'], lvl['quat'], f'{arm} turn level', speed=0.02,
                              via=[(p, q) for p, q in reversed(lvl['via'])],
                              length=0.05) is None:
                node.move_joints(arm, lvl['joints'], f'{arm} turn level', via_home=False)
            lt['turned'] = False
            node.settle(arm, lvl['joints'], timeout=1.5)
        # Back to pre_pick the way it came (retrace; planned if that fails) ...
        # With a box held, only if that way is clear of everything WITH the
        # box: the way in was checked without it.
        self.phase(f'{arm} return')
        clear = node.path_is_clear(arm, lt['trajectory']) if holding else True
        if clear:
            self.say(f'{arm} arm: returning the way it came')
            back = node.retrace(arm, lt['trajectory'], f'{arm} return')
        else:
            self.say(f'{arm} arm: the way in is too tight with the box - planning a way back')
            back = False
        if not back:
            node.settle(arm, lt['start'], timeout=1.0)
            back = node.move_joints(arm, lt['start'], f'{arm} return', via_home=False)
        # ... then straight on to the drop pose, hold, and back to pre_pick.
        drop = self.pose(arm, 'drop_state') if args.drop_hold >= 0 else None
        if back and drop is not None and not lt.get('dropped'):
            lt['dropped'] = True                     # once per touch, also after a RETURN
            self.phase(f'{arm} to_drop')
            self.say(f'{arm} arm: to the drop pose')
            if node.move_joints(arm, drop, f'{arm} to drop', via_home=False):
                self.phase(f'{arm} drop')
                self.say(f'{arm} arm: at the drop pose; holding {args.drop_hold:g} s')
                time.sleep(args.drop_hold)
                if arm in node.held:
                    node.detach_held(arm)
                    self.say(f'{arm} arm: box dropped - out of the planners')
            else:
                self.say(f'{arm} arm: could not reach the drop pose; staying at pre_pick', True)
            self.phase(f'{arm} to_pre_pick')
            self.say(f'{arm} arm: back to pre_pick')
            back = node.move_joints(arm, lt['start'], f'{arm} return', via_home=False)
        if back and arm in node.held and drop is None:
            # no drop pose: the box goes where the system takes it from pre_pick
            node.detach_held(arm)
            self.say(f'{arm} arm: back at pre_pick with the box; released from the planners')
        if back:
            self.last_touch = None
            self.phase(f'{arm} done')
            return
        self.say(f'{arm} arm: could not get back to the start posture -- press RETURN (b) to '
                 f'try again', True)

    def _touch(self, arm, point, origin, scene=None):
        node, args = self.node, self.node.args
        start = node.arm_positions(arm)
        if start is None:
            self.say(f'no /joint_states for the {arm} arm yet', True)
            return
        if arm in node.held:
            self.say(f'{arm} arm: still holds the box from the last pick - release it first '
                     f'(Release, or /beast_arm/held_object/release)', True)
            return
        node.require_move_group()
        # Every time: a respawned move_group has forgotten it, and without it
        # the clicked surface counts as a collision and nothing solves.
        ok, message = node.ensure_grippers_in_octomap()
        if not ok:
            self.say(f'{message} -- not moving', True)
            return
        if args.close_gripper:
            self.say(f'{arm} arm: closing the gripper so the fingertips meet')
            node.close_gripper(arm)

        aim, learned, taught = self.aim_for(arm, point)

        t0 = time.monotonic()
        with self.plan_lock:                         # lets the click's own check finish
            check = self.check
            if (check is not None and check.get('plan') is not None
                    and check['pick'][2] is point and np.allclose(check['aim'], aim)
                    and time.monotonic() - check['time'] < 300.0
                    and np.max(np.abs(np.array(check['start']) - np.array(start))) < 0.003):
                plan, why = check['plan'], None          # checked on the click; nothing moved
            else:
                self.say(f'{arm} arm: checking the whole touch before moving...')
                plan, why = self.preplan(arm, aim, point, origin, start)
        if plan is None:
            self.say(f'{arm} arm: not moving -- {why}.', True)
            return
        axis, quat = plan['axis'], plan['quat']
        way = plan.get('way', axis)                  # the way the tip travels in
        standoff = plan.get('standoff', args.standoff)
        near_d, correct_d = min(NEAR, standoff), min(CORRECT_AT, standoff)
        approach, contact = plan['approach'], plan['contact']
        node.publish_target(contact, quat)
        self.say(f'{arm} arm: planned in {time.monotonic() - t0:.1f} s [{plan["label"]}]'
                 + (f'; taught correction from {taught} touches' if taught else '')
                 + ('; AT THE EDGE OF REACH - a correction may not fit' if plan['edge']
                    else ''))

        # From here on the arm is out: touch() brings it back whatever happens.
        self.last_touch = {'arm': arm, 'trajectory': plan['trajectory'], 'start': start,
                           'above': plan['above'], 'quat': quat, 'out_to': approach,
                           'out': True, 'level': plan.get('level')}
        self.phase(f'{arm} approach')
        ok, _ = node.execute(plan['trajectory'], f'{arm} approach')
        if not ok:
            self.say(f'{arm} arm: the approach did not complete', True)
            return
        lvl = plan.get('level')
        if lvl is not None:
            # TOP DOWN: arrived level, high above the box; turn the wrist down
            # here, clear of everything, then on as usual
            node.settle(arm, lvl['joints'], timeout=args.settle_time)
            self.phase(f'{arm} turn_down')
            self.say(f'{arm} arm: above the box; turning the tool down')
            self.last_touch['turned'] = True
            if node.line_move(arm, approach, quat, f'{arm} turn down', speed=0.02,
                              via=lvl['via'], length=0.05) is None:
                # The checked turn starts from where the arm was planned to
                # be; it arrived a little off and the path did not solve
                # (2026-10-14). Up here a planned joint move does it as well.
                if not node.move_joints(arm, plan['above'], f'{arm} turn down (joints)',
                                        via_home=False):
                    self.say(f'{arm} arm: could not turn the tool down here - coming back',
                             True)
                    return
        offset = np.zeros(3)
        # Corrections happen out at the approach point (--standoff, 10 cm),
        # away from the object, and again NEAR short; from there the way in
        # is ONE straight line along the tool axis, slow at the end until it
        # feels the surface.
        node.settle(arm, plan['above'], timeout=args.settle_time)
        # TOP DOWN: the arm sags much more reaching down (36-42 mm off at
        # the approach, 2026-10-09): two corrections, and larger ones.
        self.phase(f'{arm} correct')
        off_at_approach = None
        if standoff > correct_d + 0.005:
            # (from just above, TOP DOWN, the corrections below are these)
            offset, off_at_approach = self.correct_to(
                arm, approach, axis, quat, 'the approach', plan['above'], offset,
                corrections=max(args.corrections, 2) if self.top_down else None,
                too_far=0.06 if self.top_down else 0.04)

        # The camera's check (any approach): where it sees the tip against the
        # line in, moved onto it. Without a view of the tip, TOP DOWN reads the
        # gripper pieces if it has them; else the arm model, as before.
        seen_fix = None
        cam = self._tip_vision_fix(arm, aim - way * standoff, way, approach + offset, quat,
                                   'the approach')
        if cam is not None:
            seen_fix = (cam, offset)
        elif self.top_down:
            seen_fix = self._top_vision_fix(arm, aim, way, standoff, approach, quat, offset)
        if seen_fix:
            # the camera saw where the tip really is: shift what follows by it
            shift, offset = seen_fix
            approach, contact = approach + shift, contact + shift
            self.last_touch['out_to'] = approach + offset

        short = contact - way * near_d
        stop = contact - way * correct_d
        self.last_touch['out'] = False
        self.last_touch['out_short'] = stop + offset
        ends, touched = None, False
        # In to CORRECT_AT short of the surface, every link checked (the gripper too).
        self.phase(f'{arm} in')
        near = node.line_or_step(arm, stop + offset, quat, f'{arm} in',
                              speed=args.fast_line_speed)
        if near is not None and self.arm_models.has(arm) and not seen_fix:
            # The arm model's correction where the contact really happens: the
            # plan took it NEAR short, and gravity's sag changes over the last
            # cm (2-7 mm, simulated 2026-10-12). Re-aim there and slide the
            # line sideways before going in (the tilt stays as planned).
            raw, level = plan['contact_raw']
            q_c = node.solve_ik(arm, contact, quat, near)
            cmd = (self.arm_models.command(arm, raw, matrix_from_quat(*level), q_c,
                                           node.joint_limits(arm), directions(near, q_c))
                   if q_c is not None else None)
            con = cmd[0] if cmd is not None else contact
            if np.linalg.norm(con - contact) > 0.0005:
                moved = node.line_or_step(arm, con - way * correct_d + offset, quat,
                                       f'{arm} re-aim', speed=args.fast_line_speed)
                if moved is not None:
                    contact = con
                    short = contact - way * near_d
                    stop = contact - way * correct_d
                    self.last_touch['out_short'] = stop + offset
                    near = moved
        if near is not None:
            # Where the arm really got to, CORRECT_AT short: a joint-space controller
            # under gravity stops short of its command -- 27 mm at the approach
            # on the robot, and the one correction out there did not carry
            # through the line in and the re-aim (touches 7-8 mm off,
            # 2026-10-13). Fly it out here; the contact line keeps the offset.
            # Up to 4, until within --tolerance, each after a longer settle:
            # with 2 the left arm (rubber tool) still ended 10-17 mm off -- each
            # correction took out only about half (robot, 2026-10-13).
            offset, left = self.correct_to(
                arm, stop, axis, quat, f'{correct_d * 100:.0f} cm short', near, offset,
                corrections=6, too_far=0.05, settle=max(args.settle_time, 3.0), gain=0.6)
            self.last_touch['out_short'] = stop + offset
            # the camera once more, right before the straight line in: what it
            # sees replaces the joints' idea of how far off the tip still is
            cam = self._tip_vision_fix(arm, aim - way * correct_d, way, stop + offset, quat,
                                       f'{correct_d * 100:.0f} cm short')
            if cam is not None:
                offset = offset + cam
                self.last_touch['out_short'] = stop + offset
                if self.vision_miss is not None:
                    left = self.vision_miss          # measured, not inferred from the joints
            # Never go in from where the arm is not: a TOP DOWN pick at the edge
            # of reach arrived 26 mm off, its correction did not solve, and it
            # went down anyway -- beside the box, pushing the others aside
            # (2026-10-14). The paths were checked for where it was sent.
            if left is None or left > getattr(args, 'max_miss', MAX_LEFT):
                self.say(f'{arm} arm: still {1000 * (left or 0):.0f} mm off the target '
                         f'{correct_d * 100:.0f} cm short and it cannot be corrected (edge '
                         f'of reach?) - not going in; coming back', True)
                return
            # then straight on, no more sideways moves, to NEAR short
            near = node.line_or_step(arm, short + offset, quat, f'{arm} in to {near_d * 100:.0f} cm',
                                  speed=args.fast_line_speed)
            if near is not None:
                self.last_touch['out_short'] = short + offset
        if near is not None:
            # The last NEAR, on the same line: only now may this arm's fingers
            # touch the octomap -- slowly, stopping when it feels the surface,
            # so the depth comes from the object itself.
            self.phase(f'{arm} contact')
            with node.fingers_touching(arm):
                ends, touched = node.guarded_line(
                    arm, contact + way * plan['beyond'] + offset, quat, f'{arm} contact',
                    needed=near_d + 0.003)
                if ends is not None:
                    held = time.monotonic()
                    node.settle(arm, ends, timeout=0.1)
                    how = 'felt the surface' if touched else 'no contact felt'
                    # Stopped well short of the measured surface: not a touch
                    # (a false stop once left the cup 2.4 cm above the box,
                    # reported as "felt the surface", 2026-10-14)
                    tcp_now = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
                    if touched and tcp_now is not None:
                        short_by = float((contact + offset - tcp_now[:3, 3]) @ way)
                        if short_by > 0.015:
                            how = (f'STOPPED {100 * short_by:.1f} cm before the surface - '
                                   f'not touching')
                            self.say(f'{arm} arm: stopped {100 * short_by:.1f} cm before the '
                                     f'measured surface', True)
                    self.phase(f'{arm} holding')
                    self.say(f'{arm} arm: touching ({how}); holding {args.hold:.0f} s')
                    # the box is on the tool now: its size goes into the planners,
                    # so the way back keeps it clear of everything (2026-10-21)
                    size = node.object_size
                    if touched and 'STOPPED' not in how and size is not None:
                        here = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
                        # the box matched to the measured face: turned as it was seen
                        # (2026-10-28); else laid out by the tool's direction
                        exact, pb = None, getattr(node, 'pick_box', None)
                        if here is not None and pb is not None:
                            exact = box_match.box_on_tool(here[:3, :3], pb['long'], pb['box_mm'],
                                                          args.tip_offsets[arm])
                        if here is not None and node.attach_held(arm, size['hwd'],
                                                                 args.tip_offsets[arm], exact):
                            self.last_touch.update(held=True, contact_tcp=here[:3, 3].copy(),
                                                   way=np.array(way, float))
                            h, w, d = (1000 * v for v in size['hwd'])
                            self.say(f'{arm} arm: holding the box ({h:.0f} x {w:.0f} x {d:.0f} mm,'
                                     f' {size["source"]}) - the planners keep it clear until the '
                                     f'drop')
                    tip = node.fingertip(arm)
                    seen_by = 'joints'
                    if tip is not None and self.arm_models.has(arm):
                        # where the arm model says the tip really is
                        here_q = node.arm_positions(arm)
                        tip = tip + self.arm_models.tip_shift(arm, here_q, args.tip_offsets[arm],
                                                              directions(near, here_q))
                        seen_by = 'arm model'
                    if tip is not None:
                        self.last_contact = {'arm': arm, 'tip': tip, 'aim': aim,
                                             'axis': way, 'felt': touched, 'how': how}
                        miss = aim - tip
                        along = float(miss @ way)
                        sideways = float(np.linalg.norm(miss - along * way))
                        self.detail = (f'last touch: {how}; tip {1000 * sideways:.1f} mm from the click '
                                       f'sideways ({seen_by})'
                                       + (f', {1000 * off_at_approach:.1f} mm off at the approach'
                                          if off_at_approach is not None else '')
                                       + f'; {held - t0:.1f} s to touch')
                        node.get_logger().info(self.detail)
                    if self.teach:
                        here = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
                        self.teach_touch(arm, point, learned, axis, quat,
                                         here[:3, 3] if here is not None else contact + offset,
                                         scene)
                    else:
                        time.sleep(max(0.0, args.hold - (time.monotonic() - held)))
                self.phase(f'{arm} off_surface')
                # back off the surface the way it came, fingers still allowed
                if node.line_move(arm, short + offset, quat, f'{arm} off the surface',
                                  speed=args.fast_line_speed) is not None:
                    self.last_touch['near_out'] = True
        if ends is None:
            self.say(f'{arm} arm: no straight line in; could not touch', True)
        self.last_touch['out_to'] = approach + offset
        if self.return_to_start() and ends is not None:
            self.say(f'{arm} arm: done in {time.monotonic() - t0:.1f} s. Click the next point '
                     f'and press MOVE.')
