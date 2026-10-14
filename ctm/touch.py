"""A touch: the reach check after a click, the plan of the whole touch,
and the sequence itself (approach, correct out at the approach point, one straight
line in with guarded contact, hold, back out, retrace home); plus TEACH."""

import math
import time

import cv2
import numpy as np

import robot_camera_calibration as rcc
from ctm.kinematics import directions
from ctm.common import (ARM_REACH, BASE_FRAME, NUDGE, matrix_from_quat, quat_from_matrix,
                        tool_frame_along)
from ctm.corrections import learned_correction, save_correction

# The last NEAR m onto the surface is the only part where the gripper (its
# fingers, this arm only) may touch the octomap; everything before it is
# checked for every link.
NEAR = 0.03
# The re-aim and the corrections happen this far short of the surface; from
# there in it is one straight line (the user asked for 5 cm, 2026-10-13).
CORRECT_AT = 0.05
# The most the tip may still be off the target there before going in, m.
MAX_LEFT = 0.005


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
        check.update(ok=plan is not None, plan=plan, aim=aim, start=start,
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
                   corrections=None, too_far=0.04, settle=None):
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
                node.line_move(arm, tcp_goal + best[1], quat, f'{arm} back')
                node.settle(arm, best[2])
                return best[1], best[0]
            best = (residual, offset.copy(), expected)
            if residual <= args.tolerance or attempt == corrections:
                break
            if residual > too_far:
                self.say(f'{arm} arm: {1000 * residual:.0f} mm off at {label} -- too far '
                         f'for a correction; not correcting', True)
                break
            offset += error
            self.say(f'{arm} arm: {1000 * residual:.1f} mm off at {label}; correcting')
            expected = node.line_move(arm, tcp_goal + offset, quat, f'{arm} correct')
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
                if what in steps:
                    nudge = nudge + steps[what] * NUDGE
                    node.line_move(arm, command + nudge, quat, f'{arm} nudge')
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
        for ring in sorted(rings):
            solved = []
            for label, axis, quat, _ring in rings[ring]:
                # The way the tip travels in: along the tool, or in TOP DOWN
                # straight down whatever the tool's lean -- coming in along a
                # leaned tool (20 deg: the wrist cannot point straight down
                # ~45 cm out) caught the box's near edge (2026-10-14).
                way = np.array([0.0, 0.0, -1.0]) if self.top_down else axis
                # TOP DOWN: from just above the top, not 10 cm up -- a 10 cm
                # vertical line asks the wrist to point straight down over a
                # longer stretch than the arm has out there (2026-10-14)
                contact = node.tcp_for_tip(aim - way * args.touch_offset, axis, arm)
                # TOP DOWN: as high above as the arm can point down there --
                # 12 cm, else 8, else 5: high up, pointing down puts the upper
                # arm into the unseen space above the picture (2026-10-14)
                standoff, above0 = args.standoff, None
                for standoff in ((args.top_standoff, 0.08, 0.05) if self.top_down
                                 else (args.standoff,)):
                    approach = node.tcp_for_tip(aim - way * standoff, axis, arm)
                    if self.top_down:
                        # Can the arm point this way there at all? (fast, all
                        # postures; KDL from a few seeds missed most of them,
                        # and each failure cost its whole timeout -- 2026-10-14)
                        seeds = node.fast_ik(arm, approach, quat, start)
                    else:
                        seeds = node.seeds(arm, start, approach, quat)
                    above0 = next((q for q in (node.solve_ik(arm, approach, quat, s_)
                                               for s_ in seeds) if q is not None), None)
                    if above0 is not None:
                        break
                near_d = min(NEAR, standoff)
                # Up to 3 different arm postures for this approach: from one,
                # the straight line can run into a wrist limit half-way (then
                # it is refused: joint jump), from another it goes through.
                found = []
                for seed in ([above0] if above0 is not None else []) + seeds:
                    above = node.solve_ik(arm, approach, quat, seed)
                    if above is None or any(np.max(np.abs(np.subtract(above, f))) < 0.1
                                            for f in found):
                        continue
                    # the posture NEAR cm short of the surface, fully checked:
                    # gripper included (it may only touch the clicked surface)
                    near_q = node.solve_ik(arm, contact - way * near_d, quat, above)
                    if near_q is None:
                        continue
                    app_c, con_c, quat_c = approach, contact, quat
                    if self.arm_models.has(arm):
                        # CALIBRATE ARMS: the joints that, by the arm model, put
                        # the real tool NEAR short of the surface, level -- and
                        # where the URDF puts those joints, which is what the
                        # planner gets. So the sag is taken out of the tilt as
                        # well as the position (the tool used to dip). Taken at
                        # the posture NEAR short; re-taken at contact (_touch).
                        # (gear play: each joint's last move there is the line in)
                        cmd = self.arm_models.command(arm, contact - way * near_d,
                                                      matrix_from_quat(*quat), near_q,
                                                      node.joint_limits(arm),
                                                      directions(above, near_q))
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
                    # joint travel, plus a penalty for a joint near its limit:
                    # no room left there for the line in or a correction
                    room = min(node.limit_room(arm, above), node.limit_room(arm, near_q))
                    cost = (sum(abs(x - y) for x, y in zip(above, start))
                            + 20.0 * max(0.0, 0.1 - room))
                    solved.append((cost, label, axis, quat_c, app_c, con_c, above, (contact, quat),
                                   way, standoff))
                    if len(found) >= 3:
                        break
                if not found:
                    tally['ik'] += 1
            solved.sort(key=lambda s_: s_[0])
            for _cost, label, axis, quat, approach, contact, above, raw, way, standoff in solved:
                near_d = min(NEAR, standoff)
                # Cheap checks first (~50 ms line, ~100 ms room); a cuMotion
                # plan costs 0.5 s, a failed one ~5 s.
                # The line in, fully checked -- every link, the gripper too --
                # to NEAR short of the surface; only the last NEAR (the
                # contact) lets this arm's fingers touch the map. It aims past
                # the surface; at the edge of reach 3 mm past is enough (it
                # stops on contact anyway).
                fraction, _t, near_joints = node.cartesian_plan(
                    arm, contact - way * near_d, quat, start_joints=above)
                if fraction < 0.98 or near_joints is None:
                    tally['line'] += 1
                    continue
                with node.fingers_touching(arm):
                    for beyond in (args.past_contact, min(args.past_contact, 0.003)):
                        fraction, _t, _f = node.cartesian_plan(
                            arm, contact + way * beyond, quat, start_joints=near_joints)
                        if fraction >= 0.98:
                            break
                    room = (fraction >= 0.98 and self._room_to_correct(
                        arm, contact + way * beyond, axis, quat, above))
                if fraction < 0.98:
                    tally['line'] += 1
                    continue
                plan = {'label': label, 'axis': axis, 'way': way, 'quat': quat,
                        'standoff': standoff,
                        'approach': approach,
                        'contact': contact, 'above': above, 'beyond': beyond, 'edge': False,
                        'contact_raw': raw, 'level': None}
                if self.top_down:
                    # Fly there level and turn the wrist down only up there:
                    # flown in already pointing down, the hanging hand hit the
                    # boxes on the way (2026-10-14).
                    plan['level'] = self._level_above(arm, aim, standoff, origin, above,
                                                      approach, quat)
                    if plan['level'] is None:
                        tally['line'] += 1
                        continue
                if not room and not (self.top_down and ring == 0):
                    edge.append(dict(plan, edge=True))       # only if nothing better
                    continue
                # (TOP DOWN: straight down at the edge of reach beats a lean --
                # the cup seals best flat, and if the arm cannot be corrected to
                # within MAX_LEFT there, it does not go in)
                plan['edge'] = not room
                _code, plan['trajectory'] = node.plan_joints(
                    arm, plan['level']['joints'] if plan['level'] else above)
                if plan['trajectory'] is not None:
                    return plan, None
                tally['path'] += 1
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
            return None, (f'the straight line in {how} is blocked, or runs the wrist out of '
                          f'range part-way' + (' -- try a point nearer the robot, or TOP DOWN '
                                               'off' if self.top_down else ''))
        return None, ('no collision-free path to the approach point (something is in '
                      'the way)' if tally['path'] else 'not reachable')

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
        if not lt['out']:
            self.say(f'{arm} arm: backing straight out')
            if not lt.get('near_out') and 'out_short' in lt:
                with node.fingers_touching(arm):     # still at the surface
                    if node.line_move(arm, lt['out_short'], lt['quat'], f'{arm} off the surface',
                                      speed=args.fast_line_speed) is not None:
                        lt['near_out'] = True
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
            lt['out'] = True
        node.settle(arm, lt['above'], timeout=1.5)   # plans must start from where it is
        lvl = lt.get('level')
        if lvl is not None and lt.get('turned'):
            # TOP DOWN: the tool level again, up there, before retracing the
            # (level) way it came
            self.say(f'{arm} arm: turning the tool level')
            if node.line_move(arm, lvl['tcp'], lvl['quat'], f'{arm} turn level', speed=0.02,
                              via=[(p, q) for p, q in reversed(lvl['via'])],
                              length=0.05) is None:
                node.move_joints(arm, lvl['joints'], f'{arm} turn level', via_home=False)
            lt['turned'] = False
            node.settle(arm, lvl['joints'], timeout=1.5)
        # Back to pre_pick the way it came (retrace; planned if that fails) ...
        self.say(f'{arm} arm: returning the way it came')
        back = node.retrace(arm, lt['trajectory'], f'{arm} return')
        if not back:
            node.settle(arm, lt['start'], timeout=1.0)
            back = node.move_joints(arm, lt['start'], f'{arm} return', via_home=False)
        # ... then straight on to the drop pose, hold, and back to pre_pick.
        drop = self.pose(arm, 'drop_state') if args.drop_hold >= 0 else None
        if back and drop is not None and not lt.get('dropped'):
            lt['dropped'] = True                     # once per touch, also after a RETURN
            self.say(f'{arm} arm: to the drop pose')
            if node.move_joints(arm, drop, f'{arm} to drop', via_home=False):
                self.say(f'{arm} arm: at the drop pose; holding {args.drop_hold:g} s')
                time.sleep(args.drop_hold)
            else:
                self.say(f'{arm} arm: could not reach the drop pose; staying at pre_pick', True)
            self.say(f'{arm} arm: back to pre_pick')
            back = node.move_joints(arm, lt['start'], f'{arm} return', via_home=False)
        if back:
            self.last_touch = None
            return True
        self.say(f'{arm} arm: could not get back to the start posture -- press RETURN (b) to '
                 f'try again', True)
        return False

    def _touch(self, arm, point, origin, scene=None):
        node, args = self.node, self.node.args
        start = node.arm_positions(arm)
        if start is None:
            self.say(f'no /joint_states for the {arm} arm yet', True)
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
        ok, _ = node.execute(plan['trajectory'], f'{arm} approach')
        if not ok:
            self.say(f'{arm} arm: the approach did not complete', True)
            return
        lvl = plan.get('level')
        if lvl is not None:
            # TOP DOWN: arrived level, high above the box; turn the wrist down
            # here, clear of everything, then on as usual
            node.settle(arm, lvl['joints'], timeout=args.settle_time)
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
        off_at_approach = None
        if standoff > correct_d + 0.005:
            # (from just above, TOP DOWN, the corrections below are these)
            offset, off_at_approach = self.correct_to(
                arm, approach, axis, quat, 'the approach', plan['above'], offset,
                corrections=max(args.corrections, 2) if self.top_down else None,
                too_far=0.06 if self.top_down else 0.04)

        short = contact - way * near_d
        stop = contact - way * correct_d
        self.last_touch['out'] = False
        self.last_touch['out_short'] = stop + offset
        ends, touched = None, False
        # In to CORRECT_AT short of the surface, every link checked (the gripper too).
        near = node.line_move(arm, stop + offset, quat, f'{arm} in',
                              speed=args.fast_line_speed)
        if near is not None and self.arm_models.has(arm):
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
                moved = node.line_move(arm, con - way * correct_d + offset, quat,
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
                corrections=4, too_far=0.05, settle=max(args.settle_time, 2.0))
            self.last_touch['out_short'] = stop + offset
            # Never go in from where the arm is not: a TOP DOWN pick at the edge
            # of reach arrived 26 mm off, its correction did not solve, and it
            # went down anyway -- beside the box, pushing the others aside
            # (2026-10-14). The paths were checked for where it was sent.
            if left is None or left > MAX_LEFT:
                self.say(f'{arm} arm: still {1000 * (left or 0):.0f} mm off the target '
                         f'{correct_d * 100:.0f} cm short and it cannot be corrected (edge '
                         f'of reach?) - not going in; coming back', True)
                return
            # then straight on, no more sideways moves, to NEAR short
            near = node.line_move(arm, short + offset, quat, f'{arm} in to {near_d * 100:.0f} cm',
                                  speed=args.fast_line_speed)
            if near is not None:
                self.last_touch['out_short'] = short + offset
        if near is not None:
            # The last NEAR, on the same line: only now may this arm's fingers
            # touch the octomap -- slowly, stopping when it feels the surface,
            # so the depth comes from the object itself.
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
                    self.say(f'{arm} arm: touching ({how}); holding {args.hold:.0f} s')
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
                                             'axis': way, 'felt': touched}
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
