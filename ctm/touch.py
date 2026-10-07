"""A touch: the reach check after a click, the plan of the whole touch,
and the sequence itself (approach, correct 1 cm short, guarded contact,
hold, back out, retrace home); plus TEACH."""

import time

import cv2
import numpy as np

import robot_camera_calibration as rcc
from ctm.common import ARM_REACH, BASE_FRAME, NUDGE
from ctm.corrections import learned_correction, save_correction


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
        beyond = np.linalg.norm(world - shoulder, axis=1) - args.tip_offset - ARM_REACH
        bad = ((z > 0.1) & (beyond > 0)).reshape(vs.shape).astype(np.uint8)
        return cv2.resize(bad, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    def aim_for(self, arm, point, p_cam):
        """(aim, learned, taught, markers_on): where to send the fingertip.
        The TEACH corrections always apply -- they are the best first guess.
        With calibrated flags the camera then measures the fingertip against
        the click itself and corrects whatever is left; if no flag is in view
        the taught correction still stands (it used to be dropped as soon as
        any flag was calibrated, which left the right arm with no correction
        at all when its one flag was hidden, 2026-10-07)."""
        learned, _used, taught = learned_correction(arm, point)
        markers_on = self.markers is not None and self.markers.has_arm(arm) and p_cam is not None
        return point + learned, learned, taught, markers_on

    def check_pick(self, check):
        """Right after a click: can its arm touch that point -- reach, a
        front approach, a collision-free path and the straight line in, all
        planned, nothing moved. The plan is kept for MOVE if nothing changes."""
        node = self.node
        _u, _v, point, arm, origin = check['pick']
        try:
            with self.plan_lock:
                if self.check is not check or self.busy:
                    return
                node.require_move_group()
                node.ensure_grippers_in_octomap()
                scene = self.frame
                t_wo = node.lookup(BASE_FRAME, scene[3])
                p_cam = (np.linalg.inv(t_wo)[:3, :3] @ point + np.linalg.inv(t_wo)[:3, 3]
                         if t_wo is not None else None)
                aim = self.aim_for(arm, point, p_cam)[0]
                start = node.arm_positions(arm)
                plan, why = self.preplan(arm, aim, point, origin, start)
        except Exception as exc:                     # noqa: BLE001 - only a preview
            plan, why, aim, start = None, f'could not check ({exc})', None, None
        check.update(ok=plan is not None, plan=plan, aim=aim, start=start,
                     time=time.monotonic())
        if self.check is not check or self.busy:
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
        self.in_background(lambda: self.touch(arm, point, origin, scene))

    # -- the touch ------------------------------------------------------------

    def correct_to(self, arm, tcp_goal, axis, quat, label, expected, offset=None):
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
        wanted = tcp_goal + axis * args.tip_offset
        offset = np.zeros(3) if offset is None else np.array(offset, dtype=float)
        best = None                                  # (residual, offset, joints)
        for attempt in range(args.corrections + 1):
            t0 = time.monotonic()
            joint_err = node.settle(arm, expected)
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
            if residual <= args.tolerance or attempt == args.corrections:
                break
            if residual > 0.04:
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
                    n = save_correction(arm, point, total)
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
        The flags' corrections move the line in by the arm's flex (5-10 mm);
        at the edge of reach that can leave the corrected line unreachable."""
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
            far = float(np.linalg.norm(aim - shoulder[:3, 3])) - args.tip_offset - ARM_REACH
            if far > 0:
                return None, (f'out of reach: {100 * far:.0f} cm beyond what the {arm} arm '
                              f'can reach')
        tally = {'ik': 0, 'path': 0, 'line': 0}
        edge = []                    # plans with no room for corrections, if that is all
        rings = {}
        for cand in node.candidates(point, origin):
            rings.setdefault(cand[3], []).append(cand)
        for ring in sorted(rings):
            solved = []
            for label, axis, quat, _ring in rings[ring]:
                approach = node.tcp_for_tip(aim - axis * args.standoff, axis)
                contact = node.tcp_for_tip(aim - axis * args.touch_offset, axis)
                for seed in node.seeds(arm, start):
                    above = node.solve_ik(arm, approach, quat, seed)
                    if above is None:
                        continue
                    if node.solve_ik(arm, contact, quat, above) is not None:
                        cost = sum(abs(x - y) for x, y in zip(above, start))
                        solved.append((cost, label, axis, quat, approach, contact, above))
                        break
                else:
                    tally['ik'] += 1
            solved.sort(key=lambda s_: s_[0])
            if self.markers is not None and self.markers.has_arm(arm):
                solved = self._flags_in_view_first(arm, solved)
            for _cost, label, axis, quat, approach, contact, above in solved:
                # Cheap checks first (~50 ms line, ~100 ms room); a cuMotion
                # plan costs 0.5 s, a failed one ~5 s.
                # The contact move aims past the surface; at the edge of reach
                # 3 mm past is enough (it stops on contact anyway).
                for beyond in (args.past_contact, min(args.past_contact, 0.003)):
                    fraction, _t, _f = node.cartesian_plan(arm, contact + axis * beyond, quat,
                                                           start_joints=above)
                    if fraction >= 0.98:
                        break
                if fraction < 0.98:
                    tally['line'] += 1
                    continue
                plan = {'label': label, 'axis': axis, 'quat': quat, 'approach': approach,
                        'contact': contact, 'above': above, 'beyond': beyond, 'edge': False}
                if not self._room_to_correct(arm, contact + axis * beyond, axis, quat, above):
                    edge.append(dict(plan, edge=True))       # only if nothing better
                    continue
                _code, plan['trajectory'] = node.plan_joints(arm, above)
                if plan['trajectory'] is not None:
                    return plan, None
                tally['path'] += 1
        for plan in edge:
            _code, plan['trajectory'] = node.plan_joints(arm, plan['above'])
            if plan['trajectory'] is not None:
                return plan, None
            tally['path'] += 1
        if tally['path'] == 0 and tally['line'] == 0:
            return None, 'out of reach from the front (no arm posture gets there)'
        if tally['line'] and not tally['path']:
            return None, 'the straight line in is blocked or out of reach'
        return None, ('no collision-free path to the approach point (something is in '
                      'the way)' if tally['path'] else 'not reachable')

    def touch(self, arm, point, origin, scene=None):
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

        # The target in the camera's own frame, for the marker check: it and
        # the gripper flags are then compared without going through tf.
        p_cam = None
        if scene is not None:
            t_wo = node.lookup(BASE_FRAME, scene[3])
            if t_wo is not None:
                p_cam = np.linalg.inv(t_wo)[:3, :3] @ point + np.linalg.inv(t_wo)[:3, 3]
        aim, learned, taught, markers_on = self.aim_for(arm, point, p_cam)

        t0 = time.monotonic()
        with self.plan_lock:                         # lets the click's own check finish
            check = self.check
            if (check is not None and check.get('plan') is not None
                    and check['pick'][2] is point and np.allclose(check['aim'], aim)
                    and time.monotonic() - check['time'] < 60.0
                    and np.max(np.abs(np.array(check['start']) - np.array(start))) < 0.003):
                plan, why = check['plan'], None          # checked on the click; nothing moved
            else:
                self.say(f'{arm} arm: checking the whole touch before moving...')
                plan, why = self.preplan(arm, aim, point, origin, start)
        if plan is None:
            self.say(f'{arm} arm: not moving -- {why}.', True)
            return
        axis, quat = plan['axis'], plan['quat']
        approach, contact, above = plan['approach'], plan['contact'], plan['above']
        node.publish_target(contact, quat)
        self.say(f'{arm} arm: planned in {time.monotonic() - t0:.1f} s [{plan["label"]}]'
                 + (f'; taught correction from {taught} touches' if taught else '')
                 + ('; flags will correct it' if markers_on else '')
                 + ('; AT THE EDGE OF REACH - a correction may not fit' if plan['edge']
                    else ''))

        ok, _ = node.execute(plan['trajectory'], f'{arm} approach')
        if not ok:
            self.say(f'{arm} arm: the approach did not complete', True)
            return
        offset = np.zeros(3)
        vis = np.zeros(3)            # flag measurement: real fingertip = model + vis
        off_at_approach = None
        # Straight on through the 5 cm point, no stop: one correction 1 cm
        # short does what two stops did (5 cm and 1 cm), in a third of the
        # time -- the whole touch used to take 23-28 s, half of it waiting.
        short = contact - axis * min(0.01, args.standoff / 2)
        near = node.line_move(arm, short - vis + offset, quat, f'{arm} in',
                              speed=args.fast_line_speed)
        ends, touched = None, False
        if near is not None:
            d = None
            if markers_on:
                node.settle(arm, near, timeout=args.settle_time)
                for k in range(args.corrections):    # measure, correct (, measure again)
                    d = self.marker_offset(arm, p_cam, aim, axis, short, '1 cm short')
                    if d is None:
                        break
                    if k == 0:
                        off_at_approach = float(np.linalg.norm(d - (d @ axis) * axis))
                    if np.linalg.norm(d) <= 0.001:
                        break
                    vis = vis + d
                    moved = node.line_move(arm, short - vis + offset, quat, f'{arm} flag fix',
                                           speed=args.fast_line_speed)
                    if moved is None:
                        break
                    if k + 1 < args.corrections:
                        node.settle(arm, moved, timeout=args.settle_time)
            if d is None and not np.any(vis):
                # No flag in view: at least get the joints onto their targets.
                offset, off_at_approach = self.correct_to(arm, short, axis, quat,
                                                          '1 cm short', near, offset)
            # The last centimetre: slow, and stops when it feels the surface,
            # so the depth comes from the object itself, not from the camera.
            ends, touched = node.guarded_line(
                arm, contact + axis * plan['beyond'] - vis + offset, quat, f'{arm} contact')
        if ends is None:
            self.say(f'{arm} arm: no straight line in; could not touch', True)
        else:
            held = time.monotonic()
            node.settle(arm, ends, timeout=0.1)
            how = 'felt the surface' if touched else 'no contact felt'
            self.say(f'{arm} arm: touching ({how}); holding {args.hold:.0f} s')
            tip = node.fingertip(arm)
            seen_by = 'joints'
            if tip is not None:
                if markers_on:
                    # Where the camera sees the fingertip now, against the click.
                    here = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
                    d = (self.marker_offset(arm, p_cam, aim, axis, here[:3, 3], 'the touch')
                         if here is not None else None)
                    tip = tip + (d if d is not None else vis)
                    seen_by = 'camera' if d is not None else 'joints + flags'
                miss = aim - tip
                along = float(miss @ axis)
                sideways = float(np.linalg.norm(miss - along * axis))
                self.detail = (f'last touch: {how}; tip {1000 * sideways:.1f} mm from the click '
                               f'sideways ({seen_by})'
                               + (f', flags moved it {1000 * np.linalg.norm(vis):.1f} mm'
                                  if markers_on else '')
                               + (f', {1000 * off_at_approach:.1f} mm off at 1 cm short'
                                  if off_at_approach is not None else '')
                               + f'; {held - t0:.1f} s to touch')
                node.get_logger().info(self.detail)
            if self.teach:
                here = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
                self.teach_touch(arm, point, learned, axis, quat,
                                 here[:3, 3] if here is not None else contact - vis + offset,
                                 scene)
            else:
                time.sleep(max(0.0, args.hold - (time.monotonic() - held)))
        self.say(f'{arm} arm: backing straight out')
        if node.line_move(arm, approach - vis + offset, quat, f'{arm} out',
                          speed=args.fast_line_speed) is None:
            node.move_joints(arm, above, f'{arm} retreat', velocity=args.touch_velocity,
                             via_home=False)
        self.say(f'{arm} arm: returning the way it came')
        back = node.retrace(arm, plan['trajectory'], f'{arm} return')
        if not back:
            back = node.move_joints(arm, start, f'{arm} return', via_home=False)
        if ends is not None and back:
            self.say(f'{arm} arm: done in {time.monotonic() - t0:.1f} s. Click the next point '
                     f'and press MOVE.')
        elif not back:
            self.say(f'{arm} arm: could not return to the start posture', True)
