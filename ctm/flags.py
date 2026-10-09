"""Gripper flags (ArUco): whether they will be in view, measuring the
fingertip from them, and the FLAGS calibration (roll sweeps)."""

import datetime
import math

import cv2
import numpy as np
from moveit_msgs.msg import MoveItErrorCodes

import gripper_markers as gm
import robot_camera_calibration as rcc
from ctm.corrections import replace_auto
from ctm.common import (BASE_FRAME, TCP_IN_HAND, decode_image, pixel_ray, quat_from_matrix,
                        tool_frame_along)


class FlagsMixin:
    """Part of click_to_move.App; uses self.node, self.frame, self.say..."""

    # -- gripper flags ----------------------------------------------------------

    def flag_patches(self, arm):
        """{id or name: 4x4 hand<-marker} -- calibrated flags, else the two
        nominal rail-end positions."""
        if self.markers is not None and self.markers.has_arm(arm):
            return {i: self.markers.hand_marker(i) for i in self.markers.ids(arm)}
        out = {}
        for name, y in (('end+', 0.109), ('end-', -0.109)):
            m = np.eye(4)
            m[:3, :3] = np.diag([1.0, -1.0, -1.0])           # face looking back (-Z)
            m[:3, 3] = (0.0, y, 0.004)
            out[name] = m
        return out

    def flag_visible(self, arm, joints):
        """How many of this arm's gripper flags will be in the camera's view
        with the arm at `joints`? A z-buffer of both arms' meshes (this one by
        FK at `joints`, the other where it is), against each flag's patch.
        (Unknown -> 1.)"""
        node = self.node
        if self.frame is None:
            return 1
        info, optical = self.frame[2], self.frame[3]
        t_wo = node.lookup(BASE_FRAME, optical)
        links = [f'openarm_{arm}_{s_}' for s_ in rcc.RobotSurface.LINKS]
        poses = node.fk(arm, joints, links)
        if t_wo is None or poses is None:
            return 1
        other = 'left' if arm == 'right' else 'right'
        poses.update(node.link_poses((other,)))
        pts, _n = node.robot_surface().posed(poses)
        t_ow = np.linalg.inv(t_wo)
        pc = pts @ t_ow[:3, :3].T + t_ow[:3, 3]
        pc = pc[(pc[:, 2] > 0.05) & (np.abs(pc[:, 0]) < pc[:, 2]) & (np.abs(pc[:, 1]) < pc[:, 2])]
        zb = np.full((480, 640), np.inf, np.float32)
        if len(pc):                      # cv2 refuses an empty set (arm out of view)
            px, z = rcc.project(pc, info)
            u, v = px[:, 0].astype(int), px[:, 1].astype(int)
            ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
            np.minimum.at(zb, (v[ok], u[ok]), z[ok].astype(np.float32))
        zb = cv2.erode(zb, np.ones((5, 5), np.uint8))
        hand = poses[f'openarm_{arm}_hand']
        g = np.stack(np.meshgrid(np.linspace(-0.02, 0.02, 5), np.linspace(-0.02, 0.02, 5)),
                     -1).reshape(-1, 2)
        local = np.column_stack([g, np.zeros(len(g)), np.ones(len(g))])
        count = 0
        for mk in self.flag_patches(arm).values():
            w = (hand @ mk @ local.T).T[:, :3]
            cam = w @ t_ow[:3, :3].T + t_ow[:3, 3]
            if cam[:, 2].min() < 0.05:
                continue                 # beside or behind the camera
            normal = t_ow[:3, :3] @ (hand[:3, :3] @ mk[:3, 2])
            facing = -float(normal @ (cam.mean(0) / np.linalg.norm(cam.mean(0))))
            pp, zz = rcc.project(cam, info)
            uu, vv = pp[:, 0].astype(int), pp[:, 1].astype(int)
            inside = (uu >= 0) & (uu < 640) & (vv >= 0) & (vv < 480)
            seen = np.zeros(len(cam), bool)
            seen[inside] = zz[inside] <= zb[vv[inside], uu[inside]] + 0.004
            if facing > 0.4 and seen.mean() > 0.85:
                count += 1
        return count

    def marker_offset(self, arm, p_cam, aim, axis, tcp_goal, label):
        """Where the real fingertip is, from the gripper flags, against where
        it should be at this stage. Returns d (world, m): the real fingertip
        is d away from where it should be -- or None if no flag is seen.

        Everything is compared in the camera's own frame: the target (from
        the click) and the fingertip (from the flags) are measured by the
        same camera, so its calibration, and arm flex the encoders cannot
        see, cancel out.
        """
        node = self.node
        _c, _d, info = node.latest()
        optical = self.frame[3] if self.frame is not None else 'camera_color_optical_frame'
        t_wo = node.lookup(BASE_FRAME, optical)
        hand = node.lookup(BASE_FRAME, f'openarm_{arm}_hand')
        if info is None or t_wo is None or hand is None:
            return None
        frames = []
        for _ in range(node.args.marker_frames):
            msg = node.fresh_color(timeout=1.0)
            if msg is None:
                break
            frames.append(decode_image(msg))
        r_ow = t_wo[:3, :3].T
        ids = self.markers.ids(arm)
        expected = {i: r_ow @ hand[:3, :3] @ self.markers.hand_marker(i)[:3, 2] for i in ids}
        seen = gm.observe(frames, info, ids, expected)
        if not seen:
            node.get_logger().info(f'flags at {label}: none of {ids} in view')
            return None
        tip_in_hand = np.array([0.0, 0.0, TCP_IN_HAND + node.args.tip_offset])
        hand_cam, used = gm.hand_from_flags(
            seen, {i: self.markers.hand_marker(i) for i in ids}, r_ow @ hand[:3, :3])
        tip_cam = hand_cam[:3, :3] @ tip_in_hand + hand_cam[:3, 3]
        # each flag on its own, for the agreement check
        tips = [m_cam[:3, 3] + hand_cam[:3, :3] @ (tip_in_hand - self.markers.hand_marker(i)[:3, 3])
                for i, (m_cam, _n, _s) in seen.items()]
        should = tcp_goal + axis * node.args.tip_offset          # world, model
        desired_cam = p_cam + r_ow @ (should - aim)
        d = t_wo[:3, :3] @ (tip_cam - desired_cam)
        node.get_logger().info(
            f'flags at {label}: {sorted(seen)} seen; fingertip is ({1000 * d[0]:+.1f}, '
            f'{1000 * d[1]:+.1f}, {1000 * d[2]:+.1f}) mm from where it should be'
            + (f' (flags disagree by {1000 * np.ptp(np.array(tips), axis=0).max():.1f} mm)'
               if len(tips) > 1 else ''))
        if np.linalg.norm(d) > 0.025:
            node.get_logger().warn(f'flags at {label}: {1000 * np.linalg.norm(d):.0f} mm is '
                                   f'implausible; ignored')
            return None
        return d

    def marker_targets(self, t_wo):
        """Fingertip spots for CALIBRATE MARKERS: a grid over each half of
        the image, 0.40 / 0.46 m out -- far enough that the whole gripper is
        beyond the D455's minimum range -- skipping any spot with something
        less than 6 cm behind it in the captured depth (the gripper would be
        lost against it). Interleaved so each arm's spots spread out."""
        bgr, depth, info, _frame = self.frame
        h, w = depth.shape
        out = {'left': [], 'right': []}
        for k, fv in enumerate((0.30, 0.50, 0.70)):
            for j, fu in enumerate((0.12, 0.28, 0.42, 0.58, 0.72, 0.88)):
                for wanted in ((0.40, 0.46) if (j + k) % 2 else (0.46, 0.40)):
                    u, v = fu * bgr.shape[1], fv * bgr.shape[0]
                    du, dv = int(u * w / bgr.shape[1]), int(v * h / bgr.shape[0])
                    patch = depth[max(0, dv - 8):dv + 9, max(0, du - 8):du + 9]
                    seen = patch[patch > 0.1]
                    if seen.size and np.percentile(seen, 10) < wanted + 0.06:
                        continue
                    p = t_wo[:3, :3] @ (pixel_ray(u, v, info) * wanted) + t_wo[:3, 3]
                    out['left' if fu < 0.5 else 'right'].append(('left' if fu < 0.5 else 'right', p))
        return out['right'] + out['left']

    def calibrate_markers(self):
        """CALIBRATE MARKERS (FLAGS): measure where each flag sits on its gripper.

        At a few spots 0.40-0.46 m out, the hand turns about its own axis
        (fingertip held still, -60..+60 deg in 30 deg steps) and the camera
        follows the flags round their circles (gripper_markers.flags_from_roll).
        No depth fitting: a fit of the gripper's shape to depth was tried and
        placed the flags 1-7 mm wrong in simulation (2026-10-07); the circles
        place them to ~0.5 mm. About 20 s per spot.
        """
        node, args = self.node, self.node.args
        self.say('flag calibration: taking a clean frame first...')
        self.capture()
        if self.frame is None:
            return
        _bgr, _depth, info, optical = self.frame
        t_wo = node.lookup(BASE_FRAME, optical)
        if t_wo is None:
            self.say(f'no tf for {optical}', True)
            return
        t_ow = np.linalg.inv(t_wo)
        cam = t_wo[:3, 3]
        targets = self.marker_targets(t_wo)
        if not targets:
            self.say('flag calibration needs clear space 40-50 cm in front of the camera on '
                     'both sides -- move things back and try again', True)
            return
        arms = sorted({a for a, _p in targets})
        starts = {a: node.arm_positions(a) for a in arms}
        if any(v is None for v in starts.values()):
            self.say('no /joint_states yet', True)
            return
        node.require_move_group()
        node.ensure_grippers_in_octomap()
        samples, per_arm = {}, {}
        current = None
        self.abort.clear()
        self.calibrating = True
        try:
            for k, (arm, tip) in enumerate(targets, start=1):
                if self.abort.is_set():
                    raise KeyboardInterrupt
                if per_arm.get(arm, 0) >= args.marker_poses_per_arm:
                    continue
                node.require_move_group()
                if current and current != arm:
                    node.move_joints(current, starts[current], f'{current} return')
                current = arm
                axis = (tip - cam) / np.linalg.norm(tip - cam)
                tcp = node.tcp_for_tip(tip, axis)
                # a base roll from which the whole sweep solves, both flags in view
                sweep, best = None, -1
                for roll0 in (0, 90, -90, 180):
                    seed, sols = node.arm_positions(arm), []
                    for phi in (0, -30, -60, 30, 60):
                        quat = quat_from_matrix(tool_frame_along(axis, math.radians(roll0 + phi)))
                        sol = node.solve_ik(arm, tcp, quat, seed if phi else node.arm_positions(arm))
                        if sol is None:
                            break
                        sols.append((phi, sol))
                        seed = sols[0][1]
                    if len(sols) < 5:
                        continue
                    n = self.flag_visible(arm, sols[0][1])
                    if n > best:
                        sweep, best = sorted(sols), n
                    if best >= 2:
                        break
                if sweep is None:
                    self.say(f'flags {k}: the {arm} arm cannot turn there; skipping')
                    continue
                rows = []
                for phi, joints in sweep:
                    if self.abort.is_set():
                        raise KeyboardInterrupt
                    self.say(f'flags {k}: {arm} hand turned {phi:+d} deg')
                    if node._move_joints_once(arm, joints, f'flags {k} {phi:+d}') != \
                            MoveItErrorCodes.SUCCESS:
                        continue
                    node.settle(arm, joints, timeout=3.0)
                    frames = []
                    for _ in range(args.marker_frames):
                        msg = node.fresh_color(timeout=1.0)
                        if msg is None:
                            break
                        frames.append(decode_image(msg))
                    hand = node.lookup(BASE_FRAME, f'openarm_{arm}_hand')
                    if hand is None or not frames:
                        continue
                    hand_cam = t_ow @ hand
                    expected = {i: hand_cam[:3, :3] @ gm.NOMINAL_NORMAL for i in gm.IDS[arm]}
                    rows.append((hand_cam, gm.observe(frames, info, gm.IDS[arm], expected)))
                    if frames:
                        self.calib_view = (frames[-1], None, None, None)
                found, rms = gm.flags_from_roll(rows)
                for i, mats in found.items():
                    samples.setdefault(i, []).extend(mats)
                if found:
                    per_arm[arm] = per_arm.get(arm, 0) + 1
                self.say(f'flags {k}: ' + (', '.join(f'ID {i} circle fit {r:.1f} mm'
                                                     for i, r in sorted(rms.items()))
                                           or 'no flag followed round its circle'))
        except KeyboardInterrupt:
            self.say('flag calibration aborted')
            return
        finally:
            self.calib_view = None
            self.calibrating = False
            if current and node.alive():
                node.move_joints(current, starts[current], f'{current} return')

        result, report = {}, []
        for i, mats in samples.items():
            if len(mats) < 4:
                report.append(f'ID {i}: only {len(mats)} views, not saved')
                continue
            mats = np.array(mats)
            t = np.median(mats[:, :3, 3], axis=0)
            dt = np.linalg.norm(mats[:, :3, 3] - t, axis=1)
            keep = dt < max(0.003, 3.0 * np.median(dt))
            mats = mats[keep] if keep.sum() >= 4 else mats
            m = np.eye(4)
            m[:3, :3] = gm.average_rotation(mats[:, :3, :3])
            m[:3, 3] = np.mean(mats[:, :3, 3], axis=0)
            lateral = np.linalg.norm(mats[:, :2, 3] - m[:2, 3], axis=1)
            spread = float(np.median(lateral) * 1000)
            if spread > 3.0:
                report.append(f'ID {i}: inconsistent by {spread:.0f} mm, NOT saved (loose, '
                              f'or seen too rarely)')
                continue
            result[i] = {'arm': gm.ARM_OF.get(i, '?'), 'hand_marker': m.tolist(),
                         'views': int(len(mats)), 'spread_mm': round(spread, 2)}
            report.append(f'ID {i}: {len(mats)} views, consistent to {spread:.1f} mm')
        never = [i for a in arms for i in gm.IDS[a] if i not in samples]
        if never:
            report.append(f'never seen: {never}')
        if not result:
            self.detail = '; '.join(report)
            self.say('flag calibration failed: ' + '; '.join(report) + ' -- are the flags '
                     'facing back towards the wrist, flat, and clear of the motor?', True)
            return
        merged = dict(self.markers.data) if self.markers is not None else {}
        merged.update(result)
        cal = gm.MarkerCalibration(merged)
        cal.save(note=datetime.datetime.now().isoformat(timespec='seconds'))
        self.markers = cal
        self.detail = '; '.join(report)
        self.say(f'flags calibrated ({", ".join(str(i) for i in sorted(result))}): touches now '
                 f'correct themselves with them')

    def measure_tip(self, arm):
        """Where the fingertip really is (world), from the gripper flags in a
        few fresh colour frames, or None if no calibrated flag is seen."""
        node = self.node
        _c, _d, info = node.latest()
        optical = self.frame[3] if self.frame is not None else 'camera_color_optical_frame'
        t_wo = node.lookup(BASE_FRAME, optical)
        hand = node.lookup(BASE_FRAME, f'openarm_{arm}_hand')
        if info is None or t_wo is None or hand is None or self.markers is None:
            return None, []
        frames = []
        for _ in range(node.args.marker_frames):
            msg = node.fresh_color(timeout=1.0)
            if msg is None:
                break
            frames.append(decode_image(msg))
        r_ow = t_wo[:3, :3].T
        ids = self.markers.ids(arm)
        expected = {i: r_ow @ hand[:3, :3] @ self.markers.hand_marker(i)[:3, 2] for i in ids}
        seen = gm.observe(frames, info, ids, expected)
        if not seen:
            return None, []
        hand_cam, used = gm.hand_from_flags(
            seen, {i: self.markers.hand_marker(i) for i in ids}, r_ow @ hand[:3, :3])
        tip_in_hand = np.array([0.0, 0.0, TCP_IN_HAND + node.args.tip_offset])
        tip_cam = hand_cam[:3, :3] @ tip_in_hand + hand_cam[:3, 3]
        return t_wo[:3, :3] @ tip_cam + t_wo[:3, 3], used

    def auto_calibrate(self):
        """AUTO CAL: the one-time calibration with the gripper flags.

        In normal work the arm hides its flags from the camera, so they are
        not used during touches. Instead, once, with the work area clear,
        each arm visits ~18 spots in open space in front of the camera, tool
        level as in a touch, posed so its flags face the camera; at each the
        flags give where the fingertip really is and the joints where they
        think it is. The difference -- the arm's flex and sag, and the
        camera calibration's error, which the joints cannot see -- is saved
        as this arm's "auto" corrections in touch_corrections.yaml; every
        touch after uses them. TEACH samples are kept. Rerun after moving the
        camera, the flags or the arm mounts.
        """
        node = self.node
        if self.markers is None or not all(self.markers.has_arm(a) for a in ('left', 'right')):
            self.say('AUTO CAL: first, where the flags sit on the grippers (FLAGS)...')
            self.calibrate_markers()
            if self.markers is None:
                self.say('AUTO CAL needs the gripper flags calibrated -- see the message above',
                         True)
                return
        self.say('AUTO CAL: taking a clean frame (keep the area in front of the camera clear)')
        self.capture()
        if self.frame is None or not self.ensure_pick_mode():
            return
        t_wo = node.lookup(BASE_FRAME, self.frame[3])
        targets = self.marker_targets(t_wo)
        origin = t_wo[:3, 3]
        self.abort.clear()
        self.calibrating = True
        report = []
        try:
            for arm in ('right', 'left'):
                if not self.markers.has_arm(arm):
                    report.append(f'{arm}: no calibrated flag, skipped')
                    continue
                start = self.pose(arm, 'pre_pick_state') or node.arm_positions(arm)
                measured, errors, k = [], [], 0
                spots = [p for a, p in targets if a == arm]
                for tip in spots:
                    if self.abort.is_set():
                        raise KeyboardInterrupt
                    k += 1
                    # the posture a touch here would use: level, along the
                    # camera's line of sight, from pre_pick -- with a flag in view
                    best, best_n = None, 0
                    for label, axis, quat, ring in node.candidates(tip, origin):
                        if ring != 0:
                            continue
                        sol = node.solve_ik(arm, node.tcp_for_tip(tip, axis), quat, start)
                        n = self.flag_visible(arm, sol) if sol is not None else 0
                        if n > best_n:
                            best, best_n = sol, n
                        if best_n >= 2:
                            break
                    if best is None:
                        continue
                    self.say(f'AUTO CAL {arm} {k}/{len(spots)}: moving')
                    if not node.move_joints(arm, best, f'autocal {k}', via_home=False):
                        continue
                    node.settle(arm, best, timeout=3.0)
                    model = node.fingertip(arm)
                    real, used = self.measure_tip(arm)
                    if model is None or real is None:
                        self.say(f'AUTO CAL {arm} {k}/{len(spots)}: flags not seen; skipped')
                        continue
                    error = real - model
                    if np.linalg.norm(error) > 0.06:
                        self.say(f'AUTO CAL {arm} {k}: {1000 * np.linalg.norm(error):.0f} mm is '
                                 f'implausible; skipped', True)
                        continue
                    measured.append((real, model - real))   # aim there to land here
                    errors.append(float(np.linalg.norm(error)))
                    self.say(f'AUTO CAL {arm} {k}/{len(spots)}: tip {1000 * errors[-1]:.1f} mm '
                             f'from where the joints put it (flags {used})')
                node.move_joints(arm, start, f'{arm} back to pre_pick', via_home=False)
                if len(measured) >= 4:
                    replace_auto(arm, measured)
                    report.append(f'{arm}: {len(measured)} spots, arm error median '
                                  f'{1000 * np.median(errors):.1f} mm (max '
                                  f'{1000 * max(errors):.1f}) -- saved')
                else:
                    report.append(f'{arm}: only {len(measured)} spots measured, NOT saved '
                                  f'(are its flags facing the camera?)')
        except KeyboardInterrupt:
            self.say('AUTO CAL aborted; nothing more saved')
            return
        finally:
            self.calibrating = False
        self.detail = '; '.join(report)
        self.say('AUTO CAL done: ' + '; '.join(report))

    def _flags_in_view_first(self, arm, solved):
        """`solved` in the same order, but postures where both gripper flags
        will be in view first, then one, then none -- checked lazily (~70 ms
        each). Both matters: with one flag the gripper's tilt comes from the
        joints alone, and their flex then shows at the fingertip (~3 mm per
        degree in simulation); two flags 218 mm apart measure it."""
        fewer = {1: [], 0: []}
        for s_ in solved:
            n = self.flag_visible(arm, s_[6])
            if n >= 2:
                yield s_
            else:
                fewer[n].append(s_)
        yield from fewer[1]
        yield from fewer[0]
