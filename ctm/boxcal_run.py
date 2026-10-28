"""DEPTH CHECK and CAL TIP (box): the automatic calibration (2026-10-26).

DEPTH CHECK (ctm/depth_cal.py): hold the ChArUco board (charuco_board.pdf)
in front of the camera, CAPTURE, move it, CAPTURE ... (3-5 places), FIT.

CAL TIP: the left arm holds the calibration box (calib_box_sheets.pdf, glued
on a 170 x 100 x 25 mm box) with the cup on the cross of face B, the arm at
pre_pick. The run reads the box there (how it sits on the tool), then moves
through ~60 postures across what the camera sees; at each the camera measures
the box's pose and so where the tip is (the cross) -- hundreds of measured
tips, no nudging. The arm model is fitted to them (with the CAL ARMS
recordings for the arm itself), 20 % held out first to say how good it is in
mm, then fitted to all and saved. The box is in the planners all along.
"""

import glob
import math
import os
import time
import types

import cv2
import numpy as np

import robot_camera_calibration as rcc
from ctm import armcal, calib_box as cb, depth_cal
from ctm.common import BASE_FRAME, WS, decode_image, quat_from_matrix, tool_frame_along
from ctm.kinematics import Arm, directions

BOX_POSTURES = 60
BOX_CANDIDATES = 260
BOX_MINUTES = 15
BOX_RMS_RAW_PX = 3.0        # a box pose worse than this, as built (before refining), not kept
BOX_RMS_PX = 0.8            # after refining the box's geometry: worse is not used
HOLDOUT = 0.2


class BoxCalMixin:
    """Part of click_to_move.App (with ArmCalMixin's posture tools)."""

    # -- the depth check ---------------------------------------------------------

    def depth_capture(self):
        """One placement of the board: its depth samples, kept until FIT."""
        node = self.node
        if not hasattr(self, 'depth_samples'):
            self.depth_samples = []
        depth, last = node.averaged_depth(10, raw=True)
        msg = node.fresh_color(timeout=1.5)
        _c, _d, info = node.latest()
        if depth is None or msg is None or info is None:
            self.say('DEPTH CHECK: no camera frames', True)
            return
        found = cb.board_pose(decode_image(msg), info.k, info.d)
        if found is None:
            self.say('DEPTH CHECK: the board is not seen (all of it in view, flat, 35-60 cm '
                     'away, not shiny?)', True)
            return
        board_cam, corners, n = found
        s = depth_cal.samples_from(board_cam, depth, info, corners)
        if s is None:
            self.say('DEPTH CHECK: no depth on the board there - move it a little', True)
            return
        self.depth_samples.append(s)
        err = 1000 * float(np.median(s[:, 3] - s[:, 2]))
        self.say(f'DEPTH CHECK: placement {len(self.depth_samples)}: {n} corners, {len(s)} depth '
                 f'points, depth {err:+.1f} mm off the board at {float(np.median(s[:, 2])):.2f} '
                 f'm. Move the board (other distance / side / tilt) and CAPTURE again; FIT '
                 f'after 3-5')

    def depth_fit(self):
        samples = getattr(self, 'depth_samples', [])
        if len(samples) < 3:
            self.say(f'DEPTH CHECK: {len(samples)} placements - at least 3 first', True)
            return
        coef, before, after = depth_cal.fit(samples)
        depth_cal.save(coef, before, after, len(samples), int(sum(len(s) for s in samples)))
        self.depth_samples = []
        self.say(f'DEPTH CHECK: depth now agrees with colour to {1000 * after:.1f} mm rms (was '
                 f'{1000 * before:.1f}); saved, used from the next capture. Then CAL TIP')

    # -- the box ------------------------------------------------------------------

    def measure_box(self, arm, t_wo, info, frames=6):
        """The box as the camera sees it now: {'tip': world, 'box_in_tcp',
        'rms', 'n'} or None (not seen well enough)."""
        node = self.node
        imgs, qs = [], []
        for _ in range(frames):
            msg = node.fresh_color(timeout=1.0)
            if msg is None:
                break
            imgs.append(decode_image(msg))
            qs.append(node.arm_positions(arm))
        if not imgs:
            return None
        seen = cb.detect(imgs)
        got = cb.box_pose(seen, info.k, info.d)
        if got is None:
            return None
        cam_box, rms, n = got
        big = any(cb.MARKERS[i][1] >= 0.026 for i in seen)
        if rms > BOX_RMS_RAW_PX or not (n >= 3 or (n >= 2 and big)):
            return None
        world_box = t_wo @ cam_box
        tcp = node.lookup(BASE_FRAME, f'openarm_{arm}_hand_tcp')
        q = np.mean(np.array(qs), axis=0)
        self.calib_view = (imgs[-1], None, None, None)
        return {'tip': (world_box @ np.r_[cb.TIP_IN_BOX, 1.0])[:3], 'q': q, 'rms': rms, 'n': n,
                'seen': {int(i): np.asarray(c).tolist() for i, c in seen.items()},
                't_wo': t_wo.tolist(), 'k': list(info.k), 'd': list(info.d),
                'box_in_tcp': np.linalg.inv(tcp) @ world_box if tcp is not None else None}

    def box_visible(self, arm, joints, box_in_tcp, only=None):
        """How many of the box's markers the camera will see with the arm at
        `joints` (the arms' meshes in a z-buffer, the box as measured)."""
        node = self.node
        info, optical = self.frame[2], self.frame[3]
        t_wo = node.lookup(BASE_FRAME, optical)
        links = [f'openarm_{arm}_{s_}' for s_ in rcc.RobotSurface.LINKS] + \
            [f'openarm_{arm}_hand_tcp']
        poses = node.fk(arm, joints, links)
        if t_wo is None or poses is None:
            return 0
        tcp = poses.pop(f'openarm_{arm}_hand_tcp')
        other = 'left' if arm == 'right' else 'right'
        poses.update(node.link_poses((other,)))
        pts, _n = node.robot_surface().posed(poses)
        t_ow = np.linalg.inv(t_wo)
        pc = pts @ t_ow[:3, :3].T + t_ow[:3, 3]
        pc = pc[(pc[:, 2] > 0.05) & (np.abs(pc[:, 0]) < pc[:, 2]) & (np.abs(pc[:, 1]) < pc[:, 2])]
        zb = np.full((info.height, info.width), np.inf, np.float32)
        if len(pc):
            px, z = rcc.project(pc, info)
            u, v = px[:, 0].astype(int), px[:, 1].astype(int)
            ok = (u >= 0) & (u < info.width) & (v >= 0) & (v < info.height)
            np.minimum.at(zb, (v[ok], u[ok]), z[ok].astype(np.float32))
        zb = cv2.erode(zb, np.ones((5, 5), np.uint8))
        box = tcp @ box_in_tcp
        count = 0
        for i, (face, size, corners) in cb.MARKERS.items():
            if only is not None and i not in only:
                continue
            normal = t_ow[:3, :3] @ (box[:3, :3] @ cb.FACES[face][0])
            g = corners.mean(0) + (corners - corners.mean(0)) * 0.7
            w = g @ box[:3, :3].T + box[:3, 3]
            cam = w @ t_ow[:3, :3].T + t_ow[:3, 3]
            if cam[:, 2].min() < 0.2:
                continue
            if -float(normal @ (cam.mean(0) / np.linalg.norm(cam.mean(0)))) < 0.35:
                continue
            pp, zz = rcc.project(cam, info)
            uu, vv = pp[:, 0].astype(int), pp[:, 1].astype(int)
            if not ((uu >= 2) & (uu < info.width - 2) & (vv >= 2) & (vv < info.height - 2)).all():
                continue
            if (zz <= zb[vv, uu] + 0.004).all():
                count += 1 if size < 0.026 else 2      # a big one counts double
        return count

    def calibrate_tip_box(self, arm='left'):
        """CAL TIP: the whole run (see the module's docstring)."""
        node = self.node
        self.say(f'CAL TIP: {arm} arm, box held on the cup at pre_pick...')
        if not self.ensure_pick_mode():
            return
        # Not read yet, but known roughly: face B's centre on the cup, the box
        # square to the tool. In the planners as a box big enough for any turn
        # about the tool (170 x 170 x 25), BEFORE the scene is captured, so the
        # map leaves it out and the arm may move with it.
        tip_off = float(node.args.tip_offsets[arm])
        nominal = np.eye(4)
        nominal[:3, 3] = [0.0, 0.0, tip_off - cb.TIP_IN_BOX[2]]
        node.attach_held(arm, None, None, exact=(np.eye(3), nominal[:3, 3],
                                                  np.array([cb.SIZE[0], cb.SIZE[0], cb.SIZE[2]])))
        self.abort.clear()
        self.calibrating = True
        anchors = None
        try:
            self.capture()
            if self.frame is None:
                return
            info, optical = self.frame[2], self.frame[3]
            t_wo = node.lookup(BASE_FRAME, optical)
            # where the camera sees it: at pre_pick if it can, else the nearest
            # posture that shows face B (2026-10-27: at pre_pick it cannot)
            first = self.measure_box(arm, t_wo, info, frames=10)
            if first is None:
                first = self._show_box(arm, t_wo, info, nominal)
            if first is None or first['box_in_tcp'] is None:
                self.say('CAL TIP: the camera does not see the box anywhere it was shown - is it '
                         'held with face B (the cross) on the cup?', True)
                return
            box_in_tcp = first['box_in_tcp']
            tip_on_tool = (box_in_tcp @ np.r_[cb.TIP_IN_BOX, 1.0])[:3]
            self.say(f'CAL TIP: box seen ({first["n"]} markers, {first["rms"]:.2f} px); the cross '
                     f'is {1000 * tip_on_tool[2]:.1f} mm out along the tool, '
                     f'{1000 * np.hypot(*tip_on_tool[:2]):.1f} mm off its axis')
            # now exactly as it sits
            node.attach_held(arm, None, None, exact=(box_in_tcp[:3, :3], box_in_tcp[:3, 3],
                                                      np.array(cb.SIZE)))
            anchors = self._box_postures(arm, t_wo, info, box_in_tcp)
        except KeyboardInterrupt:
            self.say('CAL TIP: aborted; nothing saved', True)
            anchors = None
        finally:
            self.calibrating = False
            self.calib_view = None
            self._back_to_pre_pick(arm)
            node.detach_held(arm)
        if not anchors:
            return
        self._fit_box(arm, anchors)

    def _show_box(self, arm, t_wo, info, nominal):
        """Move to the nearest posture where the camera should see face B
        (the box as `nominal`: its turn about the tool not known, so face B --
        which faces back along the tool -- is what is counted), and read the
        box there. The reading, or None after a few tries."""
        node = self.node
        start = node.arm_positions(arm)
        rng = np.random.default_rng(5)
        face_b = {i for i, m in cb.MARKERS.items() if m[0] == 'B'}
        cands = []
        self.say('CAL TIP: the box is not in view at pre_pick - finding a posture that shows it...')
        for tip, axis, roll in self._front_poses(arm, t_wo, info, rng):
            if self.abort.is_set():
                raise KeyboardInterrupt
            if len(cands) >= 40:
                break
            quat = quat_from_matrix(tool_frame_along(axis, math.radians(roll)))
            tcp = node.tcp_for_tip(tip, axis, arm)
            seeds = node.fast_ik(arm, tcp, quat, start, want=1)
            sol = node.solve_ik(arm, tcp, quat, seeds[0]) if seeds else None
            if sol is not None and self.box_visible(arm, sol, nominal, only=face_b) >= 4:
                cands.append(np.array(sol))
        cands.sort(key=lambda q: float(np.linalg.norm(q - np.asarray(start))))
        for n, q in enumerate(cands[:4], start=1):
            if self.abort.is_set():
                raise KeyboardInterrupt
            self.say(f'CAL TIP: showing the box to the camera ({n})...')
            if not node.move_joints(arm, q.tolist(), f'boxcal show {n}', via_home=False):
                continue
            node.settle(arm, q.tolist(), timeout=3.0)
            time.sleep(0.3)
            first = self.measure_box(arm, t_wo, info, frames=10)
            if first is not None:
                return first
        return None

    def _box_postures(self, arm, t_wo, info, box_in_tcp):
        node = self.node
        start = self.pose(arm, 'pre_pick_state') or node.arm_positions(arm)
        rng = np.random.default_rng(3)
        cands, where = [], []
        self.say('CAL TIP: choosing postures where the camera sees the box...')
        for tip, axis, roll in self._front_poses(arm, t_wo, info, rng):
            if self.abort.is_set():
                raise KeyboardInterrupt
            if len(cands) >= BOX_CANDIDATES:
                break
            quat = quat_from_matrix(tool_frame_along(axis, math.radians(roll)))
            tcp = node.tcp_for_tip(tip, axis, arm)
            seeds = node.fast_ik(arm, tcp, quat, start, want=1)
            sol = node.solve_ik(arm, tcp, quat, seeds[0]) if seeds else None
            if sol is not None and self.box_visible(arm, sol, box_in_tcp) >= 4:
                cands.append(np.array(sol))
                where.append(np.r_[np.asarray(tip) / 0.05, np.asarray(axis) / 0.35])
        if len(cands) < 10:
            self.say(f'CAL TIP: only {len(cands)} postures show the box - not enough', True)
            return None
        # the most spread out, visited nearest-first (as CAL ARMS)
        feats = np.hstack([np.array(cands) / 0.5, np.array(where)])
        chosen = [0]
        dist = np.linalg.norm(feats - feats[0], axis=1)
        while len(chosen) < min(BOX_POSTURES, len(cands)):
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(feats - feats[nxt], axis=1))
        order, here, left = [], np.array(start), set(chosen)
        while left:
            nxt = min(left, key=lambda i: np.linalg.norm(cands[i] - here))
            order.append(nxt)
            left.discard(nxt)
            here = cands[nxt]
        anchors, failed = [], 0
        t_end = time.monotonic() + 60.0 * BOX_MINUTES
        for n, i in enumerate(order, start=1):
            if self.abort.is_set():
                raise KeyboardInterrupt
            if time.monotonic() > t_end:
                self.say(f'CAL TIP: {BOX_MINUTES} min done - fitting with {len(anchors)}')
                break
            self.say(f'CAL TIP: posture {n}/{len(order)} ({len(anchors)} measured)')
            before = node.arm_positions(arm)
            if not node.move_joints(arm, cands[i].tolist(), f'boxcal {n}', via_home=False,
                                    attempts=1):
                failed += 1
                if failed == 3:
                    self.say('CAL TIP: moves keep failing; waiting for the planner...')
                    time.sleep(8.0)
                    node.require_move_group()
                elif failed >= 6:
                    self.say(f'CAL TIP: the planner is not answering - stopping with '
                             f'{len(anchors)}', True)
                    break
                continue
            failed = 0
            node.settle(arm, cands[i].tolist(), timeout=3.0)
            time.sleep(0.2)
            m = self.measure_box(arm, t_wo, info)
            if m is not None:
                anchors.append({'arm': arm, 'q': m['q'].tolist(), 'point': m['tip'].tolist(),
                                'tip_offset': float(node.args.tip_offsets[arm]),
                                'rms': m['rms'], 'n': m['n'], 'seen': m['seen'],
                                't_wo': m['t_wo'], 'k': m['k'], 'd': m['d'],
                                'dirs': (directions(before, m['q']) if before is not None
                                         else np.zeros(7)).tolist()})
        return anchors

    def _refine_box(self, anchors):
        """The box's real geometry from all the readings at once (calib_box.
        refine_geometry), then each reading's tip again with it; readings
        that still do not fit (> BOX_RMS_PX) are dropped."""
        with_px = [a for a in anchors if a.get('seen')]
        if len(with_px) < 5:
            return [a for a in anchors if a['rms'] <= BOX_RMS_PX]
        k, d = with_px[0]['k'], with_px[0]['d']
        views = [{int(i): np.asarray(c) for i, c in a['seen'].items()} for a in with_px]
        geom, before, after = cb.refine_geometry(views, k, d, log=self.node.get_logger().info)
        out = []
        for a, v in zip(with_px, views):
            got = cb.box_pose_with(v, k, d, geom)
            if got is None or got[1] > BOX_RMS_PX:
                continue
            cam_box, rms, _n = got
            world_box = np.asarray(a['t_wo']) @ cam_box
            out.append(dict(a, point=(world_box @ np.r_[cb.TIP_IN_BOX, 1.0])[:3].tolist(),
                            rms=rms))
        self.say(f'CAL TIP: the box as built: its markers fit {after:.2f} px (as drawn: '
                 f'{before:.2f}); {len(out)} of {len(anchors)} readings kept')
        return out

    def _fit_box(self, arm, anchors):
        """The arm model fitted to the measured tips (+ the CAL ARMS recordings
        for the arm), 20 % held out first for an honest error in mm."""
        node = self.node
        os.makedirs(os.path.join(WS, 'armcal_samples'), exist_ok=True)
        stamp = time.strftime('%Y-%m-%dT%H-%M-%S')
        np.save(os.path.join(WS, 'armcal_samples', f'{arm}_box_{stamp}.npy'),
                np.array(anchors, dtype=object), allow_pickle=True)
        anchors = self._refine_box(anchors)
        if len(anchors) < 10:
            self.say(f'CAL TIP: only {len(anchors)} good box readings - not enough to fit', True)
            return
        files = sorted(glob.glob(os.path.join(WS, 'armcal_samples', f'{arm}_2*.npz')))
        if not files:
            self.say('CAL TIP: no CAL ARMS recordings for this arm - run CAL ARMS once', True)
            return
        z = np.load(files[-1])
        info = types.SimpleNamespace(k=z['k'].tolist(), d=z['d'].tolist(),
                                     width=int(z['size'][0]), height=int(z['size'][1]))
        samples = [(q, {int(i): c for i, c in zip(z['ids'], pix) if not np.isnan(c).any()})
                   for q, pix in zip(z['q'], z['pixels'])]
        dirs = z['dirs'] if 'dirs' in z and len(z['dirs']) == len(samples) else \
            np.zeros((len(samples), 7))
        variant = (armcal.load_model_raw().get(arm) or {}).get('model', 'backlash')
        tip_offset = node.args.tip_offsets[arm]
        kw = dict(variant=variant, tip_offset=tip_offset, tip_sigma=(0.02, 0.02, 0.03),
                  shift_sigma=1e-5, log=node.get_logger().info)

        kin = Arm(arm)

        def tip_err(params, rows):
            delta = np.r_[params.get('tip_delta', np.zeros(3)), 0.0]
            return np.array([np.linalg.norm((kin.tcp(np.array(r['q']), params)
                                             @ (np.array([0, 0, r['tip_offset'], 1.0]) + delta))[:3]
                                            - np.array(r['point'])) for r in rows])

        self.say(f'CAL TIP: fitting {len(anchors)} measured tips (+ {len(samples)} CAL ARMS '
                 f'postures)...')
        p0, b0, st0 = armcal.fit(arm, samples, z['t_wo'], info, max_nfev=40, **kw)
        samples, dropped, keep = armcal.drop_outliers(arm, samples, p0, b0, z['t_wo'], info)
        dirs = dirs[keep]
        rng = np.random.default_rng(4)
        idx = rng.permutation(len(anchors))
        n_hold = max(3, int(round(HOLDOUT * len(anchors))))
        held, train = [anchors[i] for i in idx[:n_hold]], [anchors[i] for i in idx[n_hold:]]
        p_t, _b, _s = armcal.fit(arm, samples, z['t_wo'], info, dirs=dirs, anchors=train, **kw)
        e_hold = tip_err(p_t, held) * 1000
        before = tip_err(self.arm_models.params[arm], anchors) * 1000 \
            if self.arm_models.has(arm) else None
        p, b, st = armcal.fit(arm, samples, z['t_wo'], info, dirs=dirs, anchors=anchors, **kw)
        e_all = tip_err(p, anchors) * 1000
        st.update(model=variant, dropped=dropped, holdout_px=float('nan'),
                  box_tips=len(anchors), box_holdout_mm=float(np.mean(e_hold)),
                  box_holdout_p90_mm=float(np.percentile(e_hold, 90)),
                  box_fit_mm=float(np.mean(e_all)))
        import shutil
        if os.path.exists(armcal.MODEL_FILE):            # the model before, kept
            shutil.copy(armcal.MODEL_FILE, armcal.MODEL_FILE.replace('.yaml', f'.before_boxcal_{stamp}.yaml'))
        armcal.save_model({arm: (p, b, st)})
        self.arm_models = armcal.ArmModels()
        self.detail = (f'CAL TIP {arm}: tip error on postures NOT used in the fit: mean '
                       f'{np.mean(e_hold):.1f} mm, 90 % under {np.percentile(e_hold, 90):.1f} mm, '
                       f'max {np.max(e_hold):.1f}' + (f' (the old model: mean {np.mean(before):.1f} '
                                                     f'mm)' if before is not None else ''))
        self.say(f'CAL TIP: saved. {self.detail}. Take the box off the cup.')
