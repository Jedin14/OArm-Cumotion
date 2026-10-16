"""CALIBRATE ARMS, the run on the robot (the fit itself is ctm/armcal.py).

With two pieces (ctm/board.py, gripper_boards.pdf) on each gripper and the area
in front of the camera clear except for a flat surface (the board) to touch:

  0. set-up: the board (for the touch check) ~50 cm in front of the robot,
     the space between clear -- the strip postures happen there;
  1. postures: ~120 candidates across the workspace -- near and far, high
     and low, level, tilted down to 45 deg, turned 25 deg, three rolls --
     that the arm can reach without collisions and that show the camera at
     least 4 of the pieces' 12 markers (most of one piece: the wrist usually
     hides the other); the 70 most spread out are visited;
  2. at each: settle, 6 frames, the pieces' marker corners averaged, the
     joint angles read;
  3. the fit: postures the model cannot explain left out, then the model
     (of a few) that best predicts postures it was not fitted to;
  4. the touch check: 8 touches on the flat surface (found in the depth
     image) around the wall marker, the tool tilted +/-25 deg two ways (less
     where out of reach); where the model puts the tip
     at each contact, against the surface, says how good it is in mm. If
     that is worse than 1.5 mm the touches go into the fit as well.
  5. saved in arm_model.yaml; every touch after uses it.
"""

import math
import time

import cv2
import numpy as np

import robot_camera_calibration as rcc
from cv2 import FONT_HERSHEY_SIMPLEX as FONT_CV
from ctm import armcal, board as bd
from ctm.kinematics import directions
from ctm.common import (BASE_FRAME, WS, decode_image, pixel_ray, quat_from_matrix, rot_about,
                        tool_frame_along)

WALL_ID = 11                    # the reference marker on the wall (docs/gripper_markers.pdf)
WALL_MARKER = 0.050             # m: its black square (64 mm card)
WALL_X = 0.50                   # m: the wall is this far in front of the robot (world x)
POSTURES = 100                  # per arm, spread over everywhere the camera sees and it reaches
CANDIDATES = 320
POSTURE_MINUTES = 16            # per arm: then fit with what there is (never stuck there)
TOUCH_TILTS = (25, 18, 12)      # deg: the tool tilted this much, smaller where it does not reach
TOUCH_CHECK_MM = 1.5            # touches worse than this go into the fit


class ArmCalMixin:
    """Part of click_to_move.App."""

    # -- what the camera will see -----------------------------------------------

    def strip_visible(self, arm, joints):
        """How many of the pieces' 12 markers the camera will see with the
        arm at `joints` (both arms' meshes in a z-buffer, the pieces where
        they nominally sit)."""
        node = self.node
        info, optical = self.frame[2], self.frame[3]
        t_wo = node.lookup(BASE_FRAME, optical)
        # the pieces lie ON the gripper's rail, so the gripper itself (hand,
        # fingers) is not counted as hiding it -- the wrist and arm are
        links = [f'openarm_{arm}_{s_}' for s_ in rcc.RobotSurface.LINKS
                 if s_ not in ('hand', 'left_finger', 'right_finger')] + \
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
        pieces = [tcp @ armcal.nominal_piece_in_tcp(k) for k in range(len(bd.pieces(arm)))]
        count = 0
        for i, corners in bd.corners_3d(arm).items():
            board = pieces[bd.piece_of(arm, i)]
            normal = t_ow[:3, :3] @ board[:3, 2]
            g = corners.mean(0) + (corners - corners.mean(0)) * 0.7           # inside the marker
            w = g @ board[:3, :3].T + board[:3, 3]
            cam = w @ t_ow[:3, :3].T + t_ow[:3, 3]
            if cam[:, 2].min() < 0.2:
                continue
            if -float(normal @ (cam.mean(0) / np.linalg.norm(cam.mean(0)))) < 0.3:
                continue                                                      # edge-on / back
            pp, zz = rcc.project(cam, info)
            uu, vv = pp[:, 0].astype(int), pp[:, 1].astype(int)
            inside = (uu >= 2) & (uu < info.width - 2) & (vv >= 2) & (vv < info.height - 2)
            if not inside.all():
                continue
            if (zz <= zb[vv, uu] + 0.004).all():
                count += 1
        return count

    def find_plane(self, arm):
        """The flat surface to touch: the biggest plane facing the camera in
        this arm's half of the captured depth, 25-80 cm out. (n, d) in world
        with n towards the camera and n . p = d on it -- or None."""
        bgr, depth, info, optical = self.frame
        t_wo = self.node.lookup(BASE_FRAME, optical)
        h, w = depth.shape
        us = np.arange(0, w // 2) if arm == 'left' else np.arange(w // 2, w)
        vs, uu = np.meshgrid(np.arange(0, h, 3), us[::3], indexing='ij')
        z = depth[vs, uu].ravel()
        keep = (z > 0.25) & (z < 0.8)
        if keep.sum() < 300:
            return None
        sx, sy = info.width / float(w), info.height / float(h)
        rays = np.array([pixel_ray(u * sx, v * sy, info) for u, v in
                         zip(uu.ravel()[keep], vs.ravel()[keep])])
        cam = rays * z[keep][:, None]
        world = cam @ t_wo[:3, :3].T + t_wo[:3, 3]
        rng = np.random.default_rng(0)
        best = None
        for _ in range(400):
            p = world[rng.choice(len(world), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            inl = np.abs((world - p[0]) @ n) < 0.004
            if best is None or inl.sum() > best.sum():
                best = inl
        pts = world[best]
        centre = pts.mean(0)
        n = np.linalg.svd(pts - centre)[2][2]
        if n @ (t_wo[:3, 3] - centre) < 0:
            n = -n
        view = (centre - t_wo[:3, 3]) / np.linalg.norm(centre - t_wo[:3, 3])
        if -(n @ view) < 0.6 or len(pts) < 200:
            return None
        return n, float(n @ centre), pts

    # -- the run ----------------------------------------------------------------

    def calibrate_arms(self):
        node, args = self.node, self.node.args
        self.say('CALIBRATE ARMS: clean frame (calibration pieces on, area clear except the board)...')
        self.capture()
        if self.frame is None or not self.ensure_pick_mode():
            return
        info, optical = self.frame[2], self.frame[3]
        t_wo = node.lookup(BASE_FRAME, optical)
        wall = self.find_wall_marker(t_wo, info)
        if wall is None:
            self.say(f'CALIBRATE ARMS: marker ID {WALL_ID} not found on the wall - stick it on the '
                     f'wall {WALL_X * 100:.0f} cm in front, in view of the camera, and try again',
                     True)
            return
        if node.pieces_on and not node.set_pieces(True):
            # never move without them: the folded pieces stick out 4 cm, and a
            # run that planned without them hit the wall (2026-10-15)
            self.say('CALIBRATE ARMS: the planners did not take the calibration pieces '
                     '(is cuMotion up?) - not moving; try again in a moment', True)
            return
        self.abort.clear()
        self.calibrating = True
        if not self.confirm_wall_marker(wall, t_wo, info):
            self.calibrating = False
            self.say('CALIBRATE ARMS cancelled at the wall marker check; nothing moved')
            return
        node.robot_surface()
        results, report = {}, []
        try:
            for arm in ('right', 'left'):
                # the other arm must be back home first: never both out (the
                # right arm stayed out, stuck, while the left one ran, 2026-10-13)
                other = 'left' if arm == 'right' else 'right'
                if not self._back_to_pre_pick(other):
                    report.append(f'{other} arm could not get back to pre_pick - stopped '
                                  f'before calibrating the {arm} arm')
                    break
                samples, dirs = self._collect_postures(arm, t_wo, info)
                # home while it fits (minutes): not left out at the last posture
                self._back_to_pre_pick(arm)
                if len(samples) < 15:
                    report.append(f'{arm}: only {len(samples)} postures with the pieces seen '
                                  f'- NOT calibrated (are the pieces on and facing the wrist?)')
                    continue
                self._save_samples(arm, samples, t_wo, info, dirs)
                log = node.get_logger().info
                params, boards, stats, samples, dirs, variant = self._fit_arm(
                    arm, samples, dirs, t_wo, info)
                play = params.get('hysteresis')
                if play is not None and np.max(np.abs(play)) > 0.015:
                    j = int(np.argmax(np.abs(play)))
                    self.say(f'{arm} arm: joint {j + 1} has {abs(play[j]) * 1000:.0f} mrad of play '
                             f'that depends on which way it last turned -- check that joint, and '
                             f'that the calibration pieces on that gripper are firm', True)
                plane = self.find_plane(arm)
                # no tilted touches with a tool on: they press the rubber
                # extension sideways, and its touches are not fitted anyway
                touches = (self._touch_check(arm, plane, params, around=wall['centre'])
                           if plane is not None and not args.tools[arm]['on'] else [])
                if len(touches) >= 4:
                    err = self._touch_errors(arm, touches, plane, params, args.tip_offsets[arm])
                    stats['touch_check_mm'] = err
                    # Only a rigid tip's touches go into the fit: a tool (the
                    # left arm's vacuum cup) squashes by a different amount at
                    # each tilt, which the fit would read as arm error. The
                    # strip fit itself is about the hand, tool or not.
                    rigid = not args.tools[arm]['on']
                    if err > TOUCH_CHECK_MM and not rigid:
                        self.say(f'{arm} arm: touches disagree by {err:.1f} mm -- not used: '
                                 f'its tool is not rigid')
                    elif err > TOUCH_CHECK_MM:
                        self.say(f'{arm} arm: touches disagree by {err:.1f} mm; refitting with '
                                 f'them')
                        params, boards, stats2 = armcal.fit(
                            arm, samples, t_wo, info, log=log, touches=touches,
                            plane=plane[:2], tip_offset=args.tip_offsets[arm], dirs=dirs,
                            variant=variant)
                        stats.update(stats2)
                        stats['touch_check_mm'] = self._touch_errors(arm, touches, plane, params,
                                                                     args.tip_offsets[arm])
                else:
                    stats['touch_check_mm'] = None
                results[arm] = (params, boards, stats)
                urdf = self._urdf_touch_errors(arm, touches, plane, args.tip_offsets[arm])
                report.append(
                    f'{arm}: {stats["postures"]} postures, model {variant}, fit '
                    f'{stats["rms_px"]:.2f} px, unseen {stats["holdout_px"]:.2f} px'
                    + (f', touch check {stats["touch_check_mm"]:.1f} mm (URDF alone '
                       f'{urdf:.1f} mm)' if stats.get('touch_check_mm') is not None else
                       ', no surface found to touch-check on' if plane is None else
                       ', no tilted touch check (tool on)' if args.tools[arm]['on'] else
                       f', touch check: only {len(touches)} touches felt the surface'))
        except KeyboardInterrupt:
            self.say('CALIBRATE ARMS aborted; nothing saved')
            return
        finally:
            self.calibrating = False
            for arm in ('right', 'left'):
                start = self.pose(arm, 'pre_pick_state')
                if start is not None:
                    node.move_joints(arm, start, f'{arm} back to pre_pick', via_home=False)
        if results:
            armcal.save_model(results)
            self.arm_models = armcal.ArmModels()
            if all(self._back_to_pre_pick(a) for a in ('right', 'left')):
                report += self.wall_touches(wall, t_wo)
            else:
                report.append('an arm could not get back to pre_pick - no marker touches')
        self.detail = '; '.join(report)
        self.say('CALIBRATE ARMS: ' + ('; '.join(report) or 'nothing calibrated'),
                 not results)

    def _collect_postures(self, arm, t_wo, info, poses=None):
        """Visit postures where the camera sees this arm's pieces and record
        them: (samples, approach directions). poses: [(tip, tool axis, roll)]
        to choose from; default the front ones."""
        node = self.node
        start = self.pose(arm, 'pre_pick_state') or node.arm_positions(arm)
        rng = np.random.default_rng(1 if arm == 'left' else 2)
        if poses is None:
            poses = self._front_poses(arm, t_wo, info, rng)
        else:
            poses = list(poses)
            rng.shuffle(poses)
        cands, where = [], []
        self.say(f'{arm} arm: choosing postures where the camera sees its pieces...')
        for tip, axis, roll in poses:
            if self.abort.is_set():
                raise KeyboardInterrupt
            if len(cands) >= CANDIDATES:
                break
            quat = quat_from_matrix(tool_frame_along(axis, math.radians(roll)))
            tcp = node.tcp_for_tip(tip, axis, arm)
            # can it get there at all (fast)? then collision-checked from there
            seeds = node.fast_ik(arm, tcp, quat, start, want=1)
            sol = node.solve_ik(arm, tcp, quat, seeds[0]) if seeds else None
            if sol is not None and axis[2] < -0.6:
                # pointing down: also clear where the arm really arrives, ~4 cm
                # low (a top-down calibration at 6 cm brushed the table, 2026-10-14)
                low = node.solve_ik(arm, tcp - np.array([0.0, 0.0, 0.04]), quat, sol)
                if low is None or np.max(np.abs(np.subtract(low, sol))) > 0.4:
                    continue
            if sol is not None and self.strip_visible(arm, sol) >= 4:
                cands.append(np.array(sol))
                where.append(np.r_[np.asarray(tip) / 0.05, np.asarray(axis) / 0.35])
        return self._visit_postures(arm, cands, start, rng, where)

    def _front_poses(self, arm, t_wo, info, rng):
        """CAL ARMS's postures: [(tip, tool axis, roll)] in front of the robot."""
        origin = t_wo[:3, 3]
        # fingertip spots along camera rays in this arm's half, 32-44 cm out (the
        # strip is 11 cm behind the tip: nearer, it is too close to the camera
        # and off the picture) and always 6 cm in front of whatever the camera
        # sees there: the strip postures need open space -- the board, for the
        # touch check, goes ~50 cm out, behind them
        bgr, depth, _info, _opt = self.frame
        h, w = depth.shape
        tips = []
        # this arm's half of the picture and a little past the middle (picks
        # there go to either arm), top to bottom, near to far: everywhere the
        # camera sees that the arm may be sent to
        cols = (0.05, 0.14, 0.23, 0.32, 0.41, 0.55) if arm == 'left' else \
            (0.45, 0.59, 0.68, 0.77, 0.86, 0.95)
        for fu in cols:
            for fv in (0.15, 0.32, 0.5, 0.68, 0.85):
                du, dv = int(fu * w), int(fv * h)
                patch = depth[max(0, dv - 8):dv + 9, max(0, du - 8):du + 9]
                seen = patch[patch > 0.1]
                limit = float(np.percentile(seen, 10)) - 0.06 if seen.size else 0.6
                for dist in (0.30, 0.36, 0.42, 0.48, 0.54):
                    if dist <= limit:
                        ray = pixel_ray(fu * info.width, fv * info.height, info) * dist
                        tips.append(t_wo[:3, :3] @ ray + t_wo[:3, 3])
        # level and tilted for front picks; 70 deg down and straight down for
        # top-down ones -- one model for both (the top/bottom faces of the
        # pieces face the camera then, 2026-10-15)
        turns = [(0, 0), (0, -25), (0, -45), (25, 0), (-25, 0), (0, 15), (25, -30), (-25, -30),
                 (0, -70), (0, -90), (25, -70), (-25, -70)]
        combos = [(t, tr, roll) for t in tips for tr in turns for roll in (0, 180, 90)]
        rng.shuffle(combos)
        up = np.array([0.0, 0.0, 1.0])
        out = []
        for tip, (yaw, pitch), roll in combos:
            ray = tip - origin
            ray[2] = 0.0
            ray /= np.linalg.norm(ray)
            sidev = np.cross(up, ray)
            out.append((tip, rot_about(up, math.radians(yaw)) @ rot_about(sidev, math.radians(-pitch))
                        @ ray, roll))
        return out

    def _visit_postures(self, arm, cands, start, rng, where=None):
        """The most spread out of the candidate postures, visited and measured:
        (samples, approach directions). Spread in where the tip is and which
        way it points (5 cm ~ 20 deg ~ 1 unit) and in joint space alike -- so
        the whole region the camera sees gets covered, in all its postures."""
        node = self.node
        if not cands:
            return [], np.zeros((0, 7))
        feats = np.array(cands) / 0.5
        if where is not None and len(where) == len(cands):
            feats = np.hstack([feats, np.array(where)])
        chosen = [0]
        dist = np.linalg.norm(feats - feats[0], axis=1)
        while len(chosen) < min(POSTURES, len(cands)):
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(feats - feats[nxt], axis=1))
        order, here = [], np.array(start)
        left = set(chosen)
        while left:
            nxt = min(left, key=lambda i: np.linalg.norm(cands[i] - here))
            order.append(nxt)
            left.discard(nxt)
            here = cands[nxt]
        samples, dirs = [], []
        failed = 0
        t_end = time.monotonic() + 60.0 * POSTURE_MINUTES
        for n, i in enumerate(order, start=1):
            if self.abort.is_set():
                raise KeyboardInterrupt
            if time.monotonic() > t_end:
                self.say(f'{arm} arm: {POSTURE_MINUTES} min of postures done - fitting with '
                         f'{len(samples)}')
                break
            self.say(f'{arm} arm: posture {n}/{len(order)} ({len(samples)} measured)')
            # one cuMotion try, then OMPL: three cuMotion retries on the moves
            # it cannot plan cost up to 15 s a posture (2026-10-13)
            before = node.arm_positions(arm)
            if not node.move_joints(arm, cands[i].tolist(), f'armcal {n}', via_home=False,
                                    attempts=1):
                failed += 1
                if failed == 3:
                    # three in a row: the planner is restarting (cuMotion exits
                    # when its solver breaks; launch respawns it) -- wait for it
                    self.say(f'{arm} arm: moves keep failing; waiting for the planner...')
                    time.sleep(8.0)
                    node.require_move_group()
                elif failed >= 6:
                    self.say(f'{arm} arm: the planner is not answering - stopping this arm '
                             f'with {len(samples)} postures', True)
                    break
                continue
            failed = 0
            node.settle(arm, cands[i].tolist(), timeout=3.0)
            time.sleep(0.15)
            frames, qs = [], []
            for _ in range(6):
                msg = node.fresh_color(timeout=1.0)
                if msg is None:
                    break
                frames.append(decode_image(msg))
                qs.append(node.arm_positions(arm))
            seen = bd.detect_many(frames, arm, min_frames=4)
            if len(seen) >= 4 and qs:
                q = np.mean(np.array(qs), axis=0)
                samples.append((q, seen))
                dirs.append(directions(before, q) if before is not None else np.zeros(7))
                if frames:
                    self.calib_view = (frames[-1], None, None, None)
        self.calib_view = None
        return samples, np.array(dirs)

    def tip_from_pieces(self, arm, frames, boards, t_wo, info):
        """Where the real tip is (world), from the gripper pieces in these
        frames and where the fit put them on the hand -- measured, not
        modelled. None if no piece is seen well enough."""
        seen = bd.detect_many(frames, arm, min_frames=max(1, len(frames) // 2))
        if not seen:
            return None
        k = np.array(info.k, float).reshape(3, 3)
        d = np.array(info.d, float) if len(info.d) else np.zeros(5)
        c3 = bd.corners_3d(arm)
        tip = np.array([0.0, 0.0, self.node.args.tip_offsets[arm], 1.0])
        # plus where the anchors found the tip on the hand (REFIT): without it
        # the camera and the arm model disagreed by its 30 mm (2026-10-16)
        models = getattr(self, 'arm_models', None)
        if models is not None and arm in models.params:
            tip[:3] += models.params[arm].get('tip_delta', np.zeros(3))
        tips = []
        for piece in range(len(boards)):
            ids = [i for i in seen if bd.piece_of(arm, i) == piece]
            if len(ids) < 2:
                continue
            ok, rv, tv = cv2.solvePnP(np.concatenate([c3[i] for i in ids]),
                                      np.concatenate([seen[i] for i in ids]), k, d)
            if not ok:
                continue
            cam_piece = np.eye(4)
            cam_piece[:3, :3] = cv2.Rodrigues(rv)[0]
            cam_piece[:3, 3] = tv.ravel()
            tips.append((t_wo @ cam_piece @ np.linalg.inv(boards[piece]) @ tip)[:3])
        if not tips:
            return None
        return np.mean(tips, axis=0) + (models.shift(arm) if models is not None else 0.0)

    def _fit_arm(self, arm, samples, dirs, t_wo, info):
        """Rough fit, the postures it cannot explain left out, the model that
        best predicts unseen postures, the final fit. (params, boards, stats,
        samples kept, dirs kept, model name)"""
        node, args = self.node, self.node.args
        log = node.get_logger().info
        self.say(f'{arm} arm: fitting {len(samples)} postures...')
        params, boards, stats = armcal.fit(arm, samples, t_wo, info, log=log,  # rough
                                              tip_offset=args.tip_offsets[arm], max_nfev=40)
        samples, dropped, keep = armcal.drop_outliers(arm, samples, params, boards, t_wo, info)
        dirs = dirs[keep]
        self.say(f'{arm} arm: '
                 + (f'{dropped} postures disagree (misread or moving), left out; '
                    if dropped else '')
                 + 'choosing the model that best predicts postures it has not seen...')
        variant, scores = armcal.choose_model(arm, samples, dirs, t_wo, info, log=log)
        params, boards, stats = armcal.fit(arm, samples, t_wo, info, log=log,
                                              tip_offset=args.tip_offsets[arm],
                                              dirs=dirs, variant=variant)
        stats.update(dropped=dropped, model=variant, holdout_px=scores[variant])
        return params, boards, stats, samples, dirs, variant

    def _save_samples(self, arm, samples, t_wo, info, dirs=None):
        """The raw measurements, so a fit can be redone and looked into offline:
        armcal_samples/<arm>_<time>.npz (joints, marker pixels, camera)."""
        import datetime
        import os
        folder = os.path.join(WS, 'armcal_samples')
        os.makedirs(folder, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        ids = sorted({i for _q, px in samples for i in px})
        pix = np.full((len(samples), len(ids), 4, 2), np.nan)
        for n, (_q, px) in enumerate(samples):
            for i, c in px.items():
                pix[n, ids.index(i)] = c
        np.savez(os.path.join(folder, f'{arm}_{stamp}.npz'), q=np.array([q for q, _ in samples]),
                 ids=np.array(ids), pixels=pix, t_wo=t_wo, k=np.array(info.k, float),
                 d=np.array(info.d, float), size=np.array([info.width, info.height]),
                 dirs=np.zeros((len(samples), 7)) if dirs is None else np.asarray(dirs))

    def _holdout_px(self, arm, held, params, boards, t_wo, info):
        model = armcal.Arm(arm)
        k = np.array(info.k, float).reshape(3, 3)
        d = np.array(info.d, float) if len(info.d) else np.zeros(5)
        t_ow = np.linalg.inv(t_wo)
        c3 = bd.corners_3d(arm)
        errs = []
        for q, px in held:
            tcp = model.tcp(q, params)
            for i, c in px.items():
                t_board = tcp @ boards[bd.piece_of(arm, i)]
                w = c3[i] @ t_board[:3, :3].T + t_board[:3, 3]
                cam = w @ t_ow[:3, :3].T + t_ow[:3, 3]
                p, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d)
                errs += list(np.linalg.norm(p.reshape(-1, 2) - c, axis=1))
        return float(np.sqrt(np.mean(np.square(errs)))) if errs else float('nan')

    def _back_to_pre_pick(self, arm):
        """Bring `arm` to pre_pick and check it got there (within 0.1 rad on
        every joint -- it sags up to ~0.05 under gravity, which a 0.05 check
        read as "could not get back", 2026-10-16): a direct move, then via
        home. True when it is there."""
        node = self.node
        target = self.pose(arm, 'pre_pick_state')
        if target is None:
            return True
        here = node.arm_positions(arm)
        for via_home in (False, True):
            if here is not None and max(abs(a - b) for a, b in zip(here, target)) < 0.1:
                return True
            self.say(f'{arm} arm: back to pre_pick...')
            if node.move_joints(arm, target, f'{arm} back to pre_pick', via_home=via_home):
                node.settle(arm, target, timeout=2.0)
            here = node.arm_positions(arm)
        ok = here is not None and max(abs(a - b) for a, b in zip(here, target)) < 0.1
        if not ok:
            self.say(f'{arm} arm: could not get back to pre_pick', True)
        return ok

    # -- the wall marker ------------------------------------------------------------

    def find_wall_marker(self, t_wo, info, marker_id=WALL_ID):
        """The reference marker (ID 11, 50 mm) on the wall: its centre in world
        -- the ray through its centre pixel, met with the wall's plane from the
        depth camera -- and how that compares with what is known (the wall
        WALL_X in front) and with the marker's own size (PnP). Said in the
        window; returns {'centre', 'normal', ...} or None."""
        node = self.node
        frames = []
        for _ in range(15):
            msg = node.fresh_color(timeout=1.0)
            if msg is not None:
                frames.append(decode_image(msg))
        dictionary = bd.dictionary()
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 3
        seen = []
        for bgr in frames:
            corners, ids, _ = cv2.aruco.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                                                      dictionary, parameters=params)
            if ids is not None:
                seen += [c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if i == marker_id]
        if len(seen) < 5:
            return None
        px = np.median(np.array(seen), axis=0)
        k = np.array(info.k, float).reshape(3, 3)
        d = np.array(info.d, float) if len(info.d) else np.zeros(5)
        # its size: PnP (depth only to ~5 mm from a 50 mm marker; position across the view good)
        h = WALL_MARKER / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
        ok, rv, tv = cv2.solvePnP(obj, px, k, d, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        pnp = t_wo[:3, :3] @ tv.ravel() + t_wo[:3, 3] if ok else None
        # the wall's plane, from the depth camera around the marker
        bgr, depth, _info, _opt = self.frame
        dh, dw = depth.shape
        sx, sy = info.width / float(dw), info.height / float(dh)
        c_px = px.mean(0)
        # the plane from the depth on the marker's own card only (its 64 mm
        # card around the 50 mm square): a wider patch took in the wall
        # behind it and tilted the plane 77 deg (simulated, 2026-10-14)
        card = c_px + (px - c_px) * (64.0 / 50.0)
        r = int(np.max(np.abs(card - c_px))) + 2
        vs, us = np.mgrid[max(0, int(c_px[1] / sy) - r):min(dh, int(c_px[1] / sy) + r),
                          max(0, int(c_px[0] / sx) - r):min(dw, int(c_px[0] / sx) + r)]
        inside = np.array([cv2.pointPolygonTest(card.astype(np.float32),
                                                (float(u * sx), float(v * sy)), False) >= 0
                           for u, v in zip(us.ravel(), vs.ravel())])
        z = depth[vs, us].ravel()
        keep = (z > 0.1) & inside
        if keep.sum() < 30:
            return None
        rays = np.array([pixel_ray(u * sx, v * sy, info) for u, v in
                         zip(us.ravel()[keep], vs.ravel()[keep])])
        pts = rays * z[keep][:, None]
        world = pts @ t_wo[:3, :3].T + t_wo[:3, 3]
        centre_w = np.median(world, axis=0)
        n = np.linalg.svd(world - centre_w)[2][2]
        # once more without the points off that plane (edge pixels, noise)
        on = np.abs((world - centre_w) @ n) < 0.004
        if on.sum() >= 20:
            centre_w = np.median(world[on], axis=0)
            n = np.linalg.svd(world[on] - centre_w)[2][2]
        if n @ (t_wo[:3, 3] - centre_w) < 0:
            n = -n
        ray = t_wo[:3, :3] @ pixel_ray(c_px[0], c_px[1], info)
        t = float(n @ (centre_w - t_wo[:3, 3])) / float(n @ ray)
        centre = t_wo[:3, 3] + ray * t
        tilt = float(np.degrees(np.arccos(min(1.0, abs(n[0])))))
        self.say(f'wall marker {WALL_ID}: centre at x {centre[0] * 100:.1f} cm (wall should be '
                 f'{WALL_X * 100:.0f}: {(centre[0] - WALL_X) * 1000:+.0f} mm), y '
                 f'{centre[1] * 100:+.1f}, z {centre[2] * 100:.1f} cm; wall turned {tilt:.1f} deg '
                 f'from square'
                 + (f'; by the marker\'s size {(pnp[0] - centre[0]) * 1000:+.0f} mm deeper'
                    if pnp is not None else ''))
        node.get_logger().info(f'wall marker: centre {np.round(centre, 4).tolist()}, normal '
                               f'{np.round(n, 3).tolist()}, PnP '
                               f'{np.round(pnp, 4).tolist() if pnp is not None else None}')
        return {'centre': centre, 'normal': n, 'pnp': pnp, 'pixels': px, 'image': frames[-1],
                'tilt': tilt}

    def confirm_wall_marker(self, wall, t_wo, info):
        """Show what the camera made of the wall marker and wait for CONFIRM:
        green = the marker as detected, yellow = a 50 mm square at the measured
        spot, straight in the robot's axes (y across, z up). If the camera's
        pose is right and the marker hangs straight, they lie on each other.
        True when confirmed."""
        img = wall['image'].copy()
        k = np.array(info.k, float).reshape(3, 3)
        d = np.array(info.d, float) if len(info.d) else np.zeros(5)
        t_ow = np.linalg.inv(t_wo)
        h = WALL_MARKER / 2
        c = wall['centre']
        square = np.array([c + [0, h, h], c + [0, -h, h], c + [0, -h, -h], c + [0, h, -h]])
        cam = square @ t_ow[:3, :3].T + t_ow[:3, 3]
        proj, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d)
        proj = proj.reshape(-1, 2)
        seen = wall['pixels']
        # a 5x inset of the marker, overlays drawn at that scale
        mid = seen.mean(0)
        r = int(max(30, 1.4 * np.max(np.linalg.norm(seen - mid, axis=1))))
        x0, y0 = int(mid[0]) - r, int(mid[1]) - r
        crop = img[max(0, y0):y0 + 2 * r, max(0, x0):x0 + 2 * r].copy()
        zoom = 5
        inset = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        off = np.array([max(0, x0), max(0, y0)], float)
        for pts, colour in ((seen, (0, 255, 0)), (proj, (0, 255, 255))):
            cv2.polylines(img, [np.round(pts).astype(np.int32)], True, colour, 1)
            cv2.polylines(inset, [np.round((pts - off) * zoom).astype(np.int32)], True, colour, 2)
        cu, cv_ = np.round(seen.mean(0)).astype(int)
        cv2.drawMarker(img, (int(cu), int(cv_)), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
        ih, iw = inset.shape[:2]
        side = min(ih, iw, img.shape[0] - 70, 260)
        inset = cv2.resize(inset, (side, side))
        img[60:60 + side, img.shape[1] - side - 8:img.shape[1] - 8] = inset
        cv2.rectangle(img, (img.shape[1] - side - 8, 60), (img.shape[1] - 8, 60 + side),
                      (255, 255, 255), 1)
        px_off = float(np.mean(np.linalg.norm(proj - seen, axis=1)))
        lines = [f'wall marker {WALL_ID}: x {c[0] * 100:.1f} cm (expected {WALL_X * 100:.0f}: '
                 f'{(c[0] - WALL_X) * 1000:+.0f} mm), y {c[1] * 100:+.1f}, z {c[2] * 100:.1f} cm',
                 f'wall turned {wall["tilt"]:.1f} deg; yellow vs green {px_off:.1f} px',
                 'green = seen, yellow = robot axes. Match? CONFIRM (y) / CANCEL (x)']
        for i, text in enumerate(lines):
            cv2.putText(img, text, (8, img.shape[0] - 52 + 18 * i), FONT_CV, 0.45, (0, 0, 0), 3)
            cv2.putText(img, text, (8, img.shape[0] - 52 + 18 * i), FONT_CV, 0.45,
                        (255, 255, 255), 1)
        self.calib_view = (img, None, None, None)
        self.calib_event.clear()
        self.calib_answer = None
        self.confirming = True
        self.say(f'check the wall marker overlay (x {c[0] * 100:.1f} cm, wall turned '
                 f'{wall["tilt"]:.1f} deg), then CONFIRM or CANCEL')
        try:
            while not self.calib_event.wait(0.2):
                if self.abort.is_set():
                    return False
        finally:
            self.confirming = False
            self.calib_view = None
        return bool(self.calib_answer and self.calib_answer[0] == 'confirm')

    def wall_touches(self, wall, t_wo):
        """After the fit: each arm touches the wall marker's centre (level, the
        normal touch). The wall stops the tip, so where the model puts the
        tip then, against the wall, says the depth; sideways, look whether it
        is on the centre of the marker. Report lines."""
        out = []
        for arm in ('right', 'left'):
            if self.abort.is_set():
                break
            self.say(f'{arm} arm: touching the wall marker\'s centre to confirm...')
            self.last_contact = None
            self.check = None
            self.touch(arm, wall['centre'], t_wo[:3, 3], scene=self.frame)
            c = self.last_contact
            if c is None:
                out.append(f'{arm}: wall marker touch did not happen')
                continue
            miss = c['tip'] - wall['centre']
            n = wall['normal']
            depth = float(miss @ n)
            side = miss - depth * n
            out.append(f'{arm}: wall marker touch {"felt" if c["felt"] else "NOT felt"}, model '
                       f'tip {np.linalg.norm(side) * 1000:.1f} mm from its centre, '
                       f'{-depth * 1000:+.1f} mm into the wall (look: is it on the centre?)')
        return out

    # -- the touch check ----------------------------------------------------------

    def _touch_check(self, arm, plane, params=None, around=None):
        """Touch the surface 8 times, tool tilted 4 ways (2 each); joint angles
        at each contact. Tilted touches at the edge of reach do not solve (6
        of 8 at 25 deg, 50 cm out -- simulated 2026-10-12): each way tries
        several spots, then smaller tilts."""
        node = self.node
        n, d, pts = plane
        start = self.pose(arm, 'pre_pick_state') or node.arm_positions(arm)
        # spots on the surface nearest this arm's shoulder, spread along the
        # line to the farthest of the nearest seen points
        shoulder = node.lookup(BASE_FRAME, f'openarm_{arm}_link1')[:3, 3]
        near = pts[np.argsort(np.linalg.norm(pts - shoulder, axis=1))]
        first = near[0]                                         # nearest point seen
        second = near[np.argmax(np.linalg.norm(near[:200] - first, axis=1))]
        spots = [first + (second - first) * f for f in (0.2, 0.8, 0.5, 0.0, 1.0)]
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(up, -n)
        side /= np.linalg.norm(side)
        if around is not None:
            # around the wall marker: both arms reach it level (the spots
            # nearest the shoulder were out of reach tilted: the right arm
            # got none, 2026-10-13)
            toward = 1.0 if arm == 'left' else -1.0          # a little towards this arm
            spots = [around + side * toward * dy + up * dz
                     for dy, dz in ((0.03, 0.03), (0.03, -0.03), (-0.02, 0.0), (0.06, 0.0),
                                    (0.0, 0.05), (0.0, -0.05))] + spots
        spots = [s_ - n * float(n @ s_ - d) for s_ in spots]          # exactly on the plane
        out = []
        for about, sign in ((up, 1), (up, -1), (side, 1), (side, -1)):
            got = 0
            for tilt in TOUCH_TILTS:
                for spot in spots:
                    if got >= 2:
                        break
                    if self.abort.is_set():
                        raise KeyboardInterrupt
                    axis = rot_about(about, math.radians(sign * tilt)) @ (-n)
                    q = self._touch_once(arm, spot, axis, start, params)
                    if q is not None:
                        out.append(q)
                        got += 1
                        self.say(f'{arm} arm: touch check {len(out)}/8 ({tilt} deg)')
        node.move_joints(arm, start, f'{arm} back to pre_pick', via_home=False)
        return out

    def _touch_once(self, arm, spot, axis, start, params=None):
        node, args = self.node, self.node.args
        model = armcal.Arm(arm)
        tipv = np.array([0.0, 0.0, args.tip_offsets[arm], 1.0])
        for roll in (0, 90, 180, -90):
            quat = quat_from_matrix(tool_frame_along(axis, math.radians(roll)))
            approach = node.tcp_for_tip(spot - axis * 0.10, axis, arm)
            above = node.solve_ik(arm, approach, quat, start)
            if above is None:
                continue
            aim = spot
            if params is not None:
                # Aim with the model just fitted -- fully, depth too, at the
                # contact posture: aimed by the URDF (or sideways only) the true
                # tip, 2-5 cm off in depth, met the surface on the fast way in
                # or never reached it (3 of 8 felt, simulated 2026-10-12). The
                # angles at contact are what is measured, whatever the aim.
                q_c = above
                for _ in range(4):
                    q_c = node.solve_ik(arm, node.tcp_for_tip(aim, axis, arm), quat, q_c)
                    if q_c is None:
                        break
                    shift = (model.tcp(q_c, params) @ tipv)[:3] - (model.tcp(q_c) @ tipv)[:3]
                    new = spot - shift
                    done = np.linalg.norm(new - aim) < 0.0005
                    aim = new
                    if done:
                        break
                if q_c is None:
                    continue
                approach = node.tcp_for_tip(aim - axis * 0.10, axis, arm)
                above = node.solve_ik(arm, approach, quat, above)
                if above is None:
                    continue
            if not node.move_joints(arm, above, f'{arm} touch check', via_home=False,
                                    attempts=1):
                return None
            near = node.tcp_for_tip(aim - axis * 0.03, axis, arm)
            if node.line_move(arm, near, quat, f'{arm} in', speed=args.fast_line_speed) is None:
                return None
            q = None
            with node.fingers_touching(arm):
                ends, touched = node.guarded_line(arm, node.tcp_for_tip(aim + axis * 0.03, axis, arm),
                                                  quat, f'{arm} check contact')
                if ends is not None and touched:
                    node.settle(arm, ends, timeout=0.5)
                    q = np.array(node.arm_positions(arm))
                node.line_move(arm, near, quat, f'{arm} off', speed=args.fast_line_speed)
            node.line_move(arm, approach, quat, f'{arm} out', speed=args.fast_line_speed)
            return q
        return None

    def _touch_errors(self, arm, touches, plane, params, tip_offset):
        """rms (mm) of where the model puts the tip at each contact, off the
        surface -- after the common offset (the camera's depth bias and the
        contact's push) is taken out."""
        model = armcal.Arm(arm)
        n, d = plane[0], plane[1]
        tip = np.array([0.0, 0.0, tip_offset, 1.0])
        r = np.array([n @ (model.tcp(q, params) @ tip)[:3] - d for q in touches])
        return float(np.sqrt(np.mean((r - np.median(r)) ** 2)) * 1000)

    def _urdf_touch_errors(self, arm, touches, plane, tip_offset):
        if not touches or plane is None:
            return float('nan')
        return self._touch_errors(arm, touches, plane, None, tip_offset)
