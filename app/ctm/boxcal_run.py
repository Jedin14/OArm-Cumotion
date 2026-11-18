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
from ctm import accucal, armcal, calib_box as cb, depth_cal
from ctm.common import (BACKUP_DIR, BASE_FRAME, SAMPLES_DIR, decode_image, quat_from_matrix,
                        robot_fingerprint, tool_frame_along)
from ctm.kinematics import Arm, directions

BOX_POSTURES = 60
BOX_CANDIDATES = 260
BOX_MINUTES = 15
BOX_RMS_RAW_PX = 3.0        # a box pose worse than this, as built (before refining), not kept
BOX_RMS_PX = 0.8            # after refining the box's geometry: worse is not used
HOLDOUT = 0.2
MIN_BOX_MARKERS = 3         # a reading needs this many box markers seen (two leave the tilt loose)
DEPTH_OK = 0.004            # a depth correction is saved only if it leaves this little (m rms)
SLIP_MM = 50.0              # AUTO CAL: the box this far off its mounting (hand frame) moved on the cup


class SlipWatch:
    """AUTO CAL: did the box move on the cup? (2026-11-17)

    Each reading's box pose in the HAND frame -- the camera's box through
    the plain URDF hand pose of the reading's joints -- against the median
    of the readings on this mounting so far. Off by more than SLIP_MM, two
    readings in a row that agree with each other: the box moved, and a new
    mounting ("segment") starts with the first of them. One reading off on
    its own is a misread: it stays on the mounting (the fit drops what it
    cannot explain). Readings with fewer than MIN_BOX_MARKERS markers (their
    tilt is loose) take the current mounting and are not judged.

    Until 2026-11-16 the check compared the box in the CAMERA frame with
    the mounting's first reading: every new posture moves the box there
    (median 242 mm on the 10-30 readings), so every reading started its own
    segment and the fit could never run. In the hand frame the plain URDF's
    own error is all that moves it: on 10-30 at most 40 mm from the running
    median, nothing over 50. Smaller slips are left to the fit's drift check
    (accucal._drift flags a trend over 5 mm)."""

    def __init__(self, kin, limit_mm=SLIP_MM, min_markers=MIN_BOX_MARKERS):
        self.kin = kin
        self.limit = float(limit_mm)
        self.min_markers = int(min_markers)
        self.segment = 0
        self.ref = []               # this mounting's box positions in the hand, mm
        self.suspect = None         # (reading, position): off the mounting, not confirmed
        self.slips = []             # [(index, mm)]: the confirmed moves

    def position(self, reading):
        """The box's origin in hand_tcp (plain URDF), mm."""
        hand = self.kin.tcp(np.asarray(reading['q'], float))
        box = np.linalg.inv(hand) @ np.asarray(reading['t_wo'], float) \
            @ np.asarray(reading['cam_box'], float)
        return 1000.0 * box[:3, 3]

    def place(self, reading):
        """Set reading['segment'] (a suspect's may still change with the next
        reading). The box's shift (mm) when this reading confirms a move,
        else None."""
        reading['segment'] = self.segment
        if reading.get('cam_box') is None or int(reading.get('n') or 0) < self.min_markers:
            return None
        p = self.position(reading)
        if not self.ref:
            self.ref.append(p)
            return None
        shift = float(np.linalg.norm(p - np.median(self.ref, axis=0)))
        if shift <= self.limit:
            self.suspect = None             # a suspect before this one was a misread
            self.ref.append(p)
            return None
        if self.suspect is not None and \
                float(np.linalg.norm(p - self.suspect[1])) <= self.limit:
            first, p0 = self.suspect
            self.segment += 1
            first['segment'] = reading['segment'] = self.segment
            self.ref = [p0, p]
            self.suspect = None
            self.slips.append((first.get('index'), shift))
            return shift
        self.suspect = (reading, p)         # (an earlier suspect stays where it was)
        return None

    def finish(self):
        """At the end: a suspect never confirmed gets a mounting of its own
        (sharing one with readings it may not belong with would bend the fit;
        alone, its mount is free and it changes nothing)."""
        if self.suspect is not None:
            self.segment += 1
            self.suspect[0]['segment'] = self.segment
            self.suspect = None
        return self.segment + 1


def _keep_px(rms):
    """How far off its markers (px) a box reading may be and still be kept:
    1.5x the run's typical one, at least BOX_RMS_PX, at most BOX_RMS_RAW_PX
    (small markers seen at a slant read 1.2-1.4 px: a fixed 0.8 kept 7 of 39,
    2026-10-30)."""
    rms = [r for r in rms if np.isfinite(r)]
    if not rms:
        return BOX_RMS_PX
    return float(min(BOX_RMS_RAW_PX, max(BOX_RMS_PX, 1.5 * np.median(rms))))


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
        self.calib_last = {'when': time.monotonic(), 'n': n, 'rms': rms}   # (the live view)
        return {'tip': (world_box @ np.r_[cb.TIP_IN_BOX, 1.0])[:3], 'q': q, 'rms': rms, 'n': n,
                'cam_box': cam_box,
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

    # -- AUTO CALIBRATE: the one-button run (2026-11-16) -------------------------

    AUTOCAL_POSTURES = 90       # ~60 front + ~30 top-down, more than CAL TIP
    AUTOCAL_CANDIDATES = 400
    AUTOCAL_MINUTES = 22        # time cap; fit with what we have after this

    def autocal_box_on(self):
        """Called from the web UI when the user signals the box is on the cup."""
        self.answer('box_on')

    def calibrate_auto(self, arm='left', weigh=False):
        """AUTO CALIBRATE: the one-button run.

        Steps (exposed through self.autocal for the UI):
          weigh      -- WEIGH TOOL + APPLY WEIGHT (only when weigh=True)
          wait_box   -- wait for the user to put the box on the cup
          choose     -- choose postures where the camera sees the box
          measure    -- move to each posture and read the box
          fit        -- accucal.fit_auto: choose + fit + cross-validate the model
          done / failed / aborted

        weigh is off by default (2026-11-17): APPLY WEIGHT rewrites the URDF,
        but the motors read it only at launch, so the readings after it were
        still taken with the old gravity compensation. Weigh separately
        (WEIGH TOOL, APPLY WEIGHT, restart terminal 1), then run this. The
        model is saved with the robot as it was BEFORE any weighing here, so
        a model measured that way is not applied after the restart.
        """
        import shutil
        node = self.node
        report_id = time.strftime('%Y%m%dT%H%M%S')
        self.autocal = {'step': 'weigh', 'text': '', 'report_id': report_id}
        self.abort.clear()
        self.calibrating = True
        restart_needed = False
        weigh_text = None
        # the robot the readings are taken on: as the motors have it now
        # (a weigh below changes only the URDF on disk until a restart)
        fingerprint = robot_fingerprint(arm)
        self.autocal_readings = []          # filled as the run goes (kept on abort)

        try:
            # -- 1. weigh the tool (arm empty) ------------------------------------
            if weigh:
                self.say('AUTO CAL: weighing the tool (arm empty)...')
                ok, weigh_text, restart_needed = self.weigh_and_apply(arm)
                if self.abort.is_set():
                    raise KeyboardInterrupt
                if ok:
                    self.say(f'AUTO CAL: {weigh_text}')
                else:
                    self.say(f'AUTO CAL: weigh skipped - {weigh_text}')
                    weigh_text = f'skipped: {weigh_text}'
            else:
                weigh_text = 'skipped (weigh=False)'

            # -- 2. return to pre_pick so the box can be mounted squarely --------
            self.say('AUTO CAL: returning the arm to pre_pick...')
            self._back_to_pre_pick(arm)

            # -- 3. wait for the box on the cup -----------------------------------
            self.autocal['step'] = 'wait_box'
            self.autocal['text'] = 'put the ArUco box on the cup, then press "Box is on the cup"'
            self.say('AUTO CAL: arm at pre_pick. Put the ArUco calibration box on the cup '
                     '(cup on the cross of face B), then press "Box is on the cup" in the UI.')
            self.calib_event.clear()
            self.calib_answer = None
            # wait until the user signals or aborts
            while True:
                self.calib_event.wait(timeout=2.0)
                self.calib_event.clear()
                if self.abort.is_set():
                    raise KeyboardInterrupt
                ans = self.calib_answer
                self.calib_answer = None
                if ans is not None and ans[0] == 'box_on':
                    break

            # -- 3. capture the frame and find the box ----------------------------
            self.autocal['step'] = 'choose'
            self.autocal['text'] = 'finding the box...'
            if not self.ensure_pick_mode():
                self.say('AUTO CAL: could not enter pick mode', True)
                self.autocal['step'] = 'failed'
                return
            # attach a nominal held object so the planner leaves room for it --
            # BEFORE the capture rebuilds the map, as CAL TIP does, so the map
            # leaves the box out (2026-11-17: it was attached after, and the
            # box held at pre_pick could go into the map as an obstacle)
            tip_off = float(node.args.tip_offsets[arm])
            nominal = np.eye(4)
            nominal[:3, 3] = [0.0, 0.0, tip_off - cb.TIP_IN_BOX[2]]
            node.attach_held(arm, None, None,
                             exact=(np.eye(3), nominal[:3, 3],
                                    np.array([cb.SIZE[0], cb.SIZE[0], cb.SIZE[2]])))
            self.capture()
            if self.frame is None:
                self.say('AUTO CAL: no camera frame', True)
                self.autocal['step'] = 'failed'
                return
            info, optical = self.frame[2], self.frame[3]
            t_wo = node.lookup(BASE_FRAME, optical)
            self.calib_live = True

            # read the box where the arm is; if not visible, find a posture that shows it
            first = self.measure_box(arm, t_wo, info, frames=10)
            if first is None:
                first = self._show_box(arm, t_wo, info, nominal)
            if first is None or first['box_in_tcp'] is None:
                self.say('AUTO CAL: the camera cannot see the box - '
                         'is it held with face B (the cross) on the cup?', True)
                self.autocal['step'] = 'failed'
                return
            box_in_tcp = first['box_in_tcp']
            node.attach_held(arm, None, None,
                             exact=(box_in_tcp[:3, :3], box_in_tcp[:3, 3],
                                    np.array(cb.SIZE)))
            self.say(f'AUTO CAL: box seen ({first["n"]} markers, {first["rms"]:.2f} px); '
                     f'starting the run across the workspace...')

            # -- 4. measure postures (may span multiple box mountings/segments) ----
            readings = self._autocal_postures(arm, t_wo, info, box_in_tcp, fingerprint)

        except KeyboardInterrupt:
            # the readings so far are kept (a 22 min run was lost on any stop)
            kept = self._save_autocal_readings(arm, self.autocal_readings, partial=True)
            self.say('AUTO CAL: aborted; no model saved'
                     + (f' - the {len(self.autocal_readings)} readings so far are in {kept}'
                        if kept else ''), True)
            self.autocal['step'] = 'aborted'
            self.autocal['text'] = 'aborted'
            return
        finally:
            self.calibrating = False
            self.calib_live = False
            self.calib_view = None
            node.detach_held(arm)
            self._back_to_pre_pick(arm)

        if not readings:
            self.say('AUTO CAL: no box readings collected', True)
            self.autocal['step'] = 'failed'
            self.autocal['text'] = 'no readings'
            return

        # -- 5. fit the model (pure computation, no motion) ----------------------
        # the readings on disk first: a fit that fails or is refused must not
        # lose the run (2026-11-17: they were never saved)
        kept = self._save_autocal_readings(arm, readings)
        self.autocal['step'] = 'fit'
        self.autocal['text'] = f'fitting the model from {len(readings)} readings...'
        self.say(f'AUTO CAL: fitting the model from {len(readings)} readings'
                 + (f' (kept in {kept})' if kept else '') + '...')

        tip_offset = float(node.args.tip_offsets[arm])
        # the model in the file scored IN FULL, as it is applied (2026-11-17:
        # only base, offset and comply were passed -- the 10-30 model then
        # scored 50.3 mm on its own readings instead of 11.4, so the gate and
        # the "was X mm" were wrong)
        current_raw = (armcal.load_model_raw().get(arm) or {})
        models = getattr(self, 'arm_models', None)
        in_use = bool(models is not None and models.has(arm))

        try:
            result = accucal.fit_auto(
                arm, readings, tip_offset,
                current=current_raw or None,
                log=node.get_logger().info,
            )
        except ValueError as exc:
            self.say(f'AUTO CAL: fit failed - {exc}', True)
            self.autocal.update(step='failed', text=str(exc))
            return
        except Exception as exc:                     # noqa: BLE001
            self.say(f'AUTO CAL: unexpected fit error - {exc}', True)
            self.autocal.update(step='failed', text=str(exc))
            return

        variant = result['variant']
        val = result['val']
        baseline = result.get('baseline', {})
        # What a new model has to beat: what the picks use NOW (2026-11-17) --
        # the model in the file if it is applied, else the plain URDF (a model
        # the fingerprint check refuses, or --arm-model off, is not used: a
        # stale one that happens to score well must not block a new one).
        file_mm = (baseline.get('current') or {}).get('mean_mm')
        if in_use:
            cur_mm, cur_name = file_mm, 'the current model'
        else:
            cur_mm = (baseline.get('urdf') or {}).get('mean_mm')
            cur_name = 'the plain URDF, in use now'
            if file_mm is not None:
                node.get_logger().info(
                    f'AUTO CAL: the model in the file scores {file_mm:.1f} mm on these readings '
                    f'(not applied: {(getattr(models, "refused", None) or {}).get(arm)})')
        new_mm = val.get('mean_mm')

        # -- 6. decide whether to save -------------------------------------------
        saved = False
        why = None
        if new_mm is None:
            why = 'cross-validation produced no scores'
        elif variant == 'urdf' and not in_use:
            why = (f'the plain URDF, in use now, is as good as any model on held-out postures '
                   f'({new_mm:.1f} mm): nothing to correct')
        elif cur_mm is not None and new_mm >= cur_mm:
            why = (f'the new model ({variant}, {new_mm:.1f} mm) is not better than {cur_name} '
                   f'({cur_mm:.1f} mm) on held-out postures')
        else:
            # save: keep the existing boards (gripper pieces) from the current model file
            os.makedirs(BACKUP_DIR, exist_ok=True)
            stamp = time.strftime('%Y-%m-%dT%H-%M-%S')
            if os.path.exists(armcal.MODEL_FILE):
                shutil.copy(armcal.MODEL_FILE, os.path.join(
                    BACKUP_DIR, os.path.basename(armcal.MODEL_FILE).replace(
                        '.yaml', f'.before_autocal_{stamp}.yaml'),
                ))
            # keep existing boards (the gripper-piece poses, from CAL ARMS)
            old_raw = armcal.load_model_raw()
            old_boards = [np.asarray(b, float)
                          for b in (old_raw.get(arm) or {}).get('pieces', [])]
            params = accucal.saveable(result)
            stats = dict(result['stats'])
            # with the robot the readings were taken on (before any weighing here)
            armcal.save_model({arm: (params, old_boards, stats)}, fingerprints={arm: fingerprint})
            # (re-read in the same --arm-model mode)
            # (a lambda: its own call line -- rclpy pins one severity to each, and the
            # start-up ArmModels logged from armcal's line at warn, 2026-11-17 review)
            self.arm_models = armcal.ArmModels(getattr(models, 'mode', None),
                                               log=lambda text: node.get_logger().info(text))
            saved = True
            why = f'mean {new_mm:.1f} mm on held-out postures'
            if cur_mm is not None:
                why += f', was {cur_mm:.1f} mm ({cur_name})'
            if self.arm_models.has(arm):
                use = 'Used from the next pick.'
            elif restart_needed:
                use = ('NOT applied: it was measured before the new tool weight takes effect - '
                       'restart terminal 1, then run AUTO CAL again without weighing.')
            else:
                use = f'NOT applied: {self.arm_models.status(arm)}'
            self.say(f'AUTO CAL: saved ({variant}, {why}). {use}')

        if not saved:
            self.say(f'AUTO CAL: NOT saved - {why}', True)

        # the error-map points for the UI
        per_point = val.get('per_point') or []
        points = []
        for p in per_point:
            meas = p.get('tip_meas') or []
            err = p.get('err_mm') or []
            if len(meas) >= 3 and len(err) >= 3:
                points.append({
                    'x': meas[0], 'y': meas[1], 'z': meas[2],
                    'dx': err[0], 'dy': err[1], 'dz': err[2],
                    'e': float(np.linalg.norm(err)),
                    'kind': p.get('kind', 'front'),
                })

        self.autocal.update(
            step='done',
            text='saved' if saved else f'not saved: {why}',
            measured=len(readings),
            total=len(readings),
            eta_s=0,
            result={
                'variant': variant,
                'cv': {k: {'mean_mm': v.get('mean_mm'), 'p90_mm': v.get('p90_mm')}
                       for k, v in result.get('cv', {}).items()},
                'val': [val.get('mean_mm'), val.get('p90_mm'), val.get('max_mm')],
                'baseline': baseline,
                'compared_with': cur_name,
                'readings_file': kept,
                'saved': saved,
                'why': why,
                'drift': (result.get('drift') or {}).get('text'),
                'depth': None,
                'weigh': weigh_text,
                'restart_needed': restart_needed,
            },
            points=points,
        )

    def _autocal_postures(self, arm, t_wo, info, box_in_tcp, fingerprint=None):
        """Like _box_postures but for the auto-cal run: more postures, tracks
        the autocal state dict (measured/total/eta_s/segments), and handles
        box slippage by starting a new segment when the box moved on the cup
        (SlipWatch). Returns a list of reading dicts for accucal.fit_auto --
        the same list as self.autocal_readings, filled as it goes (an abort
        keeps what was read)."""
        node = self.node
        start = self.pose(arm, 'pre_pick_state') or node.arm_positions(arm)
        rng = np.random.default_rng(7)
        cands, where, kinds = [], [], []
        self.autocal['text'] = 'choosing postures where the camera sees the box...'
        for tip, axis, roll in self._front_poses(arm, t_wo, info, rng):
            if self.abort.is_set():
                raise KeyboardInterrupt
            if len(cands) >= self.AUTOCAL_CANDIDATES:
                break
            quat = quat_from_matrix(tool_frame_along(axis, math.radians(roll)))
            tcp = node.tcp_for_tip(tip, axis, arm)
            seeds = node.fast_ik(arm, tcp, quat, start, want=1)
            sol = node.solve_ik(arm, tcp, quat, seeds[0]) if seeds else None
            if sol is not None and self.box_visible(arm, sol, box_in_tcp) >= 4:
                cands.append(np.array(sol))
                where.append(np.r_[np.asarray(tip) / 0.05, np.asarray(axis) / 0.35])
                kinds.append('top' if np.asarray(axis)[2] < -0.6 else 'front')

        if len(cands) < 10:
            self.say(f'AUTO CAL: only {len(cands)} postures show the box - not enough', True)
            return []

        feats = np.hstack([np.array(cands) / 0.5, np.array(where)])
        chosen = []
        for kind in ('front', 'top'):
            idx = [i for i, k in enumerate(kinds) if k == kind]
            quota = (self.AUTOCAL_POSTURES - len(chosen)) if kind == 'top' \
                else self.AUTOCAL_POSTURES // 2
            if not idx or quota <= 0:
                continue
            pick = [idx[0]]
            dist = np.linalg.norm(feats[idx] - feats[idx[0]], axis=1)
            while len(pick) < min(quota, len(idx)):
                nxt = int(np.argmax(dist))
                pick.append(idx[nxt])
                dist = np.minimum(dist, np.linalg.norm(feats[idx] - feats[idx[nxt]], axis=1))
            chosen += pick

        self.say(f'AUTO CAL: {sum(kinds[i] == "front" for i in chosen)} level and '
                 f'{sum(kinds[i] == "top" for i in chosen)} top-down postures '
                 f'(of {len(cands)} that show the box)')

        order, here, left = [], np.array(start), set(chosen)
        while left:
            nxt = min(left, key=lambda i: np.linalg.norm(cands[i] - here))
            order.append(nxt)
            left.discard(nxt)
            here = cands[nxt]

        self.autocal['step'] = 'measure'
        self.autocal['total'] = len(order)
        self.autocal['measured'] = 0
        self.autocal['segments'] = [0]      # list of segment indices seen so far
        readings = getattr(self, 'autocal_readings', None)
        if readings is None:
            readings = self.autocal_readings = []
        failed = 0
        watch = SlipWatch(Arm(arm))         # the plain URDF, as the motors have it
        tip_off_r = float(node.args.tip_offsets[arm])
        t_start = time.monotonic()
        t_end = t_start + 60.0 * self.AUTOCAL_MINUTES
        for n, i in enumerate(order, start=1):
            if self.abort.is_set():
                raise KeyboardInterrupt
            if time.monotonic() > t_end:
                self.say(f'AUTO CAL: {self.AUTOCAL_MINUTES} min done - '
                         f'fitting with {len(readings)}')
                break
            done = len(readings)
            elapsed = time.monotonic() - t_start
            eta_s = int(elapsed / done * (len(order) - done)) if done else 0
            self.autocal.update(
                text=f'posture {n}/{len(order)} ({done} measured)',
                measured=done,
                eta_s=eta_s,
            )
            before = node.arm_positions(arm)
            if not node.move_joints(arm, cands[i].tolist(), f'autocal {n}',
                                    via_home=False, attempts=1):
                failed += 1
                if failed == 3:
                    self.say('AUTO CAL: moves keep failing; waiting for the planner...')
                    time.sleep(8.0)
                    node.require_move_group()
                elif failed >= 6:
                    self.say(f'AUTO CAL: the planner is not answering - '
                             f'stopping with {len(readings)}', True)
                    break
                continue
            failed = 0
            node.settle(arm, cands[i].tolist(), timeout=3.0)
            time.sleep(0.2)
            m = self.measure_box(arm, t_wo, info)
            if m is None:
                continue
            reading = {
                'q': m['q'].tolist(),
                'seen': m['seen'],
                'k': m['k'],
                'd': m['d'],
                't_wo': m['t_wo'],
                'cam_box': m['cam_box'].tolist()
                    if isinstance(m['cam_box'], np.ndarray) else m['cam_box'],
                'rms': m['rms'],
                'n': m['n'],
                'kind': kinds[i],
                'segment': 0,               # (SlipWatch sets it)
                'index': n,
                'dirs': (directions(before, m['q']) if before is not None
                         else np.zeros(7)).tolist(),
                # for a replay offline: which arm, tool and robot it was read on
                'arm': arm, 'tip_offset': tip_off_r, 'fingerprint': fingerprint,
            }
            # did the box move on the cup? (2026-11-17: in the hand frame; the
            # camera-frame check made every reading its own segment)
            # (a side check: whatever goes wrong in it, the run goes on)
            try:
                moved = watch.place(reading)
            except Exception as exc:                 # noqa: BLE001
                moved = None
                reading['segment'] = watch.segment
                node.get_logger().warn(f'AUTO CAL: slip check failed ({exc})')
            readings.append(reading)
            if moved is not None:
                self.say(f'AUTO CAL: the box moved on the cup ({moved:.0f} mm in the hand, two '
                         f'readings in a row) - a new mounting from posture '
                         f'{watch.slips[-1][0]}')
                self.autocal['segments'] = list(range(watch.segment + 1))
            # live error-map point: URDF tip minus camera-measured tip
            try:
                from ctm.kinematics import Arm as _Arm
                _kin = getattr(self, '_autocal_kin', None)
                if _kin is None:
                    self._autocal_kin = _Arm(arm)
                    _kin = self._autocal_kin
                tip_off_v = float(node.args.tip_offsets[arm])
                hand = _kin.tcp(np.asarray(m['q'], float))
                tip_urdf = (hand @ np.array([0.0, 0.0, tip_off_v, 1.0]))[:3]
                world_box = np.asarray(m['t_wo'], float) @ np.asarray(m['cam_box'], float)
                from ctm import calib_box as _cb
                tip_meas = (world_box @ np.r_[_cb.TIP_IN_BOX, 1.0])[:3]
                err = (tip_urdf - tip_meas) * 1000
                pts = list(self.autocal.get('points') or [])
                pts.append({
                    'x': float(tip_meas[0]), 'y': float(tip_meas[1]),
                    'z': float(tip_meas[2]),
                    'dx': float(err[0]), 'dy': float(err[1]), 'dz': float(err[2]),
                    'e': float(np.linalg.norm(err)),
                    'kind': kinds[i],
                })
                self.autocal['points'] = pts
            except Exception:                        # noqa: BLE001
                pass
        self.autocal['segments'] = list(range(watch.finish()))
        self.autocal['measured'] = len(readings)
        return readings

    def _save_autocal_readings(self, arm, readings, partial=False):
        """The run's readings to calibration/armcal_samples/<arm>_autocal_<stamp>.npy
        (as CAL TIP keeps its own), so a fit that fails or is refused -- or a
        stop -- does not lose them (2026-11-17). The path, or None (nothing
        to keep, or it could not be written: the run goes on regardless)."""
        if not readings:
            return None
        stamp = time.strftime('%Y-%m-%dT%H-%M-%S')
        path = os.path.join(SAMPLES_DIR, f'{arm}_autocal_{stamp}'
                            + ('_partial' if partial else '') + '.npy')
        try:
            os.makedirs(SAMPLES_DIR, exist_ok=True)
            np.save(path, np.array(list(readings), dtype=object), allow_pickle=True)
        except Exception as exc:                     # noqa: BLE001
            self.node.get_logger().error(f'AUTO CAL: readings not saved ({exc})')
            return None
        return path

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
        self.calib_live = True                           # the GUI: live camera, markers outlined
        anchors = None
        self.box_depth = []                              # depth vs markers, per posture
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
            self.calib_live = False
            self.calib_view = None
            self._back_to_pre_pick(arm)
            node.detach_held(arm)
        if self.box_depth:
            self._fit_depth_from_box()
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
        cands, where, kinds = [], [], []
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
                kinds.append('top' if np.asarray(axis)[2] < -0.6 else 'front')
        if len(cands) < 10:
            self.say(f'CAL TIP: only {len(cands)} postures show the box - not enough', True)
            return None
        # the most spread out of each kind -- as many top-down postures as level
        # ones where the camera sees the box: top picks ran on a model fitted to
        # front postures only (2026-10-29) -- visited nearest-first (as CAL ARMS)
        feats = np.hstack([np.array(cands) / 0.5, np.array(where)])
        chosen = []
        for kind in ('front', 'top'):
            idx = [i for i, k in enumerate(kinds) if k == kind]
            quota = BOX_POSTURES - len(chosen) if kind == 'top' else BOX_POSTURES // 2
            if not idx or quota <= 0:
                continue
            pick = [idx[0]]
            dist = np.linalg.norm(feats[idx] - feats[idx[0]], axis=1)
            while len(pick) < min(quota, len(idx)):
                nxt = int(np.argmax(dist))
                pick.append(idx[nxt])
                dist = np.minimum(dist, np.linalg.norm(feats[idx] - feats[idx[nxt]], axis=1))
            chosen += pick
        self.say(f'CAL TIP: {sum(kinds[i] == "front" for i in chosen)} level and '
                 f'{sum(kinds[i] == "top" for i in chosen)} top-down postures '
                 f'(of {len(cands)} that show the box)')
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
                # the depth camera on the same marker faces: its error against
                # the markers' own geometry (the depth check, no board)
                # (a side measurement: whatever goes wrong here, the run goes on)
                try:
                    depth, _last = node.averaged_depth(4, timeout=2.0, raw=True)
                    got = self._box_depth_samples(m['cam_box'], m['seen'], depth, info) \
                        if depth is not None else None
                    if got is not None:
                        self.box_depth.append(got)
                except Exception as exc:                 # noqa: BLE001
                    node.get_logger().warn(f'CAL TIP: depth on the box not read ({exc})')
                anchors.append({'arm': arm, 'q': m['q'].tolist(), 'point': m['tip'].tolist(),
                                'kind': kinds[i],
                                'tip_offset': float(node.args.tip_offsets[arm]),
                                'rms': m['rms'], 'n': m['n'], 'seen': m['seen'],
                                't_wo': m['t_wo'], 'k': m['k'], 'd': m['d'],
                                'dirs': (directions(before, m['q']) if before is not None
                                         else np.zeros(7)).tolist()})
        return anchors

    @staticmethod
    def _box_depth_samples(cam_box, seen, depth, info):
        """(x, y, z_measured, z_true) on each face of the box with markers seen:
        the face's plane from the box's pose (the markers), the depth inside
        the seen markers' outline. None if too few readings."""
        cam_box = np.asarray(cam_box, float)
        out = []
        by_face = {}
        for i, c in seen.items():
            if int(i) in cb.MARKERS:
                by_face.setdefault(cb.MARKERS[int(i)][0], []).append(np.asarray(c, float))
        for face, corners in by_face.items():
            n, _r, _d, centre = cb.FACES[face]
            plane = np.eye(4)
            plane[:3, 2] = cam_box[:3, :3] @ n
            plane[:3, 3] = (cam_box @ np.r_[centre, 1.0])[:3]
            got = depth_cal.samples_from(plane, depth, info, np.vstack(corners), shrink=0.1)
            if got is not None and len(got):
                out.append(got)
        return np.vstack(out) if out else None

    def _fit_depth_from_box(self):
        """The depth correction from the box's faces (all postures): saved only
        if it makes the depth error clearly smaller."""
        samples = [s for s in self.box_depth if s is not None and len(s)]
        points = int(sum(len(s) for s in samples))
        if len(samples) < 6 or points < 2000:
            self.say(f'CAL TIP: depth - only {len(samples)} postures with depth on the box '
                     f'({points} points): depth correction not changed')
            return
        # each posture's points round its own median (strip edges, the board
        # behind): out beyond 3 MADs
        clean = []
        for smp in samples:
            r = smp[:, 3] - smp[:, 2]
            mad = float(np.median(np.abs(r - np.median(r)))) + 1e-4
            clean.append(smp[np.abs(r - np.median(r)) < 3.0 * 1.4826 * mad])
        samples = [c for c in clean if len(c) > 50]
        coef, before, after = depth_cal.fit(samples)
        # a correction that still leaves the depth this far off from the markers
        # explains nothing reliably (11.0 -> 8.8 mm with a 3 % scale was saved and
        # had to be taken back, 2026-10-30)
        if after > DEPTH_OK or after >= 0.85 * before:
            self.say(f'CAL TIP: depth NOT changed: {1000 * before:.1f} mm rms against the '
                     f'markers, a correction leaves {1000 * after:.1f} (must be under '
                     f'{1000 * DEPTH_OK:.0f} mm and clearly better)')
            return
        depth_cal.save(coef, before, after, len(samples), points)
        self.say(f'CAL TIP: depth corrected: {1000 * before:.1f} -> {1000 * after:.1f} mm rms '
                 f'({len(samples)} postures, {points} points on the box)')

    def _refine_box(self, anchors):
        """The box's real geometry from all the readings at once (calib_box.
        refine_geometry), then each reading's tip again with it; readings
        that still do not fit (> BOX_RMS_PX) are dropped."""
        with_px = [a for a in anchors if a.get('seen') and a.get('k') is not None]
        if len(with_px) < 5:
            # nothing to refine with (readings saved before the pixels were):
            # judged as measured
            return [a for a in anchors if a['rms'] <= _keep_px([a['rms'] for a in anchors])
                    and a.get('n', MIN_BOX_MARKERS) >= MIN_BOX_MARKERS]
        k, d = with_px[0]['k'], with_px[0]['d']
        views = [{int(i): np.asarray(c) for i, c in a['seen'].items()} for a in with_px]
        geom, before, after = cb.refine_geometry(views, k, d, log=self.node.get_logger().info)
        # at least 3 markers: two small ones on one face fit their pixels as
        # well (1.8 px) but leave the tilt loose -- one such reading put the
        # tip 76 mm off the arm model, another 30 mm (2026-10-30)
        poses = [cb.box_pose_with(v, k, d, geom) for v in views]
        poses = [g if g is not None and g[2] >= MIN_BOX_MARKERS else None for g in poses]
        limit = _keep_px([g[1] for g in poses if g is not None])
        out = []
        for a, got in zip(with_px, poses):
            if got is None or got[1] > limit:
                continue
            cam_box, rms, _n = got
            world_box = np.asarray(a['t_wo']) @ cam_box
            out.append(dict(a, point=(world_box @ np.r_[cb.TIP_IN_BOX, 1.0])[:3].tolist(),
                            rms=rms))
        if geom:
            self.say(f'CAL TIP: the box as built: its markers fit {after:.2f} px (as drawn: '
                     f'{before:.2f}); {len(out)} of {len(anchors)} readings kept (up to '
                     f'{limit:.2f} px)')
        else:
            self.say(f'CAL TIP: the box as drawn (refining its faces did not fit better, or '
                     f'its cross side was seen too rarely); {len(out)} of {len(anchors)} '
                     f'readings kept (up to {limit:.2f} px)')
        return out

    def _fit_box(self, arm, anchors):
        """The arm model fitted to the measured tips (+ the CAL ARMS recordings
        for the arm), 20 % held out first for an honest error in mm."""
        node = self.node
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        stamp = time.strftime('%Y-%m-%dT%H-%M-%S')
        np.save(os.path.join(SAMPLES_DIR, f'{arm}_box_{stamp}.npy'),
                np.array(anchors, dtype=object), allow_pickle=True)
        anchors = self._refine_box(anchors)
        if len(anchors) < 10:
            self.say(f'CAL TIP: only {len(anchors)} good box readings - not enough to fit', True)
            return
        files = sorted(glob.glob(os.path.join(SAMPLES_DIR, f'{arm}_2*.npz')))
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
        old_hold = tip_err(self.arm_models.params[arm], held) * 1000 \
            if self.arm_models.has(arm) else None
        by_kind = {k: e_hold[[j for j, h in enumerate(held) if h.get('kind') == k]]
                   for k in ('front', 'top')}
        kinds_txt = ', '.join(f'{k} {np.mean(v):.1f} mm ({len(v)})' for k, v in by_kind.items()
                              if len(v))
        if old_hold is not None and np.mean(e_hold) >= np.mean(old_hold):
            self.detail = (f'CAL TIP {arm}: NOT saved - on postures not used in the fit the new '
                           f'model is {np.mean(e_hold):.1f} mm off, the current one '
                           f'{np.mean(old_hold):.1f} mm' + (f' ({kinds_txt})' if kinds_txt else ''))
            self.say(f'{self.detail}. Take the box off the cup.', True)
            return
        p, b, st = armcal.fit(arm, samples, z['t_wo'], info, dirs=dirs, anchors=anchors, **kw)
        e_all = tip_err(p, anchors) * 1000
        st.update(model=variant, dropped=dropped, holdout_px=float('nan'),
                  box_tips=len(anchors), box_holdout_mm=float(np.mean(e_hold)),
                  box_holdout_p90_mm=float(np.percentile(e_hold, 90)),
                  box_fit_mm=float(np.mean(e_all)))
        import shutil
        if os.path.exists(armcal.MODEL_FILE):            # the model before, kept in old_version/
            os.makedirs(BACKUP_DIR, exist_ok=True)
            shutil.copy(armcal.MODEL_FILE, os.path.join(
                BACKUP_DIR, os.path.basename(armcal.MODEL_FILE).replace(
                    '.yaml', f'.before_boxcal_{stamp}.yaml')))
        armcal.save_model({arm: (p, b, st)})
        # (re-read in the same --arm-model mode, 2026-11-17)
        # (a lambda: its own call line, as at AUTO CAL above)
        self.arm_models = armcal.ArmModels(getattr(self.arm_models, 'mode', None),
                                           log=lambda text: node.get_logger().info(text))
        self.detail = (f'CAL TIP {arm}: tip error on postures NOT used in the fit: mean '
                       f'{np.mean(e_hold):.1f} mm, 90 % under {np.percentile(e_hold, 90):.1f} mm, '
                       f'max {np.max(e_hold):.1f}' + (f' - {kinds_txt}' if kinds_txt else '')
                       + (f' (the old model on the same postures: {np.mean(old_hold):.1f} mm)'
                          if old_hold is not None else ''))
        self.say(f'CAL TIP: saved. {self.detail}. Take the box off the cup.')
