"""CALIBRATE / APPLY / OVERLAY: the camera pose from the arms seen in
depth, written into the URDF, cam_org.txt and the live robot."""

import datetime
import math
import os
import re
import subprocess
import time

import numpy as np
import yaml
from moveit_msgs.msg import MoveItErrorCodes

import robot_camera_calibration as rcc
from ctm.common import (BASE_FRAME, CALIBRATION_FILE, CAM_ORG, SCREW_FRAME, URDF_XACRO,
                        WS, click_shift, decode_image, pixel_ray, plain,
                        quat_from_matrix, rpy_from_matrix, tool_frame_along)


def roll_across_view(axis, ray):
    """Roll that spreads the fingers across the view, not one behind the other.

    The fingers open along the tool's +Y. Seen end-on, one hides the other and
    perspective pulls the visible one well off the midpoint (~18 px measured
    in simulation at 0.3 m); side by side, the pair's centre is the midpoint.
    """
    x0, y0, z = tool_frame_along(axis, 0.0).T
    want = np.cross(z, ray)
    if np.linalg.norm(want) < 1e-6:
        return 0.0
    want /= np.linalg.norm(want)
    return math.atan2(-(want @ x0), want @ y0)


class CameraCalibrationMixin:
    """Part of click_to_move.App; uses self.node, self.frame, self.say..."""

    def show_overlay(self):
        """Draw the robot model, posed by the encoders, onto a fresh frame:
        green where the camera's depth agrees with it (within 15 mm), red
        where it does not. If the calibration is right the model sits on
        the arms in the image; this is the check, before and after."""
        node = self.node
        self.say('loading the robot model (slow only the first time)...')
        surface = node.robot_surface()
        depth, last = node.averaged_depth(6)
        color, _d, info = node.latest()
        if depth is None or color is None or info is None:
            self.say('no camera frames', True)
            return
        optical = last.header.frame_id or info.header.frame_id
        t_wo = node.lookup(BASE_FRAME, optical)
        if t_wo is None:
            self.say(f'no tf for {optical}', True)
            return
        self.frame = (decode_image(color), depth, info, optical)
        self.pick = None
        mp, mn = surface.posed(node.link_poses())
        px, agree, seen = rcc.depth_agreement(depth, mp, mn, t_wo, info)
        if not seen.any():
            self.say('neither arm is in view', True)
            return
        self.overlay = (px[seen][::2], agree[seen][::2])
        self.say(f'model vs depth: {100 * agree[seen].mean():.0f}% of the visible arm agrees '
                 f'within 15 mm (green). Red = camera and robot disagree. o = hide')

    # -- camera calibration ----------------------------------------------------

    def calibration_targets(self, t_world_optical):
        """Candidate fingertip positions: a 3 x 4 grid over the image, each
        cell at both depths (24), at least 7 cm in front of whatever the
        captured depth sees there.

        calibrate() stops once it has enough points, so a spot the arm cannot
        reach costs a candidate rather than a point -- with 12 candidates a
        run on 2026-10-01 lost half of them and was left with the bare
        minimum of 6. Per arm, every cell is tried once before any cell is
        tried at its other depth; the right arm's spots come first, then the
        left's, so the other arm is never parked in view.
        """
        bgr, depth, info, _frame = self.frame
        h, w = depth.shape
        near, far = self.node.args.calib_depths
        passes = ([], [])
        for row, fv in enumerate((0.25, 0.50, 0.75)):
            for col, fu in enumerate((0.15, 0.38, 0.62, 0.85)):
                order = (near, far) if (row + col) % 2 == 0 else (far, near)
                for targets, wanted in zip(passes, order):
                    u, v = fu * bgr.shape[1], fv * bgr.shape[0]
                    du, dv = int(u * w / bgr.shape[1]), int(v * h / bgr.shape[0])
                    patch = depth[max(0, dv - 5):dv + 6, max(0, du - 5):du + 6]
                    seen = patch[patch > 0.1]
                    if seen.size:
                        wanted = min(wanted, float(np.median(seen)) - 0.07)
                    if wanted < 0.22:
                        continue
                    p = t_world_optical[:3, :3] @ (pixel_ray(u, v, info) * wanted) \
                        + t_world_optical[:3, 3]
                    targets.append(('left' if fu < 0.5 else 'right', p))
        every = passes[0] + passes[1]
        return ([t for t in every if t[0] == 'right'] +
                [t for t in every if t[0] == 'left'])

    def calibration_axes(self, arm, tip, camera):
        """Tool axes to try at a calibration spot, best first.

        The fingertip has to be *seen*: the hand sits 95 mm behind it along
        the axis, so an axis within 60 deg of the camera ray puts the hand in
        front of the tip. Top-down first, then across the image, then the
        diagonals between -- whichever the arm can reach.
        """
        ray = tip - camera
        ray /= np.linalg.norm(ray)
        inward = np.array([0.0, 1.0 if arm == 'right' else -1.0, 0.0])
        down = np.array([0.0, 0.0, -1.0])
        back = -np.array([ray[0], ray[1], 0.0]) / max(1e-9, math.hypot(ray[0], ray[1]))
        preferred = [down, inward, down + inward, down + back, inward + back, back,
                     -inward, down - inward, np.array([0.0, 0.0, 1.0]), inward - down]
        axes = []
        for axis in preferred:
            axis = axis / np.linalg.norm(axis)
            if math.degrees(math.acos(np.clip(axis @ ray, -1.0, 1.0))) >= 60.0:
                axes.append(axis)
        return axes

    def calibration_pose(self, arm, tip, start, camera):
        """(joints, tool axis) for a pose that shows the fingertip to the
        camera, or (None, None)."""
        node = self.node
        seeds = [start] + ([node.retract[arm]] if arm in node.retract else [])
        ray = (tip - camera) / np.linalg.norm(tip - camera)
        for axis in self.calibration_axes(arm, tip, camera):
            best = roll_across_view(axis, ray)
            for roll in (best, best + math.pi):
                quat = quat_from_matrix(tool_frame_along(axis, roll))
                tcp = node.tcp_for_tip(tip, axis)
                for seed in seeds:
                    joints = node.solve_ik(arm, tcp, quat, seed)
                    if joints is not None:
                        return joints, axis
        return None, None

    def observe_arm(self, i, total, arm, info, background, t_wo):
        """One pose's evidence for the 3D fit: the arm as the depth camera sees
        it, and the robot model where the encoders say it is."""
        node = self.node
        depth, _last = node.averaged_depth(6)
        color = node.fresh_color()
        if depth is None or color is None:
            return None
        surface = node.robot_surface()
        mp, mn = surface.posed(node.link_poses((arm,)))
        obs, mask = rcc.arm_points(depth, background, info)
        obs = rcc.near_model(obs, t_wo, mp)
        px, agree, seen = rcc.depth_agreement(depth, mp, mn, t_wo, info)
        self.overlay = (px[seen][::3], agree[seen][::3])
        self.calib_view = (decode_image(color), None, None, mask)
        if len(obs) < 300:
            self.say(f'{i}/{total}: only {len(obs)} arm points seen; skipping')
            time.sleep(0.5)
            return None
        self.last_pose_view = (decode_image(color), depth, mp, mn)
        self.say(f'{i}/{total}: {len(obs)} arm points; model/depth agreement '
                 f'{100 * agree[seen].mean() if seen.any() else 0:.0f}% before calibrating')
        time.sleep(0.5)
        return obs, mp, mn, depth

    def calibrate(self):
        node, args = self.node, self.node.args
        self.say('loading the robot model (slow only the first time)...')
        node.robot_surface()
        self.say('calibration: taking a clean frame and map first...')
        self.capture()                               # background = the scene without the arm
        if self.frame is None:
            return
        background, info, optical = self.frame[1], self.frame[2], self.frame[3]
        t_wo = node.lookup(BASE_FRAME, optical)
        t_os = node.lookup(optical, SCREW_FRAME)
        if t_wo is None or t_os is None:
            self.say(f'no tf for {optical} / {SCREW_FRAME}', True)
            return
        targets = self.calibration_targets(t_wo)
        if len(targets) < 6:
            self.say('not enough free space in front of the camera to calibrate -- clear '
                     'the view to ~0.45 m and try again', True)
            return
        arms = sorted({arm for arm, _p in targets})
        starts = {arm: node.arm_positions(arm) for arm in arms}
        if any(s is None for s in starts.values()):
            self.say('no /joint_states yet', True)
            return
        node.require_move_group()
        node.ensure_grippers_in_octomap()
        if args.close_gripper:
            for arm in arms:
                node.close_gripper(arm)

        observations, current = [], None
        self.abort.clear()
        self.calibrating = True
        try:
            per_arm = {}
            for i, (arm, tip) in enumerate(targets, start=1):
                if self.abort.is_set():
                    raise KeyboardInterrupt
                if per_arm.get(arm, 0) >= args.calib_points_per_arm:
                    continue                         # this arm has enough
                node.require_move_group()
                if current and current != arm:
                    self.say(f'{current} arm: back to where it started')
                    node.move_joints(current, starts[current], f'{current} return')
                current = arm
                self.say(f'calibration {i}/{len(targets)}: {arm} arm moving')
                joints, axis = self.calibration_pose(arm, tip, node.arm_positions(arm),
                                                     t_wo[:3, 3])
                # One direct attempt: a spot with no direct path is skipped,
                # not reached via home -- there are spare candidates.
                if joints is None or node._move_joints_once(
                        arm, joints, f'calib {i}') != MoveItErrorCodes.SUCCESS:
                    self.say(f'calibration {i}: {arm} arm cannot get there; skipping')
                    continue
                node.settle(arm, joints, timeout=8.0)   # calibration wants it truly still
                seen = self.observe_arm(i, len(targets), arm, info, background, t_wo)
                if seen is not None:
                    observations.append(seen)            # (obs, model pts, normals, depth)
                    per_arm[arm] = per_arm.get(arm, 0) + 1
        except KeyboardInterrupt:
            self.say('calibration aborted')
            return
        finally:
            self.calib_view = None
            self.calibrating = False
            if current and node.alive():
                self.say(f'{current} arm: back to where it started')
                node.move_joints(current, starts[current], f'{current} return')

        self.conclude_fitted(observations, t_wo, t_os, info)

    def conclude_fitted(self, captured, t_wo, t_os, info):
        """The 3D fit: camera pose from the arm's surfaces over every pose."""
        observations = [(obs, mp, mn) for obs, mp, mn, _depth in captured]
        arms_seen = len(observations)
        if arms_seen < 4:
            self.say(f'only {arms_seen} usable poses; 4 are needed. Clear the view in front '
                     f'of the camera and try again', True)
            return
        self.say(f'fitting the camera to {sum(len(o[0]) for o in observations)} arm points '
                 f'from {arms_seen} poses...')
        shape = self.frame[1].shape
        saved = rcc.save_run(captured, t_wo, t_os, info, shape)
        self.node.get_logger().info(f'calibration data saved to {saved}')
        solved, stats = rcc.fit_camera_multistart(observations, t_wo, info, shape)
        if getattr(self, 'last_pose_view', None) is not None:
            # The result, on the last pose's frame: the model as the new
            # camera pose places it. Judge it by eye before APPLY.
            img, depth, mp, mn = self.last_pose_view
            px, agree, seen = rcc.depth_agreement(depth, mp, mn, solved, info)
            self.frame = (img, depth, info, self.frame[3])
            self.pick = None
            self.overlay = (px[seen][::2], agree[seen][::2])
        before, within_before = rcc.alignment_error(observations, t_wo)
        after, within_after = rcc.alignment_error(observations, solved)

        def agreement(t):
            good = seen = 0
            for _obs, mp, mn, depth in captured:
                _px, agree, observed = rcc.depth_agreement(depth, mp, mn, t, info)
                good += int(agree[observed].sum())
                seen += int(observed.sum())
            return 100.0 * good / max(1, seen)
        agree_before, agree_after = agreement(t_wo), agreement(solved)
        shift = click_shift(t_wo, solved, info)
        evidence = (f'{arms_seen} arm poses, {sum(len(o[0]) for o in observations)} depth '
                    f'points fitted to the URDF meshes')
        ok = (stats['plane_rms_mm'] < 3.0 and stats['inlier_fraction'] > 0.7
              and within_after > 0.6)
        why = (f'fit rms {stats["plane_rms_mm"]:.1f} mm, {100 * within_after:.0f}% of points '
               f'within 5 mm')
        why += (f'; model/depth agreement {agree_before:.0f}% -> {agree_after:.0f}%; clicks '
                f'were {shift[0]:.0f} mm off on average, {shift[1]:.0f} mm at worst')
        self.finish_calibration(solved, t_wo, t_os, 'arm point cloud fitted to the robot model',
                                evidence, before, after, ok, why,
                                {'fit': stats, 'within_5mm_before': within_before,
                                 'within_5mm_after': within_after, 'poses': arms_seen,
                                 'agreement_before_pct': agree_before,
                                 'agreement_after_pct': agree_after,
                                 'click_shift_mm_avg': shift[0], 'click_shift_mm_max': shift[1]})

    def finish_calibration(self, solved, t_wo, t_os, method, evidence, before_mm, after_mm,
                           ok, why, extra):
        """Record, sanity-check and (unless --no-auto-apply) apply a result."""
        old = t_wo @ t_os
        new = solved @ t_os
        moved = float(np.linalg.norm(new[:3, 3] - old[:3, 3]))
        turned = math.degrees(math.acos(np.clip(
            (np.trace(old[:3, :3].T @ new[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        record = plain({
            'when': datetime.datetime.now().isoformat(timespec='seconds'),
            'method': method, 'evidence': evidence,
            'frame': SCREW_FRAME, 'parent': BASE_FRAME,
            'xyz': [round(float(v), 5) for v in new[:3, 3]],
            'rpy': [round(float(v), 5) for v in rpy_from_matrix(new[:3, :3])],
            'previous_xyz': [round(float(v), 5) for v in old[:3, 3]],
            'previous_rpy': [round(float(v), 5) for v in rpy_from_matrix(old[:3, :3])],
            'error_before_mm': round(before_mm, 1), 'error_after_mm': round(after_mm, 1),
            'moved_mm': round(1000 * moved, 1), 'turned_deg': round(turned, 2),
            **extra,
        })
        # Plain Python types only: yaml.safe_dump refuses numpy scalars, and
        # that exception used to end the run right after the solve, with an
        # empty file and nothing applied (2026-10-01).
        text = yaml.safe_dump(record, sort_keys=False)
        with open(CALIBRATION_FILE, 'w') as handle:
            handle.write(text)
        self.detail = (f'camera moves {1000 * moved:.0f} mm, turns {turned:.1f} deg; {why}')
        plausible = ok and moved < 0.12 and turned < 15.0
        if not plausible:
            if moved < 0.12 and turned < 15.0:
                # Borderline: offer it, with the overlay on screen to judge by.
                self.calibration = record
                self.say(f'not sure of this one ({why}). The model is drawn where the new '
                         f'camera pose puts it: if it sits on the arm, press APPLY.', True)
            else:
                self.calibration = None
                self.say(f'result not trusted ({why}; moves {1000 * moved:.0f} mm / '
                         f'{turned:.1f} deg); nothing changed. See camera_calibration.yaml.',
                         True)
            return
        if moved < 0.002 and turned < 0.2:
            self.say(f'camera is already right (within {1000 * moved:.1f} mm / '
                     f'{turned:.2f} deg); nothing to change')
            return
        self.calibration = record
        if self.node.args.no_auto_apply:
            self.say('calibrated. APPLY to use it.')
            return
        self.apply_calibration()

    def apply_calibration(self):
        record = self.calibration
        if record is None:
            self.say('nothing to apply -- run CALIBRATE first', True)
            return
        (x, y, z), (r, p, yw) = record['xyz'], record['rpy']
        origin = f'<origin xyz="{x:.5f} {y:.5f} {z:.5f}" rpy="{r:.5f} {p:.5f} {yw:.5f}"/>'
        # How the user measures it: from the arm origin (link0). V7 moved the
        # arm mount to (0.0126, +/-, 0.7344); the centre between the arms is y=0.0062.
        from_arm = (f'{1000 * (x - 0.0126):.2f} mm in front of and {1000 * (0.7344 - z):.2f} mm '
                    f'below the arm origin, {1000 * (y - 0.0062):+.2f} mm to the left')

        with open(CAM_ORG) as handle:
            cam_org = handle.read()
        # "V4" alone or "V4  (note...)": both are version headings.
        version = 1 + max([int(n) for n in re.findall(r'^V(\d+)\b', cam_org, re.M)] or [3])
        tag = f'V{version}'

        # URDF: the whole sensor_d455 body is ours, so replace it outright.
        with open(URDF_XACRO) as handle:
            urdf = handle.read()
        body = (f'\n    <!-- {tag}: calibrated {record["when"]} with click_to_move.py '
                f'CALIBRATE ({record["method"]},\n'
                f'         {record["error_after_mm"]} mm left). History in cam_org.txt. -->\n'
                f'    {origin}\n  ')
        urdf, n = re.subn(r'(<xacro:sensor_d455[^>]*>).*?(</xacro:sensor_d455>)',
                          lambda m: m.group(1) + body + m.group(2), urdf, flags=re.S)
        if n != 1:
            self.say('could not find xacro:sensor_d455 in v10.urdf.xacro; nothing written',
                     True)
            return

        summary = (f'current ({tag}):    {origin}\n\n'
                   f'{from_arm}\n'
                   f'X = {1000 * x:.2f} mm, Y = {1000 * y:.2f} mm, Z = {1000 * z:.2f} mm\n'
                   f'roll = {math.degrees(r):.2f} deg, pitch = {math.degrees(p):.2f} deg, '
                   f'yaw = {math.degrees(yw):.2f} deg\n'
                   f'Measured with the arm (click_to_move.py CALIBRATE), not a tape.\n')
        cam_org = re.sub(r'current \(V\d+\):.*?(?=\nhistory:)', summary.rstrip('\n') + '\n',
                         cam_org, count=1, flags=re.S)
        block = (f'\n\n{tag}\n  {origin}\n  {from_arm}\n'
                 f'  calibrated {record["when"]}: {record["evidence"]}\n'
                 f'  arm vs model {record["error_before_mm"]} mm -> {record["error_after_mm"]} mm; '
                 f'moved {record["moved_mm"]} mm, turned {record["turned_deg"]} deg '
                 f'from the previous version\n')
        cam_org = cam_org.rstrip('\n') + block

        for path, text in ((URDF_XACRO, urdf), (CAM_ORG, cam_org)):
            with open(path, 'w') as handle:
                handle.write(text)
        self.calibration = None
        self.say(f'{tag} written; rebuilding openarm_description...')
        build = subprocess.run(
            [os.path.join(WS, 'native/build_ws.sh'), '--packages-select', 'openarm_description'],
            capture_output=True, text=True)
        built = build.returncode == 0
        live, why = self.node.live_update_camera((x, y, z), (r, p, yw))
        if live:
            # The map was built through the old camera pose.
            self.capture()
        notes = [] if built else ['rebuild failed: run native/build_ws.sh --packages-select '
                                  'openarm_description']
        if live:
            self.say(f'{tag} applied and live: camera is {from_arm}. '
                     + ' '.join(notes))
        else:
            self.say(f'{tag} written ({why}); restart launch_everything to use it. '
                     + ' '.join(notes), True)
