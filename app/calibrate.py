#!/usr/bin/env python3
"""CALIBRATE: everything that calibrates the robot, in one window, apart from
the everyday tools (click_to_move.py, auto_box.py).

    native/run_calibrate.sh           (or: source native/setup.bash; python3 calibrate.py)

The window is the live camera, every ArUco marker in view outlined with its
id and what it is (green: big enough to read well, orange: small):

  CAL ARMS (f)    the arm model, front and top-down, both arms: the folded
                  pieces on both grippers, marker 11 on the wall 50 cm in
                  front -- confirm its overlay, then ~35-40 min
  CALIBRATE (c)   where the camera is, from the arms (after moving it);
                  APPLY writes it
  OVERLAY (o)     the robot model drawn over the camera image: a check
  click + MOVE    a test: the arm touches the clicked point with its tool
                  (the tool row sets it) and goes straight back to pre_pick
  ANCHOR (a)      with it on, a test touch holds: nudge the tip exactly onto
                  the clicked point (arrows, + -), SAVE -- ground truth for
                  where the TIP is (the camera sees only the pieces). Best on
                  marker 11's centre and on box tops (TOP DOWN), 3-4 per arm
  REFIT (e)       the arm model again from the last CAL ARMS's recordings
                  plus the anchors: seconds, no new run
  NAVIGATION (n), PICK MODE (p), TOP DOWN (d), RETURN (b)

Here the calibration pieces folded round each gripper's rail are in the
planners (they stick out ~4 cm past the rail's ends and towards the
fingers); click_to_move.py and auto_box.py take them out again.
"""

import threading
import time

import cv2
import numpy as np

import click_to_move as ctm_app
from ctm import armcal, board as bd
from ctm.common import FONT, WS, decode_image


class CalibrateApp(ctm_app.App):
    """click_to_move's window with the calibration tools and a live view."""

    HINT = 'click a point to test   keys: f c o a e m n p d b r'
    CALIBRATION = True

    def __init__(self, node):
        node.pieces_on = True                 # the pieces are on the grippers here
        super().__init__(node)
        self.params = cv2.aruco.DetectorParameters_create()
        self.anchor_mode = False              # ANCHOR: test touches save anchors

    # -- the buttons ----------------------------------------------------------

    def main_buttons(self, idle, grey):
        specs = [s_ for s_ in super().main_buttons(idle, grey) if s_[0] not in ('teach',)]
        n = {arm: len(armcal.load_anchors(arm)) for arm in ('left', 'right')}
        specs.insert(5, ('anchor', f'ANCHOR ON {n["left"]}/{n["right"]}' if self.anchor_mode
                         else f'ANCHOR {n["left"]}/{n["right"]}',
                         (0, 100, 220) if self.anchor_mode else (70, 70, 120)))
        specs.insert(6, ('refit', 'REFIT', (0, 140, 0) if idle and (n['left'] or n['right'])
                         else grey))
        return specs

    def press(self, name):
        if name == 'anchor':
            self.anchor_mode = not self.anchor_mode
            self.teach = self.anchor_mode     # the touch holds for the nudge
            self.say('ANCHOR on: click a precise point (marker 11 centre, a box top), MOVE; '
                     'nudge the tip exactly onto it, SAVE' if self.anchor_mode else
                     'ANCHOR off')
        elif name == 'refit':
            self.in_background(self.refit)
        else:
            super().press(name)

    def on_key(self, key):
        if key in (ord('a'), ord('A')):
            self.press('anchor')
            return True
        if key in (ord('e'), ord('E')) and not self.busy:     # (r is RECAPTURE)
            self.press('refit')
            return True
        return False

    def refit(self):
        """The arm model from the last CAL ARMS recordings plus the anchors."""
        import glob
        import os
        import types
        report, results = [], {}
        for arm in ('right', 'left'):
            anchors = armcal.load_anchors(arm)
            files = sorted(f for f in glob.glob(os.path.join(WS, 'armcal_samples',
                                                             f'{arm}_2*.npz')))
            if len(anchors) < 2 or not files:
                report.append(f'{arm}: ' + ('fewer than 2 anchors' if len(anchors) < 2
                                            else 'no CAL ARMS recordings') + ' - not refitted')
                continue
            z = np.load(files[-1])
            info = types.SimpleNamespace(k=z['k'].tolist(), d=z['d'].tolist(),
                                         width=int(z['size'][0]), height=int(z['size'][1]))
            samples = [(q, {int(i): c for i, c in zip(z['ids'], pix) if not np.isnan(c).any()})
                       for q, pix in zip(z['q'], z['pixels'])]
            if not samples or len(samples[0][1]) and \
                    bd.piece_of(arm, next(iter(samples[0][1]))) is None:
                report.append(f'{arm}: its recordings are of the old pieces - run CAL ARMS')
                continue
            dirs = z['dirs'] if 'dirs' in z and len(z['dirs']) == len(samples) else \
                np.zeros((len(samples), 7))
            before = self._anchor_error(arm, anchors)
            variant = (armcal.load_model_raw().get(arm) or {}).get('model', 'backlash')
            self.say(f'{arm} arm: refitting {len(samples)} postures + {len(anchors)} anchors...')
            log = self.node.get_logger().info
            p, b, st = armcal.fit(arm, samples, z['t_wo'], info, log=log, max_nfev=40)
            samples, dropped, keep = armcal.drop_outliers(arm, samples, p, b, z['t_wo'], info)
            p, b, st = armcal.fit(arm, samples, z['t_wo'], info, log=log, dirs=dirs[keep],
                                  variant=variant, anchors=anchors,
                                  tip_offset=self.node.args.tip_offsets[arm])
            st.update(model=variant, dropped=dropped, holdout_px=float('nan'))
            results[arm] = (p, b, st)
            report.append(f'{arm}: tip at the anchors {before:.1f} -> {st["anchor_mm"]:.1f} mm, '
                          f'pieces fit {st["rms_px"]:.2f} px')
        if results:
            armcal.save_model(results)
            self.arm_models = armcal.ArmModels()
        self.detail = '; '.join(report)
        self.say('REFIT: ' + '; '.join(report), not results)

    def _anchor_error(self, arm, anchors):
        """rms (mm) of where the current model puts the tip at the anchors."""
        if not self.arm_models.has(arm):
            return float('nan')
        kin, params = self.arm_models.arms[arm], self.arm_models.params[arm]
        delta = np.r_[params.get('tip_delta', np.zeros(3)), 0.0]
        errs = [np.linalg.norm((kin.tcp(np.array(r['q']), params)
                                @ (np.array([0.0, 0.0, r['tip_offset'], 1.0]) + delta))[:3]
                               + self.arm_models.shift(arm) - np.array(r['point']))
                for r in anchors]
        return float(np.sqrt(np.mean(np.square(errs)))) * 1000

    # -- a click: capture now (the live view is not the captured frame), then
    # -- the point and its check, as in click_to_move ----------------------------

    def on_mouse(self, event, u, v, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 <= u <= x1 and y0 <= v <= y1:
                self.press(name)
                return
        if self.busy or self.calib_view is not None or self.calibrating:
            return
        live = self.live_image()
        if live is None or v >= live.shape[0]:
            return

        def job():
            if self.mode != 'pick' and not self.ensure_pick_mode():
                return
            self.capture()
            if self.frame is None:
                return
            check = self.select_point(u, v)
            if check is not None:
                self.check_pick(check, in_job=True)
        self.in_background(job)

    # -- the live view ----------------------------------------------------------

    def live_image(self):
        color, _depth, _info = self.node.latest()
        return decode_image(color) if color is not None else None

    def draw(self):
        if (self.calib_view is not None or self.overlay is not None or self.teaching
                or self.confirming):
            return super().draw()
        img = self.live_image()
        if img is None:
            return super().draw()
        img = img.copy()
        # markers found ~6 times a second, drawn on every frame: finding them
        # on every frame kept 2+ cores busy (2026-10-16)
        now = time.monotonic()
        if now - getattr(self, '_found_at', 0.0) > 0.15:
            self._found = cv2.aruco.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                  bd.dictionary(), parameters=self.params)
            self._found_at = now
        corners, ids, _ = self._found
        seen = []
        for c, i in zip(corners or [], ids.ravel() if ids is not None else []):
            c = c.reshape(4, 2)
            size = float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))
            colour = (0, 220, 0) if size >= 20 else (0, 200, 255)
            cv2.polylines(img, [np.round(c).astype(np.int32)], True, colour, 2)
            label = f'{int(i)} {bd.name_of(int(i))}'.strip()
            x, y = int(c[:, 0].min()), max(12, int(c[:, 1].min()) - 5)
            cv2.putText(img, label, (x, y), FONT, 0.4, (0, 0, 0), 3)
            cv2.putText(img, label, (x, y), FONT, 0.4, colour, 1)
            seen.append(int(i))
        if self.pick:
            u, v = self.pick[0], self.pick[1]
            ok = self.check['ok'] if self.check is not None else None
            colour = (0, 200, 0) if ok else (0, 0, 230) if ok is False else (0, 200, 255)
            cv2.drawMarker(img, (int(u), int(v)), colour, cv2.MARKER_CROSS, 18, 2)
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        lines = [f'[{(self.mode or "?").upper()}{" TOP DOWN" if self.top_down else ""}]  '
                 f'{self.HINT}' + ('  [busy]' if self.busy else ''),
                 self.status,
                 f'markers: {len(seen)} '
                 + ('(' + ', '.join(str(i) for i in sorted(seen)[:14])
                    + (' ...' if len(seen) > 14 else '') + ')' if seen else '- none in view')]
        for k, text in enumerate(lines):
            y = 22 + 22 * k
            cv2.putText(img, text, (8, y), FONT, 0.5, (0, 0, 0), 3)
            cv2.putText(img, text, (8, y), FONT, 0.5, (255, 255, 255), 1)
        return self.with_buttons(img)


def main():
    args = ctm_app.parse_args()
    args.drop_hold = -1.0                     # a test touch goes straight back to pre_pick
    ctm_app.main(args, CalibrateApp)


if __name__ == '__main__':
    threading.current_thread().name = 'calibrate'
    main()
