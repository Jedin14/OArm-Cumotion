#!/usr/bin/env python3
"""AUTO BOX: find the box to pick with the trained detector, then pick it.

The same robot, planning and touch sequence as click_to_move.py -- only the
point comes from the box model (box_target_detector/) instead of a click.

    native/setup.bash; python3 auto_box.py

The flow:
  1. The arms boot in navigation (boot_pre_pick.py); the window starts there.
  2. PICK MODE (p): both arms to pre_pick, and the box model loads (once).
  3. DETECT BOX (space): a fresh frame (both arms at pre_pick, out of view)
     goes to the model. The target box -- the topmost, leftmost of the top
     row -- is drawn with its bounding box and centroid; the centroid's depth
     gives the world point, the half of the frame its arm. The whole pick is
     then planned without moving (reach, a collision-free path, the straight
     line in): the box turns GREEN if the arm can pick it, RED if not, and
     says why.
  4. PICK (m): the arm goes to the box exactly as click_to_move's MOVE does
     -- approach, corrections, the straight line in, stop on contact, hold,
     back out, pre_pick, drop, pre_pick.
  NAVIGATION (n) takes both arms back; TOP DOWN (d) picks a box lying flat
  from above; RETURN (b) brings an arm back if a pick stopped half-way.

Calibrate with click_to_move.py (CALIBRATE, CAL ARMS); both tools share the
camera calibration, the arm model and the tool settings.
"""

import os
import sys
import threading
import time

import cv2

import numpy as np

import click_to_move as ctm_app
from ctm.common import BASE_FRAME, FONT, pixel_ray

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'box_target_detector'))


class AutoBoxApp(ctm_app.App):
    """click_to_move's window with the box model choosing the point."""

    def __init__(self, node):
        super().__init__(node)
        self.detector = None          # BoxTargetDetector, loaded on PICK MODE
        self.detector_error = None
        self.loading = threading.Lock()
        self.box = None               # the last detection (dict from the model)

    # -- the model ----------------------------------------------------------

    def load_detector(self):
        """Load the box model once (~2 s, then ~5 ms a frame on the GPU)."""
        with self.loading:
            if self.detector is not None:
                return True
            try:
                from box_detector import BoxTargetDetector
                t0 = time.monotonic()
                self.detector = BoxTargetDetector(conf=self.node.args.box_conf)
                self.node.get_logger().info(
                    f'box model loaded in {time.monotonic() - t0:.1f} s')
                self.detector_error = None
                return True
            except Exception as exc:                 # noqa: BLE001 - shown in the window
                self.detector_error = str(exc)
                self.say(f'box model failed to load: {exc}', True)
                return False

    def enter_pick_mode(self):
        """PICK MODE: both arms to pre_pick, and the model loads meanwhile."""
        threading.Thread(target=self.load_detector, daemon=True).start()
        self.in_background(lambda: self.go_mode('pick'))

    def detect_box(self):
        """DETECT BOX: a fresh frame to the model; the target and whether its
        arm can pick it (planned, not moved)."""
        def job():
            if self.mode != 'pick' and not self.ensure_pick_mode():
                return
            if self.detector is None:
                self.say('loading the box model...')
            if not self.load_detector():
                return
            self.box = None
            self.capture()                           # arms at pre_pick: a clear view
            if self.frame is None:
                return
            t0 = time.monotonic()
            result = self.detector.detect(self.frame[0])
            took = (time.monotonic() - t0) * 1000
            if result is None:
                self.pick = self.check = None
                self.say(f'no box found ({took:.0f} ms)', True)
                return
            self.box = result
            self.say(f'box at pixel ({result["x"]:.0f}, {result["y"]:.0f}), '
                     f'{result["num_boxes"]} boxes seen, confidence '
                     f'{result["confidence"]:.2f} ({took:.0f} ms)')
            top = self.box_top(result) if self.top_down else None
            if self.top_down and top is None:
                self.say('TOP DOWN: could not find the box\'s top in the depth image; '
                         'aiming at its centre', True)
            check = self.select_point(result['x'], result['y'], world=top)
            if check is None:
                self.say('box found, but no depth at its centre - is it too close or too '
                         'far for the camera?', True)
                return
            self.check_pick(check, in_job=True)
        self.in_background(job)

    def box_top(self, result):
        """TOP DOWN: the middle of the box's top surface, world -- from the
        depth inside its bounding box: the highest points (within 1 cm of
        the top), their median. The bounding box's centre is often on the
        front face or the front edge, and the tip came down on the edge
        (2026-10-14). None if too few points."""
        bgr, depth, info, frame_id = self.frame
        t_wo = self.node.lookup(BASE_FRAME, frame_id)
        if t_wo is None:
            return None
        dh, dw = depth.shape
        bh, bw = bgr.shape[:2]
        sx, sy = dw / float(bw), dh / float(bh)
        x1, y1, x2, y2 = result['bbox']
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)         # stay off the box's outline
        us = np.arange(int((x1 + mx) * sx), int((x2 - mx) * sx), 2)
        vs = np.arange(int((y1 + my) * sy), int((y2 - my) * sy), 2)
        if not len(us) or not len(vs):
            return None
        uu, vv = np.meshgrid(us, vs)
        z = depth[vv, uu].ravel()
        keep = (z > 0.1) & (z < 4.0) & np.isfinite(z)
        if keep.sum() < 30:
            return None
        cam = np.array([pixel_ray(u / sx, v / sy, info) for u, v in
                        zip(uu.ravel()[keep], vv.ravel()[keep])]) * z[keep][:, None]
        world = cam @ t_wo[:3, :3].T + t_wo[:3, 3]
        top_z = np.percentile(world[:, 2], 95)
        top = world[world[:, 2] > top_z - 0.01]
        if len(top) < 15:
            return None
        # 2 cm in from the top's near edge, not its middle (the far part
        # needs more wrist lean than the arm has, 2026-10-14) -- but never
        # past the middle
        out = np.median(top, axis=0)
        near_edge = np.percentile(top[:, 0], 5)
        out[0] = min(out[0], near_edge + 0.02)
        return out

    def pick_box(self):
        """PICK: only a detected, pickable box."""
        if self.box is None or self.pick is None:
            self.say('press DETECT BOX first', True)
            return
        if self.check is not None and self.check.get('ok') is False:
            self.say('that box is not pickable (see why above) - DETECT BOX again', True)
            return
        super().move()

    def move(self):
        """MOVE / m / Enter: the PICK, with its checks."""
        self.pick_box()

    def on_key(self, key):
        if key == ord(' '):
            self.detect_box()
            return True
        return False

    # -- the window ---------------------------------------------------------

    HINT = 'p pick mode  space detect  m pick  n navigation  d top down  b return'

    WORDING = (('click a point, then MOVE', 'DETECT BOX, then PICK'),
               ('Click the next point and press MOVE.', 'DETECT BOX for the next one.'),
               ('click a point to start picking', 'PICK MODE to start'),
               ('press MOVE', 'press PICK'),
               ('Click a point, then MOVE.', 'DETECT BOX, then PICK.'))

    def say(self, text, error=False):
        """click_to_move's messages, in this window's words."""
        for old, new in self.WORDING:
            text = text.replace(old, new)
        super().say(text, error)


    def main_buttons(self, idle, grey):
        ok = self.check['ok'] if self.check is not None and self.box is not None else None
        ready = idle and self.mode == 'pick'
        pick_colour = (grey if not (idle and self.box is not None) else (0, 150, 0) if ok
                       else (0, 120, 200) if ok is None else (0, 0, 150))
        loaded = self.detector is not None
        specs = [('navigation', 'NAVIGATION', (160, 110, 0) if self.mode == 'navigation'
                  else (110, 80, 30) if idle else grey),
                 ('pickmode', 'PICK MODE', (0, 160, 160) if self.mode == 'pick'
                  else (30, 100, 100) if idle else grey),
                 ('detect', 'DETECT BOX' if loaded or not ready else 'DETECT (loading)',
                  (170, 60, 0) if ready else grey),
                 ('pickbox', 'PICK', pick_colour),
                 ('topdown', 'TOP DOWN ON' if self.top_down else 'TOP DOWN',
                  (0, 90, 230) if self.top_down else (70, 70, 120))]
        if self.last_touch is not None:
            specs.append(('return', 'RETURN', (0, 0, 200) if idle else grey))
        return specs

    def press(self, name):
        if name == 'pickmode':
            self.enter_pick_mode()
        elif name == 'detect':
            self.detect_box()
        elif name in ('pickbox', 'move'):
            self.pick_box()
        else:
            super().press(name)

    def on_mouse(self, event, u, v, flags, param):
        """Buttons only: the point comes from the model, not a click."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 <= u <= x1 and y0 <= v <= y1:
                self.press(name)
                return

    def draw(self):
        img = super().draw()
        if self.box is None or self.calib_view is not None or self.frame is None:
            return img
        h = self.frame[0].shape[0]
        top = img[:h]
        # every box the model found, light cyan; the target is drawn over it
        for b in self.box.get('boxes', []):
            bx1, by1, bx2, by2 = (int(round(c)) for c in b)
            cv2.rectangle(top, (bx1, by1), (bx2, by2), (255, 255, 160), 2)
        count = f'boxes: {self.box.get("num_boxes", len(self.box.get("boxes", [])))}'
        (cw, _ch), _ = cv2.getTextSize(count, FONT, 0.7, 2)
        cx0 = (top.shape[1] - cw) // 2
        cv2.putText(top, count, (cx0, 68), FONT, 0.7, (0, 0, 0), 4)
        cv2.putText(top, count, (cx0, 68), FONT, 0.7, (255, 255, 160), 2)
        ok = self.check['ok'] if self.check is not None else None
        plan = self.check.get('plan') if self.check is not None else None
        # at the edge of reach the arm's sag correction may not fit: the last
        # line can end short of the box (boxes at 55 cm, 2026-10-13)
        edge = bool(ok and plan is not None and plan.get('edge'))
        colour = ((0, 140, 255) if edge else (0, 200, 0)) if ok else \
            (0, 0, 230) if ok is False else (0, 200, 255)
        x1, y1, x2, y2 = (int(round(c)) for c in self.box['bbox'])
        cv2.rectangle(top, (x1, y1), (x2, y2), colour, 2)
        cx, cy = int(round(self.box['x'])), int(round(self.box['y']))
        cv2.drawMarker(top, (cx, cy), colour, cv2.MARKER_CROSS, 18, 2)
        state = ('EDGE OF REACH - may not reach' if edge else 'PICKABLE' if ok
                 else 'NOT PICKABLE' if ok is False else 'checking...')
        lines = [f'box {state}' + (f' - {self.pick[3]} arm' if self.pick else ''),
                 f'pixel ({cx}, {cy})  conf {self.box["confidence"]:.2f}']
        if self.pick is not None:
            p = self.pick[2]
            lines.append(f'x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f} m')
        ty = max(18, y1 - 8 - 16 * (len(lines) - 1))
        tx = min(max(4, x1), top.shape[1] - 230)
        for i, text in enumerate(lines):
            y = ty + 16 * i
            cv2.putText(top, text, (tx, y), FONT, 0.45, (0, 0, 0), 3)
            cv2.putText(top, text, (tx, y), FONT, 0.45, colour, 1)
        return img



def make_parser():
    parser = ctm_app.make_parser()
    parser.description = __doc__.split('\n')[0]
    parser.add_argument('--box-conf', type=float, default=0.5,
                        help='minimum detection confidence of the box model (0-1)')
    return parser


def main():
    args = make_parser().parse_args()
    ctm_app.finish_args(args)
    ctm_app.main(args, AutoBoxApp)


if __name__ == '__main__':
    main()
