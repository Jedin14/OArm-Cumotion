"""Take a frame, click a point on it, and an arm's fingertips touch it. cuMotion only.

No VLM, no grasp model. Bring the robot up exactly as for pick and place --
MoveIt, cuMotion, the D455 and the octomap gater all come from
launch_everything.launch.py -- then run this on top:

    native/run_launch_everything.sh              # terminal 1, unchanged
    native/run_click_to_move.sh                  # terminal 2

The window shows a captured frame, not live video.

    left click      select that pixel: marker + "x y z" in RViz and in the window
    MOVE / m        the arm on that side touches the selected point
    RECAPTURE / r   take a new frame *and* rebuild the octomap from it
    CALIBRATE / c   measure where the camera really is with the arm, and fix it
    OVERLAY / o     draw the robot model on the image: green = depth agrees
    APPLY           (only with --no-auto-apply) write the calibration
    q               quit

The x y z shown is the clicked surface in `world`, metres: origin at the base
of the stand, +x forward, +y the robot's left, +z up. The shoulders are at
(0.0126, -0.0448/+0.0572, 0.7344) since V7 (cam_org.txt).

Why a touch lands where it does -- the three errors, and what handles each
--------------------------------------------------------------------------

1. What touches. hand_tcp is 80 mm out from the hand, but the fingertips
   reach 95.4 mm (finger joint at z=0.015, finger mesh 80.4 mm beyond it; see
   openarm_hand.xacro). Aiming hand_tcp at the point drove the fingertips
   15 mm into the surface, and a stiff arm pressed into a surface slides off
   the spot. So the goal is the fingertips, `--tip-offset` beyond hand_tcp
   along the tool axis. The gripper is not commanded by default, so the point
   is aimed midway between the fingertips; `--close-gripper` shuts them
   first so they meet at one point and actually touch it.

2. The arm not arriving. The joint trajectory controller has no goal
   tolerance, so MoveIt reports SUCCESS the moment a trajectory ends; the
   hardware's integral term (KI 20 against KP 100, a 5 s time constant) is
   still pulling the last of the gravity sag out. pick_place_orchestrator
   measured tools arriving 15-36 mm short with a perfect plan. So after the
   approach the arm is left to settle, the fingertip is *measured* (tf from
   /joint_states, i.e. the encoders), and any error is flown out with a small
   straight correction; the same offset is carried into the touch. The window
   reports the residual.

3. The camera being somewhere else. The mount in the URDF is a tape
   measurement (cam_org.txt), and a centimetre or a degree there moves every
   clicked point. Nothing downstream can see that error -- the arm goes
   exactly where the camera said -- so it has to be measured. CALIBRATE does
   that by itself (robot_camera_calibration.py): it moves each arm to six
   poses in view, and at each one takes the arm as the depth camera sees it
   (thousands of points: what is now nearer than a frame taken before) and
   the robot's own model of it (the URDF visual meshes, posed by the joint
   encoders through tf). The camera pose is the rigid transform that puts
   the one onto the other, by robust point-to-plane ICP over all poses,
   started from several guesses around the tape measurement. It is a 3D fit,
   so it cannot trade camera height against tilt, and it calibrates exactly
   what a click uses. A plausible result is applied at once: written as the
   next version into cam_org.txt and v10.urdf.xacro,
   openarm_description rebuilt, robot_state_publisher given the new URDF
   live. The window reports how far off clicks were and the model/depth
   agreement before and after. OVERLAY draws the model on the image (green:
   depth agrees, red: it does not) for checking by eye whenever an arm is
   in view.

A touch (MOVE)
--------------

1. The pixel is deprojected with the aligned depth -- a per-pixel median over
   several frames, then over a small patch, undistorted -- and taken into
   `world` through tf. It goes on /click_to_move/marker.
2. The arm is chosen by the half of the frame: left half -> left arm.
3. Right after the click the whole touch is planned (reach, a front
   approach with the tool level (horizontal), turned 0/15/30 deg, a collision-free
   cuMotion path, the straight line in); MOVE turns green or red.
4. MOVE: fly the planned path to the point 10 cm out (`--standoff`); one
   correction there -- from the gripper flags if they are in view, else
   from the joints; then ONE straight line along the tool axis: fast to
   1 cm short (`--fast-line-speed`), the last centimetre slowly
   (`--contact-speed`), stopping on contact; hold `--hold` s; straight back out; retrace the approach path
   back to the starting posture. (Gripper closed first with --close-gripper.)

Code layout
-----------
click_to_move.py           the window (App), command-line options, main()
ctm/robot.py               ClickToMove node: ROS/MoveIt interface, motion primitives
ctm/touch.py               reach check, preplan, the touch sequence, TEACH
ctm/flags.py               gripper flags: visibility, measurement, FLAGS calibration
ctm/camera_calibration.py  CALIBRATE / APPLY / OVERLAY
ctm/corrections.py         TEACH corrections (touch_corrections.yaml)
ctm/common.py              constants, geometry and image helpers

Touching needs two exemptions, as in pick_place_orchestrator: the hand and
fingers of both arms may touch <octomap> (the clicked surface is in the map);
the forearm and upper arm are still checked. And joint goals rather than
cuMotion pose goals, which serve one tool_frame only and plan all 14 joints.
"""

import argparse
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

import gripper_markers as gm
from ctm.camera_calibration import CameraCalibrationMixin
from ctm.common import (FINGERTIP_BEYOND_TCP, FONT, TOOL_EXTENSION, MoveGroupDown, WINDOW,
                        decode_image)
from ctm.flags import FlagsMixin
from ctm.modes import ModesMixin, load_poses
from ctm.robot import ClickToMove
from ctm.touch import TouchMixin


class App(TouchMixin, FlagsMixin, CameraCalibrationMixin, ModesMixin):
    """The OpenCV window, and the worker threads that capture and move."""

    def __init__(self, node):
        self.node = node
        self.frame = None             # (bgr, depth, info, frame_id)
        self.pick = None              # (u, v, world point, arm, camera origin)
        self.check = None             # the reach check of the current pick (dict)
        self.unreachable = None       # (H, W) bool: surface beyond either arm's reach
        self.plan_lock = threading.Lock()
        self.last_touch = None        # an arm that is out, and how to bring it back
        self.poses = load_poses()     # {arm: {navigation_state, pre_pick_state, drop_state}}
        self.mode = None              # 'navigation' / 'pick' / None (see ctm/modes.py)
        self.buttons = {}             # name -> (x0, y0, x1, y1) in window pixels
        self.status = 'capturing...'
        self.detail = ''
        self.busy = False
        # Calibration: the view being clicked, the answer, and a result to apply.
        self.calib_view = None        # (bgr, predicted (u, v) or None, index, total)
        self.calib_answer = None
        self.calib_event = threading.Event()
        self.calibration = None
        self.calibrating = False
        self.overlay = None           # (pixels (N,2), agree (N,) bool): robot model on the image
        self.markers = gm.MarkerCalibration.load()   # None until CALIBRATE MARKERS has run
        self.teach = False            # TEACH mode: hold each touch for nudges
        self.top_down = node.args.top_down   # approach straight down instead of level
        self.teaching = False         # a touch is being taught right now
        self.abort = threading.Event()

    def say(self, text, error=False):
        self.status = text
        # Two call sites on purpose: rclpy raises "Logger severity cannot be
        # changed between calls" when one line logs at two severities, which
        # turned the first error message of every run into a crash.
        if error:
            self.node.get_logger().error(text)
        else:
            self.node.get_logger().info(text)

    def in_background(self, work):
        if self.busy:
            return
        self.busy = True

        def run():
            try:
                work()
            except MoveGroupDown:
                self.say('move_group stopped answering -- it crashed (grep "process has died" '
                         '~/.ros/log/latest/launch.log). It respawns in a few seconds; then '
                         'RECAPTURE (its octomap is gone) and try again.', True)
            except Exception as exc:                 # noqa: BLE001 - shown in the window
                self.say(f'error: {exc}', True)
            finally:
                self.busy = False
                self.calib_view = None
        threading.Thread(target=run, daemon=True).start()

    def capture(self):
        """Take a frame and rebuild the octomap from the same moment."""
        deadline = time.monotonic() + 15.0
        while True:
            color, depth, info = self.node.latest()
            if color is not None and depth is not None and info is not None:
                break
            if time.monotonic() > deadline:
                self.say('no camera frames -- is the D455 streaming?', True)
                return
            self.status = 'waiting for camera topics...'
            time.sleep(0.2)
        self.say(f'frame captured; averaging {self.node.args.depth_frames} depth frames...')
        averaged, last = self.node.averaged_depth(self.node.args.depth_frames)
        if averaged is None:
            self.say('no depth frames arrived', True)
            return
        color, _d, info = self.node.latest()
        self.frame = (decode_image(color), averaged, info,
                      last.header.frame_id or info.header.frame_id)
        self.pick = None
        self.check = None
        self.overlay = None
        try:
            self.unreachable = self.reach_map()
        except Exception as exc:                     # noqa: BLE001 - only a hint
            self.unreachable = None
            self.node.get_logger().warn(f'reach map: {exc}')
        self.say('frame captured; rebuilding the octomap...')
        message = self.node.rebuild_octomap()
        n = self.node.refresh_walls(self.frame[3])
        self.say(f'{message}' + ('' if n is None else f'; walls around the picture ({n})')
                 + '. Click a point, then MOVE.')

    # -- input ----------------------------------------------------------------

    def answer(self, what, where=None):
        if what == 'abort':
            self.abort.set()
        self.calib_answer = (what, where)
        self.calib_event.set()

    def on_mouse(self, event, u, v, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 <= u <= x1 and y0 <= v <= y1:
                self.press(name)
                return
        if self.calib_view is not None:
            return
        if self.busy or self.frame is None:
            return
        bgr, depth, info, frame_id = self.frame
        if v >= bgr.shape[0]:
            return                                   # the button bar, off a button
        dh, dw = depth.shape
        bh, bw = bgr.shape[:2]
        try:
            point, origin, z = self.node.pixel_to_world(
                int(u * dw / bw), int(v * dh / bh), depth, info, frame_id)
        except Exception as exc:                     # noqa: BLE001
            self.pick = None
            self.say(f'pixel ({u}, {v}): {exc}', True)
            return
        if self.node.args.arm != 'auto':
            arm = self.node.args.arm
        else:
            arm = 'left' if u < bw / 2.0 else 'right'
        self.pick = (u, v, point, arm, origin)
        self.node.publish_marker(point)
        where = f'x={point[0]:+.3f} y={point[1]:+.3f} z={point[2]:+.3f} (depth {z:.2f} m)'
        self.say(f'{where}, {arm} arm - checking it can be touched...')
        check = {'pick': self.pick, 'where': where, 'ok': None, 'plan': None}
        self.check = check
        if self.mode != 'pick':
            # First point: into pick mode (both arms to pre_pick), then the check.
            self.in_background(lambda: self.ensure_pick_mode() and
                               self.check_pick(check, in_job=True))
        else:
            threading.Thread(target=self.check_pick, args=(check,), daemon=True).start()

    def press(self, name):
        if self.teaching and name in ('left', 'right', 'up', 'down', 'in', 'out', 'save',
                                      'skip'):
            self.answer(name)
        elif name == 'teach':
            self.teach = not self.teach
            self.say('TEACH on: after each touch, nudge the fingertip onto the spot and SAVE'
                     if self.teach else 'TEACH off: touches use what was taught')
        elif name == 'topdown':
            if self.busy:
                return
            self.top_down = not self.top_down
            self.say('TOP DOWN on: the tool comes straight down from 10 cm above the point '
                     '(objects lying flat)' if self.top_down else
                     'TOP DOWN off: the tool approaches level, from the camera side')
            if self.pick is not None and self.mode == 'pick':
                # the last check planned the other way round: check again
                _u, _v, p, _arm, _o = self.pick
                check = {'pick': self.pick, 'ok': None, 'plan': None,
                         'where': f'x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}'}
                self.check = check
                threading.Thread(target=self.check_pick, args=(check,), daemon=True).start()
            else:
                self.check = None
        elif name == 'skip':
            self.answer('skip')
        elif name == 'abort':
            self.answer('abort')
        elif name == 'move':
            self.move()
        elif name == 'recapture':
            self.in_background(self.capture)
        elif name == 'calibrate':
            self.in_background(self.calibrate)
        elif name == 'apply':
            self.in_background(self.apply_calibration)
        elif name == 'markers':
            self.in_background(self.auto_calibrate)
        elif name == 'return':
            self.in_background(self.return_to_start)
        elif name == 'navigation':
            self.in_background(lambda: self.go_mode('navigation'))
        elif name == 'pickmode':
            self.in_background(lambda: self.go_mode('pick'))
        elif name == 'overlay':
            if self.overlay is not None:
                self.overlay = None
            else:
                self.in_background(self.show_overlay)

    # -- drawing --------------------------------------------------------------

    def draw(self):
        if self.calib_view is not None:
            img, predicted, found, mask = self.calib_view
            img = img.copy()
            if mask is not None:
                img[mask] = (0.5 * img[mask] + 0.5 * np.array([255, 120, 0])).astype(np.uint8)
            if predicted is not None and np.all(np.isfinite(predicted)):
                pu, pv = int(round(predicted[0])), int(round(predicted[1]))
                cv2.circle(img, (pu, pv), 9, (0, 255, 0), 2)
                label = 'tf says the tip is here'
                lw = cv2.getTextSize(label, FONT, 0.45, 1)[0][0]
                lx = pu + 12 if pu + 12 + lw < img.shape[1] else pu - 12 - lw
                cv2.putText(img, label, (lx, pv + 4), FONT, 0.45, (0, 255, 0), 1)
            if found is not None:
                fu, fv = int(round(found[0])), int(round(found[1]))
                cv2.drawMarker(img, (fu, fv), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        elif self.frame is None:
            img = np.zeros((480, 640, 3), np.uint8)
        else:
            img = self.frame[0].copy()
            h, w = img.shape[:2]
            if self.unreachable is not None and self.unreachable.shape == (h, w):
                bad = self.unreachable
                img[bad] = (0.55 * img[bad] + 0.45 * np.array([40, 40, 200])).astype(np.uint8)
                cv2.putText(img, 'red tint: out of reach', (w // 2 - 80, h - 10), FONT, 0.45,
                            (120, 120, 255), 1)
            cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
            cv2.putText(img, 'LEFT ARM', (8, h - 10), FONT, 0.5, (255, 255, 255), 1)
            cv2.putText(img, 'RIGHT ARM', (w - 90, h - 10), FONT, 0.5, (255, 255, 255), 1)
            if self.pick:
                u, v, p, _arm, _origin = self.pick
                cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.circle(img, (u, v), 10, (0, 255, 255), 1)
                text = f'{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}'
                (tw, _th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
                tx = u + 12 if u + 12 + tw < img.shape[1] else u - 12 - tw
                cv2.putText(img, text, (tx, v - 12), FONT, 0.5, (0, 0, 0), 3)
                cv2.putText(img, text, (tx, v - 12), FONT, 0.5, (0, 255, 255), 1)
        if self.overlay is not None:
            pts, agree = self.overlay
            for (u, v), good in zip(pts.astype(int), agree):
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    img[v, u] = (0, 255, 0) if good else (0, 0, 255)
        if self.teaching:
            head = 'arrows: nudge   +/-: in/out   Enter: SAVE   Esc: SKIP'
        elif self.calibrating:
            head = 'calibrating by itself - keep the view clear   ABORT / x'
        else:
            head = (f'[{(self.mode or "?").upper()}{" TOP DOWN" if self.top_down else ""}]  '
                    f'click a point   keys: m n p d b r c o t f')
        for i, text in enumerate((head + ('  [busy]' if self.busy else ''), self.status)):
            y = 22 + 22 * i
            cv2.putText(img, text, (8, y), FONT, 0.5, (0, 0, 0), 3)
            cv2.putText(img, text, (8, y), FONT, 0.5, (255, 255, 255), 1)
        return self.with_buttons(img)

    def with_buttons(self, img):
        """The frame with a button bar under it (OpenCV has no widgets)."""
        h, w = img.shape[:2]
        bar = np.full((84, w, 3), 40, np.uint8)
        grey = (90, 90, 90)
        if self.teaching:
            nudge = (120, 90, 0)
            specs = [(n, n.upper(), nudge) for n in ('left', 'right', 'up', 'down', 'in', 'out')]
            specs += [('save', 'SAVE', (0, 150, 0)), ('skip', 'SKIP', (90, 90, 90))]
        elif self.calibrating:
            specs = [('abort', 'ABORT', (0, 0, 170))]
        else:
            idle = not self.busy
            ok = self.check['ok'] if self.check is not None else None
            move_colour = (grey if not (idle and self.pick) else (0, 150, 0) if ok
                           else (0, 120, 200) if ok is None else (0, 0, 150))
            specs = [('move', 'MOVE', move_colour),
                     ('recapture', 'RECAPTURE', (150, 90, 0) if idle else grey),
                     ('calibrate', 'CALIBRATE', (140, 0, 140) if idle else grey),
                     ('overlay', 'OVERLAY', (0, 140, 140) if idle or self.overlay else grey),
                     ('topdown', 'TOP DOWN ON' if self.top_down else 'TOP DOWN',
                      (0, 90, 230) if self.top_down else (70, 70, 120)),
                     ('teach', 'TEACH ON' if self.teach else 'TEACH',
                      (0, 100, 220) if self.teach else (70, 70, 120)),
                     ('markers', 'AUTO CAL', (0, 140, 0) if self.markers is not None and idle
                      else (100, 100, 40) if idle else grey)]
            specs += [('navigation', 'NAVIGATION', (160, 110, 0) if self.mode == 'navigation'
                       else (110, 80, 30) if idle else grey),
                      ('pickmode', 'PICK MODE', (0, 160, 160) if self.mode == 'pick'
                       else (30, 100, 100) if idle else grey)]
            if self.last_touch is not None:
                specs.append(('return', 'RETURN', (0, 0, 200) if idle else grey))
            if self.calibration is not None:
                specs.append(('apply', 'APPLY', (0, 120, 200) if idle else grey))
        self.buttons = {}
        gap = 6
        width = min(146, (w - 16 - gap * (len(specs) - 1)) // len(specs))
        x = 8
        for name, text, colour in specs:
            x1 = x + width
            cv2.rectangle(bar, (x, 6), (x1, 44), colour, -1)
            cv2.rectangle(bar, (x, 6), (x1, 44), (255, 255, 255), 1)
            scale = 0.65
            while cv2.getTextSize(text, FONT, scale, 2)[0][0] > width - 10 and scale > 0.35:
                scale -= 0.05
            (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
            cv2.putText(bar, text, (x + (width - tw) // 2, 25 + th // 2), FONT, scale,
                        (255, 255, 255), 2)
            self.buttons[name] = (x, h + 6, x1, h + 44)
            x = x1 + gap
        if self.pick and not self.calibrating:
            _u, _v, p, arm, _o = self.pick
            info = f'{arm} arm   x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f} m (world)'
            cv2.putText(bar, info, (8, 62), FONT, 0.5, (0, 255, 255), 1)
        if self.detail:
            text = self.detail
            while cv2.getTextSize(text, FONT, 0.4, 1)[0][0] > w - 16 and len(text) > 10:
                text = text[:-4] + '...'
            cv2.putText(bar, text, (8, 78), FONT, 0.4, (200, 200, 200), 1)
        return np.vstack((img, bar))

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        def startup():
            self.say('waiting for move_group to take the gripper/octomap exemption...')
            _ok, message = self.node.ensure_grippers_in_octomap(patience=60.0)
            self.say(message)
            self.capture()
            self.mode = self.detect_mode()
            self.say({'navigation': 'navigation mode: click a point to start picking',
                      'pick': 'pick mode: click a point, then MOVE'}.get(
                self.mode, 'arms are in neither navigation nor pre_pick: press NAVIGATION (n) '
                           'or PICK MODE (p)'))
        self.in_background(startup)
        while rclpy.ok():
            cv2.imshow(WINDOW, self.draw())
            key = cv2.waitKey(30) & 0xFF
            # AUTOSIZE, not VISIBLE: GTK reports VISIBLE as 0 until the window
            # is mapped, which closed this on its first frame. AUTOSIZE is -1
            # only once the window has actually been destroyed.
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                if self.calibrating:
                    self.answer('abort')
                break
            if self.teaching:
                keys = {81: 'left', 83: 'right', 82: 'up', 84: 'down', ord('+'): 'in',
                        ord('='): 'in', ord('-'): 'out', 13: 'save', 10: 'save', 27: 'skip'}
                if key in keys:
                    self.answer(keys[key])
            elif self.calibrating:
                if key in (ord('s'), ord('S')):
                    self.answer('skip')
                elif key in (ord('x'), ord('X'), 27):
                    self.answer('abort')
            elif key in (ord('r'), ord('R')):
                self.in_background(self.capture)
            elif key in (ord('m'), ord('M'), 13, 10):
                self.move()
            elif key in (ord('c'), ord('C')):
                self.in_background(self.calibrate)
            elif key in (ord('o'), ord('O')):
                self.press('overlay')
            elif key in (ord('t'), ord('T')):
                self.press('teach')
            elif key in (ord('f'), ord('F')):
                self.press('markers')
            elif key in (ord('b'), ord('B')):
                self.press('return')
            elif key in (ord('d'), ord('D')):
                self.press('topdown')
            elif key in (ord('n'), ord('N')):
                self.press('navigation')
            elif key in (ord('p'), ord('P')):
                self.press('pickmode')
        cv2.destroyAllWindows()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm', choices=['auto', 'left', 'right'], default='auto',
                        help='auto: left half of the frame -> left arm, right half -> right')
    parser.add_argument('--standoff', type=float, default=0.10,
                        help='the tip first stops this far from the point (corrections '
                             'happen there), then moves in on one straight line, m')
    parser.add_argument('--line-speed', type=float, default=0.02,
                        help='fingertip speed for the straight move in and out, m/s')
    parser.add_argument('--touch-offset', type=float, default=0.0,
                        help='stop the fingertips this far short of the measured surface, m. '
                             'The arm is compliant (~6 N per cm of push), so 0 is safe')
    parser.add_argument('--tool-extension', type=float, default=TOOL_EXTENSION,
                        help='tool fitted past the fingertips (vacuum extension), m; the '
                             'working tip is this much further out')
    parser.add_argument('--tip-offset', type=float, default=None,
                        help='working tip beyond hand_tcp along the tool axis, m (default: '
                             'fingertips 15.4 mm + --tool-extension)')
    parser.add_argument('--close-gripper', action='store_true',
                        help='close the gripper before touching and calibrating, so the '
                             'fingertips meet at one point. Off by default: the gripper is '
                             'never commanded, and the point is aimed midway between the '
                             'fingertips')
    parser.add_argument('--hold', type=float, default=2.0, help='seconds to stay touching')
    parser.add_argument('--live-flags', action='store_true',
                        help='also measure the gripper flags during each touch (only useful '
                             'if the arm does not hide them); normally they are used once, '
                             'by AUTO CAL')
    parser.add_argument('--top-down', action='store_true',
                        help='start with TOP DOWN on: approach straight down from --standoff '
                             'above the point (objects lying flat) instead of level')
    parser.add_argument('--drop-hold', type=float, default=1.0,
                        help='after a touch: seconds at drop_state before pre_pick '
                             '(negative: skip the drop pose, return straight to pre_pick)')
    parser.add_argument('--fast-line-speed', type=float, default=0.06,
                        help='tool speed for the straight line in (to 1 cm short) and out, m/s')
    parser.add_argument('--contact-speed', type=float, default=0.025,
                        help='tool speed for the last, guarded centimetre, m/s')
    parser.add_argument('--past-contact', type=float, default=0.01,
                        help='how far past the measured surface the guarded move may go '
                             'before giving up on feeling it, m')
    parser.add_argument('--contact-torque', type=float, default=0.3,
                        help='joint torque change that counts as touching, Nm (0 = do not '
                             'watch: stop at the measured surface instead)')
    parser.add_argument('--settle-time', type=float, default=1.0,
                        help='how long to let the arm converge before measuring it, s')
    parser.add_argument('--tolerance', type=float, default=0.002,
                        help='fingertip error at the approach worth correcting, m')
    parser.add_argument('--corrections', type=int, default=1,
                        help='correction moves at the approach point, at most')
    parser.add_argument('--no-auto-apply', action='store_true',
                        help='CALIBRATE stops at the result; APPLY writes it')
    parser.add_argument('--calib-points-per-arm', type=int, default=6,
                        help='fingertips CALIBRATE collects per arm before moving on '
                             '(out of 12 candidate spots each)')
    parser.add_argument('--marker-frames', type=int, default=8,
                        help='colour frames averaged for each flag measurement')
    parser.add_argument('--marker-poses-per-arm', type=int, default=3,
                        help='poses per arm for CALIBRATE MARKERS (FLAGS)')
    parser.add_argument('--calib-depths', type=float, nargs=2, default=[0.30, 0.40],
                        metavar=('NEAR', 'FAR'),
                        help='camera distances for the calibration fingertip grid, m')
    parser.add_argument('--velocity', type=float, default=1.0,
                        help='velocity and acceleration scaling for the long moves, 0..1')
    parser.add_argument('--touch-velocity', type=float, default=0.1,
                        help='scaling for the retreat when the straight line out fails')
    parser.add_argument('--planning-time', type=float, default=5.0)
    parser.add_argument('--attempts', type=int, default=3,
                        help='cuMotion resends per goal on PLANNING_FAILED')
    parser.add_argument('--motion-timeout', type=float, default=60.0)
    parser.add_argument('--patch', type=int, default=3,
                        help='depth median half-width around the click, pixels')
    parser.add_argument('--depth-frames', type=int, default=8,
                        help='aligned depth frames median-averaged per capture')
    parser.add_argument('--octomap-settle', type=float, default=1.5,
                        help='seconds to let move_group integrate a refreshed map')
    return parser


def parse_args(argv=None):
    args = make_parser().parse_args(argv)
    if args.tip_offset is None:                  # fingertips beyond hand_tcp, plus the extension
        args.tip_offset = FINGERTIP_BEYOND_TCP + args.tool_extension
    return args


def main():
    args = parse_args()

    rclpy.init()
    node = ClickToMove(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        App(node).run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
