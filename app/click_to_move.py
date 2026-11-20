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
   the spot. So the goal is the working tip (fingertips + that arm's tool) beyond hand_tcp
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
4. MOVE: fly the planned path to the point 10 cm out (`--standoff`), aimed
   with the arm model (CAL ARMS); a correction there from the joints, and
   up to four more 3 cm short; then ONE straight line along the tool axis
   (`--fast-line-speed`), the last 3 cm slowly (`--contact-speed`),
   stopping on contact; hold `--hold` s; straight back out; retrace the approach path
   back to the starting posture. (Gripper closed first with --close-gripper.)

Code layout
-----------
click_to_move.py           the window (App), command-line options, main()
ctm/robot.py               ClickToMove node: ROS/MoveIt interface, motion primitives
ctm/touch.py               reach check, preplan, the touch sequence, TEACH
ctm/armcal.py, armcal_run.py, board.py, kinematics.py
                           CAL ARMS: the arm model, its fit and the calibration run
ctm/camera_calibration.py  CALIBRATE / APPLY / OVERLAY
ctm/corrections.py         TEACH corrections (touch_corrections.yaml)
ctm/modes.py, tools.py, walls.py
                           navigation/pick modes, gripper tools, unseen-area walls
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

from ctm import armcal
from ctm.armcal_run import ArmCalMixin
from ctm.boxcal_run import BoxCalMixin
from ctm.payload import PayloadMixin
from ctm.camera_calibration import CameraCalibrationMixin
from ctm import tools as tl
from ctm import walls
from ctm.common import BASE_FRAME, FONT, TOOL_ARM, MoveGroupDown, WINDOW, decode_image
from ctm.modes import ModesMixin, load_poses
from ctm.robot import ClickToMove
from ctm.touch import TouchMixin


class App(TouchMixin, CameraCalibrationMixin, ModesMixin, ArmCalMixin, BoxCalMixin,
          PayloadMixin):
    """The OpenCV window, and the worker threads that capture and move."""

    HINT = 'click a point   keys: m n p d b r t'
    CALIBRATION = False               # calibrate.py: CALIBRATE, OVERLAY, CAL ARMS, APPLY

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
        self.next_hint = 'Click a point, then MOVE.'   # what to do next (the GUI: detect a box)
        self.pick_tool = node.args.pick_tool   # vacuum (left only) / finger (right only) / auto
        # CALIBRATE ARMS results, if any -- applied per --arm-model (2026-11-17: auto =
        # only a model fitted on this robot as it is now; ctm/armcal.py ArmModels)
        # (its one line per arm goes to the node's log -- the terminal -- and the
        # web GUI's log shows arm_models.notes at start, 2026-11-17)
        # (a lambda: rclpy pins one severity to each call line, and ArmModels logs from
        # one line -- AUTO CAL's reload logs at info, 2026-11-17 review)
        self.arm_models = armcal.ArmModels(getattr(node.args, 'arm_model', None) or 'auto',
                                           log=lambda text: node.get_logger().warn(text))
        self.buttons = {}             # name -> (x0, y0, x1, y1) in window pixels
        self.status = 'capturing...'
        self.detail = ''
        self.busy = False
        # Calibration: the view being clicked, the answer, and a result to apply.
        self.calib_view = None        # (bgr, predicted (u, v) or None, index, total)
        self.calib_answer = None
        self.confirming = False       # CALIBRATE ARMS waiting for CONFIRM / CANCEL
        self.calib_event = threading.Event()
        self.calibration = None
        self.calibrating = False
        self.overlay = None           # (pixels (N,2), agree (N,) bool): robot model on the image
        self.teach = False            # TEACH mode: hold each touch for nudges
        self.tip_seen = None          # the camera's last view of a tip (ctm/tip_vision.py)
        self.vision_miss = None
        self.top_down = node.args.top_down   # approach straight down instead of level
        self.teaching = False         # a touch is being taught right now
        self.abort = threading.Event()
        self._job_lock = threading.Lock()

    def phase(self, step):
        """The step running now ("left contact", ...): pick_stack.py publishes it."""

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
        """work() in a thread, one job at a time. False if one runs already."""
        with self._job_lock:                         # (GUI and ROS threads both start jobs)
            if self.busy:
                return False
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
                self.phase('idle')
        threading.Thread(target=run, daemon=True).start()
        return True

    def halted(self):
        """STOP was pressed (or ABORT): a job checks this between its steps."""
        return self.abort.is_set() or self.node.halt.is_set()

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
        args = self.node.args
        # the walls' edges where the depth ends: aligned depth has none in a band
        # down each side and along the top, which the octomap cannot fill
        bounds = walls.depth_bounds(averaged) if args.walls_edges == 'depth' and \
            averaged.shape[:2] == (info.height, info.width) else None
        n = self.node.refresh_walls(self.frame[3], args.walls_ahead, args.walls_x_min, bounds)
        walls_text = ('' if n is None else
                      '; walls off: the 4D LiDAR sees round the picture' if getattr(
                          self.node, 'walls_off', False) else f'; walls around the picture ({n})')
        self.say(f'{message}' + walls_text
                 + f'. {self.next_hint}')

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
        if v >= self.frame[0].shape[0]:
            return                                   # the button bar, off a button
        check = self.select_point(u, v)
        if check is None:
            return
        if self.mode != 'pick':
            # First point: into pick mode (both arms to pre_pick), then the check.
            self.in_background(lambda: self.ensure_pick_mode() and
                               self.check_pick(check, in_job=True))
        else:
            threading.Thread(target=self.check_pick, args=(check,), daemon=True).start()

    def select_point(self, u, v, world=None):
        """Pixel (u, v) of the captured frame -> the point to touch (world,
        from the depth there, or `world` if given) and its arm (by the half
        of the frame). Sets self.pick and a fresh self.check (not run yet);
        returns the check, or None if there is no depth there."""
        bgr, depth, info, frame_id = self.frame
        dh, dw = depth.shape
        bh, bw = bgr.shape[:2]
        try:
            point, origin, z = self.node.pixel_to_world(
                int(u * dw / bw), int(v * dh / bh), depth, info, frame_id)
        except Exception as exc:                     # noqa: BLE001
            if world is None:
                self.pick = None
                self.say(f'pixel ({u:.0f}, {v:.0f}): {exc}', True)
                return None
            m = self.node.lookup(BASE_FRAME, frame_id)
            origin, z = m[:3, 3], float(np.linalg.norm(np.asarray(world) - m[:3, 3]))
        if world is not None:
            point = np.asarray(world, float)
        if self.pick_tool in TOOL_ARM:
            # the tool fixes the arm, wherever the point is: a vacuum pick is
            # the left arm's, and the right arm never moves for it (2026-10-28)
            arm = TOOL_ARM[self.pick_tool]
        elif self.node.args.arm != 'auto':
            arm = self.node.args.arm
        else:
            arm = 'left' if u < bw / 2.0 else 'right'
        self.pick = (int(u), int(v), point, arm, origin)
        self.node.publish_marker(point)
        where = f'x={point[0]:+.3f} y={point[1]:+.3f} z={point[2]:+.3f} (depth {z:.2f} m)'
        self.say(f'{where}, {arm} arm - checking it can be touched...')
        check = {'pick': self.pick, 'where': where, 'ok': None, 'plan': None}
        self.check = check
        return check

    def on_key(self, key):
        """A key the window loop does not know: True if handled (subclasses)."""
        return False

    def toggle_pieces(self):
        """ARUCO EXT: the calibration pieces (still on the grippers) in the
        planners, or not -- the same as calibrate.py has them, ~4 cm past the
        rail's ends and towards the fingers."""
        if self.busy:
            self.say('ARUCO EXT: not while an arm is moving', True)
            return
        on = not self.node.pieces_on

        def job():
            self.node._pieces_off_done = False          # (off is otherwise sent once)
            if not self.node.set_pieces(on):
                self.say('ARUCO EXT: the planners did not take the change - try again', True)
                return
            self.node.pieces_on = on
            self.say('ARUCO EXT on: the planners keep the ArUco pieces clear of everything'
                     if on else 'ARUCO EXT off: the planners know only the gripper')
            if self.pick is not None and self.mode == 'pick':
                # the last check planned without (or with) them: check again
                _u, _v, p, _arm, _o = self.pick
                check = {'pick': self.pick, 'ok': None, 'plan': None,
                         'where': f'x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}'}
                self.check = check
                self.check_pick(check, in_job=True)
        self.in_background(job)

    def change_tool(self, arm, toggle=False, step=0.0):
        """Switch an arm's tool on/off or change its length; the touches use
        the new working tip from now on."""
        if self.busy:
            self.say('tool: not while an arm is moving', True)
            return
        t = self.node.args.tools[arm]
        if toggle:
            t['on'] = not t['on']
        elif step:
            t['on'] = True
            t['length'] = float(min(tl.MAX_LENGTH, max(tl.STEP, t['length'] + step)))
        tl.apply(self.node.args, self.node.args.tools)
        tl.save(self.node.args.tools)
        # the planners' tool too (a thread: it waits on move_group and cuMotion)
        threading.Thread(target=self.node.set_tool, daemon=True).start()
        self.say(f'{arm} arm: ' + (f'tool on, {t["length"] * 1000:.0f} mm past the fingertips'
                                   if t['on'] else 'no tool: the fingertips touch')
                 + (f' ({"[ ]" if arm == "left" else ", ."} change the length)'))
        self.check = None                       # the last check was for the old tip

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
        elif name in ('tool_left', 'tool_right'):
            self.change_tool(name[5:], toggle=True)
        elif name == 'pieces':
            self.toggle_pieces()
        elif name == 'skip':
            self.answer('skip')
        elif name == 'confirm':
            self.answer('confirm')
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
            self.in_background(self.calibrate_arms)
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
                    f'{self.HINT}')
        for i, text in enumerate((head + ('  [busy]' if self.busy else ''), self.status)):
            y = 22 + 22 * i
            cv2.putText(img, text, (8, y), FONT, 0.5, (0, 0, 0), 3)
            cv2.putText(img, text, (8, y), FONT, 0.5, (255, 255, 255), 1)
        return self.with_buttons(img)

    def main_buttons(self, idle, grey):
        """[(name, label, colour)] of the button bar when nothing special is on."""
        ok = self.check['ok'] if self.check is not None else None
        move_colour = (grey if not (idle and self.pick) else (0, 150, 0) if ok
                       else (0, 120, 200) if ok is None else (0, 0, 150))
        specs = [('move', 'MOVE', move_colour),
                 ('recapture', 'RECAPTURE', (150, 90, 0) if idle else grey)]
        if self.CALIBRATION:
            specs += [('calibrate', 'CALIBRATE', (140, 0, 140) if idle else grey),
                      ('overlay', 'OVERLAY', (0, 140, 140) if idle or self.overlay else grey),
                      ('markers', 'CAL ARMS', (0, 140, 0) if self.arm_models.has('left')
                       and idle else (100, 100, 40) if idle else grey)]
        specs += [('topdown', 'TOP DOWN ON' if self.top_down else 'TOP DOWN',
                   (0, 90, 230) if self.top_down else (70, 70, 120)),
                  ('teach', 'TEACH ON' if self.teach else 'TEACH',
                   (0, 100, 220) if self.teach else (70, 70, 120))]
        specs += [('navigation', 'NAVIGATION', (160, 110, 0) if self.mode == 'navigation'
                   else (110, 80, 30) if idle else grey),
                  ('pickmode', 'PICK MODE', (0, 160, 160) if self.mode == 'pick'
                   else (30, 100, 100) if idle else grey)]
        if self.last_touch is not None:
            specs.append(('return', 'RETURN', (0, 0, 200) if idle else grey))
        if self.CALIBRATION and self.calibration is not None:
            specs.append(('apply', 'APPLY', (0, 120, 200) if idle else grey))
        return specs

    def with_buttons(self, img):
        """The frame with a button bar under it (OpenCV has no widgets)."""
        h, w = img.shape[:2]
        row2 = not self.teaching and not self.calibrating
        bar = np.full((84 + (42 if row2 else 0), w, 3), 40, np.uint8)
        grey = (90, 90, 90)
        if self.teaching:
            nudge = (120, 90, 0)
            specs = [(n, n.upper(), nudge) for n in ('left', 'right', 'up', 'down', 'in', 'out')]
            specs += [('save', 'SAVE', (0, 150, 0)), ('skip', 'SKIP', (90, 90, 90))]
        elif self.calibrating and getattr(self, 'confirming', False):
            specs = [('confirm', 'CONFIRM', (0, 150, 0)), ('abort', 'CANCEL', (0, 0, 170))]
        elif self.calibrating:
            specs = [('abort', 'ABORT', (0, 0, 170))]
        else:
            specs = self.main_buttons(not self.busy, grey)
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
        if row2:
            # the tools, on a second row: click = on/off, [ ] and , . = length
            x = 8
            for arm in ('left', 'right'):
                t = self.node.args.tools[arm]
                x1 = x + 150
                colour = (0, 110, 160) if t['on'] else (70, 70, 70)
                cv2.rectangle(bar, (x, 50), (x1, 82), colour, -1)
                cv2.rectangle(bar, (x, 50), (x1, 82), (255, 255, 255), 1)
                text = tl.label(arm, t)
                (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
                cv2.putText(bar, text, (x + (150 - tw) // 2, 66 + th // 2), FONT, 0.5,
                            (255, 255, 255), 1)
                self.buttons[f'tool_{arm}'] = (x, h + 50, x1, h + 82)
                x = x1 + gap
            if not self.CALIBRATION:                  # (calibrate.py: always on)
                # ARUCO EXT: the calibration pieces in the planners
                x1 = x + 130
                on = self.node.pieces_on
                cv2.rectangle(bar, (x, 50), (x1, 82), (0, 100, 220) if on else (70, 70, 70), -1)
                cv2.rectangle(bar, (x, 50), (x1, 82), (255, 255, 255), 1)
                text = 'ARUCO EXT ON' if on else 'ARUCO EXT OFF'
                (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
                cv2.putText(bar, text, (x + (130 - tw) // 2, 66 + th // 2), FONT, 0.5,
                            (255, 255, 255), 1)
                self.buttons['pieces'] = (x, h + 50, x1, h + 82)
                x = x1 + gap
            cv2.putText(bar, 'tool length: [ ] left   , . right', (x + 6, 70), FONT, 0.4,
                        (200, 200, 200), 1)
        y0 = 42 if row2 else 0
        if self.pick and not self.calibrating:
            _u, _v, p, arm, _o = self.pick
            info = f'{arm} arm   x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f} m (world)'
            cv2.putText(bar, info, (8, 62 + y0), FONT, 0.5, (0, 255, 255), 1)
        if self.detail:
            text = self.detail
            while cv2.getTextSize(text, FONT, 0.4, 1)[0][0] > w - 16 and len(text) > 10:
                text = text[:-4] + '...'
            cv2.putText(bar, text, (8, 78 + y0), FONT, 0.4, (200, 200, 200), 1)
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
                if key in (ord('y'), ord('Y'), 13, 10) and getattr(self, 'confirming', False):
                    self.answer('confirm')
                elif key in (ord('s'), ord('S')):
                    self.answer('skip')
                elif key in (ord('x'), ord('X'), 27):
                    self.answer('abort')
            elif key in (ord('r'), ord('R')):
                self.in_background(self.capture)
            elif key in (ord('m'), ord('M'), 13, 10):
                self.move()
            elif key in (ord('c'), ord('C')) and self.CALIBRATION:
                self.in_background(self.calibrate)
            elif key in (ord('o'), ord('O')) and self.CALIBRATION:
                self.press('overlay')
            elif key in (ord('t'), ord('T')):
                self.press('teach')
            elif key in (ord('f'), ord('F')) and self.CALIBRATION:
                self.press('markers')
            elif key in (ord('b'), ord('B')):
                self.press('return')
            elif key in (ord('d'), ord('D')):
                self.press('topdown')
            elif key in (ord('n'), ord('N')):
                self.press('navigation')
            elif key in (ord('p'), ord('P')):
                self.press('pickmode')
            elif self.on_key(key):
                pass
            elif key in (ord('['), ord(']'), ord(','), ord('.')):
                arm = 'left' if key in (ord('['), ord(']')) else 'right'
                self.change_tool(arm, step=tl.STEP if key in (ord(']'), ord('.')) else -tl.STEP)
        cv2.destroyAllWindows()


def on_off(parser, flag, default, text):
    """An on/off option (2026-11-17): `--flag on|off`, and `--no-flag` for off.
    The value stays the string 'on'/'off' (read it with an _on() helper: 'off' is a
    true string)."""
    dest = flag[2:].replace('-', '_')
    parser.add_argument(flag, dest=dest, choices=('on', 'off'), default=default,
                        help=f'{text} (default {default})')
    parser.add_argument('--no-' + flag[2:], dest=dest, action='store_const', const='off',
                        help=f'the same as {flag} off')


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm', choices=['auto', 'left', 'right'], default='auto',
                        help='auto: left half of the frame -> left arm, right half -> right')
    parser.add_argument('--pick-tool', choices=['auto', 'vacuum', 'finger'], default='auto',
                        help='vacuum: every point is the left arm\'s, the right arm never moves '
                             'for a pick; finger: the right arm\'s; auto: by --arm')
    parser.add_argument('--standoff', type=float, default=0.10,
                        help='the farthest the tip first stops from the point (corrections '
                             'happen there), then moves in on one straight line, m')
    parser.add_argument('--tip-vision', choices=['off', 'log', 'on'], default='off',
                        help='the camera\'s check of the tip before going in (ctm/tip_vision.py:'
                             ' the colour outline, plus depth where it has readings). log: '
                             'measure, report and save only; on: also move the tip onto the '
                             'line; off: not at all')
    parser.add_argument('--tip-vision-max', type=float, default=0.03,
                        help='the camera\'s view of the tip is trusted up to this far off the '
                             'line in, m (more: a misreading, the arm model is used)')
    parser.add_argument('--correction-room', type=float, default=0.025,
                        help='before moving, the whole straight line in must still solve with '
                             'its ends moved this far sideways (the corrections move it), m; '
                             'plans with only 1 cm are a last resort, none: not moving')
    parser.add_argument('--max-miss', type=float, default=0.010,
                        help='the most the tip may still be off the target 5 cm short, after '
                             'the corrections, and still go in, m')
    parser.add_argument('--min-standoff', type=float, default=0.04,
                        help='the nearest the straight line in may start from the point: the '
                             'approach point is the farthest from --standoff / --top-standoff '
                             'down to this that the arm reaches with the whole line in clear, '
                             'm. Under 3 cm (only when nothing further out works) the tool may be '
                             'near the box: the flight there is checked with the tool counted up '
                             'to its last 15 %%')
    parser.add_argument('--top-standoff', type=float, default=0.12,
                        help='TOP DOWN: the tip flies (tool level) to this far above the top, '
                             'the wrist turns down there, then it goes straight down, m')
    parser.add_argument('--top-x-offset', type=float, default=-0.015,
                        help='TOP DOWN: the point the tool goes down onto, moved this far along the '
                             'world x (straight ahead, towards the boxes), m; negative = nearer the '
                             'robot. -0.015: it went in 1.5 cm too deep (2026-11-06); 0 = as measured')
    parser.add_argument('--vacuum-lean', type=float, default=30,
                        help='TOP DOWN with a tool on: the most the tool may lean off '
                             'straight down (deg). Straight down first, then the smallest '
                             'lean that reaches; default no real cap (bellows cup). '
                             '0 = straight down only')
    parser.add_argument('--line-speed', type=float, default=0.02,
                        help='fingertip speed for the straight move in and out, m/s')
    parser.add_argument('--touch-offset', type=float, default=0.0,
                        help='stop the fingertips this far short of the measured surface, m. '
                             'The arm is compliant (~6 N per cm of push), so 0 is safe')
    for arm in ('left', 'right'):
        parser.add_argument(f'--tool-extension-{arm}', type=float, default=None,
                            help=f'tool fitted past the {arm} fingertips, m (0 = none); default: '
                                 f'as last set in the window (tools.yaml)')
    parser.add_argument('--close-gripper', action='store_true',
                        help='close the gripper before touching and calibrating, so the '
                             'fingertips meet at one point. Off by default: the gripper is '
                             'never commanded, and the point is aimed midway between the '
                             'fingertips')
    parser.add_argument('--hold', type=float, default=0.8,
                        help='seconds to stay touching (the vacuum seals in well under 1 s)')
    parser.add_argument('--top-down', action='store_true',
                        help='start with TOP DOWN on: approach straight down from --standoff '
                             'above the point (objects lying flat) instead of level')
    parser.add_argument('--drop-hold', type=float, default=1.0,
                        help='after a touch: seconds at drop_state before pre_pick '
                             '(negative: skip the drop pose, return straight to pre_pick)')
    parser.add_argument('--fast-line-speed', type=float, default=0.08,
                        help='tool speed for the straight line in (to 1 cm short) and out, m/s')
    parser.add_argument('--contact-speed', type=float, default=0.015,
                        help='tool speed for the last, guarded centimetre, m/s')
    parser.add_argument('--past-contact', type=float, default=0.02,
                        help='how far past the measured surface the guarded move may go '
                             'before giving up on feeling it, m')
    parser.add_argument('--contact-torque', type=float, default=0.3,
                        help='joint torque change that counts as touching, Nm (0 = do not '
                             'watch: stop at the measured surface instead)')
    parser.add_argument('--max-touch-torque', type=float, default=2.0,
                        help='hard limit while touching: stop at once if any joint torque '
                             'rises this much above its level before the move, Nm')
    parser.add_argument('--soft-tool-torque', type=float, default=1.0,
                        help='the torque limit while touching with a tool on (the rubber '
                             'vacuum extension must not be crushed or bent), Nm')
    parser.add_argument('--settle-time', type=float, default=1.0,
                        help='how long to let the arm converge before measuring it, s')
    parser.add_argument('--tolerance', type=float, default=0.002,
                        help='fingertip error at the approach worth correcting, m')
    parser.add_argument('--corrections', type=int, default=1,
                        help='correction moves at the approach point, at most')
    parser.add_argument('--plan-budget', type=float, default=None,
                        help='seconds the reach check may search; nothing by then: not from '
                             'here (the pick stack then asks the wheels: base_adjust). '
                             'Default 10 with --reach-search fast, 30 with full')
    parser.add_argument('--edge', choices=['try', 'adjust'], default='try',
                        help='a point reachable only at the edge of reach (no room to correct '
                             'the tool): try it, or count it as not reachable from here '
                             '(adjust: the pick stack asks the wheels to move)')
    parser.add_argument('--prefer-top', action=argparse.BooleanOptionalAction, default=True,
                        help='a FRONT target on a face narrower than --narrow-face-mm, of a box '
                             'with nothing on it and a wide top (box size known), is picked from '
                             'the TOP instead')
    parser.add_argument('--narrow-face-mm', type=float, default=45.0,
                        help='front faces narrower than this (short side, mm) prefer the top')
    parser.add_argument('--edge-face-mm', type=float, default=25.0,
                        help='an edge-of-reach plan (no room to correct) is still flown when the '
                             'face to grip is at least this wide (short side, mm). 25: the cup fits '
                             'on it -- the user asked to pick a 28 mm face there rather than move the '
                             'base (2026-10-30); the arm still goes in only within --max-miss of it')
    parser.add_argument('--min-depth', type=float, default=0.35,
                        help='nearer to the camera than this (m), its depth is not trusted: '
                             'not picked from here')
    parser.add_argument('--press-retries', type=int, default=2,
                        help='presses again, 5 mm deeper and 0.5 s longer, when the joint '
                             'torques show nothing on the cup after a press')
    parser.add_argument('--held-min-g', type=float, default=30.0,
                        help='weight on the cup (from the joint torques, held still before '
                             'and after the press) that counts as holding the box, g')
    parser.add_argument('--near-corrections', type=int, default=0,
                        help='sideways correction moves 3 cm short of the box, at most (was 2 '
                             'until 2026-11-13). 0: from 5 cm short the way in is one straight '
                             'line -- nothing moves sideways next to the neighbouring boxes')
    parser.add_argument('--face-first', action=argparse.BooleanOptionalAction, default=True,
                        help='FRONT pick of a detected box: try the approach square on to its '
                             'measured face first, before the ones along the camera\'s view')
    parser.add_argument('--direct-path', action=argparse.BooleanOptionalAction, default=False,
                        help='FRONT: when no straight line in works from any approach point '
                             '(--standoff down to --min-standoff), fly a planned path straight to '
                             '3 cm in front of the face (checked with the tool counted up to its '
                             'last 15 %%), then only the press. OFF since 2026-11-13: tried once, a box at the edge of '
                             'reach arrived 22 mm off 3 cm from it, and the tool pushed its neighbours')
    parser.add_argument('--short-corrections', type=int, default=3,
                        help='correction moves 5 cm short of the box, at most (each ~1.5 s)')
    parser.add_argument('--short-tolerance', type=float, default=0.004,
                        help='fingertip error 5 cm short worth another correction, m (the '
                             '25 mm cup centred within 4 mm still seals)')
    # -- precision (2026-11-17): each new behaviour with its switch back ---------
    parser.add_argument('--correct-mode', choices=('settle', 'hybrid', 'cartesian'),
                        default='settle',
                        help='FRONT picks, how the arrival out at the approach and 5 cm short is '
                             'handled. settle: let the arm settle onto its command and measure it, '
                             'no encoder-based corrections (they measured servo error and added it '
                             'to the command for good: offsets up to 40 mm, 8-13 s a pick, '
                             '2026-11-16); hybrid: settle, then at most one correction 5 cm short '
                             '(60 %%, capped to --correction-room); cartesian: the corrections as '
                             'before. TOP DOWN always cartesian')
    parser.add_argument('--approach-settle', type=float, default=1.5,
                        help='settle/hybrid: the longest wait for the arm to settle at the '
                             'approach point, s (2026-11-18: 1.5 -- it always ran out at 2.5 with '
                             'the tip 3-7 mm off, and the 5 cm stop is the one that decides)')
    parser.add_argument('--stop-on-motor-drop', choices=('on', 'off'), default='on',
                        help='STOP at once when an arm motor stops driving (status 1 -> 0 or a '
                             'fault; 2026-11-18: left joint2 dropped and PICK ALL went on for a '
                             'minute with the shoulder limp)')
    parser.add_argument('--short-settle', type=float, default=3.0,
                        help='settle/hybrid: the same 5 cm short of the box, s')
    parser.add_argument('--abort-joint-err', type=float, default=0.10,
                        help='after the approach and 5 cm short: any joint further than this off '
                             'its command and the arm does not go in, it comes back, rad')
    on_off(parser, '--stored-segments', 'on',
           'fly the straight line in and the press exactly as the reach check verified them '
           '(re-checked if the map changed); off: planned again from where the arm is')
    on_off(parser, '--press-hand-free', 'on',
           'the press along the tool\'s own axis may have the hand touch the map too (only '
           'where the exempt tool went first; checked coaxial within --press-coax-deg/-mm); '
           'off: only the fingers and the tool, as before')
    parser.add_argument('--press-coax-deg', type=float, default=3.0,
                        help='a press is coaxial within this angle of the tool axis, deg')
    parser.add_argument('--press-coax-mm', type=float, default=3.0,
                        help='... and this far sideways of it, mm')
    parser.add_argument('--reach-search', choices=('fast', 'full'), default='fast',
                        help='FRONT reach check. fast: every way in judged in numbers first, '
                             'move_group asked only about the best few (ctm/reach_fast.py, ~2-4 s); '
                             'full: the search as before (13-32 s a box)')
    parser.add_argument('--limit-margin', type=float, default=0.10,
                        help='fast reach check: a way in is first choice only with this much room '
                             'from every joint limit at the approach, 3 cm short and the face, rad')
    on_off(parser, '--free-roll', 'on',
           'fast reach check: also every 15 deg of roll about each approach axis')
    parser.add_argument('--arm-model', choices=armcal.MODES, default='auto',
                        help='the CALIBRATE ARMS model (calibration/arm_model.yaml). auto: only '
                             'an arm\'s model fitted on this robot as it is now (hand mount, tool, '
                             'camera, masses); on: whatever is in the file (as before 2026-11-17); '
                             'off: the plain URDF')
    parser.add_argument('--return-to', choices=('nominal', 'measured'), default='nominal',
                        help='after a pick: back to the nominal pre_pick_state (when it started '
                             'within 0.15 rad of it), or to the posture it started from')
    parser.add_argument('--no-drop-x', nargs=2, type=float, default=[87.0, 152.0],
                        metavar=('MIN', 'MAX'),
                        help='cm: while x (--vertical-topic) reads between these, a held box is '
                             'let go at pre_pick, not at the drop pose -- the box under the arms '
                             'is then too high, the drop pose would collide (2026-11-19). '
                             'No reading for 3 s: the drop pose as usual. "0 0": off')
    parser.add_argument('--vertical-topic', default='/c60000004/vertical_distance',
                        help='sensor_msgs/Range: x = range x 100 cm (robot/height_box.py reads it too)')
    on_off(parser, '--spot-check', 'on',
           'the camera on the box\'s spot before the touch and once the arm is back: the box '
           'still there -> nothing to take to the drop; gone -> to the drop even if the cup '
           'weighed nothing (the weighing misses boxes on the cup)')
    on_off(parser, '--skip-empty-drop', 'on',
           'nothing held and the cup never pressed: straight back to pre_pick, not via the drop')
    parser.add_argument('--lead-in-max', type=float, default=0.10,
                        help='a plan that starts this far from where the arm is (any joint) is led '
                             'in onto its start by a collision-checked joint move; further: '
                             'refused and planned again, rad (never more than 0.10)')
    parser.add_argument('--held-size', choices=('model', 'off'), default='model',
                        help='the box on the cup in the planners: held_object/size, else the size '
                             'the box model measured (model); off: only held_object/size')
    parser.add_argument('--held-depth-mm', type=float, default=150.0,
                        help='--held-size model: the depth taken when the model did not see it, mm')
    parser.add_argument('--held-margin-mm', type=float, default=10.0,
                        help='--held-size model: added to each side of the held box, mm')
    on_off(parser, '--map-recheck', 'on',
           'a plan checked before the map was rebuilt is checked again against the new map '
           'before it is flown (or planned again); off: flown as it was checked')
    parser.add_argument('--no-auto-apply', action='store_true',
                        help='CALIBRATE stops at the result; APPLY writes it')
    parser.add_argument('--calib-points-per-arm', type=int, default=6,
                        help='fingertips CALIBRATE collects per arm before moving on '
                             '(out of 12 candidate spots each)')
    parser.add_argument('--calib-depths', type=float, nargs=2, default=[0.30, 0.40],
                        metavar=('NEAR', 'FAR'),
                        help='camera distances for the calibration fingertip grid, m')
    parser.add_argument('--velocity', type=float, default=1.0,
                        help='velocity and acceleration scaling for the long moves, 0..1')
    parser.add_argument('--touch-velocity', type=float, default=0.1,
                        help='scaling for the retreat when the straight line out fails')
    parser.add_argument('--held-velocity', type=float, default=0.4,
                        help='velocity scaling (acceleration: its square) for the moves with the '
                             'box maybe on the cup, until the drop (the retrace out from beside the '
                             'others): at 1.0 the yank swung it into the boxes beside it '
                             '(2026-10-30); 0.5 was tried 2026-11-05 and put back (accuracy first)')
    parser.add_argument('--drop-velocity', type=float, default=0.7,
                        help='velocity scaling (acceleration too) for the move from pre_pick to the '
                             'drop pose with the box maybe on the cup: clear of the other boxes by '
                             'then, so faster than --held-velocity (0.4 took 5 s, 2026-11-05)')
    parser.add_argument('--correct-speed', type=float, default=0.02,
                        help='tool speed for the correction moves at the approach point and short '
                             'of the box, m/s. 0.04 was tried (2026-11-05): the corrections 5 cm '
                             'short converged slower (5-8 mm left after the first, against < 3.5)')
    parser.add_argument('--held-line-speed', type=float, default=0.03,
                        help='tool speed for the straight lines out with the box maybe on the '
                             'cup, m/s (--fast-line-speed otherwise)')
    parser.add_argument('--top-carry-speed', type=float, default=0.05,
                        help='after a TOP pick: tool speed of the straight line to pre_pick_top '
                             '(the tool kept pointing down, the box level), m/s')
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
    parser.add_argument('--walls-edges', choices=('depth', 'picture'), default='depth',
                        help="the unseen-space walls (green in RViz) start where the depth "
                             "ends (about 45 px left, 60 px right, 9 rows at the top of "
                             "the colour picture have none) or at the picture's edges "
                             "(as before 2026-11-05: more reach, but objects in those bands "
                             "are invisible to the planners)")
    parser.add_argument('--lidar-walls', choices=('auto', 'always'), default='auto',
                        help='auto: no unseen-space walls while the Unitree L2 LiDAR works (it '
                             'sees round the picture; /lidar/status), back on as soon as it does '
                             'not; always: the walls whatever the LiDAR does')
    parser.add_argument('--walls-ahead', type=float, default=walls.AHEAD,
                        help='the walls start this far in front of the nearest thing seen, m')
    parser.add_argument('--walls-x-min', type=float, default=walls.X_MIN,
                        help='... never nearer the robot than this, m')
    return parser


def parse_args(argv=None):
    return finish_args(make_parser().parse_args(argv))


def finish_args(args):
    """What the options imply: each arm's tools and working tip."""
    # each arm's working tip beyond hand_tcp: the fingertips, plus its tool
    args.tools = tl.load()
    # the reach check's budget: the fast search decides in ~2-4 s, and "not from here"
    # should not wait 30 s for it (2026-11-17); the full search keeps its 30 s
    if getattr(args, 'plan_budget', None) is None:
        args.plan_budget = 10.0 if getattr(args, 'reach_search', 'fast') == 'fast' else 30.0
    args.lead_in_max = min(float(getattr(args, 'lead_in_max', 0.10)), 0.10)
    for arm in ('left', 'right'):
        length = getattr(args, f'tool_extension_{arm}')
        if length is not None:
            args.tools[arm] = {'on': length > 0, 'length': length if length > 0 else 0.020}
    tl.apply(args, args.tools)
    return args


def main(args=None, app_class=None):
    args = parse_args() if args is None else args

    rclpy.init()
    node = ClickToMove(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        (app_class or App)(node).run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
