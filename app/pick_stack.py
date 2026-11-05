#!/usr/bin/env python3
"""PICK STACK: the arms as a ROS 2 module -- the same planning, touch and
drop sequence as click_to_move.py, driven by topics, with or
without a window.

    native/run_pick_stack.sh          (the robot stack must be up: run_launch_everything.sh)
    ros2 launch launch/pick_stack.launch.py gui:=auto

The interface -- everything under /beast_arm (--ns):

  arm_state            std_msgs/String, out, latched -- where the arms are,
                       on every change and every --state-period s:
                         navigation, pre_pick            at that pose
                         moving_to_navigation, moving_to_pre_pick
                         detecting, picking, returning   during a pick
                         arm_out    a pick stopped half-way: send "return"
                         unknown    neither pose (moved by hand, at boot)
  arm_state/cmd        std_msgs/String, in -- "navigation", "pre_pick",
                       "return" (bring an arm left out back), "status"
  arm_state/get        std_srvs/Trigger -- the state, on request
  arm_action           std_msgs/String, in -- "ready_to_pick" ["top"|"front"|"auto"]
                       ["vacuum"|"finger"]: both arms to pre_pick, the model
                       finds the box, the tool's arm picks it, drops it, back to
                       pre_pick, then the camera checks the box is gone. The tool
                       fixes the arm: vacuum -> left, finger -> right (default
                       --pick-tool, vacuum); the pick itself never moves the
                       other arm, a target in the right half included.
                       "stop": cancel all motion
  arm_action/status    std_msgs/String, out, latched -- busy, detecting,
                       target_found, picking, checking, pick_complete, stopped,
                       no_target: <why>, pick_failed: <why>,
                       pick_unverified: <why> (touched, but the camera could not
                       tell -- arm in view, camera moved)
  base_adjust          pick_interfaces/BaseAdjust, out (not latched: never a stale
                       order) -- after --adjust-after (2) picks in a row that did not
                       take the same box, or when it is out of reach from here: how
                       far the robot should move SIDEWAYS (direction left / right /
                       none, distance_m, left_m; never forward or a turn) so the box
                       lines up with the picking arm's sweet spot (--sweet-spot). The
                       arms do nothing more on their own: after the move,
                       ready_to_pick again
  vacuum               std_msgs/Bool, out, latched -- what the vacuum should be: on
                       just before the cup goes in, off at the drop pose (the ESP32
                       relay follows it: app/vacuum_bridge.py, in terminal 1)
  pick_result          std_msgs/String, out, latched -- each pick's outcome, JSON:
                       {picked: true/false/null, outcome (success, replaced, missed,
                       moved, dropped, wrong_box, blocked, camera_moved, unknown,
                       no_contact), contact, tool, arm, approach, xyz, box_mm,
                       message, verdict (the camera check's numbers)}
  arm_action/phase     std_msgs/String, out, latched -- the step running now,
                       "<arm> <step>": approach, turn_down, correct, in,
                       contact, holding, off_surface, back_out, box_clear,
                       turn_level, return, to_drop, drop, to_pre_pick, done;
                       "both to_navigation" / "both to_pre_pick"; "idle"
  stop                 std_msgs/Empty, in -- the same as arm_action "stop"
  held_object/size     pick_interfaces/ObjectSize, in -- height x width x
                       depth mm of the box to pick next; at the contact it is
                       attached to the tool for the planners, released at the
                       drop pose. Zeros: no object
  held_object/state    std_msgs/String, out, latched -- "none", or
                       "<arm> <h>x<w>x<d> mm"
  held_object/release  std_msgs/Empty, in -- take the held box out of the
                       planners now (it was put down some other way)
  pick_target          pick_interfaces/PickTarget, out -- what the model found,
                       with tool and arm filled in (always matching) and box_mm:
                       the known size matched to the measured face (170 x 100 x
                       25 seen as 168 x 99 -> along 170, across 100, 25 deep),
                       which is how the box is attached to the tool at contact
  status               std_msgs/String, out, latched -- all of the above as
                       one JSON object, on change and every second
  log                  std_msgs/String, out -- every message, as the window
                       shows it

The model is the <ns>/box_model/detect service (pick_interfaces/DetectTarget):
box_model_node.py, or any node that serves it; the check after the pick is its
<ns>/box_model/check (pick_interfaces/CheckPick), if served.

The window (--gui auto: if there is a display): the live camera, the state,
the last target. Keys: p pre_pick, n navigation, space ready_to_pick,
b return, q quit. Without a display it runs the same, headless.
"""

import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger

from pick_interfaces.msg import BaseAdjust, ObjectSize, PickTarget
from pick_interfaces.srv import CheckPick, DetectTarget

import click_to_move as ctm_app
from ctm import box_match
from ctm.common import BASE_FRAME, FONT, TOOL_ARM, decode_image
from ctm.robot import ClickToMove

WINDOW = 'pick stack'
MOVING = {'navigation': 'moving_to_navigation', 'pick': 'moving_to_pre_pick'}
STATE_OF_MODE = {'navigation': 'navigation', 'pick': 'pre_pick'}


class PickStack(ctm_app.App):
    """click_to_move's App, driven by topics instead of clicks."""

    def __init__(self, node):
        super().__init__(node)
        args = node.args
        self.activity = None          # what runs now: moving_to_*, detecting, picking, returning
        self.state = None             # last published arm_state
        self.target = None            # the last PickTarget
        self.result = ''              # the last arm_action/status
        self.step = 'idle'            # the last arm_action/phase
        self._held_said = None
        self.next_hint = 'Waiting for ready_to_pick.'
        self.pick_result = None       # the last pick's outcome (pick_result)
        self.node.pick_box = None     # the matched box for the next contact (ctm/touch.py)
        self.node.pick_face_mm = None  # ... and its face's short side, mm
        self._status_said = None
        ns = args.ns.rstrip('/')
        self.ns = ns
        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        pub = lambda kind, name, qos=10: node.create_publisher(kind, f'{ns}/{name}', qos)
        sub = lambda kind, name, fn: node.create_subscription(kind, f'{ns}/{name}', fn, 10,
                                                              callback_group=cb)
        self.state_pub = pub(String, 'arm_state', latched)
        self.status_pub = node.create_publisher(
            String, args.status_topic or f'{ns}/arm_action/status', latched)
        self.phase_pub = pub(String, 'arm_action/phase', latched)
        self.held_pub = pub(String, 'held_object/state', latched)
        self.json_pub = pub(String, 'status', latched)
        self.log_pub = pub(String, 'log')
        self.target_pub = pub(PickTarget, 'pick_target')
        self.result_pub = pub(String, 'pick_result', latched)
        self.base_pub = pub(BaseAdjust, 'base_adjust')   # (volatile: no stale orders)
        self.model_cmd_pub = pub(String, 'box_model/cmd')  # flag / skip a box given up on
        self.vacuum_pub = pub(Bool, 'vacuum', latched)    # the ESP32 relay: vacuum_bridge.py
        self.vacuum_on = False
        self.fails = {'xyz': None, 'outcomes': []}     # failed picks in a row, one box
        self.size_pub = pub(ObjectSize, 'held_object/size', latched)    # the GUI's dummy size
        sub(String, 'arm_state/cmd', self._on_state_cmd)
        sub(String, 'arm_action', self._on_action)
        sub(Empty, 'stop', lambda _m: self._on_action(String(data='stop')))
        sub(ObjectSize, 'held_object/size', self._on_size)
        sub(Empty, 'held_object/release', lambda _m: self.release())
        node.create_service(Trigger, f'{ns}/arm_state/get', self._on_get_state,
                            callback_group=cb)
        model = args.model_service or f'{ns}/box_model/detect'
        self.detect_client = node.create_client(DetectTarget, model, callback_group=cb)
        self.check_client = node.create_client(
            CheckPick, model.rsplit('/', 1)[0] + '/check', callback_group=cb)
        node.create_timer(0.2, self._publish_state_if_changed, callback_group=cb)
        node.create_timer(args.state_period, self.publish_state, callback_group=cb)
        self.phase('idle')

    # -- what the window used to show, now also on topics -----------------------

    def say(self, text, error=False):
        super().say(text, error)
        self.log_pub.publish(String(data=text))

    def vacuum(self, arm, on):
        """<ns>/vacuum (std_msgs/Bool, latched): on before the cup goes in, off
        at the drop. app/vacuum_bridge.py switches the ESP32 relay from it
        ('1' / '0' over USB serial)."""
        if arm != TOOL_ARM['vacuum']:
            return                                       # (the finger arm has no vacuum)
        if bool(on) != self.vacuum_on:
            self.vacuum_on = bool(on)
            self.vacuum_pub.publish(Bool(data=self.vacuum_on))
            self.node.get_logger().info(f'vacuum: {"on" if on else "off"} ({arm} arm)')

    def phase(self, step):
        """arm_action/phase: the step running now (the touch code calls it)."""
        self.step = step
        self.phase_pub.publish(String(data=step))

    # -- the held object ----------------------------------------------------------

    def _on_size(self, msg):
        hwd = (msg.height_mm / 1000.0, msg.width_mm / 1000.0, msg.depth_mm / 1000.0)
        source = msg.header.frame_id or 'topic'
        if min(hwd) <= 0.0:
            if self.node.object_size is not None:
                self.node.get_logger().info('held_object/size: no object (zeros)')
            self.node.object_size = None
            return
        if self.node.object_size is None or self.node.object_size['hwd'] != hwd:
            self.node.get_logger().info(
                f'held_object/size: {msg.height_mm:.0f} x {msg.width_mm:.0f} x '
                f'{msg.depth_mm:.0f} mm ({source})')
        self.node.object_size = {'hwd': hwd, 'source': source}

    def release(self):
        if self.node.held:
            self.node.detach_held()
            self.say('held box released from the planners')

    def held_text(self):
        if not self.node.held:
            return 'none'
        return ', '.join(f'{arm} ' + 'x'.join(f'{1000 * v:.0f}' for v in h['size']) + ' mm'
                         for arm, h in sorted(self.node.held.items()))

    def status_dict(self):
        """Everything a system watching the arms needs, as plain data."""
        t, size = self.target, self.node.object_size
        pick = None
        if self.pick is not None:
            check = self.check or {}
            pick = {'arm': self.pick[3], 'xyz': [round(float(v), 4) for v in self.pick[2]],
                    'ok': check.get('ok'), 'why': check.get('why')}
        return {
            'arm_state': self.current_state(), 'phase': self.step, 'mode': self.mode,
            'busy': self.busy, 'halted': self.node.halt.is_set(), 'result': self.result,
            'top_down': self.top_down, 'pick': pick,
            'pick_tool': self.pick_tool, 'tool_arm': TOOL_ARM.get(self.pick_tool),
            'vacuum': self.vacuum_on,
            'pick_result': self.pick_result,
            'held': {a: {'size_mm': [round(1000 * v) for v in h['size']], 'top': h['top']}
                     for a, h in self.node.held.items()},
            'object_size_mm': ([round(1000 * v) for v in size['hwd']] if size else None),
            'object_size_source': size['source'] if size else None,
            'target': ({'approach': t.approach, 'tool': t.tool, 'arm': t.arm,
                        'confidence': round(float(t.confidence), 2),
                        'xyz': [round(t.position.x, 4), round(t.position.y, 4),
                                round(t.position.z, 4)],
                        'face_mm': [round(float(v)) for v in t.face_mm],
                        'box_mm': [round(float(v)) for v in t.box_mm]}
                       if t is not None else None),
            'tools': {a: {'on': bool(v['on']), 'length_mm': round(1000 * float(v['length']))}
                      for a, v in self.node.args.tools.items()},
            'pieces_on': self.node.pieces_on, 'message': self.status,
        }

    def report(self, result):
        """arm_action/status."""
        self.result = result
        self.status_pub.publish(String(data=result))
        self.node.get_logger().info(f'arm_action/status: {result}')

    def current_state(self):
        if self.activity:
            return self.activity
        if self.last_touch is not None:
            return 'arm_out'
        self.mode = self.detect_mode()
        return STATE_OF_MODE.get(self.mode, 'unknown')

    def publish_state(self):
        self.state = self.current_state()
        self.state_pub.publish(String(data=self.state))
        if self._status_said is not None:            # every period too, for late joiners
            self.json_pub.publish(String(data=self._status_said))

    def _publish_state_if_changed(self):
        if self.current_state() != self.state:
            self.publish_state()
        held = self.held_text()
        if held != self._held_said:
            self._held_said = held
            self.held_pub.publish(String(data=held))
        status = json.dumps(self.status_dict(), default=str)
        if status != self._status_said:
            self._status_said = status
            self.json_pub.publish(String(data=status))

    def _on_get_state(self, _request, response):
        response.success, response.message = True, self.current_state()
        return response

    # -- commands ---------------------------------------------------------------

    def _on_state_cmd(self, msg):
        cmd = msg.data.strip().lower()
        if cmd in ('navigation', 'pre_pick', 'return') and not self.busy:
            self.node.resume()                       # (a refused command must not undo STOP)
        if cmd == 'status':
            self.publish_state()
        elif cmd in ('navigation', 'pre_pick'):
            if not self.go_to(cmd):
                self.report('busy')
        elif cmd == 'return':
            if not self.run(self._return, 'returning'):
                self.report('busy')
        else:
            self.say(f'arm_state/cmd: unknown "{msg.data}" (navigation, pre_pick, return, '
                     f'status)', True)

    def _on_action(self, msg):
        words = msg.data.strip().lower().split()
        if words and words[0] == 'stop':
            # always, even while busy: cancel what moves; new moves clear it
            self.node.stop_motion()
            self.abort.set()
            self.report('stopped')
            return
        if words and words[0] == 'ready_to_pick' and not self.busy:
            self.node.resume()                       # (a refused command must not undo STOP)
        if not words or words[0] != 'ready_to_pick':
            # (our own results, if arm_action/status is set to arm_action itself)
            if words and words[0].rstrip(':') not in (
                    'busy', 'detecting', 'target_found', 'picking', 'checking',
                    'pick_complete', 'pick_unverified', 'no_target', 'pick_failed',
                    'retrying', 'stopped'):
                self.say(f'arm_action: unknown "{msg.data}" (ready_to_pick '
                         f'[top|front|auto] [vacuum|finger])', True)
            return
        approach, tool = self.node.args.approach, self.pick_tool
        for word in words[1:]:
            if word in ('auto', 'top', 'front'):
                approach = word
            elif word in TOOL_ARM:
                tool = word
            elif word in ('left', 'right'):
                # the tool decides the arm: an arm that differs is not used
                if TOOL_ARM.get(tool, word) != word:
                    self.say(f'arm_action: "{word}" ignored - a {tool} pick is the '
                             f'{TOOL_ARM[tool]} arm\'s', True)
            else:
                self.report(f'pick_failed: "{word}" - ready_to_pick [top|front|auto] '
                            f'[vacuum|finger]')
                return
        if tool not in TOOL_ARM:
            self.report('pick_failed: say the tool: ready_to_pick ... vacuum|finger')
            return
        if self.busy:
            self.report('busy')
            return
        self.set_tool_mode(tool)
        if not self.run(lambda: self.ready_to_pick(approach), 'detecting'):
            self.report('busy')

    def run(self, work, activity):
        """work() in the background, with arm_state showing `activity`.
        False if something else is running."""
        if self.busy:
            self.say(f'busy ({self.current_state()}) - not now', True)
            return False

        def job():
            self.activity = activity
            try:
                work()
            finally:
                self.activity = None
                self.phase('idle')
        return self.in_background(job)

    def go_to(self, state):
        mode = 'pick' if state == 'pre_pick' else 'navigation'

        def job():
            if self.last_touch is not None and not self.return_to_start():
                return
            self.phase(f'both to_{state}')
            self.go_mode(mode)
        return self.run(job, MOVING[mode])

    def _return(self):
        self.return_to_start()

    # -- the pick ---------------------------------------------------------------

    def ready_to_pick(self, approach):
        """The pick, and again by itself while it fails: look again, pick
        again -- until it is taken, or --adjust-after failures on the box have
        asked the wheels to move (base_adjust), or STOP."""
        tries = max(1, int(getattr(self.node.args, 'adjust_after', 2) or 1))
        self.abort.clear()                               # a new command: STOP is past
        for attempt in range(1, tries + 1):
            self.pick_result = None
            self.final_attempt = attempt == tries        # earlier ones say "retrying"
            try:
                self._ready_to_pick(approach)
            except Exception as exc:                 # noqa: BLE001 - reported, not raised
                self.say(f'pick: {exc}', True)
                self.report(f'pick_failed: {exc}')
                return
            r = self.pick_result or {}
            if r.get('picked') is not False or r.get('base_adjust') or \
                    r.get('outcome') == 'arm_out' or self.abort.is_set() or \
                    self.node.halt.is_set() or self.last_touch is not None:
                return
            if attempt < tries:
                self.say(f'not picked ({r.get("outcome")}) - looking again and picking again '
                         f'by itself ({attempt + 1}/{tries})')
        self.final_attempt = True

    def _ready_to_pick(self, approach):
        if self.last_touch is not None:
            self.report('pick_failed: an arm is still out from the last pick - send '
                        'arm_state/cmd "return"')
            return
        if TOOL_ARM.get(self.pick_tool) in self.node.held:
            self.report('pick_failed: the arm still holds the last box - take it off the cup, '
                        'then held_object/release')
            return
        if self.halted():
            self.report('stopped')
            return
        if self.mode != 'pick':
            self.activity = 'moving_to_pre_pick'
            self.phase('both to_pre_pick')
            if not self.go_mode('pick'):
                self.report('stopped' if self.halted() else
                            'pick_failed: could not reach pre_pick')
                return
        # the model's frame: both arms at pre_pick, out of the view
        self.activity = 'detecting'
        self.report('detecting')
        if not self.detect_client.wait_for_service(timeout_sec=5.0):
            self.report(f'pick_failed: no model ({self.detect_client.srv_name} not served - '
                        f'is box_model_node.py running?)')
            return
        request = DetectTarget.Request(approach=approach, tool=self.pick_tool)
        # the octomap's frame meanwhile: the same still scene (~2 s saved)
        captured = threading.Thread(target=self.capture, daemon=True)
        captured.start()
        answer = self.node.wait(self.detect_client.call_async(request), 30.0)
        captured.join()
        if answer is None:
            self.report('pick_failed: the model did not answer in 30 s')
            return
        if not answer.found:
            self.say(f'no target: {answer.message}', True)
            self.report(f'no_target: {answer.message}')
            return
        target = self.target = answer.target
        self.say(f'target: {answer.message}')
        self.prepare_target(target)
        self.target_pub.publish(target)
        self.report('target_found')

        # the point (on the frame captured meanwhile) and its check
        if self.frame is None:
            self.report('pick_failed: no camera frame for the octomap')
            return
        self.top_down = target.approach == PickTarget.APPROACH_TOP
        bw = self.frame[0].shape[1]
        u = target.u * bw / float(target.image_width or bw)
        v = target.v * self.frame[0].shape[0] / float(target.image_height or
                                                      self.frame[0].shape[0])
        p = target.position
        check = self.select_point(u, v, world=np.array([p.x, p.y, p.z]))
        if check is None:
            self.report('pick_failed: no depth at the target')
            return
        self.check_pick(check, in_job=True)      # (the tool's arm: select_point)
        if self.halted():
            self.report('stopped')
            return
        if not check.get('ok'):
            why = check.get('why') or 'not reachable'
            # the wheels only when it is the robot's place (reach, edge, camera
            # too near) -- not for a check that failed (move_group down, ...)
            adjust = self.unreachable_adjust(why) if check.get('not_here') else None
            self.report(f'pick_failed: {why}' + (
                f'; base_adjust requested: {self.adjust_text(adjust)}' if adjust else ''))
            return
        self.activity = 'picking'
        self.report('picking')
        _u, _v, point, arm, origin = self.pick
        self.touch(arm, point, origin, self.frame)
        self.finish_pick(arm)

    # -- the tool, the box, the check -----------------------------------------------

    def set_tool_mode(self, tool):
        """vacuum (left arm only) / finger (right arm only)."""
        if tool != self.pick_tool:
            self.pick_tool = tool
            self.pick = self.check = None                # chosen for the other arm
            self.say(f'{tool} picks: the {TOOL_ARM[tool]} arm only - the '
                     f'{"right" if TOOL_ARM[tool] == "left" else "left"} arm is not moved '
                     f'for a pick')

    def prepare_target(self, target):
        """The model's target, made explicit: the tool and its arm (whatever
        the model said), and the known box matched to the measured face (for
        the attach at contact)."""
        tool = self.pick_tool if self.pick_tool in TOOL_ARM else (target.tool or 'vacuum')
        arm = TOOL_ARM[tool]
        if target.arm and target.arm != arm:
            self.say(f'the target named the {target.arm} arm: a {tool} pick is the {arm} '
                     f'arm\'s', True)
        target.tool, target.arm = tool, arm
        self.node.pick_box = None
        # the face's short side: the cup must stay on it (ctm/touch.py)
        sides = [float(v) for v in target.face_mm if v > 0]
        self.node.pick_face_mm = min(sides) if sides else None
        target.box_mm = [0.0, 0.0, 0.0]
        size = self.node.object_size
        face = [float(v) for v in target.face_mm]
        if size is None or min(face) <= 0:
            if size is not None:
                self.say('box size known, but the model gave no face size: attached by the '
                         'tool\'s direction', True)
            return
        known = [1000.0 * v for v in size['hwd']]
        box, err = box_match.match(known, face, loose=target.approach == PickTarget.APPROACH_TOP)
        k = ' x '.join(f'{v:.0f}' for v in known)
        if box is None:
            self.say(f'the face seen ({face[0]:.0f} x {face[1]:.0f} mm) is no face of the '
                     f'{k} mm box (off by {err:.0f} mm) - attached by the tool\'s direction',
                     True)
            return
        target.box_mm = [float(v) for v in box]
        f = target.face_long
        self.node.pick_box = {'long': np.array([f.x, f.y, f.z]), 'box_mm': box}
        self.say(f'face seen {face[0]:.0f} x {face[1]:.0f} mm = the {box[0]:.0f} x '
                 f'{box[1]:.0f} face of the {k} mm box ({err:.0f} mm off): {box[2]:.0f} mm '
                 f'deep along the tool')
        if target.approach == PickTarget.APPROACH_TOP:
            self._top_from_near_edge(target, face, box)
        elif self._narrow_front_to_top(target, face, box):
            self.node.pick_box = {'long': np.array([target.face_long.x, target.face_long.y,
                                                    target.face_long.z]),
                                  'box_mm': tuple(float(v) for v in target.box_mm)}

    def _narrow_front_to_top(self, target, face, box):
        """FRONT on a narrow face (a flat box: 169 x 30 mm for the 25 mm cup) of a box
        with nothing on it -- the picker grips the larger of the faces it SEES, and
        from ~10 cm above the stack it hardly sees the tops (2026-10-29) -- becomes
        TOP, when the matched box's top is wide: its centre is the front face's
        centre (seen head-on: accurate) raised half the box's height and moved
        back half its depth. True if it did."""
        args = self.node.args
        if not getattr(args, 'prefer_top', True):
            return False
        height, depth = box[1], box[2]                   # across the front face, into the box
        if min(face) > getattr(args, 'narrow_face_mm', 45.0) or depth < 60.0:
            return False
        n = np.array([target.face_normal.x, target.face_normal.y, target.face_normal.z])
        along = np.array([target.face_long.x, target.face_long.y, target.face_long.z])
        up = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(n) < 0.5 or abs(n @ up) > 0.5 or abs(along @ up) > 0.5:
            return False                                 # not an upright front of a flat box
        back = -(n - (n @ up) * up)
        back /= np.linalg.norm(back)
        p = np.array([target.position.x, target.position.y, target.position.z])
        top = p + up * (height / 2000.0) + back * (depth / 2000.0)
        target.position.x, target.position.y, target.position.z = (float(v) for v in top)
        target.approach = PickTarget.APPROACH_TOP
        target.face_normal.x, target.face_normal.y, target.face_normal.z = 0.0, 0.0, 1.0
        target.face_mm = [float(box[0]), float(depth)]
        target.box_mm = [float(box[0]), float(depth), float(height)]   # on the cup: top down
        self.node.pick_face_mm = float(min(box[0], depth))
        self._project_target(target)
        self.say(f'the {min(face):.0f} mm front is too narrow for the cup and nothing is on the '
                 f'box: picking its {box[0]:.0f} x {depth:.0f} mm TOP instead (from the front face '
                 f'and the box size)')
        return True

    def _project_target(self, target):
        """target.u, v: its position in the colour image (the ring the GUI draws)."""
        if self.frame is None:
            return
        _bgr, _depth, info, frame_id = self.frame
        m = self.node.lookup(frame_id, BASE_FRAME)
        if m is None or info is None:
            return
        q = m[:3, :3] @ np.array([target.position.x, target.position.y, target.position.z]) + m[:3, 3]
        if q[2] <= 0:
            return
        k = np.asarray(info.k, float).reshape(3, 3)
        target.u, target.v = float(k[0, 0] * q[0] / q[2] + k[0, 2]), float(k[1, 1] * q[1] / q[2] + k[1, 2])

    def _top_from_near_edge(self, target, face, box):
        """A top face is seen at a shallow angle (the camera ~10 cm above the box
        tops, ~14 deg): one pixel is 4-5 mm along the view, the FAR edge blurs
        into the box behind, and the picker returns the frame that saw the most
        of it -- the top read 120 mm deep for a 100 mm box and the cup went to
        its far end (2026-10-29). The NEAR edge is sharp: with the box's real
        size known, the centre is half the real depth in from it -- the target
        moves toward the camera by half what the face was seen too long."""
        cam = self.node.lookup(BASE_FRAME, 'camera_link')
        if cam is None:
            return
        p = np.array([target.position.x, target.position.y, target.position.z])
        view = p - cam[:3, 3]
        view[2] = 0.0
        if np.linalg.norm(view) < 1e-6:
            return
        view /= np.linalg.norm(view)
        n = np.array([target.face_normal.x, target.face_normal.y, target.face_normal.z])
        along = np.array([target.face_long.x, target.face_long.y, target.face_long.z])
        across = np.cross(n, along)
        if np.linalg.norm(across) < 1e-6:
            return
        across /= np.linalg.norm(across)
        seen = sorted(face, reverse=True)                    # (long, short) as measured
        # the face's edge that runs along the camera's view: its length is the one
        # the far edge stretches
        if abs(along @ view) >= abs(across @ view):
            axis, measured, real = along, seen[0], box[0]
        else:
            axis, measured, real = across, seen[1], box[1]
        excess = measured - real
        if not 3.0 < excess <= 40.0:
            return                                           # as measured (or not trusted)
        shift = -np.sign(axis @ view) * axis * (excess / 2000.0)   # toward the camera
        p = p + shift
        target.position.x, target.position.y, target.position.z = (float(v) for v in p)
        self.say(f'top seen {excess:.0f} mm longer than the box along the camera\'s view (its '
                 f'far edge blurs at this angle): aiming {excess / 2:.0f} mm nearer, at the '
                 f'real centre')

    def verify_pick(self, holding='', wait=0.5):
        """The box picker's check, the arm out of the view: (picked, outcome,
        message, verdict) -- picked None when it cannot tell or is not served.
        holding: 'held' / 'empty' from the joint torques after the press, or ''.
        wait: s first, for the arm to be still (0: it is out of the picture
        already, PICK ALL's look-ahead)."""
        if not self.check_client.wait_for_service(timeout_sec=2.0):
            return None, 'no_check', f'no check ({self.check_client.srv_name} not served)', None
        for attempt in range(2):
            time.sleep(wait if attempt == 0 else 1.5)    # the arm still; then once more
            answer = self.node.wait(self.check_client.call_async(
                CheckPick.Request(holding=holding)), 25.0)
            if answer is None:                           # served but silent: can't tell
                return None, 'check_timeout', 'the camera check did not answer', None
            if not answer.outcome:                       # the picker is down, or nothing to check
                return None, 'no_check', answer.message, None
            picked = {1: True, 0: False}.get(answer.picked)
            if picked is not None or answer.outcome != 'blocked':
                break
        try:
            verdict = json.loads(answer.verdict_json)
        except ValueError:
            verdict = None
        return picked, answer.outcome, answer.message, verdict

    def count_failure(self, result):
        """Failed picks in a row on one box (within 5 cm); at --adjust-after of
        them, the base move on base_adjust (returned as a dict, the count
        reset). A success, or another box, starts the count again."""
        after = int(getattr(self.node.args, 'adjust_after', 2) or 0)
        xyz = result.get('xyz')
        if result.get('picked') is not False or xyz is None or after <= 0:
            if result.get('picked') is True:
                self.fails = {'xyz': None, 'outcomes': []}
            return None                                  # (can't tell: not counted)
        same = self.fails['xyz'] is not None and np.linalg.norm(
            np.subtract(xyz, self.fails['xyz'])) <= 0.05
        outcomes = (self.fails['outcomes'] if same else []) + [result.get('outcome') or '?']
        self.fails = {'xyz': list(xyz), 'outcomes': outcomes}
        if len(outcomes) < after:
            self.say(f'pick failed {len(outcomes)} time(s) on this box - at {after}, the base '
                     f'is asked to move (base_adjust)')
            return None
        self.fails = {'xyz': None, 'outcomes': []}
        return self.request_base_adjust(result, outcomes)

    def unreachable_adjust(self, why):
        """The arm cannot pick the box from here at all (the reach check, from
        every approach point down to 1 cm): ask the wheels at once -- no pick
        will ever fail on it to count. The base_adjust dict, also in
        pick_result."""
        if self.pick is None or int(getattr(self.node.args, 'adjust_after', 2) or 0) <= 0:
            return None
        t = self.target
        result = {'xyz': [float(v) for v in self.pick[2]], 'arm': self.pick[3],
                  'tool': self.pick_tool, 'approach': t.approach if t is not None else None,
                  'picked': False, 'outcome': 'unreachable', 'message': why,
                  'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        adjust = self.request_base_adjust(result, ['unreachable'])
        result['base_adjust'] = adjust
        self.pick_result = result
        self.result_pub.publish(String(data=json.dumps(result, default=str)))
        return adjust

    def request_base_adjust(self, result, outcomes):
        """base_adjust: how far the base should move SIDEWAYS -- left or right
        only, never forward, back or a turn (2026-11-02) -- so the box lines up
        across the robot with the arm's sweet spot: left_m = box y - sweet y
        (the robot moving left by d moves the box d to the right in its frame).
        Under 1 cm: "none" -- sideways cannot help (the box is too near or too
        far). The dict goes into pick_result."""
        args = self.node.args
        p = np.asarray(result['xyz'], float)
        sweet = np.asarray(args.sweet_spot, float)
        left = float(p[1] - sweet[1])
        direction = 'none' if abs(left) < 0.01 else ('left' if left > 0 else 'right')
        if direction == 'none':
            left = 0.0
        msg = BaseAdjust()
        msg.header.frame_id = BASE_FRAME
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.direction, msg.distance_m, msg.left_m = direction, abs(left), left
        msg.target.x, msg.target.y, msg.target.z = (float(v) for v in p)
        msg.goal.x, msg.goal.y, msg.goal.z = float(sweet[0]), float(sweet[1]), float(p[2])
        msg.arm, msg.tool = result.get('arm') or '', result.get('tool') or ''
        msg.approach = result.get('approach') or ''
        msg.failures = len(outcomes)
        msg.reason = ', '.join(outcomes)
        self.base_pub.publish(msg)
        adjust = {'direction': direction, 'distance_m': round(abs(left), 3),
                  'left_m': round(left, 3), 'failures': len(outcomes), 'reason': msg.reason,
                  'goal': [round(float(v), 3) for v in sweet]}
        why = ('the arm cannot reach this box from here' if outcomes == ['unreachable'] else
               f'{len(outcomes)} picks in a row did not take this box ({msg.reason})')
        if direction == 'none':
            self.say(f'{why}: base_adjust sent - none: the box is already lined up sideways, and '
                     f'moving left or right cannot help (it is {p[0]:.2f} m out, '
                     f'{p[2]:.2f} m up; the arm works best {sweet[0]:.2f} m out)', True)
        else:
            self.say(f'{why}: base_adjust sent - move {direction} {abs(left):.3f} m; detect '
                     f'the box again once the base has moved', True)
        self.give_up_on_box(outcomes, direction)
        return adjust

    def give_up_on_box(self, outcomes, direction):
        """As the picker's own loop does with a box it gave up on (2026-11-03): photos
        of it for a person (box_model/cmd "flag": the scene with it marked, close up,
        the last check), and -- when moving the base cannot help -- skipped, so the
        next detection takes another box. When the base is to move, the skipped boxes
        are forgotten instead: they are remembered by their place in the picture."""
        tried = outcomes != ['unreachable']
        if tried:
            self.model_cmd_pub.publish(String(data='flag'))
        if direction == 'none':
            self.model_cmd_pub.publish(String(data='skip'))
            self.say('this box is skipped: the next detection takes another one (Forget skipped '
                     'brings it back)' + ('; photos of it saved for a person (FLAG)' if tried else ''),
                     True)
        else:
            self.model_cmd_pub.publish(String(data='reset'))

    @staticmethod
    def adjust_text(adjust):
        """base_adjust, as said in a status: "move left 0.100 m" / "none ..."."""
        if adjust['direction'] == 'none':
            return 'none (already lined up sideways; left or right cannot help)'
        return f'move {adjust["direction"]} {adjust["distance_m"]:.3f} m'

    def finish_pick(self, arm, pick=None, target=None, verified=None, count=True):
        """After touch(): arm_action/status and pick_result from the contact
        and the camera's check.

        PICK ALL (web_gui.py) finishes a pick after it has already detected the
        next box: `pick` / `target` are the picked box's (self.pick / self.target
        are the next one's by then), `verified` the camera's check it made as
        soon as the arm was out of the picture (verify_pick's answer), and
        count=False: its boxes get one try each -- a failure is not counted
        toward base_adjust, and the next box's measured face stays."""
        contact = self.last_contact
        t = self.target if target is None else target
        pick = self.pick if pick is None else pick
        felt = bool(contact and contact['felt'] and 'STOPPED' not in contact.get('how', ''))
        result = {'tool': self.pick_tool, 'arm': arm, 'contact': felt,
                  'approach': t.approach if t is not None else None,
                  'xyz': ([round(float(v), 4) for v in pick[2]] if pick else None),
                  'box_mm': ([round(float(v)) for v in t.box_mm] if t is not None else None),
                  'picked': None, 'outcome': None, 'message': None, 'verdict': None,
                  'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        if self.last_touch is not None:
            result.update(outcome='arm_out', message=f'the {arm} arm did not get back')
            status = f'pick_failed: the {arm} arm did not get back - send arm_state/cmd "return"'
        elif not getattr(self, 'touch_moved', True):
            # refused before the arm moved (STOP, a box still held, no plan):
            # not a failed pick -- not counted, not retried, no base move
            stopped = self.halted()
            result.update(outcome='stopped' if stopped else 'not_started', message=self.status)
            status = 'stopped' if stopped else f'pick_failed: not started - {self.status}'
        elif contact is None:
            result.update(picked=False, outcome='no_contact', message=self.status)
            status = f'pick_failed: the {arm} arm did not reach the box ({self.status})'
        else:
            self.activity = 'checking'
            self.report('checking')
            self.phase(f'{arm} check')
            held = contact.get('held')
            result.update(held=held, held_g=contact.get('held_g'))
            picked, outcome, message, verdict = verified if verified is not None else \
                self.verify_pick({True: 'held', False: 'empty'}.get(held, ''))
            result.update(picked=picked, outcome=outcome, message=message, verdict=verdict)
            if picked is True:
                status = 'pick_complete'
            elif picked is False:
                status = f'pick_failed: {outcome} - {message}'
            elif outcome == 'no_check':                  # no camera check: contact + weight
                ok = felt and held is not False
                result['picked'] = ok
                result['outcome'] = 'success' if ok else 'no_contact' if not felt else 'empty'
                status = 'pick_complete' if ok else \
                    f'pick_failed: {contact.get("how", "no contact felt") if not felt else "nothing on the cup"}'
            else:
                status = f'pick_unverified: {outcome} - {message}'
            self.say(f'pick check: {message}', picked is False)
        adjust = self.count_failure(result) if count else None
        if adjust is not None:
            result['base_adjust'] = adjust
            status += f'; base_adjust requested: {self.adjust_text(adjust)}'
        if result.get('picked') is False and not result.get('base_adjust') and \
                not getattr(self, 'final_attempt', True):
            status = f'retrying: {result.get("outcome")}'  # not the end: another attempt follows
        self.pick_result = result
        if count:
            self.node.pick_box = None                    # (used: the next target's own)
            self.node.pick_face_mm = None
        self.result_pub.publish(String(data=json.dumps(result, default=str)))
        self.report(status)

    # -- the window (only with a display) ---------------------------------------

    def draw(self):
        color, _d, _i = self.node.latest()
        img = decode_image(color).copy() if color is not None else \
            np.zeros((480, 640, 3), np.uint8)
        t = self.target
        if t is not None:
            x1, y1, x2, y2 = (int(round(c)) for c in t.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.drawMarker(img, (int(t.u), int(t.v)), (0, 200, 0), cv2.MARKER_CROSS, 18, 2)
        lines = [f'arm_state: {self.state or "?"}' + ('   [busy]' if self.busy else ''),
                 f'arm_action/status: {self.result or "-"}',
                 (f'target: {t.approach} x={t.position.x:+.3f} y={t.position.y:+.3f} '
                  f'z={t.position.z:+.3f}  conf {t.confidence:.2f}' if t is not None
                  else 'target: -'),
                 self.status,
                 'keys: p pre_pick  n navigation  space ready_to_pick  b return  q quit']
        for k, text in enumerate(lines):
            text = text if len(text) < 95 else text[:92] + '...'
            y = 22 + 22 * k
            cv2.putText(img, text, (8, y), FONT, 0.5, (0, 0, 0), 3)
            cv2.putText(img, text, (8, y), FONT, 0.5, (255, 255, 255), 1)
        return img

    def window(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        while rclpy.ok():
            cv2.imshow(WINDOW, self.draw())
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                break
            if key == ord('p'):
                self.go_to('pre_pick')
            elif key == ord('n'):
                self.go_to('navigation')
            elif key == ord(' '):
                self._on_action(String(data='ready_to_pick'))
            elif key == ord('b'):
                self.run(self._return, 'returning')
        cv2.destroyAllWindows()


def make_parser():
    parser = ctm_app.make_parser()
    parser.description = __doc__.split('\n')[0]
    parser.add_argument('--gui', choices=['auto', 'on', 'off'], default='auto',
                        help='the window: auto = only if there is a display')
    parser.add_argument('--approach', choices=['auto', 'top', 'front'], default='auto',
                        help='how to pick when ready_to_pick does not say: auto = as the '
                             'model decides')
    parser.add_argument('--state-period', type=float, default=1.0,
                        help='arm_state is published on every change and this often, s')
    parser.add_argument('--ns', default='/beast_arm',
                        help='every topic and service goes under this')
    parser.add_argument('--status-topic', default=None,
                        help='where the pick results go (pick_complete, ...); default '
                             '<ns>/arm_action/status')
    parser.add_argument('--model-service', default=None,
                        help='the model: a pick_interfaces/DetectTarget service; default '
                             '<ns>/box_model/detect')
    parser.add_argument('--adjust-after', type=int, default=2,
                        help='failed picks in a row on one box before base_adjust asks the '
                             'wheels to move the robot (0: never)')
    parser.add_argument('--sweet-spot', type=float, nargs=2, default=[0.45, 0.10],
                        metavar=('X', 'Y'),
                        help='where the picking arm works best, robot frame m (x forward, y '
                             'left): base_adjust brings the box there')
    parser.set_defaults(drop_hold=1.0, pick_tool='vacuum', edge='adjust')
    return parser


def has_display():
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def main():
    args = make_parser().parse_args(remove_ros_args(sys.argv)[1:])
    ctm_app.finish_args(args)
    rclpy.init(args=sys.argv)
    node = ClickToMove(args)
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    gui = args.gui == 'on' or (args.gui == 'auto' and has_display())
    try:
        stack = PickStack(node)
        # where the arms are, once /joint_states has come in
        deadline = time.monotonic() + 10.0
        while node.arm_positions('left') is None and time.monotonic() < deadline:
            time.sleep(0.2)
        stack.publish_state()
        node.get_logger().info(f'pick stack up ({"window" if gui else "headless"}); '
                               f'arm_state: {stack.state}')
        if gui:
            stack.window()
        else:
            while rclpy.ok():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        except BaseException:                        # noqa: BLE001 - a second Ctrl-C
            pass
        rclpy.try_shutdown()
        os._exit(0)                                  # (no waiting on worker threads)


if __name__ == '__main__':
    main()
