#!/usr/bin/env python3
"""PICK STACK: the arms as a ROS 2 module -- the same planning, touch and
drop sequence as click_to_move.py / auto_box.py, driven by topics, with or
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
  pick_result          std_msgs/String, out, latched -- each pick's outcome, JSON:
                       {picked: true/false/null, outcome (success, missed, moved,
                       dropped, wrong_box, blocked, camera_moved, unknown,
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
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from pick_interfaces.msg import ObjectSize, PickTarget
from pick_interfaces.srv import CheckPick, DetectTarget

import click_to_move as ctm_app
from ctm import box_match
from ctm.common import FONT, TOOL_ARM, decode_image
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
        self.pick_result = None       # the last pick's outcome (pick_result)
        self.node.pick_box = None     # the matched box for the next contact (ctm/touch.py)
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
        if cmd in ('navigation', 'pre_pick', 'return'):
            self.node.resume()
        if cmd == 'status':
            self.publish_state()
        elif cmd in ('navigation', 'pre_pick'):
            self.go_to(cmd)
        elif cmd == 'return':
            self.run(self._return, 'returning')
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
        if words and words[0] == 'ready_to_pick':
            self.node.resume()
        if not words or words[0] != 'ready_to_pick':
            # (our own results, if arm_action/status is set to arm_action itself)
            if words and words[0].rstrip(':') not in (
                    'busy', 'detecting', 'target_found', 'picking', 'checking',
                    'pick_complete', 'pick_unverified', 'no_target', 'pick_failed',
                    'stopped'):
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
        self.in_background(job)
        return True

    def go_to(self, state):
        mode = 'pick' if state == 'pre_pick' else 'navigation'

        def job():
            if self.last_touch is not None and not self.return_to_start():
                return
            self.phase(f'both to_{state}')
            self.go_mode(mode)
        self.run(job, MOVING[mode])

    def _return(self):
        self.return_to_start()

    # -- the pick ---------------------------------------------------------------

    def ready_to_pick(self, approach):
        try:
            self._ready_to_pick(approach)
        except Exception as exc:                     # noqa: BLE001 - reported, not raised
            self.say(f'pick: {exc}', True)
            self.report(f'pick_failed: {exc}')

    def _ready_to_pick(self, approach):
        if self.last_touch is not None:
            self.report('pick_failed: an arm is still out from the last pick - send '
                        'arm_state/cmd "return"')
            return
        if self.mode != 'pick':
            self.activity = 'moving_to_pre_pick'
            self.phase('both to_pre_pick')
            if not self.go_mode('pick'):
                self.report('pick_failed: could not reach pre_pick')
                return
        # the model's frame: both arms at pre_pick, out of the view
        self.activity = 'detecting'
        self.report('detecting')
        if not self.detect_client.wait_for_service(timeout_sec=5.0):
            self.report(f'pick_failed: no model ({self.detect_client.srv_name} not served - '
                        f'is box_model_node.py running?)')
            return
        request = DetectTarget.Request(approach=approach, tool=self.pick_tool)
        answer = self.node.wait(self.detect_client.call_async(request), 30.0)
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

        # the octomap from the same still scene, then the point and its check
        self.capture()
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
        if not check.get('ok'):
            self.report(f'pick_failed: {check.get("why") or "not reachable"}')
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
        target.box_mm = [0.0, 0.0, 0.0]
        size = self.node.object_size
        face = [float(v) for v in target.face_mm]
        if size is None or min(face) <= 0:
            if size is not None:
                self.say('box size known, but the model gave no face size: attached by the '
                         'tool\'s direction', True)
            return
        known = [1000.0 * v for v in size['hwd']]
        box, err = box_match.match(known, face)
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

    def verify_pick(self):
        """The box picker's check, the arm out of the view: (picked, outcome,
        message, verdict) -- picked None when it cannot tell or is not served."""
        if not self.check_client.wait_for_service(timeout_sec=2.0):
            return None, 'no_check', f'no check ({self.check_client.srv_name} not served)', None
        for attempt in range(2):
            time.sleep(0.5 if attempt == 0 else 1.5)     # the arm still; then once more
            answer = self.node.wait(self.check_client.call_async(
                CheckPick.Request(holding='')), 25.0)
            if answer is None:
                return None, 'no_check', 'the check did not answer', None
            picked = {1: True, 0: False}.get(answer.picked)
            if picked is not None or answer.outcome != 'blocked':
                break
        try:
            verdict = json.loads(answer.verdict_json)
        except ValueError:
            verdict = None
        return picked, answer.outcome, answer.message, verdict

    def finish_pick(self, arm):
        """After touch(): arm_action/status and pick_result from the contact
        and the camera's check."""
        contact, t = self.last_contact, self.target
        felt = bool(contact and contact['felt'] and 'STOPPED' not in contact.get('how', ''))
        result = {'tool': self.pick_tool, 'arm': arm, 'contact': felt,
                  'approach': t.approach if t is not None else None,
                  'xyz': ([round(float(v), 4) for v in self.pick[2]] if self.pick else None),
                  'box_mm': ([round(float(v)) for v in t.box_mm] if t is not None else None),
                  'picked': None, 'outcome': None, 'message': None, 'verdict': None,
                  'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        if self.last_touch is not None:
            result.update(outcome='arm_out', message=f'the {arm} arm did not get back')
            status = f'pick_failed: the {arm} arm did not get back - send arm_state/cmd "return"'
        elif contact is None:
            result.update(picked=False, outcome='no_contact', message=self.status)
            status = f'pick_failed: the {arm} arm did not reach the box ({self.status})'
        else:
            self.activity = 'checking'
            self.report('checking')
            self.phase(f'{arm} check')
            picked, outcome, message, verdict = self.verify_pick()
            result.update(picked=picked, outcome=outcome, message=message, verdict=verdict)
            if picked is True:
                status = 'pick_complete'
            elif picked is False:
                status = f'pick_failed: {outcome} - {message}'
            elif outcome == 'no_check':                  # no camera check: the contact says
                result['picked'] = felt
                result['outcome'] = 'success' if felt else 'no_contact'
                status = 'pick_complete' if felt else \
                    f'pick_failed: {contact.get("how", "no contact felt")}'
            else:
                status = f'pick_unverified: {outcome} - {message}'
            self.say(f'pick check: {message}', picked is False)
        self.pick_result = result
        self.node.pick_box = None                        # (used: the next target's own)
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
    parser.set_defaults(drop_hold=1.0, pick_tool='vacuum')
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
