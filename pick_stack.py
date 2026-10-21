#!/usr/bin/env python3
"""PICK STACK: the arms as a ROS 2 module -- the same planning, touch and
drop sequence as click_to_move.py / auto_box.py, driven by topics, with or
without a window.

    native/run_pick_stack.sh          (the robot stack must be up: run_launch_everything.sh)
    ros2 launch launch/pick_stack.launch.py gui:=auto

The interface (names relative: a namespace or remap moves them):

  arm_state            std_msgs/String, out -- where the arms are, on every
                       change and every --state-period s (latched):
                         navigation, pre_pick            at that pose
                         moving_to_navigation, moving_to_pre_pick
                         detecting, picking, returning   during a pick
                         arm_out    a pick stopped half-way: send "return"
                         unknown    neither pose (moved by hand, at boot)
  arm_state/cmd        std_msgs/String, in -- "navigation", "pre_pick",
                       "return" (bring an arm left out back), "status"
                       (publish arm_state now)
  arm_state/get        std_srvs/Trigger -- the state, on request
  arm_action           std_msgs/String, in -- "ready_to_pick": to pre_pick
                       if not there, the model takes a frame and finds the
                       box (top or front), the arm picks it, drops it, back
                       to pre_pick. "ready_to_pick top" / "... front"
                       overrides the model's approach; "stop" cancels
                       every motion at once (new commands clear it)
  arm_action/status    std_msgs/String, out -- the pick's progress (latched):
                         busy                 refused: something else runs
                         detecting
                         target_found         (the target is on pick_target)
                         picking
                         pick_complete        touched, dropped, back at pre_pick
                         no_target: <why>     nothing to pick in the frame
                         pick_failed: <why>
  pick_target          pick_interfaces/PickTarget -- what the model found
                       (box_model_node.py publishes it; position, approach
                       "top"/"front", arm, pixel box, confidence)
  pick_stack/log       std_msgs/String -- every message, as the window shows it

The model is the /box_model/detect service (pick_interfaces/DetectTarget):
box_model_node.py, or any node that serves it.

The window (--gui auto: if there is a display): the live camera, the state,
the last target. Keys: p pre_pick, n navigation, space ready_to_pick,
b return, q quit. Without a display it runs the same, headless.
"""

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
from std_msgs.msg import String
from std_srvs.srv import Trigger

from pick_interfaces.msg import PickTarget
from pick_interfaces.srv import DetectTarget

import click_to_move as ctm_app
from ctm.common import FONT, decode_image
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
        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = node.create_publisher(String, 'arm_state', latched)
        self.status_pub = node.create_publisher(String, args.status_topic, latched)
        self.log_pub = node.create_publisher(String, 'pick_stack/log', 10)
        self.target_pub = node.create_publisher(PickTarget, 'pick_target', 10)
        node.create_subscription(String, 'arm_state/cmd', self._on_state_cmd, 10,
                                 callback_group=cb)
        node.create_subscription(String, 'arm_action', self._on_action, 10, callback_group=cb)
        node.create_service(Trigger, 'arm_state/get', self._on_get_state, callback_group=cb)
        self.detect_client = node.create_client(DetectTarget, args.model_service,
                                                callback_group=cb)
        node.create_timer(0.2, self._publish_state_if_changed, callback_group=cb)
        node.create_timer(args.state_period, self.publish_state, callback_group=cb)

    # -- what the window used to show, now also on topics -----------------------

    def say(self, text, error=False):
        super().say(text, error)
        self.log_pub.publish(String(data=text))

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

    def _publish_state_if_changed(self):
        if self.current_state() != self.state:
            self.publish_state()

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
                    'busy', 'detecting', 'target_found', 'picking', 'pick_complete',
                    'no_target', 'pick_failed', 'stopped'):
                self.say(f'arm_action: unknown "{msg.data}" (ready_to_pick [top|front])', True)
            return
        approach = words[1] if len(words) > 1 else self.node.args.approach
        if approach not in ('auto', 'top', 'front'):
            self.report(f'pick_failed: approach must be top, front or auto, not "{approach}"')
            return
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
        self.in_background(job)
        return True

    def go_to(self, state):
        mode = 'pick' if state == 'pre_pick' else 'navigation'

        def job():
            if self.last_touch is not None and not self.return_to_start():
                return
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
            if not self.go_mode('pick'):
                self.report('pick_failed: could not reach pre_pick')
                return
        # the model's frame: both arms at pre_pick, out of the view
        self.activity = 'detecting'
        self.report('detecting')
        if not self.detect_client.wait_for_service(timeout_sec=5.0):
            self.report(f'pick_failed: no model ({self.node.args.model_service} not served - '
                        f'is box_model_node.py running?)')
            return
        request = DetectTarget.Request(approach=approach)
        answer = self.node.wait(self.detect_client.call_async(request), 30.0)
        if answer is None:
            self.report('pick_failed: the model did not answer in 30 s')
            return
        if not answer.found:
            self.say(f'no target: {answer.message}', True)
            self.report(f'no_target: {answer.message}')
            return
        target = self.target = answer.target
        self.target_pub.publish(target)
        self.say(f'target: {answer.message}')
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
        if target.arm in ('left', 'right') and target.arm != self.pick[3]:
            self.pick = self.pick[:3] + (target.arm,) + self.pick[4:]
            check['pick'] = self.pick
        self.check_pick(check, in_job=True)
        if not check.get('ok'):
            self.report(f'pick_failed: {check.get("why") or "not reachable"}')
            return
        self.activity = 'picking'
        self.report('picking')
        _u, _v, point, arm, origin = self.pick
        self.touch(arm, point, origin, self.frame)
        contact = self.last_contact
        if self.last_touch is not None:
            self.report(f'pick_failed: the {arm} arm did not get back - send '
                        f'arm_state/cmd "return"')
        elif contact is None:
            self.report(f'pick_failed: the {arm} arm did not reach the box ({self.status})')
        elif not contact['felt'] or 'STOPPED' in contact.get('how', ''):
            self.report(f'pick_failed: {contact.get("how", "no contact felt")}')
        else:
            self.report('pick_complete')

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
    parser.add_argument('--status-topic', default='arm_action/status',
                        help='where the pick results go (pick_complete, ...)')
    parser.add_argument('--model-service', default='/box_model/detect',
                        help='the model: a pick_interfaces/DetectTarget service')
    parser.set_defaults(drop_hold=1.0)
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
