#!/usr/bin/env python3
"""BOX MODEL NODE: the box picker (box_picker/) as a ROS 2 node.

The box picker -- SAM 3.1 finds each box's outline, the depth gives its 3D
faces, the picking rules choose the next box (on top first, nothing in
front, nearest, leftmost), over 5 frames, the box they agree on -- runs as
its own process (box_picker/src/pick_server.py --ros, Python 3.11 in
box_picker/src/.venv), fed by its ros_bridge.py. Its grip point (the centre
of the face, colour camera frame) is put into the world frame here, with the
depth check's correction (ctm/depth_cal.py), as a pick_interfaces/PickTarget:

    position   the centre of the face to grip, world frame, m
    approach   "front" (the face towards the camera) or "top" (from above):
               the picker's rules choose
    tool, arm  "vacuum" -> "left", "finger" -> "right" (the request's tool,
               else the `tool` parameter): always both, always matching
    face_*     the gripped face, world frame: size (long, short mm), normal,
               long edge's direction -- the pick stack matches it to the box
    u v (pixels), confidence (the share of frames that agree), num_boxes;
    bbox not given

    service   /beast_arm/box_model/detect   pick_interfaces/srv/DetectTarget
              /beast_arm/box_model/check    pick_interfaces/srv/CheckPick: after the
                                            pick, with the arm out of view, is the box
                                            gone? (the picker's pick check)
    publishes /beast_arm/box_model/target   pick_interfaces/msg/PickTarget, each detection
              /beast_arm/box_model/image    sensor_msgs/Image (latched): the picker's drawing
                                            of each answer -- every box, the chosen face
                                            -- and of each check
              /beast_arm/box_model/check_result  std_msgs/String (latched): each check,
                                            JSON {picked, outcome, message, verdict}

    parameters  ns (/beast_arm), frames (5), world_frame (world),
                picker_dir (box_picker/src), tool (vacuum)

The camera's tilt for the picker (--pitch) is read from the TF at start.
Install the picker once: cd box_picker/src && ./setup.sh

    source native/setup.bash; python3 box_model_node.py
"""

import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from rclpy.qos import DurabilityPolicy, QoSProfile
import tf2_ros

from pick_interfaces.msg import PickTarget
from pick_interfaces.srv import CheckPick, DetectTarget

from ctm import depth_cal
from ctm.common import BASE_FRAME, INFO_TOPIC, TOOL_ARM, matrix_from_quat


class BoxModelNode(Node):
    def __init__(self):
        super().__init__('box_model')
        self.declare_parameter('ns', '/beast_arm')
        self.declare_parameter('approach', 'auto')       # (kept for the launch files)
        self.declare_parameter('frames', 5)
        self.declare_parameter('tool', 'vacuum')         # when a request does not say
        self.declare_parameter('world_frame', BASE_FRAME)
        self.declare_parameter('picker_dir', os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'box_picker', 'src'))
        cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.info = None
        self.create_subscription(CameraInfo, INFO_TOPIC, self._on_info, 2, callback_group=cb)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        ns = str(self.get_parameter('ns').value).rstrip('/')
        self.target_pub = self.create_publisher(PickTarget, f'{ns}/box_model/target', 10)
        self.image_pub = self.create_publisher(
            Image, f'{ns}/box_model/image',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.check_pub = self.create_publisher(
            String, f'{ns}/box_model/check_result',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.overlay = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native', 'cache',
                                    'box_picker_overlay.jpg')
        self.detect_lock = threading.Lock()
        self.service_name = f'{ns}/box_model/detect'
        self.create_service(DetectTarget, self.service_name, self._on_detect,
                            callback_group=cb)
        self.create_service(CheckPick, f'{ns}/box_model/check', self._on_check,
                            callback_group=cb)
        self.picker = self.bridge_proc = None
        self.detector = None
        self.load_error = None
        pdir = str(self.get_parameter('picker_dir').value)
        if not os.path.exists(os.path.join(pdir, '.venv', 'bin', 'python')):
            self.load_error = (f'the box picker is not installed ({pdir}/.venv missing): '
                               f'cd box_picker/src && ./setup.sh')
            self.get_logger().error(self.load_error)
        else:
            self.ready = threading.Event()
            self.restart_lock = threading.Lock()
            self._stop_orphans()
            threading.Thread(target=self._load_picker, daemon=True).start()

    def _on_info(self, msg):
        with self.lock:
            self.info = msg

    def lookup(self, target, source):
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        except Exception:                            # noqa: BLE001
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        m = np.eye(4)
        m[:3, :3] = matrix_from_quat(q.x, q.y, q.z, q.w)
        m[:3, 3] = (t.x, t.y, t.z)
        return m

    def _on_detect(self, request, response):
        with self.detect_lock:
            try:
                found, message, target = self.detect(
                    request.approach or self.get_parameter('approach').value,
                    request.tool or str(self.get_parameter('tool').value))
            except Exception as exc:                 # noqa: BLE001 - answered, not raised
                found, message, target = False, f'detection failed: {exc}', PickTarget()
        response.found, response.message, response.target = found, message, target
        self.get_logger().info(message if found else f'no target: {message}')
        return response

    def _picker_env(self):
        """The environment for the picker's own Python 3.11: without the ROS /
        native PYTHONPATH and library paths (Python 3.10 packages, another CUDA)."""
        env = dict(os.environ)
        for k in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'VIRTUAL_ENV', 'AMENT_PREFIX_PATH',
                  'COLCON_PREFIX_PATH', 'CMAKE_PREFIX_PATH'):
            env.pop(k, None)
        return env

    def _stop_orphans(self):
        """pick_server.py / ros_bridge.py left running by a run that was killed
        (their parent gone): each picker holds ~4 GB of GPU memory, and one left
        over made the next run's first detection run out (2026-10-27)."""
        import signal
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode()
                ppid = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ('pick_server.py' in cmd or 'box_picker_server.py' in cmd
                    or 'ros_bridge.py' in cmd) and ppid == 1:
                self.get_logger().warn(f'stopping a box picker left from an earlier run: '
                                       f'pid {pid}')
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except OSError:
                    pass

    def _last_error(self):
        """The last line of the picker's own log that says what went wrong."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native', 'logs',
                            'box_picker.log')
        try:
            lines = [ln.strip() for ln in open(path, errors='replace') if ln.strip()]
        except OSError:
            return ''
        for ln in reversed(lines[-80:]):
            if 'Error' in ln or 'error' in ln or 'out of memory' in ln:
                return ln[:200]
        return lines[-1][:200] if lines else ''

    def _load_picker(self):
        """ros_bridge.py (this Python: it needs rclpy) and pick_server.py --ros
        (the picker's .venv), the camera's tilt from the TF."""
        import subprocess
        pdir = str(self.get_parameter('picker_dir').value)
        logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native', 'logs')
        os.makedirs(logdir, exist_ok=True)
        self.ready.clear()
        try:
            if self.bridge_proc is None or self.bridge_proc.poll() is not None:
                self.bridge_proc = subprocess.Popen(
                    [sys.executable, 'ros_bridge.py'], cwd=pdir,
                    stdout=open(os.path.join(logdir, 'box_picker_bridge.log'), 'w'),
                    stderr=subprocess.STDOUT)
            # the camera's tilt: the colour optical frame's forward axis
            pitch, deadline = None, time.monotonic() + 30.0
            while pitch is None and time.monotonic() < deadline:
                with self.lock:
                    info = self.info
                t_wo = self.lookup(str(self.get_parameter('world_frame').value),
                                   info.header.frame_id) if info is not None else None
                if t_wo is not None:
                    pitch = float(np.degrees(np.arcsin(np.clip(-t_wo[2, 2], -1, 1))))
                else:
                    time.sleep(0.5)
            if pitch is None:
                raise RuntimeError('no camera info / tf for the camera after 30 s')
            time.sleep(2.0)                                  # the bridge's first frames
            # our box_picker_server.py: pick_server.py's answer plus the drawing
            os.makedirs(os.path.dirname(self.overlay), exist_ok=True)
            self.picker = subprocess.Popen(
                [os.path.join(pdir, '.venv', 'bin', 'python'),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), 'box_picker_server.py'),
                 '--pitch', f'{pitch:.1f}', '--ros',
                 '--frames', str(int(self.get_parameter('frames').value)),
                 '--overlay', self.overlay],
                cwd=pdir, env=self._picker_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=open(os.path.join(logdir, 'box_picker.log'), 'w'), text=True, bufsize=1)
            self._start_reader()
            first = self._picker_line(120.0, key='ready')
            if not first or not first.get('ready'):
                raise RuntimeError(f'pick_server.py did not start ({first}); see '
                                   f'native/logs/box_picker.log')
            self.detector = 'picker'
            self.load_error = None
            self.ready.set()
            self.get_logger().info(f'box picker ready (camera pitch {pitch:.1f} deg); serving '
                                   f'{self.service_name}')
        except Exception as exc:                     # noqa: BLE001 - reported per request
            why = self._last_error()
            self.load_error = f'box picker failed to start: {exc}' + (f' - {why}' if why else '')
            self.get_logger().error(self.load_error)
            self.ready.set()                             # (waiters see load_error)

    def _start_reader(self):
        """One thread reads all of pick_server.py's stdout into a queue (two
        readers racing for one pipe lost answers)."""
        import queue
        self.lines = queue.Queue()
        proc = self.picker

        def read():
            for line in proc.stdout:
                self.lines.put(line)
            self.lines.put(None)                         # the process ended
        threading.Thread(target=read, daemon=True).start()

    def _picker_line(self, timeout, key='target'):
        """The next JSON line from pick_server.py that has `key`; None if it
        ended or `timeout` s passed (self.picker_ended says which)."""
        import json
        import queue
        deadline = time.monotonic() + timeout
        self.picker_ended = False
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return None
            if line is None:
                self.picker_ended = True
                return None
            try:
                obj = json.loads(line)
            except ValueError:
                self.get_logger().info(f'box picker said: {line.strip()[:160]}')
                continue
            if isinstance(obj, dict) and key in obj:
                return obj

    def _publish_overlay(self, ans):
        """The picker's drawing of this answer, on box_model/image."""
        import cv2
        path = ans.get('overlay')
        img = cv2.imread(path) if path else None
        if img is None:
            return
        boxes = ans.get('boxes') or []
        t = ans.get('target')
        line = (f'{len(boxes)} boxes - pick: {t["approach"].upper()} face '
                f'{t["size_mm"][0]} x {t["size_mm"][1]} mm ({t.get("agreeing", "?")}/'
                f'{t.get("frames", "?")} frames)' if t else f'{len(boxes)} boxes - nothing to pick')
        h = img.shape[0]
        cv2.rectangle(img, (0, h - 30), (img.shape[1], h), (0, 0, 0), -1)
        cv2.putText(img, line, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if t else (255, 255, 255), 2, cv2.LINE_AA)
        self._publish_image(None, img)
        self.last_answer = ans

    def _publish_image(self, path, img):
        """An image (or the file at path) on box_model/image."""
        import cv2
        if img is None:
            img = cv2.imread(path) if path else None
            if img is None:
                return
        msg = Image(height=img.shape[0], width=img.shape[1], encoding='bgr8', is_bigendian=0,
                    step=img.shape[1] * 3, data=img.tobytes())
        msg.header.stamp = self.get_clock().now().to_msg()
        with self.lock:
            info = self.info
        msg.header.frame_id = info.header.frame_id if info is not None else ''
        self.image_pub.publish(msg)

    def detect_picker(self):
        """pick_server.py's answer, into the world frame: (found, message, target)."""
        if not hasattr(self, 'ready'):
            return False, self.load_error, PickTarget()
        if not self.ready.wait(90.0):                    # still loading: wait for it
            return False, 'the box picker is still starting - try again', PickTarget()
        if self.picker is None or self.picker.poll() is not None:
            # it died (e.g. the GPU ran out): once more, then this request
            with self.restart_lock:
                if self.picker is None or self.picker.poll() is not None:
                    why = self._last_error()
                    self.get_logger().warn('the box picker had stopped' +
                                           (f' ({why})' if why else '') + ' - starting it again')
                    self._load_picker()
            if self.picker is None or self.picker.poll() is not None or self.load_error:
                return False, self.load_error or 'the box picker is not running', PickTarget()
        self.picker.stdin.write('pick\n')
        self.picker.stdin.flush()
        ans = self._picker_line(60.0)
        if ans is None:
            if getattr(self, 'picker_ended', False):
                why = self._last_error()
                return False, ('the box picker stopped while looking' +
                               (f': {why}' if why else '') + ' - it restarts on the next try'), \
                    PickTarget()
            return False, 'the box picker did not answer in 60 s', PickTarget()
        self._publish_overlay(ans)
        t = ans.get('target')
        if not t:
            return False, f'no box to pick ({ans.get("seconds", "?")} s)', PickTarget()
        frames, agreeing = int(t.get('frames', 0)), int(t.get('agreeing', 0))
        if agreeing * 2 < frames:
            return False, (f'the frames disagree ({agreeing} of {frames}: something moved) - '
                           f'ask again'), PickTarget()
        with self.lock:
            info = self.info
        world = str(self.get_parameter('world_frame').value)
        t_wo = self.lookup(world, info.header.frame_id) if info is not None else None
        if t_wo is None:
            return False, f'no tf {world} -> the camera', PickTarget()
        p = np.array([t['x_m'], t['y_m'], t['z_m']], float)
        coef = depth_cal.load()                        # the depth check, as for every depth
        if coef is not None and p[2] > 0:
            a, b, c, e = coef
            xn, yn = p[0] / p[2], p[1] / p[2]
            z = p[2] * (1 + a) + b + c * xn + e * yn
            p = np.array([xn * z, yn * z, z])
        position = t_wo[:3, :3] @ p + t_wo[:3, 3]
        target = PickTarget()
        target.header.frame_id = world
        target.header.stamp = self.get_clock().now().to_msg()
        target.position.x, target.position.y, target.position.z = (float(v) for v in position)
        target.approach = str(t.get('approach', 'front'))
        face = ans.get('face')
        if face:                                         # the face, into the world
            r = t_wo[:3, :3]
            for vec, key in ((target.face_normal, 'normal'), (target.face_long, 'long')):
                vec.x, vec.y, vec.z = (float(v) for v in r @ np.asarray(face[key], float))
            target.face_mm = [float(v) for v in face['size_mm']]
        target.confidence = float(agreeing) / max(frames, 1)
        target.u, target.v = float(t.get('u', 0.0)), float(t.get('v', 0.0))
        target.bbox = [0.0, 0.0, 0.0, 0.0]                 # the picker gives the face, not a box
        target.image_width = int(info.width) if info is not None else 0
        target.image_height = int(info.height) if info is not None else 0
        target.num_boxes = int(t.get('num_boxes', 0))
        self.target_pub.publish(target)
        size = t.get('size_mm') or ['?', '?']
        return True, (f'{target.approach}, {agreeing} of {frames} frames agree, face '
                      f'{size[0]} x {size[1]} mm, {target.num_boxes} boxes seen, at '
                      f'x={position[0]:+.3f} y={position[1]:+.3f} z={position[2]:+.3f} '
                      f'({ans.get("seconds", "?")} s)'), target

    # -- the pick check -----------------------------------------------------------

    def _on_check(self, request, response):
        import json
        with self.detect_lock:
            try:
                picked, outcome, message, verdict = self.check_pick(request.holding.strip().lower())
            except Exception as exc:                 # noqa: BLE001 - answered, not raised
                picked, outcome, message, verdict = None, '', f'check failed: {exc}', None
        response.picked = {True: 1, False: 0}.get(picked, -1)
        response.outcome, response.message = outcome, message
        response.verdict_json = json.dumps(verdict)
        self.check_pub.publish(String(data=json.dumps(
            {'picked': picked, 'outcome': outcome, 'message': message, 'verdict': verdict,
             'time': time.strftime('%Y-%m-%d %H:%M:%S')})))
        self.get_logger().info(f'pick check: {message}')
        return response

    def check_pick(self, holding):
        """The picker's pick check: (picked True/False/None, outcome, message, verdict)."""
        if not hasattr(self, 'ready') or not self.ready.is_set() or \
                self.picker is None or self.picker.poll() is not None:
            return None, '', 'no check: the box picker is not running', None
        word = holding if holding in ('held', 'empty') else ''
        self.picker.stdin.write(f'check {word}'.strip() + '\n')
        self.picker.stdin.flush()
        ans = self._picker_line(20.0, key='check')
        if ans is None:
            return None, '', 'no check: the box picker did not answer in 20 s', None
        self._publish_image(ans.get('overlay'), None)
        verdict = ans.get('check')
        if verdict is None:
            return None, '', ans.get('message', 'nothing to check'), None
        return ans.get('picked'), verdict['outcome'], ans.get('message', verdict['outcome']), verdict

    # -- a request ----------------------------------------------------------------

    def detect(self, approach, tool='vacuum'):
        if approach not in ('auto', 'top', 'front', ''):
            return False, f'approach must be auto, top or front, not "{approach}"', PickTarget()
        tool = tool.strip().lower()
        if tool not in TOOL_ARM:
            return False, f'tool must be vacuum or finger, not "{tool}"', PickTarget()
        # the picker chooses the face by its rules: a forced approach would grip
        # a face it did not measure -- only reported
        found, msg, target = self.detect_picker()
        target.tool, target.arm = tool, TOOL_ARM[tool]
        if found:
            msg += f'; {tool} -> {target.arm} arm'
        if found and approach not in ('auto', '') and target.approach != approach:
            msg += f' (asked for {approach}: the picker\'s rules chose {target.approach})'
        return found, msg, target


def main():
    rclpy.init()
    node = BoxModelNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (getattr(node, 'picker', None), getattr(node, 'bridge_proc', None)):
            try:
                if proc is not None and proc.poll() is None:
                    if proc is node.picker:
                        proc.stdin.write('quit\n')
                        proc.stdin.flush()
                        proc.wait(timeout=5)
                    else:
                        proc.terminate()
            except Exception:                        # noqa: BLE001
                if proc is not None:
                    proc.kill()
        try:
            executor.shutdown()
            node.destroy_node()
        except BaseException:                        # noqa: BLE001 - a second Ctrl-C
            pass
        rclpy.try_shutdown()
        os._exit(0)                                  # (no waiting on worker threads)


if __name__ == '__main__':
    main()
