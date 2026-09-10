#!/usr/bin/env python3
"""The pick-and-place panel, as a web page.

One process holding two things: an rclpy node subscribed to everything worth
watching, and an aiohttp server that hands it to a browser. Run it on the robot
and open it from anywhere on the network -- which is the point, because the
workspace lives on one machine and is driven from another.

    python3 pick_place_web.py                    # http://<host>:8088
    python3 pick_place_web.py --port 9000 --host 127.0.0.1

What it serves:

    /                 the page
    /api/status       one snapshot, for a first paint before the socket opens
    /api/config       GET the sequence, the steps available and the settings
                      POST a new sequence or settings; written to the config
                      file and then reloaded by the orchestrator, so the file
                      stays the single source of truth
    /api/<action>     start, abort, open_gripper, grip -- the same services
                      the Tk panel calls
    /camera.jpg       one frame. ?pinned=1 is the frame the last detection
                      was made on, which is the one worth looking at
    /urdf             the robot description, for the 3D view
    /meshes/...       the meshes it references
    /ws               live status, joint states, TCP, motion log and console

Nothing here plans or moves anything itself. Every action is a service call to
the orchestrator, which is what keeps the guards -- the travel budget, the
octomap exemption, the refusal to fly home from a low posture -- in one place
rather than duplicated behind a button.
"""

import argparse
import asyncio
import json
import os
import threading
import time

from aiohttp import web

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

import pick_place_sequence as sequence_model
from vlm_prompt import to_detection_prompt

WS = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(WS, 'pick_place_web.html')

# Where the description's meshes may come from. package:// URLs are resolved
# against these and nothing else -- the browser asks for whatever the URDF
# names, and without a whitelist that is a path traversal with extra steps.
MESH_ROOTS = {
    'openarm_description': os.path.join(WS, 'src', 'openarm_description'),
    'openarm_bimanual_moveit_config': os.path.join(
        WS, 'src', 'openarm_ros2', 'openarm_bimanual_moveit_config'),
}

ACTIONS = {
    'start': '/pick_place/start',
    'abort': '/pick_place/abort',
    'open_gripper': '/pick_place/open_gripper',
    'grip': '/pick_place/grip',
    'reload_config': '/pick_place/reload_config',
    'save_config': '/pick_place/save_config',
}


class Bridge(Node):
    """Everything the page needs, kept current and handed over on request.

    Deliberately a snapshot rather than a stream of deltas: the page can be
    opened at any moment, including mid-cycle, and "here is the whole state"
    is the only thing that renders correctly on the first frame.
    """

    def __init__(self):
        super().__init__('pick_place_web')
        self.cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()

        self.status = {}
        self.state_lines = []
        self.joints = {}
        self.efforts = {}
        self.velocities = {}
        self.frame = None                 # latest JPEG bytes
        self.frame_stamp = 0.0
        # The frame the last detection was made on, kept until the next one.
        # A live stream was the wrong thing: the interesting frame is the one
        # the arm decided from, and it stays interesting for the minute the
        # cycle then takes. Streaming it thirty times a second to show the
        # same tabletop is bandwidth spent on nothing.
        self.pinned = None
        self.pinned_id = 0
        self.pinned_at = 0.0
        self.detections = {}
        self.motions = []
        self.urdf = ''
        self.listeners = set()            # asyncio.Queue per open socket
        self.loop = None                  # set once the server is running

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/pick_place/status',
                                 self._on_status, latched,
                                 callback_group=self.cb)
        self.create_subscription(String, '/pick_place/state',
                                 self._on_state, latched,
                                 callback_group=self.cb)
        self.create_subscription(JointState, '/joint_states',
                                 self._on_joints, 10, callback_group=self.cb)
        self.create_subscription(String, '/vlm/detections',
                                 self._on_detections, 10,
                                 callback_group=self.cb)
        self.create_subscription(String, '/robot_description',
                                 self._on_urdf, latched,
                                 callback_group=self.cb)
        # The annotated frame, because it is the one that shows what the
        # detector actually decided. Falls back to raw colour if the detector
        # is not running, so the view is never simply blank.
        self.create_subscription(Image, '/vlm/debug_image',
                                 self._on_image, 1, callback_group=self.cb)
        self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._on_raw_image, 1,
            callback_group=self.cb)

        self.prompt_pub = self.create_publisher(
            String, '/pick_place/prompt', latched)
        # Not `self.clients`: rclpy.Node already owns that name, and
        # assigning over it fails at construction with a bare AttributeError.
        self.triggers = {name: self.create_client(Trigger, path,
                                                  callback_group=self.cb)
                         for name, path in ACTIONS.items()}
        self.set_params = self.create_client(
            SetParameters, '/pick_place_orchestrator/set_parameters',
            callback_group=self.cb)
        self.get_params = self.create_client(
            GetParameters, '/pick_place_orchestrator/get_parameters',
            callback_group=self.cb)

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)
        self.create_timer(0.1, self._tick, callback_group=self.cb)

        self._cv = None
        try:
            import cv2
            self._cv = cv2
        except ImportError:
            self.get_logger().warn(
                'cv2 is not importable, so the camera panel will stay empty')

    # -- inbound -----------------------------------------------------------

    def _on_status(self, msg):
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.status = status
        self.push({'kind': 'status', 'status': status})

    def _on_state(self, msg):
        line = f'{time.strftime("%H:%M:%S")}  {msg.data}'
        with self.lock:
            self.state_lines.append(line)
            del self.state_lines[:-400]
        self.push({'kind': 'log', 'line': line})

    def _on_joints(self, msg):
        with self.lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.joints[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.efforts[name] = msg.effort[i]

    def _on_detections(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.detections = payload
            # Pin the frame this was decided on. Only when something was
            # actually found: an empty detection is not worth replacing a
            # picture of the object with.
            if payload.get('detections') and self.frame is not None:
                self.pinned = self.frame
                self.pinned_id += 1
                self.pinned_at = time.time()
                pinned_id = self.pinned_id
            else:
                pinned_id = None
        if pinned_id is not None:
            self.push({'kind': 'frame', 'id': pinned_id,
                       'detections': payload.get('detections', [])})

    def _on_urdf(self, msg):
        with self.lock:
            self.urdf = msg.data

    def _on_image(self, msg):
        self._store_frame(msg, annotated=True)

    def _on_raw_image(self, msg):
        # Only when the detector is quiet: its frame is strictly better.
        if time.time() - self.frame_stamp < 2.0:
            return
        self._store_frame(msg, annotated=False)

    def _store_frame(self, msg, annotated):
        if self._cv is None:
            return
        try:
            import numpy as np
            channels = max(1, msg.step // max(1, msg.width))
            array = np.frombuffer(msg.data, dtype=np.uint8)
            array = array.reshape(msg.height, msg.width, channels)
            if msg.encoding in ('rgb8', 'rgba8'):
                array = array[:, :, ::-1] if channels == 3 \
                    else array[:, :, [2, 1, 0, 3]]
            ok, buffer = self._cv.imencode(
                '.jpg', array, [self._cv.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                return
        except Exception as exc:                     # noqa: BLE001 - reported
            self.get_logger().warn(f'could not encode a frame: {exc}',
                                   throttle_duration_sec=10.0)
            return
        with self.lock:
            self.frame = buffer.tobytes()
            if annotated:
                self.frame_stamp = time.time()

    def _tick(self):
        """The things that are polled rather than pushed: TF and the log."""
        payload = {'kind': 'tick'}
        for arm in ('left', 'right'):
            try:
                tf = self.tf_buffer.lookup_transform(
                    'world', f'openarm_{arm}_hand_tcp', rclpy.time.Time())
            except Exception:                        # noqa: BLE001
                continue
            t = tf.transform.translation
            payload[f'{arm}_tcp'] = [round(t.x, 4), round(t.y, 4),
                                     round(t.z, 4)]
        with self.lock:
            payload['joints'] = dict(self.joints)
            payload['efforts'] = dict(self.efforts)
            payload['velocities'] = dict(self.velocities)
            payload['detections'] = self.detections.get('detections', [])
        self.push(payload)

    # -- outbound ----------------------------------------------------------

    def push(self, message):
        """Fan a message out to every open socket, from the ROS thread.

        call_soon_threadsafe because this runs on an executor thread and the
        queues belong to the event loop. A socket whose queue has backed up is
        dropped rather than blocking the ROS side -- a slow browser must not
        be able to stall the node.
        """
        if self.loop is None:
            return
        try:
            self.loop.call_soon_threadsafe(self._fan_out, message)
        except RuntimeError:
            pass

    def _fan_out(self, message):
        for queue in list(self.listeners):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                self.listeners.discard(queue)

    def snapshot(self):
        with self.lock:
            return {
                'status': dict(self.status),
                'log': list(self.state_lines[-200:]),
                'joints': dict(self.joints),
                'efforts': dict(self.efforts),
                'velocities': dict(self.velocities),
                'detections': self.detections.get('detections', []),
                'motions': list(self.motions[-50:]),
                'has_camera': self.frame is not None,
                'frame_id': self.pinned_id,
            }

    # -- actions -----------------------------------------------------------

    def call(self, name, timeout=10.0):
        client = self.triggers.get(name)
        if client is None:
            return False, f'{name} is not an action'
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f'{ACTIONS[name]} is not being served -- is the ' \
                          f'orchestrator running?'
        future = client.call_async(Trigger.Request())
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return False, f'{name} did not answer in {timeout:.0f} s'
        result = future.result()
        if result is None:
            return False, f'{name} failed'
        return bool(result.success), result.message

    def publish_prompt(self, text):
        """Send what the operator typed. The orchestrator does the
        translating, so a bare `ros2 topic pub` behaves the same way."""
        self.prompt_pub.publish(String(data=text))

    def apply_settings(self, settings):
        """Set parameters on the orchestrator. Returns (ok, message)."""
        parameters = []
        for name, value in (settings or {}).items():
            entry = sequence_model.SETTINGS_BY_NAME.get(name)
            if entry is None:
                continue
            coerced, complaint = sequence_model.coerce_setting(entry, value)
            if coerced is None:
                return False, complaint or f'{name} is not settable'
            message = ParameterMsg(name=name)
            if entry['type'] == 'bool':
                message.value = ParameterValue(
                    type=ParameterType.PARAMETER_BOOL, bool_value=coerced)
            else:
                message.value = ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(coerced))
            parameters.append(message)
        if not parameters:
            return True, 'nothing to set'
        if not self.set_params.wait_for_service(timeout_sec=2.0):
            return False, 'the orchestrator is not accepting parameters'
        future = self.set_params.call_async(
            SetParameters.Request(parameters=parameters))
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(10.0):
            return False, 'setting parameters timed out'
        result = future.result()
        refused = [f'{p.name}: {r.reason}'
                   for p, r in zip(parameters, result.results)
                   if not r.successful]
        if refused:
            return False, '; '.join(refused)
        return True, f'{len(parameters)} setting(s) applied'

    def read_settings(self):
        """The live parameter values, so the page shows the robot rather than
        the file -- they differ whenever a launch argument overrode one."""
        names = [entry['name'] for entry in sequence_model.SETTINGS]
        if not self.get_params.wait_for_service(timeout_sec=2.0):
            return {}
        future = self.get_params.call_async(
            GetParameters.Request(names=names))
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(5.0):
            return {}
        result = future.result()
        if result is None:
            return {}
        values = {}
        for name, value in zip(names, result.values):
            if value.type == ParameterType.PARAMETER_BOOL:
                values[name] = value.bool_value
            elif value.type == ParameterType.PARAMETER_DOUBLE:
                values[name] = round(value.double_value, 6)
            elif value.type == ParameterType.PARAMETER_INTEGER:
                values[name] = value.integer_value
        return values


# -- the server -------------------------------------------------------------


def config_path():
    """The same file the orchestrator reads, found the same way.

    From the environment rather than a flag, because the launch file sets it
    for both processes at once -- two halves disagreeing about where the
    config lives would be the sort of bug that looks like the UI silently
    ignoring you.
    """
    name = os.environ.get('PICK_PLACE_CONFIG', 'pick_place_config.json')
    return name if os.path.isabs(name) else os.path.join(WS, name)


async def index(request):
    if not os.path.exists(PAGE):
        return web.Response(status=500, text=f'{PAGE} is missing')
    return web.FileResponse(PAGE)


async def api_status(request):
    bridge = request.app['bridge']
    snapshot = bridge.snapshot()
    snapshot['settings'] = await asyncio.to_thread(bridge.read_settings)
    return web.json_response(snapshot)


async def api_config(request):
    bridge = request.app['bridge']
    path = config_path()
    if request.method == 'GET':
        saved, problems = sequence_model.load_config(path)
        live = await asyncio.to_thread(bridge.read_settings)
        return web.json_response({
            'path': path,
            'sequence': saved['sequence'],
            'prompt': saved['prompt'],
            'problems': problems,
            'catalogue': sequence_model.describe(saved['sequence']),
            'fields': sequence_model.SETTINGS,
            'settings': live or saved['settings'],
        })

    body = await request.json()
    problems = []
    settings = body.get('settings')
    if settings:
        ok, message = await asyncio.to_thread(bridge.apply_settings, settings)
        if not ok:
            return web.json_response({'ok': False, 'error': message},
                                     status=400)
        problems.append(message)

    saved, _ = sequence_model.load_config(path)
    if 'sequence' in body:
        cleaned, complaints = sequence_model.validate_sequence(
            body['sequence'])
        # An unusual order is the operator's business; an incoherent one is
        # not an order at all. Corrections are applied and reported; an unmet
        # dependency is refused, because saving it would mean the next cycle
        # closes the jaws in mid-air and the file is then the record of a
        # decision nobody made.
        blocking = sequence_model.sequence_blockers(cleaned)
        if blocking:
            return web.json_response(
                {'ok': False, 'error': '; '.join(blocking),
                 'problems': complaints, 'sequence': cleaned}, status=400)
        saved['sequence'] = cleaned
        problems.extend(complaints)
    if isinstance(body.get('prompt'), str):
        saved['prompt'] = body['prompt']
    live = await asyncio.to_thread(bridge.read_settings)
    saved['settings'] = live or saved['settings']

    why = sequence_model.save_config(path, saved)
    if why:
        return web.json_response({'ok': False, 'error': why}, status=500)

    # The file is the source of truth, so the orchestrator re-reads it rather
    # than being told separately. If it will not (a cycle is running) the save
    # still stands and takes effect on the next one -- say which.
    ok, message = await asyncio.to_thread(bridge.call, 'reload_config')
    if not ok:
        problems.append(f'saved, but not live yet: {message}')
    return web.json_response({'ok': True, 'sequence': saved['sequence'],
                              'problems': problems, 'reloaded': ok})


async def api_action(request):
    bridge = request.app['bridge']
    name = request.match_info['name']
    if name not in ACTIONS:
        return web.json_response({'ok': False, 'error': f'no action {name}'},
                                 status=404)
    if name == 'start':
        body = await request.json() if request.can_read_body else {}
        prompt = (body or {}).get('prompt')
        if prompt:
            bridge.publish_prompt(prompt)
            # The orchestrator applies the same translation to anything
            # arriving on the topic; give it a moment to land before start
            # reads it, or the cycle runs on the previous object.
            await asyncio.sleep(0.3)
    ok, message = await asyncio.to_thread(bridge.call, name)
    return web.json_response({'ok': ok, 'message': message})


async def api_translate(request):
    """What the detector will actually be asked, for the field to show."""
    text = request.query.get('text', '')
    return web.json_response({'prompt': to_detection_prompt(text)})


async def camera(request):
    """One JPEG.

    ?pinned=1 gives the frame the last detection was made on -- the one the
    arm decided from, and the only one worth looking at while it works. Plain
    gives whatever the camera has right now, for the refresh button.

    A still rather than a stream because the picture does not change during a
    cycle and pushing it thirty times a second was bandwidth spent on nothing.
    """
    bridge = request.app['bridge']
    with bridge.lock:
        pinned = request.query.get('pinned') not in (None, '', '0')
        frame = bridge.pinned if pinned else bridge.frame
        taken = bridge.pinned_at if pinned else bridge.frame_stamp
    if frame is None:
        return web.Response(
            status=404,
            text='no frame yet -- is the camera up, and has anything been '
                 'detected?')
    return web.Response(body=frame, content_type='image/jpeg', headers={
        # The URL carries a cache-buster, but say it anyway: a stale frame
        # shown next to a live state is worse than no frame.
        'Cache-Control': 'no-store',
        'X-Frame-Taken': f'{taken:.3f}',
    })


async def urdf(request):
    bridge = request.app['bridge']
    with bridge.lock:
        text = bridge.urdf
    if not text:
        return web.Response(status=503,
                            text='/robot_description has not arrived')
    return web.Response(text=text, content_type='application/xml')


async def mesh(request):
    """A mesh the URDF names, from a whitelisted package.

    The browser asks for whatever it read in the description, so the path is
    checked against the package root it claims to be in rather than trusted.
    """
    package = request.match_info['package']
    tail = request.match_info['path']
    root = MESH_ROOTS.get(package)
    if root is None:
        return web.Response(status=404, text=f'{package} is not served')
    full = os.path.normpath(os.path.join(root, tail))
    if not full.startswith(os.path.normpath(root) + os.sep):
        return web.Response(status=403, text='outside the package')
    if not os.path.exists(full):
        return web.Response(status=404, text=f'{tail} is not there')
    return web.FileResponse(full)


async def websocket(request):
    bridge = request.app['bridge']
    socket = web.WebSocketResponse(heartbeat=20.0)
    await socket.prepare(request)
    queue = asyncio.Queue(maxsize=200)
    bridge.listeners.add(queue)
    try:
        await socket.send_json({'kind': 'snapshot', **bridge.snapshot()})
        while not socket.closed:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                continue
            await socket.send_json(message)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        bridge.listeners.discard(queue)
    return socket


def build(bridge):
    app = web.Application()
    app['bridge'] = bridge
    app.add_routes([
        web.get('/', index),
        web.get('/api/status', api_status),
        web.get('/api/config', api_config),
        web.post('/api/config', api_config),
        web.get('/api/translate', api_translate),
        web.post('/api/action/{name}', api_action),
        web.get('/camera.jpg', camera),
        web.get('/urdf', urdf),
        web.get('/meshes/{package}/{path:.*}', mesh),
        web.get('/ws', websocket),
    ])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0',
                        help='interface to bind; 127.0.0.1 for local only')
    parser.add_argument('--port', type=int, default=8088)
    args = parser.parse_args()

    rclpy.init()
    bridge = Bridge()
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    app = build(bridge)

    async def remember_loop(_app):
        bridge.loop = asyncio.get_running_loop()
    app.on_startup.append(remember_loop)

    print(f'pick and place UI on http://{args.host}:{args.port}')
    try:
        web.run_app(app, host=args.host, port=args.port,
                    print=lambda *_a, **_k: None)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
