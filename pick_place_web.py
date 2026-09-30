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

from control_msgs.msg import DynamicJointState, JointTrajectoryControllerState

try:
    from controller_manager_msgs.srv import ListHardwareComponents
except ImportError:                                  # no ros2_control here
    # Optional on purpose. It answers "are the motors on", which nothing
    # else can, but the panel's other half -- is each motor still
    # answering -- comes from /joint_states and works without it.
    ListHardwareComponents = None
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

import pick_place_sequence as sequence_model
import grasp_recipes
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

# How a motor is judged, in seconds.
#
# There is no health flag to read. openarm_hardware exports position,
# velocity and effort per joint and nothing else, so a motor that has
# stopped answering on the CAN bus does not report an error -- its three
# numbers simply stop changing, because read() copies whatever the last
# received frame held.
#
# That is still a signal, and a strong one. These are direct-drive motors
# holding an arm up against gravity: measured at rest in motion_log.jsonl,
# every joint's effort moves on every sample -- joint1 read 3.5473, 3.3099,
# 3.4418, 3.9429 Nm on four consecutive stills -- and the velocities sit at
# +/-0.01 rad/s rather than at zero. A reading that is bit-identical for
# seconds is not a still arm, it is a motor that is no longer being heard
# from.
#
# Generous, because the cost of the two mistakes is not equal: calling a
# working motor dead sends somebody to the robot for nothing.
MOTOR_QUIET_SECONDS = 4.0
# No /joint_states at all for this long and the question is not about one
# motor any more.
MOTOR_TOPIC_SECONDS = 2.0
# What each motor says about itself, decoded from the status nibble every
# Damiao feedback frame carries. openarm_can was throwing byte 0 away, so
# nothing in the ROS stack could tell a motor that is driving from one that
# is merely reporting; it is parsed now and exported by openarm_hardware as
# each joint's "status" state interface, which reaches here on
# /dynamic_joint_states.
MOTOR_STATUS = {
    0: ('off', 'not enabled'),
    1: ('live', 'enabled'),
    8: ('fault', 'over-voltage'),
    9: ('fault', 'under-voltage'),
    10: ('fault', 'over-current'),
    11: ('fault', 'driver over-temperature'),
    12: ('fault', 'motor over-temperature'),
    13: ('fault', 'lost communication'),
    14: ('fault', 'overloaded'),
}
# How stale that reading may be before it is ignored, in seconds.
MOTOR_STATUS_AGE = 2.0
# How far a joint may sit from the position it was commanded to hold, in
# radians, before it counts as not holding it.
#
# This is the question "did the reading change" cannot answer, and the
# reason it is asked separately. A motor with no torque still has an
# encoder: it reports, the numbers move -- more than a driven joint's, not
# less, because gravity is pulling it -- and every freshness check in the
# world calls it healthy. What it does not do is stay where it was put.
#
# Measured on this robot, both arms parked and holding still, 194 samples
# of each controller's own state: every joint sat within 0.00038 rad of its
# commanded position except two, at 0.00286 and 0.00506. 0.002 rad is the
# gap between those two groups, and it is about an eighth of a degree.
MOTOR_HOLD_ERROR = 0.002
# ...but only once the joint has been asked to hold still for this long.
# During a move the tracking error is the arm lagging its trajectory, which
# is ordinary and says nothing about the motor.
MOTOR_HOLD_SETTLE = 1.5
# How often the hardware components are asked for their lifecycle state.
# It is the authoritative answer to "are the motors on", and it is what
# separates a disengaged arm from a broken one -- but it is a service call,
# so it is polled slowly rather than per frame.
HARDWARE_POLL_SECONDS = 3.0

ACTIONS = {
    'start': '/pick_place/start',
    'place': '/pick_place/place',
    'place_left': '/pick_place/place_left',
    'place_right': '/pick_place/place_right',
    'stop_arm': '/pick_place/stop_arm',
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
        # Per joint: when it was last heard, and when its reading last
        # actually changed. See MOTOR_QUIET_SECONDS.
        self.motor_seen = {}
        self.joints_at = 0.0
        # {component name: lifecycle label}, from the controller manager.
        self.hardware = {}
        self.hardware_at = 0.0
        # Per joint, from its controller: how far it is from the position
        # it was told to hold, and how long it has been told to hold it.
        self.tracking = {}
        # Per joint, what the motor itself reports. See MOTOR_STATUS.
        self.reported = {}
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
        # The motor's own word for whether it is driving. Everything else
        # here is inference: an encoder reports the same whether or not
        # there is any torque behind it, and a joint carrying no load holds
        # its place on friction alone. This is the only direct answer, and
        # it is absent on a stack whose openarm_hardware does not export
        # it -- in which case the inference below is all there is, and the
        # panel says so rather than showing green.
        self.create_subscription(DynamicJointState, '/dynamic_joint_states',
                                 self._on_reported, 10,
                                 callback_group=self.cb)
        # The controller's own view, which is the only place the commanded
        # position is published. /joint_states says where a joint is;
        # this says where it was told to be.
        for arm in ('left', 'right'):
            self.create_subscription(
                JointTrajectoryControllerState,
                f'/{arm}_joint_trajectory_controller/controller_state',
                self._on_tracking, 10, callback_group=self.cb)
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

        # Whether the motors are on at all, which /joint_states cannot say:
        # a deactivated component stops being read but keeps its last values
        # on the state interfaces, so the topic carries on looking normal.
        self.hardware_client = None
        if ListHardwareComponents is not None:
            self.hardware_client = self.create_client(
                ListHardwareComponents,
                '/controller_manager/list_hardware_components',
                callback_group=self.cb)

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)
        self.create_timer(0.1, self._tick, callback_group=self.cb)
        self.create_timer(HARDWARE_POLL_SECONDS, self._poll_hardware,
                          callback_group=self.cb)

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
        now = time.time()
        with self.lock:
            self.joints_at = now
            for i, name in enumerate(msg.name):
                reading = (
                    msg.position[i] if i < len(msg.position) else None,
                    msg.velocity[i] if i < len(msg.velocity) else None,
                    msg.effort[i] if i < len(msg.effort) else None,
                )
                if reading[0] is not None:
                    self.joints[name] = reading[0]
                if reading[1] is not None:
                    self.velocities[name] = reading[1]
                if reading[2] is not None:
                    self.efforts[name] = reading[2]
                # Heard from, and heard saying something different. The
                # second is the interesting one: a motor off the bus is
                # still in msg.name, carrying the values it had when it
                # went.
                was = self.motor_seen.get(name)
                if was is None:
                    self.motor_seen[name] = {
                        'heard': now, 'moved': now, 'last': reading}
                else:
                    was['heard'] = now
                    if reading != was['last']:
                        was['moved'] = now
                        was['last'] = reading

    def _on_reported(self, msg):
        """What each motor says about itself, if the driver exports it.

        openarm_hardware publishes it as a "status" state interface per
        joint; joint_state_broadcaster puts every interface it is given on
        this topic. A joint with no such interface simply never appears
        here, which is the "cannot tell" case and is reported as exactly
        that.
        """
        now = time.time()
        with self.lock:
            for index, name in enumerate(msg.joint_names):
                if index >= len(msg.interface_values):
                    continue
                entry = msg.interface_values[index]
                for which, value in zip(entry.interface_names, entry.values):
                    if which != 'status':
                        continue
                    self.reported[name] = {'code': int(round(value)),
                                           'at': now}

    def _on_tracking(self, msg):
        """How far each joint is from where its controller told it to be.

        Both halves matter. The error is the answer to "is it holding
        position"; the reference is what says whether that question is fair
        yet, because during a move the error is the arm lagging its own
        trajectory and means nothing about the motor.
        """
        now = time.time()
        with self.lock:
            for index, name in enumerate(msg.joint_names):
                if index >= len(msg.error.positions):
                    continue
                wanted = (msg.reference.positions[index]
                          if index < len(msg.reference.positions) else None)
                entry = self.tracking.get(name)
                if entry is None:
                    entry = self.tracking[name] = {
                        'still_since': now, 'wanted': wanted}
                elif wanted is None or entry['wanted'] is None or \
                        abs(wanted - entry['wanted']) > 1e-6:
                    entry['still_since'] = now
                entry['wanted'] = wanted
                entry['error'] = abs(msg.error.positions[index])
                entry['at'] = now
                # The verdict, latched. Only ever set while the joint has
                # been asked to stand still, and only ever cleared by the
                # joint being seen to hold -- see motor_report.
                if now - entry['still_since'] > MOTOR_HOLD_SETTLE:
                    if entry['error'] > MOTOR_HOLD_ERROR:
                        if entry.get('failed_at') is None:
                            entry['failed_at'] = now
                        entry['failed_error'] = entry['error']
                    else:
                        entry['failed_at'] = None
                        entry['failed_error'] = None

    def _poll_hardware(self):
        """Ask the controller manager which hardware components are active.

        The authoritative answer to "are the motors on". An arm whose
        component is INACTIVE is disengaged -- deliberately, by Stop arm or
        by the contact guard -- which looks nothing like a motor that has
        fallen off the bus, and the panel must not say the same thing about
        both. Silent when the service is not there: this panel also runs
        against a stack with no controller manager at all.
        """
        if (self.hardware_client is None
                or not self.hardware_client.service_is_ready()):
            with self.lock:
                if self.hardware and time.time() - self.hardware_at > 10.0:
                    self.hardware = {}
            return
        future = self.hardware_client.call_async(
            ListHardwareComponents.Request())
        future.add_done_callback(self._store_hardware)

    def _store_hardware(self, future):
        try:
            result = future.result()
        except Exception:                            # noqa: BLE001
            return
        if result is None:
            return
        found = {}
        for entry in result.component:
            found[entry.name] = entry.state.label or str(entry.state.id)
        with self.lock:
            self.hardware = found
            self.hardware_at = time.time()

    def motor_report(self):
        """Every motor, and whether it is answering. Green, amber or red.

        Three states, because there are three situations and they need
        different responses:

        * `live`   -- heard from, and its reading has changed inside
                      MOTOR_QUIET_SECONDS. A direct-drive motor holding an
                      arm up is never perfectly still.
        * `quiet`  -- in /joint_states, but saying exactly the same thing it
                      said seconds ago. Either the motors are off, which the
                      component state says, or this one has stopped
                      answering.
        * `silent` -- not on the topic at all, or the topic itself has
                      stopped.
        """
        now = time.time()
        with self.lock:
            seen = {name: dict(entry)
                    for name, entry in self.motor_seen.items()}
            tracking = {name: dict(entry)
                        for name, entry in self.tracking.items()}
            reported = {name: dict(entry)
                        for name, entry in self.reported.items()}
            joints = dict(self.joints)
            efforts = dict(self.efforts)
            hardware = dict(self.hardware)
            topic_at = self.joints_at
        topic_age = now - topic_at if topic_at else None
        stale_topic = topic_age is None or topic_age > MOTOR_TOPIC_SECONDS

        rows = []
        for arm in ('left', 'right'):
            names = [f'openarm_{arm}_joint{i}' for i in range(1, 8)]
            names.append(f'openarm_{arm}_finger_joint1')
            for name in names:
                entry = seen.get(name)
                if entry is None or stale_topic:
                    state = 'silent'
                    quiet = None
                else:
                    quiet = now - entry['moved']
                    state = 'quiet' if quiet > MOTOR_QUIET_SECONDS else 'live'
                # And separately: is it where it was told to be. A motor
                # with no torque still reports -- more movement than a
                # driven one, not less -- so nothing above this can see it.
                held = tracking.get(name)
                off = None
                holding = None
                since = None
                if held and 'error' in held and now - held['at'] < 1.0:
                    off = held['error']
                    if now - held['still_since'] > MOTOR_HOLD_SETTLE:
                        holding = off <= MOTOR_HOLD_ERROR
                    elif held.get('failed_at') is not None:
                        # Mid-move, so this joint cannot be judged now --
                        # during a move the error is the arm lagging its own
                        # trajectory. But it was judged, and it failed, and
                        # a hardware fault does not heal because a cycle
                        # started. The verdict is held until the joint is
                        # next seen standing still and holding.
                        holding = False
                        off = held.get('failed_error', off)
                        since = now - held['failed_at']
                if state == 'live' and holding is False:
                    state = 'adrift'
                # The motor's own word, which outranks all of the above:
                # everything else here is inference from an encoder, and an
                # encoder reads the same whether or not there is torque
                # behind it.
                says = reported.get(name)
                code = None
                if says and now - says['at'] < MOTOR_STATUS_AGE:
                    code = says['code']
                said, why = MOTOR_STATUS.get(code, (None, None))
                if said == 'fault':
                    state = 'fault'
                elif said == 'off':
                    state = 'off'
                elif said is None and state == 'live' and holding is None:
                    # Nothing reported it, and nothing could be inferred
                    # either: the joint is answering, but whether anything
                    # is driving it is not known. Green would be a claim,
                    # and this is the case that showed a limp wrist joint
                    # as healthy -- it bears almost no gravity load, so it
                    # holds its place with no torque at all.
                    state = 'unknown'
                rows.append({
                    'name': name,
                    'arm': arm,
                    'label': ('jaws' if name.endswith('finger_joint1')
                              else 'j' + name[-1]),
                    'state': state,
                    'quiet': None if quiet is None else round(quiet, 1),
                    'position': joints.get(name),
                    'effort': efforts.get(name),
                    'error': None if off is None else round(off, 5),
                    'holding': holding,
                    # What the motor said, and in words. None when the
                    # driver does not export it.
                    'reported': code,
                    'says': why,
                    # Seconds since it was last judged, when the judgement
                    # is a held-over one rather than a fresh one.
                    'judged': None if since is None else round(since, 1),
                })
        live = sum(1 for row in rows if row['state'] == 'live')
        # Worst first, and in the same order the panel ranks them: a motor
        # saying nothing at all is a worse thing to be told than one that
        # is talking and not holding, because with the first there is no
        # way to tell which of the two it is doing.
        for state in ('silent', 'fault', 'quiet', 'adrift', 'off', 'unknown'):
            if any(row['state'] == state for row in rows):
                worst = state
                break
        else:
            worst = 'live'
        # Which arms the controller manager says are powered. Reported as
        # it comes: this maps component names to states and does not try to
        # decide what a name means.
        engaged = [{'name': name, 'state': label}
                   for name, label in sorted(hardware.items())
                   if 'openarm' in name or 'hardware' in name]
        return {
            'joints': rows,
            'live': live,
            'total': len(rows),
            'worst': worst,
            'topic_age': None if topic_age is None else round(topic_age, 2),
            'components': engaged,
            'quiet_after': MOTOR_QUIET_SECONDS,
            'hold_error': MOTOR_HOLD_ERROR,
            'tracked': bool(tracking),
            # Whether the motors are reporting their own status at all --
            # the difference between knowing and inferring.
            'reported': bool(reported),
        }

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
        payload['motors'] = self.motor_report()
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
        # Outside the lock: motor_report takes it itself, and taking it
        # twice from one thread is a deadlock waiting for a slow browser.
        motors = self.motor_report()
        with self.lock:
            return {
                'motors': motors,
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
    """The page, and never a cached copy of an older one.

    aiohttp sets Last-Modified and nothing else, so a browser applies its
    own heuristic freshness -- typically a tenth of the file's age -- and
    keeps serving what it already has without asking. That is how a panel
    comes up missing a button that is sitting in the file on disk. The
    page is a few tens of kilobytes on a local network: revalidating every
    time costs nothing worth counting.
    """
    if not os.path.exists(PAGE):
        return web.Response(status=500, text=f'{PAGE} is missing')
    return web.FileResponse(PAGE, headers={'Cache-Control': 'no-cache'})


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
    """What the detector will actually be asked, for the field to show.

    Including the recipe, when there is one. The cycle asks for the part
    worth gripping before it asks for the object -- "edge of the tape"
    rather than "tape" -- and a panel showing only half of that is a panel
    that disagrees with the log.
    """
    text = request.query.get('text', '')
    prompt = to_detection_prompt(text)
    answer = {'prompt': prompt}
    entry = grasp_recipes.recipe_for(prompt)
    # Only when it is the one that will actually be asked: a recipe whose
    # approach the cycle cannot fly is not applied -- see part_prompt.
    if entry is not None and entry.get('approach') in (None, 'top'):
        answer['part'] = grasp_recipes.part_prompt(prompt)
        answer['why'] = entry.get('why')
    return web.json_response(answer)


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
