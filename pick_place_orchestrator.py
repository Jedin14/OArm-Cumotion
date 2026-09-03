#!/usr/bin/env python3
"""VLM-guided pick and place for the 7DOF-OArm, with VLM-checked retries.

    READY -> LOCATE -> PRE_PICK -> PREGRASP -> OPEN -> DESCEND -> CLOSE
       ^                                                             |
       |                                                             v
       +----------------------- failed ----------------- VERIFY_GRASP -> LIFT
                                                                        |
                    READY -> DROP -> RELEASE -> VERIFY_PLACE <----------+

Three named postures, all joint-space goals:

  READY            ready_joint_positions, from the parameter. The observation
                   pose: the arm starts here and the object is located from
                   here, so it has to leave the camera a clear view.
  PRE_PICK         pre_pick_state, from the states file. Staging pose entered
                   after the object is located, so the approach to the object
                   starts from a known posture.
  DROP             drop_state, from the states file. Where the object is
                   released.

PRE_PICK and DROP are recorded with record_states.py rather than typed in; the
orchestrator re-reads that file at the start of every cycle, so re-recording a
pose takes effect without restarting anything. place_mode selects what DROP
means: "state" (the recorded drop_state), "ready" (the observation pose), or
"position" (a place_position in world coordinates).

Consumes /vlm/detections from VLM/vlm_detector_node.py (world-frame 3D points)
and drives the arm through MoveIt's /move_action with pipeline_id "cumotion".

Four things about this workspace decide how it talks to the robot:

* Arm motion goes through /move_action, not moveit_py -- there is no moveit_py
  in Humble. pipeline_id "cumotion" is the default pipeline registered by
  openarm_bimanual_moveit_config's demo.launch.py.

* Pose goals target openarm_<arm>_hand_tcp in `world`, because that is exactly
  what cuMotion is configured for: openarm.yml sets base_link "world" and
  ee_link "openarm_left_hand_tcp".

* The gripper is NOT commanded through MoveIt. cuMotion rejects 1-DOF grippers
  (see README), so grasping uses the controller's own GripperCommand action.

* That controller is position_controllers/GripperActionController with
  allow_stalling unset, i.e. false -- so closing onto an object ABORTS the
  action even though the grasp succeeded. The action result is therefore
  ignored, and success is judged from the finger position in /joint_states plus
  a VLM re-detection.

Verification is deliberately geometric rather than a question to the VLM: the
model here is paligemma-3b-pt-224, a pretrained checkpoint where "detect X" is
well-formed but yes/no VQA is not reliable. So instead of asking "is it in the
gripper?", it re-runs the same detection and checks where the object went:
still at the pick point means the grasp failed, near the tool frame means it
came along.

Run it via pick_place.launch.py, or standalone:

    source native/setup.bash
    python3 pick_place_orchestrator.py --ros-args \
        -p prompt:="detect screwdriver" -p arm:=right

Then start a cycle with:

    ros2 service call /pick_place/start std_srvs/srv/Trigger
"""

import math
import json
import os
import threading

import rclpy
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
)
import yaml
from moveit_msgs.srv import ApplyPlanningScene
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from vlm_prompt import to_detection_prompt

MOVEIT_SUCCESS = 1

# Distinguishes "the planner node is not in the graph" from "it did not answer",
# because only the first is worth refusing to start over.
DEAD_PLANNER = 'planner-not-running'

ATTACHED_OBJECT_ID = 'vlm_target'
TABLE_OBJECT_ID = 'work_surface'

WS = os.path.dirname(os.path.realpath(__file__))
DEFAULT_STATES_FILE = os.path.join(WS, 'pick_place_states.yaml')
PRE_PICK_STATE = 'pre_pick_state'
DROP_STATE = 'drop_state'

# Default READY pose, joint1..joint7 in radians: elbow up with the tool clear of
# the work area, so the camera sees the table unobstructed. Captured off the
# right arm while it was held in that pose, which is why the numbers are not
# round -- the RViz Joints tab showed -47, 0, 0, 133, 0, 0, -23 degrees, and
# these are the same pose to better than half a degree. Every value is inside
# the URDF limits (joint4 is 2.324 against an upper limit of 2.443).
# Jog to whatever pose you want and call /pick_place/capture_ready for its list.
READY_JOINT_POSITIONS = [
    -0.828374,
    0.000191,
    -0.000191,
    2.324140,
    -0.000191,
    -0.000191,
    -0.391966,
]

# Retry ladder. Each entry is a whole fresh attempt: re-detect, then apply these
# modifiers. Escalating beats repeating -- a failed grasp usually has a single
# cause, and the two most common ones here are minAreaRect's 90-degree axis
# ambiguity and depth read off the object's top surface sitting too high.
STRATEGIES = [
    {'name': 'nominal', 'yaw_offset': 0.0, 'z_offset': 0.0, 'refresh_octomap': False},
    {'name': 'redetect', 'yaw_offset': 0.0, 'z_offset': 0.0, 'refresh_octomap': False},
    {'name': 'yaw+90', 'yaw_offset': math.pi / 2, 'z_offset': 0.0, 'refresh_octomap': False},
    {'name': 'lower-8mm', 'yaw_offset': 0.0, 'z_offset': -0.008, 'refresh_octomap': False},
    {'name': 'yaw+90-lower', 'yaw_offset': math.pi / 2, 'z_offset': -0.008, 'refresh_octomap': False},
    {'name': 'ready-refresh', 'yaw_offset': 0.0, 'z_offset': 0.0, 'refresh_octomap': True},
]


def quat_mul(a, b):
    """(x, y, z, w) * (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def top_down_quat(yaw):
    """Tool orientation with hand_tcp's +Z pointing straight down.

    hand_tcp sits 0.08 m along +Z of openarm_<arm>_hand (see openarm.urdf), so
    +Z is the approach axis and the fingers close along +/-Y (finger_joint1's
    axis is 0 -1 0 in the hand frame).

    R = Rz(yaw) * Ry(pi) puts the tool Z on world -Z and the finger closing
    direction at yaw + 90deg. Feeding the detected object axis in as `yaw`
    therefore closes the fingers across the object, which is what we want.
    """
    q_pitch = (0.0, 1.0, 0.0, 0.0)                    # pi about Y
    q_yaw = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    return quat_mul(q_yaw, q_pitch)


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class PickPlaceOrchestrator(Node):

    def __init__(self):
        super().__init__('pick_place_orchestrator')
        self.cb = ReentrantCallbackGroup()

        self.declare_parameter('arm', 'right')
        self.declare_parameter('prompt', 'detect screwdriver')
        self.declare_parameter('pipeline_id', 'cumotion')
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('velocity_scaling', 0.15)
        self.declare_parameter('acceleration_scaling', 0.15)
        self.declare_parameter('motion_timeout', 60.0)

        self.declare_parameter('ready_joint_positions', READY_JOINT_POSITIONS)
        self.declare_parameter('approach_height', 0.05)
        self.declare_parameter('grasp_z_offset', -0.005)
        self.declare_parameter('min_grasp_z', 0.01)
        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('orientation_tolerance', 0.05)

        self.declare_parameter('states_file', DEFAULT_STATES_FILE)
        self.declare_parameter('place_mode', 'state')
        self.declare_parameter('place_position', [0.35, 0.30, 0.25])
        self.declare_parameter('place_yaw', 0.0)
        self.declare_parameter('place_approach_height', 0.15)
        self.declare_parameter('place_radius', 0.15)

        self.declare_parameter('gripper_open', 0.044)      # finger_joint1 upper limit
        self.declare_parameter('gripper_close', 0.0)
        self.declare_parameter('gripper_max_effort', 20.0)
        self.declare_parameter('gripper_settle_time', 1.0)
        self.declare_parameter('grasp_finger_min', 0.003)
        self.declare_parameter('grasp_finger_max', 0.040)

        self.declare_parameter('detect_timeout', 15.0)
        self.declare_parameter('object_moved_eps', 0.05)
        self.declare_parameter('object_hold_radius', 0.15)
        self.declare_parameter('attached_object_size', [0.05, 0.05, 0.05])

        self.declare_parameter('use_table_collision', False)
        self.declare_parameter('table_z', 0.0)
        self.declare_parameter('table_size', [1.2, 1.2, 0.02])
        self.declare_parameter('refresh_octomap_frames', 3)
        self.declare_parameter('auto_start', False)

        p = self.get_parameter
        self.arm = p('arm').value
        if self.arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'")
        self.group = f'{self.arm}_arm'
        self.tcp_frame = f'openarm_{self.arm}_hand_tcp'
        self.hand_link = f'openarm_{self.arm}_hand'
        self.finger_joint = f'openarm_{self.arm}_finger_joint1'
        self.arm_joints = [f'openarm_{self.arm}_joint{i}' for i in range(1, 8)]
        self.touch_links = [
            self.hand_link,
            f'openarm_{self.arm}_left_finger',
            f'openarm_{self.arm}_right_finger',
        ]
        self.base_frame = 'world'

        self.place_mode = p('place_mode').value
        if self.place_mode not in ('state', 'ready', 'position'):
            raise ValueError("place_mode must be 'state', 'ready' or 'position'")
        self.states_file = p('states_file').value

        self.prompt = p('prompt').value
        if not self.prompt.lower().startswith('detect'):
            self.prompt = f'detect {self.prompt}'

        self._lock = threading.Lock()
        self._latest_detections = None
        self._finger_position = None
        self._failures = []
        self._arm_positions = {}
        self._busy = False
        self._abort = threading.Event()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.move_client = ActionClient(self, MoveGroup, '/move_action',
                                        callback_group=self.cb)
        self.gripper_client = ActionClient(
            self, GripperCommand, f'/{self.arm}_gripper_controller/gripper_cmd',
            callback_group=self.cb)
        self.scene_client = self.create_client(ApplyPlanningScene,
                                               '/apply_planning_scene',
                                               callback_group=self.cb)
        self.octomap_client = self.create_client(Trigger, '/octomap_gater/refresh',
                                                 callback_group=self.cb)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.prompt_pub = self.create_publisher(String, '/vlm/prompt', latched)
        self.state_pub = self.create_publisher(String, '/pick_place/state', latched)

        self.create_subscription(String, '/pick_place/prompt', self._on_prompt, 10,
                                 callback_group=self.cb)
        self.create_subscription(String, '/vlm/detections', self._on_detections, 10,
                                 callback_group=self.cb)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10,
                                 callback_group=self.cb)

        self.create_service(Trigger, '/pick_place/start', self._srv_start,
                            callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/abort', self._srv_abort,
                            callback_group=self.cb)
        self.create_service(Trigger, '/pick_place/capture_ready',
                            self._srv_capture_ready, callback_group=self.cb)

        self._set_state('IDLE')
        self.get_logger().info(
            f'arm={self.arm} group={self.group} tcp={self.tcp_frame} '
            f'pipeline={p("pipeline_id").value}')
        ready = [round(v, 4) for v in p('ready_joint_positions').value]
        self.get_logger().info(f'ready pose (rad) {ready}')
        if self.place_mode == 'state':
            self.get_logger().info(f'states file {self.states_file}')
            states = self.load_states()
            for name in (PRE_PICK_STATE, DROP_STATE):
                entry = states.get(name)
                if entry is None:
                    self.get_logger().warn(
                        f'{name} is not recorded yet -- run '
                        f'"python3 record_states.py {name}" before starting a '
                        'cycle')
                else:
                    self.get_logger().info(
                        f'{name} tool at {entry.get("tcp_xyz")}')
        elif self.place_mode == 'ready':
            self.get_logger().warn(
                'place_mode is "ready": the object is released at the ready pose, '
                'so it drops from whatever height that pose holds the tool at. '
                'Check what is underneath before the first run.')
        else:
            place = p('place_position').value
            self.get_logger().warn(
                f'place_position is {list(place)} -- verify this is reachable for '
                f'the {self.arm} arm in RViz before running on hardware')

        if p('auto_start').value:
            self.create_timer(3.0, self._auto_start_once, callback_group=self.cb)

    # -- plumbing ------------------------------------------------------------

    def _note_failure(self, stage, message):
        self._failures.append((stage, message))

    def _failure_summary(self):
        """Which stage failed most often -- not merely which failed last.

        Attempts can fail at different stages: a marginal pre_pick plan on one,
        an out-of-reach object on the next. Reporting only the last one is
        actively misleading -- it once pointed at pre_pick_state, which failed
        twice, while the pre-grasp failed three times because the object was
        90 mm beyond the arm. So report the dominant stage, and count the rest.
        """
        if not self._failures:
            return 'no attempt ran'
        counts = {}
        for stage, message in self._failures:
            entry = counts.setdefault(stage, [0, message])
            entry[0] += 1
        stage, (count, message) = max(counts.items(), key=lambda item: item[1][0])
        summary = f'{count} of {len(self._failures)} attempts failed at {stage}'
        others = ', '.join(f'{s} x{c}' for s, (c, _m) in sorted(counts.items())
                           if s != stage)
        if others:
            summary += f' (also {others})'
        return f'{summary}. {message}'

    def _set_state(self, state, detail=''):
        text = state if not detail else f'{state}: {detail}'
        self.state_pub.publish(String(data=text))
        self.get_logger().info(f'[{state}] {detail}'.rstrip())

    def _await(self, future, timeout):
        """Block a worker thread on an rclpy future.

        spin_until_future_complete cannot be used here: the executor is already
        spinning on the main thread. add_done_callback plus an Event is the
        thread-safe equivalent.
        """
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return None
        return future.result()

    def _on_prompt(self, msg):
        """Retarget the next cycle, so the UI can change what to pick.

        Refused while a cycle is running: self.prompt is what detect() matches
        replies against, and swapping it mid-cycle would leave the retry ladder
        hunting for a different object than the one it started on.
        """
        # Normalised the same way the panel does, so that publishing
        # "pick up the wrench" straight onto this topic behaves identically to
        # typing it into the panel.
        wanted = to_detection_prompt(msg.data)
        if not wanted:
            self.get_logger().warn(f'ignoring prompt "{msg.data}": no object in it')
            return
        with self._lock:
            if self._busy:
                self.get_logger().warn(
                    f'ignoring prompt "{wanted}": a cycle is already running')
                return
            self.prompt = wanted
        self.get_logger().info(f'prompt is now "{wanted}"')
        self.prompt_pub.publish(String(data=wanted))

    def _on_detections(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('malformed JSON on /vlm/detections')
            return
        with self._lock:
            self._latest_detections = payload

    def _on_joint_states(self, msg):
        with self._lock:
            if self.finger_joint in msg.name:
                self._finger_position = msg.position[msg.name.index(self.finger_joint)]
            for joint in self.arm_joints:
                if joint in msg.name:
                    self._arm_positions[joint] = msg.position[msg.name.index(joint)]

    def _auto_start_once(self):
        for timer in list(self.timers):
            timer.cancel()
        self._start_cycle()

    def _srv_start(self, _request, response):
        response.success, response.message = self._start_cycle()
        return response

    def _srv_capture_ready(self, _request, response):
        """Report the arm's current joint values as a ready_joint_positions list.

        Jog the arm to the pose you want in RViz, call this, and paste the list
        back as -p ready_joint_positions:="[...]". Beats reading angles off the
        Joints tab, which only shows them rounded to the degree.
        """
        with self._lock:
            missing = [j for j in self.arm_joints if j not in self._arm_positions]
            values = [self._arm_positions.get(j) for j in self.arm_joints]
        if missing:
            response.success = False
            response.message = f'no /joint_states yet for {missing}'
            return response
        formatted = '[' + ', '.join(f'{v:.6f}' for v in values) + ']'
        self.get_logger().info(f'ready_joint_positions:="{formatted}"')
        response.success = True
        response.message = formatted
        return response

    def _srv_abort(self, _request, response):
        self._abort.set()
        response.success = True
        response.message = 'abort requested'
        return response

    def _start_cycle(self):
        with self._lock:
            if self._busy:
                return False, 'a cycle is already running'
            self._busy = True
        self._abort.clear()
        threading.Thread(target=self._run_cycle, daemon=True).start()
        return True, 'cycle started'

    # -- motion primitives ---------------------------------------------------

    def _base_request(self):
        req = MoveGroup.Goal().request
        req.group_name = self.group
        req.pipeline_id = self.get_parameter('pipeline_id').value
        req.num_planning_attempts = 1
        req.allowed_planning_time = self.get_parameter('planning_time').value
        req.max_velocity_scaling_factor = self.get_parameter('velocity_scaling').value
        req.max_acceleration_scaling_factor = \
            self.get_parameter('acceleration_scaling').value
        req.start_state.is_diff = True
        req.workspace_parameters.header.frame_id = self.base_frame
        for corner, sign in ((req.workspace_parameters.min_corner, -1.0),
                             (req.workspace_parameters.max_corner, 1.0)):
            corner.x = corner.y = corner.z = sign * 1.5
        return req

    def _send_move_goal(self, request, label):
        if not self.move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('/move_action unavailable -- is move_group running?')
            return False

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        handle = self._await(self.move_client.send_goal_async(goal), 15.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: goal rejected by move_group')
            return False

        result = self._await(handle.get_result_async(),
                             self.get_parameter('motion_timeout').value)
        if result is None:
            self.get_logger().error(f'{label}: timed out waiting for execution')
            return False
        code = result.result.error_code.val
        if code != MOVEIT_SUCCESS:
            self.get_logger().error(f'{label}: MoveIt error code {code}')
            return False
        return True

    def move_to_pose(self, position, quat, label):
        req = self._base_request()
        pos_tol = self.get_parameter('position_tolerance').value
        ori_tol = self.get_parameter('orientation_tolerance').value

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.tcp_frame
        pc.weight = 1.0
        region = SolidPrimitive()
        region.type = SolidPrimitive.BOX
        region.dimensions = [2 * pos_tol] * 3
        pc.constraint_region.primitives.append(region)
        target = PoseStamped().pose
        target.position.x, target.position.y, target.position.z = position
        target.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(target)

        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.tcp_frame
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = quat
        oc.absolute_x_axis_tolerance = ori_tol
        oc.absolute_y_axis_tolerance = ori_tol
        oc.absolute_z_axis_tolerance = ori_tol
        oc.weight = 1.0

        req.goal_constraints = [Constraints(position_constraints=[pc],
                                            orientation_constraints=[oc])]
        self.get_logger().info(
            f'{label}: xyz=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})')
        return self._send_move_goal(req, label)

    def _move_to_joints(self, positions, label):
        """Joint-space goal, used for every named posture.

        A joint goal is the one kind cuMotion takes for either arm regardless of
        its ee_link: it is turned into a pose by running FK on the merged goal
        state, so no link name is compared. It is also repeatable -- the same
        seven numbers give the same posture every time, which a pose goal on a
        redundant 7-DOF arm does not.
        """
        if len(positions) != len(self.arm_joints):
            self.get_logger().error(
                f'{label} needs {len(self.arm_joints)} joint values, got '
                f'{len(positions)}')
            return False
        req = self._base_request()
        constraints = Constraints()
        for name, value in zip(self.arm_joints, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints = [constraints]
        return self._send_move_goal(req, label)

    def move_to_ready(self):
        positions = list(self.get_parameter('ready_joint_positions').value)
        return self._move_to_joints(positions, 'READY')

    def move_to_state(self, name, states):
        """Replay a pose recorded by record_states.py."""
        entry = states.get(name)
        if not entry or not entry.get('joints'):
            self.get_logger().error(
                f'{name} is not in {self.states_file}; record it with '
                f'"python3 record_states.py {name}"')
            return False
        return self._move_to_joints(list(entry['joints']), name.upper())

    def command_gripper(self, position, label):
        """Drive the gripper and wait for it to settle.

        The result status is intentionally discarded: this controller has
        allow_stalling false, so a successful grasp -- fingers stopped by the
        object -- comes back ABORTED. finger_position() is the real signal.
        """
        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f'/{self.arm}_gripper_controller/gripper_cmd unavailable')
            return False
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = self.get_parameter('gripper_max_effort').value
        handle = self._await(self.gripper_client.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{label}: gripper goal rejected')
            return False
        self._await(handle.get_result_async(), 10.0)
        self._abort.wait(self.get_parameter('gripper_settle_time').value)
        self.get_logger().info(f'{label}: finger at {self.finger_position()}')
        return True

    def finger_position(self):
        with self._lock:
            return self._finger_position

    # -- planning scene ------------------------------------------------------

    def _apply_scene(self, scene, label):
        if not self.scene_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'{label}: /apply_planning_scene unavailable')
            return False
        request = ApplyPlanningScene.Request()
        request.scene = scene
        result = self._await(self.scene_client.call_async(request), 5.0)
        if result is None or not result.success:
            self.get_logger().warn(f'{label}: planning scene update rejected')
            return False
        return True

    def attach_object(self):
        """Tell cuMotion the gripper is now holding something.

        Without this the planner drags an invisible object through the octomap
        on the way to the box.
        """
        size = list(self.get_parameter('attached_object_size').value)
        aco = AttachedCollisionObject()
        aco.link_name = self.hand_link
        aco.touch_links = self.touch_links
        aco.object.id = ATTACHED_OBJECT_ID
        aco.object.header.frame_id = self.tcp_frame
        aco.object.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = size
        aco.object.primitives.append(primitive)
        pose = PoseStamped().pose
        pose.orientation.w = 1.0
        aco.object.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        return self._apply_scene(scene, 'ATTACH')

    def detach_object(self):
        aco = AttachedCollisionObject()
        aco.link_name = self.hand_link
        aco.object.id = ATTACHED_OBJECT_ID
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        return self._apply_scene(scene, 'DETACH')

    def add_table(self):
        """Optional explicit collision box for the work surface.

        Worth turning on with octomap:=static, where the map can legitimately be
        empty and nothing else stops the tool descending through the table.
        """
        if not self.get_parameter('use_table_collision').value:
            self.get_logger().info(
                'use_table_collision is false: relying on the octomap alone for '
                'the work surface')
            return True
        table_z = self.get_parameter('table_z').value
        size = list(self.get_parameter('table_size').value)
        obj = CollisionObject()
        obj.id = TABLE_OBJECT_ID
        obj.header.frame_id = self.base_frame
        obj.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = size
        obj.primitives.append(primitive)
        pose = PoseStamped().pose
        pose.position.z = table_z - size[2] / 2.0     # top face at table_z
        pose.orientation.w = 1.0
        obj.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        return self._apply_scene(scene, 'TABLE')

    def refresh_octomap(self):
        if not self.octomap_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(
                '/octomap_gater/refresh unavailable (octomap:=live, or the '
                'gater is not running) -- skipping refresh')
            return False
        result = self._await(self.octomap_client.call_async(Trigger.Request()), 5.0)
        return bool(result and result.success)

    def _planner_parameter(self, name):
        """One parameter off the cuMotion node.

        Queried one name per call on purpose: if any name in a GetParameters
        request is declared-but-unset, the whole reply comes back with values=[]
        rather than a PARAMETER_NOT_SET entry per name -- measured against the
        running planner. Batching tool_frame (usually unset) with robot (always
        set) therefore returns nothing for either.

        Returns DEAD_PLANNER if the node is not there at all, None if it is
        there but did not answer, '' if the parameter is unset, otherwise its
        value.
        """
        client = self.create_client(GetParameters,
                                   '/cumotion_planner/get_parameters',
                                   callback_group=self.cb)
        try:
            if not client.wait_for_service(timeout_sec=5.0):
                return DEAD_PLANNER
            result = self._await(
                client.call_async(GetParameters.Request(names=[name])), 5.0)
        finally:
            self.destroy_client(client)
        if result is None:
            return None
        if not result.values:
            return ''
        return result.values[0].string_value

    def planner_ee_link(self):
        """Which link cuMotion will actually accept Cartesian goals for.

        tool_frame wins if it is set; otherwise it is the ee_link in the robot
        config the node was launched with. Returns None if neither could be
        determined.
        """
        tool_frame = self._planner_parameter('tool_frame')
        if tool_frame is DEAD_PLANNER or tool_frame is None:
            return tool_frame
        if tool_frame:
            return tool_frame
        robot_file = self._planner_parameter('robot')
        if robot_file is DEAD_PLANNER or not robot_file:
            return None
        try:
            import yaml
            with open(robot_file) as handle:
                config = yaml.safe_load(handle)
            return config['robot_cfg']['kinematics']['ee_link']
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f'could not read ee_link from {robot_file}: {exc}')
            return None

    def load_states(self):
        """Read the recorded poses. Re-read every cycle, on purpose.

        That way re-recording a pose with record_states.py takes effect on the
        next pick without restarting the stack -- which matters because the
        stack takes a PaliGemma load to come back up.
        """
        path = self.states_file
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().error(f'could not read {path}: {exc}')
            return {}
        recorded_arm = data.get('arm')
        if recorded_arm and recorded_arm != self.arm:
            # Joint values are per-arm: the arms are mirrored, so replaying a
            # left-arm recording on the right arm is a different posture, not a
            # mirrored one.
            self.get_logger().error(
                f'{path} was recorded for the {recorded_arm} arm but this is the '
                f'{self.arm} arm; re-record it with '
                f'"python3 record_states.py --arm {self.arm}"')
            return {}
        return data.get('states') or {}

    def check_states(self, states):
        """Every pose the cycle will need, before it moves anything."""
        needed = [PRE_PICK_STATE]
        if self.place_mode == 'state':
            needed.append(DROP_STATE)
        missing = [n for n in needed if not (states.get(n) or {}).get('joints')]
        if missing:
            self.get_logger().error(
                f'missing {", ".join(missing)} in {self.states_file}. Record with: '
                f'python3 record_states.py {" ".join(missing)}')
            return False
        return True

    def check_planner_tool_frame(self):
        """cuMotion has exactly one ee_link and rejects pose goals aimed elsewhere.

        openarm.yml sets ee_link to openarm_left_hand_tcp, so with the default
        config every right-arm pose goal comes back INVALID_LINK_NAME -- but only
        at PREGRASP, after the gripper has already opened. Catching it up front
        turns that into one line of text instead of a half-executed cycle.

        The node reads tool_frame once at construction, so this cannot be fixed
        at runtime: it needs a relaunch of the robot.
        """
        if self.get_parameter('pipeline_id').value != 'cumotion':
            return True
        ee_link = self.planner_ee_link()
        if ee_link is DEAD_PLANNER:
            # Worth a hard stop rather than a warning: with no planner every
            # pose goal fails, so the cycle would walk the whole retry ladder
            # and report "every pick strategy was exhausted" -- which reads as
            # a grasping problem when the planner simply is not running. It has
            # crashed here before (SIGFPE, exit code -8), so this is a state
            # the stack really does reach.
            self.get_logger().error(
                '/cumotion_planner is not running -- nothing can be planned. '
                'Check the launch output for "cumotion_goal_set_planner_node ... '
                'process has died", and restart the robot.')
            return False
        if ee_link is None:
            self.get_logger().warn(
                'could not determine cuMotion\'s ee_link; continuing without the '
                'check. A mismatch shows up as INVALID_LINK_NAME at PREGRASP.')
            return True
        if ee_link != self.tcp_frame:
            self.get_logger().error(
                f'cuMotion takes Cartesian goals for {ee_link}, but this cycle '
                f'needs {self.tcp_frame}. Relaunch the robot with '
                f'tool_frame:={self.tcp_frame} '
                '(native/run_launch_everything.sh '
                f'tool_frame:={self.tcp_frame}), or run this with arm:='
                f'{"left" if self.arm == "right" else "right"}.')
            return False
        self.get_logger().info(f'cuMotion plans Cartesian goals for {ee_link}')
        return True

    # -- perception ----------------------------------------------------------

    def detect(self, min_count=1):
        """Wait for a detection message newer than now and matching our prompt."""
        self.prompt_pub.publish(String(data=self.prompt))
        started = self.get_clock().now().nanoseconds * 1e-9
        timeout = self.get_parameter('detect_timeout').value
        deadline = started + timeout
        while not self._abort.is_set():
            with self._lock:
                payload = self._latest_detections
            fresh = (payload is not None
                     and payload.get('stamp', 0.0) > started
                     and payload.get('prompt') == self.prompt)
            if fresh and len(payload.get('detections', [])) >= min_count:
                return payload
            if self.get_clock().now().nanoseconds * 1e-9 > deadline:
                if fresh:
                    return payload          # fresh but empty: nothing was seen
                self.get_logger().error(
                    'no fresh /vlm/detections -- is VLM/run_vlm_detector.sh running?')
                return None
            self._abort.wait(0.1)
        return None

    def tcp_position(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as exc:
            self.get_logger().warn(f'no {self.tcp_frame} transform: {exc}')
            return None
        t = tf.transform.translation
        return (t.x, t.y, t.z)

    # -- grasp geometry ------------------------------------------------------

    def grasp_from_detection(self, detection, strategy):
        point = detection['point']
        axis_yaw = detection.get('axis_yaw')
        if axis_yaw is None:
            axis_yaw = 0.0
            self.get_logger().warn('no object axis from the VLM; assuming yaw 0')
        yaw = axis_yaw + strategy['yaw_offset']

        grasp_z = point[2] + self.get_parameter('grasp_z_offset').value \
            + strategy['z_offset']
        min_z = self.get_parameter('min_grasp_z').value
        if grasp_z < min_z:
            self.get_logger().warn(
                f'grasp z {grasp_z:.3f} below min_grasp_z {min_z:.3f}; clamping')
            grasp_z = min_z

        grasp = (point[0], point[1], grasp_z)
        pregrasp = (point[0], point[1],
                    grasp_z + self.get_parameter('approach_height').value)
        return grasp, pregrasp, top_down_quat(yaw), point

    # -- verification --------------------------------------------------------

    def verify_grasp(self, pick_point):
        """Two independent checks, because either alone has a blind spot.

        The finger check misses the case where a fingertip stalls on the
        object's edge without capturing it; the re-detection check misses when
        the object is occluded by the gripper. Requiring the finger check and
        no-object-left-behind together is the combination that actually
        distinguishes "held" from "closed on air".
        """
        finger = self.finger_position()
        lo = self.get_parameter('grasp_finger_min').value
        hi = self.get_parameter('grasp_finger_max').value
        if finger is None:
            self.get_logger().warn('no finger feedback on /joint_states')
            finger_ok = False
        else:
            finger_ok = lo < finger < hi
            self.get_logger().info(
                f'finger {finger:.4f} in ({lo:.4f}, {hi:.4f}) -> '
                f'{"holding" if finger_ok else "empty"}')
        if not finger_ok:
            return False

        payload = self.detect(min_count=0)
        if payload is None:
            self.get_logger().warn('no detection for grasp check; trusting fingers')
            return True

        detections = payload.get('detections', [])
        eps = self.get_parameter('object_moved_eps').value
        left_behind = [d for d in detections if dist(d['point'], pick_point) < eps]
        if left_behind:
            self.get_logger().warn(
                f'object still at the pick point ({dist(left_behind[0]["point"], pick_point):.3f} m) '
                '-- grasp failed')
            return False

        tcp = self.tcp_position()
        if tcp is not None and detections:
            radius = self.get_parameter('object_hold_radius').value
            nearest = min(dist(d['point'], tcp) for d in detections)
            if nearest < radius:
                self.get_logger().info(f'object {nearest:.3f} m from the tool: held')
                return True
            self.get_logger().info(
                f'nearest detection {nearest:.3f} m from the tool; the gripper is '
                'probably occluding it -- trusting fingers')
        return True

    def verify_place(self, reference=None):
        """Was the object seen near where it was released?

        `reference` is where the release actually happened -- the tool position
        in place_mode "ready", place_position otherwise. Advisory either way: a
        box occludes what is inside it.
        """
        if reference is None:
            reference = list(self.get_parameter('place_position').value)
        payload = self.detect(min_count=0)
        if payload is None:
            return False
        radius = self.get_parameter('place_radius').value
        for d in payload.get('detections', []):
            if dist(d['point'][:2], reference[:2]) < radius:
                self.get_logger().info('object seen where it was released')
                return True
        self.get_logger().warn(
            'object not seen at the release point -- it may be occluded, or it '
            'rolled after the drop')
        return False

    # -- the cycle -----------------------------------------------------------

    def _run_cycle(self):
        try:
            self._cycle()
        except Exception as exc:
            self.get_logger().error(f'cycle crashed: {exc}')
            self._set_state('FAILED', str(exc))
        finally:
            with self._lock:
                self._busy = False

    def _cycle(self):
        if not self.check_planner_tool_frame():
            self._set_state('FAILED', 'the planner is unusable for this arm')
            return
        self._failures = []
        states = self.load_states()
        if not self.check_states(states):
            self._set_state('FAILED', 'recorded states are missing')
            return
        self.add_table()

        # Start from the ready pose: it is the posture the rest of the cycle
        # assumes, and it clears the arm out of the camera's view of the table
        # before the first detection.
        self._set_state('READY', 'moving to the observation pose')
        if not self.move_to_ready():
            self._set_state('FAILED', 'could not reach the ready pose')
            return

        picked_point = None
        for attempt, strategy in enumerate(STRATEGIES):
            if self._abort.is_set():
                self._set_state('ABORTED')
                return
            self._set_state('ATTEMPT',
                            f'{attempt + 1}/{len(STRATEGIES)} ({strategy["name"]})')
            if attempt > 0:
                if not self.move_to_ready():
                    continue
                if strategy['refresh_octomap']:
                    self.refresh_octomap()
            picked_point = self._attempt_pick(strategy, states)
            if picked_point is not None:
                break
        else:
            self._set_state(
                'FAILED',
                f'every pick strategy was exhausted -- {self._failure_summary()}')
            self.move_to_ready()
            return

        self._set_state('READY', 'carrying the object')
        if not self.move_to_ready():
            self._set_state('FAILED', 'could not return to ready while holding')
            return

        if not self._place(states):
            self._set_state('FAILED', 'place failed')
            return

        # place_mode "ready" releases at the observation pose, so the arm is
        # already where this would send it.
        if self.place_mode != 'ready':
            self._set_state('READY', 'cycle complete')
            self.move_to_ready()
        self._set_state('DONE')

    def _attempt_pick(self, strategy, states):
        self._set_state('LOCATE', self.prompt)
        payload = self.detect(min_count=1)
        if payload is None or not payload.get('detections'):
            self._set_state('LOCATE', 'nothing detected')
            self._note_failure('LOCATE', (
                f'nothing matched "{self.prompt}" -- check /vlm/debug_image, and '
                'that the detector has finished loading'))
            return None

        detection = payload['detections'][0]
        grasp, pregrasp, quat, point = self.grasp_from_detection(detection, strategy)
        self.get_logger().info(
            f'target at {point} axis_yaw={detection.get("axis_yaw")} '
            f'depth={detection["depth_m"]} m from {detection["depth_px"]} px')

        # Staged through pre_pick_state rather than going straight at the
        # object: the observation pose is chosen to keep the arm out of the
        # camera's view, which is not necessarily a good posture to start a
        # descent from.
        self._set_state('PRE_PICK')
        if not self.move_to_state(PRE_PICK_STATE, states):
            self._note_failure('PRE_PICK', (
                f'could not plan to {PRE_PICK_STATE}; re-record it somewhere the '
                'arm can reach from the ready pose'))
            return None

        self._set_state('PREGRASP')
        if not self.move_to_pose(pregrasp, quat, 'PREGRASP'):
            # By far the most common real cause, and the one that used to hide
            # behind "every pick strategy was exhausted": the object is simply
            # outside the arm's envelope. Say where it was and what the limit is
            # instead of making the reader go and measure it.
            self._note_failure('PREGRASP', (
                f'could not plan to the pre-grasp above '
                f'({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) -- most often '
                f'the object is out of the {self.arm} arm\'s reach. Verify with: '
                f'python3 native/tests/check_reachability.py --arm {self.arm} '
                f'--point {pregrasp[0]:.3f} {pregrasp[1]:.3f} {pregrasp[2]:.3f}'))
            return None

        # Opened here rather than before the approach: the fingers are already
        # over the object, so an open gripper cannot catch anything on the way
        # in, and a gripper that fails to open costs no motion.
        self._set_state('OPEN_GRIPPER')
        if not self.command_gripper(self.get_parameter('gripper_open').value, 'OPEN'):
            return None

        self._set_state('DESCEND')
        if not self.move_to_pose(grasp, quat, 'DESCEND'):
            self._note_failure('DESCEND', (
                f'reached the pre-grasp but could not descend to '
                f'z={grasp[2]:.3f} -- the octomap may have the object itself in '
                'it, or the grasp is below the work surface'))
            return None

        self._set_state('CLOSE_GRIPPER')
        if not self.command_gripper(self.get_parameter('gripper_close').value, 'CLOSE'):
            return None

        self._set_state('LIFT')
        if not self.move_to_pose(pregrasp, quat, 'LIFT'):
            return None

        self._set_state('VERIFY_GRASP')
        if not self.verify_grasp(point):
            self._note_failure('VERIFY_GRASP', (
                'the gripper closed but nothing was held -- check '
                'grasp_finger_min against your object, and the grasp height'))
            self._set_state('VERIFY_GRASP', 'failed, retrying')
            self.command_gripper(self.get_parameter('gripper_open').value, 'OPEN')
            return None

        self.attach_object()
        self._set_state('VERIFY_GRASP', 'confirmed')
        return point

    def _place(self, states):
        if self.place_mode == 'state':
            return self._place_at_state(states)
        if self.place_mode == 'ready':
            return self._place_at_ready()
        place = list(self.get_parameter('place_position').value)
        quat = top_down_quat(self.get_parameter('place_yaw').value)
        above = (place[0], place[1],
                 place[2] + self.get_parameter('place_approach_height').value)

        self._set_state('OVER_BOX')
        if not self.move_to_pose(above, quat, 'OVER_BOX'):
            return False
        if not self.move_to_pose(tuple(place), quat, 'PLACE'):
            return False

        self._set_state('RELEASE')
        if not self.command_gripper(self.get_parameter('gripper_open').value, 'OPEN'):
            return False
        self.detach_object()

        self._set_state('RETREAT')
        if not self.move_to_pose(above, quat, 'RETREAT'):
            return False

        self._set_state('VERIFY_PLACE')
        self.verify_place(place)     # advisory: the box usually occludes the object
        return True

    def _place_at_state(self, states):
        """Carry to the recorded drop_state and release there."""
        self._set_state('DROP', 'moving to the recorded drop pose')
        if not self.move_to_state(DROP_STATE, states):
            return False

        release_point = self.tcp_position()
        self._set_state('RELEASE', 'at drop_state')
        if not self.command_gripper(self.get_parameter('gripper_open').value, 'OPEN'):
            return False
        self.detach_object()

        self._set_state('VERIFY_PLACE')
        if release_point is None:
            self.get_logger().warn(
                f'no {self.tcp_frame} transform at release; skipping the check')
        else:
            self.verify_place(release_point)
        return True

    def _place_at_ready(self):
        """Release from the ready pose, where the cycle has already returned to.

        No approach or retreat move: the pose is the drop point, so the object
        falls the distance between the tool and whatever is below it.
        """
        release_point = self.tcp_position()

        self._set_state('RELEASE', 'at the ready pose')
        if not self.command_gripper(self.get_parameter('gripper_open').value, 'OPEN'):
            return False
        self.detach_object()

        self._set_state('VERIFY_PLACE')
        if release_point is None:
            self.get_logger().warn(
                f'no {self.tcp_frame} transform at release; skipping the check')
        else:
            self.verify_place(release_point)
        return True


def main():
    rclpy.init()
    node = PickPlaceOrchestrator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._abort.set()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
