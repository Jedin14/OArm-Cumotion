#!/usr/bin/env python3
"""Run a whole pick-and-place cycle against a fake robot and check the sequence.

No hardware, no MoveIt, no camera, no PaliGemma. Everything the orchestrator
talks to is stubbed here -- /move_action, the gripper action, /joint_states, TF,
the planning scene, cuMotion's parameters and /vlm/detections -- so the cycle
runs for real and every goal it sends is recorded.

What that buys over the geometry tests: those check the grasp maths in
isolation, this checks the *order*, which is the part that regressed when
pre_pick_state and drop_state were added. It asserts the exact sequence of
goals, that PRE_PICK and DROP replay the recorded joint values, and that
PREGRASP and LIFT sit approach_height above the grasp while DESCEND sits on it.

    source native/setup.bash && python3 native/tests/test_pick_cycle.py
"""

import importlib.util
import json
import os
import sys
import threading
import time

import rclpy
import yaml
from control_msgs.action import GripperCommand
from geometry_msgs.msg import TransformStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import ApplyPlanningScene
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

ARM = 'right'
OBJECT_POINT = [0.35, -0.18, 0.05]
OBJECT_YAW = 0.4
APPROACH_HEIGHT = 0.05
GRASP_Z_OFFSET = -0.005
PRE_PICK_JOINTS = [-0.5, 0.1, -0.2, 1.9, 0.05, -0.3, 0.7]
DROP_JOINTS = [0.9, 0.2, -0.1, 1.5, 0.0, 0.2, -0.4]
READY_JOINTS = [-0.828374, 0.000191, -0.000191, 2.324140,
                -0.000191, -0.000191, -0.391966]

OPEN_FINGER = 0.044
HOLDING_FINGER = 0.02          # inside (grasp_finger_min, grasp_finger_max)

failures = []


def is_numeric(values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in values)


def check(label, got, want, tol=1e-6):
    if isinstance(want, (list, tuple)):
        if want and is_numeric(want):
            ok = (len(got) == len(want)
                  and all(abs(a - b) <= tol for a, b in zip(got, want)))
        else:
            ok = list(got) == list(want)
    elif isinstance(want, float):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print(('pass  ' if ok else 'FAIL  ') + label)
    if not ok:
        print(f'        got  {got}\n        want {want}')
        failures.append(label)


class FakeRobot(Node):
    """Every interface the orchestrator needs, and a log of what it asked for."""

    def __init__(self):
        super().__init__('fake_robot')
        cb = ReentrantCallbackGroup()

        self.goals = []                    # MoveGroup goals, in order
        self.gripper_commands = []
        self.finger = OPEN_FINGER
        self.holding = False
        self.lock = threading.Lock()

        self.tf = TransformBroadcaster(self)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.detections_pub = self.create_publisher(String, '/vlm/detections', 10)
        self.create_subscription(String, '/vlm/prompt', self._on_prompt, 10,
                                 callback_group=cb)
        self.prompt = 'detect screwdriver'

        ActionServer(self, MoveGroup, '/move_action', self._on_move,
                     callback_group=cb)
        ActionServer(self, GripperCommand,
                     f'/{ARM}_gripper_controller/gripper_cmd', self._on_gripper,
                     callback_group=cb)
        self.create_service(ApplyPlanningScene, '/apply_planning_scene',
                            self._on_scene, callback_group=cb)
        self.create_service(GetParameters, '/cumotion_planner/get_parameters',
                            self._on_get_parameters, callback_group=cb)
        self.create_service(Trigger, '/octomap_gater/refresh',
                            self._on_refresh, callback_group=cb)

        self.create_timer(0.05, self._tick, callback_group=cb)

    # -- published state ---------------------------------------------------

    def _tick(self):
        now = self.get_clock().now()
        stamp = now.to_msg()

        js = JointState()
        js.header.stamp = stamp
        js.name = ([f'openarm_{ARM}_joint{i}' for i in range(1, 8)]
                   + [f'openarm_{ARM}_finger_joint1'])
        with self.lock:
            finger = self.finger
        js.position = list(READY_JOINTS) + [finger]
        self.joint_pub.publish(js)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'world'
        t.child_frame_id = f'openarm_{ARM}_hand_tcp'
        t.transform.translation.x = 0.3
        t.transform.translation.y = -0.2
        t.transform.translation.z = 0.3
        t.transform.rotation.w = 1.0
        self.tf.sendTransform(t)

        # Once the gripper has closed on it, the object is no longer lying at
        # the pick point -- which is exactly what verify_grasp checks for.
        with self.lock:
            holding = self.holding
        payload = {
            'stamp': now.nanoseconds * 1e-9,
            'prompt': self.prompt,
            'detections': [] if holding else [{
                'point': list(OBJECT_POINT),
                'axis_yaw': OBJECT_YAW,
                'depth_m': 0.62,
                'depth_px': 812,
                'axis_source': 'depth',
            }],
        }
        self.detections_pub.publish(String(data=json.dumps(payload)))

    def _on_prompt(self, msg):
        self.prompt = msg.data

    # -- served interfaces -------------------------------------------------

    def _on_move(self, goal_handle):
        request = goal_handle.request.request
        constraints = request.goal_constraints[0]
        entry = {'group': request.group_name}
        if constraints.joint_constraints:
            entry['kind'] = 'joint'
            entry['joints'] = [jc.position for jc in constraints.joint_constraints]
            entry['names'] = [jc.joint_name for jc in constraints.joint_constraints]
        else:
            pose = constraints.position_constraints[0]
            point = pose.constraint_region.primitive_poses[0].position
            quat = constraints.orientation_constraints[0].orientation
            entry['kind'] = 'pose'
            entry['link'] = pose.link_name
            entry['xyz'] = [point.x, point.y, point.z]
            entry['quat'] = [quat.x, quat.y, quat.z, quat.w]
        with self.lock:
            self.goals.append(entry)

        goal_handle.succeed()
        result = MoveGroup.Result()
        result.error_code.val = MoveItErrorCodes.SUCCESS
        return result

    def _on_gripper(self, goal_handle):
        position = goal_handle.request.command.position
        with self.lock:
            self.gripper_commands.append(position)
            # Closing onto the object stalls the fingers short of the command,
            # which is what a real grasp looks like on this controller.
            if position < OPEN_FINGER / 2:
                self.finger = HOLDING_FINGER
                self.holding = True
            else:
                self.finger = OPEN_FINGER
                self.holding = False
        goal_handle.succeed()
        result = GripperCommand.Result()
        result.position = position
        result.reached_goal = True
        return result

    def _on_scene(self, _request, response):
        response.success = True
        return response

    def _on_get_parameters(self, request, response):
        for name in request.names:
            value = ParameterValue()
            if name == 'tool_frame':
                value.type = ParameterType.PARAMETER_STRING
                value.string_value = f'openarm_{ARM}_hand_tcp'
            else:
                value.type = ParameterType.PARAMETER_NOT_SET
            response.values.append(value)
        return response

    def _on_refresh(self, _request, response):
        response.success = True
        return response


def write_states(path):
    document = {
        'arm': ARM,
        'states': {
            'pre_pick_state': {'joints': list(PRE_PICK_JOINTS)},
            'drop_state': {'joints': list(DROP_JOINTS)},
        },
    }
    with open(path, 'w') as handle:
        yaml.safe_dump(document, handle, sort_keys=False)


def load_orchestrator():
    # Loading by path does not put the workspace on sys.path, and the
    # orchestrator imports vlm_prompt from beside itself.
    if WS not in sys.path:
        sys.path.insert(0, WS)
    path = os.path.join(WS, 'pick_place_orchestrator.py')
    spec = importlib.util.spec_from_file_location('pick_place_orchestrator', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['pick_place_orchestrator'] = module
    spec.loader.exec_module(module)
    return module


def main():
    states_path = os.path.join(WS, 'native', 'tests', '.test_states.yaml')
    write_states(states_path)

    rclpy.init(args=[
        '--ros-args',
        '-p', f'arm:={ARM}',
        '-p', f'states_file:={states_path}',
        '-p', 'place_mode:=state',
        '-p', f'approach_height:={APPROACH_HEIGHT}',
        '-p', f'grasp_z_offset:={GRASP_Z_OFFSET}',
        '-p', 'gripper_settle_time:=0.05',
        '-p', 'detect_timeout:=8.0',
    ])
    module = load_orchestrator()

    robot = FakeRobot()
    orchestrator = module.PickPlaceOrchestrator()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(robot)
    executor.add_node(orchestrator)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    states = []
    orchestrator.create_subscription(
        String, '/pick_place/state', lambda m: states.append(m.data), 10)

    try:
        time.sleep(2.0)                    # let discovery settle

        orchestrator._on_prompt(String(data='pick up the wrench'))
        check('a conversational prompt on the topic is normalised',
              orchestrator.prompt, 'detect wrench')
        robot.prompt = 'detect wrench'

        ok, message = orchestrator._start_cycle()
        check('cycle starts', ok, True)

        deadline = time.time() + 90
        while time.time() < deadline:
            if states and states[-1].split(':')[0] in ('DONE', 'FAILED', 'ABORTED'):
                break
            time.sleep(0.2)

        steps = [s.split(':')[0].strip() for s in states]
        print('\nstates:', ' -> '.join(steps), '\n')
        check('cycle reached DONE', steps[-1] if steps else None, 'DONE')

        # The sequence the cycle is supposed to walk, in order. Checked as a
        # subsequence so retries or extra detail lines cannot break it.
        wanted = ['READY', 'LOCATE', 'PRE_PICK', 'PREGRASP', 'OPEN_GRIPPER',
                  'DESCEND', 'CLOSE_GRIPPER', 'LIFT', 'VERIFY_GRASP', 'READY',
                  'DROP', 'RELEASE', 'VERIFY_PLACE', 'DONE']
        index, missing = 0, []
        for step in wanted:
            while index < len(steps) and steps[index] != step:
                index += 1
            if index == len(steps):
                missing.append(step)
            index += 1
        check('states appear in the documented order', missing, [])

        with robot.lock:
            goals = list(robot.goals)
            grips = list(robot.gripper_commands)

        kinds = [g['kind'] for g in goals]
        check('eight goals: ready, pre_pick, 3 poses, ready, drop, ready',
              kinds, ['joint', 'joint', 'pose', 'pose', 'pose',
                      'joint', 'joint', 'joint'])

        check('all goals target the right arm group',
              sorted({g['group'] for g in goals}), [f'{ARM}_arm'])
        check('pose goals target the right tool frame',
              sorted({g['link'] for g in goals if g['kind'] == 'pose'}),
              [f'openarm_{ARM}_hand_tcp'])

        check('goal 1 is the ready pose', goals[0]['joints'], READY_JOINTS, 1e-5)
        check('goal 2 replays pre_pick_state', goals[1]['joints'], PRE_PICK_JOINTS)
        check('pre_pick goal names the arm joints', goals[1]['names'],
              [f'openarm_{ARM}_joint{i}' for i in range(1, 8)])

        grasp_z = OBJECT_POINT[2] + GRASP_Z_OFFSET
        above = grasp_z + APPROACH_HEIGHT
        check('PREGRASP is approach_height above the grasp',
              goals[2]['xyz'], [OBJECT_POINT[0], OBJECT_POINT[1], above], 1e-9)
        check('DESCEND is on the grasp', goals[3]['xyz'],
              [OBJECT_POINT[0], OBJECT_POINT[1], grasp_z], 1e-9)
        check('LIFT returns to the pre-grasp height',
              goals[4]['xyz'], [OBJECT_POINT[0], OBJECT_POINT[1], above], 1e-9)
        check('the three object goals share one orientation',
              goals[2]['quat'], goals[3]['quat'], 1e-9)
        check('grasp orientation is the detected yaw',
              list(goals[2]['quat']), list(module.top_down_quat(OBJECT_YAW)), 1e-9)

        check('goal 6 returns to ready carrying', goals[5]['joints'],
              READY_JOINTS, 1e-5)
        check('goal 7 replays drop_state', goals[6]['joints'], DROP_JOINTS)
        check('goal 8 returns to ready', goals[7]['joints'], READY_JOINTS, 1e-5)

        check('gripper opened, closed, then opened to release',
              [round(g, 3) for g in grips], [0.044, 0.0, 0.044])

        # Guards. These are what stand between an unrecorded pose and the arm
        # moving somewhere nobody chose.
        check('an empty states file is refused',
              orchestrator.check_states({}), False)
        check('a states file missing drop_state is refused',
              orchestrator.check_states(
                  {'pre_pick_state': {'joints': PRE_PICK_JOINTS}}), False)
        check('a complete states file passes',
              orchestrator.check_states(
                  {'pre_pick_state': {'joints': PRE_PICK_JOINTS},
                   'drop_state': {'joints': DROP_JOINTS}}), True)

        with open(states_path, 'w') as handle:
            yaml.safe_dump({'arm': 'left', 'states': {
                'pre_pick_state': {'joints': PRE_PICK_JOINTS},
                'drop_state': {'joints': DROP_JOINTS}}}, handle)
        check('a file recorded for the other arm is not replayed',
              orchestrator.load_states(), {})

        # A dead planner must stop the cycle, not warn and walk the whole retry
        # ladder: with nothing planning, "every pick strategy was exhausted"
        # reads as a grasping problem when the planner simply is not running.
        real_lookup = orchestrator._planner_parameter
        orchestrator._planner_parameter = lambda name: module.DEAD_PLANNER
        check('a missing planner node is detected',
              orchestrator.planner_ee_link(), module.DEAD_PLANNER)
        check('and refuses the cycle', orchestrator.check_planner_tool_frame(), False)
        orchestrator._planner_parameter = real_lookup
        check('a live planner still passes',
              orchestrator.check_planner_tool_frame(), True)

        before = len(goals)
        orchestrator._cycle()
        with robot.lock:
            after = len(robot.goals)
        check('a refused cycle sends no goals at all', after, before)
        check('and reports why', states[-1].split(':')[0].strip(), 'FAILED')
    finally:
        orchestrator._abort.set()
        executor.shutdown()
        robot.destroy_node()
        orchestrator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if os.path.exists(states_path):
            os.remove(states_path)

    print()
    if failures:
        print(f'{len(failures)} check(s) failed: {", ".join(failures)}')
        return 1
    print('the whole cycle ran in the documented order')
    return 0


if __name__ == '__main__':
    sys.exit(main())
