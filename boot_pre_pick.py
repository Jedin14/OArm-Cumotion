#!/usr/bin/env python3
"""Walk both arms to their recorded pre_pick_state once the robot is up.

Started by launch_everything.launch.py (boot_pre_pick:=true, the default). It
waits for move_group, cuMotion and /joint_states, then sends each arm to the
pre_pick_state in pick_place_states_<arm>.yaml: slowly, as a cuMotion joint goal,
one arm at a time, and then exits. An arm that is already there, or has no
recording, is left alone.

pick_place_demo.launch.py turns this off, because the orchestrator does its own
boot walk (boot_walk) to the same poses and two nodes driving the arms at once
would fight.

A joint goal per arm rather than one 14-joint goal: the MoveIt groups are
left_arm and right_arm, and cuMotion merges a 7-joint goal into the current
full state, so the other arm holds still while one moves.
"""

import argparse
import os
import sys
import threading
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from sensor_msgs.msg import JointState

WS = os.path.dirname(os.path.abspath(__file__))
STATE = 'pre_pick_state'
# The folded posture both arms can reach from almost anywhere
# (pick_place_orchestrator's home_joint_positions). A direct move that
# cannot be planned -- an arm left hanging by the torso needs a detour
# longer than cuRobo's optimiser will find -- goes via here instead.
HOME = [0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0]


def wait(future, timeout):
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    return future.result() if done.wait(timeout) else None


class BootPrePick(Node):
    def __init__(self, args):
        super().__init__('boot_pre_pick')
        self.args = args
        self.joints = {}
        self.lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self.move_client = ActionClient(self, MoveGroup, '/move_action')

    def _on_joints(self, msg):
        with self.lock:
            self.joints.update(zip(msg.name, msg.position))

    def current(self, names):
        with self.lock:
            if all(n in self.joints for n in names):
                return [self.joints[n] for n in names]
        return None

    def target(self, arm):
        path = os.path.join(WS, f'pick_place_states_{arm}.yaml')
        try:
            with open(path) as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().warn(f'{arm}: could not read {path}: {exc}')
            return None
        if data.get('arm', arm) != arm:
            self.get_logger().warn(f'{arm}: {path} was recorded for the {data["arm"]} arm')
            return None
        joints = ((data.get('states') or {}).get(STATE) or {}).get('joints')
        if not joints or len(joints) != 7:
            self.get_logger().warn(
                f'{arm}: no {STATE} in {path}; left where it is. Record one with '
                f'"python3 record_states.py --arm {arm} {STATE}"')
            return None
        return [float(v) for v in joints]

    def move(self, arm, positions):
        """One cuMotion joint goal. Returns a MoveIt error code (or None)."""
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = f'{arm}_arm'
        req.pipeline_id = 'cumotion'
        req.num_planning_attempts = 1
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = self.args.velocity
        req.max_acceleration_scaling_factor = self.args.velocity
        req.start_state.is_diff = True
        constraints = Constraints()
        for i, value in enumerate(positions, start=1):
            jc = JointConstraint()
            jc.joint_name = f'openarm_{arm}_joint{i}'
            jc.position = value
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        handle = wait(self.move_client.send_goal_async(goal), 15.0)
        if handle is None or not handle.accepted:
            return None
        result = wait(handle.get_result_async(), 90.0)
        if result is None:
            handle.cancel_goal_async()
            return None
        return result.result.error_code.val

    def run(self):
        log = self.get_logger()
        deadline = time.monotonic() + self.args.timeout
        log.info('waiting for move_group...')
        while not self.move_client.wait_for_server(timeout_sec=2.0):
            if time.monotonic() > deadline:
                log.error('move_group never came up; arms not moved')
                return False
        ok = True
        for arm in self.args.order:
            names = [f'openarm_{arm}_joint{i}' for i in range(1, 8)]
            goal = self.target(arm)
            if goal is None:
                continue
            while self.current(names) is None:
                if time.monotonic() > deadline:
                    log.error(f'{arm}: no /joint_states; not moved')
                    return False
                time.sleep(0.5)
            if max(abs(a - b) for a, b in zip(self.current(names), goal)) < 0.02:
                log.info(f'{arm}: already at {STATE}')
                continue
            # cuMotion loads its CUDA kernels after move_group is up and
            # answers nothing meanwhile; keep asking until it plans or the
            # deadline passes. A failed plan never moves the arm.
            tries = 0
            while True:
                log.info(f'{arm}: moving to {STATE} (velocity {self.args.velocity:.2f})')
                code = self.move(arm, goal)
                if code == MoveItErrorCodes.SUCCESS:
                    log.info(f'{arm}: at {STATE}')
                    break
                tries += 1
                if code in (MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.FAILURE) and tries >= 2:
                    log.info(f'{arm}: no direct path to {STATE}; going via home')
                    if self.move(arm, HOME) == MoveItErrorCodes.SUCCESS:
                        continue
                if time.monotonic() > deadline:
                    log.error(f'{arm}: could not reach {STATE} (last code {code})')
                    ok = False
                    break
                log.warn(f'{arm}: not yet (code {code}); cuMotion may still be '
                         f'warming up, retrying in {self.args.retry:.0f} s')
                time.sleep(self.args.retry)
        return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--velocity', type=float, default=0.15,
                        help='velocity and acceleration scaling, 0..1')
    parser.add_argument('--order', nargs='+', choices=['left', 'right'],
                        default=['right', 'left'])
    parser.add_argument('--timeout', type=float, default=180.0,
                        help='give up this many seconds after start')
    parser.add_argument('--retry', type=float, default=5.0)
    args, _ros = parser.parse_known_args()

    rclpy.init()
    node = BootPrePick(args)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        ok = node.run()
    except KeyboardInterrupt:
        ok = False
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
