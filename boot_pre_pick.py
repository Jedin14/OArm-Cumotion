#!/usr/bin/env python3
"""Walk both arms to a recorded pose (navigation_state) once the robot is up.

Started by launch_everything.launch.py (boot_pre_pick:=true, the default),
then exits. The arms boot into navigation_state (--state picks another);
click_to_move takes them to pre_pick when you start picking. An arm that is already there, or has no recording in
pick_place_states_<arm>.yaml, is left alone.

Fast path (both arms at once, ~4 s after launch):
    MoveIt's OMPL pipeline plans each arm (plan only; it needs no warm-up),
    the two plans are merged onto one time line and the merged motion is
    collision-checked state by state -- both arms together, so they cannot
    meet -- then executed as one trajectory on both controllers.

Fallback (as before): one cuMotion joint goal per arm, retried while cuMotion
warms up (~27 s of CUDA set-up after launch), via home if there is no direct
path. The boot used to take ~52 s this way (2026-10-07: cuMotion warm-up,
then the right arm at 15 % speed, then the left).

Turn it off (boot_pre_pick:=false) when something else parks the arms:
two nodes driving the arms at once would fight.
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
import rclpy
import yaml
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetStateValidity
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

WS = os.path.dirname(os.path.abspath(__file__))
# The folded posture both arms can reach from almost anywhere. A direct cuMotion move
# that cannot be planned goes via here instead.
HOME = [0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0]
ARRIVED = 0.02            # rad: an arm this close to pre_pick is left alone
SAMPLE = 0.05             # s: merged trajectory resolution and check spacing


def wait(future, timeout):
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    return future.result() if done.wait(timeout) else None


def arm_joints(arm):
    return [f'openarm_{arm}_joint{i}' for i in range(1, 8)]


def seconds(point):
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9


def merge(trajectories):
    """Several joint trajectories (disjoint joints) -> one, on a common time
    line sampled every SAMPLE s; each holds its last point once it is done."""
    names, tables = [], []
    for jt in trajectories:
        t = np.array([seconds(p) for p in jt.points])
        q = np.array([list(p.positions) for p in jt.points])
        names += list(jt.joint_names)
        tables.append((t, q))
    end = max(t[-1] for t, _q in tables)
    times = np.append(np.arange(0.0, end, SAMPLE), end)
    q = np.hstack([np.column_stack([np.interp(times, t, qq[:, j]) for j in range(qq.shape[1])])
                   for t, qq in tables])
    v = np.gradient(q, times, axis=0)
    v[0] = v[-1] = 0.0
    out = JointTrajectory()
    out.joint_names = names
    for ti, qi, vi in zip(times, q, v):
        p = JointTrajectoryPoint(positions=qi.tolist(), velocities=vi.tolist())
        p.time_from_start.sec = int(ti)
        p.time_from_start.nanosec = int((ti % 1.0) * 1e9)
        out.points.append(p)
    return out


class BootPrePick(Node):
    def __init__(self, args):
        super().__init__('boot_pre_pick')
        self.args = args
        self.joints = {}
        self.lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.validity = self.create_client(GetStateValidity, '/check_state_validity')

    # -- inputs ---------------------------------------------------------------

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
        joints = ((data.get('states') or {}).get(self.args.state) or {}).get('joints')
        if not joints or len(joints) != 7:
            self.get_logger().warn(
                f'{arm}: no {self.args.state} in {path}; left where it is. Record one with '
                f'"python3 record_states.py --arm {arm} {self.args.state}"')
            return None
        return [float(v) for v in joints]

    # -- MoveIt ---------------------------------------------------------------

    def goal(self, arm, positions, pipeline, plan_only):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = f'{arm}_arm'
        req.pipeline_id = pipeline
        req.num_planning_attempts = 3 if pipeline == 'ompl' else 1
        req.allowed_planning_time = 2.0 if pipeline == 'ompl' else 5.0
        req.max_velocity_scaling_factor = self.args.velocity
        req.max_acceleration_scaling_factor = self.args.velocity
        req.start_state.is_diff = True
        constraints = Constraints()
        for name, value in zip(arm_joints(arm), positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints = [constraints]
        goal.planning_options.plan_only = plan_only
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def send(self, goal, timeout=90.0):
        """(MoveIt error code or None, planned trajectory or None)."""
        handle = wait(self.move_client.send_goal_async(goal), 15.0)
        if handle is None or not handle.accepted:
            return None, None
        result = wait(handle.get_result_async(), timeout)
        if result is None:
            handle.cancel_goal_async()
            return None, None
        res = result.result
        return res.error_code.val, res.planned_trajectory.joint_trajectory

    def collision_free(self, jt):
        """Every SAMPLE-s state of the merged motion, both arms together."""
        if not self.validity.wait_for_service(timeout_sec=5.0):
            return False
        for p in jt.points:
            req = GetStateValidity.Request()
            req.robot_state.is_diff = True
            req.robot_state.joint_state.name = list(jt.joint_names)
            req.robot_state.joint_state.position = list(p.positions)
            res = wait(self.validity.call_async(req), 2.0)
            if res is None or not res.valid:
                bodies = sorted({(c.contact_body_1, c.contact_body_2)
                                 for c in (res.contacts if res else [])})[:3]
                self.get_logger().warn(f'merged motion collides at {seconds(p):.2f} s: {bodies}')
                return False
        return True

    def execute(self, jt):
        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = RobotTrajectory(joint_trajectory=jt)
        handle = wait(self.execute_client.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            return False
        result = wait(handle.get_result_async(), seconds(jt.points[-1]) + 20.0)
        return result is not None and result.result.error_code.val == MoveItErrorCodes.SUCCESS

    # -- the two ways there ---------------------------------------------------

    def together(self, goals):
        """Fast path: OMPL plans, merged, checked, executed. True if done."""
        log = self.get_logger()
        plans = []
        for arm, positions in goals.items():
            code, jt = self.send(self.goal(arm, positions, 'ompl', plan_only=True), 15.0)
            if code != MoveItErrorCodes.SUCCESS or jt is None or not jt.points:
                log.warn(f'{arm}: OMPL could not plan to {self.args.state} (code {code})')
                return False
            plans.append(jt)
        merged = merge(plans)
        if not self.collision_free(merged):
            return False
        log.info(f'{" and ".join(goals)}: moving to {self.args.state} together '
                 f'({seconds(merged.points[-1]):.1f} s, velocity {self.args.velocity:.2f})')
        return self.execute(merged)

    def one_by_one(self, goals, deadline):
        """Fallback: one cuMotion goal per arm, retried while it warms up."""
        log = self.get_logger()
        ok = True
        for arm, positions in goals.items():
            tries = 0
            while True:
                log.info(f'{arm}: moving to {self.args.state} with cuMotion')
                code, _jt = self.send(self.goal(arm, positions, 'cumotion', plan_only=False))
                if code == MoveItErrorCodes.SUCCESS:
                    log.info(f'{arm}: at {self.args.state}')
                    break
                tries += 1
                if code in (MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.FAILURE) \
                        and tries >= 2:
                    log.info(f'{arm}: no direct path to {self.args.state}; going via home')
                    home, _ = self.send(self.goal(arm, HOME, 'cumotion', plan_only=False))
                    if home == MoveItErrorCodes.SUCCESS:
                        continue
                if time.monotonic() > deadline:
                    log.error(f'{arm}: could not reach {self.args.state} (last code {code})')
                    ok = False
                    break
                log.warn(f'{arm}: not yet (code {code}); cuMotion may still be warming up, '
                         f'retrying in {self.args.retry:.0f} s')
                time.sleep(self.args.retry)
        return ok

    def run(self):
        log = self.get_logger()
        start = time.monotonic()
        deadline = start + self.args.timeout
        log.info('waiting for move_group...')
        while not self.move_client.wait_for_server(timeout_sec=1.0):
            if time.monotonic() > deadline:
                log.error('move_group never came up; arms not moved')
                return False
        goals = {}
        for arm in self.args.order:
            target = self.target(arm)
            if target is None:
                continue
            while self.current(arm_joints(arm)) is None:
                if time.monotonic() > deadline:
                    log.error(f'{arm}: no /joint_states; not moved')
                    return False
                time.sleep(0.2)
            if max(abs(a - b) for a, b in zip(self.current(arm_joints(arm)), target)) < ARRIVED:
                log.info(f'{arm}: already at {self.args.state}')
                continue
            goals[arm] = target
        if not goals:
            return True
        if self.together(goals):
            log.info(f'at {self.args.state} {time.monotonic() - start:.1f} s after start')
            return True
        log.warn('falling back to cuMotion, one arm at a time')
        left = {a: g for a, g in goals.items()
                if max(abs(x - y) for x, y in zip(self.current(arm_joints(a)), g)) >= ARRIVED}
        ok = self.one_by_one(left, deadline)
        log.info(f'boot walk finished {time.monotonic() - start:.1f} s after start')
        return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--state', default='navigation_state',
                        help='pose from pick_place_states_<arm>.yaml to walk to')
    parser.add_argument('--velocity', type=float, default=0.4,
                        help='velocity and acceleration scaling, 0..1')
    parser.add_argument('--order', nargs='+', choices=['left', 'right'],
                        default=['right', 'left'])
    parser.add_argument('--timeout', type=float, default=180.0,
                        help='give up this many seconds after start')
    parser.add_argument('--retry', type=float, default=3.0)
    args, _ros = parser.parse_known_args()

    rclpy.init()
    node = BootPrePick(args)
    executor = MultiThreadedExecutor(num_threads=3)
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
