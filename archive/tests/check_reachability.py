#!/usr/bin/env python3
"""Ask the real planner whether the arm can get to a pose, before planning to it.

    source native/setup.bash
    python3 native/tests/check_reachability.py --point 0.54 -0.02 0.35
    python3 native/tests/check_reachability.py --sweep --z 0.42
    python3 native/tests/check_reachability.py --heights --y 0.15

Needs move_group and the cuMotion planner running
(native/run_launch_everything.sh). Everything here is plan-only: nothing moves.

It asks **cuMotion**, via a plan_only goal on /move_action, because that is what
the orchestrator will use. Do not use /compute_ik for this: the configured
solver is kdl_kinematics_plugin with a 5 ms timeout (config/kinematics.yaml),
which fails constantly on this redundant 7-DOF arm and will tell you a perfectly
reachable pose is impossible. `--ik` is kept only to show that contrast.

What this arm is actually fussy about is *orientation*, not distance. joint6 is
limited to +/-45 degrees, so a strictly top-down tool at full forward extension
runs out of wrist while the position itself is fine -- which is why a pose you
can reach by teleoperation can still come back IK_FAIL when you demand top-down.
--point sweeps approach azimuth and tilt for exactly this reason: read it as
"which approach angles work here", not "can the arm reach here".

Pair it with VLM/pixel_to_world.py: that tells you where something is, this
tells you how the arm can get to it.
"""

import argparse
import importlib.util
import math
import os
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SUCCESS = 1


def load_orchestrator():
    # Loading by path does not put the workspace on sys.path, and the
    # orchestrator imports vlm_prompt from beside itself.
    if WS not in sys.path:
        sys.path.insert(0, WS)
    path = os.path.join(WS, 'pick_place_orchestrator.py')
    spec = importlib.util.spec_from_file_location('pick_place_orchestrator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tilted_quat(orch, azimuth, tilt):
    """Tool +Z = Rz(azimuth) . (sin tilt, 0, -cos tilt). tilt 0 is straight down."""
    qz = (0.0, 0.0, math.sin(azimuth / 2), math.cos(azimuth / 2))
    a = math.pi - tilt
    qy = (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
    return orch.quat_mul(qz, qy)


class ReachChecker(Node):

    def __init__(self, arm='left', pipeline='cumotion', use_ik=False):
        super().__init__('check_reachability')
        self.group = f'{arm}_arm'
        self.link = f'openarm_{arm}_hand_tcp'
        self.pipeline = pipeline
        self.use_ik = use_ik

        if use_ik:
            self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
            if not self.ik_client.wait_for_service(timeout_sec=10.0):
                raise SystemExit('no /compute_ik -- is move_group running?')
        else:
            self.move_client = ActionClient(self, MoveGroup, '/move_action')
            if not self.move_client.wait_for_server(timeout_sec=15.0):
                raise SystemExit('no /move_action -- is move_group running?')

    def solve(self, xyz, quat):
        """True if the pose is achievable. Nothing is executed."""
        return self._ask_ik(xyz, quat) if self.use_ik else self._ask_planner(xyz, quat)

    def _ask_ik(self, xyz, quat):
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group
        request.ik_request.ik_link_name = self.link
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = 1
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z) = xyz
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = quat
        request.ik_request.pose_stamped = pose

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        result = future.result()
        return bool(result and result.error_code.val == SUCCESS)

    def _ask_planner(self, xyz, quat):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.group
        req.pipeline_id = self.pipeline
        req.num_planning_attempts = 1
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.1
        req.max_acceleration_scaling_factor = 0.1
        req.start_state.is_diff = True
        req.workspace_parameters.header.frame_id = 'world'
        for corner, sign in ((req.workspace_parameters.min_corner, -1.5),
                             (req.workspace_parameters.max_corner, 1.5)):
            corner.x = corner.y = corner.z = sign

        pc = PositionConstraint()
        pc.header.frame_id = 'world'
        pc.link_name = self.link
        pc.weight = 1.0
        region = SolidPrimitive()
        region.type = SolidPrimitive.BOX
        region.dimensions = [0.02] * 3
        pc.constraint_region.primitives.append(region)
        target = PoseStamped().pose
        target.position.x, target.position.y, target.position.z = xyz
        target.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(target)

        oc = OrientationConstraint()
        oc.header.frame_id = 'world'
        oc.link_name = self.link
        (oc.orientation.x, oc.orientation.y,
         oc.orientation.z, oc.orientation.w) = quat
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0

        req.goal_constraints = [Constraints(position_constraints=[pc],
                                            orientation_constraints=[oc])]
        goal.planning_options.plan_only = True          # never executes
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        send = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=20.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=40.0)
        result = result_future.result()
        return bool(result and result.result.error_code.val == SUCCESS)


def report_point(checker, orch, xyz):
    """One point, swept over orientation, so a failure is clearly geometric."""
    print(f'\ntarget ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f})   # = IK solution')
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
    print('  tilt\\az' + ''.join(f'{a:>6}' for a in azimuths))
    any_ok = False
    for tilt in (0, 30, 60):
        row = ''
        for az in azimuths:
            ok = checker.solve(xyz, tilted_quat(orch, math.radians(az),
                                               math.radians(tilt)))
            any_ok = any_ok or ok
            row += '     #' if ok else '     .'
        print(f'  {tilt:>4}   ' + row)

    if any_ok:
        top_down = checker.solve(xyz, orch.top_down_quat(0.0))
        if top_down:
            print('\n  reachable, and the top-down grasp the orchestrator uses works')
        else:
            print('\n  reachable only at an angle -- the orchestrator only does '
                  'top-down, so treat this as unusable')
    else:
        print('\n  NOT reachable at any orientation. Raise it: this arm loses '
              'reach as the tool goes lower.')
    return any_ok


def report_sweep(checker, orch, z, xs, ys):
    print(f'\n{checker.group}, top-down, tool z={z:.3f}   (# = IK solution)')
    print('        y=' + ''.join(f'{y:+6.2f}' for y in ys))
    for x in xs:
        row = ''.join('     #' if checker.solve((x, y, z),
                                                orch.top_down_quat(0.0))
                      else '     .' for y in ys)
        print(f'  x={x:.2f} ' + row)


def report_heights(checker, orch, y, xs, zs):
    print(f'\n{checker.group}, top-down, y={y:+.2f}   (# = IK solution)')
    print('   tool z=' + ''.join(f'{z:>7.2f}' for z in zs))
    for x in xs:
        row = ''.join('      #' if checker.solve((x, y, z),
                                                 orch.top_down_quat(0.0))
                      else '      .' for z in zs)
        print(f'  x={x:.2f} ' + row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='check one point over all orientations')
    parser.add_argument('--sweep', action='store_true',
                        help='map the xy plane at --z')
    parser.add_argument('--heights', action='store_true',
                        help='map x against tool height at --y')
    parser.add_argument('--z', type=float, default=0.42, help='height for --sweep')
    parser.add_argument('--y', type=float, default=0.15, help='y for --heights')
    parser.add_argument('--arm', default='left', choices=('left', 'right'))
    parser.add_argument('--pipeline', default='cumotion',
                        help='planning pipeline to ask (cumotion, ompl)')
    parser.add_argument('--ik', action='store_true',
                        help='use /compute_ik (KDL) instead -- unreliable here, '
                             'kept for comparison only')
    args, _ = parser.parse_known_args()

    orch = load_orchestrator()
    rclpy.init()
    try:
        checker = ReachChecker(args.arm, args.pipeline, args.ik)
        if args.point:
            report_point(checker, orch, tuple(args.point))
        if args.sweep:
            report_sweep(checker, orch, args.z,
                         [0.25, 0.30, 0.35, 0.40, 0.45],
                         [-0.20, -0.10, 0.00, 0.10, 0.20, 0.30])
        if args.heights:
            report_heights(checker, orch, args.y,
                           [0.30, 0.35, 0.40, 0.45, 0.50],
                           [0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
        if not (args.point or args.sweep or args.heights):
            parser.print_help()
        return 0
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
