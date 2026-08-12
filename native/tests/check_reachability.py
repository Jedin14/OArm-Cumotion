#!/usr/bin/env python3
"""Ask MoveIt whether the arm can actually get to a point, before you plan to it.

    source native/setup.bash
    python3 native/tests/check_reachability.py --point 0.38 0.15 0.40
    python3 native/tests/check_reachability.py --sweep --z 0.42
    python3 native/tests/check_reachability.py --heights --y 0.15

Needs move_group running (native/run_launch_everything.sh).

Why this exists: this arm's usable envelope is much smaller than its 0.80 m
reach suggests. The shoulders sit at z=0.698 and joint6 is limited to +/-45
degrees, so pointing the tool straight down while extended forward runs out of
wrist long before it runs out of arm. The practical consequence is that reach
*shrinks as the tool goes lower* -- a point that is fine at z=0.45 can be
impossible at z=0.35. Guessing from arm length gets this wrong every time, and
the symptom is an opaque MoveIt error code halfway through a pick.

Pair it with VLM/pixel_to_world.py: that tells you where something is, this
tells you whether the arm can get there.
"""

import argparse
import importlib.util
import math
import os
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
NO_IK_SOLUTION = -31


def load_orchestrator():
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

    def __init__(self, arm='left'):
        super().__init__('check_reachability')
        self.group = f'{arm}_arm'
        self.link = f'openarm_{arm}_hand_tcp'
        self.client = self.create_client(GetPositionIK, '/compute_ik')
        if not self.client.wait_for_service(timeout_sec=10.0):
            raise SystemExit('no /compute_ik -- is move_group running?')

    def solve(self, xyz, quat, timeout=1):
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group
        request.ik_request.ik_link_name = self.link
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = timeout
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        (pose.pose.position.x, pose.pose.position.y,
         pose.pose.position.z) = xyz
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = quat
        request.ik_request.pose_stamped = pose

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        result = future.result()
        return result.error_code.val if result else None


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
                                                math.radians(tilt))) == 1
            any_ok = any_ok or ok
            row += '     #' if ok else '     .'
        print(f'  {tilt:>4}   ' + row)

    if any_ok:
        top_down = checker.solve(xyz, orch.top_down_quat(0.0)) == 1
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
                                                orch.top_down_quat(0.0)) == 1
                      else '     .' for y in ys)
        print(f'  x={x:.2f} ' + row)


def report_heights(checker, orch, y, xs, zs):
    print(f'\n{checker.group}, top-down, y={y:+.2f}   (# = IK solution)')
    print('   tool z=' + ''.join(f'{z:>7.2f}' for z in zs))
    for x in xs:
        row = ''.join('      #' if checker.solve((x, y, z),
                                                 orch.top_down_quat(0.0)) == 1
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
    args, _ = parser.parse_known_args()

    orch = load_orchestrator()
    rclpy.init()
    try:
        checker = ReachChecker(args.arm)
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
