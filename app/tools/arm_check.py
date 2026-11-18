#!/usr/bin/env python3
"""Rest check for one arm: how still it holds, and how far off its command.

    source native/setup.bash
    python3 app/tools/arm_check.py --arm left --seconds 60

Read-only (2026-11-17): it only LISTENS -- /joint_states, the arm's trajectory
controller state (/<arm>_joint_trajectory_controller/controller_state) and the
motors' status (/dynamic_joint_states) -- for --seconds, then prints per joint:
the mean position and how far that is from the nearest named pose
(robot/pick_place_states_<arm>.yaml), the peak-to-peak and the step-to-step
jitter in mrad, and the command error (the controller's reference minus the
measured position). It sends no command, no goal, no service call: run it with
the arm parked (pre_pick) and the stack up.

The limits it checks are the ones the 2026-11-17 plan set for the driver
changes: peak-to-peak under 2 mrad on joints 1-4 and under 6 mrad on the wrist
(5-7). Exit status: 0 within them, 1 not, 2 no data.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'robot')
PTP_LIMIT_MRAD = (2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0)


def arm_joints(arm):
    return [f'openarm_{arm}_joint{i}' for i in range(1, 8)]


def load_poses(arm):
    """{name: [7 joints]} from robot/pick_place_states_<arm>.yaml ({} if unreadable)."""
    try:
        import yaml
        with open(os.path.join(ROBOT_DIR, f'pick_place_states_{arm}.yaml')) as f:
            states = yaml.safe_load(f).get('states', {})
        return {k: [float(v) for v in s['joints']] for k, s in states.items()
                if isinstance(s, dict) and len(s.get('joints', [])) == 7}
    except Exception:                                # noqa: BLE001 - optional
        return {}


def summarise(arm, positions, efforts, state, status, seconds):
    """The numbers, as a dict (also what --json writes)."""
    names = arm_joints(arm)
    out = {'arm': arm, 'seconds': seconds, 'samples': len(positions), 'joints': []}
    if len(positions) < 2:
        return out
    q = np.array(positions)
    mean = q.mean(0)
    ptp = 1000.0 * np.ptp(q, axis=0)
    std = 1000.0 * q.std(0)
    jitter = 1000.0 * np.sqrt(np.mean(np.diff(q, axis=0) ** 2, axis=0))
    tau = np.array(efforts).mean(0) if len(efforts) else [math.nan] * 7
    err = np.array([e for e in state['error'] if e is not None]) if state['error'] else None
    ref = np.array([r for r in state['reference'] if r is not None]) if state['reference'] \
        else None
    poses = load_poses(arm)
    near = None
    if poses:
        name = min(poses, key=lambda k: float(np.max(np.abs(np.subtract(poses[k], mean)))))
        near = {'name': name,
                'off_mrad': [round(1000.0 * (m - p), 1) for m, p in zip(mean, poses[name])]}
    out['pose'] = near
    for i, n in enumerate(names):
        j = {'joint': n, 'mean': round(float(mean[i]), 4), 'ptp_mrad': round(float(ptp[i]), 2),
             'std_mrad': round(float(std[i]), 2), 'jitter_mrad': round(float(jitter[i]), 3),
             'effort': round(float(tau[i]), 3), 'ptp_limit_mrad': PTP_LIMIT_MRAD[i]}
        if err is not None and len(err):
            j['cmd_err_mean_mrad'] = round(1000.0 * float(err[:, i].mean()), 2)
            j['cmd_err_max_mrad'] = round(1000.0 * float(np.abs(err[:, i]).max()), 2)
        if ref is not None and len(ref):
            j['command'] = round(float(ref[:, i].mean()), 4)
            j['command_ptp_mrad'] = round(1000.0 * float(np.ptp(ref[:, i])), 2)
        s = status.get(n)
        if s:
            j['status_first'], j['status_last'], j['status_changes'] = s[0], s[1], s[2]
        out['joints'].append(j)
    out['within_limits'] = bool(all(j['ptp_mrad'] < j['ptp_limit_mrad'] for j in out['joints']))
    return out


def print_report(r):
    print(f"{r['arm']} arm, {r['seconds']:.0f} s, {r['samples']} joint-state samples")
    if not r['joints']:
        print('  no /joint_states for this arm - is the stack up (and the same ROS_DOMAIN_ID)?')
        return
    if r.get('pose'):
        print(f"  nearest named pose: {r['pose']['name']}")
    head = ('joint', 'mean rad', 'p-p mrad', 'limit', 'jitter', 'cmd err', 'max err',
            'vs pose', 'effort', 'status')
    print('  ' + ' '.join(f'{h:>9}' for h in head))
    for i, j in enumerate(r['joints']):
        pose = r['pose']['off_mrad'][i] if r.get('pose') else math.nan
        status = (f"{j['status_last']:g}" + (f" ({j['status_changes']}x)"
                                             if j.get('status_changes') else '')
                  if 'status_last' in j else '-')
        row = (f'j{i + 1}', f"{j['mean']:+.4f}", f"{j['ptp_mrad']:.2f}",
               f"<{j['ptp_limit_mrad']:g}", f"{j['jitter_mrad']:.3f}",
               f"{j.get('cmd_err_mean_mrad', math.nan):+.1f}",
               f"{j.get('cmd_err_max_mrad', math.nan):.1f}", f'{pose:+.1f}',
               f"{j['effort']:+.3f}", status)
        flag = '' if j['ptp_mrad'] < j['ptp_limit_mrad'] else '   <-- over'
        print('  ' + ' '.join(f'{c:>9}' for c in row) + flag)
    print('  within the rest limits' if r['within_limits'] else
          '  NOT within the rest limits (p-p: < 2 mrad j1-j4, < 6 mrad j5-j7)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm', choices=('left', 'right'), default='left')
    parser.add_argument('--seconds', type=float, default=60.0)
    parser.add_argument('--json', default=None, help='also write the numbers here')
    args = parser.parse_args()

    import rclpy
    from control_msgs.msg import DynamicJointState, JointTrajectoryControllerState
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    names = arm_joints(args.arm)
    positions, efforts = [], []
    state = {'error': [], 'reference': []}
    status = {}                                     # joint -> [first, last, changes]

    rclpy.init()
    node = Node('arm_check')

    def on_joints(msg):
        idx = {n: k for k, n in enumerate(msg.name)}
        if all(n in idx for n in names):
            positions.append([msg.position[idx[n]] for n in names])
            if len(msg.effort) == len(msg.name):
                efforts.append([msg.effort[idx[n]] for n in names])

    def pick(point, joint_names):
        idx = {n: k for k, n in enumerate(joint_names)}
        if not point.positions or not all(n in idx for n in names):
            return None
        return [point.positions[idx[n]] for n in names]

    def on_state(msg):
        jn = list(msg.joint_names)
        ref, fb = msg.reference, msg.feedback
        if not ref.positions:                       # (older controllers: desired/actual)
            ref, fb = msg.desired, msg.actual
        r, f = pick(ref, jn), pick(fb, jn)
        e = pick(msg.error, jn)
        if e is None and r is not None and f is not None:
            e = [a - b for a, b in zip(r, f)]
        if r is not None:
            state['reference'].append(r)
        if e is not None:
            state['error'].append(e)

    def on_dynamic(msg):
        for n, iv in zip(msg.joint_names, msg.interface_values):
            if n not in names or 'status' not in iv.interface_names:
                continue
            v = float(iv.values[list(iv.interface_names).index('status')])
            s = status.setdefault(n, [v, v, 0])
            if v != s[1] and not (math.isnan(v) and math.isnan(s[1])):
                s[2] += 1
            s[1] = v

    node.create_subscription(JointState, '/joint_states', on_joints, 50)
    node.create_subscription(JointTrajectoryControllerState,
                             f'/{args.arm}_joint_trajectory_controller/controller_state',
                             on_state, 50)
    node.create_subscription(DynamicJointState, '/dynamic_joint_states', on_dynamic, 50)
    print(f'listening for {args.seconds:.0f} s (read-only: nothing is sent)...', flush=True)
    t_end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    report = summarise(args.arm, positions, efforts, state, status, args.seconds)
    print_report(report)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=1)
    if not report['joints']:
        return 2
    return 0 if report['within_limits'] else 1


if __name__ == '__main__':
    sys.exit(main())
