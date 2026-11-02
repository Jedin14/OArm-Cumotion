"""WEIGH TOOL: the real mass and centre of mass of everything after joint 7
(2026-10-27).

The hardware holds each joint with Kp = 100 Nm/rad plus a gravity
feed-forward from the URDF's masses (v10_simple_hardware.cpp, KDL, chain to
the hand) and a slow integral. The left wrist and gripper became a cylinder
and a 4.3 cm vacuum tool, the fingers went -- but the URDF kept the old
masses, so the feed-forward is wrong by up to ~4 Nm: the arm arrived 30 mm
off (38 mrad short) and the integral needed seconds to pull it in.

Held STILL, a motor's torque is not the gravity load: the gears' friction
holds part of it (still, the left arm read 74 % of the load, +-0.7 Nm, one
posture 3.75 Nm off -- 2026-10-27). So at each posture every joint moves a
little, slowly, one way and back: friction adds one way and subtracts the
other, and the mean of the two is gravity. Against the
URDF's prediction, what is missing is linear in 4 unknowns: the extra mass
after joint 7 (dm) and dm times its centre of mass in the hand frame (v):

    tau_j = z_j . ((p_hand - o_j) x G) dm + (R_hand^T (G x z_j)) . v

-- 10-20 still postures, least squares. The result is written into the left
hand's inertial in openarm_hand.xacro and openarm.urdf (the finger masses
to ~0: they are gone), and the hardware uses it after a restart.
"""

import math
import os
import re
import subprocess
import time

import numpy as np
import yaml

from ctm.common import BASE_FRAME, WS, quat_from_matrix, tool_frame_along
from ctm.kinematics import G, Arm, frames_batch, gravity_torque_batch

FILE = os.path.join(WS, 'payload_{arm}.yaml')
XACRO = os.path.join(WS, 'src/openarm_description/urdf/ee/openarm_hand.xacro')
URDF = os.path.join(WS, 'openarm.urdf')
POSES = 16


def regressor(kin, q):
    """(N, 7, 4): each joint's torque per unit of (dm, v) at postures q."""
    frames, joints = frames_batch(kin, q)
    h = kin.links.index(f'openarm_{kin.arm}_hand')
    rot, pos = frames[h][:, :3, :3], frames[h][:, :3, 3]
    a = np.zeros((len(pos), 7, 4))
    for k, (origin, axis) in enumerate(joints):
        a[:, k, 0] = np.einsum('ni,ni->n', axis, np.cross(pos - origin, G))
        a[:, k, 1:] = np.einsum('nji,nj->ni', rot, np.cross(G[None], axis))
    return a


def fit(arm, qs, taus):
    """qs, taus: (N, 7) joints and gravity torques (friction already taken
    out). Links 1-6 as the URDF has them; the body after joint 7 -- link7 and
    the hand move as one, both were replaced -- fitted: its mass and centre
    of mass (hand frame). (Letting links 1-6 scale too fitted no better and
    gave a 55 g body with its centre 34 cm off: not identifiable, 2026-10-27.)"""
    import copy
    kin = Arm(arm)
    qs, taus = np.asarray(qs, float), np.asarray(taus, float)
    model = gravity_torque_batch(kin, qs)
    sign = -1.0 if np.sum(model * taus) < 0 else 1.0     # holding torque vs load
    meas = sign * taus
    h = kin.links.index(f'openarm_{arm}_hand')
    l7 = kin.links.index(f'openarm_{arm}_link7')
    base = copy.deepcopy(kin)
    base.masses[h] = 0.0
    base.masses[l7] = 0.0
    t_base = gravity_torque_batch(base, qs)
    a = regressor(kin, qs)
    rows = slice(0, 6)
    M = a[:, rows, :].reshape(-1, 4)
    b = (meas - t_base)[:, rows].reshape(-1)
    x, *_ = np.linalg.lstsq(M, b, rcond=None)
    for _ in range(2):                                      # down-weight outliers
        r = M @ x - b
        w = 1.0 / np.maximum(np.abs(r), np.percentile(np.abs(r), 50))
        x, *_ = np.linalg.lstsq(M * w[:, None], b * w, rcond=None)
    m_new, v = float(x[0]), x[1:]
    c_new = v / m_new if m_new > 0.02 else np.asarray(kin.coms[h])
    before = float(np.sqrt(np.mean((model[:, rows] - meas[:, rows]) ** 2)))
    after = float(np.sqrt(np.mean((M @ x - b) ** 2)))
    return {'sign': float(sign), 'rms_before_nm': before, 'rms_after_nm': after,
            'link7_mass_was': float(kin.masses[l7]),
            'hand_mass_was': float(kin.masses[h]),
            'hand_com_was': [float(c) for c in kin.coms[h]],
            'hand_mass': float(m_new), 'hand_com': [float(c) for c in c_new],
            'postures': int(len(qs))}


def _inertial(mass, com, indent):
    """An <inertial> for the hand: the mass at com, the inertia of a 30 mm
    cylinder 0.24 m long about it (dynamics only: gravity needs mass + com)."""
    r, length = 0.015, 0.24
    ixx = mass * (3 * r * r + length * length) / 12.0
    izz = mass * r * r / 2.0
    s = ' ' * indent
    return (f'{s}<inertial>\n'
            f'{s}  <origin rpy="0 0 0" xyz="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}"/>\n'
            f'{s}  <mass value="{mass:.4f}"/>\n'
            f'{s}  <inertia ixx="{ixx:.3e}" ixy="0" ixz="0" iyy="{ixx:.3e}" iyz="0" izz="{izz:.3e}"/>\n'
            f'{s}</inertial>')


def apply(arm, result, rebuild=True):
    """The fitted mass into the left hand's inertial (xacro + openarm.urdf),
    the finger masses to ~0; rebuild openarm_description. Returns a message."""
    if arm != 'left':
        return 'only the left arm (its gripper was replaced) is written'
    mass, com = result['hand_mass'], result['hand_com']
    if not 0.05 < mass < 3.0:
        return f'not written: {mass:.2f} kg is not plausible - weigh again'
    stamp = time.strftime('%Y%m%d-%H%M%S')
    # openarm.urdf: the left hand's and fingers' <inertial> blocks
    s = open(URDF).read()
    open(URDF + f'.before_weigh_{stamp}', 'w').write(s)
    for link, m, c in ((f'openarm_{arm}_hand', mass, com),
                       (f'openarm_{arm}_link7', 0.001, (0, 0, 0.05)),
                       (f'openarm_{arm}_left_finger', 0.001, (0, 0, 0)),
                       (f'openarm_{arm}_right_finger', 0.001, (0, 0, 0))):
        a = s.index(f'<link name="{link}">')
        b = s.index('</link>', a)
        block = s[a:b]
        new = _inertial(m, c, 4)
        block = (re.sub(r'\s*<inertial>.*?</inertial>', '\n' + new, block, flags=re.S)
                 if '<inertial>' in block else block.rstrip() + '\n' + new + '\n  ')
        s = s[:a] + block + s[b:]
    open(URDF, 'w').write(s)
    # the xacro: the left hand (explicit), the left fingers (tiny)
    x = open(XACRO).read()
    open(XACRO + f'.before_weigh_{stamp}', 'w').write(x)
    left = x.index("<xacro:if value=\"${arm_prefix == 'left_'}\">")
    hand_end = x.index('</link>', left)
    seg = x[left:hand_end]
    seg = re.sub(r'<xacro:ee-inertials name="hand" />|<inertial>.*?</inertial>',
                 '\n' + _inertial(mass, com, 8).lstrip() + '\n', seg, count=1, flags=re.S)
    x = x[:left] + seg + x[hand_end:]
    fl = x.index("<xacro:if value=\"${arm_prefix == 'left_'}\">", left + len(seg))
    fe = x.index('</xacro:if>', fl)
    fseg = re.sub(r'<xacro:ee-inertials name="(left|right)_finger"/>|<inertial>.*?</inertial>',
                  _inertial(0.001, (0, 0, 0), 8).lstrip(), x[fl:fe], flags=re.S)
    x = x[:fl] + fseg + x[fe:]
    open(XACRO, 'w').write(x)
    # link7 (left): its mass is in the hand's now (one body after joint 7)
    arm_xacro = os.path.join(WS, 'src/openarm_description/urdf/arm/openarm_arm.xacro')
    ax = open(arm_xacro).read()
    open(arm_xacro + f'.before_weigh_{stamp}', 'w').write(ax)
    l7 = ax.index('<link name="${prefix}link7">')
    l7e = ax.index('</link>', l7)
    ax = ax[:l7] + re.sub(r'<xacro:openarm-inertials name="link7"[^>]*/>|<inertial>.*?</inertial>',
                          _inertial(0.001, (0, 0, 0.05), 8).lstrip(), ax[l7:l7e], count=1,
                          flags=re.S) + ax[l7e:]
    open(arm_xacro, 'w').write(ax)
    msg = (f'everything after joint 7 now weighs {mass:.3f} kg in the hand (was '
           f'{result.get("link7_mass_was", 0) + result["hand_mass_was"]:.3f}: link7 + hand + '
           f'fingers), centre of mass ({1000 * com[0]:.0f}, {1000 * com[1]:.0f}, '
           f'{1000 * com[2]:.0f}) mm in the hand frame')
    if rebuild:
        done = subprocess.run([os.path.join(WS, 'native', 'build_ws.sh'), '--packages-select',
                               'openarm_description'], capture_output=True, text=True)
        msg += ('; openarm_description rebuilt' if done.returncode == 0
                else '; REBUILD FAILED: run native/build_ws.sh --packages-select openarm_description')
    return msg + ' - restart run_launch_everything.sh to use it'


class PayloadMixin:
    """Part of click_to_move.App: WEIGH TOOL."""

    def weigh_tool(self, arm='left'):
        """Still postures, the motors' torques there, the fit. Nothing written
        until APPLY WEIGHT."""
        node = self.node
        self.say(f'WEIGH TOOL: {arm} arm, nothing held on the cup - choosing postures...')
        if not self.ensure_pick_mode():
            return
        self.capture()
        if self.frame is None:
            return
        info, optical = self.frame[2], self.frame[3]
        t_wo = node.lookup(BASE_FRAME, optical)
        start = node.arm_positions(arm)
        rng = np.random.default_rng(7)
        cands = []
        for tip, axis, roll in self._front_poses(arm, t_wo, info, rng):
            if len(cands) >= 120:
                break
            quat = quat_from_matrix(tool_frame_along(axis, math.radians(roll)))
            tcp = node.tcp_for_tip(tip, axis, arm)
            seeds = node.fast_ik(arm, tcp, quat, start, want=1)
            sol = node.solve_ik(arm, tcp, quat, seeds[0]) if seeds else None
            if sol is not None:
                cands.append(np.array(sol))
        if len(cands) < 8:
            self.say(f'WEIGH TOOL: only {len(cands)} postures reachable - not enough', True)
            return
        # spread out in joint space (each joint's load changes with all of them)
        chosen = [0]
        dist = np.linalg.norm(np.array(cands) - cands[0], axis=1)
        while len(chosen) < min(POSES, len(cands)):
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(np.array(cands) - cands[nxt], axis=1))
        self.abort.clear()
        self.calibrating = True
        qs, taus = [], []
        try:
            for n, i in enumerate(chosen, start=1):
                if self.abort.is_set():
                    raise KeyboardInterrupt
                self.say(f'WEIGH TOOL: posture {n}/{len(chosen)} ({len(qs)} measured)')
                if not node.move_joints(arm, cands[i].tolist(), f'weigh {n}', via_home=False,
                                        attempts=1):
                    continue
                node.settle(arm, cands[i].tolist(), timeout=3.0)
                got = self._both_ways(arm)
                if got is not None:
                    qs.append(got[0])
                    taus.append(got[1])
        except KeyboardInterrupt:
            self.say('WEIGH TOOL: aborted; nothing written', True)
            return
        finally:
            self.calibrating = False
            self._back_to_pre_pick(arm)
        if len(qs) < 8:
            self.say(f'WEIGH TOOL: only {len(qs)} postures measured - not enough', True)
            return
        np.savez(os.path.join(WS, 'armcal_samples',
                              f'{arm}_weigh_{time.strftime("%Y-%m-%dT%H-%M-%S")}.npz'),
                 q=np.array(qs), tau=np.array(taus))
        result = fit(arm, qs, taus)
        with open(FILE.format(arm=arm), 'w') as handle:
            handle.write('# WEIGH TOOL (ctm/payload.py): after joint 7, measured\n'
                         + yaml.safe_dump(result, sort_keys=False))
        self.payload_result = (arm, result)
        self.say(f'WEIGH TOOL: everything after joint 7 weighs {result["hand_mass"]:.3f} kg (the '
                 f'URDF says {result["link7_mass_was"] + result["hand_mass_was"]:.3f}); gravity '
                 f'torques now '
                 f'{result["rms_after_nm"]:.2f} Nm off (were {result["rms_before_nm"]:.2f}). '
                 f'APPLY WEIGHT writes it into the URDF')

    def _both_ways(self, arm, delta=0.08, speed=0.05):
        """Every joint a little way and back, slowly (joint steps, checked);
        the torques while moving, each way. Their mean: gravity, the gears'
        friction taken out. (joints, torques) at the middle, or None."""
        import threading
        node = self.node
        q0 = np.asarray(node.arm_positions(arm), float)
        lim = node.joint_limits(arm)
        # each joint away from its nearer limit
        u = np.array([1.0 if (hi - q) > (q - lo) else -1.0 for q, (lo, hi) in zip(q0, lim)]) \
            if lim else np.ones(7)
        ways = []
        for target in (q0 + u * delta, q0):
            samples, stop = [], threading.Event()

            def record():
                while not stop.is_set():
                    q, e = node.arm_positions(arm), node.arm_efforts(arm)
                    if q is not None and e is not None:
                        samples.append((time.monotonic(), q, e))
                    time.sleep(0.02)
            th = threading.Thread(target=record, daemon=True)
            th.start()
            done = node.joint_step(arm, target, 'weigh move', max_step=delta + 0.01,
                                   speed=speed)
            stop.set()
            th.join(1.0)
            if done is None or len(samples) < 10:
                return None
            t = np.array([s_[0] for s_ in samples])
            q = np.array([s_[1] for s_ in samples])
            e = np.array([s_[2] for s_ in samples])
            v = np.linalg.norm(np.gradient(q, t, axis=0), axis=1)
            fast = v > 0.6 * v.max()                  # the steady middle of the move
            if fast.sum() < 5:
                return None
            ways.append((q[fast].mean(axis=0), e[fast].mean(axis=0)))
        node.settle(arm, q0.tolist(), timeout=1.0)
        return (ways[0][0] + ways[1][0]) / 2.0, (ways[0][1] + ways[1][1]) / 2.0

    def apply_weight(self):
        got = getattr(self, 'payload_result', None)
        if got is None:
            try:
                arm = 'left'
                got = (arm, yaml.safe_load(open(FILE.format(arm=arm))))
            except (OSError, yaml.YAMLError):
                self.say('APPLY WEIGHT: weigh the tool first', True)
                return
        arm, result = got
        self.say('APPLY WEIGHT: ' + apply(arm, result))
