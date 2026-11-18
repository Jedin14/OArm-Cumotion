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
hand's inertial in openarm_hand.xacro and openarm.urdf (link7 and the finger
masses to ~0: link7 and the hand are one body, weighed together), and the
hardware uses it after a rebuild of openarm_description and a restart.

Why APPLY WEIGHT never got written (2026-11-16, from the logs):
- 10-27 12:41: the fit of that day wrote "hand = everything after joint 7
  minus link7's URDF mass"; the first weighing's torques were friction-ridden
  (0.68 Nm left of 1.10) and the total came out at link7's 0.466 kg, so the
  hand got -0.0002 kg -> "-0.00 kg is not plausible". Fixed the same day:
  link7 + hand are one body.
- 10-27 15:56: the second weighing crashed writing payload_left.yaml:
  yaml.safe_dump cannot write numpy floats (kin.coms[h][0] = numpy 0.0 ->
  "cannot represent an object 0.0"). Fixed with float()s, the yaml re-fitted
  and written by hand -- but APPLY WEIGHT was never pressed again.
- Still wrong until today: the arm xacro's path was fixed inside apply() (a
  test on copies wrote the real file), apply() said nothing a caller could
  check, a fit that explained nothing would have been written, the rebuild
  could hang, and the cached kinematics kept the old masses.
"""

import copy
import math
import os
import re
import subprocess
import time

import numpy as np
import yaml

from ctm.common import (BACKUP_DIR, BASE_FRAME, CALIB_DIR, OPENARM_URDF, SAMPLES_DIR, WS,
                        plain,
                        quat_from_matrix,
                        tool_frame_along)
from ctm.kinematics import G, Arm, frames_batch, gravity_torque_batch

FILE = os.path.join(CALIB_DIR, 'payload_{arm}.yaml')
XACRO = os.path.join(WS, 'src/openarm_description/urdf/ee/openarm_hand.xacro')
ARM_XACRO = os.path.join(WS, 'src/openarm_description/urdf/arm/openarm_arm.xacro')
# what the motors' robot_description is made from at launch (copies, not links)
INSTALLED = os.path.join(WS, 'native/install/openarm_description/share/openarm_description/urdf')
URDF = OPENARM_URDF
POSES = 16
LEFT_IF = "<xacro:if value=\"${arm_prefix == 'left_'}\">"

# A fit goes into the URDF only when it is a measurement (2026-11-16): a mass
# a 30 mm cylinder and a cup can have, and torques it explains -- the
# friction-ridden first weighing of 10-27 left 0.68 Nm of 1.10.
MASS_RANGE = (0.05, 1.5)          # kg after joint 7
RMS_MAX = 0.5                     # Nm the fit may leave
COM_MAX = 0.2                     # m from the hand frame
# Re-weighed within this of what the URDF has: nothing to write, no restart.
SAME_KG, SAME_M = 0.03, 0.02
DUMMY = 0.001                     # kg left on link7 and the fingers
REBUILD_S = 900


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
    gave a 55 g body with its centre 34 cm off: not identifiable, 2026-10-27.)
    Plain Python floats only: yaml cannot write numpy's."""
    kin = Arm(arm, URDF)
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
    return plain({'sign': float(sign), 'rms_before_nm': before, 'rms_after_nm': after,
                  'link7_mass_was': float(kin.masses[l7]),
                  'hand_mass_was': float(kin.masses[h]),
                  'hand_com_was': [float(c) for c in kin.coms[h]],
                  'hand_mass': float(m_new), 'hand_com': [float(c) for c in c_new],
                  'postures': int(len(qs))})


def check(result):
    """Why `result` must not go into the URDF, or None."""
    nan = float('nan')
    mass = float(result.get('hand_mass', nan))
    before = float(result.get('rms_before_nm', nan))
    after = float(result.get('rms_after_nm', nan))
    com = np.asarray(result.get('hand_com', [nan] * 3), float)
    if not MASS_RANGE[0] < mass < MASS_RANGE[1]:      # (NaN fails every test here)
        return (f'{mass:.2f} kg is not plausible (expected {MASS_RANGE[0]:.2f}-'
                f'{MASS_RANGE[1]:.1f} kg after joint 7)')
    if com.shape != (3,) or not np.all(np.isfinite(com)) or np.linalg.norm(com) > COM_MAX:
        return f'its centre of mass {np.round(1000 * com).tolist()} mm is not plausible'
    if not after < RMS_MAX:
        return f'the fit still leaves {after:.2f} Nm of the torques (limit {RMS_MAX:.1f})'
    if not after < 0.5 * before:
        return f'the fit explains too little of the torques ({before:.2f} -> {after:.2f} Nm)'
    return None


def save_result(arm, result):
    """payload_<arm>.yaml: what APPLY WEIGHT writes later."""
    with open(FILE.format(arm=arm), 'w') as handle:
        handle.write('# WEIGH TOOL (ctm/payload.py): after joint 7, measured\n'
                     + yaml.safe_dump(plain(result), sort_keys=False))


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


def _swap_inertials(text, mass, com, macro=None, count=0):
    """text with its <inertial> blocks (or the xacro calls `macro` makes them
    with) replaced by ours, each at the indent of what it replaces -- so a
    second APPLY WEIGHT writes the same file again (the old one added blank
    lines every time). ValueError when there is none to replace."""
    what = r'<inertial>.*?</inertial>' + (f'|{macro}' if macro else '')
    new, n = re.subn(rf'^([ \t]*)(?:{what})', lambda m: _inertial(mass, com, len(m.group(1))),
                     text, count=count, flags=re.S | re.M)
    if n == 0:
        raise ValueError(f'no inertial to replace in: {text[:60]!r}...')
    return new


def _urdf_after_j7(arm):
    """(link7 + hand + fingers kg, hand centre of mass, link7 kg) as the
    URDF has them now (the fingers ride on the hand, ctm/kinematics.py)."""
    kin = Arm(arm, URDF)
    h = kin.links.index(f'openarm_{arm}_hand')
    l7 = kin.links.index(f'openarm_{arm}_link7')
    return kin.masses[l7] + kin.masses[h], np.asarray(kin.coms[h], float), kin.masses[l7]


def _built():
    """True when the installed xacros are the source ones (the motors read
    the installed at launch)."""
    try:
        return all(open(src).read() == open(os.path.join(INSTALLED, sub)).read()
                   for src, sub in ((XACRO, 'ee/openarm_hand.xacro'),
                                    (ARM_XACRO, 'arm/openarm_arm.xacro')))
    except OSError:
        return False


def _new_urdf(s, arm, mass, com):
    """openarm.urdf (the Python kinematics and the planners): the four links."""
    for link, m, c in ((f'openarm_{arm}_hand', mass, com),
                       (f'openarm_{arm}_link7', DUMMY, (0, 0, 0.075)),
                       (f'openarm_{arm}_left_finger', DUMMY, (0, 0, 0)),
                       (f'openarm_{arm}_right_finger', DUMMY, (0, 0, 0))):
        a = s.index(f'<link name="{link}">')
        b = s.index('</link>', a)
        block = s[a:b]
        block = (_swap_inertials(block, m, c, count=1) if '<inertial>' in block
                 else block.rstrip() + '\n' + _inertial(m, c, 4) + '\n  ')
        s = s[:a] + block + s[b:]
    return s


def _new_hand_xacro(x, mass, com):
    """The hand xacro (the motors, after a rebuild): the left hand, then the
    left fingers (the second left-only block)."""
    left = x.index(LEFT_IF)
    end = x.index('</link>', left)
    x = x[:left] + _swap_inertials(x[left:end], mass, com, r'<xacro:ee-inertials name="hand"\s*/>',
                                   count=1) + x[end:]
    fl = x.index(LEFT_IF, x.index('</link>', left))
    fe = x.index('</xacro:if>', fl)
    return x[:fl] + _swap_inertials(x[fl:fe], DUMMY, (0, 0, 0),
                                    r'<xacro:ee-inertials name="(?:left|right)_finger"\s*/>') + x[fe:]


def _new_arm_xacro(ax):
    """The arm xacro: the LEFT arm's link7 (inside its own xacro:if; the
    right arm's link7 comes from the same file and is left alone)."""
    l7 = ax.index('<link name="${prefix}link7">', ax.index(LEFT_IF))
    l7e = ax.index('</link>', l7)
    return ax[:l7] + _swap_inertials(ax[l7:l7e], DUMMY, (0, 0, 0.075),
                                     r'<xacro:openarm-inertials name="link7"[^>]*/>',
                                     count=1) + ax[l7e:]


def _write_files(arm, mass, com):
    """The body after joint 7 into the hand's inertial (openarm.urdf, the
    hand xacro), link7 and the fingers to DUMMY. All three new texts are made
    first, so a file that does not look as expected (ValueError) leaves all
    three as they were; each is backed up to old_version/ and written only if
    it changes. The paths are read from the module at the call (a test points
    them all at copies -- the arm xacro's used to be fixed in here)."""
    new = {}
    for path, change in ((URDF, lambda t: _new_urdf(t, arm, mass, com)),
                         (XACRO, lambda t: _new_hand_xacro(t, mass, com)),
                         (ARM_XACRO, _new_arm_xacro)):
        with open(path) as handle:
            old = handle.read()
        new[path] = (old, change(old))
    stamp = time.strftime('%Y%m%d-%H%M%S')
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for path, (old, text) in new.items():
        if text == old:
            continue
        keep = os.path.join(BACKUP_DIR, f'{os.path.basename(path)}.before_weigh_{stamp}')
        if not os.path.exists(keep):            # two writes in one second: keep the first
            with open(keep, 'w') as handle:
                handle.write(old)
        with open(path, 'w') as handle:
            handle.write(text)


def _rebuild():
    """openarm_description into native/install. '' or what went wrong."""
    try:
        done = subprocess.run([os.path.join(WS, 'native', 'build_ws.sh'), '--packages-select',
                               'openarm_description'], capture_output=True, text=True,
                              timeout=REBUILD_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if done.returncode == 0:
        return ''
    lines = (done.stderr or done.stdout or '').strip().splitlines()
    return lines[-1] if lines else f'exit code {done.returncode}'


def write_weight(arm, result, rebuild=True):
    """APPLY WEIGHT: the fitted body after joint 7 into the URDF and the
    xacros, then openarm_description rebuilt. (ok, written, text): ok False =
    refused (nothing written, text says why); written True = the motors'
    description changes, so terminal 1 must be restarted."""
    if arm != 'left':
        return False, False, 'only the left arm (its gripper was replaced) is written'
    why = check(result)
    if why:
        return False, False, f'not written: {why} - weigh again'
    mass, com = float(result['hand_mass']), np.asarray(result['hand_com'], float)
    was, was_com, was_l7 = _urdf_after_j7(arm)
    text = (f'everything after joint 7 weighs {mass:.3f} kg, centre of mass '
            f'({1000 * com[0]:.0f}, {1000 * com[1]:.0f}, {1000 * com[2]:.0f}) mm in the hand frame')
    same = (was_l7 < 0.01 and abs(was - mass) < SAME_KG
            and np.linalg.norm(was_com - com) < SAME_M)
    if same and _built():
        return True, False, text + ' - already in the URDF and built: nothing to change'
    if not same:
        _write_files(arm, mass, com)
        text += f' (the URDF had {was:.3f}: link7 + hand + fingers) - written'
    else:
        text += ' - in the URDF already, not built yet'
    if not rebuild:
        return True, True, (text + '; now run native/build_ws.sh --packages-select '
                            'openarm_description and restart terminal 1')
    err = _rebuild()
    if err:
        return True, True, (text + f'; REBUILD FAILED ({err}): run native/build_ws.sh '
                            '--packages-select openarm_description, then restart terminal 1')
    return True, True, (text + '; openarm_description rebuilt - restart terminal 1 '
                        '(native/run_launch_everything.sh) so the motors use it')


def apply(arm, result, rebuild=True):
    """APPLY WEIGHT as a message (write_weight's text)."""
    return write_weight(arm, result, rebuild)[2]


class PayloadMixin:
    """Part of click_to_move.App: WEIGH TOOL."""

    def weigh_tool(self, arm='left'):
        """Still postures, the motors' torques there, the fit. Nothing written
        until APPLY WEIGHT."""
        self.say(f'WEIGH TOOL: {arm} arm, nothing held on the cup - choosing postures...')
        if not self.ensure_pick_mode():
            return
        self.abort.clear()
        try:
            got = self._weigh_measure(arm, 'WEIGH TOOL')
        except KeyboardInterrupt:
            self.say('WEIGH TOOL: aborted; nothing written', True)
            return
        if isinstance(got, str):
            self.say(f'WEIGH TOOL: {got}', True)
            return
        result = self._weigh_fit(arm, *got)
        self.say(f'WEIGH TOOL: everything after joint 7 weighs {result["hand_mass"]:.3f} kg (the '
                 f'URDF says {result["link7_mass_was"] + result["hand_mass_was"]:.3f}); gravity '
                 f'torques now '
                 f'{result["rms_after_nm"]:.2f} Nm off (were {result["rms_before_nm"]:.2f}). '
                 f'APPLY WEIGHT writes it into the URDF')

    def weigh_and_apply(self, arm='left'):
        """The auto calibration's first step (2026-11-16): WEIGH TOOL and APPLY
        WEIGHT in one, inside the job that runs now (no job of its own), the
        arm EMPTY. (ok, text, restart_needed); restart_needed when the
        motors' description was changed (they read it only at launch). STOP
        raises KeyboardInterrupt to the caller, nothing written."""
        if arm != 'left':
            return False, 'only the left arm (its gripper was replaced) is weighed', False
        self.say(f'WEIGH TOOL: {arm} arm, nothing held on the cup - choosing postures...')
        got = self._weigh_measure(arm, 'WEIGH TOOL')
        if isinstance(got, str):
            return False, f'not weighed: {got}', False
        result = self._weigh_fit(arm, *got)
        ok, written, text = write_weight(arm, result, rebuild=True)
        text = (f'{text} (torques {result["rms_before_nm"]:.2f} -> '
                f'{result["rms_after_nm"]:.2f} Nm off)')
        if written:
            self._reload_kinematics(arm)
        return ok, text, written

    def _weigh_measure(self, arm, label):
        """The postures, the moves, the torques both ways -- inside the job
        that runs now. (qs, taus) arrays, or why not (str). STOP raises
        KeyboardInterrupt; the arm goes back to pre_pick either way."""
        node = self.node
        self.capture()
        if self.frame is None:
            return 'no camera frame'
        cands = self._weigh_postures(arm)
        if len(cands) < 8:
            return f'only {len(cands)} postures reachable - not enough'
        # spread out in joint space (each joint's load changes with all of them)
        chosen = [0]
        dist = np.linalg.norm(np.array(cands) - cands[0], axis=1)
        while len(chosen) < min(POSES, len(cands)):
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(np.array(cands) - cands[nxt], axis=1))
        was = self.calibrating                     # the auto calibration's own flag stays
        self.calibrating = True
        qs, taus = [], []
        try:
            for n, i in enumerate(chosen, start=1):
                if self.abort.is_set():
                    raise KeyboardInterrupt
                self.say(f'{label}: posture {n}/{len(chosen)} ({len(qs)} measured)')
                if not node.move_joints(arm, cands[i].tolist(), f'weigh {n}', via_home=False,
                                        attempts=1):
                    continue
                node.settle(arm, cands[i].tolist(), timeout=3.0)
                got = self._both_ways(arm)
                if got is not None:
                    qs.append(got[0])
                    taus.append(got[1])
        finally:
            self.calibrating = was
            self._back_to_pre_pick(arm)
        if len(qs) < 8:
            return f'only {len(qs)} postures measured - not enough'
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        np.savez(os.path.join(SAMPLES_DIR, f'{arm}_weigh_{time.strftime("%Y-%m-%dT%H-%M-%S")}.npz'),
                 q=np.array(qs), tau=np.array(taus))
        return np.array(qs), np.array(taus)

    def _weigh_postures(self, arm):
        """Up to 120 reachable postures in front of the camera (joints)."""
        node = self.node
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
        return cands

    def _weigh_fit(self, arm, qs, taus):
        """The fit, kept for APPLY WEIGHT (memory and payload_<arm>.yaml).
        The memory first: a yaml that cannot be written must not lose it."""
        result = fit(arm, qs, taus)
        self.payload_result = (arm, result)
        save_result(arm, result)
        return result

    def _reload_kinematics(self, arm):
        """The Python side re-reads the masses: the arm models (their sag uses
        gravity torques) and the node's cached chains (the held-weight check)."""
        from ctm import armcal
        self.arm_models = armcal.ArmModels()
        getattr(self.node, '_kin', {}).pop(arm, None)

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
        """APPLY WEIGHT: the last WEIGH TOOL (this run's, else the yaml)."""
        got = getattr(self, 'payload_result', None)
        if got is None:
            try:
                arm = 'left'
                with open(FILE.format(arm=arm)) as handle:
                    got = (arm, yaml.safe_load(handle))
            except (OSError, yaml.YAMLError):
                self.say('APPLY WEIGHT: weigh the tool first', True)
                return
        arm, result = got
        ok, written, text = write_weight(arm, result)
        if written:
            self._reload_kinematics(arm)
        self.say('APPLY WEIGHT: ' + text, not ok)
