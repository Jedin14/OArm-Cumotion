"""The arm as the URDF describes it, plus what calibration adds to it.

Chain: world -> openarm_<arm>_hand_tcp, read from openarm.urdf (fixed and
revolute joints with their origins and axes, and every link's mass and
centre of mass). On top of the URDF, a calibrated arm has:

    base      6 numbers: a small rotation (rad, as a vector, about the
              shoulder) and shift (m) of the arm's base, relative to the
              camera's idea of the world
    offset    7 joint zero offsets (rad): the encoder reads q, the joint is
              at q + offset
    comply    7 joint compliances (rad per N m): under gravity each joint
              gives a little, by comply * (gravity torque on it) -- the
              elastic sag past the encoder (gearbox, belts). What the
              controller lets sag before the encoder (its gravity
              compensation is never perfect) the encoders already read.
    backlash  7 (rad): each gearbox's play, taken up on the side gravity
              pulls: backlash * tanh(tau / BACKLASH_TORQUE)
    geom      (7, 6): small errors in each joint's mounting -- rotation (3,
              rad, as a vector) and shift (3, m) of its origin in its parent
              link: the links' real lengths and axes. Joint 1's row stays zero
              (the base correction covers it).

So the real joint angle is q + offset + comply * tau + backlash *
tanh(tau / BACKLASH_TORQUE), tau the gravity torque, and the real tool pose
follows from the chain with those angles and mountings, behind the base
correction. With all of them zero this is exactly the URDF.
"""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from ctm.common import WS

G = np.array([0.0, 0.0, -9.81])
BACKLASH_TORQUE = 0.5           # N m: the play is taken up by about this much torque


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rotvec(v):
    """Rotation matrix of a rotation vector (axis * angle)."""
    v = np.asarray(v, float)
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * kx @ kx


def _origin(el):
    o = el.find('origin')
    xyz = [float(v) for v in (o.get('xyz') or '0 0 0').split()] if o is not None else [0, 0, 0]
    rpy = [float(v) for v in (o.get('rpy') or '0 0 0').split()] if o is not None else [0, 0, 0]
    m = np.eye(4)
    m[:3, :3] = _rpy(*rpy)
    m[:3, 3] = xyz
    return m


class Arm:
    """One arm's chain from the URDF: steps [(fixed 4x4, axis or None)],
    each step's child link mass/centre of mass, and the 7 joints' indices."""

    def __init__(self, arm, urdf_path=None):
        self.arm = arm
        root = ET.parse(urdf_path or os.path.join(WS, 'openarm.urdf')).getroot()
        joints = {j.find('child').get('link'): j for j in root.findall('joint')}
        links = {l.get('name'): l for l in root.findall('link')}
        chain = []
        link = f'openarm_{arm}_hand_tcp'
        while link in joints:
            chain.append(joints[link])
            link = joints[link].find('parent').get('link')
        chain.reverse()
        self.steps, self.masses, self.coms, self.names = [], [], [], []
        self.links = [j.find('child').get('link') for j in chain]     # each step's child
        for j in chain:
            axis = None
            if j.get('type') in ('revolute', 'continuous'):
                axis = np.array([float(v) for v in j.find('axis').get('xyz').split()])
                self.names.append(j.get('name'))
            self.steps.append((_origin(j), axis))
            child = links.get(j.find('child').get('link'))
            inertial = child.find('inertial') if child is not None else None
            if inertial is not None and inertial.find('mass') is not None:
                self.masses.append(float(inertial.find('mass').get('value')))
                self.coms.append(_origin(inertial)[:3, 3])
            else:
                self.masses.append(0.0)
                self.coms.append(np.zeros(3))
        # the fingers ride on the hand: add their mass there
        for name, lk in links.items():
            if re.match(rf'openarm_{arm}_(left|right)_finger$', name):
                inertial = lk.find('inertial')
                if inertial is not None and inertial.find('mass') is not None:
                    hand = [i for i, j in enumerate(chain)
                            if j.find('child').get('link') == f'openarm_{arm}_hand']
                    if hand:
                        self.masses[hand[0]] += float(inertial.find('mass').get('value'))
        assert len(self.names) == 7, self.names
        # the shoulder: joint 1's origin, fixed in the world -- base corrections
        # turn about it (about the world origin they were tangled with joint 1)
        self.shoulder = self.frames(np.zeros(7))[1][0][0]

    def frames(self, q, geom=None):
        """World poses of every step's child frame, and of each joint (its
        frame and axis in world), for joint angles q (7) and joint mounting
        errors geom ((7, 6) or None)."""
        t = np.eye(4)
        out, joints = [], []
        k = 0
        for fixed, axis in self.steps:
            t = t @ fixed
            if axis is not None and geom is not None and np.any(geom[k]):
                err = np.eye(4)
                err[:3, :3] = rotvec(geom[k][:3])
                err[:3, 3] = geom[k][3:]
                t = t @ err
            if axis is not None:
                joints.append((t[:3, 3].copy(), t[:3, :3] @ axis))
                rot = np.eye(4)
                rot[:3, :3] = rotvec(axis * q[k])
                t = t @ rot
                k += 1
            out.append(t.copy())
        return out, joints

    def gravity_torque(self, q):
        """Torque gravity puts on each of the 7 joints (N m), at angles q."""
        frames, joints = self.frames(q)
        tau = np.zeros(7)
        coms = [f[:3, :3] @ c + f[:3, 3] for f, c in zip(frames, self.coms)]
        # which step each joint sits in, so only links beyond it count
        step_of = [i for i, (_f, a) in enumerate(self.steps) if a is not None]
        for k, (origin, axis) in enumerate(joints):
            for i in range(step_of[k], len(self.steps)):
                if self.masses[i] > 0:
                    tau[k] += axis @ np.cross(coms[i] - origin, self.masses[i] * G)
        return tau

    def tcp(self, q, params=None, dirs=None):
        """4x4 world <- hand_tcp for encoder angles q, with calibration params
        (dict: base (6), offset (7), comply (7), and optionally backlash (7),
        geom (7, 6), hysteresis (7) with dirs (7): the direction each joint
        last moved in) or the plain URDF if None."""
        q = np.asarray(q, float)
        if params is None:
            return self.frames(q)[0][-1]
        if dirs is not None and 'hysteresis' in params:
            q = q + np.asarray(params['hysteresis']) * np.asarray(dirs, float)
        tau = self.gravity_torque(q + np.asarray(params['offset']))
        real = q + np.asarray(params['offset']) + np.asarray(params['comply']) * tau
        if 'backlash' in params:
            real = real + np.asarray(params['backlash']) * np.tanh(tau / BACKLASH_TORQUE)
        t = self.frames(real, params.get('geom'))[0][-1]
        base = np.eye(4)
        base[:3, :3] = rotvec(params['base'][:3])
        base[:3, 3] = self.shoulder - base[:3, :3] @ self.shoulder + params['base'][3:]
        return base @ t


    def ik(self, target, q0, params=None, limits=None, dirs=None):
        """Encoder angles putting hand_tcp at `target` (4x4) for this model,
        by least squares from q0. (A plain numeric solver: for calibration
        and tests, not for planning.)"""
        from scipy.optimize import least_squares

        def res(q):
            t = self.tcp(q, params, dirs)
            # the whole matrix: the skew part alone reads zero at 180 deg off
            return np.r_[(t[:3, 3] - target[:3, 3]) * 10.0,
                         (t[:3, :3] - target[:3, :3]).ravel()]
        lo, hi = (np.array(limits).T if limits is not None else (-np.inf, np.inf))
        sol = least_squares(res, np.clip(q0, lo + 1e-6, hi - 1e-6) if limits is not None else q0,
                            bounds=(lo, hi))
        # position error (m) and orientation error (rad), as one number
        return sol.x, float(np.linalg.norm(sol.fun[:3]) / 10.0 + np.linalg.norm(sol.fun[3:]))


def _rot_batch(axis, angles):
    """(N, 3, 3) rotations about one unit axis by N angles."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s_, c_ = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3) + s_ * k + (1 - c_) * (k @ k)


def frames_batch(arm, q, geom=None):
    """Arm.frames for N postures at once: ([(N, 4, 4)] per step,
    [(origin (N, 3), axis (N, 3))] per joint)."""
    q = np.atleast_2d(np.asarray(q, float))
    n = len(q)
    t = np.repeat(np.eye(4)[None], n, axis=0)
    out, joints = [], []
    k = 0
    for fixed, axis in arm.steps:
        t = t @ fixed
        if axis is not None and geom is not None and np.any(geom[k]):
            err = np.eye(4)
            err[:3, :3] = rotvec(geom[k][:3])
            err[:3, 3] = geom[k][3:]
            t = t @ err
        if axis is not None:
            joints.append((t[:, :3, 3].copy(), t[:, :3, :3] @ axis))
            rot = np.repeat(np.eye(4)[None], n, axis=0)
            rot[:, :3, :3] = _rot_batch(axis, q[:, k])
            t = t @ rot
            k += 1
        out.append(t)
    return out, joints


def gravity_torque_batch(arm, q):
    """Arm.gravity_torque for N postures: (N, 7)."""
    frames, joints = frames_batch(arm, q)
    coms = [f[:, :3, :3] @ c + f[:, :3, 3] for f, c in zip(frames, arm.coms)]
    step_of = [i for i, (_f, a) in enumerate(arm.steps) if a is not None]
    tau = np.zeros((len(frames[0]), 7))
    for k, (origin, axis) in enumerate(joints):
        for i in range(step_of[k], len(arm.steps)):
            if arm.masses[i] > 0:
                tau[:, k] += np.einsum('ni,ni->n', axis,
                                       np.cross(coms[i] - origin, arm.masses[i] * G))
    return tau


def tcp_batch(arm, q, params=None, dirs=None):
    """Arm.tcp for N postures: (N, 4, 4). dirs (N, 7): the direction each
    joint last moved in (+1 / -1 / 0), for params['hysteresis'] -- gear play
    taken up by the motion, not by gravity."""
    q = np.atleast_2d(np.asarray(q, float))
    if params is None:
        return frames_batch(arm, q)[0][-1]
    if dirs is not None and 'hysteresis' in params:
        q = q + np.asarray(params['hysteresis']) * dirs
    offset = np.asarray(params['offset'])
    tau = gravity_torque_batch(arm, q + offset)
    real = q + offset + np.asarray(params['comply']) * tau
    if 'backlash' in params:
        real = real + np.asarray(params['backlash']) * np.tanh(tau / BACKLASH_TORQUE)
    t = frames_batch(arm, real, params.get('geom'))[0][-1]
    base = np.eye(4)
    base[:3, :3] = rotvec(params['base'][:3])
    base[:3, 3] = arm.shoulder - base[:3, :3] @ arm.shoulder + params['base'][3:]
    return base @ t


def ik_batch(arm, target, seeds, limits, iters=25, tol=1e-4):
    """Plain-URDF IK for hand_tcp at `target` (4x4) from many starting
    postures at once: damped least squares with the exact (geometric)
    Jacobian, all seeds in one batch -- milliseconds, so the planner can ask
    "can the arm get there at all?" before any slow, collision-aware IK.
    Returns the solutions (k, 7), distinct by 0.2 rad, best first."""
    q = np.array(seeds, float)
    lo, hi = np.array(limits, float).T
    q = np.clip(q, lo, hi)
    pt, rt = target[:3, 3], target[:3, :3]
    for _ in range(iters):
        frames, joints = frames_batch(arm, q)
        t = frames[-1]
        p_err = pt - t[:, :3, 3]
        r_rel = rt @ np.transpose(t[:, :3, :3], (0, 2, 1))       # target * current^T
        w_err = 0.5 * np.stack([r_rel[:, 2, 1] - r_rel[:, 1, 2], r_rel[:, 0, 2] - r_rel[:, 2, 0],
                                r_rel[:, 1, 0] - r_rel[:, 0, 1]], axis=1)
        err = np.concatenate([p_err, w_err], axis=1)                # (n, 6)
        jac = np.zeros((len(q), 6, 7))
        for k, (origin, axis) in enumerate(joints):
            jac[:, :3, k] = np.cross(axis, t[:, :3, 3] - origin)
            jac[:, 3:, k] = axis
        jt = np.transpose(jac, (0, 2, 1))
        dq = np.linalg.solve(jt @ jac + 1e-4 * np.eye(7), jt @ err[:, :, None])[:, :, 0]
        q = np.clip(q + dq, lo, hi)
    frames, _j = frames_batch(arm, q)
    t = frames[-1]
    pos = np.linalg.norm(pt - t[:, :3, 3], axis=1)
    rot = np.linalg.norm(t[:, :3, :3] - rt, axis=(1, 2))
    good = np.flatnonzero((pos < tol) & (rot < 1e-3))
    good = good[np.argsort(pos[good] + rot[good])]
    out = []
    for i in good:
        if all(np.max(np.abs(q[i] - o)) > 0.2 for o in out):
            out.append(q[i])
    return np.array(out).reshape(-1, 7)


def directions(before, after, still=0.005):
    """+1 / -1 per joint: which way it moved from `before` to `after` (0 if
    it hardly moved -- its gear play stays where it was, unknown)."""
    step = np.asarray(after, float) - np.asarray(before, float)
    out = np.sign(step)
    out[np.abs(step) < still] = 0.0
    return out


def zero_params():
    return {'base': np.zeros(6), 'offset': np.zeros(7), 'comply': np.zeros(7),
            'backlash': np.zeros(7), 'geom': np.zeros((7, 6))}
