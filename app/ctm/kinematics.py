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

import re
import xml.etree.ElementTree as ET

import numpy as np

from ctm.common import OPENARM_URDF

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
        root = ET.parse(urdf_path or OPENARM_URDF).getroot()
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


# -- batched rows and straight lines, for the fast reach check (2026-11-17) ----
# ik_batch asks one pose from many seeds; the reach check asks hundreds of poses
# (every way in, every roll) at once, and then whether the straight lines in and
# onto the box stay solvable all the way -- the question MoveIt's Cartesian path
# answers one 50 ms timeout per failed waypoint (55.7 s of a 12 min run went on
# failed lines, 2026-11-16). Plain URDF, joint limits, no collisions: these only
# throw out what cannot work; MoveIt still checks what is flown.


def _chain(arm):
    """The arm's steps folded for speed: [(fixed rotation 3x3, fixed shift 3,
    joint axis 3)] per joint and the fixed (rotation, shift) after the last --
    the plain URDF, as frames_batch with geom None. Kept on the Arm."""
    got = arm.__dict__.get('_folded')
    if got is not None:
        return got
    joints, pending = [], np.eye(4)
    for fixed, axis in arm.steps:
        pending = pending @ fixed
        if axis is not None:
            joints.append((pending[:3, :3].copy(), pending[:3, 3].copy(),
                           np.asarray(axis, float) / np.linalg.norm(axis)))
            pending = np.eye(4)
    arm._folded = (joints, (pending[:3, :3].copy(), pending[:3, 3].copy()))
    return arm._folded


def _fk_jacobian(arm, q):
    """hand_tcp poses (N, 4, 4) and geometric Jacobians (N, 6, 7) at q (N, 7):
    frames_batch's chain, the rotation kept as three column arrays and each
    joint turned by Rodrigues on them -- about 4x faster for the thousands of
    rows the reach check asks at once (13 -> ~3 ms for 8832 rows)."""
    q = np.atleast_2d(np.asarray(q, float))
    n = len(q)
    joints, (tail_r, tail_p) = _chain(arm)
    cols = [np.zeros((n, 3)) for _ in range(3)]          # the columns of R (world <- link)
    for i in range(3):
        cols[i][:, i] = 1.0
    p = np.zeros((n, 3))
    origins, axes = [], []
    for k, (fr, fp, axis) in enumerate(joints):
        p = p + cols[0] * fp[0] + cols[1] * fp[1] + cols[2] * fp[2]
        cols = [cols[0] * fr[0, j] + cols[1] * fr[1, j] + cols[2] * fr[2, j] for j in range(3)]
        a = cols[0] * axis[0] + cols[1] * axis[1] + cols[2] * axis[2]
        origins.append(p)
        axes.append(a)
        c_, s_ = np.cos(q[:, k])[:, None], np.sin(q[:, k])[:, None]
        m = int(np.argmax(np.abs(axis)))
        if abs(abs(axis[m]) - 1.0) < 1e-12:
            # about a link axis (every OpenArm joint): only the other two columns turn
            s_ = s_ if axis[m] > 0 else -s_
            i, j = (m + 1) % 3, (m + 2) % 3
            cols[i], cols[j] = cols[i] * c_ + cols[j] * s_, cols[j] * c_ - cols[i] * s_
        else:
            # R @ rot(axis, q): each column c -> c cos + (a x c) sin + a (a.c)(1 - cos)
            cols = [c * c_ + np.cross(a, c) * s_
                    + a * (np.einsum('ni,ni->n', a, c)[:, None] * (1 - c_)) for c in cols]
    p = p + cols[0] * tail_p[0] + cols[1] * tail_p[1] + cols[2] * tail_p[2]
    cols = [cols[0] * tail_r[0, j] + cols[1] * tail_r[1, j] + cols[2] * tail_r[2, j]
            for j in range(3)]
    t = np.zeros((n, 4, 4))
    t[:, :3, 0], t[:, :3, 1], t[:, :3, 2], t[:, :3, 3] = cols[0], cols[1], cols[2], p
    t[:, 3, 3] = 1.0
    jac = np.empty((n, 6, 7))
    for k, (origin, a) in enumerate(zip(origins, axes)):
        jac[:, :3, k] = np.cross(a, p - origin)
        jac[:, 3:, k] = a
    return t, jac


def _pose_error(t, targets):
    """(N, 6) error of poses t against targets: position (m) and the small-angle
    rotation vector (rad); and (N,) position / rotation error norms -- the
    rotation as ik_batch measures it (the matrices' Frobenius distance)."""
    p_err = targets[:, :3, 3] - t[:, :3, 3]
    r_rel = targets[:, :3, :3] @ np.transpose(t[:, :3, :3], (0, 2, 1))
    w_err = 0.5 * np.stack([r_rel[:, 2, 1] - r_rel[:, 1, 2], r_rel[:, 0, 2] - r_rel[:, 2, 0],
                            r_rel[:, 1, 0] - r_rel[:, 0, 1]], axis=1)
    pos = np.linalg.norm(p_err, axis=1)
    rot = np.linalg.norm(t[:, :3, :3] - targets[:, :3, :3], axis=(1, 2))
    return np.concatenate([p_err, w_err], axis=1), pos, rot


def _dls(jac, err, damping=1e-4, cap=None):
    """Damped least-squares joint step (N, 7), each row's largest joint step
    capped to `cap` rad (a far seed otherwise overshoots into a joint limit).
    J^T (J J^T + damping)^-1 e -- the same step as ik_batch's (J^T J +
    damping)^-1 J^T e, a 6x6 solve instead of 7x7 (a third faster)."""
    jt = np.transpose(jac, (0, 2, 1))
    dq = (jt @ np.linalg.solve(jac @ jt + damping * np.eye(6), err[:, :, None]))[:, :, 0]
    if cap is not None:
        big = np.max(np.abs(dq), axis=1, keepdims=True)
        dq = dq * np.minimum(1.0, cap / np.maximum(big, 1e-12))
    return dq


def joint_margin(q, limits):
    """(N, 7) how far each joint is from its nearer limit, rad."""
    lo, hi = np.array(limits, float).T
    q = np.asarray(q, float)
    return np.minimum(q - lo, hi - q)


def ik_batch_rows(arm, targets, seeds, limits, iters=30, tol=1e-4, rot_tol=1e-3,
                  all_seeds=False, cap=0.4, give_up=None):
    """Plain-URDF IK for many poses at once: row i of `targets` (N, 4, 4) is
    solved from every seed of row i of `seeds` ((N, S, 7), or (S, 7): the same
    seeds for every row) -- damped least squares, joints clipped to `limits`
    every iteration (as KDL keeps them), rows dropped from the batch once they
    have converged (pos < tol m, rotation < rot_tol).

    Returns the best seed's answer per row: (q (N, 7), ok (N,), err (N,)) --
    best = converged with the most room to its joint limits (a posture with
    room to move), else the smallest error. all_seeds=True: every seed's
    answer instead, (q (N, S, 7), ok (N, S), err (N, S)). err = position error
    (m) + rotation error.

    give_up: [(iterations, metres)] -- a seed still that far off after that many
    iterations is dropped (left unconverged): most random seeds end on a joint
    limit and never converge, and iterating them is most of the time. The
    reach check uses ((8, 0.05), (16, 0.01)): 85 instead of 215 ms for 384
    poses x 23 seeds, the same poses solved but 0-5 of ~200 (2026-11-17)."""
    targets = np.asarray(targets, float).reshape(-1, 4, 4)
    n = len(targets)
    seeds = np.asarray(seeds, float)
    if seeds.ndim == 2:
        seeds = np.broadcast_to(seeds[None], (n,) + seeds.shape)
    s = seeds.shape[1]
    lo, hi = np.array(limits, float).T
    q = np.clip(seeds.reshape(n * s, 7), lo, hi)
    goal = np.repeat(targets, s, axis=0)
    pos = np.full(n * s, np.inf)
    rot = np.full(n * s, np.inf)
    active = np.arange(n * s)
    for it in range(iters + 1):
        if not active.size:
            break
        t, jac = _fk_jacobian(arm, q[active])
        err, p, r = _pose_error(t, goal[active])
        pos[active], rot[active] = p, r
        if it == iters:
            break
        left = ~((p < tol) & (r < rot_tol))
        for after, far in (give_up or ()):
            if it >= after:
                left &= p <= far
        active, jac, err = active[left], jac[left], err[left]
        if active.size:
            q[active] = np.clip(q[active] + _dls(jac, err, cap=cap), lo, hi)
    ok = (pos < tol) & (rot < rot_tol)
    err = pos + rot
    if all_seeds:
        return q.reshape(n, s, 7), ok.reshape(n, s), err.reshape(n, s)
    q, ok, err = q.reshape(n, s, 7), ok.reshape(n, s), err.reshape(n, s)
    room = joint_margin(q.reshape(-1, 7), limits).min(axis=1).reshape(n, s)
    score = np.where(ok, room, -1e3 - err)
    best = np.argmax(score, axis=1)
    rows = np.arange(n)
    return q[rows, best], ok[rows, best], err[rows, best]


def _log_rot(r):
    """(N, 3) rotation vectors of rotations r (N, 3, 3)."""
    cos = np.clip((np.trace(r, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cos)
    w = np.stack([r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]],
                 axis=1)
    s = np.sin(ang)
    out = np.where((s > 1e-9)[:, None], w * (ang / np.maximum(2.0 * s, 1e-12))[:, None], 0.5 * w)
    # (near 180 deg the skew part vanishes: the axis from the symmetric part)
    for i in np.flatnonzero((s <= 1e-4) & (ang > 3.0)):
        b = (r[i] + np.eye(3)) / 2.0
        j = int(np.argmax(np.diag(b)))
        axis = b[:, j] / np.sqrt(max(b[j, j], 1e-12))
        out[i] = axis / np.linalg.norm(axis) * ang[i]
    return out


def _exp_rot(w):
    """(N, 3, 3) rotations of rotation vectors w (N, 3)."""
    a = np.linalg.norm(w, axis=1)
    k = w / np.maximum(a, 1e-12)[:, None]
    kx = np.zeros((len(w), 3, 3))
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    kx[:, 1, 0], kx[:, 2, 0], kx[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s_, c_ = np.sin(a)[:, None, None], np.cos(a)[:, None, None]
    return np.eye(3) + s_ * kx + (1 - c_) * (kx @ kx)


def line_walk(arm, q0, T0, T1, step=0.002, limits=None, max_dq_per_mm=0.03, iters=4,
              tol=1e-4, rot_tol=1e-3):
    """Walk hand_tcp in a straight line from T0 to T1 ((N, 4, 4) or one (4, 4)
    for every row), row i starting from the joints q0[i] (N, 7), which should
    put it at T0: every `step` m one warm-started damped-least-squares solve
    (up to `iters` iterations, from the step before), joints clipped to their
    limits, as MoveIt's Cartesian path tracks its line with KDL. A row fails at
    the first step that does not converge or that moves any joint more than
    max_dq_per_mm rad per mm of the line -- the app's JUMP, 0.15 rad per 5 mm
    step (robot.py): a branch flip or a joint run onto its limit.

    Returns (ok (N,), q_end (N, 7), margin (N,), path (K + 1, N, 7)): q_end the
    last good joints (the end for an ok row), margin the least room any joint
    had to its limit along the good part (rad), path every step's joints (NaN
    from where a row failed)."""
    if limits is None:
        raise ValueError('line_walk needs the joint limits')
    q = np.array(q0, float).reshape(-1, 7)
    n = len(q)
    T0 = np.broadcast_to(np.asarray(T0, float), (n, 4, 4))
    T1 = np.broadcast_to(np.asarray(T1, float), (n, 4, 4))
    lo, hi = np.array(limits, float).T
    q = np.clip(q, lo, hi)
    p0, p1 = T0[:, :3, 3], T1[:, :3, 3]
    r0 = T0[:, :3, :3]
    w = _log_rot(np.transpose(r0, (0, 2, 1)) @ T1[:, :3, :3])   # in T0's frame
    length = np.linalg.norm(p1 - p0, axis=1)
    turn = np.linalg.norm(w, axis=1)
    # one step count for all rows (a shorter row takes shorter steps); a turn in
    # place goes in 0.5 deg steps
    k_all = int(max(1, np.ceil(max(float(np.max(length, initial=0.0)) / step,
                                   float(np.max(turn, initial=0.0)) / np.radians(0.5)))))
    step_mm = np.maximum(length / k_all * 1000.0, step * 1000.0 * (length < 1e-9))
    path = np.full((k_all + 1, n, 7), np.nan)
    # the start must be where it says
    t, _j = _fk_jacobian(arm, q)
    _e, p, r = _pose_error(t, np.array(T0))
    alive = (p < 10 * tol) & (r < 10 * rot_tol)
    path[0, alive] = q[alive]
    margin = np.where(alive, joint_margin(q, limits).min(axis=1), -np.inf)
    for k in range(1, k_all + 1):
        idx = np.flatnonzero(alive)
        if not idx.size:
            break
        f = k / k_all
        goal = np.empty((len(idx), 4, 4))
        goal[:, 3] = (0.0, 0.0, 0.0, 1.0)
        goal[:, :3, 3] = p0[idx] + f * (p1[idx] - p0[idx])
        goal[:, :3, :3] = r0[idx] @ _exp_rot(f * w[idx])
        qa = q[idx].copy()
        for it in range(iters + 1):
            t, jac = _fk_jacobian(arm, qa)
            err, p, r = _pose_error(t, goal)
            conv = (p < tol) & (r < rot_tol)
            if it == iters or conv.all():
                break
            dq = _dls(jac, err, cap=0.15)
            dq[conv] = 0.0
            qa = np.clip(qa + dq, lo, hi)
        jump = np.max(np.abs(qa - q[idx]), axis=1)
        good = conv & (jump <= max_dq_per_mm * step_mm[idx] + 1e-12)
        alive[idx[~good]] = False
        g = idx[good]
        q[g] = qa[good]
        path[k, g] = qa[good]
        margin[g] = np.minimum(margin[g], joint_margin(qa[good], limits).min(axis=1))
    return alive, q, margin, path
