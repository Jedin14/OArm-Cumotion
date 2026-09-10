"""Forward kinematics and posture choice, straight off the robot description.

No ROS and no planner: this reads the URDF text, walks the joint tree, and
solves for arm postures numerically. It exists so the orchestrator can *choose*
which of an arm's many solutions to use, which neither cuMotion nor
``/compute_ik`` will do -- both hand back the first solution they find.

Why that matters, measured on this robot. The cycle asks for a tool pose:
position above the object, gripper pointing straight down, jaws at the object's
yaw. That is six constraints on a seven-joint arm, and at the object's actual
position it is nearly infeasible. Sampling 400 random seeds:

    tool position + point down + jaw yaw   124/400 solved,
                                           *all 124* within 0.10 rad of a
                                           joint limit, best margin 0.007 rad

There is no comfortable posture. The one the robot found had joint3 0.013 rad
and joint5 0.018 rad from their stops -- and a Cartesian path cannot continue
once a joint it needs is against a stop, which is why straight-line descents
kept returning ``fraction=0.0`` for the last stretch and falling back to curved
hops.

Partitioning at the wrist changes that. joint7 is a *pitch* (axis ``0 1 0``)
carrying the whole hand: the tool sits 180.1 mm from joint7's centre, so that
one joint swings the tool through a 180 mm arc. Ask instead for

    joint7's centre at the right point, and link6 oriented so that joint7
    *can* tilt the hand to vertical

-- which leaves joint7 out of the solve and drops one orientation constraint:

    wrist position + tiltable link6         223/400 solved,
                                            43 of them clear of every limit,
                                            best margin 0.489 rad

and the resulting posture, with joint7 then set to -82.2 deg, puts the tool
0.3 mm from the target pointing exactly straight down, with 0.137 rad of
margin on its worst joint -- ten times what the robot managed on its own.

So the approach is planned for the wrist, joint7 is tilted afterwards, and the
descent starts from a posture that has room to move.
"""

import math
import xml.etree.ElementTree as ET

ARM_DOF = 7


def _need_numpy():
    try:
        import numpy as np
    except ImportError as exc:                            # pragma: no cover
        raise RuntimeError(
            'arm_kinematics needs numpy; install it or set '
            'approach_frame:=tool to use the planner-only path') from exc
    return np


def rpy_matrix(roll, pitch, yaw):
    """Fixed-axis roll-pitch-yaw, the URDF convention."""
    np = _need_numpy()
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_matrix(axis, angle):
    """Rotation by `angle` about `axis`, Rodrigues."""
    np = _need_numpy()
    a = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(a)
    if norm == 0:
        return np.eye(3)
    a = a / norm
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def principal_rotation(kind, angle):
    """Rotation about +x (0), +y (1) or +z (2), filled directly.

    The general Rodrigues form in axis_matrix costs about ten numpy calls;
    this costs one array build. Worth having because every joint on this arm
    turns about a principal axis of its own frame, and the rotation is
    rebuilt inside pose() -- which the numerical Jacobian calls eight times
    per iteration, eighty iterations per seed, forty-eight seeds per
    orientation probed.
    """
    np = _need_numpy()
    c, s = math.cos(angle), math.sin(angle)
    if kind == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if kind == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def principal_axis(axis):
    """(which principal axis this is, whether it points backwards) or None.

    None for a genuinely oblique axis, which then takes the general path.
    """
    for index in range(3):
        others = [abs(axis[other]) for other in range(3) if other != index]
        if abs(abs(axis[index]) - 1.0) < 1e-9 and max(others) < 1e-9:
            return index, axis[index] < 0.0
    return None


def quat_matrix(quat):
    """Rotation matrix from (x, y, z, w)."""
    np = _need_numpy()
    x, y, z, w = quat
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_error(have, want):
    """Rotation vector taking `have` to `want`, length = angle in radians."""
    np = _need_numpy()
    E = want.T @ have
    angle = math.acos(max(-1.0, min(1.0, (np.trace(E) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    return axis * (angle / (2.0 * math.sin(angle)))


class ArmChain:
    """One arm of the description, with the pieces needed to pick a posture."""

    def __init__(self, urdf_text, arm):
        self.arm = arm
        self.joint_names = [f'openarm_{arm}_joint{i}' for i in range(1, 8)]
        self.tool_link = f'openarm_{arm}_hand_tcp'
        # joint7's rotation centre. The tool hangs off it, so this is the point
        # the approach is planned for -- see the module docstring.
        self.wrist_link = f'openarm_{arm}_link7'
        # The last link before joint7. Its y-axis *is* joint7's axis, so
        # whether the hand can be tilted to vertical is a statement about this
        # link's orientation.
        self.forearm_link = f'openarm_{arm}_link6'
        self._parse(urdf_text)

    # -- description ------------------------------------------------------

    def _parse(self, urdf_text):
        np = _need_numpy()
        self.joints = {}
        self._by_child = {}
        for joint in ET.fromstring(urdf_text).findall('joint'):
            name = joint.get('name')
            child = joint.find('child')
            parent = joint.find('parent')
            if child is None or parent is None:
                continue
            origin = joint.find('origin')
            axis = joint.find('axis')
            entry = {
                'name': name,
                'type': joint.get('type'),
                'parent': parent.get('link'),
                'child': child.get('link'),
                'xyz': [float(v) for v in (origin.get('xyz', '0 0 0')
                                           if origin is not None
                                           else '0 0 0').split()],
                'rpy': [float(v) for v in (origin.get('rpy', '0 0 0')
                                           if origin is not None
                                           else '0 0 0').split()],
                'axis': [float(v) for v in (axis.get('xyz', '0 0 1')
                                            if axis is not None
                                            else '0 0 1').split()],
            }
            limit = joint.find('limit')
            if limit is not None and limit.get('lower') is not None:
                entry['limits'] = (float(limit.get('lower')),
                                   float(limit.get('upper')))
            self.joints[name] = entry
            self._by_child[entry['child']] = entry

        missing = [j for j in self.joint_names if j not in self.joints]
        if missing:
            raise ValueError(f'the description has no {missing[0]}')
        self.limits = np.array([self.joints[j].get('limits', (-math.pi, math.pi))
                                for j in self.joint_names])
        # Chains are fixed by the description, so resolve them once.
        self._chains = {link: self._chain_to(link)
                        for link in (self.tool_link, self.wrist_link,
                                     self.forearm_link)}
        # Filled on first use by pose(), which is where the shape of a chain
        # actually costs anything.
        self._compiled = {}
        self.wrist_lever = self._measure_lever()

    def _chain_to(self, link):
        chain, cur, guard = [], link, 0
        while cur in self._by_child and guard < 64:
            entry = self._by_child[cur]
            chain.append(entry)
            cur = entry['parent']
            guard += 1
        chain.reverse()
        return chain

    def _measure_lever(self):
        """How far the tool sits from joint7's centre, metres."""
        np = _need_numpy()
        zero = np.zeros(ARM_DOF)
        return float(np.linalg.norm(self.pose(self.tool_link, zero)[:3, 3]
                                    - self.pose(self.wrist_link, zero)[:3, 3]))

    # -- kinematics -------------------------------------------------------

    def compiled(self, link):
        """The chain to `link` as fixed transforms with joints between them.

        T = F0 . R(qa) . F1 . R(qb) . ... . Fn, with every fixed piece built
        once here and runs of them multiplied together. A 28-entry chain
        with seven joints in it then costs seven rotations and eight matrix
        multiplies per pose instead of twenty-eight of each.

        Worth the machinery because pose() is the innermost function of the
        whole reach check: the numerical Jacobian calls it eight times per
        iteration, up to eighty iterations per seed, forty-eight seeds per
        orientation, ten orientations per pre-flight. Measured on this
        description, compiling took a pose from 89 to 24 us.
        """
        np = _need_numpy()
        steps, fixed = [], np.eye(4)
        for entry in (self._chains.get(link) or self._chain_to(link)):
            L = np.eye(4)
            L[:3, :3] = rpy_matrix(*entry['rpy'])
            L[:3, 3] = entry['xyz']
            fixed = fixed @ L
            if entry['type'] in ('revolute', 'continuous', 'prismatic'):
                index = (self.joint_names.index(entry['name'])
                         if entry['name'] in self.joint_names else None)
                steps.append({
                    'fixed': fixed,
                    'type': entry['type'],
                    'axis': np.asarray(entry['axis'], dtype=float),
                    'index': index,
                    'principal': principal_axis(entry['axis']),
                })
                fixed = np.eye(4)
        steps.append({'fixed': fixed, 'type': None})
        return steps

    def pose(self, link, q):
        """World 4x4 of `link` at joint vector `q` (joint1..joint7)."""
        steps = self._compiled.get(link)
        if steps is None:
            steps = self._compiled[link] = self.compiled(link)
        T = steps[0]['fixed'].copy()
        for i in range(len(steps) - 1):
            step = steps[i]
            index = step['index']
            value = float(q[index]) if index is not None else 0.0
            if step['type'] == 'prismatic':
                # A pure translation along the joint axis, expressed in the
                # frame the arm has reached: the rotation is untouched.
                T[:3, 3] = T[:3, 3] + T[:3, :3] @ (step['axis'] * value)
            else:
                principal = step['principal']
                if principal is None:
                    R = axis_matrix(step['axis'], value)
                else:
                    kind, flip = principal
                    R = principal_rotation(kind, -value if flip else value)
                # T @ [[R, 0], [0, 1]] touches the rotation block alone, so
                # this is a 3x3 multiply rather than a 4x4 one.
                T[:3, :3] = T[:3, :3] @ R
            # The next fixed piece, which is already one matrix however
            # many links it was built from.
            T = T @ steps[i + 1]['fixed']
        return T

    def _joint_frames(self, q):
        """World axis and origin of each arm joint, plus each link pose.

        One pass, because the Jacobians below all need the same thing: a
        revolute joint's contribution is built from its world axis and the
        position of its origin.
        """
        np = _need_numpy()
        angles = dict(zip(self.joint_names, [float(v) for v in q]))
        axes, origins, poses = {}, {}, {}
        T = np.eye(4)
        for entry in self._chains[self.tool_link]:
            L = np.eye(4)
            L[:3, :3] = rpy_matrix(*entry['rpy'])
            L[:3, 3] = entry['xyz']
            T = T @ L
            if entry['type'] in ('revolute', 'continuous'):
                axes[entry['name']] = T[:3, :3] @ np.asarray(entry['axis'],
                                                             dtype=float)
                origins[entry['name']] = T[:3, 3].copy()
                R = np.eye(4)
                R[:3, :3] = axis_matrix(entry['axis'],
                                        angles.get(entry['name'], 0.0))
                T = T @ R
            elif entry['type'] == 'prismatic':
                R = np.eye(4)
                R[:3, 3] = [v * angles.get(entry['name'], 0.0)
                            for v in entry['axis']]
                T = T @ R
            poses[entry['child']] = T.copy()
        return axes, origins, poses

    def margins(self, q):
        """Distance from each joint to its nearer limit, radians."""
        np = _need_numpy()
        q = np.asarray(q, dtype=float)
        return np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q)

    def worst_margin(self, q):
        return float(self.margins(q).min())

    def numeric_jacobian(self, residual, step=1e-6):
        """Finite-difference Jacobian of a residual closure."""
        np = _need_numpy()

        def jacobian(q):
            q = np.asarray(q, dtype=float)
            base = residual(q)
            Jac = np.zeros((len(base), ARM_DOF))
            for i in range(ARM_DOF):
                nudge = np.zeros(ARM_DOF)
                nudge[i] = step
                Jac[:, i] = (residual(q + nudge) - base) / step
            return Jac

        return jacobian

    # -- objectives -------------------------------------------------------
    #
    # Each returns a (residual, jacobian) pair of closures over q alone, so
    # solve() does not have to know what is being asked for.

    def objective_tool_pose(self, position, quat):
        """The full tool pose: position, and the gripper pointing as given.

        Six constraints on seven joints. This is what the cycle asks for
        today, and at the measured object position 67 of the 74 solutions
        found were within 0.10 rad of a joint limit -- so it matters a great
        deal *which* one is used, which is what roomiest() is for.
        """
        np = _need_numpy()
        want_p = np.asarray(position, dtype=float)
        want_R = quat_matrix(quat)

        def residual(q):
            M = self.pose(self.tool_link, q)
            return np.concatenate([M[:3, 3] - want_p,
                                   rotation_error(M[:3, :3], want_R)])

        # Numerical, deliberately. The position rows really are the geometric
        # Jacobian -- checked to 3e-7 against finite differences -- but the
        # orientation rows are not: a rotation-vector error does not
        # differentiate to the angular Jacobian anywhere except near zero, and
        # substituting one for the other made this objective converge 0 times
        # in 48 where finite differences converge about one seed in five. Six
        # extra forward-kinematics passes per iteration is the price of it
        # working at all, and this runs once per pick.
        jacobian = self.numeric_jacobian(residual)
        return residual, jacobian

    def objective_wrist(self, wrist_position, jaw_axis=None):
        """Wrist at a point, hand able to tilt to the approach axis, and --
        given `jaw_axis` -- the jaws lined up across the object.

        joint7 appears in none of it, which is the point: the wrist point lies
        on that joint's axis, and link6 is upstream of it.

        The jaw constraint is free. joint7 turns *about* link6's y-axis, so it
        cannot move it -- measured: swinging joint7 over its whole range
        leaves link6's y and the hand's y identical to four decimals, while
        the tool's z swings from +0.765 to -0.988. The fingers slide along the
        hand's y (finger_joint1's axis is 0 -1 0), so the direction the jaws
        close in is fixed by joints 1-6 and nothing else.

        Which means leaving it out was a mistake, not a saving: with only
        "link6's y is horizontal" the azimuth was whatever the solver landed
        on, so the gripper closed at an arbitrary angle to the object. On a
        screwdriver that is a full straight-line descent onto the right point
        followed by the jaws closing on nothing.

        Constraining it to +/-jaw_axis rather than +jaw_axis because a
        parallel gripper closes on the same two faces either way round.
        """
        np = _need_numpy()
        want = np.asarray(wrist_position, dtype=float)
        axis = None
        if jaw_axis is not None:
            axis = np.asarray(jaw_axis, dtype=float)
            axis = axis / np.linalg.norm(axis)

        def residual(q):
            wrist = self.pose(self.wrist_link, q)[:3, 3]
            forearm = self.pose(self.forearm_link, q)[:3, :3]
            y6 = forearm[:, 1]
            if axis is None:
                # Just horizontal: enough for joint7 to reach vertical.
                return np.concatenate([wrist - want, [y6[2]]])
            sign = 1.0 if float(y6 @ axis) >= 0.0 else -1.0
            return np.concatenate([wrist - want, y6 - sign * axis])

        def jacobian(q):
            axes, origins, poses = self._joint_frames(q)
            wrist = poses[self.wrist_link][:3, 3]
            y6 = poses[self.forearm_link][:3, 1]
            rows = 4 if axis is None else 6
            Jac = np.zeros((rows, ARM_DOF))
            for i, name in enumerate(self.joint_names):
                if i >= ARM_DOF - 1:
                    continue            # joint7 moves neither term
                a = axes[name]
                Jac[:3, i] = np.cross(a, wrist - origins[name])
                turn = np.cross(a, y6)
                if axis is None:
                    Jac[3, i] = turn[2]
                else:
                    Jac[3:, i] = turn
            return Jac

        return residual, jacobian

    # -- solving ----------------------------------------------------------

    def solve(self, residual, jacobian, seed, iterations=80, tolerance=1e-5,
              max_step=0.2, damping=1e-6):
        """Damped least squares, clipped into the joint limits.

        Clipped rather than penalised: a posture outside the limits is not a
        posture, and restarting from another seed is cheap.
        """
        np = _need_numpy()
        q = np.clip(np.asarray(seed, dtype=float).copy(),
                    self.limits[:, 0], self.limits[:, 1])
        error = residual(q)
        for _ in range(iterations):
            if np.linalg.norm(error) < tolerance:
                break
            Jac = jacobian(q)
            JT = Jac.T
            try:
                step = JT @ np.linalg.solve(
                    Jac @ JT + damping * np.eye(Jac.shape[0]), -error)
            except np.linalg.LinAlgError:                 # pragma: no cover
                break
            q = np.clip(q + np.clip(step, -max_step, max_step),
                        self.limits[:, 0], self.limits[:, 1])
            error = residual(q)
        return q, float(np.linalg.norm(error))

    def postures(self, residual, jacobian, seeds=48, iterations=80,
                 tolerance=1e-4, extra_seeds=(), rng_seed=0, keep=None,
                 distinct=0.15):
        """Converged postures, most joint-limit headroom first.

        Near-duplicates are merged -- two seeds landing in the same basin are
        the same posture, and a caller trying alternatives wants genuinely
        different arm configurations, not eight copies of one.

        Deterministic: same call, same list. A solver that answers differently
        each time is how a descent ends up reconfiguring 157 degrees between
        two waypoints 20 mm apart.

        Returns (list of (joints, margin), converged, tried).
        """
        np = _need_numpy()
        rng = np.random.default_rng(rng_seed)
        seeds_used = [np.asarray(s, dtype=float) for s in extra_seeds]
        seeds_used += [rng.uniform(self.limits[:, 0], self.limits[:, 1])
                       for _ in range(max(0, seeds))]
        found, converged = [], 0
        for seed in seeds_used:
            q, err = self.solve(residual, jacobian, seed, iterations=iterations)
            if err > tolerance:
                continue
            converged += 1
            found.append((q, self.worst_margin(q)))
        found.sort(key=lambda pair: -pair[1])
        kept = []
        for q, margin in found:
            if any(float(np.abs(q - other).max()) < distinct
                   for other, _ in kept):
                continue
            kept.append((q, margin))
            if keep is not None and len(kept) >= keep:
                break
        return kept, converged, len(seeds_used)

    def roomiest(self, residual, jacobian, seeds=48, iterations=80,
                 tolerance=1e-4, extra_seeds=(), rng_seed=0):
        """Just the best of postures(). (joints or None, margin, converged,
        tried)."""
        kept, converged, tried = self.postures(
            residual, jacobian, seeds=seeds, iterations=iterations,
            tolerance=tolerance, extra_seeds=extra_seeds, rng_seed=rng_seed,
            keep=1)
        if not kept:
            return None, -math.inf, converged, tried
        return kept[0][0], kept[0][1], converged, tried

    # -- what the orchestrator asks for -----------------------------------

    def wrist_point_for(self, tool_position, quat):
        """Where joint7's centre has to be to put the tool at that pose.

        The tool sits `wrist_lever` along the tool frame's z, so the wrist is
        that far back along it. For a gripper pointing straight down this is
        simply `wrist_lever` higher.
        """
        np = _need_numpy()
        R = quat_matrix(quat)
        return np.asarray(tool_position, dtype=float) - R[:, 2] * self.wrist_lever

    def jaw_axis_for(self, quat):
        """World direction the jaws close along for that grasp.

        The tool frame's y: finger_joint1's axis is 0 -1 0 in the hand frame,
        so this is the line the fingers travel. Projected flat, because it is
        compared against link6's y-axis, which the tilt requires to be
        horizontal.
        """
        np = _need_numpy()
        y = quat_matrix(quat)[:, 1].copy()
        y[2] = 0.0
        norm = float(np.linalg.norm(y))
        if norm < 1e-9:
            return None                  # jaws close vertically: no azimuth
        return y / norm

    def approach_posture(self, tool_position, quat, seeds=48, extra_seeds=(),
                         rng_seed=0, jaw_axis=None):
        """Roomiest posture putting the *wrist* where that tool pose needs it.

        joint7 is not part of the objective, so it comes back wherever the
        solve left it; tilt_for_direction() then points the hand.
        """
        residual, jacobian = self.objective_wrist(
            self.wrist_point_for(tool_position, quat), jaw_axis=jaw_axis)
        return self.roomiest(residual, jacobian, seeds=seeds,
                             extra_seeds=extra_seeds, rng_seed=rng_seed)

    def tool_posture(self, tool_position, quat, seeds=48, extra_seeds=(),
                     rng_seed=0):
        """Roomiest posture achieving the full tool pose, joint7 included."""
        residual, jacobian = self.objective_tool_pose(tool_position, quat)
        return self.roomiest(residual, jacobian, seeds=seeds,
                             extra_seeds=extra_seeds, rng_seed=rng_seed)

    def approach_candidates(self, tool_position, quat, wrist=True, seeds=48,
                            extra_seeds=(), rng_seed=0, keep=8,
                            jaw_axis=None, align_tolerance=0.05):
        """Usable postures for that approach, best first.

        In wrist mode the tilt is worked out *here*, and postures joint7
        cannot tilt into line are dropped before the ranking. That order
        matters more than it looks. Ranking by the untilted posture's
        headroom and filtering afterwards discards the good answers: measured
        at the transit point above the object, the seven roomiest solutions
        all needed joint7 beyond its own +/-90 degree limit and landed 15 to
        25 degrees short of vertical, while the first *usable* one sat at rank
        eight with 0.240 rad of headroom. Keeping the top six by the wrong
        score kept six unusable postures and threw the usable ones away.

        So the score is the headroom of the posture that will actually be
        commanded, tilt included.

        Returns (list of (joints, tilt, margin), converged, tried), where
        `joints` is the solve's own seven values and `tilt` is what joint7
        should be set to (None in tool mode, where it is already solved).
        """
        approach = quat_matrix(quat)[:, 2]
        if wrist:
            residual, jacobian = self.objective_wrist(
                self.wrist_point_for(tool_position, quat), jaw_axis=jaw_axis)
        else:
            residual, jacobian = self.objective_tool_pose(tool_position, quat)
        found, converged, tried = self.postures(
            residual, jacobian, seeds=seeds, extra_seeds=extra_seeds,
            rng_seed=rng_seed, keep=None)

        usable = []
        for joints, margin in found:
            if not wrist:
                usable.append((list(joints), None, margin))
                continue
            tilt, error = self.tilt_for_direction(list(joints), approach)
            if error > align_tolerance:
                continue
            tilted = list(joints)
            tilted[ARM_DOF - 1] = tilt
            usable.append((list(joints), tilt, self.worst_margin(tilted)))
        usable.sort(key=lambda entry: -entry[2])
        return usable[:keep] if keep else usable, converged, tried

    def tilt_for_direction(self, q, direction=(0.0, 0.0, -1.0), samples=721):
        """joint7 that points the tool along `direction`, and how well.

        Scanned rather than solved: it is one bounded joint, the objective is
        not convex in it, and 721 samples of a 3.14 rad range costs nothing.
        Ties are broken by joint-limit headroom, so the tilt does not park
        joint7 against a stop for no reason.
        """
        np = _need_numpy()
        want = np.asarray(direction, dtype=float)
        want = want / np.linalg.norm(want)
        low, high = self.limits[ARM_DOF - 1]
        best, best_key = None, None
        trial = np.asarray(q, dtype=float).copy()
        for value in np.linspace(low, high, samples):
            trial[ARM_DOF - 1] = value
            z = self.pose(self.tool_link, trial)[:3, 2]
            error = float(np.arccos(max(-1.0, min(1.0, float(z @ want)))))
            # Round the alignment so that near-equal tilts are separated by
            # headroom instead of floating-point noise.
            key = (round(error, 4), -min(value - low, high - value))
            if best_key is None or key < best_key:
                best, best_key = float(value), key
        trial[ARM_DOF - 1] = best
        z = self.pose(self.tool_link, trial)[:3, 2]
        return best, float(np.arccos(max(-1.0, min(1.0, float(z @ want)))))


def chain_from_urdf(urdf_text, arm):
    """Convenience: None instead of raising, for callers that can carry on."""
    try:
        return ArmChain(urdf_text, arm)
    except (ET.ParseError, ValueError, RuntimeError):
        return None
