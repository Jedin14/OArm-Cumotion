#!/usr/bin/env python3
"""Find where the camera is by fitting what it sees of the arm to the robot model.

Why this and not the fingertip: the first CALIBRATE picked one pixel per
pose -- "the fingertip" -- out of a 424x240 depth silhouette and solved the
camera pose from those pixels (solvePnP). On 2026-10-01 that left 6.5 px of
error with 3 of 12 points thrown out as outliers: one pixel per pose from a
blurry silhouette is not a precise enough measurement to calibrate from.

Here every pose contributes thousands of measurements. The depth camera sees
the gripper and forearm as a point cloud; the robot knows exactly where
those surfaces are -- the URDF's visual meshes, posed by the joint encoders
through tf. The camera pose is whatever rigid transform puts the observed
cloud onto the model surfaces (point-to-plane ICP over all poses at once).
It is a 3D fit, so it cannot trade camera height against tilt the way a 2D
fit at one depth can, and it calibrates exactly what a click uses: pixel ->
depth -> camera -> world.

Pieces:
    RobotSurface        points + normals sampled on each link's visual mesh
    arm_points()        the observed arm: depth nearer than a background frame
    visible()           model points the camera can actually see (z-buffer)
    fit_camera()        robust point-to-plane ICP over many poses
    depth_agreement()   per-pixel check of model vs depth, for the overlay
"""

import hashlib
import math
import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

WS = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(WS, 'native', 'cache')


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin_matrix(element):
    m = np.eye(4)
    o = element.find('origin') if element is not None else None
    if o is not None:
        m[:3, 3] = [float(v) for v in (o.get('xyz') or '0 0 0').split()]
        m[:3, :3] = rpy_matrix(*[float(v) for v in (o.get('rpy') or '0 0 0').split()])
    return m


def resolve(filename):
    path = re.sub(r'^package://openarm_description', os.path.join(WS, 'src', 'openarm_description'),
                  filename)
    return path.replace('file://', '')


class RobotSurface:
    """Surface samples of each arm link's visual mesh, in that link's frame.

    Visual, not collision, meshes: the collision ones are simplified hulls
    (link6 is 600 faces against 96k), and a calibration fitted to a hull
    would be off by however far the hull is from the real part.
    """

    LINKS = ['link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'left_finger', 'right_finger']

    def __init__(self, urdf_xml, arms=('left', 'right'), points_per_m2=400000.0):
        import trimesh                               # only needed here
        root = ET.fromstring(urdf_xml)
        links = {link.get('name'): link for link in root.findall('link')}
        wanted = [f'openarm_{arm}_{short}' for arm in arms for short in self.LINKS]
        # Keyed on the arm links only: APPLY changes the camera joint in the
        # URDF, and that must not throw away a minute of mesh sampling.
        key = hashlib.sha1((''.join(ET.tostring(links[n]).decode() for n in wanted if n in links)
                            + repr(points_per_m2)).encode()).hexdigest()[:12]
        cache = os.path.join(CACHE_DIR, f'robot_surface_{key}.npz')
        self.links = {}                              # name -> (points (N,3), normals (N,3))
        if os.path.exists(cache):
            data = np.load(cache)
            for name in data.files:
                if name.endswith('__p'):
                    base = name[:-3]
                    self.links[base] = (data[name], data[base + '__n'])
            return
        meshes = {}
        rng = np.random.default_rng(0)
        for arm in arms:
            for short in self.LINKS:
                name = f'openarm_{arm}_{short}'
                link = links.get(name)
                if link is None:
                    continue
                pts, nrm = [], []
                for visual in link.findall('visual'):
                    mesh = visual.find('geometry/mesh')
                    if mesh is None:
                        continue
                    path = resolve(mesh.get('filename'))
                    if path not in meshes:
                        meshes[path] = trimesh.load(path, force='mesh')
                    g = meshes[path].copy()
                    scale = np.array([float(v) for v in (mesh.get('scale') or '1 1 1').split()])
                    g.apply_scale(scale)             # a negative axis mirrors (right finger)
                    g.apply_transform(origin_matrix(visual))
                    n = max(200, int(g.area * points_per_m2))
                    np.random.seed(int(rng.integers(1 << 30)))   # repeatable samples
                    p, faces = trimesh.sample.sample_surface(g, n)
                    pts.append(np.asarray(p))
                    nrm.append(np.asarray(g.face_normals[faces]))
                if pts:
                    self.links[name] = (np.concatenate(pts).astype(np.float32),
                                        np.concatenate(nrm).astype(np.float32))
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, **{f'{k}__p': v[0] for k, v in self.links.items()},
                            **{f'{k}__n': v[1] for k, v in self.links.items()})

    def posed(self, link_poses):
        """World points and normals for {link name: 4x4 world<-link}."""
        pts, nrm = [], []
        for name, pose in link_poses.items():
            if name not in self.links or pose is None:
                continue
            p, n = self.links[name]
            pts.append(p @ pose[:3, :3].T + pose[:3, 3])
            nrm.append(n @ pose[:3, :3].T)
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.concatenate(pts), np.concatenate(nrm)


def intrinsics(info):
    k = np.array(info.k, dtype=np.float64).reshape(3, 3)
    d = np.array(info.d, dtype=np.float64) if len(info.d) else np.zeros(5)
    return k, d


def project(points_cam, info):
    """Optical-frame points -> pixels (with lens distortion) and depth."""
    k, d = intrinsics(info)
    z = points_cam[:, 2]
    px, _ = cv2.projectPoints(points_cam.reshape(-1, 1, 3).astype(np.float64),
                              np.zeros(3), np.zeros(3), k, d)
    return px.reshape(-1, 2), z


def arm_points(depth, background, info, stride=2, min_change=0.02, max_range=1.2,
               min_range=0.25):
    """Observed arm points in the optical frame: what is now nearer than in
    the background frame (the scene without the arm there).

    Flying pixels -- the D455 smears depth across silhouette edges, and at
    424x240 upsampled to 640x480 that smear is several pixels wide -- are
    dropped by a local-range test, because they lie between the arm and the
    wall and would pull the fit towards the camera.
    """
    k, d = intrinsics(info)
    # Nothing nearer than min_range: below the D455's minimum depth (~0.2 m
    # at 424x240) the readings are unreliable, and the forearm passes that
    # close to the camera at many calibration poses.
    valid = (depth > min_range) & (depth < max_range)
    changed = valid & ((background <= 0.1) | (background - depth > min_change))
    local_max = cv2.dilate(np.where(valid, depth, 0).astype(np.float32), np.ones((5, 5)))
    local_min = cv2.erode(np.where(valid, depth, 10).astype(np.float32), np.ones((5, 5)))
    smooth = (local_max - local_min) < 0.012
    mask = changed & smooth
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    sub = np.zeros_like(mask)
    sub[::stride, ::stride] = mask[::stride, ::stride]
    v, u = np.nonzero(sub)
    if not len(u):
        return np.zeros((0, 3)), mask
    rays = cv2.undistortPoints(np.stack([u, v], 1).astype(np.float64).reshape(-1, 1, 2),
                               k, d).reshape(-1, 2)
    z = depth[v, u].astype(np.float64)
    return np.column_stack([rays * z[:, None], z]), mask


def visible(points_w, normals_w, t_world_optical, info, shape, cell=3, tol=0.004):
    """Which model points the camera sees: inside the image and not behind
    another part of the model (a z-buffer over all samples -- see fit_camera
    for why the normals are not used)."""
    t_ow = np.linalg.inv(t_world_optical)
    pc = points_w @ t_ow[:3, :3].T + t_ow[:3, 3]
    keep = (pc[:, 2] > 0.1) & (np.abs(pc[:, 0]) < pc[:, 2]) & (np.abs(pc[:, 1]) < pc[:, 2])
    idx = np.flatnonzero(keep)
    if not len(idx):
        return idx
    px, z = project(pc[idx], info)
    h, w = shape
    inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    idx, px, z = idx[inside], px[inside], z[inside]
    cu = (px[:, 0] // cell).astype(int)
    cv_ = (px[:, 1] // cell).astype(int)
    zbuf = np.full(((h + cell - 1) // cell, (w + cell - 1) // cell), np.inf)
    np.minimum.at(zbuf, (cv_, cu), z)
    return idx[z <= zbuf[cv_, cu] + tol]


def _so3_exp(w):
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


COARSE_TO_FINE = (0.10, 0.10, 0.08, 0.08, 0.06, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03,
                  0.025, 0.02, 0.02, 0.015, 0.015, 0.012, 0.012, 0.01, 0.01, 0.008,
                  0.008, 0.008, 0.006, 0.006, 0.006, 0.006)


def fit_camera(observations, t_world_optical, info, shape, log=None, schedule=COARSE_TO_FINE,
               lock_world_x=False, lock_translation=False):
    """Robust point-to-plane ICP for the camera pose over many arm poses.

    observations: [(observed optical-frame points (N,3), model world points
    (M,3), model world normals (M,3))], one per pose. The transform is
    updated about the camera centre, so a rotation does not drag the camera
    sideways. Each step: visible model points, nearest-neighbour matches
    under the current distance gate, Huber-weighted point-to-plane normal
    equations (with a small point-to-point term so flat faces cannot slide).

    Returns (t_world_optical, stats).
    """
    from scipy.spatial import cKDTree
    t = t_world_optical.copy()
    trees = {}
    stats = {}
    for it, gate in enumerate(schedule):
        rows, rhs, wts = [], [], []
        dists = []
        centre = t[:3, 3].copy()
        for i, (obs, mp, mn) in enumerate(observations):
            if i not in trees:
                # Every surface sample, front and back. The visual meshes are
                # not watertight and carry internal parts, so their normals
                # are only ~60% outward -- no use for deciding what faces the
                # camera (doing so biased the fit by 7 mm / 2 deg in
                # simulation). Matching observed points to their nearest
                # model point needs no visibility: the surface the camera
                # sees is the nearest one.
                trees[i] = (cKDTree(mp), mp, mn) if len(mp) >= 50 else None
            if trees[i] is None or not len(obs):
                continue
            tree, vp, vn = trees[i]
            q = obs @ t[:3, :3].T + t[:3, 3]
            dist, j = tree.query(q, distance_upper_bound=gate)
            ok = np.isfinite(dist)
            if not ok.any():
                continue
            q, m, n = q[ok], vp[j[ok]], vn[j[ok]]
            dists.append(dist[ok])
            r_pl = np.einsum('ij,ij->i', q - m, n)
            arm = q - centre
            j_pl = np.hstack([np.cross(arm, n), n])
            huber = 0.004
            w_pl = np.where(np.abs(r_pl) < huber, 1.0, huber / np.maximum(np.abs(r_pl), 1e-9))
            rows.append(j_pl)
            rhs.append(-r_pl)
            wts.append(w_pl)
            # weak point-to-point term, three rows per pair
            for axis in range(3):
                e = np.zeros(3)
                e[axis] = 1.0
                rows.append(np.hstack([np.cross(arm, e), np.tile(e, (len(q), 1))]))
                rhs.append(-(q - m)[:, axis])
                # Point-to-point dominates while the gate is wide (it pulls in
                # from far off: a 65 mm / 9 deg start got stuck at 15 mm with
                # point-to-plane alone), and only steadies once it is tight.
                wts.append(np.full(len(q), 1.0 if gate > 0.03 else 0.05))
        if not rows:
            raise RuntimeError('no observed points matched the robot model')
        a = np.vstack(rows)
        b = np.concatenate(rhs)
        w = np.concatenate(wts)
        if lock_world_x:
            # the camera's forward position is fixed (see fit_camera_multistart)
            a[:, 3] = 0.0
        if lock_translation:
            # position fixed (a measured mount); only the orientation is fitted
            a[:, 3:] = 0.0
        aw = a * w[:, None]
        x = np.linalg.solve(aw.T @ a + 1e-9 * np.eye(6), aw.T @ b)
        rot = _so3_exp(x[:3])
        new = np.eye(4)
        new[:3, :3] = rot @ t[:3, :3]
        new[:3, 3] = rot @ (t[:3, 3] - centre) + centre + x[3:]
        t = new
        d = np.concatenate(dists)
        stats = {'iteration': it, 'gate_mm': gate * 1000, 'matched': int(len(d)),
                 'rms_mm': float(np.sqrt(np.mean(d ** 2)) * 1000),
                 'step_mm': float(np.linalg.norm(x[3:]) * 1000),
                 'step_deg': float(math.degrees(np.linalg.norm(x[:3])))}
        if log:
            log(stats)
    # final residuals at a 10 mm gate, point-to-plane
    residuals, total = [], 0
    for i, (obs, mp, mn) in enumerate(observations):
        if trees.get(i) is None or not len(obs):
            continue
        tree, vp, vn = trees[i]
        q = obs @ t[:3, :3].T + t[:3, 3]
        dist, j = tree.query(q, distance_upper_bound=0.01)
        ok = np.isfinite(dist)
        total += len(q)
        residuals.append(np.abs(np.einsum('ij,ij->i', q[ok] - vp[j[ok]], vn[j[ok]])))
    res = np.concatenate(residuals) if residuals else np.zeros(1)
    stats['plane_rms_mm'] = float(np.sqrt(np.mean(res ** 2)) * 1000)
    stats['inlier_fraction'] = float(len(res) / max(1, total))
    return t, stats


def fit_camera_multistart(observations, t_world_optical, info, shape, log=None):
    """fit_camera from the given pose and from offsets around it; best wins.

    ICP is local. The starting pose is a tape measurement and could be off
    by several centimetres and degrees, in exactly the up/down-and-tilt
    direction that is hardest to see from one depth, so the fit is also
    started from the camera 4 cm higher and lower and tilted 6 deg up and
    down. The result with the most points within 6 mm, then the lowest
    point-to-plane rms, is kept.
    """
    starts = [(0, 0, 0, 0)]
    for dz in (-0.04, 0.04):
        for pitch in (-6.0, 0.0, 6.0):
            starts.append((dz, pitch, 0, 0))
    starts += [(0, -6.0, 0, 0), (0, 6.0, 0, 0), (0, 0, 4.0, 0), (0, 0, -4.0, 0)]
    best = None
    for dz, pitch, yaw, _ in starts:
        t0 = t_world_optical.copy()
        rot = rpy_matrix(0.0, math.radians(pitch), math.radians(yaw))
        t0[:3, :3] = rot @ t0[:3, :3]
        t0[2, 3] += dz
        try:
            t, st = fit_camera(observations, t0, info, shape)
        except RuntimeError:
            continue
        score = (round(st['inlier_fraction'], 3), -st['plane_rms_mm'])
        if log:
            log({'start': (dz, pitch, yaw), **st})
        if best is None or score > best[0]:
            best = (score, t, st)
    if best is None:
        raise RuntimeError('no start converged')
    return best[1], best[2]


def alignment_error(observations, t_world_optical, gate=0.05):
    """How far the camera's view of the arm sits from the robot model.

    Median |point-to-plane distance| of observed points to their nearest
    model point (mm), and the fraction within 5 mm. Points with nothing
    within `gate` are left out (clutter).
    """
    from scipy.spatial import cKDTree
    res = []
    for obs, mp, mn in observations:
        if not len(obs) or not len(mp):
            continue
        q = obs @ t_world_optical[:3, :3].T + t_world_optical[:3, 3]
        dist, j = cKDTree(mp).query(q, distance_upper_bound=gate)
        ok = np.isfinite(dist)
        res.append(np.abs(np.einsum('ij,ij->i', q[ok] - mp[j[ok]], mn[j[ok]])))
    if not res:
        return float('nan'), 0.0
    r = np.concatenate(res)
    return float(np.median(r) * 1000), float(np.mean(r < 0.005))


def near_model(obs, t_world_optical, model_points, radius=0.12):
    """Observed points within `radius` of the model under the current
    camera estimate: drops people and anything else that moved."""
    from scipy.spatial import cKDTree
    if not len(obs) or not len(model_points):
        return obs
    q = obs @ t_world_optical[:3, :3].T + t_world_optical[:3, 3]
    dist, _ = cKDTree(model_points).query(q, distance_upper_bound=radius)
    return obs[np.isfinite(dist)]


def depth_agreement(depth, points_w, normals_w, t_world_optical, info, tol=0.015):
    """Project the visible model; per point, does the camera's depth agree?

    Returns (pixels (N,2), agree (N,) bool, observed (N,) bool). A point whose
    pixel has no depth, or is behind something else, counts as unobserved.
    """
    idx = visible(points_w, normals_w, t_world_optical, info, depth.shape)
    if not len(idx):
        return np.zeros((0, 2)), np.zeros(0, bool), np.zeros(0, bool)
    t_ow = np.linalg.inv(t_world_optical)
    pc = points_w[idx] @ t_ow[:3, :3].T + t_ow[:3, 3]
    px, z = project(pc, info)
    u = np.clip(px[:, 0].astype(int), 0, depth.shape[1] - 1)
    v = np.clip(px[:, 1].astype(int), 0, depth.shape[0] - 1)
    d = depth[v, u]
    observed = (d > 0.1) & (d > z - 0.05)            # not hidden behind something
    return px, observed & (np.abs(d - z) < tol), observed


def save_run(captured, t_world_optical, t_optical_screw, info, shape):
    """Everything a fit needs, saved so a run can be refitted offline
    (refit()) instead of the arms doing it all again."""
    import datetime
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, 'calibration_run_%s.npz'
                        % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    data = {'t_world_optical': t_world_optical, 't_optical_screw': t_optical_screw,
            'k': np.array(info.k, dtype=np.float64), 'd': np.array(info.d, dtype=np.float64),
            'shape': np.array(shape), 'n': np.array(len(captured))}
    for i, (obs, mp, mn, depth) in enumerate(captured):
        data[f'obs{i}'] = obs.astype(np.float32)
        data[f'mp{i}'] = mp.astype(np.float32)
        data[f'mn{i}'] = mn.astype(np.float32)
        data[f'depth{i}'] = depth.astype(np.float16)
    np.savez_compressed(path, **data)
    return path
