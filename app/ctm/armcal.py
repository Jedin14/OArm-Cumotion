"""CALIBRATE ARMS: one model of where each arm's tool really is, for any posture.

Once, with two pieces (ctm/board.py) on each gripper, each arm visits ~40
postures across the workspace -- near and far, high and low, level and
tilted down -- and at each the camera reads the strip's marker corners and
the arm its joint angles. One least-squares fit then finds, per arm
(ctm/kinematics.py):

    base      where the arm's base really is relative to the camera (6)
    offset    each joint's zero offset (7)
    comply    each joint's give under gravity (7): the sag, for any posture
    backlash  each gearbox's play, taken up the way gravity pulls (7)
    geom      each joint's mounting errors: the links' real lengths and axes
              (6 x joints 2-7)
    board     where each piece sits on the gripper (6 each)

by matching every detected corner's pixel with where the model puts it.
Held-out postures (not used in the fit) then say how good it is, in mm.

Afterwards no strip is needed: before each touch the model says where the
tip will really be for the chosen joints, and the target is shifted so that
it lands on the click (model_correction). Saved in arm_model.yaml; rerun
after moving the camera, an arm or a gripper.
"""

import datetime
import os

import cv2
import numpy as np
import yaml

from ctm import board as bd
from ctm.common import WS, plain
from ctm.kinematics import Arm, rotvec, tcp_batch

MODEL_FILE = os.path.join(WS, 'arm_model.yaml')
PRIOR_WEIGHT = float(os.environ.get('ARMCAL_PRIOR_WEIGHT', '0.4'))
TOUCH_MM = 2.0                  # a touch's contact depth: this many mm count as one pixel (its noise is ~2x a corner's)
PLANE_SIGMA = 0.01              # m: depth camera bias + contact push, taken up by the fit

# prior widths: how far each parameter may plausibly be from the URDF
PRIOR = {'base_rot': 0.03, 'base_pos': 0.03, 'offset': 0.05, 'comply': 0.02,
         'backlash': 0.01, 'geom_rot': 0.01, 'geom_pos': 0.003, 'hysteresis': 0.01,
         # The strip lies flat on the back of the rail: tilted out of that plane by
         # a degree or so at most, turned within it (about the tool axis) freely.
         # This is what lets the fit tell a tilted strip from a wrist-joint offset
         # -- they tilt the strip alike, but only the joint moves the tip, 10 cm
         # away: left free, the tip came out ~9 mm wrong (simulated, 2026-10-12).
         # Position: along and across the rail from where the camera first saw
         # the piece (only roughly, through the URDF), its height off the rail
         # from the rail itself.
         'board_tilt': 0.008, 'board_turn': 0.3, 'board_pos': 0.02, 'board_height': 0.005}
RAIL_Z = -0.08 + 0.004          # hand_tcp -> the back of the rail, along the tool axis


FACE_HALF = 0.025                # m: the rail's half thickness, nominally (the flaps' offset)
FACE_NORMALS = {'back': (0.0, 0.0, -1.0), 'top': (1.0, 0.0, 0.0), 'bottom': (-1.0, 0.0, 0.0)}


def nominal_piece_in_tcp(k):
    """Face k as nominally mounted (bd.face_of): the back face on the back of
    the rail, looking at the wrist (hand -z); the top and bottom faces folded
    over the rail's two long sides (hand +x and -x), reaching towards the
    fingers. RAIL_Y either side of the hand's middle, along the rail (hand
    y). Face frame: origin bottom-left, x along the rail, z out of the face."""
    side, kind = bd.face_of(k)
    y_c = bd.RAIL_Y if side == 0 else -bd.RAIL_Y
    x_b = np.array([0.0, 1.0, 0.0])
    z_b = np.array(FACE_NORMALS[kind])
    y_b = np.cross(z_b, x_b)
    m = np.eye(4)
    m[:3, :3] = np.column_stack([x_b, y_b, z_b])
    centre = (np.array([0.0, y_c, RAIL_Z]) if kind == 'back' else
              np.array([z_b[0] * FACE_HALF, y_c, RAIL_Z + bd.CUT[1] / 2]))
    m[:3, 3] = centre - m[:3, :3] @ np.array([bd.SIZE[0] / 2, bd.SIZE[1] / 2, 0.0])
    return m


def _snap_to_rail(m):
    """A face pose turned square to the hand: its normal onto the nearest of
    the three a face can have (hand -z, +x, -x), keeping its turn and place;
    a back face also moved onto the rail's plane."""
    half = np.array([bd.SIZE[0] / 2, bd.SIZE[1] / 2, 0.0])
    centre = m[:3, 3] + m[:3, :3] @ half
    n = m[:3, 2]
    want = max((np.array(v) for v in FACE_NORMALS.values()), key=lambda v: float(n @ v))
    axis = np.cross(n, want)
    s_, c_ = np.linalg.norm(axis), float(n @ want)
    turn = rotvec(axis / s_ * np.arctan2(s_, c_)) if s_ > 1e-9 else np.eye(3)
    out = np.eye(4)
    out[:3, :3] = turn @ m[:3, :3]
    if want[2] < -0.5:                       # the back face: on the rail
        centre[2] = RAIL_Z
    out[:3, 3] = centre - out[:3, :3] @ half
    return out


def face_sigma(init, prior):
    """Prior widths of a face's 6 pose numbers (rotation vector, shift; hand
    frame): turned freely about its own normal, tilted off it little; slid
    freely in its plane; off it little -- a back face (on the rail) 5 mm, a
    flap (the rail's thickness not known) 10 mm."""
    a = int(np.argmax(np.abs(init[:3, 2])))
    rot = [prior['board_tilt']] * 3
    rot[a] = prior['board_turn']
    pos = [prior['board_pos']] * 3
    pos[a] = prior['board_height'] if a == 2 else 0.01
    return rot + pos


def initial_pieces(arm, samples, t_wo, info):
    """Where each piece sits on the gripper, first guess: from the camera's
    view of it (each posture's pose of the piece, through the plain URDF),
    the median over postures, snapped flat onto the rail. So it does not
    matter which side or which way up a piece went on. Nominal for a piece
    never seen."""
    model = Arm(arm)
    k = np.array(info.k, float).reshape(3, 3)
    d = np.array(info.d, float) if len(info.d) else np.zeros(5)
    c3 = bd.corners_3d(arm)
    out = []
    for p_k in range(len(bd.pieces(arm))):
        rots, pos = [], []
        for q, px in samples:
            ids = [i for i in sorted(px) if bd.piece_of(arm, i) == p_k]
            if len(ids) < 2:
                continue
            ok, rv, tv = cv2.solvePnP(np.concatenate([c3[i] for i in ids]),
                                      np.concatenate([px[i] for i in ids]), k, d)
            if not ok:
                continue
            cam_piece = np.eye(4)
            cam_piece[:3, :3] = cv2.Rodrigues(rv)[0]
            cam_piece[:3, 3] = tv.ravel()
            m = np.linalg.inv(model.tcp(q)) @ t_wo @ cam_piece
            rots.append(m[:3, :3])
            pos.append(m[:3, 3])
        if not rots:
            out.append(nominal_piece_in_tcp(p_k))
            continue
        u, _s, vt = np.linalg.svd(np.sum(rots, axis=0))          # the mean rotation
        m = np.eye(4)
        m[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        m[:3, 3] = np.median(pos, axis=0)
        out.append(_snap_to_rail(m))
    return out


# Models CALIBRATE ARMS chooses between, per arm, by how well each predicts
# postures it was not fitted to: {name: (prior changes, uses directions)}.
# On the robot (2026-10-13) the left arm did best with gravity backlash
# alone, the right with link errors -- and the right wrist (joint 6) showed
# 35 mrad of play that depends on which way it last turned.
VARIANTS = {'backlash': ({'geom_rot': 1e-6, 'geom_pos': 1e-7}, False),
            'backlash+play': ({'geom_rot': 1e-6, 'geom_pos': 1e-7}, True),
            'links': ({'geom_rot': 0.003, 'geom_pos': 0.001}, False)}
N_ARM = 6 + 7 + 7 + 7 + 6 * 6    # base, offset, comply, backlash, geom of joints 2-7


def _unpack(x, n_pieces):
    geom = np.zeros((7, 6))
    geom[1:] = np.reshape(x[27:N_ARM], (6, 6))
    params = {'base': x[0:6], 'offset': x[6:13], 'comply': x[13:20], 'backlash': x[20:27],
              'geom': geom}
    return params, [x[N_ARM + 6 * k:N_ARM + 6 + 6 * k] for k in range(n_pieces)]


def arm_sigma(prior=None):
    """Prior widths of the arm's parameters, in _unpack's order."""
    p = PRIOR if prior is None else prior
    return np.concatenate([[p['base_rot']] * 3, [p['base_pos']] * 3,
                           [p['offset']] * 7, [p['comply']] * 7,
                           [p['backlash']] * 7,
                           ([p['geom_rot']] * 3 + [p['geom_pos']] * 3) * 6])


def _board_pose(v, init):
    m = np.eye(4)
    m[:3, :3] = rotvec(v[:3]) @ init[:3, :3]
    m[:3, 3] = init[:3, 3] + v[3:]
    return m


def predict(arm_model, x, q, t_ow, k, d, ids, inits):
    """Pixels where the model puts these marker corners (N*4, 2)."""
    params, vs = _unpack(x, len(inits))
    tcp = arm_model.tcp(q, params)
    pieces = [tcp @ _board_pose(v, init) for v, init in zip(vs, inits)]
    c3 = bd.corners_3d(arm_model.arm)
    world = np.concatenate([c3[i] @ pieces[bd.piece_of(arm_model.arm, i)][:3, :3].T
                            + pieces[bd.piece_of(arm_model.arm, i)][:3, 3] for i in ids])
    cam = world @ t_ow[:3, :3].T + t_ow[:3, 3]
    px, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d)
    return px.reshape(-1, 2)


ANCHOR_MM = 1.0                 # an anchor's tip: this many mm count as one corner pixel
ANCHORS_FILE = os.path.join(WS, 'armcal_anchors.yaml')


def load_anchors(arm=None):
    """[{arm, q, point, tip_offset, when}]: taught points -- the real tip was
    nudged exactly onto `point` (world, from the camera) with the joints at q."""
    try:
        with open(ANCHORS_FILE) as handle:
            rows = (yaml.safe_load(handle) or {}).get('anchors', [])
    except (OSError, yaml.YAMLError):
        rows = []
    return [r for r in rows if arm is None or r.get('arm') == arm]


def save_anchor(arm, q, point, tip_offset):
    rows = load_anchors()
    rows.append(plain({'arm': arm, 'q': [float(v) for v in q],
                       'point': [float(v) for v in point], 'tip_offset': float(tip_offset),
                       'when': datetime.datetime.now().isoformat(timespec='seconds')}))
    with open(ANCHORS_FILE, 'w') as handle:
        handle.write('# CAL ARMS anchors (calibrate.py ANCHOR): the real tip nudged exactly onto\n'
                     '# a camera-measured point, with these joints. REFIT uses them.\n'
                     + yaml.safe_dump({'anchors': rows}, sort_keys=False))
    return sum(1 for r in rows if r['arm'] == arm)


def fit(arm, samples, t_wo, info, log=print, touches=None, plane=None, tip_offset=0.0354,
        max_nfev=400, dirs=None, variant=None, anchors=None, tip_sigma=(0.003, 0.003, 0.010),
        shift_sigma=0.05):
    """samples: [(q (7), {marker id: (4, 2) pixels})] -- the strip postures.
    touches: [q (7)] at contact with `plane` (unit normal n, d: n . p = d,
    world, from the depth camera), the tool tilted both ways.

    The strip alone cannot tell a strip tilted about joint 7's axis from a
    joint-7 zero error (same strip, but only the joint moves the tip, ~18 cm
    away: up to 58 mm of tip per 0.3 px of strip -- simulated 2026-10-12).
    Touches with the tool tilted +/-25 deg settle it: a tip that is really
    off sideways reaches the plane early one way and late the other. A free
    plane offset takes up the depth camera's bias and the contact's push.

    variant: a VARIANTS name (its priors; directions only if it uses them).
    anchors: load_anchors() rows for this arm -- ground truth for where the
    TIP is (the camera sees only the pieces: where the tip sits relative to
    them is otherwise an assumption, and a wrong one put front touches 3 cm
    low, 2026-10-16).

    Returns (params dict, [each piece's 4x4 in hand_tcp], stats)."""
    from scipy.optimize import least_squares
    prior = dict(PRIOR)
    if variant is not None:
        prior.update(VARIANTS[variant][0])
        if not VARIANTS[variant][1]:
            dirs = None
    model = Arm(arm)
    k = np.array(info.k, float).reshape(3, 3)
    d = np.array(info.d, float) if len(info.d) else np.zeros(5)
    t_ow = np.linalg.inv(t_wo)
    inits = initial_pieces(arm, samples, t_wo, info)
    n_b = N_ARM + 6 * len(inits)
    n_h = 7 if dirs is not None else 0                   # direction play, after the pieces
    # with anchors: where the tip really is on the hand (3, hand_tcp frame) --
    # the camera fixes the hand, only the anchors can say where its tip is --
    # and a world shift (3) between the arm model (from the colour image) and
    # the clicked points (from depth)
    n_t = 6 if anchors else 0
    sigma = np.concatenate([arm_sigma(prior)]
                           + [face_sigma(init, prior) for init in inits])
    meas = [(np.asarray(q, float), sorted(px), np.concatenate([px[i] for i in sorted(px)]))
            for q, px in samples]
    touches = [np.asarray(q, float) for q in (touches or [])]
    n_plane, d_plane = (np.asarray(plane[0], float), float(plane[1])) if plane else (None, 0.0)
    tip = np.array([0.0, 0.0, tip_offset, 1.0])
    # The tip on the hand: a few mm sideways (the cup is centred between the
    # fingers), 1 cm along the tool (how far the bellows reach). Free (5 cm)
    # it took a 30 mm SIDEWAYS offset from 4 anchors all at the wall: level,
    # a hand offset and a world one look the same, and pointing down the
    # hand one put the tool 3 cm beside the box (2026-10-16). The world shift
    # takes what the anchors cannot place -- pointing down it is vertical,
    # and the contact takes that up. Anchors in other turns (TOP DOWN) tell
    # the two apart.
    # (the calibration box measures the tip itself: a looser tip prior, and
    # no colour/depth shift -- all of it is the colour camera, 2026-10-26)
    sigma = np.r_[sigma, [prior['hysteresis']] * n_h,
                  (list(tip_sigma) + [shift_sigma] * 3)[:n_t]]
    n_x = n_b + n_h + n_t + (1 if touches else 0)
    sigma_all = np.r_[sigma, [PLANE_SIGMA]] if touches else sigma
    dirs_all = np.asarray(dirs, float) if dirs is not None else None

    # every corner of every posture as one batch (one posture at a time
    # took 3-9 min a fit on the robot, 2026-10-13): its posture, its piece,
    # its point on the piece, in the order of `meas`
    c3 = bd.corners_3d(arm)
    q_all = np.array([q for q, _i, _o in meas])
    post = np.concatenate([[n] * 4 * len(ids) for n, (_q, ids, _o) in enumerate(meas)])
    piece = np.concatenate([[bd.piece_of(arm, i)] * 4 for _q, ids, _o in meas for i in ids])
    local = np.concatenate([c3[i] for _q, ids, _o in meas for i in ids])
    obs_all = np.concatenate([obs for _q, _i, obs in meas])
    q_touch = np.array(touches) if touches else None
    anchors = list(anchors or [])
    q_anchor = np.array([r['q'] for r in anchors]) if anchors else None
    p_anchor = np.array([r['point'] for r in anchors]) if anchors else None
    tip_anchor = (np.array([[0.0, 0.0, r['tip_offset'], 1.0] for r in anchors])
                  if anchors else None)

    def residuals(x):
        params, vs = _unpack(x, len(inits))
        if n_h:
            params['hysteresis'] = x[n_b:n_b + 7]
        tcp = tcp_batch(model, q_all, params, dirs_all)
        pieces = np.array([_board_pose(v, init) for v, init in zip(vs, inits)])
        t = tcp[post] @ pieces[piece]
        world = np.einsum('mij,mj->mi', t[:, :3, :3], local) + t[:, :3, 3]
        cam = world @ t_ow[:3, :3].T + t_ow[:3, 3]
        px, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d)
        r = [(px.reshape(-1, 2) - obs_all).ravel()]
        if anchors:
            t_a = tcp_batch(model, q_anchor, params)
            tip_d = np.r_[x[n_b + n_h:n_b + n_h + 3], 0.0]
            tips_a = (np.einsum('nij,nj->ni', t_a, tip_anchor + tip_d)[:, :3]
                      + x[n_b + n_h + 3:n_b + n_h + 6])
            r.append(((tips_a - p_anchor) * 1000.0 / ANCHOR_MM).ravel())
        if touches:
            tips = (tcp_batch(model, q_touch, params) @ tip)[:, :3]
            r.append((tips @ n_plane - d_plane - x[-1]) * 1000.0 / TOUCH_MM)  # mm -> "px"
        r.append(x / sigma_all * PRIOR_WEIGHT)                      # priors, in "pixels"
        # (anchors sit between the corners and the priors: their count is
        # small; n_px below still counts only the corners)
        return np.concatenate(r)

    sol = least_squares(residuals, np.zeros(n_x), loss='huber', f_scale=1.5,
                        x_scale=sigma_all, max_nfev=max_nfev)
    params, vs = _unpack(sol.x, len(inits))
    if n_h:
        params['hysteresis'] = sol.x[n_b:n_b + 7]
    if n_t:
        params['tip_delta'] = sol.x[n_b + n_h:n_b + n_h + 3]
        params['anchor_shift'] = sol.x[n_b + n_h + 3:n_b + n_h + 6]
    n_px = sum(len(obs) for _q, _i, obs in meas) * 2
    res = sol.fun[:n_px].reshape(-1, 2)
    stats = {'rms_px': float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))),
             'corners': int(len(res)), 'postures': len(samples), 'touches': len(touches)}
    log(f'{arm}: fit rms {stats["rms_px"]:.2f} px')
    n_a = 3 * len(anchors)
    if anchors:
        ar = sol.fun[n_px:n_px + n_a].reshape(-1, 3) * ANCHOR_MM          # mm
        stats['anchors'] = len(anchors)
        stats['anchor_mm'] = float(np.sqrt(np.mean(np.sum(ar ** 2, axis=1))))
    if touches:
        tr = sol.fun[n_px + n_a:n_px + n_a + len(touches)] * TOUCH_MM
        stats['touch_rms_mm'] = float(np.sqrt(np.mean(tr ** 2)))
        stats['plane_offset_mm'] = float(sol.x[-1] * 1000)
    return ({kk: np.asarray(v, float) for kk, v in params.items()},
            [_board_pose(v, init) for v, init in zip(vs, inits)], stats)


def posture_px(arm, samples, params, boards, t_wo, info, dirs=None):
    """Each posture's rms corner error (px) under a fitted model."""
    model = Arm(arm)
    k = np.array(info.k, float).reshape(3, 3)
    d = np.array(info.d, float) if len(info.d) else np.zeros(5)
    t_ow = np.linalg.inv(t_wo)
    c3 = bd.corners_3d(arm)
    out = []
    tcps = tcp_batch(model, np.array([q for q, _px in samples]), params,
                     np.asarray(dirs, float) if dirs is not None else None)
    for (q, px), tcp in zip(samples, tcps):
        errs = []
        for i, c in px.items():
            tb = tcp @ boards[bd.piece_of(arm, i)]
            cam = (c3[i] @ tb[:3, :3].T + tb[:3, 3]) @ t_ow[:3, :3].T + t_ow[:3, 3]
            p, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d)
            errs += list(np.linalg.norm(p.reshape(-1, 2) - c, axis=1))
        out.append(float(np.sqrt(np.mean(np.square(errs)))))
    return np.array(out)


def choose_model(arm, samples, dirs, t_wo, info, folds=3, log=print):
    """The VARIANTS model that best predicts postures it was not fitted to
    (each posture held out once, `folds` fits per model). Returns (name,
    {name: unseen rms px})."""
    idx = np.arange(len(samples))
    scores = {}
    for name in VARIANTS:
        errs = []
        for f in range(folds):
            tr, he = idx[idx % folds != f], idx[f::folds]
            p, b, _st = fit(arm, [samples[i] for i in tr], t_wo, info, log=lambda *_: None,
                            dirs=dirs[tr], variant=name, max_nfev=150)
            errs += list(posture_px(arm, [samples[i] for i in he], p, b, t_wo, info,
                                    dirs[he] if VARIANTS[name][1] else None))
        scores[name] = float(np.sqrt(np.mean(np.square(errs))))
        log(f'{arm}: model {name}: {scores[name]:.2f} px on unseen postures')
    return min(scores, key=scores.get), scores


def drop_outliers(arm, samples, params, boards, t_wo, info):
    """The samples without the postures the model cannot explain: more than
    3x the median error and over 3 px (a misread marker, an arm still
    swinging -- one such posture at 15.6 px against 1.5 px typical put the
    left arm's check at 7 px, 2026-10-13). Returns (kept, dropped count)."""
    err = posture_px(arm, samples, params, boards, t_wo, info)
    bad = err > max(3.0, 3.0 * float(np.median(err)))
    return [s_ for s_, b in zip(samples, bad) if not b], int(bad.sum()), ~bad


def tip_error_mm(arm, samples, params, boards, t_wo, info, tip_offset):
    """For held-out samples: how far the model (and the plain URDF) put the
    tip from where the pieces say it is, in mm (median, max) -- per posture
    the best-seen piece's pose solved, with the model's placement of it."""
    model = Arm(arm)
    k = np.array(info.k, float).reshape(3, 3)
    d = np.array(info.d, float) if len(info.d) else np.zeros(5)
    c3 = bd.corners_3d(arm)
    tip_tcp = np.array([0.0, 0.0, tip_offset, 1.0])
    out = {'model': [], 'urdf': []}
    for q, px in samples:
        by_piece = {}
        for i in px:
            by_piece.setdefault(bd.piece_of(arm, i), []).append(i)
        p_k, ids = max(by_piece.items(), key=lambda kv: len(kv[1]))
        ok, rv, tv = cv2.solvePnP(np.concatenate([c3[i] for i in ids]),
                                  np.concatenate([px[i] for i in ids]), k, d)
        if not ok:
            continue
        cam_board = np.eye(4)
        cam_board[:3, :3] = cv2.Rodrigues(rv)[0]
        cam_board[:3, 3] = tv.ravel()
        real_tcp = t_wo @ cam_board @ np.linalg.inv(boards[p_k])
        real_tip = (real_tcp @ tip_tcp)[:3]
        for name, p in (('model', params), ('urdf', None)):
            out[name].append(1000 * np.linalg.norm((model.tcp(q, p) @ tip_tcp)[:3] - real_tip))
    return {n: (float(np.median(v)), float(np.max(v))) if v else (None, None)
            for n, v in out.items()}


# -- the saved model ------------------------------------------------------------

def save_model(models):
    """models: {arm: (params, boards, stats)}; an arm not in it keeps what
    the file has."""
    data = {}
    for arm, (params, boards, stats) in models.items():
        data[arm] = plain({'base': params['base'].tolist(), 'offset': params['offset'].tolist(),
                           'comply': params['comply'].tolist(),
                           'backlash': params['backlash'].tolist(),
                           'geom': params['geom'].tolist(),
                           **({'hysteresis': params['hysteresis'].tolist()}
                              if 'hysteresis' in params else {}),
                           **({kk: params[kk].tolist() for kk in ('tip_delta', 'anchor_shift')
                               if kk in params}),
                           'pieces': [b.tolist() for b in boards],
                           **stats})
    data['when'] = datetime.datetime.now().isoformat(timespec='seconds')
    old = load_model_raw()
    for arm in ('left', 'right'):
        if arm not in data and arm in old:
            data[arm] = old[arm]
    with open(MODEL_FILE, 'w') as handle:
        handle.write('# Arm models from click_to_move CALIBRATE ARMS (ctm/armcal.py).\n'
                     + yaml.safe_dump(data, sort_keys=False))


def load_model_raw():
    try:
        with open(MODEL_FILE) as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}


class ArmModels:
    """The calibrated arms, and the correction a touch needs from them."""

    def __init__(self):
        raw = load_model_raw()
        self.when = raw.get('when')
        self.params = {}
        for arm in ('left', 'right'):
            if arm in raw:
                self.params[arm] = {kk: np.asarray(raw[arm][kk], float)
                                    for kk in ('base', 'offset', 'comply', 'backlash', 'geom',
                                               'hysteresis', 'tip_delta', 'anchor_shift')
                                    if kk in raw[arm]}
        self.arms = {arm: Arm(arm) for arm in self.params}
        # where each gripper piece sits on the hand (CAL ARMS): for reading
        # the real tip off the pieces live (TOP DOWN, at the hover point)
        # (only for the pieces as they are now: a model fitted with another
        # layout -- the old flat pieces -- does not know where these faces are)
        self.pieces = {arm: [np.asarray(b, float) for b in raw[arm]['pieces']]
                       for arm in ('left', 'right') if arm in raw and 'pieces' in raw[arm]
                       and len(raw[arm]['pieces']) == len(bd.pieces(arm))}

    def has(self, arm):
        return arm in self.params

    def tag(self, arm):
        """What TEACH samples taught on top of this arm's model are marked with."""
        return self.when if arm in self.params else None

    def tip_shift(self, arm, q, tip_offset, dirs=None):
        """Where the tip really is minus where the URDF puts it, for encoder
        angles q (world, m). Aim at target - tip_shift to land on target."""
        if arm not in self.params:
            return np.zeros(3)
        a = self.arms[arm]
        tip = np.array([0.0, 0.0, tip_offset, 1.0])
        delta = np.r_[self.params[arm].get('tip_delta', np.zeros(3)), 0.0]   # anchors
        return ((a.tcp(q, self.params[arm], dirs) @ (tip + delta))[:3] + self.shift(arm)
                - (a.tcp(q) @ tip)[:3])

    def shift(self, arm):
        """The world shift from the anchors (m): the arm model's tip plus this
        is where the depth camera sees it."""
        return self.params.get(arm, {}).get('anchor_shift', np.zeros(3))

    def command(self, arm, tcp, rot, seed, limits=None, dirs=None):
        """The hand_tcp pose (position, 3x3) to send the planner -- which
        believes the URDF -- so that the real tool ends up at `tcp` with
        orientation `rot`: the joints the model says do that, and where the
        URDF puts those joints. Corrects the tilt as well as the position
        (gravity sags the tool down, not only sideways). None if the model's
        inverse kinematics does not converge."""
        if arm not in self.params:
            return np.asarray(tcp, float), np.asarray(rot, float)
        a = self.arms[arm]
        want = np.eye(4)
        want[:3, :3] = rot
        # the tip off where the URDF has it on the hand (anchors): move the
        # hand so that the real tip lands where the nominal one was aimed
        want[:3, 3] = np.asarray(tcp, float) - np.asarray(rot) @ self.params[arm].get(
            'tip_delta', np.zeros(3)) - self.shift(arm)
        q, err = a.ik(want, np.asarray(seed, float), self.params[arm], limits, dirs)
        if err > 1e-4:
            return None
        t = a.tcp(q)
        return t[:3, 3], t[:3, :3]
