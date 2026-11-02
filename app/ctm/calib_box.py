"""The calibration box and the depth board (2026-10-26).

THE BOX: a cuboid 170 x 100 x 25 mm (the size of the boxes picked), ArUco
markers on all six faces, held by the vacuum cup at the centre of one big
face (a crosshair and a 25 mm ring show where). Wherever the arm holds it,
the camera reads its pose from whichever markers it sees, and so the point
where the cup holds it -- the tip -- measured, not modelled. Even if the
bellows let the box tilt, that point is where the tip is.

    box frame: origin at the centre; x along 170, y along 100, z along 25.
    face B (z = -12.5): the cup's face, crosshair at its centre = THE TIP
    face A (z = +12.5): opposite; C/D: the long sides (y = +-50); E/F: the
    ends (x = +-85).

Markers: DICT_4X4_250, ids 100-133 (the first 100 of that family are the
4X4_100 the gripper pieces and the wall marker use: no clash).

THE BOARD: an A4 ChArUco board (DICT_5X5_100, 7 x 5 squares of 36 mm): the
colour image gives its exact pose, the depth image its surface -- the depth
check (ctm/depth_cal.py).

    python3 -m ctm.calib_box        writes calib_box_sheets.pdf and charuco_board.pdf
"""

import os

import cv2
import numpy as np

from ctm.common import WS

SIZE = (0.170, 0.100, 0.025)                 # m: x, y, z
TIP_IN_BOX = np.array([0.0, 0.0, -SIZE[2] / 2])   # the crosshair on face B
FIRST_ID = 100
BOX_SHEETS = os.path.join(WS, 'calib_box_sheets.pdf')
BOARD_SHEET = os.path.join(WS, 'charuco_board.pdf')

# faces: name -> (outward normal n, panel right r, panel down d, centre); d x r = n
_X, _Y, _Z = np.eye(3)
FACES = {
    'A': (+_Z, +_X, -_Y, +_Z * SIZE[2] / 2),
    'B': (-_Z, +_X, +_Y, -_Z * SIZE[2] / 2),
    'C': (+_Y, +_X, +_Z, +_Y * SIZE[1] / 2),
    'D': (-_Y, -_X, +_Z, -_Y * SIZE[1] / 2),
    'E': (+_X, -_Y, +_Z, +_X * SIZE[0] / 2),
    'F': (-_X, +_Y, +_Z, -_X * SIZE[0] / 2),
}
FACE_TEXT = {'A': 'A - top (opposite the cup)', 'B': 'B - CUP SIDE (cup on the cross)',
             'C': 'C - long side', 'D': 'D - long side', 'E': 'E - end', 'F': 'F - end'}
# markers per face: (size m, [(right, down) centres in m on the panel])
_LAYOUT = {
    'A': (0.034, [(x, y) for y in (-0.023, 0.023) for x in (-0.063, -0.021, 0.021, 0.063)]),
    'B': (0.026, [(x, y) for y in (-0.025, 0.025) for x in (-0.064, -0.034, 0.034, 0.064)]),
    'C': (0.019, [(x, 0.0) for x in (-0.0725, -0.0435, -0.0145, 0.0145, 0.0435, 0.0725)]),
    'D': (0.019, [(x, 0.0) for x in (-0.0725, -0.0435, -0.0145, 0.0145, 0.0435, 0.0725)]),
    'E': (0.019, [(y, 0.0) for y in (-0.030, 0.0, 0.030)]),
    'F': (0.019, [(y, 0.0) for y in (-0.030, 0.0, 0.030)]),
}
PANEL = {'A': (SIZE[0], SIZE[1]), 'B': (SIZE[0], SIZE[1]), 'C': (SIZE[0], SIZE[2]),
         'D': (SIZE[0], SIZE[2]), 'E': (SIZE[1], SIZE[2]), 'F': (SIZE[1], SIZE[2])}


def dictionary():
    return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)


def markers():
    """{id: (face, size, 4x3 corners in the box frame, OpenCV corner order)}"""
    out, mid = {}, FIRST_ID
    for face in 'ABCDEF':
        n, r, d, c = FACES[face]
        size, centres = _LAYOUT[face]
        for (pr, pd) in centres:
            m = c + r * pr + d * pd
            h = size / 2
            out[mid] = (face, size, np.array([m - r * h - d * h, m + r * h - d * h,
                                              m + r * h + d * h, m - r * h + d * h]))
            mid += 1
    return out


MARKERS = markers()
LAST_ID = max(MARKERS)


def name_of(i):
    return f'box {MARKERS[i][0]}' if i in MARKERS else ''


def detect(frames, min_frames=None):
    """{id: (4, 2) pixel corners} of the box's markers, averaged over the
    frames each is seen in (sub-pixel; at least min_frames of them)."""
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 3
    seen = {}
    for img in frames:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary(), parameters=params)
        for c, i in zip(corners or [], ids.ravel() if ids is not None else []):
            if int(i) in MARKERS:
                seen.setdefault(int(i), []).append(c.reshape(4, 2))
    need = min_frames if min_frames is not None else max(1, (len(frames) + 1) // 2)
    return {i: np.mean(v, axis=0) for i, v in seen.items() if len(v) >= need}


def box_pose(seen, k, d):
    """The box's pose in the camera frame (4x4) from its seen markers, the
    rms reprojection error (px) and how many markers; None if too few (2
    markers, or one if it is big)."""
    ids = sorted(seen)
    if not ids or (len(ids) < 2 and MARKERS[ids[0]][1] < 0.03):
        return None
    obj = np.concatenate([MARKERS[i][2] for i in ids]).astype(np.float64)
    img = np.concatenate([seen[i] for i in ids]).astype(np.float64)
    k = np.asarray(k, float).reshape(3, 3)
    d = np.asarray(d, float) if len(d) else np.zeros(5)
    ok, rv, tv = cv2.solvePnP(obj, img, k, d, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj, img, k, d, rv, tv)
    proj, _ = cv2.projectPoints(obj, rv, tv, k, d)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rv)[0]
    t[:3, 3] = tv.ravel()
    return t, rms, len(ids)


# -- the box as it was really built ------------------------------------------------
# The panels are glued by hand and a real box is not exactly 170 x 100 x 25:
# every box pose came out ~1.2 px off its markers (0.2-0.4 is what good
# targets give) and the tips 7 mm off together (Cal tip, 2026-10-27). From
# all the postures' pictures at once, each face's real pose on the box is
# solved (face B fixed: the cross on it IS the tip), with the box's pose per
# picture -- bundle adjustment.

def corners_with(geometry=None):
    """{id: 4x3 corners in the box frame} with each face moved by
    geometry[face] (a 4x4 in the box frame) if given."""
    out = {}
    for i, (face, _size, c) in MARKERS.items():
        g = (geometry or {}).get(face)
        out[i] = c if g is None else c @ g[:3, :3].T + g[:3, 3]
    return out


def box_pose_with(seen, k, d, geometry=None):
    """box_pose() with the refined geometry."""
    ids = sorted(seen)
    if not ids:
        return None
    pts = corners_with(geometry)
    obj = np.concatenate([pts[i] for i in ids]).astype(np.float64)
    img = np.concatenate([np.asarray(seen[i]) for i in ids]).astype(np.float64)
    k = np.asarray(k, float).reshape(3, 3)
    d = np.asarray(d, float) if len(d) else np.zeros(5)
    if len(obj) < 6:
        return None
    ok, rv, tv = cv2.solvePnP(obj, img, k, d, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj, img, k, d, rv, tv)
    proj, _ = cv2.projectPoints(obj, rv, tv, k, d)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rv)[0]
    t[:3, 3] = tv.ravel()
    return t, rms, len(ids)


def refine_geometry(views, k, d, log=print):
    """views: [{id: (4, 2) pixels}] (one per posture). Each face but B gets a
    6-DoF correction (priors: 0.1 rad, 10 mm); every view its box pose.
    Returns ({face: 4x4}, rms before px, rms after px)."""
    from scipy.optimize import least_squares
    k = np.asarray(k, float).reshape(3, 3)
    d = np.asarray(d, float) if len(d) else np.zeros(5)
    faces = [f for f in 'ACDEF' if any(MARKERS[i][0] == f for v in views for i in v)]
    poses = []
    for v in views:
        got = box_pose_with(v, k, d)
        poses.append(got[0] if got else None)
    use = [i for i, p in enumerate(poses) if p is not None]
    views = [views[i] for i in use]
    poses = [poses[i] for i in use]
    if len(views) < 5 or not faces:
        return {}, float('nan'), float('nan')
    nf = len(faces)
    x0 = np.zeros(6 * nf + 6 * len(views))
    for j, p in enumerate(poses):
        x0[6 * nf + 6 * j:6 * nf + 6 * j + 3] = cv2.Rodrigues(p[:3, :3])[0].ravel()
        x0[6 * nf + 6 * j + 3:6 * nf + 6 * j + 6] = p[:3, 3]
    obs = [(j, i, np.asarray(px, float)) for j, v in enumerate(views) for i, px in v.items()]

    def geometry(x):
        g = {}
        for n, f in enumerate(faces):
            m = np.eye(4)
            m[:3, :3] = cv2.Rodrigues(x[6 * n:6 * n + 3])[0]
            m[:3, 3] = x[6 * n + 3:6 * n + 6]
            g[f] = m
        return g

    def residuals(x):
        pts = corners_with(geometry(x))
        r = []
        for j, i, px in obs:
            rv = x[6 * nf + 6 * j:6 * nf + 6 * j + 3]
            tv = x[6 * nf + 6 * j + 3:6 * nf + 6 * j + 6]
            proj, _ = cv2.projectPoints(pts[i], rv, tv, k, d)
            r.append((proj.reshape(-1, 2) - px).ravel())
        prior = np.concatenate([np.r_[x[6 * n:6 * n + 3] / 0.1, x[6 * n + 3:6 * n + 6] / 0.010]
                                for n in range(nf)])
        return np.concatenate(r + [prior])

    n_px = sum(px.size for _j, _i, px in obs)
    before = residuals(x0)[:n_px]
    sol = least_squares(residuals, x0, loss='huber', f_scale=1.0, max_nfev=200)
    after = sol.fun[:n_px]
    rms = lambda r: float(np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, axis=1))))
    g = geometry(sol.x)
    log(f'box geometry: {len(views)} views, faces {"".join(faces)}: {rms(before):.2f} -> '
        f'{rms(after):.2f} px; ' + ', '.join(f'{f} moved {1000 * np.linalg.norm(m[:3, 3]):.1f} mm '
                                         f'{np.degrees(np.linalg.norm(cv2.Rodrigues(m[:3, :3])[0])):.1f} deg'
                                         for f, m in g.items()))
    return g, rms(before), rms(after)


# -- the depth board --------------------------------------------------------------

BOARD_SQUARES = (7, 5)
BOARD_SQUARE = 0.036
BOARD_MARKER = 0.027


def board():
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard_create(BOARD_SQUARES[0], BOARD_SQUARES[1], BOARD_SQUARE,
                                         BOARD_MARKER, d), d


def board_pose(img, k, d):
    """The board's pose in the camera frame (4x4), its corner pixels and how
    many corners; None if fewer than 8 corners."""
    b, dic = board()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    params = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dic, parameters=params)
    if ids is None or len(ids) < 4:
        return None
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, b)
    if cc is None or n < 8:
        return None
    k = np.asarray(k, float).reshape(3, 3)
    d = np.asarray(d, float) if len(d) else np.zeros(5)
    ok, rv, tv = cv2.aruco.estimatePoseCharucoBoard(cc, ci, b, k, d, None, None)
    if not ok:
        return None
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rv)[0]
    t[:3, 3] = tv.ravel()
    return t, cc.reshape(-1, 2), int(n)


# -- the printable sheets -----------------------------------------------------------

def _marker_png(i, px):
    img = cv2.aruco.drawMarker(dictionary(), i, px)
    return cv2.imencode('.png', img)[1].tobytes()


def write_box_sheets(path=BOX_SHEETS):
    """A4: the six faces at exact size, to glue on a 170 x 100 x 25 mm box.
    Each panel says which face it is and which face each of its edges meets,
    so they go on the right way round."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    names = {tuple(np.round(v[0], 3)): f for f, v in FACES.items()}
    c = canvas.Canvas(path, pagesize=A4)
    pw, ph = A4

    def panel(face, cx, cy):
        w, h = (v * 1000 for v in PANEL[face])
        x0, y0 = cx - w / 2 * mm, cy - h / 2 * mm
        c.setLineWidth(0.4)
        c.setDash(3, 2)
        c.rect(x0, y0, w * mm, h * mm)
        c.setDash()
        size, centres = _LAYOUT[face]
        for (pr, pd), mid in zip(centres, [i for i, m in MARKERS.items() if m[0] == face]):
            s = size * 1000
            img = ImageReader(io.BytesIO(_marker_png(mid, 400)))
            c.drawImage(img, cx + (pr * 1000 - s / 2) * mm, cy - (pd * 1000 + s / 2) * mm,
                        s * mm, s * mm)
        n, r, d, _c = FACES[face]
        # which face each edge meets (the panel's top edge = -d, ...): in the
        # title, outside the panel -- inside, the text sat on the markers
        edge = {'top': -d, 'bottom': d, 'left': -r, 'right': r}
        meet = {k: names.get(tuple(np.round(v, 3)), '?') for k, v in edge.items()}
        c.setFont('Helvetica-Bold', 7)
        c.drawString(x0, y0 + h * mm + 3, FACE_TEXT[face])
        c.setFont('Helvetica', 6.5)
        c.drawRightString(x0 + w * mm, y0 + h * mm + 3,
                          f'top edge meets {meet["top"]}, bottom edge meets {meet["bottom"]}, '
                          f'left {meet["left"]}, right {meet["right"]}')
        if face == 'B':
            c.setLineWidth(0.6)
            c.circle(cx, cy, 12.5 * mm)
            c.line(cx - 18 * mm, cy, cx + 18 * mm, cy)
            c.line(cx, cy - 18 * mm, cx, cy + 18 * mm)
            c.setFont('Helvetica', 5)
            c.drawCentredString(cx, cy - 16 * mm, 'cup here (25 mm)')

    def scale(y):
        c.setLineWidth(0.8)
        c.line(20 * mm, y, 120 * mm, y)
        for t in range(0, 101, 10):
            c.line((20 + t) * mm, y, (20 + t) * mm, y + (3 if t % 50 else 5) * mm)
        c.setFont('Helvetica', 7)
        c.drawString(122 * mm, y, '100 mm - check before cutting (print at 100 %)')

    c.setFont('Helvetica-Bold', 10)
    c.drawString(20 * mm, ph - 15 * mm, 'Calibration box 170 x 100 x 25 mm - faces A and B')
    c.setFont('Helvetica', 7)
    c.drawString(20 * mm, ph - 20 * mm, 'Cut on the dashed lines, glue flat on the box faces. '
                 'Edge labels say which face each edge meets.')
    panel('A', pw / 2, ph - 85 * mm)
    panel('B', pw / 2, ph - 200 * mm)
    scale(20 * mm)
    c.showPage()
    c.setFont('Helvetica-Bold', 10)
    c.drawString(20 * mm, ph - 15 * mm, 'Calibration box - sides C, D (170 x 25) and ends E, F '
                 '(100 x 25)')
    panel('C', pw / 2, ph - 50 * mm)
    panel('D', pw / 2, ph - 95 * mm)
    panel('E', pw / 2, ph - 140 * mm)
    panel('F', pw / 2, ph - 185 * mm)
    scale(20 * mm)
    c.showPage()
    c.save()
    return path


def write_board_sheet(path=BOARD_SHEET):
    """A4 landscape: the ChArUco board at exact size."""
    import io
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    b, _d = board()
    wmm, hmm = BOARD_SQUARES[0] * BOARD_SQUARE * 1000, BOARD_SQUARES[1] * BOARD_SQUARE * 1000
    img = b.draw((int(wmm * 10), int(hmm * 10)))
    c = canvas.Canvas(path, pagesize=landscape(A4))
    pw, ph = landscape(A4)
    c.drawImage(ImageReader(io.BytesIO(cv2.imencode('.png', img)[1].tobytes())),
                (pw - wmm * mm) / 2, (ph - hmm * mm) / 2 + 5 * mm, wmm * mm, hmm * mm)
    c.setFont('Helvetica', 7)
    c.drawString(20 * mm, 8 * mm, f'Depth check board: {BOARD_SQUARES[0]} x {BOARD_SQUARES[1]} '
                 f'squares of {BOARD_SQUARE * 1000:.0f} mm (check with a ruler), DICT_5X5_100. '
                 'Print at 100 %, mount flat on cardboard / foam board.')
    c.showPage()
    c.save()
    return path


if __name__ == '__main__':
    print(write_box_sheets())
    print(write_board_sheet())
