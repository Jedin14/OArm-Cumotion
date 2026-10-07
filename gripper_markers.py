#!/usr/bin/env python3
"""ArUco flags on the grippers: the print sheet, and finding them in the camera image.

Why markers: in a front approach the depth camera cannot see the gripper (the
hand is nearer than its minimum range and hides the fingertips), and the arm
bends in its gearboxes by up to ~30 mm where the encoders cannot see it. The
colour camera can see a flag on the gripper at any range. Each flag is a
printed marker on a stiff card fixed to one end of the gripper rail, facing
back towards the wrist; with one on each end, one is in view in every
front-approach posture (simulated: 20/20 targets, every reachable roll).

    python3 gripper_markers.py --sheet      # writes gripper_markers.pdf to print
    python3 gripper_markers.py --live       # camera view: which flags it can see

Where each flag is relative to the fingertips is measured by the robot itself
(click_to_move.py, CALIBRATE MARKERS), so the mounting only has to be rigid,
not precise.
"""

import argparse
import os

import cv2
import numpy as np

WS = os.path.dirname(os.path.abspath(__file__))
DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_MM = 50.0                       # black square, edge to edge
BORDER_MM = 7.0                        # white margin around it on the card
IDS = {'left': (10, 11), 'right': (20, 21)}
ARM_OF = {i: arm for arm, ids in IDS.items() for i in ids}


def dictionary():
    return cv2.aruco.getPredefinedDictionary(DICTIONARY)


def detect(bgr, info, ids_wanted=None):
    """{id: (rvec, tvec, corners (4,2))} for each marker found, in the colour
    camera's optical frame (metres). Pose by IPPE for a square: the four
    corners at their sub-pixel positions, lens distortion included."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary(), parameters=params)
    out = {}
    if ids is None:
        return out
    k = np.array(info.k, dtype=np.float64).reshape(3, 3)
    d = np.array(info.d, dtype=np.float64) if len(info.d) else np.zeros(5)
    half = MARKER_MM / 2000.0
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
                   dtype=np.float64)
    for c, i in zip(corners, ids.ravel()):
        i = int(i)
        if ids_wanted is not None and i not in ids_wanted:
            continue
        img = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, d, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            out[i] = (rvec.ravel(), tvec.ravel(), img)
    return out


# Where a flag nominally sits in the hand frame: off the end of the rail, the
# printed face looking back along -Z (towards the wrist). Only used to pick the
# right one of the two pose solutions before a flag has been calibrated.
NOMINAL_NORMAL = np.array([0.0, 0.0, -1.0])
TIP_IN_HAND = np.array([0.0, 0.0, 0.0955])   # fingertip midpoint (openarm_hand.xacro)
CALIBRATION_FILE = os.path.join(WS, 'marker_calibration.yaml')


def detect_both(bgr, info, ids_wanted=None):
    """Like detect(), but with both IPPE solutions per marker:
    {id: ([(rvec, tvec, reprojection error), ...], corners)}. A small marker
    seen nearly face-on has two poses that fit its corners almost equally
    well, mirrored in tilt; the caller picks with what it expects."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary(), parameters=params)
    out = {}
    if ids is None:
        return out
    k = np.array(info.k, dtype=np.float64).reshape(3, 3)
    d = np.array(info.d, dtype=np.float64) if len(info.d) else np.zeros(5)
    half = MARKER_MM / 2000.0
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
                   dtype=np.float64)
    for c, i in zip(corners, ids.ravel()):
        i = int(i)
        if ids_wanted is not None and i not in ids_wanted:
            continue
        img = c.reshape(4, 2).astype(np.float64)
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, k, d,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
        sols = [(np.asarray(rvecs[j]).ravel(), np.asarray(tvecs[j]).ravel(),
                 float(np.asarray(errs).ravel()[j]) if errs is not None else 0.0)
                for j in range(n)]
        if sols:
            out[i] = (sols, img)
    return out


def average_rotation(mats):
    """Chordal mean of rotation matrices."""
    u, _s, vt = np.linalg.svd(np.sum(mats, axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def observe(frames, info, ids_wanted, expected_normal):
    """Marker poses averaged over several colour frames.

    expected_normal: {id: unit vector, camera frame} -- where the marker's
    face is expected to point (from the joint readings); of the two pose
    solutions per frame, the one whose normal is nearer it is kept, and a
    frame where even that one is more than 25 deg off is dropped.
    Returns {id: (4x4 camera<-marker, frames used, apparent size px)}.
    """
    by_id = {}
    for bgr in frames:
        for i, (sols, img) in detect_both(bgr, info, ids_wanted).items():
            want = expected_normal.get(i)
            best = None
            for rvec, tvec, err in sols:
                m = marker_matrix(rvec, tvec)
                normal = m[:3, 2]
                score = -float(normal @ want) if want is not None else err
                if best is None or score < best[0]:
                    best = (score, m)
            if want is not None and -best[0] < np.cos(np.radians(25)):
                continue
            size = float(np.mean([np.linalg.norm(img[a] - img[(a + 1) % 4]) for a in range(4)]))
            by_id.setdefault(i, []).append((best[1], size))
    out = {}
    for i, rows in by_id.items():
        mats = [r[0] for r in rows]
        t = np.median([m[:3, 3] for m in mats], axis=0)
        m = np.eye(4)
        m[:3, :3] = average_rotation([x[:3, :3] for x in mats])
        m[:3, 3] = t
        out[i] = (m, len(rows), float(np.mean([r[1] for r in rows])))
    return out


def _align(a, b):
    """The smallest rotation taking direction a to direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def hand_from_flags(seen, hand_markers, r_hand):
    """The hand's pose in the camera frame from the flags it carries.

    seen: observe() output; hand_markers: {id: 4x4 hand<-marker};
    r_hand: the hand's rotation in the camera frame from the joint readings.

    Only the flags' *positions* are used. A 50 mm marker seen face-on gives
    its position to a fraction of a millimetre across the image but its tilt
    only to a degree or two, and the fingertip is 9 cm from the flags, so
    the marker's own rotation would put 2-4 mm of noise on the fingertip
    (simulated, 2026-10-07). With both flags in view, the line between them
    (218 mm long) corrects the joint-reading rotation in the two directions
    it can see; the third -- tilt about that line -- stays as the joints say.
    (Blending in the flags' own reading of that tilt helped the typical case
    a little and the worst case not at all: with 2 deg of flex, worst miss
    4.3 mm without, 5.0 mm at 30 % weight, 7.1 mm at full.)
    Returns (4x4 camera<-hand, ids used).
    """
    ids = [i for i in seen if i in hand_markers]
    if not ids:
        return None, []
    r = np.asarray(r_hand, float)
    pos = {i: seen[i][0][:3, 3] for i in ids}
    local = {i: hand_markers[i][:3, 3] for i in ids}
    if len(ids) >= 2:
        a, b = ids[0], ids[1]
        if np.linalg.norm(local[a] - local[b]) > 0.05:
            r = _align(r @ (local[a] - local[b]), pos[a] - pos[b]) @ r
    origin = np.mean([pos[i] - r @ local[i] for i in ids], axis=0)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, origin
    return out, ids


def fit_circle(points):
    """3D circle through points: (centre, unit normal, radius, rms m)."""
    c0 = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - c0)
    e1, e2, normal = vt[0], vt[1], vt[2]
    xy = np.column_stack([(points - c0) @ e1, (points - c0) @ e2])
    a = np.column_stack([2 * xy, np.ones(len(xy))])
    (cx, cy, k), *_ = np.linalg.lstsq(a, (xy ** 2).sum(axis=1), rcond=None)
    radius = float(np.sqrt(max(1e-12, k + cx * cx + cy * cy)))
    rms = float(np.sqrt(np.mean((np.linalg.norm(xy - (cx, cy), axis=1) - radius) ** 2)))
    return c0 + cx * e1 + cy * e2, normal, radius, rms


def flags_from_roll(samples):
    """Where each flag sits on the hand, from one roll sweep.

    samples: [(4x4 camera<-hand from the joint readings, observe() output)],
    the hand turned about its own Z axis with the fingertip held still. Each
    flag then draws a circle round that axis; the circles' common centre and
    normal *are* the axis, in the camera frame, measured from flag positions
    alone (sub-millimetre across the image) -- the arm's flex does not enter,
    since it hardly changes while only the wrist turns. Along the axis, and
    the roll about it, still come from the joints: an error along the axis
    is taken up by the contact move, and a roll error is the same for both
    flags, which cancels when both are in view (hand_from_flags()).

    Returns ({id: [4x4 hand<-marker, one per sample]}, {id: circle rms mm}).
    """
    by_id = {}
    for k, (_hand, seen) in enumerate(samples):
        for i, (m, _n, _size) in seen.items():
            by_id.setdefault(i, []).append((k, m))
    fits = {}
    for i, rows in by_id.items():
        if len(rows) < 4:            # 3 points always fit a circle exactly: no check
            continue
        pts = np.array([m[:3, 3] for _k, m in rows])
        centre, normal, radius, rms = fit_circle(pts)
        # an arc too short to place its centre (< ~50 deg) is no use
        chord = max(np.linalg.norm(a - b) for a in pts for b in pts)
        if radius < 0.02 or chord < 0.8 * radius or rms > 0.003:
            continue
        fits[i] = (centre, normal, rms, rows)
    if not fits:
        return {}, {}
    z_enc = np.mean([h[:3, 2] for h, _s in samples], axis=0)
    normals = [f[1] * (1.0 if f[1] @ z_enc > 0 else -1.0) for f in fits.values()]
    axis = np.mean(normals, axis=0)
    axis /= np.linalg.norm(axis)
    centre = np.mean([f[0] for f in fits.values()], axis=0)
    out = {}
    for i, (_c, _n, _rms, rows) in fits.items():
        for k, m in rows:
            hand = samples[k][0]
            h = np.eye(4)
            h[:3, :3] = _align(hand[:3, 2], axis) @ hand[:3, :3]
            h[:3, 3] = centre + axis * float((hand[:3, 3] - centre) @ axis)
            out.setdefault(i, []).append(np.linalg.inv(h) @ m)
    return out, {i: 1000 * f[2] for i, f in fits.items()}


class MarkerCalibration:
    """Where each flag sits on its gripper: {id: 4x4 hand<-marker}, from
    CALIBRATE MARKERS."""

    def __init__(self, data):
        self.data = data or {}

    @classmethod
    def load(cls, path=CALIBRATION_FILE):
        import yaml
        try:
            with open(path) as handle:
                raw = yaml.safe_load(handle) or {}
        except (OSError, Exception):
            return None
        markers = {int(k): v for k, v in (raw.get('markers') or {}).items()}
        return cls(markers) if markers else None

    def save(self, path=CALIBRATION_FILE, note=''):
        import yaml
        text = yaml.safe_dump({'markers': self.data}, sort_keys=False)
        with open(path, 'w') as handle:
            handle.write('# Gripper flags: pose of each marker in its hand frame, measured by\n'
                         '# click_to_move.py CALIBRATE MARKERS. ' + note + '\n' + text)

    def ids(self, arm):
        return [i for i, v in self.data.items() if v.get('arm') == arm]

    def has_arm(self, arm):
        return bool(self.ids(arm))

    def hand_marker(self, marker_id):
        return np.array(self.data[marker_id]['hand_marker'], dtype=float)


def marker_matrix(rvec, tvec):
    m = np.eye(4)
    m[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    m[:3, 3] = tvec
    return m


def write_sheet(path):
    """An A4 PDF with the four markers at exact size, cut lines, a scale
    check and the mounting instructions."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    import io
    from PIL import Image

    page = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    page.setFont('Helvetica-Bold', 15)
    page.drawString(18 * mm, height - 20 * mm, 'Gripper marker flags for click_to_move')
    page.setFont('Helvetica', 9.5)
    lines = [
        'PRINT AT 100% / "Actual size" (no "fit to page"). Check: the bar below must measure exactly 100 mm.',
        'Two flags per gripper: LEFT arm = IDs 10 and 11, RIGHT arm = IDs 20 and 21 (either ID on either end).',
        '1. Cut each card out on its outline and glue it FLAT onto something stiff, 2-3 mm thick',
        '   (foam board, plastic, acrylic or a 3D-printed plate). Paper or thin card will bend: not good enough.',
        '2. Fix one card to EACH END of the gripper rail (the bar the fingers slide along), sticking out',
        '   sideways beyond the end, with the printed side facing BACK towards the wrist / arm',
        '   (i.e. away from the fingertips). Keep it clear of the fingers when they open fully.',
        '3. Make it RIGID: hot glue, cable ties or a screw. It must not wobble or shift when the arm moves.',
        '   The exact angle and position do not matter -- the robot measures them (CALIBRATE MARKERS).',
        '   If a flag is ever bumped or moved, run CALIBRATE MARKERS again.',
    ]
    y = height - 30 * mm
    for line in lines:
        page.drawString(18 * mm, y, line)
        y -= 5 * mm
    # scale check
    y -= 4 * mm
    page.setLineWidth(1.2)
    page.line(18 * mm, y, 118 * mm, y)
    for x in (18, 68, 118):
        page.line(x * mm, y - 2 * mm, x * mm, y + 2 * mm)
    page.drawString(122 * mm, y - 1.5 * mm, '<- must be exactly 100 mm')
    # the cards
    card = MARKER_MM + 2 * BORDER_MM
    positions = [(25, 60), (25 + card + 20, 60), (25, 60 + card + 25), (25 + card + 20, 60 + card + 25)]
    labels = [(10, 'LEFT arm'), (11, 'LEFT arm'), (20, 'RIGHT arm'), (21, 'RIGHT arm')]
    for (x0, y0), (marker_id, arm) in zip(positions, labels):
        pixels = 600
        img = cv2.aruco.drawMarker(dictionary(), marker_id, pixels)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='PNG')
        buf.seek(0)
        page.setLineWidth(0.4)
        page.setDash(3, 2)
        page.rect(x0 * mm, y0 * mm, card * mm, card * mm)
        page.setDash()
        page.drawImage(ImageReader(buf), (x0 + BORDER_MM) * mm, (y0 + BORDER_MM) * mm,
                       MARKER_MM * mm, MARKER_MM * mm)
        page.setFont('Helvetica', 8)
        page.drawString(x0 * mm, (y0 - 4) * mm,
                        f'ID {marker_id} - {arm} - marker {MARKER_MM:.0f} mm, card {card:.0f} mm')
    page.setFont('Helvetica-Oblique', 8)
    page.drawString(18 * mm, 15 * mm, 'Dashed line = cut line. Do not trim into the white margin: '
                    'the detector needs it.')
    page.showPage()
    page.save()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sheet', action='store_true', help='write gripper_markers.pdf')
    parser.add_argument('--live', action='store_true',
                        help='show the camera with every detected flag outlined')
    args = parser.parse_args()
    if args.sheet:
        print(write_sheet(os.path.join(WS, 'gripper_markers.pdf')))
    if args.live:
        live()


def live():
    """Camera view with the flags it finds: green = detected (ID, size in px),
    red = square-ish shapes it rejected. Every flag should be green with the
    arms at pre-pick and wherever they touch. q / Esc quits."""
    import rclpy
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init()
    node = rclpy.create_node('flag_check')
    got = {}
    node.create_subscription(Image, '/camera/camera/color/image_raw',
                             lambda m: got.__setitem__('img', m), 2)
    node.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                             lambda m: got.__setitem__('info', m), 2)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        msg = got.pop('img', None)
        if msg is None:
            continue
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, dictionary(), parameters=cv2.aruco.DetectorParameters_create())
        cv2.aruco.drawDetectedMarkers(img, rejected, borderColor=(0, 0, 255))
        seen = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids, borderColor=(0, 255, 0))
            for c, i in zip(corners, ids.ravel()):
                size = np.linalg.norm(c[0][0] - c[0][1])
                seen.append(f'{int(i)}({ARM_OF.get(int(i), "?")} {size:.0f}px)')
        missing = [i for ids_ in IDS.values() for i in ids_
                   if ids is None or i not in ids.ravel()]
        text = 'seen: ' + (' '.join(seen) or 'none') + '   missing: ' + str(missing)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('flag check', img)
        if cv2.waitKey(1) in (27, ord('q')):
            break
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
