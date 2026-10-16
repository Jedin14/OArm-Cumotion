"""The calibration pieces: two per gripper, one either side of the motor in
the middle, each folded round the rail into three faces:

    top     folds up from the back face's upper long edge (faces the ceiling
            with the gripper level)
    back    on the back of the rail, facing the wrist (the camera when the
            arm reaches forward) -- the old single face
    bottom  folds down from its lower long edge (faces the floor)

Each face: 3 x 2 markers of 20 mm, 4 mm apart, 75 x 50 mm with its white
border; a piece is 75 x 150 mm, printed at 100 %, folded along the two
dashed lines. With the tool pointing down the back face looks up, under the
forearm, and the camera rarely sees it; the top and bottom faces then look
sideways and it does (2026-10-15). Faces at right angles also fix the
hand's pose far better from one view than one flat card.

Every face is its own target: the fit (ctm/armcal.py) finds each face's
pose on the gripper, so the folds need not be exact -- only flat, rigid
and fixed. (20 mm is about the smallest the 640 x 480 camera reads at 50
cm: ~2.5 px per marker cell.)

Marker ids (4x4, the 100-marker set; its first 50 are the old set, so the
wall marker 11 is the same): left gripper 12-47, right gripper 48-83, six
per face.

    python3 -m ctm.board --sheet      # writes gripper_boards.pdf to print
    python3 -m ctm.board --live       # camera view: every marker it sees, with its id
"""

import argparse
import os

import cv2
import numpy as np

from ctm.common import WS

MARKERS = (3, 2)                # per face: along the rail, across it
MARKER = 0.020                  # m
GAP = 0.004                     # m
FACES = ('top', 'back', 'bottom')                   # a piece's faces, top to bottom on the sheet
# each face's first id: piece 1's top, back, bottom, then piece 2's
FIRST = {'left': (12, 18, 24, 30, 36, 42), 'right': (48, 54, 60, 66, 72, 78)}
RAIL_Y = 0.055                  # m: a piece's centre this far either side of the hand's middle (nominal)
CUT = (0.075, 0.050)            # m: one face, white border included
SHEET = os.path.join(WS, 'gripper_boards.pdf')
SIZE = (MARKERS[0] * MARKER + (MARKERS[0] - 1) * GAP, MARKERS[1] * MARKER + (MARKERS[1] - 1) * GAP)
PER_PIECE = MARKERS[0] * MARKERS[1]


def dictionary():
    return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)


def pieces(arm):
    """The first ids of this arm's faces, in order -- face k is FACES[k % 3]
    of piece k // 3 (piece 0 nominally on the hand's +y side)."""
    return FIRST[arm]


def face_of(k):
    """(piece 0/1, 'top'/'back'/'bottom') of face index k."""
    return k // len(FACES), FACES[k % len(FACES)]


def board(first):
    return cv2.aruco.GridBoard_create(MARKERS[0], MARKERS[1], MARKER, GAP, dictionary(), first)


def piece_of(arm, marker_id):
    """Which face (index into pieces(arm)) of this arm a marker id is on, or None."""
    for k, first in enumerate(FIRST[arm]):
        if first <= marker_id < first + PER_PIECE:
            return k
    return None


def corners_3d(arm):
    """{marker id: (4, 3) corners in its piece's frame (origin at the piece's
    bottom-left, x along the rail, z out of the printed face)}"""
    out = {}
    for first in FIRST[arm]:
        b = board(first)
        out.update({int(i): np.asarray(c, float).reshape(4, 3)
                    for i, c in zip(b.ids.ravel(), b.objPoints)})
    return out


def detect(bgr, arm):
    """{marker id: (4, 2) corner pixels} of this arm's pieces in one image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # A 2 px window: the default 5 px, on markers ~25 px across, wandered into
    # the marker's own pattern -- 1.34 px rms corner error against 0.32 px
    # (rendered tests, 2026-10-12).
    params.cornerRefinementWinSize = 2
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary(), parameters=params)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for c, i in zip(corners, ids.ravel())
            if piece_of(arm, int(i)) is not None}


def detect_many(frames, arm, min_frames=3):
    """Corner pixels averaged over several frames of a still scene:
    {marker id: (4, 2)} for markers seen in at least min_frames of them."""
    seen = {}
    for bgr in frames:
        for i, c in detect(bgr, arm).items():
            seen.setdefault(i, []).append(c)
    return {i: np.median(np.array(cs), axis=0) for i, cs in seen.items() if len(cs) >= min_frames}


def write_sheet(path=SHEET):
    """A4, one page per gripper: its two pieces at exact size (three faces
    each, fold lines dashed), labelled, with a 100 mm scale check."""
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    bw, bh = SIZE[0] * 1000, SIZE[1] * 1000                 # one face's markers, mm
    fw, fh = CUT[0] * 1000, CUT[1] * 1000                   # one face with its border, mm
    for arm in ('left', 'right'):
        c.setFont('Helvetica-Bold', 14)
        c.drawString(20 * mm, h - 20 * mm, f'{arm.upper()} gripper - calibration pieces '
                                           f'(click_to_move CAL ARMS)')
        c.setFont('Helvetica', 9.5)
        lines = ['Print at 100 % / "Actual size" (no fit-to-page). Check: the bar below must '
                 'measure exactly 100 mm.',
                 'Cut along the solid outline (75 x 150 mm). Fold along the two dashed lines, '
                 '90 degrees, printed side out.',
                 'BACK (middle) goes flat on the back of the rail, facing the wrist, one piece '
                 'each side of the motor;',
                 'TOP folds up and over the rail\'s upper side, BOTTOM down over its lower side. '
                 'Stiff card, glued flat;',
                 'the faces must not move or bend. Exact fold angles do not matter - every face '
                 'is calibrated on its own.',
                 f'{arm.upper()} pieces on the {arm.upper()} gripper only. Keep them on: the '
                 'top-down picks read them.']
        for i, line in enumerate(lines):
            c.drawString(20 * mm, h - (30 + 5.5 * i) * mm, line)
        y = h - 68 * mm
        c.setLineWidth(1)
        c.line(20 * mm, y, 120 * mm, y)
        for x in (20, 120):
            c.line(x * mm, y - 2 * mm, x * mm, y + 2 * mm)
        c.drawString(125 * mm, y - 1.5 * mm, '100 mm')
        firsts = FIRST[arm]
        for piece in range(2):
            x0 = (40 + piece * 88) * mm                     # the cut outline's left edge
            top = h - 85 * mm                               # its top edge
            c.setFont('Helvetica-Bold', 11)
            c.drawString(x0, top + 3 * mm, f'{arm.upper()} piece {piece + 1}')
            for f, kind in enumerate(FACES):
                first = firsts[piece * len(FACES) + f]
                px = 20                                     # image pixels per mm
                img = board(first).draw((int(bw * px), int(bh * px)), marginSize=0,
                                        borderBits=1)
                ok, png = cv2.imencode('.png', img)
                fy = top - (f + 1) * fh * mm                # this face's bottom edge
                c.drawImage(ImageReader(io.BytesIO(png.tobytes())),
                            x0 + (fw - bw) / 2 * mm, fy + (fh - bh) / 2 * mm, bw * mm, bh * mm)
                c.setFont('Helvetica-Bold', 8)
                c.drawRightString(x0 - 1.5 * mm, fy + fh / 2 * mm, kind.upper())
                if f:
                    c.setDash(3, 2)                         # fold line
                    c.line(x0, fy + fh * mm, x0 + fw * mm, fy + fh * mm)
                    c.setDash()
            c.rect(x0, top - 3 * fh * mm, fw * mm, 3 * fh * mm)      # cut outline
        c.showPage()
    c.save()
    return path


NAMES = {11: 'wall (CAL ARMS)'}


def name_of(marker_id):
    for arm in ('left', 'right'):
        k = piece_of(arm, marker_id)
        if k is not None:
            piece, kind = face_of(k)
            return f'{arm} piece {piece + 1} {kind}'
    return NAMES.get(marker_id, '')


def live():
    """Show the colour camera with every 4x4 marker found: outline, id, what
    it is, and its size in pixels (under ~20 px it reads badly). q quits."""
    import rclpy
    from sensor_msgs.msg import Image

    from ctm.common import COLOR_TOPIC, decode_image
    rclpy.init()
    node = rclpy.create_node('marker_live')
    latest = {}
    node.create_subscription(Image, COLOR_TOPIC, lambda m: latest.update(msg=m), 2)
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 2
    window = 'markers (q quits)'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if 'msg' not in latest:
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                continue
            img = decode_image(latest['msg'])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary(), parameters=params)
            found = []
            for c, i in zip(corners or [], ids.ravel() if ids is not None else []):
                c = c.reshape(4, 2)
                size = float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))
                colour = (0, 220, 0) if size >= 20 else (0, 200, 255)
                cv2.polylines(img, [np.round(c).astype(np.int32)], True, colour, 2)
                label = f'{int(i)} {name_of(int(i))} {size:.0f}px'.replace('  ', ' ')
                x, y = int(c[:, 0].min()), int(c[:, 1].min()) - 6
                cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
                found.append(int(i))
            head = f'seen: {sorted(found) if found else "none"}   (green >= 20 px, orange small)'
            cv2.putText(img, head, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, head, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow(window, img)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sheet', action='store_true', help='write gripper_boards.pdf')
    parser.add_argument('--live', action='store_true',
                        help='camera view with every marker it sees (q quits)')
    opts = parser.parse_args()
    if opts.sheet:
        print(write_sheet())
    if opts.live:
        live()
