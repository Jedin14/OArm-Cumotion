"""The calibration pieces: two small ArUco grids per gripper, on the back of
the rail (facing the wrist, i.e. the camera when the arm reaches forward),
one either side of the motor in the middle.

Each piece: 3 x 2 markers of 20 mm, 4 mm apart, on a 75 x 50 mm card (the
white border included), printed at 100 %. (20 mm is about the smallest the
640 x 480 camera reads at 50 cm: ~2.5 px per marker cell.) Every visible marker gives its own four corners, so one piece in view
is enough. Where exactly each piece sits on the rail does not matter: the
fit (ctm/armcal.py) finds each piece's pose on the gripper, starting from
what the camera sees. They must lie flat on the rail and not move.

Marker ids: left gripper 22-27 and 28-33, right gripper 34-39 and 40-45
(the old flags were 10/11/20/21). (One 182 mm strip across the rail was the
first design -- it does not fit past the motor. A ChArUco strip of this
height has too few inner corners, and 18 mm markers are too small to read
from 40-50 cm: both tried in simulation, 2026-10-12.)

    python3 -m ctm.board --sheet      # writes gripper_boards.pdf to print
"""

import argparse
import os

import cv2
import numpy as np

from ctm.common import WS

MARKERS = (3, 2)                # per piece: along the rail, across it
MARKER = 0.020                  # m
GAP = 0.004                     # m
FIRST = {'left': (22, 28), 'right': (34, 40)}       # each piece's first id
RAIL_Y = 0.055                  # m: a piece's centre this far either side of the hand's middle (nominal)
CUT = (0.075, 0.050)            # m: the card, white border included (fits the rail either side of the motor)
SHEET = os.path.join(WS, 'gripper_boards.pdf')
SIZE = (MARKERS[0] * MARKER + (MARKERS[0] - 1) * GAP, MARKERS[1] * MARKER + (MARKERS[1] - 1) * GAP)
PER_PIECE = MARKERS[0] * MARKERS[1]


def dictionary():
    return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def pieces(arm):
    """The first ids of this arm's pieces, in order (piece 0 sits on the hand's
    +y side, piece 1 on its -y side, nominally)."""
    return FIRST[arm]


def board(first):
    return cv2.aruco.GridBoard_create(MARKERS[0], MARKERS[1], MARKER, GAP, dictionary(), first)


def piece_of(arm, marker_id):
    """Which piece (0, 1) of this arm a marker id is on, or None."""
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
    """A4: all four pieces at exact size, labelled, with a 100 mm scale check."""
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    c.setFont('Helvetica-Bold', 14)
    c.drawString(20 * mm, h - 20 * mm, 'Gripper calibration pieces (click_to_move CALIBRATE ARMS)')
    c.setFont('Helvetica', 9.5)
    lines = ['Four 75 x 50 mm pieces. Print at 100 % / "Actual size" (no fit-to-page). Check: the bar below must measure '
             'exactly 100 mm.',
             'Cut along the dashed lines, keeping the white border. Glue each piece flat onto stiff '
             'board (2-3 mm), no bubbles or curl.',
             'Two pieces per gripper: one on the BACK of the rail (the side facing the wrist) LEFT '
             'of the motor, one RIGHT of it,',
             'printed side towards the wrist, flat on the rail. Which piece goes on which side, and '
             'which way up, does not matter --',
             'but LEFT pieces on the LEFT gripper, RIGHT pieces on the RIGHT one. They must not move '
             'during the run.',
             'Only needed for the one-time calibration: take them off afterwards (put them back the '
             'same way to recalibrate).']
    for i, line in enumerate(lines):
        c.drawString(20 * mm, h - (30 + 5.5 * i) * mm, line)
    c.setLineWidth(1)
    y = h - 70 * mm
    c.line(20 * mm, y, 120 * mm, y)
    for x in (20, 120):
        c.line(x * mm, y - 2 * mm, x * mm, y + 2 * mm)
    c.drawString(125 * mm, y - 1.5 * mm, '100 mm')
    bw, bh = SIZE[0] * 1000, SIZE[1] * 1000
    for row, arm in enumerate(('left', 'right')):
        for k, first in enumerate(FIRST[arm]):
            px = 20                                     # pixels per mm in the image
            img = board(first).draw((int(bw * px), int(bh * px)), marginSize=0, borderBits=1)
            ok, png = cv2.imencode('.png', img)
            top = h - (95 + row * 80) * mm
            x0 = (25 + k * 100) * mm
            c.drawImage(ImageReader(io.BytesIO(png.tobytes())), x0, top - bh * mm,
                        bw * mm, bh * mm)
            mx, my = (CUT[0] * 1000 - bw) / 2, (CUT[1] * 1000 - bh) / 2
            c.setDash(2, 2)
            c.rect(x0 - mx * mm, top - (bh + my) * mm, CUT[0] * 1000 * mm, CUT[1] * 1000 * mm)
            c.setDash()
            c.setFont('Helvetica-Bold', 11)
            c.drawString(x0 - mx * mm, top + (my + 3) * mm,
                         f'{arm.upper()} gripper, piece {k + 1}')
    c.showPage()
    c.save()
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sheet', action='store_true', help='write gripper_boards.pdf')
    if parser.parse_args().sheet:
        print(write_sheet())
