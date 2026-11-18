"""Which face of a box the robot grips (its pick face): the rules in one place.
docs/picking_rules.md. faces.py finds the faces and measures them; this decides.

FRONT RULES  (the face turned to the camera, within FACE_ANGLE_DEG of facing it)
    A front can always be gripped. A box turned away with no front or top: its face turned
    most towards the camera counts as its front (faces.analyse).

TOP RULES  (the face on top, within FACE_ANGLE_DEG of level)
    A top found on a box that also has a front counts only if it really is this box's top and
    is measured well (faces.analyse):
    - as wide as the front along the edge they share (else it's a sliver of the box under it);
    - its front edge seen at FLAT_VIEW_DEG or more (flatter, it is a few rows of noisy depth:
      it is then dropped, and its pixels go back to the front).

BASE RULE  (a box on the base: no box under it)
    The larger of its front and top. A flat box lying on the floor is gripped from the top
    when its top is bigger than its thin front.

DEFAULT: a box standing or lying on another box is gripped by its FRONT, whatever its top
    shows (the robot would have to reach in over the stack for a top). If no front was found
    for it, only a top, the top is used as a last resort and the box is flagged (top_only):
    the robot still picks it, and a person is told.

TOP FACES (top_faces=True, the window's Advanced tab: F, or --top-faces): every box as on the
    base, the larger of its front and top wherever it is (how it worked before 5 Nov 2026).
"""


def pick_face(faces, on_box, top_faces=False):
    """The pick face among a box's faces (the front and top that passed the rules above),
    and whether it is a top used only because no front was found (flag it).
    on_box: the box rests on another box. -> (face or None, top_only)"""
    fronts = [f for f in faces if f.kind == "front"]
    tops = [f for f in faces if f.kind == "top"]
    larger = max(fronts + tops, key=lambda f: f.area_mm2, default=None)
    if top_faces or not on_box:                        # base rule (and top faces mode)
        return larger, False
    if fronts:                                         # on a box: front rule
        return max(fronts, key=lambda f: f.area_mm2), False
    return larger, larger is not None                  # on a box, only a top: last resort, flagged
