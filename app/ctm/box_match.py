"""The known box against the face the camera measured (2026-10-28).

The size is known (held_object/size: height x width x depth, e.g. 170 x 100 x
25 mm); the box picker measures only the face it will grip, as (long, short)
in any orientation -- 168 x 99 is the 170 x 100 face, 171 x 24 the 170 x 25
one. The face that fits best within the tolerance is the one gripped, the
box's third size goes along the tool, and the face's long edge (measured, in
the world) says how the box turns about the tool.
"""

import itertools

import numpy as np

TOL_MM = 15.0        # a face edge may be off by this much ...
TOL_SHARE = 0.12     # ... or this share of its length, whichever is larger


def match(size_hwd_mm, face_mm):
    """(along the long edge, across it, along the tool) mm and the worst edge
    error mm, or (None, error) when no face of the box fits."""
    long_, short = sorted((float(v) for v in face_mm), reverse=True)
    if long_ <= 0 or short <= 0:
        return None, None
    best = None
    for i, j, k in itertools.permutations(range(3)):
        a, b, c = (float(size_hwd_mm[n]) for n in (i, j, k))
        if a < b:
            continue
        err = max(abs(long_ - a), abs(short - b))
        ok = abs(long_ - a) <= max(TOL_MM, TOL_SHARE * a) and \
            abs(short - b) <= max(TOL_MM, TOL_SHARE * b)
        if best is None or (ok, -err) > (best[0], -best[2]):
            best = (ok, (a, b, c), err)
    ok, dims, err = best
    return (dims if ok else None), err


def box_on_tool(rot_tcp_world, long_world, box_mm, tip_offset):
    """The matched box hanging off the tip, in hand_tcp: (rotation, centre,
    extents m), its gripped face on the tip, its long edge as measured
    (turned square to the tool). None if that edge runs along the tool."""
    z = np.array([0.0, 0.0, 1.0])
    x = np.asarray(rot_tcp_world, float).T @ np.asarray(long_world, float)
    x = x - (x @ z) * z
    if np.linalg.norm(x) < 0.3:
        return None
    x /= np.linalg.norm(x)
    rot = np.column_stack((x, np.cross(z, x), z))
    ext = np.asarray(box_mm, float) / 1000.0
    return rot, z * (tip_offset + ext[2] / 2.0), ext
