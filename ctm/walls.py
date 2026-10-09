"""Walls where the camera cannot see: above, below and beside its picture.

The camera's view is a pyramid. In front of the robot, everything below the
plane of the picture's bottom edge (the table under the scene, which the
camera never sees), above the plane of its top edge, and beyond its left and
right edges is unknown, so the arm must treat it as solid. Inside the
picture the octomap says what is there. (Walls over and under every object
were tried first and blocked space the camera does see -- 2026-10-09.)

The walls start a little in front of the nearest thing the camera saw (and
never nearer the robot than X_MIN), so the postures near the robot -- home,
navigation, pre_pick, drop -- stay clear. They go into the planning scene as
one collision object ("unseen_walls"): RViz shows them, and MoveIt and
cuMotion both avoid them. Per 2 cm slice: a box above, below, and to
each side.
"""

import numpy as np

TOP = 1.4                       # m: the upper walls go up to here
X_MIN = 0.30                    # m: walls never start nearer the robot than this
X_MAX = 1.0                     # m: nor reach further than this
AHEAD = 0.05                    # m: start this far in front of the nearest thing seen
Y_MAX = 0.9                     # m: half width
SLICE = 0.02                    # m: slice thickness along x
WALL_ID = 'unseen_walls'


def nearest_seen(octomap_msg, origin):
    """x of the nearest occupied cell in front of the robot, or None."""
    from isaac_ros_cumotion import scene_world      # 2 s import: only when used
    centres, _sizes = scene_world.decode_octomap(octomap_msg)
    if not len(centres):
        return None
    c = centres + origin
    keep = ((c[:, 0] > 0.15) & (c[:, 0] < X_MAX) & (np.abs(c[:, 1]) < Y_MAX)
            & (c[:, 2] > 0.10) & (c[:, 2] < TOP))
    return float(c[keep, 0].min()) if keep.any() else None


def boxes(t_world_optical, info, x_start):
    """[(centre xyz, size xyz)]: per slice, a box from the floor up to the
    picture's bottom edge, one from its top edge up to TOP, and one beyond
    each side edge out to Y_MAX."""
    k = np.array(info.k, dtype=float).reshape(3, 3)
    # the picture's edges, from the real optical centre (not the middle:
    # 6 px off on this camera, which let the far top walls graze the view)
    tan_up, tan_down = k[1, 2] / k[1, 1], (info.height - k[1, 2]) / k[1, 1]
    tan_left, tan_right = k[0, 2] / k[0, 0], (info.width - k[0, 2]) / k[0, 0]
    r_ow = t_world_optical[:3, :3].T
    eye = t_world_optical[:3, 3]
    # optical y points down: below the picture is y > tan_down z, above y < -tan_up z
    below = r_ow[1] - tan_down * r_ow[2]
    above = -r_ow[1] - tan_up * r_ow[2]
    # optical x points right: beside it is x > tan_right z or x < -tan_left z
    side_a = r_ow[0] - tan_right * r_ow[2]
    side_b = -r_ow[0] - tan_left * r_ow[2]

    def edge_z(a, x, y):
        """z on the plane a . (p - eye) = 0 at (x, y)."""
        return eye[2] - (a[0] * (x - eye[0]) + a[1] * (y - eye[1])) / a[2]

    def edge_y(a, x, z):
        """y on the plane a . (p - eye) = 0 at (x, z)."""
        return eye[1] - (a[0] * (x - eye[0]) + a[2] * (z - eye[2])) / a[1]

    out = []
    for x0 in np.arange(max(X_MIN, x_start), X_MAX, SLICE):
        x1 = x0 + SLICE
        corners = [(x, y) for x in (x0, x1) for y in (-Y_MAX, Y_MAX)]
        low = min(edge_z(below, x, y) for x, y in corners)    # stays out of the picture
        high = max(edge_z(above, x, y) for x, y in corners)
        cx = (x0 + x1) / 2.0
        if low > SLICE / 2:
            out.append(((cx, 0.0, low / 2.0), (SLICE, 2 * Y_MAX, low)))
        if TOP - high > SLICE / 2:
            out.append(((cx, 0.0, (high + TOP) / 2.0), (SLICE, 2 * Y_MAX, TOP - high)))
        # the sides: from the picture's edge out to Y_MAX, floor to TOP; the
        # edge taken at its widest over the slice, so no wall enters the view
        for a in (side_a, side_b):
            ys = [edge_y(a, x, z) for x in (x0, x1) for z in (0.0, TOP)]
            if np.mean(ys) > eye[1]:                       # the left (+y) edge
                inner = max(ys)
                if Y_MAX - inner > SLICE / 2:
                    out.append(((cx, (inner + Y_MAX) / 2.0, TOP / 2.0),
                                (SLICE, Y_MAX - inner, TOP)))
            else:                                          # the right (-y) edge
                inner = min(ys)
                if inner + Y_MAX > SLICE / 2:
                    out.append(((cx, (inner - Y_MAX) / 2.0, TOP / 2.0),
                                (SLICE, inner + Y_MAX, TOP)))
    return out
