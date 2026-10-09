"""Walls where the camera cannot see: above and below its picture.

The camera's view is a pyramid. In front of the robot, everything below the
plane of the picture's bottom edge (the table under the scene, which the
camera never sees) and above the plane of its top edge is unknown, so the
arm must treat it as solid. Inside the picture the octomap says what is
there. (Walls over and under every object were tried first and blocked space
the camera does see -- 2026-10-09.)

The walls start a little in front of the nearest thing the camera saw (and
never nearer the robot than X_MIN), so the postures near the robot -- home,
navigation, pre_pick, drop -- stay clear. They go into the planning scene as
one collision object ("unseen_walls"): RViz shows them, and MoveIt and
cuMotion both avoid them. One box per 2 cm slice, above and below.
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
    picture's bottom edge and one from its top edge up to TOP."""
    k = np.array(info.k, dtype=float).reshape(3, 3)
    tan_v = info.height / 2.0 / k[1, 1]
    r_ow = t_world_optical[:3, :3].T
    eye = t_world_optical[:3, 3]
    # optical y points down: below the picture is y > tan_v z, above is y < -tan_v z
    below = r_ow[1] - tan_v * r_ow[2]
    above = -r_ow[1] - tan_v * r_ow[2]

    def edge_z(a, x, y):
        """z on the plane a . (p - eye) = 0 at (x, y)."""
        return eye[2] - (a[0] * (x - eye[0]) + a[1] * (y - eye[1])) / a[2]

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
    return out
