"""The obstacles move_group checks a path against, for cuMotion to plan around.

cuMotion only modelled the robot. It planned straight through the octomap,
the camera and the stand, and move_group -- which checks every returned path
against its full planning scene -- threw the paths away: "Computed path is
not valid", <octomap> vs openarm_left_link7, camera_link vs
openarm_right_link6, openarm_body_link0 vs openarm_right_link5
(2026-10-06). Every goal failed and the arm never moved.

The planning scene, octomap included, already arrives with every request
(the MoveIt plugin sends it as planning_scene_diff); this turns it into
cuRobo world geometry:

    decode_octomap()   octomap_msgs binary OcTree -> occupied voxel boxes
    ShadowedMap        the octomap's cells plus the space hidden behind them
                       from the camera, as one closed surface mesh
    StaticObstacles    the camera box (from tf) and the stand column
                       (from the stand's collision mesh)
"""

import hashlib
import math
import os

import numpy as np


def decode_octomap(octomap_msg):
    """Occupied leaves of a binary octomap_msgs/Octomap: (centres (N,3), sizes (N,)).

    The binary OcTree stream is depth-first; each node is two bytes holding
    2 bits per child: 00 unknown, 01 occupied leaf, 10 free leaf, 11 inner
    node (its own two bytes follow, in child order). The root is the 2^16
    voxel cube centred on the origin; child i sits at +/- a quarter of its
    parent's size along x (bit 0), y (bit 1) and z (bit 2).
    """
    if not octomap_msg.data:
        return np.zeros((0, 3)), np.zeros(0)
    data = np.frombuffer(bytes(bytearray(x & 0xFF for x in octomap_msg.data)), np.uint8)
    res = float(octomap_msg.resolution)
    if not octomap_msg.binary:
        return _decode_full(data, res)
    centres, sizes = [], []
    pos = 0
    # (centre, size) of nodes whose bytes are next, in stream order
    stack = [(np.zeros(3), res * 65536.0)]
    offsets = np.array([[(i & 1) * 2 - 1, ((i >> 1) & 1) * 2 - 1, ((i >> 2) & 1) * 2 - 1]
                        for i in range(8)], dtype=np.float64)
    while stack and pos + 1 < len(data):
        centre, size = stack.pop()
        bits = int(data[pos]) | (int(data[pos + 1]) << 8)
        pos += 2
        inner = []
        for i in range(8):
            pair = (bits >> (2 * i)) & 0b11
            if pair == 0:
                continue
            child = centre + offsets[i] * (size / 4.0)
            if pair == 0b10:                         # bit i*2+1 set only: occupied leaf
                centres.append(child)
                sizes.append(size / 2.0)
            elif pair == 0b11:
                inner.append((child, size / 2.0))
        # depth first, children in order: push reversed so child 0 pops first
        stack.extend(reversed(inner))
    if not centres:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(centres), np.array(sizes)


def _decode_full(data, res, threshold=0.0):
    """The full (non-binary) OcTree stream move_group sends: depth first, each
    node a float32 log-odds then one byte with a bit per existing child.
    A node without children whose log-odds is above `threshold` (0 = 50%)
    is an occupied leaf."""
    centres, sizes = [], []
    offsets = np.array([[(i & 1) * 2 - 1, ((i >> 1) & 1) * 2 - 1, ((i >> 2) & 1) * 2 - 1]
                        for i in range(8)], dtype=np.float64)
    values = data
    stack = [(np.zeros(3), res * 65536.0)]
    pos = 0
    n = len(values)
    while stack and pos + 5 <= n:
        centre, size = stack.pop()
        logodds = float(np.frombuffer(values[pos:pos + 4].tobytes(), np.float32)[0])
        bits = int(values[pos + 4])
        pos += 5
        if bits == 0:
            if logodds > threshold:
                centres.append(centre)
                sizes.append(size)
            continue
        kids = [(centre + offsets[i] * (size / 4.0), size / 2.0) for i in range(8)
                if bits & (1 << i)]
        stack.extend(reversed(kids))
    if not centres:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(centres), np.array(sizes)


def quat_wxyz(rotation):
    m = rotation
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    return [(m[k, j] - m[j, k]) / s] + q


class StaticObstacles:
    """The camera and the stand column, which move_group knows as robot
    links and cuMotion did not know at all."""

    # realsense2_description _d455: camera_link collision box and its offset
    CAMERA_BOX = (0.026, 0.124, 0.029)
    CAMERA_OFFSET = (-0.00845, -0.0475, 0.0)
    STAND_TOP = 0.62     # below the shoulder housing, which arm link0/1 overlap

    def __init__(self, urdf_path, logger):
        self.logger = logger
        self.stand = None
        try:
            self.stand = self._stand_column(urdf_path)
        except Exception as exc:                     # noqa: BLE001
            logger.warn(f'stand column not modelled for cuMotion: {exc}')

    def _stand_column(self, urdf_path):
        import trimesh
        import xml.etree.ElementTree as ET
        root = ET.parse(urdf_path).getroot()
        link = [l for l in root.findall('link') if l.get('name') == 'openarm_body_link0'][0]
        mesh_el = link.find('collision/geometry/mesh')
        name = mesh_el.get('filename')
        ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        path = name.replace('package://openarm_description',
                            os.path.join(ws, 'src', 'openarm_description'))
        mesh = trimesh.load(path, force='mesh')
        mesh.apply_scale([float(v) for v in (mesh_el.get('scale') or '1 1 1').split()])
        column = trimesh.intersections.slice_mesh_plane(mesh, [0, 0, -1], [0, 0, self.STAND_TOP],
                                                        cap=True)
        self.logger.info(f'stand column for cuMotion: {len(column.faces)} faces below '
                         f'z={self.STAND_TOP}')
        return np.asarray(column.vertices), np.asarray(column.faces)

    def eye(self, tf_buffer, base_frame):
        """Where the depth camera looks from, in base_frame, or None."""
        import rclpy
        for frame in ('camera_color_optical_frame', 'camera_depth_optical_frame', 'camera_link'):
            try:
                t = tf_buffer.lookup_transform(base_frame, frame,
                                               rclpy.time.Time()).transform.translation
                return np.array([t.x, t.y, t.z])
            except Exception:                        # noqa: BLE001
                continue
        return None

    def camera(self, tf_buffer, base_frame):
        """Camera box pose [x y z qw qx qy qz] in base_frame, or None."""
        try:
            tf = tf_buffer.lookup_transform(base_frame, 'camera_link', __import__('rclpy').time.Time())
        except Exception:                            # noqa: BLE001
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        centre = rot @ np.array(self.CAMERA_OFFSET) + [t.x, t.y, t.z]
        return list(centre) + [w, x, y, z]


def voxel_cells(centres, sizes, res):
    """Occupied leaves of any size -> unique integer cells (i, j, k) at the
    finest resolution; cell (i, j, k) spans [i*res, (i+1)*res) on each axis."""
    if not len(centres):
        return np.zeros((0, 3), np.int64)
    scale = np.maximum(1, np.round(sizes / res).astype(np.int64))
    out = []
    for k in np.unique(scale):
        sel = scale == k
        lo = np.round((centres[sel] - (k * res) / 2.0) / res).astype(np.int64)
        grid = np.stack(np.meshgrid(*[np.arange(k)] * 3, indexing='ij'), -1).reshape(-1, 3)
        out.append((lo[:, None, :] + grid[None]).reshape(-1, 3))
    return np.unique(np.concatenate(out), axis=0)


def add_shadow(cells, res, eye, length):
    """`cells` plus every cell up to `length` behind them, seen from `eye`.

    The camera sees the front surface of an object and nothing behind it;
    the octomap holds that surface as a skin one voxel thick. Without the
    space behind it, a plan may swing the arm round the far side of an
    object -- through the part nobody has seen. Front approach only, so
    the hidden side is simply treated as solid.
    """
    if not len(cells) or length <= 0:
        return cells
    centres = (cells + 0.5) * res
    rays = centres - eye
    rays /= np.maximum(1e-9, np.linalg.norm(rays, axis=1))[:, None]
    steps = np.arange(res / 2.0, length + 1e-9, res / 2.0)
    behind = centres[:, None, :] + rays[:, None, :] * steps[None, :, None]
    behind = np.floor(behind / res).astype(np.int64).reshape(-1, 3)
    return np.unique(np.concatenate([cells, behind]), axis=0)


_FACES = [  # (direction, the face's four corner offsets, counter-clockwise from outside)
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
]


def _keys(ijk):
    b = ijk + (1 << 20)
    return (b[:, 0] << 42) | (b[:, 1] << 21) | b[:, 2]


def cells_mesh(cells, res):
    """Closed surface of a set of cells -> (vertices, faces): only the faces
    between an occupied cell and a free one, corners shared, so the mesh is
    watertight and a fraction of the size of one box per cell."""
    if not len(cells):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    keys = np.sort(_keys(cells))
    quads = []
    for direction, corners in _FACES:
        exposed = cells[~_in_sorted(_keys(cells + direction), keys)]
        quads.append(exposed[:, None, :] + np.array(corners)[None])
    corners = np.concatenate(quads)                       # (F, 4, 3) integer corners
    uniq, inverse = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    quad = inverse.reshape(-1, 4)
    faces = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])
    return uniq * res, faces


def _in_sorted(values, sorted_keys):
    idx = np.clip(np.searchsorted(sorted_keys, values), 0, len(sorted_keys) - 1)
    return sorted_keys[idx] == values


class ShadowedMap:
    """The octomap as cuMotion sees it: cropped to the arms' reach, the
    space behind each surface (from the camera) filled in, and carved clear
    round the grippers. The shadowed cells are kept between plans while the
    map and the camera stay put."""

    SHADOW = float(os.environ.get('CUMOTION_SHADOW', '0.12'))   # metres behind a surface

    def __init__(self):
        self._key = None
        self._cells = np.zeros((0, 3), np.int64)
        self._res = 0.02

    def cells(self, centres, sizes, res, eye, crop):
        key = (len(centres), float(np.sum(centres)) if len(centres) else 0.0, res,
               None if eye is None else tuple(np.round(eye, 3)))
        if key != self._key:
            keep = crop(centres)
            cells = voxel_cells(centres[keep], sizes[keep], res)
            surface = len(cells)
            if eye is not None:
                cells = add_shadow(cells, res, np.asarray(eye, float), self.SHADOW)
            self._key, self._cells, self._res = key, cells, res
            self.counts = (surface, len(cells))
        return self._cells

    def mesh(self, centres, sizes, res, eye, crop, keep_clear, clear_radius=0.10,
             spheres=None, sphere_margin=0.03):
        """spheres: (N, 4) x y z r -- the arm where it is; cells within r +
        sphere_margin of any are cleared too (the arm is physically there)."""
        cells = self.cells(centres, sizes, res, eye, crop)
        if len(cells) and (len(keep_clear) or spheres is not None):
            centre = (cells + 0.5) * res
            keep = np.ones(len(cells), bool)
            for p in keep_clear:
                keep &= np.linalg.norm(centre - p, axis=1) > clear_radius
            if spheres is not None:
                for x, y, z, r in np.asarray(spheres, float):
                    if r > 0:
                        keep &= np.linalg.norm(centre - (x, y, z), axis=1) > r + sphere_margin
            cells = cells[keep]
        return cells_mesh(cells, res), len(cells)


class OctomapCache:
    """decode_octomap() once per distinct map, not once per plan."""

    def __init__(self):
        self._key = None
        self._value = (np.zeros((0, 3)), np.zeros(0))

    def get(self, octomap_msg):
        key = hashlib.sha1(bytes(bytearray(x & 0xFF for x in octomap_msg.data))).hexdigest() \
            if octomap_msg.data else None
        if key != self._key:
            self._key = key
            self._value = decode_octomap(octomap_msg)
        return self._value
