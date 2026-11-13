"""Sparse voxel hashing and connected-component clustering (numpy only)."""

import itertools

import numpy as np

_BITS = 21
_OFF = 1 << (_BITS - 1)          # +-1,048,576 voxels per axis
_MASK = (1 << _BITS) - 1
_SX = np.int64(1) << np.int64(2 * _BITS)
_SY = np.int64(1) << np.int64(_BITS)


def voxel_indices(points: np.ndarray, size: float) -> np.ndarray:
    return np.floor(points[:, :3] / size).astype(np.int64)


def pack(idx: np.ndarray) -> np.ndarray:
    i = idx + _OFF
    return (i[:, 0] << (2 * _BITS)) | (i[:, 1] << _BITS) | i[:, 2]


def unpack(keys: np.ndarray) -> np.ndarray:
    k = np.asarray(keys, dtype=np.int64)
    return np.stack([(k >> (2 * _BITS)) & _MASK, (k >> _BITS) & _MASK, k & _MASK], axis=1) - _OFF


def voxel_keys(points: np.ndarray, size: float) -> np.ndarray:
    return pack(voxel_indices(points, size))


def key_offsets(radius: int = 1, half: bool = False) -> np.ndarray:
    """Additive key offsets of the (2r+1)^3 neighbourhood (excluding 0).

    Packed keys are linear in the indices, so key(i + d) == key(i) + offset(d).
    half=True keeps one of each +-d pair (enough for building undirected edges).
    """
    offs = []
    for d in itertools.product(range(-radius, radius + 1), repeat=3):
        if d == (0, 0, 0):
            continue
        if half and d < (0, 0, 0):
            continue
        offs.append(d[0] * _SX + d[1] * _SY + d[2])
    return np.array(offs, dtype=np.int64)


def isin_sorted(keys: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    if sorted_set.size == 0:
        return np.zeros(keys.shape, dtype=bool)
    pos = np.searchsorted(sorted_set, keys)
    pos = np.minimum(pos, sorted_set.size - 1)
    return sorted_set[pos] == keys


def dilate(sorted_keys: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0 or sorted_keys.size == 0:
        return sorted_keys
    offs = np.concatenate([[0], key_offsets(radius)])
    return np.unique((sorted_keys[:, None] + offs[None, :]).ravel())


def connected_components(points: np.ndarray, voxel_size: float, radius: int = 1):
    """Label points by 26-connected (radius 1) occupied voxels.

    Returns (labels (N,), n_components). Labels are 0..n-1.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    keys = voxel_keys(points, voxel_size)
    uniq, inv = np.unique(keys, return_inverse=True)
    v = uniq.size
    src, dst = [], []
    for off in key_offsets(radius, half=True):
        nb = uniq + off
        pos = np.minimum(np.searchsorted(uniq, nb), v - 1)
        hit = uniq[pos] == nb
        if np.any(hit):
            src.append(np.flatnonzero(hit))
            dst.append(pos[hit])
    lab = np.arange(v, dtype=np.int64)
    if src:
        a = np.concatenate(src)
        b = np.concatenate(dst)
        while True:
            m = np.minimum(lab[a], lab[b])
            new = lab.copy()
            np.minimum.at(new, a, m)
            np.minimum.at(new, b, m)
            # pointer jumping
            while True:
                j = new[new]
                if np.array_equal(j, new):
                    break
                new = j
            if np.array_equal(new, lab):
                break
            lab = new
    _, comp = np.unique(lab, return_inverse=True)
    return comp[inv].astype(np.int64), int(comp.max()) + 1 if v else 0
