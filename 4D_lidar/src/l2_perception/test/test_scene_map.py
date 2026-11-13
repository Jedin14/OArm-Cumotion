import numpy as np
import pytest

from l2_perception.scene_map import (BandedVoxelMap, band_average, band_edges, band_of,
                                     split_bands)

SIZES = [0.005, 0.01, 0.02, 0.04, 0.08]
L2_DENSITY = 3700.0          # points / sr / s, measured on the real L2 (UART)


def test_band_edges_give_constant_angular_resolution():
    e = band_edges(SIZES, 60.0, L2_DENSITY)
    assert e.size == 4 and np.all(np.diff(e) > 0)
    assert np.allclose(np.asarray(SIZES[:-1]) / e, SIZES[0] / e[0])   # size / range constant
    assert band_of(np.array([0.1, e[0] - 1e-6, e[0] + 1e-6, 1e3]), e).tolist() == [0, 0, 1, 4]
    with pytest.raises(ValueError):
        BandedVoxelMap([0.02, 0.01])


def test_insert_merges_and_averages():
    m = BandedVoxelMap(SIZES, 60.0, L2_DENSITY)
    p = np.array([[1.0012, 0.0, 0.0], [1.0018, 0.0, 0.0], [20.0, 0.0, 0.0]])
    r = np.linalg.norm(p, axis=1)
    m.insert(p[:1], np.array([10.0]), r[:1], 0.0)
    m.insert(p, np.array([20.0, 30.0, 40.0]), r, 1.0)
    c, inten, band = m.points()
    assert len(m) == 2 and np.all(np.diff(m.keys) > 0)
    assert np.allclose(c[0], [(1.0012 * 2 + 1.0018) / 3, 0, 0]) and np.isclose(inten[0], 20.0)
    assert band.tolist() == [band_of(r[:1], m.edges)[0], 4]
    assert m.cnt.tolist() == [3, 1]


def test_forget_and_touch():
    m = BandedVoxelMap(SIZES, 10.0, L2_DENSITY)          # forget_time = 20 s
    m.insert(np.array([[2.0, 0, 0]]), np.zeros(1), np.array([2.0]), 0.0)
    m.insert(np.array([[0, 3.0, 0]]), np.zeros(1), np.array([3.0]), 15.0)
    m.forget(25.0)
    assert len(m) == 1                                  # the first was not seen for 25 s
    m.touch(1000.0)
    m.forget(1010.0)
    assert len(m) == 1


def _wall_coverage(dist, seconds, sizes, fill_time, splat=1.5, half=0.2, seed=0):
    """Fraction of a wall patch (facing the lidar at `dist`) covered by the drawn squares."""
    rng = np.random.default_rng(seed)
    m = BandedVoxelMap(sizes, fill_time, L2_DENSITY)
    per_s = L2_DENSITY * (2 * half) ** 2 / dist ** 2    # points the scan puts on the patch
    for t in range(int(seconds)):
        n = rng.poisson(per_s)
        p = np.column_stack([np.full(n, dist) + rng.normal(0, 0.003, n),
                             rng.uniform(-half, half, n), rng.uniform(-half, half, n)])
        m.insert(p, np.zeros(n), np.linalg.norm(p, axis=1), float(t))
    c, _, band = m.points()
    res = 0.002
    g = int(2 * half / res)
    cov = np.zeros((g, g), dtype=bool)
    for (y, z), b in zip(c[:, 1:], band):
        h = splat * sizes[b] / 2
        i0, i1 = int((y - h + half) / res), int(np.ceil((y + h + half) / res))
        j0, j1 = int((z - h + half) / res), int(np.ceil((z + h + half) / res))
        cov[max(i0, 0):max(i1, 0), max(j0, 0):max(j1, 0)] = True
    inner = slice(g // 10, g - g // 10)                  # ignore the patch border
    return cov[inner, inner].mean(), sizes[band_of(np.array([dist]), m.edges)[0]]


@pytest.mark.parametrize("dist", [0.8, 1.9, 3.5, 7.0])
def test_wall_is_gap_free_after_fill_time_at_every_range(dist):
    cov, size = _wall_coverage(dist, 60, SIZES, 60.0)
    assert cov > 0.95, (dist, size, cov)


def test_old_fixed_1cm_view_left_gaps_beyond_3m():
    """The previous view (1 cm everywhere, 20 s, 1.2 cm squares): 96 % at 1.9 m, 67 % at
    3.5 m, 24 % at 7 m."""
    assert _wall_coverage(3.5, 20, [0.01], 60.0, splat=1.2)[0] < 0.8
    assert _wall_coverage(7.0, 20, [0.01], 60.0, splat=1.2)[0] < 0.35


def test_band_average_and_split():
    e = band_edges([0.01, 0.04], 3.0, L2_DENSITY)
    p = np.array([[0.5, 0, 0], [0.5004, 0, 0], [6.0, 0, 0], [6.01, 0, 0]])
    c, v, band = band_average(p, np.array([1.0, 3.0, 5.0, 7.0]), np.linalg.norm(p, axis=1), e,
                              [0.01, 0.04])
    assert band.tolist() == [0, 1] and np.allclose(v, [2.0, 6.0])
    assert split_bands(band, 2) == [(0, 1), (1, 2)]
