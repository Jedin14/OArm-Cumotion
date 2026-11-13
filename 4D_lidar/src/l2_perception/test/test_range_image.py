import numpy as np

from l2_perception.range_image import (RangeImageMap, fill_holes, grid_points, remove_speckles)


def _dir(az_deg, el_deg):
    a, e = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _pixel(m, az_deg, el_deg):
    pix, _ = m.pixels(_dir(az_deg, el_deg)[None])
    return divmod(int(pix[0]), m.W)


def test_front_surface_skips_one_off_hits_and_averages_noise():
    rng = np.random.default_rng(0)
    m = RangeImageMap(0.2)
    d = _dir(30.1, 10.1)
    wall = (3.0 + rng.normal(0, 0.003, 20))[:, None] * d       # 20 hits, sigma 3 mm
    m.insert(wall, 0.0)
    m.insert(1.5 * d[None], 1.0)                               # dust: one hit in front
    lone = _dir(-60.1, 20.1)
    m.insert(2.0 * lone[None], 1.0)                            # a pixel seen only once
    img, v0 = m.depth()
    v, u = _pixel(m, 30.1, 10.1)
    assert abs(img[v - v0, u] - 3.0) < 0.002                   # mean of the wall, dust skipped
    v, u = _pixel(m, -60.1, 20.1)
    assert np.isclose(img[v - v0, u], 2.0)                     # too sparse to judge: kept
    assert np.isfinite(img).sum() == 2


def test_nearer_solid_surface_wins():
    m = RangeImageMap(0.2)
    d = _dir(100.1, -5.1)
    m.insert(np.full((10, 1), 4.0) * d, 0.0)
    m.insert(np.array([[2.0], [2.004], [1.997]]) * d, 0.0)
    img, v0 = m.depth()
    v, u = _pixel(m, 100.1, -5.1)
    assert abs(img[v - v0, u] - 2.0) < 0.003


def test_forget_and_touch():
    m = RangeImageMap(0.2, forget_time=20.0)
    m.insert(2.0 * _dir(0.1, 0.1)[None], 0.0)
    m.insert(3.0 * _dir(90.1, 0.1)[None], 15.0)
    m.forget(25.0)
    assert len(m) == 1
    m.touch(1000.0)
    m.forget(1010.0)
    assert len(m) == 1


def test_fill_bridges_scan_line_gaps_but_not_depth_edges():
    H, W = 30, 40
    wall = np.full((H, W), 3.0, dtype=np.float32) + np.linspace(0, 0.06, H)[:, None]   # tilted
    truth = wall.copy()
    truth[:, 25:] = 1.5                                        # a box in front, right side
    lines = np.full((H, W), np.inf, dtype=np.float32)
    lines[::3] = truth[::3]                                    # only every 3rd row is scanned
    out = fill_holes(lines, max_gap=3)
    inner = slice(0, 28)                                       # rows bracketed by two lines
    assert np.isfinite(out[inner]).mean() > 0.99
    assert np.abs(out[inner] - truth[inner])[np.isfinite(out[inner])].max() < 0.005
    # across the vertical box edge: no depths in between foreground and background
    vals = out[np.isfinite(out)]
    assert not np.any((vals > 1.6) & (vals < 2.9))


def test_fill_does_not_bridge_wide_gaps_and_wraps_azimuth():
    d = np.full((5, 20), np.inf, dtype=np.float32)
    d[2, 0], d[2, 6] = 2.0, 2.0                                # 5 empty pixels between: too wide
    d[2, 18], d[2, 1] = 2.0, 2.0                               # 19, (0), 1: across the 360 deg seam
    out = fill_holes(d, max_gap=3)
    assert np.isfinite(out[2, 19]) and np.isclose(out[2, 19], 2.0)
    assert not np.isfinite(out[2, 3])


def test_speckles_removed_thin_lines_kept():
    d = np.full((10, 20), np.inf, dtype=np.float32)
    d[2, 2] = 1.0                                              # lone speckle
    d[6, 2:13] = 2.0                                           # one-pixel-wide line (cable)
    d[6, 7] = 5.0                                              # a stray range on the line
    out = remove_speckles(d)
    assert not np.isfinite(out[2, 2]) and not np.isfinite(out[6, 7])
    assert np.isfinite(out[6, [3, 4, 5, 9, 10, 11]]).all()


def test_grid_points_lie_on_pixel_rays_and_thin_towards_the_pole():
    m = RangeImageMap(0.5)
    pts = np.array([2.0 * _dir(45.0, 30.0), 5.0 * _dir(-170.0, -20.0)])
    m.insert(pts, 0.0)
    img, v0 = m.depth()
    rel, r, vv, uu = grid_points(m, img, v0)
    assert rel.shape == (2, 3) and np.allclose(sorted(r), [2.0, 5.0])
    for p in pts:                                              # within half a pixel of the hit
        q = rel[np.argmin(np.linalg.norm(rel - p, axis=1))]
        ang = np.arccos(np.clip(q @ p / np.linalg.norm(q) / np.linalg.norm(p), -1, 1))
        assert np.rad2deg(ang) < 0.5 * 0.5 * np.sqrt(2) + 1e-6
    keep = m.pole_keep(0, m.H)
    assert keep[m.H // 2].all() and keep[0].sum() < 10         # equator: all, near zenith: few
    full = np.ones((m.H, m.W), dtype=np.float32)
    rel, _, vv, _ = grid_points(m, full, 0)
    per_row = np.bincount(vv, minlength=m.H)
    el = np.pi / 2 - (np.arange(m.H) + 0.5) * m.dv
    assert np.abs(per_row - m.W * np.cos(el)).max() <= 1.0       # W cos(el) points per row


def test_l2_like_scan_lines_become_a_gap_free_surface():
    """The L2 scans fixed elevation lines ~0.6 deg apart (measured; the pattern repeats). Waiting
    never fills the rows between them; the image cloud bridges them on surfaces."""
    rng = np.random.default_rng(1)
    m = RangeImageMap(0.2)
    for t in range(30):                                        # 30 s, dense along each line
        el = np.repeat(np.arange(-9.9, 10.0, 0.6), 200)          # lines mid-row (not on a boundary)
        el = el + rng.normal(0, 0.01, el.size)
        az = rng.uniform(-10, 10, el.size)
        dirs = np.stack([_dir(a, e) for a, e in zip(az, el)])
        r = 3.0 / dirs[:, 0] + rng.normal(0, 0.003, el.size)   # wall x = 3 m
        m.insert(dirs * r[:, None], float(t))
    raw, v0 = m.depth()
    img = remove_speckles(fill_holes(raw, 3))
    rows = slice(_pixel(m, 0, 9.0)[0] - v0, _pixel(m, 0, -9.0)[0] - v0)
    cols = slice(_pixel(m, 9.0, 0)[1], _pixel(m, -9.0, 0)[1])
    assert np.isfinite(raw[rows, cols]).mean() < 0.5          # scan lines only
    assert np.isfinite(img[rows, cols]).mean() > 0.97         # one surface
    rel, r, vv, uu = grid_points(m, img, v0)
    sel = np.abs(rel[:, 1] / rel[:, 0]) < np.tan(np.deg2rad(8))
    sel &= np.abs(rel[:, 2]) < 3.0 * np.tan(np.deg2rad(8))
    assert np.abs(rel[sel, 0] - 3.0).max() < 0.02             # flat: noise averaged, no bulges
