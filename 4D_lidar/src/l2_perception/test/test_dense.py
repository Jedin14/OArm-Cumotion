import numpy as np

from l2_perception.dense_node import turbo, voxel_average


def test_voxel_average_merges_and_denoises():
    rng = np.random.default_rng(0)
    # a flat wall at x = 5 seen in 20 clouds with 2 cm range noise
    n = 20000
    p = np.column_stack([5.0 + rng.normal(0, 0.02, n), rng.uniform(-1, 1, n), rng.uniform(0, 2, n)])
    ids = np.repeat(np.arange(20), n // 20)
    out, inten = voxel_average(p, np.full(n, 10.0), ids, 0.05, 1)
    assert out.shape[0] < n / 3                         # merged: one point per voxel
    assert np.std(out[:, 0]) < 1.2 * np.std(p[:, 0]) + 0.01   # surface stays where it was
    assert np.allclose(inten, 10.0)


def test_min_hits_drops_single_cloud_voxels():
    static = np.tile(np.array([[1.0, 1.0, 1.0]]), (10, 1))
    mover = np.array([[3.0, 0.0, 1.0]])                 # seen in one cloud only
    p = np.vstack([static, mover])
    ids = np.concatenate([np.arange(10), [4]])
    out, _ = voxel_average(p, np.zeros(11), ids, 0.1, 2)
    assert out.shape[0] == 1 and np.allclose(out[0], [1, 1, 1])


def test_turbo_near_red_far_blue():
    near, far = turbo(np.array([0.95, 0.1]))            # the section dense_node uses
    assert near[0] > near[2] and far[2] > far[0]


def test_depth_panorama_and_auto_elevation():
    from l2_perception.dense_node import auto_elevation, depth_panorama
    rng = np.random.default_rng(1)
    # upright lidar in a cylinder-shaped room of radius 4 m: only elevations 0..60 deg seen
    az = rng.uniform(-np.pi, np.pi, 200000)
    el = rng.uniform(0, np.deg2rad(60), 200000)
    r = 4.0 / np.cos(el)
    rel = np.column_stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)])
    lo, hi = auto_elevation(rel)
    assert -10 <= np.rad2deg(lo) <= 0 and 60 <= np.rad2deg(hi) <= 70
    d = depth_panorama(rel, r, 512, 128, lo, hi, 2)
    filled = np.isfinite(d)
    assert filled.mean() > 0.8
    # bottom rows (horizon) are ~4 m, top rows farther
    assert abs(np.nanmedian(np.where(filled[-20:], d[-20:], np.nan)) - 4.2) < 0.4


# --------------------------------------------------------------------------- static / live layers
L2_DENSITY = 3700.0          # points / sr / s, measured on the real L2 (UART)


def _person_frames(rng, s_of_t, R=2.5, frames=40, dt=0.1):
    """Front face (0.5 m wide, 1.7 m tall) of a person on a circle of radius R around the lidar,
    sampled like the L2 scan. s_of_t: arc position of the body centre. Returns points, stamps,
    `now` and the arc coordinate of every point."""
    per_frame = int(L2_DENSITY * (0.5 * 1.7) / R ** 2 * dt)
    pts, ts, arc = [], [], []
    for k in range(frames):
        t = k * dt
        n = rng.poisson(per_frame)
        a = s_of_t(t) + rng.uniform(-0.25, 0.25, n)    # arc coordinate of each hit
        phi = a / R
        r = R + rng.normal(0, 0.003, n)
        pts.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), rng.uniform(0.0, 1.7, n)]))
        ts.append(np.full(n, t))
        arc.append(a)
    return np.concatenate(pts), np.concatenate(ts), (frames - 1) * dt, np.concatenate(arc)


def _tau(r):
    from l2_perception.dense_node import confirm_time
    return confirm_time(r, 0.15, L2_DENSITY, 4.0, 0.2, 1.5)


def test_confirm_time_grows_with_range_and_is_clamped():
    t = _tau(np.array([0.5, 2.5, 4.0, 30.0]))
    assert t[0] == 0.2 and t[-1] == 1.5
    assert 0.2 < t[2] < 1.5 and t[1] <= t[2]


def test_live_layer_walking_person_leaves_no_trail():
    from l2_perception.dense_node import live_select
    rng = np.random.default_rng(3)
    v = 1.2                                             # brisk walk across the view
    p, t, now, arc = _person_frames(rng, lambda t: v * t)
    idx = live_select(p, t, now, np.zeros(3), 3.0, 0.15, _tau)
    body = v * now
    assert arc[idx].min() > body - 0.25 - 0.7           # trail behind the body < 0.7 m ...
    assert arc[idx].max() < body + 0.25 + 0.01
    plain = arc[now - t <= 3.0]                         # ... a plain 3 s exposure: 3.6 m
    assert plain.min() < body - 0.25 - 3.0
    assert idx.size > 100                               # the current body is still well covered


def test_live_layer_standing_person_accumulates_detail():
    from l2_perception.dense_node import live_select
    rng = np.random.default_rng(4)
    p, t, now, _ = _person_frames(rng, lambda t: 0.0)
    held = np.flatnonzero(now - t <= 3.0)
    idx = live_select(p, t, now, np.zeros(3), 3.0, 0.15, _tau)
    assert idx.size > 0.92 * held.size                  # ~3 s of points kept on a still person
    one_frame = np.count_nonzero(t == now)
    assert idx.size > 20 * one_frame


def test_static_mask_routes_background_and_obstacle_boxes():
    from l2_perception.dense_node import static_mask
    from l2_perception.voxel import dilate, voxel_keys
    rng = np.random.default_rng(5)
    wall = np.column_stack([np.full(3000, 5.0), rng.uniform(-2, 2, 3000), rng.uniform(0, 2, 3000)])
    bg = dilate(np.unique(voxel_keys(wall, 0.1)), 1)
    person = np.column_stack([np.full(300, 3.0), rng.uniform(-0.25, 0.25, 300), rng.uniform(0, 1.7, 300)])
    leaning = np.array([[4.93, 1.0, 1.0]])              # in the dilated wall, but part of a person
    pts = np.vstack([wall, person, leaning])
    box = (np.array([4.75, 1.0, 0.9]), np.array([0.3, 0.5, 1.6]), 0.0)
    s = static_mask(pts, bg, 0.1, [])
    assert s[:3000].all() and not s[3000:3300].any() and s[-1]
    s = static_mask(pts, bg, 0.1, [box])
    assert not s[-1]                                    # inside a tracked box -> live
    behind = np.abs(wall[:, 1] - 1.0) <= 0.25 + 0.15    # wall hidden behind the person's box
    assert s[:3000][~behind].all() and s[:3000][behind].mean() < 0.2
    s = static_mask(pts, None, 0.1, [box])              # no background model: boxes only
    assert s[3000:3300].all() and not s[-1]


def test_live_layer_box_gate_cuts_the_trail_far_away():
    """At 5 m the scan needs ~1.2 s to re-check a cell; the tracked box knows sooner."""
    from l2_perception.dense_node import in_boxes, live_select
    rng = np.random.default_rng(6)
    v, R = 1.2, 5.0
    p, t, now, arc = _person_frames(rng, lambda t: v * t, R=R)
    no_gate = live_select(p, t, now, np.zeros(3), 3.0, 0.15, _tau)
    lag = 0.15                                          # box from a 0.3 s detection window
    phi = (v * now - lag) / R
    box = (np.array([R * np.cos(phi), R * np.sin(phi), 0.875]), np.array([0.3, 0.5 + 2 * lag, 1.75]),
           phi)                                         # box x along the radius, y along the walk
    gate = in_boxes(p, [box], 0.15, 0.3)
    gated = live_select(p, t, now, np.zeros(3), 3.0, 0.15, _tau, gate, 0.2)
    trail = lambda idx: (v * now - 0.25) - arc[idx].min()
    assert trail(no_gate) > 0.9
    assert trail(gated) < 0.5
    on_body = np.flatnonzero(np.abs(arc - v * now) <= 0.25)
    fresh_on_body = on_body[now - t[on_body] <= 0.2]
    assert np.isin(fresh_on_body, gated).all()          # the body itself is fully shown


def test_live_layer_box_gate_keeps_detail_of_a_standing_person():
    from l2_perception.dense_node import in_boxes, live_select
    rng = np.random.default_rng(7)
    p, t, now, _ = _person_frames(rng, lambda t: 0.0)
    box = (np.array([2.5, 0.0, 0.875]), np.array([0.3, 0.5, 1.75]), 0.0)
    gate = in_boxes(p, [box], 0.15, 0.3)
    idx = live_select(p, t, now, np.zeros(3), 3.0, 0.15, _tau, gate, 0.2)
    assert idx.size > 0.92 * np.count_nonzero(now - t <= 3.0)
    gone = live_select(p, t, now, np.zeros(3), 3.0, 0.15, _tau, np.zeros(len(p), bool), 0.2)
    assert gone.size == np.count_nonzero(now - t <= 0.2)   # no box any more: only the latest


def test_static_mask_keeps_floor_under_a_box_static():
    from l2_perception.dense_node import static_mask
    from l2_perception.voxel import dilate, voxel_keys
    g = np.random.default_rng(8)
    floor = np.column_stack([g.uniform(-2, 2, 4000), g.uniform(-2, 2, 4000), g.normal(0, 0.005, 4000)])
    bg = dilate(np.unique(voxel_keys(floor, 0.1)), 1)
    feet = np.array([[0.0, 0.0, 0.12], [0.1, 0.0, 0.08]])  # shoes in the dilated floor zone
    box = (np.array([0.0, 0.0, 0.9]), np.array([0.4, 0.4, 1.6]), 0.0)   # body above 0.1 m
    s = static_mask(np.vstack([floor, feet]), bg, 0.1, [box], 0.15, 0.3, 0.05)
    assert s[:4000].all()                               # floor under the person stays static
    assert not s[4000:].any()                           # feet go live


def test_height_colors_red_low_yellow_mid_green_high():
    from l2_perception.dense_node import height_colors, pack_rgb
    c = height_colors(np.array([-5.0, 0.0, 1.0, 2.0, 9.0]), 0.0, 2.0)
    assert c[0].tolist() == [255, 0, 0] and c[1].tolist() == [255, 0, 0]
    assert c[2].tolist() == [255, 255, 0]
    assert c[3].tolist() == [0, 255, 0] and c[4].tolist() == [0, 255, 0]
    assert pack_rgb(c[:1]).view(np.uint32)[0] == 0xFF0000
