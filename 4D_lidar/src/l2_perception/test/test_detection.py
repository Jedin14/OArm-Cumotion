import numpy as np

from l2_perception.background import VoxelBackground
from l2_perception.detection import Tracker, cluster
from l2_perception.voxel import connected_components, dilate, isin_sorted, pack, unpack, voxel_keys


def room_frame(rng, n=6000):
    """One sparse frame of a static room (floor, 2 walls, a pillar)."""
    k = n // 4
    floor = np.column_stack([rng.uniform(-10, 10, k), rng.uniform(-10, 10, k), rng.normal(0, 0.01, k)])
    w1 = np.column_stack([np.full(k, 10.0) + rng.normal(0, 0.01, k), rng.uniform(-10, 10, k), rng.uniform(0, 3, k)])
    w2 = np.column_stack([rng.uniform(-10, 10, k), np.full(k, -10.0) + rng.normal(0, 0.01, k), rng.uniform(0, 3, k)])
    pil = np.column_stack([rng.uniform(3, 3.5, k), rng.uniform(3, 3.5, k), rng.uniform(0, 3, k)])
    return np.vstack([floor, w1, w2, pil])


def box_points(rng, lo, hi, n=300):
    return rng.uniform(lo, hi, size=(n, 3))


def test_pack_unpack_roundtrip():
    idx = np.array([[0, 0, 0], [-5, 7, -1000000], [1000000, -1000000, 3]])
    assert np.array_equal(unpack(pack(idx)), idx)


def test_dilate_and_membership():
    k = voxel_keys(np.array([[0.05, 0.05, 0.05]]), 0.1)
    d = dilate(np.sort(k), 1)
    assert d.size == 27
    q = np.array([[0.15, 0.15, 0.15], [0.35, 0.05, 0.05]])
    assert list(isin_sorted(voxel_keys(q, 0.1), d)) == [True, False]


def test_connected_components_separates_blobs():
    rng = np.random.default_rng(0)
    a = rng.uniform([0, 0, 0], [1, 1, 1], (200, 3))
    b = rng.uniform([3, 0, 0], [4, 1, 1], (200, 3))
    lab, n = connected_components(np.vstack([a, b]), 0.2)
    assert n == 2
    assert len(set(lab[:200])) == 1 and len(set(lab[200:])) == 1 and lab[0] != lab[200]


def test_background_then_detect_new_object():
    rng = np.random.default_rng(1)
    bg = VoxelBackground(voxel_size=0.1, slice_time=1.0, min_slices=2, dilation=1)
    bg.start_learning(10.0)
    t = 0.0
    done = False
    while not done:
        frame = room_frame(rng)
        if 2.0 < t < 2.4:  # a person walks through during learning (transient)
            frame = np.vstack([frame, box_points(rng, [0, 0, 0], [0.5, 0.5, 1.7])])
        done = bg.add_frame(frame, t)
        t += 0.1
    assert bg.ready

    # static scene only: (almost) nothing is foreground
    static = np.vstack([room_frame(rng) for _ in range(3)])
    fg = static[~bg.is_background(static)]
    fg = fg[fg[:, 2] > 0.1]
    assert cluster(fg, np.array([0, 0, 3.0]), min_points=8) == []

    # a new box appears where the transient person was: must be detected
    obj = box_points(rng, [0.0, 0.0, 0.0], [0.6, 0.4, 1.0])
    scene = np.vstack([static, obj])
    fg = scene[~bg.is_background(scene)]
    fg = fg[fg[:, 2] > 0.1]
    cl = cluster(fg, np.array([0, 0, 3.0]), min_points=8)
    assert len(cl) == 1
    c = cl[0]
    assert np.allclose(c.centroid[:2], [0.3, 0.2], atol=0.1)
    assert 0.8 < c.size[2] < 1.0
    assert c.num_points > 200


def test_background_save_load(tmp_path):
    rng = np.random.default_rng(2)
    bg = VoxelBackground()
    bg.start_learning(3.0)
    t = 0.0
    while not bg.add_frame(room_frame(rng), t):
        t += 0.1
    f = str(tmp_path / "bg.npz")
    bg.save(f, "abc")
    bg2 = VoxelBackground()
    assert bg2.load(f, "abc") == ""
    assert np.array_equal(bg2.keys, bg.keys)
    assert bg2.load(f, "different") != ""          # mount changed -> refuse stale model
    assert VoxelBackground(voxel_size=0.2).load(f) != ""


def test_tracker_ids_and_confirmation():
    rng = np.random.default_rng(3)
    tr = Tracker(gate=1.0, min_hits=3, max_missed_time=0.5)
    ids = []
    for k in range(10):
        p = box_points(rng, [k * 0.1, 0, 0], [k * 0.1 + 0.5, 0.5, 1.7])
        tracks = tr.update(cluster(p, np.zeros(3)), k * 0.1)
        assert len(tracks) == 1
        ids.append(tracks[0].id)
        assert tracks[0].confirmed == (k >= 2)
    assert len(set(ids)) == 1
    assert abs(tracks[0].velocity[0] - 1.0) < 0.3
    # disappears longer than max_missed_time -> new id afterwards
    tr.update([], 2.0)
    t2 = tr.update(cluster(box_points(rng, [0, 0, 0], [0.5, 0.5, 1.7]), np.zeros(3)), 2.1)
    assert t2[0].id != ids[0]
