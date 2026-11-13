import numpy as np
from std_msgs.msg import Header

from l2_perception import pointcloud as pc
from l2_perception.stamp_guard import StampGuard


def test_cloud_roundtrip():
    n = 1000
    rng = np.random.default_rng(0)
    cols = {"x": rng.normal(size=n).astype(np.float32), "y": rng.normal(size=n).astype(np.float32),
            "z": rng.normal(size=n).astype(np.float32),
            "intensity": rng.uniform(0, 255, n).astype(np.float32),
            "time": np.linspace(0, 0.1, n).astype(np.float32),
            "ring": rng.integers(0, 18, n).astype(np.uint16)}
    msg = pc.make_cloud(Header(frame_id="f"), cols)
    assert msg.point_step == 22 and msg.width == n
    arr = pc.to_struct(msg)
    for k, v in cols.items():
        assert np.array_equal(arr[k], v)
    assert np.allclose(pc.xyz(arr)[:, 0], cols["x"])
    assert np.all(pc.field_or(arr, "missing", 7.0) == 7.0)


def test_driver_layout_with_padding():
    # l2_driver layout: x y z intensity time (float32) ring (uint16) + 2 pad bytes = 24
    from sensor_msgs.msg import PointCloud2, PointField
    n = 3
    raw = np.zeros(n, dtype=np.dtype({"names": ["x", "y", "z", "intensity", "time", "ring"],
                                      "formats": ["<f4"] * 5 + ["<u2"],
                                      "offsets": [0, 4, 8, 12, 16, 20], "itemsize": 24}))
    raw["x"] = [1, 2, 3]
    raw["time"] = [0.0, 0.05, 0.1]
    m = PointCloud2(height=1, width=n, point_step=24, row_step=24 * n, is_dense=True)
    m.fields = [PointField(name=k, offset=o, datatype=t, count=1) for k, o, t in
                [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7), ("time", 16, 7), ("ring", 20, 4)]]
    m.data = raw.tobytes()
    a = pc.to_struct(m)
    assert list(a["x"]) == [1, 2, 3] and np.isclose(a["time"][2], 0.1)


def test_stamp_guard():
    g = StampGuard("x", max_jump=1.0)
    assert [g.check(t) for t in (10.0, 10.1, 10.1, 10.0, 15.0, 15.1)] == \
        ["ok", "ok", "duplicate", "backward", "jump", "ok"]
