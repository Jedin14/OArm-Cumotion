import numpy as np
import pytest

from l2_perception.geometry import rpy_to_matrix
from l2_perception.mount import MountCalibration, calibrate, tilt_change_deg

G = 9.80665


def make_scene(R_wl, height, n=20000, seed=0, accel_bias=(0, 0, 0)):
    """Floor + two walls seen from a lidar at (0,0,height) with orientation R_wl."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(-8, 8, n), rng.uniform(-8, 8, n), rng.normal(0, 0.01, n)])
    wall1 = np.column_stack([np.full(n // 4, 8.0), rng.uniform(-8, 8, n // 4), rng.uniform(0, 3, n // 4)])
    wall2 = np.column_stack([rng.uniform(-8, 8, n // 4), np.full(n // 4, -6.0), rng.uniform(0, 3, n // 4)])
    table = np.column_stack([rng.uniform(1, 2, n // 8), rng.uniform(1, 2, n // 8), np.full(n // 8, 0.8)])
    world = np.vstack([floor, wall1, wall2, table])
    t = np.array([0.0, 0.0, height])
    pts_l = (world - t) @ R_wl               # world -> lidar
    acc = R_wl.T @ np.array([0, 0, G]) + np.array(accel_bias)
    return pts_l, acc


@pytest.mark.parametrize("rpy_deg,height", [
    ((0.0, 0.0, 0.0), 1.2),          # upright on a pole
    ((3.0, -4.0, 30.0), 2.0),        # tilted bracket
    ((182.0, -3.0, 25.0), 3.0),      # hanging upside-down from the ceiling
    ((-90.0, 0.0, 0.0), 2.5),        # on a wall, looking sideways
])
def test_levels_any_mount(rpy_deg, height):
    R = rpy_to_matrix(*np.deg2rad(rpy_deg))
    pts, acc = make_scene(R, height)
    cal = calibrate(acc, pts, log=lambda *_: None)
    world = cal.apply(pts)
    floor = world[np.abs(world[:, 2]) < 0.1]
    assert floor.shape[0] > 15000
    assert abs(np.median(floor[:, 2])) < 0.01                   # floor at z = 0
    assert abs(cal.z - height) < 0.02
    # walls stay vertical: fitted normal of floor points is +z
    c = floor - floor.mean(axis=0)
    n = np.linalg.svd(c, full_matrices=False)[2][2]
    assert np.rad2deg(np.arccos(abs(n[2]))) < 0.2


def test_ground_reference_cancels_accelerometer_bias():
    R = rpy_to_matrix(*np.deg2rad([2.0, 1.0, 0.0]))
    pts, acc = make_scene(R, 2.0, accel_bias=(0.2, -0.15, 0.0))   # ~1.5 deg of accel bias
    g = calibrate(acc, pts, level_reference="gravity", log=lambda *_: None)
    f = calibrate(acc, pts, level_reference="ground", log=lambda *_: None)

    def floor_tilt(cal):
        fl = cal.apply(pts[:20000])          # make_scene puts the n floor points first
        n = np.linalg.svd(fl - fl.mean(axis=0), full_matrices=False)[2][2]
        return np.rad2deg(np.arccos(abs(n[2])))
    assert floor_tilt(g) > 0.8
    assert floor_tilt(f) < 0.1


def test_lowest_plane_is_ground_not_table():
    R = np.eye(3)
    pts, acc = make_scene(R, 1.5)
    cal = calibrate(acc, pts, log=lambda *_: None)
    assert abs(cal.z - 1.5) < 0.02


def test_yaml_roundtrip_and_tilt_monitor(tmp_path):
    R = rpy_to_matrix(*np.deg2rad([5.0, -2.0, 0.0]))
    pts, acc = make_scene(R, 2.0)
    cal = calibrate(acc, pts, yaw=np.deg2rad(15), x=1.0, log=lambda *_: None)
    f = tmp_path / "m.yaml"
    cal.save(str(f))
    back = MountCalibration.load(str(f))
    assert np.allclose(back.rotation(), cal.rotation(), atol=1e-9)
    assert np.allclose(back.translation(), cal.translation())
    assert tilt_change_deg(back, acc) < 1e-6
    R2 = rpy_to_matrix(*np.deg2rad([5.8, -2.0, 0.0]))       # bracket bent by 0.8 deg
    acc2 = R2.T @ np.array([0, 0, G])
    assert 0.7 < tilt_change_deg(back, acc2) < 0.9


def test_no_floor_falls_back_to_gravity():
    R = rpy_to_matrix(*np.deg2rad([4.0, 2.0, 0.0]))
    acc = R.T @ np.array([0, 0, G])
    cal = calibrate(acc, np.zeros((0, 3)), mount_height=2.2, log=lambda *_: None)
    assert cal.source == "gravity"
    assert abs(cal.z - 2.2) < 1e-9
    up = cal.rotation() @ (acc / np.linalg.norm(acc))
    assert np.allclose(up, [0, 0, 1], atol=1e-9)
