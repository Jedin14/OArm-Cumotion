import numpy as np

from l2_perception.geometry import rotate_by_rotvec, rotvec_to_matrix
from l2_perception.vibration import VibrationCompensator, VibrationConfig


def simulate(duration, rate=400.0, amp_deg=(0.15, 0.10, 0.05), freq=(7.0, 12.0, 3.0),
             bias_dps=(0.3, -0.2, 0.1), noise=0.003, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / rate)
    amp = np.deg2rad(amp_deg)
    ph = rng.uniform(0, 2 * np.pi, 3)
    arg = 2 * np.pi * np.array(freq)[None] * t[:, None] + ph[None]
    v = amp[None] * np.sin(arg)
    w = amp[None] * 2 * np.pi * np.array(freq)[None] * np.cos(arg)
    gyro = w + np.deg2rad(bias_dps)[None] + rng.normal(0, noise, w.shape)
    return t, v, gyro


def test_compensates_vibration():
    t, v, gyro = simulate(30.0)
    vc = VibrationCompensator()
    for ti, g in zip(t, gyro):
        vc.add_imu(1.7e9 + ti, g)
    # evaluate over the last 2 s at random instants
    q = 1.7e9 + np.random.default_rng(1).uniform(t[-1] - 2.0, t[-1], 2000)
    est = vc.rotvec_at(q)
    truth = np.stack([np.interp(q - 1.7e9, t, v[:, i]) for i in range(3)], axis=1)
    err = np.rad2deg(np.linalg.norm(est - truth, axis=1))
    raw = np.rad2deg(np.linalg.norm(truth, axis=1))
    assert np.sqrt(np.mean(err ** 2)) < 0.25 * np.sqrt(np.mean(raw ** 2))  # >4x reduction


def test_point_correction_end_to_end():
    t, v, gyro = simulate(10.0, seed=3)
    vc = VibrationCompensator()
    for ti, g in zip(t, gyro):
        vc.add_imu(ti, g)
    # a wall point 10 m away observed at many instants from the vibrating lidar
    p_true = np.array([10.0, 2.0, -1.0])
    times = np.linspace(8.0, 9.9, 500)
    vv = np.stack([np.interp(times, t, v[:, i]) for i in range(3)], axis=1)
    measured = np.array([rotvec_to_matrix(x).T @ p_true for x in vv])
    corrected = vc.compensate(measured, times)
    spread_raw = np.linalg.norm(measured - p_true, axis=1).max()
    spread_fix = np.linalg.norm(corrected - p_true, axis=1).max()
    assert spread_raw > 0.025
    assert spread_fix < 0.35 * spread_raw


def test_cannot_drift_with_bias_and_no_vibration():
    # one simulated hour of a perfectly still sensor with a large, changing gyro bias
    rate = 200.0
    vc = VibrationCompensator()
    rng = np.random.default_rng(7)
    max_angle = 0.0
    n = int(3600 * rate)
    for k in range(n):
        tt = k / rate
        bias = np.deg2rad([0.5, -0.4, 0.3]) * (1 + 0.5 * np.sin(2 * np.pi * tt / 900))
        vc.add_imu(tt, bias + rng.normal(0, 0.003, 3))
        if k % 200 == 0 and vc.ready:
            max_angle = max(max_angle, float(np.linalg.norm(vc.rotvec_at(np.array([tt]))[0])))
    # bounded by (bias-tracking lag) x tau ~ 0.05 deg for this extreme 0.25 deg/s bias swing;
    # an integrating (LIO-style) estimator would be off by hundreds of degrees after an hour
    assert np.rad2deg(max_angle) < 0.1


def test_bounded_even_with_garbage_gyro():
    vc = VibrationCompensator(VibrationConfig(max_angle_deg=1.0))
    for k in range(4000):
        vc.add_imu(k * 0.0025, np.array([5.0, -3.0, 2.0]) if k > 400 else np.zeros(3))
    assert np.rad2deg(np.linalg.norm(vc.rotvec_at(np.array([10.0]))[0])) <= 1.0 + 1e-9


def test_gap_resets_and_duplicates_ignored():
    vc = VibrationCompensator()
    t = 0.0
    for _ in range(800):
        vc.add_imu(t, np.array([0.0, 0.0, 0.0]))
        t += 0.0025
    vc.add_imu(t - 0.0025, np.array([9.0, 9.0, 9.0]))   # duplicate / backwards: ignored
    assert vc.dropped == 1
    vc.add_imu(t + 0.5, np.zeros(3))                     # 0.5 s gap
    assert vc.resets == 1


def test_disabled_is_identity():
    vc = VibrationCompensator(VibrationConfig(enabled=False))
    p = np.random.default_rng(0).normal(size=(10, 3))
    assert np.array_equal(vc.compensate(p, np.zeros(10)), p)


def test_rotate_by_rotvec_matches_matrix():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(50, 3))
    rv = rng.normal(scale=0.3, size=(50, 3))
    ref = np.array([rotvec_to_matrix(r) @ x for r, x in zip(rv, p)])
    assert np.allclose(rotate_by_rotvec(p, rv), ref)
