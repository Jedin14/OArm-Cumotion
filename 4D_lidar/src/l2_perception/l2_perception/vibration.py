"""Drift-free vibration compensation for a STATIONARY lidar.

Why this cannot drift
---------------------
SLAM / LIO (Point-LIO, FAST-LIO) integrate the IMU into a pose, so any bias,
timestamp jitter or bad initialisation becomes position drift that grows
without bound. A stationary lidar has a *known, constant* mean pose (the mount
calibration), so nothing needs to be integrated into a pose at all.

Vibration only adds a small, zero-mean, fast rotation on top of that constant
pose. It is estimated with a *leaky* (high-pass) integration of the gyro:

    theta_k = exp(-dt / tau) * theta_{k-1} + (omega_k - bias) * dt

Any error (gyro bias, noise, a bad sample, an IMU gap) decays with time
constant tau back to zero instead of accumulating, and |theta| is hard-clamped.
The gyro bias is tracked continuously as the long-term mean of the gyro,
which is exactly zero for a stationary sensor. The worst case is therefore
"no compensation", never "drift".

theta(t) is the rotation from the instantaneous (vibrating) lidar frame to its
nominal frame, so a point measured at time t is corrected with
    p_nominal = Exp(theta(t)) * p_measured
using its own per-point timestamp.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .geometry import rotate_by_rotvec


@dataclass
class VibrationConfig:
    enabled: bool = True
    highpass_time: float = 0.5     # [s] tau; cut-off 1/(2 pi tau) = 0.32 Hz
    bias_time: float = 30.0        # [s] gyro-bias tracker time constant
    bias_init_time: float = 1.0    # [s] initial bias average before compensating
    max_angle_deg: float = 2.0     # hard clamp on the estimated deflection
    max_gap: float = 0.05          # [s] IMU gap that resets the estimate to zero
    history: float = 3.0           # [s] of estimates kept for per-point lookup


class VibrationCompensator:
    def __init__(self, cfg: VibrationConfig = VibrationConfig()):
        self.cfg = cfg
        self._max_angle = np.deg2rad(cfg.max_angle_deg)
        self.bias = np.zeros(3)
        self._bias_ready = False
        self._bias_sum = np.zeros(3)
        self._bias_n = 0
        self._bias_t0 = None
        self.resets = 0
        self.dropped = 0
        self.clamped = 0
        self.reset(full=True)

    # ------------------------------------------------------------------ state
    def reset(self, full: bool = False):
        """Forget the deflection estimate. full=True also re-learns the gyro bias."""
        self._theta = np.zeros(3)
        self._last_t = None
        self._last_w = None
        self._t = deque()
        self._th = deque()
        if full:
            self._bias_ready = False
            self._bias_sum[:] = 0.0
            self._bias_n = 0
            self._bias_t0 = None

    @property
    def ready(self) -> bool:
        return self._bias_ready and len(self._t) >= 2

    def coverage_end(self):
        return self._t[-1] if self._t else None

    # ------------------------------------------------------------------ input
    def add_imu(self, t: float, gyro) -> None:
        w_raw = np.asarray(gyro, dtype=float)
        if self._last_t is not None:
            dt = t - self._last_t
            if dt <= 0.0:
                self.dropped += 1        # duplicate / out of order: ignore
                return
            if dt > self.cfg.max_gap:
                self.resets += 1         # cannot integrate across a gap: restart at zero
                self.reset(full=False)

        if not self._bias_ready:
            if self._bias_t0 is None:
                self._bias_t0 = t
            self._bias_sum += w_raw
            self._bias_n += 1
            if t - self._bias_t0 >= self.cfg.bias_init_time and self._bias_n >= 10:
                self.bias = self._bias_sum / self._bias_n
                self._bias_ready = True
            self._last_t = t
            self._last_w = None
            return

        if self._last_t is None:
            self._last_t = t
            self._last_w = w_raw - self.bias
            self._push(t)
            return

        dt = t - self._last_t
        # gyro bias = long-term mean of the gyro (true mean rate of a fixed mount is 0)
        self.bias += (dt / self.cfg.bias_time) * (w_raw - self.bias)
        w = w_raw - self.bias
        w_prev = self._last_w if self._last_w is not None else w
        self._theta = np.exp(-dt / self.cfg.highpass_time) * self._theta + 0.5 * (w + w_prev) * dt
        n = np.linalg.norm(self._theta)
        if n > self._max_angle:
            self._theta *= self._max_angle / n
            self.clamped += 1
        self._last_t = t
        self._last_w = w
        self._push(t)

    def _push(self, t):
        self._t.append(t)
        self._th.append(self._theta.copy())
        while len(self._t) > 2 and self._t[-1] - self._t[0] > self.cfg.history:
            self._t.popleft()
            self._th.popleft()

    # ------------------------------------------------------------------ output
    def rotvec_at(self, times: np.ndarray) -> np.ndarray:
        """Deflection estimate at each time (N,) -> (N,3). Held at the ends; zero if not ready."""
        times = np.asarray(times, dtype=float)
        if not self.cfg.enabled or not self.ready:
            return np.zeros((times.size, 3))
        t = np.fromiter(self._t, dtype=float, count=len(self._t))
        th = np.array(self._th)
        return np.stack([np.interp(times, t, th[:, i]) for i in range(3)], axis=1)

    def compensate(self, points: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Undo vibration for points (N,3) measured at absolute times (N,)."""
        if not self.cfg.enabled or not self.ready or points.shape[0] == 0:
            return points
        return rotate_by_rotvec(points, self.rotvec_at(times))

    def recent_amplitude_deg(self, window: float = 1.0) -> float:
        """RMS deflection over the last `window` seconds (diagnostics)."""
        if len(self._t) < 2:
            return 0.0
        t_end = self._t[-1]
        vals = [np.linalg.norm(th) for tt, th in zip(self._t, self._th) if t_end - tt <= window]
        return float(np.rad2deg(np.sqrt(np.mean(np.square(vals))))) if vals else 0.0
