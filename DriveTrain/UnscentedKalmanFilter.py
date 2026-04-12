import numpy as np


class UKF:
    """
    Unscented Kalman Filter for a skid-steer robot.

    State vector: [x, y, theta]
        x, y   - position in meters
        theta  - heading in radians, kept in (-pi, pi]

    Coordinate convention:
        +X = right, +Y = down, +theta = CCW

    Three inputs drive the filter:
        predict()       - called on every encoder tick (high rate, ~50 ms)
        update_uwb()    - called on every UWB fix (low rate, ~2 s)
        update_accel()  - called on every IMU heading fix (~50 ms)
    """

    # UKF tuning knobs
    ALPHA = 1e-3   # spread of sigma points around mean (small = tightly packed)
    BETA  = 2.0    # optimal for Gaussian priors
    KAPPA = 0.0    # secondary scaling; 0 is a safe default for state estimation

    # UKF intermediate parameters
    L       = 3
    _lambda = ALPHA**2 * (L + KAPPA) - L

    # Small nugget added to P before Cholesky to guard against numerical drift
    # making the covariance matrix non-positive-definite during UWB dropouts.
    _P_NUGGET = 1e-4

    def __init__(self, initial_state: np.ndarray, P0: np.ndarray, Q: np.ndarray,
                 R_uwb: np.ndarray, R_accel: np.ndarray):
        self.x      = initial_state.copy().astype(np.float64)
        self.P      = P0.copy().astype(np.float64)
        self.Q      = Q.copy().astype(np.float64)
        self.R_uwb  = R_uwb.copy().astype(np.float64)
        self.R_accel = R_accel.copy().astype(np.float64)

        self._wm = np.full(2 * self.L + 1, 0.5 / (self.L + self._lambda))
        self._wc = np.full(2 * self.L + 1, 0.5 / (self.L + self._lambda))
        self._wm[0] = self._lambda / (self.L + self._lambda)
        self._wc[0] = self._lambda / (self.L + self._lambda) + (1 - self.ALPHA**2 + self.BETA)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(theta: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return float(np.arctan2(np.sin(theta), np.cos(theta)))

    def _mean_angle(self, angles: np.ndarray, weights: np.ndarray) -> float:
        """
        Compute a weighted circular mean of angles.
        Naively averaging angles is wrong near +/-pi -- this isn't.
        """
        sin_sum = np.dot(weights, np.sin(angles))
        cos_sum = np.dot(weights, np.cos(angles))
        return float(np.arctan2(sin_sum, cos_sum))

    def _sigma_points(self) -> np.ndarray:
        """
        Generate 2L+1 sigma points from current mean and covariance.

        P is symmetrised before the Cholesky decomposition to correct for
        any floating-point asymmetry that accumulates over many predict steps.
        A small diagonal nugget (_P_NUGGET * I) is added to guarantee
        positive-definiteness during extended UWB dropouts.
        """

        P_safe = (self.P + self.P.T) / 2.0
        P_safe += self._P_NUGGET * np.eye(self.L)

        try:
            A = np.linalg.cholesky((self.L + self._lambda) * P_safe)
        except np.linalg.LinAlgError:
            print("[UKF] WARNING: P not positive definite after nugget -- resetting covariance")
            self.P = np.diag([0.5, 0.5, 0.1])
            P_safe = (self.P + self.P.T) / 2.0 + self._P_NUGGET * np.eye(self.L)
            A = np.linalg.cholesky((self.L + self._lambda) * P_safe)

        sp = np.empty((2 * self.L + 1, self.L))
        sp[0] = self.x
        for i in range(self.L):
            sp[i + 1]          = self.x + A[:, i]
            sp[i + 1 + self.L] = self.x - A[:, i]
        return sp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, v_l: float, v_r: float, wheel_base: float, dt: float):
        """
        Prediction step -- driven by left/right wheel speeds from encoders.

        Convention: +X right, +Y down, +theta CCW.
          x_new = x + v * cos(theta) * dt
          y_new = y - v * sin(theta) * dt   (minus because +Y is down)
          w = (v_r - v_l) / wheelbase       (CCW positive: right faster -> left turn)
        """
        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / wheel_base

        sp = self._sigma_points()

        sp_pred = np.empty_like(sp)
        for i, s in enumerate(sp):
            sp_pred[i, 0] = s[0] + v * np.cos(s[2]) * dt
            sp_pred[i, 1] = s[1] - v * np.sin(s[2]) * dt   # Y-down
            sp_pred[i, 2] = self._wrap_angle(s[2] + w * dt)

        x_pred    = np.dot(self._wm, sp_pred)
        x_pred[2] = self._mean_angle(sp_pred[:, 2], self._wm)

        P_pred = self.Q.copy()
        for i in range(2 * self.L + 1):
            d    = sp_pred[i] - x_pred
            d[2] = self._wrap_angle(d[2])
            P_pred += self._wc[i] * np.outer(d, d)

        self.x = x_pred
        self.P = P_pred

    def update_uwb(self, uwb_x: float, uwb_y: float):
        """UWB Update Step -- x,y position fix."""
        z  = np.array([uwb_x, uwb_y])
        sp = self._sigma_points()

        z_sigma = sp[:, :2]
        z_pred  = np.dot(self._wm, z_sigma)

        Pyy = self.R_uwb.copy()
        Pxy = np.zeros((self.L, 2))

        for i in range(2 * self.L + 1):
            dz    = z_sigma[i] - z_pred
            dx    = sp[i] - self.x
            dx[2] = self._wrap_angle(dx[2])
            Pyy  += self._wc[i] * np.outer(dz, dz)
            Pxy  += self._wc[i] * np.outer(dx, dz)

        K         = Pxy @ np.linalg.inv(Pyy)
        self.x    = self.x + K @ (z - z_pred)
        self.x[2] = self._wrap_angle(self.x[2])
        self.P    = self.P - K @ Pyy @ K.T

    def update_accel(self, accel_theta: float):
        """
        IMU heading update step.

        Expects accel_theta in the same convention as the state: CCW positive.
        If your IMU returns a different convention, negate before calling this.
        """
        # z  = np.array([accel_theta])
        # sp = self._sigma_points()

        # z_sigma = sp[:, 2]
        # z_pred  = self._mean_angle(z_sigma, self._wm)

        # Pyy = self.R_accel.copy()
        # Pxy = np.zeros((self.L, 1))

        # for i in range(2 * self.L + 1):
        #     dz   = self._wrap_angle(z_sigma[i] - z_pred)
        #     dx   = sp[i] - self.x
        #     dx[2] = self._wrap_angle(dx[2])
        #     Pyy += self._wc[i] * np.array([[dz**2]])
        #     Pxy += self._wc[i] * dx.reshape(-1, 1) * dz

        # K          = Pxy / Pyy[0, 0]
        # innovation = self._wrap_angle(float(z[0]) - z_pred)
        # self.x     = self.x + K.flatten() * innovation
        # self.x[2]  = self._wrap_angle(self.x[2])
        # self.P     = self.P - K @ Pyy @ K.T
        self.x[2] = accel_theta

    @property
    def state(self) -> np.ndarray:
        """Return a copy of the current state estimate [x, y, theta]."""
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Return a copy of the current covariance matrix."""
        return self.P.copy()