import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np

class UKF:
    """
    Unscented Kalman Filter for a skid-steer robot.

    State vector: [x, y, theta]
        x, y   - position in meters
        theta  - heading in radians, kept in (-pi, pi]

    Three inputs drive the filter:
        predict()       - called on every encoder tick (high rate, ~50 ms)
        update_uwb()    - called on every UWB fix (low rate, ~2 s)
        update_accel()  - called on every Accelerometer fix (faster rate, ~50 ms)
    """

    # UKF tuning knobs -------------------------------------------------------
    ALPHA = 1e-3   # spread of sigma points around mean (small = tightly packed)
    BETA  = 2.0    # optimal for Gaussian priors
    KAPPA = 0.0    # secondary scaling; 0 is a safe default for state estimation
    
    # UKF intermediate parameters --------------------------------------------
    L = 3
    _lambda = ALPHA**2 * (L + KAPPA) - L
    
    def __init__(self, initial_state: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_uwb: np.ndarray, R_accel: np.ndarray):
        """
        Parameters
        ----------
        initial_state : (3,) array  [x0, y0, theta0]
        P0            : (3,3) initial state covariance
        Q             : (3,3) process noise covariance  (encoder / slip noise)
        R_uwb         : (2,2) UWB measurement noise covariance
        R_accel       : (1,1) Accelerometer measurement noise covariance
        """
        self.x = initial_state.copy().astype(np.float64)
        self.P = P0.copy().astype(np.float64)
        self.Q = Q.copy().astype(np.float64)
        self.R_uwb = R_uwb.copy().astype(np.float64)
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
        Naively averaging angles is wrong near ±pi — this isn't.
        """
        sin_sum = np.dot(weights, np.sin(angles))
        cos_sum = np.dot(weights, np.cos(angles))
        return float(np.arctan2(sin_sum, cos_sum))

    def _sigma_points(self) -> np.ndarray:
        """Generate 2L+1 sigma points from current mean and covariance."""
        # Cholesky of scaled covariance; clip for numerical safety
        A = np.linalg.cholesky((self.L + self._lambda) * self.P)
        sp = np.empty((2 * self.L + 1, self.L))
        sp[0] = self.x
        for i in range(self.L):
            sp[i + 1]     = self.x + A[:, i]
            sp[i + 1 + self.L] = self.x - A[:, i]
        return sp
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, v_l: float, v_r: float, wheel_base: float, dt: float):
        """
        Prediction step — driven by left/right wheel speeds from encoders.

        Parameters
        ----------
        v_l        : left  wheel linear speed  (m/s)
        v_r        : right wheel linear speed  (m/s)
        wheel_base : track width               (m)
        dt         : time elapsed since last call (s)
        """
        
        # Robot-level velocity commands from differential kinematics
        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / wheel_base

        sp = self._sigma_points()  

        # Propagate each sigma point through the nonlinear motion model
        sp_pred = np.empty_like(sp)
        for i, s in enumerate(sp):
            sp_pred[i, 0] = s[0] + v * np.cos(s[2]) * dt
            sp_pred[i, 1] = s[1] + v * np.sin(s[2]) * dt
            sp_pred[i, 2] = self._wrap_angle(s[2] + w * dt)

        # Predicted mean (use circular mean for the angle component)
        x_pred = np.dot(self._wm, sp_pred)
        x_pred[2] = self._mean_angle(sp_pred[:, 2], self._wm)

        # Predicted covariance + process noise
        P_pred = self.Q.copy()
        for i in range(2 * self.L + 1):
            d = sp_pred[i] - x_pred
            d[2] = self._wrap_angle(d[2])          # angle residual must be wrapped
            P_pred += self._wc[i] * np.outer(d, d)

        self.x = x_pred
        self.P = P_pred

    def update_uwb(self, uwb_x: float, uwb_y: float):
        """
        UWB Update Step — driven by x,y,theta estimated values from UWB Anchors.

        Parameters
        ----------
        uwb_x        : Estimated X position of UWB anchors (m)
        uwb_y        : Estimated Y position of UWB anchors (m)
        """     
    
        z = np.array([uwb_x, uwb_y])
        sp = self._sigma_points()

        # Predicted measurement mean
        z_sigma = sp[:, :2]
        z_pred = np.dot(self._wm, z_sigma)

        # Innovation covariance Pyy and cross-covariance Pxy
        Pyy = self.R_uwb.copy()
        Pxy = np.zeros((self.L, 2))

        for i in range (2 * self.L + 1):
            dz = z_sigma[i] - z_pred
            dx = sp[i] - self.x
            dx[2] = self._wrap_angle(dx[2])
            Pyy += self._wc[i] * np.outer(dz, dz)
            Pxy += self._wc[i] * np.outer(dx, dz)

        K = Pxy @ np.linalg.inv(Pyy)
        innovation = z - z_pred
        self.x = self.x + K @ innovation
        self.x[2] = self._wrap_angle(self.x[2])
        self.P = self.P - K @ Pyy @ K.T

    def update_accel(self, accel_theta: float):
        """
        Accelerometer Update Step — driven by theta estimated values from Accelerometer.

        Parameters
        ----------
        accel_theta    : Estimate theta from accelerometer (rad)
        """     

        z = np.array([accel_theta])
        sp = self._sigma_points()

        # Predicted measurement mean
        z_sigma = sp[:, 2]
        z_pred = self._mean_angle(z_sigma, self._wm)  # circular mean for θ

        # Innovation covariance Pyy and cross-covariance Pxy
        Pyy = self.R_accel.copy()
        Pxy = np.zeros((self.L, 1))

        for i in range (2 * self.L + 1):
            dz = self._wrap_angle(z_sigma[i] - z_pred)
            dx = sp[i] - self.x
            dx[2] = self._wrap_angle(dx[2])
            Pyy += self._wc[i] * np.array([[dz**2]])
            Pxy += self._wc[i] * dx.reshape(-1, 1) * dz

        K = Pxy / Pyy
        innovation = self._wrap_angle(float(z[0]) - z_pred)
        self.x = self.x + K.flatten() * innovation
        self.x[2] = self._wrap_angle(self.x[2])
        self.P = self.P - K @ Pyy @ K.T

    @property
    def state(self) -> np.ndarray:
        """Return a copy of the current state estimate [x, y, theta]."""
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Return a copy of the current covariance matrix."""
        return self.P.copy()
