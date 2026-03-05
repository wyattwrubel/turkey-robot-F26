import sys
import os
sys.path.append(os.path.dirname(__file__))
import DifferentialDrive as DD
import UWBDriver
import DCMotorDriver
import numpy as np
import asyncio
import time

class UKF:
    """
    Unscented Kalman Filter for a skid-steer robot.

    State vector: [x, y, theta]
        x, y   - position in meters
        theta  - heading in radians, kept in (-pi, pi]

    Two inputs drive the filter:
        predict()       - called on every encoder tick (high rate, ~50 ms)
        update_uwb()    - called on every UWB fix (low rate, ~2 s)
    """

    # UKF tuning knobs -------------------------------------------------------
    ALPHA = 1e-3   # spread of sigma points around mean (small = tightly packed)
    BETA  = 2.0    # optimal for Gaussian priors
    KAPPA = 0.0    # secondary scaling; 0 is a safe default for state estimation

    def __init__(self, initial_state: np.ndarray, P0: np.ndarray, Q: np.ndarray, R_uwb: np.ndarray):
        """
        Parameters
        ----------
        initial_state : (3,) array  [x0, y0, theta0]
        P0            : (3,3) initial state covariance
        Q             : (3,3) process noise covariance  (encoder / slip noise)
        R_uwb         : (2,2) UWB measurement noise covariance
        """
        self.n = 3
        self.x = initial_state.copy().astype(np.float64)
        self.P = P0.copy().astype(np.float64)
        self.Q = Q.copy().astype(np.float64)
        self.R_uwb = R_uwb.copy().astype(np.float64)

        # Pre-compute lambda and weights (they never change)
        n = self.n
        lam = self.ALPHA**2 * (n + self.KAPPA) - n
        self._lambda = lam

        self._wm = np.full(2 * n + 1, 0.5 / (n + lam))
        self._wc = np.full(2 * n + 1, 0.5 / (n + lam))
        self._wm[0] = lam / (n + lam)
        self._wc[0] = lam / (n + lam) + (1 - self.ALPHA**2 + self.BETA)

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
        """Generate 2n+1 sigma points from current mean and covariance."""
        n = self.n
        # Cholesky of scaled covariance; clip for numerical safety
        A = np.linalg.cholesky((n + self._lambda) * self.P)
        sp = np.empty((2 * n + 1, n))
        sp[0] = self.x
        for i in range(n):
            sp[i + 1]     = self.x + A[:, i]
            sp[i + 1 + n] = self.x - A[:, i]
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
        for i in range(2 * self.n + 1):
            d = sp_pred[i] - x_pred
            d[2] = self._wrap_angle(d[2])          # angle residual must be wrapped
            P_pred += self._wc[i] * np.outer(d, d)

        self.x = x_pred
        self.P = P_pred

    def update_uwb(self, uwb_x: float, uwb_y: float, uwb_theta: float):
        z = np.array([uwb_x, uwb_y, uwb_theta])

        sp = self._sigma_points()

        # Measurement model: now h(s) = [x, y, θ] — full state observable
        z_sigma = sp[:, :3]                        # shape (2n+1, 3)

        # Predicted measurement mean
        z_pred = np.dot(self._wm, z_sigma)         # shape (3,)
        z_pred[2] = self._mean_angle(sp[:, 2], self._wm)  # circular mean for θ

        # Innovation covariance S and cross-covariance Pxy
        S   = self.R_uwb.copy()                    # (3,3)
        Pxy = np.zeros((self.n, 3))                # (3,3)
        for i in range(2 * self.n + 1):
            dz = z_sigma[i] - z_pred               # (3,)
            dz[2] = self._wrap_angle(dz[2])        # wrap θ residual in measurement too
            dx = sp[i] - self.x                    # (3,)
            dx[2] = self._wrap_angle(dx[2])
            S   += self._wc[i] * np.outer(dz, dz)
            Pxy += self._wc[i] * np.outer(dx, dz)

        # Kalman gain
        K = Pxy @ np.linalg.inv(S)                 # (3,3)

        # State and covariance update
        innovation = z - z_pred
        innovation[2] = self._wrap_angle(innovation[2])   # wrap θ innovation
        self.x = self.x + K @ innovation
        self.x[2] = self._wrap_angle(self.x[2])
        self.P = self.P - K @ S @ K.T

    @property
    def state(self) -> np.ndarray:
        """Return a copy of the current state estimate [x, y, theta]."""
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Return a copy of the current covariance matrix."""
        return self.P.copy()

# ---------------------------------------------------------------------------

class MotorDriver:
    def __init__(self, PosePublisher, Simulated=False, Verbose=False):
        self.PosePublisher  = PosePublisher
        self.SimulatedData  = Simulated
        self.Verbose        = Verbose

        if not Simulated:
            self.UWB      = UWBDriver.UWB()
            self.DCDriver = DCMotorDriver.DC_Driver()

        self.MotorSpeeds = np.array([0.0, 0.0], dtype=np.float64)

        # Timing constants
        self.WheelBase            = 0.5   # meters — REPLACE with real track width
        self.PosePeriod           = 0.2   # s
        self.UWBSamplePeriod      = 2.0   # s
        self.ReadMotorDataPeriod  = 0.05  # s
        self.UpdateMotorPeriod    = 0.2   # s

        self.running  = True
        self.UI_lock  = False

        # asyncio lock — prevents concurrent access to UKF state
        self._state_lock = asyncio.Lock()

        # ---- UKF initialisation ----
        initial_state = np.array([2.0, 2.0, 0.0])

        P0 = np.diag([0.5, 0.5, 0.1])          # initial position uncertainty ~0.7 m, heading ~0.3 rad
        
        # Process noise Q: tune higher if wheel slip is severe.
        # Units: m² for x/y, rad² for theta.
        Q  = np.diag([0.02, 0.02, 0.01])
        
        # UWB measurement noise R: reflects your anchor geometry & multipath.
        # 0.1 → std dev ≈ 0.32 m; tighten if your UWB is more accurate.
        R_uwb = np.diag([0.1, 0.1])

        self.ukf = UKF(initial_state, P0, Q, R_uwb)

    # ------------------------------------------------------------------
    # Command receiver
    # ------------------------------------------------------------------

    async def MotorCommandReceiver(self, data):
        try:
            cmd_id = data["Id"]
        except KeyError:
            print(f"DriveTrainDriver: 'Id' key missing in {data}")
            return

        if cmd_id == "UnlockUI":
            self.UI_lock = False
            return

        try:
            cmd_data = data["Cmd"]
        except KeyError:
            print(f"DriveTrainDriver: 'Cmd' key missing in {data}")
            return

        if cmd_id == "UI":
            self.UI_lock   = True
            self.MotorSpeeds = cmd_data
        elif cmd_id == "System":
            if not self.UI_lock:
                self.MotorSpeeds = cmd_data
        else:
            print(f"DriveTrainDriver: Invalid cmd_id '{cmd_id}' in {data}")

    # ------------------------------------------------------------------
    # Async runner — each coroutine owns its own loop
    # ------------------------------------------------------------------

    async def DriveTrainDriver_Runner(self):
        await asyncio.gather(
            self.RobotPosePublisher(),
            self.ReadUWBPosition(),
            self.UpdateMotorSpeeds(),
            self.ReadMotorPosition(),
        )

    async def RobotPosePublisher(self):
        while self.running:
            async with self._state_lock:
                state = self.ukf.state
            await self.PosePublisher.publish(state)
            await asyncio.sleep(self.PosePeriod)

    async def ReadUWBPosition(self):
        """
        Periodically request a UWB fix and fuse it into the UKF via the
        measurement update step.  We no longer overwrite the state —
        the filter blends UWB with encoder odometry automatically.
        """
        while self.running:
            # UpdatePosition now returns a raw (x, y) fix, not a full state
            uwb_x, uwb_y = self.UWB.UpdatePosition()

            async with self._state_lock:
                self.ukf.update_uwb(uwb_x, uwb_y)

            if self.Verbose:
                print(f"[UWB ] fix  x={uwb_x:.3f}  y={uwb_y:.3f}  "
                      f"state={np.round(self.ukf.state, 3)}")

            await asyncio.sleep(self.UWBSamplePeriod)

    async def UpdateMotorSpeeds(self):
        """Send commanded speeds to the motor driver (or simulate)."""
        while self.running:
            if self.SimulatedData:
                async with self._state_lock:
                    current = self.ukf.state
                new_state = DD.differential_drive(
                    current, self.MotorSpeeds, self.WheelBase, self.UpdateMotorPeriod
                )
                async with self._state_lock:
                    # In simulation the motion model IS the ground truth; 
                    # inject it directly so the predict step stays consistent.
                    self.ukf.x = new_state
            else:
                self.DCDriver.UpdateMotorSpeeds(self.MotorSpeeds)

            await asyncio.sleep(self.UpdateMotorPeriod)

    async def ReadMotorPosition(self):
        """
        High-rate encoder loop.  Drives the UKF prediction step.
        Only runs when using real hardware.
        """
        if self.SimulatedData:
            return

        last_time = time.time()
        while self.running:
            now          = time.time()
            dt           = now - last_time
            last_time    = now

            # AverageLinearSpeed() returns (v_left, v_right) in m/s
            v_l, v_r = self.DCDriver.AverageLinearSpeed()

            async with self._state_lock:
                self.ukf.predict(v_l, v_r, self.WheelBase, dt)

            if self.Verbose:
                print(f"[ENC ] v_l={v_l:.3f}  v_r={v_r:.3f}  dt={dt*1e3:.1f}ms  "
                      f"state={np.round(self.ukf.state, 3)}")

            await asyncio.sleep(self.ReadMotorDataPeriod)