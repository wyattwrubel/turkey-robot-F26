import sys
import os
sys.path.append(os.path.dirname(__file__))
import DCMotorDriver
import numpy as np
import asyncio
import time
import csv
from datetime import datetime


class MotorDriver:
    """
    Drivetrain manager for a 4-wheel skid-steer robot.

    Localization strategy:
        - Dead-reckoning only: [x, y, theta] integrated from encoder odometry.
        - Encoder predict step fires ONLY when driving straight
          (|v_l - v_r| / avg_speed < _straight_threshold).
          During turns, encoder slip is too high to trust -- heading comes
          exclusively from the accelerometer.
        - Accelerometer: directly sets theta as ground truth on every reading.
          No filtering, no weighted blend -- whatever the IMU says is the angle.
        - GPS (via Drivetrain_GPSReceiver, same interface as old UWB callback):
            * First fix ever: hard-sets x, y as the true origin.
            * Subsequent fixes: soft nudge -- pulls DR position toward GPS
              by a small alpha weight (_gps_alpha). GPS is inaccurate so
              alpha is kept small to avoid jerky corrections.

    No UKF, no covariance matrix, no uncertainty accumulation.

    Speed ramping:
        All motor commands pass through a ramp filter in UpdateMotorSpeeds.
        _current_speeds tracks what is actually being commanded to the hardware;
        MotorSpeeds is the target reference set by MotorCommandReceiver.

        Each tick, _current_speeds steps toward MotorSpeeds at a rate
        determined by the active ramp mode:

          - Normal accel  : ramp_rate_accel  (m/s2) -- default, gradual spin-up
          - Deceleration  : ramp_rate_decel  (m/s2) -- more aggressive, faster stop
          - Point turn    : ramp_rate_turn   (m/s2) -- slowest, protects traction
                            during in-place spins (v_l and v_r opposite signs,
                            both above point_turn_threshold)

        Both sides are stepped as a proportional unit each tick: the side
        requiring the largest step determines the rate, and the other side
        is scaled by the same factor. This preserves the left/right ratio
        throughout the ramp so the robot tracks the reference heading.

        Snap zone (both sides < snap_threshold = 1.0 m/s):
            Target is applied instantly with no ramp, except when both sides
            are also below DC_Driver.ol_zero_threshold (0.5 m/s), in which
            case DC_Driver sends an open-loop duty 0 command instead.

        Ramp mode selection per tick (priority order):
          1. Both sides below snap_threshold          -> instant snap
          2. Both target speeds zero                  -> decel rate
          3. v_l and v_r opposite signs, both above
             point_turn_threshold                     -> turn rate
          4. _current_speeds moving away from zero    -> accel rate
          5. Otherwise                                -> decel rate
    """

    # -- Localization parameters ---------------------------------------
    _straight_threshold = 0.15
    _gps_alpha          = 0.05

    # -- Timing periods (seconds) --------------------------------------
    PosePeriod          = 0.01
    ReadMotorDataPeriod = 0.05
    UpdateMotorPeriod   = 0.1
    ReadCurrentPeriod   = 0.1
    SlipPublishPeriod   = 1.0   # kept for external compatibility; no-op

    # -- Safety limits -------------------------------------------------
    _MaxSpeedAllowed = 8.7   # m/s -- matches DC_Driver.max_linear_speed

    # -- Speed ramp parameters ----------------------------------------
    #
    # All rates are in m/s per second (m/s2).
    # The ramp runs every UpdateMotorPeriod seconds, so per-tick step:
    #   step = rate * UpdateMotorPeriod
    #
    # ramp_rate_accel : spin-up rate for normal forward/reverse driving
    # ramp_rate_decel : speed reduction / stopping (more aggressive)
    # ramp_rate_turn  : in-place spin (opposite-sign wheels, slowest)
    # point_turn_threshold : min |v| on each side to enter turn mode
    #
    # snap_threshold:
    #   If BOTH target speeds are below this, ramp is bypassed and the
    #   target is applied instantly. Matches DC_Driver.snap_threshold.
    #   Below DC_Driver.ol_zero_threshold (0.5), DC_Driver sends OL duty 0.
    #
    ramp_rate_accel      = 5.0   # m/s2
    ramp_rate_decel      = 10.0   # m/s2
    ramp_rate_turn       = 1.0   # m/s2
    point_turn_threshold = 0.1   # m/s
    snap_threshold       = 1.0   # m/s -- must match DC_Driver.snap_threshold

    # -- Wheel geometry ------------------------------------------------
    WheelBase = 0.6   # metres between left and right contact patches

    # -- Simulation parameters ----------------------------------------
    _sim_omega_scale = 0.5

    def __init__(self, PosePublisher, SlipPublisher, Simulated=False, Verbose=False, EncoderLogging=False):
        self.PosePublisher  = PosePublisher
        self.SlipPublisher  = SlipPublisher   # kept for API compatibility; unused internally
        self.SimulatedData  = Simulated
        self.Verbose        = Verbose
        self.EncoderLogging = EncoderLogging

        # -- Dead-reckoning state -------------------------------------
        self._dr_state    = np.array([15.0, 75.0, 0], dtype=np.float64)
        self._gps_initialised = False

        # -- Runtime state --------------------------------------------
        self.MotorSpeeds     = np.array([0.0, 0.0], dtype=np.float64)  # target reference
        self._current_speeds = np.array([0.0, 0.0], dtype=np.float64)  # ramped command
        self.running         = True
        self.UI_lock         = False
        self._state_lock     = asyncio.Lock()

        # -- Encoder logger setup -------------------------------------
        self._log_file   = None
        self._log_writer = None
        if self.EncoderLogging and not self.SimulatedData:
            log_filename = datetime.now().strftime(
                "/home/turkeysquad/ME450/MainDesign/DriveTrain/Encoder/encoder_log_%Y%m%d_%H%M%S.csv"
            )
            self._log_file   = open(log_filename, "w", newline="")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow([
                "timestamp",
                "v_l_ms", "v_r_ms", "v_avg_ms",
                "is_straight",
                "dr_x", "dr_y", "dr_theta_deg",
            ])
            self._log_file.flush()
            print(f"[LOG  ] Encoder logging enabled -> {log_filename}")

        if not Simulated:
            self.DCDriver = DCMotorDriver.DC_Driver()

    # ------------------------------------------------------------------
    # Angle utility
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(a: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return float(np.arctan2(np.sin(a), np.cos(a)))

    def _log_row(self, v_l: float, v_r: float, going_straight: bool, state: np.ndarray) -> None:
        if not self.EncoderLogging or self._log_writer is None:
            return
        now   = datetime.now()
        ts    = now.strftime("%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        v_avg = (abs(v_l) + abs(v_r)) / 2.0
        self._log_writer.writerow([
            ts,
            f"{v_l:.4f}", f"{v_r:.4f}", f"{v_avg:.4f}",
            going_straight,
            f"{state[0]:.4f}", f"{state[1]:.4f}", f"{np.degrees(state[2]):.2f}",
        ])
        self._log_file.flush()

    def close(self) -> None:
        self.running = False
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file   = None
            self._log_writer = None
            print("[LOG  ] Encoder log closed.")

    # ------------------------------------------------------------------
    # Dead-reckoning integrator
    # ------------------------------------------------------------------
    def _dr_predict(self, v_l: float, v_r: float, dt: float) -> None:
        x, y, th = self._dr_state
        v = (v_l + v_r) / 2.0
        self._dr_state[0] = x + v * np.cos(th) * dt
        self._dr_state[1] = y - v * np.sin(th) * dt
        if self.SimulatedData:
            omega = (v_r - v_l) / self.WheelBase
            self._dr_state[2] = self._wrap(th + omega * self._sim_omega_scale * dt)

    @staticmethod
    def _is_driving_straight(v_l: float, v_r: float, threshold: float) -> bool:
        avg = (abs(v_l) + abs(v_r)) / 2.0
        if avg < 1e-4:
            return True
        return abs(v_l - v_r) / avg < threshold

    # ------------------------------------------------------------------
    # Speed ramp helper
    # ------------------------------------------------------------------
    def _ramp_speeds(self, current: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """
        Step _current_speeds toward target by one ramp tick.

        Snap zone (checked first):
            If both target speeds are below snap_threshold in magnitude,
            bypass the ramp entirely and apply the target instantly.
            This covers:
              - Both < ol_zero_threshold (0.5): DC_Driver will send OL duty 0
              - Both < snap_threshold (1.0):    DC_Driver sends CL cmd 35 instantly

        Ramp mode selection:
            1. Both targets zero or decelerating  -> decel rate
            2. Point turn (opposite signs, both   -> turn rate
               above point_turn_threshold)
            3. Any side accelerating              -> accel rate
            4. Otherwise                          -> decel rate

        Proportional stepping:
            Both sides scaled by the same factor each tick so the
            left/right ratio is always preserved during ramp.
        """
        t_l, t_r = target[0], target[1]
        c_l, c_r = current[0], current[1]

        # -- Snap zone -------------------------------------------------
        if abs(t_l) < self.snap_threshold and abs(t_r) < self.snap_threshold:
            return target.copy()

        # -- Select ramp rate ------------------------------------------
        both_zero    = (t_l == 0.0 and t_r == 0.0)
        point_turn   = (
            t_l * t_r < 0
            and abs(t_l) >= self.point_turn_threshold
            and abs(t_r) >= self.point_turn_threshold
        )
        accelerating = (
            abs(c_l) < abs(t_l) - 1e-6
            or abs(c_r) < abs(t_r) - 1e-6
        )

        if both_zero or not accelerating:
            rate = self.ramp_rate_decel
        elif point_turn:
            rate = self.ramp_rate_turn
        else:
            rate = self.ramp_rate_accel

        max_step = rate * dt

        # -- Proportional step -----------------------------------------
        delta         = target - current
        max_delta_mag = np.max(np.abs(delta))

        if max_delta_mag < 1e-9:
            return target.copy()

        if max_delta_mag <= max_step:
            return target.copy()
        else:
            scale = max_step / max_delta_mag
            return current + delta * scale

    # ------------------------------------------------------------------
    # Robot State Publisher
    # ------------------------------------------------------------------
    async def RobotPosePublisher(self):
        while self.running:
            async with self._state_lock:
                state = self._dr_state.copy()
            await self.PosePublisher.publish(state)
            if self.Verbose:
                print(f"[DR  ] state: x={state[0]:.3f}  y={state[1]:.3f}  "
                      f"th={np.degrees(state[2]):.1f}deg")
            await asyncio.sleep(self.PosePeriod)

    # ------------------------------------------------------------------
    # Receiver Functions
    # ------------------------------------------------------------------
    async def Drivetrain_GPSReceiver(self, data):
        gps = np.array(data, dtype=np.float64)
        gps_x, gps_y = gps[0], gps[1]

        async with self._state_lock:
            if not self._gps_initialised:
                self._dr_state[0] = gps_x
                self._dr_state[1] = gps_y
                self._gps_initialised = True
                if self.Verbose:
                    print(f"[GPS ] First fix -- origin set: x={gps_x:.3f}  y={gps_y:.3f}")
            else:
                self._dr_state[0] += self._gps_alpha * (gps_x - self._dr_state[0])
                self._dr_state[1] += self._gps_alpha * (gps_y - self._dr_state[1])
                if self.Verbose:
                    print(f"[GPS ] Nudge: gps=({gps_x:.3f}, {gps_y:.3f})  "
                          f"dr=({self._dr_state[0]:.3f}, {self._dr_state[1]:.3f})")

    async def Drivetrain_UWBReceiver(self, data):
        """Alias for Drivetrain_GPSReceiver -- routes old UWB callback to GPS logic."""
        await self.Drivetrain_GPSReceiver(data)

    async def Drivetrain_AccelReceiver(self, data):
        theta = float(np.squeeze(data))
        async with self._state_lock:
            self._dr_state[2] = self._wrap(theta)
        if self.Verbose:
            print(f"[ACCEL] heading set: {np.degrees(theta):.1f}deg")

    async def ReadMotorPosition(self):
        if self.SimulatedData:
            return

        last_time = time.time()
        while self.running:
            now       = time.time()
            dt        = now - last_time
            last_time = now

            v_l, v_r = self.DCDriver.AverageLinearSpeed()
            if v_l is None or v_r is None:
                await asyncio.sleep(self.ReadMotorDataPeriod)
                continue

            going_straight = self._is_driving_straight(v_l, v_r, self._straight_threshold)

            async with self._state_lock:
                if going_straight:
                    self._dr_predict(v_l, v_r, dt)
                state = self._dr_state.copy()

            self._log_row(v_l, v_r, going_straight, state)

            if self.Verbose:
                tag = "STRAIGHT" if going_straight else "TURNING "
                print(f"[ENC ] {tag} -- v_l={v_l:.3f}  v_r={v_r:.3f}  dt={dt*1e3:.1f}ms")

            await asyncio.sleep(self.ReadMotorDataPeriod)

    # ------------------------------------------------------------------
    # Motor Speed Updates
    # ------------------------------------------------------------------
    async def UpdateMotorSpeeds(self):
        """
        Ramp _current_speeds toward MotorSpeeds and send to hardware each tick.

        _ramp_speeds handles snap (< snap_threshold instant) and ramp selection.
        DC_Driver.UpdateMotorSpeeds handles the OL duty 0 zone (< ol_zero_threshold).
        """
        last_time = time.time()
        while self.running:
            now       = time.time()
            dt        = now - last_time
            last_time = now

            target = self.MotorSpeeds.copy()

            self._current_speeds = self._ramp_speeds(self._current_speeds, target, dt)

            safe_speeds = np.clip(
                self._current_speeds,
                -self._MaxSpeedAllowed,
                self._MaxSpeedAllowed,
            )

            if self.SimulatedData:
                async with self._state_lock:
                    self._dr_predict(safe_speeds[0], safe_speeds[1], self.UpdateMotorPeriod)
            else:
                self.DCDriver.UpdateMotorSpeeds(safe_speeds)

            if self.Verbose:
                tgt = self.MotorSpeeds
                cur = self._current_speeds
                print(f"[RAMP] target=({tgt[0]:.3f}, {tgt[1]:.3f})  "
                      f"cmd=({cur[0]:.3f}, {cur[1]:.3f})")

            await asyncio.sleep(self.UpdateMotorPeriod)

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    async def DriveTrainDriver_Runner(self):
        await asyncio.gather(
            self.RobotPosePublisher(),
            self.ReadMotorPosition(),
            self.UpdateMotorSpeeds(),
            self.SlipPublisherBroadcast(),
        )

    async def SlipPublisherBroadcast(self):
        """
        No-op stub kept for external API compatibility.
        Slip detection and stall monitoring handled by RoboClaw firmware.
        """
        while self.running:
            await asyncio.sleep(self.SlipPublishPeriod)

    # ------------------------------------------------------------------
    # Motor Command Receiver
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
            self.UI_lock = True
            self.MotorSpeeds = np.array(cmd_data, dtype=np.float64)
        elif cmd_id == "System":
            if not self.UI_lock:
                self.MotorSpeeds = np.array(cmd_data, dtype=np.float64)
        else:
            print(f"DriveTrainDriver: Invalid cmd_id '{cmd_id}' in {data}")