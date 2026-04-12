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
    """

    # ── Localization parameters ───────────────────────────────────────
    # Fraction threshold for straight-driving detection.
    # If |v_l - v_r| / avg_speed < this, encoders update localization.
    # 0.15 = 15% difference allowed before we consider it a turn.
    _straight_threshold = 0.15

    # GPS soft nudge weight per fix.
    # Each GPS reading pulls DR position by: dr += alpha * (gps - dr)
    # 0.05 = 5% trust per fix. Small because GPS is inaccurate.
    # Increase toward 0.2 if GPS is reliable; decrease toward 0.01 if noisy.
    _gps_alpha = 0.05

    # ── Timing periods (seconds) ──────────────────────────────────────
    PosePeriod          = 0.01
    ReadMotorDataPeriod = 0.05
    UpdateMotorPeriod   = 0.1
    ReadCurrentPeriod   = 0.1
    SlipPublishPeriod   = 1.0

    # ── Safety limits ─────────────────────────────────────────────────
    _MaxSpeedAllowed        = 3.0   # m/s hard clamp on motor commands
    _Max_Current_Threshold  = 40    # Amps -- stall detection ceiling
    _StallDebounceLimit     = 5     # consecutive over-current scans before stop

    # ── Slip detection ────────────────────────────────────────────────
    _SlipWarnThreshold = 0.4
    _SlipStopThreshold = 0.7
    _SlipDeadband      = 0.05   # m/s -- ignore slip check below this speed

    _SlipNormal = 1.0
    _SlipWarn   = 0.7
    _SlipStop   = 0.0

    # ── Wheel geometry (must match DCMotorDriver) ─────────────────────
    WheelBase = 0.6   # metres between left and right contact patches -- UPDATE THIS

    # ── Simulation parameters ─────────────────────────────────────────
    # Omega (turn rate) scale factor applied in simulated mode only.
    # 0.1 = 10% of ideal diff-drive turn rate, replicating heading drift
    # from wheel slip observed on real hardware. Increase toward 1.0 for
    # an ideal (slip-free) simulation; decrease toward 0.0 for heavy slip.
    _sim_omega_scale = 0.06

    def __init__(self, PosePublisher, SlipPublisher, Simulated=False, Verbose=False, EncoderLogging=False):
        self.PosePublisher  = PosePublisher
        self.SlipPublisher  = SlipPublisher
        self.SimulatedData  = Simulated
        self.Verbose        = Verbose
        self.EncoderLogging = EncoderLogging

        # ── Dead-reckoning state ──────────────────────────────────────
        # Default position until first GPS fix sets x, y as true origin.
        # Theta is always overridden by the accelerometer.
        self._dr_state = np.array([10.0, 75.0, 45.0], dtype=np.float64)  # [x, y, theta_rad]
        self._gps_initialised = False   # True after first GPS fix sets x, y

        # ── Runtime state ─────────────────────────────────────────────
        self.MotorSpeeds     = np.array([0.0, 0.0], dtype=np.float64)
        self._SlipModifier   = self._SlipNormal
        self._stall_debounce = 0
        self._last_currents  = [None, None, None, None]  # FL, FR, RL, RR -- updated by _CurrentMonitor
        self.running         = True
        self.UI_lock         = False
        self._state_lock     = asyncio.Lock()

        # ── Encoder logger setup ──────────────────────────────────────
        # File created only if EncoderLogging=True.
        # Named encoder_log_YYYYMMDD_HHMMSS.csv so each run is separate.
        self._log_file   = None
        self._log_writer = None
        if self.EncoderLogging:
            log_filename = datetime.now().strftime("encoder_log_%Y%m%d_%H%M%S.csv")
            self._log_file   = open(log_filename, "w", newline="")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow([
                "timestamp",
                "v_l_ms", "v_r_ms", "v_avg_ms",
                "is_straight",
                "dr_x", "dr_y", "dr_theta_deg",
                "current_fl_A", "current_fr_A", "current_rl_A", "current_rr_A",
                "slip_modifier",
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
        """Wrap angle to (-π, π]."""
        return float(np.arctan2(np.sin(a), np.cos(a)))

    def _log_row(self, v_l: float, v_r: float, going_straight: bool, state: np.ndarray) -> None:
        """
        Write one row to the encoder log CSV.
        Called every encoder read cycle (~50ms) when EncoderLogging=True.

        Columns:
            timestamp        -- DD HH:MM:SS.mmm
            v_l_ms           -- left wheel speed  (m/s, robot frame)
            v_r_ms           -- right wheel speed (m/s, robot frame)
            v_avg_ms         -- (|v_l| + |v_r|) / 2
            is_straight      -- True if DR predict fired this tick
            dr_x, dr_y       -- dead-reckoning position at this tick
            dr_theta_deg     -- heading in degrees (easier to read than radians)
            current_fl/fr/rl/rr_A  -- last known motor currents (100ms cache)
            slip_modifier    -- 1.0 / 0.7 / 0.0  (normal / warn / stop)
        """
        if not self.EncoderLogging or self._log_writer is None:
            return
        now = datetime.now()
        ts  = now.strftime("%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        v_avg = (abs(v_l) + abs(v_r)) / 2.0
        fl, fr, rl, rr = self._last_currents
        self._log_writer.writerow([
            ts,
            f"{v_l:.4f}", f"{v_r:.4f}", f"{v_avg:.4f}",
            going_straight,
            f"{state[0]:.4f}", f"{state[1]:.4f}", f"{np.degrees(state[2]):.2f}",
            f"{fl:.3f}" if fl is not None else "None",
            f"{fr:.3f}" if fr is not None else "None",
            f"{rl:.3f}" if rl is not None else "None",
            f"{rr:.3f}" if rr is not None else "None",
            f"{self._SlipModifier:.1f}",
        ])
        self._log_file.flush()

    def close(self) -> None:
        """
        Cleanly close the log file. Call this on shutdown.
        Stops the async runner loop and flushes/closes the CSV if open.
        """
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
        """
        Euler-integrate differential drive kinematics into _dr_state.

        Hardware mode (SimulatedData=False):
            Only called when driving straight (|v_l - v_r| / avg < threshold).
            Theta is NOT integrated here -- it comes exclusively from the
            accelerometer via Drivetrain_AccelReceiver.

        Simulation mode (SimulatedData=True):
            Full differential drive model -- x, y, AND theta are all integrated.
            Called unconditionally (no straight-only gate) since there is no
            real IMU to provide heading.
            Omega is scaled by _sim_omega_scale (default 0.1) to replicate the
            heading drift caused by wheel slip seen on real hardware.

        Convention: +X right, +Y down, +theta CCW.
          v     = (v_l + v_r) / 2
          omega = (v_r - v_l) / WheelBase
          x    += v * cos(theta) * dt
          y    -= v * sin(theta) * dt   (minus because +Y is down)
          theta += omega * dt  (sim only, scaled by _sim_omega_scale)
        """
        x, y, th = self._dr_state
        v = (v_l + v_r) / 2.0
        self._dr_state[0] = x + v * np.cos(th) * dt
        self._dr_state[1] = y - v * np.sin(th) * dt
        if self.SimulatedData:
            omega = (v_r - v_l) / self.WheelBase
            self._dr_state[2] = self._wrap(th + omega * self._sim_omega_scale * dt)
        # hardware: theta left unchanged -- accel is ground truth for heading

    @staticmethod
    def _is_driving_straight(v_l: float, v_r: float, threshold: float) -> bool:
        """
        Returns True if the speed differential is within the straight-driving
        threshold, meaning encoder odometry is reliable enough to use.

        Avoids division by zero when both speeds are near zero.
        """
        avg = (abs(v_l) + abs(v_r)) / 2.0
        if avg < 1e-4:
            return True   # essentially stopped -- no integrate needed but not turning
        return abs(v_l - v_r) / avg < threshold

    # ------------------------------------------------------------------
    # Robot State Publisher
    # ------------------------------------------------------------------
    async def RobotPosePublisher(self):
        """Publishes dead-reckoning pose at PosePeriod."""
        while self.running:
            async with self._state_lock:
                state = self._dr_state.copy()
            await self.PosePublisher.publish(state)
            if self.Verbose:
                print(f"[DR  ] state: x={state[0]:.3f}  y={state[1]:.3f}  "
                      f"th={np.degrees(state[2]):.1f}°")
            await asyncio.sleep(self.PosePeriod)

    # ------------------------------------------------------------------
    # Receiver Functions
    # ------------------------------------------------------------------
    async def Drivetrain_GPSReceiver(self, data):
        """
        GPS position fix receiver (replaces old UWB callback).

        First call ever: hard-sets _dr_state x, y to the GPS reading,
        establishing the true origin.

        Subsequent calls: soft nudge --
            dr_x += _gps_alpha * (gps_x - dr_x)
            dr_y += _gps_alpha * (gps_y - dr_y)

        GPS does not update theta -- heading comes from the accelerometer.

        Args:
            data: array-like [gps_x, gps_y]
        """
        gps = np.array(data, dtype=np.float64)
        gps_x, gps_y = gps[0], gps[1]

        async with self._state_lock:
            if not self._gps_initialised:
                # First fix: treat as ground truth, set true origin
                self._dr_state[0] = gps_x
                self._dr_state[1] = gps_y
                self._gps_initialised = True
                if self.Verbose:
                    print(f"[GPS ] First fix -- origin set: x={gps_x:.3f}  y={gps_y:.3f}")
            else:
                # Subsequent fixes: soft nudge toward GPS
                self._dr_state[0] += self._gps_alpha * (gps_x - self._dr_state[0])
                self._dr_state[1] += self._gps_alpha * (gps_y - self._dr_state[1])
                if self.Verbose:
                    print(f"[GPS ] Nudge: gps=({gps_x:.3f}, {gps_y:.3f})  "
                          f"dr=({self._dr_state[0]:.3f}, {self._dr_state[1]:.3f})")

    # Keep old name as alias so existing wiring doesn't break
    async def Drivetrain_UWBReceiver(self, data):
        """Alias for Drivetrain_GPSReceiver -- routes old UWB callback to GPS logic."""
        await self.Drivetrain_GPSReceiver(data)

    async def Drivetrain_AccelReceiver(self, data):
        """
        Accelerometer heading receiver.

        The accelerometer delivers heading directly in radians (same CCW-positive
        convention as the rest of the system). This is treated as absolute ground
        truth -- it overwrites theta in _dr_state with no filtering or blending.

        Args:
            data: scalar heading in radians
        """
        theta = float(data)
        async with self._state_lock:
            self._dr_state[2] = self._wrap(theta)
        if self.Verbose:
            print(f"[ACCEL] heading set: {np.degrees(theta):.1f}°")

    async def ReadMotorPosition(self):
        """
        High-rate encoder loop.

        Reads left/right wheel speeds from encoders and integrates dead-reckoning
        ONLY when driving straight (within _straight_threshold).
        During turns the encoder step is skipped entirely -- heading drift from
        wheel slip would corrupt the position estimate.

        When EncoderLogging=True, writes one CSV row per tick regardless of
        whether the DR predict fired, so turning periods are visible in the log.
        """
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

            # Log every tick (straight or turning) so the full picture is captured
            self._log_row(v_l, v_r, going_straight, state)

            if self.Verbose:
                tag = "STRAIGHT" if going_straight else "TURNING "
                print(f"[ENC ] {tag} -- v_l={v_l:.3f}  v_r={v_r:.3f}  dt={dt*1e3:.1f}ms")

            await asyncio.sleep(self.ReadMotorDataPeriod)

    # ------------------------------------------------------------------
    # Background current monitor (feeds logger cache, also used by stall check)
    # ------------------------------------------------------------------
    async def _CurrentMonitor(self):
        """
        Reads motor currents at ReadCurrentPeriod (100ms) and caches the result
        in _last_currents. The encoder logger snapshots this cache each tick
        without adding extra serial load.

        Also handles stall detection (replaces the old CheckStall loop).
        """
        if self.SimulatedData:
            return
        while self.running:
            currents = self.DCDriver.ReadMotorCurrent()
            self._last_currents = currents

            valid_currents = [c for c in currents if c is not None]
            if valid_currents and max(valid_currents) > self._Max_Current_Threshold:
                self._stall_debounce += 1
                if self._stall_debounce >= self._StallDebounceLimit:
                    print(f"[STALL] HARD STOP -- current {max(valid_currents):.1f} A "
                          f"exceeded {self._Max_Current_Threshold} A for "
                          f"{self._StallDebounceLimit} scans")
                    self._SlipModifier = self._SlipStop
            else:
                self._stall_debounce = 0

            await asyncio.sleep(self.ReadCurrentPeriod)

    # ------------------------------------------------------------------
    # Motor Speed Updates
    # ------------------------------------------------------------------
    async def UpdateMotorSpeeds(self):
        """
        Send commanded speeds to the motor driver (or simulate).

        Simulation mode: integrates the full diff-drive model (x, y, theta)
        unconditionally on every tick -- no straight-only gate since there is
        no real IMU. Theta is slip-scaled via _sim_omega_scale.

        Hardware mode: sends speed commands to DCDriver; DR integration happens
        separately in ReadMotorPosition at encoder-read rate.
        """
        while self.running:
            if self.SimulatedData:
                v_l = float(self.MotorSpeeds[0])
                v_r = float(self.MotorSpeeds[1])
                async with self._state_lock:
                    self._dr_predict(v_l, v_r, self.UpdateMotorPeriod)
            else:
                safe_speeds = np.clip(
                    self.MotorSpeeds * self._SlipModifier,
                    -self._MaxSpeedAllowed,
                    self._MaxSpeedAllowed,
                )
                self.DCDriver.UpdateMotorSpeeds(safe_speeds)

            await asyncio.sleep(self.UpdateMotorPeriod)

    # ------------------------------------------------------------------
    # Main runner and slip broadcaster
    # ------------------------------------------------------------------
    async def DriveTrainDriver_Runner(self):
        await asyncio.gather(
            self.RobotPosePublisher(),
            self.ReadMotorPosition(),
            self.UpdateMotorSpeeds(),
            self._CurrentMonitor(),
            self.SlipPublisherBroadcast(),
        )

    async def SlipPublisherBroadcast(self):
        while self.running:
            if self._SlipModifier == self._SlipStop:
                await self.SlipPublisher.publish(2)
            elif self._SlipModifier == self._SlipWarn:
                await self.SlipPublisher.publish(1)
            else:
                await self.SlipPublisher.publish(0)
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