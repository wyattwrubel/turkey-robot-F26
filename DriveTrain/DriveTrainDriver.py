import sys
import os
sys.path.append(os.path.dirname(__file__))
import DCMotorDriver
import numpy as np
import asyncio
import time
import UnscentedKalmanFilter as UKF

class MotorDriver:
    def __init__(self, PosePublisher, SlipPublisher, Simulated=False, Verbose=False, SkipUKFUpdates=False):
        self.PosePublisher  = PosePublisher
        self.SlipPublisher  = SlipPublisher
        self.SimulatedData  = Simulated
        self.Verbose        = Verbose
        self.SkipUKFUpdates = SkipUKFUpdates
        
        initial_state = np.array([20.0, 15.0, 0.0])
        P0    = np.diag([0.5, 0.5, 0.1])
        Q     = np.diag([0.00025, 0.00025, 0.0017])
        R_uwb = np.diag([0.02, 0.02])
        R_accel = np.diag([0.003])

        self.UKF = UKF.UKF(initial_state, P0, Q, R_uwb, R_accel)

        # ── Dead-reckoning state (used when SkipUKFUpdates=True) ──
        # Simple odometry integration — no covariance, no uncertainty buildup
        self._dr_state = initial_state.copy()   # [x, y, theta]

        if not Simulated:
            self.DCDriver = DCMotorDriver.DC_Driver()

        self.MotorSpeeds = np.array([0.0, 0.0], dtype=np.float64)

        self.WheelBase            = 0.7 # UPDATE THIS!

        self.PosePeriod           = 0.01
        self.UWBSamplePeriod      = 0.5
        self.ReadMotorDataPeriod  = 0.05
        self.UpdateMotorPeriod    = 0.1
        self.ReadCurrentPeriod    = 0.1
        self.SlipPublishPeriod    = 1

        self._MaxSpeedAllowed       = 3.0

        self._Max_Current_Threshold = 40
        self._StallDebounceLimit    = 5
        self._stall_debounce        = 0

        self._SlipWarnThreshold = 0.4
        self._SlipStopThreshold = 0.7
        self._SlipDeadband      = 0.05

        self._SlipNormal = 1.0
        self._SlipWarn   = 0.7
        self._SlipStop   = 0.0
        self._SlipModifier = self._SlipNormal

        self._enc_speed_accum = []
        self._prev_uwb_pos    = initial_state[:2].copy()
        self._prev_uwb_time   = time.time()

        self.running  = True
        self.UI_lock  = False
        self._state_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Angle utility
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(a: float) -> float:
        """Wrap angle to (-π, π] — safe across any wrap boundary."""
        return np.arctan2(np.sin(a), np.cos(a))

    # ------------------------------------------------------------------
    # Dead-reckoning integrator (no UKF, no covariance)
    # ------------------------------------------------------------------
    def _dr_predict(self, v_l, v_r, dt):
        """
        Euler-integrate the differential drive kinematics directly.
        No covariance matrix — uncertainty never accumulates.
        Uses same Y-down convention as the rest of the system.
        """
        x, y, th = self._dr_state
        v     = (v_l + v_r) / 2.0
        omega = (v_r - v_l) / self.WheelBase
        self._dr_state[0] = x  + v * np.cos(th) * dt
        self._dr_state[1] = y  - v * np.sin(th) * dt
        self._dr_state[2] = self._wrap(th + omega * dt)

    @property
    def _current_state(self):
        """Single access point — returns DR state or UKF state depending on mode."""
        if self.SkipUKFUpdates:
            return self._dr_state.copy()
        return self.UKF.state.copy()

    # ------------------------------------------------------------------
    # Robot State Publisher
    # ------------------------------------------------------------------
    async def RobotPosePublisher(self):
        """
        Publishes robot pose at PosePeriod.

        SkipUKFUpdates=False (default): publishes full UKF state (predict + correct).
        SkipUKFUpdates=True:            publishes pure odometry dead-reckoning —
                                        no UKF involved at all, so covariance never
                                        grows and pose estimate never degrades.
        """
        while self.running:
            async with self._state_lock:
                state = self._current_state
            await self.PosePublisher.publish(state)
            if self.Verbose and self.SkipUKFUpdates:
                print(f"[DR  ] Dead-reckoning state: {np.round(state, 3)}")
            await asyncio.sleep(self.PosePeriod)

    # ------------------------------------------------------------------
    # Receiver Functions
    # ------------------------------------------------------------------
    async def Drivetrain_UWBReceiver(self, data):
        """
        UWB correction — skipped in DR mode.
        Slip detection still runs since UWB ground truth is still valid.
        """
        UWB_positions = data.copy().astype(np.float64)

        if not self.SkipUKFUpdates:
            async with self._state_lock:
                self.UKF.update_uwb(UWB_positions[0], UWB_positions[1])

        now   = time.time()
        dt    = now - self._prev_uwb_time
        v_uwb = np.hypot(UWB_positions[0] - self._prev_uwb_pos[0],
                         UWB_positions[1] - self._prev_uwb_pos[1]) / dt

        if self._enc_speed_accum:
            v_enc = float(np.mean(self._enc_speed_accum))
            if abs(v_enc) > self._SlipDeadband:
                slip = abs((v_enc - v_uwb) / v_enc)
                if slip >= self._SlipStopThreshold:
                    print(f"[SLIP] HARD STOP -- ratio {slip:.2f}")
                    self._SlipModifier = self._SlipStop
                elif slip >= self._SlipWarnThreshold:
                    print(f"[SLIP] SOFT DEGRADE -- ratio {slip:.2f}")
                    self._SlipModifier = self._SlipWarn
                else:
                    self._SlipModifier = self._SlipNormal
            else:
                self._SlipModifier = self._SlipNormal

        self._enc_speed_accum = []
        self._prev_uwb_pos    = UWB_positions
        self._prev_uwb_time   = now

    async def Drivetrain_AccelReceiver(self, data):
        """Accel correction — skipped entirely in DR mode."""
        if self.SkipUKFUpdates:
            return
        theta = data.copy().astype(np.float64)
        async with self._state_lock:
            self.UKF.update_accel(theta)

    async def ReadMotorPosition(self):
        """
        High-rate encoder loop.
        UKF mode:  drives UKF predict step as before.
        DR mode:   drives _dr_predict directly — no covariance involved.
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

            v_avg = (abs(v_l) + abs(v_r)) / 2.0
            if v_avg > self._SlipDeadband:
                async with self._state_lock:
                    if self.SkipUKFUpdates:
                        self._dr_predict(v_l, v_r, dt)
                    else:
                        self.UKF.predict(v_l, v_r, self.WheelBase, dt)

            self._enc_speed_accum.append(v_avg)

            if self.Verbose:
                state = self._current_state
                print(f"[ENC ] v_l={v_l:.3f}  v_r={v_r:.3f}  dt={dt*1e3:.1f}ms  "
                      f"state={np.round(state, 3)}")

            await asyncio.sleep(self.ReadMotorDataPeriod)

    # ------------------------------------------------------------------
    # Motor Speed Updates
    # ------------------------------------------------------------------
    async def UpdateMotorSpeeds(self):
        """
        Send commanded speeds to the motor driver (or simulate).
        DR mode uses _dr_predict for simulated odometry integration.
        """
        while self.running:
            if self.SimulatedData:
                v_l = float(self.MotorSpeeds[0])
                v_r = float(self.MotorSpeeds[1])
                if (abs(v_l) + abs(v_r)) / 2.0 > self._SlipDeadband:
                    async with self._state_lock:
                        if self.SkipUKFUpdates:
                            self._dr_predict(v_l, v_r, self.UpdateMotorPeriod)
                        else:
                            self.UKF.predict(v_l, v_r, self.WheelBase, self.UpdateMotorPeriod)
            else:
                safe_speeds = np.clip(
                    self.MotorSpeeds * self._SlipModifier,
                    -self._MaxSpeedAllowed,
                    self._MaxSpeedAllowed,
                )
                self.DCDriver.UpdateMotorSpeeds(safe_speeds)

            await asyncio.sleep(self.UpdateMotorPeriod)

    # ------------------------------------------------------------------
    # Stall Detection  (unchanged)
    # ------------------------------------------------------------------
    async def CheckStall(self):
        if self.SimulatedData:
            return
        while self.running:
            currents      = self.DCDriver.ReadMotorCurrent()
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
    # Unchanged methods
    # ------------------------------------------------------------------
    async def DriveTrainDriver_Runner(self):
        await asyncio.gather(
            self.RobotPosePublisher(),
            self.ReadMotorPosition(),
            self.UpdateMotorSpeeds(),
            self.CheckStall(),
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