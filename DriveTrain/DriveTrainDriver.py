import sys
import os
sys.path.append(os.path.dirname(__file__))
import DCMotorDriver
import numpy as np
import asyncio
import time
import UnscentedKalmanFilter as UKF

class MotorDriver:
    def __init__(self, PosePublisher, Simulated=False, Verbose=False):
        self.PosePublisher  = PosePublisher
        self.SimulatedData  = Simulated
        self.Verbose        = Verbose

        # ---- UKF initialisation ----
        initial_state = np.array([2.0, 2.0, 0.0])
        P0    = np.diag([0.5, 0.5, 0.1])
        Q     = np.diag([0.00025, 0.00025, 0.0017])
        R_uwb = np.diag([0.02, 0.02])
        R_accel = np.diag([0.003])

        # UKF always created — needed by both real and simulated modes
        self.UKF = UKF.UKF(initial_state, P0, Q, R_uwb, R_accel)

        if not Simulated:
            self.DCDriver = DCMotorDriver.DC_Driver()

        self.MotorSpeeds = np.array([0.0, 0.0], dtype=np.float64)

        # Timing constants
        self.WheelBase            = 0.7   # meters — TODO REPLACE with real track width
        self.PosePeriod           = 0.2   # s | 5 Hz
        self.UWBSamplePeriod      = 0.5   # s | 2 Hz
        self.ReadMotorDataPeriod  = 0.05  # s | 20 Hz
        self.UpdateMotorPeriod    = 0.2   # s | 5 Hz
        self.ReadCurrentPeriod    = 0.1   # s | 10 Hz

        self._MaxSpeedAllowed        = 3.0  # m/s
        self._AntiSlipSpeedModifier  = 0.5

        self._Max_Current_Threshold = 40    # A
        self._StallDebounceLimit    = 5     # Overcurrent Counts Needed

        # Slip detection — thresholds from Li & Wang (2014), optimal traction band 0.1–0.4
        self._SlipWarnThreshold   = 0.3
        self._SlipStopThreshold   = 0.5
        self._SlipDeadband        = 0.05  # m/s — skip ratio computation below this speed

        # Slip state — applied in UpdateMotorSpeeds at send time
        self._SlipModifier        = 1.0   # 1.0 = normal, 0.5 = degrade, 0.0 = stop

        # Accumulator for encoder speed between UWB updates
        self._enc_speed_accum     = []
        self._prev_uwb_pos        = initial_state[:2].copy()
        self._prev_uwb_time       = time.time()

        self.running  = True
        self.UI_lock  = False

        # asyncio lock — prevents concurrent access to UKF state
        self._state_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Command Receiver
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
            self.UI_lock     = True
            self.MotorSpeeds = cmd_data
        elif cmd_id == "System":
            if not self.UI_lock:
                self.MotorSpeeds = cmd_data
        else:
            print(f"DriveTrainDriver: Invalid cmd_id '{cmd_id}' in {data}")

    # ------------------------------------------------------------------
    # Async Runner
    # ------------------------------------------------------------------
    async def DriveTrainDriver_Runner(self):
        await asyncio.gather(
            self.RobotPosePublisher(),
            self.ReadMotorPosition(),
            self.UpdateMotorSpeeds(),
            self.CheckStall(),
        )

    # ------------------------------------------------------------------
    # Robot State Publisher
    # ------------------------------------------------------------------
    async def RobotPosePublisher(self):
        while self.running:
            async with self._state_lock:
                state = self.UKF.state.copy()
            await self.PosePublisher.publish(state)
            await asyncio.sleep(self.PosePeriod)

    # ------------------------------------------------------------------
    # Receiver Functions
    # ------------------------------------------------------------------
    async def Drivetrain_UWBReceiver(self, data):
        """
        Receives UWB Data From UWB Publisher.
        Data: 2 x 1 Numpy Array (X [m], Y [m])

        Also serves as the slip detection trigger — compares UWB-derived
        ground truth speed against the average encoder speed accumulated
        since the last UWB update. Sets _SlipModifier which is applied
        at send time in UpdateMotorSpeeds.
        """
        UWB_positions = data.copy().astype(np.float64)
        async with self._state_lock:
            self.UKF.update_uwb(UWB_positions[0], UWB_positions[1])

        # --- Slip detection ---
        now   = time.time()
        dt    = now - self._prev_uwb_time
        v_uwb = np.hypot(UWB_positions[0] - self._prev_uwb_pos[0],
                         UWB_positions[1] - self._prev_uwb_pos[1]) / dt

        if self._enc_speed_accum:
            v_enc = float(np.mean(self._enc_speed_accum))

            if abs(v_enc) > self._SlipDeadband:
                slip = (v_enc - v_uwb) / v_enc

                if slip >= self._SlipStopThreshold:
                    print(f"[SLIP] HARD STOP — ratio {slip:.2f}")
                    self._SlipModifier = 0.0
                elif slip >= self._SlipWarnThreshold:
                    print(f"[SLIP] SOFT DEGRADE — ratio {slip:.2f}")
                    self._SlipModifier = self._AntiSlipSpeedModifier
                else:
                    self._SlipModifier = 1.0

        # Reset accumulator for next UWB window
        self._enc_speed_accum = []
        self._prev_uwb_pos    = UWB_positions
        self._prev_uwb_time   = now

    async def Drivetrain_AccelReceiver(self, data):
        """
        Receives Accelerometer Data From Accel Publisher.
        Data: 1 x 1 Numpy Array (theta [radians])
        """
        theta = data.copy().astype(np.float64)
        async with self._state_lock:
            self.UKF.update_accel(theta)

    async def ReadMotorPosition(self):
        """
        High-rate encoder loop. Drives the UKF prediction step.
        Only runs when using real hardware.
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

            async with self._state_lock:
                self.UKF.predict(v_l, v_r, self.WheelBase, dt)

            # Accumulate scalar encoder speed for slip check in UWBReceiver
            self._enc_speed_accum.append((abs(v_l) + abs(v_r)) / 2.0)

            if self.Verbose:
                print(f"[ENC ] v_l={v_l:.3f}  v_r={v_r:.3f}  dt={dt*1e3:.1f}ms  "
                      f"state={np.round(self.UKF.state, 3)}")

            await asyncio.sleep(self.ReadMotorDataPeriod)

    # ------------------------------------------------------------------
    # Motor Speed Updates
    # ------------------------------------------------------------------
    async def UpdateMotorSpeeds(self):
        """Send commanded speeds to the motor driver (or simulate)."""
        while self.running:
            if self.SimulatedData:
                async with self._state_lock:
                    current = self.UKF.state.copy()

                x_next    = np.zeros_like(current)
                v_avg     = (self.MotorSpeeds[0] + self.MotorSpeeds[1]) / 2.0
                x_next[0] = current[0] + self.UpdateMotorPeriod * np.cos(current[2]) * v_avg
                x_next[1] = current[1] + self.UpdateMotorPeriod * np.sin(current[2]) * v_avg
                x_next[2] = current[2] + self.UpdateMotorPeriod * (self.MotorSpeeds[1] - self.MotorSpeeds[0]) / self.WheelBase

                async with self._state_lock:
                    self.UKF.x = x_next
            else:
                safe_speeds = np.clip(self.MotorSpeeds * self._SlipModifier, -self._MaxSpeedAllowed, self._MaxSpeedAllowed)
                self.DCDriver.UpdateMotorSpeeds(safe_speeds)

            await asyncio.sleep(self.UpdateMotorPeriod)
    
    # ------------------------------------------------------------------
    # Motor Speed Updates
    # ------------------------------------------------------------------
    async def CheckStall(self):
        """
        Polls all four motor currents at ReadCurrentPeriod.
        If any motor exceeds _StallCurrentThreshold for _StallDebounceLimit
        consecutive scans (0.5 s at 20 Hz), sets _SlipModifier to 0.0
        to hard-stop all motors via UpdateMotorSpeeds.
        """
        if self.SimulatedData:
            return
 
        while self.running:
            currents = self.DCDriver.ReadMotorCurrent()
 
            if currents is not None and np.max(currents) > self._Max_Current_Threshold:
                self._stall_debounce += 1
                if self._stall_debounce >= self._StallDebounceLimit:
                    print(f"[STALL] HARD STOP — current {np.max(currents):.1f} A "
                          f"exceeded {self._Max_Current_Threshold} A for "
                          f"{self._StallDebounceLimit} scans")
                    self._SlipModifier = 0.0
            else:
                self._stall_debounce = 0
 
            await asyncio.sleep(self.ReadCurrentPeriod)