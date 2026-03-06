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
        P0 = np.diag([0.5, 0.5, 0.1]) # initial position uncertainty ~0.7 m, heading ~0.3 rad
        
        # Process noise Q: tune higher if wheel slip is severe.
        Q = np.diag([0.00025, 0.00025, 0.0017]) # X,Y 10 cm deviation over 2 seconds, theta 15 degrees over 2 seconds [20 Hz update]
        
        # UWB measurement noise R: reflects your anchor geometry & multipath.
        R_uwb = np.diag([0.02, 0.02]) # 0.008 -> 0.2 m std every 1 seconds [2 samples at 2 Hz]

        # Accelerometer measurement noise R? 
        R_accel = np.diag([0.003]) #rad² for theta [100 Hz]

        if not Simulated:
            self.DCDriver = DCMotorDriver.DC_Driver()
            self.UKF = UKF.UKF(initial_state, P0, Q, R_uwb, R_accel)         

        self.MotorSpeeds = np.array([0.0, 0.0], dtype=np.float64)

        # Timing constants
        self.WheelBase            = 0.7   # meters — TODO REPLACE with real track width
        self.PosePeriod           = 0.2   # s | 5 Hz
        self.UWBSamplePeriod      = 0.5   # s | 2 Hz 
        self.ReadMotorDataPeriod  = 0.05  # s | 20 Hz
        self.UpdateMotorPeriod    = 0.2   # s | 5 Hz

        self.running  = True
        self.UI_lock  = False

        # asyncio lock — prevents concurrent access to UKF state
        self._state_lock = asyncio.Lock()

        # ---- UKF initialisation ----
        initial_state = np.array([2.0, 2.0, 0.0])

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
            self.UI_lock   = True
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
            self.Drivetrain_UWBReceiver(),
            self.Drivetrain_AccelReceiver(),
            self.ReadMotorPosition(),
            self.UpdateMotorSpeeds(),
        )

    # ------------------------------------------------------------------
    # Robot State Publisher: 3 x 1 Numpy Array (X [m], Y [m], theta [rad])
    # ------------------------------------------------------------------
    async def RobotPosePublisher(self):
        while self.running:
            async with self._state_lock:
                state = self.ukf.state
            await self.PosePublisher.publish(state)
            await asyncio.sleep(self.PosePeriod)

    # ------------------------------------------------------------------
    # Receiver Functions. UWB and Accel from Publisher. Read Motor Position Sync Call To DCMotorDriver
    # ------------------------------------------------------------------
    async def Drivetrain_UWBReceiver(self, data):
        """ 
        Receives UWB Data From UWB Publisher. Updates X,Y Internal Estimation
        Data: 2 x 1 Numpy Array (X [m], Y [m])  
        """
        
        UWB_positions = data.copy().astype(np.float64)

        async with self._state_lock:
            self.UKF.update_uwb(UWB_positions[0], UWB_positions[1])

    async def Drivetrain_AccelReceiver(self, data):
        """ 
        Receives Accelerometer Data From Accel Publisher. Updates Theta Internal Estimation
        Data: 1 x 1 Numpy Array (theta [radians])  
        """

        theta = data.copy().astype(np.float64)

        async with self._state_lock:
            self.UKF.update_uwb(theta)
        
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
                self.UKF.predict(v_l, v_r, self.WheelBase, dt)

            if self.Verbose:
                print(f"[ENC ] v_l={v_l:.3f}  v_r={v_r:.3f}  dt={dt*1e3:.1f}ms  "
                      f"state={np.round(self.UKF.state, 3)}")

            await asyncio.sleep(self.ReadMotorDataPeriod)

    # ------------------------------------------------------------------
    # Process MotorSpeed Update Commands
    # ------------------------------------------------------------------
    async def UpdateMotorSpeeds(self):
        """Send commanded speeds to the motor driver (or simulate)."""
        while self.running:
            if self.SimulatedData:
                async with self._state_lock:
                    current = self.UKF.state

                x_next = np.zeros_like(current)
                x_next[0] = current[0] + self.UpdateMotorPeriod * (np.cos(current[2])/2 * self.MotorSpeeds[0] + np.cos(current[2])/2 * self.MotorSpeeds[1])
                x_next[1] = current[1] + self.UpdateMotorPeriod * (np.sin(current[2])/2 * self.MotorSpeeds[0] + np.sin(current[2])/2 * self.MotorSpeeds[1])
                x_next[2] = current[2] + self.UpdateMotorPeriod * 1/self.WheelBase * (self.MotorSpeeds[1] - self.MotorSpeeds[0])
                new_state = x_next

                async with self._state_lock:
                    self.UKF.x = new_state
            else:
                self.DCDriver.UpdateMotorSpeeds(self.MotorSpeeds)

            await asyncio.sleep(self.UpdateMotorPeriod)