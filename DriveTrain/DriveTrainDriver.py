import sys
import os
sys.path.append(os.path.dirname(__file__))
import DifferentialDrive as DD
import UWBDriver
import DCMotorDriver
import numpy as np
import asyncio
import time

class MotorDriver: 
    def __init__(self, PosePublisher, Simulated=False, Verbose=False):
        self.PosePublisher = PosePublisher
        self.SimulatedData = Simulated # Bool to pull data from UWB + Encoders or State Space Model
        self.Verbose = Verbose

        if(Simulated == False):
            self.UWB = UWBDriver.UWB()
            self.DCDriver = DCMotorDriver.DC_Driver()

        self.MotorSpeeds = np.array([0,0], dtype=np.float32)
        self.CurrentState = np.array([2, 2, 0], dtype=np.float32)

        # TODO TO BE REVISED WHEN REAL NUMBERS COME OUT
        self.WheelLength = 1 # meters
        
        self.PosePeriod = 0.2 # seconds
        self.UWBSamplePeriod = 2 # seconds
        self.ReadMotorDataPeriod = 0.05 # seconds
        self.UpdateMotorPeriod = 0.2 # seconds

        self.running = True
        self.UI_lock = False

        # Variables for UKF
        P = np.eye(3) * 0.1
        Q = np.diag([0.02, 0.02, 0.01])
        
    async def MotorCommandReceiver(self, data):        
        try:
            cmd_id = data["Id"]
        except KeyError:
           print(f"DriveTrainDriver: 'Id' key missing in {data}")

        if(cmd_id == "UnlockUI"):
            self.UI_lock = False
        else:
            try:
                cmd_data = data["Cmd"]
            except KeyError:
                print(f"DriveTrainDriver: 'Data' key missing in {data}")
        
            if(cmd_id == "UI"):
                self.UI_lock = True
                self.MotorSpeeds = cmd_data
            elif(cmd_id == "System"):
                if(self.UI_lock == False):
                    self.MotorSpeeds = cmd_data
            else:
                print(f"DriveTrainDriver: Invalid 'Data' key in {data}")
        
    async def DriveTrainDriver_Runner(self):
        while self.running:
            await asyncio.gather(
                self.RobotPosePublisher(), 
                self.ReadUWBPosition(),
                self.UpdateMotorSpeeds(),
                self.ReadMotorPosition(),
            )

    async def RobotPosePublisher(self):
        while self.running:
            await self.PosePublisher.publish(self.CurrentState)
            await asyncio.sleep(self.PosePeriod)

    async def ReadUWBPosition(self):
        while self.running:
            # READ UWB ANCHOR POSITIONS TODO
            self.CurrentState = self.UWB.UpdatePosition(self.CurrentState)

    # UPDATES MOTOR COMMANDS WITH SPEEDS EVERY 0.2 SECONDS
    async def UpdateMotorSpeeds(self):
        while self.running:
            if self.SimulatedData:
                self.CurrentState = DD.differential_drive(self.CurrentState, self.MotorSpeeds, self.WheelLength, self.UpdateMotorPeriod)
            else:
                self.DCDriver.UpdateMotorSpeeds(self.MotorSpeeds)

            await asyncio.sleep(self.UpdateMotorPeriod)

    # READS MOTOR ENCODER SPEEDS EVERY 0.05 SECONDS
    async def ReadMotorPosition(self):
        if not self.SimulatedData:
            last_time = time.time()
            while self.running:
                now = time.time()
                time_elapsed = now - last_time
                last_time = now

                # Read motor speeds from encoder speeds
                leftspeed, rightspeed = self.DCDriver.AverageLinearSpeed()

                self.ukf_predict(leftspeed, rightspeed, time_elapsed)

                await asyncio.sleep(self.ReadMotorDataPeriod) 

    # TODO NEED TO VALIDATE THAT THIS WORKS? 
    def ukf_predict(self, v_l, v_r, dt):
        n = 3
        alpha = 1e-3
        beta = 2
        kappa = 0
        lambda_ = alpha**2 * (n + kappa) - n
        
        # weights
        wm = np.full(2*n + 1, 1/(2*(n+lambda_)))
        wc = np.full(2*n + 1, 1/(2*(n+lambda_)))
        wm[0] = lambda_/(n+lambda_)
        wc[0] = lambda_/(n+lambda_) + (1-alpha**2+beta)
        
        # compute v, w from wheel velocities
        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / self.WheelLength
        
        # generate sigma points
        S = np.linalg.cholesky((n + lambda_) * self.P)
        sigma_points = [self.CurrentState]
        for i in range(n):
            sigma_points.append(self.CurrentState + S[:, i])
            sigma_points.append(self.CurrentState - S[:, i])
        sigma_points = np.array(sigma_points)
        
        # propagate each sigma point through motion model
        sigma_pred = []
        for s in sigma_points:
            x_new = s[0] + v * np.cos(s[2]) * dt
            y_new = s[1] + v * np.sin(s[2]) * dt
            theta_new = s[2] + w * dt
            sigma_pred.append([x_new, y_new, theta_new])
        sigma_pred = np.array(sigma_pred)
        
        # predicted mean
        x_pred = np.zeros(n)
        for i in range(2*n + 1):
            x_pred += wm[i] * sigma_pred[i]
        
        # predicted covariance
        P_pred = self.Q.copy()
        for i in range(2*n + 1):
            d = sigma_pred[i] - x_pred
            P_pred += wc[i] * np.outer(d, d)
        
        self.CurrentState = x_pred
        self.P = P_pred