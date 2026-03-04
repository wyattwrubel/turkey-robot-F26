import sys
import os
sys.path.append(os.path.dirname(__file__))
import DriveTrain.SimulatedDynamics as SimulatedDynamics
import UWBDriver
import DCMotorDriver

import asyncio
import numpy as np
import asyncio
import time

class MotorDriver: 
    def __init__(self, PosePublisher, UWBPublisher, Simulated=False):
        self.PosePublisher = PosePublisher
        self.UWBPublisher = UWBPublisher

        self.SimulatedData = Simulated
        if(Simulated == False):
            self.UWB = UWBDriver.UWB()
            self.DCDriver = DCMotorDriver.DC_Driver()

        self.MotorSpeeds = np.array([0,0], dtype=np.float32)
        self.CurrentState = np.array([2, 2, 0], dtype=np.float32)

        # TODO TO BE REVISED WHEN REAL NUMBERS COME OUT
        self.WheelLength = 1 # meters
        self.SimulationSamplePeriod = 0.1 # seconds
        self.PosePeriod = 0.1 # seconds
        self.ReadMotorSamplePeriod = 0.01 # seconds
        self.UWBSamplePeriod = 2 # seconds
        
        self.running = True
        self.UI_lock = False
        self.emergency_mode = False
        
    async def MotorCommandReceiver(self, data):        
        try:
            cmd_id = data["Id"]
        except KeyError:
           print(f"Bad Motor Command ID Received: 'Id' key missing in {data}")

        if(cmd_id == "UnlockUI"):
            self.UI_lock = False
        else:
            try:
                cmd_data = data["Cmd"]
            except KeyError:
                print(f"Bad Motor Command Data Received: 'Data' key missing in {data}")
        
            if(cmd_id == "UI"):
                self.UI_lock = True
                self.MotorSpeeds = cmd_data
            elif(cmd_id == "System"):
                if(self.UI_lock == False):
                    self.MotorSpeeds = cmd_data
            else:
                print("Invalid Data Received", data)

    async def DriveTrainDriver_Runner(self):
        while self.running:
            await asyncio.gather(
                self.RobotPosePublisher(), 
                self.UpdateMotor(),
                self.ReadMotorData(),
            )

    async def RobotPosePublisher(self):
        while self.running:
            await self.PosePublisher.publish(self.CurrentState)
            await asyncio.sleep(self.PosePeriod)

    async def UpdateMotor(self):
        while self.running:
            if self.SimulatedData:
                self.CurrentState = SimulatedDynamics.differential_drive(self.CurrentState, self.MotorSpeeds, self.WheelLength, self.SimulationSamplePeriod)
                await asyncio.sleep(self.SimulationSamplePeriod)
            else:
                if(self.emergency_mode == False):
                    self.DCDriver.UpdateMotorSpeeds(self.MotorSpeeds)
            
    async def ReadMotorData(self):
        if not self.SimulatedData:
            last_time = time.time()
            while self.running:
                now = time.time()
                timeelapsed = now - last_time
                last_time = now  # update for next iteration

                # Read motor speeds
                leftspeed, rightspeed = self.DCDriver.AverageLinearSpeed()

                if abs(leftspeed - self.MotorSpeeds[0]) > 0.5 or abs(rightspeed - self.MotorSpeeds[1]) > 0.5:
                    self.emergency_mode = True
                    slowed_speeds = np.array([0.5, 0.5], dtype=np.float32)
                    self.DCDriver.UpdateMotorSpeeds(slowed_speeds)
                else:
                    self.emergency_mode = False

                # BUG TODO FIX THIS BUG
                self.CurrentState = SimulatedDynamics.differential_drive(
                    self.CurrentState,
                    self.MotorSpeeds,
                    self.WheelLength,
                    timeelapsed
                )

                await asyncio.sleep(self.ReadMotorSamplePeriod) 

    async def ReadUWBPosition(self):
        while self.running:
            # READ UWB ANCHOR POSITIONS TODO
            self.CurrentState[0:2] = self.UWB.UpdatePosition()
            await asyncio.sleep(self.UWBSamplePeriod)
