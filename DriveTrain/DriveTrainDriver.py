import sys
import os
sys.path.append(os.path.dirname(__file__))
import DriveTrain.Simulated_Dynamics as SimulatedDynamics

import asyncio
import numpy as np

class MotorDriver: 

    def __init__(self, PosePublisher, motorcommand_publisher):
        self.PosePublisher = PosePublisher
        motorcommand_publisher.subscribe(self.motorcommand_receiver)
        
        self.SimDynamics = SimulatedDynamics
        self.MotorSpeeds = np.array([0,0], dtype=np.float32)
        self.CurrentState = np.array([2, 2, 0], dtype=np.float32)

        # TODO TO BE REVISED WHEN REAL NUMBERS COME OUT
        self.WheelLength = 1
        self.SamplingPeriod = 0.1
        self.PosePeriod = 0.1
        
        self.running = True
        self.UI_lock = False
        self.SimulatedData = True

    async def motorcommand_receiver(self, data):        
        
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
                self.motorspeeds = cmd_data

            elif(cmd_id == "System"):
                if(self.UI_lock == False):
                    self.motorspeeds = cmd_data
            else:
                print("Invalid Data Received", data)

    async def robotpose_transmitter(self):
        while self.running:
            await self.PosePublisher.publish(self.CurrentState)
            await asyncio.sleep(self.PosePeriod)

    async def increment_dynamics(self):
        while self.running:
            if self.SimulatedData:
                self.CurrentState = self.SimDynamics.differential_drive(self.CurrentState, self.MotorSpeeds, self.WheelLength, self.SamplingPeriod)
            else:
                # self.CurrentState = # PROCESS FROM 
                print()
            await asyncio.sleep(self.SamplingPeriod)

    