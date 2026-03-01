import asyncio
from . import Dynamics_Sim
import numpy as np

class MotorDriver: 
    def __init__(self, pose_publisher, motorcommand_publisher):
        self.pose_publisher = pose_publisher
        motorcommand_publisher.subscribe(self.motorcommand_receiver)
        
        self.SimDynamics = Dynamics_Sim
        self.motorspeeds = np.array([0,0], dtype=np.float32)
        self.state = np.array([2, 2, 0], dtype=np.float32)

        # TO BE REVISED WHEN REAL NUMBERS COME OUT
        self.wheel_length = 1
        self.sampling_period = 0.1
        
        self.running = True
        self.UI_lock = False

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

            elif(cmd_id == "Nav"):
                if(self.UI_lock == False):
                    self.motorspeeds = cmd_data
            else:
                print("Invalid Data Received", data)

    async def robotpose_transmitter(self):
        while self.running:
            await self.pose_publisher.publish(self.state)
            await asyncio.sleep(0.1)

    async def increment_dynamics(self):
        while self.running:
            self.state = self.SimDynamics.differential_drive(self.state, self.motorspeeds, self.wheel_length, self.sampling_period)
            await asyncio.sleep(self.sampling_period)

            