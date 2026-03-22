import asyncio
import numpy as np

#TODO NEED TO ADD CASADI AND IMPLEMENT THIS THING
class MPC_loop:
    
    Tracked_Trajectory = []
    Desired_State = np.array([0, 0, 0], dtype=np.float32)
    Verbose = False

    def Orient_CMD_Receiver(self, data):
        self.Desired_State = data
        self.Tracked_Trajectory = []

    def Move_CMD_Receiver(self, data):
        self.Tracked_Trajectory = data
        self.Desired_State = np.array([0, 0, 0], dtype=np.float32)

    async def MPC_Main(self):
        if(self.Verbose):
            print("TRACKING ", self.Tracked_Trajectory)
            print("Driving to ", self.Desired_State)
            
    async def DriveTrain_Sender(self):
        while True:
            await asyncio.sleep(0.1)
            # print()
            await self.MPC_Main()
    
