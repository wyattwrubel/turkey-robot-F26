import asyncio
import numpy as np

from . import MapUpdater
from . import MovementExecutor
from . import PathGenerator

class TrajectoryTracker:
    Max_Allowed_Speed = 3
    MPC_Loop_Running = False

    Current_State = np.array([0, 0, 0], dtype=np.float32)
    Verbose = False

    Map = MapUpdater.Map()

    def __init__(self, DriveTrainPub, MotionCommandPub, MPCFeedback_Navigation_pub, Verbose=False):
        self.MotionCommandPub = MotionCommandPub
        self.Verbose = Verbose

        self.MPC = MovementExecutor.MPC_loop(self, DriveTrainPub, MPCFeedback_Navigation_pub)
        asyncio.create_task(self.MPC.DriveTrain_Sender())

    # From vision: dynamic objects in field of view
    async def VisionObject_Receiver(self, data):
        if self.Verbose: print("VisionObject Received: ", data)
        
        try:
            input_id = data["Id"]
        except KeyError:
           print(f"Bad VisionObject ID Received: 'Id' key missing in {data}")


        if(input_id == "DynamicObjects"):
            try:
                turkeydata = data["Turkeys"]
            except KeyError:
                print(f"Bad VisionObject Cmd Received: 'Turkeys' key missing in {data}")
                return
            try:
                humandata = data["Humans"]
            except KeyError:
                print(f"Bad VisionObject Cmd Received: 'Humans' key missing in {data}")
                return  

            turkeydata = np.array(turkeydata)
            humandata = np.array(humandata)
            if turkeydata.size == 0:
                turkeydata = turkeydata.reshape(0, 3)
            if humandata.size == 0:
                humandata = humandata.reshape(0, 3)

            if(turkeydata.shape[1] != 3 or humandata.shape[1] != 3):
                print(f"Invalid shape dimension for Human or Turkey received")
                return
        
            self.Map.ClearDynamicObstacles()
        
            if(turkeydata.shape[0] != 0):
                for t_idx in turkeydata.astype(int):
                    self.Map.UpdateTurkey(t_idx[0], t_idx[1])

            if(humandata.shape[0] != 0):
                for h_idx in humandata.astype(int):
                    self.Map.UpdateHuman(h_idx[0], h_idx[1])

        else:
            print(f"Bad VisionObject Cmd Received: {input_id}")

        if self.Verbose: print(self.Map.Zone_Map)
            
    # From drivetrain: Current position x, y, theta
    async def RobotPose_Receiver(self, data):    
        self.Current_State = data

    # From state planner: Desired commands
    async def TrajectoryCMD_Receiver(self, data):
        if self.Verbose: (f"Processing TrajectoryCMD {data}" )
        print("RECEIVED ", data)

        try:
            input_id = data["Id"]
        except KeyError:
           print(f"Bad Trajectory Command ID Received: 'Id' key missing in {data}")

        if(input_id == "Orient"):
            await self._Orient_CMD_Receiver(data)
            await self.MotionCommandPub.publish(data)
        elif(input_id == "Move"):
            await self._Move_CMD_Receiver(data)
            await self.MotionCommandPub.publish(data)
        elif(input_id == "SpeedCap"):
            self._SpeedCap_CMD_Receiver(data)
            await self.MotionCommandPub.publish(data)
        else:
            print(f"Bad Trajectory Command ID Received: {input_id}")

    # Used to Traverse Zone For Chase Routines
    async def _Move_CMD_Receiver(self, data):
        try: 
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Trajectory Command ID Received: 'Cmd' key missing in {data}")

        trajectory = PathGenerator.GenerateTrajectory(self.Map.Zone_Map, self.Current_State, input_cmd)
        await self.MPC.Move_CMD_Receiver(trajectory)
        
        if self.Verbose:
            start = trajectory[0]
            end = trajectory[-1]
            for y in range(self.Map.Zone_Map.shape[0]):
                row_str = ""
                for x in range(self.Map.Zone_Map.shape[1]):
                    if (x, y) == start:
                        row_str += "S "
                    elif (x, y) == end:
                        row_str += "E "
                    elif (x, y) in trajectory:
                        row_str += "* "
                    else:
                        row_str += f"{self.Map.Zone_Map[y, x]} "
                print(row_str)
            print(trajectory)

    # Used to Max Fine Adjustments For Rotation and Alert Routines
    async def _Orient_CMD_Receiver(self, data):    
        try: 
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Trajectory Command ID Received: 'Cmd' key missing in {data}")

        await self.MPC.Orient_CMD_Receiver(input_cmd)

    # Regulates Maximum Speed Traversed By The Robot
    def _SpeedCap_CMD_Receiver(self, data):
        try: 
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Trajectory Command ID Received: 'Cmd' key missing in {data}")

        try:
            self.Max_Allowed_Speed = float(input_cmd)
        except ValueError:
            print(f"Bad Trajectory Cmd Received: {input_cmd}")