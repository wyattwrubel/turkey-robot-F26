import asyncio
import numpy as np

from . import MapUpdater
from . import MovementExecutor
from . import PathGenerator


class TrajectoryTracker:
    Max_Allowed_Speed = 3
    MPC_Loop_Running  = False

    Current_State = np.array([0, 0, 0], dtype=np.float32)
    Verbose       = False

    def __init__(self, DriveTrainPub, MotionCommandPub, Verbose=False):
        self.DriveTrainPub    = DriveTrainPub
        self.MotionCommandPub = MotionCommandPub
        self.Verbose          = Verbose

        self.Map = MapUpdater.Map()
        self.MPC = MovementExecutor.MPC_loop()

        # Wire PD controller drive output -> drivetrain publisher
        self.MPC._drive_callback = DriveTrainPub.publish
        self.MPC.Verbose         = Verbose

    # ------------------------------------------------------------------
    # Receivers
    # ------------------------------------------------------------------

    async def VisionObject_Receiver(self, data):
        if self.Verbose:
            print("VisionObject Received: ", data)

        try:
            input_id = data["Id"]
        except KeyError:
            print(f"Bad VisionObject ID Received: 'Id' key missing in {data}")
            return

        if input_id == "DynamicObjects":
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
            humandata  = np.array(humandata)

            if turkeydata.size == 0:
                turkeydata = turkeydata.reshape(0, 3)
            if humandata.size == 0:
                humandata = humandata.reshape(0, 3)

            if turkeydata.ndim < 2 or humandata.ndim < 2:
                print("Invalid shape dimension for Human or Turkey received")
                return
            if turkeydata.shape[1] != 3 or humandata.shape[1] != 3:
                print("Invalid shape dimension for Human or Turkey received")
                return

            self.Map.ClearDynamicObstacles()

            if turkeydata.shape[0] != 0:
                for t_idx in turkeydata:
                    self.Map.UpdateTurkey(t_idx[0], t_idx[1])

            if humandata.shape[0] != 0:
                for h_idx in humandata:
                    self.Map.UpdateHuman(h_idx[0], h_idx[1])

        else:
            print(f"Bad VisionObject Cmd Received: {input_id}")

        if self.Verbose:
            print(self.Map.Zone_Map)

    async def RobotPose_Receiver(self, data):
        self.Current_State = np.asarray(data, dtype=np.float32).flatten()
        # Keep PD controller's pose in sync
        self.MPC.Current_State = self.Current_State.astype(np.float64)

    async def TrajectoryCMD_Receiver(self, data):
        if self.Verbose:
            print(f"Processing TrajectoryCMD {data}")

        try:
            input_id = data["Id"]
        except KeyError:
            print(f"Bad Trajectory Command ID Received: 'Id' key missing in {data}")
            return

        if input_id == "Orient":
            self._Orient_CMD_Receiver(data)
        elif input_id == "Move":
            self._Move_CMD_Receiver(data)
        elif input_id == "SpeedCap":
            self._SpeedCap_CMD_Receiver(data)
        else:
            print(f"Bad Trajectory Command ID Received: {input_id}")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _Move_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Trajectory Command ID Received: 'Cmd' key missing in {data}")
            return

        trajectory = PathGenerator.GenerateTrajectory(
            self.Map.Zone_Map, self.Current_State, input_cmd
        )

        if trajectory is None:
            print(f"[Navigation] PathGenerator returned None for cmd={input_cmd} "
                  f"from state={self.Current_State[:2]}")
            return

        self.MPC.Move_CMD_Receiver(trajectory)

        if self.Verbose:
            start = trajectory[0]
            end   = trajectory[-1]
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

    def _Orient_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Trajectory Command ID Received: 'Cmd' key missing in {data}")
            return

        self.MPC.Orient_CMD_Receiver(input_cmd)

    def _SpeedCap_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Trajectory Command ID Received: 'Cmd' key missing in {data}")
            return

        try:
            speed = float(np.asarray(input_cmd).flatten()[0])
            self.Max_Allowed_Speed = speed
            self.MPC.SpeedCap_Receiver(speed)
        except (ValueError, IndexError):
            print(f"Bad Trajectory Cmd Received: {input_cmd}")