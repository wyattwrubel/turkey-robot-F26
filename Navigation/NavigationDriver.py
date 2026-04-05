import asyncio
import numpy as np

from . import MapUpdater
from . import MovementExecutor
from . import PathGenerator
from .DataLogger import NavigationLogger


class TrajectoryTracker:
    Max_Allowed_Speed = 3
    MPC_Loop_Running  = False

    Current_State = np.array([0, 0, 0], dtype=np.float32)
    Verbose       = False

    def __init__(self, DriveTrainPub, MotionCommandPub,
                 MPCFeedback_Navigation_pub, Verbose=False):
        self.MotionCommandPub = MotionCommandPub
        self.Verbose          = Verbose

        # Map (includes Zone_Map + Obstacle_Rects)
        self.Map = MapUpdater.Map()

        # Logger — one file per run
        self.Logger = NavigationLogger()

        # MPC controller
        self.MPC = MovementExecutor.MPC_loop(
            self, DriveTrainPub, MPCFeedback_Navigation_pub
        )

        # Last known turkey position for live parameter passing
        self._turkey_pos       = None   # (tx, ty) or None

        # Replan threshold — only rerun Theta* if target moves more than
        # this distance from the last planned target.  Prevents waypoint
        # oscillation from high-frequency turkey position updates.
        self._last_planned_target = None  # (tx, ty) of last Theta* call
        self._replan_threshold    = 3.0   # metres

        asyncio.create_task(
            self.MPC.DriveTrain_Sender(logger=self.Logger)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Vision receiver — dynamic objects
    # ─────────────────────────────────────────────────────────────────────────

    async def VisionObject_Receiver(self, data):
        if self.Verbose:
            print("VisionObject Received:", data)

        try:
            input_id = data["Id"]
        except KeyError:
            print(f"Bad VisionObject ID Received: 'Id' key missing in {data}")
            return

        if input_id != "DynamicObjects":
            print(f"Bad VisionObject Cmd Received: {input_id}")
            return

        try:
            turkeydata = np.array(data["Turkeys"])
            humandata  = np.array(data["Humans"])
        except KeyError as e:
            print(f"Bad VisionObject Cmd Received: {e} key missing")
            return

        if turkeydata.size == 0:
            turkeydata = turkeydata.reshape(0, 3)
        if humandata.size == 0:
            humandata = humandata.reshape(0, 3)

        if turkeydata.ndim < 2 or humandata.ndim < 2:
            print("Invalid shape dimension for Human or Turkey received")
            return
        if (turkeydata.size > 0 and turkeydata.shape[1] != 3) or \
           (humandata.size  > 0 and humandata.shape[1]  != 3):
            print("Invalid shape dimension for Human or Turkey received")
            return

        # ── Update Zone_Map dynamic layer ─────────────────────────────────────
        self.Map.ClearDynamicObstacles()

        dynamic_obs_for_mpc = []

        if turkeydata.shape[0] > 0:
            for t_idx in turkeydata.astype(int):
                self.Map.UpdateTurkey(t_idx[0], t_idx[1])
                dynamic_obs_for_mpc.append((float(t_idx[0]), float(t_idx[1])))

        if humandata.shape[0] > 0:
            for h_idx in humandata.astype(int):
                self.Map.UpdateHuman(h_idx[0], h_idx[1])
                dynamic_obs_for_mpc.append((float(h_idx[0]), float(h_idx[1])))

        # ── Update MPC with live dynamic obstacle positions ───────────────────
        self.MPC.set_dynamic_obstacles(dynamic_obs_for_mpc)

        # ── Update live turkey position for FOV cost ──────────────────────────
        if turkeydata.shape[0] > 0:
            # Track the closest turkey to the robot
            robot_xy  = self.Current_State[:2]
            dists     = [np.linalg.norm(robot_xy - t[:2]) for t in turkeydata]
            primary   = turkeydata[int(np.argmin(dists))]
            tx, ty    = float(primary[0]), float(primary[1])
            self._turkey_pos = (tx, ty)
            self.MPC.set_turkey(tx, ty)
        else:
            self._turkey_pos          = None
            self._last_planned_target = None   # force full replan on next Move
            self.MPC.clear_turkey()

        if self.Verbose:
            print(self.Map.Zone_Map)

    # ─────────────────────────────────────────────────────────────────────────
    # Drivetrain receiver — current pose
    # ─────────────────────────────────────────────────────────────────────────

    async def RobotPose_Receiver(self, data):
        self.Current_State = data

    # ─────────────────────────────────────────────────────────────────────────
    # State planner receiver — trajectory commands
    # ─────────────────────────────────────────────────────────────────────────

    async def TrajectoryCMD_Receiver(self, data):
        if self.Verbose:
            print(f"Processing TrajectoryCMD {data}")

        try:
            input_id = data["Id"]
        except KeyError:
            print(f"Bad Trajectory Command ID Received: 'Id' key missing in {data}")
            return

        if input_id == "Orient":
            await self._Orient_CMD_Receiver(data)
            await self.MotionCommandPub.publish(data)
        elif input_id == "Move":
            await self._Move_CMD_Receiver(data)
            await self.MotionCommandPub.publish(data)
        elif input_id == "SpeedCap":
            self._SpeedCap_CMD_Receiver(data)
            await self.MotionCommandPub.publish(data)
        else:
            print(f"Bad Trajectory Command ID Received: {input_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal command handlers
    # ─────────────────────────────────────────────────────────────────────────

    async def _Move_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Trajectory Command Received: 'Cmd' key missing in {data}")
            return

        target = np.array(input_cmd[:2], dtype=float)

        # Replan threshold: during chase, suppress Theta* replanning if the
        # target hasn't moved significantly.  The FOV heading cost already
        # tracks the turkey live each tick, so frequent replanning just causes
        # waypoint index oscillation and unnecessary pivots.
        if self._last_planned_target is not None:
            dist_moved = np.linalg.norm(target - self._last_planned_target)
            if dist_moved < self._replan_threshold and self.MPC._mode in ("trajectory", "pivot"):
                # Target hasn't moved enough — update turkey FOV only, skip replan
                if self.Verbose:
                    print(f"[Nav] Replan suppressed (target moved {dist_moved:.2f}m < {self._replan_threshold}m)")
                return

        self._last_planned_target = target.copy()

        # Run Theta* from current position to target
        waypoints = PathGenerator.GenerateTrajectory(
            self.Map, self.Current_State, input_cmd
        )

        if self.Verbose:
            print(f"[Nav] Theta* produced {len(waypoints)} waypoints: {waypoints}")

        # Pass chase flag to MPC so it can adjust pivot threshold.
        # During chase, only pivot for extreme misalignment (>120deg) —
        # the FOV cost handles smaller heading corrections naturally.
        self.MPC.set_chase_pivot_mode(self._turkey_pos is not None)

        await self.MPC.Move_CMD_Receiver(waypoints)

    async def _Orient_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Trajectory Command Received: 'Cmd' key missing in {data}")
            return

        await self.MPC.Orient_CMD_Receiver(input_cmd)

    def _SpeedCap_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Trajectory Command Received: 'Cmd' key missing in {data}")
            return

        try:
            self.Max_Allowed_Speed = float(input_cmd)
        except ValueError:
            print(f"Bad Trajectory Cmd Received: {input_cmd}")

    # ─────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────────────

    def shutdown(self):
        """Call this on clean exit to flush the log file."""
        self.Logger.close()