import asyncio
import numpy as np

from . import MapUpdater
from . import MovementExecutor
from . import PathGenerator
from .DataLogger import NavigationLogger


class TrajectoryTracker:
    Max_Allowed_Speed = 3
    Current_State     = np.array([0, 0, 0], dtype=np.float32)
    Verbose           = False

    def __init__(self, DriveTrainPub, MotionCommandPub,
                 MPCFeedback_Navigation_pub, Verbose=False):
        self.MotionCommandPub = MotionCommandPub
        self.Verbose          = Verbose

        self.Map    = MapUpdater.Map()
        self.Logger = NavigationLogger()
        self.MPC    = MovementExecutor.MPC_loop(
            self, DriveTrainPub, MPCFeedback_Navigation_pub)

        self._dynamic_obs         = []
        self._turkey_pos          = None
        self._last_planned_target = None
        self._replan_threshold    = 3.0

        # Stores the most recent chase A* path for old-waypoint bias on replan
        self._chase_waypoints = []

        asyncio.create_task(self.MPC.DriveTrain_Sender(logger=self.Logger))

    # ── Vision receiver ───────────────────────────────────────────────────────

    async def VisionObject_Receiver(self, data):
        if self.Verbose:
            print("VisionObject Received:", data)
        try:
            input_id = data["Id"]
        except KeyError:
            print(f"Bad VisionObject ID: 'Id' missing in {data}"); return

        if input_id != "DynamicObjects":
            print(f"Bad VisionObject Cmd: {input_id}"); return

        try:
            turkeydata = np.array(data["Turkeys"])
            humandata  = np.array(data["Humans"])
        except KeyError as e:
            print(f"Bad VisionObject Cmd: {e} missing"); return

        if turkeydata.size == 0: turkeydata = turkeydata.reshape(0, 3)
        if humandata.size  == 0: humandata  = humandata.reshape(0, 3)

        if turkeydata.ndim < 2 or humandata.ndim < 2:
            print("Invalid shape for Human or Turkey"); return
        if (turkeydata.size > 0 and turkeydata.shape[1] != 3) or \
           (humandata.size  > 0 and humandata.shape[1]  != 3):
            print("Invalid shape for Human or Turkey"); return

        self.Map.ClearDynamicObstacles()
        self._dynamic_obs = []

        if turkeydata.shape[0] > 0:
            for t in turkeydata.astype(int):
                self.Map.UpdateTurkey(t[0], t[1])
                self._dynamic_obs.append((float(t[0]), float(t[1])))

        if humandata.shape[0] > 0:
            for h in humandata.astype(int):
                self.Map.UpdateHuman(h[0], h[1])
                self._dynamic_obs.append((float(h[0]), float(h[1])))

        if turkeydata.shape[0] > 0:
            rxy   = np.array(self.Current_State[:2], dtype=float)
            dists = [np.linalg.norm(rxy - t[:2]) for t in turkeydata]
            p     = turkeydata[int(np.argmin(dists))]
            self._turkey_pos = (float(p[0]), float(p[1]))
            self.MPC.set_turkey(*self._turkey_pos)
        else:
            self._turkey_pos          = None
            self._last_planned_target = None
            self._chase_waypoints     = []
            self.MPC.clear_turkey()

        if self.Verbose:
            print(self.Map.Zone_Map)

    # ── Drivetrain receiver ───────────────────────────────────────────────────

    async def RobotPose_Receiver(self, data):
        self.Current_State = data

    # ── State planner receiver ────────────────────────────────────────────────

    async def TrajectoryCMD_Receiver(self, data):
        if self.Verbose:
            print(f"Processing TrajectoryCMD {data}")
        try:
            input_id = data["Id"]
        except KeyError:
            print(f"Bad TrajectoryCMD: 'Id' missing in {data}"); return

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
            print(f"Bad TrajectoryCMD Id: {input_id}")

    # ── Internal handlers ─────────────────────────────────────────────────────

    async def _Move_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Move CMD: 'Cmd' missing in {data}"); return

        target = np.array(input_cmd[:2], dtype=float)

        if self._last_planned_target is not None:
            moved = np.linalg.norm(target - self._last_planned_target)
            if moved < self._replan_threshold and \
               self.MPC._mode in ("trajectory", "chase", "pivot"):
                if self.Verbose:
                    print(f"[Nav] Replan suppressed (moved {moved:.2f}m)")
                if self._turkey_pos is not None:
                    self.MPC.set_turkey(*self._turkey_pos)
                return

        self._last_planned_target = target.copy()

        if self._turkey_pos is not None:
            await self._chase_move(target)
        else:
            await self._scan_move(target)

    async def _scan_move(self, target):
        wps = PathGenerator.GenerateTrajectory(
            self.Map, self.Current_State, target)
        if self.Verbose:
            print(f"[Nav] A* (scan) → {len(wps)} waypoints")
        await self.MPC.Move_CMD_Receiver(wps, target, chase=False)

    async def _chase_move(self, target):
        """
        Chase mode: A* from current position directly to the turkey, biased
        toward the previous chase path for continuity.  No path truncation —
        the robot runs to the end of the path, then orients toward the turkey.
        """
        turkey = np.array(self._turkey_pos, dtype=float)

        wps = PathGenerator.GenerateTrajectory(
            self.Map,
            self.Current_State,
            turkey,
            old_waypoints=self._chase_waypoints if self._chase_waypoints else None,
        )

        if not wps:
            if self.Verbose:
                print("[Nav] Chase A* returned empty path")
            return

        self._chase_waypoints = wps   # store for next replan bias

        if self.Verbose:
            print(f"[Nav] A* (chase) → {len(wps)} waypoints to turkey at {turkey}")

        await self.MPC.Move_CMD_Receiver(wps, tuple(turkey), chase=True)

    async def replan_to(self, target):
        """
        Called by MPC_loop on stall recovery.  Replans A* from current
        position to target and updates the MPC plan.
        """
        if target is None:
            return
        if self._turkey_pos is not None:
            await self._chase_move(np.array(target, dtype=float))
        else:
            wps = PathGenerator.GenerateTrajectory(
                self.Map, self.Current_State, target)
            if wps:
                print(f"[Nav] Stall replan → {len(wps)} waypoints")
                await self.MPC.Move_CMD_Receiver(wps, target, chase=False)

    async def _Orient_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad Orient CMD: 'Cmd' missing in {data}"); return
        await self.MPC.Orient_CMD_Receiver(input_cmd)

    def _SpeedCap_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"Bad SpeedCap CMD: 'Cmd' missing in {data}"); return
        try:
            self.MPC.set_speed_cap(float(input_cmd))
            self.Max_Allowed_Speed = float(input_cmd)   # keep local attr in sync
        except ValueError:
            print(f"Bad SpeedCap value: {input_cmd}")

    def shutdown(self):
        self.Logger.close()