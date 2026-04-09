import asyncio
import numpy as np

from . import MapUpdater
from . import MovementExecutor
from . import PathGenerator
from .DataLogger import NavigationLogger

REPLAN_THRESHOLD = 3.0   # metres — suppress replan if target hasn't moved this far


class TrajectoryTracker:
    Max_Allowed_Speed = 3
    Current_State     = np.array([0, 0, 0], dtype=np.float32)

    def __init__(self, DriveTrainPub, MotionCommandPub,
                 MPCFeedback_Navigation_pub, Verbose=False, EnableLogging=True):
        self.MotionCommandPub = MotionCommandPub
        self.Verbose          = Verbose

        self.Map    = MapUpdater.Map()
        self.Logger = NavigationLogger() if EnableLogging else None
        self.MPC    = MovementExecutor.MPC_loop(
            self, DriveTrainPub, MPCFeedback_Navigation_pub,
            verbose=Verbose)

        # Used by stall-replan only — tracks the last planned destination so
        # replan_to() can re-issue the same move without needing external input.
        self._last_target       = None
        self._last_face_heading = None

        # Stores the most recent A* path for old-waypoint bias on replan.
        self._last_waypoints = []

        # Count of dynamic obstacles last seen — used only for logging.
        self._n_dynamic_obs = 0

        asyncio.create_task(self.MPC.DriveTrain_Sender(logger=self.Logger))

    # ── Vision receiver ───────────────────────────────────────────────────────
    # Vision data is used ONLY to update the obstacle map.
    # The MPC and A* never see turkey/human positions directly.

    async def VisionObject_Receiver(self, data):
        if self.Verbose:
            print("[Nav] VisionObject received:", data)
        try:
            input_id = data["Id"]
        except KeyError:
            print(f"[Nav] Bad VisionObject: 'Id' missing in {data}"); return

        if input_id != "DynamicObjects":
            print(f"[Nav] Bad VisionObject Id: {input_id}"); return

        try:
            turkeydata = np.array(data["Turkeys"])
            humandata  = np.array(data["Humans"])
        except KeyError as e:
            print(f"[Nav] Bad VisionObject: {e} missing"); return

        if turkeydata.size == 0: turkeydata = turkeydata.reshape(0, 3)
        if humandata.size  == 0: humandata  = humandata.reshape(0, 3)

        if turkeydata.ndim < 2 or humandata.ndim < 2:
            print("[Nav] Invalid shape for Human or Turkey data"); return
        if (turkeydata.size > 0 and turkeydata.shape[1] != 3) or \
           (humandata.size  > 0 and humandata.shape[1]  != 3):
            print("[Nav] Invalid shape for Human or Turkey data"); return

        self.Map.ClearDynamicObstacles()
        n = 0

        for t in turkeydata.astype(int):
            self.Map.UpdateTurkey(t[0], t[1])
            n += 1

        for h in humandata.astype(int):
            self.Map.UpdateHuman(h[0], h[1])
            n += 1

        self._n_dynamic_obs = n

        if self.Verbose:
            print(f"[Nav] Map updated: {n} dynamic obstacles")

    # ── Drivetrain receiver ───────────────────────────────────────────────────

    async def RobotPose_Receiver(self, data):
        self.Current_State = data

    # ── State planner receiver ────────────────────────────────────────────────

    async def TrajectoryCMD_Receiver(self, data):
        if self.Verbose:
            print(f"[Nav] TrajectoryCMD received: {data}")
        try:
            input_id = data["Id"]
        except KeyError:
            print(f"[Nav] Bad TrajectoryCMD: 'Id' missing in {data}"); return

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
            print(f"[Nav] Bad TrajectoryCMD Id: {input_id}")

    # ── Internal handlers ─────────────────────────────────────────────────────

    async def _Move_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"[Nav] Bad Move CMD: 'Cmd' missing in {data}"); return

        target = np.array(input_cmd[:2], dtype=float)

        # Suppress replan if the target hasn't moved much and MPC is active.
        if self._last_target is not None:
            moved = np.linalg.norm(target - self._last_target)
            if moved < REPLAN_THRESHOLD and \
               self.MPC._mode in ("trajectory", "orient"):
                if self.Verbose:
                    print(f"[Nav] Replan suppressed (target moved {moved:.2f}m, "
                          f"MPC mode={self.MPC._mode})")
                return

        self._last_target = target.copy()

        # Run A* from current position to the requested target.
        # GenerateTrajectory snaps the goal if it is inside an obstacle and
        # returns the closest reachable path — the MPC then drives there.
        wps = PathGenerator.GenerateTrajectory(
            self.Map,
            self.Current_State,
            target,
            old_waypoints=self._last_waypoints if self._last_waypoints else None,
            verbose=self.Verbose,
        )

        if not wps:
            if self.Verbose:
                print("[Nav] A* returned empty path — ignoring move command")
            return

        self._last_waypoints = wps

        # If the path end differs from the requested target by more than one
        # grid cell, the goal was snapped (target is in/near an obstacle).
        # Compute a face_heading so the robot orients toward the original
        # target after arriving at the nearest reachable cell.
        path_end   = np.array(wps[-1], dtype=float)
        target_arr = np.array(target,  dtype=float)
        face_heading = None
        if np.linalg.norm(path_end - target_arr) > 1.0:
            dx = target_arr[0] - path_end[0]
            dy = target_arr[1] - path_end[1]
            face_heading = float(np.arctan2(-dy, dx))
            if self.Verbose:
                print(f"[Nav] Goal snapped: path ends at {tuple(wps[-1])}, "
                      f"face_heading={np.degrees(face_heading):.1f} deg toward "
                      f"original target {tuple(target_arr.astype(int))}")

        self._last_face_heading = face_heading

        if self.Verbose:
            print(f"[Nav] A* → {len(wps)} waypoints, "
                  f"face_heading={'%.1f deg' % np.degrees(face_heading) if face_heading is not None else 'none'}")

        await self.MPC.Move_CMD_Receiver(wps, tuple(target), face_heading=face_heading)

    async def _Orient_CMD_Receiver(self, data):
        try:
            data["Cmd"]
        except KeyError:
            print(f"[Nav] Bad Orient CMD: 'Cmd' missing in {data}"); return
        await self.MPC.Orient_CMD_Receiver(data["Cmd"])

    def _SpeedCap_CMD_Receiver(self, data):
        try:
            input_cmd = data["Cmd"]
        except KeyError:
            print(f"[Nav] Bad SpeedCap CMD: 'Cmd' missing in {data}"); return
        try:
            self.MPC.set_speed_cap(float(input_cmd))
            self.Max_Allowed_Speed = float(input_cmd)
        except ValueError:
            print(f"[Nav] Bad SpeedCap value: {input_cmd}")

    # ── Stall recovery ────────────────────────────────────────────────────────

    async def replan_to(self, target):
        """
        Called by MPC_loop on stall detection.  Replans A* from current
        position to the last known target and re-issues Move_CMD_Receiver.
        Reuses the last face_heading so snapped-goal orientation is preserved.
        """
        if target is None:
            return

        wps = PathGenerator.GenerateTrajectory(
            self.Map,
            self.Current_State,
            target,
            old_waypoints=self._last_waypoints if self._last_waypoints else None,
            verbose=self.Verbose,
        )

        if not wps:
            if self.Verbose:
                print("[Nav] Stall replan: A* returned empty path")
            return

        self._last_waypoints = wps

        if self.Verbose:
            print(f"[Nav] Stall replan → {len(wps)} waypoints")

        await self.MPC.Move_CMD_Receiver(
            wps, target, face_heading=self._last_face_heading)

        # Push the updated path to the UI — Move_CMD_Receiver calls
        # _pub_reference internally, but only after a mode transition.
        # Re-publishing here ensures the display stays current after a replan.
        await self.MPC._pub_reference(wps)

        # Move_CMD_Receiver already clears the cooldown flag via the normal
        # plan-acceptance path, so no extra action needed here.

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        if self.Logger is not None:
            self.Logger.close()