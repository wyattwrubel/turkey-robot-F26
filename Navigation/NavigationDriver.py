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
                 MPCFeedback_Navigation_pub, Verbose=False, EnableLogging=False):
        self.MotionCommandPub = MotionCommandPub
        self.Verbose          = Verbose

        self.Map    = MapUpdater.Map()
        self.Logger = NavigationLogger() if EnableLogging else None
        self.MPC    = MovementExecutor.MPC_loop(
            self, DriveTrainPub, MPCFeedback_Navigation_pub,
            verbose=Verbose)

        self._last_target       = None
        self._last_face_heading = None
        self._last_waypoints    = []

        # Count of dynamic obstacles last seen — used for logging.
        # Written by VisionObject_Receiver; read by MPC_loop.
        self._n_dynamic_obs = 0

        # Latest raw turkey list from vision — list of [x, y, fov_angle].
        # Used by _Move_CMD_Receiver when chase_mode is requested.
        self._latest_turkeys: list = []

        asyncio.create_task(self.MPC.DriveTrain_Sender(logger=self.Logger))

    # ── Vision receiver ───────────────────────────────────────────────────────

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

        # Store latest turkey list for chase-mode targeting.
        self._latest_turkeys = turkeydata.tolist() if turkeydata.size > 0 else []

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

        target     = np.array(input_cmd[:2], dtype=float)
        chase_mode = data.get("Mode", "") == "Chase"

        # ── Direct-chase path ─────────────────────────────────────────────────
        if chase_mode:
            await self._handle_chase_move(target, data)
            return

        # ── Standard A* path ──────────────────────────────────────────────────

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

        # Identify if this target corresponds to a turkey (for A* early-exit).
        # We use the turkey closest to the robot, since that is also what
        # StatePlanner sends as the Move target during Chase_Routine.
        turkey_target_xy = self._nearest_turkey_to_robot()

        wps = PathGenerator.GenerateTrajectory(
            self.Map,
            self.Current_State,
            target,
            old_waypoints=self._last_waypoints if self._last_waypoints else None,
            turkey_target_xy=turkey_target_xy,
            verbose=self.Verbose,
        )

        if not wps:
            if self.Verbose:
                print("[Nav] A* returned empty path — ignoring move command")
            return

        self._last_waypoints = wps

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

        await self.MPC.Move_CMD_Receiver(wps, tuple(target),
                                         face_heading=face_heading,
                                         chase_mode=False)

    async def _handle_chase_move(self, target: np.ndarray, data: dict):
        """
        Evaluate whether direct-chase mode is viable for the given target.

        Conditions for engaging direct-chase:
          1. Robot is within CHASE_ENGAGE_DIST of the target (turkey position).
          2. The corridor between robot and turkey is clear of static obstacles.

        If both conditions are met, sends the chase command directly to MPC.
        Otherwise falls back to a normal A* Move.

        The nearest turkey's full [x, y, fov_angle] data is forwarded to MPC
        so the chase solver can use the FOV angle for centering cost.
        """
        from . import MovementExecutor as ME

        robot_xy = np.array(self.Current_State[:2], dtype=float)
        dist     = np.linalg.norm(target - robot_xy)

        # Find the turkey closest to the robot (same criterion as StatePlanner).
        turkey_data = self._nearest_turkey_to_robot(return_full=True)

        corridor_ok = False
        if dist <= ME.CHASE_ENGAGE_DIST:
            corridor_ok = ME._corridor_clear(
                self.Map.Zone_Map, robot_xy, target)

        if dist <= ME.CHASE_ENGAGE_DIST and corridor_ok:
            if self.Verbose:
                print(f"[Nav] Direct-chase engaged: dist={dist:.2f}m, "
                      f"corridor clear, turkey_data={turkey_data}")

            # Chase mode does not use an A* path — pass empty waypoints.
            await self.MPC.Move_CMD_Receiver(
                waypoints=[],
                target_xy=tuple(target),
                face_heading=None,
                chase_mode=True,
                turkey_data=turkey_data,
            )
        else:
            # Conditions not met — fall back to standard A* trajectory.
            reason = (f"dist={dist:.2f}m > {ME.CHASE_ENGAGE_DIST}m"
                      if dist > ME.CHASE_ENGAGE_DIST
                      else "corridor blocked")
            if self.Verbose:
                print(f"[Nav] Direct-chase not viable ({reason}) — "
                      f"falling back to A*")

            # Run A* toward the turkey target.
            turkey_xy = np.array(turkey_data[:2]) if turkey_data is not None else None

            wps = PathGenerator.GenerateTrajectory(
                self.Map,
                self.Current_State,
                target,
                old_waypoints=self._last_waypoints if self._last_waypoints else None,
                turkey_target_xy=turkey_xy,
                verbose=self.Verbose,
            )

            if not wps:
                if self.Verbose:
                    print("[Nav] A* (chase fallback) returned empty path — ignored")
                return

            self._last_waypoints = wps
            self._last_target    = target.copy()

            await self.MPC.Move_CMD_Receiver(
                wps, tuple(target),
                face_heading=None,
                chase_mode=False,
            )

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

    # ── Turkey helpers ────────────────────────────────────────────────────────

    def _nearest_turkey_to_robot(self, return_full: bool = False):
        """
        Return the turkey closest to the robot from the latest vision frame.

        Parameters
        ----------
        return_full : bool
            If True, returns the full [x, y, fov_angle] row.
            If False, returns [x, y] only (for A* turkey_target_xy).

        Returns None if no turkeys are currently visible.
        """
        if not self._latest_turkeys:
            return None
        robot_xy = np.array(self.Current_State[:2], dtype=float)
        nearest  = min(self._latest_turkeys,
                       key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_xy))
        return nearest if return_full else nearest[:2]

    # ── Stall recovery ────────────────────────────────────────────────────────

    async def replan_to(self, target):
        """
        Called by MPC_loop on stall detection or post-escape recovery.
        Replans A* from current position to `target` and re-issues Move_CMD_Receiver.
        """
        if target is None:
            return

        turkey_target_xy = self._nearest_turkey_to_robot()

        wps = PathGenerator.GenerateTrajectory(
            self.Map,
            self.Current_State,
            target,
            old_waypoints=self._last_waypoints if self._last_waypoints else None,
            turkey_target_xy=turkey_target_xy,
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
            wps, target, face_heading=self._last_face_heading,
            chase_mode=False)

        await self.MPC._pub_reference(wps)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        if self.Logger is not None:
            self.Logger.close()