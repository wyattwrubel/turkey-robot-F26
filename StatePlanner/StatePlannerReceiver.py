"""
StatePlannerReceiver.py

Architecture: event-driven single-task state machine.

Ownership rules:
  - StatePlannerReceiver owns ALL state transitions, movement commands
    (MotionCommand_Publisher), speed caps, timers, and path selection.
  - ModuleScan / ModuleAlert / ModuleChase own ONLY canine behavior:
    poses, audio, and servo commands. They never publish movement.
  - ZoneChecker is used exclusively here to evaluate turkey position.

State machine:
  IDLE  -> SCAN  : on UnlockUI with no turkey in zone
  IDLE  -> CHASE : on UnlockUI with turkey already in zone
  SCAN  -> CHASE : turkey detected inside zone (immediate preempt)
  SCAN  -> SCAN  : turkey detected outside zone (ignored)
  CHASE -> ALERT : turkey lost visual  (end_of_chase=False, lose_visual path)
  CHASE -> ALERT : turkey exits zone   (end_of_chase=True,  boundary path)
  ALERT -> CHASE : turkey reappears inside zone (immediate interrupt)
  ALERT -> SCAN  : alert timer expires with no turkey reacquired

Chase navigation:
  Chase_Routine runs two parallel tasks:
    Nav task      — sends first Move immediately, then every 1 s toward
                    nearest-to-robot turkey. On lost visual, navigates to last
                    known pos and transitions to Alert when robot stops moving
                    (< STOPPED_THRESHOLD_M displacement in STOPPED_WINDOW_S).
    Behavior task — loops _aggressive_pursuit / _burst_attack based on
                    distance to nearest turkey; never blocks the nav task.
    _growl_and_chase fires as a background task so it does not delay Move.
  VisionObject_Receiver is memory-only: stores turkeys/ghosts, updates
  last_known_turkey_pos, never publishes movement.

Alert paths:
  Path1  (end_of_chase=True  — turkey crossed boundary):
    Orient toward last known turkey (inverted-Y heading), then:
    TerritorialWarning(30s) -> ShortWarningBurst(30s) ->
    TerritorialWarning(30s) -> ShortWarningBurst(30s)   [2 min total]
  Path2  (lose-visual, odd  attempt):
    TerritorialWarning behavior + continuous 90 deg rotation until timer expires
  Path3  (lose-visual, even attempt):
    ShortWarningBurst behavior  + continuous 90 deg rotation until timer expires

  Path2/Path3 alternate from the very first lose-visual event.
  Turkey outside zone during Alert does NOT reset the timer.
  Turkey inside zone during Alert immediately cancels and enters Chase.

Waypoint resumption:
  _scan_idx and _scan_target are preserved across interruptions.
  The robot always resumes toward the waypoint it was heading to;
  waypoints are never skipped.
"""

import asyncio
import numpy as np
from enum import Enum, auto

from StatePlanner.ModuleScan  import ModuleScan
from StatePlanner.ModuleChase import ModuleChase
from StatePlanner.ModuleAlert import ModuleAlert
from StatePlanner.ZoneChecker import ZoneChecker


class Mode(Enum):
    IDLE  = auto()
    SCAN  = auto()
    CHASE = auto()
    ALERT = auto()
    STOP  = auto()


class StatePlannerReceiver:

    ARRIVE_THRESHOLD_M       = 1.0    # metres — considered "arrived"
    GHOST_DEDUP_RADIUS_M     = 5.0    # metres — minimum separation between ghost points
    CHASE_NAV_RATE_S         = 0.5    # seconds — Move replan interval in chase
    STOPPED_WINDOW_S         = 1.0    # seconds — interval to measure robot displacement
    STOPPED_THRESHOLD_M      = 0.2    # metres — displacement below this = robot stopped

    def __init__(self, MacroState_Publisher, Canine_Publisher, MotionCommand_Publisher, Verbose=False):
        self.MacroState_Publisher    = MacroState_Publisher
        self.Canine_Publisher        = Canine_Publisher
        self.MotionCommand_Publisher = MotionCommand_Publisher
        self.Verbose                 = Verbose

        self.Current_RobotPose = np.array([0.0, 0.0, 0.0])
        self.IsManualMode      = True
        self.WaypointQueue     = self._load_waypoints('/home/turkeysquad/ME450/MainDesign/Navigation/Waypoints.csv')
        self.Turkeys           = []
        self.GhostQueue        = []

        self.ZoneChecker   = ZoneChecker()
        self._module_scan  = ModuleScan(self)
        self._module_chase = ModuleChase(self)
        self._module_alert = ModuleAlert(self)

        # Single source of truth for mode
        self._mode         : Mode             = Mode.IDLE
        self._current_task : asyncio.Task | None = None

        # Waypoint cursor — preserved across interruptions
        self._scan_idx    = 0
        self._scan_target = None

        # Alert lose-visual alternation counter
        # Odd  -> Path2 (TerritorialWarning -> WatchfulDog)
        # Even -> Path3 (ShortWarningBurst  -> WatchfulDog)
        self._lose_visual_count = 0

        # Turkey tracking
        self.turkey_detected       = False
        self.last_known_turkey_pos = None

        # Set when any turkey is confirmed inside the zone; cleared otherwise
        self._turkey_in_zone_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Waypoints
    # ------------------------------------------------------------------

    def _load_waypoints(self, path: str) -> list:
        waypoints = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                waypoints.append(np.array([float(parts[1]), float(parts[2])]))
        return waypoints

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _dbg(self, msg: str) -> None:
        if self.Verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def _cancel_current_task(self) -> None:
        """Fire-and-forget cancel. Never await a task you just cancelled."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self._current_task = None

    def _start_task(self, coro) -> None:
        self._cancel_current_task()
        self._current_task = asyncio.get_event_loop().create_task(coro)

    # ------------------------------------------------------------------
    # Mode entry helpers — always synchronous, always call _start_task
    # ------------------------------------------------------------------

    def _enter_scan(self) -> None:
        self._dbg(f"[Mode] -> SCAN  (waypoint idx={self._scan_idx})")
        self._mode = Mode.SCAN
        self._start_task(self.Scan_Routine())

    def _enter_chase(self) -> None:
        self._dbg("[Mode] -> CHASE")
        self._mode = Mode.CHASE
        self._start_task(self.Chase_Routine())

    def _enter_alert(self, end_of_chase: bool) -> None:
        self._dbg(f"[Mode] -> ALERT  (end_of_chase={end_of_chase})")
        self._mode = Mode.ALERT
        self._start_task(self.Alert_Routine(end_of_chase=end_of_chase))

    def _enter_stop(self) -> None:
        self._dbg("[Mode] -> STOP")
        self._mode = Mode.STOP
        self._start_task(self.Stop_Routine())

    # ------------------------------------------------------------------
    # Receivers
    # ------------------------------------------------------------------

    async def RobotPose_Receiver(self, data) -> None:
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 3:
            self.Current_RobotPose = arr[:3]
        else:
            print(f"[StatePlanner] RobotPose_Receiver: invalid data '{data}'")

    async def VisionObject_Receiver(self, data: dict) -> None:
        if self.IsManualMode:
            return

        turkeys = np.asarray(data.get("Turkeys", []), dtype=float)
        ghosts  = np.asarray(data.get("Ghosts",  []), dtype=float)
        self.Turkeys         = turkeys.tolist() if turkeys.size > 0 else []
        self.turkey_detected = len(self.Turkeys) > 0

        # Determine nearest turkey and update zone event / last known pos
        turkey_in_zone = False
        if self.turkey_detected:
            robot_pos = np.array(self.Current_RobotPose[:2])
            nearest   = min(self.Turkeys, key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_pos))
            self.last_known_turkey_pos = np.array(nearest[:2])
            turkey_in_zone = self.ZoneChecker.is_inside(nearest[0], nearest[1])

        # Maintain event flag for routines that poll it
        if turkey_in_zone:
            self._turkey_in_zone_event.set()
        else:
            self._turkey_in_zone_event.clear()

        # Enqueue any ghost positions for scan routing
        for g in (ghosts.tolist() if ghosts.size > 0 else []):
            self._try_enqueue_ghost(g[0], g[1])

        await self._arbitrate(turkey_in_zone)

    async def MacroState_Receiver(self, data: dict) -> None:
        """
        Handles manual UI commands and the UnlockUI / re-lock flow.
        Substates dispatched here are for manual testing only.
        """
        state = data.get("Id", "")

        if state == "UnlockUI":
            self.IsManualMode = False
            self._mode        = Mode.IDLE
            self._dbg("[StatePlanner] UnlockUI — starting auto")
            await self._arbitrate(turkey_in_zone=self._turkey_in_zone_event.is_set())
            return

        # Any other command locks to manual mode
        self.IsManualMode = True
        self._cancel_current_task()
        self._mode = Mode.IDLE

        dispatch = {
            "Scan":               lambda: self._start_task(self.Scan_Routine()),
            "Chase":              lambda: self._start_task(self._manual_chase_cycle()),
            "Alert":              lambda: self._start_task(self.Alert_Routine()),
            "Stop":               lambda: self._start_task(self.Stop_Routine()),
            "TravelWaypoint":     lambda: self._start_task(self._module_scan._travel_to_waypoint()),
            "StationaryScan":     lambda: self._start_task(self._module_scan._stationary_scan()),
            "StationaryCrouch":   lambda: self._start_task(self._module_scan._stationary_crouch()),
            "AggressivePursuit":  lambda: self._start_task(self._module_chase._aggressive_pursuit()),
            "BurstAttack":        lambda: self._start_task(self._module_chase._burst_attack()),
            "GrowlAndChase":      lambda: self._start_task(self._module_chase._growl_and_chase()),
            "WatchfulDog":        lambda: self._start_task(self._module_alert._watchful_dog()),
            "TerritorialWarning": lambda: self._start_task(self._module_alert._territorial_warning()),
            "ShortWarningBurst":  lambda: self._start_task(self._module_alert._short_warning_burst()),
        }
        if state in dispatch:
            dispatch[state]()
        else:
            print(f"[StatePlanner] MacroState_Receiver: unknown state '{state}'")

    # ------------------------------------------------------------------
    # Arbitration — called every vision frame, must be instant
    # ------------------------------------------------------------------

    async def _arbitrate(self, turkey_in_zone: bool) -> None:
        if self.IsManualMode:
            return

        if self.Verbose:
            self._dbg(
                f"[Arbitrate] mode={self._mode.name} | in_zone={turkey_in_zone} | "
                f"detected={self.turkey_detected} | "
                f"last_pos={np.round(self.last_known_turkey_pos, 2) if self.last_known_turkey_pos is not None else None}"
            )

        if turkey_in_zone:
            if self._mode in (Mode.SCAN, Mode.ALERT):
                self._dbg(f"[Arbitrate] {self._mode.name} -> CHASE (turkey inside zone)")
                self._enter_chase()
            return

        if self._mode == Mode.IDLE:
            self._dbg("[Arbitrate] IDLE -> SCAN")
            self._enter_scan()

    # ------------------------------------------------------------------
    # Scan Routine
    # ------------------------------------------------------------------

    async def Scan_Routine(self) -> None:
        await self.MacroState_Publisher.publish({"Id": "Scan"})

        if self._scan_target is None:
            self._scan_target = self.WaypointQueue[self._scan_idx % len(self.WaypointQueue)]

        self._dbg(f"[Scan] Resuming — idx={self._scan_idx}, target={np.round(self._scan_target, 2)}")

        while True:
            self._dbg(f"[Scan] Traveling to waypoint {self._scan_idx}: {np.round(self._scan_target, 2)}")
            await self.MotionCommand_Publisher.publish(
                {"Id": "Move", "Cmd": self._scan_target[:2].tolist()}
            )
            await self._SpeedCap_Publish(10.0)
            await self._module_scan._travel_to_waypoint()

            while True:
                await asyncio.sleep(0.25)
                robot_pos = np.array(self.Current_RobotPose[:2])
                dist      = np.linalg.norm(self._scan_target[:2] - robot_pos)
                self._dbg(f"[Scan] dist_to_waypoint={dist:.2f}m")
                if dist < self.ARRIVE_THRESHOLD_M:
                    break

            self._dbg(f"[Scan] Arrived at waypoint {self._scan_idx} — spinning 360°")
            base_heading  = float(self.Current_RobotPose[2])
            even_waypoint = (self._scan_idx % 2 == 0)

            if even_waypoint:
                await self._module_scan._stationary_scan()
            else:
                await self._module_scan._stationary_crouch()

            for i in range(1, 5):
                robot_pos      = np.array(self.Current_RobotPose[:3])
                target_heading = ((base_heading + i * np.pi / 2) + np.pi) % (2 * np.pi) - np.pi
                await self.MotionCommand_Publisher.publish(
                    {"Id": "Orient", "Cmd": [robot_pos[0], robot_pos[1], target_heading]}
                )
                await asyncio.sleep(1.0)

            self._scan_idx   += 1
            self._scan_target = self._next_target()
            self._dbg(f"[Scan] Next target: idx={self._scan_idx}, pos={np.round(self._scan_target, 2)}")

    # ------------------------------------------------------------------
    # Chase Routine
    # ------------------------------------------------------------------

    async def Chase_Routine(self) -> None:
        await self.MacroState_Publisher.publish({"Id": "Chase"})

        if not self.Turkeys and self.last_known_turkey_pos is None:
            self._dbg("[Chase] No turkeys and no last known pos — back to Scan")
            self._enter_scan()
            return

        self._dbg(f"[Chase] Starting — {len(self.Turkeys)} turkey(s) visible")

        # Shared signal: nav loop sets this before returning
        # False = lost visual, True = turkey exited zone
        end_of_chase_result: list = [None]

        def _nearest_turkey() -> np.ndarray | None:
            if not self.Turkeys:
                return None
            robot_pos = np.array(self.Current_RobotPose[:2])
            nearest   = min(self.Turkeys, key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_pos))
            return np.array(nearest[:2])

        async def _chase_nav_loop() -> None:
            """
            Publishes Move toward nearest in-zone turkey every 0.5 s.
            Detects zone-exit and lost-visual; sets end_of_chase_result and returns.
            """
            while True:
                # Lost visual — navigate to last known, wait until stopped
                if not self.turkey_detected or not self.Turkeys:
                    self._dbg("[Chase/Nav] Lost visual — navigating to last known pos")
                    reacquired = await self._nav_to_last_known()
                    if reacquired and self._turkey_in_zone_event.is_set():
                        self._dbg("[Chase/Nav] Turkey reacquired — resuming")
                        continue
                    self._dbg("[Chase/Nav] Turkey not reacquired — Alert")
                    end_of_chase_result[0] = False
                    return

                turkey_pos = _nearest_turkey()
                if turkey_pos is None:
                    await asyncio.sleep(self.CHASE_NAV_RATE_S)
                    continue

                if self.ZoneChecker.is_outside(turkey_pos[0], turkey_pos[1]):
                    self._dbg("[Chase/Nav] Turkey exited zone — Alert (end_of_chase)")
                    end_of_chase_result[0] = True
                    return

                await self.MotionCommand_Publisher.publish(
                    {"Id": "Move", "Cmd": turkey_pos.tolist()}
                )
                self._dbg(f"[Chase/Nav] Move -> {np.round(turkey_pos, 2)}")
                await asyncio.sleep(self.CHASE_NAV_RATE_S)

        async def _chase_canine_loop() -> None:
            """
            Plays growl sequence once, then loops pursuit/attack behavior
            based on distance to nearest turkey. Fully independent of nav.
            """
            await self._SpeedCap_Publish(0.5)
            await self._module_chase._growl_and_chase()
            await self._SpeedCap_Publish(3.0)

            while True:
                turkey_pos = _nearest_turkey()
                if turkey_pos is None:
                    await asyncio.sleep(self.CHASE_NAV_RATE_S)
                    continue

                robot_pos = np.array(self.Current_RobotPose[:2])
                dist      = np.linalg.norm(turkey_pos - robot_pos)

                if dist > self._module_chase.TURKEY_NEAR_THRESHOLD_M:
                    self._dbg(f"[Chase/Canine] dist={dist:.1f}m — AggressivePursuit")
                    await self._module_chase._aggressive_pursuit()
                else:
                    self._dbg(f"[Chase/Canine] dist={dist:.1f}m — BurstAttack")
                    await self._module_chase._burst_attack()

        loop = asyncio.get_event_loop()
        nav    = loop.create_task(_chase_nav_loop())
        canine = loop.create_task(_chase_canine_loop())

        try:
            await asyncio.shield(nav)
        except asyncio.CancelledError:
            pass
        finally:
            nav.cancel()
            canine.cancel()

        if end_of_chase_result[0] is not None:
            self._dbg("[Chase] Stopping before Alert transition")
            await self.MacroState_Publisher.publish({"Id": "Stop"})
            await self._Pose_Publish("IdlePose")
            await asyncio.sleep(0.5)
            self._enter_alert(end_of_chase=end_of_chase_result[0])

    # ------------------------------------------------------------------
    # Alert Routine
    # ------------------------------------------------------------------

    async def Alert_Routine(self, end_of_chase: bool = False) -> None:
        await self.MacroState_Publisher.publish({"Id": "Alert"})
        await self._SpeedCap_Publish(3.0)
        self._dbg(f"[Alert] Starting — end_of_chase={end_of_chase}")

        if end_of_chase:
            async def _orient_track_loop() -> None:
                """Re-orient toward turkey every 1 s while it remains visible.
                Silently stops publishing if turkey leaves sight — warning
                substates continue unaffected."""
                while True:
                    await asyncio.sleep(1.0)
                    if not self.turkey_detected or self.last_known_turkey_pos is None:
                        self._dbg("[Alert/Path1] Turkey out of sight — orient tracking paused")
                        await asyncio.sleep(1.0)
                        continue
                    rx, ry  = float(self.Current_RobotPose[0]), float(self.Current_RobotPose[1])
                    tx, ty  = float(self.last_known_turkey_pos[0]), float(self.last_known_turkey_pos[1])
                    heading = np.arctan2(-(ty - ry), tx - rx)
                    heading = ((heading + np.pi) % (2 * np.pi)) - np.pi
                    self._dbg(f"[Alert/Path1] Orient -> ({tx:.2f},{ty:.2f}), heading={np.degrees(heading):.1f}°")
                    await self.MotionCommand_Publisher.publish(
                        {"Id": "Orient", "Cmd": [rx, ry, heading]}
                    )

            # Initial orient before substates begin
            if self.last_known_turkey_pos is not None:
                rx, ry    = float(self.Current_RobotPose[0]), float(self.Current_RobotPose[1])
                tx, ty    = float(self.last_known_turkey_pos[0]), float(self.last_known_turkey_pos[1])
                heading   = np.arctan2(-(ty - ry), tx - rx)
                heading   = ((heading + np.pi) % (2 * np.pi)) - np.pi
                self._dbg(f"[Alert] Path1 — initial orient toward ({tx:.2f},{ty:.2f}), heading={np.degrees(heading):.1f}°")
                await self.MotionCommand_Publisher.publish(
                    {"Id": "Orient", "Cmd": [rx, ry, heading]}
                )

            self._dbg("[Alert] Path1 — boundary chase exit (2 min)")
            orient_task = asyncio.get_event_loop().create_task(_orient_track_loop())
            try:
                await self._module_alert._territorial_warning()
                await self._module_alert._short_warning_burst()
                await self._module_alert._territorial_warning()
                await self._module_alert._short_warning_burst()
            finally:
                orient_task.cancel()

        else:
            self._lose_visual_count += 1
            use_path2 = (self._lose_visual_count % 2 == 1)

            if self.last_known_turkey_pos is not None:
                self._dbg(f"[Alert] Navigating to last known pos {np.round(self.last_known_turkey_pos, 2)}")
                await self.MotionCommand_Publisher.publish(
                    {"Id": "Move", "Cmd": self.last_known_turkey_pos[:2].tolist()}
                )

            # Duration matches the behavior substate that runs alongside rotations
            alert_duration_s = 30.0
            step_s           = 1.0

            async def _rotation_loop() -> None:
                """Continuous 90° rotation steps for the duration of the alert."""
                base_heading = float(self.Current_RobotPose[2])
                step = 0
                elapsed = 0.0
                while elapsed < alert_duration_s:
                    step += 1
                    robot_pos      = np.array(self.Current_RobotPose[:3])
                    target_heading = ((base_heading + step * np.pi / 2) + np.pi) % (2 * np.pi) - np.pi
                    await self.MotionCommand_Publisher.publish(
                        {"Id": "Orient", "Cmd": [robot_pos[0], robot_pos[1], target_heading]}
                    )
                    self._dbg(f"[Alert/Rotate] step={step}, heading={np.degrees(target_heading):.1f}°")
                    await asyncio.sleep(step_s)
                    elapsed += step_s

            if use_path2:
                self._dbg(f"[Alert] Path2 (lose-visual #{self._lose_visual_count}) — TerritorialWarning + rotation")
                rot = asyncio.get_event_loop().create_task(_rotation_loop())
                try:
                    await self._module_alert._territorial_warning()
                finally:
                    rot.cancel()
            else:
                self._dbg(f"[Alert] Path3 (lose-visual #{self._lose_visual_count}) — ShortWarningBurst + rotation")
                rot = asyncio.get_event_loop().create_task(_rotation_loop())
                try:
                    await self._module_alert._short_warning_burst()
                finally:
                    rot.cancel()

        self._dbg("[Alert] Complete — resuming Scan")
        self._enter_scan()

    # ------------------------------------------------------------------
    # Manual Chase Cycle (manual mode only)
    # ------------------------------------------------------------------

    async def _manual_chase_cycle(self) -> None:
        """
        Manual Chase button — cycles through the three chase substates on a
        5 s hard-cancel each, with a 1 s idle between.
        Order: AggressivePursuit -> BurstAttack -> GrowlAndChase -> repeat.
        """
        substates = [
            ("AggressivePursuit", self._module_chase._aggressive_pursuit),
            ("BurstAttack",       self._module_chase._burst_attack),
            ("GrowlAndChase",     self._module_chase._growl_and_chase),
        ]
        idx = 0
        while True:
            name, coro_fn = substates[idx % len(substates)]
            self._dbg(f"[ManualChase] Starting {name} (5 s)")
            task = asyncio.get_event_loop().create_task(coro_fn())
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                self._dbg(f"[ManualChase] {name} timed out — idle 1 s")
            except asyncio.CancelledError:
                task.cancel()
                return
            else:
                self._dbg(f"[ManualChase] {name} completed early — idle 1 s")

            await asyncio.sleep(1.0)
            idx += 1

    # ------------------------------------------------------------------
    # Stop Routine
    # ------------------------------------------------------------------

    async def Stop_Routine(self) -> None:
        await self.MacroState_Publisher.publish({"Id": "Stop"})
        await self._Pose_Publish("IdlePose")
        await self._Audio_Publish("Stop")

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    async def _nav_to_last_known(self) -> bool:
        """
        Navigate toward last known turkey position.
        Returns True if turkey reacquired, False when robot stops moving
        (displacement < STOPPED_THRESHOLD_M over STOPPED_WINDOW_S).
        Waits one extra STOPPED_WINDOW_S grace period before the first
        measurement to allow the MPC time to begin moving.
        """
        if self.last_known_turkey_pos is None:
            return False

        await self.MotionCommand_Publisher.publish(
            {"Id": "Move", "Cmd": self.last_known_turkey_pos[:2].tolist()}
        )

        # Grace period — let MPC start moving before first displacement check
        await asyncio.sleep(self.STOPPED_WINDOW_S)
        if self.turkey_detected:
            return True

        while True:
            pos_before = np.array(self.Current_RobotPose[:2])
            await asyncio.sleep(self.STOPPED_WINDOW_S)

            if self.turkey_detected:
                return True

            pos_after    = np.array(self.Current_RobotPose[:2])
            displacement = np.linalg.norm(pos_after - pos_before)
            self._dbg(f"[NavToLast] displacement={displacement:.3f}m over {self.STOPPED_WINDOW_S}s")

            if displacement < self.STOPPED_THRESHOLD_M:
                self._dbg("[NavToLast] Robot stopped — turkey not reacquired")
                return False

    def _next_target(self) -> np.ndarray:
        if self.GhostQueue:
            ghost = self.GhostQueue.pop(0)
            self._dbg(f"[NextTarget] Ghost {ghost}, {len(self.GhostQueue)} remaining")
            return np.array(ghost)
        target = self.WaypointQueue[self._scan_idx % len(self.WaypointQueue)]
        self._dbg(f"[NextTarget] Waypoint idx={self._scan_idx} -> {np.round(target, 2)}")
        return target

    def _try_enqueue_ghost(self, x: float, y: float) -> None:
        candidate = np.array([x, y])
        if any(np.linalg.norm(np.array(g) - candidate) < self.GHOST_DEDUP_RADIUS_M for g in self.GhostQueue):
            return
        self.GhostQueue.append([x, y])
        self._dbg(f"[Ghost] Enqueued ({x:.2f},{y:.2f}), queue={len(self.GhostQueue)}")

    # ------------------------------------------------------------------
    # Publisher helpers
    # ------------------------------------------------------------------

    async def _SpeedCap_Publish(self, speed: float) -> None:
        await self.MotionCommand_Publisher.publish({"Id": "SpeedCap", "Cmd": speed})

    async def _Audio_Publish(self, audio: str) -> None:
        src = "UI" if self.IsManualMode else "System"
        await self.Canine_Publisher.publish({"Id": src, "Type": "Audio", "Data": audio})

    async def _Pose_Publish(self, pose: str) -> None:
        src = "UI" if self.IsManualMode else "System"
        await self.Canine_Publisher.publish({"Id": src, "Type": "Pose", "Data": pose})

    async def _ServoCommand_Publish(self, servoname: str, angle: float) -> None:
        src = "UI" if self.IsManualMode else "System"
        await self.Canine_Publisher.publish({"Id": src, "Type": "Servo", "Name": servoname, "Data": angle})