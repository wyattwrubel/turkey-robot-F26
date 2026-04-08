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
  VisionObject_Receiver publishes Move commands toward the turkey at a
  rate-limited 0.5 s cadence while in CHASE mode. The Chase_Routine task
  manages only: initial growl, behavior selection, lost-visual/exit logic.

Alert paths:
  Path1  (end_of_chase=True  — turkey crossed boundary):
    TerritorialWarning(30s) -> ShortWarningBurst(30s) ->
    TerritorialWarning(30s) -> ShortWarningBurst(30s)   [2 min total]
  Path2  (lose-visual, odd  attempt):
    TerritorialWarning(30s) -> WatchfulDog(30s)          [1 min total]
  Path3  (lose-visual, even attempt):
    ShortWarningBurst(30s)  -> WatchfulDog(30s)          [1 min total]

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

    ARRIVE_THRESHOLD_M    = 1.0    # metres — considered "arrived"
    GHOST_DEDUP_RADIUS_M  = 5.0    # metres — minimum separation between ghost points
    NAV_TO_LAST_TIMEOUT_S = 15.0   # seconds — nav-to-last-known timeout
    CHASE_NAV_RATE_S      = 0.5    # seconds — minimum interval between chase Move publishes

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

        # Chase nav rate-limiting
        self._last_chase_nav_time = 0.0

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
        self._last_chase_nav_time = 0.0
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

        # Determine nearest turkey and read zone status directly from map
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

        # Chase nav: rate-limited Move publish toward turkey
        if self._mode == Mode.CHASE and turkey_in_zone and self.last_known_turkey_pos is not None:
            now = asyncio.get_event_loop().time()
            if now - self._last_chase_nav_time >= self.CHASE_NAV_RATE_S:
                self._last_chase_nav_time = now
                await self.MotionCommand_Publisher.publish(
                    {"Id": "Move", "Cmd": self.last_known_turkey_pos.tolist()}
                )

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
            "Chase":              lambda: self._start_task(self.Chase_Routine()),
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
            if self._mode == Mode.SCAN:
                self._dbg("[Arbitrate] SCAN -> CHASE (turkey inside zone)")
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
            await self._SpeedCap_Publish(2.0)
            await self._module_scan._travel_to_waypoint()

            while True:
                await asyncio.sleep(0.25)
                robot_pos = np.array(self.Current_RobotPose[:2])
                dist      = np.linalg.norm(self._scan_target[:2] - robot_pos)
                self._dbg(f"[Scan] dist_to_waypoint={dist:.2f}m")
                if dist < self.ARRIVE_THRESHOLD_M:
                    break

            self._dbg(f"[Scan] Arrived at waypoint {self._scan_idx} — spinning 360°")
            base_heading = float(self.Current_RobotPose[2])

            for i in range(1, 5):
                if i % 2 == 0:
                    await self._module_scan._stationary_scan()
                else:
                    await self._module_scan._stationary_crouch()

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
        await self._SpeedCap_Publish(3.0)

        if not self.Turkeys and self.last_known_turkey_pos is None:
            self._dbg("[Chase] No turkeys and no last known pos — back to Scan")
            self._enter_scan()
            return

        self._dbg(f"[Chase] Starting — {len(self.Turkeys)} turkey(s) visible")
        await self._module_chase._growl_and_chase()

        while True:
            if not self.turkey_detected or not self.Turkeys:
                self._dbg("[Chase] Lost visual — navigating to last known pos")
                arrived = await self._nav_to_last_known(timeout=self.NAV_TO_LAST_TIMEOUT_S)

                if self.turkey_detected and self._turkey_in_zone_event.is_set():
                    self._dbg("[Chase] Turkey reacquired inside zone — continuing chase")
                    continue

                self._dbg(f"[Chase] Turkey not reacquired (arrived={arrived}) — entering Alert")
                self._enter_alert(end_of_chase=False)
                return

            robot_pos  = np.array(self.Current_RobotPose[:2])
            nearest    = min(self.Turkeys, key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_pos))
            turkey_pos = np.array(nearest[:2])
            dist       = np.linalg.norm(turkey_pos - robot_pos)

            if self.ZoneChecker.is_outside(turkey_pos[0], turkey_pos[1]):
                self._dbg("[Chase] Turkey exited zone — entering Alert (end_of_chase)")
                self._enter_alert(end_of_chase=True)
                return

            if dist > self._module_chase.TURKEY_NEAR_THRESHOLD_M:
                self._dbg(f"[Chase] dist={dist:.1f}m > threshold — AggressivePursuit")
                await self._module_chase._aggressive_pursuit()
            else:
                self._dbg(f"[Chase] dist={dist:.1f}m <= threshold — BurstAttack")
                await self._module_chase._burst_attack()

    # ------------------------------------------------------------------
    # Alert Routine
    # ------------------------------------------------------------------

    async def Alert_Routine(self, end_of_chase: bool = False) -> None:
        await self.MacroState_Publisher.publish({"Id": "Alert"})
        await self._SpeedCap_Publish(3.0)
        self._dbg(f"[Alert] Starting — end_of_chase={end_of_chase}")

        if end_of_chase:
            self._dbg("[Alert] Path1 — boundary chase exit (2 min)")
            await self._module_alert._territorial_warning()
            await self._module_alert._short_warning_burst()
            await self._module_alert._territorial_warning()
            await self._module_alert._short_warning_burst()

        else:
            self._lose_visual_count += 1
            use_path2 = (self._lose_visual_count % 2 == 1)

            if self.last_known_turkey_pos is not None:
                self._dbg(f"[Alert] Navigating to last known pos {np.round(self.last_known_turkey_pos, 2)}")
                await self.MotionCommand_Publisher.publish(
                    {"Id": "Move", "Cmd": self.last_known_turkey_pos[:2].tolist()}
                )

            if use_path2:
                self._dbg(f"[Alert] Path2 (lose-visual #{self._lose_visual_count}) — TerritorialWarning -> WatchfulDog")
                await self._module_alert._territorial_warning()
                await self._module_alert._watchful_dog()
            else:
                self._dbg(f"[Alert] Path3 (lose-visual #{self._lose_visual_count}) — ShortWarningBurst -> WatchfulDog")
                await self._module_alert._short_warning_burst()
                await self._module_alert._watchful_dog()

        self._dbg("[Alert] Complete — resuming Scan")
        self._enter_scan()

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

    async def _nav_to_last_known(self, timeout: float) -> bool:
        if self.last_known_turkey_pos is None:
            return False

        await self.MotionCommand_Publisher.publish(
            {"Id": "Move", "Cmd": self.last_known_turkey_pos[:2].tolist()}
        )

        elapsed  = 0.0
        interval = 0.25
        while elapsed < timeout:
            await asyncio.sleep(interval)
            elapsed += interval
            if self.turkey_detected:
                return True
            rp = np.array(self.Current_RobotPose[:2])
            if np.linalg.norm(self.last_known_turkey_pos[:2] - rp) < self.ARRIVE_THRESHOLD_M:
                return True

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