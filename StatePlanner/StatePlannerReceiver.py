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
  CHASE -> ALERT : turkey lost visual for > LOST_VISUAL_DEBOUNCE_S seconds
  CHASE -> ALERT : turkey exits zone   (end_of_chase=True,  boundary path)
  ALERT -> CHASE : turkey reappears inside zone (immediate interrupt)
  ALERT -> SCAN  : alert timer expires with no turkey reacquired

Chase navigation:
  Chase_Routine runs two parallel tasks:
    Nav task      — sends first Move immediately, then every 0.5 s toward
                    nearest-to-robot turkey.  If turkey is absent from the
                    vision frame for > LOST_VISUAL_DEBOUNCE_S continuously,
                    nav immediately stops and transitions to Alert.
                    Any reappearance within the debounce window resets the
                    timer and resumes chasing — no path-following on lost visual.
    Behavior task — loops _aggressive_pursuit / _burst_attack based on
                    distance to nearest turkey; never blocks the nav task.
    _growl_and_chase fires as a background task so it does not delay Move.
  Approach shape (CHASE_APPROACH, X2): "zigzag" sends one Move per leg from
  ZigZag.py, alternating left/right of the line of sight within the camera's
  FOV, then the straight chase for the last few metres; "straight" is W26's
  Move-toward-the-bird every 0.5 s.
  F8 combos (CHASE_RANDOM_COMBOS, X3): each chase picks one row of
  Combos.COMBOS (approach x speed profile x opening pose x sound) from a
  shuffle bag, so no two chases in a row look the same; each pick is logged.

  VisionObject_Receiver is memory-only: stores turkeys/ghosts, updates
  last_known_turkey_pos and last_known_fov_angle, never publishes movement.

Alert paths:
  Path1  (end_of_chase=True  — turkey crossed boundary):
    Orient toward last known turkey (inverted-Y heading), then:
    TerritorialWarning(30s) -> ShortWarningBurst(30s) ->
    TerritorialWarning(30s) -> ShortWarningBurst(30s)   [2 min total]
  Path2  (lose-visual, odd  attempt):
    Direction-biased 8-step rotation (2 full circles) toward the side the
    turkey was last seen, then TerritorialWarning behavior.
  Path3  (lose-visual, even attempt):
    Direction-biased 8-step rotation (2 full circles) toward the side the
    turkey was last seen, then ShortWarningBurst behavior.
  Path2/Path3 alternate from the very first lose-visual event.
  Turkey outside zone during Alert does NOT reset the timer.
  Turkey inside zone during Alert immediately cancels and enters Chase.

Waypoint resumption:
  _scan_idx and _scan_target are preserved across interruptions.
  The robot always resumes toward the waypoint it was heading to;
  waypoints are never skipped.

Deter mode (deter_mode=True, DR2 plan X5 / C8):
  Stationary detect-and-deter with the drivetrain off. UnlockUI arms
  ModuleDeter.run() (head toward the turkey and bark); any other MacroState
  disarms it. Vision goes only to ModuleDeter; the SCAN/CHASE/ALERT machine
  never runs, Scan and Alert requests are ignored, and the MotionCommand
  publisher is replaced by a no-op, so nothing here can move the wheels.

Geofence (Geofence_Receiver, DR2 plan X6 / C4):
  Every MotionCommand passes through GeofenceGate. near_edge caps speed at
  0.5 m/s and clips Move targets into the polygon; outside / unknown / a stale
  feed (and a bad fix with gps_real) latch STOP: SpeedCap 0 and no motion until
  AUTO is pressed with the geofence OK again. No Geofence messages at all =
  nothing enforced, so profiles without a geofence run unchanged.
"""

from __future__ import annotations   # lets "X | None" hints run on a laptop's Python 3.9

import asyncio
import math
import numpy as np
from enum import Enum, auto

from StatePlanner.ZigZag      import ZigZag
from StatePlanner.Combos      import Combo, ComboPicker, SPEED_PROFILES, fixed_combo
from StatePlanner.GeofenceGate import GeofenceGate, evaluate as evaluate_geofence
from StatePlanner.ModuleScan  import ModuleScan
from StatePlanner.ModuleChase import ModuleChase
from StatePlanner.ModuleAlert import ModuleAlert
from StatePlanner.ModuleDeter import ModuleDeter
from StatePlanner.ZoneChecker import ZoneChecker
from config.paths import WAYPOINTS_CSV


class Mode(Enum):
    IDLE  = auto()
    SCAN  = auto()
    CHASE = auto()
    ALERT = auto()
    STOP  = auto()
    DETER = auto()


class _NoMotionPublisher:
    """Stands in for the MotionCommand publisher in deter mode (C8): drops every message."""

    def subscribe(self, callback) -> None:
        pass

    async def publish(self, data) -> None:
        print(f"[StatePlanner] deter mode: MotionCommand dropped {data}")


def _resolve_clip_target(geofence):
    """clip_target(x, y) from the object Main.py passes, else Navigation/Geofence.py (C4), else None."""
    if geofence is not None:
        return getattr(geofence, "clip_target", None)
    try:
        from Navigation import Geofence as _geofence_module
    except ImportError:
        return None
    return getattr(_geofence_module, "clip_target", None)


class StatePlannerReceiver:

    ARRIVE_THRESHOLD_M       = 1.0    # metres — considered "arrived"
    HEADING_THRESHOLD_RAD    = 0.035  # radians — exit threshold (~2°, matches PIVOT_ALIGN_DEG)
    HEADING_REENGAGE_RAD     = 0.10   # radians — re-publish Orient only if error exceeds this (~6°)
    HEADING_TIMEOUT_S        = 10.0   # seconds — max wait per Orient step before giving up
    HEADING_POLL_S           = 0.05   # seconds — RobotPose poll interval inside heading waits
    GHOST_DEDUP_RADIUS_M     = 5.0    # metres — minimum separation between ghost points
    CHASE_NAV_RATE_S         = 0.5    # seconds — Move replan interval in chase
    STOPPED_WINDOW_S         = 1.0    # seconds — interval to measure robot displacement
    STOPPED_THRESHOLD_M      = 0.2    # metres — displacement below this = robot stopped

    # ── Speed caps (m/s) — tune these at the top level ────────────────────────
    SPEED_SCAN               = 1.0    # cruising speed during waypoint scan
    SPEED_ALERT              = 1.0    # speed cap during alert routines
    SPEED_CHASE_GROWL        = 1.0    # slow speed during initial growl sequence
    SPEED_CHASE              = 2.0    # full chase pursuit speed

    # Minimum approach distance during chase.
    # When the robot is within this radius of the turkey, speed cap drops to 0
    # (robot holds position).  Speed is restored as soon as the turkey moves
    # back beyond this distance.
    # Keep equal to MovementExecutor.CHASE_MIN_DIST (one hold distance, C7).
    CHASE_STOP_DIST_M        = 4.0    # metres

    # F8 combos (X3): each chase picks approach x speed profile x opening pose x
    # sound from Combos.COMBOS. False = W26's chase with CHASE_APPROACH below.
    CHASE_RANDOM_COMBOS      = True
    # Chase approach shape (X2) when combos are off: "zigzag" (ZigZag.py) or "straight" (W26).
    CHASE_APPROACH           = "zigzag"
    ZIGZAG_TICK_S            = 0.1    # seconds — leg-switch check interval
    ZIGZAG_SWITCH_EARLY_M    = 0.3    # metres — start the next leg this far before the vertex
    ZIGZAG_BEARING_GUARD_DEG = 4.0    # end a leg early if the bird gets this close to the FOV edge
    ZIGZAG_TURKEY_MOVED_M    = 2.0    # metres — replan the leg if the bird moved this far
    ZIGZAG_LEG_TIMEOUT_S     = 12.0   # seconds — replan if a leg takes longer than this
    ZIGZAG_RESTART_M         = 10.0   # metres — bird this far during the final approach → zig again

    # Lost-visual debounce: turkey must be absent for this long continuously
    # before the chase nav loop treats it as lost and transitions to Alert.
    # Any reappearance within this window resets the timer.
    LOST_VISUAL_DEBOUNCE_S   = 1.5

    # Direction-biased rotation in Alert Path2/Path3.
    # Total steps = ALERT_ROTATION_STEPS (8 = two full 360° rotations).
    # Direction is determined by last_known_fov_angle (+ = left, as in C2):
    #   fov_angle > 0  → turkey was left  of centre → turn left  (positive steps, CCW)
    #   fov_angle < 0  → turkey was right of centre → turn right (negative steps, CW)
    ALERT_ROTATION_STEPS     = 8
    ALERT_ROTATION_STEP_DEG  = 90.0

    # MacroStates that start routines which drive the wheels; ignored in deter mode.
    DETER_BLOCKED_STATES     = ("Scan", "Alert")

    def __init__(self, MacroState_Publisher, Canine_Publisher, MotionCommand_Publisher,
                 Verbose=False, deter_mode=False, geofence=None, gps_real=False):
        """
        geofence: optional object with clip_target(x, y) (Navigation/Geofence.py, C4);
                  if None, Navigation/Geofence.py's module-level clip_target is used if present.
        gps_real: True with --gps real; then a Fix worse than rtk_float means STOP (C4).
        """
        self.MacroState_Publisher    = MacroState_Publisher
        self.Canine_Publisher        = Canine_Publisher
        self.deter_mode              = deter_mode
        self.gps_real                = gps_real
        self._geofence_gate          = (None if deter_mode else
                                        GeofenceGate(MotionCommand_Publisher,
                                                     clip_target=_resolve_clip_target(geofence)))
        self.MotionCommand_Publisher = (_NoMotionPublisher() if deter_mode
                                        else self._geofence_gate)
        self.Verbose                 = Verbose

        # Geofence bookkeeping (Geofence_Receiver)
        self._geofence_decision: str | None = None   # "inside" | "near_edge" | "stop"
        self._geofence_reason:   str        = ""
        self._geofence_last_rx:  float      = 0.0
        self._geofence_watchdog: asyncio.Task | None = None

        self.Current_RobotPose = np.array([0.0, 0.0, 0.0])
        self.IsManualMode      = True
        self.WaypointQueue     = self._load_waypoints(str(WAYPOINTS_CSV))
        self.Turkeys           = []
        self.GhostQueue        = []

        self.ZoneChecker   = ZoneChecker()
        self._module_scan  = ModuleScan(self)
        self._module_chase = ModuleChase(self)
        self._module_alert = ModuleAlert(self)
        self._module_deter = ModuleDeter(self)
        self._zigzag       = ZigZag(hold_m=self.CHASE_STOP_DIST_M)
        self._combo_picker = ComboPicker()
        self._combo: Combo | None = None
        self._chase_count  = 0

        # Single source of truth for mode
        self._mode         : Mode             = Mode.IDLE
        self._current_task : asyncio.Task | None = None

        # Waypoint cursor — preserved across interruptions
        self._scan_idx    = 0
        self._scan_target = None

        # Alert lose-visual alternation counter
        self._lose_visual_count = 0

        # Turkey tracking
        self.turkey_detected       = False
        self.last_known_turkey_pos = None

        # Last known FOV angle of the nearest turkey (col 2 of Turkeys array).
        # Positive = turkey was left  of camera centre.
        # Negative = turkey was right of camera centre.
        # Updated whenever a turkey is visible; preserved on loss so Alert
        # rotation knows which direction to spin.
        self.last_known_fov_angle: float = 0.0

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
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self._current_task = None

    def _start_task(self, coro) -> None:
        self._cancel_current_task()
        self._current_task = asyncio.get_event_loop().create_task(coro)

    # ------------------------------------------------------------------
    # Mode entry helpers
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
    # Heading helper
    # ------------------------------------------------------------------

    async def _wait_for_heading(self, target_rad: float) -> bool:
        """
        Poll RobotPose theta until within HEADING_THRESHOLD_RAD of target_rad,
        or until HEADING_TIMEOUT_S elapses.

        Re-publishes Orient only when the heading error exceeds
        HEADING_REENGAGE_RAD — this keeps the MPC pivot alive if the robot
        overshoots or drifts, without hammering Orient_CMD_Receiver every tick
        (which resets u_prev and err_history and causes its own delays).

        Returns True if heading reached, False on timeout.
        """
        elapsed   = 0.0
        robot_pos = np.array(self.Current_RobotPose[:3])
        drive_heading = (target_rad + np.pi) % (2 * np.pi) - np.pi

        # Publish once immediately to kick the pivot off.
        await self.MotionCommand_Publisher.publish(
            {"Id": "Orient",
             "Cmd": [float(robot_pos[0]), float(robot_pos[1]), drive_heading]}
        )

        while elapsed < self.HEADING_TIMEOUT_S:
            await asyncio.sleep(self.HEADING_POLL_S)
            elapsed += self.HEADING_POLL_S

            current = float(self.Current_RobotPose[2])
            err     = (target_rad - current + np.pi) % (2 * np.pi) - np.pi
            err_abs = abs(err)

            if err_abs < self.HEADING_THRESHOLD_RAD:
                self._dbg(
                    f"[Heading] Reached {np.degrees(target_rad):.1f}° "
                    f"(err={np.degrees(err):.2f}°) in {elapsed:.2f}s"
                )
                return True

            # Only re-publish if error has grown back above re-engage threshold.
            # This lets the MPC pivot run to completion without being reset every
            # 50ms, while still correcting overshoots or drift.
            if err_abs > self.HEADING_REENGAGE_RAD:
                robot_pos = np.array(self.Current_RobotPose[:3])
                await self.MotionCommand_Publisher.publish(
                    {"Id": "Orient",
                     "Cmd": [float(robot_pos[0]), float(robot_pos[1]), drive_heading]}
                )

        self._dbg(
            f"[Heading] Timeout ({self.HEADING_TIMEOUT_S}s) waiting for "
            f"{np.degrees(target_rad):.1f}° — proceeding anyway"
        )
        return False

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
        if self.deter_mode:
            self._module_deter.on_vision(data)   # stores only; never awaits
            return

        if self.IsManualMode:
            return

        turkeys = np.asarray(data.get("Turkeys", []), dtype=float)
        ghosts  = np.asarray(data.get("Ghosts",  []), dtype=float)
        self.Turkeys         = turkeys.tolist() if turkeys.size > 0 else []
        self.turkey_detected = len(self.Turkeys) > 0

        # Determine nearest turkey, update zone event, last known pos and FOV angle.
        turkey_in_zone = False
        if self.turkey_detected:
            robot_pos = np.array(self.Current_RobotPose[:2])
            nearest   = min(self.Turkeys,
                            key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_pos))
            self.last_known_turkey_pos = np.array(nearest[:2])

            # Col 2 is the relative angle from camera centre (radians).
            # Positive = left of centre, negative = right of centre.
            # Only update when turkey is visible so we preserve the last-seen
            # direction for Alert rotation after the turkey disappears.
            self.last_known_fov_angle = float(nearest[2])

            turkey_in_zone = self.ZoneChecker.is_inside(nearest[0], nearest[1])

        if turkey_in_zone:
            self._turkey_in_zone_event.set()
        else:
            self._turkey_in_zone_event.clear()

        for g in (ghosts.tolist() if ghosts.size > 0 else []):
            self._try_enqueue_ghost(g[0], g[1])

        await self._arbitrate(turkey_in_zone)

    async def MacroState_Receiver(self, data: dict) -> None:
        state = data.get("Id", "")

        if state == "UnlockUI":
            gate = self._geofence_gate
            if gate is not None and gate.latched:
                if self._geofence_decision == "stop":
                    print(f"[Geofence] AUTO refused — geofence says {self._geofence_reason}; "
                          f"drive back inside in manual first")
                    return
                gate.release()
                print("[Geofence] STOP released by AUTO")
            self.IsManualMode = False
            if self.deter_mode:
                self._dbg("[StatePlanner] UnlockUI — deter mode armed")
                self._mode = Mode.DETER
                self._start_task(self._module_deter.run())
                return
            self._mode        = Mode.IDLE
            self._dbg("[StatePlanner] UnlockUI — starting auto")
            await self._arbitrate(turkey_in_zone=self._turkey_in_zone_event.is_set())
            return

        self.IsManualMode = True
        self._cancel_current_task()
        self._mode = Mode.IDLE
        # Vision is ignored in manual, so "bird in zone" would go stale. W26 kept it,
        # and AUTO with it still set left the planner stuck in IDLE.
        self._turkey_in_zone_event.clear()

        if self.deter_mode and state in self.DETER_BLOCKED_STATES:
            print(f"[StatePlanner] deter mode: '{state}' ignored (it drives the wheels)")
            return

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

    async def Geofence_Receiver(self, data: dict) -> None:
        """Contract C4: Geofence at 10 Hz from Navigation/Geofence.py (wired in Main.py)."""
        gate = self._geofence_gate
        if gate is None:                 # deter mode: the wheels never move anyway
            return
        if not isinstance(data, dict) or data.get("Id") != "Geofence":
            print(f"[Geofence] ignored malformed message {data}")
            return

        self._geofence_last_rx = asyncio.get_event_loop().time()
        if not gate.active:
            gate.active = True
            print("[Geofence] first message — geofence enforced from now on")
            self._geofence_watchdog = asyncio.get_event_loop().create_task(
                self._geofence_watchdog_loop())

        decision, reason = evaluate_geofence(data, self.gps_real)
        await self._apply_geofence(decision, reason, data)

    async def _apply_geofence(self, decision: str, reason: str, data: dict | None = None) -> None:
        gate = self._geofence_gate
        if (decision, reason) != (self._geofence_decision, self._geofence_reason):
            extra = ""
            if data is not None:
                dist = data.get("DistToEdge_m")
                if isinstance(dist, (int, float)):
                    extra += f", {dist:+.2f} m to edge"
                extra += f", fix {data.get('Fix')}"
            print(f"[Geofence] {self._geofence_decision or 'start'} -> {decision} "
                  f"({reason}{extra}) | mode {self._mode.name}")
            self._geofence_decision, self._geofence_reason = decision, reason

        if decision == "stop":
            if not gate.latched:
                await gate.latch(reason)
                self._enter_stop()
            return
        await gate.set_near_edge(decision == "near_edge")

    async def _geofence_watchdog_loop(self) -> None:
        """After the first Geofence message, silence for GeofenceGate.STALE_S counts as unknown."""
        loop = asyncio.get_event_loop()
        while True:
            await asyncio.sleep(0.2)
            if loop.time() - self._geofence_last_rx > GeofenceGate.STALE_S:
                await self._apply_geofence(
                    "stop", f"no Geofence message for > {GeofenceGate.STALE_S:.0f} s")

    # ------------------------------------------------------------------
    # Arbitration
    # ------------------------------------------------------------------

    async def _arbitrate(self, turkey_in_zone: bool) -> None:
        if self.IsManualMode or self.deter_mode:
            return

        if self.Verbose:
            self._dbg(
                f"[Arbitrate] mode={self._mode.name} | in_zone={turkey_in_zone} | "
                f"detected={self.turkey_detected} | "
                f"last_pos={np.round(self.last_known_turkey_pos, 2) if self.last_known_turkey_pos is not None else None} | "
                f"fov_angle={np.degrees(self.last_known_fov_angle):.1f}deg"
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
            await self._SpeedCap_Publish(self.SPEED_SCAN)
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
                robot_pos    = np.array(self.Current_RobotPose[:3])
                wait_heading = (base_heading + i * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
                drive_heading = (wait_heading + np.pi) % (2 * np.pi) - np.pi
                await self.MotionCommand_Publisher.publish(
                    {"Id": "Orient", "Cmd": [robot_pos[0], robot_pos[1], drive_heading]}
                )
                await self._wait_for_heading(wait_heading)
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

        end_of_chase_result: list = [None]

        # F8 combo for this chase (X3): approach shape, speed profile, opening pose + sound.
        combo    = self._pick_combo()
        approach = combo.approach
        # The profile sets "target"; the nav loop holds at 0 near the bird
        # ("holding") and the profile never raises speed while it does.
        chase_speed = {"target": self._speed_level(SPEED_PROFILES[combo.speed][0][1]),
                       "holding": False}

        # Zig-zag approach state for this chase (used when approach == "zigzag").
        self._zigzag.reset()
        zz = {"leg": None, "target": None, "turkey_ref": None, "t_leg": 0.0,
              "final": False, "t_final": 0.0}

        def _nearest_turkey() -> np.ndarray | None:
            if not self.Turkeys:
                return None
            robot_pos = np.array(self.Current_RobotPose[:2])
            nearest   = min(self.Turkeys,
                            key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_pos))
            return np.array(nearest[:2])

        async def _chase_nav_loop() -> None:
            """
            Approaches the nearest in-zone turkey. The combo's approach "straight"
            publishes Move toward it every CHASE_NAV_RATE_S (W26). "zigzag"
            publishes one Move per zig-zag leg, then the same straight chase
            for the last few metres (see _zigzag_step).

            Lost-visual handling:
              A 0.5 s countdown timer is reset every time a turkey is visible.
              If the timer expires without a visible turkey frame, the robot
              stops (SpeedCap 0.0) and transitions to Alert immediately.
              Business as usual while turkey is continuously in frame.

            Zone-exit handling is unchanged: if the turkey is visible but has
            crossed the zone boundary, end_of_chase_result[0] = True.
            """
            # Deadline: asyncio time by which a turkey must appear or we stop.
            # Initialised to now + debounce so the loop starts with a full window.
            deadline:  float = (asyncio.get_event_loop().time()
                               + self.LOST_VISUAL_DEBOUNCE_S)
            stopped:   bool  = False   # True once SpeedCap 0 sent for lost-visual
            too_close: bool  = False   # True once SpeedCap 0 sent for proximity

            while True:
                now = asyncio.get_event_loop().time()

                if self.turkey_detected and self.Turkeys:
                    # Turkey visible — reset the deadline and restore speed if
                    # we had previously lost visual.
                    deadline = now + self.LOST_VISUAL_DEBOUNCE_S
                    if stopped:
                        self._dbg("[Chase/Nav] Turkey back in frame — resuming chase")
                        chase_speed["holding"] = False
                        await self._SpeedCap_Publish(chase_speed["target"])
                        stopped   = False
                        too_close = False   # re-evaluate proximity from scratch

                    turkey_pos = _nearest_turkey()
                    if turkey_pos is None:
                        await asyncio.sleep(self.CHASE_NAV_RATE_S)
                        continue

                    if self.ZoneChecker.is_outside(turkey_pos[0], turkey_pos[1]):
                        self._dbg("[Chase/Nav] Turkey exited zone — Alert (end_of_chase)")
                        end_of_chase_result[0] = True
                        return

                    # ── Proximity stop ────────────────────────────────────────
                    # If the robot is within CHASE_STOP_DIST_M, cut speed to 0
                    # and suppress the Move command.  Resume as soon as the
                    # turkey moves back outside the radius.
                    robot_pos   = np.array(self.Current_RobotPose[:2])
                    dist_to_turkey = np.linalg.norm(turkey_pos - robot_pos)

                    if dist_to_turkey < self.CHASE_STOP_DIST_M:
                        if not too_close:
                            self._dbg(
                                f"[Chase/Nav] Turkey within {self.CHASE_STOP_DIST_M}m "
                                f"(dist={dist_to_turkey:.2f}m) — holding position"
                            )
                            chase_speed["holding"] = True
                            await self._SpeedCap_Publish(0.0)
                            too_close = True
                        await asyncio.sleep(self.CHASE_NAV_RATE_S)
                        continue

                    # Turkey is beyond the stop radius — resume if we were holding.
                    if too_close:
                        self._dbg(
                            f"[Chase/Nav] Turkey moved beyond {self.CHASE_STOP_DIST_M}m "
                            f"(dist={dist_to_turkey:.2f}m) — resuming"
                        )
                        chase_speed["holding"] = False
                        await self._SpeedCap_Publish(chase_speed["target"])
                        too_close = False

                    if approach == "zigzag":
                        await self._zigzag_step(zz, turkey_pos, now)
                        await asyncio.sleep(self.ZIGZAG_TICK_S)
                    else:
                        await self._publish_chase_move(turkey_pos)
                        await asyncio.sleep(self.CHASE_NAV_RATE_S)

                else:
                    # Turkey not in frame this tick.
                    if (approach == "zigzag" and zz["leg"] is not None
                            and self.last_known_turkey_pos is not None):
                        # A zig took the bird out of frame: stop zigging and head
                        # straight back toward where it was last seen. Zig-zag
                        # replans from wherever the robot is once it's back.
                        self._dbg("[ZigZag] Bird left the frame mid-leg — turning back toward it")
                        zz.update(leg=None, turkey_ref=None)
                        await self._publish_chase_move(self.last_known_turkey_pos)
                    if now >= deadline:
                        # Timer expired — stop and transition to Alert.
                        if not stopped:
                            self._dbg(
                                f"[Chase/Nav] Turkey absent for "
                                f"{self.LOST_VISUAL_DEBOUNCE_S}s — "
                                f"stopping, transitioning to Alert"
                            )
                            chase_speed["holding"] = True
                            await self.MotionCommand_Publisher.publish(
                                {"Id": "SpeedCap", "Cmd": 0.0}
                            )
                            stopped = True
                        end_of_chase_result[0] = False
                        return
                    # Still within window — poll fast and wait for turkey to return.
                    await asyncio.sleep(self.HEADING_POLL_S)

        async def _chase_speed_loop() -> None:
            """Walks the combo's speed profile; never raises speed while holding."""
            t0 = asyncio.get_event_loop().time()
            for t_step, level in SPEED_PROFILES[combo.speed]:
                delay = t0 + t_step - asyncio.get_event_loop().time()
                if delay > 0:
                    await asyncio.sleep(delay)
                chase_speed["target"] = self._speed_level(level)
                if not chase_speed["holding"]:
                    await self._SpeedCap_Publish(chase_speed["target"])

        async def _chase_canine_loop() -> None:
            await self._module_chase._opening(combo.pose, combo.sound)

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

        loop   = asyncio.get_event_loop()
        nav    = loop.create_task(_chase_nav_loop())
        speed  = loop.create_task(_chase_speed_loop())
        canine = loop.create_task(_chase_canine_loop())

        try:
            await asyncio.shield(nav)
        except asyncio.CancelledError:
            pass
        finally:
            nav.cancel()
            speed.cancel()
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
        await self._SpeedCap_Publish(self.SPEED_ALERT)
        self._dbg(f"[Alert] Starting — end_of_chase={end_of_chase}")

        if end_of_chase:
            # ── Path 1: turkey crossed zone boundary ──────────────────────────
            async def _orient_track_loop() -> None:
                while True:
                    await asyncio.sleep(2.0)
                    if not self.turkey_detected or self.last_known_turkey_pos is None:
                        self._dbg("[Alert/Path1] Turkey out of sight — orient tracking paused")
                        await asyncio.sleep(2.0)
                        continue
                    rx, ry        = float(self.Current_RobotPose[0]), float(self.Current_RobotPose[1])
                    tx, ty        = float(self.last_known_turkey_pos[0]), float(self.last_known_turkey_pos[1])
                    wait_heading  = np.arctan2(-(ty - ry), tx - rx)
                    wait_heading  = ((wait_heading + np.pi) % (2 * np.pi)) - np.pi
                    drive_heading = (wait_heading + np.pi) % (2 * np.pi) - np.pi
                    self._dbg(f"[Alert/Path1] Orient -> ({tx:.2f},{ty:.2f}), heading={np.degrees(wait_heading):.1f}°")
                    await self.MotionCommand_Publisher.publish(
                        {"Id": "Orient", "Cmd": [rx, ry, drive_heading]}
                    )
                    await self._wait_for_heading(wait_heading)

            if self.last_known_turkey_pos is not None:
                rx, ry        = float(self.Current_RobotPose[0]), float(self.Current_RobotPose[1])
                tx, ty        = float(self.last_known_turkey_pos[0]), float(self.last_known_turkey_pos[1])
                wait_heading  = np.arctan2(-(ty - ry), tx - rx)
                wait_heading  = ((wait_heading + np.pi) % (2 * np.pi)) - np.pi
                drive_heading = (wait_heading + np.pi) % (2 * np.pi) - np.pi
                self._dbg(f"[Alert] Path1 — initial orient toward ({tx:.2f},{ty:.2f}), "
                           f"heading={np.degrees(wait_heading):.1f}°")
                await self.MotionCommand_Publisher.publish(
                    {"Id": "Orient", "Cmd": [rx, ry, drive_heading]}
                )
                await self._wait_for_heading(wait_heading)

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
            # ── Path 2 / Path 3: lost visual ──────────────────────────────────
            self._lose_visual_count += 1
            use_path2 = (self._lose_visual_count % 2 == 1)

            # Determine rotation direction from the last known FOV angle.
            # fov_angle is + = left (C2) and +theta is CCW, so we turn toward
            # the side the turkey was last seen:
            # fov_angle > 0 → turkey was left  of centre → turn left  (step = +90°)
            # fov_angle < 0 → turkey was right of centre → turn right (step = -90°)
            # fov_angle == 0 is excluded by design (only lost when not in frame).
            step_sign   = 1.0 if self.last_known_fov_angle > 0.0 else -1.0
            step_rad    = step_sign * np.radians(self.ALERT_ROTATION_STEP_DEG)
            direction   = "left" if step_sign > 0 else "right"
            self._dbg(
                f"[Alert] Path{'2' if use_path2 else '3'} "
                f"(lose-visual #{self._lose_visual_count}) — "
                f"{self.ALERT_ROTATION_STEPS} × {self.ALERT_ROTATION_STEP_DEG}° "
                f"turning {direction} "
                f"(last_fov={np.degrees(self.last_known_fov_angle):.1f}°)"
            )

            async def _directed_rotation() -> None:
                """
                Execute ALERT_ROTATION_STEPS × 90° turns in the direction the
                turkey was last seen (2 full rotations).  Uses closed-loop
                Orient + _wait_for_heading so each step actually settles before
                the next fires.  The coroutine completes after all steps are
                done — it does NOT loop indefinitely.
                """
                base_heading = float(self.Current_RobotPose[2])
                for step in range(1, self.ALERT_ROTATION_STEPS + 1):
                    robot_pos     = np.array(self.Current_RobotPose[:3])
                    # Accumulate steps from the base heading in the chosen direction.
                    wait_heading  = (base_heading + step * step_rad + np.pi) % (2 * np.pi) - np.pi
                    drive_heading = (wait_heading + np.pi) % (2 * np.pi) - np.pi
                    self._dbg(
                        f"[Alert/Rotate] step={step}/{self.ALERT_ROTATION_STEPS} "
                        f"heading={np.degrees(wait_heading):.1f}° ({direction})"
                    )
                    await self.MotionCommand_Publisher.publish(
                        {"Id": "Orient", "Cmd": [robot_pos[0], robot_pos[1], drive_heading]}
                    )
                    await self._wait_for_heading(wait_heading)

            # Run the directed rotation, then the behavior substate.
            # The rotation task is awaited (not fire-and-forget) so we know
            # exactly when all 8 steps are done before the substate begins.
            # Both are cancellable by the outer _arbitrate preempt.
            await _directed_rotation()

            if use_path2:
                self._dbg(f"[Alert] Path2 — TerritorialWarning")
                await self._module_alert._territorial_warning()
            else:
                self._dbg(f"[Alert] Path3 — ShortWarningBurst")
                await self._module_alert._short_warning_burst()

        self._dbg("[Alert] Complete — resuming Scan")
        self._enter_scan()

    # ------------------------------------------------------------------
    # Manual Chase Cycle (manual mode only)
    # ------------------------------------------------------------------

    async def _manual_chase_cycle(self) -> None:
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

            await asyncio.sleep(2.0)
            idx += 1

    # ------------------------------------------------------------------
    # Stop Routine
    # ------------------------------------------------------------------

    async def Stop_Routine(self) -> None:
        # Hold the wheels. Without this the MPC keeps driving the last path after
        # MANUAL / Stop (W26 only stopped the dog).
        if not self.deter_mode:
            await self._SpeedCap_Publish(0.0)
        await self.MacroState_Publisher.publish({"Id": "Stop"})
        await self._Pose_Publish("IdlePose")
        await self._Audio_Publish("Stop")

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _pick_combo(self) -> Combo:
        """Next F8 combo (X3); every pick is logged."""
        combo = (self._combo_picker.pick() if self.CHASE_RANDOM_COMBOS
                 else fixed_combo(self.CHASE_APPROACH))
        self._chase_count += 1
        self._combo = combo
        print(f"[F8] chase #{self._chase_count}: combo {combo.describe()}")
        return combo

    def _speed_level(self, level: str) -> float:
        """Speed-profile level -> m/s, using only the existing chase limits (C7)."""
        return self.SPEED_CHASE if level == "full" else self.SPEED_CHASE_GROWL

    async def _publish_chase_move(self, turkey_pos) -> None:
        """W26's straight chase: Move toward the bird; Navigation goes direct inside 5 m."""
        await self.MotionCommand_Publisher.publish(
            {"Id": "Move", "Mode": "Chase", "Cmd": np.asarray(turkey_pos, dtype=float).tolist()}
        )
        self._dbg(f"[Chase/Nav] Move -> {np.round(turkey_pos, 2)}")

    async def _zigzag_step(self, zz: dict, turkey_pos: np.ndarray, now: float) -> None:
        """
        One tick of the zig-zag approach (X2). Publishes a plain Move only when
        a new leg starts: when the robot passes the current vertex, when the
        bird nears the edge of the image on the side this leg pushes it, when
        the bird has moved, or when a leg times out. When ZigZag has no leg
        left (a few metres out), finishes with the straight chase, which holds
        at CHASE_STOP_DIST_M.
        """
        robot = np.array(self.Current_RobotPose[:2], dtype=float)

        if zz["final"]:
            if np.linalg.norm(turkey_pos - robot) > self.ZIGZAG_RESTART_M:
                self._dbg("[ZigZag] Bird pulled away — zig-zag again")
                self._zigzag.reset()
                zz.update(final=False, leg=None, target=None, turkey_ref=None)
            else:
                if now - zz["t_final"] >= self.CHASE_NAV_RATE_S:
                    await self._publish_chase_move(turkey_pos)
                    zz["t_final"] = now
                return

        leg, reason = zz["leg"], "start"
        if leg is not None:
            bearing = self._bearing_of_turkey_at(turkey_pos)
            edge    = math.radians(self._zigzag.hfov_deg / 2.0 - self.ZIGZAG_BEARING_GUARD_DEG)
            along   = float(np.dot(robot - leg["start"], leg["dir"]))
            if along >= leg["length"] - self.ZIGZAG_SWITCH_EARLY_M:
                reason = "vertex"
            elif bearing is not None and -leg["side"] * bearing > edge:
                # On a leg to side s the bird sits at bearing ~ -s * offset.
                reason = "near FOV edge"
            elif np.linalg.norm(turkey_pos - zz["turkey_ref"]) > self.ZIGZAG_TURKEY_MOVED_M:
                reason = "bird moved"
            elif now - zz["t_leg"] > self.ZIGZAG_LEG_TIMEOUT_S:
                reason = "timeout"
            else:
                return

        nxt = self._zigzag.next_leg(robot, turkey_pos, last_target=zz["target"])
        if nxt is None:
            self._dbg(f"[ZigZag] {np.linalg.norm(turkey_pos - robot):.1f} m out — "
                      f"final straight approach")
            zz.update(final=True, leg=None, t_final=now)
            await self._publish_chase_move(turkey_pos)
            return

        info = self._zigzag.legs[-1]
        self._dbg(f"[ZigZag] leg {len(self._zigzag.legs)} ({reason}): {info['side']} "
                  f"{info['offset_deg']} deg x {info['length_m']} m -> {info['end']}")
        zz.update(leg=nxt, target=nxt["target"],
                  turkey_ref=np.array(turkey_pos, dtype=float), t_leg=now)
        await self.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": nxt["target"].tolist()})

    def _bearing_of_turkey_at(self, turkey_pos) -> float | None:
        """Camera bearing (col 2, + = left) of the visible turkey at turkey_pos, if any."""
        if not self.Turkeys:
            return None
        t = min(self.Turkeys, key=lambda e: np.linalg.norm(np.array(e[:2]) - turkey_pos))
        return float(t[2]) if len(t) >= 3 else None

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
        if any(np.linalg.norm(np.array(g) - candidate) < self.GHOST_DEDUP_RADIUS_M
               for g in self.GhostQueue):
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