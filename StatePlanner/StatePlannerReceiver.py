"""
StatePlannerReceiver.py

Receives sensor and UI messages, arbitrates which mode is active,
and delegates execution to the appropriate module.

Inputs:  MacroState, RobotPose, VisionObject
Outputs: MacroState, Canine, MotionCommand (via modules)

GhostQueue semantics:
    Turkeys that were seen but have since left the zone boundary.
    These are priority investigation targets for ModuleScan.
    Deduplicated within GHOST_DEDUP_RADIUS_M to avoid redundant entries
    from the same turkey being seen across multiple frames.
"""

import asyncio
import numpy as np
from enum import Enum, auto

from StatePlanner.ModuleScan  import ModuleScan
from StatePlanner.ModuleChase import ModuleChase
from StatePlanner.ModuleAlert import ModuleAlert
from StatePlanner.ZoneChecker import ZoneChecker


# ------------------------------------------------------------------
# Mode Enum
# ------------------------------------------------------------------

class Mode(Enum):
    SCAN  = auto()
    CHASE = auto()
    ALERT = auto()

# ------------------------------------------------------------------
# StatePlannerReceiver
# ------------------------------------------------------------------


        # # Republish Move toward turkey every 0.5s for 18s
        # elapsed = 0.0
        # while elapsed < 18.0:
        #     target = self._nearest_turkey_position()
        #     if target is not None:
        #         await self._Move_Publish(np.array([target[0], target[1], 0.0]))
        #     await asyncio.sleep(0.5)
        #     elapsed += 0.5


        # # Republish Move during burst
        # target = self._nearest_turkey_position()
        # if target is not None:
        #     await self._Move_Publish(np.array([target[0], target[1], 0.0]))

    # ------------------------------------------------------------------
    # Turkey Helpers
    # ------------------------------------------------------------------


class StatePlannerReceiver:

    # Minimum distance (meters) between ghost entries -- prevents duplicate
    # ghost waypoints from the same turkey seen across multiple frames
    GHOST_DEDUP_RADIUS_M = 5.0

    def __init__(
        self,
        MacroState_Publisher,
        Canine_Publisher,
        MotionCommand_Publisher,
        Verbose: bool = False,
    ):
        self.MacroState_Publisher    = MacroState_Publisher
        self.Canine_Publisher        = Canine_Publisher
        self.MotionCommand_Publisher = MotionCommand_Publisher
        self.Verbose                 = Verbose

        # Shared state -- written by receivers, read by modules via parent reference
        self.Current_RobotPose = np.array([0.0, 0.0, 0.0])  # [X, Y, theta]
        self.IsManualMode      = False
        self.WaypointQueue     = []   # list of [X, Y] -- injected externally
        self.Turkeys           = []   # list of [X, Y, angle_rad] -- currently visible in world coords

        # GhostQueue: last known positions of turkeys that left the zone.
        # list of [X, Y]. ModuleScan navigates these in order, highest priority first.
        self.GhostQueue = []

        # Zone boundary -- raises RuntimeError on missing/malformed file
        self.ZoneChecker = ZoneChecker()

        # Modules -- instantiated once, tasks created/destroyed on mode switches
        self._module_scan  = ModuleScan(self)
        self._module_chase = ModuleChase(self)
        self._module_alert = ModuleAlert(self)

        # Active task
        self._current_mode: Mode | None         = None
        self._current_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Mode Arbitration
    # ------------------------------------------------------------------

    async def _switch_mode(self, new_mode: Mode) -> None:
        """Cancel the running module task and spin up the new one."""

        if new_mode == self._current_mode:
            return

        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        self._current_mode = new_mode

        module_map = {
            Mode.SCAN:  self._module_scan,
            Mode.CHASE: self._module_chase,
            Mode.ALERT: self._module_alert,
        }

        mode_map = {
            Mode.SCAN: "Scan",
            Mode.CHASE: "Chase",
            Mode.ALERT: "Alert",
        }
        
        await self.MacroState_Publisher.publish({"Id": mode_map[new_mode]})

        self._current_task = asyncio.create_task(module_map[new_mode].run())
        self._current_task.add_done_callback(self._on_module_exit)

        if self.Verbose:
            print(f"[StatePlanner] Switched to mode: {new_mode.name}")

    def _on_module_exit(self, task: asyncio.Task) -> None:
        """
        Called when a module task ends naturally (not cancelled).
        Schedules re-arbitration -- primarily used when Alert finishes
        its 2-minute sequence.
        """
        if task.cancelled():
            return
        if task.exception():
            print(f"[StatePlanner] Module {self._current_mode} crashed: {task.exception()}")
            return
        asyncio.create_task(self._rearbitrate())

    async def _rearbitrate(self) -> None:
        """
        Decide next mode after a natural module exit or UnlockUI.

        Priority order:
          1. Turkey visible inside zone  -> CHASE
          2. Turkey visible outside zone -> ALERT
          3. Default                     -> SCAN (ModuleScan drains GhostQueue first)
        """
        turkey_in_zone  = any(self.ZoneChecker.is_inside(t[0], t[1])  for t in self.Turkeys)
        turkey_out_zone = any(self.ZoneChecker.is_outside(t[0], t[1]) for t in self.Turkeys)

        if turkey_in_zone:
            await self._switch_mode(Mode.CHASE)
        elif turkey_out_zone:
            await self._switch_mode(Mode.ALERT)
        else:
            await self._switch_mode(Mode.SCAN)

    # ------------------------------------------------------------------
    # Ghost Queue Management
    # ------------------------------------------------------------------

    def _try_enqueue_ghost(self, x: float, y: float) -> None:
        """
        Add [X, Y] to GhostQueue only if no existing ghost entry
        is within GHOST_DEDUP_RADIUS_M. Prevents the same turkey
        from generating redundant investigation waypoints across frames.
        """
        candidate = np.array([x, y])
        for ghost in self.GhostQueue:
            if np.linalg.norm(np.array(ghost) - candidate) < self.GHOST_DEDUP_RADIUS_M:
                return  # duplicate -- skip
        self.GhostQueue.append([x, y])
        if self.Verbose:
            print(f"[StatePlanner] Ghost enqueued at ({x:.2f}, {y:.2f}). "
                  f"Queue length: {len(self.GhostQueue)}")

    # ------------------------------------------------------------------
    # Receiver Functions
    # ------------------------------------------------------------------

    async def MacroState_Receiver(self, data: dict) -> None:
        """High-level system state from User Interface."""

        mainstates = {"Scan", "Chase", "Alert"}

        allowed_substates = {
            "TravelWaypoint", "StationaryScan", "StationaryCrouch",
            "AggressivePursuit", "BurstAttack", "GrowlAndChase",
            "WatchfulDog", "TerritorialWarning", "ShortWarningBurst", "Stop"
        }

        state = data.get("Id", "")

        if state == "UnlockUI":
            self.IsManualMode = False
            await self._rearbitrate()
            return

        self.IsManualMode = True

        mode_map = {
            "Scan":  Mode.SCAN,
            "Chase": Mode.CHASE,
            "Alert": Mode.ALERT,
        }

        if state in mainstates:
            await self._switch_mode(mode_map[state])

        elif state in allowed_substates:
            substate_map = {
                "TravelWaypoint":     lambda: self._module_scan._travel_to(
                                          self._module_scan._current_waypoint() or np.zeros(2)),
                "StationaryScan":     self._module_scan._stationary_scan,
                "StationaryCrouch":   self._module_scan._stationary_crouch,
                "AggressivePursuit":  self._module_chase._aggressive_pursuit,
                "BurstAttack":        self._module_chase._burst_attack,
                "GrowlAndChase":      self._module_chase._growl_and_chase,
                "WatchfulDog":        self._module_alert._watchful_dog,
                "TerritorialWarning": self._module_alert._territorial_warning,
                "ShortWarningBurst":  self._module_alert._short_warning_burst,
                "Stop":               self.StopAll,
            }

            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
                try:
                    await self._current_task
                except asyncio.CancelledError:
                    pass

            self._current_task = asyncio.create_task(substate_map[state]())
            self._current_task.add_done_callback(self._on_module_exit)

        else:
            print(f"[StatePlanner] MacroState_Receiver: invalid data '{data}'")

    async def StopAll(self):
        await self._module_scan._Pose_Publish("IdlePose")
        await self._module_scan._Audio_Publish("Stop")
        await self.MacroState_Publisher.publish({"Id": "Stop"})

    async def RobotPose_Receiver(self, data) -> None:
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 3:
            self.Current_RobotPose = arr[:3]
        else:
            print(f"[StatePlanner] RobotPose_Receiver: invalid data '{data}'")

    async def VisionObject_Receiver(self, data: dict) -> None:
        turkeys = np.asarray(data.get("Turkeys", []), dtype=float)

        self.Turkeys = turkeys.tolist() if turkeys.size > 0 else []

        # Turkeys seen outside the zone become ghost investigation targets
        for t in self.Turkeys:
            if self.ZoneChecker.is_outside(t[0], t[1]):
                self._try_enqueue_ghost(t[0], t[1])

        if not self.IsManualMode:
            turkey_in_zone  = any(self.ZoneChecker.is_inside(t[0], t[1])  for t in self.Turkeys)
            turkey_out_zone = any(self.ZoneChecker.is_outside(t[0], t[1]) for t in self.Turkeys)

            if turkey_in_zone:
                await self._switch_mode(Mode.CHASE)
            elif turkey_out_zone:
                await self._switch_mode(Mode.ALERT)
            else:
                await self._switch_mode(Mode.SCAN)