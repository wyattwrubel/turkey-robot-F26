"""
StatePlannerReceiver.py
"""

import asyncio
import numpy as np
from enum import Enum, auto

from StatePlanner.ModuleScan  import ModuleScan
from StatePlanner.ModuleChase import ModuleChase
from StatePlanner.ModuleAlert import ModuleAlert
from StatePlanner.ZoneChecker import ZoneChecker

class Mode(Enum):
    SCAN  = auto()
    CHASE = auto()
    ALERT = auto()
    STOP  = auto()

class StatePlannerReceiver:

    GHOST_DEDUP_RADIUS_M   = 5.0
    CHASE_GOAL_THRESHOLD_M = 0.5
    CHASE_GOAL_UPDATE_RATE = 2.0

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

        self._scan_routine_idx  = 0
        self._alert_routine_idx = 0
        self._current_mode      = None
        self._current_submode   = None
        self._current_task: asyncio.Task | None = None
        self._active_auto_mode: Mode | None = None
        self._alert_running     = False
        self.target             = None

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

    def _dbg(self, msg: str) -> None:
        if self.Verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _cancel_current_task(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        self._current_task = None

    async def _switch_to(self, coro) -> None:
        self._dbg(f"[SwitchTo] target={np.round(self.target, 2) if self.target is not None else None}")
        await self.Stop_Routine()
        await self._cancel_current_task()
        self._current_task = asyncio.get_event_loop().create_task(coro)

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
        self.Turkeys = turkeys.tolist() if turkeys.size > 0 else []
        for g in (ghosts.tolist() if ghosts.size > 0 else []):
            self._try_enqueue_ghost(g[0], g[1])
        await self._arbitrate_auto()

    async def MacroState_Receiver(self, data: dict) -> None:
        mainstates = {"Scan", "Chase", "Alert", "Stop"}
        allowed_substates = {
            "TravelWaypoint", "StationaryScan", "StationaryCrouch",
            "AggressivePursuit", "BurstAttack", "GrowlAndChase",
            "WatchfulDog", "TerritorialWarning", "ShortWarningBurst",
        }
        state = data.get("Id", "")

        if state == "UnlockUI":
            self.IsManualMode      = False
            self._current_mode     = None
            self._current_submode  = None
            self._active_auto_mode = None
            self._alert_running    = False
            await self._arbitrate_auto()
            return

        self.IsManualMode = True
        if state in mainstates:
            self._current_mode, self._current_submode = state, None
        elif state in allowed_substates:
            self._current_mode, self._current_submode = None, state
        else:
            print(f"[StatePlanner] MacroState_Receiver: unknown state '{state}'")
            return
        await self._dispatch_manual()

    # ------------------------------------------------------------------
    # Manual dispatch
    # ------------------------------------------------------------------

    async def _dispatch_manual(self) -> None:
        mode, submode = self._current_mode, self._current_submode
        self._dbg(f"[StatePlanner] Manual dispatch — mode={mode}, submode={submode}")

        dispatch = {
            "Scan":                 lambda: self._switch_to(self.Scan_Routine()),
            "Chase":                lambda: self._switch_to(self.Chase_Routine()),
            "Alert":                lambda: self._switch_to(self.Alert_Routine()),
            "Stop":                 lambda: self.Stop_Routine(),
            "TravelWaypoint":       lambda: self._switch_to(self._module_scan._travel_to_waypoint()),
            "StationaryScan":       lambda: self._switch_to(self._module_scan._stationary_scan()),
            "StationaryCrouch":     lambda: self._switch_to(self._module_scan._stationary_crouch()),
            "AggressivePursuit":    lambda: self._switch_to(self._module_chase._aggressive_pursuit()),
            "BurstAttack":          lambda: self._switch_to(self._module_chase._burst_attack()),
            "GrowlAndChase":        lambda: self._switch_to(self._module_chase._growl_and_chase()),
            "WatchfulDog":          lambda: self._switch_to(self._module_alert._watchful_dog()),
            "TerritorialWarning":   lambda: self._switch_to(self._module_alert._territorial_warning()),
            "ShortWarningBurst":    lambda: self._switch_to(self._module_alert._short_warning_burst()),
        }
        key = mode or submode
        if key in dispatch:
            await dispatch[key]()

    # ------------------------------------------------------------------
    # Automatic arbitration
    # Chase preempts all. Alert is Chase-only cooldown. Default is Scan.
    # Chase triggers ONLY on is_inside() — visible-but-outside never chases.
    # ------------------------------------------------------------------

    async def _arbitrate_auto(self) -> None:
        turkey_in_zone = any(self.ZoneChecker.is_inside(t[0], t[1]) for t in self.Turkeys)

        if self.Verbose:
            status = [f"({t[0]:.2f},{t[1]:.2f})->{'IN' if self.ZoneChecker.is_inside(t[0], t[1]) else 'OUT'}" for t in self.Turkeys]
            print(f"[Arbitrate] mode={self._active_auto_mode} | alert_running={self._alert_running} | "
                  f"in_zone={turkey_in_zone} | turkeys={status or 'none'} | "
                  f"target={np.round(self.target, 2) if self.target is not None else None}")

        if turkey_in_zone:
            if self._active_auto_mode != Mode.CHASE:
                self._alert_running    = False
                self._active_auto_mode = Mode.CHASE
                self._dbg("[Arbitrate] -> CHASE")
                await self._switch_to(self.Chase_Routine())
            return

        if self._alert_running:
            return

        if self._active_auto_mode == Mode.CHASE:
            self._active_auto_mode = Mode.ALERT
            self._dbg("[Arbitrate] -> ALERT (end_of_chase)")
            await self._switch_to(self.Alert_Routine(end_of_chase=True))
            return

        if self._active_auto_mode != Mode.SCAN:
            self._active_auto_mode = Mode.SCAN
            self._dbg("[Arbitrate] -> SCAN")
            await self._switch_to(self.Scan_Routine())

    # ------------------------------------------------------------------
    # MacroStates
    # ------------------------------------------------------------------

    async def Scan_Routine(self):
        await self.MacroState_Publisher.publish({"Id": "Scan"})
        if self.target is None:
            self.target = self._next_target()
        self._dbg(f"[Scan] Starting — target={np.round(self.target, 2)}, idx={self._scan_routine_idx}")

        await self.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": self.target[:2].tolist()})
        await self._module_scan._travel_to_waypoint()

        while True:
            robot_pos = np.array(self.Current_RobotPose[:3])
            dist      = np.linalg.norm(self.target[:2] - robot_pos[:2])
            self._dbg(f"[Scan] dist={dist:.2f}m target={np.round(self.target[:2], 2)} pos={np.round(robot_pos[:2], 2)}")

            if dist < 1:
                self._dbg("[Scan] Reached waypoint — spinning")
                base_heading = float(self.Current_RobotPose[2])
                for i in range(1, 5):
                    if i % 4 == 0:
                        await self._module_scan._stationary_scan()
                    elif i % 4 == 2:
                        await self._module_scan._stationary_crouch()
                    robot_pos      = np.array(self.Current_RobotPose[:3])
                    target_heading = ((base_heading + i * np.pi / 2) + np.pi) % (2 * np.pi) - np.pi
                    await self.MotionCommand_Publisher.publish({"Id": "Orient", "Cmd": [robot_pos[0], robot_pos[1], target_heading]})
                    await asyncio.sleep(1)

                self.target = self._next_target()
                self._dbg(f"[Scan] Next target={np.round(self.target, 2)}, idx={self._scan_routine_idx}")
                await self.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": self.target[:2].tolist()})
                await self._module_scan._travel_to_waypoint()

            await asyncio.sleep(0.5)

    async def Chase_Routine(self):
        await self.MacroState_Publisher.publish({"Id": "Chase"})
        if not self.Turkeys:
            self._dbg("[Chase] No turkeys at startup — bailing")
            self._active_auto_mode = None
            return
        self._dbg(f"[Chase] Starting — {len(self.Turkeys)} turkey(s)")

        async def _move_loop():
            while True:
                robot_pos      = np.array(self.Current_RobotPose[:2])
                nearest_turkey = min(self.Turkeys, key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_pos))
                self.target    = np.array(nearest_turkey[:2])
                await self.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": self.target.tolist()})
                self._dbg(f"[Chase] Tracking {np.round(nearest_turkey[:2], 2)}, robot={np.round(robot_pos, 2)}")
                await asyncio.sleep(1.0)

        async def _gesture_loop():
            while True:
                if self.Turkeys is None:
                    return
                robot_pos      = np.array(self.Current_RobotPose[:2])
                nearest_turkey = min(self.Turkeys, key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_pos))
                dist           = np.linalg.norm(np.array(nearest_turkey[:2]) - robot_pos)
                self._dbg(f"[Chase] dist={dist:.2f}m")
                if dist > 5:
                    await self._module_chase._growl_and_chase()
                    await self._module_chase._aggressive_pursuit()
                else:
                    await self._module_chase._burst_attack()

        await asyncio.gather(_move_loop(), _gesture_loop())

    async def Alert_Routine(self, end_of_chase: bool = False):
        """
        Chase cooldown with two paths:
          end_of_chase=True  → TerminalChaseRoutine (turkey left zone)
          end_of_chase=False → LoseVisualRoutine    (lost visual mid-chase)
        Both hand off to Scan when complete.
        """
        self._alert_running = True
        await self.MacroState_Publisher.publish({"Id": "Alert"})
        self._dbg(f"[Alert] Starting — end_of_chase={end_of_chase}, target={np.round(self.target, 2) if self.target is not None else None}")

        async def _spin_loop():
            if self.target is not None:
                self._dbg(f"[Alert] Moving toward {np.round(self.target[:2], 2)}, waiting 10s")
                await self.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": self.target[:2].tolist()})
                await asyncio.sleep(10)
            base_heading = float(self.Current_RobotPose[2])
            self._dbg(f"[Alert] Spinning at {np.round(self.Current_RobotPose[:2], 2)}")
            for i in range(1, 12):
                robot_pos      = np.array(self.Current_RobotPose[:3])
                target_heading = ((base_heading + i * np.pi / 2) + np.pi) % (2 * np.pi) - np.pi
                await self.MotionCommand_Publisher.publish({"Id": "Orient", "Cmd": [robot_pos[0], robot_pos[1], target_heading]})
                await asyncio.sleep(1)
            self._dbg("[Alert] Spin complete")

        async def _gesture_loop():
            if end_of_chase:
                self._dbg("[Alert] TerminalChaseRoutine")
                await self._module_alert._territorial_warning()
                await self._module_alert._short_warning_burst()
                await self._module_alert._territorial_warning()
                await self._module_alert._short_warning_burst()
            else:
                self._alert_routine_idx += 1
                self._dbg(f"[Alert] LoseVisualRoutine idx={self._alert_routine_idx}")
                if self._alert_routine_idx % 2 == 1:
                    await self._module_alert._territorial_warning()
                    await self._module_alert._watchful_dog()
                else:
                    await self._module_alert._short_warning_burst()
                    await self._module_alert._watchful_dog()
            self._dbg("[Alert] Gestures complete")

        await asyncio.gather(_spin_loop(), _gesture_loop())

        self._alert_running    = False
        self._active_auto_mode = Mode.SCAN
        self.target            = self._next_target()
        self._dbg(f"[Alert] Complete — handing off to Scan, target={np.round(self.target, 2)}")
        await self._switch_to(self.Scan_Routine())

    def _next_target(self) -> np.ndarray:
        if self.GhostQueue:
            ghost = self.GhostQueue.pop(0)
            self._dbg(f"[NextTarget] Ghost {ghost}, {len(self.GhostQueue)} remaining")
            return np.array(ghost)
        target = self.WaypointQueue[self._scan_routine_idx % len(self.WaypointQueue)]
        self._dbg(f"[NextTarget] Waypoint idx={self._scan_routine_idx} -> {np.round(target, 2)}")
        self._scan_routine_idx += 1
        return target

    async def _orient_toward(self, target: np.ndarray) -> None:
        robot_pos = np.array(self.Current_RobotPose[:3])
        heading   = np.arctan2(-(target[1] - robot_pos[1]), target[0] - robot_pos[0])
        await self.MotionCommand_Publisher.publish({"Id": "Orient", "Cmd": [robot_pos[0], robot_pos[1], heading]})

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def Stop_Routine(self) -> None:
        await self.MacroState_Publisher.publish({"Id": "Stop"})
        await self.Canine_Publisher.publish({"Id": "UI", "Type": "Pose", "Data": "IdlePose"})
        await self._Audio_Publish("Stop")
        self._current_mode    = None
        self._current_submode = None
        self._alert_running   = False

    def _try_enqueue_ghost(self, x: float, y: float) -> None:
        candidate = np.array([x, y])
        if any(np.linalg.norm(np.array(g) - candidate) < self.GHOST_DEDUP_RADIUS_M for g in self.GhostQueue):
            return
        self.GhostQueue.append([x, y])
        self._dbg(f"[Ghost] Enqueued ({x:.2f},{y:.2f}), queue={len(self.GhostQueue)}")

    async def _SpeedCap_Publish(self, data: np.ndarray) -> None:
        await self.MotionCommand_Publisher.publish({"Id": "SpeedCap", "Cmd": data})

    async def _Audio_Publish(self, audio: str) -> None:
        src = "UI" if self.IsManualMode else "System"
        await self.Canine_Publisher.publish({"Id": src, "Type": "Audio", "Data": audio})

    async def _Pose_Publish(self, pose: str) -> None:
        src = "UI" if self.IsManualMode else "System"
        await self.Canine_Publisher.publish({"Id": src, "Type": "Pose", "Data": pose})

    async def _ServoCommand_Publish(self, servoname: str, angle: float) -> None:
        src = "UI" if self.IsManualMode else "System"
        await self.Canine_Publisher.publish({"Id": src, "Type": "Servo", "Name": servoname, "Data": angle})