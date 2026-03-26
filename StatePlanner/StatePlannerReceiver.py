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

    GHOST_DEDUP_RADIUS_M = 5.0

    def __init__(self, MacroState_Publisher, Canine_Publisher, MotionCommand_Publisher, Verbose=False):
        self.MacroState_Publisher    = MacroState_Publisher
        self.Canine_Publisher        = Canine_Publisher
        self.MotionCommand_Publisher = MotionCommand_Publisher
        self.Verbose                 = Verbose

        self.Current_RobotPose = np.array([0.0, 0.0, 0.0])
        self.IsManualMode      = False
        self.WaypointQueue = [np.array([10, 10]), np.array([20, 20])]
        self.Turkeys           = []
        self.GhostQueue        = []

        self.ZoneChecker    = ZoneChecker()
        self._module_scan   = ModuleScan(self)
        self._module_chase  = ModuleChase(self)
        self._module_alert  = ModuleAlert(self)

        self._scan_routine_idx = 0 

        self._current_mode    = None
        self._current_submode = None
        self._current_task: asyncio.Task | None = None
        self._active_auto_mode: Mode | None = None  # Track what auto is currently running

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
        """Cancel the running task and start a new one."""
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
            await self._arbitrate_auto()
            return

        self.IsManualMode = True

        if state in mainstates:
            self._current_mode    = state
            self._current_submode = None
        elif state in allowed_substates:
            self._current_submode = state
            self._current_mode    = None
        else:
            print(f"[StatePlanner] MacroState_Receiver: unknown state '{state}'")
            return

        await self._dispatch_manual()

    # ------------------------------------------------------------------
    # Manual dispatch
    # ------------------------------------------------------------------

    async def _dispatch_manual(self) -> None:
        mode    = self._current_mode
        submode = self._current_submode

        if self.Verbose:
            print(f"[StatePlanner] Manual dispatch — mode={mode}, submode={submode}")

        if mode == "Scan":
            await self._switch_to(self.Scan_Routine())
        elif mode == "Chase":
            await self._switch_to(self.Chase_Routine())
        elif mode == "Alert":
            await self._switch_to(self.Alert_Routine())
        elif mode == "Stop":
            await self.Stop_Routine()

        elif submode == "TravelWaypoint":
            await self._switch_to(self._module_scan._travel_to_waypoint())
        elif submode == "StationaryScan":
            await self._switch_to(self._module_scan._stationary_scan())
        elif submode == "StationaryCrouch":
            await self._switch_to(self._module_scan._stationary_crouch())
        elif submode == "AggressivePursuit":
            await self._switch_to(self._module_chase._aggressive_pursuit())
        elif submode == "BurstAttack":
            await self._switch_to(self._module_chase._burst_attack())
        elif submode == "GrowlAndChase":
            await self._switch_to(self._module_chase._growl_and_chase())
        elif submode == "WatchfulDog":
            await self._switch_to(self._module_alert._watchful_dog())
        elif submode == "TerritorialWarning":
            await self._switch_to(self._module_alert._territorial_warning())
        elif submode == "ShortWarningBurst":
            await self._switch_to(self._module_alert._short_warning_burst())

    # ------------------------------------------------------------------
    # Automatic arbitration  (Chase > Alert > Scan)
    # ------------------------------------------------------------------

    async def _arbitrate_auto(self) -> None:
        """Called whenever sensor data changes. Switches task only if mode changed."""
        
        if any(self.ZoneChecker.is_inside(turkey[0], turkey[1]) for turkey in self.Turkeys):
            target = Mode.CHASE
        elif (self._active_auto_mode == Mode.CHASE):
            target = Mode.ALERT
        elif (self._current_task == None):
            target = Mode.SCAN

        if target == self._active_auto_mode:
            return

        self._active_auto_mode = target

        if self.Verbose and target != None:
            print(f"[StatePlanner] Auto mode → {target.name}")

        if target == Mode.CHASE:
            await self._switch_to(self.Chase_Routine())
        elif target == Mode.ALERT:
            await self._switch_to(self.Alert_Routine())
        else:
            await self._switch_to(self.Scan_Routine())

    # ------------------------------------------------------------------
    # MacroStates
    # ------------------------------------------------------------------

    async def Scan_Routine(self):

        await self.MacroState_Publisher.publish({"Id": "Scan"})
        if (self.Verbose):
            print("[StatePlannerReceiver] Starting Scan Routine")        
        
        # If at a waypoint: Finish scan cycle

        # IF new new waypoint: Grab next spot to investigate from ghost
        # If now new ghosts: Grab from next in queue
        
        target = None

        if(self.GhostQueue):
            target = self.GhostQueue[0]
            self.GhostQueue.pop(0)
        else:
            target = self.WaypointQueue[self._scan_routine_idx % len(self.WaypointQueue)]
            self._scan_routine_idx+=1

        while True: 
            
            robot_pos = np.array(self.Current_RobotPose[:3])
            distance_to_waypoint = np.linalg.norm(np.array(target[:2]) - robot_pos)

            if(distance_to_waypoint < 1):
                # Initiate scan routine: 3 full rotations, 90 degree turns, 2 second delay per, gesture every other
                for i in range(1,12):
                    if i % 4 == 0: 
                        await self.Scan_Routine._stationary_scan()
                    elif i % 4 == 2:
                        await self.Scan_Routine._stationary_crouch()
                        robot_pos[2] = ((robot_pos[2] + np.pi) + np.pi) % (2 * np.pi) - np.pi

                    await self.MotionCommand_Publisher.publish({"Id": "Orient", "Cmd": robot_pos[:3]})  
                    await asyncio.sleep(2)
                if(self.GhostQueue):
                    target = self.GhostQueue[0]
                    self.GhostQueue.pop(0)
                else:
                    target = self.WaypointQueue[self._scan_routine_idx % len(self.WaypointQueue)]
                    self._scan_routine_idx+=1
                    await self.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": robot_pos[2]})  
                    await self.Scan_Routine._travel_to_waypoint()
            else:
                await asyncio.sleep(0.5)                
            
        
    async def Chase_Routine(self):
        await self.MacroState_Publisher.publish({"Id": "Chase"})

        if not self.Turkeys: 
            return

        robot_pos = np.array(self.Current_RobotPose[:2])  # [x, y]
        nearest_turkey = min(self.Turkeys, key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_pos))
        await self.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": nearest_turkey})

        distance_away = np.linalg.norm(np.array(nearest_turkey[:2] - robot_pos))

        if(distance_away > 5):
            TurkeyNear = False
        else:
            TurkeyNear = True

        if (self.Verbose):
            print("[StatePlannerReceiver] Starting Chase Routine")        
        while True: 
            if(TurkeyNear == False):
                if (self.Verbose):
                    print("[StatePlannerReceiver] Starting Chase GrowlAndChase")

                await self._module_chase._growl_and_chase()
                
                if (self.Verbose):
                    print("[StatePlannerReceiver] Starting Chase AggressivePursuit")

                await self._module_chase._aggressive_pursuit()
            else:
                if (self.Verbose):
                    print("[StatePlannerReceiver] Starting Chase BurstAttack")

                await self._module_chase._burst_attack()
            
    async def Alert_Routine(self):
        await self.MacroState_Publisher.publish({"Id": "Alert"})
        if (self.Verbose):
            print("[StatePlannerReceiver] Starting Alert Routine")
        
        if(self.Turkeys):
            EndOfChase=True
        else:
            EndOfChase=False
            
        if(EndOfChase == True):
            if (self.Verbose):
                print("[StatePlannerReceiver] Starting Alert TerminalChaseRoutine")
                
            await self._module_alert._territorial_warning()
            await self._module_alert._short_warning_burst()
            await self._module_alert._territorial_warning()
            await self._module_alert._short_warning_burst()
        else:
            if (self.Verbose):
                print("[StatePlannerReceiver] Starting Alert LoseVisualRoutine")
        
            self._scan_routine_idx+=1

            if(self._scan_routine_idx % 2 == 1):
                await self._module_alert._territorial_warning()
                await self._module_alert._watchful_dog()
            else:
                await self._module_alert._short_warning_burst()
                await self._module_alert._watchful_dog()
        
        await self._switch_to(self.Scan_Routine())


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    
    async def Stop_Routine(self) -> None:
        await self.MacroState_Publisher.publish({"Id": "Stop"})
        await self.Canine_Publisher.publish({"Id": "UI", "Type": "Pose", "Data": "IdlePose"})
        await self._Audio_Publish("Stop")
        self._current_mode    = None
        self._current_submode = None
        await self._cancel_current_task()

    def _try_enqueue_ghost(self, x: float, y: float) -> None:
        candidate = np.array([x, y])
        for ghost in self.GhostQueue:
            if np.linalg.norm(np.array(ghost) - candidate) < self.GHOST_DEDUP_RADIUS_M:
                return
        self.GhostQueue.append([x, y])
        if self.Verbose:
            print(f"[StatePlanner] Ghost enqueued at ({x:.2f}, {y:.2f}). "
                  f"Queue length: {len(self.GhostQueue)}")

    async def _SpeedCap_Publish(self, data: np.ndarray) -> None:
        await self.MotionCommand_Publisher.publish({"Id": "SpeedCap", "Cmd": data})

    async def _Audio_Publish(self, audio: str) -> None:
        if(self.IsManualMode == True):
            await self.Canine_Publisher.publish({"Id": "UI", "Type": "Audio", "Data": audio})
        else:
            await self.Canine_Publisher.publish({"Id": "System", "Type": "Audio", "Data": audio})

    async def _Pose_Publish(self, pose: str) -> None:
        if(self.IsManualMode == True):
            await self.Canine_Publisher.publish({"Id": "UI", "Type": "Pose", "Data": pose})
        else:
            await self.Canine_Publisher.publish({"Id": "System", "Type": "Pose", "Data": pose})


    async def _ServoCommand_Publish(self, servoname: str, angle: float) -> None:
        if(self.IsManualMode == True):
            await self.Canine_Publisher.publish({"Id": "UI", "Type": "Servo", "Name": servoname, "Data": angle})
        else:
            await self.Canine_Publisher.publish({"Id": "System", "Type": "Servo", "Name": servoname, "Data": angle})
