"""
ModuleAlert.py

ALERT mode: turkey detected outside the zone boundary.
2-minute alert window cycling Path1 -> Path2 -> Path3 -> Path1.

On natural exit, StatePlannerReceiver._on_module_exit fires _rearbitrate()
which will go to SCAN (ghost or waypoint) unless a turkey is still visible.

Inputs (via parent):
    parent.GhostQueue  -- informational only in this module

Outputs:
    MotionCommand: SpeedCap
    Canine:        Pose, Servo, Audio
"""

import asyncio
import numpy as np

class ModuleAlert:

    def __init__(self, parent):
        self.parent    = parent
        self._path_idx = 0   # cycles Path1 -> Path2 -> Path3 -> Path1

    # ------------------------------------------------------------------
    # Entry Point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Runs one 2-minute alert path then exits naturally.
        _on_module_exit in StatePlannerReceiver handles re-arbitration.

        Path1: TerritorialWarning, ShortWarningBurst, TerritorialWarning, ShortWarningBurst
        Path2: TerritorialWarning, WatchfulDog
        Path3: ShortWarnngBurst, WatchfulDog
        """
        
        try:
            border_distance = self.parent.ZoneChecker.border_distance(self.parent.Current_RobotPose[0], self.parent.Current_RobotPose[1])
            if(border_distance < 2):
                await self._territorial_warning()
                await self._short_warning_burst()
                await self._territorial_warning()
                await self._short_warning_burst()
            else:
                if(self._path_idx % 2 == 1):
                    await self._territorial_warning()
                    await self._watchful_dog()
                else:
                    await self._short_warning_burst()
                    await self._watchful_dog()
                self._path_idx+=1

        except asyncio.CancelledError:
            await self._SpeedCap_Publish(0.0)
            raise

    # ------------------------------------------------------------------
    # Substates
    # ------------------------------------------------------------------

    async def _watchful_dog(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "WatchfulDog"})
        await self._Pose_Publish("AlertPose")
        await self._SpeedCap_Publish(3.0)
        await self._Pose_Publish("HeadScan")
        
        for i in range(0,6):  
            await self._Audio_Publish("BarkShort")
            await asyncio.sleep(5)

    async def _territorial_warning(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "TerritorialWarning"})
        await self._SpeedCap_Publish(3.0)
        await self._Pose_Publish("AlertStep")
        
        for i in range(0,5):  
            await self._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2)
            await self._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1)
            await self._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1)
            await self._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2)


    async def _short_warning_burst(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "ShortWarningBurst"})
        await self._SpeedCap_Publish(3.0)

        for i in range(5):
            await self._Audio_Publish("BarkBurst")
            await self._ServoCommand_Publish("Head", 40.0)
            await self._Pose_Publish("AlertStep")
            await asyncio.sleep(3.0)
            await self._Pose_Publish("AlertPose")
            await self._ServoCommand_Publish("Head", -40.0)
            await self._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(3.0)

    # ------------------------------------------------------------------
    # Publish Helpers
    # ------------------------------------------------------------------

    async def _SpeedCap_Publish(self, data: np.ndarray) -> None:
        await self.parent.MotionCommand_Publisher.publish({"Id": "SpeedCap", "Cmd": data})

    async def _Audio_Publish(self, audio: str) -> None:
        await self.parent.Canine_Publisher.publish({"Id": "System", "Type": "Audio", "Data": audio})

    async def _ServoCommand_Publish(self, servoname: str, angle: float) -> None:
        await self.parent.Canine_Publisher.publish({"Id": "System", "Type": "Servo", "Name": servoname, "Data": angle})

    async def _Pose_Publish(self, pose: str) -> None:
        await self.parent.Canine_Publisher.publish({"Id": "System", "Type": "Pose", "Data": pose})