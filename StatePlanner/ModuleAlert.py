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

        Path1: WatchfulDog, TerritorialWarning, WatchfulDog,        WatchfulDog
        Path2: WatchfulDog, TerritorialWarning, ShortWarningBurst,  WatchfulDog
        Path3: ShortWarningBurst, WatchfulDog,  ShortWarningBurst,  WatchfulDog
        """
        alert_paths = [
            [self._watchful_dog, self._territorial_warning, self._watchful_dog,        self._watchful_dog],
            [self._watchful_dog, self._territorial_warning, self._short_warning_burst,  self._watchful_dog],
            [self._short_warning_burst, self._watchful_dog, self._short_warning_burst,  self._watchful_dog],
        ]

        try:
            path = alert_paths[self._path_idx % 3]
            self._path_idx += 1

            for substate_fn in path:
                await substate_fn()

            # Natural exit -- _on_module_exit will call _rearbitrate()

        except asyncio.CancelledError:
            await self._SpeedCap_Publish(np.array([0.0]))
            raise

    # ------------------------------------------------------------------
    # Substates
    # ------------------------------------------------------------------

    async def _watchful_dog(self) -> None:
        """WatchfulDog: AlertPose, HeadScan, BarkShort every 5s for 30s."""
        await self._Pose_Publish("AlertPose")
        await self._Pose_Publish("HeadScan")

        elapsed    = 0.0
        bark_timer = 0.0
        while elapsed < 30.0:
            await asyncio.sleep(0.5)
            elapsed    += 0.5
            bark_timer += 0.5
            if bark_timer >= 5.0:
                await self._Audio_Publish("BarkShort")
                bark_timer = 0.0

    async def _territorial_warning(self) -> None:
        """TerritorialWarning: AlertStep, Growl, slow head oscillate +-30 deg over 30s."""
        await self._Pose_Publish("AlertStep")
        await self._Audio_Publish("Growl")

        elapsed    = 0.0
        head_timer = 0.0
        head_dir   = 1
        while elapsed < 30.0:
            await asyncio.sleep(0.5)
            elapsed    += 0.5
            head_timer += 0.5
            if head_timer >= 5.0:
                await self._ServoCommand_Publish("Head", 30.0 * head_dir)
                head_dir  *= -1
                head_timer = 0.0

    async def _short_warning_burst(self) -> None:
        """ShortWarningBurst: 5x [BarkBurst, Head +40, AlertStep, wait 3s, AlertPose, Head -40, BarkBurst, wait 3s]."""
        for _ in range(5):
            await self._Audio_Publish("BarkBurst")
            await self._ServoCommand_Publish("Head", 40.0)
            await self._Pose_Publish("AlertStep")
            await asyncio.sleep(3.0)
            await self._Pose_Publish("AlertPose")
            await self._ServoCommand_Publish("Head", -40.0)
            await self._Audio_Publish("BarkBurst")
            await asyncio.sleep(3.0)

    # ------------------------------------------------------------------
    # Publish Helpers
    # ------------------------------------------------------------------

    async def _SpeedCap_Publish(self, data: np.ndarray) -> None:
        self.parent.MotionCommand_Publisher({"Id": "SpeedCap", "Cmd": data})

    async def _Audio_Publish(self, audio: str) -> None:
        self.parent.Canine_Publisher({"Id": "System", "Type": "Audio", "Data": audio})

    async def _ServoCommand_Publish(self, servoname: str, angle: float) -> None:
        self.parent.Canine_Publisher({"Id": "System", "Type": "Servo", "Name": servoname, "Data": angle})

    async def _Pose_Publish(self, pose: str) -> None:
        self.parent.Canine_Publisher({"Id": "System", "Type": "Pose", "Data": pose})