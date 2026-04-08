"""
ModuleAlert.py

Canine behavior module for ALERT state.
Owns ONLY: poses, audio, servo commands.
All movement, timing, and path selection is owned by StatePlannerReceiver.

Each substate runs its full fixed duration and then returns.
StatePlannerReceiver sequences them and handles interruption via task cancellation.
"""

import asyncio


class ModuleAlert:

    def __init__(self, parent):
        self.parent = parent

    # ------------------------------------------------------------------
    # Substates — 30 seconds each, behavior only
    # ------------------------------------------------------------------

    async def _watchful_dog(self) -> None:
        """
        WatchfulDog — 30 seconds.
        AlertPose + HeadScan. Short barks every 5 seconds.
        Used as the wind-down substate in Path2 and Path3 (lose visual).
        """
        await self.parent.MacroState_Publisher.publish({"Id": "WatchfulDog"})
        await self.parent._Pose_Publish("AlertPose")
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")

        for _ in range(2):
            await self.parent._Audio_Publish("BarkShort")
            await asyncio.sleep(5.0)

    async def _territorial_warning(self) -> None:
        """
        TerritorialWarning — 30 seconds.
        AlertStep + HeadScan. Escalating bark pattern repeated twice.
        Used in Path1 (turkey outside zone) and Path2 (lose visual).
        """
        await self.parent.MacroState_Publisher.publish({"Id": "TerritorialWarning"})
        await self.parent._Pose_Publish("AlertStep")
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")

        for _ in range(2):
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2.0)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.0)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.0)
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2.0)

    async def _short_warning_burst(self) -> None:
        """
        ShortWarningBurst — 30 seconds.
        AlertStep. Head snaps + BarkBurst pattern repeated twice.
        Used in Path1 (turkey outside zone) and Path3 (lose visual).
        """
        await self.parent.MacroState_Publisher.publish({"Id": "ShortWarningBurst"})
        await self.parent._Pose_Publish("AlertStep")

        for _ in range(2):
            await self.parent._Audio_Publish("BarkBurst")
            await self.parent._ServoCommand_Publish("Head", 40.0)
            await asyncio.sleep(3.0)
            await self.parent._ServoCommand_Publish("Head", -40.0)
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(3.0)