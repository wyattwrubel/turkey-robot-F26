import asyncio
import numpy as np

class ModuleAlert:

    def __init__(self, parent):
        self.parent    = parent
        self._path_idx = 0   # cycles Path1 -> Path2 -> Path3 -> Path1

    # ------------------------------------------------------------------
    # Substates
    # ------------------------------------------------------------------

    async def _watchful_dog(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "WatchfulDog"})
        await self.parent._Pose_Publish("AlertPose")
        await self.parent._SpeedCap_Publish(3.0)
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")
        
        for i in range(0,6):  
            await self.parent._Audio_Publish("BarkShort")
            await asyncio.sleep(5)

    async def _territorial_warning(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "TerritorialWarning"})
        await self.parent._SpeedCap_Publish(3.0)
        await self.parent._Pose_Publish("AlertStep")
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")
        
        for i in range(0,5):  
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1)
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2)

    async def _short_warning_burst(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "ShortWarningBurst"})
        await self.parent._SpeedCap_Publish(3.0)
        await self.parent._Pose_Publish("AlertStep")

        for i in range(4):
            await self.parent._Audio_Publish("BarkBurst")
            await self.parent._ServoCommand_Publish("Head", 40.0)
            await asyncio.sleep(5.0)
            await self.parent._ServoCommand_Publish("Head", -40.0)
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2.0)

