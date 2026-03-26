import asyncio
import random
import numpy as np

class ModuleChase:

    TURKEY_NEAR_THRESHOLD_M = 5.0   # meters -- far: AggressivePursuit, close: BurstAttack

    def __init__(self, parent):
        self.parent = parent

    # ------------------------------------------------------------------
    # Substates
    # ------------------------------------------------------------------
    async def _growl_and_chase(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "GrowlAndChase"})
        await self.parent._SpeedCap_Publish(0.5)
        await self.parent._Audio_Publish("Growl")
        await self.parent._Pose_Publish("AggressivePose")
        await self.parent._ServoCommand_Publish("Head", 0.0)
        await asyncio.sleep(3.0)

    async def _aggressive_pursuit(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "AggressivePursuit"})
        await self.parent._SpeedCap_Publish(3.0)
        await self.parent._Pose_Publish("ExcitedGait")
        await self.parent._ServoCommand_Publish("Head", random.uniform(-40.0, 40.0))
        await self.parent._Audio_Publish("Stop")
        await asyncio.sleep(1)
        
        while True: 
            await self.parent._Audio_Publish("AggressiveBark")
            await asyncio.sleep(18)

    async def _burst_attack(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "BurstAttack"})
        await self.parent._SpeedCap_Publish(3.0)
        await self.parent._Pose_Publish("ExcitedGait")

        while True: 
            await self.parent._ServoCommand_Publish("Head", 30.0)
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(1.8)
            await self.parent._ServoCommand_Publish("Head", -30.0)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.8)
            await self.parent._ServoCommand_Publish("Head", 30.0)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.8)
            await self.parent._ServoCommand_Publish("Head", -30.0)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.8)
            await self.parent._ServoCommand_Publish("Head", 30.0)
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(1.8)
            await self.parent._ServoCommand_Publish("Head", -30.0)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.8)
