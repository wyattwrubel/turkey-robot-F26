"""
ModuleChase.py

Canine behavior module for CHASE state.
Owns ONLY: poses, audio, servo commands.
All movement (Move, SpeedCap) is issued by StatePlannerReceiver.

IMPORTANT — loop design:
  Each behavior coroutine completes exactly ONE cycle then returns.
  StatePlannerReceiver calls them in a loop and re-evaluates the
  distance threshold between each cycle. This allows clean switching
  between AggressivePursuit and BurstAttack without mid-cycle cancellation.
"""

import asyncio
import random


class ModuleChase:

    # Distance threshold: > this -> AggressivePursuit, <= this -> BurstAttack
    TURKEY_NEAR_THRESHOLD_M = 10.0

    def __init__(self, parent):
        self.parent = parent

    # ------------------------------------------------------------------
    # Substates — each runs ONE cycle then returns
    # ------------------------------------------------------------------

    async def _growl_and_chase(self) -> None:
        """
        GrowlAndChase — single one-shot trigger on first sight of turkey.
        AggressivePose + Growl audio. Runs once for 3 seconds.
        W26's chase opening; the UI's GrowlAndChase button still runs it.
        """
        await self._opening("AggressivePose", "Growl")

    async def _opening(self, pose: str, sound: str) -> None:
        """
        Chase opening (X3): the combo's pose + sound, head centred, 3 seconds.
        Reported to the UI as the GrowlAndChase substate.
        """
        await self.parent.MacroState_Publisher.publish({"Id": "GrowlAndChase"})
        await self.parent._Pose_Publish(pose)
        await self.parent._ServoCommand_Publish("Head", 0.0)
        await self.parent._Audio_Publish(sound)
        await asyncio.sleep(3.0)

    async def _aggressive_pursuit(self) -> None:
        """
        AggressivePursuit — ONE cycle (~18 seconds).
        ExcitedGait + random head + AggressiveBark.
        Used when turkey is > TURKEY_NEAR_THRESHOLD_M away.
        Returns after one cycle so StatePlannerReceiver can re-evaluate distance.
        """
        await self.parent.MacroState_Publisher.publish({"Id": "AggressivePursuit"})
        await self.parent._Pose_Publish("ExcitedGait")
        await self.parent._ServoCommand_Publish("Head", random.uniform(-40.0, 40.0))
        await self.parent._Audio_Publish("AggressiveBark")
        await asyncio.sleep(18.0)
        # Returns — caller re-evaluates distance and calls again or switches

    async def _burst_attack(self) -> None:
        """
        BurstAttack — ONE full cycle (~21.6 seconds).
        ExcitedGait + alternating head snaps + bark pattern.
        Used when turkey is <= TURKEY_NEAR_THRESHOLD_M away.
        Returns after one cycle so StatePlannerReceiver can re-evaluate distance.
        """
        await self.parent.MacroState_Publisher.publish({"Id": "BurstAttack"})
        await self.parent._Pose_Publish("ExcitedGait")

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
        # Returns — caller re-evaluates distance and calls again or switches