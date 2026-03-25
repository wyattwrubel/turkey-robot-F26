"""
ModuleChase.py

CHASE mode: pursue a turkey detected inside the zone boundary.

Inputs (via parent):
    parent.Current_RobotPose  -- [X, Y, theta]
    parent.Turkeys            -- list of [X, Y, angle_rad] in world coords

Outputs:
    MotionCommand: Move, SpeedCap
    Canine:        Pose, Servo, Audio
"""

import asyncio
import random
import numpy as np


class ModuleChase:

    TURKEY_NEAR_THRESHOLD_M = 5.0   # meters -- far: AggressivePursuit, close: BurstAttack

    def __init__(self, parent):
        self.parent = parent

    # ------------------------------------------------------------------
    # Entry Point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main chase loop.

        On entry, if turkey is far (>5m): fire GrowlAndChase once, loop AggressivePursuit.
        if turkey is near (<5m): loop BurstAttack
            Far  (>5m) -> AggressivePursuit
            Near (<5m) -> BurstAttack
        """
        try:
            if self._nearest_turkey_distance() > self.TURKEY_NEAR_THRESHOLD_M:
                await self._growl_and_chase()
                await self._aggressive_pursuit()
            else:
                await self._burst_attack()

        except asyncio.CancelledError:
            await self._SpeedCap_Publish(0.0)
            raise

    # ------------------------------------------------------------------
    # Substates
    # ------------------------------------------------------------------

    async def _growl_and_chase(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "GrowlAndChase"})
        await self._SpeedCap_Publish(0.5)
        await self._Audio_Publish("Growl")
        await self._Pose_Publish("AggressivePose")
        await self._ServoCommand_Publish("Head", 0.0)
        await asyncio.sleep(3.0)

    async def _aggressive_pursuit(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "AggressivePursuit"})
        await self._SpeedCap_Publish(3.0)
        await self._Pose_Publish("ExcitedGait")
        await self._ServoCommand_Publish("Head", random.uniform(-40.0, 40.0))
        await self._Audio_Publish("Stop")
        await asyncio.sleep(1)
        
        while True: 
            await self._Audio_Publish("AggressiveBark")
            await asyncio.sleep(18)

    async def _burst_attack(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "BurstAttack"})
        await self._SpeedCap_Publish(3.0)
        await self._Pose_Publish("ExcitedGait")

        while True: 
            await self._ServoCommand_Publish("Head", 30.0)
            await self._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(1.8)
            await self._ServoCommand_Publish("Head", -30.0)
            await self._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.8)
            await self._ServoCommand_Publish("Head", 30.0)
            await self._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.8)
            await self._ServoCommand_Publish("Head", -30.0)
            await self._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.8)
            await self._ServoCommand_Publish("Head", 30.0)
            await self._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(1.8)
            await self._ServoCommand_Publish("Head", -30.0)
            await self._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1.8)


    # ------------------------------------------------------------------
    # Publish Helpers
    # ------------------------------------------------------------------

    async def _Move_Publish(self, data: np.ndarray) -> None:
        await self.parent.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": data})

    async def _SpeedCap_Publish(self, data: np.ndarray) -> None:
        await self.parent.MotionCommand_Publisher.publish({"Id": "SpeedCap", "Cmd": data})

    async def _Audio_Publish(self, audio: str) -> None:
        await self.parent.Canine_Publisher.publish({"Id": "System", "Type": "Audio", "Data": audio})

    async def _ServoCommand_Publish(self, servoname: str, angle: float) -> None:
        await self.parent.Canine_Publisher.publish({"Id": "System", "Type": "Servo", "Name": servoname, "Data": angle})

    async def _Pose_Publish(self, pose: str) -> None:
        await self.parent.Canine_Publisher.publish({"Id": "System", "Type": "Pose", "Data": pose})

    # ------------------------------------------------------------------
    # Turkey Helpers
    # ------------------------------------------------------------------

    def _nearest_turkey_position(self) -> list | None:
        """Returns [X, Y] of the nearest turkey, or None if no turkeys visible."""
        if not self.parent.Turkeys:
            return None
        robot_xy = self.parent.Current_RobotPose[:2]
        return min(self.parent.Turkeys, key=lambda t: np.linalg.norm(np.array(t[:2]) - robot_xy))

    def _nearest_turkey_distance(self) -> float:
        """Euclidean distance (meters) to the nearest turkey."""
        target = self._nearest_turkey_position()

        print("TARGET ", target)
        print(self.parent.Turkeys)
        print(self.parent.Current_RobotPose[:2])
        
        if target is None:
            return float("inf")
        return float(np.linalg.norm(np.array(target[:2]) - self.parent.Current_RobotPose[:2]))
