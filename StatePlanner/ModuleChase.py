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

    TURKEY_NEAR_THRESHOLD_M = 10.0   # meters -- far: AggressivePursuit, close: BurstAttack

    def __init__(self, parent):
        self.parent = parent

    # ------------------------------------------------------------------
    # Entry Point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main chase loop.

        On entry, if turkey is far (>10m): fire GrowlAndChase once.
        Each iteration re-evaluates distance:
            Far  (>10m) -> AggressivePursuit
            Close (<10m) -> BurstAttack
        """
        try:
            if self._nearest_turkey_distance() > self.TURKEY_NEAR_THRESHOLD_M:
                await self._growl_and_chase()

            while True:
                if self._nearest_turkey_distance() > self.TURKEY_NEAR_THRESHOLD_M:
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
        """GrowlAndChase: 1 m/s, Growl, WalkGait, Head +30->0 deg, wait 3s, AggressiveBark, ExcitedGait."""
        await self.parent.MacroState_Publisher.publish({"Id": "GrowlAndChase"})
        await self._SpeedCap_Publish(1.0)
        await self._Audio_Publish("Growl")
        await self._Pose_Publish("WalkGait")
        await self._ServoCommand_Publish("Head", 30.0)
        await asyncio.sleep(1.5)
        await self._ServoCommand_Publish("Head", 0.0)
        await asyncio.sleep(3.0)
        await self._Audio_Publish("AggressiveBark")
        await self._Pose_Publish("ExcitedGait")

    async def _aggressive_pursuit(self) -> None:
        """
        AggressivePursuit: 3 m/s, ExcitedGait, random head oscillate +-40 deg,
        continuously republish Move toward nearest turkey, AggressiveBark, wait 18s.
        """
        await self.parent.MacroState_Publisher.publish({"Id": "AggressivePursuit"})
        await self._SpeedCap_Publish(3.0)
        await self._Pose_Publish("ExcitedGait")
        await self._ServoCommand_Publish("Head", random.uniform(-40.0, 40.0))
        await self._Audio_Publish("AggressiveBark")

        # Republish Move toward turkey every 0.5s for 18s
        elapsed = 0.0
        while elapsed < 18.0:
            target = self._nearest_turkey_position()
            if target is not None:
                await self._Move_Publish(np.array([target[0], target[1], 0.0]))
            await asyncio.sleep(0.5)
            elapsed += 0.5

    async def _burst_attack(self) -> None:
        """
        BurstAttack: 3 m/s, ExcitedGait, continuously pursue turkey,
        head snaps +45 and -30 deg, BarkBurst, 2s waits.
        """
        await self.parent.MacroState_Publisher.publish({"Id": "BurstAttack"})
        await self._SpeedCap_Publish(3.0)
        await self._Pose_Publish("ExcitedGait")

        # Republish Move during burst
        target = self._nearest_turkey_position()
        if target is not None:
            await self._Move_Publish(np.array([target[0], target[1], 0.0]))

        await self._ServoCommand_Publish("Head", 45.0)
        await self._Audio_Publish("BarkBurst")
        await asyncio.sleep(2.0)

        target = self._nearest_turkey_position()
        if target is not None:
            await self._Move_Publish(np.array([target[0], target[1], 0.0]))

        await self._ServoCommand_Publish("Head", -30.0)
        await self._Audio_Publish("BarkBurst")
        await asyncio.sleep(2.0)

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
        if target is None:
            return float("inf")
        return float(np.linalg.norm(np.array(target[:2]) - self.parent.Current_RobotPose[:2]))

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