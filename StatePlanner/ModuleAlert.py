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
        
        for i in range(3):
            await self.parent._Audio_Publish("BarkShort")
            await asyncio.sleep(5)

    async def _territorial_warning(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "TerritorialWarning"})
        await self.parent._SpeedCap_Publish(3.0)
        await self.parent._Pose_Publish("AlertStep")
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")
        
        for i in range(2):
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1)
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2)

    async def _Pose_setup_lose_visual(self) -> None:
        """Set alert pose and speed before the concurrent watchful_dog + navigate phase."""
        await self.parent.MacroState_Publisher.publish({"Id": "WatchfulDog"})
        await self.parent._Pose_Publish("AlertPose")
        await self.parent._SpeedCap_Publish(3.0)

    async def _lose_visual_routine(self) -> None:
        """
        Navigate to last known turkey position and wait for arrival.
        Runs concurrently with _watchful_dog in Alert_Routine (LoseVisual path).
        """
        last_pos = self.parent.last_known_turkey_pos
        if last_pos is not None:
            self.parent._dbg(f"[Alert/LoseVisual] Navigating to last known pos {np.round(last_pos, 2)}")
            await self.parent.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": last_pos[:2].tolist()})
            await self.parent._Pose_Publish("HeadScan")
            # Wait until arrival (15 s timeout)
            for _ in range(30):
                robot_pos = np.array(self.parent.Current_RobotPose[:2])
                if np.linalg.norm(last_pos[:2] - robot_pos) < 1.0:
                    break
                await asyncio.sleep(0.5)

    async def _spin_360(self) -> None:
        """Full 360° spin in 4 × 90° Orient steps — mirrors Scan_Routine spin."""
        self.parent._dbg("[Alert] Spinning 360°")
        base_heading = float(self.parent.Current_RobotPose[2])
        for i in range(1, 5):
            if i % 4 == 0:
                await self.parent._module_scan._stationary_scan()
            elif i % 4 == 2:
                await self.parent._module_scan._stationary_crouch()
            robot_pos      = np.array(self.parent.Current_RobotPose[:3])
            target_heading = ((base_heading + i * np.pi / 2) + np.pi) % (2 * np.pi) - np.pi
            await self.parent.MotionCommand_Publisher.publish(
                {"Id": "Orient", "Cmd": [robot_pos[0], robot_pos[1], target_heading]}
            )
            await asyncio.sleep(1)

    async def _short_warning_burst(self) -> None:

        await self.parent.MacroState_Publisher.publish({"Id": "ShortWarningBurst"})
        await self.parent._SpeedCap_Publish(3.0)
        await self.parent._Pose_Publish("AlertStep")

        for i in range(2):
            await self.parent._Audio_Publish("BarkBurst")
            await self.parent._ServoCommand_Publish("Head", 40.0)
            await asyncio.sleep(5.0)
            await self.parent._ServoCommand_Publish("Head", -40.0)
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2.0)