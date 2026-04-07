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
        
        for i in range(0, 2):  
            await self.parent._Audio_Publish("BarkShort")
            await asyncio.sleep(5)

    async def _territorial_warning(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "TerritorialWarning"})
        await self.parent._SpeedCap_Publish(3.0)
        await self.parent._Pose_Publish("AlertStep")
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")
        
        for i in range(0, 1):  
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1)
            await self.parent._Audio_Publish("BarkShort_3x")
            await asyncio.sleep(1)
            await self.parent._Audio_Publish("BarkShort_2x2")
            await asyncio.sleep(2)

    async def _lose_visual_routine(self) -> None:
        """
        Navigate to last known turkey position, then:
          - Concurrently: watchful_dog pose setup AND 360° spin
          - Sequentially after: territorial warning bark sequence
        """
        last_pos = self.parent.last_known_turkey_pos

        async def _navigate_and_spin():
            await self.parent.MacroState_Publisher.publish({"Id": "WatchfulDog"})
            await self.parent._Pose_Publish("AlertPose")
            await self.parent._SpeedCap_Publish(3.0)

            if last_pos is not None:
                self.parent._dbg(f"[Alert/LoseVisual] Navigating to last known pos {np.round(last_pos, 2)}")
                await self.parent.MotionCommand_Publisher.publish({"Id": "Move", "Cmd": last_pos[:2].tolist()})
                await self.parent._Pose_Publish("HeadScan")

                # Wait until we arrive (or time out after 15 s)
                for _ in range(30):
                    robot_pos = np.array(self.parent.Current_RobotPose[:2])
                    if np.linalg.norm(last_pos[:2] - robot_pos) < 1.0:
                        break
                    await asyncio.sleep(0.5)

            # 360° spin
            self.parent._dbg("[Alert/LoseVisual] Arrived — spinning 360°")
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

        async def _watchful_pose():
            """Run watchful_dog barks concurrently with navigation/spin."""
            await self._watchful_dog()

        await asyncio.gather(_navigate_and_spin(), _watchful_pose())