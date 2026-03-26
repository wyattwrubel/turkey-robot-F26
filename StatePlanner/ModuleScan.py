import numpy as np

class ModuleScan:

    def __init__(self, parent):
        self.parent = parent

    # ------------------------------------------------------------------
    # Substates
    # ------------------------------------------------------------------

    async def _travel_to_waypoint(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "TravelWaypoint"})
        await self._SpeedCap_Publish(2.0)
        await self._Pose_Publish("WalkGait")
        await self._Pose_Publish("HeadScan")
        
    async def _stationary_scan(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "StationaryScan"})
        await self._Pose_Publish("IdlePose")
        await self._Pose_Publish("HeadScan")

    async def _stationary_crouch(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "StationaryCrouch"})
        await self._Pose_Publish("AlertPose")

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