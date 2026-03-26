import numpy as np
import asyncio

class ModuleScan:

    def __init__(self, parent):
        self.parent = parent

    # ------------------------------------------------------------------
    # Substates
    # ------------------------------------------------------------------

    async def _travel_to_waypoint(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "TravelWaypoint"})
        await self.parent._SpeedCap_Publish(2.0)
        await self.parent._Pose_Publish("WalkGait")
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")
        
    async def _stationary_scan(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "StationaryScan"})
        await self.parent._Pose_Publish("IdlePose")
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")

    async def _stationary_crouch(self) -> None:
        await self.parent.MacroState_Publisher.publish({"Id": "StationaryCrouch"})
        await self.parent._Pose_Publish("AlertPose")
