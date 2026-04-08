"""
ModuleScan.py

Canine behavior module for SCAN state.
Owns ONLY: poses, audio, servo commands.
All movement (Move, Orient, SpeedCap) is issued by StatePlannerReceiver.
"""

import asyncio


class ModuleScan:

    def __init__(self, parent):
        self.parent = parent

    # ------------------------------------------------------------------
    # Substates — behavior only, no navigation
    # ------------------------------------------------------------------

    async def _travel_to_waypoint(self) -> None:
        """Canine behavior while traveling between waypoints."""
        await self.parent.MacroState_Publisher.publish({"Id": "TravelWaypoint"})
        await self.parent._Pose_Publish("WalkGait")
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")

    async def _stationary_scan(self) -> None:
        """Canine behavior during an even rotation step at a waypoint."""
        await self.parent.MacroState_Publisher.publish({"Id": "StationaryScan"})
        await self.parent._Pose_Publish("IdlePose")
        await asyncio.sleep(0.1)
        await self.parent._Pose_Publish("HeadScan")

    async def _stationary_crouch(self) -> None:
        """Canine behavior during an odd rotation step at a waypoint."""
        await self.parent.MacroState_Publisher.publish({"Id": "StationaryCrouch"})
        await self.parent._Pose_Publish("AlertPose")