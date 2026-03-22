"""
ModuleScan.py

SCAN mode: patrol waypoints and scan for targets.

GhostQueue priority: if GhostQueue is non-empty, navigate to ghost
locations first. On arrival, perform a full 360-degree scan, then
drop that ghost. Once GhostQueue is empty, resume normal WaypointQueue
patrol.

Inputs (via parent):
    parent.Current_RobotPose  -- [X, Y, theta]
    parent.WaypointQueue      -- list of [X, Y], injected externally
    parent.GhostQueue         -- list of [X, Y], drained here in SCAN mode

Outputs:
    MotionCommand: Move, Orient, SpeedCap
    Canine:        Pose, Servo, Audio
"""

import asyncio
import numpy as np


class ModuleScan:

    WAYPOINT_ARRIVAL_THRESHOLD_M = 0.5   # meters -- close enough to consider arrived
    GHOST_CLEAR_RADIUS_M         = 3.0   # meters -- close enough to begin ghost scan

    def __init__(self, parent):
        self.parent = parent

        # Persists across task restarts
        self._waypoint_idx = 0

    # ------------------------------------------------------------------
    # Entry Point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main scan loop.

        Each iteration:
          1. If GhostQueue non-empty: navigate to ghost, do full 360 scan, clear ghost
          2. Otherwise: travel to next waypoint, do standard scan rotations
        """
        try:
            while True:
                if self.parent.GhostQueue:
                    await self._investigate_ghost()
                else:
                    await self._patrol_waypoint()

        except asyncio.CancelledError:
            await self._SpeedCap_Publish(np.array([0.0]))
            raise

    # ------------------------------------------------------------------
    # Ghost Investigation
    # ------------------------------------------------------------------

    async def _investigate_ghost(self) -> None:
        """
        Navigate to GhostQueue[0]. On arrival within GHOST_CLEAR_RADIUS_M,
        perform a full 360-degree scan, then remove ghost from queue.
        """
        ghost_target = np.array(self.parent.GhostQueue[0])

        if self.parent.Verbose:
            print(f"[Scan] Investigating ghost at {ghost_target}. "
                  f"Remaining in queue: {len(self.parent.GhostQueue)}")

        await self._travel_to(ghost_target, threshold=self.GHOST_CLEAR_RADIUS_M)
        await self._full_360_scan()

        self.parent.GhostQueue.pop(0)

        if self.parent.Verbose:
            print(f"[Scan] Ghost cleared. Remaining: {len(self.parent.GhostQueue)}")

    async def _full_360_scan(self) -> None:
        """
        Perform 3 full 360-degree rotations in place.
        Each rotation steps by scan_rotation_rad, alternating
        StationaryScan and StationaryCrouch at each step.
        """
        steps_per_rotation = max(1, round((2 * np.pi) / self.scan_rotation_rad))

        if self.parent.Verbose:
            print(f"[Scan] Full 360 scan x3: {steps_per_rotation} steps of "
                  f"{np.rad2deg(self.scan_rotation_rad):.1f} deg each")

        step_count = 0
        for _ in range(3):
            for i in range(steps_per_rotation):
                target_theta = self.parent.Current_RobotPose[2] + (self.scan_rotation_rad * (i + 1))
                await self._Orient_Publish(np.array([
                    self.parent.Current_RobotPose[0],
                    self.parent.Current_RobotPose[1],
                    target_theta,
                ]))
                await asyncio.sleep(1.0)

                if step_count % 2 == 0:
                    await self._stationary_scan()
                else:
                    await self._stationary_crouch()

                step_count += 1

    # ------------------------------------------------------------------
    # Waypoint Patrol
    # ------------------------------------------------------------------

    async def _patrol_waypoint(self) -> None:
        """Travel to next waypoint and perform standard scan rotations."""
        waypoint = self._current_waypoint()
        if waypoint is None:
            if self.parent.Verbose:
                print("[Scan] No waypoints in queue -- waiting")
            await asyncio.sleep(1.0)
            return

        await self._travel_to(waypoint, threshold=self.WAYPOINT_ARRIVAL_THRESHOLD_M)
        await self._scan_rotations()
        self._advance_waypoint()

    async def _scan_rotations(self) -> None:
        """
        Waypoint scan: 3 x (wait 0.5s, rotate 90 deg CCW, wait 0.5s).
        Alternates StationaryScan / StationaryCrouch at each step.
        Covers 270 deg total at each waypoint stop.
        """
        rotation_rad = np.deg2rad(90.0)   # 90 deg CCW per step
        steps        = 3                   # 3 rotations per waypoint

        for i in range(steps):
            await asyncio.sleep(0.5)       # camera dwell before rotation
            target_theta = self.parent.Current_RobotPose[2] + (rotation_rad * (i + 1))
            await self._Orient_Publish(np.array([
                self.parent.Current_RobotPose[0],
                self.parent.Current_RobotPose[1],
                target_theta,
            ]))
            await asyncio.sleep(0.5)       # camera dwell after rotation

            if i % 2 == 0:
                await self._stationary_scan()
            else:
                await self._stationary_crouch()

    # ------------------------------------------------------------------
    # Shared Travel
    # ------------------------------------------------------------------

    async def _travel_to(self, target: np.ndarray, threshold: float) -> None:
        """
        WalkGait, publish Move toward target, head oscillate +-45 deg every 3s,
        BarkShort every 10s. Returns once within threshold meters of target.
        """
        await self._SpeedCap_Publish(np.array([1.0]))
        await self._Pose_Publish("WalkGait")
        await self._Move_Publish(np.array([target[0], target[1], 0.0]))

        bark_timer = 0.0
        head_timer = 0.0
        head_dir   = 1

        while True:
            await asyncio.sleep(0.5)
            bark_timer += 0.5
            head_timer += 0.5

            if head_timer >= 3.0:
                await self._ServoCommand_Publish("Head", 45.0 * head_dir)
                head_dir  *= -1
                head_timer = 0.0

            if bark_timer >= 10.0:
                await self._Audio_Publish("BarkShort")
                bark_timer = 0.0

            dist = np.linalg.norm(self.parent.Current_RobotPose[:2] - target[:2])
            if dist <= threshold:
                break

    # ------------------------------------------------------------------
    # Substates
    # ------------------------------------------------------------------

    async def _stationary_scan(self) -> None:
        """StationaryScan: AlertPose, HeadScan, BarkShort, pause 4s."""
        await self._Pose_Publish("AlertPose")
        await self._Pose_Publish("HeadScan")
        await self._Audio_Publish("BarkShort")
        await asyncio.sleep(4.0)

    async def _stationary_crouch(self) -> None:
        """StationaryCrouch: AggressivePose, pause 1s."""
        await self._Pose_Publish("AggressivePose")
        await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Waypoint Helpers
    # ------------------------------------------------------------------

    def _current_waypoint(self) -> np.ndarray | None:
        if not self.parent.WaypointQueue:
            return None
        return np.array(self.parent.WaypointQueue[self._waypoint_idx % len(self.parent.WaypointQueue)])

    def _advance_waypoint(self) -> None:
        if self.parent.WaypointQueue:
            self._waypoint_idx = (self._waypoint_idx + 1) % len(self.parent.WaypointQueue)

    # ------------------------------------------------------------------
    # Publish Helpers
    # ------------------------------------------------------------------

    async def _Move_Publish(self, data: np.ndarray) -> None:
        await self.parent.MotionCommand_Publisher({"Id": "Move", "Cmd": data})

    async def _Orient_Publish(self, data: np.ndarray) -> None:
        await self.parent.MotionCommand_Publisher({"Id": "Orient", "Cmd": data})

    async def _SpeedCap_Publish(self, data: np.ndarray) -> None:
        await self.parent.MotionCommand_Publisher({"Id": "SpeedCap", "Cmd": data})

    async def _Audio_Publish(self, audio: str) -> None:
        await self.parent.Canine_Publisher({"Id": "System", "Type": "Audio", "Data": audio})

    async def _ServoCommand_Publish(self, servoname: str, angle: float) -> None:
        await self.parent.Canine_Publisher({"Id": "System", "Type": "Servo", "Name": servoname, "Data": angle})

    async def _Pose_Publish(self, pose: str) -> None:
        await self.parent.Canine_Publisher({"Id": "System", "Type": "Pose", "Data": pose})