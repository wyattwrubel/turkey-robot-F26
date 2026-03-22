"""
main.py — CDS Control Server entry point

Usage:
    python main.py
    Open http://localhost:8000

Simulation scenario (SimulatedVision):
    0s  - 10s  : No turkeys visible        -> SCAN mode, patrol waypoints
    10s - 20s  : Turkey inside zone        -> CHASE mode
    20s - 30s  : Turkey outside zone       -> ALERT mode, ghost enqueued
    30s - 40s  : No turkeys visible        -> SCAN mode, investigate ghost
    40s+       : Repeats
"""

import asyncio
import sys
import os
import math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UserInterface.App                  import CDSControlServer
from Navigation.NavigationDriver        import TrajectoryTracker
from Canine.CanineDriver                import CanineDriver
from DriveTrain.DriveTrainDriver        import MotorDriver
from Sensors.AccelPublisher             import IMUDriver
from Sensors.UWBPublisher               import UWB
from StatePlanner.StatePlannerReceiver  import StatePlannerReceiver
from Vision.SimpleVision                import Vision


# ── Publisher ─────────────────────────────────────────────────────────────────

class AsyncPublisher:
    def __init__(self, name: str = ""):
        self._name        = name
        self._subscribers = []

    def subscribe(self, callback):
        self._subscribers.append(callback)

    async def publish(self, data):
        for cb in self._subscribers:
            try:
                await cb(data)
            except Exception as e:
                print(f"[{self._name}] Subscriber error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():

    # ── Publishers ────────────────────────────────────────────────────────────

    VisionObject_Vision_pub          = AsyncPublisher("VisionObject")
    VideoFeed_Vision_pub             = AsyncPublisher("VideoFeed")

    Canine_UI_pub                    = AsyncPublisher("UI_Canine")
    Canine_StatePlanner_pub          = AsyncPublisher("StatePlanner_Canine")

    ServoFeedback_Canine_pub         = AsyncPublisher("CanineServoAngles")

    DriveCommand_Navigation_pub      = AsyncPublisher("Nav_DriveCommand")
    DriveCommand_UI_pub              = AsyncPublisher("UI_DriveCommand")

    SlipFeedback_Drivetrain_pub      = AsyncPublisher("DrivetrainSlip")

    RobotPose_Drivetrain_pub         = AsyncPublisher("RobotPose")

    MacroState_StatePlanner_pub      = AsyncPublisher("StatePlanner_MacroState")
    MacroState_UI_pub                = AsyncPublisher("UI_MacroState")

    MotionCommand_Navigation_pub     = AsyncPublisher("Nav_MotionCommand")
    MotionCommand_StatePlanner_pub   = AsyncPublisher("StatePlanner_MotionCommand")
    MotionCommand_UI_pub             = AsyncPublisher("UI_MotionCommand")

    UWB_UWBPublisher_pub             = AsyncPublisher("UWB")
    Accelerometer_AccelPublisher_pub = AsyncPublisher("IMU")

    # ── Module Instantiation ──────────────────────────────────────────────────

    # Vision handles both VideoFeed and VisionObject internally
    VideoVision = Vision(VideoFeed_Vision_pub, VisionObject_Vision_pub)

    Canine     = CanineDriver(ServoFeedback_Canine_pub, Verbose=False)
    Drivetrain = MotorDriver(RobotPose_Drivetrain_pub, SlipFeedback_Drivetrain_pub, Simulated=True)
    Navigation = TrajectoryTracker(DriveCommand_Navigation_pub, MotionCommand_Navigation_pub, Verbose=False)

    StatePlanner = StatePlannerReceiver(
        MacroState_StatePlanner_pub.publish,
        Canine_StatePlanner_pub.publish,
        MotionCommand_StatePlanner_pub.publish,
        Verbose=True,
    )

    # Inject patrol waypoints inside the zone
    StatePlanner.WaypointQueue = [
        [5.0,  5.0],
        [35.0, 5.0],
        [35.0, 70.0],
        [5.0,  70.0],
    ]

    UWB_Anchors   = UWB(UWB_UWBPublisher_pub)
    Accelerometer = IMUDriver(Accelerometer_AccelPublisher_pub, Simulation=True)

    UserInterface = CDSControlServer(
        DriveCommand_UI_pub,
        Canine_UI_pub,
        MacroState_UI_pub,
        MotionCommand_UI_pub,
    )

    # ── Subscriptions ─────────────────────────────────────────────────────────

    """VisionObject"""
    VisionObject_Vision_pub.subscribe(UserInterface.VisionObject_Receiver)
    VisionObject_Vision_pub.subscribe(Navigation.VisionObject_Receiver)
    VisionObject_Vision_pub.subscribe(StatePlanner.VisionObject_Receiver)

    """VideoFeed"""
    VideoFeed_Vision_pub.subscribe(UserInterface.VideoFeed_Receiver)

    """Canine"""
    Canine_UI_pub.subscribe(Canine.CanineDriver_Receiver)
    Canine_StatePlanner_pub.subscribe(Canine.CanineDriver_Receiver)

    """ServoFeedback"""
    ServoFeedback_Canine_pub.subscribe(UserInterface.diagnostics.CanineAngle_Receiver)

    """DriveCommand"""
    DriveCommand_Navigation_pub.subscribe(Drivetrain.MotorCommandReceiver)
    DriveCommand_UI_pub.subscribe(Drivetrain.MotorCommandReceiver)

    """SlipFeedback"""
    SlipFeedback_Drivetrain_pub.subscribe(UserInterface.diagnostics.Slip_Receiver)

    """RobotPose"""
    RobotPose_Drivetrain_pub.subscribe(Navigation.RobotPose_Receiver)
    RobotPose_Drivetrain_pub.subscribe(UserInterface.diagnostics.RobotPose_Receiver)
    RobotPose_Drivetrain_pub.subscribe(StatePlanner.RobotPose_Receiver)

    """MacroState"""
    MacroState_StatePlanner_pub.subscribe(UserInterface.diagnostics.MacroState_Receiver)
    MacroState_UI_pub.subscribe(StatePlanner.MacroState_Receiver)

    """MotionCommand"""
    MotionCommand_UI_pub.subscribe(Navigation.TrajectoryCMD_Receiver)
    MotionCommand_Navigation_pub.subscribe(UserInterface.diagnostics.MotionCommand_Receiver)
    MotionCommand_StatePlanner_pub.subscribe(Navigation.TrajectoryCMD_Receiver)

    """UWB"""
    UWB_UWBPublisher_pub.subscribe(Drivetrain.Drivetrain_UWBReceiver)
    UWB_UWBPublisher_pub.subscribe(UserInterface.diagnostics.UWB_Sensors_Receiver)

    """Accelerometer"""
    Accelerometer_AccelPublisher_pub.subscribe(Drivetrain.Drivetrain_AccelReceiver)
    Accelerometer_AccelPublisher_pub.subscribe(UserInterface.diagnostics.Accelerometer_Receiver)

    print("[MAIN] CDS Control Server starting")
    print("[MAIN] UI -> http://localhost:8000")
    print("[MAIN] Simulated vision cycle: Scan(10s) -> Chase(10s) -> Alert(10s) -> Ghost(10s)")

    await asyncio.gather(
        UserInterface.serve(),
        VideoVision.run(),
        Drivetrain.DriveTrainDriver_Runner(),
        Canine.PoseDriver.MotorPositions_Runner(),
        # UWB_Anchors.run(),       # requires real serial hardware (/tmp/vserial1)
        # Accelerometer.run(),
    )


if __name__ == "__main__":
    asyncio.run(main())