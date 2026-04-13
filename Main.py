import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UserInterface.App import CDSControlServer
from Navigation.NavigationDriver  import TrajectoryTracker
from Canine.CanineDriver import CanineDriver
from DriveTrain.DriveTrainDriver import MotorDriver
from Sensors.AccelPublisher import IMUDriver
from Sensors.PositioningPublisher import PositioningDriver
from StatePlanner.StatePlannerReceiver import StatePlannerReceiver

from Vision.Vision import VisionObject
from Vision.SimpleVision import FakeVision

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

async def main():

    # ─────────────────────────────────────────────────────────────────────────
    # ASYNC PUBLISHERS
    # ─────────────────────────────────────────────────────────────────────────

    # VisionObject Protocol
    VisionObject_Vision_pub = AsyncPublisher("VideoObject")
    TESTING_VisionObject_Vision_pub = AsyncPublisher("TestVideoObject")

    # VideoFeed Protocol
    VideoFeed_Vision_pub = AsyncPublisher("VideoFeed")
    TESTING_VideoFeed_Vision_pub = AsyncPublisher("TestVideoFeed")

    # Canine Protocol
    Canine_UI_pub = AsyncPublisher("UI_Canine")
    Canine_StatePlanner_pub = AsyncPublisher("StatePlanner_Canine")
    Canine_Canine_pub = AsyncPublisher("Canine_Canine")

    # ServoFeedback Protocol
    ServoFeedback_Canine_pub = AsyncPublisher("CanineServoAngles")

    # DriveCommand Protocol
    DriveCommand_Navigation_pub = AsyncPublisher("Nav_DriveCommand")
    DriveCommand_UI_pub  = AsyncPublisher("UI_DriveCommand")

    # SlipFeedback Protocol
    SlipFeedback_Drivetrain_pub = AsyncPublisher("DrivetrainSlip")

    # RobotPose Protocol
    RobotPose_Drivetrain_pub = AsyncPublisher("RobotPose")

    # MacroState
    MacroState_StatePlanner = AsyncPublisher("StatePlanner_MacroState")
    MacroState_UI_pub = AsyncPublisher("UI_MacroState")

    # MotionCommand
    MotionCommand_Navigation_pub = AsyncPublisher("Nav_MotionCommand")
    MotionCommand_StatePlanner_pub = AsyncPublisher("StatePlanner_MotionCommand")
    MotionCommand_UI_pub = AsyncPublisher("UI_MotionCommand")

    # MPCFeedback
    MPCFeedback_Navigation_pub = AsyncPublisher("Nav_MPC")

    # UWB
    GPS_GPSPublisher_pub = AsyncPublisher("GPS")

    # Accelerometer
    Accelerometer_AccelPublisher_pub = AsyncPublisher("IMU")

    # ─────────────────────────────────────────────────────────────────────────
    # SOFTWARE MODULES
    # ─────────────────────────────────────────────────────────────────────────

    Vision = VisionObject(VideoFeed_Vision_pub, VisionObject_Vision_pub, Simulated=False)

    TestVision = FakeVision(TESTING_VideoFeed_Vision_pub, TESTING_VisionObject_Vision_pub)

    Canine = CanineDriver(ServoFeedback_Canine_pub, Canine_Canine_pub, Simulated=False, Verbose=False)

    Drivetrain = MotorDriver(RobotPose_Drivetrain_pub, SlipFeedback_Drivetrain_pub, Simulated=True, EncoderLogging=True)

    Navigation = TrajectoryTracker(DriveCommand_Navigation_pub, MotionCommand_Navigation_pub, MPCFeedback_Navigation_pub, Verbose=False)

    StatePlanner = StatePlannerReceiver(MacroState_StatePlanner, Canine_StatePlanner_pub, MotionCommand_StatePlanner_pub, Verbose=False)

    PositionDriver = PositioningDriver(GPS_GPSPublisher_pub, Simulation=True, Verbose=True)

    Accelerometer = IMUDriver(Accelerometer_AccelPublisher_pub, Simulation=True, Log=True)

    UserInterface = CDSControlServer(DriveCommand_UI_pub, Canine_UI_pub, MacroState_UI_pub, MotionCommand_UI_pub)

    # ─────────────────────────────────────────────────────────────────────────
    # SUBSCRIPTION PATHS
    # ─────────────────────────────────────────────────────────────────────────

    """VisionObject"""
    VisionObject_Vision_pub.subscribe(UserInterface.VisionObject_Receiver)
    VisionObject_Vision_pub.subscribe(Navigation.VisionObject_Receiver)
    VisionObject_Vision_pub.subscribe(StatePlanner.VisionObject_Receiver)

    TESTING_VisionObject_Vision_pub.subscribe(UserInterface.VisionObject_Receiver)
    TESTING_VisionObject_Vision_pub.subscribe(Navigation.VisionObject_Receiver)
    TESTING_VisionObject_Vision_pub.subscribe(StatePlanner.VisionObject_Receiver)

    """VideoFeed"""
    VideoFeed_Vision_pub.subscribe(UserInterface.VideoFeed_Receiver)
    TESTING_VideoFeed_Vision_pub.subscribe(UserInterface.VideoFeed_Receiver)

    """Canine"""
    Canine_UI_pub.subscribe(Canine.CanineDriver_Receiver)
    Canine_StatePlanner_pub.subscribe(Canine.CanineDriver_Receiver)
    Canine_StatePlanner_pub.subscribe(UserInterface.diagnostics.CaninePose_Receiver)
    Canine_Canine_pub.subscribe(UserInterface.diagnostics.CaninePose_Receiver)

    """ServoFeedback"""
    ServoFeedback_Canine_pub.subscribe(UserInterface.diagnostics.CanineAngle_Receiver)

    """DriveCommand"""
    DriveCommand_Navigation_pub.subscribe(Drivetrain.MotorCommandReceiver)
    DriveCommand_Navigation_pub.subscribe(UserInterface.diagnostics.DriveCommand_Receiver)
    DriveCommand_UI_pub.subscribe(Drivetrain.MotorCommandReceiver)

    """SlipFeedback"""
    SlipFeedback_Drivetrain_pub.subscribe(UserInterface.diagnostics.Slip_Receiver)

    """RobotPose"""
    RobotPose_Drivetrain_pub.subscribe(Navigation.RobotPose_Receiver)
    RobotPose_Drivetrain_pub.subscribe(UserInterface.diagnostics.RobotPose_Receiver)
    RobotPose_Drivetrain_pub.subscribe(Vision.RobotPose_Receiver)
    RobotPose_Drivetrain_pub.subscribe(StatePlanner.RobotPose_Receiver)

    """MacroState"""
    MacroState_StatePlanner.subscribe(UserInterface.diagnostics.MacroState_Receiver)
    MacroState_UI_pub.subscribe(StatePlanner.MacroState_Receiver)

    """MotionCommand"""
    MotionCommand_UI_pub.subscribe(Navigation.TrajectoryCMD_Receiver)
    MotionCommand_Navigation_pub.subscribe(UserInterface.diagnostics.MotionCommand_Receiver)
    MotionCommand_StatePlanner_pub.subscribe(Navigation.TrajectoryCMD_Receiver)

    """MPCFeedback"""
    MPCFeedback_Navigation_pub.subscribe(UserInterface.diagnostics.TrajectoryTracker_Receiver)

    """GPS X,Y Values"""
    GPS_GPSPublisher_pub.subscribe(Drivetrain.Drivetrain_UWBReceiver)
    GPS_GPSPublisher_pub.subscribe(UserInterface.diagnostics.UWB_Sensors_Receiver)

    """Accelerometer Theta Values"""
    Accelerometer_AccelPublisher_pub.subscribe(Drivetrain.Drivetrain_AccelReceiver)
    Accelerometer_AccelPublisher_pub.subscribe(UserInterface.diagnostics.Accelerometer_Receiver)

    print("[MAIN] CDS Control Server Starting")

    await asyncio.gather(
        UserInterface.serve(),
        Drivetrain.DriveTrainDriver_Runner(),
        Canine.PoseDriver.MotorPositions_Runner(),
        Canine.Audio._monitor(),
        TestVision.VisionRunner(),
        Navigation.MPC.DriveTrain_Sender(),
        Accelerometer.Runner(),
        PositionDriver.Runner(),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down.")