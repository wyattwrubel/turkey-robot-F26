"""
main.py — CDS Control Server entry point

Wires the UI server to your real system modules.
Search for ── FILL IN ── to find every placeholder that needs a real import.

Usage:
    python main.py
    Open http://localhost:8000

Publisher / Receiver wiring summary
─────────────────────────────────────────────────────────────────────
  INBOUND  (system → UI)
    robot_pose_pub   → ui.diagnostics.RobotPose_Receiver     [X, Y, theta]
    uwb_pub          → ui.diagnostics.UWB_Sensors_Receiver   [UWB_X, UWB_Y]
    imu_pub          → ui.diagnostics.Accelerometer_Receiver [theta]
    vision_pub       → ui.VisionObject_Receiver               {Turkeys, Humans}
    macrostate_pub   → ui.diagnostics.MacroState_Receiver     {Id}
    navigation_pub   → ui.map.Navigation_Receiver             {Id, Cmd}
    canine_pub       → ui.controls.Canine_Receiver            {Id, Type, Data}
    video_pub        → ui.VideoFeed_Receiver                  JPEG bytes

  OUTBOUND (UI → system)
    drivetrain_pub   → your drivetrain driver receiver
    canine_pub_out   → your canine driver receiver
    macrostate_pub_out → your state planner receiver
    navigation_pub_out → your trajectory tracker receiver
─────────────────────────────────────────────────────────────────────
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UserInterface.App import CDSControlServer
from Navigation.NavigationDriver  import TrajectoryTracker
from Canine.CanineDriver import CanineDriver
from DriveTrain.DriveTrainDriver import MotorDriver
from Sensors.AccelPublisher import IMUDriver
from Sensors.UWBPublisher import UWB



# STATE PLANNER IMPORT TODO
from StatePlanner.StatePlannerReceiver import StatePlannerReceiver


# TODO FAKE VISION REMOVE EVENTUALLY
from Vision.SimpleVision import Vision
# VISION IMPORT NEEDED TODO
from Vision.Vision import VisionObject

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

    # ── Inbound publishers (system → UI) ──────────────────────────────────────
    # These are owned by your hardware/algorithm modules and published into.
    # Subscribe the UI receivers below so the UI stays in sync.
    
    # VisionObject Protocol
    VisionObject_Vision_pub = AsyncPublisher("VideoObject") # Not passed into module TODO
    TESTING_VisionObject_Vision_pub = AsyncPublisher("TestVideoObject") # Not passed into module TODO

    # VideoFeed Protocol
    VideoFeed_Vision_pub = AsyncPublisher("VideoFeed")
    TESTING_VideoFeed_Vision_pub = AsyncPublisher("TestVideoFeed")

    # Canine Protocol
    Canine_UI_pub = AsyncPublisher("UI_Canine")
    Canine_StatePlanner_pub = AsyncPublisher("StatePlanner_Canine") # Not passed into module TODO
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

    # UWB
    UWB_UWBPublisher_pub = AsyncPublisher("UWB")
    
    # Accelerometer
    Accelerometer_AccelPublisher_pub = AsyncPublisher("IMU")
    
    # ── Instantiate Modules ───────────────
    # VISION OBJECT TODO ADD HERE

    # REPLACE WITH REAL VIDEOFEED TODO
    VisionObj = VisionObject(VideoFeed_Vision_pub, VisionObject_Vision_pub)
    VisionObj.start()

    vision = Vision(TESTING_VideoFeed_Vision_pub, TESTING_VisionObject_Vision_pub)
    
    # REAL MODULES TODO REPLACE THIS HEADER
    Canine = CanineDriver(ServoFeedback_Canine_pub, Canine_Canine_pub, Verbose=False)
    Drivetrain = MotorDriver(RobotPose_Drivetrain_pub, SlipFeedback_Drivetrain_pub, Simulated=True)

    Navigation = TrajectoryTracker(DriveCommand_Navigation_pub, MotionCommand_Navigation_pub, Verbose=False)

    # STATE PLANNER TODO
    StatePlanner = StatePlannerReceiver(MacroState_StatePlanner, Canine_StatePlanner_pub, MotionCommand_StatePlanner_pub, Verbose=False)

    # SENSORS
    UWB_Anchors = UWB(UWB_UWBPublisher_pub)
    Accelerometer = IMUDriver(Accelerometer_AccelPublisher_pub, Simulation=True)

    # User Interface
    UserInterface = CDSControlServer(DriveCommand_UI_pub, Canine_UI_pub, MacroState_UI_pub, MotionCommand_UI_pub)


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
    DriveCommand_UI_pub.subscribe(Drivetrain.MotorCommandReceiver)
 
    """SlipFeedback"""
    SlipFeedback_Drivetrain_pub.subscribe(UserInterface.diagnostics.Slip_Receiver)

    """RobotPose"""
    RobotPose_Drivetrain_pub.subscribe(Navigation.RobotPose_Receiver)
    RobotPose_Drivetrain_pub.subscribe(UserInterface.diagnostics.RobotPose_Receiver)
    RobotPose_Drivetrain_pub.subscribe(VisionObj.RobotPose_Receiver)
    RobotPose_Drivetrain_pub.subscribe(StatePlanner.RobotPose_Receiver)

    """MacroState"""
    MacroState_StatePlanner.subscribe(UserInterface.diagnostics.MacroState_Receiver)
    MacroState_UI_pub.subscribe(StatePlanner.MacroState_Receiver)

    """MotionCommand"""
    MotionCommand_UI_pub.subscribe(Navigation.TrajectoryCMD_Receiver)
    MotionCommand_Navigation_pub.subscribe(UserInterface.diagnostics.MotionCommand_Receiver)
    MotionCommand_StatePlanner_pub.subscribe(Navigation.TrajectoryCMD_Receiver)
     
    """UWB X,Y Values"""
    UWB_UWBPublisher_pub.subscribe(Drivetrain.Drivetrain_UWBReceiver)
    UWB_UWBPublisher_pub.subscribe(UserInterface.diagnostics.UWB_Sensors_Receiver)

    """Accelerometer Theta Values"""
    Accelerometer_AccelPublisher_pub.subscribe(Drivetrain.Drivetrain_AccelReceiver)
    Accelerometer_AccelPublisher_pub.subscribe(UserInterface.diagnostics.Accelerometer_Receiver)
    
    print("[MAIN] CDS Control Server starting")
    print("[MAIN] UI → http://localhost:8000")

    await asyncio.gather(
        UserInterface.serve(),

        # TODO FAKE VISION
        # vision.VisionRunner(),

        # SIMULATED MODULES
        Drivetrain.DriveTrainDriver_Runner(),
        Canine.PoseDriver.MotorPositions_Runner(),
        Canine.Audio._monitor(),

        Navigation.MPC.DriveTrain_Sender(),
        # ── FILL IN: add your real module coroutines here ──────────────────
        #
        # ukf.run(),
        # uwb_sensor.run(),
        # imu_sensor.run(),
        # detector.run(),
        # planner.run(),
        # tracker.run(),
        # canine.run(),
        # camera.run(),
        # motor.run(),
        #
        # ──────────────────────────────────────────────────────────────────
    )


if __name__ == "__main__":
    asyncio.run(main())