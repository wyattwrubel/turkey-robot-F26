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
# VISION IMPORT NEEDED TODO

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
    Drivetrain_robotpose_pub = AsyncPublisher("RobotPose")      # ── FILL IN: owned by your UKF / odometry module
    Sensor_uwb_pub        = AsyncPublisher("UWB")       # ── FILL IN: owned by your UWB sensor module
    Sensor_imu_pub        = AsyncPublisher("IMU")       # ── FILL IN: owned by your IMU module
    Navigation_drivetrain_pub     = AsyncPublisher("NavDrivetrain")    # ── FILL IN: owned by your trajectory tracker
    # Stateplanner_canine_pub  = AsyncPublisher("Canine") # ── FILL IN: owned by your canine driver
    # Stateplanner_macrostate_pub = AsyncPublisher("MacroState") # ── FILL IN: owned by your state planner
    # VISION 
    # STATE PLANNER
    # ── Instantiate Modules ───────────────
    
    # ── Outbound publishers (UI → system) ─────────────────────────────────────
    UI_drivetrain_pub  = AsyncPublisher("UI_Drivetrain")
    UI_canine_pub  = AsyncPublisher("UI_Canine")
    UI_macrostate_pub  = AsyncPublisher("UI_MacroState")
    UI_navigation_pub  = AsyncPublisher("UI_Navigation")

    Accelerometer = IMUDriver(Sensor_imu_pub, Simulation=True)
    UWB_Anchors = UWB(Sensor_uwb_pub)
    Drivetrain = MotorDriver(Drivetrain_robotpose_pub, Simulated=True)
    Navigation = TrajectoryTracker(Navigation_drivetrain_pub)
    Canine = CanineDriver(Verbose=True)
    # VISION 
    # STATE PLANNER

    # ── Instantiate UI server ──────────────────────────────────────────────────
    UserInterface = CDSControlServer(UI_drivetrain_pub, UI_canine_pub, UI_macrostate_pub, UI_navigation_pub)

    # ─────────────────────────────────────────────────────────────────────────
    """Vision Object"""
    # <VISION>.subscribe(Navigation.VisionObject_Receiver)
    # <VISION>.subscribe(UserInterface.VisionObject_Receiver)
    # <VISION>.subscribe(<STATE PLANNER>)

    """Canine """
    UI_canine_pub.subscribe(Canine.CanineDriver_Receiver)
    # <STATEPLANNER>.subscribe(Canine.CanineDriver_Receiver)

    """Drivetrain """
    Navigation_drivetrain_pub.subscribe(Drivetrain.MotorCommandReceiver)
    UI_drivetrain_pub.subscribe(Drivetrain.MotorCommandReceiver)
 
    """Robot Pose"""
    Drivetrain_robotpose_pub.subscribe(Navigation.RobotPose_Receiver)
    Drivetrain_robotpose_pub.subscribe(UserInterface.diagnostics.RobotPose_Receiver)
    # Drivetrain_robotpose_pub.subscribe("VISION")
    # Drivetrain_robotpose_pub.subscribe(STATE PLANNER)

    """Macro State"""
    # UI_macrostate_pub.subscribe(<STATE PLANNER>)
 
    """Navigation"""
    # STATE PLANNER
    UI_navigation_pub.subscribe(Navigation.TrajectoryCMD_Receiver)
     
    """Accelerometer Theta Values"""
    Sensor_imu_pub.subscribe(Drivetrain.Drivetrain_AccelReceiver)
    Sensor_imu_pub.subscribe(UserInterface.diagnostics.Accelerometer_Receiver)
    # Sensor_imu_pub.subscribe("VISION")

    """UWB X,Y Values"""
    Sensor_uwb_pub.subscribe(Drivetrain.Drivetrain_UWBReceiver)
    Sensor_uwb_pub.subscribe(UserInterface.diagnostics.UWB_Sensors_Receiver)
    # Sensor_uwb_pub.subscribe("VISION")
    # ─────────────────────────────────────────────────────────────────────────


    print("[MAIN] CDS Control Server starting")
    print("[MAIN] UI → http://localhost:8000")

    await asyncio.gather(
        UserInterface.serve(),

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