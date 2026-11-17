"""
Main.py: starts the robot software in one of four run profiles (DR2 plan, contract C1).

    python Main.py --profile full     everything (W26's setup); needs the drivetrain battery
    python Main.py --profile vision   camera + detector + UI; drivetrain may be unpowered
    python Main.py --profile deter    vision + the dog (head + bark); wheels never commanded
    python Main.py --profile sim      laptop, no hardware: fake turkey, simulated drive and dog

Options:
    --ui ops|present which page port 8000 serves: the full operator UI (default) or the
                     presentation copy with no zone map (UserInterface/WebsitePresent)
    --gps sim|real   real GPS receiver (full only; default sim)
    --imu sim|real   real IMU (full only; default sim until the gyro-heading work, W1, lands:
                     today's IMU mode is the magnetometer heading the robot's motors corrupt)
    --run-for N      stop after N seconds (used by tests/check_profiles.py)

A profile builds only the modules it lists, and every subscription below skips a module the
profile left out, so nothing assumes another module exists.
"""

import argparse
import asyncio
import contextlib
import logging
import operator
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UserInterface.App import CDSControlServer
from Navigation.NavigationDriver  import TrajectoryTracker
from Navigation.Geofence import Geofence
from Canine.CanineDriver import CanineDriver
from DriveTrain.DriveTrainDriver import MotorDriver
from Sensors.AccelPublisher import IMUDriver
from Sensors.PositioningPublisher import PositioningDriver
from StatePlanner.StatePlannerReceiver import StatePlannerReceiver

from Vision.Vision import VisionObject
from Vision.SimpleVision import FakeVision

# What each profile builds (contract C1). None = module left out.
#   vision: "camera" = the robot's camera (VisionObject), "fake" = FakeVision's fixed turkey
#   canine / drivetrain: "real" hardware or "sim"
#   sensors: GPS + IMU drivers (real or simulated per --gps / --imu)
#   The robot geofence (Navigation/Geofence.py, C4) runs whenever a drivetrain gives a pose.
PROFILES = {
    "full":   dict(vision="camera", canine="real", drivetrain="real", navigation=True,  planner="normal", sensors=True),
    "vision": dict(vision="camera", canine=None,   drivetrain=None,   navigation=False, planner=None,     sensors=False),
    "deter":  dict(vision="camera", canine="real", drivetrain=None,   navigation=False, planner="deter",  sensors=False),
    "sim":    dict(vision="fake",   canine="sim",  drivetrain="sim",  navigation=True,  planner="normal", sensors=True),
}


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


def rx(module, name):
    """A module's receiver method, or None when the profile left the module out."""
    return None if module is None else operator.attrgetter(name)(module)


def link(publisher, *receivers):
    for receiver in receivers:
        if receiver is not None:
            publisher.subscribe(receiver)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Start the turkey robot software.")
    ap.add_argument("--profile", choices=PROFILES, default="full")
    ap.add_argument("--ui", choices=("ops", "present"), default="ops",
                    help="page served on port 8000: operator UI or the presentation copy")
    ap.add_argument("--gps", choices=("sim", "real"), default="sim")
    ap.add_argument("--imu", choices=("sim", "real"), default="sim")
    ap.add_argument("--run-for", type=float, default=None, metavar="SECONDS")
    args = ap.parse_args(argv)
    if args.profile != "full" and "real" in (args.gps, args.imu):
        ap.error("--gps real and --imu real only apply to --profile full")
    return args


async def main(args):
    p = PROFILES[args.profile]
    print(f"[MAIN] Profile '{args.profile}': " + ", ".join(f"{k}={v}" for k, v in p.items())
          + f", gps={args.gps if p['sensors'] else None}, imu={args.imu if p['sensors'] else None}")

    # ─────────────────────────────────────────────────────────────────────────
    # ASYNC PUBLISHERS
    # ─────────────────────────────────────────────────────────────────────────

    VisionObject_Vision_pub = AsyncPublisher("VideoObject")
    VideoFeed_Vision_pub = AsyncPublisher("VideoFeed")

    Canine_UI_pub = AsyncPublisher("UI_Canine")
    Canine_StatePlanner_pub = AsyncPublisher("StatePlanner_Canine")
    Canine_Canine_pub = AsyncPublisher("Canine_Canine")

    ServoFeedback_Canine_pub = AsyncPublisher("CanineServoAngles")

    DriveCommand_Navigation_pub = AsyncPublisher("Nav_DriveCommand")
    DriveCommand_UI_pub  = AsyncPublisher("UI_DriveCommand")

    SlipFeedback_Drivetrain_pub = AsyncPublisher("DrivetrainSlip")

    RobotPose_Drivetrain_pub = AsyncPublisher("RobotPose")

    MacroState_StatePlanner = AsyncPublisher("StatePlanner_MacroState")
    MacroState_UI_pub = AsyncPublisher("UI_MacroState")

    MotionCommand_Navigation_pub = AsyncPublisher("Nav_MotionCommand")
    MotionCommand_StatePlanner_pub = AsyncPublisher("StatePlanner_MotionCommand")
    MotionCommand_UI_pub = AsyncPublisher("UI_MotionCommand")

    MPCFeedback_Navigation_pub = AsyncPublisher("Nav_MPC")

    GPS_GPSPublisher_pub = AsyncPublisher("GPS")

    Accelerometer_AccelPublisher_pub = AsyncPublisher("IMU")

    Geofence_pub = AsyncPublisher("Geofence")

    # ─────────────────────────────────────────────────────────────────────────
    # SOFTWARE MODULES (only the ones this profile lists)
    # ─────────────────────────────────────────────────────────────────────────

    if p["vision"] == "camera":
        Vision = VisionObject(VideoFeed_Vision_pub, VisionObject_Vision_pub, Simulated=False)
    else:
        Vision = FakeVision(VideoFeed_Vision_pub, VisionObject_Vision_pub)

    Canine = None
    if p["canine"]:
        Canine = CanineDriver(ServoFeedback_Canine_pub, Canine_Canine_pub,
                              Simulated=(p["canine"] == "sim"), Verbose=False)

    Drivetrain = None
    if p["drivetrain"]:
        simulated = p["drivetrain"] == "sim"
        Drivetrain = MotorDriver(RobotPose_Drivetrain_pub, SlipFeedback_Drivetrain_pub,
                                 Simulated=simulated, EncoderLogging=not simulated)

    # TrajectoryTracker starts the MPC drive loop itself (NavigationDriver.py). W26 also
    # started it from the gather below, so the loop ran twice (found by Xinyuan).
    Navigation = None
    if p["navigation"]:
        Navigation = TrajectoryTracker(DriveCommand_Navigation_pub, MotionCommand_Navigation_pub,
                                       MPCFeedback_Navigation_pub, Verbose=False)

    # Fix type until the F9P reports its own (fix / carrSoln): "sim" or the old puck's "gps".
    Fence = None
    if Drivetrain:
        Fence = Geofence(Geofence_pub, fix="gps" if p["sensors"] and args.gps == "real" else "sim")

    # The planner gates every motion command on the fence (C4): near_edge caps speed,
    # outside / unknown latch STOP, and with --gps real a fix worse than rtk_float stops too.
    StatePlanner = None
    if p["planner"] == "normal":
        StatePlanner = StatePlannerReceiver(MacroState_StatePlanner, Canine_StatePlanner_pub,
                                            MotionCommand_StatePlanner_pub, Verbose=False,
                                            geofence=Fence, gps_real=(p["sensors"] and args.gps == "real"))
    elif p["planner"] == "deter":
        # No fence in deter: the wheels are never built (C8).
        StatePlanner = StatePlannerReceiver(MacroState_StatePlanner, Canine_StatePlanner_pub,
                                            MotionCommand_StatePlanner_pub, Verbose=False,
                                            deter_mode=True)

    PositionDriver = Accelerometer = None
    if p["sensors"]:
        PositionDriver = PositioningDriver(GPS_GPSPublisher_pub, Simulation=(args.gps == "sim"), Verbose=True)
        Accelerometer = IMUDriver(Accelerometer_AccelPublisher_pub, Simulation=(args.imu == "sim"),
                                  Log=(args.imu == "real"))

    UI = CDSControlServer(DriveCommand_UI_pub, Canine_UI_pub, MacroState_UI_pub, MotionCommand_UI_pub,
                          site=args.ui)

    # ─────────────────────────────────────────────────────────────────────────
    # SUBSCRIPTION PATHS (a receiver on a missing module is skipped)
    # ─────────────────────────────────────────────────────────────────────────

    """VisionObject"""
    link(VisionObject_Vision_pub,
         rx(UI, "VisionObject_Receiver"),
         rx(Navigation, "VisionObject_Receiver"),
         rx(StatePlanner, "VisionObject_Receiver"))

    """VideoFeed"""
    link(VideoFeed_Vision_pub, rx(UI, "VideoFeed_Receiver"))

    """Canine"""
    link(Canine_UI_pub, rx(Canine, "CanineDriver_Receiver"))
    link(Canine_StatePlanner_pub, rx(Canine, "CanineDriver_Receiver"),
         rx(UI, "diagnostics.CaninePose_Receiver"))
    link(Canine_Canine_pub, rx(UI, "diagnostics.CaninePose_Receiver"))

    """ServoFeedback"""
    link(ServoFeedback_Canine_pub, rx(UI, "diagnostics.CanineAngle_Receiver"))

    """DriveCommand"""
    link(DriveCommand_Navigation_pub, rx(Drivetrain, "MotorCommandReceiver"),
         rx(UI, "diagnostics.DriveCommand_Receiver"))
    link(DriveCommand_UI_pub, rx(Drivetrain, "MotorCommandReceiver"))

    """SlipFeedback"""
    link(SlipFeedback_Drivetrain_pub, rx(UI, "diagnostics.Slip_Receiver"))

    """RobotPose"""
    link(RobotPose_Drivetrain_pub,
         rx(Navigation, "RobotPose_Receiver"),
         rx(UI, "diagnostics.RobotPose_Receiver"),
         rx(Vision, "RobotPose_Receiver") if p["vision"] == "camera" else None,
         rx(StatePlanner, "RobotPose_Receiver"),
         rx(Fence, "RobotPose_Receiver"))

    """MacroState"""
    link(MacroState_StatePlanner, rx(UI, "diagnostics.MacroState_Receiver"))
    link(MacroState_UI_pub, rx(StatePlanner, "MacroState_Receiver"))

    """MotionCommand"""
    link(MotionCommand_UI_pub, rx(Navigation, "TrajectoryCMD_Receiver"))
    link(MotionCommand_Navigation_pub, rx(UI, "diagnostics.MotionCommand_Receiver"))
    link(MotionCommand_StatePlanner_pub, rx(Navigation, "TrajectoryCMD_Receiver"))

    """MPCFeedback"""
    link(MPCFeedback_Navigation_pub, rx(UI, "diagnostics.TrajectoryTracker_Receiver"))

    """GPS X,Y Values"""
    link(GPS_GPSPublisher_pub, rx(Drivetrain, "Drivetrain_UWBReceiver"),
         rx(UI, "diagnostics.UWB_Sensors_Receiver"))

    """Geofence (C4)"""
    link(Geofence_pub, rx(StatePlanner, "Geofence_Receiver"), rx(UI, "Geofence_Receiver"))

    """Accelerometer Theta Values"""
    link(Accelerometer_AccelPublisher_pub, rx(Drivetrain, "Drivetrain_AccelReceiver"),
         rx(UI, "diagnostics.Accelerometer_Receiver"))

    # ─────────────────────────────────────────────────────────────────────────
    # RUN
    # ─────────────────────────────────────────────────────────────────────────

    tasks = [UI.serve()]
    if Drivetrain:
        tasks.append(Drivetrain.DriveTrainDriver_Runner())
    if Canine:
        tasks += [Canine.PoseDriver.MotorPositions_Runner(), Canine.Audio._monitor()]
    if p["vision"] == "fake":
        tasks.append(Vision.VisionRunner())
    if p["sensors"]:
        tasks += [Accelerometer.Runner(), PositionDriver.Runner()]
    if Fence:
        tasks.append(Fence.Runner())

    print("[MAIN] CDS Control Server Starting")
    running = asyncio.gather(*tasks)
    try:
        if args.run_for is None:
            await running
            return
        await asyncio.wait({running}, timeout=args.run_for)
        if running.done():
            running.result()                      # a module crashed early: raise it
        # Timed stop. Cancelling uvicorn logs a CancelledError traceback that isn't a failure.
        logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
        print(f"[MAIN] --run-for {args.run_for:g} s reached, stopped.")
    finally:
        # Every exit (timed stop, Ctrl-C, a crash): release the camera and the Hailo
        # before Python exits, or HailoRT aborts the process (exit 134).
        stop = getattr(Vision, "stop", None)
        if stop is not None:
            stop()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down.")
