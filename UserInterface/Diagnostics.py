"""
PanelDiagnostics.py — System Diagnostics

Owns everything related to the diagnostics panel:
  - Receiving sensor data (UWB, IMU, fused pose) and writing it to shared state
  - Receiving MacroState channel messages and updating sys_state
  - Receiving VisionObject channel messages and broadcasting them to clients

Inbound receivers (subscribe in main.py):
    pose_pub.subscribe(ui.diagnostics.RobotPose_Receiver)
    uwb_pub.subscribe(ui.diagnostics.UWB_Sensors_Receiver)
    imu_pub.subscribe(ui.diagnostics.Accelerometer_Receiver)
    macrostate_pub.subscribe(ui.diagnostics.MacroState_Receiver)
    vision_pub.subscribe(ui.diagnostics.VisionObject_Receiver)

No outbound publishing — diagnostics is display only.
"""

import numpy as np

from Robot import Robot


class PanelDiagnostics:

    def __init__(self, robot: Robot):
        self._robot = robot

    # ── Inbound receivers ─────────────────────────────────────────────────────

    async def RobotPose_Receiver(self, data) -> None:
        """
        Fused pose estimate.
        Expected: 1×3 array [X (m), Y (m), theta (rad)]
        """
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 3:
            self._robot.set_robot_pose(arr[0], arr[1], arr[2])
        else:
            print(f"[UI RobotPose_Receiver] Invalid Data '{data}'")   

    async def CanineAngle_Receiver(self, data) -> None:
        """
        Reported servo angles from canine driver
        Name: "FrontLeft", "FrontRight", "RearLeft", "RearRight", "Head"
        Angle: float (theta)
        """
        Name = data.get("Name", "")
        Angle = data.get("Angle", "")

        if Name is not None and Angle is not None: 
            self._robot.set_servo(Name, Angle)
        else:
            print(f"[UI CanineAngle_Receiver] Invalid Data '{data}'")   

    async def CaninePose_Receiver(self, data) -> None: 
        """
        Recceives poses from state planner: 
        Id: "System" 
        Type: "Pose" 
        Data: IdlePose, AlertPose, AggressivePose, WalkGait, ExcitedGait, AlertStep
        """
        
        LegalPoses = {"Stop", "IdlePose", "AlertPose", "AggressivePose", "WalkGait", "ExcitedGait", "AlertStep"}
        Pose = data.get("Data", "")
        if Pose in LegalPoses: 
            self._robot.set_pose_cmd(Pose)
            
    async def UWB_Sensors_Receiver(self, data) -> None:
        """
        Raw UWB position.
        Expected: 1×2 array [UWB_X (m), UWB_Y (m)]
        """
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 2:
            self._robot.set_uwb(arr[0], arr[1])
        else:
            print(f"[UI UWB_Sensors_Receiver] Invalid Data '{data}'")   

    async def Accelerometer_Receiver(self, data) -> None:
        """
        Raw IMU heading.
        Expected: 1×1 array [theta (rad)]
        """
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 1:
            self._robot.set_imu(arr[0])
        else:
            print(f"[UI Accelerometer_Receiver] Invalid Data '{data}'")    

    async def MacroState_Receiver(self, data: dict) -> None:
        """
        High-level system state from state planner.

          Id="Scan"     → sys_state = "Scanning"
          Id="Chase"    → sys_state = "Chase"
          Id="Alert"    → sys_state = "Alert"
          Id="UnlockUI" → sys_state = "Scanning"  (auto mode restored)
        """
        mainstates = {
            "Scan":     "Scanning",
            "Chase":    "Chase",
            "Alert":    "Alert",
            "UnlockUI": "Scanning",
        }

        allowed_substates = {
            "TravelWaypoint", "StationaryScan", "StationaryCrouch", 
            "AggressivePursuit",  "BurstAttack", "GrowlAndChase", 
            "WatchfulDog", "TerritorialWarning", "ShortWarningBurst",
        }

        state = mainstates.get(data.get("Id", ""))
        substate = data.get("Id", "")
        if state:
            self._robot.set_sys_state(state)
        elif (substate in allowed_substates):
            self._robot.set_sub_state(substate)
        else:
            print(f"[UI MacroState_Receiver] Invalid Data '{data}'")    

    async def VisionObject_Receiver(self, data: dict, broadcast_fn) -> None:
        """
        Detected object positions — broadcast directly to clients for map display.

          Turkeys: T×3 array [x, y, angle_rad]
          Humans:  P×3 array [x, y, angle_rad]

        broadcast_fn — pass CDSControlServer._broadcast
        """
        turkeys = np.asarray(data.get("Turkeys", []), dtype=float)
        humans  = np.asarray(data.get("Humans",  []), dtype=float)
        await broadcast_fn({
            "type":    "vision",
            "turkeys": turkeys.tolist() if turkeys.size > 0 else [],
            "humans":  humans.tolist()  if humans.size  > 0 else [],
        })

    async def Slip_Receiver(self, data: dict) -> None: 
        """
        Slipping Reporting From Drivetrain: 
        0: No slip
        1: Warning of Slip
        2: Heavy slip, requires manual inspection
        """
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 1:
            self._robot.set_drivetrain_status(arr[0])
        else:
            print(f"[UI Slip_Receiver] Invalid Data '{data}'")

    async def MotionCommand_Receiver(self, data: dict) -> None:
        """
        Id: "Orient", "Move"
        Cmd: 1 x 3 numpy: [X, Y, theta]

        Id: "SpeedCap"
        Cmd: 1 x 1 numpy: [motor speed (m/s)]
        """
        print("UI MOTIONCOMMAND ", data)

        id  = data.get("Id",  "")
        cmd = data.get("Cmd", "")

        if id == "Move":
            target = np.asarray(cmd, dtype=float).flatten()
            self._robot.set_target(target[0], target[1], target[2], nav_type="Move")
        elif id == "Orient":
            target = np.asarray(cmd, dtype=float).flatten()
            self._robot.set_target(target[0], target[1], target[2], nav_type="Orient")
        elif id == "SpeedCap":
            speed = float(np.asarray(cmd, dtype=float).flatten()[0])

            self._robot.set_speed(speed)
        else:
            print(f"[UI MotionCommand_Receiver] Unknown Id '{id}'")