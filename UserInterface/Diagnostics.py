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

    async def UWB_Sensors_Receiver(self, data) -> None:
        """
        Raw UWB position.
        Expected: 1×2 array [UWB_X (m), UWB_Y (m)]
        """
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 2:
            self._robot.set_uwb(arr[0], arr[1])

    async def Accelerometer_Receiver(self, data) -> None:
        """
        Raw IMU heading.
        Expected: 1×1 array [theta (rad)]
        """
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 1:
            self._robot.set_imu(arr[0])

    async def MacroState_Receiver(self, data: dict) -> None:
        """
        High-level system state from state planner.

          Id="Scan"     → sys_state = "Scanning"
          Id="Chase"    → sys_state = "Chase"
          Id="Alert"    → sys_state = "Alert"
        """
        mapping = {
            "Scan":     "Scanning",
            "Chase":    "Chase",
            "Alert":    "Alert",
        }
        state = mapping.get(data.get("Id", ""))
        if state:
            self._robot.set_sys_state(state)

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