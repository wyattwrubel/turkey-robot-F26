"""
PanelControls.py — Manual Controls

Owns everything related to the manual controls panel:
  - Receiving Canine channel messages from the system (pose, servo, audio)
    and updating shared state
  - Handling all inbound UI messages from the browser (key drive, speed,
    servo, pose, audio, macrostate, estop)
  - All outbound publisher stubs — the only place that calls publish()

Inbound receiver (subscribe in main.py):
    canine_pub.subscribe(ui.controls.Canine_Receiver)

Outbound publishers are passed in at construction and called here:
    drivetrain_publisher   → motor commands, speed cap
    canine_publisher       → pose, servo, audio commands
    macrostate_publisher   → Chase / Scan / Alert commands
    navigation_publisher   → Move, Orient, UnlockUI commands

KEY_DIRECTIONS maps WASD keys to normalised [left, right] motor values.
Speed limit scaling is applied here before publishing to hardware.

Mode / estop invariants
───────────────────────
  estop_engage()     → _estop=True,  ui_locked=True   (motors zeroed)
  estop_release()    → _estop=False, ui_locked=True   (stays manual)
  toggle_mode        → if locked:  unlock only if _estop is clear
                       if unlocked: lock unconditionally
  unlock_navigation  → no-op + returns False if _estop still active

Resume sequence after estop:
  1. Operator presses RESET  → estop_reset  → _estop cleared, stays manual
  2. Operator presses AUTO   → toggle_mode  → unlocks only if _estop clear
"""

import json

from fastapi import WebSocket

from Robot import Robot

# Normalised direction vectors [left_motor, right_motor] for each drive key.
# Speed limit scaling is applied before publishing.
KEY_DIRECTIONS = {
    "w": ( 1.0,  1.0),   # forward
    "s": (-1.0, -1.0),   # reverse
    "a": (-0.2,  0.2),   # turn left
    "d": ( 0.2, -0.2),   # turn right
}


class PanelControls:

    def __init__(self, robot: Robot,
                 drivetrain_publisher,
                 canine_publisher,
                 macrostate_publisher,
                 navigation_publisher):
        self._robot          = robot
        self._drivetrain_pub = drivetrain_publisher
        self._canine_pub     = canine_publisher
        self._macrostate_pub = macrostate_publisher
        self._navigation_pub = navigation_publisher

    # ═══════════════════════════════════════════════════════════
    # INBOUND — system → UI state
    # ═══════════════════════════════════════════════════════════

    async def Canine_Receiver(self, data: dict) -> None:
        """
        Canine channel messages from the autonomous system.
          Type="Pose"   Data="Pose1".."Pose5"
          Type="Servo"  Name=servo_id  Data=angle (float)
          Type="Audio"  Data=audio_id
        """
        if data.get("Id") == "UnlockUI":
            return

        msg_type = data.get("Type", "")

        if msg_type == "Pose":
            self._robot.set_pose_cmd(data.get("Data", ""))
        elif msg_type == "Servo":
            self._robot.set_servo(data.get("Name", ""), float(data.get("Data", 0.0)))
        elif msg_type == "Audio":
            self._robot.set_audio(data.get("Data", ""))

    # ═══════════════════════════════════════════════════════════
    # INBOUND — browser → server
    # ═══════════════════════════════════════════════════════════

    async def handle_ui_message(self, data: dict) -> None:
        """
        Route a WebSocket message from the browser.
        All state is written to Robot first, then published to hardware.
        The 10 Hz broadcast loop reflects the result back to all clients.
        """
        msg_type = data.get("type", "")

        # ── Drive ────────────────────────────────────────────────
        
        if msg_type == "key_command":
            await self._handle_key_command(data)

        elif msg_type == "speed_limit":
            speed = float(data.get("speed", 1.5))
            self._robot.set_speed(speed)
            await self._pub_speed(speed)

        # ── Canine ───────────────────────────────────────────────
        elif msg_type == "servo_angle":
            servo_id = data.get("servo", "")
            angle    = float(data.get("angle", 0.0))
            self._robot.set_servo(servo_id, angle)
            await self._pub_servo(servo_id, angle)

        elif msg_type == "pose_command":
            pose = data.get("pose", "")
            self._robot.set_pose_cmd(pose)
            await self._pub_pose(pose)

        elif msg_type == "audio_play":
            sound = data.get("sound", "")
            self._robot.set_audio(sound)
            await self._pub_audio(sound)

        # ── Mission ──────────────────────────────────────────────
        elif msg_type == "macrostate_command":
            state = data.get("state", "")
            self._robot.set_sys_state(state)
            await self._macrostate_pub.publish({"Id": state})

        elif msg_type == "macrostate_sub_command":
            await self._macrostate_pub.publish({"Id": data.get("state", "")})

        elif msg_type == "waypoint_move":
            await self._handle_waypoint_move(data)

        # ── Safety ───────────────────────────────────────────────
        elif msg_type == "estop":
            self._robot.estop_engage()
            await self._pub_estop()

        elif msg_type == "estop_reset":
            self._robot.estop_release()
            # Stays in manual — operator must explicitly press AUTO to resume.

        elif msg_type == "toggle_mode":
            await self._handle_toggle_mode()

        elif msg_type == "heartbeat":
            self._robot.heartbeat()

    # ═══════════════════════════════════════════════════════════
    # PRIVATE HANDLERS
    # ═══════════════════════════════════════════════════════════

    async def _handle_key_command(self, data: dict) -> None:
        """
        Drive key press/release.
          press   → scale by speed_limit and publish
          release → zero motors
        """
        key   = data.get("key",   "")
        state = data.get("state", "")

        if state == "press" and key in KEY_DIRECTIONS:
            left, right = KEY_DIRECTIONS[key]
            left  *= self._robot.speed_limit
            right *= self._robot.speed_limit
            self._robot.set_motors(left, right)
            await self._pub_drivetrain("UI", left, right)

        elif state == "release":
            self._robot.set_motors(0.0, 0.0)
            await self._pub_drivetrain("UI", 0.0, 0.0)

    async def _handle_toggle_mode(self) -> None:
        """
        Switch between manual and auto.
        Switching TO auto is blocked if estop is still active —
        unlock_navigation() returns False in that case and nothing is published.
        """
        if self._robot.ui_locked:
            if not self._robot.unlock_navigation():
                print("[UI SAFETY] toggle_mode → AUTO blocked — estop still active")
                return
            await self._drivetrain_pub.publish({"Id": "UnlockUI"})
            await self._canine_pub.publish({"Id": "UnlockUI"})
            await self._macrostate_pub.publish({"Id": "UnlockUI"})
        else:
            self._robot.lock_navigation()
            await self._macrostate_pub.publish({"Id": "Stop"})

    async def _handle_waypoint_move(self, data: dict) -> None:
        """
        Send the robot to a map waypoint.
        Blocked unless in auto mode (ui_locked=False).
        """
        if self._robot.ui_locked:
            print("[WAYPOINT] Blocked — robot is in manual mode")
            return
        x     = float(data.get("x", 0.0))
        y     = float(data.get("y", 0.0))
        label = data.get("label", "?")
        self._robot.set_target(x, y, theta=0.0, nav_type="Move")
        await self._pub_navigation("Move", [x, y, 0.0])
        print(f"[WAYPOINT] Move → {label}  ({x:.2f}, {y:.2f})")

    # ═══════════════════════════════════════════════════════════
    # OUTBOUND PUBLISHERS
    # ═══════════════════════════════════════════════════════════

    async def _pub_drivetrain(self, source: str, left: float, right: float) -> None:
        await self._drivetrain_pub.publish({"Id": source, "Cmd": [left, right]})

    async def _pub_pose(self, pose_id: str) -> None:
        await self._canine_pub.publish({"Id": "UI", "Type": "Pose", "Data": pose_id})

    async def _pub_servo(self, servo_id: str, angle: float) -> None:
        await self._canine_pub.publish({"Id": "UI", "Type": "Servo", "Name": servo_id, "Data": angle})

    async def _pub_audio(self, audio_id: str) -> None:
        await self._canine_pub.publish({"Id": "UI", "Type": "Audio", "Data": audio_id})

    async def _pub_navigation(self, cmd_type: str, cmd: list) -> None:
        await self._navigation_pub.publish({"Id": cmd_type, "Cmd": cmd})

    async def _pub_speed(self, speed: float) -> None:
        await self._navigation_pub.publish({"Id": "SpeedCap", "Cmd": speed})

    # ═══════════════════════════════════════════════════════════
    # ON-CONNECT SYNC
    # ═══════════════════════════════════════════════════════════

    async def sync_client(self, ws: WebSocket) -> None:
        """Push current servo positions to a newly connected client."""
        await ws.send_text(json.dumps(self._robot.to_servo_msg()))

    async def _pub_estop(self) -> None:
        """Single point of truth for all estop downstream consequences."""
        await self._pub_drivetrain("UI", 0.0, 0.0)
        await self._navigation_pub.publish({"Id": "SpeedCap", "Cmd": 0})
        await self._macrostate_pub.publish({"Id": "Stop"})