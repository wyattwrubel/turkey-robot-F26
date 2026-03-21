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
        self._robot           = robot
        self._drivetrain_pub  = drivetrain_publisher
        self._canine_pub      = canine_publisher
        self._macrostate_pub  = macrostate_publisher
        self._navigation_pub  = navigation_publisher

    # ── Inbound receiver (system → UI state) ─────────────────────────────────

    async def Canine_Receiver(self, data: dict) -> None:
        """
        Canine channel messages from the autonomous system.

          Type="Pose"   Data="Pose1".."Pose5"
          Type="Servo"  Name=servo_id  Data=angle (float)
          Type="Audio"  Data=audio_id
          Id="UnlockUI" → ignored
        """
        if data.get("Id") == "UnlockUI":
            return

        msg_type = data.get("Type", "")

        if msg_type == "Pose":
            self._robot.set_pose_cmd(data.get("Data", ""))

        elif msg_type == "Servo":
            self._robot.set_servo(
                data.get("Name", ""),
                float(data.get("Data", 0.0)),
            )

        elif msg_type == "Audio":
            self._robot.set_audio(data.get("Data", ""))

    # ── Inbound UI message handler (browser → server) ────────────────────────

    async def handle_ui_message(self, data: dict) -> None:
        """
        Route a WebSocket message from the browser.
        Updates shared robot state then publishes to the appropriate channel.
        """
        msg_type = data.get("type", "")

        if msg_type == "key_command":
            await self._handle_key_command(data)

        elif msg_type == "speed_limit":
            speed = float(data.get("speed", 1.5))
            self._robot.set_speed(speed)
            await self._pub_speed(speed)

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

        elif msg_type == "macrostate_command":
            state = data.get("state", "")
            self._robot.set_sys_state(state)
            await self._macrostate_pub.publish({"Id": state})

        elif msg_type == "macrostate_sub_command":
            state = data.get("state", "")
            await self._macrostate_pub.publish({"Id": state})

        elif msg_type == "estop":
            self._robot.estop_engage()
            await self._pub_drivetrain("UI", 0.0, 0.0)

        elif msg_type == "estop_reset":
            self._robot.estop_release()

        elif msg_type == "waypoint_move":
            if not self._robot.ui_locked:
                print("[WAYPOINT] Blocked — not in manual mode")
                return
            x     = float(data.get("x", 0.0))
            y     = float(data.get("y", 0.0))
            label = data.get("label", "?")
            self._robot.set_target(x, y, theta=0.0, nav_type="Move")
            await self._pub_navigation("Move", [x, y, 0.0])
            print(f"[WAYPOINT] Move → {label}  ({x}, {y})")

        elif msg_type == "lock_ui":
            # Operator switched to manual mode — lock navigation target
            self._robot.lock_navigation()

        elif msg_type == "unlock_ui":
            self._robot.unlock_navigation()
            await self._drivetrain_pub.publish({"Id": "UnlockUI"})
            await self._canine_pub.publish({"Id": "UnlockUI"})
            await self._macrostate_pub.publish({"Id": "UnlockUI"})
            await self._navigation_pub.publish({"Id": "UnlockUI"})

    async def _handle_key_command(self, data: dict) -> None:
        """
        Drive key press/release from the browser.

          state="press"   key in w/a/s/d  → scale by speed_limit, publish motors
          state="release" any key or "all" → zero motors
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
            # Handles both single key release and "all" bulk release on mode switch
            self._robot.set_motors(0.0, 0.0)
            await self._pub_drivetrain("UI", 0.0, 0.0)

    # ── Outbound publisher stubs ──────────────────────────────────────────────
    # Add any scaling, clamping, or logging here before commands reach hardware.

    async def _pub_drivetrain(self, source: str, left: float, right: float) -> None:
        """Drivetrain channel — motor command."""
        await self._drivetrain_pub.publish({
            "Id":  source,
            "Cmd": [left, right],
        })

    async def _pub_pose(self, pose_id: str) -> None:
        """Canine channel — pose command."""
        await self._canine_pub.publish({
            "Id":   "UI",
            "Type": "Pose",
            "Data": pose_id,
        })

    async def _pub_servo(self, servo_id: str, angle: float) -> None:
        """Canine channel — raw servo command."""
        await self._canine_pub.publish({
            "Id":   "UI",
            "Type": "Servo",
            "Name": servo_id,
            "Data": angle,
        })

    async def _pub_audio(self, audio_id: str) -> None:
        """Canine channel — audio command."""
        await self._canine_pub.publish({
            "Id":   "UI",
            "Type": "Audio",
            "Data": audio_id,
        })

    async def _pub_navigation(self, cmd_type: str, cmd: list) -> None:
        """Navigation channel — Move or Orient command."""
        await self._navigation_pub.publish({
            "Id":  cmd_type,
            "Cmd": cmd,
        })

    async def _pub_speed(self, speed: float) -> None:
        """Drivetrain channel — speed cap."""
        await self._navigation_pub.publish({
            "Id":  "SpeedCap",
            "Cmd": speed,
        })


    # ── On-connect sync ───────────────────────────────────────────────────────

    async def sync_client(self, ws: WebSocket) -> None:
        """Send current servo positions to a newly connected client."""
        await ws.send_text(json.dumps(self._robot.to_servo_msg()))