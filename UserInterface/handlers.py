"""
handlers.py — Incoming WebSocket message routing

Each message type from the UI is dispatched here.
Add new message types by adding an elif block.
"""

from fastapi import WebSocket

from logger import log_cmd, log_warn, log_err, log_evt
from robot  import RobotInterface


async def handle_message(ws: WebSocket, data: dict, robot: RobotInterface) -> None:
    """
    Route an incoming WebSocket message to the correct robot method.
    Called once per message received from the browser.
    """
    msg_type = data.get("type", "unknown")

    # ── Drivetrain ────────────────────────────────────────────
    if msg_type == "motor_command":
        cmd      = data.get("Cmd", [0.0, 0.0])
        speed_ms = float(data.get("speed_ms", robot.speed_limit))
        left     = float(cmd[0])
        right    = float(cmd[1])

        robot.apply_motor_command(left, right, speed_ms)
        log_cmd("MOTOR", f"L={left:+.2f}  R={right:+.2f}  speed_limit={speed_ms:.1f} m/s")

    # ── Speed limit ───────────────────────────────────────────
    elif msg_type == "speed_limit":
        speed = float(data.get("speed", 1.5))
        robot.apply_speed_limit(speed)
        log_cmd("SPEED", f"Limit set -> {speed:.1f} m/s")

    # ── Servo angle ───────────────────────────────────────────
    elif msg_type == "servo_angle":
        servo_id = data.get("servo", "?")
        angle    = float(data.get("angle", 0.0))
        robot.apply_servo_angle(servo_id, angle)
        log_cmd("SERVO", f"{servo_id} -> {angle:.1f} deg")

    # ── Pose preset ───────────────────────────────────────────
    elif msg_type == "pose_command":
        pose = data.get("pose", "?")
        robot.apply_pose(pose)
        log_evt("POSE", f"Executing -> {pose}")

    # ── Audio ─────────────────────────────────────────────────
    elif msg_type == "audio_play":
        sound = data.get("sound", "?")
        robot.apply_audio(sound)
        log_evt("AUDIO", f"Playing -> {sound}")

    # ── Mode switching ────────────────────────────────────────
    elif msg_type == "unlock":
        log_warn("MODE", "Switched to MANUAL — UI inputs active")

    elif msg_type == "lock_manual":
        robot.apply_motor_command(0.0, 0.0, 0.0)
        log_warn("MODE", "Switched to AUTO — motors zeroed, UI inputs locked")

    # ── Emergency stop ────────────────────────────────────────
    elif msg_type == "estop":
        robot.apply_estop()
        log_err("ESTOP", "EMERGENCY STOP ENGAGED — all motors halted")

    elif msg_type == "estop_reset":
        log_warn("ESTOP", "E-stop reset — system ready")

    # ── Unknown ───────────────────────────────────────────────
    else:
        log_warn("WS", f"Unknown message type: '{msg_type}'  full={data}")
