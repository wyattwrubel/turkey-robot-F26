"""
robot.py — Robot state interface

This is YOUR integration point. Every method marked TODO is
where you wire in your real MotorDriver, UKF, servo driver, etc.

When you're ready to connect your real code:

    1. Import your MotorDriver at the top of this file
    2. Replace each TODO method body with the real call
    3. Update to_state_msg() to pull live values from UKF

Everything else in the server (handlers, broadcast, app)
calls only the methods defined here — nothing else needs
to change when you swap mock → real.
"""

import math
import random
import time


class RobotInterface:
    """
    Abstract interface the server expects.
    Your real implementation should provide all these methods.
    MockRobotState below is a drop-in stand-in for testing.
    """

    def tick(self, dt: float) -> None:
        """Called at 10 Hz by the broadcast loop. Update internal state."""
        raise NotImplementedError

    def apply_motor_command(self, left: float, right: float, speed_ms: float) -> None:
        """Receive a normalised [-1, 1] motor command with a speed cap in m/s."""
        raise NotImplementedError

    def apply_speed_limit(self, speed: float) -> None:
        """Set the global speed limit in m/s."""
        raise NotImplementedError

    def apply_servo_angle(self, servo_id: str, angle: float) -> None:
        """Command a single servo to an absolute angle in degrees."""
        raise NotImplementedError

    def apply_pose(self, pose_id: str) -> None:
        """Execute a named pose preset (e.g. 'POSE_1')."""
        raise NotImplementedError

    def apply_audio(self, sound_id: str) -> None:
        """Play a named audio clip (e.g. 'BARK')."""
        raise NotImplementedError

    def apply_estop(self) -> None:
        """Immediately halt all motors."""
        raise NotImplementedError

    def to_state_msg(self) -> dict:
        """Return a dict ready to JSON-serialise and push to the UI."""
        raise NotImplementedError

    def to_servo_state_msg(self) -> dict:
        """Return current servo angles for UI sync on connect."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
class MockRobotState(RobotInterface):
    """
    Simulated robot — used for UI development and testing.
    Drives a slow circle so the map panel animates.

    Replace with your real implementation when hardware is ready.
    """

    def __init__(self):
        # ── UKF pose ──────────────────────────────────────────
        self.x     = 2.0
        self.y     = 2.0
        self.theta = 0.0

        # ── UKF covariance ────────────────────────────────────
        self.sigma_x     = 0.08
        self.sigma_y     = 0.07
        self.sigma_theta = 0.021

        # ── Raw sensor readings ───────────────────────────────
        self.uwb_x   = 2.0
        self.uwb_y   = 2.0
        self.uwb_age = 0.3

        self.imu_theta = 0.0
        self.cal_sys   = 3
        self.cal_gyro  = 3
        self.cal_mag   = 3

        # ── Mission state ─────────────────────────────────────
        self.dist_target   = None
        self.target_status = "no target acquired"
        self.wp_id         = "WP 1 / 7"
        self.wp_dist       = 12.4
        self.wp_bearing    = 45.0

        # ── Motor commands ────────────────────────────────────
        self.cmd_left    = 0.0
        self.cmd_right   = 0.0
        self.drive_source = "IDLE"
        self.speed_limit  = 1.5

        # ── Deterrent ─────────────────────────────────────────
        self.det_status = "IDLE"
        self.det_count  = 0
        self.det_last   = "\u2014"

        # ── Servos ────────────────────────────────────────────
        self.servo_angles = {
            "SERVO_1": 90.0,
            "SERVO_2": 90.0,
            "SERVO_3": 135.0,
            "SERVO_4": 0.0,
            "SERVO_5": 90.0,
        }

        self._t = 0.0

        # ── Active pose / audio (for UI feedback) ─────────────
        self.active_pose  = None   # e.g. "POSE_1", cleared after timeout
        self.active_audio = None   # e.g. "BARK", cleared after timeout
        self._pose_timer  = 0.0
        self._audio_timer = 0.0

    # ── Simulation tick ───────────────────────────────────────
    def tick(self, dt: float = 0.1) -> None:
        """Simulates a slow circle. Remove when using real hardware."""
        self._t += dt

        self.x     = 20.0 + 5.0 * math.cos(self._t * 0.1)
        self.y     = 10.0 + 5.0 * math.sin(self._t * 0.1)
        self.theta = (self._t * 0.1) % (2 * math.pi)

        self.uwb_x     = self.x + random.gauss(0, 0.05)
        self.uwb_y     = self.y + random.gauss(0, 0.05)
        self.uwb_age   = random.uniform(0.1, 0.6)
        self.imu_theta = self.theta + random.gauss(0, 0.005)

        self.sigma_x     = 0.06 + 0.02 * abs(math.sin(self._t * 0.2))
        self.sigma_y     = 0.06 + 0.02 * abs(math.cos(self._t * 0.2))
        self.sigma_theta = 0.018 + 0.005 * abs(math.sin(self._t))

        self.dist_target   = 8.4 + 2.0 * math.sin(self._t * 0.15)
        self.target_status = "acquired" if self.dist_target < 10 else "searching"

        # decay pose/audio active flags after 2s
        if self.active_pose is not None:
            self._pose_timer -= dt
            if self._pose_timer <= 0:
                self.active_pose = None
        if self.active_audio is not None:
            self._audio_timer -= dt
            if self._audio_timer <= 0:
                self.active_audio = None

    # ── TODO: wire these into your real drivers ───────────────

    def apply_motor_command(self, left: float, right: float, speed_ms: float) -> None:
        """
        TODO: replace with real call, e.g.
            await self.motor_driver.MotorCommandReceiver({
                "Id": "UI", "Cmd": [left, right]
            })
        """
        self.cmd_left    = left
        self.cmd_right   = right
        self.drive_source = "UI"

    def apply_speed_limit(self, speed: float) -> None:
        """
        TODO: forward to motor driver speed cap if applicable.
        """
        self.speed_limit = speed

    def apply_servo_angle(self, servo_id: str, angle: float) -> None:
        """
        TODO: replace with real servo driver call, e.g.
            self.servo_driver.set_angle(servo_id, angle)
        """
        self.servo_angles[servo_id] = angle

    def apply_pose(self, pose_id: str) -> None:
        """
        TODO: replace with real pose executor, e.g.
            self.pose_manager.execute(pose_id)
        Sets active_pose so the UI highlights the active button for 2s.
        """
        self.active_pose  = pose_id
        self._pose_timer  = 2.0

    def apply_audio(self, sound_id: str) -> None:
        """
        TODO: replace with real audio driver call, e.g.
            self.audio_driver.play(sound_id)
        Sets active_audio so the UI highlights the active button for 2s.
        """
        self.active_audio  = sound_id
        self._audio_timer  = 2.0

    def apply_estop(self) -> None:
        """
        TODO: also send estop signal to hardware watchdog if you have one.
        """
        self.cmd_left    = 0.0
        self.cmd_right   = 0.0
        self.drive_source = "ESTOP"

    # ── Message serialisers ───────────────────────────────────

    def to_state_msg(self) -> dict:
        """
        Builds the JSON blob pushed to the UI at 10 Hz.
        TODO: replace attribute reads with live UKF / sensor values, e.g.
            "x": self.ukf.state[0],
            "y": self.ukf.state[1],
        """
        return {
            "type":          "state",
            # raw sensors
            "uwb_x":         round(self.uwb_x,   3),
            "uwb_y":         round(self.uwb_y,   3),
            "uwb_age":       round(self.uwb_age,  2),
            "imu_theta":     round(self.imu_theta, 4),
            "cal_sys":       self.cal_sys,
            "cal_gyro":      self.cal_gyro,
            "cal_mag":       self.cal_mag,
            # UKF fused pose
            "x":             round(self.x,     3),
            "y":             round(self.y,     3),
            "theta":         round(self.theta, 4),
            # UKF uncertainty
            "sigma_x":       round(self.sigma_x,     4),
            "sigma_y":       round(self.sigma_y,     4),
            "sigma_theta":   round(self.sigma_theta, 4),
            # mission
            "dist_target":   round(self.dist_target, 2) if self.dist_target is not None else None,
            "target_status": self.target_status,
            "wp_id":         self.wp_id,
            "wp_dist":       round(self.wp_dist,    1),
            "wp_bearing":    round(self.wp_bearing, 1),
            # commands
            "cmd_left":      round(self.cmd_left,  3),
            "cmd_right":     round(self.cmd_right, 3),
            "drive_source":  self.drive_source,
            "speed_ms":      self.speed_limit,
            # deterrent
            "det_status":    self.det_status,
            "det_count":     self.det_count,
            "det_last":      self.det_last,
            # pose / audio active state
            "active_pose":   self.active_pose,
            "active_audio":  self.active_audio,
            # servos
            "servos": [
                {"id": k, "angle": v}
                for k, v in self.servo_angles.items()
            ],
        }

    def to_servo_state_msg(self) -> dict:
        return {
            "type":   "servo_state",
            "servos": [
                {"id": k, "angle": v}
                for k, v in self.servo_angles.items()
            ],
        }