"""
robot.py — Robot state interface

Variable → Protocol mapping
────────────────────────────────────────────────────────────────
  x, y, theta             ← RobotPose Channel (Ch. 4)
                             array [X, Y, theta]

  sigma_x/y/theta         ← UncertaintyPose Channel (new)
                             array [sigma_x, sigma_y, sigma_theta]

  uwb_x, uwb_y            ← UWB_Values Channel (Ch. 8.1)
                             array [UWB_X, UWB_Y]
  uwb_age                 ← derived: time since last UWB_Values message

  imu_theta               ← Acceler_Values Channel (Ch. 8.2)
                             array [theta]

  cmd_left, cmd_right     ← DriveTrain Channel (Ch. 3)
                             Cmd [left, right]
  drive_source            ← DriveTrain Channel Id field ("UI" / "System")

  speed_limit             ← Navigation SpeedCap (Ch. 6.3)
                             Cmd [speed_m/s]

  active_pose             ← Canine Channel, Type "Pose"  (Ch. 2.1)
  active_audio            ← Canine Channel, Type "Audio" (Ch. 2.3)
  servo_angles            ← Canine Channel, Type "Servo" (Ch. 2.2)

  destination             ← Navigation Channel (Chase/Scan/Alert only)
  sys_state               ← MacroState Channel: "Chase"|"Scanning"|"Alert"|None
  nav_cmd                 ← Navigation Channel (Move/Orient — target coords/state)
                             Move:   {"type":"Move",   "x":float, "y":float}
                             Orient: {"type":"Orient", "x":float, "y":float, "theta":float}
  wp_id, wp_dist          ← WaypointInfo Protocol (separate channel)
────────────────────────────────────────────────────────────────

State-change callbacks (register in app.py or main.py)
────────────────────────────────────────────────────────────────
  robot.register_motor_callback(cb)       fired on apply_motor_command
  robot.register_speed_callback(cb)       fired on apply_speed_limit
  robot.register_servo_callback(cb)       fired on apply_servo_angle
  robot.register_pose_callback(cb)        fired on apply_pose
  robot.register_audio_callback(cb)       fired on apply_audio
  robot.register_estop_callback(cb)       fired on apply_estop
  robot.register_mission_callback(cb)     fired on apply_mission_state
────────────────────────────────────────────────────────────────
"""

import asyncio
import time


# ── Callback helper ───────────────────────────────────────────────────────────

def _fire(callbacks: list, data: dict) -> None:
    """Schedule all callbacks as asyncio tasks (non-blocking, fire-and-forget)."""
    loop = asyncio.get_event_loop()
    for cb in callbacks:
        loop.create_task(cb(data))


# ─────────────────────────────────────────────────────────────────────────────
class RobotInterface:
    """
    Abstract interface the server expects.
    Subclass this (or use RobotState directly) when connecting real hardware.
    """

    def __init__(self):
        self._motor_callbacks   = []
        self._speed_callbacks   = []
        self._servo_callbacks   = []
        self._pose_callbacks    = []
        self._audio_callbacks   = []
        self._estop_callbacks   = []
        self._mission_callbacks = []

    def register_motor_callback(self, cb)   -> None: self._motor_callbacks.append(cb)
    def register_speed_callback(self, cb)   -> None: self._speed_callbacks.append(cb)
    def register_servo_callback(self, cb)   -> None: self._servo_callbacks.append(cb)
    def register_pose_callback(self, cb)    -> None: self._pose_callbacks.append(cb)
    def register_audio_callback(self, cb)   -> None: self._audio_callbacks.append(cb)
    def register_estop_callback(self, cb)   -> None: self._estop_callbacks.append(cb)
    def register_mission_callback(self, cb) -> None: self._mission_callbacks.append(cb)

    def tick(self, dt: float)                                    -> None: raise NotImplementedError
    def apply_motor_command(self, left, right, source)           -> None: raise NotImplementedError
    def apply_speed_limit(self, speed: float)                    -> None: raise NotImplementedError
    def apply_servo_angle(self, servo_id: str, angle: float)     -> None: raise NotImplementedError
    def apply_pose(self, pose_id: str)                           -> None: raise NotImplementedError
    def apply_audio(self, sound_id: str)                         -> None: raise NotImplementedError
    def apply_estop(self)                                        -> None: raise NotImplementedError
    def apply_robot_pose(self, x, y, theta)                      -> None: raise NotImplementedError
    def apply_uncertainty(self, sx, sy, stheta)                  -> None: raise NotImplementedError
    def apply_uwb(self, uwb_x, uwb_y)                           -> None: raise NotImplementedError
    def apply_imu(self, imu_theta)                               -> None: raise NotImplementedError
    def apply_mission_state(self, **kwargs)                      -> None: raise NotImplementedError
    def to_state_msg(self)                                       -> dict: raise NotImplementedError
    def to_servo_state_msg(self)                                 -> dict: raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class RobotState(RobotInterface):
    """
    Live robot state — all fields are written by their respective protocol
    receivers in app.py. tick() no longer generates fake data; it only
    advances the pose/audio timeout timers.
    """

    def __init__(self):
        super().__init__()

        # ── RobotPose Channel (Ch. 4) ──────────────────────────
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        # ── UncertaintyPose Channel (new) ──────────────────────
        self.sigma_x     = 0.0
        self.sigma_y     = 0.0
        self.sigma_theta = 0.0

        # ── UWB_Values Channel (Ch. 8.1) ──────────────────────
        self.uwb_x            = 0.0
        self.uwb_y            = 0.0
        self._uwb_last_update = None   # timestamp of last UWB message

        # ── Acceler_Values Channel (Ch. 8.2) ──────────────────
        self.imu_theta = 0.0

        # ── DriveTrain Channel (Ch. 3) ────────────────────────
        self.cmd_left     = 0.0
        self.cmd_right    = 0.0
        self.drive_source = None     # "UI" or "System" — None until first command
        self._estop_active = False   # when True, apply_motor_command is a no-op

        # ── Navigation SpeedCap (Ch. 6.3) ─────────────────────
        self.speed_limit = 1.5

        # ── Navigation / MacroState Channel ──────────────────
        self.destination   = None    # metres to current target (Chase/Scan/Alert only)
        self.sys_state     = None    # "Chase" | "Scanning" | "Alert" | None
        self.nav_cmd       = None    # Move/Orient target: dict or None
        # ── WaypointInfo Protocol (separate) ──────────────────
        self.wp_id         = "—"    # e.g. 'WP 3 / 7'
        self.wp_dist       = 0.0    # metres to waypoint (from WaypointInfo)

        # ── Canine Channel (Ch. 2) ────────────────────────────
        self.servo_angles = {
            "FrontLeft":  90.0,
            "FrontRight": 90.0,
            "RearLeft":   135.0,
            "RearRight":  0.0,
            "Head":       90.0,
        }
        self.active_pose  = None
        self.active_audio = None
        self._pose_timer  = 0.0
        self._audio_timer = 0.0

    # ── Tick — timers only, no simulation ────────────────────────────────────
    def tick(self, dt: float = 0.1) -> None:
        """
        Called at 10 Hz by the broadcast loop.
        Only manages pose/audio highlight timeouts — all state is now
        written directly by the protocol receivers in app.py.
        """
        if self.active_pose is not None:
            self._pose_timer -= dt
            if self._pose_timer <= 0:
                self.active_pose = None

        if self.active_audio is not None:
            self._audio_timer -= dt
            if self._audio_timer <= 0:
                self.active_audio = None

    # ── uwb_age property — derived from last message timestamp ───────────────
    @property
    def uwb_age(self) -> float:
        """Seconds since the last UWB_Values message. None if never received."""
        if self._uwb_last_update is None:
            return None
        return round(time.monotonic() - self._uwb_last_update, 2)

    # ── Protocol receivers (called by app.py subscriber methods) ─────────────

    def apply_robot_pose(self, x: float, y: float, theta: float) -> None:
        """RobotPose Channel (Ch. 4): array [X, Y, theta]"""
        self.x     = x
        self.y     = y
        self.theta = theta

    def apply_uncertainty(self, sx: float, sy: float, stheta: float) -> None:
        """UncertaintyPose Channel: array [sigma_x, sigma_y, sigma_theta]"""
        self.sigma_x     = sx
        self.sigma_y     = sy
        self.sigma_theta = stheta

    def apply_uwb(self, uwb_x: float, uwb_y: float) -> None:
        """UWB_Values Channel (Ch. 8.1): array [UWB_X, UWB_Y]"""
        self.uwb_x            = uwb_x
        self.uwb_y            = uwb_y
        self._uwb_last_update = time.monotonic()

    def apply_imu(self, imu_theta: float) -> None:
        """Acceler_Values Channel (Ch. 8.2): array [theta]"""
        self.imu_theta = imu_theta

    def apply_motor_command(self, left: float, right: float, source: str) -> None:
        """
        DriveTrain Channel (Ch. 3).
        source = Id field from the protocol message ("UI", "System").
        No-op while estop is active — call apply_estop_release() first.
        """
        if self._estop_active:
            return   # hold motors at zero until estop is released

        self.cmd_left     = left
        self.cmd_right    = right
        self.drive_source = source

        _fire(self._motor_callbacks, {
            "cmd_left":     self.cmd_left,
            "cmd_right":    self.cmd_right,
            "drive_source": self.drive_source,
        })

    def apply_speed_limit(self, speed: float) -> None:
        """Navigation SpeedCap (Ch. 6.3): Cmd [speed_m/s]"""
        self.speed_limit = speed

        _fire(self._speed_callbacks, {"speed_limit": self.speed_limit})

    def apply_mission_state(self, **kwargs) -> None:
        """
        MacroState / Navigation Channel — mission fields.
        Accepts any subset of:
          sys_state    — "Chase" | "Scanning" | "Alert" | None
                         (Move and Orient do NOT set sys_state)
          destination  — metres to current target.
                         Set by Chase (hypot(X,Y)), cleared to None by Scan/Alert.
                         Move and Orient never touch destination.
          nav_cmd      — dict describing the current Move/Orient target, or None.
                         Move:   {"type":"Move",   "x":float, "y":float}
                         Orient: {"type":"Orient", "x":float, "y":float, "theta":float}
        wp_id and wp_dist are written separately by apply_waypoint_info().
        """
        for field in ("destination", "sys_state", "nav_cmd"):
            if field in kwargs:
                setattr(self, field, kwargs[field])

        _fire(self._mission_callbacks, {
            "destination": self.destination,
            "sys_state":   self.sys_state,
            "nav_cmd":     self.nav_cmd,
            "wp_id":       self.wp_id,
            "wp_dist":     self.wp_dist,
        })

    def apply_waypoint_info(self, wp_id: str, wp_dist: float) -> None:
        """
        WaypointInfo Protocol — separate channel for waypoint progress.
        wp_id   — label e.g. 'WP 3 / 7'
        wp_dist — metres remaining to the waypoint
        """
        self.wp_id   = wp_id
        self.wp_dist = wp_dist

        _fire(self._mission_callbacks, {
            "destination": self.destination,
            "sys_state":   self.sys_state,
            "nav_cmd":     self.nav_cmd,
            "wp_id":       self.wp_id,
            "wp_dist":     self.wp_dist,
        })

    def apply_servo_angle(self, servo_id: str, angle: float) -> None:
        """Canine Channel, Type 'Servo' (Ch. 2.2)"""
        self.servo_angles[servo_id] = angle

        _fire(self._servo_callbacks, {
            "servo_id": servo_id,
            "angle":    angle,
            "servos":   dict(self.servo_angles),
        })

    def apply_pose(self, pose_id: str) -> None:
        """
        Canine Channel, Type 'Pose' (Ch. 2.1).
        pose_id canonical format: 'Pose1'..'Pose5'
        Sets active_pose and a 2s highlight timer — broadcast loop clears it after timeout.
        Called from both Canine_Receiver (system) and handlers.py (UI browser).
        """
        self.active_pose = pose_id
        self._pose_timer = 2.0

        _fire(self._pose_callbacks, {"active_pose": self.active_pose})

    def apply_audio(self, sound_id: str) -> None:
        """Canine Channel, Type 'Audio' (Ch. 2.3)"""
        self.active_audio = sound_id
        self._audio_timer = 2.0

        _fire(self._audio_callbacks, {"active_audio": self.active_audio})

    def apply_estop_engage(self) -> None:
        """
        Engage emergency stop.
        Zeros motors, sets _estop_active = True so apply_motor_command
        becomes a no-op until apply_estop_release() is called.
        """
        self._estop_active = True
        self.cmd_left      = 0.0
        self.cmd_right     = 0.0
        self.drive_source  = None    # motors are held; source is internal

        _fire(self._estop_callbacks, {
            "cmd_left":     self.cmd_left,
            "cmd_right":    self.cmd_right,
            "drive_source": self.drive_source,
        })

    def apply_estop_release(self) -> None:
        """
        Release emergency stop.
        Clears the hold so motor commands are accepted again.
        drive_source is reset to None until the next motor command arrives.
        """
        self._estop_active = False
        self.drive_source  = None

        _fire(self._estop_callbacks, {
            "cmd_left":     self.cmd_left,
            "cmd_right":    self.cmd_right,
            "drive_source": self.drive_source,
        })

    # ── Message serialisers ───────────────────────────────────────────────────

    def to_state_msg(self) -> dict:
        return {
            "type": "state",

            # ── Raw sensors ───────────────────────────────────
            "uwb_x":       round(self.uwb_x,    3),
            "uwb_y":       round(self.uwb_y,    3),
            "uwb_age":     self.uwb_age,            # None until first UWB msg
            "imu_theta":   round(self.imu_theta, 4),

            # ── UKF fused pose ────────────────────────────────
            "x":           round(self.x,     3),
            "y":           round(self.y,     3),
            "theta":       round(self.theta, 4),

            # ── UKF uncertainty (sigma_x/y used for confidence indicator) ──
            "sigma_x":     round(self.sigma_x, 4),
            "sigma_y":     round(self.sigma_y, 4),

            # ── Mission ───────────────────────────────────────
            "destination": round(self.destination, 2) if self.destination is not None else None,
            "sys_state":   self.sys_state,
            "nav_cmd":     self.nav_cmd,
            "wp_id":       self.wp_id,
            "wp_dist":     round(self.wp_dist, 1),

            # ── Drivetrain ────────────────────────────────────
            "cmd_left":     round(self.cmd_left,  3),
            "cmd_right":    round(self.cmd_right, 3),
            "drive_source": self.drive_source,
            "speed_ms":     self.speed_limit,

            # ── Canine ────────────────────────────────────────
            "active_pose":  self.active_pose,
            "active_audio": self.active_audio,
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


# ── Backwards-compatible alias ────────────────────────────────────────────────
# MockRobotState is kept so existing imports don't break while you migrate.
# Once app.py is updated to use RobotState directly, this can be removed.
MockRobotState = RobotState