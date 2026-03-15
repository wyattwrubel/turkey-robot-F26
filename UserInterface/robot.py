"""
robot.py — Robot state container

Holds all live state written by inbound receivers in app.py.
Provides apply_*() methods for clean writes and serialisers for broadcast.

State fields and their sources
───────────────────────────────────────────────────────────────
  x, y, theta       ← RobotPose_Receiver      [X, Y, theta (rad)]
  uwb_x, uwb_y      ← UWB_Sensors_Receiver    [UWB_X, UWB_Y (m)]
  imu_theta         ← Accelerometer_Receiver  [theta (rad)]
  cmd_left/right    ← UI motor commands via handlers
  speed_limit       ← UI speed commands via handlers
  sys_state         ← MacroState_Receiver     "Chase"|"Scanning"|"Alert"
  target            ← Navigation_Receiver     {x, y, theta} from Move/Orient
  servo_angles      ← Canine_Receiver / UI    per-servo float angles
  active_pose       ← Canine_Receiver / UI    "Pose1".."Pose5", clears after 2s
  active_audio      ← Canine_Receiver / UI    audio id, clears after 2s
───────────────────────────────────────────────────────────────
"""


class RobotState:

    def __init__(self):
        # ── Fused pose (RobotPose channel) ────────────────────
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        # ── Raw sensors ────────────────────────────────────────
        self.uwb_x     = 0.0
        self.uwb_y     = 0.0
        self.imu_theta = 0.0

        # ── Drivetrain ─────────────────────────────────────────
        self.cmd_left      = 0.0
        self.cmd_right     = 0.0
        self.speed_limit   = 1.5
        self._estop_active = False

        # ── Mission ────────────────────────────────────────────
        self.sys_state = "Scanning"
        self.target    = {"x": 0.0, "y": 0.0, "theta": 0.0}

        # ── Canine ─────────────────────────────────────────────
        self.servo_angles = {
            "FrontLeft":  0.0,
            "FrontRight": 0.0,
            "RearLeft":   0.0,
            "RearRight":  0.0,
            "Head":       0.0,
        }
        self.active_pose  = None
        self.active_audio = None
        self._pose_timer  = 0.0
        self._audio_timer = 0.0

    # ── Tick — called at 10 Hz by broadcast loop ──────────────
    def tick(self, dt: float = 0.1) -> None:
        """Advance pose/audio highlight timers. Clears them when expired."""
        if self.active_pose is not None:
            self._pose_timer -= dt
            if self._pose_timer <= 0:
                self.active_pose = None

        if self.active_audio is not None:
            self._audio_timer -= dt
            if self._audio_timer <= 0:
                self.active_audio = None

    # ── Sensor writes ──────────────────────────────────────────
    def apply_robot_pose(self, x: float, y: float, theta: float) -> None:
        self.x, self.y, self.theta = x, y, theta

    def apply_uwb(self, uwb_x: float, uwb_y: float) -> None:
        self.uwb_x, self.uwb_y = uwb_x, uwb_y

    def apply_imu(self, imu_theta: float) -> None:
        self.imu_theta = imu_theta

    # ── Drivetrain writes ──────────────────────────────────────
    def apply_motor_command(self, left: float, right: float) -> None:
        """No-op while estop is active."""
        if self._estop_active:
            return
        self.cmd_left  = left
        self.cmd_right = right

    def apply_speed_limit(self, speed: float) -> None:
        self.speed_limit = speed

    def apply_estop_engage(self) -> None:
        self._estop_active = True
        self.cmd_left      = 0.0
        self.cmd_right     = 0.0

    def apply_estop_release(self) -> None:
        self._estop_active = False

    # ── Mission writes ─────────────────────────────────────────
    def apply_sys_state(self, state: str) -> None:
        self.sys_state = state

    def apply_target(self, x: float, y: float, theta: float = 0.0) -> None:
        self.target = {"x": round(x, 3), "y": round(y, 3), "theta": round(theta, 4)}

    # ── Canine writes ──────────────────────────────────────────
    def apply_servo_angle(self, servo_id: str, angle: float) -> None:
        self.servo_angles[servo_id] = angle

    def apply_pose(self, pose_id: str) -> None:
        self.active_pose = pose_id
        self._pose_timer = 2.0

    def apply_audio(self, sound_id: str) -> None:
        self.active_audio = sound_id
        self._audio_timer = 2.0

    # ── Serialisers ────────────────────────────────────────────
    def to_state_msg(self) -> dict:
        """Full state broadcast sent to browser at 10 Hz."""
        return {
            "type":         "state",
            "uwb_x":        round(self.uwb_x,     3),
            "uwb_y":        round(self.uwb_y,     3),
            "imu_theta":    round(self.imu_theta,  4),
            "x":            round(self.x,          3),
            "y":            round(self.y,          3),
            "theta":        round(self.theta,       4),
            "macrostate":   self.sys_state,
            "target":       self.target,
            "cmd_left":     round(self.cmd_left,   3),
            "cmd_right":    round(self.cmd_right,  3),
            "speed_ms":     self.speed_limit,
            "active_pose":  self.active_pose,
            "active_audio": self.active_audio,
            "servos": [{"id": k, "angle": v} for k, v in self.servo_angles.items()],
        }

    def to_servo_state_msg(self) -> dict:
        """Servo positions only — sent once on client connect."""
        return {
            "type":   "servo_state",
            "servos": [{"id": k, "angle": v} for k, v in self.servo_angles.items()],
        }