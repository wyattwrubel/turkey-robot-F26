"""
robot.py — Shared robot state

All fields written by panel receivers. Read by the 10 Hz broadcast loop.

  x, y, theta           ← diagnostics.RobotPose_Receiver
  uwb_x, uwb_y          ← diagnostics.UWB_Sensors_Receiver
  imu_theta             ← diagnostics.Accelerometer_Receiver
  cmd_left/right        ← controls.handle_ui_message
  speed_limit           ← controls.handle_ui_message / zonemap.Navigation_Receiver
  sys_state             ← diagnostics.MacroState_Receiver
  target                ← zonemap.Navigation_Receiver
  servo_angles          ← controls.Canine_Receiver / handle_ui_message
  active_pose           ← controls.Canine_Receiver / handle_ui_message
  active_audio          ← controls.Canine_Receiver / handle_ui_message
  drivetrain_status     ← set externally by drivetrain subsystem
                          0 = nominal (green dot)
                          1 = warning (amber dot)
                          2 = error   (red dot)
"""

class Robot:

    def __init__(self):
        self.x          = 0.0
        self.y          = 0.0
        self.theta      = 0.0

        self.uwb_x      = 0.0
        self.uwb_y      = 0.0
        self.imu_theta  = 0.0

        self.cmd_left      = 0.0
        self.cmd_right     = 0.0
        self.speed_limit   = 1.5
        self._estop        = False

        self.sys_state    = "Scanning"
        self.sub_state    = "Idle"
        self.target       = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.nav_cmd_type = None   # "Move" | "Orient" | None
        self.ui_locked    = False  # True in manual mode — blocks Navigation_Receiver from overwriting target

        self.servo_angles = {
            "FrontLeft": 0.0, "FrontRight": 0.0,
            "RearLeft":  0.0, "RearRight":  0.0,
            "Head":      0.0,
        }
        self.active_pose  = None
        self.active_audio = None
        self._audio_timer = 0.0

        # 0 = nominal, 1 = warning, 2 = error
        self.drivetrain_status = 0

    def tick(self, dt: float = 0.1) -> None:
        """Advance audio highlight timer. Called at 10 Hz.
        Note: active_pose is now a persistent state machine — it does NOT time out.
        It remains set until explicitly changed via set_pose_cmd().
        """
        if self.active_audio is not None:
            self._audio_timer -= dt
            if self._audio_timer <= 0:
                self.active_audio = None

    # ── Sensors ───────────────────────────────────────────────
    def set_robot_pose(self, x, y, theta): self.x, self.y, self.theta = x, y, theta
    def set_uwb(self, x, y):               self.uwb_x, self.uwb_y = x, y
    def set_imu(self, theta):              self.imu_theta = theta

    # ── Drivetrain ────────────────────────────────────────────
    def set_motors(self, left, right):
        if not self._estop:
            self.cmd_left, self.cmd_right = left, right

    def set_speed(self, speed):            self.speed_limit = speed
    def estop_engage(self):                self._estop = True;  self.cmd_left = self.cmd_right = 0.0
    def estop_release(self):               self._estop = False

    def set_drivetrain_status(self, status: int) -> None:
        """
        Update drivetrain health indicator.
          0 — nominal  (green dot in UI)
          1 — warning  (amber dot in UI)
          2 — error    (red dot in UI)
        Hook this into your drivetrain subsystem / fault monitor.
        """
        assert status in (0, 1, 2), f"drivetrain_status must be 0, 1, or 2 — got {status}"
        self.drivetrain_status = status

    # ── Mission ───────────────────────────────────────────────
    def set_sys_state(self, state):        self.sys_state = state
    def lock_navigation(self):              self.ui_locked = True
    def unlock_navigation(self):            self.ui_locked = False
    def set_target(self, x, y, theta=0.0, nav_type=None):
        self.target       = {"x": round(x, 3), "y": round(y, 3), "theta": round(theta, 4)}
        self.nav_cmd_type = nav_type

    # ── Canine ────────────────────────────────────────────────
    def set_servo(self, servo_id, angle):  self.servo_angles[servo_id] = angle
    def set_pose_cmd(self, pose_id):
        """
        Transition to a new pose state.
        active_pose persists indefinitely — it is a state, not a timed flash.
        Pass None to clear (e.g. on estop or system reset).
        """
        self.active_pose = pose_id

    def set_audio(self, audio_id):         self.active_audio = audio_id; self._audio_timer = 2.0

    # ── Serialisers ───────────────────────────────────────────
    def to_state_msg(self) -> dict:
        return {
            "type":               "state",
            "uwb_x":              round(self.uwb_x,    3),
            "uwb_y":              round(self.uwb_y,    3),
            "imu_theta":          round(self.imu_theta, 4),
            "x":                  round(self.x,         3),
            "y":                  round(self.y,         3),
            "theta":              round(self.theta,      4),
            "macrostate":         self.sys_state,
            "nav_cmd_type":       self.nav_cmd_type,
            "target":             self.target,
            "cmd_left":           round(self.cmd_left,  3),
            "cmd_right":          round(self.cmd_right, 3),
            "speed_ms":           self.speed_limit,
            "active_pose":        self.active_pose,
            "active_audio":       self.active_audio,
            "drivetrain_status":  self.drivetrain_status,
            "servos": [{"id": k, "angle": v} for k, v in self.servo_angles.items()],
        }

    def to_servo_msg(self) -> dict:
        return {
            "type":   "servo_state",
            "servos": [{"id": k, "angle": v} for k, v in self.servo_angles.items()],
        }