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
  reference_trajectory  ← diagnostics.TrajectoryTracker_Receiver (6.1)
  mpc_horizon           ← diagnostics.TrajectoryTracker_Receiver (6.2)

Mode / estop invariants
───────────────────────
  estop_engage()       → _estop=True,  ui_locked=True  (motors zeroed)
  estop_release()      → _estop=False, ui_locked=True  (stays manual)
  unlock_navigation()  → sets ui_locked=False only if _estop is clear
                         returns True on success, False if blocked
  lock_navigation()    → always succeeds, ui_locked=True
"""

import numpy as np
import time

class Robot:

    def __init__(self):
        self.x         = 0.0
        self.y         = 0.0
        self.theta     = 0.0

        self.uwb_x     = 0.0
        self.uwb_y     = 0.0
        self.imu_theta = 0.0

        self.cmd_left    = 0.0
        self.cmd_right   = 0.0
        self.speed_limit = 5
        self._estop      = False

        self.sys_state    = "Stop"
        self.sub_state    = "Idle"
        self.target       = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.nav_cmd_type = None      # "Move" | "Orient" | None
        self.ui_locked    = True      # True = manual, False = auto

        self.servo_angles = {
            "FrontLeft": 0.0, "FrontRight": 0.0,
            "RearLeft":  0.0, "RearRight":  0.0,
            "Head":      0.0,
        }
        self.active_pose  = None
        self.active_audio = None
        self._audio_timer = 0.0

        self.drivetrain_status = 0    # 0=nominal, 1=warning, 2=error (nothing sets this; see sources)

        # Header status dots for the data sources. mode comes from Main.py's profile and flags:
        # "real" = hardware expected, "sim" = simulated, "off" = not running in this profile.
        # last_seen is when that source's data last reached the UI; a real source with no
        # data for SOURCE_STALE_S shows as an error on the page.
        self.sources   = {"gps": "off", "imu": "off", "drivetrain": "off"}
        self.last_seen = {name: None for name in self.sources}

        self.reference_trajectory = []   # full reference path   — N×2 [x, y]
        self.mpc_horizon          = []   # live lookahead horizon — N×2 [x, y]

        self._last_heartbeat = time.monotonic()
        self.HEARTBEAT_TIMEOUT = 1.0   # seconds
        self._auto_ever_engaged = False

    # ── Sensors ───────────────────────────────────────────────
    def set_robot_pose(self, x, y, theta): self.x, self.y, self.theta = x, y, theta
    def set_uwb(self, x, y):               self.uwb_x, self.uwb_y = x, y
    def set_imu(self, theta):              self.imu_theta = theta

    # ── Drivetrain ────────────────────────────────────────────
    def set_motors(self, left, right):
        if self._estop and (left != 0.0 or right != 0.0):
            return  # block drive commands only, never block zeroing
        self.cmd_left, self.cmd_right = left, right

    def set_speed(self, speed):
        self.speed_limit = speed

    def set_drivetrain_status(self, status: int) -> None:
        assert status in (0, 1, 2), f"drivetrain_status must be 0, 1, or 2 — got {status}"
        self.drivetrain_status = status

    # ── Safety ────────────────────────────────────────────────
    def estop_engage(self):
        self._estop    = True
        self.ui_locked = True
        self.cmd_left  = 0.0
        self.cmd_right = 0.0

    def estop_release(self):
        """Clear estop only. Robot stays in manual until operator explicitly switches to auto."""
        self._estop = False

    # ── Mode ──────────────────────────────────────────────────
    def lock_navigation(self):
        self.ui_locked = True

    def unlock_navigation(self) -> bool:
        if self._estop:
            print("[UI SAFETY] unlock_navigation blocked — estop still active")
            return False
        self.ui_locked          = False
        self._auto_ever_engaged = True   # ← arm the watchdog on first auto
        return True

    # ── Mission ───────────────────────────────────────────────
    def set_sys_state(self, state):  self.sys_state = state
    def set_sub_state(self, state):  self.sub_state = state

    def set_target(self, x, y, theta=0.0, nav_type=None):
        self.target       = {"x": round(x, 3), "y": round(y, 3), "theta": round(theta, 4)}
        self.nav_cmd_type = nav_type

    # ── Trajectory Tracking (Protocol 6B) ────────────────────
    def set_reference_trajectory(self, arr: np.ndarray) -> None:
        """6.1 — Full reference path. arr is N×2 [x, y]."""
        self.reference_trajectory = arr.tolist() if arr.size > 0 else []

    def set_mpc_horizon(self, arr: np.ndarray) -> None:
        """6.2 — Live lookahead horizon. arr is N×2 [x, y]."""
        self.mpc_horizon = arr.tolist() if arr.size > 0 else []

    def clear_trajectories(self) -> None:
        """6.3 — Wipe both trajectory fields."""
        self.reference_trajectory = []
        self.mpc_horizon          = []

    # ── Canine ────────────────────────────────────────────────
    def set_servo(self, servo_id, angle):
        self.servo_angles[servo_id] = angle

    def set_pose_cmd(self, pose_id):
        """Persistent pose state — not a timed flash. Pass None to clear."""
        self.active_pose = pose_id

    def set_audio(self, audio_id):
        self.active_audio = audio_id
        self._audio_timer = 2.0

    def heartbeat(self):
        self._last_heartbeat = time.monotonic()

    def check_heartbeat(self) -> bool:
        """Only active after auto mode has been engaged at least once."""
        if not self._auto_ever_engaged:
            return True
        return (time.monotonic() - self._last_heartbeat) < self.HEARTBEAT_TIMEOUT

    # ── Data sources (header dots) ────────────────────────────
    SOURCE_STALE_S = 2.0

    def set_sources(self, sources: dict) -> None:
        for name, mode in sources.items():
            if name in self.sources and mode in ("real", "sim", "off"):
                self.sources[name] = mode

    def mark_seen(self, name: str) -> None:
        if name in self.last_seen:
            self.last_seen[name] = time.monotonic()

    def sources_msg(self) -> dict:
        now = time.monotonic()
        return {name: {"mode": mode,
                       "age": None if self.last_seen[name] is None
                              else round(now - self.last_seen[name], 1)}
                for name, mode in self.sources.items()}

    # ── Serialisers ───────────────────────────────────────────
    def to_state_msg(self) -> dict:
        return {
            "type":                 "state",
            "uwb_x":                round(self.uwb_x,    3),
            "uwb_y":                round(self.uwb_y,    3),
            "imu_theta":            round(self.imu_theta, 4),
            "x":                    round(self.x,         3),
            "y":                    round(self.y,         3),
            "theta":                round(self.theta,     4),
            "macrostate":           self.sys_state,
            "substate":             self.sub_state,
            "nav_cmd_type":         self.nav_cmd_type,
            "target":               self.target,
            "cmd_left":             round(self.cmd_left,  3),
            "cmd_right":            round(self.cmd_right, 3),
            "speed_ms":             self.speed_limit,
            "active_pose":          self.active_pose,
            "active_audio":         self.active_audio,
            "drivetrain_status":    self.drivetrain_status,
            "reference_trajectory": self.reference_trajectory,
            "mpc_horizon":          self.mpc_horizon,
            "servos":               [{"id": k, "angle": v} for k, v in self.servo_angles.items()],
            "is_manual":            self.ui_locked or self._estop,
            "is_estopped": self._estop,
            "sources":              self.sources_msg(),
        }

    def to_servo_msg(self) -> dict:
        return {
            "type":   "servo_state",
            "servos": [{"id": k, "angle": v} for k, v in self.servo_angles.items()],
        }