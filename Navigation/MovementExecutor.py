import asyncio
import numpy as np


class MPC_loop:
    """
    PD controller replacing the MPC stub.

    Move mode:   follows a list of (x, y) waypoints from PathGenerator
    Orient mode: rotates in place to a target theta

    Outputs differential drive commands {"Id": "System", "Cmd": [left, right]}
    via _drive_callback, set by NavigationDriver after instantiation.
    """

    # PD gains
    KP_LINEAR   = 0.8
    KP_ANGULAR  = 1.2
    KD_ANGULAR  = 0.1

    MAX_SPEED   = 1.0   # m/s per wheel — overridden by SpeedCap

    ARRIVAL_THRESHOLD_M   = 0.5   # metres — waypoint considered reached
    ARRIVAL_THRESHOLD_RAD = 0.05  # radians — orientation considered aligned

    def __init__(self):
        self.Tracked_Trajectory = []          # list of (x, y) grid cells
        self.Desired_State      = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.Current_State      = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.Verbose            = False

        self._mode              = "idle"      # "move", "orient", "idle"
        self._waypoint_idx      = 0
        self._prev_heading_err  = 0.0
        self._drive_callback    = None        # async callable set by NavigationDriver
        self._speed_cap         = self.MAX_SPEED

    # ------------------------------------------------------------------
    # Command Receivers
    # ------------------------------------------------------------------

    def Orient_CMD_Receiver(self, data):
        self.Desired_State      = np.asarray(data, dtype=np.float64).flatten()
        self.Tracked_Trajectory = []
        self._waypoint_idx      = 0
        self._prev_heading_err  = 0.0
        self._mode              = "orient"
        if self.Verbose:
            print(f"[PD] Orient -> theta={np.rad2deg(float(self.Desired_State[2])):.1f} deg")

    def Move_CMD_Receiver(self, data):
        self.Tracked_Trajectory = data if data is not None else []
        self.Desired_State      = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._waypoint_idx      = 0
        self._prev_heading_err  = 0.0
        self._mode              = "move" if self.Tracked_Trajectory else "idle"
        if self.Verbose and self.Tracked_Trajectory:
            end = self.Tracked_Trajectory[-1]
            print(f"[PD] Move -> {len(self.Tracked_Trajectory)} waypoints, "
                  f"destination=({end[0]}, {end[1]})")

    def SpeedCap_Receiver(self, speed: float):
        self._speed_cap = float(np.clip(speed, 0.0, self.MAX_SPEED))

    # ------------------------------------------------------------------
    # PD helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_angle(a: float) -> float:
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    def _wheel_speeds(self, forward: float, turn: float):
        cap   = self._speed_cap
        left  = float(np.clip(forward - turn, -cap, cap))
        right = float(np.clip(forward + turn, -cap, cap))
        return left, right

    # ------------------------------------------------------------------
    # Control computations
    # ------------------------------------------------------------------

    def _compute_move(self):
        if not self.Tracked_Trajectory or self._waypoint_idx >= len(self.Tracked_Trajectory):
            if self.Verbose:
                print("[PD] Trajectory complete")
            self._mode = "idle"
            return 0.0, 0.0

        # Current target waypoint
        tx, ty  = float(self.Tracked_Trajectory[self._waypoint_idx][0]), \
                  float(self.Tracked_Trajectory[self._waypoint_idx][1])
        cx, cy  = float(self.Current_State[0]), float(self.Current_State[1])
        ctheta  = float(self.Current_State[2])

        dx, dy   = tx - cx, ty - cy
        distance = float(np.hypot(dx, dy))

        # Advance to next waypoint if close enough
        if distance < self.ARRIVAL_THRESHOLD_M:
            self._waypoint_idx += 1
            self._prev_heading_err = 0.0
            if self._waypoint_idx >= len(self.Tracked_Trajectory):
                if self.Verbose:
                    print("[PD] Destination reached")
                self._mode = "idle"
                return 0.0, 0.0
            # Recalculate with new waypoint
            tx, ty   = float(self.Tracked_Trajectory[self._waypoint_idx][0]), \
                       float(self.Tracked_Trajectory[self._waypoint_idx][1])
            dx, dy   = tx - cx, ty - cy
            distance = float(np.hypot(dx, dy))

        desired_heading = float(np.arctan2(dy, dx))
        heading_err     = self._normalize_angle(desired_heading - ctheta)
        heading_err_d   = heading_err - self._prev_heading_err
        self._prev_heading_err = heading_err

        alignment = max(0.0, float(np.cos(heading_err)))
        forward   = min(self.KP_LINEAR * distance * alignment, self._speed_cap)
        turn      = float(np.clip(
            self.KP_ANGULAR * heading_err + self.KD_ANGULAR * heading_err_d,
            -self._speed_cap, self._speed_cap
        ))

        left, right = self._wheel_speeds(forward, turn)

        if self.Verbose:
            print(
                f"[PD] MOVE  wp={self._waypoint_idx}/{len(self.Tracked_Trajectory)}  "
                f"dist={distance:.2f}m  hdg={np.rad2deg(heading_err):+.1f}deg  "
                f"L={left:+.3f}  R={right:+.3f}"
            )

        return left, right

    def _compute_orient(self):
        target_theta  = float(self.Desired_State[2])
        ctheta        = float(self.Current_State[2])
        heading_err   = self._normalize_angle(target_theta - ctheta)
        heading_err_d = heading_err - self._prev_heading_err
        self._prev_heading_err = heading_err

        if abs(heading_err) < self.ARRIVAL_THRESHOLD_RAD:
            if self.Verbose:
                print(f"[PD] Aligned to theta={np.rad2deg(target_theta):.1f} deg")
            self._mode = "idle"
            return 0.0, 0.0

        turn = float(np.clip(
            self.KP_ANGULAR * heading_err + self.KD_ANGULAR * heading_err_d,
            -self._speed_cap, self._speed_cap
        ))

        left, right = self._wheel_speeds(0.0, turn)

        if self.Verbose:
            print(
                f"[PD] ORIENT  hdg_err={np.rad2deg(heading_err):+.1f}deg  "
                f"L={left:+.3f}  R={right:+.3f}"
            )

        return left, right

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def MPC_Main(self):
        if self._mode == "move":
            left, right = self._compute_move()
        elif self._mode == "orient":
            left, right = self._compute_orient()
        else:
            return  # idle -- no output

        if self._drive_callback is not None:
            cmd = np.array([left, right], dtype=np.float64)
            await self._drive_callback({"Id": "System", "Cmd": cmd})

    async def DriveTrain_Sender(self):
        while True:
            await asyncio.sleep(0.1)
            await self.MPC_Main()