import csv
import os
import time
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR = "/home/turkeysquad/ME450/MainDesign/Navigation/logs"

COLUMNS = [
    "timestamp_s",          # monotonic time since logger init (seconds)
    "wall_time",            # human-readable HH:MM:SS.mmm
    # Robot state
    "x",
    "y",
    "theta_rad",
    "theta_deg",
    # MPC mode
    "mode",                 # idle | trajectory | orient | pivot | escape
    # Current target waypoint
    "waypoint_x",
    "waypoint_y",
    # Obstacle proximity
    "edt_distance_m",       # EDT distance to nearest static obstacle
    # Control outputs
    "vl",                   # left wheel speed (m/s)
    "vr",                   # right wheel speed (m/s)
    "v_fwd",                # (vl + vr) / 2
    "omega",                # (vr - vl) / L  — yaw rate
    # Dynamic obstacles
    "n_dynamic_obstacles",
    # Solver diagnostics
    "solve_time_ms",        # wall-clock time IPOPT took this tick
    "warm_start_used",      # True / False
    # Replan events
    "replan_triggered",     # True on ticks where A* was re-run
    "n_waypoints",          # number of waypoints in current path
]


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class NavigationLogger:
    """
    Lightweight CSV logger for the navigation stack.

    Usage
    -----
    logger = NavigationLogger()

    logger.log(
        state         = [x, y, theta],
        mode          = "trajectory",
        waypoint      = (wx, wy),
        edt_distance  = 2.4,
        vl            = 1.2,
        vr            = 1.4,
        n_dynamic     = 2,
        solve_time_ms = 14.3,
        warm_start    = True,
        replan        = False,
        n_waypoints   = 5,
        L             = 0.7,
    )

    logger.close()
    """

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename      = f"navigation_{timestamp_str}.csv"
        self._path    = os.path.join(LOG_DIR, filename)

        self._start_time = time.monotonic()
        self._file   = open(self._path, "w", newline="", buffering=1)  # line-buffered
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()

        print(f"[Logger] Logging to {self._path}")

    # ─────────────────────────────────────────────────────────────────────────

    def log(self,
            state:         list,
            mode:          str,
            waypoint:      tuple,
            edt_distance:  float,
            vl:            float,
            vr:            float,
            n_dynamic:     int,
            solve_time_ms: float,
            warm_start:    bool,
            replan:        bool,
            n_waypoints:   int,
            L:             float = 0.7):
        """
        Write one row to the CSV.  All parameters are positional-keyword;
        the old `turkey_pos` kwarg has been removed — it was being passed by
        MovementExecutor but not accepted here, silently swallowing every
        logger.log() call via a TypeError and producing an empty log file.
        """
        import math

        now       = time.monotonic()
        elapsed   = now - self._start_time
        wall_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        theta_wrapped = math.atan2(math.sin(theta), math.cos(theta))

        v_fwd = (vl + vr) / 2.0
        omega = (vr - vl) / L

        row = {
            "timestamp_s":           round(elapsed, 4),
            "wall_time":             wall_time,
            "x":                     round(x, 4),
            "y":                     round(y, 4),
            "theta_rad":             round(theta_wrapped, 5),
            "theta_deg":             round(math.degrees(theta_wrapped), 3),
            "mode":                  mode,
            "waypoint_x":            round(float(waypoint[0]), 3) if waypoint is not None else float("nan"),
            "waypoint_y":            round(float(waypoint[1]), 3) if waypoint is not None else float("nan"),
            "edt_distance_m":        round(float(edt_distance), 4),
            "vl":                    round(float(vl), 4),
            "vr":                    round(float(vr), 4),
            "v_fwd":                 round(v_fwd, 4),
            "omega":                 round(omega, 5),
            "n_dynamic_obstacles":   int(n_dynamic),
            "solve_time_ms":         round(float(solve_time_ms), 3),
            "warm_start_used":       warm_start,
            "replan_triggered":      replan,
            "n_waypoints":           int(n_waypoints),
        }

        self._writer.writerow(row)

    # ─────────────────────────────────────────────────────────────────────────

    def close(self):
        self._file.flush()
        self._file.close()
        print(f"[Logger] Closed log: {self._path}")

    @property
    def path(self):
        return self._path