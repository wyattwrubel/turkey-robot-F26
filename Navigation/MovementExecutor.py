import asyncio
import time
import numpy as np
from collections import deque
import sys
sys.path.insert(0, '/home/turkeysquad/ME450/MainDesign/casadi')
if 'casadi' in sys.modules:
    del sys.modules['casadi']
import casadi as ca

# =============================================================================
# Config
# =============================================================================
WHEELBASE = 0.7
L         = WHEELBASE
DT        = 0.1
N         = 8

# WARNING: these are intentionally wide for testing.  Call set_speed_cap()
# at startup to apply real operating limits (e.g. 3.0 m/s).
V_MAX_DEFAULT = 20.0
V_MIN_DEFAULT = -20.0

# Arrival: robot is considered to have reached the end of the A* path when it
# is within this distance of the final waypoint.
PATH_END_DIST = 1.0   # metres (1 grid cell)

STALL_WINDOW = 30     # ticks
STALL_DELTA  = 0.3    # metres — minimum progress over STALL_WINDOW ticks

LOOKAHEAD_DIST = 2.5  # metres

# Pre-move pivot: only fire if heading error exceeds this.  Raised from 10 deg
# to 70 deg so the robot skips the pivot on shallow angles and drives directly.
PIVOT_THRESHOLD_DEG    = 70.0

# Loosened from 3° to 10° — now tightened back to 2° for aggressive closed-loop
# alignment.  The P controller runs continuously and re-reads live pose every
# DT tick, so the robot keeps correcting until it genuinely reaches this threshold.
PIVOT_ALIGN_DEG        =  2.0

# Closed-loop P gain for pivot.  Output is wheel speed (m/s) via:
#   wheel_speed = clip(PIVOT_KP * |error_rad|, PIVOT_MIN_SPEED, v_max)
# Raised from 3.0 to 8.0: a 90° error → ~12.6 m/s before clip, ensuring the
# robot always commands full available speed on large errors.  The live heading
# error is re-read from RobotPose every DT tick so this is truly closed-loop.
PIVOT_KP               =  8.0

# Minimum wheel speed during a pivot so small residual errors still overcome
# drivetrain stiction.  Applied after the P formula, before v_max clipping.
PIVOT_MIN_SPEED        =  0.3   # m/s — tune to your drivetrain's stall threshold

PIVOT_ALIGN_SEGMENTS   =  3     # leading path segments used for pre-move heading

# Pivot stall: abort pivot if heading error hasn't improved by this many degrees
# over PIVOT_STALL_WINDOW ticks.
PIVOT_STALL_WINDOW     = 15     # ticks (~1.5 s at DT=0.1)
PIVOT_STALL_DEG        =  5.0   # minimum heading progress over the window

# Terminal orient: abort after this many consecutive stall windows, then idle.
# 3 × 1.5 s ≈ 4.5 s total before giving up.
PIVOT_STALL_MAX_ATTEMPTS = 3

# Obstacle escape (targeted cell + MPC):
ESCAPE_EDT_FLOOR      = 2.0   # minimum EDT a candidate cell must have
ESCAPE_BEHIND_WEIGHT  = 3.0   # how strongly cells behind the robot are preferred
ESCAPE_SEARCH_RADIUS  = 12    # BFS radius in grid cells

# Direct-chase mode parameters:
# Activated when Mode=="Chase" flag is in the Move command AND the robot is
# within CHASE_ENGAGE_DIST of the turkey AND the corridor is clear.
CHASE_ENGAGE_DIST      = 5.0   # metres — switch to direct-chase inside this radius
CHASE_CORRIDOR_WIDTH   = 1.0   # metres — half-width of obstacle corridor check
CHASE_CORRIDOR_STEP    = 1.0   # metres — sample interval along corridor
CHASE_TARGET_MOVE_DIST = 1.0   # metres — recompute intercept if turkey moves this far
CHASE_ARRIVE_DIST      = 1.5   # metres — stop chasing when this close
CHASE_MIN_DIST         = 2.0   # metres — hold position when closer than this

# Direct-chase MPC cost weights (override run weights for close-range pursuit).
W_CHASE_XY       =  5.0   # position tracking toward turkey
W_CHASE_HEADING  = 20.0   # stronger heading weight to keep turkey centred in FOV
W_CHASE_FOV      = 15.0   # cost on camera angle offset (col 2 of turkey array)
W_CHASE_OMEGA    =  1.0
W_CHASE_INPUT_DIFF = 0.5

# Standard trajectory MPC weights
W_XY_RUN     =  5.0
W_XY_TERM    = 80.0
W_HEADING    = 12.0
W_OMEGA      =  1.0
W_INPUT_DIFF =  0.5

# =============================================================================
# Helpers
# =============================================================================
def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

def get_state(parent):
    raw = np.array(parent.Current_State, dtype=np.float64)
    return np.array([raw[0], raw[1], wrap(raw[2])])

def _path_entry_heading(robot_xy, waypoints, n_segments=PIVOT_ALIGN_SEGMENTS):
    """
    Compute the heading the robot should face before beginning to track a path.
    Takes the mean direction of the first n_segments segments, weighted by
    length, to avoid reacting to a single tiny 1-cell step.
    Returns heading in radians, or None if path has fewer than 2 points.
    """
    pts = [np.array(wp, dtype=float) for wp in waypoints]
    if len(pts) < 2:
        return None

    start = 0
    for i, pt in enumerate(pts):
        if np.linalg.norm(pt - np.array(robot_xy)) > 0.5:
            start = i
            break

    vx_sum, vy_sum, total_len = 0.0, 0.0, 0.0
    segments_used = 0
    for i in range(start, min(start + n_segments, len(pts) - 1)):
        seg = pts[i + 1] - pts[i]
        sl  = np.linalg.norm(seg)
        if sl < 1e-6:
            continue
        vx_sum    += seg[0]
        vy_sum    += seg[1]
        total_len += sl
        segments_used += 1

    if segments_used == 0 or total_len < 1e-6:
        return None

    return float(np.arctan2(-vy_sum, vx_sum))


def _find_escape_cell(map_obj, robot_xy, robot_heading, verbose=False):
    """
    Find the best cell for the robot to drive to in order to escape an obstacle.

    BFS outward from the robot's current grid cell.  Each candidate cell is
    scored as:

        score = edt(cell) + ESCAPE_BEHIND_WEIGHT * cos(angle_to_cell - (heading + π))

    The cosine term is maximised when the cell is directly behind the robot
    (opposite to its current heading), so the escape naturally retraces the
    approach path rather than powering through the obstacle.

    Hard floor: only cells with EDT >= ESCAPE_EDT_FLOOR are eligible.

    Fallback: if no cell clears the floor (robot is deeply boxed in), returns
    the single highest-EDT reachable cell so there is always somewhere to go.

    Returns (int x, int y) grid cell, or None if the map is completely blocked.
    """
    from . import PathGenerator as PG

    gz       = map_obj.Zone_Map
    edt_grid = PG._get_edt(gz)
    h, w     = gz.shape

    gx = int(np.clip(round(robot_xy[0]), 0, w - 1))
    gy = int(np.clip(round(robot_xy[1]), 0, h - 1))

    behind_angle = wrap(robot_heading + np.pi)

    candidates    = []
    fallback_best  = None
    fallback_score = -np.inf

    visited = {(gx, gy)}
    queue   = [(gx, gy, 0)]

    while queue:
        next_queue = []
        for cx, cy, depth in queue:
            if depth > ESCAPE_SEARCH_RADIUS:
                continue

            cell_edt = float(edt_grid[cy, cx])

            dx = cx - gx
            dy = cy - gy
            if dx == 0 and dy == 0:
                behind_score = 0.0
            else:
                angle_to_cell = np.arctan2(-dy, dx)
                behind_score  = float(np.cos(angle_to_cell - behind_angle))

            score = cell_edt + ESCAPE_BEHIND_WEIGHT * behind_score

            if cell_edt >= ESCAPE_EDT_FLOOR:
                candidates.append((score, (cx, cy)))
            elif score > fallback_score:
                fallback_score = score
                fallback_best  = (cx, cy)

            for ndx, ndy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + ndx, cy + ndy
                if (nx, ny) not in visited and 0 <= nx < w and 0 <= ny < h:
                    cell = gz[ny, nx]
                    if cell != 1 and cell != -1:
                        visited.add((nx, ny))
                        next_queue.append((nx, ny, depth + 1))

        queue = next_queue

    if candidates:
        best = max(candidates, key=lambda c: c[0])[1]
        if verbose:
            print(f"[MPC] Escape cell: {best}  "
                  f"edt={float(edt_grid[best[1], best[0]]):.1f}  "
                  f"({len(candidates)} candidates, "
                  f"behind={np.degrees(behind_angle):.1f} deg)")
        return best

    if fallback_best is not None:
        if verbose:
            print(f"[MPC] Escape cell (fallback — no EDT>={ESCAPE_EDT_FLOOR} found): "
                  f"{fallback_best}  edt={float(edt_grid[fallback_best[1], fallback_best[0]]):.1f}")
        return fallback_best

    if verbose:
        print("[MPC] Escape: no reachable cell found — fully boxed in")
    return None


def _corridor_clear(map_grid, robot_xy, turkey_xy, half_width=CHASE_CORRIDOR_WIDTH,
                    step=CHASE_CORRIDOR_STEP):
    """
    Sample cells along the straight line from robot_xy to turkey_xy at `step`
    metre intervals, checking a ±half_width perpendicular band at each sample.

    Returns True if no sampled cell contains a static obstacle (1 or -1).
    Dynamic obstacles (2) are intentionally ignored — the turkey itself is
    dynamic and we don't want to abort the chase due to its own footprint.
    """
    h, w   = map_grid.shape
    vec    = np.array(turkey_xy, dtype=float) - np.array(robot_xy, dtype=float)
    dist   = np.linalg.norm(vec)
    if dist < 1e-6:
        return True

    unit    = vec / dist
    perp    = np.array([-unit[1], unit[0]])  # 90° rotation

    n_steps = max(1, int(dist / step))
    offsets = np.arange(-int(half_width), int(half_width) + 1)

    for i in range(n_steps + 1):
        t      = min(i * step, dist)
        centre = np.array(robot_xy, dtype=float) + unit * t
        for o in offsets:
            pt = centre + perp * o
            gx = int(np.clip(round(pt[0]), 0, w - 1))
            gy = int(np.clip(round(pt[1]), 0, h - 1))
            cell = map_grid[gy, gx]
            if cell == 1 or cell == -1:
                return False
    return True


# =============================================================================
# Unified MPC solver (standard trajectory)
# =============================================================================

def build_solver(n_steps=N):
    n = n_steps; ns = 3; nu = 2

    xs, ys, ths = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('th')
    vls, vrs    = ca.SX.sym('vl'), ca.SX.sym('vr')
    vs  = (vls + vrs) / 2
    ws  = (vrs - vls) / L
    f   = ca.Function('f',
        [ca.vertcat(xs, ys, ths), ca.vertcat(vls, vrs)],
        [ca.vertcat(xs, ys, ths) + DT * ca.vertcat(
            vs * ca.cos(ths), -vs * ca.sin(ths), ws)])

    X   = ca.SX.sym('X', ns, n + 1)
    U   = ca.SX.sym('U', nu, n)
    PX0 = ca.SX.sym('X0', ns)
    Pl  = ca.SX.sym('l',  2)
    Pup = ca.SX.sym('up', nu)
    p   = ca.vertcat(PX0, Pl, Pup)

    obj = 0
    geq = [X[:, 0] - PX0]

    for k in range(n):
        xk, yk, thk = X[0, k], X[1, k], X[2, k]
        vlk, vrk     = U[0, k], U[1, k]

        obj += W_XY_RUN * ((xk - Pl[0])**2 + (yk - Pl[1])**2)

        wdx  = Pl[0] - xk
        wdy  = Pl[1] - yk
        wreg = ca.sqrt(wdx**2 + wdy**2 + 1e-6)
        aw   = ca.atan2(-wdy / wreg, wdx / wreg)
        obj += W_HEADING * ca.atan2(ca.sin(aw - thk), ca.cos(aw - thk))**2

        obj += W_OMEGA * ((vrk - vlk) / L)**2

        upk  = Pup if k == 0 else U[:, k - 1]
        obj += W_INPUT_DIFF * ca.sumsqr(U[:, k] - upk)

        geq.append(X[:, k + 1] - f(X[:, k], U[:, k]))

    obj += W_XY_TERM * ((X[0, n] - Pl[0])**2 + (X[1, n] - Pl[1])**2)

    ov  = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nd  = ov.numel()
    cs  = ns * (n + 1)

    lbx = -np.inf * np.ones(nd); ubx = np.inf * np.ones(nd)
    lbx[cs:] = V_MIN_DEFAULT;    ubx[cs:] = V_MAX_DEFAULT

    gc  = ca.vertcat(*geq)
    ne  = gc.numel()
    lbg = np.zeros(ne); ubg = np.zeros(ne)

    slv = ca.nlpsol('solver', 'ipopt',
        {'f': obj, 'x': ov, 'g': gc, 'p': p},
        {'ipopt.print_level': 0, 'print_time': 0,
         'ipopt.tol': 1e-2, 'ipopt.max_iter': 50,
         'ipopt.warm_start_init_point': 'yes',
         'ipopt.mu_strategy': 'adaptive'})

    return dict(solver=slv, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg,
                n_dec=nd, ctrl_start=cs, ns=ns, nu=nu, N=n, n_p=p.numel())


# =============================================================================
# Direct-chase MPC solver
# =============================================================================
# Parameter vector: X0(3) | turkey_xy(2) | fov_angle(1) | u_prev(2)  (total: 8)
# fov_angle: relative angle of turkey from camera centre (radians, col 2 of
#            the Turkeys array).  Zero means turkey is centred in FOV.
# =============================================================================

def build_chase_solver(n_steps=N):
    n = n_steps; ns = 3; nu = 2

    xs, ys, ths = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('th')
    vls, vrs    = ca.SX.sym('vl'), ca.SX.sym('vr')
    vs  = (vls + vrs) / 2
    ws  = (vrs - vls) / L
    f   = ca.Function('f',
        [ca.vertcat(xs, ys, ths), ca.vertcat(vls, vrs)],
        [ca.vertcat(xs, ys, ths) + DT * ca.vertcat(
            vs * ca.cos(ths), -vs * ca.sin(ths), ws)])

    X      = ca.SX.sym('X',   ns, n + 1)
    U      = ca.SX.sym('U',   nu, n)
    PX0    = ca.SX.sym('X0',  ns)
    Ptxy   = ca.SX.sym('txy', 2)       # turkey world position
    Pfov   = ca.SX.sym('fov', 1)       # current FOV angle offset (radians)
    Pup    = ca.SX.sym('up',  nu)
    p      = ca.vertcat(PX0, Ptxy, Pfov, Pup)

    obj = 0
    geq = [X[:, 0] - PX0]

    for k in range(n):
        xk, yk, thk = X[0, k], X[1, k], X[2, k]
        vlk, vrk     = U[0, k], U[1, k]

        # Drive toward turkey XY
        obj += W_CHASE_XY * ((xk - Ptxy[0])**2 + (yk - Ptxy[1])**2)

        # Align robot heading toward turkey world position
        wdx  = Ptxy[0] - xk
        wdy  = Ptxy[1] - yk
        wreg = ca.sqrt(wdx**2 + wdy**2 + 1e-6)
        aw   = ca.atan2(-wdy / wreg, wdx / wreg)
        obj += W_CHASE_HEADING * ca.atan2(ca.sin(aw - thk), ca.cos(aw - thk))**2

        # FOV centering: penalise camera angle offset directly.
        # Pfov is the *current* measured offset — we want it driven to zero.
        # We apply it at every step with a decaying weight so the solver
        # prioritises early correction (when the offset is most observable).
        decay = float(n - k) / float(n)
        obj  += W_CHASE_FOV * decay * Pfov**2

        # Smoothness
        obj += W_CHASE_OMEGA * ((vrk - vlk) / L)**2
        upk  = Pup if k == 0 else U[:, k - 1]
        obj += W_CHASE_INPUT_DIFF * ca.sumsqr(U[:, k] - upk)

        geq.append(X[:, k + 1] - f(X[:, k], U[:, k]))

    # Terminal cost
    obj += W_CHASE_XY * 10.0 * ((X[0, n] - Ptxy[0])**2 + (X[1, n] - Ptxy[1])**2)

    ov  = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nd  = ov.numel()
    cs  = ns * (n + 1)

    lbx = -np.inf * np.ones(nd); ubx = np.inf * np.ones(nd)
    lbx[cs:] = V_MIN_DEFAULT;    ubx[cs:] = V_MAX_DEFAULT

    gc  = ca.vertcat(*geq)
    ne  = gc.numel()
    lbg = np.zeros(ne); ubg = np.zeros(ne)

    slv = ca.nlpsol('chase_solver', 'ipopt',
        {'f': obj, 'x': ov, 'g': gc, 'p': p},
        {'ipopt.print_level': 0, 'print_time': 0,
         'ipopt.tol': 1e-2, 'ipopt.max_iter': 50,
         'ipopt.warm_start_init_point': 'yes',
         'ipopt.mu_strategy': 'adaptive'})

    return dict(solver=slv, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg,
                n_dec=nd, ctrl_start=cs, ns=ns, nu=nu, N=n, n_p=p.numel())


# =============================================================================
# Warm-start helpers
# =============================================================================

def _seed_guess(state, mpc):
    x0 = np.zeros(mpc['n_dec'])
    for k in range(mpc['N'] + 1):
        x0[k * mpc['ns']:k * mpc['ns'] + mpc['ns']] = state
    return x0

def _shift_guess(sol_flat, mpc):
    ns, nu, n, cs = mpc['ns'], mpc['nu'], mpc['N'], mpc['ctrl_start']
    X  = sol_flat[:cs].reshape(ns, n + 1, order='F')
    U  = sol_flat[cs:].reshape(nu, n,     order='F')
    x0 = np.zeros(mpc['n_dec'])
    x0[:cs] = np.hstack([X[:, 1:], X[:, -1:]]).flatten(order='F')
    x0[cs:] = np.hstack([U[:, 1:], U[:, -1:]]).flatten(order='F')
    return x0

# =============================================================================
# Lookahead
# =============================================================================

def _advance_waypoints(robot_xy, waypoints):
    """
    Drop waypoints the robot has already passed, keeping at least one.
    A waypoint is passed when the robot is within PATH_END_DIST of it
    AND it is not the last waypoint (the goal).
    """
    while len(waypoints) > 1:
        dist = np.linalg.norm(np.array(robot_xy) - np.array(waypoints[0], dtype=float))
        if dist < PATH_END_DIST:
            waypoints.pop(0)
        else:
            break
    return waypoints


def _lookahead_point(robot_xy, waypoints, dist=LOOKAHEAD_DIST):
    """
    Project a point `dist` metres ahead along the pruned A* path.
    Returns (lookahead_xy, at_end).
    at_end is True when robot is within PATH_END_DIST of the final waypoint.
    """
    if not waypoints:
        return np.array(robot_xy), False

    pts    = [np.array(wp, dtype=float) for wp in waypoints]
    at_end = np.linalg.norm(np.array(robot_xy) - pts[-1]) < PATH_END_DIST

    remaining = dist
    for i in range(len(pts) - 1):
        seg = pts[i + 1] - pts[i]
        sl  = np.linalg.norm(seg)
        if sl < 1e-6:
            continue
        if sl >= remaining:
            return pts[i] + seg * (remaining / sl), at_end
        remaining -= sl
    return pts[-1], at_end

# =============================================================================
# Controller
# =============================================================================

class MPC_loop:

    def __init__(self, parent, Drivetrain_pub, MPCFeedback_Navigation_pub,
                 verbose=False):
        self.Drivetrain_pub  = Drivetrain_pub
        self.MPCFeedback_pub = MPCFeedback_Navigation_pub
        self._parent         = parent
        self._verbose        = verbose

        self._mode      = "idle"
        self._waypoints = []
        self._target    = None

        # Heading to face at path end (set when goal was snapped).
        self._face_heading = None

        self._pivot_target_hdg         = 0.0
        self._pivot_is_terminal_orient = False
        self._pivot_err_history        = deque(maxlen=PIVOT_STALL_WINDOW)
        # Count of consecutive stall windows for terminal orient abort logic.
        self._pivot_stall_attempts     = 0

        self._dist_history = deque(maxlen=STALL_WINDOW)
        self._solving      = False

        self._stall_replan_task     = None
        self._stall_replan_cooldown = False

        # Escape state
        self._escape_target_cell = None

        self._guess       = None
        self._chase_guess = None
        self._u_prev      = np.zeros(2)

        self._v_max = V_MAX_DEFAULT
        self._v_min = V_MIN_DEFAULT

        # ── Direct-chase state ─────────────────────────────────────────────
        # Set True when the current Move command carries Mode=="Chase".
        self._chase_mode            = False
        # Locked intercept point — recomputed only when turkey moves > threshold.
        self._chase_intercept       = None   # np.ndarray [x, y] or None
        # Last turkey position used to compute the intercept.
        self._chase_last_turkey_pos = None   # np.ndarray [x, y] or None
        # Latest FOV angle offset from the turkey array (col 2).
        self._chase_fov_angle       = 0.0

        print(f"[MPC] Building standard solver (N={N})...")
        self._mpc = build_solver(N)
        print(f"[MPC] Building chase solver (N={N})...")
        self._chase_mpc = build_chase_solver(N)
        print(f"[MPC] Solvers ready. n_dec={self._mpc['n_dec']}")

    # ── Speed cap ─────────────────────────────────────────────────────────────

    def set_speed_cap(self, cap: float):
        """Symmetrically clamp wheel speeds to [-cap, +cap]."""
        cap = float(cap)
        cap = max(0.0, min(cap, V_MAX_DEFAULT))
        self._v_max =  cap
        self._v_min = -cap
        for mpc in (self._mpc, self._chase_mpc):
            cs = mpc['ctrl_start']
            mpc['lbx'][cs:] = self._v_min
            mpc['ubx'][cs:] = self._v_max
        if self._verbose:
            print(f"[MPC] Speed cap updated: [{self._v_min:.2f}, {self._v_max:.2f}] m/s")

    # ── Protocol helpers ──────────────────────────────────────────────────────

    async def _pub_reference(self, wps):
        await self.MPCFeedback_pub.publish(
            {"Id": "Reference",
             "Cmd": [[float(x), float(y)] for x, y in wps]})

    async def _pub_clear(self):
        await self.MPCFeedback_pub.publish({"Id": "ClearTrajectory"})

    # ── Command receivers ─────────────────────────────────────────────────────

    async def Orient_CMD_Receiver(self, data):
        new_hdg = wrap(float(data[2]))

        # If already pivoting to the same heading (within PIVOT_ALIGN_DEG),
        # just refresh the target without resetting history or u_prev.
        # This lets _wait_for_heading re-publish Orient for drift correction
        # without destroying the warm state of an in-progress pivot.
        if (self._mode == "pivot"
                and not self._pivot_is_terminal_orient
                and abs(np.degrees(wrap(new_hdg - self._pivot_target_hdg)))
                    < PIVOT_ALIGN_DEG):
            self._pivot_target_hdg = new_hdg   # refresh (effectively a no-op)
            return

        # New or significantly different heading — full reset.
        if self._stall_replan_task is not None and not self._stall_replan_task.done():
            self._stall_replan_task.cancel()
            self._stall_replan_task = None

        self._waypoints             = []
        self._target                = None
        self._face_heading          = None
        self._dist_history.clear()
        self._stall_replan_cooldown = False
        self._escape_target_cell    = None
        self._chase_mode            = False
        self._chase_intercept       = None
        self._chase_last_turkey_pos = None

        self._u_prev = np.zeros(2)
        self._guess  = None

        self._mode                     = "pivot"
        self._pivot_is_terminal_orient = False
        self._pivot_target_hdg         = new_hdg
        self._pivot_err_history.clear()
        self._pivot_stall_attempts     = 0

        await self._pub_clear()

        if self._verbose:
            print(f"[MPC] Orient — pivoting to "
                  f"{np.degrees(self._pivot_target_hdg):.1f} deg")

    async def Move_CMD_Receiver(self, waypoints, target_xy, face_heading=None,
                                chase_mode=False, turkey_data=None):
        """
        Begin tracking a new A* path, or enter direct-chase mode.

        Parameters
        ----------
        waypoints    : list of (int x, int y)
        target_xy    : (x, y) — original requested destination
        face_heading : float (radians) or None — terminal orient heading
        chase_mode   : bool — if True, attempt direct-chase (no A* path needed)
        turkey_data  : list/array [x, y, fov_angle] or None — live turkey data
                       used by direct-chase; fov_angle is col 2 of Turkeys array
        """
        if not waypoints and not chase_mode:
            if self._verbose:
                print("[MPC] Move_CMD_Receiver: empty waypoints and not chase — ignored")
            return

        dest_changed = (
            self._target is None
            or np.linalg.norm(np.array(target_xy) - np.array(self._target)) > 3.0
        )

        self._target       = target_xy
        self._face_heading = face_heading
        self._chase_mode   = chase_mode

        # Update live turkey data for chase solver if provided.
        if turkey_data is not None:
            td = np.array(turkey_data, dtype=float)
            new_turkey_xy = td[:2]
            self._chase_fov_angle = float(td[2]) if len(td) > 2 else 0.0

            # Recompute intercept only if turkey has moved significantly.
            if (self._chase_last_turkey_pos is None or
                    np.linalg.norm(new_turkey_xy - self._chase_last_turkey_pos)
                    > CHASE_TARGET_MOVE_DIST):
                self._chase_intercept       = new_turkey_xy.copy()
                self._chase_last_turkey_pos = new_turkey_xy.copy()
                self._chase_guess           = None   # invalidate warm start
                if self._verbose:
                    print(f"[MPC] Chase intercept updated: {self._chase_intercept}")

        if chase_mode:
            # In direct-chase mode we bypass the A* path entirely.
            # The corridor check was already done in TrajectoryTracker; we trust it.
            self._waypoints = waypoints   # may be empty
            self._guess     = None
            if dest_changed:
                self._dist_history.clear()
            self._stall_replan_cooldown = False
            self._mode = "chase"
            if self._verbose:
                print(f"[MPC] Direct-chase mode engaged. "
                      f"Intercept={self._chase_intercept}")
            return

        # ── Standard A* path tracking ──────────────────────────────────────
        self._waypoints = waypoints
        self._guess     = None   # invalidate warm start on new plan

        if self._escape_target_cell is not None:
            if self._verbose:
                print("[MPC] Move received during escape — updating target, "
                      "deferring until escape completes")
            if dest_changed:
                self._dist_history.clear()
            return

        self._stall_replan_cooldown = False

        if dest_changed:
            self._dist_history.clear()

        state = get_state(self._parent)
        rxy   = state[:2]

        hdg = _path_entry_heading(rxy, waypoints, n_segments=PIVOT_ALIGN_SEGMENTS)

        if hdg is None:
            self._mode = "trajectory"
            await self._pub_reference(waypoints)
            return

        err = wrap(hdg - state[2])
        if abs(np.degrees(err)) > PIVOT_THRESHOLD_DEG:
            already_pivoting_correctly = (
                self._mode == "pivot"
                and not self._pivot_is_terminal_orient
                and abs(np.degrees(wrap(hdg - self._pivot_target_hdg))) < PIVOT_THRESHOLD_DEG
            )
            if not already_pivoting_correctly:
                self._pivot_target_hdg         = hdg
                self._pivot_is_terminal_orient = False
                self._pivot_err_history.clear()
                self._pivot_stall_attempts     = 0
                self._mode                     = "pivot"
                if self._verbose:
                    print(f"[MPC] Pre-move pivot: {np.degrees(err):.1f} deg error "
                          f"(threshold={PIVOT_THRESHOLD_DEG} deg)")
            elif self._verbose:
                print("[MPC] Already pivoting to similar heading — suppressing update")
            await self._pub_reference(waypoints)
            return

        self._mode = "trajectory"
        await self._pub_reference(waypoints)

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def MPC_Main(self, logger=None):
        if self._solving:
            return
        self._solving = True
        try:
            await self._MPC_Main_inner(logger=logger)
        finally:
            self._solving = False

    async def _MPC_Main_inner(self, logger=None):

        # ── Pivot ──────────────────────────────────────────────────────────────
        if self._mode == "pivot":
            state   = get_state(self._parent)
            err     = wrap(self._pivot_target_hdg - state[2])
            err_deg = abs(np.degrees(err))

            self._pivot_err_history.append(err_deg)

            # Check for a stall only once the window is full.
            if len(self._pivot_err_history) == PIVOT_STALL_WINDOW:
                window_improvement = self._pivot_err_history[0] - err_deg
                if window_improvement < PIVOT_STALL_DEG:
                    self._pivot_stall_attempts += 1
                    self._pivot_err_history.clear()   # start a fresh window

                    if self._verbose:
                        print(f"[MPC] Pivot stall #{self._pivot_stall_attempts} "
                              f"(improvement={window_improvement:.1f} deg < "
                              f"{PIVOT_STALL_DEG} deg required)")

                    abort = (self._pivot_stall_attempts >= PIVOT_STALL_MAX_ATTEMPTS
                             if self._pivot_is_terminal_orient
                             else True)   # non-terminal: abort immediately (original behaviour)

                    if abort:
                        if self._verbose:
                            if self._pivot_is_terminal_orient:
                                print(f"[MPC] Terminal orient stall — "
                                      f"aborted after {self._pivot_stall_attempts} attempts, "
                                      f"going idle")
                            else:
                                print("[MPC] Pivot stall — aborting, resuming trajectory")
                        self._pivot_err_history.clear()
                        self._pivot_stall_attempts = 0
                        self._u_prev = np.zeros(2)
                        self._guess  = None
                        self._mode   = ("trajectory"
                                        if self._waypoints and not self._pivot_is_terminal_orient
                                        else "idle")
                        if self._mode == "idle":
                            await self._send(np.zeros(2))
                            await self._pub_clear()
                        return

            if err_deg < PIVOT_ALIGN_DEG:
                if self._pivot_is_terminal_orient:
                    self._mode                     = "idle"
                    self._pivot_is_terminal_orient = False
                    self._face_heading             = None
                    self._dist_history.clear()
                    self._pivot_stall_attempts     = 0
                    await self._send(np.zeros(2))
                    await self._pub_clear()
                    if self._verbose:
                        print("[MPC] Terminal orient complete — idle.")
                else:
                    self._mode = "trajectory"
                    if self._verbose:
                        print("[MPC] Pre-move pivot complete — trajectory.")
                self._pivot_err_history.clear()
                self._pivot_stall_attempts = 0
                self._u_prev = np.zeros(2)
                self._guess  = None

                if logger:
                    logger.log(state=state, mode="idle",
                        waypoint=tuple(self._target) if self._target is not None else (0, 0),
                        edt_distance=self._parent.Map.grid_edt_at(*state[:2]),
                        vl=0.0, vr=0.0,
                        n_dynamic=self._parent._n_dynamic_obs,
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=len(self._waypoints))
            else:
                # Closed-loop pure-P pivot.
                # Re-reads live heading error from RobotPose every DT tick.
                # Apply a minimum speed floor so stiction doesn't stall small errors.
                omega    = PIVOT_KP * err
                raw_spd  = abs(omega) * L / 2
                spd      = max(raw_spd, PIVOT_MIN_SPEED)
                sign     = 1.0 if omega >= 0 else -1.0
                u = np.array([
                    np.clip(-sign * spd, self._v_min, self._v_max),
                    np.clip( sign * spd, self._v_min, self._v_max),
                ])
                await self._send(u)
                if logger:
                    logger.log(state=state, mode="pivot",
                        waypoint=tuple(self._target) if self._target is not None else (0, 0),
                        edt_distance=self._parent.Map.grid_edt_at(*state[:2]),
                        vl=float(u[0]), vr=float(u[1]),
                        n_dynamic=self._parent._n_dynamic_obs,
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=len(self._waypoints))
            return

        # ── Direct-chase ───────────────────────────────────────────────────────
        if self._mode == "chase":
            state = get_state(self._parent)
            rxy   = state[:2]

            if self._target is None or self._chase_intercept is None:
                self._mode = "idle"
                await self._send(np.zeros(2))
                return

            turkey_xy = self._chase_intercept
            dist_to_turkey = np.linalg.norm(rxy - turkey_xy)

            # ── Arrival ───────────────────────────────────────────────────────
            if dist_to_turkey < CHASE_ARRIVE_DIST:
                if self._verbose:
                    print(f"[MPC] Chase: arrived within {CHASE_ARRIVE_DIST}m — idle")
                self._mode        = "idle"
                self._chase_mode  = False
                await self._send(np.zeros(2))
                await self._pub_clear()
                if logger:
                    logger.log(state=state, mode="idle",
                        waypoint=tuple(turkey_xy),
                        edt_distance=self._parent.Map.grid_edt_at(*rxy),
                        vl=0.0, vr=0.0,
                        n_dynamic=self._parent._n_dynamic_obs,
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=0)
                return

            # ── Corridor re-check — fall back to A* if obstacle appears ───────
            gz = self._parent.Map.Zone_Map
            if not _corridor_clear(gz, rxy, turkey_xy):
                if self._verbose:
                    print("[MPC] Chase: obstacle detected in corridor — "
                          "falling back to A*")
                self._mode        = "trajectory"
                self._chase_mode  = False
                self._chase_guess = None
                # Trigger an immediate replan via the parent.
                if (self._target is not None
                        and (self._stall_replan_task is None
                             or self._stall_replan_task.done())):
                    self._stall_replan_task = asyncio.create_task(
                        self._parent.replan_to(self._target))
                return

            # ── Minimum distance guard ────────────────────────────────────────
            # If the robot is within CHASE_MIN_DIST of the turkey, hold position.
            # No solve, no wheel commands — just send zero velocity.
            if dist_to_turkey < CHASE_MIN_DIST:
                if self._verbose:
                    print(f"[MPC] Chase: within min dist "
                          f"({dist_to_turkey:.2f}m < {CHASE_MIN_DIST}m) — holding")
                await self._send(np.zeros(2))
                if logger:
                    logger.log(state=state, mode="chase",
                        waypoint=tuple(turkey_xy),
                        edt_distance=self._parent.Map.grid_edt_at(*rxy),
                        vl=0.0, vr=0.0,
                        n_dynamic=self._parent._n_dynamic_obs,
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=0)
                return

            # ── Chase MPC solve ────────────────────────────────────────────────
            warm = self._chase_guess is not None
            if not warm:
                self._chase_guess = _seed_guess(state, self._chase_mpc)

            fov_angle = self._chase_fov_angle
            pvec = np.concatenate([state, turkey_xy,
                                   np.array([fov_angle]), self._u_prev])

            t0  = time.perf_counter()
            sol = self._chase_mpc['solver'](
                x0=self._chase_guess,
                lbx=self._chase_mpc['lbx'], ubx=self._chase_mpc['ubx'],
                lbg=self._chase_mpc['lbg'], ubg=self._chase_mpc['ubg'],
                p=pvec)
            ms  = (time.perf_counter() - t0) * 1000

            flat              = sol['x'].full().flatten()
            u                 = flat[self._chase_mpc['ctrl_start']:
                                     self._chase_mpc['ctrl_start'] + self._chase_mpc['nu']]
            self._chase_guess = _shift_guess(flat, self._chase_mpc)
            self._u_prev      = u.copy()

            if self._verbose:
                print(f"[MPC] Chase dist={dist_to_turkey:.2f}m "
                      f"fov={np.degrees(fov_angle):.1f}deg "
                      f"u=[{u[0]:.2f},{u[1]:.2f}] solve={ms:.1f}ms warm={warm}")

            if logger:
                logger.log(state=state, mode="chase",
                    waypoint=tuple(turkey_xy),
                    edt_distance=self._parent.Map.grid_edt_at(*rxy),
                    vl=float(u[0]), vr=float(u[1]),
                    n_dynamic=self._parent._n_dynamic_obs,
                    solve_time_ms=ms, warm_start=warm,
                    replan=False, n_waypoints=0)

            await self._send(u)
            return

        if self._mode != "trajectory":
            return

        state = get_state(self._parent)
        rxy   = state[:2]
        edt   = self._parent.Map.grid_edt_at(*rxy)

        if self._target is None:
            self._mode = "idle"; return

        # ── Obstacle escape (targeted cell + MPC) ────────────────────────────
        from . import PathGenerator as PG

        gz       = self._parent.Map.Zone_Map
        giy      = int(np.clip(round(rxy[1]), 0, gz.shape[0] - 1))
        gix      = int(np.clip(round(rxy[0]), 0, gz.shape[1] - 1))
        grid_edt = float(PG._get_edt(gz)[giy, gix])
        in_obstacle = grid_edt == 0.0

        if in_obstacle and self._escape_target_cell is None:
            target_cell = _find_escape_cell(
                self._parent.Map, rxy, state[2], verbose=self._verbose)

            if target_cell is None:
                await self._send(np.zeros(2))
                if logger:
                    logger.log(state=state, mode="escape",
                        waypoint=tuple(self._target) if self._target is not None else (0, 0),
                        edt_distance=grid_edt, vl=0.0, vr=0.0,
                        n_dynamic=self._parent._n_dynamic_obs,
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=0)
                return

            self._escape_target_cell    = target_cell
            self._guess                 = None
            self._u_prev                = np.zeros(2)
            self._stall_replan_cooldown = False
            if self._verbose:
                print(f"[MPC] Obstacle escape: EDT=0 at ({rxy[0]:.1f},{rxy[1]:.1f}) "
                      f"— driving to escape cell {target_cell} "
                      f"(edt={float(PG._get_edt(gz)[target_cell[1], target_cell[0]]):.1f})")

        if self._escape_target_cell is not None:
            escape_cell = self._escape_target_cell

            if grid_edt >= ESCAPE_EDT_FLOOR and not in_obstacle:
                if self._verbose:
                    print(f"[MPC] Escape cleared (EDT={grid_edt:.1f} >= {ESCAPE_EDT_FLOOR}) "
                          f"— replanning from ({rxy[0]:.1f},{rxy[1]:.1f})")
                self._escape_target_cell = None
                await self._send(np.zeros(2))
                if (self._target is not None
                        and (self._stall_replan_task is None
                             or self._stall_replan_task.done())):
                    self._stall_replan_task = asyncio.create_task(
                        self._parent.replan_to(self._target))
                return

            escape_xy = np.array(escape_cell, dtype=float)
            warm = self._guess is not None
            if not warm:
                self._guess = _seed_guess(state, self._mpc)
            pvec = np.concatenate([state, escape_xy, self._u_prev])

            t0  = time.perf_counter()
            sol = self._mpc['solver'](
                x0=self._guess,
                lbx=self._mpc['lbx'], ubx=self._mpc['ubx'],
                lbg=self._mpc['lbg'], ubg=self._mpc['ubg'],
                p=pvec)
            ms  = (time.perf_counter() - t0) * 1000

            flat         = sol['x'].full().flatten()
            u            = flat[self._mpc['ctrl_start']:self._mpc['ctrl_start'] + self._mpc['nu']]
            self._guess  = _shift_guess(flat, self._mpc)
            self._u_prev = u.copy()

            if self._verbose:
                dist_esc = np.linalg.norm(rxy - escape_xy)
                print(f"[MPC] Escape driving to {escape_cell} "
                      f"dist={dist_esc:.2f}m edt={grid_edt:.2f} "
                      f"u=[{u[0]:.2f},{u[1]:.2f}] solve={ms:.1f}ms")

            if logger:
                logger.log(state=state, mode="escape",
                    waypoint=tuple(escape_cell),
                    edt_distance=grid_edt, vl=float(u[0]), vr=float(u[1]),
                    n_dynamic=self._parent._n_dynamic_obs,
                    solve_time_ms=ms, warm_start=warm,
                    replan=False, n_waypoints=len(self._waypoints))

            await self._send(u)
            return

        # ── Guard: empty waypoints ────────────────────────────────────────────
        if not self._waypoints:
            if self._verbose:
                print("[MPC] trajectory with empty waypoints — going idle")
            self._mode = "idle"
            await self._send(np.zeros(2))
            await self._pub_clear()
            return

        # ── Prune passed waypoints ────────────────────────────────────────────
        self._waypoints = _advance_waypoints(rxy, self._waypoints)

        # ── Lookahead and end-of-path detection ───────────────────────────────
        lxy, at_end = _lookahead_point(rxy, self._waypoints)

        # ── Arrival ───────────────────────────────────────────────────────────
        if at_end:
            await self._send(np.zeros(2))

            if self._face_heading is not None:
                if self._verbose:
                    print(f"[MPC] Path end — terminal orient to "
                          f"{np.degrees(self._face_heading):.1f} deg")
                self._pivot_target_hdg         = self._face_heading
                self._pivot_is_terminal_orient = True
                self._pivot_err_history.clear()
                self._pivot_stall_attempts     = 0
                self._waypoints                = []
                self._mode                     = "pivot"
            else:
                if self._verbose:
                    print("[MPC] Path end reached — idle")
                self._mode         = "idle"
                self._dist_history.clear()
                await self._pub_clear()

            if logger:
                logger.log(state=state, mode="idle",
                    waypoint=tuple(lxy),
                    edt_distance=edt, vl=0.0, vr=0.0,
                    n_dynamic=self._parent._n_dynamic_obs,
                    solve_time_ms=0.0, warm_start=False,
                    replan=False, n_waypoints=len(self._waypoints))
            return

        # ── Stall detection (disabled in chase mode) ───────────────────────────
        dist_to_end = np.linalg.norm(rxy - np.array(self._waypoints[-1], dtype=float))
        self._dist_history.append(dist_to_end)
        stalled = False

        if not self._chase_mode:
            stalled = (
                not self._stall_replan_cooldown
                and len(self._dist_history) == STALL_WINDOW
                and (self._dist_history[0] - dist_to_end) < STALL_DELTA
            )

            if stalled:
                if self._verbose:
                    print(f"[MPC] Stall detected (dist_to_end={dist_to_end:.2f}m) — replanning")
                self._dist_history.clear()
                self._guess                 = None
                self._stall_replan_cooldown = True
                self._stall_replan_task     = asyncio.create_task(
                    self._parent.replan_to(self._target))

        # ── Solve ─────────────────────────────────────────────────────────────
        warm = self._guess is not None
        if not warm:
            self._guess = _seed_guess(state, self._mpc)
        pvec = np.concatenate([state, lxy, self._u_prev])

        t0  = time.perf_counter()
        sol = self._mpc['solver'](
            x0=self._guess,
            lbx=self._mpc['lbx'], ubx=self._mpc['ubx'],
            lbg=self._mpc['lbg'], ubg=self._mpc['ubg'],
            p=pvec)
        ms  = (time.perf_counter() - t0) * 1000

        flat         = sol['x'].full().flatten()
        u            = flat[self._mpc['ctrl_start']:self._mpc['ctrl_start'] + self._mpc['nu']]
        self._guess  = _shift_guess(flat, self._mpc)
        self._u_prev = u.copy()

        if self._verbose:
            print(f"[MPC] mode={self._mode} dist_to_end={dist_to_end:.2f}m edt={edt:.2f}m "
                  f"state=({state[0]:.2f},{state[1]:.2f},{np.degrees(state[2]):.1f}deg) "
                  f"u=[{u[0]:.2f},{u[1]:.2f}] solve={ms:.1f}ms warm={warm} "
                  f"vlim=[{self._v_min:.1f},{self._v_max:.1f}]")

        if logger:
            logger.log(state=state, mode=self._mode, waypoint=tuple(lxy),
                edt_distance=edt, vl=float(u[0]), vr=float(u[1]),
                n_dynamic=self._parent._n_dynamic_obs,
                solve_time_ms=ms, warm_start=warm,
                replan=stalled, n_waypoints=len(self._waypoints))

        await self._send(u)

    async def _send(self, u):
        await self.Drivetrain_pub.publish(
            {"Id": "System", "Cmd": np.array(u, dtype=np.float32).tolist()})

    async def DriveTrain_Sender(self, logger=None):
        while True:
            t0 = asyncio.get_event_loop().time()
            await self.MPC_Main(logger=logger)
            await asyncio.sleep(
                max(0.0, DT - (asyncio.get_event_loop().time() - t0)))