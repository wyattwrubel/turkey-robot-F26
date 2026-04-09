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
DT        = 0.2
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

PIVOT_THRESHOLD_DEG    = 10.0
PIVOT_ALIGN_DEG        =  3.0
PIVOT_KP               =  3.0
PIVOT_ALIGN_SEGMENTS   =  3     # number of leading path segments used to compute pre-move heading

# Pivot stall: if heading error hasn't improved by this many degrees over
# PIVOT_STALL_WINDOW ticks, abort the pivot and proceed to trajectory.
PIVOT_STALL_WINDOW     = 15     # ticks (~3 s at DT=0.2)
PIVOT_STALL_DEG        =  5.0   # minimum heading progress required over the window

# Obstacle escape: straight-line drive used when EDT == 0 (robot inside obstacle).
# Speed is capped and only forward motion is commanded (no reversing, no turning).
ESCAPE_SPEED           =  0.5   # m/s — slow and deliberate
ESCAPE_HEADING_TOL_DEG = 15.0   # align to escape heading within this before driving

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

    Takes the mean direction vector of the first `n_segments` path segments
    (or fewer if the path is short), weighted by segment length.  This is more
    robust than using the single nearest waypoint because it accounts for the
    initial curvature of the path rather than reacting to a single 1-cell step.

    Returns the heading in radians, or None if the path has fewer than 2 points.
    """
    pts = [np.array(wp, dtype=float) for wp in waypoints]
    if len(pts) < 2:
        return None

    # Skip waypoints that are trivially close to the robot to avoid
    # a near-zero segment distorting the average.
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


def _escape_heading(map_obj, robot_xy, verbose=False):
    """
    Compute the best heading to drive straight out of the hard-block zone.

    Uses the grid EDT (same metric as A* and the escape trigger) so the probe
    and the trigger always agree on what counts as 'blocked' vs 'clear'.
    Probes 16 directions at ~22.5° spacing, scoring each by:
      1. How quickly it exits the PLANNING_CLEARANCE zone (fewer steps = better)
      2. How many additional steps of clear space exist beyond the exit point
         (more depth = leads further into open space, not just past a wall face)
    Hard-wall cells (1 or -1) terminate a probe immediately.
    """
    from . import PathGenerator as PG

    edt_grid = PG._get_edt(map_obj.Zone_Map)
    h, w     = map_obj.Zone_Map.shape
    rx, ry   = float(robot_xy[0]), float(robot_xy[1])

    candidates = []   # (exit_step, -clearance_depth, angle)

    for angle_deg in range(0, 360, 23):   # ~16 directions at 22.5° spacing
        angle = np.radians(angle_deg)
        dx    =  np.cos(angle)
        dy    = -np.sin(angle)   # robot coord convention (y flipped)

        exit_step       = None
        clearance_depth = 0

        for step in range(1, 41):   # up to 20m at 0.5m steps
            cx_i = int(np.clip(round(rx + dx * step * 0.5), 0, w - 1))
            cy_i = int(np.clip(round(ry + dy * step * 0.5), 0, h - 1))

            cell = map_obj.Zone_Map[cy_i, cx_i]
            if cell == 1 or cell == -1:
                break   # hard wall — no exit this way

            edt_val = float(edt_grid[cy_i, cx_i])
            if edt_val > PG.PLANNING_CLEARANCE:
                if exit_step is None:
                    exit_step = step
                clearance_depth += 1   # accumulate open steps beyond exit

        if exit_step is not None:
            candidates.append((exit_step, -clearance_depth, angle))

    if not candidates:
        if verbose:
            print("[MPC] _escape_heading: no exit direction found — all blocked")
        return None

    # Prefer fewest steps to exit; break ties by most clearance depth
    candidates.sort(key=lambda c: (c[0], c[1]))
    best_angle = candidates[0][2]

    if verbose:
        print(f"[MPC] _escape_heading: {len(candidates)} directions viable, "
              f"best={np.degrees(best_angle):.1f}° "
              f"(exit_step={candidates[0][0]}, depth={-candidates[0][1]})")

    return float(best_angle)

    if verbose:
        print(f"[MPC] _escape_heading: best={np.degrees(best_angle):.1f}° "
              f"(exit_step={candidates[0][0]}, "
              f"clearance={-candidates[0][1]}), "
              f"total candidates={len(candidates)}")

    return float(best_angle)

# =============================================================================
# Unified MPC solver
# =============================================================================
# Obstacle avoidance is fully delegated to A*.
# Speed limits are applied by patching lbx/ubx before each solve.
#
# Parameter vector: X0(3) | lookahead(2) | u_prev(2)   (total: 7)
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
    Drop all waypoints the robot has already passed, keeping at least one.

    A waypoint is considered passed when the robot is within PATH_END_DIST of
    it AND it is not the last waypoint (the goal).  This ensures `waypoints[-1]`
    always refers to the true destination, so `at_end` detection is reliable
    regardless of how many intermediate points the path contains.

    Mutates `waypoints` in-place and returns it.
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
    Project a point `dist` metres ahead along the (already-pruned) A* path.

    Because _advance_waypoints has already removed passed points, waypoints[0]
    is always the next relevant point and waypoints[-1] is the true goal.
    `at_end` is True when the robot is within PATH_END_DIST of the final waypoint.

    Returns
    -------
    (lookahead_xy, at_end)
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

        # Heading the robot should face at the end of the current path.
        # Set by Move_CMD_Receiver when a face_heading is provided; None means
        # no terminal orient is required (plain trajectory mode).
        self._face_heading = None

        self._pivot_target_hdg = 0.0
        # True while executing the post-arrival orient so we land in idle,
        # not back in trajectory.
        self._pivot_is_terminal_orient = False

        # Pivot stall detection: track error magnitude over the last window
        # of ticks so we can abort a spinning pivot that isn't converging.
        self._pivot_err_history = deque(maxlen=PIVOT_STALL_WINDOW)

        self._dist_history = deque(maxlen=STALL_WINDOW)
        self._solving      = False

        # Stall-replan state: track the in-flight task so Orient can cancel it,
        # and a cooldown flag to prevent back-to-back replan storms.
        self._stall_replan_task     = None
        self._stall_replan_cooldown = False

        # Obstacle escape state: heading is locked on first entry and held
        # until the robot fully clears the obstacle, preventing oscillation
        # caused by recomputing a slightly-different heading every tick.
        self._escape_hdg_locked = None

        self._guess  = None
        self._u_prev = np.zeros(2)

        self._v_max = V_MAX_DEFAULT
        self._v_min = V_MIN_DEFAULT

        print(f"[MPC] Building unified solver (N={N})...")
        self._mpc = build_solver(N)
        print(f"[MPC] Solver ready. n_dec={self._mpc['n_dec']}")

    # ── Speed cap ─────────────────────────────────────────────────────────────

    def set_speed_cap(self, cap: float):
        """
        Symmetrically clamp wheel speeds to [-cap, +cap].
        Patches lbx/ubx in-place — effective on the very next solve.
        """
        cap = float(cap)
        cap = max(0.0, min(cap, V_MAX_DEFAULT))
        self._v_max =  cap
        self._v_min = -cap
        cs = self._mpc['ctrl_start']
        self._mpc['lbx'][cs:] = self._v_min
        self._mpc['ubx'][cs:] = self._v_max
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
        # Cancel any in-flight stall replan so it doesn't overwrite the orient.
        if self._stall_replan_task is not None and not self._stall_replan_task.done():
            self._stall_replan_task.cancel()
            self._stall_replan_task = None

        # Clear all move state so the MPC doesn't resume trajectory after orient.
        self._waypoints             = []
        self._target                = None
        self._face_heading          = None
        self._dist_history.clear()
        self._stall_replan_cooldown = False
        self._escape_hdg_locked     = None

        # Reset solver to avoid stale warm-start influencing the pivot.
        self._u_prev = np.zeros(2)
        self._guess  = None

        self._mode                     = "pivot"
        self._pivot_is_terminal_orient = False   # external orient — normal completion
        self._pivot_target_hdg         = wrap(float(data[2]))

        # Notify UI that the trajectory has been cancelled.
        await self._pub_clear()

        if self._verbose:
            print(f"[MPC] Orient interrupt — move cancelled, "
                  f"pivoting to {np.degrees(self._pivot_target_hdg):.1f} deg")

    async def Move_CMD_Receiver(self, waypoints, target_xy,
                                face_heading=None):
        """
        Begin tracking a new A* path.

        Parameters
        ----------
        waypoints    : list of (int x, int y)
        target_xy    : (x, y) — the original requested destination (may differ
                       from waypoints[-1] if the goal was snapped)
        face_heading : float (radians) or None — if provided, the robot will
                       pivot to this heading after arriving at path end.
                       Used when the caller wants the robot to face a point
                       that is behind an obstacle (e.g. a turkey outside the
                       driveable zone): A* routes to the nearest reachable cell
                       and the MPC then orients toward the original target.
        """
        if not waypoints:
            if self._verbose:
                print("[MPC] Move_CMD_Receiver: empty waypoints — ignored")
            return

        dest_changed = (
            self._target is None
            or np.linalg.norm(np.array(target_xy) - np.array(self._target)) > 3.0
        )

        self._waypoints    = waypoints
        self._target       = target_xy
        self._face_heading = face_heading
        self._guess        = None   # invalidate warm start on new plan

        # If escape is currently active, update the target and waypoints so the
        # robot resumes toward the right goal after clearing the obstacle, but
        # don't reset the escape heading or trigger a pre-move pivot — doing so
        # interrupts the escape and causes the spin/replan/re-enter cycle.
        if self._escape_hdg_locked is not None:
            if self._verbose:
                print("[MPC] Move received during escape — updating target/waypoints, "
                      "deferring pivot until obstacle cleared")
            if dest_changed:
                self._dist_history.clear()
            return

        # Clear stall cooldown and escape heading so the new path gets a
        # fresh detection window and clean escape state.
        self._stall_replan_cooldown = False
        self._escape_hdg_locked     = None

        if dest_changed:
            self._dist_history.clear()

        state = get_state(self._parent)
        rxy   = state[:2]

        # Compute pre-move heading from the average direction of the first
        # PIVOT_ALIGN_SEGMENTS path segments (weighted by length).
        # This prevents donuts caused by aligning to a single 1-cell step that
        # doesn't represent the path's true initial direction.
        hdg = _path_entry_heading(rxy, waypoints, n_segments=PIVOT_ALIGN_SEGMENTS)

        if hdg is None:
            # Degenerate path (single point) — go straight to trajectory mode.
            self._mode = "trajectory"
            await self._pub_reference(waypoints)
            return

        err = wrap(hdg - state[2])
        if abs(np.degrees(err)) > PIVOT_THRESHOLD_DEG:
            # If we're already mid-pivot toward a similar heading, don't
            # reset the target — that's the root cause of the spinning 360s
            # seen when repeated Move CMDs arrive during a pre-move pivot.
            already_pivoting_correctly = (
                self._mode == "pivot"
                and not self._pivot_is_terminal_orient
                and abs(np.degrees(wrap(hdg - self._pivot_target_hdg))) < PIVOT_THRESHOLD_DEG
            )
            if not already_pivoting_correctly:
                self._pivot_target_hdg         = hdg
                self._pivot_is_terminal_orient = False
                self._pivot_err_history.clear()   # fresh stall window for new target
                self._mode                     = "pivot"
                if self._verbose:
                    print(f"[MPC] Pre-move pivot engaged: {np.degrees(err):.1f} deg "
                          f"(averaged over first {PIVOT_ALIGN_SEGMENTS} segments)")
            elif self._verbose:
                print(f"[MPC] Already pivoting to similar heading — suppressing target update")
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
            state = get_state(self._parent)
            err   = wrap(self._pivot_target_hdg - state[2])
            err_deg = abs(np.degrees(err))

            # ── Pivot stall detection ─────────────────────────────────────────
            # If the heading error hasn't improved meaningfully over the last
            # PIVOT_STALL_WINDOW ticks, the pivot is stuck (drivetrain not
            # responding, or repeated Move CMDs keep shifting the target).
            # Abort: clear the error history and jump to trajectory so the
            # robot keeps moving rather than spinning indefinitely.
            self._pivot_err_history.append(err_deg)
            pivot_stalled = (
                len(self._pivot_err_history) == PIVOT_STALL_WINDOW
                and (self._pivot_err_history[0] - err_deg) < PIVOT_STALL_DEG
            )
            if pivot_stalled:
                if self._verbose:
                    print(f"[MPC] Pivot stall detected (err={err_deg:.1f} deg, "
                          f"no improvement over {PIVOT_STALL_WINDOW} ticks) — "
                          f"aborting pivot, resuming trajectory")
                self._pivot_err_history.clear()
                self._u_prev = np.zeros(2)
                self._guess  = None
                # Jump straight to trajectory if we have waypoints, else idle.
                self._mode = "trajectory" if self._waypoints else "idle"
                if self._mode == "idle":
                    await self._send(np.zeros(2))
                    await self._pub_clear()
                return

            if err_deg < PIVOT_ALIGN_DEG:
                if self._pivot_is_terminal_orient:
                    # Post-arrival orient complete — go fully idle.
                    self._mode                     = "idle"
                    self._pivot_is_terminal_orient = False
                    self._face_heading             = None
                    self._dist_history.clear()
                    await self._send(np.zeros(2))
                    await self._pub_clear()
                    if self._verbose:
                        print("[MPC] Terminal orient complete — idle.")
                else:
                    # Pre-move align complete — begin tracking.
                    self._mode = "trajectory"
                    if self._verbose:
                        print("[MPC] Pre-move pivot complete — trajectory.")
                self._pivot_err_history.clear()
                self._u_prev = np.zeros(2)
                self._guess  = None

                if logger:
                    logger.log(state=state, mode="idle",
                        waypoint=tuple(self._target) if self._target is not None else (0, 0),
                        turkey_pos=None,
                        edt_distance=self._parent.Map.grid_edt_at(*state[:2]),
                        vl=0.0, vr=0.0,
                        n_dynamic=len(self._parent._dynamic_obs),
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=len(self._waypoints))
            else:
                # Pivot still running — apply proportional controller.
                omega = PIVOT_KP * err
                u = np.array([np.clip(-omega * L / 2, self._v_min, self._v_max),
                               np.clip( omega * L / 2, self._v_min, self._v_max)])
                await self._send(u)
                if logger:
                    logger.log(state=state, mode="pivot",
                        waypoint=tuple(self._target) if self._target is not None else (0, 0),
                        turkey_pos=None,
                        edt_distance=self._parent.Map.grid_edt_at(*state[:2]),
                        vl=float(u[0]), vr=float(u[1]),
                        n_dynamic=len(self._parent._dynamic_obs),
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=len(self._waypoints))
            return

        if self._mode != "trajectory":
            return

        state = get_state(self._parent)
        rxy   = state[:2]
        edt   = self._parent.Map.grid_edt_at(*rxy)

        if self._target is None:
            self._mode = "idle"; return

        # ── Obstacle escape ───────────────────────────────────────────────────
        # Use the grid EDT (same source as A* and _is_blocked) as the trigger.
        # min_rect_dist can return 0 inside freely navigable cells if OBSTACLE_RECTS
        # doesn't match the Zone_Map grid — causing endless spurious escapes in
        # open space. The grid EDT is the single authoritative obstacle source.
        from . import PathGenerator as PG
        gz  = self._parent.Map.Zone_Map
        giy = int(np.clip(round(rxy[1]), 0, gz.shape[0] - 1))
        gix = int(np.clip(round(rxy[0]), 0, gz.shape[1] - 1))
        grid_edt     = float(PG._get_edt(gz)[giy, gix])
        in_obstacle  = grid_edt <= PG.PLANNING_CLEARANCE

        if in_obstacle:
            # Lock the escape heading on first entry so we don't recompute a
            # slightly-different angle every tick and perpetually re-align.
            if self._escape_hdg_locked is None:
                self._escape_hdg_locked = _escape_heading(
                    self._parent.Map, rxy, verbose=self._verbose)
                if self._verbose:
                    if self._escape_hdg_locked is not None:
                        print(f"[MPC] Obstacle escape: grid_edt={grid_edt:.2f}m <= "
                              f"{PG.PLANNING_CLEARANCE}m — locking escape heading "
                              f"{np.degrees(self._escape_hdg_locked):.1f} deg")
                    else:
                        print(f"[MPC] Obstacle escape: grid_edt={grid_edt:.2f}m — "
                              "all directions blocked")

            esc_hdg = self._escape_hdg_locked
            u_esc = np.zeros(2)

            if esc_hdg is None:
                # Fully surrounded — stop and wait for the map to update.
                await self._send(np.zeros(2))
                if self._verbose:
                    print("[MPC] Obstacle escape: all directions blocked — stopped")
            else:
                hdg_err = wrap(esc_hdg - state[2])
                if abs(np.degrees(hdg_err)) > ESCAPE_HEADING_TOL_DEG:
                    # Align to escape heading first.
                    omega = PIVOT_KP * hdg_err
                    u_esc = np.array([np.clip(-omega * L / 2, self._v_min, self._v_max),
                                      np.clip( omega * L / 2, self._v_min, self._v_max)])
                    if self._verbose:
                        print(f"[MPC] Obstacle escape: aligning to {np.degrees(esc_hdg):.1f} deg "
                              f"(err={np.degrees(hdg_err):.1f} deg)")
                else:
                    # Heading close enough — drive forward, no differential.
                    spd = min(ESCAPE_SPEED, self._v_max)
                    u_esc = np.array([spd, spd])
                    if self._verbose:
                        print(f"[MPC] Obstacle escape: driving forward at {spd:.2f} m/s")
                await self._send(u_esc)

            if logger:
                logger.log(state=state, mode="escape",
                    waypoint=tuple(self._target) if self._target is not None else (0, 0),
                    turkey_pos=None,
                    edt_distance=edt, vl=float(u_esc[0]), vr=float(u_esc[1]),
                    n_dynamic=len(self._parent._dynamic_obs),
                    solve_time_ms=0.0, warm_start=False,
                    replan=False, n_waypoints=len(self._waypoints))
            return
        else:
            # Robot is clear of the hard-block zone — release escape state.
            if self._escape_hdg_locked is not None:
                if self._verbose:
                    print("[MPC] Obstacle escape: cleared — replanning from recovery position")
                self._escape_hdg_locked     = None
                self._stall_replan_cooldown = False
                # Replan from current position so A* routes cleanly away from
                # the obstacle rather than resuming the old path that drove into it.
                # Only fire if we have a valid target and are not already replanning.
                if (self._target is not None
                        and (self._stall_replan_task is None
                             or self._stall_replan_task.done())):
                    self._stall_replan_task = asyncio.create_task(
                        self._parent.replan_to(self._target))

        # Guard: empty waypoints → idle
        if not self._waypoints:
            if self._verbose:
                print("[MPC] trajectory with empty waypoints — going idle")
            self._mode = "idle"
            await self._send(np.zeros(2))
            await self._pub_clear()
            return

        # ── Prune passed waypoints ────────────────────────────────────────────
        # Remove waypoints the robot has already reached from the front of the
        # list so that pts[-1] always refers to the true goal and at_end
        # detection is reliable on long paths.
        self._waypoints = _advance_waypoints(rxy, self._waypoints)

        # ── Lookahead and end-of-path detection ───────────────────────────────
        lxy, at_end = _lookahead_point(rxy, self._waypoints)

        # ── Arrival ───────────────────────────────────────────────────────────
        if at_end:
            await self._send(np.zeros(2))

            if self._face_heading is not None:
                # Terminal orient requested (e.g. face a turkey outside zone).
                # Run exactly once, then idle — no replan loop possible.
                if self._verbose:
                    print(f"[MPC] Path end — terminal orient to "
                          f"{np.degrees(self._face_heading):.1f} deg")
                self._pivot_target_hdg         = self._face_heading
                self._pivot_is_terminal_orient = True
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
                    turkey_pos=None,
                    edt_distance=edt, vl=0.0, vr=0.0,
                    n_dynamic=len(self._parent._dynamic_obs),
                    solve_time_ms=0.0, warm_start=False,
                    replan=False, n_waypoints=len(self._waypoints))
            return

        # ── Stall detection ───────────────────────────────────────────────────
        dist_to_end = np.linalg.norm(rxy - np.array(self._waypoints[-1], dtype=float))
        self._dist_history.append(dist_to_end)
        stalled = (
            not self._stall_replan_cooldown
            and len(self._dist_history) == STALL_WINDOW
            and (self._dist_history[0] - dist_to_end) < STALL_DELTA
        )

        if stalled:
            if self._verbose:
                print(f"[MPC] Stall detected (dist_to_end={dist_to_end:.2f}m) — replanning")
            self._dist_history.clear()
            self._guess             = None
            self._stall_replan_cooldown = True   # suppress further replans until resolved
            self._stall_replan_task = asyncio.create_task(
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
                turkey_pos=None,
                edt_distance=edt, vl=float(u[0]), vr=float(u[1]),
                n_dynamic=len(self._parent._dynamic_obs),
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