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

PIVOT_THRESHOLD_DEG = 10.0
PIVOT_ALIGN_DEG     =  3.0
PIVOT_KP            =  3.0

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

def _lookahead_point(robot_xy, robot_theta, waypoints, dist=LOOKAHEAD_DIST):
    """
    Project a point dist metres ahead along the A* path.

    Forward-skip is softened: instead of skipping all waypoints with a
    negative dot product (which could skip past imminent waypoints after a
    sharp turn), we only skip waypoints that are both behind the robot AND
    further from the path end than the closest forward waypoint.  In practice
    this means we walk from the beginning of the remaining path and take the
    first waypoint that is either ahead or the closest one behind.

    Returns
    -------
    (lookahead_xy, at_end)
      lookahead_xy : np.array([x, y])
      at_end       : True when the robot is within PATH_END_DIST of the
                     final waypoint.  False for empty waypoints (caller guards).
    """
    if not waypoints:
        return np.array(robot_xy), False

    pts    = [np.array(wp, dtype=float) for wp in waypoints]
    at_end = np.linalg.norm(np.array(robot_xy) - pts[-1]) < PATH_END_DIST

    fwd = np.array([np.cos(robot_theta), -np.sin(robot_theta)])

    # Find the first waypoint that is strictly ahead of the robot.
    # If none are ahead, use the nearest one to avoid overshooting corners.
    start = len(pts) - 1
    min_dist = np.inf
    for i, pt in enumerate(pts):
        if np.dot(fwd, pt - robot_xy) >= -0.5:   # softened: was >= 0
            start = i
            break
        d = np.linalg.norm(pt - robot_xy)
        if d < min_dist:
            min_dist = d
            start    = i

    remaining = dist
    for i in range(start, len(pts) - 1):
        seg = pts[i + 1] - pts[i]; sl = np.linalg.norm(seg)
        if sl < 1e-6: continue
        if sl >= remaining: return pts[i] + seg * (remaining / sl), at_end
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

        self._dist_history = deque(maxlen=STALL_WINDOW)
        self._solving      = False

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
        self._mode                    = "pivot"
        self._pivot_is_terminal_orient = False   # external orient — normal completion
        self._pivot_target_hdg        = wrap(float(data[2]))

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

        if dest_changed:
            self._dist_history.clear()

        state = get_state(self._parent)
        rxy   = state[:2]
        fwd   = np.array([np.cos(state[2]), -np.sin(state[2])])

        # Pre-move pivot if robot is badly misaligned with the first forward waypoint
        pivot_wp = next(
            (np.array(wp, dtype=float) for wp in waypoints
             if np.dot(fwd, np.array(wp, dtype=float) - rxy) > 0),
            np.array(waypoints[-1], dtype=float)
        )
        dx, dy = pivot_wp[0] - rxy[0], pivot_wp[1] - rxy[1]
        hdg    = np.arctan2(-dy, dx)
        err    = wrap(hdg - state[2])
        if abs(np.degrees(err)) > PIVOT_THRESHOLD_DEG:
            self._pivot_target_hdg         = hdg
            self._pivot_is_terminal_orient = False
            self._mode                     = "pivot"
            if self._verbose:
                print(f"[MPC] Pre-move pivot engaged: {np.degrees(err):.1f} deg")
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

            if abs(np.degrees(err)) < PIVOT_ALIGN_DEG:
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
                self._u_prev = np.zeros(2)
                self._guess  = None

                if logger:
                    logger.log(state=state, mode="idle",
                        waypoint=tuple(self._target) if self._target is not None else (0, 0),
                        turkey_pos=None,
                        edt_distance=self._parent.Map.min_rect_dist(*state[:2]),
                        vl=0.0, vr=0.0,
                        n_dynamic=len(self._parent._dynamic_obs),
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=len(self._waypoints))
            else:
                omega = PIVOT_KP * err
                u = np.array([np.clip(-omega * L / 2, self._v_min, self._v_max),
                               np.clip( omega * L / 2, self._v_min, self._v_max)])
                await self._send(u)
                if logger:
                    logger.log(state=state, mode="pivot",
                        waypoint=tuple(self._target) if self._target is not None else (0, 0),
                        turkey_pos=None,
                        edt_distance=self._parent.Map.min_rect_dist(*state[:2]),
                        vl=float(u[0]), vr=float(u[1]),
                        n_dynamic=len(self._parent._dynamic_obs),
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=len(self._waypoints))
            return

        if self._mode != "trajectory":
            return

        state = get_state(self._parent)
        rxy   = state[:2]
        edt   = self._parent.Map.min_rect_dist(*rxy)

        if self._target is None:
            self._mode = "idle"; return

        # Guard: empty waypoints → idle
        if not self._waypoints:
            if self._verbose:
                print("[MPC] trajectory with empty waypoints — going idle")
            self._mode = "idle"
            await self._send(np.zeros(2))
            await self._pub_clear()
            return

        # ── Lookahead and end-of-path detection ───────────────────────────────
        lxy, at_end = _lookahead_point(rxy, state[2], self._waypoints)

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
        stalled = (len(self._dist_history) == STALL_WINDOW
                   and (self._dist_history[0] - dist_to_end) < STALL_DELTA)

        if stalled:
            if self._verbose:
                print(f"[MPC] Stall detected (dist_to_end={dist_to_end:.2f}m) — replanning")
            self._dist_history.clear()
            self._guess = None
            asyncio.create_task(self._parent.replan_to(self._target))

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