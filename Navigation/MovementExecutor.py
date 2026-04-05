import asyncio
import time
import numpy as np
import sys
sys.path.insert(0, '/home/turkeysquad/ME450/MainDesign/casadi')
if 'casadi' in sys.modules:
    del sys.modules['casadi']

import casadi as ca

# =============================================================================
# Config
# =============================================================================

WHEELBASE    = 0.7
L            = WHEELBASE
DT           = 0.1
N            = 15           # MPC horizon — reduce to 10 if Pi 5 is slow

V_MAX        =  3.0
V_MIN        = -3.0

FINAL_DIST   = 0.8          # m — final waypoint arrival threshold
STALL_TICKS  = 15           # ticks without 5cm improvement → declare arrival

# Pivot
PIVOT_THRESHOLD_DEG = 45.0
PIVOT_ALIGN_DEG     =  8.0
PIVOT_KP            =  3.0

# Cost weights
W_XY_RUN     =  5.0
W_XY_TERM    = 80.0
W_FOV        =  8.0   # chase mode: heading error to turkey
W_HEADING    = 12.0   # scan mode:  heading error to waypoint
W_OMEGA      =  1.0
W_INPUT_DIFF =  0.5
W_SLACK      = 50.0   # unified slack penalty (fixed + dynamic obstacles)

# Obstacle safety
R_HARD    = 0.8    # m — safety margin for all obstacles (fixed + dynamic)
CBF_GAMMA = 0.6    # CBF decay rate ∈ (0,1]
R_DYN     = 1.5    # m — safety radius around dynamic obstacles


# =============================================================================
# Helpers
# =============================================================================

def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def get_state(parent) -> np.ndarray:
    raw = np.array(parent.Current_State, dtype=np.float64)
    return np.array([raw[0], raw[1], wrap(raw[2])])


def heading_to_waypoint(robot_xy, wp):
    dx = wp[0] - robot_xy[0]
    dy = wp[1] - robot_xy[1]
    return np.arctan2(-dy, dx)   # y-down CCW-positive convention


# =============================================================================
# Rectangle CBF distance (CasADi symbolic)
# =============================================================================

def rect_dist_sym(px, py, x0, x1, y0, y1):
    """
    Algebraic distance from symbolic point (px, py) to axis-aligned rectangle
    [x0,x1] x [y0,y1].  Returns a CasADi expression.

    Outside the rectangle: positive Euclidean distance to nearest edge.
    On the boundary: 0.
    Inside: technically 0 for fmax formulation (we never plan inside).

    Regularised with 1e-6 inside sqrt so gradients are finite everywhere.
    """
    dx = ca.fmax(float(x0) - px, ca.fmax(0.0, px - float(x1)))
    dy = ca.fmax(float(y0) - py, ca.fmax(0.0, py - float(y1)))
    return ca.sqrt(dx**2 + dy**2 + 1e-6)


# =============================================================================
# MPC solver builder
# =============================================================================

def build_mpc_solver(obstacle_rects, max_dynamic_obstacles: int = 5):
    """
    Build the CasADi / IPOPT NLP.

    Fixed obstacles  — rectangle CBF with slack:
        dist(x_k, rect_r) - R_HARD + S_rect[r] >= 0   for each rect r, step k
        cost += W_SLACK * S_rect[r]^2

        One slack per rectangle (shared across horizon steps) — tight enough
        for safety, minimal extra variables (8 slacks total).

    Dynamic obstacles — standard CBF with slack:
        h(x_{k+1}) - (1-gamma)*h(x_k) + S_dyn[m] >= 0
        cost += W_SLACK * S_dyn[m]^2

    Heading:
        Scan  (chase=0): W_HEADING * err_to_waypoint^2
        Chase (chase=1): W_FOV    * err_to_turkey^2

    Decision variables: X_mpc, U_mpc, S_rect, S_dyn
    """
    n_s, n_u = 3, 2
    R  = len(obstacle_rects)    # number of fixed obstacle rectangles
    M  = max_dynamic_obstacles

    # Kinematics: y-down CCW-positive
    x_s, y_s, th_s = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('th')
    vl_s, vr_s     = ca.SX.sym('vl'), ca.SX.sym('vr')
    v_s     = (vl_s + vr_s) / 2
    omega_s = (vr_s - vl_s) / L
    f_disc  = ca.Function('f',
        [ca.vertcat(x_s, y_s, th_s), ca.vertcat(vl_s, vr_s)],
        [ca.vertcat(x_s, y_s, th_s) + DT * ca.vertcat(
            v_s * ca.cos(th_s),
            -v_s * ca.sin(th_s),
            omega_s
        )]
    )

    # Decision variables
    X_mpc   = ca.SX.sym('X',      n_s, N + 1)
    U_mpc   = ca.SX.sym('U',      n_u, N)
    S_rect  = ca.SX.sym('S_rect', R)      # one slack per fixed obstacle rect
    S_dyn   = ca.SX.sym('S_dyn',  M)      # one slack per dynamic obstacle

    # Parameters
    # Layout: X0(3), wp(2), turkey(2), chase(1), dyn_obs(2M), h0(M), uprev(2)
    P_X0    = ca.SX.sym('X0',    n_s)
    P_wp    = ca.SX.sym('wp',    2)
    P_turk  = ca.SX.sym('turk',  2)
    P_chase = ca.SX.sym('chase', 1)
    P_dyn   = ca.SX.sym('dyn',   2 * M)
    P_h0    = ca.SX.sym('h0',    M)
    P_uprev = ca.SX.sym('uprev', n_u)
    p_all   = ca.vertcat(P_X0, P_wp, P_turk, P_chase, P_dyn, P_h0, P_uprev)

    # Cost and equality constraints
    obj  = 0
    g_eq = [X_mpc[:, 0] - P_X0]

    # Slack costs — applied once (not per horizon step, slacks are shared)
    for r in range(R):
        obj += W_SLACK * S_rect[r]**2
    for m in range(M):
        obj += W_SLACK * S_dyn[m]**2

    for k in range(N):
        xk  = X_mpc[0, k]
        yk  = X_mpc[1, k]
        thk = X_mpc[2, k]
        vlk = U_mpc[0, k]
        vrk = U_mpc[1, k]

        # XY running cost
        obj += W_XY_RUN * ((xk - P_wp[0])**2 + (yk - P_wp[1])**2)

        # Scan heading (chase=0)
        wp_dx  = P_wp[0]   - xk
        wp_dy  = P_wp[1]   - yk
        wp_reg = ca.sqrt(wp_dx**2 + wp_dy**2 + 1e-6)
        ang_wp = ca.atan2(-wp_dy / wp_reg, wp_dx / wp_reg)
        err_wp = ca.atan2(ca.sin(ang_wp - thk), ca.cos(ang_wp - thk))
        obj   += (1 - P_chase) * W_HEADING * err_wp**2

        # Chase heading (chase=1)
        tk_dx  = P_turk[0] - xk
        tk_dy  = P_turk[1] - yk
        tk_reg = ca.sqrt(tk_dx**2 + tk_dy**2 + 1e-6)
        ang_t  = ca.atan2(-tk_dy / tk_reg, tk_dx / tk_reg)
        err_fov = ca.atan2(ca.sin(ang_t - thk), ca.cos(ang_t - thk))
        obj    += P_chase * W_FOV * err_fov**2

        # Turn-rate damping
        obj += W_OMEGA * ((vrk - vlk) / L)**2

        # Input smoothness
        u_prev_k = P_uprev if k == 0 else U_mpc[:, k - 1]
        obj     += W_INPUT_DIFF * ca.sumsqr(U_mpc[:, k] - u_prev_k)

        # Dynamics
        g_eq.append(X_mpc[:, k + 1] - f_disc(X_mpc[:, k], U_mpc[:, k]))

    # Terminal XY cost
    obj += W_XY_TERM * (
        (X_mpc[0, N] - P_wp[0])**2 + (X_mpc[1, N] - P_wp[1])**2
    )

    # Inequality constraints
    g_ineq = []

    # Fixed obstacle rectangle CBFs
    # dist(x_k, rect_r) - R_HARD + S_rect[r] >= 0
    # One slack per rectangle shared across all horizon steps — the slack
    # value must satisfy the tightest step, so it's conservative but lean.
    for r, (x0, x1, y0, y1) in enumerate(obstacle_rects):
        for k in range(N):
            xk = X_mpc[0, k]
            yk = X_mpc[1, k]
            d  = rect_dist_sym(xk, yk, x0, x1, y0, y1)
            g_ineq.append(d - R_HARD + S_rect[r])

    # Dynamic obstacle CBFs
    # h(x_{k+1}) - (1-gamma)*h(x_k) + S_dyn[m] >= 0
    for m in range(M):
        ox = P_dyn[2 * m]
        oy = P_dyn[2 * m + 1]
        for k in range(N):
            xk  = X_mpc[0, k];   yk  = X_mpc[1, k]
            xk1 = X_mpc[0, k+1]; yk1 = X_mpc[1, k+1]
            hk  = (xk  - ox)**2 + (yk  - oy)**2 - R_DYN**2
            hk1 = (xk1 - ox)**2 + (yk1 - oy)**2 - R_DYN**2
            g_ineq.append(hk1 - (1 - CBF_GAMMA) * hk + S_dyn[m])

    # Pack NLP
    opt_vars = ca.vertcat(
        ca.reshape(X_mpc,  -1, 1),
        ca.reshape(U_mpc,  -1, 1),
        S_rect,
        S_dyn,
    )
    n_dec = opt_vars.numel()

    ctrl_start       = n_s * (N + 1)
    slack_rect_start = ctrl_start + n_u * N
    slack_dyn_start  = slack_rect_start + R

    lbx = -np.inf * np.ones(n_dec)
    ubx =  np.inf * np.ones(n_dec)
    lbx[ctrl_start:slack_rect_start] = V_MIN
    ubx[ctrl_start:slack_rect_start] = V_MAX
    lbx[slack_rect_start:]           = 0.0   # all slacks >= 0

    g_eq_cat   = ca.vertcat(*g_eq)
    g_ineq_cat = ca.vertcat(*g_ineq)
    g_all      = ca.vertcat(g_eq_cat, g_ineq_cat)

    n_eq   = g_eq_cat.numel()
    n_ineq = g_ineq_cat.numel()
    lbg    = np.concatenate([np.zeros(n_eq), np.zeros(n_ineq)])
    ubg    = np.concatenate([np.zeros(n_eq), np.inf * np.ones(n_ineq)])

    solver = ca.nlpsol('solver', 'ipopt',
        {'f': obj, 'x': opt_vars, 'g': g_all, 'p': p_all},
        {
            'ipopt.print_level':           0,
            'print_time':                  0,
            'ipopt.tol':                   1e-3,
            'ipopt.max_iter':              80,
            'ipopt.warm_start_init_point': 'yes',
        }
    )

    return dict(
        solver           = solver,
        lbx              = lbx,  ubx              = ubx,
        lbg              = lbg,  ubg              = ubg,
        n_dec            = n_dec,
        ctrl_start       = ctrl_start,
        slack_rect_start = slack_rect_start,
        slack_dyn_start  = slack_dyn_start,
        n_s=n_s, n_u=n_u, R=R, M=M,
        n_p=p_all.numel(),
    )


# =============================================================================
# Parameter vector
# =============================================================================

def build_param_vector(state, waypoint, turkey_xy, chase_mode,
                       dynamic_obstacles, M):
    dyn_padded = list(dynamic_obstacles) + [(-999.0, -999.0)] * (M - len(dynamic_obstacles))
    dyn_flat   = np.array([(ox, oy) for ox, oy in dyn_padded]).flatten()
    rx, ry     = state[0], state[1]
    h0_vals    = np.array([(rx-ox)**2 + (ry-oy)**2 - R_DYN**2
                           for ox, oy in dyn_padded])
    turkey     = np.array(turkey_xy, dtype=float) if turkey_xy is not None else np.zeros(2)
    return np.concatenate([
        state,
        np.array(waypoint, dtype=float),
        turkey,
        [1.0 if chase_mode else 0.0],
        dyn_flat, h0_vals,
        np.zeros(2),    # u_prev placeholder
    ])


# =============================================================================
# Warm-start helpers
# =============================================================================

def _make_state_seeded_guess(state, n_dec, n_s, ctrl_start):
    x0 = np.zeros(n_dec)
    for k in range(N + 1):
        x0[k * n_s : k * n_s + n_s] = state
    return x0


def _shift_warm_start(sol_flat, n_s, n_u, ctrl_start,
                      slack_rect_start, slack_dyn_start, n_dec):
    X_flat  = sol_flat[:ctrl_start].reshape(n_s, N + 1, order='F')
    U_flat  = sol_flat[ctrl_start:slack_rect_start].reshape(n_u, N, order='F')
    X_shift = np.hstack([X_flat[:, 1:], X_flat[:, -1:]])
    U_shift = np.hstack([U_flat[:, 1:], U_flat[:, -1:]])
    x0      = np.zeros(n_dec)
    x0[:ctrl_start]                 = X_shift.flatten(order='F')
    x0[ctrl_start:slack_rect_start] = U_shift.flatten(order='F')
    x0[slack_rect_start:]           = sol_flat[slack_rect_start:]  # copy slacks
    return x0


# =============================================================================
# Controller
# =============================================================================

class MPC_loop:

    def __init__(self, parent, Drivetrain_pub, MPCFeedback_Navigation_pub):
        self.Drivetrain_pub  = Drivetrain_pub
        self.MPCFeedback_pub = MPCFeedback_Navigation_pub
        self._parent         = parent

        self._mode            = "idle"

        self._waypoints       = []
        self._wp_idx          = 0

        self._turkey_pos      = None
        self._chase_mode      = False
        self._dynamic_obs     = []

        self._pivot_target_hdg    = 0.0
        self._pivot_threshold_deg = PIVOT_THRESHOLD_DEG

        self._last_dist   = np.inf
        self._stall_count = 0

        self._init_guess  = None
        self._u_prev      = np.zeros(2)

        print("[MPC] Building solver...")
        rects     = parent.Map.Obstacle_Rects
        self._mpc = build_mpc_solver(rects, max_dynamic_obstacles=5)
        print(f"[MPC] Solver ready. {len(rects)} fixed obstacle rects, "
              f"n_dec={self._mpc['n_dec']}")

    # ── Public setters ────────────────────────────────────────────────────────

    def set_turkey(self, tx, ty):
        self._turkey_pos = (float(tx), float(ty))
        self._chase_mode = True

    def clear_turkey(self):
        self._turkey_pos = None
        self._chase_mode = False

    def set_dynamic_obstacles(self, obs_list):
        self._dynamic_obs = obs_list

    def set_chase_pivot_mode(self, chasing: bool):
        self._pivot_threshold_deg = 120.0 if chasing else PIVOT_THRESHOLD_DEG

    # ── Protocol 6 ───────────────────────────────────────────────────────────

    async def _proto_6_1_reference_trajectory(self, waypoints):
        await self.MPCFeedback_pub.publish({
            "Id": "Reference",
            "Cmd": [[float(x), float(y)] for x, y in waypoints]
        })

    async def _proto_6_3_clear_trajectory(self):
        await self.MPCFeedback_pub.publish({"Id": "ClearTrajectory"})

    # ── Command receivers ─────────────────────────────────────────────────────

    async def Orient_CMD_Receiver(self, data):
        self._mode = "orient"
        target_hdg = wrap(float(data[2]))
        while True:
            state = get_state(self._parent)
            err   = wrap(target_hdg - state[2])
            if abs(err) < np.radians(PIVOT_ALIGN_DEG):
                await self._send(np.zeros(2))
                self._mode = "idle"
                return
            omega = PIVOT_KP * err
            await self._send(np.array([
                np.clip(-omega * L / 2, V_MIN, V_MAX),
                np.clip( omega * L / 2, V_MIN, V_MAX),
            ]))
            await asyncio.sleep(DT)

    async def Move_CMD_Receiver(self, new_waypoints):
        if not new_waypoints:
            return

        state    = get_state(self._parent)
        robot_xy = state[:2]

        dists    = [np.linalg.norm(robot_xy - np.array(wp)) for wp in new_waypoints]
        best_idx = int(np.argmin(dists))

        self._waypoints   = new_waypoints
        self._wp_idx      = best_idx
        self._last_dist   = np.inf
        self._stall_count = 0

        if self._chase_mode:
            # During chase, skip pivot — FOV cost handles heading
            self._mode = "trajectory"
        else:
            target_wp   = np.array(new_waypoints[best_idx], dtype=float)
            desired_hdg = heading_to_waypoint(robot_xy, target_wp)
            hdg_err     = wrap(desired_hdg - state[2])
            if abs(np.degrees(hdg_err)) > self._pivot_threshold_deg:
                self._pivot_target_hdg = desired_hdg
                self._mode             = "pivot"
                print(f"[MPC] Pivot engaged: hdg_err={np.degrees(hdg_err):.1f}deg")
            else:
                self._mode = "trajectory"

        await self._proto_6_1_reference_trajectory(new_waypoints)

    # ── Control loop ──────────────────────────────────────────────────────────

    async def MPC_Main(self, logger=None):

        # Pivot
        if self._mode == "pivot":
            state   = get_state(self._parent)
            hdg_err = wrap(self._pivot_target_hdg - state[2])
            if abs(np.degrees(hdg_err)) < PIVOT_ALIGN_DEG:
                print("[MPC] Pivot complete.")
                self._mode       = "trajectory"
                self._init_guess = None
                self._u_prev     = np.zeros(2)
            else:
                omega = PIVOT_KP * hdg_err
                u = np.array([
                    np.clip(-omega * L / 2, V_MIN, V_MAX),
                    np.clip( omega * L / 2, V_MIN, V_MAX),
                ])
                await self._send(u)
                if logger:
                    logger.log(
                        state=state, mode="pivot",
                        waypoint=self._waypoints[self._wp_idx] if self._waypoints else (0,0),
                        turkey_pos=self._turkey_pos,
                        edt_distance=self._parent.Map.min_rect_dist(*state[:2]),
                        vl=float(u[0]), vr=float(u[1]),
                        n_dynamic=len(self._dynamic_obs),
                        solve_time_ms=0.0, warm_start=False,
                        replan=False, n_waypoints=len(self._waypoints),
                    )
            return

        if self._mode != "trajectory":
            return
        if not self._waypoints:
            self._mode = "idle"
            return

        state      = get_state(self._parent)
        M          = self._mpc['M']
        current_wp = np.array(self._waypoints[self._wp_idx], dtype=float)
        is_final   = (self._wp_idx == len(self._waypoints) - 1)
        dist_to_wp = np.linalg.norm(state[:2] - current_wp)

        # Dot-product waypoint advance
        if not is_final:
            next_wp  = np.array(self._waypoints[self._wp_idx + 1], dtype=float)
            to_robot = state[:2] - current_wp
            to_next  = next_wp  - current_wp
            if dist_to_wp < 0.1 or np.dot(to_robot, to_next) > 0:
                self._wp_idx += 1
                current_wp    = np.array(self._waypoints[self._wp_idx], dtype=float)
                is_final      = (self._wp_idx == len(self._waypoints) - 1)
                dist_to_wp    = np.linalg.norm(state[:2] - current_wp)

                if not self._chase_mode:
                    desired_hdg = heading_to_waypoint(state[:2], current_wp)
                    hdg_err     = wrap(desired_hdg - state[2])
                    if abs(np.degrees(hdg_err)) > self._pivot_threshold_deg:
                        self._pivot_target_hdg = desired_hdg
                        self._mode             = "pivot"
                        self._init_guess       = None
                        self._u_prev           = np.zeros(2)
                        print(f"[MPC] Mid-path pivot: {np.degrees(hdg_err):.1f}deg")
                        return

        # Final waypoint arrival / stall
        if is_final:
            if dist_to_wp < FINAL_DIST:
                arrived = True
            elif abs(self._last_dist - dist_to_wp) < 0.05:
                self._stall_count += 1
                arrived = self._stall_count >= STALL_TICKS
            else:
                self._stall_count = 0
                arrived           = False
            self._last_dist = dist_to_wp
        else:
            arrived = False

        if arrived:
            print(f"[MPC] Arrived at {tuple(current_wp)} dist={dist_to_wp:.2f}m")
            self._mode        = "idle"
            self._stall_count = 0
            self._last_dist   = np.inf
            await self._send(np.zeros(2))
            await self._proto_6_3_clear_trajectory()
            if logger:
                logger.log(
                    state=state, mode="idle",
                    waypoint=tuple(current_wp),
                    turkey_pos=self._turkey_pos,
                    edt_distance=self._parent.Map.min_rect_dist(*state[:2]),
                    vl=0.0, vr=0.0,
                    n_dynamic=len(self._dynamic_obs),
                    solve_time_ms=0.0, warm_start=False,
                    replan=False, n_waypoints=len(self._waypoints),
                )
            return

        # Build parameter vector
        dyn_list = self._dynamic_obs[:M]
        p_vec    = build_param_vector(
            state=state, waypoint=current_wp,
            turkey_xy=self._turkey_pos, chase_mode=self._chase_mode,
            dynamic_obstacles=dyn_list, M=M,
        )
        p_vec[-2] = self._u_prev[0]
        p_vec[-1] = self._u_prev[1]

        # Initial guess
        warm = (self._init_guess is not None)
        if not warm:
            self._init_guess = _make_state_seeded_guess(
                state, self._mpc['n_dec'],
                self._mpc['n_s'], self._mpc['ctrl_start'],
            )

        # Solve
        t0  = time.perf_counter()
        sol = self._mpc['solver'](
            x0=self._init_guess,
            lbx=self._mpc['lbx'], ubx=self._mpc['ubx'],
            lbg=self._mpc['lbg'], ubg=self._mpc['ubg'],
            p=p_vec,
        )
        solve_ms = (time.perf_counter() - t0) * 1000.0

        sol_flat = sol['x'].full().flatten()
        u        = sol_flat[self._mpc['ctrl_start'] :
                            self._mpc['ctrl_start'] + self._mpc['n_u']]

        # Shift warm start
        self._init_guess = _shift_warm_start(
            sol_flat,
            self._mpc['n_s'],   self._mpc['n_u'],
            self._mpc['ctrl_start'],
            self._mpc['slack_rect_start'],
            self._mpc['slack_dyn_start'],
            self._mpc['n_dec'],
        )
        self._u_prev = u.copy()

        # Log
        if logger:
            fov_err = float('nan')
            if self._turkey_pos is not None:
                tx, ty  = self._turkey_pos
                ang     = np.arctan2(-(ty - state[1]), tx - state[0])
                fov_err = wrap(ang - state[2])
            logger.log(
                state=state, mode="trajectory",
                waypoint=tuple(current_wp),
                turkey_pos=self._turkey_pos,
                edt_distance=self._parent.Map.min_rect_dist(*state[:2]),
                vl=float(u[0]), vr=float(u[1]),
                n_dynamic=len(self._dynamic_obs),
                solve_time_ms=solve_ms, warm_start=warm,
                replan=False, n_waypoints=len(self._waypoints),
            )

        print(f"[MPC] wp={self._wp_idx}/{len(self._waypoints)-1} "
              f"dist={dist_to_wp:.2f}m "
              f"{'chase' if self._chase_mode else 'scan '} "
              f"state=({state[0]:.2f},{state[1]:.2f},{np.degrees(state[2]):.1f}deg) "
              f"u=[{u[0]:.2f},{u[1]:.2f}] "
              f"solve={solve_ms:.1f}ms warm={warm}")

        await self._send(u)

    async def _send(self, u):
        await self.Drivetrain_pub.publish({
            "Id": "System", "Cmd": np.array(u, dtype=np.float32).tolist()
        })

    async def DriveTrain_Sender(self, logger=None):
        while True:
            await asyncio.sleep(DT)
            await self.MPC_Main(logger=logger)