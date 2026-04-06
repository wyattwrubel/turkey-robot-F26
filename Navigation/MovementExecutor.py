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

WHEELBASE   = 0.7
L           = WHEELBASE
DT          = 0.1

V_MAX       =  3.0
V_MIN       = -3.0

FINAL_DIST        = 0.8   # m — final waypoint arrival threshold
STALL_TICKS       = 15    # ticks without 5cm improvement → declare arrival
LIGHT_SOLVER_DIST = 4.0   # m — switch to lightweight solver (final wp only)

N_FULL  = 15   # full solver horizon
N_LIGHT =  6   # lightweight solver horizon (final 4m approach)

# Pivot
PIVOT_THRESHOLD_DEG = 45.0
PIVOT_ALIGN_DEG     =  8.0
PIVOT_KP            =  3.0

# Cost weights
W_XY_RUN     =  5.0
W_XY_TERM    = 80.0
W_FOV        =  8.0
W_HEADING    = 12.0
W_OMEGA      =  1.0
W_INPUT_DIFF =  0.5
W_SLACK      = 50.0    # dynamic obstacle CBF slack penalty

# Grid-lookup obstacle penalties
# Applied per horizon step based on Zone_Map cell value at that position.
# Flags are pre-computed in NumPy from the warm-start guess each tick and
# passed as concrete parameters — no symbolic indexing required.
W_OBS_STATIC  = 900.0   # penalty per step landing on a static obstacle  (==1)
W_OBS_DYNAMIC = 400.0   # penalty per step landing on a dynamic obstacle (==2)
# (CBF already handles dynamic obstacles; this is a belt-and-suspenders cost.)

# Dynamic obstacle CBF
CBF_GAMMA = 0.6
R_DYN     = 1.5


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
    return np.arctan2(-dy, dx)


# =============================================================================
# Grid-lookup obstacle flags — computed in NumPy each tick
# =============================================================================

def compute_obs_flags(X_guess, zone_map, n_steps):
    """
    For each of the N horizon steps, round the (x, y) position from the
    warm-start trajectory to the nearest grid cell and look up Zone_Map.

    Returns
    -------
    static_flags  : (N,) float array — 1.0 where cell == 1, else 0.0
    dynamic_flags : (N,) float array — 1.0 where cell == 2, else 0.0

    Parameters
    ----------
    X_guess   : flat decision-variable array (from previous solution / seed)
    zone_map  : (H, W) int array  — Map.Zone_Map
    n_steps   : horizon length N
    """
    H, W = zone_map.shape
    n_s  = 3   # state dimension

    static_flags  = np.zeros(n_steps, dtype=np.float64)
    dynamic_flags = np.zeros(n_steps, dtype=np.float64)

    for k in range(n_steps):
        # States are packed column-major: X[:, k] starts at k*n_s
        xk = X_guess[k * n_s]
        yk = X_guess[k * n_s + 1]

        col = int(round(xk))
        row = int(round(yk))

        # Clamp to valid grid range
        col = int(np.clip(col, 0, W - 1))
        row = int(np.clip(row, 0, H - 1))

        cell = zone_map[row, col]
        if cell == 1:
            static_flags[k]  = 1.0
        elif cell == 2:
            dynamic_flags[k] = 1.0

    return static_flags, dynamic_flags


# =============================================================================
# MPC solver builder
# =============================================================================

def build_mpc_solver(n_steps: int, max_dynamic_obstacles: int = 5):
    """
    Build one CasADi / IPOPT NLP.

    Obstacle avoidance strategy:
      - Fixed obstacles: grid-lookup flags passed as parameters each tick.
        Per-step cost += W_OBS_STATIC  * p_static[k]
                       + W_OBS_DYNAMIC * p_dynamic[k]
        Flags are concrete floats (0 or 1) — no symbolic indexing.
      - Dynamic obstacles: CBF with one slack per obstacle (always active),
        plus the dynamic grid flag cost above as extra discouragement.

    Parameters
    ----------
    n_steps               : horizon length (N_FULL or N_LIGHT)
    max_dynamic_obstacles : CBF slots for dynamic obstacles
    """
    N   = n_steps
    n_s = 3
    n_u = 2
    M   = max_dynamic_obstacles

    # Kinematics
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
    X_mpc  = ca.SX.sym('X',     n_s, N + 1)
    U_mpc  = ca.SX.sym('U',     n_u, N)
    S_dyn  = ca.SX.sym('S_dyn', M)

    # Parameters layout:
    #   X0          (3)
    #   wp          (2)
    #   turkey      (2)
    #   chase       (1)
    #   dyn_obs     (2M)
    #   h0          (M)
    #   uprev       (2)
    #   static_flags  (N)   ← 1.0 if horizon step k is on a static obstacle cell
    #   dynamic_flags (N)   ← 1.0 if horizon step k is on a dynamic obstacle cell
    P_X0          = ca.SX.sym('X0',           n_s)
    P_wp          = ca.SX.sym('wp',           2)
    P_turk        = ca.SX.sym('turk',         2)
    P_chase       = ca.SX.sym('chase',        1)
    P_dyn         = ca.SX.sym('dyn',          2 * M)
    P_h0          = ca.SX.sym('h0',           M)
    P_uprev       = ca.SX.sym('uprev',        n_u)
    P_static_flags  = ca.SX.sym('s_flags',   N)   # concrete 0/1 per step
    P_dynamic_flags = ca.SX.sym('d_flags',   N)   # concrete 0/1 per step

    p_all = ca.vertcat(
        P_X0, P_wp, P_turk, P_chase,
        P_dyn, P_h0, P_uprev,
        P_static_flags, P_dynamic_flags,
    )

    # Cost and dynamics equality
    obj  = 0
    g_eq = [X_mpc[:, 0] - P_X0]

    # Dynamic obstacle slack cost
    for m in range(M):
        obj += W_SLACK * S_dyn[m]**2

    for k in range(N):
        xk  = X_mpc[0, k];  yk  = X_mpc[1, k];  thk = X_mpc[2, k]
        vlk = U_mpc[0, k];  vrk = U_mpc[1, k]

        # XY running cost
        obj += W_XY_RUN * ((xk - P_wp[0])**2 + (yk - P_wp[1])**2)

        # ── Grid-lookup obstacle penalty ──────────────────────────────────────
        # P_static_flags[k] and P_dynamic_flags[k] are concrete 0.0 or 1.0
        # values passed as parameters — no symbolic indexing required.
        obj += W_OBS_STATIC  * P_static_flags[k]
        obj += W_OBS_DYNAMIC * P_dynamic_flags[k]

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

        # Turn-rate + smoothness
        obj += W_OMEGA * ((vrk - vlk) / L)**2
        u_prev_k = P_uprev if k == 0 else U_mpc[:, k - 1]
        obj     += W_INPUT_DIFF * ca.sumsqr(U_mpc[:, k] - u_prev_k)

        # Dynamics equality
        g_eq.append(X_mpc[:, k + 1] - f_disc(X_mpc[:, k], U_mpc[:, k]))

    # Terminal XY cost
    obj += W_XY_TERM * (
        (X_mpc[0, N] - P_wp[0])**2 + (X_mpc[1, N] - P_wp[1])**2
    )

    # Dynamic obstacle CBF inequalities
    g_ineq = []
    for m in range(M):
        ox = P_dyn[2 * m];  oy = P_dyn[2 * m + 1]
        for k in range(N):
            xk  = X_mpc[0, k];   yk  = X_mpc[1, k]
            xk1 = X_mpc[0, k+1]; yk1 = X_mpc[1, k+1]
            hk  = (xk  - ox)**2 + (yk  - oy)**2 - R_DYN**2
            hk1 = (xk1 - ox)**2 + (yk1 - oy)**2 - R_DYN**2
            g_ineq.append(hk1 - (1 - CBF_GAMMA) * hk + S_dyn[m])

    # Pack NLP
    opt_vars = ca.vertcat(
        ca.reshape(X_mpc, -1, 1),
        ca.reshape(U_mpc, -1, 1),
        S_dyn,
    )
    n_dec       = opt_vars.numel()
    ctrl_start  = n_s * (N + 1)
    slack_start = ctrl_start + n_u * N

    lbx = -np.inf * np.ones(n_dec)
    ubx =  np.inf * np.ones(n_dec)
    lbx[ctrl_start:slack_start] = V_MIN
    ubx[ctrl_start:slack_start] = V_MAX
    lbx[slack_start:]           = 0.0

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
        solver      = solver,
        lbx         = lbx,   ubx         = ubx,
        lbg         = lbg,   ubg         = ubg,
        n_dec       = n_dec,
        ctrl_start  = ctrl_start,
        slack_start = slack_start,
        n_s         = n_s,
        n_u         = n_u,
        N           = N,
        M           = M,
        n_p         = p_all.numel(),
    )


# =============================================================================
# Parameter vector
# =============================================================================

def build_param_vector(state, waypoint, turkey_xy, chase_mode,
                       dynamic_obstacles, M,
                       static_flags, dynamic_flags):
    """
    Pack all MPC parameters into a flat numpy vector.

    static_flags  : (N,) float array — 1.0 where step k hits a static obstacle
    dynamic_flags : (N,) float array — 1.0 where step k hits a dynamic obstacle
    """
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
        np.zeros(2),                          # u_prev placeholder
        static_flags.astype(np.float64),      # N values
        dynamic_flags.astype(np.float64),     # N values
    ])


# =============================================================================
# Warm-start helpers
# =============================================================================

def _make_state_seeded_guess(state, mpc_dict):
    n_dec = mpc_dict['n_dec']
    n_s   = mpc_dict['n_s']
    N     = mpc_dict['N']
    x0    = np.zeros(n_dec)
    for k in range(N + 1):
        x0[k * n_s : k * n_s + n_s] = state
    return x0


def _shift_warm_start(sol_flat, mpc_dict):
    n_s         = mpc_dict['n_s']
    n_u         = mpc_dict['n_u']
    N           = mpc_dict['N']
    ctrl_start  = mpc_dict['ctrl_start']
    slack_start = mpc_dict['slack_start']
    n_dec       = mpc_dict['n_dec']

    X_flat  = sol_flat[:ctrl_start].reshape(n_s, N + 1, order='F')
    U_flat  = sol_flat[ctrl_start:slack_start].reshape(n_u, N, order='F')
    X_shift = np.hstack([X_flat[:, 1:], X_flat[:, -1:]])
    U_shift = np.hstack([U_flat[:, 1:], U_flat[:, -1:]])

    x0 = np.zeros(n_dec)
    x0[:ctrl_start]            = X_shift.flatten(order='F')
    x0[ctrl_start:slack_start] = U_shift.flatten(order='F')
    x0[slack_start:]           = sol_flat[slack_start:]
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

        self._init_guess_full  = None
        self._init_guess_light = None
        self._u_prev           = np.zeros(2)
        self._last_solver_was_light = False

        M = 5
        print(f"[MPC] Building full solver  (N={N_FULL})...")
        self._mpc_full  = build_mpc_solver(N_FULL,  max_dynamic_obstacles=M)
        print(f"[MPC] Building light solver (N={N_LIGHT})...")
        self._mpc_light = build_mpc_solver(N_LIGHT, max_dynamic_obstacles=M)
        print(f"[MPC] Both solvers ready. "
              f"full n_dec={self._mpc_full['n_dec']}, "
              f"light n_dec={self._mpc_light['n_dec']}")

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
            self._mode = "trajectory"
        else:
            target_wp   = np.array(new_waypoints[best_idx], dtype=float)
            desired_hdg = heading_to_waypoint(robot_xy, target_wp)
            hdg_err     = wrap(desired_hdg - state[2])
            if abs(np.degrees(hdg_err)) > self._pivot_threshold_deg:
                self._pivot_target_hdg = desired_hdg
                self._mode             = "pivot"
                print(f"[MPC] Pivot engaged: {np.degrees(hdg_err):.1f}deg")
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
                self._mode             = "trajectory"
                self._init_guess_full  = None
                self._init_guess_light = None
                self._u_prev           = np.zeros(2)
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
        M          = self._mpc_full['M']
        current_wp = np.array(self._waypoints[self._wp_idx], dtype=float)
        is_final   = (self._wp_idx == len(self._waypoints) - 1)
        dist_to_wp = np.linalg.norm(state[:2] - current_wp)

        # Dot-product advance
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
                        self._init_guess_full  = None
                        self._init_guess_light = None
                        self._u_prev           = np.zeros(2)
                        print(f"[MPC] Mid-path pivot: {np.degrees(hdg_err):.1f}deg")
                        return

        # Arrival check
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

        # ── Solver selection ──────────────────────────────────────────────────
        use_light = is_final and (dist_to_wp < LIGHT_SOLVER_DIST)
        mpc       = self._mpc_light if use_light else self._mpc_full

        if use_light and not self._last_solver_was_light:
            self._init_guess_light = None
            print(f"[MPC] -> Light solver at dist={dist_to_wp:.2f}m")
        elif not use_light and self._last_solver_was_light:
            self._init_guess_full = None
        self._last_solver_was_light = use_light

        init_guess = self._init_guess_light if use_light else self._init_guess_full

        # ── Initial guess (needed before flag computation) ────────────────────
        warm = (init_guess is not None)
        if not warm:
            init_guess = _make_state_seeded_guess(state, mpc)

        # ── Grid-lookup obstacle flags (NumPy, no CasADi) ────────────────────
        # Uses the warm-start trajectory to predict which horizon steps land on
        # obstacle cells.  Lag of one tick is negligible at 0.1s / 3m/s max.
        N = mpc['N']
        static_flags, dynamic_flags = compute_obs_flags(
            init_guess, self._parent.Map.Zone_Map, N
        )

        # Log how many steps are flagged — useful for debugging near obstacles
        n_static_flagged  = int(static_flags.sum())
        n_dynamic_flagged = int(dynamic_flags.sum())
        if n_static_flagged > 0:
            print(f"[MPC] WARNING: {n_static_flagged} horizon steps on static obstacle cells")
        if n_dynamic_flagged > 0:
            print(f"[MPC] NOTE: {n_dynamic_flagged} horizon steps on dynamic obstacle cells")

        # ── Parameter vector ──────────────────────────────────────────────────
        dyn_list = self._dynamic_obs[:M]
        p_vec    = build_param_vector(
            state=state, waypoint=current_wp,
            turkey_xy=self._turkey_pos, chase_mode=self._chase_mode,
            dynamic_obstacles=dyn_list, M=M,
            static_flags=static_flags,
            dynamic_flags=dynamic_flags,
        )
        # u_prev is at a fixed offset: X0(3)+wp(2)+turkey(2)+chase(1)+dyn(2M)+h0(M)
        # With M=5: offset = 3+2+2+1+10+5 = 23
        _uprev_idx = 3 + 2 + 2 + 1 + 2 * M + M
        p_vec[_uprev_idx]     = self._u_prev[0]
        p_vec[_uprev_idx + 1] = self._u_prev[1]

        # ── Solve ─────────────────────────────────────────────────────────────
        t0  = time.perf_counter()
        sol = mpc['solver'](
            x0=init_guess,
            lbx=mpc['lbx'], ubx=mpc['ubx'],
            lbg=mpc['lbg'], ubg=mpc['ubg'],
            p=p_vec,
        )
        solve_ms = (time.perf_counter() - t0) * 1000.0

        sol_flat = sol['x'].full().flatten()
        u        = sol_flat[mpc['ctrl_start'] : mpc['ctrl_start'] + mpc['n_u']]

        new_guess = _shift_warm_start(sol_flat, mpc)
        if use_light:
            self._init_guess_light = new_guess
        else:
            self._init_guess_full  = new_guess
        self._u_prev = u.copy()

        # ── Log ───────────────────────────────────────────────────────────────
        if logger:
            fov_err = float('nan')
            if self._turkey_pos is not None:
                tx, ty  = self._turkey_pos
                ang     = np.arctan2(-(ty - state[1]), tx - state[0])
                fov_err = wrap(ang - state[2])
            logger.log(
                state=state,
                mode="trajectory_light" if use_light else "trajectory",
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
              f"{'LIGHT' if use_light else 'full '} "
              f"{'chase' if self._chase_mode else 'scan '} "
              f"state=({state[0]:.2f},{state[1]:.2f},{np.degrees(state[2]):.1f}deg) "
              f"u=[{u[0]:.2f},{u[1]:.2f}] "
              f"solve={solve_ms:.1f}ms warm={warm} "
              f"obs_flags=[s:{n_static_flagged},d:{n_dynamic_flagged}]")

        await self._send(u)

    async def _send(self, u):
        await self.Drivetrain_pub.publish({
            "Id": "System", "Cmd": np.array(u, dtype=np.float32).tolist()
        })

    async def DriveTrain_Sender(self, logger=None):
        while True:
            await asyncio.sleep(DT)
            await self.MPC_Main(logger=logger)