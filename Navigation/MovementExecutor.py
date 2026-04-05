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

FINAL_DIST        = 0.8
STALL_TICKS       = 15
LIGHT_SOLVER_DIST = 6.0   # widened: cheap solver takes over sooner

N_FULL  = 8    # was 15 — halves constraint count, faster near obstacles
N_LIGHT = 5    # was 6

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
W_SLACK_DYN  = 50.0   # dynamic obstacle CBF slack penalty
W_SLACK_FIX  = 80.0   # fixed obstacle CBF slack penalty (higher — walls are hard)

# Fixed obstacle CBF
N_FIX_OBS    = 3      # nearest fixed rects passed to solver each tick
CBF_GAMMA_FX = 0.5    # decay rate for fixed-obstacle CBF
R_FIX        = 0.9    # m — safety radius around fixed obstacles

# Dynamic obstacle CBF
M_DYN        = 3      # was 5 — saves M*N inequality constraints
CBF_GAMMA_DY = 0.6
R_DYN        = 1.5

# u_prev offset in parameter vector: X0(3)+wp(2)+turkey(2)+chase(1)+dyn(2*KD)+h0d(KD)
# With KD=3: 3+2+2+1+6+3 = 17
_UPREV_IDX = 3 + 2 + 2 + 1 + 2 * M_DYN + M_DYN


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
# Nearest fixed-obstacle rects — vectorised NumPy, called each tick
# =============================================================================

def nearest_fix_obs_rects(robot_xy: np.ndarray,
                           obstacle_rects,
                           n: int = N_FIX_OBS) -> np.ndarray:
    """
    Return the n nearest obstacle rects as a (n, 4) array [x0, x1, y0, y1],
    sorted nearest-first. Padded with far sentinels if fewer than n rects exist.
    Uses np.argpartition for O(K) selection — typically <0.1 ms with 100+ rects.
    """
    sentinel = np.array([1e6, 1e6 + 1, 1e6, 1e6 + 1], dtype=np.float64)

    if not obstacle_rects:
        return np.tile(sentinel, (n, 1))

    rects = np.asarray(obstacle_rects, dtype=np.float64)   # (K, 4)
    dx    = np.maximum(0.0, np.maximum(rects[:, 0] - robot_xy[0],
                                       robot_xy[0] - rects[:, 1]))
    dy    = np.maximum(0.0, np.maximum(rects[:, 2] - robot_xy[1],
                                       robot_xy[1] - rects[:, 3]))
    dist  = np.hypot(dx, dy)

    k   = min(n, len(rects))
    idx = np.argpartition(dist, range(k))[:k]
    idx = idx[np.argsort(dist[idx])]

    chosen = rects[idx]
    if k < n:
        chosen = np.vstack([chosen, np.tile(sentinel, (n - k, 1))])

    return chosen   # (n, 4)


# =============================================================================
# MPC solver builder
# =============================================================================

def build_mpc_solver(n_steps: int,
                     n_fix: int = N_FIX_OBS,
                     n_dyn: int = M_DYN):
    """
    Lean NLP with two classes of CBF safety constraint and no transcendental
    obstacle cost terms.

    Fixed obstacle CBF
    ------------------
    h(x,y,rect) = dist²(point, rect) - R_FIX²   (squared distance to rect surface)
    CBF:  h(k+1) >= (1 - gamma) * h(k) - slack
    One slack per rect shared across all steps.

    Dynamic obstacle CBF
    --------------------
    Same structure, M_DYN=3 slots instead of 5.

    Cost function
    -------------
    Purely quadratic + atan2 heading — no Gaussian/log-barrier terms.
    Simpler Hessian structure means L-BFGS approximates it accurately
    in far fewer iterations, especially near obstacles.
    """
    N   = n_steps
    n_s = 3
    n_u = 2
    KF  = n_fix
    KD  = n_dyn

    # ── Kinematics ────────────────────────────────────────────────────────────
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

    # ── Decision variables ────────────────────────────────────────────────────
    X_mpc  = ca.SX.sym('X',     n_s, N + 1)
    U_mpc  = ca.SX.sym('U',     n_u, N)
    S_fix  = ca.SX.sym('S_fix', KF)   # one slack per fixed rect
    S_dyn  = ca.SX.sym('S_dyn', KD)   # one slack per dynamic obstacle

    # ── Parameters ───────────────────────────────────────────────────────────
    # Layout: X0(3)|wp(2)|turkey(2)|chase(1)|dyn(2*KD)|h0d(KD)|uprev(2)|fix_rects(4*KF)
    P_X0    = ca.SX.sym('X0',    n_s)
    P_wp    = ca.SX.sym('wp',    2)
    P_turk  = ca.SX.sym('turk',  2)
    P_chase = ca.SX.sym('chase', 1)
    P_dyn   = ca.SX.sym('dyn',   2 * KD)
    P_h0d   = ca.SX.sym('h0d',   KD)
    P_uprev = ca.SX.sym('uprev', n_u)
    P_frect = ca.SX.sym('frect', 4 * KF)   # [x0,x1,y0,y1] per rect, flat

    p_all = ca.vertcat(
        P_X0, P_wp, P_turk, P_chase,
        P_dyn, P_h0d, P_uprev,
        P_frect,
    )

    # ── Cost ─────────────────────────────────────────────────────────────────
    obj  = 0
    for i in range(KF):
        obj += W_SLACK_FIX * S_fix[i]**2
    for i in range(KD):
        obj += W_SLACK_DYN * S_dyn[i]**2

    g_eq = [X_mpc[:, 0] - P_X0]

    for k in range(N):
        xk  = X_mpc[0, k];  yk  = X_mpc[1, k];  thk = X_mpc[2, k]
        vlk = U_mpc[0, k];  vrk = U_mpc[1, k]

        obj += W_XY_RUN * ((xk - P_wp[0])**2 + (yk - P_wp[1])**2)

        # Scan heading
        wp_dx  = P_wp[0] - xk;  wp_dy = P_wp[1] - yk
        wp_reg = ca.sqrt(wp_dx**2 + wp_dy**2 + 1e-6)
        ang_wp = ca.atan2(-wp_dy / wp_reg, wp_dx / wp_reg)
        err_wp = ca.atan2(ca.sin(ang_wp - thk), ca.cos(ang_wp - thk))
        obj   += (1 - P_chase) * W_HEADING * err_wp**2

        # Chase heading
        tk_dx   = P_turk[0] - xk;  tk_dy = P_turk[1] - yk
        tk_reg  = ca.sqrt(tk_dx**2 + tk_dy**2 + 1e-6)
        ang_t   = ca.atan2(-tk_dy / tk_reg, tk_dx / tk_reg)
        err_fov = ca.atan2(ca.sin(ang_t - thk), ca.cos(ang_t - thk))
        obj    += P_chase * W_FOV * err_fov**2

        obj += W_OMEGA * ((vrk - vlk) / L)**2
        u_prev_k = P_uprev if k == 0 else U_mpc[:, k - 1]
        obj     += W_INPUT_DIFF * ca.sumsqr(U_mpc[:, k] - u_prev_k)

        g_eq.append(X_mpc[:, k + 1] - f_disc(X_mpc[:, k], U_mpc[:, k]))

    obj += W_XY_TERM * (
        (X_mpc[0, N] - P_wp[0])**2 + (X_mpc[1, N] - P_wp[1])**2
    )

    # ── Fixed obstacle CBF ────────────────────────────────────────────────────
    g_ineq = []
    for i in range(KF):
        rx0 = P_frect[4 * i + 0];  rx1 = P_frect[4 * i + 1]
        ry0 = P_frect[4 * i + 2];  ry1 = P_frect[4 * i + 3]

        def rect_dist2(px, py):
            ddx = ca.fmax(rx0 - px, ca.fmax(0.0, px - rx1))
            ddy = ca.fmax(ry0 - py, ca.fmax(0.0, py - ry1))
            return ddx**2 + ddy**2

        for k in range(N):
            hk  = rect_dist2(X_mpc[0, k],   X_mpc[1, k])   - R_FIX**2
            hk1 = rect_dist2(X_mpc[0, k+1], X_mpc[1, k+1]) - R_FIX**2
            g_ineq.append(hk1 - (1 - CBF_GAMMA_FX) * hk + S_fix[i])

    # ── Dynamic obstacle CBF ──────────────────────────────────────────────────
    for i in range(KD):
        ox = P_dyn[2 * i];  oy = P_dyn[2 * i + 1]
        for k in range(N):
            xk  = X_mpc[0, k];   yk  = X_mpc[1, k]
            xk1 = X_mpc[0, k+1]; yk1 = X_mpc[1, k+1]
            hk  = (xk  - ox)**2 + (yk  - oy)**2 - R_DYN**2
            hk1 = (xk1 - ox)**2 + (yk1 - oy)**2 - R_DYN**2
            g_ineq.append(hk1 - (1 - CBF_GAMMA_DY) * hk + S_dyn[i])

    # ── Pack NLP ──────────────────────────────────────────────────────────────
    opt_vars = ca.vertcat(
        ca.reshape(X_mpc, -1, 1),
        ca.reshape(U_mpc, -1, 1),
        S_fix,
        S_dyn,
    )
    n_dec      = opt_vars.numel()
    ctrl_start = n_s * (N + 1)
    sfix_start = ctrl_start + n_u * N
    sdyn_start = sfix_start + KF

    lbx = -np.inf * np.ones(n_dec)
    ubx =  np.inf * np.ones(n_dec)
    lbx[ctrl_start:sfix_start] = V_MIN
    ubx[ctrl_start:sfix_start] = V_MAX
    lbx[sfix_start:]           = 0.0   # all slacks non-negative

    g_eq_cat   = ca.vertcat(*g_eq)
    g_ineq_cat = ca.vertcat(*g_ineq)
    g_all      = ca.vertcat(g_eq_cat, g_ineq_cat)

    n_eq   = g_eq_cat.numel()
    n_ineq = g_ineq_cat.numel()
    lbg    = np.concatenate([np.zeros(n_eq),   np.zeros(n_ineq)])
    ubg    = np.concatenate([np.zeros(n_eq),   np.inf * np.ones(n_ineq)])

    solver = ca.nlpsol('solver', 'ipopt',
        {'f': obj, 'x': opt_vars, 'g': g_all, 'p': p_all},
        {
            'ipopt.print_level':                0,
            'print_time':                       0,
            'ipopt.tol':                        5e-3,
            'ipopt.max_iter':                   50,
            'ipopt.warm_start_init_point':      'yes',
            'ipopt.mu_strategy':                'adaptive',
            'ipopt.hessian_approximation':      'limited-memory',
            'ipopt.limited_memory_max_history': 6,
        }
    )

    return dict(
        solver     = solver,
        lbx        = lbx,   ubx = ubx,
        lbg        = lbg,   ubg = ubg,
        n_dec      = n_dec,
        ctrl_start = ctrl_start,
        sfix_start = sfix_start,
        sdyn_start = sdyn_start,
        n_s        = n_s,
        n_u        = n_u,
        N          = N,
        KF         = KF,
        KD         = KD,
        n_p        = p_all.numel(),
    )


# =============================================================================
# Parameter vector
# =============================================================================

def build_param_vector(state, waypoint, turkey_xy, chase_mode,
                       dynamic_obstacles, fix_rects: np.ndarray):
    """
    fix_rects : (N_FIX_OBS, 4) from nearest_fix_obs_rects(), cols [x0,x1,y0,y1]
    """
    KD         = M_DYN
    dyn_padded = list(dynamic_obstacles) + [(-999.0, -999.0)] * (KD - len(dynamic_obstacles))
    dyn_flat   = np.array([(ox, oy) for ox, oy in dyn_padded]).flatten()
    rx, ry     = state[0], state[1]
    h0d        = np.array([(rx - ox)**2 + (ry - oy)**2 - R_DYN**2
                            for ox, oy in dyn_padded])
    turkey     = np.array(turkey_xy, dtype=float) if turkey_xy is not None else np.zeros(2)

    return np.concatenate([
        state,
        np.array(waypoint, dtype=float),
        turkey,
        [1.0 if chase_mode else 0.0],
        dyn_flat, h0d,
        np.zeros(2),          # u_prev placeholder — filled in MPC_Main
        fix_rects.flatten(),  # 4 * N_FIX_OBS
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
    n_s        = mpc_dict['n_s']
    n_u        = mpc_dict['n_u']
    N          = mpc_dict['N']
    ctrl_start = mpc_dict['ctrl_start']
    sfix_start = mpc_dict['sfix_start']
    n_dec      = mpc_dict['n_dec']

    X_flat  = sol_flat[:ctrl_start].reshape(n_s, N + 1, order='F')
    U_flat  = sol_flat[ctrl_start:sfix_start].reshape(n_u, N, order='F')
    X_shift = np.hstack([X_flat[:, 1:], X_flat[:, -1:]])
    U_shift = np.hstack([U_flat[:, 1:], U_flat[:, -1:]])

    x0 = np.zeros(n_dec)
    x0[:ctrl_start]            = X_shift.flatten(order='F')
    x0[ctrl_start:sfix_start]  = U_shift.flatten(order='F')
    x0[sfix_start:]            = sol_flat[sfix_start:]   # carry slacks forward
    return x0


# =============================================================================
# Controller
# =============================================================================

class MPC_loop:

    def __init__(self, parent, Drivetrain_pub, MPCFeedback_Navigation_pub):
        self.Drivetrain_pub  = Drivetrain_pub
        self.MPCFeedback_pub = MPCFeedback_Navigation_pub
        self._parent         = parent

        self._mode        = "idle"
        self._waypoints   = []
        self._wp_idx      = 0

        self._turkey_pos  = None
        self._chase_mode  = False
        self._dynamic_obs = []

        self._pivot_target_hdg    = 0.0
        self._pivot_threshold_deg = PIVOT_THRESHOLD_DEG

        self._last_dist   = np.inf
        self._stall_count = 0

        self._init_guess_full  = None
        self._init_guess_light = None
        self._u_prev           = np.zeros(2)
        self._last_solver_was_light = False

        print(f"[MPC] Building full solver  (N={N_FULL}, KF={N_FIX_OBS}, KD={M_DYN})...")
        self._mpc_full  = build_mpc_solver(N_FULL)
        print(f"[MPC] Building light solver (N={N_LIGHT}, KF={N_FIX_OBS}, KD={M_DYN})...")
        self._mpc_light = build_mpc_solver(N_LIGHT)
        print(f"[MPC] Ready. full n_dec={self._mpc_full['n_dec']}  "
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
                        waypoint=self._waypoints[self._wp_idx] if self._waypoints else (0, 0),
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

        # ── Nearest fixed rects (vectorised NumPy) ────────────────────────────
        fix_rects = nearest_fix_obs_rects(state[:2], self._parent.Map.Obstacle_Rects)

        # ── Parameter vector ──────────────────────────────────────────────────
        p_vec = build_param_vector(
            state=state, waypoint=current_wp,
            turkey_xy=self._turkey_pos, chase_mode=self._chase_mode,
            dynamic_obstacles=self._dynamic_obs[:M_DYN],
            fix_rects=fix_rects,
        )
        p_vec[_UPREV_IDX]     = self._u_prev[0]
        p_vec[_UPREV_IDX + 1] = self._u_prev[1]

        # ── Initial guess ─────────────────────────────────────────────────────
        warm = (init_guess is not None)
        if not warm:
            init_guess = _make_state_seeded_guess(state, mpc)

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