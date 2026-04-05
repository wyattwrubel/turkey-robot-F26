import asyncio
import numpy as np
import sys
sys.path.insert(0, '/home/turkeysquad/ME450/MainDesign/casadi')
if 'casadi' in sys.modules:
    del sys.modules['casadi']

import casadi as ca

from scipy.interpolate import interp1d


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
WHEELBASE    = 0.7
DT           = 0.1
V_MAX        = 3.0
V_MIN        = -3.0
ARRIVAL_DIST = 0.5

N  = 30

# ── Cost weights ──────────────────────────────────────────────────────────────
# Cross-track (lateral) error is penalized hard — this implicitly enforces
# heading because the only way to reduce it is to point the right direction.
# Along-track error is light — we don't care much if we're a bit behind/ahead.
# Theta is NOT penalized at all — it's an emergent consequence of XY tracking.
W_CROSS     = 80.0    # lateral deviation penalty  ← heavy
W_ALONG     = 5.0     # along-path deviation        ← light
W_TERMINAL  = 3.0     # terminal cost multiplier on both
R           = np.diag([0.3, 0.3])
OMEGA_WEIGHT = 1.5    # turn-rate damping: penalises (vr-vl)²/L²
# ─────────────────────────────────────────────────────────────────────────────

L  = WHEELBASE

PIVOT_THRESHOLD_DEG = 45.0
PIVOT_ALIGN_DEG     = 10.0
PIVOT_OMEGA         = 3.0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

def heading_to(from_xy, to_xy):
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return np.arctan2(-dy, dx)

def build_reference(waypoints_xy: np.ndarray, dt: float, total_time: float) -> np.ndarray:
    keep = np.ones(len(waypoints_xy), dtype=bool)
    for i in range(1, len(waypoints_xy)):
        if np.linalg.norm(waypoints_xy[i] - waypoints_xy[i-1]) < 1e-6:
            keep[i] = False
    waypoints_xy = waypoints_xy[keep]

    if len(waypoints_xy) < 2:
        th  = 0.0
        M   = max(1, int(total_time / dt))
        row = np.array([waypoints_xy[0][0], waypoints_xy[0][1], th])
        return np.tile(row, (M, 1))

    dists = np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1)
    arc   = np.concatenate([[0], np.cumsum(dists)])
    arc_n = arc / arc[-1]

    t_dense = np.arange(0, total_time, dt)
    t_n     = (t_dense / total_time).clip(0, 1)

    kind = 'cubic' if len(waypoints_xy) >= 4 else 'linear'

    xs = interp1d(arc_n, waypoints_xy[:, 0], kind=kind)(t_n)
    ys = interp1d(arc_n, waypoints_xy[:, 1], kind=kind)(t_n)

    dx  = np.diff(xs, append=xs[-1])
    dy  = np.diff(ys, append=ys[-1])
    ths = np.arctan2(-dy, dx)
    ths = np.unwrap(ths)

    return np.column_stack([xs, ys, ths])

def nominal_input_from_ref(ref_slice: np.ndarray, v_fwd: float) -> np.ndarray:
    u_ref = np.zeros((N, 2))
    for k in range(N):
        dth       = ref_slice[k+1, 2] - ref_slice[k, 2] if k+1 < len(ref_slice) else 0.0
        omega_ref = wrap(dth) / DT
        vr        = np.clip(v_fwd + omega_ref * L / 2, V_MIN, V_MAX)
        vl        = np.clip(v_fwd - omega_ref * L / 2, V_MIN, V_MAX)
        u_ref[k]  = [vl, vr]
    return u_ref

def get_ref_slice(ref_idx, ref_traj, v_fwd):
    end = min(ref_idx + N + 1, len(ref_traj))
    pad = N + 1 - (end - ref_idx)
    xref  = np.vstack([ref_traj[ref_idx:end], np.tile(ref_traj[-1], (pad, 1))])
    u_ref = nominal_input_from_ref(xref, v_fwd)
    return xref, u_ref

def pivot_wheels(heading_error_rad: float) -> np.ndarray:
    sign = np.sign(heading_error_rad)
    vl   = np.clip(-sign * PIVOT_OMEGA * L / 2, V_MIN, V_MAX)
    vr   = np.clip( sign * PIVOT_OMEGA * L / 2, V_MIN, V_MAX)
    return np.array([vl, vr])


# ─────────────────────────────────────────────
# MPC solver
# ─────────────────────────────────────────────
def build_mpc_solver():
    """
    Cost is expressed in path-frame coordinates (along-track, cross-track)
    rather than world-frame (x, y, theta).

    At each horizon step k, given the reference heading theta_ref[k]:
      - along-track error  = (dx)*cos(theta_ref) - (dy)*sin(theta_ref)  -- how far behind/ahead
      - cross-track error  = (dx)*sin(theta_ref) + (dy)*cos(theta_ref)  -- lateral deviation

    Cross-track is penalized heavily (W_CROSS), along-track lightly (W_ALONG).
    Theta is NOT in the cost — heading alignment is implicit.

    theta_ref values are passed as parameters (one per horizon step) so the
    rotation is fixed at solve time — no trig inside the optimisation variables,
    keeping the problem cleanly convex in the position states.
    """
    n_states, n_controls = 3, 2

    x_s, y_s, th_s = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('theta')
    vl_s, vr_s      = ca.SX.sym('vl'), ca.SX.sym('vr')
    states  = ca.vertcat(x_s, y_s, th_s)
    inputs  = ca.vertcat(vl_s, vr_s)

    v      = (vl_s + vr_s) / 2
    omega  = (vr_s - vl_s) / L
    f_expr = ca.vertcat(v * ca.cos(th_s), -v * ca.sin(th_s), omega)
    f_disc = ca.Function('f', [states, inputs], [states + DT * f_expr])

    X_mpc  = ca.SX.sym('X', n_states,   N + 1)
    U_mpc  = ca.SX.sym('U', n_controls, N)

    # Parameters: initial state, reference XY (N+1), reference theta (N+1),
    #             reference inputs (N)
    X0_p      = ca.SX.sym('X0',      n_states)
    Xref_p    = ca.SX.sym('Xref',    2,   N + 1)   # xy only
    ThRef_p   = ca.SX.sym('ThRef',   1,   N + 1)   # theta_ref per step
    Uref_p    = ca.SX.sym('Uref',    n_controls, N)
    p_all     = ca.vertcat(
        X0_p,
        ca.reshape(Xref_p,  -1, 1),
        ca.reshape(ThRef_p, -1, 1),
        ca.reshape(Uref_p,  -1, 1),
    )

    obj = 0
    g   = [X_mpc[:, 0] - X0_p]

    for k in range(N):
        dx  = X_mpc[0, k] - Xref_p[0, k]
        dy  = X_mpc[1, k] - Xref_p[1, k]
        th  = ThRef_p[0, k]               # reference heading at step k (fixed param)

        # Rotate world-frame error into path frame
        e_along = dx * ca.cos(th) - dy * ca.sin(th)
        e_cross = dx * ca.sin(th) + dy * ca.cos(th)

        e_u     = U_mpc[:, k] - Uref_p[:, k]
        omega_k = (U_mpc[1, k] - U_mpc[0, k]) / L

        obj += (W_ALONG * e_along**2
                + W_CROSS * e_cross**2
                + ca.mtimes([e_u.T, R, e_u])
                + OMEGA_WEIGHT * omega_k**2)

        g.append(X_mpc[:, k+1] - f_disc(X_mpc[:, k], U_mpc[:, k]))

    # Terminal — same path-frame cost, scaled up
    dx_N  = X_mpc[0, N] - Xref_p[0, N]
    dy_N  = X_mpc[1, N] - Xref_p[1, N]
    th_N  = ThRef_p[0, N]
    e_along_N = dx_N * ca.cos(th_N) - dy_N * ca.sin(th_N)
    e_cross_N = dx_N * ca.sin(th_N) + dy_N * ca.cos(th_N)
    obj += W_TERMINAL * (W_ALONG * e_along_N**2 + W_CROSS * e_cross_N**2)

    opt_vars   = ca.vertcat(ca.reshape(X_mpc, -1, 1), ca.reshape(U_mpc, -1, 1))
    n_dec      = opt_vars.numel()
    lbx        = -np.inf * np.ones(n_dec)
    ubx        =  np.inf * np.ones(n_dec)
    ctrl_start = n_states * (N + 1)
    for i in range(N):
        for j in range(n_controls):
            idx      = ctrl_start + i * n_controls + j
            lbx[idx] = V_MIN
            ubx[idx] = V_MAX

    g_concat = ca.vertcat(*g)
    n_g      = g_concat.numel()

    nlp    = {'f': obj, 'x': opt_vars, 'g': g_concat, 'p': p_all}
    solver = ca.nlpsol('solver', 'ipopt', nlp,
                       {'ipopt.print_level': 0, 'print_time': 0})

    return {
        'solver':      solver,
        'f_disc':      f_disc,
        'lbx':         lbx,
        'ubx':         ubx,
        'lbg':         np.zeros(n_g),
        'ubg':         np.zeros(n_g),
        'n_dec':       n_dec,
        'ctrl_start':  ctrl_start,
        'n_controls':  n_controls,
        'n_states':    n_states,
    }

def build_param_vector(state, xref, u_ref):
    """
    Pack the parameter vector for the new solver signature.
    ThRef is passed separately so the solver can rotate errors into path frame.
    """
    xy_ref = xref[:, :2]          # (N+1, 2)
    th_ref = xref[:, 2:3]         # (N+1, 1)  — unwrapped is fine, only cos/sin used
    p = np.concatenate([
        state,
        xy_ref.flatten(),
        th_ref.flatten(),
        u_ref.flatten(),
    ])
    return p


# ─────────────────────────────────────────────
# Main controller class
# ─────────────────────────────────────────────
class MPC_loop:

    def __init__(self, parent, Drivetrain_pub, MPCFeedback_Navigation_pub):
        self.Drivetrain_pub  = Drivetrain_pub
        self.MPCFeedback_pub = MPCFeedback_Navigation_pub
        self._parent         = parent

        self._mode       = "idle"
        self._ref_traj   = None
        self._ref_idx    = 0
        self._v_fwd      = 0.5
        self._init_guess = None
        self._target_hdg = 0.0

        print("[MPC] Building solver…")
        self._mpc = build_mpc_solver()
        print("[MPC] Solver ready.")

    # ─────────────────────────────────────────────
    # Protocol 6
    # ─────────────────────────────────────────────

    async def _proto_6_1_reference_trajectory(self, ref_traj):
        await self.MPCFeedback_pub.publish({
            "Id": "Reference", "Cmd": ref_traj[:, :2].tolist()
        })

    async def _proto_6_2_mpc_feedback(self, xref_slice):
        await self.MPCFeedback_pub.publish({
            "Id": "FeedbackMPC", "Cmd": xref_slice[:, :2].tolist()
        })

    async def _proto_6_3_clear_trajectory(self):
        await self.MPCFeedback_pub.publish({"Id": "ClearTrajectory"})

    # ─────────────────────────────────────────────
    # Command receivers
    # ─────────────────────────────────────────────

    async def Orient_CMD_Receiver(self, data):
        print("[MPC] Fixed/orient mode — WIP")
        self._mode = "idle"

    async def Move_CMD_Receiver(self, data):
        robot = np.array(self._parent.Current_State, dtype=np.float64)

        waypoints_xy = np.array([[float(x), float(y)] for x, y in data])
        waypoints_xy = np.vstack([robot[:2], waypoints_xy])

        total_dist = np.sum(np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1))
        self._v_fwd  = np.clip(total_dist / (total_dist / 2.8), 0.3, V_MAX)
        total_time   = total_dist / self._v_fwd + 5.0

        self._ref_traj   = build_reference(waypoints_xy, DT, total_time)
        self._ref_idx    = 0
        self._init_guess = np.zeros(self._mpc['n_dec'])

        initial_ref_hdg = wrap(self._ref_traj[0, 2])
        hdg_err         = wrap(initial_ref_hdg - robot[2])

        if abs(np.degrees(hdg_err)) > PIVOT_THRESHOLD_DEG:
            self._target_hdg = initial_ref_hdg
            self._mode       = "pivot"
            print(f"[MPC] Pivot engaged — hdg_err={np.degrees(hdg_err):.1f}°")
        else:
            self._mode = "trajectory"

        await self._proto_6_1_reference_trajectory(self._ref_traj)

    # ─────────────────────────────────────────────
    # Control loop
    # ─────────────────────────────────────────────

    async def MPC_Main(self):
        state = np.array(self._parent.Current_State, dtype=np.float64)

        # ── Pivot ─────────────────────────────────────────────────────────────
        if self._mode == "pivot":
            hdg_err = wrap(self._target_hdg - state[2])
            if abs(np.degrees(hdg_err)) < PIVOT_ALIGN_DEG:
                print("[MPC] Pivot done — switching to trajectory")
                self._mode = "trajectory"
            else:
                u = pivot_wheels(hdg_err)
                print(f"[MPC] PIVOT | "
                      f"state=({state[0]:6.2f}, {state[1]:6.2f}, {np.degrees(state[2]):7.2f}°) | "
                      f"target={np.degrees(self._target_hdg):7.2f}° | "
                      f"err={np.degrees(hdg_err):7.2f}° | "
                      f"u=[{u[0]:5.2f}, {u[1]:5.2f}]")
                await self._send(u)
                return
        # ─────────────────────────────────────────────────────────────────────

        if self._mode == "trajectory":
            final = self._ref_traj[-1]
            dist  = np.linalg.norm(state[:2] - final[:2])

            if self._ref_idx >= len(self._ref_traj) - 1 and dist < ARRIVAL_DIST:
                print(f"[MPC] Arrived at ({final[0]:.2f}, {final[1]:.2f}), err={dist:.3f} m")
                self._mode = "idle"
                await self._send(np.zeros(2))
                await self._proto_6_3_clear_trajectory()
                return

            xref, u_ref = get_ref_slice(self._ref_idx, self._ref_traj, self._v_fwd)
            p_vec       = build_param_vector(state, xref, u_ref)

            sol = self._mpc['solver'](
                x0  = self._init_guess,
                lbx = self._mpc['lbx'],
                ubx = self._mpc['ubx'],
                lbg = self._mpc['lbg'],
                ubg = self._mpc['ubg'],
                p   = p_vec,
            )

            sol_flat = sol['x'].full().flatten()
            cs       = self._mpc['ctrl_start']
            nc       = self._mpc['n_controls']
            u        = sol_flat[cs : cs + nc]

            xref_now = xref[0]
            # Compute path-frame errors for logging
            dx       = state[0] - xref_now[0]
            dy       = state[1] - xref_now[1]
            th       = xref_now[2]
            e_along  = dx * np.cos(th) - dy * np.sin(th)
            e_cross  = dx * np.sin(th) + dy * np.cos(th)

            print(f"[MPC] idx={self._ref_idx:04d} | "
                  f"state=({state[0]:6.2f}, {state[1]:6.2f}, {np.degrees(state[2]):7.2f}°) | "
                  f"ref=  ({xref_now[0]:6.2f}, {xref_now[1]:6.2f}, "
                  f"{np.degrees(wrap(xref_now[2])):7.2f}°) | "
                  f"e_along={e_along:6.3f} e_cross={e_cross:6.3f} | "
                  f"u=[{u[0]:5.2f}, {u[1]:5.2f}]")

            self._init_guess = sol_flat
            self._ref_idx    = min(self._ref_idx + 1, len(self._ref_traj) - 1)

            await self._send(u)
            await self._proto_6_2_mpc_feedback(xref)

        elif self._mode == "fixed":
            print("[MPC] Fixed/orient mode — WIP")
            self._mode = "idle"

    async def _send(self, u, **_):
        cmd = np.array(u, dtype=np.float32)
        await self.Drivetrain_pub.publish({"Id": "System", "Cmd": cmd.tolist()})

    async def DriveTrain_Sender(self):
        while True:
            await asyncio.sleep(DT)
            await self.MPC_Main()