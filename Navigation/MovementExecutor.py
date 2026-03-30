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
N            = 30
L            = WHEELBASE

# Cost weights — Q penalises [x, y, theta], R penalises [vl, vr]
Q  = np.diag([10.0, 10.0, 5.0])   # state tracking
Qf = 3.0 * Q                       # terminal state (scaled Q)
R  = np.diag([0.3, 0.3])           # control effort


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

def build_reference(waypoints_xy: np.ndarray, dt: float, total_time: float) -> np.ndarray:
    """Interpolate waypoints into a dense (x, y, theta) reference trajectory."""
    # Remove duplicate consecutive waypoints
    keep = np.ones(len(waypoints_xy), dtype=bool)
    for i in range(1, len(waypoints_xy)):
        if np.linalg.norm(waypoints_xy[i] - waypoints_xy[i-1]) < 1e-6:
            keep[i] = False
    waypoints_xy = waypoints_xy[keep]

    if len(waypoints_xy) < 2:
        M = max(1, int(total_time / dt))
        row = np.array([waypoints_xy[0][0], waypoints_xy[0][1], 0.0])
        return np.tile(row, (M, 1))

    dists = np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1)
    arc   = np.concatenate([[0], np.cumsum(dists)])
    arc_n = arc / arc[-1]

    t_dense = np.arange(0, total_time, dt)
    t_n     = (t_dense / total_time).clip(0, 1)
    kind    = 'cubic' if len(waypoints_xy) >= 4 else 'linear'

    xs = interp1d(arc_n, waypoints_xy[:, 0], kind=kind)(t_n)
    ys = interp1d(arc_n, waypoints_xy[:, 1], kind=kind)(t_n)

    # Compute heading from finite differences.
    # Use central differences in the interior to reduce endpoint noise,
    # then fix the final point to match its neighbour (not extrapolate).
    dx      = np.gradient(xs)
    dy      = np.gradient(ys)
    ths_raw = np.arctan2(-dy, dx)

    # Unwrap for continuity, then pin the last sample to its neighbour to
    # prevent np.unwrap introducing a phantom ±2π jump at the tail.
    ths = np.unwrap(ths_raw)
    ths[-1] = ths[-2]   # kill end-of-path discontinuity

    return np.column_stack([xs, ys, ths])


# ─────────────────────────────────────────────
# Continuous heading tracking
# ─────────────────────────────────────────────
def unwrap_state_heading(state: np.ndarray, prev_theta_continuous: float) -> np.ndarray:
    """
    Replace the wrapped angle in `state` with a continuous angle that is
    consistent with `prev_theta_continuous`.  This prevents the ±π flip
    (common when driving West) from causing the MPC to spin 360°.
    """
    delta = wrap(state[2] - prev_theta_continuous)
    continuous_theta = prev_theta_continuous + delta
    return np.array([state[0], state[1], continuous_theta])


# ─────────────────────────────────────────────
# MPC solver — barebones Q/R formulation
# ─────────────────────────────────────────────
def build_mpc_solver():
    """
    Standard reference-tracking NLP:

        min  Σ_{k=0}^{N-1} [ eₖᵀ Q eₖ + uₖᵀ R uₖ ] + e_Nᵀ Qf e_Nᵀ
        s.t. x_{k+1} = f(xₖ, uₖ)          (RK4 unicycle dynamics)
             V_MIN ≤ vl, vr ≤ V_MAX

    where eₖ = xₖ − xref_k  (states: x, y, θ — all continuous/unwrapped).

    Parameters packed as: [x0(3), xref_flat(3*(N+1)), uref_flat(2*N)]
    """
    n_s, n_u = 3, 2

    # ── Symbolic state / input ──────────────────────────────────────────────
    x_s  = ca.SX.sym('x');  y_s = ca.SX.sym('y');  th_s = ca.SX.sym('th')
    vl_s = ca.SX.sym('vl'); vr_s = ca.SX.sym('vr')
    states = ca.vertcat(x_s, y_s, th_s)
    inputs = ca.vertcat(vl_s, vr_s)

    v     = (vl_s + vr_s) / 2
    omega = (vr_s - vl_s) / L
    f_expr = ca.vertcat(v * ca.cos(th_s), -v * ca.sin(th_s), omega)
    f_disc = ca.Function('f', [states, inputs], [states + DT * f_expr])

    # ── Decision variables ──────────────────────────────────────────────────
    X_mpc = ca.SX.sym('X', n_s, N + 1)
    U_mpc = ca.SX.sym('U', n_u, N)

    # ── Parameters ──────────────────────────────────────────────────────────
    X0_p   = ca.SX.sym('X0',   n_s)
    Xref_p = ca.SX.sym('Xref', n_s, N + 1)
    Uref_p = ca.SX.sym('Uref', n_u, N)
    p_all  = ca.vertcat(X0_p,
                        ca.reshape(Xref_p, -1, 1),
                        ca.reshape(Uref_p, -1, 1))

    # ── Cost weights as CasADi DM ───────────────────────────────────────────
    Q_ca  = ca.DM(Q)
    Qf_ca = ca.DM(Qf)
    R_ca  = ca.DM(R)

    obj = 0
    g   = [X_mpc[:, 0] - X0_p]   # initial-state constraint

    for k in range(N):
        e_x = X_mpc[:, k] - Xref_p[:, k]
        e_u = U_mpc[:, k] - Uref_p[:, k]
        obj += ca.mtimes([e_x.T, Q_ca, e_x]) + ca.mtimes([e_u.T, R_ca, e_u])
        g.append(X_mpc[:, k + 1] - f_disc(X_mpc[:, k], U_mpc[:, k]))

    # Terminal cost
    e_N = X_mpc[:, N] - Xref_p[:, N]
    obj += ca.mtimes([e_N.T, Qf_ca, e_N])

    # ── Bounds ──────────────────────────────────────────────────────────────
    opt_vars   = ca.vertcat(ca.reshape(X_mpc, -1, 1), ca.reshape(U_mpc, -1, 1))
    n_dec      = opt_vars.numel()
    lbx        = -np.inf * np.ones(n_dec)
    ubx        =  np.inf * np.ones(n_dec)
    ctrl_start = n_s * (N + 1)
    for i in range(N):
        for j in range(n_u):
            idx      = ctrl_start + i * n_u + j
            lbx[idx] = V_MIN
            ubx[idx] = V_MAX

    g_concat = ca.vertcat(*g)
    n_g      = g_concat.numel()

    nlp    = {'f': obj, 'x': opt_vars, 'g': g_concat, 'p': p_all}
    solver = ca.nlpsol('solver', 'ipopt', nlp,
                       {'ipopt.print_level': 0, 'print_time': 0})

    return {
        'solver':     solver,
        'f_disc':     f_disc,
        'lbx':        lbx,
        'ubx':        ubx,
        'lbg':        np.zeros(n_g),
        'ubg':        np.zeros(n_g),
        'n_dec':      n_dec,
        'ctrl_start': ctrl_start,
        'n_u':        n_u,
        'n_s':        n_s,
    }


def build_param_vector(state, xref, u_ref):
    return np.concatenate([state, xref.flatten(), u_ref.flatten()])


def get_ref_slice(ref_idx, ref_traj, v_fwd):
    """Return a (N+1, 3) reference window and nominal inputs."""
    end  = min(ref_idx + N + 1, len(ref_traj))
    pad  = (N + 1) - (end - ref_idx)
    xref = np.vstack([ref_traj[ref_idx:end], np.tile(ref_traj[-1], (pad, 1))])

    # Nominal inputs: maintain v_fwd, derive omega from reference heading change
    u_ref = np.zeros((N, 2))
    for k in range(N):
        dth       = xref[k + 1, 2] - xref[k, 2] if k + 1 < len(xref) else 0.0
        omega_ref = dth / DT
        vr        = np.clip(v_fwd + omega_ref * L / 2, V_MIN, V_MAX)
        vl        = np.clip(v_fwd - omega_ref * L / 2, V_MIN, V_MAX)
        u_ref[k]  = [vl, vr]

    return xref, u_ref


# ─────────────────────────────────────────────
# Main controller class
# ─────────────────────────────────────────────
class MPC_loop:

    def __init__(self, parent, Drivetrain_pub, MPCFeedback_Navigation_pub):
        self.Drivetrain_pub  = Drivetrain_pub
        self.MPCFeedback_pub = MPCFeedback_Navigation_pub
        self._parent         = parent

        self._mode            = "idle"
        self._ref_traj        = None
        self._ref_idx         = 0
        self._v_fwd           = 0.5
        self._init_guess      = None
        self._theta_continuous = 0.0   # tracks continuous heading across steps

        print("[MPC] Building solver…")
        self._mpc = build_mpc_solver()
        print("[MPC] Solver ready.")

    # ─── Protocol 6 ──────────────────────────────────────────────────────────

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

    # ─── Command receivers ────────────────────────────────────────────────────

    async def Orient_CMD_Receiver(self, data):
        # data: target heading in radians
        target_hdg = float(data)
        KP = 3.0  # tune this — units: (m/s wheel speed) / rad

        while True:
            state = np.array(self._parent.Current_State, dtype=np.float64)
            err   = wrap(target_hdg - state[2])

            if abs(err) < np.radians(3.0):   # 3° deadband
                await self._send(np.zeros(2))
                return

            omega = KP * err
            vr =  np.clip( omega * L / 2, V_MIN, V_MAX)
            vl =  np.clip(-omega * L / 2, V_MIN, V_MAX)
            await self._send(np.array([vl, vr]))
            await asyncio.sleep(DT)

    async def Move_CMD_Receiver(self, data):
        robot = np.array(self._parent.Current_State, dtype=np.float64)

        waypoints_xy = np.array([[float(x), float(y)] for x, y in data])
        waypoints_xy = np.vstack([robot[:2], waypoints_xy])

        total_dist = np.sum(np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1))
        self._v_fwd  = np.clip(total_dist / (total_dist / 2.8), 0.3, V_MAX)
        total_time   = total_dist / self._v_fwd

        self._ref_traj   = build_reference(waypoints_xy, DT, total_time)
        self._ref_idx    = 0
        self._init_guess = np.zeros(self._mpc['n_dec'])

        # Seed the continuous-heading tracker, then shift the whole reference
        # trajectory so its heading branch matches the robot's current θ.
        # Without this, a robot at θ=+179° facing a ref at θ=-179° would see
        # a spurious 358° error on step 0 and spin before moving.
        self._theta_continuous = robot[2]
        ref_th0    = self._ref_traj[0, 2]
        th_offset  = robot[2] - wrap(robot[2] - ref_th0)
        self._ref_traj[:, 2] += (th_offset - ref_th0)

        self._mode = "trajectory"
        await self._proto_6_1_reference_trajectory(self._ref_traj)

    # ─── Control loop ─────────────────────────────────────────────────────────

    async def MPC_Main(self):
        raw_state = np.array(self._parent.Current_State, dtype=np.float64)

        # Convert wrapped angle → continuous angle to avoid ±π jumps
        state = unwrap_state_heading(raw_state, self._theta_continuous)
        self._theta_continuous = state[2]   # update tracker

        if self._mode != "trajectory":
            return

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
        nu       = self._mpc['n_u']
        u        = sol_flat[cs : cs + nu]

        ex = state[0] - xref[0, 0]
        ey = state[1] - xref[0, 1]
        et = np.degrees(wrap(state[2] - xref[0, 2]))

        print(f"[MPC] idx={self._ref_idx:04d} | "
              f"state=({state[0]:6.2f}, {state[1]:6.2f}, {np.degrees(state[2]):7.2f}°) | "
              f"ref=  ({xref[0,0]:6.2f}, {xref[0,1]:6.2f}, {np.degrees(xref[0,2]):7.2f}°) | "
              f"err=({ex:6.3f}, {ey:6.3f}, {et:6.2f}°) | "
              f"u=[{u[0]:5.2f}, {u[1]:5.2f}]")

        self._init_guess = sol_flat
        self._ref_idx    = min(self._ref_idx + 1, len(self._ref_traj) - 1)

        await self._send(u)
        await self._proto_6_2_mpc_feedback(xref)

    async def _send(self, u):
        cmd = np.array(u, dtype=np.float32)
        await self.Drivetrain_pub.publish({"Id": "System", "Cmd": cmd.tolist()})

    async def DriveTrain_Sender(self):
        while True:
            await asyncio.sleep(DT)
            await self.MPC_Main()