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
ARRIVAL_DIST = 0.1
N            = 10
L            = WHEELBASE

Q  = np.diag([10.0, 10.0, 15.0])
Qf = 3.0 * Q
R  = np.diag([3, 3])

REPLAN_DIST_THRESHOLD = 1.0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def build_reference(waypoints_xy: np.ndarray, dt: float, total_time: float) -> np.ndarray:
    """Interpolate waypoints into a dense (x, y, theta) reference.
    Headings are raw arctan2 — always in (-π, π], never unwrapped."""
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

    dx  = np.gradient(xs)
    dy  = np.gradient(ys)
    ths = np.arctan2(-dy, dx)   # stays in (-π, π] — no unwrap
    ths[-1] = ths[-2]

    return np.column_stack([xs, ys, ths])


def get_state(parent) -> np.ndarray:
    """Fresh sensor read with heading wrapped to (-π, π]. No accumulator."""
    raw = np.array(parent.Current_State, dtype=np.float64)
    return np.array([raw[0], raw[1], wrap(raw[2])])


# ─────────────────────────────────────────────
# MPC solver — wrapped angle error in CasADi cost
# ─────────────────────────────────────────────
def build_mpc_solver():
    n_s, n_u = 3, 2

    x_s  = ca.SX.sym('x');  y_s = ca.SX.sym('y');  th_s = ca.SX.sym('th')
    vl_s = ca.SX.sym('vl'); vr_s = ca.SX.sym('vr')
    states = ca.vertcat(x_s, y_s, th_s)
    inputs = ca.vertcat(vl_s, vr_s)

    v      = (vl_s + vr_s) / 2
    omega  = (vr_s - vl_s) / L
    f_expr = ca.vertcat(v * ca.cos(th_s), -v * ca.sin(th_s), omega)
    f_disc = ca.Function('f', [states, inputs], [states + DT * f_expr])

    X_mpc  = ca.SX.sym('X',     n_s, N + 1)
    U_mpc  = ca.SX.sym('U',     n_u, N)
    X0_p   = ca.SX.sym('X0',    n_s)
    Xref_p = ca.SX.sym('Xref',  n_s, N + 1)
    Uref_p = ca.SX.sym('Uref',  n_u, N)

    p_all = ca.vertcat(X0_p,
                       ca.reshape(Xref_p, -1, 1),
                       ca.reshape(Uref_p, -1, 1))

    qx,  qy,  qth  = float(Q[0,0]),  float(Q[1,1]),  float(Q[2,2])
    qxf, qyf, qthf = float(Qf[0,0]), float(Qf[1,1]), float(Qf[2,2])
    r0,  r1        = float(R[0,0]),   float(R[1,1])

    def ang_err(a, b):
        """Shortest-arc angular error, result in (-π, π]."""
        d = a - b
        return ca.atan2(ca.sin(d), ca.cos(d))

    obj = 0
    g   = [X_mpc[:, 0] - X0_p]

    for k in range(N):
        xk, uk   = X_mpc[:, k],   U_mpc[:, k]
        xrk, urk = Xref_p[:, k],  Uref_p[:, k]
        ex  = xk[0] - xrk[0]
        ey  = xk[1] - xrk[1]
        eth = ang_err(xk[2], xrk[2])          # ← wrapped, never winds up
        eu0 = uk[0] - urk[0]
        eu1 = uk[1] - urk[1]
        obj += qx*ex**2 + qy*ey**2 + qth*eth**2 + r0*eu0**2 + r1*eu1**2
        g.append(X_mpc[:, k+1] - f_disc(X_mpc[:, k], U_mpc[:, k]))

    eN  = X_mpc[:, N]
    xrN = Xref_p[:, N]
    obj += (qxf*(eN[0]-xrN[0])**2 + qyf*(eN[1]-xrN[1])**2
            + qthf*ang_err(eN[2], xrN[2])**2)

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
    nlp = {'f': obj, 'x': opt_vars, 'g': g_concat, 'p': p_all}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.print_level':           0,
        'print_time':                  0,
        'ipopt.tol':                   1e-3,
        'ipopt.max_iter':              50,
        'ipopt.warm_start_init_point': 'yes',
    })

    return {
        'solver':     solver,
        'f_disc':     f_disc,
        'lbx':        lbx,
        'ubx':        ubx,
        'lbg':        np.zeros(g_concat.numel()),
        'ubg':        np.zeros(g_concat.numel()),
        'n_dec':      n_dec,
        'ctrl_start': ctrl_start,
        'n_u':        n_u,
        'n_s':        n_s,
    }


def build_param_vector(state, xref, u_ref):
    return np.concatenate([state, xref.flatten(), u_ref.flatten()])


def get_ref_slice(ref_idx, ref_traj, v_fwd):
    end  = min(ref_idx + N + 1, len(ref_traj))
    pad  = (N + 1) - (end - ref_idx)
    xref = np.vstack([ref_traj[ref_idx:end], np.tile(ref_traj[-1], (pad, 1))])

    u_ref = np.zeros((N, 2))
    for k in range(N):
        # Use wrapped diff so dth is always in (-π, π]
        dth       = wrap(xref[k+1, 2] - xref[k, 2]) if k+1 < len(xref) else 0.0
        omega_ref = dth / DT
        vr        = np.clip(v_fwd + omega_ref * L / 2, V_MIN, V_MAX)
        vl        = np.clip(v_fwd - omega_ref * L / 2, V_MIN, V_MAX)
        u_ref[k]  = [vl, vr]

    return xref, u_ref


def _shift_warm_start(sol_flat, n_s, n_u, ctrl_start):
    g = np.zeros_like(sol_flat)
    g[:ctrl_start - n_s]           = sol_flat[n_s:ctrl_start]
    g[ctrl_start - n_s:ctrl_start] = sol_flat[ctrl_start - n_s:ctrl_start]
    g[ctrl_start:-n_u]             = sol_flat[ctrl_start + n_u:]
    g[-n_u:]                       = sol_flat[-n_u:]
    # Wrap heading slots so they don't drift in the warm-start vector
    for k in range(N + 1):
        g[k * n_s + 2] = wrap(g[k * n_s + 2])
    return g


# ─────────────────────────────────────────────
# Main controller class
# ─────────────────────────────────────────────
class MPC_loop:

    def __init__(self, parent, Drivetrain_pub, MPCFeedback_Navigation_pub):
        self.Drivetrain_pub  = Drivetrain_pub
        self.MPCFeedback_pub = MPCFeedback_Navigation_pub
        self._parent         = parent

        self._mode           = "idle"
        self._ref_traj       = None
        self._ref_idx        = 0
        self._v_fwd          = 0.5
        self._init_guess     = None   # None = cold start not yet done
        self._last_target_xy = None

        # No _theta_continuous — all headings stay in (-π, π] via wrap()

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
        target_hdg = wrap(float(data[2]))
        KP = 3.0

        while True:
            state = get_state(self._parent)
            err   = wrap(target_hdg - state[2])

            if abs(err) < np.radians(3.0):
                await self._send(np.zeros(2))
                return

            omega = KP * err
            vr =  np.clip( omega * L / 2, V_MIN, V_MAX)
            vl =  np.clip(-omega * L / 2, V_MIN, V_MAX)
            await self._send(np.array([vl, vr]))
            await asyncio.sleep(DT)

    async def Move_CMD_Receiver(self, data):
        state        = get_state(self._parent)
        waypoints_xy = np.array([[float(x), float(y)] for x, y in data])
        new_target   = waypoints_xy[-1]

        # ── Incremental replan ────────────────────────────────────────────────
        if (self._mode == "trajectory"
                and self._ref_traj is not None
                and self._last_target_xy is not None
                and np.linalg.norm(new_target - self._last_target_xy) < REPLAN_DIST_THRESHOLD):

            remaining = self._ref_traj[self._ref_idx:]
            new_tail  = np.array([[new_target[0], new_target[1], remaining[-1, 2]]])
            self._ref_traj       = np.vstack([remaining, new_tail])
            self._ref_idx        = 0
            self._last_target_xy = new_target
            # _init_guess intentionally untouched
            return

        # ── Full rebuild ──────────────────────────────────────────────────────
        is_cold_start = self._init_guess is None

        if is_cold_start:
            origin_xy = state[:2]
        else:
            # Start new trajectory from current ref point, not robot position,
            # so ref[0] doesn't teleport and _ref_idx=0 stays valid.
            origin_xy = self._ref_traj[self._ref_idx, :2]

        waypoints_xy = np.vstack([origin_xy, waypoints_xy])
        total_dist   = np.sum(np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1))
        self._v_fwd  = np.clip(total_dist / (total_dist / 2.8), 0.3, V_MAX)
        total_time   = total_dist / self._v_fwd

        # build_reference returns wrapped headings — no alignment step needed,
        # no continuous heading to maintain, no wind-up possible.
        self._ref_traj = build_reference(waypoints_xy, DT, total_time)

        if is_cold_start:
            dists          = np.linalg.norm(self._ref_traj[:, :2] - state[:2], axis=1)
            self._ref_idx  = int(np.argmin(dists))
            self._ref_traj = self._ref_traj[self._ref_idx:]
            self._ref_idx  = 0
            self._init_guess = np.zeros(self._mpc['n_dec'])
        else:
            # Replan: trajectory starts at current ref point so index is 0
            self._ref_idx = 0
            # Keep _init_guess — solver warm-starts from previous solution

        self._last_target_xy = new_target
        self._mode           = "trajectory"

        await self._proto_6_1_reference_trajectory(self._ref_traj)

    # ─── Control loop ─────────────────────────────────────────────────────────

    async def MPC_Main(self):
        state = get_state(self._parent)   # fresh wrapped heading every tick

        if self._mode != "trajectory":
            return

        final = self._ref_traj[-1]
        dist  = np.linalg.norm(state[:2] - final[:2])

        if self._ref_idx >= len(self._ref_traj) - 1 and dist < ARRIVAL_DIST:
            # print(f"[MPC] Arrived at ({final[0]:.2f}, {final[1]:.2f}), err={dist:.3f} m")
            self._mode       = "idle"
            self._init_guess = None   # clean cold start for next mission
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
        ns       = self._mpc['n_s']
        u        = sol_flat[cs : cs + nu]

        ex = state[0] - xref[0, 0]
        ey = state[1] - xref[0, 1]
        et = np.degrees(wrap(state[2] - xref[0, 2]))

        # print(f"[MPC] idx={self._ref_idx:04d} | "
        #       f"state=({state[0]:6.2f}, {state[1]:6.2f}, {np.degrees(state[2]):7.2f}°) | "
        #       f"ref=  ({xref[0,0]:6.2f}, {xref[0,1]:6.2f}, {np.degrees(xref[0,2]):7.2f}°) | "
        #       f"err=({ex:6.3f}, {ey:6.3f}, {et:6.2f}°) | "
        #       f"u=[{u[0]:5.2f}, {u[1]:5.2f}]")

        self._init_guess = _shift_warm_start(sol_flat, ns, nu, cs)

        lookahead     = self._ref_traj[self._ref_idx:]
        dists         = np.linalg.norm(lookahead[:, :2] - state[:2], axis=1)
        closest       = int(np.argmin(dists))
        self._ref_idx = min(self._ref_idx + max(1, closest), len(self._ref_traj) - 1)

        await self._send(u)
        await self._proto_6_2_mpc_feedback(xref)

    async def _send(self, u):
        cmd = np.array(u, dtype=np.float32)
        await self.Drivetrain_pub.publish({"Id": "System", "Cmd": cmd.tolist()})

    async def DriveTrain_Sender(self):
        while True:
            await asyncio.sleep(DT)
            await self.MPC_Main()