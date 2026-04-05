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
L            = WHEELBASE
DT           = 0.1
V_MAX        = 3.0
V_MIN        = -3.0
ARRIVAL_DIST = 0.1
N            = 10

Q  = np.diag([10.0, 10.0, 15.0])
Qf = 3.0 * Q
R  = np.diag([3.0, 3.0])


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def get_state(parent) -> np.ndarray:
    raw = np.array(parent.Current_State, dtype=np.float64)
    return np.array([raw[0], raw[1], wrap(raw[2])])


def build_reference(waypoints_xy: np.ndarray, dt: float, total_time: float) -> np.ndarray:
    """Interpolate waypoints into a dense (x, y, theta) reference."""
    # Drop duplicate consecutive points
    keep = np.ones(len(waypoints_xy), dtype=bool)
    for i in range(1, len(waypoints_xy)):
        if np.linalg.norm(waypoints_xy[i] - waypoints_xy[i - 1]) < 1e-6:
            keep[i] = False
    waypoints_xy = waypoints_xy[keep]

    if len(waypoints_xy) < 2:
        row = np.array([waypoints_xy[0][0], waypoints_xy[0][1], 0.0])
        return np.tile(row, (max(1, int(total_time / dt)), 1))

    dists = np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1)
    arc   = np.concatenate([[0], np.cumsum(dists)])
    arc_n = arc / arc[-1]

    t_dense = np.arange(0, total_time, dt)
    t_n     = (t_dense / total_time).clip(0, 1)
    kind    = 'cubic' if len(waypoints_xy) >= 4 else 'linear'

    xs  = interp1d(arc_n, waypoints_xy[:, 0], kind=kind)(t_n)
    ys  = interp1d(arc_n, waypoints_xy[:, 1], kind=kind)(t_n)
    ths = np.arctan2(-np.gradient(ys), np.gradient(xs))
    ths[-1] = ths[-2]

    return np.column_stack([xs, ys, ths])


def get_ref_slice(ref_idx, ref_traj, v_fwd):
    """Extract an N+1 reference window and compute feedforward u_ref."""
    end  = min(ref_idx + N + 1, len(ref_traj))
    pad  = (N + 1) - (end - ref_idx)
    xref = np.vstack([ref_traj[ref_idx:end], np.tile(ref_traj[-1], (pad, 1))])

    u_ref = np.zeros((N, 2))
    for k in range(N):
        dth       = wrap(xref[k + 1, 2] - xref[k, 2])
        omega_ref = dth / DT
        u_ref[k]  = [
            np.clip(v_fwd - omega_ref * L / 2, V_MIN, V_MAX),
            np.clip(v_fwd + omega_ref * L / 2, V_MIN, V_MAX),
        ]
    return xref, u_ref


# ─────────────────────────────────────────────
# MPC solver
# ─────────────────────────────────────────────
def build_mpc_solver():
    n_s, n_u = 3, 2

    # Symbolic variables
    states = ca.SX.sym('s', n_s)
    inputs = ca.SX.sym('u', n_u)
    x_s, y_s, th_s = states[0], states[1], states[2]
    vl_s, vr_s     = inputs[0], inputs[1]

    v     = (vl_s + vr_s) / 2
    omega = (vr_s - vl_s) / L
    f_disc = ca.Function('f', [states, inputs], [
        states + DT * ca.vertcat(v * ca.cos(th_s), -v * ca.sin(th_s), omega)
    ])

    # Decision variables
    X_mpc  = ca.SX.sym('X',    n_s, N + 1)
    U_mpc  = ca.SX.sym('U',    n_u, N)
    X0_p   = ca.SX.sym('X0',   n_s)
    Xref_p = ca.SX.sym('Xref', n_s, N + 1)
    Uref_p = ca.SX.sym('Uref', n_u, N)
    p_all  = ca.vertcat(X0_p,
                        ca.reshape(Xref_p, -1, 1),
                        ca.reshape(Uref_p, -1, 1))

    def ang_err(a, b):
        d = a - b
        return ca.atan2(ca.sin(d), ca.cos(d))

    # Cost and equality constraints
    obj = 0
    g   = [X_mpc[:, 0] - X0_p]          # initial state constraint

    for k in range(N):
        ex  = X_mpc[0, k] - Xref_p[0, k]
        ey  = X_mpc[1, k] - Xref_p[1, k]
        eth = ang_err(X_mpc[2, k], Xref_p[2, k])
        eu  = U_mpc[:, k] - Uref_p[:, k]
        obj += (Q[0,0]*ex**2 + Q[1,1]*ey**2 + Q[2,2]*eth**2
                + R[0,0]*eu[0]**2 + R[1,1]*eu[1]**2)
        g.append(X_mpc[:, k + 1] - f_disc(X_mpc[:, k], U_mpc[:, k]))

    # Terminal cost
    ex  = X_mpc[0, N] - Xref_p[0, N]
    ey  = X_mpc[1, N] - Xref_p[1, N]
    eth = ang_err(X_mpc[2, N], Xref_p[2, N])
    obj += Qf[0,0]*ex**2 + Qf[1,1]*ey**2 + Qf[2,2]*eth**2

    # Pack and set bounds
    opt_vars = ca.vertcat(ca.reshape(X_mpc, -1, 1), ca.reshape(U_mpc, -1, 1))
    n_dec    = opt_vars.numel()
    lbx      = -np.inf * np.ones(n_dec)
    ubx      =  np.inf * np.ones(n_dec)
    cs       = n_s * (N + 1)                  # index where controls start
    lbx[cs:] = V_MIN
    ubx[cs:] = V_MAX

    g_cat = ca.vertcat(*g)
    solver = ca.nlpsol('solver', 'ipopt',
                       {'f': obj, 'x': opt_vars, 'g': g_cat, 'p': p_all},
                       {'ipopt.print_level': 0, 'print_time': 0,
                        'ipopt.tol': 1e-3, 'ipopt.max_iter': 50})

    return dict(solver=solver, lbx=lbx, ubx=ubx,
                lbg=np.zeros(g_cat.numel()), ubg=np.zeros(g_cat.numel()),
                n_dec=n_dec, ctrl_start=cs, n_s=n_s, n_u=n_u)


# ─────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────
class MPC_loop:

    def __init__(self, parent, Drivetrain_pub, MPCFeedback_Navigation_pub):
        self.Drivetrain_pub  = Drivetrain_pub
        self.MPCFeedback_pub = MPCFeedback_Navigation_pub
        self._parent         = parent

        self._mode    = "idle"
        self._ref_traj = None
        self._ref_idx  = 0
        self._v_fwd    = 0.5

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
        self._mode     = "idle"
        self._ref_traj = None
        target_hdg = wrap(float(data[2]))
        KP = 3.0

        while True:
            state = get_state(self._parent)
            err   = wrap(target_hdg - state[2])
            if abs(err) < np.radians(3.0):
                await self._send(np.zeros(2))
                return
            omega = KP * err
            await self._send(np.array([
                np.clip(-omega * L / 2, V_MIN, V_MAX),
                np.clip( omega * L / 2, V_MIN, V_MAX),
            ]))
            await asyncio.sleep(DT)

    async def Move_CMD_Receiver(self, data):
        state        = get_state(self._parent)
        waypoints_xy = np.array([[float(x), float(y)] for x, y in data])

        # Always do a full rebuild from current robot position
        waypoints_xy = np.vstack([state[:2], waypoints_xy])
        total_dist   = np.sum(np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1))
        self._v_fwd  = np.clip(total_dist / (total_dist / 2.8), 0.3, V_MAX)
        total_time   = total_dist / self._v_fwd

        self._ref_traj = build_reference(waypoints_xy, DT, total_time)
        self._ref_idx  = 0
        self._mode     = "trajectory"

        await self._proto_6_1_reference_trajectory(self._ref_traj)

    # ─── Control loop ─────────────────────────────────────────────────────────

    async def MPC_Main(self):
        if self._mode != "trajectory":
            return

        state = get_state(self._parent)
        final = self._ref_traj[-1]

        # Arrival check
        if (self._ref_idx >= len(self._ref_traj) - 1
                and np.linalg.norm(state[:2] - final[:2]) < ARRIVAL_DIST):
            print(f"[MPC] Arrived at ({final[0]:.2f}, {final[1]:.2f})")
            self._mode = "idle"
            await self._send(np.zeros(2))
            await self._proto_6_3_clear_trajectory()
            return

        xref, u_ref = get_ref_slice(self._ref_idx, self._ref_traj, self._v_fwd)
        p_vec = np.concatenate([state, xref.flatten(), u_ref.flatten()])

        sol = self._mpc['solver'](
            x0  = np.zeros(self._mpc['n_dec']),   # cold start every tick
            lbx = self._mpc['lbx'],
            ubx = self._mpc['ubx'],
            lbg = self._mpc['lbg'],
            ubg = self._mpc['ubg'],
            p   = p_vec,
        )

        sol_flat = sol['x'].full().flatten()
        cs, nu   = self._mpc['ctrl_start'], self._mpc['n_u']
        u        = sol_flat[cs : cs + nu]

        # Advance ref index to closest point on remaining trajectory
        dists         = np.linalg.norm(self._ref_traj[self._ref_idx:, :2] - state[:2], axis=1)
        self._ref_idx = min(self._ref_idx + int(np.argmin(dists)), len(self._ref_traj) - 1)

        await self._send(u)
        await self._proto_6_2_mpc_feedback(xref)

    async def _send(self, u):
        await self.Drivetrain_pub.publish({
            "Id": "System", "Cmd": np.array(u, dtype=np.float32).tolist()
        })

    async def DriveTrain_Sender(self):
        while True:
            await asyncio.sleep(DT)
            await self.MPC_Main()