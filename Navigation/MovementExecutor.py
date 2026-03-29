import asyncio
import numpy as np
import sys
sys.path.insert(0, '/home/turkeysquad/ME450/MainDesign/casadi')
# Also remove the venv casadi if it snuck in already
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

N  = 20                        # MPC horizon steps
Q  = np.diag([5.0, 5.0, 1.5])
R  = np.diag([0.5, 0.5])
L  = WHEELBASE

# ─────────────────────────────────────────────
# Helpers  (stateless, no Ghost)
# ─────────────────────────────────────────────
def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

def heading_to(from_xy, to_xy):
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]   # already positive downward
    return np.arctan2(-dy, dx)    # ← remove the negative sign that was here

def path_to_refs(path):
    """Convert a list of (x,y) waypoints into (x, y, theta) reference rows."""
    pts  = [(float(x), float(y)) for x, y in path]
    refs = []
    for i, (x, y) in enumerate(pts):
        if i < len(pts) - 1:
            th = heading_to((x, y), pts[i + 1])
        else:
            th = refs[-1][2] if refs else 0.0
        refs.append(np.array([x, y, th]))
    return refs

def build_reference(waypoints_xy: np.ndarray, dt: float, total_time: float) -> np.ndarray:
    # Remove duplicate or near-duplicate points that cause interp1d to fail
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

    # Use cubic if enough points, fall back to linear otherwise
    kind = 'cubic' if len(waypoints_xy) >= 4 else 'linear'

    xs = interp1d(arc_n, waypoints_xy[:, 0], kind=kind)(t_n)
    ys = interp1d(arc_n, waypoints_xy[:, 1], kind=kind)(t_n)

    dx  = np.diff(xs, append=xs[-1])
    dy  = np.diff(ys, append=ys[-1])
    ths = np.arctan2(-dy, dx)
    ths = np.unwrap(ths)

    return np.column_stack([xs, ys, ths])

def nominal_input_from_ref(ref_slice: np.ndarray, v_fwd: float) -> np.ndarray:
    """
    Invert the kinematic model to get feed-forward (vl, vr) along the reference.
    v_fwd is the average forward speed (derived from path length / total time).
    """
    u_ref = np.zeros((N, 2))
    for k in range(N):
        dth       = ref_slice[k+1, 2] - ref_slice[k, 2] if k+1 < len(ref_slice) else 0.0
        omega_ref = wrap(dth) / DT
        vr        = np.clip(v_fwd + omega_ref * L / 2, V_MIN, V_MAX)
        vl        = np.clip(v_fwd - omega_ref * L / 2, V_MIN, V_MAX)
        u_ref[k]  = [vl, vr]
    return u_ref

def build_param_vector(x_cur, ref_idx, ref_traj, v_fwd):
    """Slice reference horizon and pack the IPOPT parameter vector."""
    end  = min(ref_idx + N + 1, len(ref_traj))
    pad  = N + 1 - (end - ref_idx)
    xref = np.vstack([ref_traj[ref_idx:end],
                      np.tile(ref_traj[-1], (pad, 1))])
    u_ref = nominal_input_from_ref(xref, v_fwd)
    p     = np.concatenate([x_cur, xref.flatten(), u_ref.flatten()])
    return p, xref, u_ref

# ─────────────────────────────────────────────
# MPC solver  (built once, reused every step)
# ─────────────────────────────────────────────
def build_mpc_solver():
    """
    Construct and return the CasADi IPOPT solver.
    Called once at startup — structure never changes, only p changes per step.
    """
    n_states, n_controls = 3, 2

    # Symbolic state / input
    x_s, y_s, th_s = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('theta')
    vl_s, vr_s      = ca.SX.sym('vl'), ca.SX.sym('vr')
    states  = ca.vertcat(x_s, y_s, th_s)
    inputs  = ca.vertcat(vl_s, vr_s)

    # Discrete nonlinear dynamics (Euler)
    v         = (vl_s + vr_s) / 2
    omega     = (vr_s - vl_s) / L
    f_expr    = ca.vertcat(v * ca.cos(th_s), -v * ca.sin(th_s), omega)
    f_disc    = ca.Function('f', [states, inputs], [states + DT * f_expr])

    # Decision variables
    X_mpc = ca.SX.sym('X', n_states,   N + 1)
    U_mpc = ca.SX.sym('U', n_controls, N)

    # Parameters: initial state + reference states + reference inputs
    X0_p   = ca.SX.sym('X0',   n_states)
    Xref_p = ca.SX.sym('Xref', n_states,   N + 1)
    Uref_p = ca.SX.sym('Uref', n_controls, N)
    p_all  = ca.vertcat(X0_p,
                        ca.reshape(Xref_p, -1, 1),
                        ca.reshape(Uref_p, -1, 1))

    # Cost + constraints
    obj = 0
    g   = [X_mpc[:, 0] - X0_p]

    for k in range(N):
        dx   = X_mpc[0, k] - Xref_p[0, k]
        dy   = X_mpc[1, k] - Xref_p[1, k]
        dth  = X_mpc[2, k] - Xref_p[2, k]
        dth_w = dth - 2*ca.pi * ca.floor((dth + ca.pi) / (2*ca.pi))

        e_x  = ca.vertcat(dx, dy, dth_w)
        e_u  = U_mpc[:, k] - Uref_p[:, k]
        obj += ca.mtimes([e_x.T, Q, e_x]) + ca.mtimes([e_u.T, R, e_u])
        g.append(X_mpc[:, k+1] - f_disc(X_mpc[:, k], U_mpc[:, k]))

    # Terminal cost
    dth_N  = X_mpc[2, N] - Xref_p[2, N]
    dth_wN = dth_N - 2*ca.pi * ca.floor((dth_N + ca.pi) / (2*ca.pi))
    e_xN   = ca.vertcat(X_mpc[0, N] - Xref_p[0, N],
                        X_mpc[1, N] - Xref_p[1, N],
                        dth_wN)
    obj += ca.mtimes([e_xN.T, Q, e_xN])

    # Bounds
    opt_vars  = ca.vertcat(ca.reshape(X_mpc, -1, 1), ca.reshape(U_mpc, -1, 1))
    n_dec     = opt_vars.numel()
    lbx       = -np.inf * np.ones(n_dec)
    ubx       =  np.inf * np.ones(n_dec)
    ctrl_start = n_states * (N + 1)
    for i in range(N):
        for j in range(n_controls):
            idx      = ctrl_start + i * n_controls + j
            lbx[idx] = V_MIN
            ubx[idx] = V_MAX

    g_concat = ca.vertcat(*g)
    n_g      = g_concat.numel()

    nlp = {'f': obj, 'x': opt_vars, 'g': g_concat, 'p': p_all}
    solver = ca.nlpsol('solver', 'ipopt', nlp,
                       {'ipopt.print_level': 0, 'print_time': 0})

    meta = {
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
    return meta

# ─────────────────────────────────────────────
# Main controller class
# ─────────────────────────────────────────────
class MPC_loop:

    def __init__(self, parent, Drivetrain_pub, MPCFeedback_Navigation_pub):
        self.Drivetrain_pub  = Drivetrain_pub
        self.MPCFeedback_pub = MPCFeedback_Navigation_pub
        self._parent         = parent

        self._mode       = "idle"
        self._ref_traj   = None   # dense (M, 3) reference
        self._ref_idx    = 0      # current position along ref_traj
        self._v_fwd      = 0.5   # updated when a new path is loaded
        self._init_guess = None   # warm-start vector

        print("[MPC] Building solver…")
        self._mpc = build_mpc_solver()
        print("[MPC] Solver ready.")

    # ─────────────────────────────────────────────
    # Protocol 6 — MPC Feedback Protocol
    # ─────────────────────────────────────────────

    async def _proto_6_1_reference_trajectory(self, ref_traj: np.ndarray):
        """6.1 ReferenceTrajectory — fires once on new Move command."""
        await self.MPCFeedback_pub.publish({
            "Id":  "Reference",
            "Cmd": ref_traj[:, :2].tolist(),
        })

    async def _proto_6_2_mpc_feedback(self, xref_slice: np.ndarray):
        """6.2 FeedbackMPC — fires every control timestep."""
        await self.MPCFeedback_pub.publish({
            "Id":  "FeedbackMPC",
            "Cmd": xref_slice[:, :2].tolist(),
        })

    async def _proto_6_3_clear_trajectory(self):
        """6.3 ClearTrajectory — fires on arrival."""
        await self.MPCFeedback_pub.publish({
            "Id": "ClearTrajectory",
        })

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

        total_dist  = np.sum(np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1))
        self._v_fwd = np.clip(total_dist / (total_dist / 2.7), 0.3, V_MAX)  # target 2.7 m/s
        total_time  = total_dist / self._v_fwd + 5.0

        self._ref_traj   = build_reference(waypoints_xy, DT, total_time)
        self._ref_idx    = 0
        self._init_guess = np.zeros(self._mpc['n_dec'])
        self._mode       = "trajectory"

        await self._proto_6_1_reference_trajectory(self._ref_traj)
    # ─────────────────────────────────────────────
    # Control loop
    # ─────────────────────────────────────────────

    async def MPC_Main(self):
        state = np.array(self._parent.Current_State, dtype=np.float64)

        if self._mode == "trajectory":
            final = self._ref_traj[-1]
            dist  = np.linalg.norm(state[:2] - final[:2])

            # Arrival check
            if self._ref_idx >= len(self._ref_traj) - 1 and dist < ARRIVAL_DIST:
                print(f"[MPC] Arrived at ({final[0]:.2f}, {final[1]:.2f}), err={dist:.3f} m")
                self._mode = "idle"
                await self._send(np.zeros(2))
                await self._proto_6_3_clear_trajectory()
                return

            # Build parameter vector for this timestep
            p_vec, xref_slice, _ = build_param_vector(
                state, self._ref_idx, self._ref_traj, self._v_fwd
            )

            # Solve
            sol = self._mpc['solver'](
                x0   = self._init_guess,
                lbx  = self._mpc['lbx'],
                ubx  = self._mpc['ubx'],
                lbg  = self._mpc['lbg'],
                ubg  = self._mpc['ubg'],
                p    = p_vec,
            )

            sol_flat = sol['x'].full().flatten()
            cs       = self._mpc['ctrl_start']
            nc       = self._mpc['n_controls']
            u        = sol_flat[cs : cs + nc]

            # ── debug ──────────────────────────────────────────────
            p_vec_dbg  = p_vec  # already built above
            xref_now   = xref_slice[0]
            xref_next  = xref_slice[1] if len(xref_slice) > 1 else xref_slice[0]

            print(f"[MPC] idx={self._ref_idx:04d} | "
                f"state=({state[0]:6.2f}, {state[1]:6.2f}, {np.degrees(state[2]):7.2f}°) | "
                f"ref=  ({xref_now[0]:6.2f}, {xref_now[1]:6.2f}, {np.degrees(xref_now[2]):7.2f}°) | "
                f"u=[{u[0]:5.2f}, {u[1]:5.2f}] | "
                f"hdg_err={np.degrees(wrap(state[2] - xref_now[2])):7.2f}°")
            # ── end debug ──────────────────────────────────────────

            # Warm-start next iteration with current solution
            self._init_guess = sol_flat

            # Advance reference index by one step
            self._ref_idx = min(self._ref_idx + 1, len(self._ref_traj) - 1)

            await self._send(u)
            await self._proto_6_2_mpc_feedback(xref_slice)

        elif self._mode == "fixed":
            print("[MPC] Fixed/orient mode — WIP")
            self._mode = "idle"

    async def _send(self, u, **_):
        """Publish wheel velocities to the drivetrain."""
        cmd = np.array(u, dtype=np.float32)
        await self.Drivetrain_pub.publish({"Id": "System", "Cmd": cmd.tolist()})

    async def DriveTrain_Sender(self):
        while True:
            await asyncio.sleep(DT)
            await self.MPC_Main()