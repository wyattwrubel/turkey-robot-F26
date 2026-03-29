import asyncio
import os
import contextlib
import numpy as np

# --- Config ---
WHEELBASE    = 0.7
DT           = 0.1
V_MAX        = 3.0
V_MIN        = -3.0
ARRIVAL_DIST = 0.5
GHOST_SPEED  = 2.8

# Proportional unicycle gains
K_V  = 1.2   # forward speed per meter of position error
K_W  = 3.0   # turn rate per radian of heading error
K_LA = 0.8   # lookahead blend: how much to chase ghost vs point-to-goal


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

def heading_to(from_xy, to_xy):
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return np.arctan2(-dy, dx)

def unicycle_to_wheels(v, omega):
    vl = np.clip(v - omega * WHEELBASE / 2, V_MIN, V_MAX)
    vr = np.clip(v + omega * WHEELBASE / 2, V_MIN, V_MAX)
    return np.array([vl, vr])

def path_to_refs(path):
    pts = [(float(x), float(y)) for x, y in path]
    refs = []
    for i, (x, y) in enumerate(pts):
        if i < len(pts) - 1:
            th = heading_to((x, y), pts[i + 1])
        else:
            th = refs[-1][2] if refs else 0.0
        refs.append(np.array([x, y, th]))
    return refs


class Ghost:
    def __init__(self):
        self._refs = []
        self._seg  = 0
        self._t    = 0.0
        self.done  = False

    def load(self, refs):
        self._refs = refs
        self._seg  = 0
        self._t    = 0.0
        self.done  = False

    def _seg_dur(self, i):
        l = np.linalg.norm(self._refs[i+1][:2] - self._refs[i][:2])
        return max(l / GHOST_SPEED, 1e-6)

    def _current(self):
        if self._seg >= len(self._refs) - 1:
            return self._refs[-1].copy()
        dur   = self._seg_dur(self._seg)
        alpha = np.clip(self._t / dur, 0.0, 1.0)
        p, q  = self._refs[self._seg], self._refs[self._seg + 1]
        return np.array([
            p[0] + alpha * (q[0] - p[0]),
            p[1] + alpha * (q[1] - p[1]),
            q[2],
        ])

    def advance(self):
        if self.done:
            return
        self._t += DT
        while self._seg < len(self._refs) - 1:
            if self._t <= self._seg_dur(self._seg):
                break
            self._t   -= self._seg_dur(self._seg)
            self._seg += 1
        if self._seg >= len(self._refs) - 1:
            self.done = True

    def lookahead(self, n_steps):
        """Return ghost position n_steps ahead without advancing."""
        seg, t, done = self._seg, self._t, self.done
        for _ in range(n_steps):
            self.advance()
        target = self._current()
        self._seg, self._t, self.done = seg, t, done
        return target


def control(state, ghost_current, ghost_ahead):
    """
    Pure nonproprietary nonlinear tracking controller.
    Blends two targets:
      1. ghost_current — the immediate reference (keeps robot on path)
      2. ghost_ahead   — a lookahead point (gives smooth curved approach)

    This is the standard pure-pursuit + heading control used in real
    differential drive robots. No linearization, no QP, always stable.
    """
    x, y, th = state

    # Vector to current ghost position
    dx_c = ghost_current[0] - x
    dy_c = ghost_current[1] - y
    dist_c = np.hypot(dx_c, dy_c)

    # Vector to lookahead ghost position
    dx_a = ghost_ahead[0] - x
    dy_a = ghost_ahead[1] - y
    dist_a = np.hypot(dx_a, dy_a)

    # Heading toward lookahead point (pure pursuit arc)
    th_pursuit = heading_to((x, y), ghost_ahead[:2])

    # Reference heading from path
    th_ref = ghost_current[2]

    # Blend: when close to ghost, weight toward path heading;
    # when far, weight toward pursuit heading to cut back to path
    alpha = np.clip(dist_c / 2.0, 0.0, 1.0)  # 0=on path, 1=far off
    th_target = wrap((1 - alpha) * th_ref + alpha * th_pursuit)

    # Heading error
    h_err = wrap(th_target - th)

    # Forward speed: full speed when heading is good, reduce when turning hard
    # This prevents the robot from driving off path while trying to turn
    heading_factor = np.cos(h_err) ** 2   # smooth speed reduction for large angles
    v = np.clip(K_V * max(dist_c, dist_a * 0.5) * heading_factor, 0.2, V_MAX)

    # Turn rate
    omega = np.clip(K_W * h_err, -2 * V_MAX / WHEELBASE, 2 * V_MAX / WHEELBASE)

    return unicycle_to_wheels(v, omega)


class MPC_loop:

    def __init__(self, parent, Drivetrain_pub):
        self.Drivetrain_pub = Drivetrain_pub
        self._parent = parent
        self._last_u = np.zeros(2)
        self._mode   = "idle"
        self._goal   = np.zeros(3)
        self._ghost  = Ghost()

    def Orient_CMD_Receiver(self, data):
        self._goal   = np.array(data, dtype=np.float64)
        self._last_u = np.zeros(2)
        self._mode   = "fixed"

    def Move_CMD_Receiver(self, data):
        robot    = np.array(self._parent.Current_State, dtype=np.float64)
        refs     = path_to_refs(data)
        start_th = heading_to(robot[:2], refs[0][:2])
        refs     = [np.array([robot[0], robot[1], start_th])] + refs
        self._ghost.load(refs)
        self._last_u = np.zeros(2)
        self._mode   = "trajectory"

    async def MPC_Main(self):
        state = np.array(self._parent.Current_State, dtype=np.float64)

        if self._mode == "trajectory":
            final = self._ghost._refs[-1]
            dist  = np.linalg.norm(state[:2] - final[:2])

            if self._ghost.done and dist < ARRIVAL_DIST:
                print(f"[CTRL] Arrived at ({final[0]:.2f}, {final[1]:.2f}), err={dist:.3f}m")
                self._last_u = np.zeros(2)
                self._mode   = "idle"
                await self._send(np.zeros(2))
                return

            ghost_now   = self._ghost.lookahead(0)
            ghost_ahead = self._ghost.lookahead(8)   # ~0.8s lookahead

            u = control(state, ghost_now, ghost_ahead)
            self._last_u = u
            await self._send(u, ghost_now, state)
            self._ghost.advance()

        elif self._mode == "fixed":
            dist   = np.linalg.norm(state[:2] - self._goal[:2])
            err_th = abs(wrap(state[2] - self._goal[2]))
            if dist < ARRIVAL_DIST and err_th < np.radians(5):
                print(f"[CTRL] Goal reached, err={dist:.3f}m")
                self._last_u = np.zeros(2)
                self._mode   = "idle"
                await self._send(np.zeros(2))
                return
            # Point toward goal with heading correction
            th_goal  = heading_to(state[:2], self._goal[:2])
            h_err    = wrap(th_goal - state[2])
            hf       = np.cos(h_err) ** 2
            v        = np.clip(K_V * dist * hf, 0.2, V_MAX)
            omega    = np.clip(K_W * h_err, -2*V_MAX/WHEELBASE, 2*V_MAX/WHEELBASE)
            u        = unicycle_to_wheels(v, omega)
            self._last_u = u
            await self._send(u, self._goal, state)

    async def _send(self, u, ref=None, state=None):
        cmd = np.array(u, dtype=np.float32)
        if ref is not None and state is not None:
            err = np.linalg.norm(state[:2] - ref[:2])
            print(f"[DriveCommand | Cmd: {cmd} | "
                  f"Pos: ({state[0]:.2f}, {state[1]:.2f}) θ={np.degrees(state[2]):.1f}° | "
                  f"Ref: ({ref[0]:.2f}, {ref[1]:.2f}) θ={np.degrees(ref[2]):.1f}° | "
                  f"Err: {err:.2f}m]")
        else:
            print(f"[DriveCommand | Cmd: {cmd}]")
        await self.Drivetrain_pub.publish({"Id": "System", "Cmd": cmd.tolist()})

    async def DriveTrain_Sender(self):
        while True:
            await asyncio.sleep(DT)
            await self.MPC_Main()