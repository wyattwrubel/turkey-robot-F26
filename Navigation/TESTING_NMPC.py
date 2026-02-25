import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time

# ===== Parameters =====
start_time = time.perf_counter()
dt = 0.1
N = 10
nx = 3
nu = 2
sim_time = 15
steps = int(sim_time / dt)

# ===== Model =====
x = ca.MX.sym("x", nx)
u = ca.MX.sym("u", nu)

f = ca.vertcat(
    u[0]*ca.cos(x[2]),
    u[0]*ca.sin(x[2]),
    u[1]
)

x_next = x + dt*f
f_d = ca.Function("f_d", [x,u], [x_next])

# ===== Decision Variables =====
X = ca.MX.sym("X", nx, N+1)
U = ca.MX.sym("U", nu, N)
P = ca.MX.sym("P", nx + (N+1)*nx)

Q = np.diag([10,10,2])
R = np.diag([0.1,0.1])

obj = 0
g = []

# initial condition
g.append(X[:,0] - P[0:nx])

for k in range(N):
    xk = X[:,k]
    uk = U[:,k]
    xref = P[nx + k*nx : nx + (k+1)*nx]

    obj += ca.mtimes([(xk-xref).T, Q, (xk-xref)])
    obj += ca.mtimes([uk.T, R, uk])

    g.append(X[:,k+1] - f_d(xk,uk))

g = ca.vertcat(*g)

OPT = ca.vertcat(
    ca.reshape(X, nx*(N+1), 1),
    ca.reshape(U, nu*N, 1)
)

nlp = {"f": obj, "x": OPT, "g": g, "p": P}

solver = ca.nlpsol("solver", "ipopt", nlp,
                   {"ipopt.print_level":0,
                    "print_time":0})

# ===== Bounds =====
x_min = [-10,-10,-np.pi]
x_max = [10,10,np.pi]
u_min = [-2,-2]
u_max = [2,2]

lbx = []
ubx = []

for _ in range(N+1):
    lbx += x_min
    ubx += x_max

for _ in range(N):
    lbx += u_min
    ubx += u_max

lbx = np.array(lbx)
ubx = np.array(ubx)

lbg = np.zeros(g.shape[0])
ubg = np.zeros(g.shape[0])

# ===== Reference Trajectory =====
def ref(t):
    r = 4
    w = 0.3
    return np.array([
        r*np.cos(w*t),
        r*np.sin(w*t),
        w*t + np.pi/2
    ])

# ===== Simulation =====
x_current = np.array([0.,0.,0.])

state_hist = []
ref_hist = []

init_guess = np.zeros(OPT.shape[0])

for i in range(steps):

    t0 = i*dt
    ref_traj = np.array([ref(t0+k*dt) for k in range(N+1)])

    p = np.concatenate([x_current, ref_traj.flatten()])

    sol = solver(
        x0=init_guess,
        lbx=lbx,
        ubx=ubx,
        lbg=lbg,
        ubg=ubg,
        p=p
    )
    if i == 1:
        end_time = time.perf_counter()
        print(f"Execution time was {end_time - start_time} seconds")

    sol_x = sol["x"].full().flatten()

    u_opt = sol_x[nx*(N+1): nx*(N+1)+nu]

    x_current = np.array(f_d(x_current, u_opt).full()).flatten()

    state_hist.append(x_current)
    ref_hist.append(ref(t0))

    init_guess = sol_x  # warm start

state_hist = np.array(state_hist)
ref_hist = np.array(ref_hist)


# ===== Plots =====
plt.figure()
plt.plot(ref_hist[:,0], ref_hist[:,1], '--')
plt.plot(state_hist[:,0], state_hist[:,1])
plt.title("NMPC Trajectory Tracking")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

plt.figure()
plt.plot(state_hist[:,0] - ref_hist[:,0])
plt.title("X Tracking Error")
plt.grid()

plt.figure()
plt.plot(state_hist[:,1] - ref_hist[:,1])
plt.title("Y Tracking Error")
plt.grid()

plt.show()