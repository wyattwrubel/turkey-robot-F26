import casadi as ca
import numpy as np

class DiffDriveMPC:
    def __init__(self, waypoints, dt=0.2, horizon=10, vmax=1.0, omegamax=np.pi/2):
        """
        waypoints: numpy array of shape (N,2) representing (x,y) targets
        dt: timestep
        horizon: MPC horizon (number of steps)
        vmax: max linear velocity
        omegamax: max angular velocity
        """
        self.waypoints = waypoints
        self.dt = dt
        self.N = horizon
        self.vmax = vmax
        self.omegamax = omegamax

        # Dimensions
        self.nx = 3  # [x, y, theta]
        self.nu = 2  # [v, omega]

        # Define symbolic variables
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')
        self.state = ca.vertcat(self.x, self.y, self.theta)

        self.v = ca.SX.sym('v')
        self.omega = ca.SX.sym('omega')
        self.control = ca.vertcat(self.v, self.omega)

        # Dynamics
        self.f = ca.Function('f', [self.state, self.control], 
                             [ca.vertcat(self.x + self.v*ca.cos(self.theta)*self.dt,
                                         self.y + self.v*ca.sin(self.theta)*self.dt,
                                         self.theta + self.omega*self.dt)])

        # Build MPC problem
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.N+1)  # state over horizon
        self.U = self.opti.variable(self.nu, self.N)    # control over horizon
        self.X0 = self.opti.parameter(self.nx)
        self.ref = self.opti.parameter(2, self.N)  # x,y reference

        # Cost function: sum of squared errors + control effort
        self.cost = 0
        Q = np.diag([10, 10, 1])
        R = np.diag([1, 0.5])
        for k in range(self.N):
            e = self.X[0:2,k] - self.ref[:,k]
            self.cost += ca.mtimes([e.T, Q[0:2,0:2], e])  # position error
            u = self.U[:,k]
            self.cost += ca.mtimes([u.T, R, u])           # control penalty

        self.opti.minimize(self.cost)

        # Dynamics constraints
        for k in range(self.N):
            x_next = self.X[:,k] + ca.vertcat(self.U[0,k]*ca.cos(self.X[2,k])*self.dt,
                                              self.U[0,k]*ca.sin(self.X[2,k])*self.dt,
                                              self.U[1,k]*self.dt)
            self.opti.subject_to(self.X[:,k+1] == x_next)

        # Input constraints
        self.opti.subject_to(self.opti.bounded(-vmax, self.U[0,:], vmax))
        self.opti.subject_to(self.opti.bounded(-omegamax, self.U[1,:], omegamax))

        # Initial state constraint
        self.opti.subject_to(self.X[:,0] == self.X0)

        # Solver options
        opts = {'print_time': False, 'ipopt': {'print_level': 0}}
        self.opti.solver('ipopt', opts)

    def solve(self, x0, idx_start=0):
        """
        x0: current state [x,y,theta]
        idx_start: starting index in the waypoint array
        """
        # Build reference trajectory over horizon
        N = self.N
        ref_traj = np.zeros((2,N))
        for k in range(N):
            idx = min(idx_start + k, len(self.waypoints)-1)
            ref_traj[:,k] = self.waypoints[idx]

        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.ref, ref_traj)

        sol = self.opti.solve()
        u_opt = sol.value(self.U[:,0])  # apply only first control
        x_next = sol.value(self.X[:,1])

        return u_opt, x_next
        

if __name__ == "__main__":
    # Example usage
    waypoints = np.array([
        (2, 2), (3, 3), (3, 4), (3, 5), (3, 6), (4, 7), (4, 8), (4, 9),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 11), (10, 12), (11, 12),
        (12, 12), (13, 12), (14, 12), (15, 13), (16, 12), (16, 11),
        (17, 10), (18, 9), (18, 8), (18, 7), (17, 6), (17, 5)
    ])
    mpc = DiffDriveMPC(waypoints, dt=0.2, horizon=10)

    # Simulate from initial state
    x = np.array([2.0, 2.0, 0.0])
    idx = 0
    traj = [x.copy()]
    for t in range(100):
        u, x = mpc.solve(x, idx)
        traj.append(x.copy())
        # advance waypoint index if close enough
        if np.linalg.norm(x[0:2]-waypoints[idx]) < 0.3 and idx < len(waypoints)-1:
            idx += 1

    traj = np.array(traj)
    print(traj)