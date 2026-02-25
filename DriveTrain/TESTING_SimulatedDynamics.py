import numpy as np
import matplotlib.pyplot as plt

def differential_drive(x, u, l, dt=0.1):
    # x | [x,y,theta]  u | [vL, vR]  l = distance between wheels
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + dt * (np.cos(x[2])/2 * u[0] + np.cos(x[2])/2 * u[1])
    x_next[1] = x[1] + dt * (np.sin(x[2])/2 * u[0] + np.sin(x[2])/2 * u[1])
    x_next[2] = x[2] + dt * 1/l * (u[1] - u[0])
    return x_next

# ----------------- Simulation parameters -----------------
dt = 0.1               # timestep
t_final = 10.0          # total simulation time
l = 0.5                # track width (meters)
x = np.array([0.0, 0.0, 0.0])  # initial state
u = np.array([2, 2])       # left/right wheel velocities (m/s)

# storage for plotting
states = [x.copy()]
time = [0.0]

# ----------------- Simulation loop -----------------
t = 0.0
while t < t_final:

    if t < 3 and t > 2:
        u = np.array([1, 2])

    x = differential_drive(x, u, l, dt)
    states.append(x.copy())
    t += dt
    time.append(t)

states = np.array(states)

# ----------------- Plotting -----------------
plt.figure(figsize=(18,4))

# Trajectory plot
plt.subplot(1,2,1)
plt.plot(states[:,0], states[:,1], marker='o')
plt.xlabel("X position [m]")
plt.ylabel("Y position [m]")
plt.title("Robot Trajectory")
plt.axis("equal")

# Heading plot
plt.subplot(1,2,2)
plt.plot(time, states[:,2])
plt.xlabel("Time [s]")
plt.ylabel("Heading [rad]")
plt.title("Robot Heading over Time")

plt.tight_layout()
plt.show()