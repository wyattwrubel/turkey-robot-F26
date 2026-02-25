import numpy as np

def differential_drive(x, u, l, dt=0.1):
    # x | [x,y,theta]  u | [vL, vR]  l = distance between wheels
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + dt * (np.cos(x[2])/2 * u[0] + np.cos(x[2])/2 * u[1])
    x_next[1] = x[1] + dt * (np.sin(x[2])/2 * u[0] + np.sin(x[2])/2 * u[1])
    x_next[2] = x[2] + dt * 1/l * (u[1] - u[0])

    return x_next
