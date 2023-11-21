import numpy as np
from numpy import pi
from numpy import cos, sin, tan
from scipy.integrate import ode
from scipy.integrate import odeint
from dynamics import Inp2Omega, Rot, Trot, diffEqn
from agents import *
import matplotlib.pyplot as plt
from params import *

# Setpoints
X_des = 0;
Y_des = 0;
Z_des = 2;




def main():
    
    phi_des = -3.78 * (np.pi / 180)
    theta_des = 5 * (np.pi / 180)
    psi_des = 10 * (np.pi / 180);
    
    Z_des = 5
    
    u1 = mass * g;
    u2 = 0;
    u3 = 0;
    u4 = 0;
    
    t0 = 0
    t1 = 20
    dt = 0.01
    t = np.arange(t0, t1, dt)
    
    sol = np.empty((len(t), 12))
    # state = [phi, theta, psi, p, q, r, u, v, w, X, Y, Z]
    state = [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0]
    TC = [5, 5, 5, 0, 0, 0]; # Terminal Conditions
    Omega, w1, w2, w3, w4 = Inp2Omega(u1, u2, u3, u4, Ct, Ctheta, l)
    params = [mass, g, Ixx, Iyy, Izz, l, J, Ct, Ctheta, u1, u2, u3, u4]
    
    solver = ode(diffEqn)
    solver.set_integrator('dop853', atol = 10**(-15), rtol = 10**(-15))
    solver.set_f_params(params)
    solver.set_initial_value(state, t0)
    iter = 1;
    sol[0] = state;
    quad = drone(state, TC, params, PID_controller_gains)
    while solver.successful() and iter < len(t):
        quad.PID_update(state, phi_des, theta_des, psi_des, Z_des, dt)
        solver.integrate(t[iter])
        
        solver.set_f_params(quad.drone_params)
        sol[iter] = solver.y
        state = solver.y
        iter = iter + 1
        #print(state)
        
    return t, sol
        
    

t, sol = main()
# Roll 
plt.subplot(2, 3, 1)
plt.plot(t, sol[:, 0])

# Pitch
plt.subplot(2, 3, 2)
plt.plot(t, sol[:, 1])

# Yaw
plt.subplot(2, 3, 3)
plt.plot(t, sol[:, 2])

plt.subplot(2, 3, 4)
plt.plot(t, sol[:, 9])

plt.subplot(2, 3, 5)
plt.plot(t, sol[:, 10])

plt.subplot(2, 3, 6)
plt.plot(t, sol[:, 11])

plt.show()
    