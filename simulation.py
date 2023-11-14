import numpy as np
from numpy import pi
from numpy import cos, sin, tan
from scipy.integrate import ode
from scipy.integrate import odeint
from dynamics import Inp2Omega, Rot, Trot, diffEqn
import matplotlib.pyplot as plt
from params import *

# Setpoints
X_des = 0;
Y_des = 0;
Z_des = 2;




def main():
    
    phi_des = 5 * (np.pi / 180)
    theta_des = 5 * (np.pi / 180)
    psi_des = 10 * (np.pi / 180);
    
    
    Z_err_int = 0; Z_err_dot = 0; Z_err_prev = 0;
    phi_err_int = 0; phi_err_dot = 0; phi_err_prev = 0;
    theta_err_int = 0; theta_err_dot = 0; theta_err_prev = 0;
    psi_err_int = 0; psi_err_dot = 0; psi_err_prev = 0;

    p_err_int = 0; p_err_dot = 0; p_err_prev = 0;
    q_err_int = 0; q_err_dot = 0; q_err_prev = 0;
    r_err_int = 0; r_err_dot = 0; r_err_prev = 0;

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
    
    Omega, w1, w2, w3, w4 = Inp2Omega(u1, u2, u3, u4, Ct, Ctheta, l)
    params = [mass, g, Ixx, Iyy, Izz, l, J, Ct, Ctheta, u1, u2, u3, u4]
    
    solver = ode(diffEqn)
    solver.set_integrator('dop853', atol = 10**(-15), rtol = 10**(-15))
    solver.set_f_params(params)
    solver.set_initial_value(state, t0)
    iter = 1;
    sol[0] = state;
    
    while solver.successful() and iter < len(t):
        # Update Parameters
        #------------------- PID loop
        # Keep track of X/Y_err_int and X_err and dt and X/Y_err_prev
        # z_err_prev, z_err_int, z_err,
        phi = state[0]; theta = state[1]; psi = state[2];
        p = state[3]; q = state[4]; r = state[5];
        u = state[6]; v = state[7]; w = state[8];
        X = state[9]; Y = state[10]; Z = state[11];
        
        R_mat = Rot(phi, theta, psi)
        T_mat = Trot(phi, theta);
        
        Z_err = Z_des - Z;
        Z_err_int = Z_err_int + (Z_err*dt);
        Z_err_dot = (Z_err - Z_err_prev) / dt;
        Z_err_prev = Z_err;
        
        u1 = (1/(cos(phi)*cos(theta)))*((Kp_Z * Z_err) + (Ki_Z * Z_err_int) + (Kd_Z * Z_err_dot) + (mass*g));
        
        
        phi_err = phi_des - phi;
        phi_err_int = phi_err_int + (phi_err*dt);
        phi_err_dot = (phi_err - phi_err_prev) / dt;
        phi_err_prev = phi_err;
        
        theta_err = theta_des - theta;
        theta_err_int = theta_err_int + (theta_err*dt);
        theta_err_dot = (theta_err - theta_err_prev) / dt;
        theta_err_prev = theta_err;
        
        psi_err = psi_des - psi;
        psi_err_int = psi_err_int + (psi_err*dt);
        psi_err_dot = (psi_err - psi_err_prev) / dt;
        psi_err_prev = psi_err;
        
        p_des = (Kp_phi * phi_err) + (Ki_phi * phi_err_int) + (Kd_phi * phi_err_dot); 
        q_des = (Kp_theta * theta_err) + (Ki_theta * theta_err_int) + (Kd_theta * theta_err_dot); 
        r_des = (Kp_psi * psi_err) + (Ki_psi * psi_err_int) + (Kd_psi * psi_err_dot);
        
        
        p_err = p_des - p;
        p_err_int = p_err_int + (p_err*dt);
        p_err_dot = (p_err - p_err_prev) / dt;
        p_err_prev = p_err;
        
        q_err = q_des - q;
        q_err_int = q_err_int + (q_err*dt);
        q_err_dot = (q_err - q_err_prev) / dt;
        q_err_prev = q_err;
        
        r_err = r_des - r;
        r_err_int = r_err_int + (r_err*dt);
        r_err_dot = (r_err - r_err_prev) / dt;
        r_err_prev = r_err;
        
        u2 = (Kp_p * p_err) + (Ki_p * p_err_int) + (Kd_p * p_err_dot);
        u3 = (Kp_q * q_err) + (Ki_q * q_err_int) + (Kd_q * q_err_dot);
        u4 = (Kp_r * r_err) + (Ki_r * r_err_int) + (Kd_r * r_err_dot);
        Omega, w1, w2, w3, w4 = Inp2Omega(u1, u2, u3, u4, Ct, Ctheta, l)
        
        solver.integrate(t[iter])
        params = [mass, g, Ixx, Iyy, Izz, l, J, Ct, Ctheta, u1, u2, u3, u4]
        solver.set_f_params(params)
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
    