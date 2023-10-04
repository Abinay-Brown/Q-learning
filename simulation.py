import numpy as np
from numpy import pi
from numpy import cos, sin, tan
from scipy.integrate import ode
from scipy.integrate import odeint
from dynamics import Inp2Omega, Rot, Trot, diffEqn
import matplotlib.pyplot as plt


# Constants
# Mass Properties
mass = 0.698;
Ixx = 0.0034;
Iyy = 0.0034;
Izz = 0.006;

J = 1.302*10**-6;
w1 = 0;
w2 = 0;
w3 = 0;
w4 = 0; 
Ct = 7.6184*(10**(-8))*(60/(2*pi))**2; #Thrust Coefficient
Ctheta = 2.6839*(10**(-9))*(60/(2*pi))**2; # Moment Coefficient
l = 0.171;
g = 9.81;



# PID gains
Kp_X = 0; Ki_X = 0; Kd_X = 0;
Kp_Y = 0; Ki_Y = 0; Kd_Y = 0;
Kp_z = 0; Ki_z = 0; Kd_z = 0;

Kp_phi = 0; Ki_phi = 0; Kd_phi = 0;
Kp_theta = 0; Ki_theta = 0; Kd_theta = 0;
Kp_psi = 0; Ki_psi = 0; Kd_psi = 0;

Kp_p = 0; Ki_p = 0; Kd_p = 0;
Kp_q = 0; Ki_q = 0; Kd_q = 0;
Kp_r = 0; Ki_r = 0; Kd_r = 0;

# PID Error Variables
X_err_int = 0; X_err_dot = 0; X_err_prev = 0;
Y_err_int = 0; Y_err_dot = 0; Y_err_prev = 0;

phi_err_int = 0; phi_err_dot = 0; phi_err_prev = 0;
theta_err_int = 0; theta_err_dot = 0; theta_err_prev = 0;
psi_err_int = 0; psi_err_dot = 0; psi_err_prev = 0;

p_err_int = 0; p_err_dot = 0; p_err_prev = 0;
q_err_int = 0; q_err_dot = 0; q_err_prev = 0;
r_err_int = 0; r_err_dot = 0; r_err_prev = 0;

# Setpoints
X_des = 0;
Y_des = 0;
Z_des = 5;




def main():
    phi_des = 5 * (np.pi / 180)
    theta_des = 5 * (np.pi / 180)
    psi_des = 0;
    z_err_int = 0; z_err_dot = 0; z_err_prev = 0;
    phi_err_int = 0; phi_err_dot = 0; phi_err_prev = 0;
    theta_err_int = 0; theta_err_dot = 0; theta_err_prev = 0;
    psi_err_int = 0; psi_err_dot = 0; psi_err_prev = 0;

    p_err_int = 0; p_err_dot = 0; p_err_prev = 0;
    q_err_int = 0; q_err_dot = 0; q_err_prev = 0;
    r_err_int = 0; r_err_dot = 0; r_err_prev = 0;

    Kp_X = 0; Ki_X = 0; Kd_X = 0;
    Kp_Y = 0; Ki_Y = 0; Kd_Y = 0;
    Kp_z = 0.15; Ki_z = 0.01; Kd_z = 1.0;

    Kp_phi = 0.005; Ki_phi = 0.000; Kd_phi = 0.01;
    Kp_theta = 0; Ki_theta = 0; Kd_theta = 0;
    Kp_psi = 0; Ki_psi = 0; Kd_psi = 0;

    Kp_p = 0.1; Ki_p = 0.000; Kd_p = 0.0;
    Kp_q = 0; Ki_q = 0; Kd_q = 0;
    Kp_r = 0; Ki_r = 0; Kd_r = 0;

    u1 = mass * g;
    u2 = 0;
    u3 = 0;
    u4 = 0;
    
    t0 = 0
    t1 = 10
    dt = 0.01
    t = np.arange(t0, t1, dt)
    
    sol = np.empty((len(t), 12))
    # state = [phi, theta, psi, p, q, r, u, v, w, X, Y, Z]
    state = [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0]
    
    Omega, w1, w2, w3, w4 = Inp2Omega(u1, u2, u3, u4, Ct, Ctheta, l)
    params = [mass, g, Ixx, Iyy, Izz, J, Omega, u1, u2, u3, u4]
    
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
        
        pos = np.dot(R_mat.T, np.array([X, Y, Z]).T)
        z = pos[2]
        
        pos_des = np.dot(R_mat.T, np.array([X_des, Y_des, Z_des]).T)
        z_des = pos_des[2]
        
        '''
        X_err = X_des - X;
        X_err_int = X_err_int + (X_err*dt);
        X_err_dot = (X_err - X_err_prev) / dt; 
        X_err_prev = X_err;
        
        Y_err = Y_des - Y;
        Y_err_int = Y_err_int + (Y_err*dt);
        Y_err_dot = (Y_err - Y_err_prev) / dt;
        Y_err_prev = Y_err;
        
        
        theta_des = (Kp_X * X_err) + (Ki_X * X_err_int) + (Kd_X * X_err_dot);
        phi_des =   (Kp_Y * Y_err) + (Ki_Y * Y_err_int) + (Kd_Y * Y_err_dot);
        '''
        
        z_err = z_des - z;
        z_err_int = z_err_int + (z_err*dt);
        z_err_dot = (z_err - z_err_prev) / dt;
        z_err_prev = z_err;
        
        u1 = (1/(cos(phi)*cos(theta)))*((Kp_z * z_err) + (Ki_z * z_err_int) + (Kd_z * z_err_dot) + (mass*g));
        
        
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
        print(p_err)
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
        u3 = (Kp_r * r_err) + (Ki_r * r_err_int) + (Kd_r * r_err_dot);
        Omega, w1, w2, w3, w4 = Inp2Omega(u1, u2, u3, u4, Ct, Ctheta, l)
        
        solver.integrate(t[iter])
        params = [mass, g, Ixx, Iyy, Izz, J, Omega, u1, u2, u3, u4]
        solver.set_f_params(params)
        sol[iter] = solver.y
        state = solver.y
        iter = iter + 1
        #print(state)
        
    plt.subplot(2, 1, 1)
    plt.plot(t, sol[:, 0])
    plt.subplot(2, 1, 2)
    plt.plot(t, sol[:, 3])
    plt.show()
    return
        
    

main()