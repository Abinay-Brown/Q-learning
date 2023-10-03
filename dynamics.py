import numpy as np
from numpy import pi
from numpy import sin, cos, tan, sqrt


def diffEqn(state, t, params):
    m = params[0]
    g = params[1];
    
    Ixx = params[2];
    Iyy = params[3];
    Izz = params[4];    
    J = params[5];

    Omega = params[6];

    phi = state[0];
    theta = state[1];
    psi = state[2];
    
    p = state[3];
    q = state[4];
    r = state[5];
    
    u = state[6];
    v = state[7];
    w = state[8];
    
    X = state[9];
    Y = state[10];
    Z = state[11];
    
    R_mat = rot(phi, theta, psi)
    Vels = np.dot(R_mat, np.array([u, v, w]).T)
    U = Vels[0]
    V = Vels[1]
    W = Vels[2]

    pos = np.dot(R_mat.T, np.array([X, Y, Z]).T)
    x = pos[0];
    y = pos[1];
    z = pos[2];

    pos_des = np.dot(R_mat.T, np.array([X_des, Y_des, Z_des]).T)
    x_des = pos_des[0];
    y_des = pos_des[1];
    z_des = pos_des[2];

    #------------------- PID loop
    # Keep track of X/Y_err_int and X_err and dt and X/Y_err_prev
    # z_err_prev, z_err_int, z_err,
    X_err = X_des - X;
    X_err_int = X_err_int + (X_err*dt);
    X_err_dot = (X_err - X_err_prev) / dt; 
    X_err_prev = X_prev;

    Y_err = Y_des - Y;
    Y_err_int = Y_err_int + (Y_err*dt);
    Y_err_dot = (Y_err - Y_err_prev) / dt;
    Y_err_prev = Y_err;

    theta_des = (Kp_X * X_err) + (Ki_X * X_err_int) + (Kd_X * X_err_dot);
    phi_des =   (Kp_Y * Y_err) + (Ki_Y * Y_err_int) + (Kd_Y * Y_err_dot);

    z_err = z_des - z;
    z_err_int = z_err_int + (z_err*dt);
    z_err_dot = (z_err - z_err_prev) / dt;
    z_err_prev = z_err;

    U1 = (1/(cos(phi)*cos(theta)))*((Kp_z * z_err) + (Ki_z * z_err_int) + (Kd_z * z_err_dot) + (m*g));
    
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

    U2 = (Kp_p * p_err) + (Ki_p * p_err_int) + (Kd_p * p_err_dot);
    U3 = (Kp_q * q_err) + (Ki_q * q_err_int) + (Kd_q * q_err_dot);
    U3 = (Kp_r * r_err) + (Ki_r * r_err_int) + (Kd_r * r_err_dot);
    
    #----------------------
    p_dot = (((Iyy-Izz)/Ixx)*q*r) + (J*q*Omega/Ixx) + (U2/Ixx);
    q_dot = (((Izz-Ixx)/Iyy)*p*r) - (J*p*Omega/Iyy) + (U3/Iyy);
    r_dot = (((Ixx-Iyy)/Izz)*p*q) + (U4/Izz);
    
    T_mat = self.Trot(phi, theta, psi)
    thetas = np.dot(Tmat, np.array([p, q, r]).T)
    
    phi_dot = thetas[0];
    theta_dot = thetas[1];
    psi_dot = thetas[2];
    
    u_dot = (v*r) - (w*q) + (g*sin(theta));
    v_dot = (w*p) - (u*r) - (g*cos(theta)*sin(phi));
    w_dot = (u*q) - (v*p) - (g*cos(theta)*cos(phi)) + (U1/m);
    
    
    state_dot = [phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot, u_dot, v_dot, w_dot, U, V, W]
    return state_dot


def Inp2Omega(U1, U2, U3, U4, Ct, Ctheta, l):
    mat = np.array([[0.25, 0, -0.5, -0.25], [0.25, 0.5, 0, 0.25], [0.25, 0, 0.5, -0.25], [0.25, -0.5, 0, 0.25]])
    Omegas = np.matmul(mat, np.array([U1/Ct, U2/Ct/l, U3/Ct/l, U4/Ctheta]).T);
    w1 = sqrt(Omegas[0]);
    w2 = sqrt(Omegas[1]);
    w3 = sqrt(Omegas[2]);
    w4 = sqrt(Omegas[3]);
    Omega = sqrt(Omegas[0])-sqrt(Omegas[1])+sqrt(Omegas[2])-sqrt(Omegas[3]);
    if (Omegas[0]<0 or Omegas[1]<0 or Omegas[2]<0 or Omegas[3]<0):
        print('Error! Aggressive Maneuver Soln not Found')
    
    return Omega, w1, w2, w3, w4
    
def Trot(phi, theta):
    T_mat = np.array([[1, sin(phi)*tan(theta),\
    cos(phi)*tan(theta)], [0, cos(phi), -sin(phi)],\
    [0, (sin(phi)/cos(theta)), (cos(phi)/cos(theta))]]);
    print(T_mat)
    return T_mat;

def Rot(phi, theta, psi):
    R_x = np.array([[1, 0, 0],[0, cos(phi), -sin(phi)],[0, sin(phi), cos(phi)]]);
    R_y = np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]);
    R_z = np.array([[cos(psi),-sin(psi),0],[sin(psi),cos(psi),0],[0,0,1]]);
    mat = np.matmul(R_z, np.matmul(R_y, R_x));
    print(mat)
    return mat;