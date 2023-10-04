import numpy as np
from numpy import pi
from numpy import sin, cos, tan, sqrt


def diffEqn(t, state, params):
    
    m = params[0]
    g = params[1];
    
    Ixx = params[2];
    Iyy = params[3];
    Izz = params[4];    
    J = params[5];

    Omega = params[6];
    
    u1 = params[7]
    u2 = params[8]
    u3 = params[9]
    u4 = params[10]

    phi, theta, psi = state[0:3];
    
    p, q, r = state[3:6];
    
    u, v, w = state[6:9];
    
    X, Y, Z = state[9:12];
    
    p_dot = (((Iyy-Izz)/Ixx)*q*r) + (J*q*Omega/Ixx) + (u2/Ixx);
    q_dot = (((Izz-Ixx)/Iyy)*p*r) - (J*p*Omega/Iyy) + (u3/Iyy);
    r_dot = (((Ixx-Iyy)/Izz)*p*q) + (u4/Izz);
    
    T_mat = Trot(phi, theta)
    thetas = np.dot(T_mat, np.array([p, q, r]).T)
    
    phi_dot = thetas[0];
    theta_dot = thetas[1];
    psi_dot = thetas[2];
    
    u_dot = (v*r) - (w*q) + (g*sin(theta));
    v_dot = (w*p) - (u*r) - (g*cos(theta)*sin(phi));
    w_dot = (u*q) - (v*p) - (g*cos(theta)*cos(phi)) + (u1/m);
    
    R_mat = Rot(phi, theta, psi)
    Vels = np.dot(R_mat, np.array([u, v, w]).T)
    U = Vels[0]
    V = Vels[1]
    W = Vels[2]

    
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
    return T_mat;

def Rot(phi, theta, psi):
    R_x = np.array([[1, 0, 0],[0, cos(phi), -sin(phi)],[0, sin(phi), cos(phi)]]);
    R_y = np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]);
    R_z = np.array([[cos(psi),-sin(psi),0],[sin(psi),cos(psi),0],[0,0,1]]);
    mat = np.matmul(R_z, np.matmul(R_y, R_x));
    return mat;