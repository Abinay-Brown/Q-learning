import numpy as np
import control as ct
from params import *
from numpy import sin, cos, tan, ones, dot
def LQR_gain():
    n, m = 3, 1;
    A = np.array([[-1.01887, 0.90506, -0.00215], [0.82225, -1.07741, -0.17555],[ 0, 0, -1]]);
    B = np.array([[0.0],[0.0],[1.0]]);
    M = np.eye(n, n);
    R = 0.1*np.eye(m, m);
    L = ct.lqr(A, B, M, R)
    return L

# Linearized inertial frame
def state_space(state, inputs):
    x, y, z, xd, yd, zd = state;
    phi, theta, psi, U1 = inputs;
    
    A = np.zeros((6, 6))
    B = np.zeros((6, 4))
    
    
    A[0, 3] = 1;
    A[1, 4] = 1;
    A[2, 5] = 1;
    
    B[3, 0] = (U1/mass)*((cos(phi)*sin(psi))-(cos(psi)*sin(phi)*sin(theta)))
    B[4, 0] =-(U1/mass)*((cos(phi)*cos(psi))+(sin(phi)*sin(psi)*sin(theta)))
    B[5, 0] =-(U1/mass)*((cos(theta)*sin(phi)))
    
    B[3, 1] = (U1/mass)*(cos(phi)*cos(psi)*cos(theta))
    B[4, 1] = (U1/mass)*(cos(phi)*cos(theta)*sin(psi))
    B[5, 1] =-(U1/mass)*(cos(phi)*sin(theta))
    
    B[3, 2] = (U1/mass)*((cos(psi)*sin(phi))-(cos(phi)*sin(psi)*sin(theta)))
    B[4, 2] = (U1/mass)*((sin(phi)*sin(psi))+(cos(phi)*cos(psi)*sin(theta)))
    
    B[3, 3] = (1/mass)*((sin(phi)*sin(psi))+(cos(phi)*cos(psi)*sin(theta)))
    B[4, 3] =-(1/mass)*((cos(psi)*sin(phi))-(cos(phi)*sin(psi)*sin(theta)))
    B[5, 3] = (1/mass)*(cos(phi)*cos(theta))
    
    return A, B
def babyCT_b(t, state, params):
    
    xf, T, Tf, A, B, M, R, Pt, percent, amplitude, alpha_a, alpha_c, 
    
    return
    
def Q_learning_dynamics(x1, x2, S):
    n = 6
    m = 4
    
    M = 10 * np.eye(6)
    R = 2 * np.eye(4)
    Pt = 0.5 * np.eye(6)

    
    # ODE parameter
    Tf, T, N = 10, 0.05, 200 # finite Horizon
    alpha_c, alpha_a = 90, 1.2
    amplitude, percent = 0.1, 50
    
    xf = np.array([x2[0], x2[1], x2[2], 0, 0, 0])
    x0 = np.array([x1[0], x1[1], x1[1], 0, 0, 0])
    
    # Critic Weights Wc0 = WcT*v(t)-> (n+m)(n+m+1)/2x1 = (6+4)(6+4+1)/2x1=55x1
    Wc0  = ones((55, 1))

    # Actor Weights n x m
    Wa10 = ones((n, ))
    Wa20 = ones((n, ))
    Wa30 = ones((n, ))
    Wa40 = ones((n, ))


    # make sure the dimension multiplication is correct
    
    u0 = [dot(Wa10.T, x0-xf), dot(Wa20.T, x0-xf), dot(Wa30.T, x0-xf), dot(Wa40.T, x0-xf)]
    
    # Variables to record (Check if dimensions are correct for appending)
    t_save = [0, ]
    x_save = [[x0; Wc0; Wa10; Wa20; Wa30; Wa40],]
    uvec = [u0, ]    
    return