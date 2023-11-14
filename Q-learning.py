import numpy as np
import control as ct
from params import *
from dynamics import *
from numpy import sin, cos, tan, ones, dot, hstack, vstack, zeros
from numpy.linalg import norm
from scipy.interpolate import CubicSpline


def interp2PWC(y, xi, xf):
    row = len(y)
    if row == 1:
        xdata = np.linspace(-1.0, xf, row + 1)
        itp = CubicSpline([xdata[0], xdata[-1]], [y[0], y[-1]], bc_type='clamped')
    else:
        xdata = np.linspace(xi, xf, row)
        itp = CubicSpline(xdata, y, bc_type='clamped')

    return itp(xdata)


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
    # Extract Drone dynamics Parameters
    # Extract Q-learning dynamics parameters
    xf, T, Tf, M, R, Pt, percent, amplitude, alpha_a, alpha_c, u1_del, u2_del, u3_del, u4_del, x1_del, x2_del, x3_del, x4_del, x5_del, x6_del, p_del, u_save = params 
    n, m = 6, 4
    # state = [phi,theta, psi, p, q, r, u, v, w, X, Y, Z, Wc0(55), Wa0(6), Wa10(6), Wa20(6), Wa30(6), Wa40(6), P(1)]
    # x = [X, Y, Z, U, V, W]
    
    phi, theta, psi, p, q, r, u, v, w, X, Y, Z = state[0:12]
    Wc = state[12:67]
    Wa10 = state[67: 73]
    Wa20 = state[73: 79]
    Wa30 = state[79: 85]
    Wa40 = state[85: 91]
    P = state[91]
    
    
    R_mat = Rot(phi, theta, psi)
    Vels = np.dot(R_mat, np.array([u, v, w]).T)
    U = Vels[0]
    V = Vels[1]
    W = Vels[2]
    
    # Q-learning state vector is global position and velocity
    x = np.array([[X, Y, Z, U, V, W]]).T
    
    # Updated Control
    ud = np.array([dot(Wa10.T, x-xf), dot(Wa20.T, x-xf), dot(Wa30.T, x-xf), dot(Wa40.T, x-xf)])
    
    ud_del = zeros(m, 1)
    x_del  = zeros(n, 1)
    
    ud_del[0, 0] = u1_del(t-T)
    ud_del[1, 0] = u2_del(t-T)
    ud_del[2, 0] = u3_del(t-T)
    ud_del[3, 0] = u4_del(t-T)
    
    x_del[0, 0] = x1_del(t-T)
    x_del[1, 0] = x2_del(t-T)
    x_del[2, 0] = x3_del(t-T)
    x_del[3, 0] = x4_del(t-T)
    x_del[4, 0] = x5_del(t-T)
    x_del[5, 0] = x6_del(t-T)
    p_del_val = p_del(t-T)
    

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
    
    xf = np.array([x2[0], x2[1], x2[2], 0, 0, 0]).reshape((6, 1))
    xi = np.array([x1[0], x1[1], x1[1], 0, 0, 0]).reshape((6, 1))
    
    # Critic Weights Wc0 = WcT*v(t)-> (n+m)(n+m+1)/2x1 = (6+4)(6+4+1)/2x1=55x1
    Wc0  = ones((55, 1))

    # Actor Weights n x m
    # Each of Actor weights column is taken as a vector because integrators need vectors not matrices
    Wa10 = ones((n, 1))
    Wa20 = ones((n, 1))
    Wa30 = ones((n, 1))
    Wa40 = ones((n, 1))


    # make sure the dimension multiplication is correct
    
    u0 = np.array([dot(Wa10.T, xi-xf), dot(Wa20.T, xi-xf), dot(Wa30.T, xi-xf), dot(Wa40.T, xi-xf)])
    
    # Appending the vectors after each row
    t_save = hstack(([0],))
    u_save = vstack((u0.T, ))
    x_save = hstack((xi.T, Wc0.T))
    x_save = hstack((x_save, Wa10.T))
    x_save = hstack((x_save, Wa20.T))
    x_save = hstack((x_save, Wa30.T))
    x_save = hstack((x_save, Wa40.T))
    x_save = hstack((x_save, np.array([[0]])))
    
    # Interpolate functions
    u1_del = interp2PWC(u_save[:, 0], -1, 1)
    u2_del = interp2PWC(u_save[:, 1], -1, 1)
    u3_del = interp2PWC(u_save[:, 2], -1, 1)
    u4_del = interp2PWC(u_save[:, 3], -1, 1)
    
    x1_del = interp2PWC(x_save[:, 0], -1, 1)
    x2_del = interp2PWC(x_save[:, 1], -1, 1)
    x3_del = interp2PWC(x_save[:, 2], -1, 1)
    x4_del = interp2PWC(x_save[:, 3], -1, 1)
    x5_del = interp2PWC(x_save[:, 4], -1, 1)
    x6_del = interp2PWC(x_save[:, 5], -1, 1)
    
    xdist = norm(xi[0:3, 0] - xf[0:3, 0])
    error = 0.25
    maxIter = 10000
    
    for iter in range(1, maxIter+1):
        t = [(iter-1)*T, iter*T]
        params = [xf, T, Tf, M, R, Pt, percent, amplitude, alpha_a, alpha_c, u1_del, u2_del, u3_del, u4_del, x1_del, x2_del, x3_del, x4_del, x5_del, x6_del, p_del, u_save]
        
        
       
    return

