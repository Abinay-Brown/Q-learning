import numpy as np
import control as ct
from params import *
from dynamics import *
from numpy import sin, cos, tan, ones, dot, hstack, vstack, zeros, square
from numpy.linalg import norm, inv
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



def babyCT_b(t, state, params):

    # Extract Q-learning dynamics parameters
    # state = [phi,theta, psi, p, q, r, u, v, w, X, Y, Z, Wc0(55), Wa0(6), Wa10(6), Wa20(6), Wa30(6), Wa40(6), P(1)]
    xf, T, Tf, M, R, Pt, percent, amplitude, alpha_a, alpha_c, u1_del, u2_del, u3_del, u4_del, x1_del, x2_del, x3_del, x4_del, x5_del, x6_del, p_del, u_save = params 
    n, m = 6, 4
    
    
    # Extract Drone dynamics Parameters
    drone_state = state[0:12]
    phi, theta, psi, p, q, r, u, v, w, X, Y, Z = state[0:12]

    # Q-learning Actor and critic weights
    Wc = state[12:67].reshape(55, 1)
    Wa10 = state[67: 73].reshape(6, 1)
    Wa20 = state[73: 79].reshape(6, 1)
    Wa30 = state[79: 85].reshape(6, 1)
    Wa40 = state[85: 91].reshape(6, 1)
    P = state[91]
    
    # Q-learning state vector is global position and velocity
    # x = [X, Y, Z, U, V, W]
    R_mat = Rot(phi, theta, psi)
    Vels = np.dot(R_mat, np.array([u, v, w]).T)
    U = Vels[0]
    V = Vels[1]
    W = Vels[2]
    x = np.array([X, Y, Z, U, V, W]).reshape(6, 1)
    
    # Updated Control
    ud = np.array([dot(Wa10.T, x-xf), dot(Wa20.T, x-xf), dot(Wa30.T, x-xf), dot(Wa40.T, x-xf)]).reshape(4, 1)
    
    # Interpolate delayed control and state
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
    
    # Kronecker Products
    diff = x-xf
    U = np.array([diff[0], diff[1], diff[2], diff[3], diff[4], diff[5], ud[0], ud[1], ud[2],ud[3], ud[4]]).T
    UkU = np.array([U[0]**2, U[0]*U[1], U[0]*U[2], U[0]*U[3], U[0]*U[4], U[0]*U[5], U[0]*U[6], U[0]*U[7], U[0]*U[8], U[0]*U[9],\
    U[1]**2, U[1]*U[2], U[1]*U[3], U[1]*U[4], U[1]*U[5], U[1]*U[6], U[1]*U[7], U[1]*U[8], U[0]*U[9],\
    U[2]**2, U[2]*U[3], U[2]*U[4], U[2]*U[5], U[2]*U[6], U[2]*U[7], U[2]*U[8], U[2]*U[9],\
    U[3]**2, U[3]*U[4], U[3]*U[5], U[3]*U[6], U[3]*U[7], U[3]*U[8], U[4]*U[9],\
    U[4]**2, U[4]*U[5], U[4]*U[6], U[4]*U[7], U[4]*U[8], U[4]*U[9],\
    U[5]**2, U[5]*U[6], U[5]*U[7], U[5]*U[8], U[6]*U[9],\
    U[6]**2, U[6]*U[7], U[6]*U[8], U[6]*U[9],\
    U[7]**2, U[7]*U[8], U[7]*U[9],\
    U[8]**2, U[8]*U[9],\
    U[9]**2]).reshape(55, 1)	
    
    Udel = hstack((x_del.T, ud_del.T))
    UkUdel = np.array([Udel[0]**2, Udel[0]*Udel[1], Udel[0]*Udel[2], Udel[0]*Udel[3], Udel[0]*Udel[4], Udel[0]*Udel[5], Udel[0]*Udel[6], Udel[0]*Udel[7], Udel[0]*Udel[8], Udel[0]*Udel[9],\
    Udel[1]**2, Udel[1]*Udel[2], Udel[1]*Udel[3], Udel[1]*Udel[4], Udel[1]*Udel[5], Udel[1]*Udel[6], Udel[1]*Udel[7], Udel[1]*Udel[8], Udel[0]*Udel[9],\
    Udel[2]**2, Udel[2]*Udel[3], Udel[2]*Udel[4], Udel[2]*Udel[5], Udel[2]*Udel[6], Udel[2]*Udel[7], Udel[2]*Udel[8], Udel[2]*Udel[9],\
    Udel[3]**2, Udel[3]*Udel[4], Udel[3]*Udel[5], Udel[3]*Udel[6], Udel[3]*Udel[7], Udel[3]*Udel[8], Udel[4]*Udel[9],\
    Udel[4]**2, Udel[4]*Udel[5], Udel[4]*Udel[6], Udel[4]*Udel[7], Udel[4]*Udel[8], Udel[4]*Udel[9],\
    Udel[5]**2, Udel[5]*Udel[6], Udel[5]*Udel[7], Udel[5]*Udel[8], Udel[6]*Udel[9],\
    Udel[6]**2, Udel[6]*Udel[7], Udel[6]*Udel[8], Udel[6]*Udel[9],\
    Udel[7]**2, Udel[7]*Udel[8], Udel[7]*Udel[9],\
    Udel[8]**2, Udel[8]*Udel[9],\
    Udel[9]**2]).reshape(55, 1)	
    
    Quu = np.array([[Wc[-10], Wc[-9], Wc[-8], Wc[-7]],
                    [Wc[-9], Wc[-6], Wc[-5], Wc[-4]],
                    [Wc[-8], Wc[-5], Wc[-3], Wc[-2]],
                    [Wc[-7], Wc[-4], Wc[-2], Wc[-1]]])
    Quu_inv = inv(Quu)
    Qux = np.array([[Wc[6], Wc[15], Wc[23], Wc[30], Wc[36], Wc[41]],
                    [Wc[7], Wc[16], Wc[24], Wc[31], Wc[37], Wc[42]],
                    [Wc[8], Wc[17], Wc[25], Wc[32], Wc[38], Wc[43]],
                    [Wc[9], Wc[18], Wc[26], Wc[33], Wc[39], Wc[44]]])

    # Integral Reinforcement Dynamics
    dP = 0.5*((dot(x.T, dot(M, x)))+ dot(ud.T, dot(R, ud)))
    sig = UkU - UkUdel
    sig_f = UkU
    
    # Critic Errors
    ec1 = P - p_del_val + dot(Wc.T, sig)
    ec2 = 0.5*dot(x.T, dot(Pt, x)) - dot(Wc.T, sig_f)

    # Actor Errors
    Wa = np.hstack((Wa10, Wa20, Wa30, Wa40))
    ea = dot(Wa.T, x) + dot(Quu_inv, dot(Qux, x))
    #term = dot(Quu_inv, dot(Qux, x))
    #ea1 = dot(Wa10.T, x) + term[0, :]
    #ea2 = dot(Wa20.T, x) + term[1, :]
    #ea3 = dot(Wa30.T, x) + term[2, :]
    #ea4 = dot(Wa40.T, x) + term[3, :]


    # Critic Dynamics
    # Check dimensions of output
    dWc = -alpha_c * ((ec1*sig/square(1+dot(sig.T, sig)))+(ec2*sig_f/square(1+dot(sig_f.T, sig_f))))
    
    # Actor Dynamics
    
    dWa = -alpha_a * dot(x, ea.T)
    dWa10 = dWa[:, 0]
    dWa20 = dWa[:, 1]
    dWa30 = dWa[:, 2]
    dWa40 = dWa[:, 3]

    unew = zeros(m, 1)
    # Persistence Excitation

    if t <= (percent/100)*Tf:
        unew[0, 0] = ud[0] + (amplitude*exp(-0.009*t)*2*(sin(t)**2*cos(t)+sin(2*t)**2*cos(0.1*t)+sin(-1.2*t)**2*cos(0.5*t)+sin(t)**5+sin(1.12*t)**2+cos(2.4*t)*sin(2.4*t)**3))
        unew[1, 0] = ud[1] + (amplitude*exp(-0.009*t)*2*(sin(t)**2*cos(t)+sin(2*t)**2*cos(0.1*t)+sin(-1.2*t)**2*cos(0.5*t)+sin(t)**5+sin(1.12*t)**2+cos(2.4*t)*sin(2.4*t)**3))
        unew[2, 0] = ud[2] + (amplitude*exp(-0.009*t)*2*(sin(t)**2*cos(t)+sin(2*t)**2*cos(0.1*t)+sin(-1.2*t)**2*cos(0.5*t)+sin(t)**5+sin(1.12*t)**2+cos(2.4*t)*sin(2.4*t)**3))
        unew[3, 0] = ud[4] + (amplitude*exp(-0.009*t)*2*(sin(t)**2*cos(t)+sin(2*t)**2*cos(0.1*t)+sin(-1.2*t)**2*cos(0.5*t)+sin(t)**5+sin(1.12*t)**2+cos(2.4*t)*sin(2.4*t)**3))
    else:
        unew = ud

    # Convert [phi, theta, psi, altitude].T desired to u1, u2, u3, u4 moment and thrust control inputs using PID
    PID_update(state, unew) 

    # Do usave in here
    
    drone_params = [mass, g, Ixx, Iyy, Izz, l, J, Ct, Ctheta, u1, u2, u3, u4]
    drone_state_dot = dynamics(t, drone_state, drone_params)
    phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot, u_dot, v_dot, w_dot, U, V, W = drone_state_dot[0:12]
    
    dx = np.array([phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot, u_dot, v_dot, w_dot, U, V, W]).reshape(1, 12)

    # dWc -> 1x55
    # dWa10 -> 1x6 
    dotx = hstack((dx, dWc, dWa10.T, dWa20.T, dWa30.T, dWa40.T, dP))
    return dotx
    

def Q_learning_dynamics(x1, x2, S):
    
    # State Vector =  [X, Y, Z, VX, VY, VZ]
    # Control Input = [phi, theta, psi, altitude]

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
    x_save = hstack((xi.T, Wc0.T, Wa10.T, Wa20.T, Wa30.T, Wa40.T, np.array([[0]])))

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

