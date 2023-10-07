import numpy as np
from params import *
from numpy import pi
from numpy import sin, cos, tan, sqrt
from scipy import integrate

# Controller Adaptation from: https://wilselby.com/research/arducopter/controller-design/


class drone:
    params = {};
    control_inputs = {};
    pid_gains = {};

    # Initialize initial conditions, terminal conditions, and quadrotor parameters 
    def __init__(self, IC, TC, params, gains):
        # Initial Conditions
        # IC = [phi, theta, psi, p, q, r, u, v, w, X, Y, Z] -> Euler angles, Euler rates, Velocity Body Frame, and Global Position
        
        # Terminal Conditions
        # TC = [X, Y, Z, VX, VY, VZ] -> Global Position and Velocity
        
        # Quadrotor properties and initial parameters
        # params = [mass, g, Ixx, Iyy, Izz, l, J, Ct, Ctheta, u1, u2, u3, u4]

        # PID gains 
        # gains = [[Kp_Z, Ki_Z, Kd_Z], 
        #          [Kp_phi, Ki_phi, Kd_phi],
        #          [Kp_theta, Ki_theta, Kd_theta],
        #          [Kp_psi, Ki_psi, Kd_psi],
        #          [Kp_p, Ki_p, Kd_p],
        #          [Kp_q, Ki_q, Kd_q],
        #          [Kp_r, Ki_r, Kd_r]]


    def diff(self):
        return

IC = [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0]; # Initial Conditions
TC = [5, 5, 5, 0, 0, 0]; # Terminal Conditions
params = [mass, g, Ixx, Iyy, Izz, l, J, Ct, Ctheta, mass*g, 0, 0, 0] # Quadrotor parameters
gains = [[Kp_Z, Ki_Z, Kd_Z],\
[Kp_phi, Ki_phi, Kd_phi],\
[Kp_theta, Ki_theta, Kd_theta],\
[Kp_psi, Ki_psi, Kd_psi],\
[Kp_p, Ki_p, Kd_p],\
[Kp_q, Ki_q, Kd_q],\
[Kp_r, Ki_r, Kd_r]]


agent = drone(IC, TC, params, gains)