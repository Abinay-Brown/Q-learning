import numpy as np
from params import *
from numpy import pi
from numpy import sin, cos, tan, sqrt
from scipy import integrate
from dynamics import *

# Controller Adaptation from: https://wilselby.com/research/arducopter/controller-design/


class drone:
    
    # Initialize initial conditions, terminal conditions, and quadrotor parameters 
    def __init__(self, IC, TC, params, gains):
        # Initial Conditions
        # IC = [phi, theta, psi, p, q, r, u, v, w, X, Y, Z] -> Euler angles, Euler rates, Velocity Body Frame, and Global Position
        self.IC = IC;

        # Terminal Conditions
        # TC = [X, Y, Z, VX, VY, VZ] -> Global Position and Velocity
        self.TC = TC;
        # Quadrotor properties and initial parameters
        # params = [mass, g, Ixx, Iyy, Izz, l, J, Ct, Ctheta, u1, u2, u3, u4]
        self.drone_params = params
        # PID gains 
        # gains = [[Kp_Z, Ki_Z, Kd_Z], 
        #          [Kp_phi, Ki_phi, Kd_phi],
        #          [Kp_theta, Ki_theta, Kd_theta],
        #          [Kp_psi, Ki_psi, Kd_psi],
        #          [Kp_p, Ki_p, Kd_p],
        #          [Kp_q, Ki_q, Kd_q],
        #          [Kp_r, Ki_r, Kd_r]]
        self.alt_gains = gains[0, :]
        self.phi_gains = gains[1, :]
        self.theta_gains = gains[2, :]
        self.psi_gains = gains[3, :]
        self.p_gains = gains[4, :]
        self.q_gains = gains[5, :]
        self.r_gains = gains[6, :]
        
        # Initialize PID variables
        self.Z_err_int = 0; 
        self.Z_err_dot = 0; 
        self.Z_err_prev = 0;
        
        self.phi_err_int = 0;
        self.phi_err_dot = 0;
        self.phi_err_prev = 0;
        
        self.theta_err_int = 0;
        self.theta_err_dot = 0;
        self.theta_err_prev = 0;
        
        self.psi_err_int = 0;
        self.psi_err_dot = 0;
        self.psi_err_prev = 0;
        
        self.p_err_int = 0;
        self.p_err_dot = 0;
        self.p_err_prev = 0;
        
        self.q_err_int = 0;
        self.q_err_dot = 0;
        self.q_err_prev = 0;
        
        self.r_err_int = 0;
        self.r_err_dot = 0;
        self.r_err_prev = 0;
    
    def PID_update(self, state, phi_req, theta_req, psi_req, alt_req, dt):
        phi, theta, psi, p, q, r, u, v, w, X, Y, Z = state[0:12]
        mass, g, Ixx, Iyy, Izz, l, J, Ct, Ctheta, u1, u2, u3, u4 = self.drone_params[0:13]
        mass = self.drone_params[0]
        g = self.drone_params[1]  

        R_mat = Rot(phi, theta, psi)
        T_mat = Trot(phi, theta)

        Z_err = alt_req - Z;
        self.Z_err_int = self.Z_err_int + (Z_err*dt);
        Z_err_dot = (Z_err - self.Z_err_prev) / dt;
        self.Z_err_prev = Z_err;

        u1 = (1/(cos(phi)*cos(theta)))*((self.alt_gains[0] * Z_err) + (self.alt_gains[1] * self.Z_err_int) + (self.alt_gains[2] * Z_err_dot) + (mass*g));

        phi_err = phi_req - phi;
        self.phi_err_int = self.phi_err_int + (phi_err*dt);
        phi_err_dot = (phi_err - self.phi_err_prev) / dt;
        self.phi_err_prev = phi_err;

        theta_err = theta_req - theta;
        self.theta_err_int = self.theta_err_int + (theta_err*dt);
        theta_err_dot = (theta_err - self.theta_err_prev) / dt;
        self.theta_err_prev = theta_err;
        
        psi_err = psi_req - psi;
        self.psi_err_int = self.psi_err_int + (psi_err*dt);
        psi_err_dot = (psi_err - self.psi_err_prev) / dt;
        self.psi_err_prev = psi_err;
        
        p_req = (self.phi_gains[0] * phi_err) + (self.phi_gains[1] * self.phi_err_int) + (self.phi_gains[2] * phi_err_dot); 
        q_req = (self.theta_gains[0] * theta_err) + (self.theta_gains[1] * self.theta_err_int) + (self.theta_gains[2] * theta_err_dot); 
        r_req = (self.psi_gains[0] * psi_err) + (self.psi_gains[1] * self.psi_err_int) + (self.psi_gains[2] * psi_err_dot);
        
        p_err = p_req - p;
        self.p_err_int = self.p_err_int + (p_err*dt);
        p_err_dot = (p_err - self.p_err_prev) / dt;
        self.p_err_prev = p_err;
        
        q_err = q_req - q;
        self.q_err_int = self.q_err_int + (q_err*dt);
        q_err_dot = (q_err - self.q_err_prev) / dt;
        self.q_err_prev = q_err;
        
        r_err = r_req - r;
        self.r_err_int = self.r_err_int + (r_err*dt);
        r_err_dot = (r_err - self.r_err_prev) / dt;
        self.r_err_prev = r_err;
        
        u2 = (self.p_gains[0] * p_err) + (self.p_gains[1] * self.p_err_int) + (self.p_gains[2] * p_err_dot);
        u3 = (self.q_gains[0] * q_err) + (self.q_gains[1] * self.q_err_int) + (self.q_gains[2] * q_err_dot);
        u4 = (self.r_gains[0] * r_err) + (self.r_gains[1] * self.r_err_int) + (self.r_gains[2] * r_err_dot);

        Omega, w1, w2, w3, w4 = Inp2Omega(u1, u2, u3, u4, Ct, Ctheta, l)
        self.drone_params[9] = u1;
        self.drone_params[10] = u2;
        self.drone_params[11] = u3;
        self.drone_params[12] = u4;

        return u1, u2, u3, u4

        



    def diff(self):
        return

