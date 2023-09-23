import numpy as np
from numpy import pi
from numpy import sin, cos, tan, sqrt
from scipy import integrate

#https://wilselby.com/research/arducopter/controller-design/
class drone:
    params = {};
    control_inputs = {};
    pid_gains = {};
    def __init__(self, initial_conditions):
        # Mass Properties
        self.params['mass'] = 0.698;
        self.params['Ixx'] = 0.0034;
        self.params['Iyy'] = 0.0034;
        self.params['Izz'] = 0.006;
        
        #Rotor inertia
        self.params['J'] = 1.302*10**-6;
        self.params['w1'] = 0;
        self.params['w2'] = 0;
        self.params['w3'] = 0;
        self.params['w4'] = 0; 

        self.params['Ct'] = 7.6184*(10**(-8))*(60/(2*pi))**2; #Thrust Coefficient
        self.params['Ctheta'] = 2.6839*(10**(-9))*(60/(2*pi))**2; # Moment Coefficient
        self.params['l'] = 0.171;
        self.params['g'] = 9.81;


        self.control_inputs['U1'] = self.params['mass'] * self.params['g'];
        self.control_inputs['U2'] = 0;
        self.control_inputs['U3'] = 0;
        self.control_inputs['U4'] = 0;

        self.params['Omega'] = self.Inp2Omega(self.control_inputs['U1'], self.control_inputs['U2'], self.control_inputs['U3'], self.control_inputs['U4'])

        self.pid_gains['Kp_p']
        self.pid_gains['Ki_p']
        self.pid_gains['Kd_p']

        self.pid_gains['Kp_q']
        self.pid_gains['Ki_q']
        self.pid_gains['Kd_q']

        self.pid_gains['Kp_r']
        self.pid_gains['Ki_r']
        self.pid_gains['Kd_r']
        
        self.state = []

    def diffEqn(self, state, t):
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

        U2 = self.control_inputs['U2'];
        U3 = self.control_inputs['U3'];
        U4 = self.control_inputs['U4'];
        Ixx = self.control_inputs['Ixx'];
        Iyy = self.control_inputs['Iyy'];
        Izz = self.control_inputs['Izz'];

        Omega = self.Inp2Omega(U1, U2, U3, U4)
        p_dot = (((Iyy-Izz)/Ixx)*q*r) + (J*q*Omega/Ixx) + (U2/Ixx);
        q_dot = (((Izz-Ixx)/Iyy)*p*r) - (J*p*Omega/Iyy) + (U3/Iyy);
        r_dot = (((Ixx-Iyy)/Izz)*p*q) + (U4/Izz);   

        T_mat = self.Trot(phi, theta, psi)



    def Trot(self, phi, theta):
        T_mat = np.array([[1, sin(phi)*tan(theta),\
         cos(phi)*tan(theta)], [0, cos(phi), -sin(phi)],\
          [0, (sin(phi)/cos(theta)), (cos(phi)/cos(theta))]]);
        print(T_mat)
        return T_mat;

    def Rot(self, phi, theta, psi):
        R_x = np.array([[1, 0, 0],[0, cos(phi), -sin(phi)],[0, sin(phi), cos(phi)]]);
        R_y = np.array([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]);
        R_z = np.array([[cos(psi),-sin(psi),0],[sin(psi),cos(psi),0],[0,0,1]]);
        mat = np.matmul(R_z, np.matmul(R_y, R_x));
        print(mat)
        return mat;

    def Inp2Omega(self, U1, U2, U3, U4):
        Ct = self.params['Ct'];
        Ctheta = self.params['Ctheta'];
        l = self.params['l'];

        mat = np.array([[0.25, 0, -0.5, -0.25], [0.25, 0.5, 0, 0.25], [0.25, 0, 0.5, -0.25], [0.25, -0.5, 0, 0.25]])
        Omegas = np.matmul(mat, np.array([U1/Ct, U2/Ct/l, U3/Ct/l, U4/Ctheta]).T);
        self.params['w1'] = sqrt(Omegas[0]);
        self.params['w2'] = sqrt(Omegas[1]);
        self.params['w3'] = sqrt(Omegas[2]);
        self.params['w4'] = sqrt(Omegas[3]);

        self.params['Omega'] = sqrt(Omegas[0])-sqrt(Omegas[1])+sqrt(Omegas[2])-sqrt(Omegas[3]);
        if (Omegas[0]<0 or Omegas[1]<0 or Omegas[2]<0 or Omegas[3]<0):
            print('Error! Aggressive Maneuver Soln not Found')
        
        return self.params['Omega'];


agent1 = drone([0, 0, 0]);

#agent1.Rot(0.3452, 0.6342, 0.1231)

#agent1.Rot(0.1245, 0, 0)