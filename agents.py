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

#agent1 = drone([0, 0, 0]);

#agent1.Rot(0.3452, 0.6342, 0.1231)

#agent1.Rot(0.1245, 0, 0)