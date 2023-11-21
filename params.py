from numpy import pi
import numpy as np

# DEFAULT QUADROTOR CONSTANTS
# Mass Properties
mass = 0.698;
Ixx = 0.0034;
Iyy = 0.0034;
Izz = 0.006;
        
#Rotor inertia
J = 1.302*10**-6;
w1 = 0;
w2 = 0;
w3 = 0;
w4 = 0; 

Ct = 7.6184*(10**(-8))*(60/(2*pi))**2; #Thrust Coefficient
Ctheta = 2.6839*(10**(-9))*(60/(2*pi))**2; # Moment Coefficient
l = 0.171;
g = 9.81;

# DEFAULT PID GAINS [Kp, Ki, Kd] gains
alt_gains = np.array([0.1, 0.0, 0.5])
phi_gains = np.array([5.5, 0.005, 0.1])
theta_gains = np.array([6.5, 0.001, 0.1])
psi_gains = np.array([0.5, 0.00, 0.02])

p_gain = np.array([0.05, 0.0001, 0])
q_gain = np.array([0.05, 0.0001, 0])
r_gain = np.array([0.05, 0.0001, 0])

PID_controller_gains = np.vstack((alt_gains, phi_gains, theta_gains, psi_gains, p_gain, q_gain, r_gain))
