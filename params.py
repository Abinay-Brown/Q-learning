from numpy import pi


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

# DEFAULT PID GAINS
Kp_Z = 0.1; Ki_Z = 0.0; Kd_Z = 0.5;

Kp_phi = 5.5; Ki_phi = 0.005; Kd_phi = 0.1;
Kp_theta = 6.5; Ki_theta = 0.001; Kd_theta = 0.1;
Kp_psi = 0.5; Ki_psi = 0.00; Kd_psi = 0.02;

Kp_p = 0.05; Ki_p = 0.0001; Kd_p = 0.0;
Kp_q = 0.05; Ki_q = 0.0001; Kd_q = 0.0;
Kp_r = 0.05; Ki_r = 0.0001; Kd_r = 0.0 ;
