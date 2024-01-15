
# --------------- Add Package and Workspace Directories to Path -------------- #
import os
import sys
amls_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ws_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, amls_pkg_path)
sys.path.insert(0, ws_path)

import numpy as np
from numpy import linalg as LA

from math import pi

from amls.dynamics.environmental_params import env_1

class QuadParameters:
    '''
    Class used to organize the needed parameters to define a UAV
    '''

    def __init__(self) -> None:

        # UAV geometry
        self.d: float = None # Distance from UAV CoM to motors (m)
        self.r: float = None # UAV blade radius (m)
        self.theta_rot: float = np.radians(90) # Ang spacing b/t rotors (rad)

        # UAV mass and inertia
        self.m: float = None # UAV mass (kg)
        self.Ixx: float = None # Ixx term (kg*m^2)
        self.Ixy: float = None # Ixy term (kg*m^2)
        self.Ixz: float = None # Ixz term (kg*m^2)
        self.Iyx: float = None # Iyx term (kg*m^2)
        self.Iyy: float = None # Iyy term (kg*m^2)
        self.Iyz: float = None # Iyz term (kg*m^2)
        self.Izx: float = None # Izx term (kg*m^2)
        self.Izy: float = None # Izy term (kg*m^2)
        self.Izz: float = None # Izz term (kg*m^2)

        # Blade parameters
        self.cT: float = None # Thrust coefficient (unitless) [4, 2.31]

        # Motor parameters
        self.w_max: float = None # Maximum motor angular velocity (rad/sec)
        self.w_min: float = 0.05 # Min motor angular velocity (rad/sec) - 
                                 # to stop div by 0 err
        self.w_dir: np.ndarray = None # +1 or -1 direction of rotation

        # Drag parameters
        self.A1c: float = None # Diagonal terms in "blade flapping" drag
        self.A1s: float = None # Off diagonal terms in "blade flapping" drag
        self.dx: float = None # x-dimension term in drag matrix
        self.dy: float = None # y-dimension term in drag matrix

        # Optional blade flap parameters
        self.theta_0: float = None # Blade root angle (rad)
        self.theta_tip: float = None # Blade tip angle (rad)
        self.c: float = None # Blade chord (m)
        self.m_b: float = None # Blade mass (m)
        self.Cl_alpha: float = None # Lift-slope gradient (unitless)

    # Initialize basic computed values
    def comp_values(self, env_params):

        self.A = pi*self.r**2 # Blade area (m^2)

        # Rotor position vectors and and matrix
        self.rotor1_pos = self.d*np.array((np.cos(self.theta_rot/2), 
            np.sin(self.theta_rot/2), 0)) # Rotor 1 position (m)
        self.rotor2_pos = self.d*np.array((-np.cos(self.theta_rot/2), 
            np.sin(self.theta_rot/2), 0)) # Rotor 2 position (m)
        self.rotor3_pos = self.d*np.array((-np.cos(self.theta_rot/2), 
            -np.sin(self.theta_rot/2), 0)) # Rotor 3 position (m)
        self.rotor4_pos = self.d*np.array((np.cos(self.theta_rot/2), 
            -np.sin(self.theta_rot/2), 0)) # Rotor 4 position (m)
        self.D = np.row_stack((self.rotor1_pos, self.rotor2_pos, 
            self.rotor3_pos, self.rotor4_pos))

        # Form inertia matrix
        self.I = np.array([[self.Ixx, self.Ixy, self.Ixz], 
            [self.Iyx, self.Iyy, self.Iyz], [self.Izx, self.Izy, self.Izz]])
        self.I_inv = LA.inv(self.I)

        # Lump parameters
        self.cQ = self.cT*np.sqrt(self.cT/2) # Torque coeff (unitless) [4, 2.34]
        self.CT = self.cT*env_params.rho*self.A*self.r**2
        self.CQ = self.cQ*env_params.rho*self.A*self.r**3

        # Optional blade flap computations
        if self.theta_0 is not None:
            self.theta_tw = self.theta_tip - self.theta_0 # Twist ang (rad/r)
            self.Ib = (1/4)*self.m_b*self.r**2 # Blade I a/b flap axis (kg*m^2)
            # Locke number (unitless) [4, 4.37]
            self.gamma = (env_params.rho*self.Cl_alpha*self.c*self.r**4)/self.Ib 

# ---------------------- 3DR Iris Quad Parameters [3,5] ---------------------- #
iris_params = QuadParameters()

# UAV geometry
iris_params.d = 0.256 # Distance from UAV CoM to motors (m) [3]
iris_params.r = 0.127 # UAV blade radius (m) [3]

# UAV mass and inertia [3]
iris_params.m = 1.5 # UAV mass (kg)
iris_params.Ixx = 0.029125 # Ixx term
iris_params.Ixy = 0 # Ixy term
iris_params.Ixz = 0 # Ixz term
iris_params.Iyx = 0 # Iyx term
iris_params.Iyy = 0.029125 # Iyy term
iris_params.Iyz = 0 # Iyz term
iris_params.Izx = 0 # Izx term
iris_params.Izy = 0 # Izy term
iris_params.Izz = 0.055225 # Izz term

# Blade parameters
iris_params.c = 0.018 # Blade chord (m) [5]
iris_params.m_b = 0.005 # Blade mass (m) [3]
iris_params.theta_0 = np.radians(14.6) # Blade root angle (rad) [5]
iris_params.theta_tip = np.radians(6.8) # Blade tip angle (rad) [5]
iris_params.Cl_alpha = 5.5 # Lift-slope gradient[5]
iris_params.cT= 0.0048 # Thrust coefficient, Eq. (3) in [1], [5]

# Motor parameters
iris_params.w_max = 20000*0.10472 # Max X500v2 motor angular velocity (rad/sec)
iris_params.w_dir = np.array([1, -1, 1, -1])

# Drag parameters
iris_params.A1c = 0.1
iris_params.A1s = 0.01
iris_params.dx = 0.1
iris_params.dy = 0.1

iris_params.comp_values(env_1)

# Set params from [3]
iris_params.CT = 5.84e-6
iris_params.CQ = 0.06*iris_params.CT
iris_params.D = np.array([[0.13, 0.22, 0], 
                          [-0.13, 0.22, 0], 
                          [-0.13, -0.22, 0], 
                          [0.13, -0.22, 0]])

# -------------------- Modified Holybro X500v2 Parameters -------------------- #

x500_exp_params = QuadParameters()

# UAV geometry
x500_exp_params.d = 0.25 # Distance from UAV CoM to motors (m) 
x500_exp_params.r = 0.127 # UAV blade radius (m) 

# UAV mass and inertia 
x500_exp_params.m = 2.35 # UAV mass (kg)
x500_exp_params.Ixx = 0.0308 # Ixx term
x500_exp_params.Ixy = 0 # Ixy term
x500_exp_params.Ixz = 0 # Ixz term
x500_exp_params.Iyx = 0 # Iyx term
x500_exp_params.Iyy = 0.0354 # Iyy term
x500_exp_params.Iyz = 0 # Iyz term
x500_exp_params.Izx = 0 # Izx term
x500_exp_params.Izy = 0 # Izy term
x500_exp_params.Izz = 0.0428 # Izz term

# Motor parameters
x500_exp_params.w_max = 10000*0.10472 # Max X500v2 motor ang velocity (rad/sec)
x500_exp_params.w_dir = np.array([1, -1, 1, -1])
x500_exp_params.cT= 0.0048

# Drag parameters
x500_exp_params.A1c = 0.02147611
x500_exp_params.A1s = 0.01277551 # Not curenntly used
x500_exp_params.dx = 0.02441327
x500_exp_params.dy = 0.01487329

# x500_exp_params.A1c = 0.01471036
# x500_exp_params.A1s = 0.01277551 # Not curenntly used
# x500_exp_params.dx = 0.01953956
# x500_exp_params.dy = 0.00333595

x500_exp_params.comp_values(env_1)

# Set needed overide params
x500_exp_params.CT = 1.17901547e-05
x500_exp_params.CQ = 1.66125963e-07
# x500_exp_params.D[:, 2] = -0.05








