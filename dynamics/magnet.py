
'''
Module contains the implementation of the magnet model class
'''

# --------------------------------- Imports ---------------------------------- #

# Standard imports
import numpy as np

from scipy.spatial.transform import Rotation as Rot

# Add workspace directory to the path
import os
import sys
sim_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, sim_pkg_path)

# Workspace package imports
from amls.dynamics.quadcopter_params import QuadParameters
from amls.dynamics.quadcopter_params import iris_params, x500_exp_params
from amls.dynamics.ground_vehicle import dock_size
from amls_impulse_contact.contact import x500_cont_points


# ----------------------- Default Class Init Arguments ----------------------- #

default_cont_size = -np.ones(2) # Default size -1 means ground plane


# -------------------------- Magnet Class Definition ------------------------- #

class MagnetLanding():
    '''
    Class used to organize the methods of the model used to represent forces
    and moments from the magnetic landing gear
    '''

    # Insert class-wide variables here

    def __init__(self, points_B: np.ndarray, strength_vec: np.ndarray, 
                 quad_params: QuadParameters, epsilon_c: float = 0.001, 
                 cont_size: np.ndarray = default_cont_size) -> None:
        '''
        Constructor method

        Required Inputs:
            points_B: 2d np matrix of coords, x, y, z, by row
            strengths: 1d np vec of mag holding forces
            quad_params: QuadParameters object from quadcopter_parameters.py

        Optional Inputs:

        '''

        self.points_B = points_B
        self.num_points = self.points_B.shape[0]
        self.strength_vec = strength_vec
        self.quad_params = quad_params

        self.epsilon_c = epsilon_c # Contact zone tolerance
        self.cont_size = cont_size # Size of the contact area x and y

    def compute_forces(self, cont_mag: np.ndarray, R_B_G: Rot, 
                       R_G_W: Rot) -> np.ndarray:
        '''
        Method to compute the forces on a uav from magnetic landing gear

        Required Inputs:
            cont_mag: 1d numpy array of booleans for which magnets in contact

        '''

        # Initialize needed variables
        dstates_dt = np.zeros(13, dtype=float) # State vec time deriv
        F_arr_G = np.zeros((self.num_points, 3)) # Forces from magnets
        F_arr_B = np.zeros((self.num_points, 3)) # Forces from magnets
        tau_arr_B = np.zeros((self.num_points, 3)) # Moments from magnets

        R_G_B = R_B_G.inv()

        if any(cont_mag): # If any feet are in contact

            for i in range(self.num_points): # Loop through magnets

                p_B = self.points_B[i]

                if cont_mag[i]: # If in contact

                    # Scale magnitude of magnet based on UAV orientation
                    vec_z_dock_G = R_B_G.apply(np.array([0, 0, 1]))
                    scale_factor = vec_z_dock_G[2] # z component in G 

                    # Apply magnet force
                    F_nom_G = np.array([0, 0, self.strength_vec[i]])
                    F_arr_G[i] = scale_factor*F_nom_G

                    # Compute torque
                    F_arr_B[i] = R_G_B.apply(F_arr_G[i])
                    tau_arr_B[i] = np.cross(p_B, F_arr_B[i])

            F_G = np.sum(F_arr_G, axis=0)
            F_W = R_G_W.apply(F_G)

            # Apply forces and torques to systemoutput states
            accel_W = (1/self.quad_params.m)*F_W
            alpha_B = self.quad_params.I_inv@np.sum(tau_arr_B, axis=0)

            # Assign to state derivative vector
            dstates_dt[7:10] = accel_W 
            dstates_dt[10:] = alpha_B

        return dstates_dt
    
    def mag_check(self, points_G: np.ndarray, t: float = 0):
        '''
        Method to check for magnet contact state for all points
        '''

        # Check if z-component of point positions are within the contact bound
        z_check = points_G[:, 2] > -self.epsilon_c

        # Check if points are withing contact zone x & y bounds (if defined)
        if self.cont_size[0] != -1: # If there is a defined cont size

            # Contact zone sizing variables
            cs_x = self.cont_size[0] # Size in x-dimension
            cs_y = self.cont_size[1] # Size in y-dimension
            
            # Check zone dimensions
            x_check_neg = points_G[:, 0] < 0.5*cs_x
            x_check_pos = points_G[:, 0] > -0.5*cs_x
            y_check_neg = points_G[:, 1] < 0.5*cs_y
            y_check_pos = points_G[:, 1] > -0.5*cs_y

            zone_check = np.all([x_check_neg, x_check_pos, 
                                 y_check_neg, y_check_pos], axis=0)

        else: 

            zone_check = np.full(self.num_points, True, dtype=bool)

        # Combine the three checks
        all_check = np.all([z_check, zone_check], axis=0) 

        return all_check

# --------------------- Configured Magnetic Landing Gear --------------------- #

# Magnet model setup based on iris UAV parameters
mag_contact_points_iris = \
    np.array([[iris_params.d/2, iris_params.d/2, iris_params.d/3],
              [-iris_params.d/2, iris_params.d/2, iris_params.d/3], 
              [-iris_params.d/2, -iris_params.d/2, iris_params.d/3],
              [iris_params.d/2, -iris_params.d/2, iris_params.d/3]]) 
contact_size1 = dock_size # 30 inches to m
mag_strength1 = 2*43.86 # Force in N - K&J D66-N52 magnet x 2 per foot
strength_vec = mag_strength1*np.ones(mag_contact_points_iris.shape[0])
mag_land_iris = MagnetLanding(mag_contact_points_iris, strength_vec, 
                              iris_params, cont_size=contact_size1)

# Magnet model setup based on experimental x500 parameters
mag_contact_points_x500 = x500_cont_points
contact_size1 = dock_size 
mag_strength1 = 2*43.86 # Force in N - K&J D66-N52 magnet x 2 per foot
strength_vec = mag_strength1*np.ones(mag_contact_points_x500.shape[0])
mag_land_x500 = MagnetLanding(mag_contact_points_x500, strength_vec, 
                              x500_exp_params, cont_size=contact_size1)

# Magnet model setup based on experimental x500 parameters
mag_contact_points_x500 = x500_cont_points
contact_size1 = dock_size 
mag_strength2 = 2*80.5 # Force in N - K&J D88-N52 magnet x 2 per foot
strength_vec2 = mag_strength2*np.ones(mag_contact_points_x500.shape[0])
mag_land_x500_strong = MagnetLanding(mag_contact_points_x500, strength_vec2, 
                                    x500_exp_params, cont_size=contact_size1)

# Zero-strength magnet model setup based on experimental x500 parameters
mag_contact_points_x500 = x500_cont_points
contact_size1 = dock_size 
mag_strength1 = 0 # Force in N 
strength_vec = mag_strength1*np.ones(mag_contact_points_x500.shape[0])
mag_0_x500 = MagnetLanding(mag_contact_points_x500, strength_vec, 
                           x500_exp_params, cont_size=contact_size1)

# Magnet model for partial magnetic landing gear
mag_cont_points_partial = iris_params.D[:2]
strength_vec_partial = mag_strength1*np.ones(mag_cont_points_partial.shape[0])
mag_land_partial = MagnetLanding(mag_cont_points_partial, strength_vec_partial, 
                                 iris_params, cont_size=contact_size1)

