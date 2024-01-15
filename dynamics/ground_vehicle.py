
'''
Module that contains the dynamics for the ground vehicle. It is a basic 6dof 
double integrator, though it is quite common that states will be directly
set in the function calling the ground vehicle dynamics (eg. directly setting
linear velocity or orientation)
'''

# --------------------------------- Imports ---------------------------------- #

# Standard Imports
import numpy as np
from numpy import linalg as LA

from scipy.spatial.transform import Rotation as Rot

# ----------------------- Default Class Init Arguments ----------------------- #

default_debug_ctl = { # Default debug settings in dict
    "compute": False # compute_dynamics method
}

default_contact_size = np.array([1, 1])

# ---------------------- Ground Vehicle Class Definition --------------------- #

class GroundVechicleDynamics:
    '''
    Class used to organize the dynamics of the mobile ground vehicle.

    Coordinate Frames:
    G - Body-fixed to ground vehicle
    W - Inertial frameß

    Notes:
    - All coordinate frames x-forward, y-right, z-down
    '''

     # Class-wide variables

    def __init__(self, contact_size: np.ndarray = default_contact_size, 
                 debug_ctl: dict = default_debug_ctl) -> None:
        '''
        Constructor method
        '''

        self.contact_size = contact_size # Size of the landing dock
        self.debug_ctl = debug_ctl # Debug control dict

    def compute_dynamics(self, t: float, state_vec: np.ndarray, 
                         u: np.ndarray = np.zeros(6)) -> np.ndarray:
        '''
        Calculate the system dynamics for the ground vehicle

        Assuming velocity states are constant, and therefore initial 
        velocity is commanded for the entire trajectory

        Required Inputs:
        t: float of the time    
        state_vec: 1d numpy vector, state vector: x, y, z, qx, qy, qz, qw, 
            dx_dt, dy_dt, dz_dt, omega_x, omega_y, omega_z

        Optional Inputs:
        u: 1d numpy vector: ux, uy, uz, uphi, utheta, upsi accelerations along
            axes and about axes
        '''

        # Enforce unit magnitude quaternion orientation
        qx = state_vec[3]
        qy = state_vec[4]
        qz = state_vec[5]
        qw = state_vec[6]
        q = np.array([qx, qy, qz, qw])
        q_unit = q/LA.norm(q)
        state_vec[3:7] = q_unit

        # Assign state vector to variables for readability
        x = state_vec[0]
        y = state_vec[1]
        z = state_vec[2]
        qx = state_vec[3]
        qy = state_vec[4]
        qz = state_vec[5]
        qw = state_vec[6]
        dx_dt_G = state_vec[7]
        dy_dt_G = state_vec[8]
        dz_dt_G = state_vec[9]
        om_x_G = state_vec[10]
        om_y_G = state_vec[11]
        om_z_G = state_vec[12]

        # Form needed vectors and matrices from states
        q = np.array([qx, qy, qz, qw])
        R_G_W = Rot.from_quat(q) # G->W
        pos_W = np.array([x, y, z])
        vel_G = np.array([dx_dt_G, dy_dt_G, dz_dt_G])
        vel_W = R_G_W.apply(vel_G)
        omega_G = np.array([om_x_G, om_y_G, om_z_G])

        # Compile system dynamics output
        dstates_dt = np.zeros(13, dtype=float) # Init return deriv state vec

        # Unforced system dynamics
        dstates_dt[0] = vel_W[0] # x velocity
        dstates_dt[1] = vel_W[1] # y velocity
        dstates_dt[2] = vel_W[2] # z velocity
        dstates_dt[3] = 0.5*(om_x_G*qw + om_z_G*qy - om_y_G*qz) # dqx_dt
        dstates_dt[4] = 0.5*(om_y_G*qw - om_z_G*qx + om_x_G*qz) # dqy_dt
        dstates_dt[5] = 0.5*(om_z_G*qw + om_y_G*qx - om_x_G*qy) # dqz_dt
        dstates_dt[6] = 0.5*(-om_x_G*qx - om_y_G*qy - om_z_G*qz) # dqw_dt
        
        # Forced system dynamics from control vector
        dstates_dt[7] = u[0] # x acceleration
        dstates_dt[8] = u[1] # y acceleration
        dstates_dt[9] = u[2] # z acceleration
        dstates_dt[10] = u[3] # alpha x angular acceleration
        dstates_dt[11] = u[4] # alpha y angular acceleration
        dstates_dt[12] = u[5] # alpha z angular acceleration

        return dstates_dt

# ------------------------ Configured Ground Vehicles ------------------------ #

dock_size = np.array([0.762, 0.762]) # 30 inches to m
dock_size = np.array([0.914, 0.914]) # 30 inches to m
ground_vehicle_1 = GroundVechicleDynamics(contact_size=dock_size)



