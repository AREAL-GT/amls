
'''
This module contains useful functions for controller development and 
implementation
'''

# --------------------------------- Imports ---------------------------------- #

import numpy as np
from numpy import linalg as LA


# -------------------------------- Functions --------------------------------- #

def descent_rate(pos_err: np.ndarray, alt: float) -> float:
    '''
    This function implements a position-error based scaling for commanding a 
    particular descent rate
    '''

    # Set descent rate parameters
    # alt_err_bound = 0.5
    max_descent_rate = 0.4
    dock_rad = 0.38
    rad_scale = 2.0
    scaled_dock = dock_rad*rad_scale

    # Compute lateral tracking error
    lat_err = LA.norm(pos_err[:2])

    # Use tracking error to compute commanded descent rate
    if alt < 0.5:
        descent_scale = (scaled_dock - lat_err)/scaled_dock
    else: # Use normal dock size at low altitude
        descent_scale = (dock_rad - lat_err)/dock_rad

    # Constrain to [0,1]
    descent_scale = np.clip(descent_scale, 0, 1)

    rate = descent_scale*max_descent_rate

    return rate

def quat_mult(q1: np.ndarray, q2: np.ndarray):
    '''
    Function to perform quaternion multiplication: q1*q2
    Assuming quaternion order is qw, qx, qy, qz
    '''

    Q_q1 = np.array([[q1[0], -q1[1], -q1[2], -q1[3]],
                     [q1[1], q1[0], -q1[3], q1[2]],
                     [q1[2], q1[3], q1[0], -q1[1]],
                     [q1[3], -q1[2], q1[1], q1[0]]])
    
    q_out = Q_q1@q2
    
    return q_out

def q_adjoint(q):
    '''
    Function to make a quaternion an adjoint
    Assuming quaternion order is qw, qx, qy, qz
    '''

    q_adj = np.insert(-q[1:], 0, q[0])

    return q_adj