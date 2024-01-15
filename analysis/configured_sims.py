'''
This module contains configured simulation classes. These can be directly 
imported by other simulation files and run
'''

# --------------------------------- Imports ---------------------------------- #

# Add package and workspace directories to the path
import os
import sys
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)

# Standard package imports
import numpy as np
from numpy import linalg as LA

import pandas as pd

from scipy.spatial.transform import Rotation as Rot

from types import MethodType

from copy import deepcopy

# Workspace package imports
from amls_sim import simulator

from amls_impulse_contact.contact import uav_contact_iris, \
    uav_contact_iris_noz, uav_contact_x500_exp

# from amls.dynamics import quadcopter
from amls.dynamics.quadcopter_params import iris_params, x500_exp_params
from amls.dynamics.quadcopter import quad_iris, quad_x500
from amls.dynamics.magnet import mag_land_iris, mag_land_partial, \
    mag_land_x500, mag_land_x500_strong, mag_0_x500
from amls.dynamics.ground_vehicle import ground_vehicle_1

# from amls.control.px4_stock import px4_mod_iris,
from amls.control.px4_mod import px4_mod_iris, px4_mod_x500
from amls.control.geometric_control import geom_x500
from amls.control import utility_functions as uf_con

# ------------------------ Select Configured Objects ------------------------- #

# Select the configured controller, dynamics, contact model, etc. to use in
# the configured simulations

dyn_use = quad_x500
params_use = x500_exp_params
control_use = geom_x500
contact_use = uav_contact_x500_exp
mag_use = mag_land_x500

# ------------------------- System Dynamics Methods -------------------------- #

def sysdyn_uav_fall_mag(self, t: float, x: np.ndarray):
    '''
    Method mixin used by the numerical integrator, including quadcopter system
    dynamics with no propeller inputs and ground

    State vector:
      UAV states: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, om_x, om_y, om_z
      GV states: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, om_x, om_y, om_z
    '''

    # Divide states into uav and ground vehicle
    x_uav = x[0:13]
    x_gv = x[13:]

    # Set control inputs for uav and ground vehicle
    u_uav = params_use.w_min*np.array([1, -1, 1, -1])
    u_gv_lin_vel = np.array([0, 0, 0]) # Assuming little gv movement
    u_gv_ang_vel = np.array([0, 0, 0])
    u_gv = np.concatenate([u_gv_lin_vel, u_gv_ang_vel])

    # Compute the uav dynamics
    dx_dt_uav = dyn_use.compute_dynamics(t, x_uav, u_uav) # Quad dynamics

    # Compute mag forcing
    R_B_W = Rot.from_quat(x_uav[3:7])
    R_G_W = Rot.from_quat(x_gv[3:7])
    R_W_G = R_G_W.inv()
    R_B_G = R_W_G*R_B_W
    dx_dt_mag = mag_use.compute_forces(self.cont_mag, R_B_G, R_G_W) 
    dx_dt_uav += dx_dt_mag # Add forcing from magnets

    # Compute the ground vehicle dynamics
    dx_dt_gv = ground_vehicle_1.compute_dynamics(t, x_gv, u_gv)

    # Concatenate dynamics into output vector
    dx_dt = np.concatenate([dx_dt_uav, dx_dt_gv])

    return dx_dt 

def sysdyn_uav_fall_partial_mag(self, t: float, x: np.ndarray):
    '''
    Method mixin used by the numerical integrator, including quadcopter system
    dynamics with no propeller inputs and ground

    State vector:
      UAV states: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, om_x, om_y, om_z
      GV states: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, om_x, om_y, om_z
    '''

    # Divide states into uav and ground vehicle
    x_uav = x[0:13]
    x_gv = x[13:]

    # Set control inputs for uav and ground vehicle
    u_uav = params_use.w_min*np.array([1, -1, 1, -1])
    u_gv_lin_vel = np.array([0, 0, 0]) # Assuming little gv movement
    u_gv_ang_vel = np.array([0, 0, 0])
    u_gv = np.concatenate([u_gv_lin_vel, u_gv_ang_vel])

    # Compute the uav dynamics
    dx_dt_uav = dyn_use.compute_dynamics(t, x_uav, u_uav) # Quad dynamics

    # Compute mag forcing
    R_B_W = Rot.from_quat(x_uav[3:7])
    R_G_W = Rot.from_quat(x_gv[3:7])
    R_W_G = R_G_W.inv()
    R_B_G = R_W_G*R_B_W
    dx_dt_mag = mag_land_partial.compute_forces(self.cont_mag, R_B_G, R_G_W) 
    dx_dt_uav += dx_dt_mag # Add forcing from magnets

    # Compute the ground vehicle dynamics
    dx_dt_gv = ground_vehicle_1.compute_dynamics(t, x_gv)

    # Concatenate dynamics into output vector
    dx_dt = np.concatenate([dx_dt_uav, dx_dt_gv])

    return dx_dt 

def sysdyn_uav_fall_nomag(self, t: float, x: np.ndarray):
    '''
    Method mixin used by the numerical integrator, including quadcopter system
    dynamics with no propeller inputs and ground

    State vector:
      UAV states: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, om_x, om_y, om_z
      GV states: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, om_x, om_y, om_z
    '''

    # Divide states into uav and ground vehicle
    x_uav = x[0:13]
    x_gv = x[13:]

    # Set control inputs for uav and ground vehicle
    u_uav = params_use.w_min*np.array([1, -1, 1, -1])
    u_gv_lin_vel = np.array([0, 0, 0]) # Assuming little gv movement
    u_gv_ang_vel = np.array([0, 0, 0])
    u_gv = np.concatenate([u_gv_lin_vel, u_gv_ang_vel])

    # Compute the uav dynamics
    dx_dt_uav = dyn_use.compute_dynamics(t, x_uav, u_uav) # Quad dynamics

    # Compute the ground vehicle dynamics
    dx_dt_gv = ground_vehicle_1.compute_dynamics(t, x_gv)

    # Concatenate dynamics into output vector
    dx_dt = np.concatenate([dx_dt_uav, dx_dt_gv])

    return dx_dt 

def sysdyn_uav_auton(self, t: float, x: np.ndarray):
    '''
    Method mixin used by the numerical integrator, including quadcopter system
    dynamics with control inputs and ground vehicle

    State vector:
      UAV states: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, om_x, om_y, om_z
      GV states: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, om_x, om_y, om_z
    '''

    # Divide states into uav and ground vehicle
    x_uav = x[0:13]
    x_gv = x[13:]

    # Compute the uav dynamics
    dx_dt_uav = dyn_use.compute_dynamics(t, x_uav, self.u_uav) # Quad dyn

    # Compute mag forcing
    R_B_W = Rot.from_quat(x_uav[3:7])
    R_G_W = Rot.from_quat(x_gv[3:7])
    R_W_G = R_G_W.inv()
    R_B_G = R_W_G*R_B_W
    dx_dt_mag = mag_use.compute_forces(self.cont_mag, R_B_G, R_G_W) 
    dx_dt_uav += dx_dt_mag # Add forcing from magnets

    # Compute the ground vehicle dynamics
    dx_dt_gv = ground_vehicle_1.compute_dynamics(t, x_gv, u=self.u_gv)

    # Concatenate dynamics into output vector
    dx_dt = np.concatenate([dx_dt_uav, dx_dt_gv])

    return dx_dt 

def sysdyn_uav_ctl_flight(self, t: float, x_uav: np.ndarray):
    '''
    Method mixin used by the numerical integrator, including quadcopter system
    dynamics with control inputs in free flight with no gv

    State vector:
      UAV states: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, om_x, om_y, om_z
    '''

    # Compute the uav dynamics
    dx_dt_uav = dyn_use.compute_dynamics(t, x_uav, self.u_uav) # Quad dyn

    return dx_dt_uav

# ------------------------ State Modification Methods ------------------------ #

def state_mod_uav_fall_mag(self, t: float, x: np.ndarray, i: int):
    '''
    Method for direct change of system states and parameters:
    - Contact model on object velocities
    '''

    # Divide states into uav and ground vehicle
    x_uav = x[0:13]
    x_gv = x[13:]

    # Enforce unit magnitude quaternion orientation on uav and ground vehicle
    q_unit = x_uav[3:7]/LA.norm(x_uav[3:7])
    x_uav[3:7] = q_unit
    q_unit = x_gv[3:7]/LA.norm(x_gv[3:7])
    x_gv[3:7] = q_unit

    # Isolate out needed uav states and variables for contact model inputs
    pos_uav_W = x_uav[:3] # UAV states
    R_Q_W = Rot.from_quat(x_uav[3:7])
    lin_vel_uav_W = x_uav[7:10]
    ang_vel_uav_Q = x_uav[10:] 

    # Apply contact model to system dynamics
    del_lin_vel_uav_W, del_ang_vel_uav_Q, del_z_pos_uav_W, cont_check_vec, \
        df_cont_tracking = contact_use.contact_main(pos_uav_W, R_Q_W, 
            lin_vel_uav_W, ang_vel_uav_Q, cont_states=x_gv, t=t)
    
    # Generate magnet contact check vector
    self.cont_mag = mag_use.mag_check(contact_use.points_G, t)

    # Combine state changes back together into output vector
    uav_state_changes = np.concatenate((del_z_pos_uav_W, np.zeros(4), 
                                        del_lin_vel_uav_W, del_ang_vel_uav_Q))
    x_out = np.concatenate((x_uav + uav_state_changes, x_gv))

    return x_out

def state_mod_uav_fall_partial_mag(self, t: float, x: np.ndarray, i: int):
    '''
    Method for direct change of system states and parameters:
    - Contact model on object velocities
    '''

    # Divide states into uav and ground vehicle
    x_uav = x[0:13]
    x_gv = x[13:]

    # Enforce unit magnitude quaternion orientation on uav and ground vehicle
    q_unit = x_uav[3:7]/LA.norm(x_uav[3:7])
    x_uav[3:7] = q_unit
    q_unit = x_gv[3:7]/LA.norm(x_gv[3:7])
    x_gv[3:7] = q_unit

    # Isolate out needed uav states and variables for contact model inputs
    pos_uav_W = x_uav[:3] # UAV states
    R_Q_W = Rot.from_quat(x_uav[3:7])
    lin_vel_uav_W = x_uav[7:10]
    ang_vel_uav_Q = x_uav[10:] 

    # Apply contact model to system dynamics
    del_lin_vel_uav_W, del_ang_vel_uav_Q, del_z_pos_uav_W, cont_check_vec, \
        df_cont_tracking = contact_use.contact_main(pos_uav_W, R_Q_W, 
            lin_vel_uav_W, ang_vel_uav_Q, cont_states=x_gv, t=t)
    
    # Generate magnet contact check vector
    self.cont_mag = \
        mag_land_partial.mag_check(contact_use.points_G[:2], t)

    # Combine state changes back together into output vector
    uav_state_changes = np.concatenate((del_z_pos_uav_W, np.zeros(4), 
                                        del_lin_vel_uav_W, del_ang_vel_uav_Q))
    x_out = np.concatenate((x_uav + uav_state_changes, x_gv))

    return x_out

def state_mod_uav_fall_nomag(self, t: float, x: np.ndarray, i: int):
    '''
    Method for direct change of system states and parameters:
    - Contact model on object velocities
    '''

    # Divide states into uav and ground vehicle
    x_uav = x[0:13]
    x_gv = x[13:]

    # Enforce unit magnitude quaternion orientation on uav and ground vehicle
    q_unit = x_uav[3:7]/LA.norm(x_uav[3:7])
    x_uav[3:7] = q_unit
    q_unit = x_gv[3:7]/LA.norm(x_gv[3:7])
    x_gv[3:7] = q_unit

    # Isolate out needed uav states and variables for contact model inputs
    pos_uav_W = x_uav[:3] # UAV states
    R_Q_W = Rot.from_quat(x_uav[3:7])
    lin_vel_uav_W = x_uav[7:10]
    ang_vel_uav_Q = x_uav[10:] 

    # Apply contact model to system dynamics
    del_lin_vel_uav_W, del_ang_vel_uav_Q, del_z_pos_uav_W, cont_check_vec, \
        df_cont_tracking = contact_use.contact_main(pos_uav_W, R_Q_W, 
            lin_vel_uav_W, ang_vel_uav_Q, cont_states=x_gv, t=t)

    # Combine state changes back together into output vector
    uav_state_changes = np.concatenate((del_z_pos_uav_W, np.zeros(4), 
                                        del_lin_vel_uav_W, del_ang_vel_uav_Q))
    x_out = np.concatenate((x_uav + uav_state_changes, x_gv))

    return x_out

def state_mod_uav_auton(self, t: float, x: np.ndarray, i: int):
    '''
    Method for direct change of system states and parameters:
    - Contact model on object velocities
    - Set controller commands
    '''

    # Divide states into uav and ground vehicle
    x_uav = x[0:13]
    x_gv = x[13:]

    pos_gv_W = x_gv[:3]

    # Enforce unit magnitude quaternion orientation on uav and ground vehicle
    q_unit = x_uav[3:7]/LA.norm(x_uav[3:7])
    x_uav[3:7] = q_unit
    q_unit = x_gv[3:7]/LA.norm(x_gv[3:7])
    x_gv[3:7] = q_unit

    # Isolate out needed uav states and variables for contact model inputs
    pos_uav_W = x_uav[:3] # UAV states
    R_Q_W = Rot.from_quat(x_uav[3:7])
    lin_vel_uav_W = x_uav[7:10]
    ang_vel_uav_Q = x_uav[10:] 

    # Make version of x_gv with lin vel in W frame instead of G
    x_gv_cont = deepcopy(x_gv).astype(float)
    R_G_W = Rot.from_quat(x_gv_cont[3:7])
    lin_vel_gv_W = R_G_W.apply(x_gv_cont[7:10])
    x_gv_cont[7:10] = lin_vel_gv_W

    # Apply contact model to system dynamics
    del_lin_vel_uav_W, del_ang_vel_uav_Q, del_z_pos_uav_W, cont_check_vec, \
        df_cont_tracking = contact_use.contact_main(pos_uav_W, R_Q_W, 
            lin_vel_uav_W, ang_vel_uav_Q, cont_states=x_gv_cont, t=t)
    
    # Generate magnet contact check vector
    self.cont_mag = mag_use.mag_check(contact_use.points_G, t)

    # Compute the direct change of gv angular velocity for rad of curve
    if self.rad_des == -1: # Assign to straight line
        x_gv[10:] = 0
    else: # Compute angular velocity for radius of curvature
        dx_dt_G = x_gv[7]
        om_z_G = dx_dt_G/self.rad_des
        x_gv[12] = om_z_G

    # Combine state changes back together into output vector
    uav_state_changes = np.concatenate((del_z_pos_uav_W, np.zeros(4), 
                                        del_lin_vel_uav_W, del_ang_vel_uav_Q))
    x_out = np.concatenate((x_uav + uav_state_changes, x_gv)).astype(float)

    # Compute controller action given the states this timestep
    cont_check = any(cont_check_vec)

    if cont_check: # Flip contact state to true if contact on any point
        if not self.in_contact:
            print('[state_mod] Touchdown at t = %0.4f' % t)
            self.in_contact = True
            self.td_time = t

    if self.in_contact: # If any contact ever with landing zone
        self.u_uav = params_use.w_min*np.array([1, -1, 1, -1])
        self.T_vec = control_use.Gamma@self.u_uav
        setpoint_vec = self.setpoint_df.iloc[i, 1:]

    else: # In free flight

        # Position setpoint to track gv
        init_z = self.state_traj.iloc[0, 3]
        # pos_set = pos_gv_W + np.array([0, 0, 1.0*init_z])
        pos_set = pos_gv_W + np.array([0, 0, x_out[2]])

        b1d = R_G_W.apply(np.array([1, 0, 0]))

        # Compute commanded descent rate
        pos_err = x_out[:3] - pos_set
        descent_rate = uf_con.descent_rate(pos_err, x_out[2])

        # Add small time limit to allow some settling
        if self.a_cmd_gv[0] > 0:
            t_accel = self.vgoal_gv[0]/self.a_cmd_gv[0]
        else:
            t_accel = 0

        if t < t_accel:
            descent_rate = 0

        # Compile control command dictionary
        control_dict = {
            'pos': pos_set,
            'vel': np.array([x_gv_cont[7], x_gv_cont[8], descent_rate]),
            'accel': None,
            'quat': None,
            'omega': None,
            'heading_vec': np.array([b1d, [0, 0 ,0], [0, 0 ,0]]),
            'psi': 0,
            'Tvec': None,
            'Th': None,
            'rotor': None
        }

        self.u_uav, self.T_vec, setpoint_vec = self.uav_ctl_use.control_main(t, 
            x_out, control_dict)
        
    # Compute the ground vehicle acceleration controls
    vel_gv_G = x_gv[7:10]
    om_gv_G = x_gv[10:]

    vel_check = vel_gv_G < self.vgoal_gv[:3]

    om_check = om_gv_G < self.vgoal_gv[3:]
    self.u_gv[:3] = vel_check*self.a_cmd_gv[:3]
    self.u_gv[3:] = om_check*self.a_cmd_gv[3:]

    # Save controller actions and setpoints to dataframe output
    self.setpoint_df.iloc[i+1, 0] = t
    self.setpoint_df.iloc[i+1, 1:] = setpoint_vec
    self.ctl_df.iloc[i+1, 0] = t
    self.ctl_df.iloc[i+1, 1:5] = self.u_uav
    self.ctl_df.iloc[i+1, 5:] = self.T_vec

    return x_out

def state_mod_uav_vel_ramp(self, t: float, x_uav: np.ndarray, i: int):
    '''
    Method for direct change of system states and parameters for the uav in
    free flight:
    - Enforce unit-magnitude quaternions
    - Set controller commands for linear velocity ramp up
    '''

    # Enforce unit magnitude quaternion orientation on uav and ground vehicle
    q_unit = x_uav[3:7]/LA.norm(x_uav[3:7])
    x_uav[3:7] = q_unit

    # Compute controller action given the states this timestep
    pos_set = np.array([0, 0, -2.0]) # Position setpoint, vel will override

    # Compile control command dictionary
    vel_cmd_x = self.vel_cmd[0]
    vel_cmd_y = self.vel_cmd[1]
    # vel_cmd_z = self.vel_cmd[2]
    control_dict = {
        'pos': pos_set,
        'vel': (vel_cmd_x, vel_cmd_y, None),
        'accel': None,
        'quat': None,
        'omega': None,
        'Tvec': None,
        'Th': None,
        'rotor': None
    }

    psi_set = 0

    self.u_uav, self.T_vec, setpoint_vec = self.uav_ctl_use.control_main(t, 
        x_uav, control_dict, psi_set)

    # Save controller actions and setpoints to dataframe output
    self.setpoint_df.iloc[i+1, 0] = t
    self.setpoint_df.iloc[i+1, 1:] = setpoint_vec
    self.ctl_df.iloc[i+1, 0] = t
    self.ctl_df.iloc[i+1, 1:5] = self.u_uav
    self.ctl_df.iloc[i+1, 5:] = self.T_vec

    return x_uav

def state_mod_uav_drag(self, t: float, x_uav: np.ndarray, i: int):
    '''
    Method for direct change of system states and parameters for the uav in
    free flight:
    - Enforce unit-magnitude quaternions
    - Set controller commands for stabilization with initial velocity
    '''

    # Enforce unit magnitude quaternion orientation on uav and ground vehicle
    q_unit = x_uav[3:7]/LA.norm(x_uav[3:7])
    x_uav[3:7] = q_unit

    # Compute controller action given the states this timestep
    pos_set = np.array([x_uav[0], x_uav[1], -2.0]) 

    # Compile control command dictionary
    quat_cmd = self.quat_cmd
    Th_cmd = self.Th_cmd

    control_dict = {
        'pos': pos_set,
        'vel': (None, None, None),
        'accel': None,
        'quat': quat_cmd,
        'omega': None,
        'Tvec': None,
        'Th': Th_cmd,
        'rotor': None
    }

    psi_set = 0

    self.u_uav, self.T_vec, setpoint_vec = self.uav_ctl_use.control_main(t, 
        x_uav, control_dict, psi_set)

    # Save controller actions and setpoints to dataframe output
    self.setpoint_df.iloc[i+1, 0] = t
    self.setpoint_df.iloc[i+1, 1:] = setpoint_vec
    self.ctl_df.iloc[i+1, 0] = t
    self.ctl_df.iloc[i+1, 1:5] = self.u_uav
    self.ctl_df.iloc[i+1, 5:] = self.T_vec

    return x_uav

# ------------------ State-Based Termination Check Methods ------------------- #

def term_check_landing(self, t, state_vec):
    '''
    Method to check early termination of the simulation based on a sucessful
    landing on the ground vehicle
    The checks performed for successful landing are:
    - Linear velocity uav-gv state match
    - Angular velocity uav-gv state match
    - uav com within contact zone and at correct z height
    - small time elapsed
    '''

    term_check = False # Initialize to no termination
    
    # Define tolerances for landing state agreement
    tol_lin_vel = 0.1
    tol_ang_vel = 0.75
    tol_pos = 0.003
    time_tol = 0.25

    # Check linear velocity between uav and gv
    lin_vel_uav_W = state_vec[7:10].astype(float)
    lin_vel_gv_G = state_vec[20:23].astype(float)
    R_G_W = Rot.from_quat(state_vec[16:20])
    lin_vel_gv_W = R_G_W.apply(lin_vel_gv_G)
    del_lin_vel = abs(lin_vel_uav_W - lin_vel_gv_W)
    lin_vel_check = np.all(del_lin_vel < tol_lin_vel)

    # Check angular velocity between uav and gv
    ang_vel_uav = state_vec[10:13]
    ang_vel_gv = state_vec[23:]
    del_ang_vel = abs(ang_vel_uav - ang_vel_gv)
    ang_vel_check = np.all(del_ang_vel < tol_ang_vel)

    # Check uav com in collision zone (or exempt if ground plane)
    pos_uav = state_vec[0:3]
    pos_gv = state_vec[13:16]
    del_pos = pos_uav - pos_gv
    if self.contact_size[0] == -1: # If a ground plane

        lat_check = True # Lateral check always true

    else: # If a defined contact size

        # Contact zone sizing variables
        cs_x = self.contact_size[0] # Size in x-dimension
        cs_y = self.contact_size[1] # Size in y-dimension

        # Check zone dimensions
        x_check_neg = del_pos[0] < 0.5*cs_x
        x_check_pos = del_pos[0] > -0.5*cs_x
        y_check_neg = del_pos[1] < 0.5*cs_y
        y_check_pos = del_pos[1] > -0.5*cs_y

        lat_check = np.all([x_check_neg, x_check_pos, 
                            y_check_neg, y_check_pos], axis=0)
        
    # Check z position of uav
    del_z_gear = del_pos[2] - (-self.landing_gear_height)

    if abs(del_z_gear) < tol_pos: # If within tolerance
        alt_check = True
    else:
        alt_check = False

    # Form overall state-based check vector
    state_check_vec = [lin_vel_check, ang_vel_check, lat_check, alt_check]

    if all(state_check_vec): # If landed states agree

        del_t = t - self.term_time

        if del_t > time_tol:
            
            print(' Successful Landing: sim end at t = %0.4f' % t)
            term_check = True # End simulation

    else: # If landed states do not agree

        self.term_time = t # Reset termination time check

    if 'td_time' in self.__dict__: # If touchdown time variable exists
        if self.td_time is not None: # If a touchdown time has been set
            
            # End simulation if time tolerance past touchdown
            if t - self.td_time > 1.0:
                term_check = True 
                print(' Touchdown Timeout: sim end at t = %0.4f' % t)

    return term_check

# ------------------------- Sim Reset Mixin Methods -------------------------- #

def reset_ctl(self) -> None:
    '''
    Method to reset the sim setpoint and control dataframes
    '''

    # Reset setpoint dataframe
    self.setpoint_df = pd.DataFrame(columns=self.setpoint_cols, 
                                    index=range(self.n))
    self.setpoint_df.iloc[0, 0] = self.tspan[0]
    self.setpoint_df.iloc[0, 1:4] = np.zeros(3)
    self.setpoint_df.iloc[0, 4:8] = np.array([0, 0, 0, 1])
    self.setpoint_df.iloc[0, 8:] = np.zeros(6)

    # Reset control dataframe
    self.T_vec = self.uav_ctl_use.T_vec_last # Initialize Tvec variable
    
    self.ctl_df = pd.DataFrame(columns=self.ctl_cols, index=range(self.n))
    self.ctl_df.iloc[0, 0] = self.tspan[0]
    self.ctl_df.iloc[0, 1:] = np.zeros(8)

    self.in_contact = False
    self.td_time = None

    self.uav_ctl_use.reset_ctl() # Reset controller object

# ---------------- Sim Config: PX4 Controlled UAV with Magnets --------------- #

# Initial simulation time parameters (Possibly overwritten in batch run)
tspan = (0, 1.0)
timestep = 0.001

# State Vector: 
#   x_uav, y_uav, z_uav, qx_uav, qy_uav, qz_uav, qw_uav, 
#       dx_dt_uav, dy_dt_uav, dz_dt_uav, omega_x_uav, omega_y_uav, omega_z_uav
#   x_gv, y_gv, z_gv, qx_gv, qy_gv, qz_gv, qw_gv, 
#       dx_dt_gv, dy_dt_gv, dz_dt_gv, omega_x_gv, omega_y_gv, omega_z_gv

state_names = ['x_uav', 'y_uav', 'z_uav', 
               'qx_uav', 'qy_uav', 'qz_uav', 'qw_uav', 
               'dx_uav', 'dy_uav', 'dz_uav', 
               'om_x_uav', 'om_y_uav', 'om_z_uav',
               'x_gv', 'y_gv', 'z_gv', 
               'qx_gv', 'qy_gv', 'qz_gv', 'qw_gv', 
               'dx_gv', 'dy_gv', 'dz_gv', 
               'om_x_gv', 'om_y_gv', 'om_z_gv']

# Placeholder initial condition vector (overwritten in batch run)
x0 = np.zeros(len(state_names))

# Turn off default sim outputs
output_ctl_1 = { # Default simulation output control for class
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'default', # Name for output file, codes unset
    'file': False, # Save output file
    'plots': False # Save state plot
}

# Initialize simulation
sim_uav_auton = simulator.Simulator(tspan, x0, timestep=timestep, 
                            state_names=state_names, output_ctl=output_ctl_1)

# Overwrite default state modification and system dynamics methods
sim_uav_auton.state_mod_fun = MethodType(state_mod_uav_auton, sim_uav_auton)
sim_uav_auton.sysdyn = MethodType(sysdyn_uav_auton, sim_uav_auton)

# Overwrite default termination check method
sim_uav_auton.term_check = MethodType(term_check_landing, sim_uav_auton)

# Set controller to use
sim_uav_auton.uav_ctl_use = control_use

# Add control reset method
sim_uav_auton.reset_ctl = MethodType(reset_ctl, sim_uav_auton)

# Add additional required parameters to the sim class
sim_uav_auton.cont_mag = np.full(contact_use.num_points, False, dtype=bool)
sim_uav_auton.contact_size = ground_vehicle_1.contact_size
sim_uav_auton.landing_gear_height = contact_use.points_B[0, 2]

sim_uav_auton.td_time = None

sim_uav_auton.u_uav = control_use.w_vec_last # Initialize u variable
sim_uav_auton.u_gv = np.zeros(6) # Initialize gv control commands
# sim_uav_auton.a_cmd_gv = np.array([2.5, 0, 0, 0, 0, 0]) # Set gv accel commands
# sim_uav_auton.vgoal_gv = np.array([1.0, 0, 0, 0, 0, 0]) # gv target velocities
# sim_uav_auton.rad_des = 10 # Radius in m, or None for straight line

# Initialize setpoint variables
sim_uav_auton.setpoint_cols = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 
                               'dx', 'dy', 'dz', 'om_x', 'om_y', 'om_z']
sim_uav_auton.setpoint_df = pd.DataFrame(columns=sim_uav_auton.setpoint_cols, 
                                    index=range(sim_uav_auton.n))
sim_uav_auton.setpoint_df.iloc[0, 0] = tspan[0]
pos_set = np.zeros(3)
sim_uav_auton.setpoint_df.iloc[0, 1:4] = pos_set
sim_uav_auton.setpoint_df.iloc[0, 4:8] = np.array([0, 0, 0, 1])
sim_uav_auton.setpoint_df.iloc[0, 8:] = np.zeros(6)

sim_uav_auton.T_vec = control_use.T_vec_last # Initialize Tvec variable
sim_uav_auton.ctl_cols = ['t', 'u1', 'u2', 'u3', 'u4', 
                          'Th', 'tau_x', 'tau_y', 'tau_z']
sim_uav_auton.ctl_df = pd.DataFrame(columns=sim_uav_auton.ctl_cols, 
                                    index=range(sim_uav_auton.n))
sim_uav_auton.ctl_df.iloc[0, 0] = tspan[0]
sim_uav_auton.ctl_df.iloc[0, 1:] = np.zeros(8)

# sim_uav_auton.uav_params = iris_params # Pass iris params to state mod method
sim_uav_auton.in_contact = False
num_points = contact_use.points_B.shape[0]
sim_uav_auton.cont_mag = np.full(num_points, False, dtype=bool)

# -------------- Sim Config: Unpowered Falling UAV with Magnets -------------- #

# Initial simulation time parameters (Possibly overwritten in batch run)
tspan = (0, 1.0)
timestep = 0.0005

# State Vector: 
#   x_uav, y_uav, z_uav, qx_uav, qy_uav, qz_uav, qw_uav, 
#       dx_dt_uav, dy_dt_uav, dz_dt_uav, omega_x_uav, omega_y_uav, omega_z_uav
#   x_gv, y_gv, z_gv, qx_gv, qy_gv, qz_gv, qw_gv, 
#       dx_dt_gv, dy_dt_gv, dz_dt_gv, omega_x_gv, omega_y_gv, omega_z_gv

state_names = ['x_uav', 'y_uav', 'z_uav', 
               'qx_uav', 'qy_uav', 'qz_uav', 'qw_uav', 
               'dx_uav', 'dy_uav', 'dz_uav', 
               'om_x_uav', 'om_y_uav', 'om_z_uav',
               'x_gv', 'y_gv', 'z_gv', 
               'qx_gv', 'qy_gv', 'qz_gv', 'qw_gv', 
               'dx_gv', 'dy_gv', 'dz_gv', 
               'om_x_gv', 'om_y_gv', 'om_z_gv']

# Placeholder initial condition vector (overwritten in batch run)
x0 = np.zeros(len(state_names))

# Turn off default sim outputs
output_ctl_1 = { # Default simulation output control for class
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'default', # Name for output file, codes unset
    'file': False, # Save output file
    'plots': False # Save state plot
}

# Initialize simulation
sim_uav_fall_mag = simulator.Simulator(tspan, x0, timestep=timestep, 
                                   state_names=state_names, 
                                   output_ctl=output_ctl_1)

# Overwrite default state modification and system dynamics methods
sim_uav_fall_mag.state_mod_fun = MethodType(state_mod_uav_fall_mag, 
                                            sim_uav_fall_mag)
sim_uav_fall_mag.sysdyn = MethodType(sysdyn_uav_fall_mag, sim_uav_fall_mag)

# Overwrite default termination check method
sim_uav_fall_mag.term_check = MethodType(term_check_landing, sim_uav_fall_mag)

# Add additional required parameters to the sim class
sim_uav_fall_mag.cont_mag = np.full(uav_contact_iris.num_points, False,
                                     dtype=bool)
sim_uav_fall_mag.contact_size = ground_vehicle_1.contact_size
sim_uav_fall_mag.landing_gear_height = contact_use.points_B[0, 2]

# ---------- Sim Config: Unpowered Falling UAV with Partial Magnets ---------- #

# Initial simulation time parameters (Possibly overwritten in batch run)
tspan = (0, 1.0)
timestep = 0.0005

# State Vector: 
#   x_uav, y_uav, z_uav, qx_uav, qy_uav, qz_uav, qw_uav, 
#       dx_dt_uav, dy_dt_uav, dz_dt_uav, omega_x_uav, omega_y_uav, omega_z_uav
#   x_gv, y_gv, z_gv, qx_gv, qy_gv, qz_gv, qw_gv, 
#       dx_dt_gv, dy_dt_gv, dz_dt_gv, omega_x_gv, omega_y_gv, omega_z_gv

state_names = ['x_uav', 'y_uav', 'z_uav', 
               'qx_uav', 'qy_uav', 'qz_uav', 'qw_uav', 
               'dx_uav', 'dy_uav', 'dz_uav', 
               'om_x_uav', 'om_y_uav', 'om_z_uav',
               'x_gv', 'y_gv', 'z_gv', 
               'qx_gv', 'qy_gv', 'qz_gv', 'qw_gv', 
               'dx_gv', 'dy_gv', 'dz_gv', 
               'om_x_gv', 'om_y_gv', 'om_z_gv']

# Placeholder initial condition vector (overwritten in batch run)
x0 = np.zeros(len(state_names))

# Turn off default sim outputs
output_ctl_1 = { # Default simulation output control for class
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'default', # Name for output file, codes unset
    'file': False, # Save output file
    'plots': False # Save state plot
}

# Initialize simulation
sim_uav_fall_partial_mag = simulator.Simulator(tspan, x0, timestep=timestep, 
                            state_names=state_names, output_ctl=output_ctl_1)

# Overwrite default state modification and system dynamics methods
sim_uav_fall_partial_mag.state_mod_fun = \
    MethodType(state_mod_uav_fall_partial_mag, sim_uav_fall_partial_mag)

sim_uav_fall_partial_mag.sysdyn = MethodType(sysdyn_uav_fall_partial_mag, 
                                             sim_uav_fall_partial_mag)

# Overwrite default termination check method
sim_uav_fall_partial_mag.term_check = MethodType(term_check_landing, 
                                                 sim_uav_fall_mag)

# Add additional required parameters to the sim class
sim_uav_fall_partial_mag.cont_mag = np.full(2, False, dtype=bool)
sim_uav_fall_partial_mag.contact_size = ground_vehicle_1.contact_size
sim_uav_fall_partial_mag.landing_gear_height = \
    contact_use.points_B[0, 2]

# ------------ Sim Config: Unpowered Falling UAV without Magnets ------------- #
# Initial simulation time parameters (Possibly overwritten in batch run)
tspan = (0, 1.0)
timestep = 0.0005

# State Vector: 
#   x_uav, y_uav, z_uav, qx_uav, qy_uav, qz_uav, qw_uav, 
#       dx_dt_uav, dy_dt_uav, dz_dt_uav, omega_x_uav, omega_y_uav, omega_z_uav
#   x_gv, y_gv, z_gv, qx_gv, qy_gv, qz_gv, qw_gv, 
#       dx_dt_gv, dy_dt_gv, dz_dt_gv, omega_x_gv, omega_y_gv, omega_z_gv

state_names = ['x_uav', 'y_uav', 'z_uav', 
               'qx_uav', 'qy_uav', 'qz_uav', 'qw_uav', 
               'dx_uav', 'dy_uav', 'dz_uav', 
               'om_x_uav', 'om_y_uav', 'om_z_uav',
               'x_gv', 'y_gv', 'z_gv', 
               'qx_gv', 'qy_gv', 'qz_gv', 'qw_gv', 
               'dx_gv', 'dy_gv', 'dz_gv', 
               'om_x_gv', 'om_y_gv', 'om_z_gv']

# Placeholder initial condition vector (overwritten in batch run)
x0 = np.zeros(len(state_names))

# Turn off default sim outputs
output_ctl_1 = { # Default simulation output control for class
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'default', # Name for output file, codes unset
    'file': False, # Save output file
    'plots': False # Save state plot
}

# Initialize simulation
sim_uav_fall_nomag = simulator.Simulator(tspan, x0, timestep=timestep, 
    state_names=state_names, output_ctl=output_ctl_1)

# Overwrite default state modification and system dynamics methods
sim_uav_fall_nomag.state_mod_fun = MethodType(state_mod_uav_fall_nomag, 
                                              sim_uav_fall_nomag)
sim_uav_fall_nomag.sysdyn = MethodType(sysdyn_uav_fall_nomag, 
                                       sim_uav_fall_nomag)

# Overwrite default termination check method
sim_uav_fall_nomag.term_check = MethodType(term_check_landing, 
                                           sim_uav_fall_nomag)

# Add additional required parameters to the sim class
sim_uav_fall_nomag.contact_size = ground_vehicle_1.contact_size
sim_uav_fall_nomag.landing_gear_height = contact_use.points_B[0, 2]

# -------------- Sim Config: PX4 Controlled UAV Velocity Ramp Up ------------- #

# Initial simulation time parameters (Possibly overwritten in batch run)
tspan = (0, 1.0)
timestep = 0.0005

# State Vector: 
#   x_uav, y_uav, z_uav, qx_uav, qy_uav, qz_uav, qw_uav, 
#       dx_dt_uav, dy_dt_uav, dz_dt_uav, omega_x_uav, omega_y_uav, omega_z_uav

state_names = ['x_uav', 'y_uav', 'z_uav', 
               'qx_uav', 'qy_uav', 'qz_uav', 'qw_uav', 
               'dx_uav', 'dy_uav', 'dz_uav', 
               'om_x_uav', 'om_y_uav', 'om_z_uav']

# Placeholder initial condition vector (overwritten in batch run)
x0 = np.zeros(len(state_names))

# Turn off default sim outputs
output_ctl_1 = { # Default simulation output control for class
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'default', # Name for output file, codes unset
    'file': False, # Save output file
    'plots': False # Save state plot
}

# Initialize simulation
sim_uav_vel_ramp = simulator.Simulator(tspan, x0, timestep=timestep, 
                            state_names=state_names, output_ctl=output_ctl_1)

# Overwrite default state modification and system dynamics methods
sim_uav_vel_ramp.state_mod_fun = MethodType(state_mod_uav_vel_ramp, 
                                            sim_uav_vel_ramp)
sim_uav_vel_ramp.sysdyn = MethodType(sysdyn_uav_ctl_flight, sim_uav_vel_ramp)

# Set controller to use
sim_uav_vel_ramp.uav_ctl_use = control_use # Modified PX4

# Add control reset method
sim_uav_vel_ramp.reset_ctl = MethodType(reset_ctl, sim_uav_vel_ramp)

sim_uav_vel_ramp.u_uav = control_use.w_vec_last # Initialize u variable

# Initialize setpoint variables
sim_uav_vel_ramp.setpoint_cols = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 
                                  'dx', 'dy', 'dz', 'om_x', 'om_y', 'om_z']
sim_uav_vel_ramp.setpoint_df = \
    pd.DataFrame(columns=sim_uav_vel_ramp.setpoint_cols, 
                 index=range(sim_uav_vel_ramp.n))
sim_uav_vel_ramp.setpoint_df.iloc[0, 0] = tspan[0]
pos_set = np.zeros(3)
sim_uav_vel_ramp.setpoint_df.iloc[0, 1:4] = pos_set
sim_uav_vel_ramp.setpoint_df.iloc[0, 4:8] = np.array([0, 0, 0, 1])
sim_uav_vel_ramp.setpoint_df.iloc[0, 8:] = np.zeros(6)

sim_uav_vel_ramp.T_vec = control_use.T_vec_last # Initialize Tvec variable
sim_uav_vel_ramp.ctl_cols = ['t', 'u1', 'u2', 'u3', 'u4', 
                             'Th', 'tau_x', 'tau_y', 'tau_z']
sim_uav_vel_ramp.ctl_df = pd.DataFrame(columns=sim_uav_vel_ramp.ctl_cols, 
                                       index=range(sim_uav_vel_ramp.n))
sim_uav_vel_ramp.ctl_df.iloc[0, 0] = tspan[0]
sim_uav_vel_ramp.ctl_df.iloc[0, 1:] = np.zeros(8)

# ----------------- Sim Config: PX4 Controlled UAV Drag Study ---------------- #

# Initial simulation time parameters (Possibly overwritten in batch run)
tspan = (0, 1.0)
timestep = 0.0005

# State Vector: 
#   x_uav, y_uav, z_uav, qx_uav, qy_uav, qz_uav, qw_uav, 
#       dx_dt_uav, dy_dt_uav, dz_dt_uav, omega_x_uav, omega_y_uav, omega_z_uav

state_names = ['x_uav', 'y_uav', 'z_uav', 
               'qx_uav', 'qy_uav', 'qz_uav', 'qw_uav', 
               'dx_uav', 'dy_uav', 'dz_uav', 
               'om_x_uav', 'om_y_uav', 'om_z_uav']

# Placeholder initial condition vector (overwritten in batch run)
x0 = np.zeros(len(state_names))

# Turn off default sim outputs
output_ctl_1 = { # Default simulation output control for class
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'default', # Name for output file, codes unset
    'file': False, # Save output file
    'plots': False # Save state plot
}

# Initialize simulation
sim_uav_drag = simulator.Simulator(tspan, x0, timestep=timestep, 
                                   state_names=state_names, 
                                   output_ctl=output_ctl_1)

# Overwrite default state modification and system dynamics methods
sim_uav_drag.state_mod_fun = MethodType(state_mod_uav_drag, sim_uav_drag)
sim_uav_drag.sysdyn = MethodType(sysdyn_uav_ctl_flight, sim_uav_drag)

# Set controller to use
sim_uav_drag.uav_ctl_use = control_use # Modified PX4

# Add control reset method
sim_uav_drag.reset_ctl = MethodType(reset_ctl, sim_uav_drag)

sim_uav_drag.u_uav = control_use.w_vec_last # Initialize u variable

# Initialize setpoint variables
sim_uav_drag.setpoint_cols = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 
                               'dx', 'dy', 'dz', 'om_x', 'om_y', 'om_z']
sim_uav_drag.setpoint_df = pd.DataFrame(columns=sim_uav_drag.setpoint_cols,
                                        index=range(sim_uav_drag.n))
sim_uav_drag.setpoint_df.iloc[0, 0] = tspan[0]
pos_set = np.zeros(3)
sim_uav_drag.setpoint_df.iloc[0, 1:4] = pos_set
sim_uav_drag.setpoint_df.iloc[0, 4:8] = np.array([0, 0, 0, 1])
sim_uav_drag.setpoint_df.iloc[0, 8:] = np.zeros(6)

sim_uav_drag.T_vec = control_use.T_vec_last # Initialize Tvec variable
sim_uav_drag.ctl_cols = ['t', 'u1', 'u2', 'u3', 'u4', 
                         'Th', 'tau_x', 'tau_y', 'tau_z']
sim_uav_drag.ctl_df = pd.DataFrame(columns=sim_uav_drag.ctl_cols, 
                                   index=range(sim_uav_drag.n))
sim_uav_drag.ctl_df.iloc[0, 0] = tspan[0]
sim_uav_drag.ctl_df.iloc[0, 1:] = np.zeros(8)


















