
'''
This module contains a subclass that inherets from and modifies the stock 
PX4 control code in px4_stock.py
'''

# --------------------------------- Imports ---------------------------------- #

# Standard imports
import numpy as np
from numpy import linalg as LA

from scipy.spatial.transform import Rotation as Rot

# from math import sin, cos, acos, radians, degrees

from amls.control.control_params import PX4Parameters
from amls.control.control_params import px4_ctl_param_iris, px4_ctl_param_x500

# from amls.control.utility_functions import quat_mult, q_adjoint
from amls.control.px4_stock import PX4Control

from amls.dynamics.quadcopter_params import QuadParameters
from amls.dynamics.environmental_params import EnvParameters
from amls.dynamics.quadcopter_params import iris_params, x500_exp_params
from amls.dynamics.environmental_params import env_1

# ----------------------- Default Class Init Arguments ----------------------- #

default_debug_ctl = { # Default debug settings in dict
    "main": False, # control_main method
    "accel_att": False # accel_yaw_to_quat method
}

# ----------------------------- Class Definition ----------------------------- #

class PX4ControlMod(PX4Control):
    '''
    This subclass inherits the stock PX4 control algorithms class, and shadows
    any methods to be modified
    '''

    def __init__(self, ctl_params: PX4Parameters, quad_params: QuadParameters, 
                 env_params: EnvParameters, 
                 debug_ctl: dict = default_debug_ctl) -> None:
        
        super().__init__(ctl_params, quad_params, env_params, debug_ctl)

        self.pos_err_int = np.zeros(3) # Position error integral initialized 0
        self.pos_err_prev = np.zeros(3) # Position error previous initialized 0
        
        # self.ctl_params.pos_err_int_lim = 10 # Position integral anti windup
        # self.ctl_params.pos_x_I = 0.1 # Position integral gains
        # self.ctl_params.pos_y_I = 0.1
        # self.ctl_params.pos_z_I = 0.01
        # self.ctl_params.pos_x_D = 0.5 # Position derivative gains
        # self.ctl_params.pos_y_D = 0.5
        # self.ctl_params.pos_z_D = 0.0

    def control_main(self, t: float, state_vec: np.ndarray, 
                     setpoint_dict: dict) -> np.ndarray:
        '''
        Primary method call for the PX4 controllers, called by the simulation
        system dynamics loop to generate the control inputs for the quad

        Required Inputs:
            t: float, timestamp for this control call
            state_vec: 1d numpy vector of the quad states
                state vector: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, 
                              omega_x, omega_y, omega_z
            setpoint_dict:
                'pos': np vec [x, y, z]
                'vel': tuple (x, y, z) or None in elements, 
                'accel': np vec [x, y, z] or None
                'quat': np vec [qx, qy, qz, qw] or None
                'omega': np vec [x, y, z] or None
                'Tvec': np vec [T, taux, tauy, tauz]
                'rotor': np vec [w1, w2, ..., wn] or None
        '''

        # UAV Position control loop [50Hz]
        pos_dt = t - self.pos_prev_t
        if pos_dt >= (1/self.ctl_params.pos_freq):

            self.pos_prev_t = t # Save time of this call

            pos_set = setpoint_dict['pos']
            self.pos_cmd = pos_set # Update position setpoint
            pos_meas = state_vec[0:3] # Isolate position states
            # vel_meas = state_vec[7:10] # Isolate velocity states

            # The the velocity feedforward terms
            vel_gv_G = state_vec[20:23].astype(float)
            R_G_W = Rot.from_quat(state_vec[16:20])
            vel_gv_W = R_G_W.apply(vel_gv_G)

            self.vel_cmd = self.position_controller(pos_meas, self.pos_cmd, 
                                                    pos_dt, vel_ff=vel_gv_W)

            # Overwrite velocity command elements if direct control
            if not setpoint_dict['vel'][0] is None: # x velocity
                self.vel_cmd[0] = setpoint_dict['vel'][0]

            if not setpoint_dict['vel'][1] is None: # y velocity
                self.vel_cmd[1] = setpoint_dict['vel'][1]

            if not setpoint_dict['vel'][2] is None: # z velocity
                self.vel_cmd[2] = setpoint_dict['vel'][2]

        # Linear velocity control [50Hz]
        vel_dt = t - self.vel_prev_t
        if vel_dt >= (1/self.ctl_params.vel_freq):

            # # Overwrite velocity command elements if direct control
            # if not setpoint_dict['vel'][0] is None: # x velocity
            #     self.vel_cmd[0] = setpoint_dict['vel'][0]

            # if not setpoint_dict['vel'][1] is None: # y velocity
            #     self.vel_cmd[1] = setpoint_dict['vel'][1]

            # if not setpoint_dict['vel'][2] is None: # z velocity
            #     self.vel_cmd[2] = setpoint_dict['vel'][2]

            # if t > 2.0:
            #     print('vel control')
            #     print(self.vel_cmd)
            #     input('hold')

            self.vel_prev_t = t # Save time of call
            vel_meas = state_vec[7:10] # Isolate velocity states
            self.accel_cmd = self.velocity_controller(vel_meas, self.vel_cmd, 
                                                      vel_dt)
            
            # Overwrite acceleration command if direct control
            if not setpoint_dict['accel'] is None: 
                self.accel_cmd = setpoint_dict['accel']
            
        # Attitude control [250Hz]
        attitude_dt = t - self.att_prev_t
        if attitude_dt >= (1/self.ctl_params.att_freq):

            self.att_prev_t = t # Save time of call
            psi_set = setpoint_dict['psi']

            # Calc commanded orientation quat and thrust for accel command
            self.q_cmd, self.Th_cmd = self.accel_yaw_to_quat(t, state_vec, 
                self.accel_cmd, psi_set)
            
            # Overwrite orientation command if direct control
            if not setpoint_dict['quat'] is None: 
                self.q_cmd = setpoint_dict['quat']

            # Overwrite thrust command if direct control
            if not setpoint_dict['Th'] is None: 
                self.Th_cmd = setpoint_dict['Th']

            # Run attitude controller to generate angular velocity setpoints
            self.omega_cmd = self.attitude_controller(state_vec, self.q_cmd)

            # Overwrite angular velocity command if direct control
            if not setpoint_dict['omega'] is None: 
                self.omega_cmd = setpoint_dict['omega']

        # Angular rate control loop timing (1000Hz)
        ang_rate_dt = t - self.ang_rate_prev_t
        if ang_rate_dt >= (1/self.ctl_params.ang_rate_freq):

            self.ang_rate_prev_t = t # Save time of call

            omega_meas = state_vec[10:13] # Measured vector of ang velocities
            
            # Run angular rate controller to generate torque setpoints
            tau_vec = self.angular_rate_controller(omega_meas, self.omega_cmd, 
                ang_rate_dt)

            # Append total thrust setpoint to desired torques
            T_vec = np.append(self.Th_cmd, tau_vec)

            # Overwrite Tvec command if direct control
            if not setpoint_dict['Tvec'] is None: 
                T_vec = setpoint_dict['Tvec']

            # Run control allocation to determine motor velocities
            w_vec = self.control_allocation(t, T_vec)

            # Overwrite rotor velocity command if direct control
            if not setpoint_dict['rotor'] is None: 
                w_vec = setpoint_dict['rotor']

            # if self.debug_dict["main"]:
            #     msg = ("t = %0.3f" % t + " omega_meas = " + 
            #         np.array2string(omega_meas) + " omega_set = " + 
            #         np.array2string(self.omega_set) + " T_vec = " + 
            #         np.array2string(T_vec) + " w_vec = " +
            #         np.array2string(w_vec))
            #     print(msg) # Insert angular rate debug here
           
        else: # If not time, pass last w_vec setpoint through

            # Assign Tvec and w_vec from previous time
            T_vec = self.T_vec_last
            w_vec = self.w_vec_last

        self.T_vec_last = T_vec # Save to previous command vector
        self.w_vec_last = w_vec # Save to previous command vector

        # Save setpoints to vector
        num_states = 13
        setpoint_vec = np.zeros(num_states) # Initialize state setpoint vector
        setpoint_vec[0:3] = self.pos_cmd # Position
        setpoint_vec[3:6] = self.q_cmd[1:] # qx, qy, qz
        setpoint_vec[6] = self.q_cmd[0] # qw
        setpoint_vec[7:10] = self.vel_cmd # Linear velocity
        setpoint_vec[10:] = self.omega_cmd # Angular velocity

        return w_vec, T_vec, setpoint_vec # Return w_vec, Tvec, and setpoints

    def position_controller(self, pos_meas, pos_set, dt, vel_ff = np.zeros(3)):
        '''
        UAV position PI controller implementation.
        '''

        params = self.ctl_params # Copy for conciseness

        pos_err = pos_set - pos_meas # Calculate error in position states

        mag_err = LA.norm(pos_err) # Position error saturation
        if mag_err > self.pos_sat:
            pos_err = self.pos_sat*(pos_err/mag_err)

        self.pos_err_int += pos_err.astype(float) # Add integral term

        # Integral anti-windup check
        for i in range(3):
            self.pos_err_int[i] = min(self.pos_err_int[i], 
                                      params.pos_err_int_lim)
            
        # Compute derivative of position error
        dpos_err_dt = (pos_err - self.pos_err_prev)/dt
        self.pos_err_prev = pos_err

        # Apply control gains
        P_term = np.array([pos_err[0]*params.pos_x_P, 
                           pos_err[1]*params.pos_y_P, 
                           pos_err[2]*params.pos_z_P])
        
        I_term = np.array([self.pos_err_int[0]*params.pos_x_P*params.pos_x_I, 
                           self.pos_err_int[1]*params.pos_y_P*params.pos_y_I, 
                           self.pos_err_int[2]*params.pos_z_P*params.pos_z_I])
        
        D_term = np.array([dpos_err_dt[0]*params.pos_x_P*params.pos_x_D, 
                           dpos_err_dt[1]*params.pos_y_P*params.pos_y_D, 
                           dpos_err_dt[2]*params.pos_z_P*params.pos_z_D])

        vel_fb = P_term + I_term + D_term

        vel_cmd = np.add(vel_fb, vel_ff)

        return vel_cmd
    
    def velocity_controller(self, vel_meas, vel_set, dt):
        '''
        Linear velocity PID controller implementation.
        '''

        params = self.ctl_params # Copy for conciseness
    
        vel_err = vel_set - vel_meas # Calculate error in velocity states

        self.vel_err_int += vel_err.astype(float) # Add integral term

        # Calculate state derivative with finite difference
        dvel_dt = (vel_meas - self.vel_prev_state)/dt

        # Integral anti-windup check
        for i in range(3):
            
            self.vel_err_int[i] = min(self.vel_err_int[i], 
                                      params.vel_err_int_lim)

        # Apply control gains
        P_term = np.array([vel_err[0]*params.vel_x_P, 
                           vel_err[1]*params.vel_y_P, 
                           vel_err[2]*params.vel_z_P])
        
        I_term = np.array([self.vel_err_int[0]*params.vel_x_P*params.vel_x_I, 
                           self.vel_err_int[1]*params.vel_y_P*params.vel_y_I, 
                           self.vel_err_int[2]*params.vel_z_P*params.vel_z_I])
        
        D_term = np.array([dvel_dt[0]*params.vel_x_P*params.vel_x_D, 
                           dvel_dt[1]*params.vel_y_P*params.vel_y_D, 
                           dvel_dt[2]*params.vel_z_P*params.vel_z_D])

        accel_cmd = P_term + I_term - D_term # Assign to output vector

        # accel_cmd[2] += -9.8 # Feed forward gravity in z-direction

        self.vel_prev_state = vel_meas # Save to previous state

        return accel_cmd

    def reset_ctl(self) -> None:
        '''
        Method to resent controller parameters for repeated use in batch
        simulation runs
        '''

        # Reset setpoint variables
        self.pos_cmd = np.zeros(3) # UAV position setpoint in {A}
        self.vel_cmd = np.zeros(3) # Linear velocity setpoint in {A}
        self.accel_cmd = np.array([0, 0, -9.8]) # Linear accel setpoint in {A}
        self.q_cmd = np.array([1, 0, 0, 0]) # Attitude quaternion
        self.omega_cmd = np.zeros(3) # Angular velocity setpoint
        self.Th_cmd = 9.8*self.quad_params.m # Thrust

        # Reset variables for storing previous state information
        self.vel_prev_state = np.zeros(3) # Velocity
        self.ang_rate_prev_state = np.zeros(3) # Angular velocity
        self.w_vec_last = np.array([0.10473, -0.10473, 0.10473, -0.10473]) 
        self.T_vec_last = np.zeros(4) # Thrust and torque commands

        # Rest integral term variables
        self.pos_err_int = np.zeros(3) # Position error integral initialized 0
        self.vel_err_int = np.zeros(3) # Velocity error integral initialized 0
        self.ang_rate_err_int = np.zeros(3) # Velocity error integral init 0

        # Reset timing variables
        self.pos_prev_t = 0 # Time for last control loop run
        self.vel_prev_t = 0 # Time for last control loop run
        self.att_prev_t = 0 # Time for last control loop run
        self.ang_rate_prev_t = 0 # Time for last control loop run

# -------------------------- Configured Controllers -------------------------- #

# Controller from iris parameters
px4_mod_iris = PX4ControlMod(px4_ctl_param_iris, iris_params, env_1)
px4_mod_iris.pos_cmd = np.zeros(3)
px4_mod_iris.q_cmd = np.array([1, 0, 0, 0]) # qx, qy, qz, qw for controller

# Controller from x500 parameters
px4_mod_x500 = PX4ControlMod(px4_ctl_param_x500, x500_exp_params, env_1)
px4_mod_x500.pos_cmd = np.zeros(3)
px4_mod_x500.q_cmd = np.array([1, 0, 0, 0]) # qx, qy, qz, qw for controller







