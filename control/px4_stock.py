
# --------------------------------- Imports ---------------------------------- #

# Standard imports
import numpy as np
from numpy import linalg as LA

from scipy.spatial.transform import Rotation as Rot

from math import sin, cos, acos, radians, degrees

from control.control_params import PX4Parameters
# from control.control_params import px4_ctl_param_1

from control.utility_functions import quat_mult, q_adjoint

from dynamics.quadcopter_params import QuadParameters
from dynamics.environmental_params import EnvParameters
# from dynamics.quadcopter_params import iris_params, x500_exp_params
# from dynamics.environmental_params import env_1

# ----------------------- Default Class Init Arguments ----------------------- #

default_debug_ctl = { # Default debug settings in dict
    "main": False, # control_main method
    "accel_att": False # accel_yaw_to_quat method
}

# ----------------------------- Class Definition ----------------------------- #

class PX4Control:
    '''
    Class used to emulate the PX4 control framework for uax simulation
    '''

    def __init__(self, ctl_params: PX4Parameters, quad_params: QuadParameters, 
                 env_params: EnvParameters, 
                 debug_ctl: dict = default_debug_ctl) -> None:

        # Assign initialization arguements to class
        self.ctl_params = ctl_params # Controller parameters
        self.quad_params = quad_params # Quadcopter parameters
        self.env_params = env_params # Environmental parameters
        self.debug_ctl = debug_ctl # Debug control dict

        # Setpoint variables
        self.pos_cmd = np.zeros(3) # UAV position setpoint in {A}
        self.vel_cmd = np.zeros(3) # Linear velocity setpoint in {A}
        self.accel_cmd = np.array([0, 0, -9.8]) # Linear accel setpoint in {A}
        self.q_cmd = np.array([0, 0, 0, 1])
        self.omega_cmd = np.zeros(3) # Angular velocity setpoint
        self.Th_cmd = 9.8*self.quad_params.m # Thrust

        # Variables for storing previous state information
        self.vel_prev_state = np.zeros(3) # Velocity
        self.ang_rate_prev_state = np.zeros(3) # Angular velocity
        self.w_vec_last = np.array([0.10473, -0.10473, 0.10473, -0.10473]) 
        self.T_vec_last = np.zeros(4) # Thrust and torque commands

        # Integral term variables
        self.vel_err_int = np.zeros(3) # Velocity error integral initialized 0
        self.ang_rate_err_int = np.zeros(3) # Velocity error integral init 0

        # Timing variables
        self.pos_prev_t = 0 # Time for last control loop run
        self.vel_prev_t = 0 # Time for last control loop run
        self.att_prev_t = 0 # Time for last control loop run
        self.ang_rate_prev_t = 0 # Time for last control loop run

        # Control allocation parameters
        theta = np.radians(45)
        CT = quad_params.CT
        d = quad_params.d
        CQ = quad_params.CQ
        self.Gamma = np.array([
            [CT, CT, CT, CT], 
            [-sin(theta)*d*CT, -sin(theta)*d*CT, sin(theta)*d*CT, 
                sin(theta)*d*CT], 
            [cos(theta)*d*CT, -cos(theta)*d*CT, -cos(theta)*d*CT, 
                cos(theta)*d*CT], 
            [-CQ, CQ, -CQ, CQ]]
        )
        self.Gamma_inv = LA.inv(self.Gamma)

        # Saturation variables
        self.tilt_sat = radians(35) # Maximum tilt from vertical PX4 default 35
        self.pos_sat = 1 # Maximum magnitude position error

    def control_main(self, t: float, state_vec: np.ndarray, 
                     setpoint_dict: dict, psi_set: float) -> np.ndarray:
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
            self.vel_cmd = self.position_controller(pos_meas, self.pos_cmd)

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

    def position_controller(self, pos_meas, pos_set):
        '''
        UAV position P controller implementation.
        '''

        pos_err = pos_set - pos_meas # Calculate error in velocity states

        mag_err = LA.norm(pos_err) # Position error saturation
        if mag_err > self.pos_sat:
            pos_err = self.pos_sat*(pos_err/mag_err)

        # Apply control gains
        P_term = np.array([pos_err[0]*self.ctl_params.pos_x_P, 
                           pos_err[1]*self.ctl_params.pos_y_P, 
                           pos_err[2]*self.ctl_params.pos_z_P])

        vel_cmd = P_term

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
    
    def accel_yaw_to_quat(self, t, state_vec, a_cmd, psi_cmd):
        '''
        Method adapting algorithms from [7]. Equation numbers (XX) correspond 
        to [7]. Turns commanded acceleration vector into commanded quaternion
        '''

        # Convert commanded acceleration to unit vector and magnitude
        # - Negative of acceleration to match z-vector of uav
        # mag_accel_cmd = LA.norm(a_cmd) # Magnitude of desired accel (44)
        # if mag_accel_cmd > 0:
        #     e_cmdz_A = -a_cmd / mag_accel_cmd # Unit vector accel direction (43)
        # else:
        #     e_cmdz_A = np.zeros(3)

        # Use hover acceleration opposite of gravity to compute orientation
        a_cmd_ornt = np.array([a_cmd[0], a_cmd[1], -9.81])

        # Negative of acceleration to match z-vector of uav
        mag_a_cmd_ornt = LA.norm(a_cmd_ornt)
        if mag_a_cmd_ornt > 0:
            e_cmdz_A = -a_cmd_ornt / mag_a_cmd_ornt # Unit vector 
        else:
            e_cmdz_A = np.zeros(3) # Fill as zeros placeholder

        # Maximum tilt angle saturation
        e_cmdz_A = self.tilt_limit(e_cmdz_A, self.tilt_sat)

        # Isolate quaternion from state vec
        q_state = state_vec[3:7] # x, y, z, w
        R = Rot.from_quat(q_state) # Make scipy rotation object Q->W

        q_use = np.insert(state_vec[3:6], 0, state_vec[6]) # qw first in [7]

        # Find body-fixed z-vector in inertial frame
        e_z_A = R.apply(np.array([0, 0, 1]))

        # ----- Reduced attitude control solution
        # Calculate alpha between current and commanded z-axis
        alpha = acos(np.minimum(e_z_A@e_cmdz_A, 1.0))

        # Calculate q_err_red
        cross_term = np.cross(e_z_A, e_cmdz_A)
        norm_cross = LA.norm(cross_term)
        if norm_cross > 0:
            k_vec = cross_term / norm_cross
        else:
            k_vec = np.zeros(3)

        q_err_red = np.array([cos(alpha/2), 
                              sin(alpha/2)*k_vec[0],
                              sin(alpha/2)*k_vec[1], 
                              sin(alpha/2)*k_vec[2]])

        # Calculate q_cmd_red
        q_cmd_red = quat_mult(q_use, q_err_red)

        # Just using reduced attitude solution for now (6/16/23)
        # Full attitude solution moved to archive
    
        # Calculate mixed-solution q_cmd
        q_cmd = q_cmd_red
        
        self.q_cmd = q_cmd # Set to controller class

        # Use the vertical acceleration command to set the thrust
        accel_cmd_tot = -9.81 + a_cmd[2]
        th_cmd_W = -self.quad_params.m*accel_cmd_tot # Flip sign for throttle
        th_cmd_W = max(th_cmd_W, 0.0) # Bound by 0 because no downward motors
        # th_cmd_W = self.quad_params.m*abs(-9.81 + a_cmd[2])
        z_W = np.array([0, 0, 1])
        zQ_Q = np.array([0, 0, 1])
        zQ_W = R.apply(zQ_Q)

        alpha_curr = acos(np.minimum(z_W@zQ_W, 1.0))
        Th_cmd = th_cmd_W/cos(alpha_curr)

        if self.debug_ctl["accel_att"]:
            print('t = %0.3f | q_cmd_red | q_cmd_full | q_cmd' % t)
            print(q_cmd_red)
            print('N/A')
            print(q_cmd)

        return q_cmd, Th_cmd

    def attitude_controller(self, state_vec, q_cmd):
        '''
        Implementation of the quaternion-based attitude p-controller used by 
        PX4. Algorithm detailed in [7].
        '''

        params = self.ctl_params

        q_use = np.insert(state_vec[3:6], 0, state_vec[6]) # qw first in [7]

        # q = np.concatenate([state_vec[6], state_vec[3:6]]) # qw first in [7]

        # Calculate invese of q
        q_conj = np.array([q_use[0], -q_use[1], -q_use[2], -q_use[3]])
        q_norm = LA.norm(q_use)
        q_inv = q_conj/q_norm

        # Calculate q_e
        q_e = quat_mult(q_inv, q_cmd)

        # Calculate omega_set
        q_e_imag = q_e[1:]
        omega_set = params.attitude_P*(2/params.attitude_time_const)* \
            np.sign(q_e[0])*q_e_imag

        return omega_set
    
    def angular_rate_controller(self, omega_meas, omega_des, dt):
        '''
        PID control implementation for angular rate controller. Takes in
        desired and measured angular rates, and outputs vector of torques
        '''

        params = self.ctl_params

        # Calculate error in angular velocity
        omega_err = omega_des - omega_meas
        self.ang_rate_err_int += omega_err.astype(float) # Add to integral sum

        # Calculate state derivative with finite difference
        d_dt_omega = (omega_meas - self.ang_rate_prev_state)/dt
        
        # Integral anti-windup check
        for i in range(3):
            self.ang_rate_err_int[i] = min(self.ang_rate_err_int[i], 
                params.ang_rate_err_int_lim)
            
        # Apply control gains
        P_term = np.array([omega_err[0]*params.ang_rate_x_P, 
                           omega_err[1]*params.ang_rate_y_P, 
                           omega_err[2]*params.ang_rate_z_P])
        I_term = np.array(
            [self.ang_rate_err_int[0]*params.ang_rate_x_P*params.ang_rate_x_I, 
             self.ang_rate_err_int[1]*params.ang_rate_y_P*params.ang_rate_y_I, 
             self.ang_rate_err_int[2]*params.ang_rate_z_P*params.ang_rate_z_I])
        D_term = np.array(
            [d_dt_omega[0]*params.ang_rate_x_P*(params.ang_rate_x_D), 
             d_dt_omega[1]*params.ang_rate_y_P*(params.ang_rate_y_D), 
             d_dt_omega[2]*params.ang_rate_z_P*(params.ang_rate_z_D)])
        
        # Assign to output vector
        tau_vec = P_term + I_term - D_term

        self.ang_rate_prev_state = omega_meas # Save to previous state

        return tau_vec
    
    def control_allocation(self, t: float, T_vec: np.ndarray):
        '''
        Method to be called to compute control allocation for the uav. Outputs
        a vector of rotor velocities required for a desired vector of torques
        and overall thrust 
        
        The desired vector is given: T_vec = [T_tot tau1 tau2 tau3]

        The output vector is given: w_vec = [w1 w2 w3 w4]
        '''

        w_vec_sq = self.Gamma_inv@T_vec # Control allocation

        w_vec_sq[w_vec_sq < 0] = 0.0

        w_vec = np.sqrt(w_vec_sq) # Account for magnitude squared

        w_vec[1] *= -1 # Correct spin of propellers 2 and 4
        w_vec[3] *= -1

        # Add min and max saturation for motors
        for i in range(4):

            # Maximum saturation
            if abs(w_vec[i]) > self.quad_params.w_max:
                warn_msg = ("[Ctl-control_allocation] t = %0.3f: " \
                                "Motor %i maximum saturation") % (t, i)
                # warnings.warn(warn_msg)
                w_vec[i] = self.quad_params.w_dir[i]*self.quad_params.w_max
                w_vec[i] *= 0.999

            # Minimum saturation (small overage to avoid warnings)
            if abs(w_vec[i]) < self.quad_params.w_min:
                warn_msg = ("[Ctl-control_allocation] t = %0.3f: " \
                                "Motor %i minimum saturation") % (t, i)
                # warnings.warn(warn_msg)
                w_vec[i] = self.quad_params.w_dir[i]*self.quad_params.w_min
                w_vec[i] *= 1.001

        return w_vec

    def tilt_limit(self, e_cmd: np.ndarray, alpha_max: float):
        '''
        Method to enforce a maximum tilt saturation on the commanded attitude

        Algorithm largely from [7]
        '''

        debug_print = False

        ez_I = np.array([0, 0, 1])

        alpha_tilt = acos(ez_I@e_cmd) # Compute commanded tilt (56)
        alpha_tilt_deg = degrees(alpha_tilt)

        if abs(alpha_tilt) > alpha_max: # If commaned tilt exceeds maximum

            # Compute rotation axis (57)
            k = np.cross(ez_I, e_cmd) / LA.norm(np.cross(ez_I, e_cmd))

            # Compute tilt quat (58)
            alpha_del = alpha_max - alpha_tilt # Changed from [7]
            q_tilt = np.insert(k*sin(alpha_del/2), 0 , cos(alpha_del/2)) 

            # Compute new thrust direction (59)
            p_command = np.insert(e_cmd, 0, 0)
            q_tilt_adj = q_adjoint(q_tilt)
            prod1 = quat_mult(p_command, q_tilt_adj)
            p_thrust = quat_mult(q_tilt, prod1)

            e_new_acmd = p_thrust[1:]
            e_new_acmd /= LA.norm(e_new_acmd)
            alpha_limit = acos(ez_I@e_new_acmd)
            alpha_limit_deg = degrees(alpha_limit)

            if debug_print: # If debug prints turned on
                
                alpha_max_deg = degrees(alpha_max)
                msg1 = ('[Tilt Limit] Commanded tilt = %0.3f deg. Maximum ' \
                        'allowable tilt = %0.3f deg.') % \
                        (alpha_tilt_deg, alpha_max_deg)
                msg2 = ('[Tilt Limit] Commanded change in tilt = %0.3f ' \
                        'deg.') % degrees(alpha_del)
                msg3 = '[Tilt Limit] Limited tilt angle = %0.3f deg' % \
                    alpha_limit_deg
                msg4 = '[Tilt Limit] Commanded acceleration vector = ' + \
                    np.array_str(e_cmd)
                msg5 = '[Tilt Limit] Adjusted acceleration vector = ' + \
                    np.array_str(e_new_acmd)
                msg6 = ('[Tilt Limit] Magnitude of adjusted accel vector' \
                        ' = %0.3f') % LA.norm(e_new_acmd)
                
                print(msg1)
                print(msg2)
                print(msg3)
                print(msg4)
                print(msg5)
                print(msg6)

        else: # If within limits pass through

            e_new_acmd = e_cmd
            
        return e_new_acmd
    
    def reset_ctl(self) -> None:
        '''
        Method to resent controller parameters for repeated use in batch
        simulation runs
        '''

        # Reset setpoint variables
        self.pos_cmd = np.zeros(3) # UAV position setpoint in {A}
        self.vel_cmd = np.zeros(3) # Linear velocity setpoint in {A}
        self.accel_cmd = np.array([0, 0, -9.8]) # Linear accel setpoint in {A}
        self.q_cmd = np.array([1, 0, 0, 0]) # Attitude quat order diff control
        self.omega_cmd = np.zeros(3) # Angular velocity setpoint
        self.Th_cmd = 9.8*self.quad_params.m # Thrust

        # Reset variables for storing previous state information
        self.vel_prev_state = np.zeros(3) # Velocity
        self.ang_rate_prev_state = np.zeros(3) # Angular velocity
        self.w_vec_last = np.array([0.10473, -0.10473, 0.10473, -0.10473]) 
        self.T_vec_last = np.zeros(4) # Thrust and torque commands

        # Rest integral term variables
        self.vel_err_int = np.zeros(3) # Velocity error integral initialized 0
        self.ang_rate_err_int = np.zeros(3) # Velocity error integral init 0

        # Reset timing variables
        self.pos_prev_t = 0 # Time for last control loop run
        self.vel_prev_t = 0 # Time for last control loop run
        self.att_prev_t = 0 # Time for last control loop run
        self.ang_rate_prev_t = 0 # Time for last control loop run

# -------------------------- Configured Controllers -------------------------- #








