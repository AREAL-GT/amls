
# --------------------------------- Imports ---------------------------------- #

# Standard imports
import numpy as np
from numpy import linalg as LA

from scipy.spatial.transform import Rotation as Rot

import warnings

from math import sin, cos, atan2, isinf

# amls package imports
from amls.dynamics.quadcopter_params import QuadParameters
from amls.dynamics.environmental_params import EnvParameters
from amls.dynamics.quadcopter_params import iris_params, x500_exp_params
from amls.dynamics.environmental_params import env_1

# ----------------------- Default Class Init Arguments ----------------------- #

# Default arguements for the class
default_debug_ctl = { # Default debug settings in dict
    "compute": False # compute_dynamics method
}

# ------------------------ Quadcopter Class Definition ----------------------- #

class QuadcopterDynamics:
    '''
    Class used to organize the dynamic equations of motion of a quadrotor uav
    based on the models from [1,2].

    Notes:
    - All coordinate frames x-forward, y-right, z-down
    '''

    # Insert class-wide variables here

    def __init__(self, quad_params: QuadParameters, env_params: EnvParameters, 
                 blade_flap: bool = False, 
                 debug_ctl: dict = default_debug_ctl) -> None:
        '''
        Constructor method

        Required Inputs:
            quad_params: QuadParameters object from quadcopter_parameters.py
            env_params: EnvParameters object from environmental_parameters.py

        '''

        self.quad_params = quad_params # Quadcopter parameters
        self.env_params = env_params # Environmental parameters
        self.debug_ctl = debug_ctl # Debug control dict
        self.blade_flap = blade_flap # Blade flap model-component control

    def compute_dynamics(self, t: float, state_vec: np.ndarray, 
                         u: np.ndarray) -> np.ndarray:
        '''
        Calculate the system dynamics for the quadrotor uav

        Required Inputs:
            state_vec: 1d numpy vector, state vector: x, y, z, qx, qy, qz, qw, 
                dx_dt, dy_dt, dz_dt, omega_x, omega_y, omega_z
            u: 1d numpy vector cooresponding to the 4 rotor rotation velocities,
                defined as all positive values, directions taken from 
                quad_params.w_dir

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
        dx_dt = state_vec[7]
        dy_dt = state_vec[8]
        dz_dt = state_vec[9]
        om_x = state_vec[10]
        om_y = state_vec[11]
        om_z = state_vec[12]

        # Form needed vectors and matrices from states
        q = np.array([qx, qy, qz, qw])
        R = Rot.from_quat(q) # B->A
        # pos_A = np.array([x, y, z])
        vel_W = np.array([dx_dt, dy_dt, dz_dt])
        vel_Q = R.apply(vel_W, inverse=True)
        omega_Q = np.array([om_x, om_y, om_z])

        # Motor saturation and floor check
        if all(u == 0): # If uncontrolled

            u = self.quad_params.w_min*np.ones(4) # Set to minimum values

        else: # If controlled loop through all motors
            
            for i in range(4):

                w_i = u[i] # Particular rotor velocity

                if w_i > self.quad_params.w_max: # Check motor saturation

                    w_i = self.quad_params.w_max

                    warn_msg = ("[Quad-compute_dynamics] t = %0.3f: " \
                                "Motor %i maximum saturation") % (t, i)
                    # print(warn_msg)

                elif abs(w_i) < self.quad_params.w_min: # Check minimum rotation

                    w_i = np.sign(w_i)*self.quad_params.w_min

                    warn_msg = ("[Quad-compute_dynamics] t = %0.3f: " \
                                "Motor %i minimum saturation") % (t, i)
                    # print(warn_msg)

                u[i] = w_i # Assign back to control vector

        # Compute thrust and drag for each rotor
        T_arr = np.zeros((4, 3))
        Q_arr = np.zeros((4, 3))
        tau_arr = np.zeros((4, 3))
        T_dir_vec = np.array([0, 0, -1])

        if self.blade_flap: # If blade flapping enabled
            a1s = np.zeros(4)
            b1s = np.zeros(4)

        for i in range(0, 4): # Loop through each rotor

            # Optional blade flapping computations:
            if self.blade_flap:
                
                # Constant parameters
                mu_bound = 1e6 # Assign numerical saturation for mu
                lam_bound = 1e6 # Assign numerical saturation for lambda
                theta_0 = self.quad_params.theta_0 # Blade root angle
                theta_tw = self.quad_params.theta_tw # Twist angle per radius
                gamma = self.quad_params.gamma

                # Compute initial values and rotations
                v_tip = abs(u[i])*self.quad_params.r # u_i*radius = v_tip
                V_rotor_B = np.cross(omega_Q, self.quad_params.D[i]) + vel_Q 
                mu = LA.norm(V_rotor_B[0:2]) / v_tip # Planar mu
                lam = V_rotor_B[2] / v_tip
                j = atan2(V_rotor_B[1], V_rotor_B[0])
                J = np.array([[cos(j), -sin(j)], [sin(j),  cos(j)]]) # R_R_Q

                # Saturation and infinite bounds on mu and lam
                if isinf(mu):

                    mu = mu_bound

                elif mu > mu_bound:

                    mu = mu_bound

                if isinf(lam):

                    lam = lam_bound

                elif abs(lam) > lam_bound:

                    lam = np.sign(lam)*lam_bound

                b0 = (mu*((8*theta_0/3) + (2*theta_tw) + (-2*lam)))/ \
                    (1 - 0.5*mu**2)
                beta = np.array([b0, 0])
                beta_B = J.T@beta # Rotate into the {B} frame

                # Add addtional uav dynamics from [2]
                a1s[i] = beta_B[0] - 16/gamma/u[i]*omega_Q[1]
                b1s[i] = beta_B[1] - 16/gamma/u[i]*omega_Q[0]

                T_dir_vec = np.array([-cos(b1s[i])*sin(a1s[i]), 
                                      sin(b1s[i]), 
                                      -cos(a1s[i])*cos(b1s[i])])

            # Compute rotor thrust adjusted by flapping angles
            CT = self.quad_params.CT
            T_arr[i] = CT*u[i]**2*T_dir_vec

            # Compute rotor drag torque
            CQ = self.quad_params.CQ
            Q_arr[i, 2] = -CQ*self.quad_params.w_dir[i]*u[i]**2

            # Compute torques from rotor thrust
            tau_arr[i] = np.cross(self.quad_params.D[i], T_arr[i])

        # Compute linear drag term
        u_mean = np.mean(np.abs(u))
        A1c = self.quad_params.A1c
        A1s = self.quad_params.A1s
        dx = self.quad_params.dx
        dy = self.quad_params.dy
        # flap_mat = np.array([[A1c, -A1s, 0], [A1s, A1c, 0], [0, 0, 0]])
        flap_mat = np.array([[A1c, 0, 0], [0, A1c, 0], [0, 0, 0]])
        A_flap = (1/(u_mean*self.quad_params.r))*flap_mat
        D = A_flap + np.diag([dx, dy, 0])

        # Compile system dynamics output
        dstates_dt = np.zeros(13, dtype=float) # Init return deriv state vec

        # Unforced system dynamics
        dstates_dt[0] = dx_dt # x velocity
        dstates_dt[1] = dy_dt # y velocity
        dstates_dt[2] = dz_dt # z velocity
        dstates_dt[3] = 0.5*(om_x*qw + om_z*qy - om_y*qz) # dqx_dt [6]
        dstates_dt[4] = 0.5*(om_y*qw - om_z*qx + om_x*qz) # dqy_dt
        dstates_dt[5] = 0.5*(om_z*qw + om_y*qx - om_x*qy) # dqz_dt
        dstates_dt[6] = 0.5*(-om_x*qx - om_y*qy - om_z*qz) # dqw_dt
        dstates_dt[7] = 0 # x acceleration
        dstates_dt[8] = 0 # y acceleration
        dstates_dt[9] = 0 # z acceleration
        # dstates_dt[10] # alpha x angular acceleration
        # dstates_dt[11] # alpha y angular acceleration
        # dstates_dt[12] # alpha z angular acceleration

        # Forced system dynamics
        sys_force = np.zeros(13, dtype=float) # Initialze forcing input vector
        T_sum_Q = np.sum(T_arr, axis=0)
        F_Q = T_sum_Q - np.abs(T_sum_Q[2])*D@vel_Q
        F_W = R.apply(F_Q)
        eq_accel_W = (1/self.quad_params.m)*F_W
        eq_accel_W[2] += 9.81 # Add gravity in 
        sys_force[7:10] = eq_accel_W # Assign to forcing vector

        I = self.quad_params.I
        I_inv = self.quad_params.I_inv
        cross_term = np.cross(omega_Q, I@omega_Q) 
        alpha_Q = -I_inv@cross_term + I_inv@(np.sum(tau_arr, axis=0) + 
            np.sum(Q_arr, axis=0))
        sys_force[10:] = alpha_Q

        dstates_dt += sys_force

        return dstates_dt

# -------------------------- Configured Quadcopters -------------------------- #

# 3DR Iris Quadcopter (parameters from PX4 gazebo package)
quad_iris = QuadcopterDynamics(iris_params, env_1, blade_flap=False)

# Modified Holybro x500v2 Quadcopter
quad_x500 = QuadcopterDynamics(x500_exp_params, env_1, blade_flap=False)





        









