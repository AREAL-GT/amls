
'''
This module contains the class implementation of the geometric controllers
developed by T Lee and his lab
'''

# --------------------------------- Imports ---------------------------------- #

# Standard imports
import numpy as np
from numpy import linalg as LA
import scipy.constants as consts
from scipy.spatial.transform import Rotation as Rot
from math import sin, cos, acos

# Workspace imports
from amls.control.control_params import GeometricParameters
from amls.control.control_params import geom_ctl_param_def, geom_ctl_param_x500
from amls.dynamics.quadcopter_params import QuadParameters
from amls.dynamics.quadcopter_params import x500_exp_params
from amls.dynamics.environmental_params import EnvParameters


# ----------------------- Default Class Init Arguments ----------------------- #


# ----------------------------- Class Definition ----------------------------- #

class GeometricControl:
    '''
    Class to implement the geometric controllers developed by T Lee and his lab
    '''

    def __init__(self, ctl_params: GeometricParameters, 
                 quad_params: QuadParameters) -> None:
        '''
        Constructor to initialize needed terms for the controller class
        '''

        # Assign initialization arguements to class
        self.ctl_params = ctl_params # Controller parameters
        self.quad_params = quad_params # Quadcopter parameters
        # self.env_params = env_params # Environmental parameters

        # Setpoint variables
        self.pos_cmd = np.zeros(3) # UAV position setpoint in {A}
        self.vel_cmd = np.zeros(3) # Linear velocity setpoint in {A}
        self.q_cmd = np.array([0, 0, 0, 1])

        # Variables for storing previous state information
        self.w_vec_last = self.quad_params.w_min*np.array([1, -1, 1, -1]) 
        self.T_vec_last = np.zeros(4) # Thrust and torque commands

        # Integral term variables
        self.ei = np.zeros(3) # Position error integral
        self.eI1 = 0 # X-axis attitude error integral
        self.eI2 = 0 # Y-axis attitude error integral
        self.eIy = 0 # Yaw attitude error integral

        # Timing variables
        self.geom_prev_t = 0 # Time for last control loop run

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
            [-CQ, CQ, -CQ, CQ]])
        self.Gamma_inv = LA.inv(self.Gamma)

    def control_main(self, t: float, state_vec: np.ndarray, 
                     setpoint_dict: dict) -> np.ndarray:
        '''
        The primary method call for the geometric controller

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
                'psi': float or None
                'heading_vec': np 3x3 vec Q-frame x-axis: b1d, b1d_dot, b1d_ddot 
                    or None
                'omega': np vec [x, y, z] or None
                'Tvec': np vec [T, taux, tauy, tauz]
                'rotor': np vec [w1, w2, ..., wn] or None
        '''

        # Geometric controller call loop [200Hz]
        geom_dt = t - self.geom_prev_t
        if geom_dt >= 1/self.ctl_params.geom_freq:

            # Pull setpoints from dictionary
            self.pos_cmd = setpoint_dict['pos']
            self.vel_cmd = setpoint_dict['vel']
            b1d = setpoint_dict['heading_vec'][0]
            b1d_dot = setpoint_dict['heading_vec'][1]
            b1d_ddot = setpoint_dict['heading_vec'][2]

            self.geom_prev_t = t # Save time of this call

            # Compute the Q to W rotation
            q = state_vec[3:7]
            R_Q_W = Rot.from_quat(q)

            # Setup desired unit vectors
            # e1 = np.array([1, 0, 0])
            # e2 = np.array([0, 1, 0])
            e3 = np.array([0, 0, 1])
            # b1 = R_Q_W.apply(e1)
            # b2 = R_Q_W.apply(e2)
            b3 = R_Q_W.apply(e3)

            # Compute the thrust controller inputs
            ex = state_vec[:3].astype(float) - self.pos_cmd
            ev = state_vec[7:10].astype(float) - self.vel_cmd
            om = state_vec[10:13]

            # Call thrust controller
            A, A_dot, A_ddot = self.thrust_control_yaw_decoupled(ex, ev, R_Q_W, 
                                                                 om, geom_dt)

            # Compute the attitude controller inputs
            b3c, b3c_dot, b3c_ddot = self.deriv_unit_vec(-A, -A_dot, -A_ddot)

            A2 = -self.hat(b1d)@b3c
            A2_dot = -self.hat(b1d_dot)@b3c - self.hat(b1d)@b3c_dot
            A2_ddot = -self.hat(b1d_ddot)@b3c - 2*self.hat(b1d_dot)@b3c_dot - \
                self.hat(b1d)@b3c_ddot

            b2c, b2c_dot, b2c_ddot = self.deriv_unit_vec(A2, A2_dot, A2_ddot)
            
            b2c_vec = np.array([b2c, b2c_dot, b2c_ddot])
            b3c_vec = np.array([b3c, b3c_dot, b3c_ddot])

            b1c, b1c_dot, b1c_ddot = self.b1c_gen(b2c_vec, b3c_vec)

            Rc_Q_W = Rot.from_matrix([b1c, b2c, b3c])
            self.q_cmd = Rc_Q_W.inv().as_quat()
            Rc_Q_W_dot = Rot.from_matrix([b1c_dot, b2c_dot, b3c_dot])
            Rc_Q_W_ddot = Rot.from_matrix([b1c_ddot, b2c_ddot, b3c_ddot])

            R_int = Rc_Q_W.inv()*Rc_Q_W_dot
            wc = self.vee(R_int.as_matrix())

            R_int2 = Rc_Q_W.inv()*Rc_Q_W_ddot
            wc_dot = self.vee(R_int2.as_matrix() - self.hat(wc)@self.hat(wc))
            wc3 = np.dot(b3, Rc_Q_W.apply(wc))
            wc3_dot = np.dot(b3, Rc_Q_W.apply(wc_dot)) + \
                np.dot(R_Q_W.apply(self.hat(om))@e3, Rc_Q_W.apply(wc))

            # Call M1/M2 attitude controller
            M1, M2 = self.m1m2_control_yaw_decoupled(b3c, b3c_dot, b3c_ddot, 
                                                     R_Q_W, om, geom_dt)

            # Call M3 yaw controller
            M3 = self.m3_control_yaw_decoupled(wc3, wc3_dot, b1c, R_Q_W, om, 
                                               geom_dt)

            # Compute the thrust force f
            f = -np.dot(A, b3)
            # th_cmd_W = -A[2]
            # z_W = np.array([0, 0, 1])
            # zQ_Q = np.array([0, 0, 1])
            # zQ_W = R_Q_W .apply(zQ_Q)
            # alpha_curr = acos(np.minimum(z_W@zQ_W, 1.0))
            # f = th_cmd_W/cos(alpha_curr)   

            # Call control allocation
            T_vec = np.array([f, M1, M2, M3])
            w_vec = self.control_allocation(t, T_vec)

        else: # If not time, pass last w_vec setpoint through

            # Assign Tvec and w_vec from previous time
            T_vec = self.T_vec_last
            w_vec = self.w_vec_last

        # Save to previous command vectors
        self.T_vec_last = T_vec 
        self.w_vec_last = w_vec

        # Save setpoints to vector
        num_states = 13
        setpoint_vec = np.zeros(num_states) # Initialize state setpoint vector
        setpoint_vec[0:3] = self.pos_cmd # Position
        setpoint_vec[3:7] = self.q_cmd
        setpoint_vec[7:10] = self.vel_cmd # Linear velocity
        # setpoint_vec[10:] = np.zeros(3) # Angular velocity unused

        return w_vec, T_vec, setpoint_vec # Return w_vec, Tvec, and setpoints

    def thrust_control_yaw_decoupled(self, ex: np.ndarray, ev: np.ndarray, 
                R_Q_W: Rot, om: np.ndarray, dt: float, xd_ddot = np.zeros(3), 
                xd_dddot = np.zeros(3), xd_ddddot = np.zeros(3)) -> np.ndarray:
        '''
        This method takes in the position-related error vectors from the UAV,
        computes the desired theoretical force vector A, and uses that to 
        return the desired z-axis b3d and rotor thrust f
        '''

        # Divide out the needed controller gains and system constants
        kx = self.ctl_params.kx
        kv = self.ctl_params.kv
        ki = self.ctl_params.ki
        sigma = self.ctl_params.sigma
        c1 = self.ctl_params.c1
        m = self.quad_params.m
        g = consts.g

        ex = ex.astype(float)
        ev = ev.astype(float)

        # Divide out needed UAV states
        # x = state_vec_uav[:3]
        # q = state_vec_uav[3:7]
        # dx_dt = state_vec_uav[7:10]
        # om = state_vec_uav[10:]
        # R_Q_W = Rot.from_quat(q) # Quad to inertial rotation

        # Setup desired unit vectors
        # e1 = np.array([1, 0, 0])
        # e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])

        # b1 = R_Q_W.apply(e1)
        # b2 = R_Q_W.apply(e2)
        b3 = R_Q_W.apply(e3)
        b3_dot = R_Q_W.as_matrix()*self.hat(om)@e3

        # Compute alternative UAV body-fixed integral components
        R_W_Q = R_Q_W.inv()
        ex_Q = R_W_Q.apply(ex)
        ev_Q = R_W_Q.apply(ev)
        ei_dot_Q = (ev_Q + np.multiply(c1, ex_Q))
        ei_Q = (dt*ei_dot_Q)

        self.ei += ei_Q
        np.clip(self.ei, -sigma, sigma, out=self.ei)

        ei_Q_sat = self.sat_dot(sigma, self.ei, ei_dot_Q)

        ei_W = R_Q_W.apply(self.ei)
        ei_dot_W = R_Q_W.apply(ei_Q_sat)

        # Compute the integral error term
        # self.ei += dt*(ev + np.multiply(c1, ex)).astype(float)
        # ei_dot = ev + np.multiply(c1, ex)
        # np.clip(self.ei, -sigma, sigma, out=self.ei)

        # Form the theoretical force vector A
        # A = -np.multiply(kx, ex) - np.multiply(kv, ev) - \
        #     np.multiply(ki, self.ei) - m*g*e3 + m*xd_ddot
        A = -np.multiply(kx, ex) - np.multiply(kv, ev) - \
            np.multiply(ki, ei_W) - m*g*e3 + m*xd_ddot
        

        # ex_z = np.array([0, 0, ex[2]])
        # ev_z = np.array([0, 0, ev[2]])
        # ei_z = np.array([0, 0, self.ei[2]])
        # A_z = -np.multiply(kx, ex_z) - np.multiply(kv, ev_z) - \
        #     np.multiply(ki, ei_z) - m*g*e3 + m*xd_ddot
        
        # Form A_dot
        f = -np.dot(A, b3)
        ea = g*e3 - (f/m)*b3  - xd_ddot # + x_delta / m - x_delta unknown
        # A_dot = -kx*ev - kv*ea + m*xd_dddot - \
        #     ki*self.sat_dot(sigma, self.ei, ei_dot)
        A_dot = -kx*ev - kv*ea + m*xd_dddot - ki*ei_dot_W

        # Form A_ddot
        ei_ddot = ea + np.multiply(c1, ev)
        f_dot = -np.dot(A_dot, b3) - np.dot(A, b3_dot)
        eb = -(f_dot/m)*b3 - (f/m)*b3_dot - xd_dddot
        A_ddot = -kx*ea - kv*eb + m*xd_ddddot - \
            ki*self.sat_dot(sigma, self.ei, ei_ddot)
        
        return A, A_dot, A_ddot

    def m1m2_control_yaw_decoupled(self, b3d: np.ndarray, b3d_dot: np.ndarray, 
        b3d_ddot: np.ndarray, R_Q_W: Rot, om: np.ndarray, 
        dt: float) -> np.ndarray:
        '''
        This method takes current UAV attitude state information, along with
        the desired z-axis b3d and computes the required pitch and roll
        moments M1 and M2
        '''

        # Divide out the needed controller gains and system constants
        kb = self.ctl_params.kb
        kw = self.ctl_params.kw
        kI = self.ctl_params.kI
        c2 = self.ctl_params.c2
        sigma2 = self.ctl_params.sigma2
        J1 = self.quad_params.I[0, 0]
        J3 = self.quad_params.I[2, 2]
        
        # Setup desired unit vectors
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        b1 = R_Q_W.apply(e1)
        b2 = R_Q_W.apply(e2)
        b3 = R_Q_W.apply(e3)

        # Compute the needed angular velocity terms
        w12 = om[0]*b1 + om[1]*b2
        w12d = self.hat(b3d)@b3d_dot
        w12d_dot = self.hat(b3d)@b3d_ddot
        b3_dot = self.hat(w12)@b3

        # Compute the attitude error terms
        eb = np.cross(b3d, b3)
        ew = w12 + (self.hat(b3)@self.hat(b3))@w12d

        # Compute the integral attitude error terms
        self.eI1 += dt*np.dot(ew + c2*eb, b1)
        self.eI2 += dt*np.dot(ew + c2*eb, b2)
        self.eI1 = max(min(self.eI1, sigma2), -sigma2)
        self.eI2 = max(min(self.eI2, sigma2), -sigma2)
        # np.clip(self.eI1, -sigma2, sigma2, out=self.eI1)
        # np.clip(self.eI2, -sigma2, sigma2, out=self.eI2)

        # Compute the overall control moment tau
        tau = -np.multiply(kb, eb) - np.multiply(kw, ew) - kI*self.eI1*b1 - \
            - kI*self.eI2*b2 - J1*np.dot(b3, w12d)*b3_dot - \
            J1*(self.hat(b3)@self.hat(b3))@w12d_dot
        
        # Convert tau into M1 and M2
        tau1 = np.dot(tau, b1)
        tau2 = np.dot(tau, b2)

        M1 = tau1 + J3*om[1]*om[2]
        M2 = tau2 - J3*om[0]*om[2]

        return M1, M2

    def m3_control_yaw_decoupled(self, wc3: float, wc3_dot: float, 
        b1c: np.ndarray, R_Q_W: Rot, om: np.ndarray, dt: float) -> np.ndarray:
        '''
        This method takes the current UAV attitude along with a desired heading
        vector, and computes the required yaw moment M3
        '''

        # Divide out the needed controller gains and system constants
        ky = self.ctl_params.ky
        kwy = self.ctl_params.kwy
        kIy = self.ctl_params.kIy
        c3 = self.ctl_params.c3
        J3 = self.quad_params.I[2, 2]

        # Setup desired unit vectors
        # e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        # e3 = np.array([0, 0, 1])
        # b1 = R_Q_W.apply(e1)
        b2 = R_Q_W.apply(e2)
        # b3 = R_Q_W.apply(e3)

        # Compute the yaw error terms
        ey = -np.dot(b2, b1c)
        ewy = om[2] - wc3

        # Compute the yaw integral error term
        self.eIy += dt*(ewy + c3*ey)

        # Compute the moment M3
        M3 = -ky*ey - kwy*ewy - kIy*self.eIy + J3*wc3_dot

        return M3

    def control_allocation(self, t: float, T_vec: np.ndarray) -> np.ndarray:
        '''
        Method to be called to compute control allocation for the uav. Outputs
        a vector of rotor velocities required for a desired vector of torques
        and overall thrust 
        
        The desired vector is given: T_vec = [T_tot M1 M2 M3]

        The output vector is given: w_vec = [w1 w2 w3 w4]
        '''

        w_vec_sq = self.Gamma_inv@T_vec # Control allocation

        w_vec_sq[w_vec_sq < 0] = 0.0 # Apply minimum bounds on squared

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

    def reset_ctl(self) -> np.ndarray:
        '''
        Method to resent controller parameters for repeated use in batch
        simulation runs
        '''

        # Reset variables for storing previous state information
        self.w_vec_last = self.quad_params.w_min*np.array([1, -1, 1, -1]) 
        self.T_vec_last = np.zeros(4) # Thrust and torque commands

        # Rest setpoint variables
        self.pos_cmd = np.zeros(3) # UAV position setpoint in {A}
        self.vel_cmd = np.zeros(3) # Linear velocity setpoint in {A}
        self.q_cmd = np.array([0, 0, 0, 1])

        # Rest integral term variables
        self.ei = np.zeros(3) # Position error integral
        self.eI1 = 0 # X-axis attitude error integral
        self.eI2 = 0 # Y-axis attitude error integral
        self.eIy = 0 # Yaw attitude error integral

        # Reset timing variables
        self.geom_prev_t = 0

    def hat(self, vec: np.ndarray) -> np.ndarray:
        '''
        This method implements the hat operator from the T Lee papers. The hat
        operator simply turns a vector of length 3 into a 3x3 skew-symetric
        matrix
        '''

        skew_sym_mat = np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]])
        
        return skew_sym_mat
    
    def vee(self, mat: np.ndarray) -> np.ndarray:
        '''
        This method implements the vee operator from the Lee papers. It takes
        a skew-symetric matrix and turns it into a vector
        '''

        vec = np.array([-mat[1, 2], mat[0, 2], -mat[0, 1]])

        return vec
    
    def deriv_unit_vec(self, A: np.ndarray, A_dot: np.ndarray, 
                       A_ddot: np.ndarray) -> np.ndarray:
        '''
        This method takes in a vector and its derivatives and gives the unit-
        vector derivatives as outputs
        '''

        nA = LA.norm(A)

        b = A/nA
        b_dot = A_dot/nA - (A*np.dot(A, A_dot))/(nA**3)
        b_ddot = A_ddot/nA - \
                 A_dot*((2 * np.dot(A, A_dot)) / (nA**3)) - \
                 A*((np.dot(A_dot, A_dot) + np.dot(A, A_dot)) / (nA**3)) + \
                 A*((3*np.dot(A, A_dot)**2)/(nA**5))

        return b.astype(float), b_dot.astype(float), b_ddot.astype(float)

    def b1c_gen(self, b2c_vec: np.ndarray, b3c_vec: np.ndarray) -> np.ndarray:
        '''
        This method computes the commanded vectors and their derivatives for
        b1c from b2c and b3c vectors and derivatives
        '''

        # Split out the input vectors
        b2c = b2c_vec[0]
        b2c_dot = b2c_vec[1]
        b2c_ddot = b2c_vec[2]

        b3c = b3c_vec[0]
        b3c_dot = b3c_vec[1]
        b3c_ddot = b3c_vec[2]

        # Compute the b1c terms
        b1c = self.hat(b2c)@b3c
        b1c_dot = self.hat(b2c_dot)@b3c + self.hat(b2c)@b3c_dot
        b1c_ddot = self.hat(b2c_ddot)@b3c + 2*self.hat(b2c_dot)@b3c_dot + \
            self.hat(b2c)@b3c_ddot
        
        return b1c, b1c_dot, b1c_ddot
    
    def sat_dot(self, sigma: np.ndarray, y: np.ndarray, 
                y_dot: np.ndarray) -> np.ndarray:
        '''
        Implementation of the satdot function from the Lee papers with a vector
        sigma
        '''

        z = np.zeros(len(y))

        for i in range(len(y)):
            if (y[i] < sigma[i]) and (y[i] > -sigma[i]):
                z[i] = y_dot[i]

        return z

# -------------------------- Configured Controllers -------------------------- #

# Controller configured with default paper tunings
geom_default = GeometricControl(geom_ctl_param_def, x500_exp_params)

# Controller configured with x500v2 tunings
geom_x500 = GeometricControl(geom_ctl_param_x500, x500_exp_params)
