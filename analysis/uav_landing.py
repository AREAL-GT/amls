
'''
Module that runs a single trial of a quadcopter UAV autonomously landing on a
contact zone (ground vehicle). The contact zone can be defined as a plane, or 
have a specific size. The contact zone can also be defined as moving or static.
This script tests the full autonomous landing process in simulation. The 
magnet model and contact models are also included.
'''

# --------------------------------- Imports ---------------------------------- #

# Standard imports
import numpy as np
from numpy import linalg as LA

import pandas as pd

from scipy.spatial.transform import Rotation as Rot

from math import degrees, pi

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = ["Latin Modern Roman"]
from mpl_toolkits.mplot3d import axes3d
from matplotlib.patches import Polygon
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation

from types import MethodType

# Add package and workspace directories to the path
import os
import sys
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)

# Workspace package imports
from amls_impulse_contact import contact
from amls_impulse_contact import utility_functions as uf_cont

from amls_sim import simulator

from dynamics import quadcopter
from dynamics import magnet
from dynamics.quadcopter_params import iris_params
from dynamics.environmental_params import env_1

from control import px4_stock
from control.control_params import px4_ctl_param_1


# --------------------- Quadcopter Object Configuration ---------------------- #

quad_debug_ctl = { # Quad model debug print settings
    "compute": False # compute_dynamics method
}

quad_iris = quadcopter.QuadcopterDynamics(iris_params, env_1, quad_debug_ctl)


# ------------------- Quadcopter Controller Configuration -------------------- #

control_debug_ctl = { # Controller debug print settings
    "main": False, # control_main method
    "accel_att": False # accel_yaw_to_quat method
}

px4_stock_1 = px4_stock.PX4Control(px4_ctl_param_1, iris_params, env_1, 
                                   control_debug_ctl)

# ----------------------- Contact Object Configuration ----------------------- #

contact_debug_ctl = { # Debug settings dict
    "main": False, # contact_main method
    "contact": False, # contact_calc methods
    "count_every": False, # contact_count method every call
    "count_pos": False, # contact_count method positive counts
    "phase": False, # compression/extension/restitution methods
    "deriv": False, # restitution_deriv/compression_deriv methods
    "ancillary": False, # Ancillary helper methods
    "file": False # Save debug prints to a text file
}

leg_dim = iris_params.d # 9.5 inches to m
uav_contact_points_Q = np.array([[leg_dim/2, leg_dim/2, leg_dim/3], # x,y,z
                                 [-leg_dim/2, leg_dim/2, leg_dim/3], 
                                 [-leg_dim/2, -leg_dim/2, leg_dim/3],
                                 [leg_dim/2, -leg_dim/2, leg_dim/3]]) 
m = 3*iris_params.m
I = 1.5*iris_params.I
e1 = 0.1 # Coefficient of restitution
mu1 = 1.0 # Coefficient of friction
contact_size = np.array([0.762, 0.762])
uav_contact = contact.ImpulseContact(uav_contact_points_Q, m, I, e=e1, 
                                     cont_size=contact_size, mu=mu1,
                                     debug_ctl=contact_debug_ctl)

# ------------------- Magnetic Landing Gear Configuration -------------------- #

mag_strength = 43.86 # Force in N - K&J D66-N52 magnet
strength_vec = mag_strength*np.ones(uav_contact_points_Q.shape[0])
mag_land = magnet.MagnetLanding(uav_contact_points_Q, strength_vec, iris_params)
mag_land.cont_size = contact_size

# ------------- Overall System Dynamics - Controlled Quadcopter -------------- #

def sysdyn_px4_quad(self, t, x):
    '''
    Function call used by the numerical integrator, including quadcopter system
    dynamics and px4 control scheme
    '''

    dx_dt = quad_iris.compute_dynamics(t, x, self.u) # Quadcopter dynamics

    dx_dt_mag = mag_land.compute_forces(t, x, self.cont_mag) # Magnet forcing
    dx_dt += dx_dt_mag # Add accels from magnet forces and moments

    return dx_dt 

# -------------------- Direct State Modification Function -------------------- #

pos_set_sim = np.array([0.0, 0.0, -0.5*leg_dim/3])
cont_pos_W1 = np.array([0, 0, 0])

def state_mod(self, t, x, i):
    '''
    Method for direct change of system states and parameters:
    - Contact model on object velocities
    - Control inputs due to system states
    '''

    debug = False
    no_thrust = False

    # Enforce unit magnitude quaternion orientation
    qx = x[3]
    qy = x[4]
    qz = x[5]
    qw = x[6]
    q = np.array([qx, qy, qz, qw])
    q_unit = q/LA.norm(q)
    x[3:7] = q_unit

    # Isolate out needed states and variables for clarity
    pos_W = x[:3] # UAV states
    R_Q_W = Rot.from_quat(x[3:7])
    lin_vel_W = x[7:10]
    ang_vel_Q = x[10:] 

    cont_pos_W = cont_pos_W1 # Ground vehicle states
    cont_quat_W = Rot.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
    cont_lin_vel_W = np.array([0, 0, 0])
    cont_ang_vel_G = np.array([0, 0, 0])
    cont_states = np.concatenate([cont_pos_W, cont_quat_W, cont_lin_vel_W, 
                                  cont_ang_vel_G])

    # Apply contact model to system dynamics
    del_lin_vel_W, del_ang_vel_Q, del_z_pos_W, cont_check_vec, \
        df_cont_tracking = uav_contact.contact_main(pos_W, R_Q_W, lin_vel_W, 
            ang_vel_Q, cont_states=cont_states, t=t)
    
    self.cont_mag = mag_land.mag_check(uav_contact.points_G, t)
    
    if debug: # Contact debug prints
        print('t = %f, del_vel = %f' % (t, del_lin_vel_W[2]))
    
    x_out = x + np.concatenate((del_z_pos_W, np.zeros(4), del_lin_vel_W, 
                                del_ang_vel_Q))

    # Compute controller action given the states this timestep
    cont_check = any(cont_check_vec)

    if cont_check: # Flip contact state to true if contact on any point
        if not self.in_contact:
            print('[state_mod] Touchdown at t = %0.4f' % t)
        self.in_contact = True

    if self.in_contact: # If any contact ever with landing zone
        self.u = iris_params.w_min*np.array([1, -1, 1, -1])
        self.T_vec = px4_stock_1.Gamma@self.u
        setpoint_vec = self.setpoint_df.iloc[i, 1:]

    else: # In free flight
        pos_set = pos_set_sim
        psi_set = 0
        self.u, self.T_vec, setpoint_vec = \
            px4_stock_1.control_main(t, x_out, pos_set, psi_set)

    if no_thrust:
        self.u = iris_params.w_min*np.array([1, -1, 1, -1])
        self.T_vec = px4_stock_1.Gamma@self.u
        setpoint_vec = self.setpoint_df.iloc[i, 1:]

    # Save controller actions and setpoints to dataframe output
    self.setpoint_df.iloc[i+1, 0] = t
    self.setpoint_df.iloc[i+1, 1:] = setpoint_vec
    self.ctl_df.iloc[i+1, 0] = t
    self.ctl_df.iloc[i+1, 1:5] = self.u
    self.ctl_df.iloc[i+1, 5:] = self.T_vec

    return x_out

# --------------------- Configure and Perform Simulation --------------------- #

# Setup simulation control parameters
output_ctl1 = { # Output file control
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'landing_test.csv', # Name for output file
    'file': True, # Save output file
    'plots': True # Save output plots
}

trial_name = 'uav_land_a05_x02_y02_d3'
ani_name = trial_name + '.mp4'
save_fig = False

tspan = (0, 1.95)
timestep = 0.0005

# State Vector: x, y, z, qx, qy, qz, qw, dx_dt, dy_dt, dz_dt, 
#   omega_x, omega_y, omega_z
state_names = ['x', 'y', 'z', 
               'qx', 'qy', 'qz', 'qw', 
               'dx', 'dy', 'dz', 
               'om_x', 'om_y', 'om_z']
pos0 = np.array([0.2, 0.2, -0.5])
R_init = Rot.from_euler('xyz', [0, 0, 0], degrees=True)
q0 = R_init.as_quat()
vel0 = np.array([0.0, 0.0, 0])
omega0 = np.array([0.0, 0, 0])
x0 = np.concatenate((pos0, q0, vel0, omega0))

# Initialize simulation
sim_land = simulator.Simulator(tspan, x0, timestep=timestep, 
                               state_names=state_names, output_ctl=output_ctl1)

# Overwrite default state modification and system dynamics methods
sim_land.state_mod_fun = MethodType(state_mod, sim_land)
sim_land.sysdyn = MethodType(sysdyn_px4_quad, sim_land)

# Initialize needed custom sim object variables
sim_land.u = px4_stock_1.w_vec_last # Initialize u variable
setpoint_cols = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'dx', 'dy', 'dz', 
                 'om_x', 'om_y', 'om_z']
sim_land.setpoint_df = pd.DataFrame(columns=setpoint_cols, 
                                    index=range(sim_land.n))
sim_land.setpoint_df.iloc[0, 0] = tspan[0]
sim_land.setpoint_df.iloc[0, 1:4] = pos_set_sim
sim_land.setpoint_df.iloc[0, 4:8] = np.array([0, 0, 0, 1])
sim_land.setpoint_df.iloc[0, 8:] = np.zeros(6)

sim_land.T_vec = px4_stock_1.T_vec_last # Initialize Tvec variable
ctl_cols = ['t', 'u1', 'u2', 'u3', 'u4', 'Th', 'tau_x', 'tau_y', 'tau_z']
sim_land.ctl_df = pd.DataFrame(columns=ctl_cols, index=range(sim_land.n))
sim_land.ctl_df.iloc[0, 0] = tspan[0]
sim_land.ctl_df.iloc[0, 1:] = np.zeros(8)

# sim_land.uav_params = iris_params # Pass iris params to state mod method
sim_land.in_contact = False
num_points = uav_contact_points_Q.shape[0]
sim_land.cont_mag = np.full(num_points, False, dtype=bool)

# Initialize conditions on controller object
px4_stock_1.pos_cmd = pos_set_sim
px4_stock_1.q_cmd = np.array([1, 0, 0, 0]) # qw, qx, qy, qz for controller

sim_land.compute() # Compute the simulation

landing_results_df = sim_land.state_traj
setpoint_df = sim_land.setpoint_df
ctl_df = sim_land.ctl_df

# ------------------------------ Post Processing ----------------------------- #

# Add euler angles to the trajectory dataframe
eul_conv = lambda row: pd.Series(Rot.from_quat([row.qx, row.qy, 
                                row.qz, row.qw]).as_euler('xyz', degrees=True))
euler_df = landing_results_df.apply(eul_conv, axis=1)

euler_df.columns = ['phi', 'theta', 'psi'] # Rename the cols in the Euler df
landing_results_df = pd.concat([landing_results_df, euler_df], axis=1)

# Add euler angles to the setpoint dataframe
eul_conv = lambda row: pd.Series(Rot.from_quat([row.qx, row.qy, 
                                row.qz, row.qw]).as_euler('xyz', degrees=True))
euler_df = setpoint_df.apply(eul_conv, axis=1)

euler_df.columns = ['phi', 'theta', 'psi'] # Rename the cols in the Euler df
setpoint_df = pd.concat([setpoint_df, euler_df], axis=1)

# Create dataframe with contact point positions and velocities
name_list_points = uf_cont.point_df_name(uav_contact_points_Q.shape[0])
point_df = pd.DataFrame(columns=name_list_points, 
                        index=landing_results_df.index)
point_df.t = landing_results_df.t

# Loop through results df and generate contact data
for i in range(len(landing_results_df.t)): # For each rod position

    # Position and orientation
    uav_pos_W = landing_results_df.iloc[i, 1:4].to_numpy().astype(float)
    uav_q = landing_results_df.iloc[i, 4:8].to_numpy().astype(float)
    R_Q_W = Rot.from_quat(uav_q)

    # Linear and angular velocity
    uav_vel_W = landing_results_df.iloc[i, 8:11].to_numpy().astype(float)
    uav_angvel_Q = landing_results_df.iloc[i, 11:14].to_numpy().astype(float)

    for ii in range(uav_contact_points_Q.shape[0]): # For each contact point

        # Compute contact point position
        point_ii_Q = uav_contact_points_Q[ii].astype(float)
        point_ii_pos_W = uav_pos_W + R_Q_W.apply(point_ii_Q)
        point_df.iloc[i, (6*ii + 1):(6*ii + 4)] = point_ii_pos_W

        # Compute contact point velocity
        cross_term = np.cross(uav_angvel_Q, point_ii_Q)
        vel_ii_W = uav_vel_W + R_Q_W.apply(cross_term)

        if ii < uav_contact_points_Q.shape[0]:
            point_df.iloc[i, (6*ii + 4):(6*ii + 7)] = vel_ii_W
        else:
            point_df.iloc[i, (6*ii + 4):] = vel_ii_W

# --------------------------------- Plotting --------------------------------- #

# Overall plot settings
fig_scale = 1.0
fig_dpi = 96 # PowerPoint default
font_suptitle = 28
font_subtitle = 22
font_axlabel = 20
font_axtick = 18
font_txt = 14
ax_lw = 2
plt_lw = 3
setpoint_lw = 1
fig_x_size = 19.995
fig_y_size = 9.36

# Plot position states and setpoints
fig_pos, (ax_x_pos, ax_y_pos, ax_z_pos) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_pos.suptitle('UAV Position States', fontsize=font_suptitle, 
                 fontweight='bold')

# x position
ax_x_pos.plot(landing_results_df.t, landing_results_df.x, linewidth=plt_lw) 
ax_x_pos.plot(setpoint_df.t, setpoint_df.x, linewidth=setpoint_lw, color='red')
ax_x_pos.set_title('X Position', fontsize=font_subtitle, fontweight='bold')
ax_x_pos.set_ylabel('Position (m)', fontsize=font_axlabel, fontweight='bold')
ax_x_pos.grid()
# ax_x_pos.set_ylim(-2.5, 2.5)
ax_x_pos.legend(['position', 'setpoint'], fontsize=18)

# y position
ax_y_pos.plot(landing_results_df.t, landing_results_df.y, linewidth=plt_lw) 
ax_y_pos.plot(setpoint_df.t, setpoint_df.y, linewidth=setpoint_lw, color='red')
ax_y_pos.set_title('Y Position', fontsize=font_subtitle, fontweight='bold')
ax_y_pos.set_ylabel('Position (m)', fontsize=font_axlabel, fontweight='bold')
ax_y_pos.grid()
# ax_y_pos.set_ylim(-2.5, 2.5)
ax_y_pos.legend(['position', 'setpoint'], fontsize=18)

# z position
ax_z_pos.plot(landing_results_df.t, landing_results_df.z, linewidth=plt_lw) 
ax_z_pos.plot(setpoint_df.t, setpoint_df.z, linewidth=setpoint_lw, color='red')
ax_z_pos.set_title('Z Position', fontsize=font_subtitle, fontweight='bold')
ax_z_pos.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_pos.set_ylabel('Position (m)', fontsize=font_axlabel, fontweight='bold')
ax_z_pos.grid()
ax_z_pos.invert_yaxis()
ax_z_pos.legend(['position', 'setpoint'], fontsize=18)
ax_z_pos.set_ylim(0.5, -5)


# Plot velocity states and setpoints
fig_vel, (ax_x_vel, ax_y_vel, ax_z_vel) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_vel.suptitle('UAV Velocity States', fontsize=font_suptitle, 
                 fontweight='bold')

# x velocity
ax_x_vel.plot(landing_results_df.t, landing_results_df.dx, linewidth=plt_lw) 
ax_x_vel.plot(setpoint_df.t, setpoint_df.dx, linewidth=setpoint_lw, color='red')
ax_x_vel.set_title('X Velocity', fontsize=font_subtitle, fontweight='bold')
ax_x_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, fontweight='bold')
ax_x_vel.grid()
# ax_x_vel.set_ylim(-1.0, 1.0)
ax_x_vel.legend(['velocity', 'setpoint'], fontsize=18)

# y velocity
ax_y_vel.plot(landing_results_df.t, landing_results_df.dy, linewidth=plt_lw) 
ax_y_vel.plot(setpoint_df.t, setpoint_df.dy, linewidth=setpoint_lw, color='red')
ax_y_vel.set_title('Y Velocity', fontsize=font_subtitle, fontweight='bold')
ax_y_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, fontweight='bold')
ax_y_vel.grid()
# ax_y_vel.set_ylim(-1.0, 1.0)
ax_y_vel.legend(['velocity', 'setpoint'], fontsize=18)

# z velocity
ax_z_vel.plot(landing_results_df.t, landing_results_df.dz, linewidth=plt_lw) 
ax_z_vel.plot(setpoint_df.t, setpoint_df.dz, linewidth=setpoint_lw, color='red')
ax_z_vel.set_title('Z Velocity', fontsize=font_subtitle, fontweight='bold')
ax_z_vel.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, fontweight='bold')
ax_z_vel.grid()
# ax_z_vel.set_ylim(-1.0, 1.0)
ax_z_vel.invert_yaxis()
ax_z_vel.legend(['velocity', 'setpoint'], fontsize=18)


# Plot attitude quaternion and setpoints
fig_quat, (ax_qx, ax_qy, ax_qz, ax_qw) = \
    plt.subplots(4, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_quat.suptitle('UAV Quaternion States', fontsize=font_suptitle, 
                 fontweight='bold')

# qx
ax_qx.plot(landing_results_df.t, landing_results_df.qx, linewidth=plt_lw) 
ax_qx.plot(setpoint_df.t, setpoint_df.qx, linewidth=setpoint_lw, color='red')
ax_qx.set_title('X Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qx.grid()
# ax_qx.set_ylim(-1.1, 1.1)
ax_qx.legend(['$q_x$', 'setpoint'], fontsize=18)

# qy
ax_qy.plot(landing_results_df.t, landing_results_df.qy, linewidth=plt_lw) 
ax_qy.plot(setpoint_df.t, setpoint_df.qy, linewidth=setpoint_lw, color='red')
ax_qy.set_title('Y Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qy.grid()
# ax_qy.set_ylim(-1.1, 1.1)
ax_qy.legend(['$q_y$', 'setpoint'], fontsize=18)

# qz
ax_qz.plot(landing_results_df.t, landing_results_df.qz, linewidth=plt_lw) 
ax_qz.plot(setpoint_df.t, setpoint_df.qz, linewidth=setpoint_lw, color='red')
ax_qz.set_title('Z Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qz.grid()
# ax_qz.set_ylim(-1.1, 1.1)
ax_qz.legend(['$q_z$', 'setpoint'], fontsize=18)

# qw
ax_qw.plot(landing_results_df.t, landing_results_df.qw, linewidth=plt_lw) 
ax_qw.plot(setpoint_df.t, setpoint_df.qw, linewidth=setpoint_lw, color='red')
ax_qw.set_title('W Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qw.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_qw.grid()
# ax_qw.set_ylim(-1.1, 1.1)
ax_qw.legend(['$q_w$', 'setpoint'], fontsize=18)


# Plot attitude euler angles and setpoints
fig_eul, (ax_phi, ax_theta, ax_psi) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_eul.suptitle('UAV Euler Angles', fontsize=font_suptitle, fontweight='bold')
ax_eul_list = [ax_phi, ax_theta, ax_psi]

# Phi (roll) angle
ax_phi.plot(landing_results_df.t, landing_results_df.phi, linewidth=plt_lw) 
ax_phi.plot(setpoint_df.t, setpoint_df.phi, linewidth=setpoint_lw, color='red')
ax_phi.set_title('Roll Angle ($\Phi$)', fontsize=font_subtitle, 
                 fontweight='bold')
ax_phi.set_ylabel('Angle (deg)', fontsize=font_axlabel,
                    fontweight='bold')
ax_phi.grid()
# ax_phi.set_ylim(-60, 60)
ax_phi.legend(['$\Phi$', 'setpoint'], fontsize=18)

# Theta (pitch) angle
ax_theta.plot(landing_results_df.t, landing_results_df.theta, linewidth=plt_lw) 
ax_theta.plot(setpoint_df.t, setpoint_df.theta, linewidth=setpoint_lw, 
             color='red')
ax_theta.set_title('Pitch Angle ($\Theta$)', fontsize=font_subtitle, 
                   fontweight='bold')
ax_theta.set_ylabel('Angle (deg)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_theta.grid()
# ax_theta.set_ylim(-60, 60)
ax_theta.legend(['$\Theta$', 'setpoint'], fontsize=18)

# Psi (yaw) angle
ax_psi.plot(landing_results_df.t, landing_results_df.psi, linewidth=plt_lw) 
ax_psi.plot(setpoint_df.t, setpoint_df.psi, linewidth=setpoint_lw, color='red')
ax_psi.set_title('Yaw Angle ($\Psi$)', fontsize=font_subtitle, 
                 fontweight='bold')
ax_psi.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_psi.set_ylabel('Angle (deg)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_psi.grid()
# ax_psi.set_ylim(-60, 60)
ax_psi.legend(['$\Psi$', 'setpoint'], fontsize=18)


# Plot angular rates and setpoints
fig_om, (ax_om_x, ax_om_y, ax_om_z) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_om.suptitle('UAV Angular Velocity States', fontsize=font_suptitle, 
                 fontweight='bold')
ax_om_list = [ax_om_x, ax_om_y, ax_om_z]

# x angular velocity
ax_om_x.plot(landing_results_df.t, (180/pi)*landing_results_df.om_x, 
             linewidth=plt_lw) 
ax_om_x.plot(setpoint_df.t, (180/pi)*setpoint_df.om_x, linewidth=setpoint_lw, 
             color='red')
ax_om_x.set_title('$\omega_x$', fontsize=font_subtitle, fontweight='bold')
ax_om_x.set_ylabel('Ang. Vel. (deg/s)', fontsize=font_axlabel,
                    fontweight='bold')
ax_om_x.grid()
# ax_om_x.set_ylim(-60, 60)
ax_om_x.legend(['$\omega_x$', 'setpoint'], fontsize=18)

# y angular velocity
ax_om_y.plot(landing_results_df.t, (180/pi)*landing_results_df.om_y, 
             linewidth=plt_lw) 
ax_om_y.plot(setpoint_df.t, (180/pi)*setpoint_df.om_y, linewidth=setpoint_lw, 
             color='red')
ax_om_y.set_title('$\omega_y$', fontsize=font_subtitle, fontweight='bold')
ax_om_y.set_ylabel('Ang. Vel. (deg/s)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_om_y.grid()
# ax_om_y.set_ylim(-60, 60)
ax_om_y.legend(['$\omega_y$', 'setpoint'], fontsize=18)

# z angular velocity
ax_om_z.plot(landing_results_df.t, (180/pi)*landing_results_df.om_z, 
             linewidth=plt_lw) 
ax_om_z.plot(setpoint_df.t, (180/pi)*setpoint_df.om_z, linewidth=setpoint_lw, 
             color='red')
ax_om_z.set_title('$\omega_z$', fontsize=font_subtitle, fontweight='bold')
ax_om_z.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_om_z.set_ylabel('Ang. Vel. (deg/s)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_om_z.grid()
# ax_om_z.set_ylim(-60, 60)
ax_om_z.legend(['$\omega_z$', 'setpoint'], fontsize=18)


# Plot commanded body torques and total thrust
fig_T, (ax_Th, ax_tau_x, ax_tau_y, ax_tau_z) = \
    plt.subplots(4, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_T.suptitle('Commanded Thrust and Body-Fixed Torques', 
                fontsize=font_suptitle, fontweight='bold')

# Thrust
ax_Th.plot(ctl_df.t, ctl_df.Th, linewidth=plt_lw) 
ax_Th.set_title('Total Thrust', fontsize=font_subtitle, fontweight='bold')
ax_Th.grid()
ax_Th.set_ylabel('Force (N)', fontsize=font_axlabel, fontweight='bold')
# ax_Th.set_ylim(-1.1, 1.1)
ax_Th.legend(['thrust'], fontsize=18)

# Tau x
ax_tau_x.plot(ctl_df.t, ctl_df.tau_x, linewidth=plt_lw) 
ax_tau_x.set_title('X-Axis Torque', fontsize=font_subtitle, fontweight='bold')
ax_tau_x.grid()
ax_tau_x.set_ylabel('Moment (Nm)', fontsize=font_axlabel, fontweight='bold')
# ax_tau_x.set_ylim(-1.1, 1.1)
ax_tau_x.legend([r'$\tau_x$'], fontsize=18)

# Tau y
ax_tau_y.plot(ctl_df.t, ctl_df.tau_y, linewidth=plt_lw) 
ax_tau_y.set_title('Y-Axis Torque', fontsize=font_subtitle, fontweight='bold')
ax_tau_y.grid()
ax_tau_y.set_ylabel('Moment (Nm)', fontsize=font_axlabel, fontweight='bold')
# ax_tau_y.set_ylim(-1.1, 1.1)
ax_tau_y.legend([r'$\tau_y$'], fontsize=18)

# Tau z
ax_tau_z.plot(ctl_df.t, ctl_df.tau_z, linewidth=plt_lw) 
ax_tau_z.set_title('Z-Axis Torque', fontsize=font_subtitle, fontweight='bold')
ax_tau_z.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_tau_z.grid()
ax_tau_z.set_ylabel('Moment (Nm)', fontsize=font_axlabel, fontweight='bold')
# ax_tau_z.set_ylim(-1.1, 1.1)
ax_tau_z.legend([r'$\tau_z$'], fontsize=18)

# Plot motor angular velocities
fig_u, (ax_u1, ax_u2, ax_u3, ax_u4) = \
    plt.subplots(4, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_T.suptitle('Commanded Motor Velocities', fontsize=font_suptitle, 
               fontweight='bold')

# Motor 1
ax_u1.plot(ctl_df.t, 9.54927*ctl_df.u1, linewidth=plt_lw) 
ax_u1.set_title('Motor 1', fontsize=font_subtitle, fontweight='bold')
ax_u1.grid()
ax_u1.set_ylabel('Rot. (rpm)', fontsize=font_axlabel, fontweight='bold')
# ax_u1.set_ylim(-1.1, 1.1)
ax_u1.legend(['$u_1$'], fontsize=18)

# Motor 1
ax_u2.plot(ctl_df.t, 9.54927*ctl_df.u2, linewidth=plt_lw) 
ax_u2.set_title('Motor 2', fontsize=font_subtitle, fontweight='bold')
ax_u2.grid()
ax_u2.set_ylabel('Rot. (rpm)', fontsize=font_axlabel, fontweight='bold')
# ax_u2.set_ylim(-1.1, 1.1)
ax_u2.legend(['$u_2$'], fontsize=18)

# Motor 1
ax_u3.plot(ctl_df.t, 9.54927*ctl_df.u3, linewidth=plt_lw) 
ax_u3.set_title('Motor 3', fontsize=font_subtitle, fontweight='bold')
ax_u3.grid()
ax_u3.set_ylabel('Rot. (rpm)', fontsize=font_axlabel, fontweight='bold')
# ax_u3.set_ylim(-1.1, 1.1)
ax_u3.legend(['$u_3$'], fontsize=18)

# Motor 1
ax_u4.plot(ctl_df.t, 9.54927*ctl_df.u4, linewidth=plt_lw) 
ax_u4.set_title('Motor 4', fontsize=font_subtitle, fontweight='bold')
ax_u4.grid()
ax_u4.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_u4.set_ylabel('Rot. (rpm)', fontsize=font_axlabel, fontweight='bold')
# ax_u4.set_ylim(-1.1, 1.1)
ax_u4.legend(['$u_4$'], fontsize=18)

# Plot the contact point positions 
point_cols = point_df.columns.values.tolist()
point_cols = ['${0}$'.format(col) for col in point_cols]
fig_cont_pos, (ax_x_cont_pos, ax_y_cont_pos, ax_z_cont_pos) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_cont_pos.suptitle('Contact Point Position States', fontsize=font_suptitle, 
                 fontweight='bold')

# x position
ax_x_cont_pos.plot(point_df.t, point_df.iloc[:, 1::6], linewidth=plt_lw) 
ax_x_cont_pos.set_title('X Position', fontsize=font_subtitle, fontweight='bold')
ax_x_cont_pos.set_ylabel('Position (m)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_x_cont_pos.grid()
# ax_x_pos.set_ylim(-2.5, 2.5)
ax_x_cont_pos.legend(point_cols[1::6], fontsize=18)

# y position
ax_y_cont_pos.plot(point_df.t, point_df.iloc[:, 2::6], linewidth=plt_lw) 
ax_y_cont_pos.set_title('Y Position', fontsize=font_subtitle, fontweight='bold')
ax_y_cont_pos.set_ylabel('Position (m)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_y_cont_pos.grid()
# ax_y_pos.set_ylim(-2.5, 2.5)
ax_y_cont_pos.legend(point_cols[2::6], fontsize=18)

# z position
ax_z_cont_pos.plot(point_df.t, point_df.iloc[:, 3::6], linewidth=plt_lw) 
ax_z_cont_pos.set_title('Z Position', fontsize=font_subtitle, fontweight='bold')
ax_z_cont_pos.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_cont_pos.set_ylabel('Position (m)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_z_cont_pos.grid()
ax_z_cont_pos.invert_yaxis()
ax_z_cont_pos.legend(point_cols[3::6], fontsize=18)
# ax_z_cont_pos.set_ylim(0.5, -5)


# Plot the contact point velocities
fig_cont_vel, (ax_x_cont_vel, ax_y_cont_vel, ax_z_cont_vel) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_cont_vel.suptitle('Contact Point Velocity States', fontsize=font_suptitle, 
                 fontweight='bold')

# x velocity
ax_x_cont_vel.plot(point_df.t, point_df.iloc[:, 4::6], linewidth=plt_lw) 
ax_x_cont_vel.set_title('X Velocity', fontsize=font_subtitle, fontweight='bold')
ax_x_cont_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_x_cont_vel.grid()
# ax_x_vel.set_ylim(-2.5, 2.5)
ax_x_cont_vel.legend(point_cols[4::6], fontsize=18)

# y velocity
ax_y_cont_vel.plot(point_df.t, point_df.iloc[:, 5::6], linewidth=plt_lw) 
ax_y_cont_vel.set_title('Y Velocity', fontsize=font_subtitle, fontweight='bold')
ax_y_cont_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_y_cont_vel.grid()
# ax_y_vel.set_ylim(-2.5, 2.5)
ax_y_cont_vel.legend(point_cols[5::6], fontsize=18)

# z velocity
ax_z_cont_vel.plot(point_df.t, point_df.iloc[:, 6::6], linewidth=plt_lw) 
ax_z_cont_vel.set_title('Z Velocity', fontsize=font_subtitle, fontweight='bold')
ax_z_cont_vel.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_cont_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, 
                         fontweight='bold')
ax_z_cont_vel.grid()
ax_z_cont_vel.invert_yaxis()
ax_z_cont_vel.legend(point_cols[6::6], fontsize=18)
# ax_z_cont_vel.set_ylim(0.5, -5)


# -------------------------- 3D Animation Generation ------------------------- #

# 3D Animation Parameters
uav_zorder = 10
patch_zorder = 1
axes_zorder = 5
txt_zorder = 6

sim_length = tspan[1]
sim_period = timestep
sim_freq = 1/sim_period
sim_steps = int(sim_length*sim_freq)
video_freq = 120
sample_freq = sim_freq/video_freq
num_frames = int(sim_length*video_freq)
sample_idx = np.linspace(0, sim_steps-1, num_frames)
sample_idx = np.around(sample_idx).astype(int)

# Initialize 3d figure
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(projection='3d')

axes_mag = 1 # Magnitude of axes on coordinate frame origins

# Plot inertial frame W axis
ax_3d.plot((0, axes_mag), (0, 0), (0, 0), 'r', linewidth=plt_lw, 
           zorder=axes_zorder)
ax_3d.plot((0, 0), (0, axes_mag), (0, 0), 'g', linewidth=plt_lw, 
           zorder=axes_zorder)
ax_3d.plot((0, 0), (0, 0), (0, axes_mag), 'b', linewidth=plt_lw, 
           zorder=axes_zorder)
ax_3d.text(-0.3, -0.1, 0, 'W', fontsize=font_txt, fontweight='bold', 
           zorder=txt_zorder)

axes_mag = 1 # Magnitude of coordinate frame arms

# Plot contact zone
cont_zone = np.array([[cont_pos_W1[0] + contact_size[0]/2, 
                       cont_pos_W1[1] + contact_size[1]/2],
                      [cont_pos_W1[0] - contact_size[0]/2,
                       cont_pos_W1[1] + contact_size[1]/2],
                      [cont_pos_W1[0] - contact_size[0]/2,
                       cont_pos_W1[1] - contact_size[1]/2], 
                      [cont_pos_W1[0] + contact_size[0]/2,
                       cont_pos_W1[1] - contact_size[1]/2],
                      [cont_pos_W1[0] + contact_size[0]/2, 
                       cont_pos_W1[1] + contact_size[1]/2]])

dock = Polygon(cont_zone, closed=True, alpha=0.8, facecolor='y', 
               edgecolor='k', zorder=1)
ax_3d.add_patch(dock)
art3d.pathpatch_2d_to_3d(dock, z=0, zdir="z")

# Plot ground vehicle (contact zone) frame G axis
ax_3d.plot((cont_pos_W1[0], cont_pos_W1[0] + axes_mag), 
           (cont_pos_W1[1], cont_pos_W1[1]), 
           (cont_pos_W1[2], cont_pos_W1[2]), 
           'r', linewidth=plt_lw, zorder=axes_zorder)
ax_3d.plot((cont_pos_W1[0], cont_pos_W1[0]), 
           (cont_pos_W1[1], cont_pos_W1[1] + axes_mag), 
           (cont_pos_W1[2], cont_pos_W1[2]), 
           'g', linewidth=plt_lw, zorder=axes_zorder)
ax_3d.plot((cont_pos_W1[0], cont_pos_W1[0]), 
           (cont_pos_W1[1], cont_pos_W1[1]), 
           (cont_pos_W1[2], cont_pos_W1[2] + axes_mag), 
           'b', linewidth=plt_lw, zorder=axes_zorder)
ax_3d.text(cont_pos_W1[0] - 0.3, cont_pos_W1[1] - 0.1, cont_pos_W1[2] + 0, 
           'G', fontsize=font_txt, fontweight='bold', zorder=txt_zorder)

# Plot simple UAV shape - contact points
lg_arm1_x = np.array([landing_results_df.x[0], point_df.pa_x[0]])
lg_arm1_y = np.array([landing_results_df.y[0], point_df.pa_y[0]])
lg_arm1_z = np.array([landing_results_df.z[0], point_df.pa_z[0]])
lg_arm2_x = np.array([landing_results_df.x[0], point_df.pb_x[0]])
lg_arm2_y = np.array([landing_results_df.y[0], point_df.pb_y[0]])
lg_arm2_z = np.array([landing_results_df.z[0], point_df.pb_z[0]])
lg_arm3_x = np.array([landing_results_df.x[0], point_df.pc_x[0]])
lg_arm3_y = np.array([landing_results_df.y[0], point_df.pc_y[0]])
lg_arm3_z = np.array([landing_results_df.z[0], point_df.pc_z[0]])
lg_arm4_x = np.array([landing_results_df.x[0], point_df.pd_x[0]])
lg_arm4_y = np.array([landing_results_df.y[0], point_df.pd_y[0]])
lg_arm4_z = np.array([landing_results_df.z[0], point_df.pd_z[0]])
l_c1 = ax_3d.plot(lg_arm1_x, lg_arm1_y, lg_arm1_z, 'k', linewidth=plt_lw, 
                  zorder=uav_zorder)[0] 
l_c2 = ax_3d.plot(lg_arm2_x, lg_arm2_y, lg_arm2_z, 'k', linewidth=plt_lw, 
                  zorder=uav_zorder)[0] 
l_c3 = ax_3d.plot(lg_arm3_x, lg_arm3_y, lg_arm3_z, 'k', linewidth=plt_lw, 
                  zorder=uav_zorder)[0] 
l_c4 = ax_3d.plot(lg_arm4_x, lg_arm4_y, lg_arm4_z, 'k', linewidth=plt_lw, 
                  zorder=uav_zorder)[0] 

# Plot simple UAV shape - UAV cross-arms
q = np.array([landing_results_df.qx[0], landing_results_df.qy[0], 
              landing_results_df.qz[0], landing_results_df.qw[0]])
R_Q_W = Rot.from_quat(q)
uav_arm1_x = np.array([landing_results_df.x[0] + 
                       R_Q_W.apply(iris_params.D[0])[0], 
                       landing_results_df.x[0] + 
                       R_Q_W.apply(iris_params.D[2])[0]])
uav_arm1_y = np.array([landing_results_df.y[0] + 
                       R_Q_W.apply(iris_params.D[0])[1], 
                       landing_results_df.y[0] + 
                       R_Q_W.apply(iris_params.D[2])[1]])
uav_arm1_z = np.array([landing_results_df.z[0] + 
                       R_Q_W.apply(iris_params.D[0])[2], 
                       landing_results_df.z[0] + 
                       R_Q_W.apply(iris_params.D[2])[2]])
uav_arm2_x = np.array([landing_results_df.x[0] + 
                       R_Q_W.apply(iris_params.D[1])[0], 
                       landing_results_df.x[0] + 
                       R_Q_W.apply(iris_params.D[3])[0]])
uav_arm2_y = np.array([landing_results_df.y[0] + 
                       R_Q_W.apply(iris_params.D[1])[1], 
                       landing_results_df.y[0] + 
                       R_Q_W.apply(iris_params.D[3])[1]])
uav_arm2_z = np.array([landing_results_df.z[0] + 
                       R_Q_W.apply(iris_params.D[1])[2], 
                       landing_results_df.z[0] + 
                       R_Q_W.apply(iris_params.D[3])[2]])
l_u1 = ax_3d.plot(uav_arm1_x, uav_arm1_y, uav_arm1_z, 'dimgrey', 
                  linewidth=plt_lw, zorder=uav_zorder)[0] 
l_u2 = ax_3d.plot(uav_arm2_x, uav_arm2_y, uav_arm2_z, 'dimgrey', 
                  linewidth=plt_lw, zorder=uav_zorder)[0]

# Compile lines into a list
lines = [l_c1, l_c2, l_c3, l_c4, l_u1, l_u2]

# 3D plot formatting
ax_3d.set_xlabel('X-Axis (m)', fontsize=font_axlabel, fontweight='bold')
ax_3d.set_ylabel('Y-Axis (m)', fontsize=font_axlabel, fontweight='bold')
ax_3d.set_zlabel('Z-Axis (m)', fontsize=font_axlabel, fontweight='bold')
ax_3d.set_xlim([-0.5, 0.5])
ax_3d.set_ylim([-0.5, 0.5])
ax_3d.set_zlim([-0.9, 0.1])

ax_3d.invert_xaxis()
ax_3d.invert_zaxis()

ax_3d.view_init(elev=30, azim=45)

def update_lines(i: int, lines: list, uav_df: pd.DataFrame, 
                 points_df: pd.DataFrame, sample_idx: np.ndarray):
    '''
    Function used to update the lines in the 3d animation of the UAV flying
    and landing
    '''

    f_i = sample_idx[i]

    print('Animating frame: %i / %i, df row: %i' % (i, len(sample_idx)-1, f_i))

    # Update landing gear lines
    lg_arm1_x = np.array([uav_df.x[f_i], points_df.pa_x[f_i]])
    lg_arm1_y = np.array([uav_df.y[f_i], points_df.pa_y[f_i]])
    lg_arm1_z = np.array([uav_df.z[f_i], points_df.pa_z[f_i]])
    lg_arm2_x = np.array([uav_df.x[f_i], points_df.pb_x[f_i]])
    lg_arm2_y = np.array([uav_df.y[f_i], points_df.pb_y[f_i]])
    lg_arm2_z = np.array([uav_df.z[f_i], points_df.pb_z[f_i]])
    lg_arm3_x = np.array([uav_df.x[f_i], points_df.pc_x[f_i]])
    lg_arm3_y = np.array([uav_df.y[f_i], points_df.pc_y[f_i]])
    lg_arm3_z = np.array([uav_df.z[f_i], points_df.pc_z[f_i]])
    lg_arm4_x = np.array([uav_df.x[f_i], points_df.pd_x[f_i]])
    lg_arm4_y = np.array([uav_df.y[f_i], points_df.pd_y[f_i]])
    lg_arm4_z = np.array([uav_df.z[f_i], points_df.pd_z[f_i]])

    lines[0].set_data(np.array([lg_arm1_x, lg_arm1_y]))
    lines[0].set_3d_properties(lg_arm1_z)

    lines[1].set_data(np.array([lg_arm2_x, lg_arm2_y]))
    lines[1].set_3d_properties(lg_arm2_z)

    lines[2].set_data(np.array([lg_arm3_x, lg_arm3_y]))
    lines[2].set_3d_properties(lg_arm3_z)

    lines[3].set_data(np.array([lg_arm4_x, lg_arm4_y]))
    lines[3].set_3d_properties(lg_arm4_z)

    # Update uav cross lines
    q = np.array([uav_df.qx[f_i], uav_df.qy[f_i], 
              uav_df.qz[f_i], uav_df.qw[f_i]])
    R_Q_W = Rot.from_quat(q)
    uav_arm1_x = np.array([uav_df.x[f_i] + 
                        R_Q_W.apply(iris_params.D[0])[0], 
                        uav_df.x[f_i] + 
                        R_Q_W.apply(iris_params.D[2])[0]])
    uav_arm1_y = np.array([uav_df.y[f_i] + 
                        R_Q_W.apply(iris_params.D[0])[1], 
                        uav_df.y[f_i] + 
                        R_Q_W.apply(iris_params.D[2])[1]])
    uav_arm1_z = np.array([uav_df.z[f_i] + 
                        R_Q_W.apply(iris_params.D[0])[2], 
                        uav_df.z[f_i] + 
                        R_Q_W.apply(iris_params.D[2])[2]])
    uav_arm2_x = np.array([uav_df.x[f_i] + 
                        R_Q_W.apply(iris_params.D[1])[0], 
                        uav_df.x[f_i] + 
                        R_Q_W.apply(iris_params.D[3])[0]])
    uav_arm2_y = np.array([uav_df.y[f_i] + 
                        R_Q_W.apply(iris_params.D[1])[1], 
                        uav_df.y[f_i] + 
                        R_Q_W.apply(iris_params.D[3])[1]])
    uav_arm2_z = np.array([uav_df.z[f_i] + 
                        R_Q_W.apply(iris_params.D[1])[2], 
                        uav_df.z[f_i] + 
                        R_Q_W.apply(iris_params.D[3])[2]])
    
    lines[4].set_data(np.array([uav_arm1_x, uav_arm1_y]))
    lines[4].set_3d_properties(uav_arm1_z)

    lines[5].set_data(np.array([uav_arm2_x, uav_arm2_y]))
    lines[5].set_3d_properties(uav_arm2_z)

    return lines

num_steps = landing_results_df.shape[0]

ani = animation.FuncAnimation(fig_3d, update_lines, num_frames, 
                              fargs=(lines, landing_results_df, point_df, 
                                     sample_idx), interval=1)


# saving to m4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=video_freq)
ani.save(ani_name, writer=writervideo)

# Save and show figures
if save_fig:
    fig_pos.savefig(trial_name + '_pos.png')
    fig_vel.savefig(trial_name + '_vel.png')
    fig_quat.savefig(trial_name + '_quat.png')
    fig_eul.savefig(trial_name + '_eul.png')
    fig_om.savefig(trial_name + '_om.png')
    fig_T.savefig(trial_name + '_T.png')
    fig_u.savefig(trial_name + '_u.png')
    fig_cont_pos.savefig(trial_name + '_cont_pos.png')
    fig_cont_vel.savefig(trial_name + '_cont_vel.png')
    plt.show()
else:
    plt.show()























