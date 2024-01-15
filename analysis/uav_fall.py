
'''
Module that runs a single trial of an unpowered UAV falling and landing on 
the contact zone. This scipt can be used to test and demonstrate:
- Multi point collision
- Magnet model
- Moving ground vehicle

It is essentially an unpowered quadcopter simulation
'''

# --------------------------------- Imports ---------------------------------- #

# Standard imports
import numpy as np
from numpy import linalg as LA

import pandas as pd

from scipy.spatial.transform import Rotation as Rot

from math import pi, radians

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

from amls.dynamics import quadcopter
from amls.dynamics.quadcopter_params import iris_params
from amls.dynamics.environmental_params import env_1
from amls.dynamics import magnet
from amls.dynamics import ground_vehicle

from amls.visualization import utility_functions as uf_viz

from amls_sim import simulator

# --------------------- Quadcopter Object Configuration ---------------------- #

quad_debug_ctl = { # Quad model debug print settings
    "compute": False # compute_dynamics method
}

quad_iris = quadcopter.QuadcopterDynamics(iris_params, env_1, quad_debug_ctl)

# ----------------------- Ground Vehicle Configuration ----------------------- #

dock_size = np.array([0.762, 0.762])
ground_vehicle_1 = ground_vehicle.GroundVechicleDynamics(contact_size=dock_size)

# ----------------------- Contact Object Configuration ----------------------- #

contact_debug_ctl = { # Debug settings dict
    "warn_mute": False, # Mute warning messages
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
uav_contact_points_Q = np.array([[leg_dim/2, leg_dim/2, leg_dim], # x,y,z by row
                                 [-leg_dim/2, leg_dim/2, leg_dim], 
                                 [-leg_dim/2, -leg_dim/2, leg_dim],
                                 [leg_dim/2, -leg_dim/2, leg_dim]]) 
uav_contact_points1_Q = np.array([[leg_dim/2, leg_dim/2, leg_dim/3],
                                  [-leg_dim/2, leg_dim/2, leg_dim/3], 
                                  [-leg_dim/2, -leg_dim/2, leg_dim/3],
                                  [leg_dim/2, -leg_dim/2, leg_dim/3]]) 
uav_contact_points_noz_Q = np.array([[leg_dim/2, leg_dim/2, 0], # x,y,z by row
                                     [-leg_dim/2, leg_dim/2, 0], 
                                     [-leg_dim/2, -leg_dim/2, 0],
                                     [leg_dim/2, -leg_dim/2, 0]]) 
uav_contact_points_Q = uav_contact_points1_Q
m = 3*iris_params.m
I = 1.5*iris_params.I
e1 = 0.1 # Coefficient of restitution
mu1 = 1.0 # Coefficient of friction
# contact_size = np.array([0.762, 0.762]) # 30 inches to m
uav_contact = contact.ImpulseContact(uav_contact_points_Q, m, I, e=e1, mu=mu1,
                                     cont_size=ground_vehicle_1.contact_size, 
                                     debug_ctl=contact_debug_ctl)

# ------------------- Magnetic Landing Gear Configuration -------------------- #

mag_strength = 0 # 43.86 # Force in N - K&J D66-N52 magnet
strength_vec = mag_strength*np.ones(uav_contact_points_Q.shape[0])
mag_land = magnet.MagnetLanding(uav_contact_points_Q, strength_vec, iris_params)
mag_land.cont_size = ground_vehicle_1.contact_size

# ------------ Overall System Dynamics - Uncontrolled Quadcopter ------------- #

def sysdyn_uav_fall(self, t, x):
    '''
    Function call used by the numerical integrator, including quadcopter system
    dynamics with no propeller inputs and ground

    State vector:
      UAV states: x, y, z, dx_dt, dy_dt, dz_dt, qx, qy, qz, qw, om_x, om_y, om_z
      GV states: x, y, z, dx_dt, dy_dt, dz_dt, qx, qy, qz, qw, om_x, om_y, om_z
    '''

    # Divide states into uav and ground vehicle
    x_uav = x[0:13]
    x_gv = x[13:]

    # Set control inputs for uav and ground vehicle
    u_uav = iris_params.w_min*np.array([1, -1, 1, -1])
    u_gv_lin_vel = np.array([0, 0, 0])
    u_gv_ang_vel = np.array([0, 0, 0])
    u_gv = np.concatenate([u_gv_lin_vel, u_gv_ang_vel])

    # Compute the uav dynamics
    dx_dt_uav = quad_iris.compute_dynamics(t, x_uav, u_uav) # Quad dynamics
    dx_dt_mag = mag_land.compute_forces(t, x_uav, self.cont_mag) # Mag forcing
    dx_dt_uav += dx_dt_mag # Add forcing from magnets

    # Compute the ground vehicle dynamics
    dx_dt_gv = ground_vehicle_1.compute_dynamics(t, x_gv, u_gv)

    # Concatenate dynamics into output vector
    dx_dt = np.concatenate([dx_dt_uav, dx_dt_gv])

    return dx_dt 

# -------------------- Direct State Modification Function -------------------- #

# cont_pos_W1 = np.array([0, 0, 0])

def state_mod(self, t, x, i):
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
    del_lin_vel_uav_W, del_ang_vel_uav_Q, del_z_pos_uav_W, df_cont_tracking = \
        uav_contact.contact_main(pos_uav_W, R_Q_W, lin_vel_uav_W, ang_vel_uav_Q, 
                                 cont_states=x_gv, t=t)
    
    # Generate magnet contact check vector
    self.cont_mag = mag_land.mag_check(uav_contact.points_G, t)

    # Combine state changes back together into output vector
    uav_state_changes = np.concatenate((del_z_pos_uav_W, np.zeros(4), 
                                        del_lin_vel_uav_W, del_ang_vel_uav_Q))
    x_out = np.concatenate((x_uav + uav_state_changes, x_gv))

    return x_out

# --------------------- Configure and Perform Simulation --------------------- #

# Setup simulation control parameters
output_ctl1 = { # Output file control
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'uav_fall_moving_test_fast.csv', # Name for output file
    'file': False, # Save output file
    'plots': True # Save output plots
}

trial_name = 'uav_fall_moving_test_fast'
ani_name = trial_name + '.mp4'
save_fig = 'none' # 'none', 'video', 'all'
show_fig = True

# Simulation time parameters
tspan = (0, 0.1)
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

# UAV States
pos0_W_uav = np.array([-0.5, 0.0, -1.0]) # x, y, z in m
R_init_Q_W_uav = Rot.from_euler('xyz', [5, 0, 0], degrees=True) # Init from eul
q0_uav = R_init_Q_W_uav.as_quat()
vel0_W_uav = np.array([0.0, 0.0, 0]) # x, y, z in m/s
omega0_Q_uav = np.radians(np.array([0.0, 0, 0])) # input deg/s, rad/s a/b x, y,z
x0_uav = np.concatenate((pos0_W_uav,  q0_uav, vel0_W_uav, omega0_Q_uav))
    
# Ground vehicle states
pos0_W_gv = np.array([0.0, 0.0, 0.0]) # x, y, z in m
R_init_G_W_gv = Rot.from_euler('xyz', [0, 0, 0], degrees=True) # Init from eul
q0_gv = R_init_G_W_gv.as_quat()
vel0_W_gv = np.array([0.0, 0.0, 0]) # x, y, z in m/s
omega0_G_gv = np.radians(np.array([0.0, 0, 0])) # input deg/s, rad/s a/b x, y,z
x0_gv = np.concatenate((pos0_W_gv,  q0_gv, vel0_W_gv, omega0_G_gv))

x0 = np.concatenate((x0_uav, x0_gv))

# Initialize simulation
sim_uav_fall = simulator.Simulator(tspan, x0, timestep=timestep, 
                                   state_names=state_names, 
                                   output_ctl=output_ctl1)

# Overwrite default state modification and system dynamics methods
sim_uav_fall.state_mod_fun = MethodType(state_mod, sim_uav_fall)
sim_uav_fall.sysdyn = MethodType(sysdyn_uav_fall, sim_uav_fall)

num_points = uav_contact_points_Q.shape[0]
sim_uav_fall.cont_mag = np.full(num_points, False, dtype=bool)

# Compute the simulation
sim_uav_fall.compute() 

# Setup output dataframes
uav_results_df = sim_uav_fall.state_traj.iloc[:, :14]
gv_results_df = sim_uav_fall.state_traj.iloc[:, 14:]
gv_results_df.insert(0, 't', uav_results_df.t)

# ------------------------------ Post Processing ----------------------------- #

# Add euler angles to the uav trajectory dataframe
eul_conv_uav = lambda row: pd.Series(Rot.from_quat([row.qx_uav, row.qy_uav, 
                        row.qz_uav, row.qw_uav]).as_euler('xyz', degrees=True))
euler_uav_df = uav_results_df.apply(eul_conv_uav, axis=1)

# Rename columns and append to uav dataframe
euler_uav_df.columns = ['phi_uav', 'theta_uav', 'psi_uav']
uav_results_df = pd.concat([uav_results_df, euler_uav_df], axis=1)

# Add euler angles to the gv trajectory dataframe
eul_conv_gv = lambda row: pd.Series(Rot.from_quat([row.qx_gv, row.qy_gv, 
                        row.qz_gv, row.qw_gv]).as_euler('xyz', degrees=True))
euler_gv_df = gv_results_df.apply(eul_conv_gv, axis=1)

# Rename columns and append to gv dataframe
euler_gv_df.columns = ['phi_gv', 'theta_gv', 'psi_gv']
gv_results_df = pd.concat([gv_results_df, euler_gv_df], axis=1)

# Create dataframe with contact point positions and velocities
name_list_points = uf_cont.point_df_name(uav_contact_points_Q.shape[0])
point_df = pd.DataFrame(columns=name_list_points, 
                        index=uav_results_df.index)
point_df.t = uav_results_df.t

# Loop through UAV results df and generate contact data
for i in range(len(uav_results_df.t)): # For each rod position

    # Position and orientation
    uav_pos_W = uav_results_df.iloc[i, 1:4].to_numpy().astype(float)
    uav_q = uav_results_df.iloc[i, 4:8].to_numpy().astype(float)
    R_Q_W = Rot.from_quat(uav_q)

    # Linear and angular velocity
    uav_vel_W = uav_results_df.iloc[i, 8:11].to_numpy().astype(float)
    uav_angvel_Q = uav_results_df.iloc[i, 11:14].to_numpy().astype(float)

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
fig_dpi = 96
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

gv_plot_ctl = 'all' # none, position, or all

# Plot position states and setpoints
fig_pos, (ax_x_pos, ax_y_pos, ax_z_pos) = plt.subplots(3, 1, 
    figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), dpi=fig_dpi, 
    sharex=True)
fig_pos.suptitle('UAV Position States', fontsize=font_suptitle, 
                 fontweight='bold')

# x position
ax_x_pos.plot(uav_results_df.t, uav_results_df.x_uav, linewidth=plt_lw)  
ax_x_pos.set_title('X Position', fontsize=font_subtitle, fontweight='bold')
ax_x_pos.set_ylabel('Position (m)', fontsize=font_axlabel, fontweight='bold')
ax_x_pos.grid()
# ax_x_pos.set_ylim(-2.5, 2.5)
ax_x_pos.legend(['position'], fontsize=18)

# y position
ax_y_pos.plot(uav_results_df.t, uav_results_df.y_uav, linewidth=plt_lw) 
ax_y_pos.set_title('Y Position', fontsize=font_subtitle, fontweight='bold')
ax_y_pos.set_ylabel('Position (m)', fontsize=font_axlabel, fontweight='bold')
ax_y_pos.grid()
# ax_y_pos.set_ylim(-2.5, 2.5)
ax_y_pos.legend(['position'], fontsize=18)

# z position
ax_z_pos.plot(uav_results_df.t, uav_results_df.z_uav, linewidth=plt_lw) 
ax_z_pos.set_title('Z Position', fontsize=font_subtitle, fontweight='bold')
ax_z_pos.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_pos.set_ylabel('Position (m)', fontsize=font_axlabel, fontweight='bold')
ax_z_pos.grid()
ax_z_pos.invert_yaxis()
ax_z_pos.legend(['position'], fontsize=18)
ax_z_pos.set_ylim(0.5, -5)
fig_pos.tight_layout()

# Plot velocity states and setpoints
fig_vel, (ax_x_vel, ax_y_vel, ax_z_vel) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_vel.suptitle('UAV Velocity States', fontsize=font_suptitle, 
                 fontweight='bold')

# x velocity
ax_x_vel.plot(uav_results_df.t, uav_results_df.dx_uav, linewidth=plt_lw) 
ax_x_vel.set_title('X Velocity', fontsize=font_subtitle, fontweight='bold')
ax_x_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, fontweight='bold')
ax_x_vel.grid()
# ax_x_vel.set_ylim(-1.0, 1.0)
ax_x_vel.legend(['velocity'], fontsize=18)

# y velocity
ax_y_vel.plot(uav_results_df.t, uav_results_df.dy_uav, linewidth=plt_lw) 
ax_y_vel.set_title('Y Velocity', fontsize=font_subtitle, fontweight='bold')
ax_y_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, fontweight='bold')
ax_y_vel.grid()
# ax_y_vel.set_ylim(-1.0, 1.0)
ax_y_vel.legend(['velocity'], fontsize=18)

# z velocity
ax_z_vel.plot(uav_results_df.t, uav_results_df.dz_uav, linewidth=plt_lw) 
ax_z_vel.set_title('Z Velocity', fontsize=font_subtitle, fontweight='bold')
ax_z_vel.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_z_vel.set_ylabel('Velocity (m/s)', fontsize=font_axlabel, fontweight='bold')
ax_z_vel.grid()
# ax_z_vel.set_ylim(-1.0, 1.0)
ax_z_vel.invert_yaxis()
ax_z_vel.legend(['velocity'], fontsize=18)
fig_vel.tight_layout()


# Plot attitude quaternion and setpoints
fig_quat, (ax_qx, ax_qy, ax_qz, ax_qw) = \
    plt.subplots(4, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_quat.suptitle('UAV Quaternion States', fontsize=font_suptitle, 
                 fontweight='bold')

# qx
ax_qx.plot(uav_results_df.t, uav_results_df.qx_uav, linewidth=plt_lw) 
ax_qx.set_title('X Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qx.grid()
# ax_qx.set_ylim(-1.1, 1.1)
ax_qx.legend(['$q_x$'], fontsize=18)

# qy
ax_qy.plot(uav_results_df.t, uav_results_df.qy_uav, linewidth=plt_lw) 
ax_qy.set_title('Y Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qy.grid()
# ax_qy.set_ylim(-1.1, 1.1)
ax_qy.legend(['$q_y$'], fontsize=18)

# qz
ax_qz.plot(uav_results_df.t, uav_results_df.qz_uav, linewidth=plt_lw) 
ax_qz.set_title('Z Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qz.grid()
# ax_qz.set_ylim(-1.1, 1.1)
ax_qz.legend(['$q_z$'], fontsize=18)

# qw
ax_qw.plot(uav_results_df.t, uav_results_df.qw_uav, linewidth=plt_lw) 
ax_qw.set_title('W Quaternion', fontsize=font_subtitle, fontweight='bold')
ax_qw.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_qw.grid()
# ax_qw.set_ylim(-1.1, 1.1)
ax_qw.legend(['$q_w$'], fontsize=18)
fig_quat.tight_layout()


# Plot attitude euler angles and setpoints
fig_eul, (ax_phi, ax_theta, ax_psi) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_eul.suptitle('UAV Euler Angles', fontsize=font_suptitle, fontweight='bold')
ax_eul_list = [ax_phi, ax_theta, ax_psi]

# Phi (roll) angle
ax_phi.plot(uav_results_df.t, uav_results_df.phi_uav, linewidth=plt_lw) 
ax_phi.set_title('Roll Angle ($\Phi$)', fontsize=font_subtitle, 
                 fontweight='bold')
ax_phi.set_ylabel('Angle (deg)', fontsize=font_axlabel, fontweight='bold')
ax_phi.grid()
# ax_phi.set_ylim(-60, 60)
ax_phi.legend(['$\Phi$'], fontsize=18)

# Theta (pitch) angle
ax_theta.plot(uav_results_df.t, uav_results_df.theta_uav, 
              linewidth=plt_lw) 
ax_theta.set_title('Pitch Angle ($\Theta$)', fontsize=font_subtitle, 
                   fontweight='bold')
ax_theta.set_ylabel('Angle (deg)', fontsize=font_axlabel, fontweight='bold')
ax_theta.grid()
# ax_theta.set_ylim(-60, 60)
ax_theta.legend(['$\Theta$'], fontsize=18)

# Psi (yaw) angle
ax_psi.plot(uav_results_df.t, uav_results_df.psi_uav, linewidth=plt_lw) 
ax_psi.set_title('Yaw Angle ($\Psi$)', fontsize=font_subtitle, 
                 fontweight='bold')
ax_psi.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_psi.set_ylabel('Angle (deg)', fontsize=font_axlabel, fontweight='bold')
ax_psi.grid()
# ax_psi.set_ylim(-60, 60)
ax_psi.legend(['$\Psi$'], fontsize=18)
fig_eul.tight_layout()


# Plot angular rates and setpoints
fig_om, (ax_om_x, ax_om_y, ax_om_z) = \
    plt.subplots(3, 1, figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                 dpi=fig_dpi, sharex=True)
fig_om.suptitle('UAV Angular Velocity States', fontsize=font_suptitle, 
                 fontweight='bold')
ax_om_list = [ax_om_x, ax_om_y, ax_om_z]

# x angular velocity
ax_om_x.plot(uav_results_df.t, (180/pi)*uav_results_df.om_x_uav, 
             linewidth=plt_lw) 
ax_om_x.set_title('$\omega_x$', fontsize=font_subtitle, fontweight='bold')
ax_om_x.set_ylabel('Ang. Vel. (deg/s)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_om_x.grid()
# ax_om_x.set_ylim(-60, 60)
ax_om_x.legend(['$\omega_x$'], fontsize=18)

# y angular velocity
ax_om_y.plot(uav_results_df.t, (180/pi)*uav_results_df.om_y_uav, 
             linewidth=plt_lw) 
ax_om_y.set_title('$\omega_y$', fontsize=font_subtitle, fontweight='bold')
ax_om_y.set_ylabel('Ang. Vel. (deg/s)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_om_y.grid()
# ax_om_y.set_ylim(-60, 60)
ax_om_y.legend(['$\omega_y$'], fontsize=18)

# z angular velocity
ax_om_z.plot(uav_results_df.t, (180/pi)*uav_results_df.om_z_uav, 
             linewidth=plt_lw) 
ax_om_z.set_title('$\omega_z$', fontsize=font_subtitle, fontweight='bold')
ax_om_z.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
ax_om_z.set_ylabel('Ang. Vel. (deg/s)', fontsize=font_axlabel, 
                   fontweight='bold')
ax_om_z.grid()
# ax_om_z.set_ylim(-60, 60)
ax_om_z.legend(['$\omega_z$'], fontsize=18)
fig_om.tight_layout()


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
fig_cont_pos.tight_layout()


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
fig_cont_vel.tight_layout()


# Ground vehicle data additions
if gv_plot_ctl == 'all':

    ax_x_pos.plot(gv_results_df.t, gv_results_df.x_gv, '--', linewidth=plt_lw)
    ax_y_pos.plot(gv_results_df.t, gv_results_df.y_gv, '--', linewidth=plt_lw)
    ax_z_pos.plot(gv_results_df.t, gv_results_df.z_gv, '--', linewidth=plt_lw)
    ax_x_pos.legend(['$x_{uav}$', '$x_{gv}$'], fontsize=18)
    ax_y_pos.legend(['$y_{uav}$', '$y_{gv}$'], fontsize=18)
    ax_z_pos.legend(['$z_{uav}$', '$z_{gv}$'], fontsize=18)

    ax_x_vel.plot(gv_results_df.t, gv_results_df.dx_gv, '--', linewidth=plt_lw)
    ax_y_vel.plot(gv_results_df.t, gv_results_df.dy_gv, '--', linewidth=plt_lw)
    ax_z_vel.plot(gv_results_df.t, gv_results_df.dz_gv, '--', linewidth=plt_lw)
    ax_x_vel.legend(['$dx_{uav}$', '$dx_{gv}$'], fontsize=18)
    ax_y_vel.legend(['$dy_{uav}$', '$dy_{gv}$'], fontsize=18)
    ax_z_vel.legend(['$dz_{uav}$', '$dz_{gv}$'], fontsize=18)

    ax_qx.plot(gv_results_df.t, gv_results_df.qx_gv, '--', linewidth=plt_lw) 
    ax_qy.plot(gv_results_df.t, gv_results_df.qy_gv, '--', linewidth=plt_lw) 
    ax_qz.plot(gv_results_df.t, gv_results_df.qz_gv, '--', linewidth=plt_lw) 
    ax_qw.plot(gv_results_df.t, gv_results_df.qw_gv, '--', linewidth=plt_lw) 
    ax_qx.legend(['$qx_{uav}$', '$qx_{gv}$'], fontsize=18)
    ax_qy.legend(['$qy_{uav}$', '$qy_{gv}$'], fontsize=18)
    ax_qz.legend(['$qz_{uav}$', '$qz_{gv}$'], fontsize=18)
    ax_qw.legend(['$qw_{uav}$', '$qw_{gv}$'], fontsize=18)

    ax_phi.plot(gv_results_df.t, gv_results_df.phi_gv, '--', linewidth=plt_lw) 
    ax_theta.plot(gv_results_df.t, gv_results_df.theta_gv, '--', 
                  linewidth=plt_lw) 
    ax_psi.plot(gv_results_df.t, gv_results_df.psi_gv, '--', linewidth=plt_lw) 
    ax_phi.legend(['$\Phi_{uav}$', '$\Phi_{gv}$'], fontsize=18)
    ax_theta.legend(['$\Theta_{uav}$', '$\Theta_{gv}$'], fontsize=18)
    ax_psi.legend(['$\Psi_{uav}$', '$\Psi_{gv}$'], fontsize=18)

    ax_om_x.plot(gv_results_df.t, (180/pi)*gv_results_df.om_x_gv, '--', 
                 linewidth=plt_lw) 
    ax_om_y.plot(gv_results_df.t, (180/pi)*gv_results_df.om_y_gv, '--', 
                 linewidth=plt_lw) 
    ax_om_z.plot(gv_results_df.t, (180/pi)*gv_results_df.om_z_gv, '--', 
                 linewidth=plt_lw) 
    ax_om_x.legend(['$\omega x_{uav}$', '$\omega x_{gv}$'], fontsize=18)
    ax_om_y.legend(['$\omega y_{uav}$', '$\omega y_{gv}$'], fontsize=18)
    ax_om_z.legend(['$\omega z_{uav}$', '$\omega z_{gv}$'], fontsize=18)

elif gv_plot_ctl == 'position':

    ax_x_pos.plot(gv_results_df.t, gv_results_df.x_gv, '--', linewidth=plt_lw)
    ax_y_pos.plot(gv_results_df.t, gv_results_df.y_gv, '--', linewidth=plt_lw)
    ax_z_pos.plot(gv_results_df.t, gv_results_df.z_gv, '--', linewidth=plt_lw)
    ax_x_pos.legend(['$x_{uav}$', '$x_{gv}$'], fontsize=18)
    ax_y_pos.legend(['$y_{uav}$', '$y_{gv}$'], fontsize=18)
    ax_z_pos.legend(['$z_{uav}$', '$z_{gv}$'], fontsize=18)

    ax_x_vel.plot(gv_results_df.t, gv_results_df.dx_gv, '--', linewidth=plt_lw)
    ax_y_vel.plot(gv_results_df.t, gv_results_df.dy_gv, '--', linewidth=plt_lw)
    ax_z_vel.plot(gv_results_df.t, gv_results_df.dz_gv, '--', linewidth=plt_lw)
    ax_x_vel.legend(['$dx_{uav}$', '$dx_{gv}$'], fontsize=18)
    ax_y_vel.legend(['$dy_{uav}$', '$dy_{gv}$'], fontsize=18)
    ax_z_vel.legend(['$dz_{uav}$', '$dz_{gv}$'], fontsize=18)

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
speed_scale = 0.5 # Scales the playback speed of the video export
sample_freq = sim_freq/video_freq
num_frames = int(sim_length*video_freq/speed_scale)
sample_idx = np.linspace(0, sim_steps-1, num_frames)
sample_idx = np.around(sample_idx).astype(int)
ani_dpi = 200

# Initialize 3d figure
fig_3d = plt.figure()
fig_3d.set_size_inches(fig_scale*fig_x_size, fig_scale*fig_y_size, True)
fig_3d.set_dpi(ani_dpi)
ax_3d = fig_3d.add_subplot(projection='3d')

axes_mag = 0.5 # Magnitude of axes on coordinate frame origins

# Plot inertial frame W axis
ax_3d.plot((0, axes_mag), (0, 0), (0, 0), 'r', linewidth=plt_lw, 
           zorder=axes_zorder)
ax_3d.plot((0, 0), (0, axes_mag), (0, 0), 'g', linewidth=plt_lw, 
           zorder=axes_zorder)
ax_3d.plot((0, 0), (0, 0), (0, axes_mag), 'b', linewidth=plt_lw, 
           zorder=axes_zorder)
ax_3d.text(-0.3, -0.1, 0, 'W', fontsize=font_txt, fontweight='bold', 
           zorder=txt_zorder)

# Plot contact zone
cont_zone = np.array([[gv_results_df.x_gv[0] +
                           ground_vehicle_1.contact_size[0]/2, 
                       gv_results_df.y_gv[0] +
                           ground_vehicle_1.contact_size[1]/2],
                      [gv_results_df.x_gv[0] -
                          ground_vehicle_1.contact_size[0]/2,
                       gv_results_df.y_gv[0] +
                           ground_vehicle_1.contact_size[1]/2],
                      [gv_results_df.x_gv[0] -
                           ground_vehicle_1.contact_size[0]/2,
                       gv_results_df.y_gv[0] -
                           ground_vehicle_1.contact_size[1]/2], 
                      [gv_results_df.x_gv[0] +
                           ground_vehicle_1.contact_size[0]/2,
                       gv_results_df.y_gv[0] -
                           ground_vehicle_1.contact_size[1]/2],
                      [gv_results_df.x_gv[0] +
                           ground_vehicle_1.contact_size[0]/2, 
                       gv_results_df.y_gv[0] +
                           ground_vehicle_1.contact_size[1]/2]])

dock = Polygon(cont_zone, closed=True, alpha=0.8, facecolor='y', 
               edgecolor='k', zorder=1)
ax_3d.add_patch(dock)
art3d.pathpatch_2d_to_3d(dock, z=0, zdir="z")

# Plot ground vehicle (contact zone) frame G axis
l_g1 = ax_3d.plot((gv_results_df.x_gv[0], gv_results_df.x_gv[0] + axes_mag), 
                  (gv_results_df.y_gv[0], gv_results_df.y_gv[0]), 
                  (gv_results_df.z_gv[0], gv_results_df.z_gv[0]), 
                  'r', linewidth=plt_lw, zorder=axes_zorder)[0]
l_g2 = ax_3d.plot((gv_results_df.x_gv[0], gv_results_df.x_gv[0]), 
                  (gv_results_df.y_gv[0], gv_results_df.y_gv[0] + axes_mag), 
                  (gv_results_df.z_gv[0], gv_results_df.z_gv[0]), 
                  'g', linewidth=plt_lw, zorder=axes_zorder)[0]
l_g3 = ax_3d.plot((gv_results_df.x_gv[0], gv_results_df.x_gv[0]), 
                  (gv_results_df.y_gv[0], gv_results_df.y_gv[0]), 
                  (gv_results_df.z_gv[0], gv_results_df.z_gv[0] + axes_mag), 
                  'b', linewidth=plt_lw, zorder=axes_zorder)[0]
t_g1 = ax_3d.text(gv_results_df.x_gv[0] - 0.3, gv_results_df.y_gv[0] - 0.1, 
                  gv_results_df.z_gv[0] + 0, 'G', fontsize=font_txt, 
                  fontweight='bold', zorder=txt_zorder)

# Plot simple UAV shape - contact points
lg_arm1_x = np.array([uav_results_df.x_uav[0], point_df.pa_x[0]])
lg_arm1_y = np.array([uav_results_df.y_uav[0], point_df.pa_y[0]])
lg_arm1_z = np.array([uav_results_df.z_uav[0], point_df.pa_z[0]])
lg_arm2_x = np.array([uav_results_df.x_uav[0], point_df.pb_x[0]])
lg_arm2_y = np.array([uav_results_df.y_uav[0], point_df.pb_y[0]])
lg_arm2_z = np.array([uav_results_df.z_uav[0], point_df.pb_z[0]])
lg_arm3_x = np.array([uav_results_df.x_uav[0], point_df.pc_x[0]])
lg_arm3_y = np.array([uav_results_df.y_uav[0], point_df.pc_y[0]])
lg_arm3_z = np.array([uav_results_df.z_uav[0], point_df.pc_z[0]])
lg_arm4_x = np.array([uav_results_df.x_uav[0], point_df.pd_x[0]])
lg_arm4_y = np.array([uav_results_df.y_uav[0], point_df.pd_y[0]])
lg_arm4_z = np.array([uav_results_df.z_uav[0], point_df.pd_z[0]])
l_c1 = ax_3d.plot(lg_arm1_x, lg_arm1_y, lg_arm1_z, 'k', linewidth=plt_lw, 
                  zorder=uav_zorder)[0] 
l_c2 = ax_3d.plot(lg_arm2_x, lg_arm2_y, lg_arm2_z, 'k', linewidth=plt_lw, 
                  zorder=uav_zorder)[0] 
l_c3 = ax_3d.plot(lg_arm3_x, lg_arm3_y, lg_arm3_z, 'k', linewidth=plt_lw, 
                  zorder=uav_zorder)[0] 
l_c4 = ax_3d.plot(lg_arm4_x, lg_arm4_y, lg_arm4_z, 'k', linewidth=plt_lw, 
                  zorder=uav_zorder)[0] 

# Plot simple UAV shape - UAV cross-arms
q = np.array([uav_results_df.qx_uav[0], uav_results_df.qy_uav[0], 
              uav_results_df.qz_uav[0], uav_results_df.qw_uav[0]])
R_Q_W = Rot.from_quat(q)
uav_arm1_x = np.array([uav_results_df.x_uav[0] + 
                       R_Q_W.apply(iris_params.D[0])[0], 
                       uav_results_df.x_uav[0] + 
                       R_Q_W.apply(iris_params.D[2])[0]])
uav_arm1_y = np.array([uav_results_df.y_uav[0] + 
                       R_Q_W.apply(iris_params.D[0])[1], 
                       uav_results_df.y_uav[0] + 
                       R_Q_W.apply(iris_params.D[2])[1]])
uav_arm1_z = np.array([uav_results_df.z_uav[0] + 
                       R_Q_W.apply(iris_params.D[0])[2], 
                       uav_results_df.z_uav[0] + 
                       R_Q_W.apply(iris_params.D[2])[2]])
uav_arm2_x = np.array([uav_results_df.x_uav[0] + 
                       R_Q_W.apply(iris_params.D[1])[0], 
                       uav_results_df.x_uav[0] + 
                       R_Q_W.apply(iris_params.D[3])[0]])
uav_arm2_y = np.array([uav_results_df.y_uav[0] + 
                       R_Q_W.apply(iris_params.D[1])[1], 
                       uav_results_df.y_uav[0] + 
                       R_Q_W.apply(iris_params.D[3])[1]])
uav_arm2_z = np.array([uav_results_df.z_uav[0] + 
                       R_Q_W.apply(iris_params.D[1])[2], 
                       uav_results_df.z_uav[0] + 
                       R_Q_W.apply(iris_params.D[3])[2]])
l_u1 = ax_3d.plot(uav_arm1_x, uav_arm1_y, uav_arm1_z, 'dimgrey', 
                  linewidth=plt_lw, zorder=uav_zorder)[0] 
l_u2 = ax_3d.plot(uav_arm2_x, uav_arm2_y, uav_arm2_z, 'dimgrey', 
                  linewidth=plt_lw, zorder=uav_zorder)[0]

# Add timescale information to the plot
timescale_msg = "Playback Speed: %0.2fx" % speed_scale
timestamp_msg = "t = %0.5fs" % uav_results_df.t[0]
t_t1 = ax_3d.text2D(0.1, 0.025, timescale_msg, transform=ax_3d.transAxes, 
                    fontsize=font_txt, fontweight='bold', zorder=txt_zorder)
t_t2 = ax_3d.text2D(0.1, 0.05, timestamp_msg, transform=ax_3d.transAxes, 
                    fontsize=font_txt, fontweight='bold', zorder=txt_zorder)

# Compile lines into a list
lines = [l_c1, l_c2, l_c3, l_c4, # Contact point lines
         l_u1, l_u2, # UAV crossbar lines
         l_g1, l_g2, l_g3, t_g1, # G-frame axes and text
         t_t1, t_t2, # Timestamp text
         dock] # Dock polygon

# 3D plot formatting
ax_3d.set_xlabel('X-Axis (m)', fontsize=font_axlabel, fontweight='bold')
ax_3d.set_ylabel('Y-Axis (m)', fontsize=font_axlabel, fontweight='bold')
ax_3d.set_zlabel('Z-Axis (m)', fontsize=font_axlabel, fontweight='bold')
ax_3d.set_xlim(-0.5)
ax_3d.set_ylim([-0.75, 0.75])
ax_3d.set_zlim([-1.4, 0.1])

addition_val = max((axes_mag, ground_vehicle_1.contact_size[0]/2))
x_aspect = 1.2*(gv_results_df.x_gv.iloc[-1] + addition_val)
ax_3d.set_box_aspect([x_aspect, 1.0, 1.0])
ax_3d.set_aspect('equal', adjustable='datalim', anchor='C')

ax_3d.invert_xaxis()
ax_3d.invert_zaxis()
ax_3d.view_init(elev=30, azim=45)
fig_3d.tight_layout()

def update_lines(i: int, lines: list, uav_df: pd.DataFrame, 
                 points_df: pd.DataFrame, sample_idx: np.ndarray):
    '''
    Function used to update the lines in the 3d animation of the UAV flying
    and landing
    '''

    df_i = sample_idx[i] # Get dataframe index for this iteration subsample

    print('Animating frame: %i / %i, df row: %i\r' % (i, len(sample_idx)-1, 
                                                      df_i), end="")

    # Update landing gear lines
    lg_arm1_x = np.array([uav_df.x_uav[df_i], points_df.pa_x[df_i]])
    lg_arm1_y = np.array([uav_df.y_uav[df_i], points_df.pa_y[df_i]])
    lg_arm1_z = np.array([uav_df.z_uav[df_i], points_df.pa_z[df_i]])
    lg_arm2_x = np.array([uav_df.x_uav[df_i], points_df.pb_x[df_i]])
    lg_arm2_y = np.array([uav_df.y_uav[df_i], points_df.pb_y[df_i]])
    lg_arm2_z = np.array([uav_df.z_uav[df_i], points_df.pb_z[df_i]])
    lg_arm3_x = np.array([uav_df.x_uav[df_i], points_df.pc_x[df_i]])
    lg_arm3_y = np.array([uav_df.y_uav[df_i], points_df.pc_y[df_i]])
    lg_arm3_z = np.array([uav_df.z_uav[df_i], points_df.pc_z[df_i]])
    lg_arm4_x = np.array([uav_df.x_uav[df_i], points_df.pd_x[df_i]])
    lg_arm4_y = np.array([uav_df.y_uav[df_i], points_df.pd_y[df_i]])
    lg_arm4_z = np.array([uav_df.z_uav[df_i], points_df.pd_z[df_i]])

    lines[0].set_data(np.array([lg_arm1_x, lg_arm1_y]))
    lines[0].set_3d_properties(lg_arm1_z)

    lines[1].set_data(np.array([lg_arm2_x, lg_arm2_y]))
    lines[1].set_3d_properties(lg_arm2_z)

    lines[2].set_data(np.array([lg_arm3_x, lg_arm3_y]))
    lines[2].set_3d_properties(lg_arm3_z)

    lines[3].set_data(np.array([lg_arm4_x, lg_arm4_y]))
    lines[3].set_3d_properties(lg_arm4_z)

    # Update uav cross lines
    q = np.array([uav_df.qx_uav[df_i], uav_df.qy_uav[df_i], 
              uav_df.qz_uav[df_i], uav_df.qw_uav[df_i]])
    R_Q_W = Rot.from_quat(q)
    uav_arm1_x = np.array([uav_df.x_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[0])[0], 
                        uav_df.x_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[2])[0]])
    uav_arm1_y = np.array([uav_df.y_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[0])[1], 
                        uav_df.y_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[2])[1]])
    uav_arm1_z = np.array([uav_df.z_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[0])[2], 
                        uav_df.z_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[2])[2]])
    uav_arm2_x = np.array([uav_df.x_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[1])[0], 
                        uav_df.x_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[3])[0]])
    uav_arm2_y = np.array([uav_df.y_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[1])[1], 
                        uav_df.y_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[3])[1]])
    uav_arm2_z = np.array([uav_df.z_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[1])[2], 
                        uav_df.z_uav[df_i] + 
                        R_Q_W.apply(iris_params.D[3])[2]])
    
    lines[4].set_data(np.array([uav_arm1_x, uav_arm1_y]))
    lines[4].set_3d_properties(uav_arm1_z)

    lines[5].set_data(np.array([uav_arm2_x, uav_arm2_y]))
    lines[5].set_3d_properties(uav_arm2_z)

    # Update the G coordinate frame
    G_xaxis_x = (gv_results_df.x_gv[df_i], gv_results_df.x_gv[df_i] + axes_mag)
    G_xaxis_y = (gv_results_df.y_gv[df_i], gv_results_df.y_gv[df_i])
    G_xaxis_z = (gv_results_df.z_gv[df_i], gv_results_df.z_gv[df_i])
    G_yaxis_x = (gv_results_df.x_gv[df_i], gv_results_df.x_gv[df_i])
    G_yaxis_y = (gv_results_df.y_gv[df_i], gv_results_df.y_gv[df_i] + axes_mag)
    G_yaxis_z = (gv_results_df.z_gv[df_i], gv_results_df.z_gv[df_i])
    G_zaxis_x = (gv_results_df.x_gv[df_i], gv_results_df.x_gv[df_i])
    G_zaxis_y = (gv_results_df.y_gv[df_i], gv_results_df.y_gv[df_i])
    G_zaxis_z = (gv_results_df.z_gv[df_i], gv_results_df.z_gv[df_i] + axes_mag)

    G_txt_x = gv_results_df.x_gv[df_i] - 0.3
    G_txt_y = gv_results_df.y_gv[df_i] - 0.1
    G_txt_z = gv_results_df.z_gv[df_i] + 0

    lines[6].set_data(np.array([G_xaxis_x, G_xaxis_y]))
    lines[6].set_3d_properties(G_xaxis_z)

    lines[7].set_data(np.array([G_yaxis_x, G_yaxis_y]))
    lines[7].set_3d_properties(G_yaxis_z)

    lines[8].set_data(np.array([G_zaxis_x, G_zaxis_y]))
    lines[8].set_3d_properties(G_zaxis_z)

    lines[9].set_position(np.array([G_txt_x, G_txt_y, G_txt_z]))

    # Update the ground vehicle patch
    dock = lines[-1]
    cont_zone = np.array([[gv_results_df.x_gv[df_i] + 
                               ground_vehicle_1.contact_size[0]/2, 
                           gv_results_df.y_gv[df_i] + 
                               ground_vehicle_1.contact_size[1]/2],
                          [gv_results_df.x_gv[df_i] - 
                               ground_vehicle_1.contact_size[0]/2,
                           gv_results_df.y_gv[df_i] + 
                               ground_vehicle_1.contact_size[1]/2],
                          [gv_results_df.x_gv[df_i] - 
                               ground_vehicle_1.contact_size[0]/2,
                           gv_results_df.y_gv[df_i] - 
                               ground_vehicle_1.contact_size[1]/2], 
                          [gv_results_df.x_gv[df_i] + 
                               ground_vehicle_1.contact_size[0]/2,
                           gv_results_df.y_gv[df_i] - 
                               ground_vehicle_1.contact_size[1]/2],
                          [gv_results_df.x_gv[df_i] + 
                               ground_vehicle_1.contact_size[0]/2, 
                           gv_results_df.y_gv[df_i] + 
                               ground_vehicle_1.contact_size[1]/2]])
    dock.remove()
    dock = Polygon(cont_zone, closed=True, alpha=0.8, facecolor='y', 
                   edgecolor='k', zorder=1)
    ax_3d.add_patch(dock)
    art3d.pathpatch_2d_to_3d(dock, z=0, zdir="z")
    lines[-1] = dock

    # Update the timestamp text
    timestamp_msg = "t = %0.5fs" % uav_results_df.t[df_i]
    lines[11].set_text(timestamp_msg)

    return lines

num_steps = uav_results_df.shape[0]

ani = animation.FuncAnimation(fig_3d, update_lines, num_frames, 
                              fargs=(lines, uav_results_df, point_df, 
                                     sample_idx), interval=1)

# Save and show figures
if save_fig == 'all':

    # saving to m4 using ffmpeg writer
    writervideo = animation.FFMpegWriter(fps=video_freq)
    ani.save(ani_name, writer=writervideo)

    fig_pos.savefig(trial_name + '_pos.png')
    fig_vel.savefig(trial_name + '_vel.png')
    fig_quat.savefig(trial_name + '_quat.png')
    fig_eul.savefig(trial_name + '_eul.png')
    fig_om.savefig(trial_name + '_om.png')
    fig_cont_pos.savefig(trial_name + '_cont_pos.png')
    fig_cont_vel.savefig(trial_name + '_cont_vel.png')

elif save_fig == 'video':

    # saving to m4 using ffmpeg writer
    writervideo = animation.FFMpegWriter(fps=video_freq)
    ani.save(ani_name, writer=writervideo)

if show_fig:
    plt.show()
