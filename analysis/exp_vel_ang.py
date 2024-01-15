
'''
This module is used to plot experimental uav flight states, and report the
desired values at particular times. These experimental trime values can then
be used for simulation comparison or other purposes
'''

# --------------------------------- Imports ---------------------------------- #

# Add package and workspace directories to the path
import os
import sys
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)

# Standard imports
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = ["Latin Modern Roman"]

import numpy as np
import pandas as pd
from math import pi
from scipy.spatial.transform import Rotation as Rot

# Workspace package imports
import amls.analysis.utility_functions as uf_an
import amls.visualization.utility_functions as uf_viz


# ------------------------- Configure Data Directory ------------------------- #

# Path to the experimental comparison data
exp_data_path = os.path.dirname(__file__) + \
    '/Experimental Comparison Data/8_29_23 - UAV Velocity Flights'
exp_file_name = 'log_1_2023-8-29-18-01-22.csv'

# ---------------------------- Raw Data Processing --------------------------- #

# Extract and seperate needed raw data
df_raw_data = pd.read_csv(exp_data_path + '/' + exp_file_name)

df_pos_ned = df_raw_data.loc[:, ['__time', 'vehicle_local_position/x', 
    'vehicle_local_position/y', 'vehicle_local_position/dist_bottom']]
df_pos_ned.dropna(inplace=True)
df_pos_ned.columns = ['t', 'N', 'E', 'D']
df_pos_ned.D *= -1

df_vel_ned = df_raw_data.loc[:, ['__time', 'vehicle_local_position/vx', 
    'vehicle_local_position/vy', 'vehicle_local_position/vz']]
df_vel_ned.dropna(inplace=True)
df_vel_ned.columns = ['t', 'vn', 've', 'vd']

df_heading = df_raw_data.loc[:, ['__time', 'vehicle_local_position/heading']]
df_heading.columns = ['t', 'heading']
df_heading.dropna(inplace=True)
df_heading.heading *= (180/pi)

# qx, qy, qz, qw - Q frame
df_attitude = df_raw_data.loc[:, ['__time', 'vehicle_attitude/q.01', 
    'vehicle_attitude/q.02', 'vehicle_attitude/q.03', 'vehicle_attitude/q.00']]
df_attitude.dropna(inplace=True)
df_attitude.columns = ['t', 'qx', 'qy', 'qz', 'qw']

# Add euler angles to attitude dataframe
df_attitude = uf_an.add_euler_quat(df_attitude)

# Rotate NED position measurements so initial heading is x-axis
heading_init = df_heading.iloc[0, 1]
R_W_ned = Rot.from_euler('xyz', [0, 0, heading_init], degrees=True)
R_ned_W = R_W_ned.inv()
ned_conv_pos = lambda row: pd.Series(R_ned_W.apply([row.N, row.E, row.D]))
df_pos_W = df_pos_ned.apply(ned_conv_pos, axis=1)
df_pos_W.insert(0, 't', df_pos_ned.loc[:, 't'].values)
df_pos_W.columns = ['t', 'x', 'y', 'z']

# Rotate NED velocity measurements for initial heading in x-axis
ned_conv_vel = lambda row: pd.Series(R_ned_W.apply([row.vn, row.ve, row.vd]))
df_vel_W = df_vel_ned.apply(ned_conv_vel, axis=1)
df_vel_W.insert(0, 't', df_vel_ned.loc[:, 't'].values)
df_vel_W.columns = ['t', 'vx', 'vy', 'vz']

# Possibly rotate the attitude angles as well

# ------------------------------- Visualization ------------------------------ #

# # Plot uav position data in rotated W frame
# uf_viz.plot_3vec(df_pos_W, 'UAV Position - Frame W', 
#                  y_label=('Position (m)', 'Position (m)', 'Position (m)'),
#                  subtitle=('X Dimension', 'Y Dimension', 'Z Dimension'))

# Plot uav velocity data in rotated W frame
uf_viz.plot_3ax(df_vel_W, 'UAV Velocity - Frame W', 
                 y_label=('Velocity (m/s)', 'Velocity (m/s)', 'Velocity (m/s)'),
                 subtitle=('X Dimension', 'Y Dimension', 'Z Dimension'))

# # Plot uav attitude euler angles
# uf_viz.plot_3vec(df_attitude.iloc[:, [0, 5, 6, 7]], 'UAV Euler Angles', 
#                  y_label=('Angle (deg)', 'Angle (deg)', 'Angle (deg)'),
#                  subtitle=('Phi', 'Theta', 'Psi'))

# Plot x and y velocity with pitch and roll angles sharing the axes
plt_dir = os.path.dirname(__file__) + '/Experimental Comparison Data/' + \
    '8_29_23 - UAV Velocity Flights/Visualizations/'
plt_name = 'f6_vel_ang.png'
plt_full = plt_dir + plt_name
velocity_tuple = (df_vel_W.iloc[:, :3],)
attitude_tuple = (df_attitude.iloc[:, [0, 6, 5]],)
uf_viz.plot_2ax(velocity_tuple, 'UAV Velocity and Orientation - Flight 6', 
                results2=attitude_tuple, subtitle=('X Axis', 'Y Axis'),
                y_label=('Velocity (m/s)', 'Velocity (m/s)', 'Angle (deg)', 
                         'Angle (deg)'), save_dir=plt_full, plot=False)

plt.show()












