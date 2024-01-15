
'''
This module is used to correlate the contact model with experimental contact 
data. The module will preprocess experimental data if needed and then run 
the simulation with initial conditions corresponding to the start of all of the
experimental cases.
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
import ahrs
from tqdm import tqdm
import timeit

# Workspace package imports
import amls.analysis.utility_functions as uf_an
import amls.visualization.utility_functions as uf_viz
from amls_impulse_contact.contact import uav_contact_x500_exp
from amls.visualization.sim_plotting import plot_cmd_dict_plot, \
    plot_cmd_dict_save
from amls.visualization import sim_plotting

# Configured simulation object imports
from amls.analysis.configured_sims import sim_uav_fall_nomag
sim_uav_fall_nomag.load_tspan((0, 2.0), timestep=0.0005)

from amls.analysis.configured_sims import sim_uav_fall_mag
sim_uav_fall_mag.load_tspan((0, 2.0), timestep=0.0005)

sim_uav_fall_use = sim_uav_fall_nomag

# ------------------ Configure Data Directory and Processing ----------------- #

# Path to the experimental comparison data
exp_data_path = os.path.dirname(__file__) + \
    '/Experimental Comparison Data/8_22_23 - UAV Drops'

# Preprocessing control parameters
raw_preprocess = False
bounds_preprocess = False
sim_comparison = True

# -------------------------- Raw Data Preprocessing -------------------------- #

if raw_preprocess:

    vicon_freq = 100 # Hz frequency of VICON system taking data

    # Get all contents of the raw data directory not including cleaned files
    exp_data_contents_clean = os.listdir(exp_data_path)
    exp_data_contents_clean = [i for i in exp_data_contents_clean 
                               if 'clean' not in i and 'bounds' not in i
                               and '.csv' in i]

    # Loop through each experimental VICON dataset
    for data in exp_data_contents_clean:

        dt = 1/vicon_freq

        # Copy raw data columns into dataframe
        df_raw_data = pd.read_csv(exp_data_path + '/' + data, skiprows=5, 
                                  header=None)
        num_steps = df_raw_data.shape[0]
        
        col_names = ['t', 'x_uav', 'y_uav', 'z_uav', 
                     'qx_uav', 'qy_uav', 'qz_uav', 'qw_uav', 
                     'dx_uav', 'dy_uav', 'dz_uav', 
                     'om_x_uav', 'om_y_uav', 'om_z_uav']
        
        df_clean_data = pd.DataFrame(index=np.arange(num_steps), 
                                     columns=col_names)
        df_clean_data.t[:] = dt*np.arange(num_steps)

        # Compute and define positions, orientations, and velocities
        x_raw = df_raw_data.iloc[:, 6]/1000
        y_raw = df_raw_data.iloc[:, 7]/1000
        z_raw = df_raw_data.iloc[:, 8]/1000
        qx_raw = df_raw_data.iloc[:, 2]
        qy_raw = df_raw_data.iloc[:, 3]
        qz_raw = df_raw_data.iloc[:, 4]
        qw_raw = df_raw_data.iloc[:, 5]
        x_uav = x_raw
        y_uav = -y_raw
        z_uav = -z_raw
        qx_uav = qx_raw
        qy_uav = -qy_raw
        qz_uav = -qz_raw
        qw_uav = qw_raw

        pos_adj = \
            uf_an.body_fixed_z_adjust(pd.concat([x_uav, y_uav, z_uav], axis=1),
                 -0.026, pd.concat([qx_uav, qy_uav, qz_uav, qw_uav], axis=1))

        dx_uav = np.gradient(x_uav, dt, edge_order=2)
        dy_uav = np.gradient(y_uav, dt, edge_order=2)
        dz_uav = np.gradient(z_uav, dt, edge_order=2)

        quats = np.array([qx_uav, qy_uav, qz_uav, qw_uav]).T
        Q = ahrs.QuaternionArray(quats)
        ang_vels = Q.angular_velocities(dt)

        om_x_uav = ang_vels[:, 0]
        om_y_uav = ang_vels[:, 1]
        om_z_uav = ang_vels[:, 2]

        # Assign to cleaned dataframe
        df_clean_data.x_uav[:] = pos_adj.iloc[:, 0]
        df_clean_data.y_uav[:] = pos_adj.iloc[:, 1]
        df_clean_data.z_uav[:] = pos_adj.iloc[:, 2]
        df_clean_data.qx_uav[:] = qx_uav
        df_clean_data.qy_uav[:] = qy_uav
        df_clean_data.qz_uav[:] = qz_uav
        df_clean_data.qw_uav[:] = qw_uav

        df_clean_data.dx_uav[:] = dx_uav
        df_clean_data.dy_uav[:] = dy_uav
        df_clean_data.dz_uav[:] = dz_uav

        df_clean_data.om_x_uav[0] = 0
        df_clean_data.om_x_uav[1:] = om_x_uav
        df_clean_data.om_y_uav[0] = 0
        df_clean_data.om_y_uav[1:] = om_y_uav
        df_clean_data.om_z_uav[0] = 0
        df_clean_data.om_z_uav[1:] = om_z_uav

        # Add euler angles to the data
        df_clean_data = uf_an.add_euler(df_clean_data)

        # Save cleaned data as new .csv with _clean at the end
        df_clean_data.to_csv(exp_data_path + '/' + data[:-4] + '_clean.csv', 
                             index=False)

# -------------------------- Raw Data Plot and Trim -------------------------- #

if bounds_preprocess:

    # Get all of the cleaned contents of the raw data directory
    exp_data_contents = os.listdir(exp_data_path)
    exp_data_contents_clean = [i for i in exp_data_contents if 'clean' in i]

    # Get bounds if the exist
    if 'bounds.csv' in exp_data_contents:
        df_bounds = pd.read_csv(exp_data_path + '/bounds.csv')
        bound_files = df_bounds.File.values

    # Loop through each cleaned file
    for data in exp_data_contents_clean:
        
        # Load in the cleaned .csv as a dataframe
        df_clean = pd.read_csv(exp_data_path + '/' + data)

        bounds = None
        if data in bound_files:
            row = df_bounds.index[df_bounds.File == data]
            bounds = df_bounds.iloc[row, 1:].values[0]

        # Plot the position, orientation, linear velocity, and angular velocity
        uf_viz.plot_3ax(df_clean.iloc[:, [0, 1, 2, 3]], title=data+' Position', 
                         bounds=bounds)
        uf_viz.plot_3ax(df_clean.iloc[:, [0, 8, 9, 10]], title='Euler', 
                         bounds=bounds)
        uf_viz.plot_3ax(df_clean.iloc[:, [0, 11, 12, 13]], 
                         title='Linear Velocity', bounds=bounds)
        uf_viz.plot_3ax(df_clean.iloc[:, [0, 14, 15, 16]], bounds=bounds,
                         title='Angular Velocity')

        plt.show()

# ---------------- Experimental-Simulation Contact Correlation --------------- #

if sim_comparison:

    param_path = os.path.dirname(__file__) + '/Simulation Parameters/'

    # Import ground vehicle initial conditions
    gv_name = 'gv_ic_fall_set1'
    gv_csv_path = param_path + gv_name + '.csv'
    df_gv_ic_fall_set1 = pd.read_csv(gv_csv_path)
    x0_gv = df_gv_ic_fall_set1.values[0]

    # Call function to setup the results directory
    uav_name = 'Not Used'
    batch_name = 'uav_cont_exp'
    root_name, txt_list = uf_an.result_dir_setup(batch_name, gv_name, uav_name)
    result_path = os.path.dirname(__file__) + '/Simulation Results/' + root_name

    # Compile list of cleaned experimental data files to use
    exp_data_contents = os.listdir(exp_data_path)
    exp_data_contents_clean = [i for i in exp_data_contents if 'clean' in i]

    # Import bounds .csv
    df_bounds = pd.read_csv(exp_data_path + '/bounds.csv')
    bound_files = df_bounds.File.values

    # Loop through all experimental datasets
    for i, exp_file in tqdm(enumerate(exp_data_contents_clean),
                            'Experimental Runs'):

        # Select bounds for this experimental file
        row = df_bounds.loc[df_bounds.File == exp_file]
        t_start = row.Start.values[0]
        t_stop = row.Stop.values[0]

        # Import experimental data and trim to start and stop times
        df_exp_data = pd.read_csv(exp_data_path + '/' + exp_file)
        df_exp_start = df_exp_data.loc[df_exp_data.t >= t_start]
        df_exp_range = df_exp_start.loc[df_exp_start.t <= t_stop]
        df_exp_range.reset_index(inplace=True, drop=True)
        df_exp_range = df_exp_range.drop(['phi_uav', 'theta_uav', 
                                          'psi_uav'],axis=1)
        start_time = df_exp_range.iloc[0, 0]
        df_exp_range.loc[:, 't'] -= start_time

        # Select initial conditions for sim from experimental data
        row_ic = df_exp_range.iloc[0]
        x0_uav = row_ic.values[1:]

        # Concatenate uav ic with gv and load into the sim
        x0 = np.concatenate((x0_uav, x0_gv))
        sim_uav_fall_use.load_x0(x0) # Load into sim

        # Compute the simulation
        try: # Try to compute the simulation

            sim_msg = "amls_exp_%i" % i
            sim_time = \
                timeit.timeit(stmt='sim_uav_fall_use.compute(desc=sim_msg)',
                globals=globals(), number=1)

        except BaseException as e: # If simulation errors out

            result_csv_name = root_name + '_exp_%i_error.csv' % i
            df_results = pd.DataFrame() # Placeholder empty DataFrame

            # Create summary text lines and add to list
            result_txt = root_name + '_exp_%i, error' % i
            txt_list.append(result_txt) # Add result to the summary list
            txt_list.append(repr(e)) # Add error message to the summary list

        else: # If simulation completes sucessfully

            result_csv_name = root_name + '_exp_%i_complete.csv' % i
            df_results = sim_uav_fall_use.state_traj

            # Create dataframe with contact point positions and velocities
            points_B = uav_contact_x500_exp.points_B
            df_sim_points = uf_an.create_contact_data(df_results, points_B)

            # Save contact point positions to .csv
            point_csv_name = root_name + '_exp_%i_contact.csv' % i
            point_full = result_path + '/' + point_csv_name
            df_sim_points.to_csv(point_full, index=False)

            # Create contact point dataframe from experimental data
            df_exp_points = uf_an.create_contact_data(df_exp_range, points_B)
            exp_point_csv_name = root_name + '_exp_%i_exp_contact.csv' % i
            exp_point_full = result_path + '/' + exp_point_csv_name
            df_exp_points.to_csv(exp_point_full, index=False)

            # Add euler angle orientations to the results and experimental dfs
            df_results = uf_an.add_euler(df_results)
            df_exp_range = uf_an.add_euler(df_exp_range)
            
            # Create summary text line and add to list
            result_txt = root_name + \
                '_exp_%i, complete in %0.3fs' % (i, sim_time)
            txt_list.append(result_txt) 

        # Save results as .csv
        result_full = result_path + '/' + result_csv_name
        df_results.to_csv(result_full, index=False)

        # Save experimental data used to dataframe and .csv
        exp_csv_name = root_name + '_exp_%i_experimental.csv' % i
        exp_full = result_path + '/' + exp_csv_name
        df_exp_range.to_csv(exp_full, index=False)

    # Save the summary list to a .txt file
    txt_name = root_name + '_summary.txt'
    txt_full_path = result_path + '/' + txt_name

    with open(txt_full_path, 'w') as fp:
        fp.write('\n'.join(txt_list))

    # ---------------------------- Visualization ----------------------------- #

    cmd_dict_use = plot_cmd_dict_save

    sim_plot = sim_plotting.SimPlotting()
    sim_plot.speed_scale = 0.5
    sim_plot.plot_main(result_path, command_dict=cmd_dict_use)





















