
'''
This module runs a batch of mobile landing simulations.

Module focus: Range of ground vehicle linear velocities and radii of curvature

'''

# --------------------------------- Imports ---------------------------------- #

# Standard imports
import pandas as pd
import os
import sys
import numpy as np
from tqdm import tqdm
import timeit

# Add package and workspace directories to the path
# sim_path_curr = os.path.abspath(os.path.dirname(__file__))
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)

# Workspace package imports
import amls.analysis.utility_functions as uf_an
from amls_impulse_contact.contact import uav_contact_x500_exp
from amls.visualization import sim_plotting
from amls.visualization.sim_plotting import plot_cmd_dict_plot, \
    plot_cmd_dict_all, plot_cmd_dict_save, plot_cmd_dict_none

# Configured simulation object imports
from amls.analysis.configured_sims import sim_uav_auton
sim_uav_auton.load_tspan((0, 30.0), timestep=0.001)

# -------- Import Simulation Iteration Parameters from External Files -------- #

param_path = os.path.dirname(__file__) + '/Simulation Parameters/'

# Import uav initial conditions
uav_name = 'uav_ic_auton_set1'
uav_csv_path = param_path + uav_name + '.csv'
df_uav_ic_fall_set1 = pd.read_csv(uav_csv_path)
 
# Import ground vehicle initial conditions
gv_name = 'gv_ic_auton_set2'
gv_csv_path = param_path + gv_name + '.csv'
df_gv_ic_fall_set1 = pd.read_csv(gv_csv_path)

# --------------------------- Perform Simulation(s) -------------------------- #

# Call function to setup the results directory
batch_name = 'uav_auton_geo'
root_name, txt_list = uf_an.result_dir_setup(batch_name, gv_name, uav_name)
result_path = os.path.dirname(__file__) + '/Simulation Results/' + root_name

# Create combined index list for iteration
gv_idx_list = range(df_gv_ic_fall_set1.shape[0])
uav_idx_list = range(df_uav_ic_fall_set1.shape[0])
combined_idx_list = [(gv, uav) for gv in gv_idx_list for uav in uav_idx_list]

# Loop through all combinations and run simulation with ICs
for indexes in tqdm(combined_idx_list, 'Batch Simulations'): 

    # Select the two initial condition rows
    gv_idx = indexes[0]
    uav_idx = indexes[1]
    gv_vec = df_gv_ic_fall_set1.iloc[gv_idx, :].to_numpy()
    uav_vec = df_uav_ic_fall_set1.iloc[uav_idx, :].to_numpy()

    # Select rows of initial condition matrices as x0 this combination
    uav_ic_vec = uav_vec
    if len(gv_vec) == 13:
        gv_ic_vec = gv_vec
    else: # If extra columns included set commanded acceleration 
        gv_ic_vec = gv_vec[:13]
        sim_uav_auton.a_cmd_gv = gv_vec[13:19] # Set gv accel commands
        sim_uav_auton.vgoal_gv = gv_vec[19:25] # gv target velocities
        sim_uav_auton.rad_des = gv_vec[25] # Radius in m, or None for straight

    x0 = np.concatenate((uav_ic_vec, gv_ic_vec))

    sim_uav_auton.load_x0(x0) # Load into sim
    sim_uav_auton.reset_ctl()

    # sim_msg = "amls_g%i_u%i" % (gv_idx, uav_idx)
    # sim_time = timeit.timeit(stmt='sim_uav_auton.compute(desc=sim_msg)',
    #                             globals=globals(), number=1)

    try: # Try to compute the simulation

        sim_msg = "amls_g%i_u%i" % (gv_idx, uav_idx)
        sim_time = timeit.timeit(stmt='sim_uav_auton.compute(desc=sim_msg)',
                                 globals=globals(), number=1)

    except BaseException as e: # If simulation errors out

        result_csv_name = root_name + '_g%i_u%i_error.csv' % (gv_idx, uav_idx)
        results_df = pd.DataFrame() # Placeholder empty DataFrame

        # Create summary text lines and add to list
        result_txt = root_name + '_g%i_u%i, error' % (gv_idx, uav_idx)
        txt_list.append(result_txt) # Add result to the summary list
        txt_list.append(repr(e)) # Add error message to the summary list

    else: # If simulation completes sucessfully

        # EVENTUALLY MAKE THIS SUCCESS/FAIL BASED ON LANDING NOT COMPLETE
        result_csv_name = root_name + '_g%i_u%i_complete.csv' %(gv_idx,uav_idx)
        results_df = sim_uav_auton.state_traj

        # Create dataframe with contact point positions and velocities
        points_B = uav_contact_x500_exp.points_B
        points_df = uf_an.create_contact_data(results_df, points_B)

        # Save contact point positions to .csv
        point_csv_name = root_name + '_g%i_u%i_contact.csv' %(gv_idx, uav_idx)
        point_full = result_path + '/' + point_csv_name
        points_df.to_csv(point_full, index=False)

        # Add euler angle orientations to the results df
        results_df = uf_an.add_euler(results_df)

        # Save setpoint dataframe to .csv
        length_res = results_df.shape[0]
        setpoint_df = sim_uav_auton.setpoint_df.iloc[:length_res]
        setpoint_df = uf_an.add_euler_setpoint(setpoint_df)
        setpoint_csv_name = root_name + '_g%i_u%i_setpoint.csv' % \
            (gv_idx, uav_idx)
        setpoint_full = result_path + '/' + setpoint_csv_name
        setpoint_df.to_csv(setpoint_full, index=False)

        # Save control input dataframe to .csv
        control_df = sim_uav_auton.ctl_df
        control_csv_name = root_name + '_g%i_u%i_control.csv' % \
            (gv_idx, uav_idx)
        control_full = result_path + '/' + control_csv_name
        control_df.to_csv(control_full, index=False)

        # Create summary text line and add to list
        result_txt = root_name + '_g%i_u%i, complete in %0.3fs' % (gv_idx,
            uav_idx, sim_time)
        txt_list.append(result_txt) 

    # Save results as .csv
    result_full = result_path + '/' + result_csv_name
    results_df.to_csv(result_full, index=False)

# Save the summary list to a .txt file
txt_name = root_name + '_summary.txt'
txt_full_path = result_path + '/' + txt_name

with open(txt_full_path, 'w') as fp:
    fp.write('\n'.join(txt_list))

# ------------------------------ Visualization ------------------------------- #

cmd_dict_use = plot_cmd_dict_none
cmd_dict_use['position'] = 'save'
cmd_dict_use['cont_position'] = 'save'
sim_plot = sim_plotting.SimPlotting()
sim_plot.speed_scale = 1.0
sim_plot.plot_main(result_path, command_dict=cmd_dict_use)












