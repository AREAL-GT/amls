
'''
This module runs autonomous UAV flights with the focus on checking and 
quantifying the effects of drag on the quadcopter
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
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)

# Workspace package imports
import amls.analysis.utility_functions as uf_an
from amls.visualization import sim_plotting
from amls.dynamics.quadcopter_params import iris_params

# Configured simulation object imports
from amls.analysis.configured_sims import sim_uav_drag
sim_uav_drag.load_tspan((0, 1.0), timestep=0.0005)

# ------------ Import Simulation IC Parameters from External Files ----------- #

param_path = os.path.dirname(__file__) + '/Simulation Parameters/'

# Import uav initial conditions
uav_name = 'uav_ic_drag_lin_vel1'
uav_csv_path = param_path + uav_name + '.csv'
df_uav_ic_fall_set1 = pd.read_csv(uav_csv_path)
uav_ic_vec = df_uav_ic_fall_set1.iloc[0, :].to_numpy()

# --------------------------- Perform Simulation(s) -------------------------- #

# Call function to setup the results directory
batch_name = 'uav_drag_lin_vel'
gv_name = 'Not Used'
root_name, txt_list = uf_an.result_dir_setup(batch_name, gv_name, uav_name)
result_path = os.path.dirname(__file__) + '/Simulation Results/' + root_name

# Reset the simulation initial conditions and control
x0 = uav_ic_vec
sim_uav_drag.load_x0(x0) 
sim_uav_drag.reset_ctl()
sim_uav_drag.quat_cmd = np.array([1, 0, 0, 0]) # Controller qw first
sim_uav_drag.Th_cmd = iris_params.m*9.8

try: # Try to compute the simulation

    sim_msg = "amls drag vel = %i" % (x0[7])
    sim_time = timeit.timeit(stmt='sim_uav_drag.compute(desc=sim_msg)',
                             globals=globals(), number=1)

except BaseException as e: # If simulation errors out

    result_csv_name = root_name + '_vel%i_error.csv' % (x0[7])
    results_df = pd.DataFrame() # Placeholder empty DataFrame

    # Create summary text lines and add to list
    result_txt = root_name + '_vel%i_error.csv' % (x0[7])
    txt_list.append(result_txt) # Add result to the summary list
    txt_list.append(repr(e)) # Add error message to the summary list

else: # If simulation completes sucessfully

    result_csv_name = root_name + '_vel%i_complete.csv' % (x0[7])
    results_df = sim_uav_drag.state_traj

    # Add euler angle orientations to the results df
    results_df = uf_an.add_euler(results_df)

    # Save setpoint dataframe to .csv
    length_res = results_df.shape[0]
    setpoint_df = sim_uav_drag.setpoint_df.iloc[:length_res]
    setpoint_df = uf_an.add_euler_setpoint(setpoint_df)
    setpoint_csv_name = root_name + '_vel%i_setpoint.csv' % (x0[7])
    setpoint_full = result_path + '/' + setpoint_csv_name
    setpoint_df.to_csv(setpoint_full, index=False)

    # Save control input dataframe to .csv
    control_df = sim_uav_drag.ctl_df
    control_csv_name = root_name + '_vel%i_control.csv' % (x0[7])
    control_full = result_path + '/' + control_csv_name
    control_df.to_csv(control_full, index=False)

    # Create summary text line and add to list
    result_txt = root_name + '_vel%i, complete in %0.3fs' % \
        (x0[7], sim_time)
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

plot_command_dict_save = { # 'all', 'plot', 'save', 'none'
    'position': 'save',
    'lin_velocity': 'save',
    'quaternion': 'save',
    'euler': 'save',
    'ang_velocity': 'save',
    'cont_position': 'save',
    'cont_velocity': 'save',
    'ctl_torques': 'save',
    'ctl_rotors': 'save',
    'animation': 'none',
    'gv_plot': 'none', # 'full_state' or 'pos_states' for gv states
    'setpoint_plot': True,
}

plot_command_dict_plot = { # 'all', 'plot', 'save', 'none'
    'position': 'plot',
    'lin_velocity': 'plot',
    'quaternion': 'plot',
    'euler': 'plot',
    'ang_velocity': 'plot',
    'cont_position': 'plot',
    'cont_velocity': 'plot',
    'ctl_torques': 'plot',
    'ctl_rotors': 'plot',
    'animation': 'none',
    'gv_plot': 'none', # 'full_state' or 'pos_states' for gv states
    'setpoint_plot': True,
}

plot_command_dict_none = { # 'all', 'plot', 'save', 'none'
    'position': 'none',
    'lin_velocity': 'none',
    'quaternion': 'none',
    'euler': 'none',
    'ang_velocity': 'none',
    'cont_position': 'none',
    'cont_velocity': 'none',
    'ctl_torques': 'none',
    'ctl_rotors': 'none',
    'animation': 'none',
    'gv_plot': 'none', # 'full_state' or 'pos_states' for gv states
    'setpoint_plot': True,
}

plot_command_dict_use = plot_command_dict_plot

sim_plot = sim_plotting.SimPlotting()
sim_plot.speed_scale = 1.0

sim_plot.plot_main(result_path, plot_command_dict_use)


















