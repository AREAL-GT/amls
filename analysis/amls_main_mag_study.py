
'''
This module runs a batch of mobile landing simulations

Module focus: UAV magnets effect on rotation when partially in contact
'''

# --------------- Add Package and Workspace Directories to Path -------------- #
import os
import sys
amls_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ws_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, amls_pkg_path)
sys.path.insert(0, ws_path)

# --------------------------------- Imports ---------------------------------- #

# Standard imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import timeit
from math import radians, degrees, sin
from scipy.spatial.transform import Rotation as Rot

# Workspace package imports
from amls_impulse_contact.contact import uav_contact_iris_noz

from amls.visualization import sim_plotting
from amls.visualization.sim_plotting import plot_cmd_dict_plot
from amls.visualization.sim_plotting import plot_cmd_dict_save

from amls.dynamics.quadcopter_params import iris_params

import amls.analysis.utility_functions as uf_an

from amls.analysis.configured_sims import sim_uav_fall_partial_mag
sim_uav_fall_partial_mag.load_tspan((0, 2.0), timestep=0.0005)

sim_uav_fall_use = sim_uav_fall_partial_mag

# -------- Import Simulation Iteration Parameters from External Files -------- #

param_path = os.path.dirname(__file__) + '/Simulation Parameters/'

# Import ground vehicle initial conditions
gv_name = 'gv_ic_fall_set1'
gv_csv_path = param_path + gv_name + '.csv'
df_gv_ic_fall_set1 = pd.read_csv(gv_csv_path)
gv_ic_vec = df_gv_ic_fall_set1.iloc[0, :].to_numpy()

# --------------------------- Perform Simulation(s) -------------------------- #

# Call function to setup the results directory
uav_name = 'Not Used'
batch_name = 'uav_fall_mag_study'
root_name, txt_list = uf_an.result_dir_setup(batch_name, gv_name, uav_name)
result_path = os.path.dirname(__file__)+'/Simulation Results/' + root_name

# Set desired uav minimum and maximum angles and spacing
min_ang = 0
max_ang = 90
ang_space = 10
ang_vec = np.arange(min_ang, max_ang + ang_space, ang_space)

msg = "Simulating angle cases: ang = " + np.array_str(ang_vec)
print(msg)

# Loop through desired list of angles of UAV orientation
for ang_i in tqdm(ang_vec, 'Batch Simulations'):

    # Compute the UAV initial conditions from this angle
    ang_i = radians(ang_i)
    z_pos = (abs(iris_params.D[0,0]))*sin(ang_i) + 0.005
    x0_pos = np.array([0, 0, -z_pos])
    x0_quat = Rot.from_euler('xyz', (ang_i, 0, 0)).as_quat()
    x0_vel = np.zeros(3)
    x0_ang_vel = np.zeros(3)
    x0 = np.concatenate((x0_pos, x0_quat, x0_vel, x0_ang_vel, gv_ic_vec))

    sim_uav_fall_use.load_x0(x0) # Load into sim

    try: # Try to compute the simulation

        sim_msg = "amls ang = %0.1f" % degrees(ang_i)
        sim_time = timeit.timeit(stmt='sim_uav_fall_use.compute(desc=sim_msg)',
                                 globals=globals(), number=1)

    except BaseException as e: # If simulation errors out

        result_csv_name = root_name + '_ang%0.1f_error.csv' % degrees(ang_i)
        results_df = pd.DataFrame() # Placeholder empty DataFrame

        # Create summary text lines and add to list
        result_txt = root_name + '_ang%0.1f, error' % degrees(ang_i)
        txt_list.append(result_txt) # Add result to the summary list
        txt_list.append(repr(e)) # Add error message to the summary list

    else: # If simulation completes sucessfully

        # EVENTUALLY MAKE THIS SUCCESS/FAIL BASED ON LANDING NOT COMPLETE
        result_csv_name = root_name + '_ang%0.1f_complete.csv' % degrees(ang_i)
        results_df = sim_uav_fall_use.state_traj

        # Create dataframe with contact point positions and velocities
        points_B = uav_contact_iris_noz.points_B
        points_df = uf_an.create_contact_data(results_df, points_B)

        # Save contact point positions to .csv
        point_csv_name = root_name + '_ang%0.1f_contact.csv' % degrees(ang_i)
        point_full = result_path + '/' + point_csv_name
        points_df.to_csv(point_full, index=False)

        # Add euler angle orientations to the results df
        results_df = uf_an.add_euler(results_df)
        
        # Create summary text line and add to list
        result_txt = root_name + '_ang%0.1f, complete in %0.3fs' % \
            (degrees(ang_i), sim_time)
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

cmd_dict_use = plot_cmd_dict_save
cmd_dict_use['ctl_torques'] = 'none'
cmd_dict_use['ctl_rotors'] = 'none'
cmd_dict_use['gv_plot'] = 'none'
cmd_dict_use['setpoint_plot'] = False

sim_plot = sim_plotting.SimPlotting()
sim_plot.speed_scale = 1.0
sim_plot.plot_main(result_path, cmd_dict_use)







