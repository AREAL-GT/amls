
'''
This module takes an exisiting set of results data and generates the desired
plots and animations for it
'''

# --------------------------------- Imports ---------------------------------- #

# Standard imports
import os
import sys

# Add package and workspace directories to the path
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)

from amls.visualization import sim_plotting
from amls.visualization.sim_plotting import plot_cmd_dict_plot, \
    plot_cmd_dict_save

# ---------------------------- Setup and Plotting ---------------------------- #

root_name = '2023_10_09 - 5m_s 10m land'
# dir_path = '/Simulation Results/'
dir_path = '/Simulation Results/Keep Long Term/2023_10_09 - ' + \
    'Final Geometric Landing Data/5m_s/'
result_path = os.path.dirname(__file__) + dir_path + root_name

cmd_dict_use = plot_cmd_dict_save

# cmd_dict_use['gv_plot'] = 'none'
# cmd_dict_use['animation'] = 'none'

sim_plot = sim_plotting.SimPlotting()
sim_plot.speed_scale = 1.0
sim_plot.plot_main(result_path, command_dict=cmd_dict_use)








