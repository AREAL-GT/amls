
import os
import sys
amls_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ws_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, amls_pkg_path)
sys.path.insert(0, ws_path)

from amls.visualization import utility_functions as uf_viz

dir_path = os.path.dirname(__file__) + '/Simulation Results/Keep' + \
    ' Long Term/2023_08_18 - Velocity ramp up trim tests 0-20 m_s/' + \
    '2023_08_18_uav_vel_ramp_0'

uf_viz.pitch_vel_plot(dir_path, plt_cmd='save')




