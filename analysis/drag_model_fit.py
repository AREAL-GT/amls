
'''
This module is used to fit and test the drag coeffecients to the uav model 
using manual flight test data gathered in experiments.
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
import numpy.linalg as LA
import pandas as pd
from math import pi

from sklearn.model_selection import train_test_split

from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares

# Workspace package imports
from amls.visualization import utility_functions as uf_viz
from amls.analysis import utility_functions as uf_an
from amls.dynamics.quadcopter_params import x500_exp_params

# ---------------------------- Residual Function ----------------------------- #

def uav_dyn_residual(beta, x_obs, y_obs):
    '''
    Contains the residual function from the UAV dynamic model used to fit
    drag coefficients.
    '''

    # Defined needed constants
    g = 9.81 # Gravitational acceleration (m/s^2)
    # Tmax = 1.332*g # Maximum thrust force x500v2 datasheet (N)
    # w_max = 9857*0.10472 # Max X500v2 motor ang velocity (rad/sec)
    r = x500_exp_params.r # Blade radius of propeller (m)
    m = x500_exp_params.m # Mass of UAV (kg)

    # Initialize vector of residuals
    residual = np.zeros(x_obs.shape[0])

    # Loop through all rows of data and compute each residual
    for i in range(x_obs.shape[0]):

        x_row = x_obs[i]
        y_row = y_obs[i]

        # Make state variables and vectors for readability
        dn_dt = x_row[0]
        de_dt = x_row[1]
        dd_dt = x_row[2]
        qx = x_row[3]
        qy = x_row[4]
        qz = x_row[5]
        qw = x_row[6]
        w1 = x_row[7]
        w2 = x_row[8]
        w3 = x_row[9]
        w4 = x_row[10]

        v_NED = np.array([dn_dt, de_dt, dd_dt])
        q_NED = np.array([qx, qy, qz, qw])
        w_vec = np.array([w1, w2, w3, w4])
        w_avg = w_vec.mean() # Linear command of thrust percentage 0-1
        R_Q_NED = Rot.from_quat(q_NED)
        v_Q = R_Q_NED.apply(v_NED, inverse=True)

        # Use models to compute thrust and rotor velocity
        T = 16.755*w_avg - 3.608
        T = 4*max(T, 0)

        u = 1196.148*w_avg**2 - 225.314*w_avg + 73.296
        u = max(u, 0)

        # Define parameters being estimated from beta vector
        # A1c = beta[0]
        # A1s = beta[1]
        # dx = beta[2]
        # dy = beta[3]

        A1c = beta[0]
        # A1s = beta[1]
        dx = beta[1]
        dy = beta[2]

        # Setup the needed matrices for drag dynamics
        # A_flap = (1/(u*r))*np.array([[A1c, -A1s, 0], [A1s, A1c, 0], 
        #                             [0, 0, 0]])
        A_flap = (1/(u*r))*np.array([[A1c, 0, 0], [0, A1c, 0], 
                                    [0, 0, 0]])
        d = np.array([[dx, 0, 0], [0, dy, 0], [0, 0, 0]])
        D = A_flap + d

        # Compute the total forces on the UAV
        F_T_Q = np.array([0, 0, -T])
        F_drag_Q = T*D@v_Q
        F_UAV_Q = F_T_Q - F_drag_Q
        F_UAV_NED = R_Q_NED.apply(F_UAV_Q)
        F_TOT_NED = F_UAV_NED + np.array([0, 0, m*g])

        # Subtract from the observed mass*acceleration to form a residual vector
        ma_obs = m*y_row
        delta = ma_obs - F_TOT_NED
        residual[i] = np.sum(np.abs(delta))

    print(np.mean(residual)/(3*m))
    return residual

# ---------------------------- Initial Data Clean ---------------------------- #

init_clean = False

if init_clean:

    dir_path = os.path.dirname(__file__) + '/Experimental Comparison Data/' + \
        '09_07_2023 - UAV Velocity Flights'
    uf_an.px4_dir_clean(dir_path)

# --------------------------- Full Timeseries Plots -------------------------- #

timeseries_plots = False

if timeseries_plots:

    # Set plotting directory options
    data_dir_8_29 = os.path.dirname(__file__) + '/Experimental Comparison ' + \
        'Data/08_29_2023 - UAV Velocity Flights/'
    data_dir_9_7 = os.path.dirname(__file__) + '/Experimental Comparison ' + \
        'Data/09_07_2023 - UAV Velocity Flights/'
    
    # Manual file plotting listing
    raw_data_files_8_29 = [
        'cleanlog_0_2023-8-29-17-04-20.csv',
        'cleanlog_1_2023-8-29-17-17-50.csv',
        'cleanlog_2_2023-8-29-17-35-26.csv',
        'cleanlog_3_2023-8-29-17-45-44.csv',
        'cleanlog_4_2023-8-29-17-57-06.csv',
        'cleanlog_5_2023-8-29-18-01-22.csv']
    raw_data_files_9_07 = [
        'cleanlog_0_2023-9-7-18-14-30.csv',
        'cleanlog_1_2023-9-7-18-16-16.csv',
        'cleanlog_2_2023-9-7-18-27-54.csv',
        'cleanlog_3_2023-9-7-18-29-48.csv']
    
    dir_use = data_dir_8_29 # Choose directory option

    # Full directory file listing
    raw_data_all = [file for file in os.listdir(dir_use) if 'cleanlog' in file]

    file_list_use = raw_data_all # Choose file list option
    raw_data_full = [dir_use + file for file in file_list_use]

    # Loop through each desired file for trimming and processing
    for i, data_full in enumerate(raw_data_full):

        df_clean = pd.read_csv(data_full)

        df_clean.phi *= (180/pi)
        df_clean.theta *= (180/pi)
        df_clean.psi *= (180/pi)
        df_clean.phi_rot *= (180/pi)
        df_clean.theta_rot *= (180/pi)
        df_clean.psi_rot *= (180/pi)
        df_clean.alpha *= (180/pi)

        plot_dir = dir_use + '/Visualizations'
        vel_ang_name = file_list_use[i][:-4] + '_vel_ang.png'
        plt_full = plot_dir + '/' + vel_ang_name

        # Plot pitch and roll angles alongside x and y velocities
        plot_cols1 = (('vx', 'vy'),)
        plot_cols2 = (('theta_rot', 'phi_rot'),)
        title = file_list_use[i][:-4] + " - Velocity and Euler Angles"
        subtitle = ('X Axis', 'Y Axis')
        y_label = ('Velocity (m/s)', 'Velocity (m/s)', 'Angle (deg)', 
                    'Angle (deg)')
        legend = ((['x velocity'], ['pitch angle']), 
                  (['y velocity'], ['roll angle']))
        save_dir = plt_full
        uf_viz.plot_2ax((df_clean,), plot_cols1, title, results2=(df_clean,),
            y_label=y_label, plot_cols2=plot_cols2, subtitle=subtitle, 
            legend=legend, save_dir=None, plot=False)
        
        # Plot pitch and roll angles alongside x and y velocities
        vel_ang_name = file_list_use[i][:-4] + '_vel_ang.png'
        plt_full = plot_dir + '/' + vel_ang_name
        fig_dict_vel_ang = {'subplot': (2,1), 
                            'title': file_list_use[i][:-4] + \
                                " - Velocity and Euler Angles",
                            'xlabel': (None, None),
                            'ylabel': (('Velocity (m/s)', 'Angle (deg)'), 
                                       ('Velocity (m/s)', 'Angle (deg)')),
                             'subtitle': ('X Axis', 'Y Axis'),
                             'bounds': (None, None),
                             'show_plt': False,
                             'save_dir': plt_full}
        data_vx = {'dataframe': df_clean, 
                   'columns': ('vx',), 
                   'subplot': 0,
                   'axis': 0,
                   'legend': ('x velocity',)}
        data_vy = {'dataframe': df_clean, 
                   'columns': ('vy',), 
                   'subplot': 1,
                   'axis': 0,
                   'legend': ('y velocity',)}
        data_theta = {'dataframe': df_clean, 
                      'columns': ('theta_rot',), 
                      'subplot': 0,
                      'axis': 1,
                      'legend': ('pitch angle',)}
        data_phi = {'dataframe': df_clean, 
                   'columns': ('phi_rot',), 
                   'subplot': 1,
                   'axis': 1,
                   'legend': ('roll angle',)}
        uf_viz.plot_timeseries((data_vx, data_vy, data_theta, data_phi), 
                               fig_dict_vel_ang)
        
        # Plot x/y/z velocities alongside accelerations
        vel_accel_name = file_list_use[i][:-4] + '_vel_accel.png'
        plt_full = plot_dir + '/' + vel_accel_name
        fig_dict_vel_accel = {'subplot': (3,1), 
                              'title': file_list_use[i][:-4] + \
                                 " - Velocity and Acceleration",
                              'xlabel': (None, None, None),
                              'ylabel': (('Velocity (m/s)', 
                                          'Acceleration (m/$s^2$)'), 
                                         ('Velocity (m/s)', 
                                          'Acceleration (m/$s^2$)'),
                                         ('Velocity (m/s)', 
                                          'Acceleration (m/$s^2$)')),
                              'subtitle': ('X Axis', 'Y Axis', 'Z Axis'),
                              'bounds': (None, None, None),
                              'show_plt': False,
                              'save_dir': plt_full}
        data_vx = {'dataframe': df_clean, 
                   'columns': ('vx',), 
                   'subplot': 0,
                   'axis': 0,
                   'legend': ('x velocity',)}
        data_vy = {'dataframe': df_clean, 
                   'columns': ('vy',), 
                   'subplot': 1,
                   'axis': 0,
                   'legend': ('y velocity',)}
        data_vz = {'dataframe': df_clean, 
                   'columns': ('vz',), 
                   'subplot': 2,
                   'axis': 0,
                   'legend': ('z velocity',)}
        data_ax = {'dataframe': df_clean, 
                   'columns': ('ax',), 
                   'subplot': 0,
                   'axis': 1,
                   'legend': ('x acceleration',)}
        data_ay = {'dataframe': df_clean, 
                   'columns': ('ay',), 
                   'subplot': 1,
                   'axis': 1,
                   'legend': ('y acceleration',)}
        data_az = {'dataframe': df_clean, 
                   'columns': ('az',), 
                   'subplot': 2,
                   'axis': 1,
                   'legend': ('z acceleration',)}
        uf_viz.plot_timeseries((data_vx, data_vy, data_vz, data_ax, data_ay, 
                                data_az), 
                               fig_dict_vel_accel)

        # Plot linear velocity alongside alpha angle with throttle as well
        linvel_alpha_name = file_list_use[i][:-4] + '_linvel_alpha.png'
        plt_full = plot_dir + '/' + linvel_alpha_name
        fig_dict_vel_tilt = {'subplot': (2,1), 
                             'title': file_list_use[i][:-4] + \
                                " - Tilt Angle and Velocity with Throttle",
                             'xlabel': (None, None),
                             'ylabel': (('Velocity (m/s)', 'Angle (deg)'), 
                                        'Throttle ($\%$)'),
                             'subtitle': ('Tilt Angle and Velocity', 
                                          'Throttle'),
                             'bounds': (None, None),
                             'show_plt': False,
                             'save_dir': plt_full}
        data_vel = {'dataframe': df_clean, 
                    'columns': ('v_lin',), 
                    'subplot': 0,
                    'axis': 0,
                    'legend': ('lin vel',)}
        data_tilt = {'dataframe': df_clean, 
                     'columns': ('alpha',), 
                     'subplot': 0,
                     'axis': 1,
                     'legend': ('tilt angle',)}
        data_throttle = {'dataframe': df_clean, 
                         'columns': ('w1', 'w2', 'w3', 'w4'), 
                         'subplot': 1,
                         'axis': 0,
                         'legend': ('w1', 'w2', 'w3', 'w4')}
        uf_viz.plot_timeseries((data_vel, data_tilt, data_throttle), 
                               fig_dict_vel_tilt)

    plt.show()

# --------------------------------- Data Trim -------------------------------- #

data_trim = False

if data_trim:

    # Set directory and file name
    data_dir = os.path.dirname(__file__) + '/Experimental Comparison ' + \
        'Data/08_29_2023 - UAV Velocity Flights/'
    filename = 'cleanlog_0_2023-8-29-17-04-20.csv'
    full_input = data_dir + filename
    df_data = pd.read_csv(full_input)

    # Set output file name and directory for trimmed data
    output_dir = os.path.dirname(__file__) + '/Experimental Comparison ' + \
        'Data/Trimmed Drag Sections/'
    output_name = 'trim_2023-8-29-17-04__383-387.csv'
    full_output = output_dir + output_name

    # Setup and call plot and trim function
    fig_dict_vel_tilt = {'subplot': (2,1), 
                         'title': filename[:-4] + \
                            " - Tilt Angle and Velocity with Throttle",
                         'xlabel': (None, None),
                         'ylabel': (('Velocity (m/s)', 'Angle (deg)'), 
                                'Throttle ($\%$)'),
                         'subtitle': ('Tilt Angle and Velocity', 
                                    'Throttle'),
                         'bounds': ((383, 387.4), (383, 387.4)),
                         'show_plt': True,
                         'save_dir': full_output}
    data_vel = {'dataframe': df_data, 
                'columns': ('v_lin',), 
                'subplot': 0,
                'axis': 0,
                'legend': ('lin vel',)}
    data_tilt = {'dataframe': df_data, 
                    'columns': ('alpha',), 
                    'subplot': 0,
                    'axis': 1,
                    'legend': ('tilt angle',)}
    data_throttle = {'dataframe': df_data, 
                        'columns': ('w1', 'w2', 'w3', 'w4'), 
                        'subplot': 1,
                        'axis': 0,
                        'legend': ('w1', 'w2', 'w3', 'w4')}
    uf_viz.plot_trim((data_vel, data_tilt, data_throttle), fig_dict_vel_tilt)

    plt.show()

# ---------------------------- Plot Trimmed Data ----------------------------- #

plot_trim_data = False

if plot_trim_data:

    # Set trimmed data directory
    trim_data_dir = os.path.dirname(__file__) + '/Experimental Comparison ' + \
                                                'Data/Trimmed Drag Sections/'
    trim_files = [file for file in os.listdir(trim_data_dir) if 'trim' in file]
    trim_paths = [trim_data_dir + file for file in trim_files]

    # Loop through all trimmed data to generate plots
    for i, path in enumerate(trim_paths):

        df_data = pd.read_csv(path)

        # Compute the acceleration as a numerical derivative of velocity
        t_vec = df_data.t.values
        vn_vec = df_data.vn.values
        ve_vec = df_data.ve.values
        vd_vec = df_data.vd.values
        t_diff = np.diff(t_vec)
        an_grad_vec = np.gradient(vn_vec, t_vec)
        ae_grad_vec = np.gradient(ve_vec, t_vec)
        ad_grad_vec = np.gradient(vd_vec, t_vec)
        df_data['an_grad'] = an_grad_vec
        df_data['ae_grad'] = ae_grad_vec
        df_data['ad_grad'] = ad_grad_vec

        # Plot the logged and computed accelerations
        accel_comp_name = trim_files[i][:-4] + '_accel_comp.png'
        plt_full = trim_data_dir + '/Visualizations/' + accel_comp_name
        fig_dict_accel_comp = {'subplot': (3,1), 
                               'title': trim_files[i][:-4] + \
                                 " - Acceleratiom Comparison",
                              'xlabel': (None, None, None),
                              'ylabel': ('Acceleration (m/$s^2$)', 
                                         'Acceleration (m/$s^2$)',
                                         'Acceleration (m/$s^2$)'),
                              'subtitle': ('N Axis', 'E Axis', 'D Axis'),
                              'bounds': (None, None, None),
                              'show_plt': False,
                              'save_dir': plt_full}
        data_an = {'dataframe': df_data, 
                   'columns': ('an', 'an_grad'), 
                   'subplot': 0,
                   'axis': 0,
                   'legend': ('an', 'an grad')}
        data_ae = {'dataframe': df_data, 
                   'columns': ('ae', 'ae_grad'), 
                   'subplot': 1,
                   'axis': 0,
                   'legend': ('ae', 'ae grad')}
        data_ad = {'dataframe': df_data, 
                   'columns': ('ad', 'ad_grad'), 
                   'subplot': 2,
                   'axis': 0,
                   'legend': ('ad', 'ad grad')}
        
        uf_viz.plot_timeseries((data_an, data_ae, data_ad), fig_dict_accel_comp)

        plt.show()

# ------------------------------ Fit Drag Model ------------------------------ #

fit_drag = True

if fit_drag:

    # Import desired data for all of the cleaned directory
    trim_data_dir = os.path.dirname(__file__) + '/Experimental Comparison ' + \
                                                'Data/Trimmed Drag Sections/'
    trim_files = [file for file in os.listdir(trim_data_dir) if 'trim' in file]
    trim_paths = [trim_data_dir + file for file in trim_files]

    df_trim_data = pd.read_csv(trim_paths[0])

    # Compute the acceleration as a numerical derivative of velocity
    t_vec = df_trim_data.t.values
    vn_vec = df_trim_data.vn.values
    ve_vec = df_trim_data.ve.values
    vd_vec = df_trim_data.vd.values
    t_diff = np.diff(t_vec)
    an_grad_vec = np.gradient(vn_vec, t_vec)
    ae_grad_vec = np.gradient(ve_vec, t_vec)
    ad_grad_vec = np.gradient(vd_vec, t_vec)
    df_trim_data['an_grad'] = an_grad_vec
    df_trim_data['ae_grad'] = ae_grad_vec
    df_trim_data['ad_grad'] = ad_grad_vec

    if len(trim_paths) > 1:

        for path in trim_paths[1:]:

            df_data = pd.read_csv(path)

            # Compute the acceleration as a numerical derivative of velocity
            t_vec = df_data.t.values
            vn_vec = df_data.vn.values
            ve_vec = df_data.ve.values
            vd_vec = df_data.vd.values
            t_diff = np.diff(t_vec)
            an_grad_vec = np.gradient(vn_vec, t_vec)
            ae_grad_vec = np.gradient(ve_vec, t_vec)
            ad_grad_vec = np.gradient(vd_vec, t_vec)
            df_data['an_grad'] = an_grad_vec
            df_data['ae_grad'] = ae_grad_vec
            df_data['ad_grad'] = ad_grad_vec

            pd.concat([df_trim_data, df_data], ignore_index=True)

    # Randomize the dataset ordering and divide into train and test subsets
    df_trim_data = df_trim_data.sample(frac=1)
    df_train, df_test = train_test_split(df_trim_data, test_size=0.2)

    # Isolate out arrays of the needed values
    X_train = df_train.loc[:, ['vn', 've', 'vd', 'qx', 'qy', 'qz', 'qw', 
                            'w1', 'w2', 'w3', 'w4']].to_numpy(copy=True)
    X_test = df_test.loc[:, ['vn', 've', 'vd', 'qx', 'qy', 'qz', 'qw', 
                            'w1', 'w2', 'w3', 'w4']].to_numpy(copy=True)
    Y_train = df_train.loc[:, ['an', 'ae', 'ad']].to_numpy(copy=True)
    Y_test = df_test.loc[:, ['an_grad', 'ae_grad', 
                             'ad_grad']].to_numpy(copy=True)

    # Define initial parameter guesses and bounds
    beta0 = np.array([1, 1, 1])
    bounds = ((0, 0, 0), np.inf)

    # Run least squares regression
    res_nls_drag = least_squares(uav_dyn_residual, beta0, 
                                 args=(X_train, Y_train), bounds=bounds)
    print(res_nls_drag.x)
    # print(res_nls_drag.fun)

# ------------------------- Drag Model Visualization ------------------------- #















