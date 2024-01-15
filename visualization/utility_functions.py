
'''
This module contains useful functions for the amls.visualization package
'''

# ---------------------------------- Imports --------------------------------- #

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

# Workspace imports
from amls.visualization.sim_plotting import plot_cmd_dict_plot, \
    plot_cmd_dict_save
import amls.visualization.utility_functions as uf_viz

# --------------------------------- Functions -------------------------------- #

# Default parameters for plot elements
default_param_dict = { 
    'fig_scale': 1.0,
    'fig_x_size': 19.995,
    'fig_y_size': 9.36,
    'fig_dpi': 96,
    'font_suptitle': 28,
    'font_subtitle': 22,
    'font_axlabel': 20,
    'font_axtick': 18,
    'font_txt': 14,
    'font_legend': 18,
    'ax_lw': 2,
    'plt_lw': 3,
}

def pitch_vel_plot(dir_path: str, param_dict: dict = default_param_dict, 
                   plt_cmd: str = 'save') -> None:
    '''
    This function takes in a directory, goes through all of the contained
    complete simulation files, and generates a plot of the agregated linear
    UAV velocity vs. pitch angle

    The argument plt_cmd = 'save', 'plot', 'all'
    '''

    # Plot save/display control parameters
    plt_save = ('all', 'save')
    plt_show = ('all', 'plot')

    # Get parameter values from input dictionary
    font_suptitle = param_dict['font_suptitle']
    font_axlabel = param_dict['font_axlabel']
    fig_x_size = param_dict['fig_x_size']
    fig_y_size = param_dict['fig_y_size']
    fig_scale= param_dict['fig_scale']
    fig_dpi = param_dict['fig_dpi']

    # Get all file names for completed simulations in directory
    dir_conts = os.listdir(dir_path)
    result_list = [i for i in dir_conts if ('complete' in i or 'success' in i 
                                            or 'fail' in i)]
    
    # Initialize results vectors
    vel_vec = np.array([])
    ang_vec = np.array([])

    # Loop through all completed simulations and compile results
    for result in result_list:

        # Import trajectory .csv as dataframe
        result_df = pd.read_csv(dir_path + '/' + result)

        # Append to results vectors
        vel_vec = np.append(vel_vec, result_df.dx_uav.iloc[-1])
        ang_vec = np.append(ang_vec, result_df.theta_uav.iloc[-1])

    # Plot the pitch angles vs. linear velocities
    fig, ax = plt.subplots(figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
                           dpi=fig_dpi)
    fig.suptitle('UAV Pitch Angle vs. Linear Velocity', fontsize=font_suptitle,
                 fontweight='bold')
    ax.scatter(vel_vec, ang_vec)

    # Plot formatting
    ax.set_xlabel('X Linear Velocity (m/s)', fontsize=font_axlabel,
                  fontweight='bold')
    ax.set_ylabel('Pitch Angle (deg)', fontsize=font_axlabel, fontweight='bold')
    ax.grid()
    
    fig.tight_layout()

    if plt_cmd in plt_save:
        fig.savefig(dir_path + '/vel_ang.png')

    if plt_cmd not in plt_show:
        plt.close(fig)

    if plt_cmd in plt_show:
        plt.show()

def plot_trim(data_dict_tup: tuple, fig_dict: dict):
    '''
    This function takes in the parameters needed to plot a timeseries but 
    makes a few changes. The save directory is used to save trimmed .csv
    data and not plots, and the plot is always shown regardless of the 
    command.
    '''

    # Isolate save directory and set plot to always show
    save_ctl = False
    fig_keys = fig_dict.keys()
    if 'save_dir' in fig_keys:
        if not fig_dict['save_dir'] == None:
            save_dir = fig_dict['save_dir']
            fig_dict['save_dir'] = None
            save_ctl = True
    
    fig_dict['show_plt'] = True

    # Call plotting function
    uf_viz.plot_timeseries(data_dict_tup, fig_dict)

    # Save trimmed .csv if a directory is defined
    if save_ctl:
        
        df_data = data_dict_tup[0]['dataframe']
        bounds = fig_dict['bounds'][0]
        
        # Trim dataframe to the desired bounds
        df_trim = df_data.loc[(df_data.t > bounds[0]) & (df_data.t < bounds[1])]
        df_trim.to_csv(save_dir, index=False)

def plot_timeseries(data_dict_tup: tuple, fig_dict: dict, 
                    param_dict: dict = default_param_dict) -> None:
    '''
    This function is used to plot timeseries data from dataframes. This 
    data can come from any column(s) of any dataframe, and be plotted on the
    primary or secondary axes of subplots in any arrangement
    '''

    #  Break out plotting parameters from dict into variables
    fig_scale = default_param_dict['fig_scale']
    fig_x_size = default_param_dict['fig_x_size']
    fig_y_size = default_param_dict['fig_y_size']
    fig_dpi = default_param_dict['fig_dpi']
    font_suptitle = default_param_dict['font_suptitle']
    font_subtitle = default_param_dict['font_subtitle']
    font_axlabel = default_param_dict['font_axlabel']
    font_legend = default_param_dict['font_legend']
    plt_lw = default_param_dict['plt_lw']

    # Break out figure setup parameters from dict into variables
    sub_arrangement = fig_dict['subplot']
    sub_count = sub_arrangement[0]*sub_arrangement[1]
    title = fig_dict['title']

    fig_keys = fig_dict.keys()

    # Check for sharex command in dict
    sharex = True
    if 'sharex' in fig_keys:
        sharex = fig_dict['sharex']

    # Create the figure and subplots
    fig, ax_prime_list = plt.subplots(sub_arrangement[0], sub_arrangement[1], 
        figsize=(fig_scale*fig_x_size, fig_scale*fig_y_size), 
        dpi=fig_dpi, sharex=sharex)
    
    # Add overall title
    fig.suptitle(title, fontsize=font_suptitle, fontweight='bold')

    # Determine which subplots need secondary axes and add them
    ax_sec_list = ['']*sub_count
    for data_dict in data_dict_tup:

        axis_idx = data_dict['axis']
        sub_idx = data_dict['subplot']
        if axis_idx == 1: # If secondary axis used add to the list
            ax_sec_list[sub_idx] = ax_prime_list[sub_idx].twinx()
    
    # Initialize list of empty lists for each subplot legend
    legend_lists = [[] for _ in range(sub_count)]

    # Loop through the input tuple of data dicts and add to plots
    for i, data_dict in enumerate(data_dict_tup):

        # Isolate important parameters this data dict
        data_keys = data_dict.keys()
        sub_idx = data_dict['subplot']
        plt_axis = data_dict['axis']
        df_data = data_dict['dataframe']

        # Determine if this data is in the legend
        legend_add = False
        if 'legend' in data_keys:
            legend_add = not data_dict['legend'] == None

        # Loop through each desired column and add to subplot
        for j, col in enumerate(data_dict['columns']):

            if legend_add: # If this data in the legend

                legend_txt = data_dict['legend'][j]

                if plt_axis == 0: # If data is on the primary axis

                    plot = ax_prime_list[sub_idx].plot(df_data.t, 
                        df_data.loc[:, col], linewidth=plt_lw, 
                        label=legend_txt)
                    legend_lists[sub_idx].append(plot[0])

                else: # If data is on the secondary axis
                    
                    color = 'C%i' % (j + 1)
                    plot = ax_sec_list[sub_idx].plot(df_data.t, 
                        df_data.loc[:, col], color, linewidth=plt_lw, 
                        label=legend_txt)
                    legend_lists[sub_idx].append(plot[0])

            else: # If this data is not in the legend
                if plt_axis == 0: # If data is on the primary axis

                    ax_prime_list[sub_idx].plot(df_data.t, df_data.loc[:, col], 
                                                linewidth=plt_lw)

                else: # If data is on the secondary axis

                    ax_sec_list[sub_idx].plot(df_data.t, df_data.loc[:, col], 
                                              linewidth=plt_lw)

    # Loop through each primary axis and add formatting
    for i, ax in enumerate(ax_prime_list):
        
        # Default formatting
        subtitle = 'Data ' + str(i)
        ax.set_title(subtitle, fontsize=font_subtitle, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=font_axlabel, fontweight='bold')
        ax.set_ylabel('Data', fontsize=font_axlabel, fontweight='bold')
        ax.grid()

        # User-defined x label
        if 'xlabel' in fig_keys:
            if not fig_dict['xlabel'][i] == None:
                ax.set_xlabel(fig_dict['xlabel'][i], fontsize=font_axlabel, 
                              fontweight='bold')

        # User-defined y label
        if 'ylabel' in fig_keys:
            if not fig_dict['ylabel'][i] == None:
                ylabel = fig_dict['ylabel'][i]
                if type(ylabel) == str:
                    ax.set_ylabel(ylabel, fontsize=font_axlabel, 
                                fontweight='bold')
                elif type(ylabel) == tuple:
                    ax.set_ylabel(ylabel[0], fontsize=font_axlabel, 
                                  fontweight='bold')
                    ax_sec = ax_sec_list[i]
                    ax_sec.set_ylabel(ylabel[1], fontsize=font_axlabel, 
                                      fontweight='bold')

        # Used-defined subtitle
        if 'subtitle' in fig_keys:
            if not fig_dict['subtitle'][i] == None:
                ax.set_title(fig_dict['subtitle'][i], fontsize=font_axlabel, 
                              fontweight='bold')

        # User-defined vertical bound markers
        if 'bounds' in fig_keys:
            if not fig_dict['bounds'][i] == None:
                bounds = fig_dict['bounds'][i]
                ax.axvline(bounds[0], color='r')
                ax.axvline(bounds[1], color='r')

        # Add legend if labels exist
        legend_data = legend_lists[i]
        if legend_data: # If there is legend data in the list
            ax.legend(handles=legend_data, fontsize=font_legend)

    fig.tight_layout()

    # Save plot
    if 'save_dir' in fig_keys:
        if not fig_dict['save_dir'] == None:
            fig.savefig(fig_dict['save_dir'])

    # Show plot
    if 'save_dir' in fig_keys:
        if not fig_dict['show_plt'] == True:
            plt.close(fig)





















