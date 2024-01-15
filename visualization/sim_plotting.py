
'''
This module contains the class used to organize plotting for the amls 
'''

# ---------------------------------- Imports --------------------------------- #

# Standard imports
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = ["Latin Modern Roman"]

from matplotlib.patches import Polygon
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation

from math import pi
import os

import numpy as np

from scipy.spatial.transform import Rotation as Rot

from copy import deepcopy

# Add package and workspace directories to the path
import os
import sys
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)

# Workspace package imports
from amls.dynamics.quadcopter_params import iris_params
from amls.dynamics.ground_vehicle import ground_vehicle_1

# ---------------------------- Default Arguments ----------------------------- #

# Default parameters for plot elements
default_param_dict = { 
    'fig_scale': 1.0,
    'fig_x_size': 19.995,
    'fig_y_size': 9.36,
    'fig_dpi': 96,
    'ani_dpi': 200,
    'font_suptitle': 28,
    'font_subtitle': 22,
    'font_axlabel': 20,
    'font_axtick': 16,
    'font_txt': 14,
    'font_legend': 20,
    'ax_lw': 2,
    'plt_lw': 3,
    'setpoint_lw': 1,
    'video_freq': 120
}

# Default commands for different plots: 'none', 'plot', 'save', 'all' 
default_command_dict = { 
    'position': 'all',
    'lin_velocity': 'all',
    'quaternion': 'all',
    'euler': 'all',
    'ang_velocity': 'all',
    'cont_position': 'all',
    'cont_velocity': 'all',
    'ctl_torques': 'all',
    'ctl_rotors': 'all',
    'animation': 'all',
    'gv_plot': 'full_state', # 'full_state' or 'pos_states' for gv states
    'setpoint_plot': True # True/False plot controller setpoints
}

default_contact_size = np.array([1, 1]) # Default contact zone size
default_uav_D = iris_params.D # Default uav prop positions

# ----------------------------- Class Definition ----------------------------- #

class SimPlotting():
    '''
    Class is used to organize the plotting and visualization for the amls data
    '''

    # Insert class-wide variables here


    def __init__(self, param_dict: dict = default_param_dict, 
                 contact_size=default_contact_size, 
                 uav_D = default_uav_D) -> None:
        '''
        Constructor method
        '''

        # Assign sizing inputs
        self.fig_scale = param_dict['fig_scale']
        self.fig_x_size = param_dict['fig_x_size']
        self.fig_y_size = param_dict['fig_y_size']
        self.fig_dpi = param_dict['fig_dpi']
        self.font_suptitle = param_dict['font_suptitle']
        self.font_subtitle = param_dict['font_subtitle']
        self.font_axlabel = param_dict['font_axlabel']
        self.font_axtick = param_dict['font_axtick']
        self.font_txt = param_dict['font_txt']
        self.font_legend = param_dict['font_legend']
        self.ax_lw = param_dict['ax_lw']
        self.plt_lw = param_dict['plt_lw']
        self.setpoint_lw = param_dict['setpoint_lw']

        # Set zorder for the animation
        self.uav_zorder = 10
        self.patch_zorder = 1
        self.axes_zorder = 5
        self.txt_zorder = 6

        # Set animation parameters
        self.ani_dpi = param_dict['ani_dpi']
        self.video_freq = param_dict['video_freq']
        self.axes_mag = 0.5
        self.contact_size = contact_size
        self.uav_D = uav_D
        self.speed_scale = 1.0

    def plot_main(self, results_direc: str, exp_data_dir: str = None,
                  command_dict: dict = default_command_dict):
        '''
        Outer level plotting method that searches a given directory, generating
        plots for the desired results .csv files
        '''

        self.any_plot_show = False # Init variable to no plots commanded shown

        # Get list of files in directory
        results_direc_conts = os.listdir(results_direc)
        result_list = [i for i in results_direc_conts if \
                       ('complete' in i or 'success' in i or 'fail' in i)]
        
        # Check if any plots are commanded to be saved
        plt_commands = command_dict.values() # All commands in dict
        plt_commands = [i for i in plt_commands if type(i)==str]
        command_check = any([i for i in plt_commands if 
                             ('save' in i or 'all' in i)])
        
        # Loop through all files in directory
        for result in result_list:

            # Initialize path for plot directory, assuming no existing folder
            #   Even if no plots saved, so direc path can be passed to the 
            #   plotting methods
            end_idx = result.rfind('_')
            result_name = result[:end_idx]
            plt_direc = result_name + '_visualizations_0'
            plot_direc_path = results_direc + '/' + plt_direc

            if command_check: # If any plots commanded to be saved

                # If a plot folder exists, increment a trailing index
                if plt_direc in results_direc_conts:
                    
                    plt_direc_list = [i for i in results_direc_conts if \
                                      (plt_direc[:-2] in i)]
                    
                    plt_direc_idx = [eval(direc[-1]) for direc in 
                                     plt_direc_list]
                    max_idx = max(plt_direc_idx)

                    plt_direc = result_name + '_visualizations_%i' % \
                        (max_idx + 1)

                plot_direc_path = results_direc + '/' + plt_direc

                # Create plot folder for given .csv
                os.mkdir(plot_direc_path)

            # Import trajectory .csv as dataframe
            result_df = pd.read_csv(results_direc + '/' + result)

            # Start dict of dataframes avaliable from .csv in directory
            df_dict = {'results': result_df}

            # If they exist, import contact, setpoint, and control .csv as df
            contact_filename = result_name + '_contact.csv'
            control_filename = result_name + '_control.csv'
            setpoint_filename = result_name + '_setpoint.csv'
            experimental_filename = result_name + '_experimental.csv'
            exp_contact_filename = result_name + '_exp_contact.csv'

            if contact_filename in results_direc_conts:
                points_df = pd.read_csv(results_direc + '/' + contact_filename)
                df_dict['contact'] = points_df

            if control_filename in results_direc_conts:
                control_df = pd.read_csv(results_direc + '/' + control_filename)
                df_dict['control'] = control_df

            if setpoint_filename in results_direc_conts:
                setpoint_df = pd.read_csv(results_direc + '/' + 
                                          setpoint_filename)
                df_dict['setpoint'] = setpoint_df

            if experimental_filename in results_direc_conts:
                experimental_df = pd.read_csv(results_direc + '/' + 
                                              experimental_filename)
                df_dict['experimental'] = experimental_df

            if exp_contact_filename in results_direc_conts:
                exp_contact_df = pd.read_csv(results_direc + '/' + 
                                             exp_contact_filename)
                df_dict['exp_contact'] = exp_contact_df

            # Run plotting method on the trajectory dataframe
            self.plot_all(df_dict, plot_direc_path, command_dict)

    def plot_all(self, df_dict, save_dir, command_dict) -> None:
        '''
        Method that organizes calls to all desired plotting methods for the 
        amls data. Also displays and/or saves plots as commanded
        '''

        # Get base name from directory
        first_idx = save_dir.rfind('/') + 1 # Start of directory name
        root_name = save_dir[first_idx:-17] # Root name w/o additional index

        print('Creating plots for case: ' + root_name)

        # Call plot methods according to command dict
        gv_command = command_dict['gv_plot']
        plt_gen = ('all', 'save', 'plot')
        plt_save = ('all', 'save')
        plt_show = ('all', 'plot')

        # Separate out trajectory csv from dict
        results_df = df_dict['results']
        df_fields = df_dict.keys()

        # Determine if contact and control dataframes are included in dict
        #   Dont check setpoints because they dont have their own plots
        contact_check = 'contact' in df_fields
        control_check = 'control' in df_fields

        sp_command = command_dict['setpoint_plot']

        # Create position plot if commanded
        if command_dict['position'] in plt_gen:

            fig_pos = self.plot_position(df_dict, gv_command, sp_command)

            if command_dict['position'] in plt_save:
                fig_pos.savefig(save_dir + '/' + root_name + '_pos.png')

            if command_dict['position'] not in plt_show:
                plt.close(fig_pos)
            else:
                self.any_plot_show = True

        # Create linear velocity plot if commanded
        if command_dict['lin_velocity'] in plt_gen:

            fig_lin_vel = self.plot_lin_velocity(df_dict, 
                                                 gv_plot_ctl=gv_command)

            if command_dict['lin_velocity'] in plt_save:
                fig_lin_vel.savefig(save_dir + '/' + root_name +'_lin_vel.png')

            if command_dict['lin_velocity'] not in plt_show:
                plt.close(fig_lin_vel)
            else:
                self.any_plot_show = True
            
        # Create quaternion plot if commanded
        if command_dict['quaternion'] in plt_gen:

            fig_quat = self.plot_quat(df_dict, gv_plot_ctl=gv_command)

            if command_dict['quaternion'] in plt_save:
                fig_quat.savefig(save_dir + '/' + root_name + '_quat.png')

            if command_dict['quaternion'] not in plt_show:
                plt.close(fig_quat)
            else:
                self.any_plot_show = True

        # Create euler plot if commanded
        if command_dict['euler'] in plt_gen:

            fig_eul = self.plot_eul(df_dict, gv_plot_ctl=gv_command)

            if command_dict['euler'] in plt_save:
                fig_eul.savefig(save_dir + '/' + root_name + '_eul.png')

            if command_dict['euler'] not in plt_show:
                plt.close(fig_eul)
            else:
                self.any_plot_show = True

        # Create angular velocity plot if commanded
        if command_dict['ang_velocity'] in plt_gen:

            fig_ang_vel = self.plot_ang_velocity(df_dict, 
                                                 gv_plot_ctl=gv_command)

            if command_dict['ang_velocity'] in plt_save:
                fig_ang_vel.savefig(save_dir + '/' + root_name+'_ang_vel.png')

            if command_dict['ang_velocity'] not in plt_show:
                plt.close(fig_ang_vel)
            else:
                self.any_plot_show = True

        # Create contact point position plot if data avaliable and commanded
        if (command_dict['cont_position'] in plt_gen) and contact_check:

            fig_cont_pos = self.plot_cont_pos(df_dict)

            if command_dict['cont_position'] in plt_save:
                fig_cont_pos.savefig(save_dir + '/'+root_name+'_cont_pos.png')

            if command_dict['cont_position'] not in plt_show:
                plt.close(fig_cont_pos)
            else:
                self.any_plot_show = True 

        # Create contact point velocity plot if commanded
        if (command_dict['cont_velocity'] in plt_gen) and contact_check:

            fig_cont_vel = self.plot_cont_vel(df_dict)

            if command_dict['cont_velocity'] in plt_save:
                fig_cont_vel.savefig(save_dir + '/'+root_name+'_cont_vel.png')

            if command_dict['cont_velocity'] not in plt_show:
                plt.close(fig_cont_vel)
            else:
                self.any_plot_show = True 

        # Create plot of commanded body torques and thrust if commanded
        if (command_dict['ctl_torques'] in plt_gen) and control_check:

            control_df = df_dict['control']
            fig_ctl_torque = self.plot_ctl_torque(control_df)

            if command_dict['ctl_torques'] in plt_save:
                fig_ctl_torque.savefig(save_dir + '/' + root_name + \
                                       '_ctl_torque.png')

            if command_dict['ctl_torques'] not in plt_show:
                plt.close(fig_ctl_torque)
            else:
                self.any_plot_show = True 

        # Create plot of control inputs (motor angular velocities) if commanded
        if (command_dict['ctl_rotors'] in plt_gen) and control_check:

            control_df = df_dict['control']
            fig_ctl_rotor = self.plot_ctl_rotor(control_df)

            if command_dict['ctl_rotors'] in plt_save:
                fig_ctl_rotor.savefig(save_dir + '/' + root_name + \
                                       '_ctl_rotor.png')

            if command_dict['ctl_rotors'] not in plt_show:
                plt.close(fig_ctl_rotor)
            else:
                self.any_plot_show = True 

        # Create G-frame position plot if commanded
        if command_dict['position_G'] in plt_gen:

            fig_pos_G = self.plot_position_G(df_dict)

            if command_dict['position_G'] in plt_save:
                fig_pos_G.savefig(save_dir + '/' + root_name + '_pos_G.png')

            if command_dict['position_G'] not in plt_show:
                plt.close(fig_pos_G)
            else:
                self.any_plot_show = True

        # Create G-frame linear velocity plot if commanded
        if command_dict['lin_velocity_G'] in plt_gen:

            fig_lin_vel_G = self.plot_lin_velocity_G(df_dict)

            if command_dict['lin_velocity_G'] in plt_save:
                fig_lin_vel_G.savefig(save_dir + '/' + root_name +
                                      '_lin_vel_G.png')

            if command_dict['lin_velocity_G'] not in plt_show:
                plt.close(fig_lin_vel_G)
            else:
                self.any_plot_show = True

        # Create animation if commanded
        if command_dict['animation'] in plt_gen:

            print('Animating case: ' + root_name)

            points_df = df_dict['contact']
            ani_3d, fig_3d = self.animate_landing(results_df, points_df)

            if command_dict['animation'] in plt_save:

                # Save to m4 using ffmpeg writer
                writervideo = animation.FFMpegWriter(fps=self.video_freq)
                ani_name = save_dir + '/' + root_name + '_animation.mp4'
                ani_3d.save(ani_name, writer=writervideo)

            if command_dict['animation'] not in plt_show:
                plt.close(fig_3d)
            else:
                self.any_plot_show = True 

        # Show plots if any are commanded to be displayed
        if self.any_plot_show: # If any plots set to be displayed
            plt.show()

    def plot_position(self, df_dict, gv_plot_ctl='full_state', 
                      sp_command=True) -> plt.figure:
        '''
        Plot uav (and gv if desired) position states and setpoints
        '''

        results_df = df_dict['results'] # Separate primary uav result df

        # Determine if setpoint results commanded and included
        df_fields = df_dict.keys()
        sp_check = 'setpoint' in df_fields
        exp_check = 'experimental' in df_fields

        # Create figure, subplots, and overall title
        fig_pos, (ax_x_pos, ax_y_pos, ax_z_pos) = plt.subplots(3, 1, 
            figsize=(self.fig_scale*self.fig_x_size, 
                     self.fig_scale*self.fig_y_size), 
            dpi=self.fig_dpi, sharex=True)
        fig_pos.suptitle('Multirotor Position States in I', 
            fontsize=self.font_suptitle, fontweight='bold')
        
        # Plot uav position states
        ax_x_pos.plot(results_df.t, results_df.x_uav, linewidth=self.plt_lw)  
        ax_y_pos.plot(results_df.t, results_df.y_uav, linewidth=self.plt_lw)
        ax_z_pos.plot(results_df.t, results_df.z_uav, linewidth=self.plt_lw) 
        legend_txt_x = ['$x_{uav}$']
        legend_txt_y = ['$y_{uav}$']
        legend_txt_z = ['$z_{uav}$']

        # Plot ground vehicle states if commanded
        if (gv_plot_ctl == 'full_state') or (gv_plot_ctl == 'pos_states'):
            ax_x_pos.plot(results_df.t, results_df.x_gv, '--', 
                          linewidth=self.plt_lw)
            ax_y_pos.plot(results_df.t, results_df.y_gv, '--', 
                          linewidth=self.plt_lw)
            ax_z_pos.plot(results_df.t, results_df.z_gv, '--', 
                          linewidth=self.plt_lw)
            legend_txt_x.append('$x_{gv}$')
            legend_txt_y.append('$y_{gv}$')
            legend_txt_z.append('$z_{gv}$')

        # Plot position setpoints if they exist and are commanded
        if sp_check and sp_command:
            setpoint_df = df_dict['setpoint']
            ax_x_pos.plot(setpoint_df.t, setpoint_df.x, color='red', 
                          linewidth=self.setpoint_lw)
            ax_y_pos.plot(setpoint_df.t, setpoint_df.y, color='red', 
                          linewidth=self.setpoint_lw)
            ax_z_pos.plot(setpoint_df.t, setpoint_df.z, color='red', 
                          linewidth=self.setpoint_lw)
            legend_txt_x.append('$x_{uav-sp}$')
            legend_txt_y.append('$y_{uav-sp}$')
            legend_txt_z.append('$z_{uav-sp}$')

        # Plot experimental position if the data exists
        if exp_check:
            experimental_df = df_dict['experimental']
            ax_x_pos.plot(experimental_df.t, experimental_df.x_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_y_pos.plot(experimental_df.t, experimental_df.y_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_z_pos.plot(experimental_df.t, experimental_df.z_uav,
                          color='peru', linewidth=self.plt_lw)
            legend_txt_x.append('$x_{uav-exp}$')
            legend_txt_y.append('$y_{uav-exp}$')
            legend_txt_z.append('$z_{uav-exp}$')

        # Add formatting and legends to plots
        ax_x_pos.set_title('X Position', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_x_pos.set_ylabel('Position (m)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_x_pos.legend(legend_txt_x, fontsize=self.font_legend)
        ax_x_pos.tick_params(axis='both', labelsize=self.font_axtick)
        ax_x_pos.grid()

        ax_y_pos.set_title('Y Position', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_y_pos.set_ylabel('Position (m)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_y_pos.legend(legend_txt_y, fontsize=self.font_legend)
        ax_y_pos.tick_params(axis='both', labelsize=self.font_axtick)
        ax_y_pos.grid()

        ax_z_pos.set_title('Z Position', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_z_pos.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_z_pos.set_ylabel('Position (m)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_z_pos.legend(legend_txt_z, fontsize=self.font_legend)
        ax_z_pos.tick_params(axis='both', labelsize=self.font_axtick)
        # ax_z_pos.set_ylim(0.5, -5)
        ax_z_pos.grid()
        
        fig_pos.tight_layout() # Remove whitespace
        
        return fig_pos

    def plot_lin_velocity(self, df_dict, gv_plot_ctl='full_state', 
                          sp_command=True) -> plt.figure:
        '''
        Plot uav (and gv if desired) linear velocity states and setpoints
        '''

        results_df = df_dict['results'] # Separate primary uav result df

        # Determine if setpoint results commanded and included
        df_fields = df_dict.keys()
        sp_check = 'setpoint' in df_fields
        exp_check = 'experimental' in df_fields

        # Create figure, subplots, and overall title
        fig_vel, (ax_x_vel, ax_y_vel, ax_z_vel) = plt.subplots(3, 1, 
            figsize=(self.fig_scale*self.fig_x_size, 
                     self.fig_scale*self.fig_y_size), 
            dpi=self.fig_dpi, sharex=True)
        fig_vel.suptitle('UAV Velocity States', fontsize=self.font_suptitle, 
                         fontweight='bold')
        
        # Plot uav velocity states
        ax_x_vel.plot(results_df.t, results_df.dx_uav, linewidth=self.plt_lw) 
        ax_y_vel.plot(results_df.t, results_df.dy_uav, linewidth=self.plt_lw) 
        ax_z_vel.plot(results_df.t, results_df.dz_uav, linewidth=self.plt_lw)
        legend_txt_x = ['$dx_{uav}$']
        legend_txt_y = ['$dy_{uav}$']
        legend_txt_z = ['$dz_{uav}$']

        # Plot ground vehicle velocity states if commanded
        if (gv_plot_ctl == 'full_state') or (gv_plot_ctl == 'pos_states'):

            # Create new dataframe with gv velocity in W frame
            G_W_conv_gv = lambda row: pd.Series(Rot.from_quat([row.qx_gv, 
                row.qy_gv, row.qz_gv, row.qw_gv]).apply(np.array([row.dx_gv, 
                row.dy_gv, row.dz_gv])))
            df_gv_vel_W = results_df.apply(G_W_conv_gv, axis=1)
            df_gv_vel_W.columns = ['dx_gv', 'dy_gv', 'dz_gv']

            ax_x_vel.plot(results_df.t, df_gv_vel_W.dx_gv, '--', 
                        linewidth=self.plt_lw)
            ax_y_vel.plot(results_df.t, df_gv_vel_W.dy_gv, '--', 
                        linewidth=self.plt_lw)
            ax_z_vel.plot(results_df.t, df_gv_vel_W.dz_gv, '--', 
                        linewidth=self.plt_lw)
            legend_txt_x.append('$dx_{gv}$')
            legend_txt_y.append('$dy_{gv}$')
            legend_txt_z.append('$dz_{gv}$')

        # Plot velocity setpoints if they exist and are commanded
        if sp_check and sp_command:
            setpoint_df = df_dict['setpoint']
            ax_x_vel.plot(setpoint_df.t, setpoint_df.dx, color='red', 
                          linewidth=self.setpoint_lw)
            ax_y_vel.plot(setpoint_df.t, setpoint_df.dy, color='red', 
                          linewidth=self.setpoint_lw)
            ax_z_vel.plot(setpoint_df.t, setpoint_df.dz, color='red', 
                          linewidth=self.setpoint_lw)
            legend_txt_x.append('$dx_{uav-sp}$')
            legend_txt_y.append('$dy_{uav-sp}$')
            legend_txt_z.append('$dz_{uav-sp}$')

        # Plot experimental velocity if the data exists
        if exp_check:
            experimental_df = df_dict['experimental']
            ax_x_vel.plot(experimental_df.t, experimental_df.dx_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_y_vel.plot(experimental_df.t, experimental_df.dy_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_z_vel.plot(experimental_df.t, experimental_df.dz_uav,
                          color='peru', linewidth=self.plt_lw)
            legend_txt_x.append('$dx_{uav-exp}$')
            legend_txt_y.append('$dy_{uav-exp}$')
            legend_txt_z.append('$dz_{uav-exp}$')

        # Add formatting and legends to plots
        ax_x_vel.set_title('X Velocity', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_x_vel.set_ylabel('Velocity (m/s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_x_vel.legend(legend_txt_x, fontsize=self.font_legend)
        ax_x_vel.grid()

        ax_y_vel.set_title('Y Velocity', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_y_vel.set_ylabel('Velocity (m/s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_y_vel.legend(legend_txt_y, fontsize=self.font_legend)
        ax_y_vel.grid()

        ax_z_vel.set_title('Z Velocity', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_z_vel.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_z_vel.set_ylabel('Velocity (m/s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_z_vel.legend(legend_txt_z, fontsize=self.font_legend)
        ax_z_vel.invert_yaxis()
        ax_z_vel.grid()

        fig_vel.tight_layout()
            
        return fig_vel

    def plot_quat(self, df_dict, gv_plot_ctl='full_state', 
                  sp_command=True) -> plt.figure:
        '''
        Plot uav (and gv if desired) quaternion states and setpoints
        '''

        results_df = df_dict['results'] # Separate primary uav result df

        # Determine if setpoint results commanded and included
        df_fields = df_dict.keys()
        sp_check = 'setpoint' in df_fields
        exp_check = 'experimental' in df_fields

        # Create figure, subplots, and overall title
        fig_quat, (ax_qx, ax_qy, ax_qz, ax_qw) = plt.subplots(4, 1, 
            figsize=(self.fig_scale*self.fig_x_size, 
                     self.fig_scale*self.fig_y_size), 
            dpi=self.fig_dpi, sharex=True)
        fig_quat.suptitle('UAV Quaternion States', fontsize=self.font_suptitle, 
                          fontweight='bold')
        
        # Plot uav quaternion states
        ax_qx.plot(results_df.t, results_df.qx_uav, linewidth=self.plt_lw) 
        ax_qy.plot(results_df.t, results_df.qy_uav, linewidth=self.plt_lw) 
        ax_qz.plot(results_df.t, results_df.qz_uav, linewidth=self.plt_lw) 
        ax_qw.plot(results_df.t, results_df.qw_uav, linewidth=self.plt_lw) 
        legend_txt_x = ['$qx_{uav}$']
        legend_txt_y = ['$qy_{uav}$']
        legend_txt_z = ['$qz_{uav}$']
        legend_txt_w = ['$qw_{uav}$']

        # Plot ground vehicle quaternion states if commanded
        if (gv_plot_ctl == 'full_state'):
            ax_qx.plot(results_df.t, results_df.qx_gv, '--', 
                       linewidth=self.plt_lw) 
            ax_qy.plot(results_df.t, results_df.qy_gv, '--', 
                       linewidth=self.plt_lw) 
            ax_qz.plot(results_df.t, results_df.qz_gv, '--', 
                       linewidth=self.plt_lw) 
            ax_qw.plot(results_df.t, results_df.qw_gv, '--', 
                       linewidth=self.plt_lw) 
            legend_txt_x.append('$qx_{gv}$')
            legend_txt_y.append('$qy_{gv}$')
            legend_txt_z.append('$qz_{gv}$')
            legend_txt_w.append('$qw_{gv}$')

        # Plot uav quaternion setpoints if they exist and are commanded
        if sp_check and sp_command:
            setpoint_df = df_dict['setpoint']
            ax_qx.plot(setpoint_df.t, setpoint_df.qx, color='red', 
                       linewidth=self.setpoint_lw)
            ax_qy.plot(setpoint_df.t, setpoint_df.qy, color='red', 
                       linewidth=self.setpoint_lw)
            ax_qz.plot(setpoint_df.t, setpoint_df.qz, color='red', 
                       linewidth=self.setpoint_lw)
            ax_qw.plot(setpoint_df.t, setpoint_df.qw, color='red', 
                       linewidth=self.setpoint_lw)
            legend_txt_x.append('$qx_{uav-sp}$')
            legend_txt_y.append('$qy_{uav-sp}$')
            legend_txt_z.append('$qz_{uav-sp}$')
            legend_txt_w.append('$qw_{uav-sp}$')

        # Plot experimental quaternion if the data exists
        if exp_check:
            experimental_df = df_dict['experimental']
            ax_qx.plot(experimental_df.t, experimental_df.qx_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_qy.plot(experimental_df.t, experimental_df.qy_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_qz.plot(experimental_df.t, experimental_df.qz_uav,
                          color='peru', linewidth=self.plt_lw)
            ax_qw.plot(experimental_df.t, experimental_df.qw_uav,
                          color='peru', linewidth=self.plt_lw)
            legend_txt_x.append('$qx_{uav-exp}$')
            legend_txt_y.append('$qy_{uav-exp}$')
            legend_txt_z.append('$qz_{uav-exp}$')
            legend_txt_w.append('$qw_{uav-exp}$')

        # Add formatting and legends to plots
        ax_qx.set_title('X Quaternion', fontsize=self.font_subtitle, 
                        fontweight='bold')
        ax_qx.legend(legend_txt_x, fontsize=self.font_legend)
        ax_qx.grid()

        ax_qy.set_title('Y Quaternion', fontsize=self.font_subtitle, 
                        fontweight='bold')
        ax_qy.legend(legend_txt_y, fontsize=self.font_legend)
        ax_qy.grid()

        ax_qz.set_title('Z Quaternion', fontsize=self.font_subtitle, 
                        fontweight='bold')
        ax_qz.legend(legend_txt_z, fontsize=self.font_legend)
        ax_qz.grid()

        ax_qw.set_title('W Quaternion', fontsize=self.font_subtitle, 
                        fontweight='bold')
        ax_qw.legend(legend_txt_w, fontsize=self.font_legend)
        ax_qw.grid()

        fig_quat.tight_layout()
            
        return fig_quat

    def plot_eul(self, df_dict, gv_plot_ctl='full_state', 
                 sp_command=True) -> plt.figure:
        '''
        Plot uav (and gv if desired) euler angle states and setpoints
        '''

        results_df = df_dict['results'] # Separate primary uav result df

        # Determine if setpoint results commanded and included
        df_fields = df_dict.keys()
        sp_check = 'setpoint' in df_fields
        exp_check = 'experimental' in df_fields

        # Create figure, subplots, and overall title
        fig_eul, (ax_phi, ax_theta, ax_psi) = plt.subplots(3, 1, 
            figsize=(self.fig_scale*self.fig_x_size, 
                     self.fig_scale*self.fig_y_size), 
            dpi=self.fig_dpi, sharex=True)
        fig_eul.suptitle('UAV Euler Angles', fontsize=self.font_suptitle, 
                         fontweight='bold')
        
        # Plot uav euler angles
        ax_phi.plot(results_df.t, results_df.phi_uav, linewidth=self.plt_lw) 
        ax_theta.plot(results_df.t, results_df.theta_uav, 
                      linewidth=self.plt_lw) 
        ax_psi.plot(results_df.t, results_df.psi_uav, linewidth=self.plt_lw) 
        legend_txt_phi = [r'$\Phi_{uav}$']
        legend_txt_theta = [r'$\Theta_{uav}$']
        legend_txt_psi = [r'$\Psi_{uav}$']

        # Plot ground vehicle euler angles if commanded
        if (gv_plot_ctl == 'full_state'):
            ax_phi.plot(results_df.t, results_df.phi_gv, '--', 
                        linewidth=self.plt_lw) 
            ax_theta.plot(results_df.t, results_df.theta_gv, '--', 
                          linewidth=self.plt_lw) 
            ax_psi.plot(results_df.t, results_df.psi_gv, '--', 
                        linewidth=self.plt_lw) 
            legend_txt_phi.append(r'$\Phi_{gv}$')
            legend_txt_theta.append(r'$\Theta_{gv}$')
            legend_txt_psi.append(r'$\Psi_{gv}$')

        # Plot euler angle setpoints if they exist and are commanded
        if sp_check and sp_command:
            setpoint_df = df_dict['setpoint']
            ax_phi.plot(setpoint_df.t, setpoint_df.phi, color='red', 
                        linewidth=self.setpoint_lw)
            ax_theta.plot(setpoint_df.t, setpoint_df.theta, color='red', 
                          linewidth=self.setpoint_lw)
            ax_psi.plot(setpoint_df.t, setpoint_df.psi, color='red', 
                        linewidth=self.setpoint_lw)
            legend_txt_phi.append(r'$\Phi_{uav-sp}$')
            legend_txt_theta.append(r'$\Theta_{uav-sp}$')
            legend_txt_psi.append(r'$\Psi_{uav-sp}$')

        # Plot experimental euler angles if the data exists
        if exp_check:
            experimental_df = df_dict['experimental']
            ax_phi.plot(experimental_df.t, experimental_df.phi_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_theta.plot(experimental_df.t, experimental_df.theta_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_psi.plot(experimental_df.t, experimental_df.psi_uav,
                          color='peru', linewidth=self.plt_lw)
            legend_txt_phi.append(r'$\Phi_{uav-exp}$')
            legend_txt_theta.append(r'$\Theta_{uav-exp}$')
            legend_txt_psi.append(r'$\Psi_{uav-exp}$')

        # Add formatting and legends to plots
        ax_phi.set_title(r'Roll Angle ($\Phi$)', fontsize=self.font_subtitle, 
                         fontweight='bold')
        ax_phi.set_ylabel('Angle (deg)', fontsize=self.font_axlabel, 
                          fontweight='bold')
        ax_phi.legend(legend_txt_phi, fontsize=self.font_legend)
        ax_phi.grid()

        ax_theta.set_title(r'Pitch Angle ($\Theta$)', 
                           fontsize=self.font_subtitle, fontweight='bold')
        ax_theta.set_ylabel('Angle (deg)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_theta.legend(legend_txt_theta, fontsize=self.font_legend)
        ax_theta.grid()

        ax_psi.set_title(r'Yaw Angle ($\Psi$)', fontsize=self.font_subtitle, 
                         fontweight='bold')
        ax_psi.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                          fontweight='bold')
        ax_psi.set_ylabel('Angle (deg)', fontsize=self.font_axlabel, 
                          fontweight='bold')
        ax_psi.legend(legend_txt_psi, fontsize=self.font_legend)
        ax_psi.grid()
        
        fig_eul.tight_layout()

        return fig_eul

    def plot_ang_velocity(self, df_dict, gv_plot_ctl='full_state', 
                          sp_command=True) -> plt.figure:
        '''
        Plot uav (and gv if desired) angular velocity states and setpoints
        '''

        results_df = df_dict['results'] # Separate primary uav result df

        # Determine if setpoint results commanded and included
        df_fields = df_dict.keys()
        sp_check = 'setpoint' in df_fields
        exp_check = 'experimental' in df_fields

        # Create figure, subplots, and overall title
        fig_om, (ax_om_x, ax_om_y, ax_om_z) = plt.subplots(3, 1, 
            figsize=(self.fig_scale*self.fig_x_size, 
                     self.fig_scale*self.fig_y_size), 
            dpi=self.fig_dpi, sharex=True)
        fig_om.suptitle('UAV Angular Velocity States', 
                        fontsize=self.font_suptitle, fontweight='bold')
        
        # Plot uav angular velocity states
        ax_om_x.plot(results_df.t, (180/pi)*results_df.om_x_uav, 
                    linewidth=self.plt_lw) 
        ax_om_y.plot(results_df.t, (180/pi)*results_df.om_y_uav, 
                    linewidth=self.plt_lw) 
        ax_om_z.plot(results_df.t, (180/pi)*results_df.om_z_uav, 
                    linewidth=self.plt_lw) 
        legend_txt_x = [r'$\omega x_{uav}$']
        legend_txt_y = [r'$\omega y_{uav}$']
        legend_txt_z = [r'$\omega z_{uav}$']

        # Plot ground vehicle states if commanded
        if (gv_plot_ctl == 'full_state'):
            ax_om_x.plot(results_df.t, (180/pi)*results_df.om_x_gv, '--', 
                         linewidth=self.plt_lw) 
            ax_om_y.plot(results_df.t, (180/pi)*results_df.om_y_gv, '--', 
                         linewidth=self.plt_lw) 
            ax_om_z.plot(results_df.t, (180/pi)*results_df.om_z_gv, '--', 
                         linewidth=self.plt_lw) 
            legend_txt_x.append(r'$\omega x_{gv}$')
            legend_txt_y.append(r'$\omega y_{gv}$')
            legend_txt_z.append(r'$\omega z_{gv}$')

        # Plot angular velocity setpoints if they exist and are commanded
        if sp_check and sp_command:
            setpoint_df = df_dict['setpoint']
            ax_om_x.plot(setpoint_df.t, (180/pi)*setpoint_df.om_x, color='red', 
                          linewidth=self.setpoint_lw)
            ax_om_y.plot(setpoint_df.t, (180/pi)*setpoint_df.om_y, color='red', 
                          linewidth=self.setpoint_lw)
            ax_om_z.plot(setpoint_df.t, (180/pi)*setpoint_df.om_z, color='red', 
                          linewidth=self.setpoint_lw)
            legend_txt_x.append(r'$\omega x_{uav-sp}$')
            legend_txt_y.append(r'$\omega y_{uav-sp}$')
            legend_txt_z.append(r'$\omega z_{uav-sp}$')

        # Plot experimental position if the data exists
        if exp_check:
            experimental_df = df_dict['experimental']
            ax_om_x.plot(experimental_df.t, (180/pi)*experimental_df.om_x_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_om_y.plot(experimental_df.t, (180/pi)*experimental_df.om_y_uav, 
                          color='peru', linewidth=self.plt_lw)
            ax_om_z.plot(experimental_df.t, (180/pi)*experimental_df.om_z_uav,
                          color='peru', linewidth=self.plt_lw)
            legend_txt_x.append(r'$\omega x_{uav-exp}$')
            legend_txt_y.append(r'$\omega y_{uav-exp}$')
            legend_txt_z.append(r'$\omega z_{uav-exp}$')

        # Add formatting and legends to plots
        ax_om_x.set_title(r'$\omega_x$', fontsize=self.font_subtitle, 
                          fontweight='bold')
        ax_om_x.set_ylabel('Ang. Vel. (deg/s)', fontsize=self.font_axlabel, 
                           fontweight='bold')
        ax_om_x.legend(legend_txt_x, fontsize=self.font_legend)
        ax_om_x.grid()

        ax_om_y.set_title(r'$\omega_y$', fontsize=self.font_subtitle, 
                          fontweight='bold')
        ax_om_y.set_ylabel('Ang. Vel. (deg/s)', fontsize=self.font_axlabel, 
                           fontweight='bold')
        ax_om_y.legend(legend_txt_y, fontsize=self.font_legend)
        ax_om_y.grid()

        ax_om_z.set_title(r'$\omega_z$', fontsize=self.font_subtitle, 
                          fontweight='bold')
        ax_om_z.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                           fontweight='bold')
        ax_om_z.set_ylabel('Ang. Vel. (deg/s)', fontsize=self.font_axlabel, 
                           fontweight='bold')
        ax_om_z.legend(legend_txt_z, fontsize=self.font_legend)
        ax_om_z.grid()

        fig_om.tight_layout()

        return fig_om

    def plot_cont_pos(self, df_dict: dict) -> plt.figure:
        '''
        Plot uav contact point position states in the V frame, relative to the
        dock
        '''

        df_results = df_dict['results'] # Separate primary uav result df
        df_points = deepcopy(df_dict['contact']) # Contact point df
        dock_dim = ground_vehicle_1.contact_size[0]

        num_points = int((df_points.shape[1] - 1)/6)

        # Subtract out G origin from W coordinates
        for j in range(num_points):

            df_points.iloc[:, 6*j + 1] = df_points.iloc[:, 6*j + 1] - \
                df_results.x_gv
            df_points.iloc[:, 6*j + 2] = df_points.iloc[:, 6*j + 2] - \
                df_results.y_gv
            df_points.iloc[:, 6*j + 3] = df_points.iloc[:, 6*j + 3] - \
                df_results.z_gv

        # Loop through all rows 
        for i in range(df_points.shape[0]):

            # Setup rotation from W to G frame
            R_W_G = Rot.from_quat(df_results.iloc[i, 20:24]).inv()

            # print(df_results.columns)
            # input('hold')

            # Loop through all contact points and rotate each point
            for j in range(num_points):

                if j < num_points:
                    df_points.iloc[i, (1 + 6*j):(6*j + 4)] = \
                        R_W_G.apply(df_points.iloc[i, (1 + 6*j):(6*j + 4)])
                else:
                    df_points.iloc[i, (1 + 6*j):] = \
                        R_W_G.apply(df_points.iloc[i, (1 + 6*j):])

        # Determine if experimental results included
        df_fields = df_dict.keys()
        exp_check = 'exp_contact' in df_fields

        # Create figure, subplots, and overall title
        point_cols = df_points.columns.values.tolist()
        point_cols = ['${0}$'.format(col) for col in point_cols]
        fig_cont_pos, (ax_x_cont_pos, ax_y_cont_pos, ax_z_cont_pos) = \
            plt.subplots(3, 1, figsize=(self.fig_scale*self.fig_x_size, 
                                        self.fig_scale*self.fig_y_size), 
                         dpi=self.fig_dpi, sharex=True)
        fig_cont_pos.suptitle('Contact Point Position States in V', 
                              fontsize=self.font_suptitle, fontweight='bold')
        
        # x position
        ax_x_cont_pos.plot(df_points.t, df_points.iloc[:, 1::6], 
                           linewidth=self.plt_lw) 
        ax_x_cont_pos.axhline(dock_dim/2)
        ax_x_cont_pos.axhline(-dock_dim/2)
        ax_x_cont_pos.set_title('X Position', fontsize=self.font_subtitle, 
                                fontweight='bold')
        ax_x_cont_pos.set_ylabel('Position (m)', fontsize=self.font_axlabel, 
                                 fontweight='bold')
        ax_x_cont_pos.tick_params(axis='both', labelsize=self.font_axtick)
        ax_x_cont_pos.grid()

        # y position
        ax_y_cont_pos.plot(df_points.t, df_points.iloc[:, 2::6], 
                           linewidth=self.plt_lw) 
        ax_y_cont_pos.axhline(dock_dim/2)
        ax_y_cont_pos.axhline(-dock_dim/2)
        ax_y_cont_pos.set_title('Y Position', fontsize=self.font_subtitle, 
                                fontweight='bold')
        ax_y_cont_pos.set_ylabel('Position (m)', fontsize=self.font_axlabel, 
                                 fontweight='bold')
        ax_y_cont_pos.tick_params(axis='both', labelsize=self.font_axtick)
        ax_y_cont_pos.grid() 

        # z position
        ax_z_cont_pos.plot(df_points.t, df_points.iloc[:, 3::6], 
                           linewidth=self.plt_lw) 
        ax_z_cont_pos.set_title('Z Position', fontsize=self.font_subtitle, 
                                fontweight='bold')
        ax_z_cont_pos.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                                 fontweight='bold')
        ax_z_cont_pos.set_ylabel('Position (m)', fontsize=self.font_axlabel, 
                                 fontweight='bold')
        ax_z_cont_pos.tick_params(axis='both', labelsize=self.font_axtick)
        ax_z_cont_pos.grid()
        ax_z_cont_pos.invert_yaxis()

        legend_txt_x = point_cols[1::6]
        legend_txt_x = ['$c_1$', '$c_2$', '$c_3$', '$c_4$']
        legend_txt_y = point_cols[2::6]
        legend_txt_y = ['$c_1$', '$c_2$', '$c_3$', '$c_4$']
        legend_txt_z = point_cols[3::6]
        legend_txt_z = ['$c_1$', '$c_2$', '$c_3$', '$c_4$']

        # Plot experimental contact positions if the data exists
        if exp_check:
            df_exp_contact = df_dict['exp_contact']
            ax_x_cont_pos.plot(df_exp_contact.t, df_exp_contact.iloc[:, 1::6], 
                               color='peru', linewidth=self.setpoint_lw)
            ax_y_cont_pos.plot(df_exp_contact.t, df_exp_contact.iloc[:, 2::6], 
                               color='peru', linewidth=self.setpoint_lw)
            ax_z_cont_pos.plot(df_exp_contact.t, df_exp_contact.iloc[:, 3::6],
                               color='peru', linewidth=self.setpoint_lw)

        # Add legends to axes
        ax_x_cont_pos.legend(legend_txt_x, fontsize=self.font_legend)
        ax_y_cont_pos.legend(legend_txt_y, fontsize=self.font_legend)
        ax_z_cont_pos.legend(legend_txt_z, fontsize=self.font_legend)

        fig_cont_pos.tight_layout()

        return fig_cont_pos

    def plot_cont_vel(self, df_dict) -> plt.figure:
        '''
        Plot uav contact point linear velocity states
        '''

        df_points = df_dict['contact']

        # Determine if experimental results included
        df_fields = df_dict.keys()
        exp_check = 'exp_contact' in df_fields

        point_cols = df_points.columns.values.tolist()
        point_cols = ['${0}$'.format(col) for col in point_cols]

        # Create figure, subplots, and overall title
        fig_cont_vel, (ax_x_cont_vel, ax_y_cont_vel, ax_z_cont_vel) = \
            plt.subplots(3, 1, figsize=(self.fig_scale*self.fig_x_size, 
                                        self.fig_scale*self.fig_y_size), 
                         dpi=self.fig_dpi, sharex=True)
        fig_cont_vel.suptitle('Contact Point Velocity States in W', 
                              fontsize=self.font_suptitle, fontweight='bold')

        # x velocity
        ax_x_cont_vel.plot(df_points.t, df_points.iloc[:, 4::6], 
                           linewidth=self.plt_lw) 
        ax_x_cont_vel.set_title('X Velocity', fontsize=self.font_subtitle, 
                                fontweight='bold')
        ax_x_cont_vel.set_ylabel('Velocity (m/s)', fontsize=self.font_axlabel, 
                                 fontweight='bold')
        ax_x_cont_vel.grid()

        # y velocity
        ax_y_cont_vel.plot(df_points.t, df_points.iloc[:, 5::6], 
                           linewidth=self.plt_lw) 
        ax_y_cont_vel.set_title('Y Velocity', fontsize=self.font_subtitle, 
                                fontweight='bold')
        ax_y_cont_vel.set_ylabel('Velocity (m/s)', fontsize=self.font_axlabel, 
                                 fontweight='bold')
        ax_y_cont_vel.grid()

        # z velocity
        ax_z_cont_vel.plot(df_points.t, df_points.iloc[:, 6::6], 
                           linewidth=self.plt_lw) 
        ax_z_cont_vel.set_title('Z Velocity', fontsize=self.font_subtitle, 
                                fontweight='bold')
        ax_z_cont_vel.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                                 fontweight='bold')
        ax_z_cont_vel.set_ylabel('Velocity (m/s)', fontsize=self.font_axlabel, 
                                 fontweight='bold')
        ax_z_cont_vel.invert_yaxis()
        ax_z_cont_vel.grid()

        legend_txt_x = point_cols[4::6]
        legend_txt_y = point_cols[5::6]
        legend_txt_z = point_cols[6::6]

        # Plot experimental contact positions if the data exists
        if exp_check:
            df_exp_contact = df_dict['exp_contact']
            ax_x_cont_vel.plot(df_exp_contact.t, df_exp_contact.iloc[:, 4::6], 
                               color='peru', linewidth=self.setpoint_lw)
            ax_y_cont_vel.plot(df_exp_contact.t, df_exp_contact.iloc[:, 5::6], 
                               color='peru', linewidth=self.setpoint_lw)
            ax_z_cont_vel.plot(df_exp_contact.t, df_exp_contact.iloc[:, 6::6],
                               color='peru', linewidth=self.setpoint_lw)

        # Add legends to axes
        ax_x_cont_vel.legend(legend_txt_x, fontsize=self.font_legend)
        ax_y_cont_vel.legend(legend_txt_y, fontsize=self.font_legend)
        ax_z_cont_vel.legend(legend_txt_z, fontsize=self.font_legend)

        fig_cont_vel.tight_layout()

        return fig_cont_vel
    
    def plot_ctl_torque(self, control_df) -> plt.figure:
        '''
        Plot uav control body-fixed torques and thrust
        '''

        # Plot commanded body torques and total thrust
        fig_T, (ax_Th, ax_tau_x, ax_tau_y, ax_tau_z) = plt.subplots(4, 1, 
            figsize=(self.fig_scale*self.fig_x_size, 
                     self.fig_scale*self.fig_y_size), 
                     dpi=self.fig_dpi, sharex=True)
        fig_T.suptitle('Commanded Thrust and Body-Fixed Torques', 
                        fontsize=self.font_suptitle, fontweight='bold')

        # Thrust
        ax_Th.plot(control_df.t, control_df.Th, linewidth=self.plt_lw) 
        ax_Th.set_title('Total Thrust', fontsize=self.font_subtitle, 
                        fontweight='bold')
        ax_Th.set_ylabel('Force (N)', fontsize=self.font_axlabel, 
                         fontweight='bold')
        ax_Th.legend(['thrust'], fontsize=self.font_legend)
        ax_Th.grid()
        
        # Tau x
        ax_tau_x.plot(control_df.t, control_df.tau_x, linewidth=self.plt_lw) 
        ax_tau_x.set_title('X-Axis Torque', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_tau_x.set_ylabel('Moment (Nm)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_tau_x.legend([r'$\tau_x$'], fontsize=self.font_legend)
        ax_tau_x.grid()

        # Tau y
        ax_tau_y.plot(control_df.t, control_df.tau_y, linewidth=self.plt_lw) 
        ax_tau_y.set_title('Y-Axis Torque', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_tau_y.set_ylabel('Moment (Nm)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_tau_y.legend([r'$\tau_y$'], fontsize=self.font_legend)
        ax_tau_y.grid()
        
        # Tau z
        ax_tau_z.plot(control_df.t, control_df.tau_z, linewidth=self.plt_lw) 
        ax_tau_z.set_title('Z-Axis Torque', fontsize=self.font_subtitle,
                           fontweight='bold')
        ax_tau_z.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_tau_z.set_ylabel('Moment (Nm)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_tau_z.legend([r'$\tau_z$'], fontsize=self.font_legend)
        ax_tau_z.grid()

        fig_T.tight_layout()

        return fig_T

    def plot_ctl_rotor(self, control_df) -> plt.figure:
        '''
        Plot uav control commanded rotor velocities
        '''

        # Plot motor angular velocities
        fig_u, (ax_u1, ax_u2, ax_u3, ax_u4) = plt.subplots(4, 1, 
            figsize=(self.fig_scale*self.fig_x_size, 
                    self.fig_scale*self.fig_y_size), 
            dpi=self.fig_dpi, sharex=True)
        fig_u.suptitle('Commanded Motor Velocities', 
                       fontsize=self.font_suptitle, fontweight='bold')

        # Motor 1
        ax_u1.plot(control_df.t, 9.54927*control_df.u1, linewidth=self.plt_lw) 
        ax_u1.set_title('Motor 1', fontsize=self.font_subtitle, 
                        fontweight='bold')
        ax_u1.set_ylabel('Rot. (rpm)', fontsize=self.font_axlabel, 
                         fontweight='bold')
        ax_u1.legend(['$u_1$'], fontsize=self.font_legend)
        ax_u1.grid()
        
        # Motor 2
        ax_u2.plot(control_df.t, 9.54927*control_df.u2, linewidth=self.plt_lw) 
        ax_u2.set_title('Motor 2', fontsize=self.font_subtitle, 
                        fontweight='bold')
        ax_u2.set_ylabel('Rot. (rpm)', fontsize=self.font_axlabel, 
                         fontweight='bold')
        ax_u2.legend(['$u_2$'], fontsize=self.font_legend)
        ax_u2.grid()
        
        # Motor 3
        ax_u3.plot(control_df.t, 9.54927*control_df.u3, linewidth=self.plt_lw) 
        ax_u3.set_title('Motor 3', fontsize=self.font_subtitle, 
                        fontweight='bold')
        ax_u3.set_ylabel('Rot. (rpm)', fontsize=self.font_axlabel, 
                         fontweight='bold')
        ax_u3.legend(['$u_3$'], fontsize=self.font_legend)
        ax_u3.grid()

        # Motor 4
        ax_u4.plot(control_df.t, 9.54927*control_df.u4, linewidth=self.plt_lw) 
        ax_u4.set_title('Motor 4', fontsize=self.font_subtitle, 
                        fontweight='bold')
        ax_u4.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                         fontweight='bold')
        ax_u4.set_ylabel('Rot. (rpm)', fontsize=self.font_axlabel, 
                         fontweight='bold')
        ax_u4.legend(['$u_4$'], fontsize=self.font_legend)
        ax_u4.grid()
        
        fig_u.tight_layout()

        return fig_u
    
    def plot_position_G(self, df_dict) -> plt.figure:
        '''
        This method plots the UAV position in the G frame, relative to the 
        ground vehicle
        '''

        df_results = deepcopy(df_dict['results']) # Primary uav result df

        # Subtract G origin from positions in W
        df_results.x_uav = df_results.x_uav -  df_results.x_gv
        df_results.y_uav = df_results.y_uav -  df_results.y_gv
        df_results.z_uav = df_results.z_uav -  df_results.z_gv

        # Create new dataframe with uav position in G frame
        W_G_conv_uav = lambda row: pd.Series(Rot.from_quat([row.qx_gv, 
            row.qy_gv, row.qz_gv, row.qw_gv]).apply(np.array([row.x_uav, 
            row.y_uav, row.z_uav]), inverse=True))
        df_uav_pos_G = df_results.apply(W_G_conv_uav, axis=1)
        df_uav_pos_G.columns = ['x_uav', 'y_uav', 'z_uav']

        # Create figure, subplots, and overall title
        fig_pos, (ax_x_pos, ax_y_pos, ax_z_pos) = plt.subplots(3, 1, 
            figsize=(self.fig_scale*self.fig_x_size, 
                     self.fig_scale*self.fig_y_size), 
            dpi=self.fig_dpi, sharex=True)
        fig_pos.suptitle('UAV Position States in G', fontweight='bold', 
                         fontsize=self.font_suptitle)
        
        # Plot uav position states
        ax_x_pos.plot(df_results.t, df_uav_pos_G.x_uav, linewidth=self.plt_lw)  
        ax_y_pos.plot(df_results.t, df_uav_pos_G.y_uav, linewidth=self.plt_lw)
        ax_z_pos.plot(df_results.t, df_uav_pos_G.z_uav, linewidth=self.plt_lw) 
        legend_txt_x = ['$x_{uav}$']
        legend_txt_y = ['$y_{uav}$']
        legend_txt_z = ['$z_{uav}$']

        # Add formatting and legends to plots
        ax_x_pos.set_title('X Position', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_x_pos.set_ylabel('Position (m)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_x_pos.legend(legend_txt_x, fontsize=self.font_legend)
        ax_x_pos.grid()

        ax_y_pos.set_title('Y Position', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_y_pos.set_ylabel('Position (m)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_y_pos.legend(legend_txt_y, fontsize=self.font_legend)
        ax_y_pos.grid()

        ax_z_pos.set_title('Z Position', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_z_pos.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_z_pos.set_ylabel('Position (m)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_z_pos.legend(legend_txt_z, fontsize=self.font_legend)
        # ax_z_pos.set_ylim(0.5, -5)
        ax_z_pos.grid()
        
        fig_pos.tight_layout() # Remove whitespace
        
        return fig_pos

    def plot_lin_velocity_G(self, df_dict) -> plt.figure:
        '''
        This method plots the UAV linear velocity in the G frame, relative
        to the ground vehicle
        '''

        df_results = deepcopy(df_dict['results']) # Primary uav result df

        # Create new dataframe with gv velocity in W frame
        G_W_conv_gv = lambda row: pd.Series(Rot.from_quat([row.qx_gv, 
            row.qy_gv, row.qz_gv, row.qw_gv]).apply(np.array([row.dx_gv, 
            row.dy_gv, row.dz_gv])))
        df_gv_vel_W = df_results.apply(G_W_conv_gv, axis=1)
        df_gv_vel_W.columns = ['dx_gv', 'dy_gv', 'dz_gv']

        # Subtract G origin velocity from velocities in W
        df_results.dx_uav = df_results.dx_uav -  df_gv_vel_W.dx_gv
        df_results.dy_uav = df_results.dy_uav -  df_gv_vel_W.dy_gv
        df_results.dz_uav = df_results.dz_uav -  df_gv_vel_W.dz_gv

        # Create new dataframe with uav velocities in G frame
        W_G_conv_uav = lambda row: pd.Series(Rot.from_quat([row.qx_gv, 
            row.qy_gv, row.qz_gv, row.qw_gv]).apply(np.array([row.dx_uav, 
            row.dy_uav, row.dz_uav]), inverse=True))
        df_uav_vel_G = df_results.apply(W_G_conv_uav, axis=1)
        df_uav_vel_G.columns = ['dx_uav', 'dy_uav', 'dz_uav']

        # Create figure, subplots, and overall title
        fig_vel, (ax_x_vel, ax_y_vel, ax_z_vel) = plt.subplots(3, 1, 
            figsize=(self.fig_scale*self.fig_x_size, 
                     self.fig_scale*self.fig_y_size), 
            dpi=self.fig_dpi, sharex=True)
        fig_vel.suptitle('UAV Velocity States in G', fontweight='bold', 
                         fontsize=self.font_suptitle)
        
        # Plot uav position states
        ax_x_vel.plot(df_results.t, df_uav_vel_G.dx_uav, linewidth=self.plt_lw)  
        ax_y_vel.plot(df_results.t, df_uav_vel_G.dy_uav, linewidth=self.plt_lw)
        ax_z_vel.plot(df_results.t, df_uav_vel_G.dz_uav, linewidth=self.plt_lw) 
        legend_txt_x = ['$x_{uav}$']
        legend_txt_y = ['$y_{uav}$']
        legend_txt_z = ['$z_{uav}$']

        # Add formatting and legends to plots
        ax_x_vel.set_title('X Velocity', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_x_vel.set_ylabel('Velocity (m/s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_x_vel.legend(legend_txt_x, fontsize=self.font_legend)
        ax_x_vel.grid()

        ax_y_vel.set_title('Y Velocity', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_y_vel.set_ylabel('Velocity (m/s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_y_vel.legend(legend_txt_y, fontsize=self.font_legend)
        ax_y_vel.grid()

        ax_z_vel.set_title('Z Velocity', fontsize=self.font_subtitle, 
                           fontweight='bold')
        ax_z_vel.set_xlabel('Time (s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_z_vel.set_ylabel('Velocity (m/s)', fontsize=self.font_axlabel, 
                            fontweight='bold')
        ax_z_vel.legend(legend_txt_z, fontsize=self.font_legend)
        ax_z_vel.invert_yaxis()
        ax_z_vel.grid()

        fig_vel.tight_layout()
            
        return fig_vel

    def animate_landing(self, results_df: pd.DataFrame, 
                        points_df: pd.DataFrame) -> animation:
        '''
        This method takes in the uav and gv trajectory csv and animates the 
        attempted landing in 3d
        '''

        # Initialize animation parameters
        tspan = (results_df.t.iloc[0], results_df.t.iloc[-1])
        timestep = tspan[1]/results_df.shape[0]
        sim_period = timestep
        sim_freq = 1/sim_period
        sim_length = tspan[1]
        sim_steps = int(sim_length*sim_freq)
        # sample_freq = sim_freq/self.video_freq
        num_frames = int(sim_length*self.video_freq/self.speed_scale)
        sample_idx = np.linspace(0, sim_steps-1, num_frames)
        sample_idx = np.around(sample_idx).astype(int)

        # Initialize 3d figure
        fig_3d = plt.figure()
        fig_3d.set_size_inches(self.fig_scale*self.fig_x_size, 
                               self.fig_scale*self.fig_y_size, True)
        fig_3d.set_dpi(self.ani_dpi)
        ax_3d = fig_3d.add_subplot(projection='3d')

        # Plot inertial frame W axis
        ax_3d.plot((0, self.axes_mag), (0, 0), (0, 0), 'r', 
                   linewidth=self.plt_lw, zorder=self.axes_zorder)
        ax_3d.plot((0, 0), (0, self.axes_mag), (0, 0), 'g', 
                   linewidth=self.plt_lw, zorder=self.axes_zorder)
        ax_3d.plot((0, 0), (0, 0), (0, self.axes_mag), 'b', 
                   linewidth=self.plt_lw, zorder=self.axes_zorder)
        ax_3d.text(-0.3, -0.1, 0, 'W', fontsize=self.font_txt, 
                   fontweight='bold', zorder=self.txt_zorder)
        
        # Compute needed rotations for ground vehicle
        q_G = np.array([results_df.qx_gv[0], results_df.qy_gv[0], 
                        results_df.qz_gv[0], results_df.qw_gv[0]])
        R_G_W = Rot.from_quat(q_G)

        # Plot contact zone
        cz0_W = np.array([results_df.x_gv[0], results_df.y_gv[0], 
                          results_df.z_gv[0]])
        cz1_G = np.array([self.contact_size[0]/2, self.contact_size[1]/2, 0])
        cz2_G = np.array([-self.contact_size[0]/2, self.contact_size[1]/2, 0])
        cz3_G = np.array([-self.contact_size[0]/2, -self.contact_size[1]/2, 0])
        cz4_G = np.array([self.contact_size[0]/2, -self.contact_size[1]/2, 0])
        cz1_W = cz0_W + R_G_W.apply(cz1_G)
        cz2_W = cz0_W + R_G_W.apply(cz2_G)
        cz3_W = cz0_W + R_G_W.apply(cz3_G)
        cz4_W = cz0_W + R_G_W.apply(cz4_G)

        cont_zone = np.array([cz1_W, cz2_W, cz3_W, cz4_W, cz1_W])[:, :2]

        dock = Polygon(cont_zone, closed=True, alpha=0.8, facecolor='y', 
                       edgecolor='k', zorder=1)
        ax_3d.add_patch(dock)
        art3d.pathpatch_2d_to_3d(dock, z=0, zdir="z")

        # Plot ground vehicle (contact zone) frame G axis
        G0_W = np.array([results_df.x_gv[0], results_df.y_gv[0], 
                        results_df.z_gv[0]])
        Gx_G = np.array([self.axes_mag, 0, 0])
        Gy_G = np.array([0, self.axes_mag, 0])
        Gz_G = np.array([0, 0, self.axes_mag])
        Gx_W = G0_W + R_G_W.apply(Gx_G)
        Gy_W = G0_W + R_G_W.apply(Gy_G)
        Gz_W = G0_W + R_G_W.apply(Gz_G)

        l_g1 = ax_3d.plot((G0_W[0], Gx_W[0]), (G0_W[1], Gx_W[1]), 
                          (G0_W[2], Gx_W[2]), 'r', linewidth=self.plt_lw, 
                          zorder=self.axes_zorder)[0]
        l_g2 = ax_3d.plot((G0_W[0], Gy_W[0]), (G0_W[1], Gy_W[1]), 
                          (G0_W[2], Gy_W[2]), 'g', linewidth=self.plt_lw, 
                          zorder=self.axes_zorder)[0]
        l_g3 = ax_3d.plot((G0_W[0], Gz_W[0]), (G0_W[1], Gz_W[1]), 
                          (G0_W[2], Gz_W[2]), 'b', linewidth=self.plt_lw, 
                          zorder=self.axes_zorder)[0]
        t_g1 = ax_3d.text(results_df.x_gv[0] - 0.3, results_df.y_gv[0] - 0.1, 
                          results_df.z_gv[0] + 0, 'G', fontsize=self.font_txt, 
                          fontweight='bold', zorder=self.txt_zorder)
        
        # Plot simple UAV shape - contact points
        lg_arm1_x = np.array([results_df.x_uav[0], points_df.pa_x[0]])
        lg_arm1_y = np.array([results_df.y_uav[0], points_df.pa_y[0]])
        lg_arm1_z = np.array([results_df.z_uav[0], points_df.pa_z[0]])
        lg_arm2_x = np.array([results_df.x_uav[0], points_df.pb_x[0]])
        lg_arm2_y = np.array([results_df.y_uav[0], points_df.pb_y[0]])
        lg_arm2_z = np.array([results_df.z_uav[0], points_df.pb_z[0]])
        lg_arm3_x = np.array([results_df.x_uav[0], points_df.pc_x[0]])
        lg_arm3_y = np.array([results_df.y_uav[0], points_df.pc_y[0]])
        lg_arm3_z = np.array([results_df.z_uav[0], points_df.pc_z[0]])
        lg_arm4_x = np.array([results_df.x_uav[0], points_df.pd_x[0]])
        lg_arm4_y = np.array([results_df.y_uav[0], points_df.pd_y[0]])
        lg_arm4_z = np.array([results_df.z_uav[0], points_df.pd_z[0]])
        l_c1 = ax_3d.plot(lg_arm1_x, lg_arm1_y, lg_arm1_z, 'k', 
                          linewidth=self.plt_lw, zorder=self.uav_zorder)[0] 
        l_c2 = ax_3d.plot(lg_arm2_x, lg_arm2_y, lg_arm2_z, 'k', 
                          linewidth=self.plt_lw, zorder=self.uav_zorder)[0] 
        l_c3 = ax_3d.plot(lg_arm3_x, lg_arm3_y, lg_arm3_z, 'k', 
                          linewidth=self.plt_lw, zorder=self.uav_zorder)[0] 
        l_c4 = ax_3d.plot(lg_arm4_x, lg_arm4_y, lg_arm4_z, 'k', 
                          linewidth=self.plt_lw, zorder=self.uav_zorder)[0] 

        # Plot simple UAV shape - UAV cross-arms
        q = np.array([results_df.qx_uav[0], results_df.qy_uav[0], 
                    results_df.qz_uav[0], results_df.qw_uav[0]])
        R_Q_W = Rot.from_quat(q)
        uav_arm1_x = np.array([results_df.x_uav[0] + 
                                   R_Q_W.apply(self.uav_D[0])[0], 
                               results_df.x_uav[0] + 
                                   R_Q_W.apply(self.uav_D[2])[0]])
        uav_arm1_y = np.array([results_df.y_uav[0] + 
                                   R_Q_W.apply(self.uav_D[0])[1], 
                               results_df.y_uav[0] + 
                                   R_Q_W.apply(self.uav_D[2])[1]])
        uav_arm1_z = np.array([results_df.z_uav[0] + 
                                   R_Q_W.apply(self.uav_D[0])[2], 
                               results_df.z_uav[0] + 
                                   R_Q_W.apply(self.uav_D[2])[2]])
        uav_arm2_x = np.array([results_df.x_uav[0] + 
                                   R_Q_W.apply(self.uav_D[1])[0], 
                               results_df.x_uav[0] + 
                                   R_Q_W.apply(self.uav_D[3])[0]])
        uav_arm2_y = np.array([results_df.y_uav[0] + 
                                   R_Q_W.apply(self.uav_D[1])[1], 
                               results_df.y_uav[0] + 
                                   R_Q_W.apply(self.uav_D[3])[1]])
        uav_arm2_z = np.array([results_df.z_uav[0] + 
                                   R_Q_W.apply(self.uav_D[1])[2], 
                               results_df.z_uav[0] + 
                                   R_Q_W.apply(self.uav_D[3])[2]])
        l_u1 = ax_3d.plot(uav_arm1_x, uav_arm1_y, uav_arm1_z, 'dimgrey', 
                        linewidth=self.plt_lw, zorder=self.uav_zorder)[0] 
        l_u2 = ax_3d.plot(uav_arm2_x, uav_arm2_y, uav_arm2_z, 'dimgrey', 
                        linewidth=self.plt_lw, zorder=self.uav_zorder)[0]
        
        # Plot grey trace of uav com
        l_u3 = ax_3d.plot(results_df.x_uav[0], results_df.y_uav[0], 
                          results_df.z_uav[0], 'lightgrey', alpha=0.5,
                          linewidth=self.plt_lw, zorder=self.uav_zorder)[0] 

        # Add timescale information to the plot
        timescale_msg = "Playback Speed: %0.2fx" % self.speed_scale
        timestamp_msg = "t = %0.5fs" % results_df.t[0]
        t_t1 = ax_3d.text2D(0.9, 0.025, timescale_msg, 
                            transform=ax_3d.transAxes, 
                            fontsize=self.font_txt, fontweight='bold', 
                            zorder=self.txt_zorder)
        t_t2 = ax_3d.text2D(0.9, 0.05, timestamp_msg, transform=ax_3d.transAxes, 
                            fontsize=self.font_txt, fontweight='bold', 
                            zorder=self.txt_zorder)

        # Compile lines into a list
        lines = [l_c1, l_c2, l_c3, l_c4, # Contact point lines
                l_u1, l_u2, l_u3, # UAV crossbar lines and com
                l_g1, l_g2, l_g3, t_g1, # G-frame axes and text
                t_t1, t_t2, # Timestamp text
                dock] # Dock polygon
        
        # 3D plot axis labels
        ax_3d.set_xlabel('X-Axis (m)', fontsize=self.font_axlabel, 
                         fontweight='bold', labelpad=15)
        ax_3d.set_ylabel('Y-Axis (m)', fontsize=self.font_axlabel, 
                         fontweight='bold')
        ax_3d.set_zlabel('Z-Axis (m)', fontsize=self.font_axlabel, 
                         fontweight='bold')
        
        # Compute 3d plot limits
        circ_rad = np.sqrt(2*(self.contact_size[0]/2)**2)
        addition_val = max((self.axes_mag, circ_rad))

        # x-axis limits
        x_max_uav = results_df.x_uav.max()
        x_min_uav = results_df.x_uav.min()
        x_max_gv = results_df.x_gv.max()
        x_min_gv = results_df.x_gv.min()
        x_max_data = max((x_max_gv, x_max_uav))
        x_min_data = min((x_min_gv, x_min_uav))

        x_max = 1.05*(x_max_data + addition_val)
        x_min = 1.05*(x_min_data - addition_val)

        # y-axis limits
        y_max_uav = results_df.y_uav.max()
        y_min_uav = results_df.y_uav.min()
        y_max_gv = results_df.y_gv.max()
        y_min_gv = results_df.y_gv.min()
        y_max_data = max((y_max_gv, y_max_uav))
        y_min_data = min((y_min_gv, y_min_uav))

        y_max = 1.05*(y_max_data + addition_val)
        y_min = 1.05*(y_min_data - addition_val)

        # z-axis limits
        # z_max_uav = results_df.z_uav.max()
        z_min_uav = results_df.z_uav.min()
        # z_max_gv = results_df.z_gv.max()
        z_min_gv = results_df.z_gv.min()
        # z_max_data = max((z_max_gv, z_max_uav))
        z_min_data = min((z_min_gv, z_min_uav))

        z_max = 0
        z_min = 1.05*(z_min_data - addition_val)

        # Set 3d animation plot limits
        x_limits = (x_min, x_max)
        y_limits = (y_min, y_max)
        z_limits = (z_min, z_max)

        ax_3d.set_xlim(x_limits)
        ax_3d.set_ylim(y_limits)
        ax_3d.set_zlim(z_limits)

        x_aspect = x_max - x_min
        y_aspect = y_max - y_min
        z_aspect = z_max - z_min
        ax_3d.set_box_aspect([x_aspect, y_aspect, z_aspect])
        ax_3d.set_aspect('equal', adjustable='box')

        ax_3d.invert_xaxis()
        ax_3d.invert_zaxis()
        ax_3d.view_init(elev=30, azim=45)
        fig_3d.tight_layout()

        # Create animation
        ani = animation.FuncAnimation(fig_3d, self.update_lines, num_frames, 
            fargs=(lines, results_df, points_df, sample_idx, ax_3d), interval=1)
        
        return ani, fig_3d
    
    def update_lines(self, i: int, lines: list, results_df: pd.DataFrame, 
        points_df: pd.DataFrame, sample_idx: np.ndarray, ax_3d):
        '''
        Method used to update the lines in the 3d animation of the UAV flying
        and landing
        '''

        df_i = sample_idx[i] # Get dataframe index for this iteration subsample

        print('Animating frame: %i / %i, df row: %i\r' % \
              (i, len(sample_idx)-1, df_i), end="")

        # Update landing gear lines
        lg_arm1_x = np.array([results_df.x_uav[df_i], points_df.pa_x[df_i]])
        lg_arm1_y = np.array([results_df.y_uav[df_i], points_df.pa_y[df_i]])
        lg_arm1_z = np.array([results_df.z_uav[df_i], points_df.pa_z[df_i]])
        lg_arm2_x = np.array([results_df.x_uav[df_i], points_df.pb_x[df_i]])
        lg_arm2_y = np.array([results_df.y_uav[df_i], points_df.pb_y[df_i]])
        lg_arm2_z = np.array([results_df.z_uav[df_i], points_df.pb_z[df_i]])
        lg_arm3_x = np.array([results_df.x_uav[df_i], points_df.pc_x[df_i]])
        lg_arm3_y = np.array([results_df.y_uav[df_i], points_df.pc_y[df_i]])
        lg_arm3_z = np.array([results_df.z_uav[df_i], points_df.pc_z[df_i]])
        lg_arm4_x = np.array([results_df.x_uav[df_i], points_df.pd_x[df_i]])
        lg_arm4_y = np.array([results_df.y_uav[df_i], points_df.pd_y[df_i]])
        lg_arm4_z = np.array([results_df.z_uav[df_i], points_df.pd_z[df_i]])

        lines[0].set_data(np.array([lg_arm1_x, lg_arm1_y]))
        lines[0].set_3d_properties(lg_arm1_z)

        lines[1].set_data(np.array([lg_arm2_x, lg_arm2_y]))
        lines[1].set_3d_properties(lg_arm2_z)

        lines[2].set_data(np.array([lg_arm3_x, lg_arm3_y]))
        lines[2].set_3d_properties(lg_arm3_z)

        lines[3].set_data(np.array([lg_arm4_x, lg_arm4_y]))
        lines[3].set_3d_properties(lg_arm4_z)

        # Update uav cross lines
        q = np.array([results_df.qx_uav[df_i], results_df.qy_uav[df_i], 
                      results_df.qz_uav[df_i], results_df.qw_uav[df_i]])
        R_Q_W = Rot.from_quat(q)
        uav_arm1_x = np.array([results_df.x_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[0])[0], 
                               results_df.x_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[2])[0]])
        uav_arm1_y = np.array([results_df.y_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[0])[1], 
                               results_df.y_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[2])[1]])
        uav_arm1_z = np.array([results_df.z_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[0])[2], 
                               results_df.z_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[2])[2]])
        uav_arm2_x = np.array([results_df.x_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[1])[0], 
                               results_df.x_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[3])[0]])
        uav_arm2_y = np.array([results_df.y_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[1])[1], 
                               results_df.y_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[3])[1]])
        uav_arm2_z = np.array([results_df.z_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[1])[2], 
                               results_df.z_uav[df_i] + 
                                   R_Q_W.apply(self.uav_D[3])[2]])
        
        lines[4].set_data(np.array([uav_arm1_x, uav_arm1_y]))
        lines[4].set_3d_properties(uav_arm1_z)

        lines[5].set_data(np.array([uav_arm2_x, uav_arm2_y]))
        lines[5].set_3d_properties(uav_arm2_z)

        # Compute needed rotations for ground vehicle
        q_G = np.array([results_df.qx_gv[df_i], results_df.qy_gv[df_i], 
                        results_df.qz_gv[df_i], results_df.qw_gv[df_i]])
        R_G_W = Rot.from_quat(q_G)

        # Update the G coordinate frame
        G0_W = np.array([results_df.x_gv[df_i], results_df.y_gv[df_i], 
                        results_df.z_gv[df_i]])
        Gx_G = np.array([self.axes_mag, 0, 0])
        Gy_G = np.array([0, self.axes_mag, 0])
        Gz_G = np.array([0, 0, self.axes_mag])
        Gx_W = G0_W + R_G_W.apply(Gx_G)
        Gy_W = G0_W + R_G_W.apply(Gy_G)
        Gz_W = G0_W + R_G_W.apply(Gz_G)

        G_txt_x = results_df.x_gv[df_i] - 0.3
        G_txt_y = results_df.y_gv[df_i] - 0.1
        G_txt_z = results_df.z_gv[df_i] + 0

        lines[7].set_data(np.array([[G0_W[0], Gx_W[0]], [G0_W[1], Gx_W[1]]]))
        lines[7].set_3d_properties(np.array([G0_W[2], Gx_W[2]]))

        lines[8].set_data(np.array([[G0_W[0], Gy_W[0]], [G0_W[1], Gy_W[1]]]))
        lines[8].set_3d_properties(np.array([G0_W[2], Gy_W[2]]))

        lines[9].set_data(np.array([[G0_W[0], Gz_W[0]], [G0_W[1], Gz_W[1]]]))
        lines[9].set_3d_properties(np.array([G0_W[2], Gz_W[2]]))

        lines[10].set_position(np.array([G_txt_x, G_txt_y, G_txt_z]))

        # Update the com trace
        lines[6].set_data(np.array([results_df.x_uav[:df_i], 
                                    results_df.y_uav[:df_i]]))
        lines[6].set_3d_properties(results_df.z_uav[:df_i])

        # Update the ground vehicle patch
        dock = lines[-1]
        cz0_W = np.array([results_df.x_gv[df_i], results_df.y_gv[df_i], 
                          results_df.z_gv[df_i]])
        cz1_G = np.array([self.contact_size[0]/2, self.contact_size[1]/2, 0])
        cz2_G = np.array([-self.contact_size[0]/2, self.contact_size[1]/2, 0])
        cz3_G = np.array([-self.contact_size[0]/2, -self.contact_size[1]/2, 0])
        cz4_G = np.array([self.contact_size[0]/2, -self.contact_size[1]/2, 0])
        cz1_W = cz0_W + R_G_W.apply(cz1_G)
        cz2_W = cz0_W + R_G_W.apply(cz2_G)
        cz3_W = cz0_W + R_G_W.apply(cz3_G)
        cz4_W = cz0_W + R_G_W.apply(cz4_G)
        cont_zone = np.array([cz1_W, cz2_W, cz3_W, cz4_W, cz1_W])[:, :2]

        dock.remove()
        dock = Polygon(cont_zone, closed=True, alpha=0.8, facecolor='y', 
                       edgecolor='k', zorder=1)
        ax_3d.add_patch(dock)
        art3d.pathpatch_2d_to_3d(dock, z=0, zdir="z")
        lines[-1] = dock

        # Update the timestamp text
        timestamp_msg = "t = %0.5fs" % results_df.t[df_i]
        lines[12].set_text(timestamp_msg)

        return lines
    
# ----------------------- Premade Command Dictionaries ----------------------- #

# Commands per field generally are: 'all', 'plot', 'save', 'none'
# gv_plot field: 'full_state', 'pos_states', 'none'
# setpoint_plot field: True,ß False

plot_cmd_dict_save = { 
    'position': 'save',
    'lin_velocity': 'save',
    'quaternion': 'save',
    'euler': 'save',
    'ang_velocity': 'save',
    'cont_position': 'save',
    'cont_velocity': 'save',
    'ctl_torques': 'save',
    'ctl_rotors': 'save',
    'position_G': 'save',
    'lin_velocity_G': 'save',
    'animation': 'save',
    'gv_plot': 'full_state', 
    'setpoint_plot': True,
    'rm_gv': False
}

plot_cmd_dict_plot = {
    'position': 'plot',
    'lin_velocity': 'plot',
    'quaternion': 'plot',
    'euler': 'plot',
    'ang_velocity': 'plot',
    'cont_position': 'plot',
    'cont_velocity': 'plot',
    'ctl_torques': 'plot',
    'ctl_rotors': 'plot',
    'position_G': 'plot',
    'lin_velocity_G': 'plot',
    'animation': 'plot',
    'gv_plot': 'full_state', 
    'setpoint_plot': True,
    'rm_gv': False
}

plot_cmd_dict_all = { 
    'position': 'all',
    'lin_velocity': 'all',
    'quaternion': 'all',
    'euler': 'all',
    'ang_velocity': 'all',
    'cont_position': 'all',
    'cont_velocity': 'all',
    'ctl_torques': 'all',
    'ctl_rotors': 'all',
    'position_G': 'all',
    'lin_velocity_G': 'all',
    'animation': 'all',
    'gv_plot': 'full_state', 
    'setpoint_plot': True,
    'rm_gv': False
}

plot_cmd_dict_none = {
    'position': 'none',
    'lin_velocity': 'none',
    'quaternion': 'none',
    'euler': 'none',
    'ang_velocity': 'none',
    'cont_position': 'none',
    'cont_velocity': 'none',
    'ctl_torques': 'none',
    'ctl_rotors': 'none',
    'position_G': 'none',
    'lin_velocity_G': 'none',
    'animation': 'none',
    'gv_plot': 'full_state', 
    'setpoint_plot': True,
    'rm_gv': False
}






