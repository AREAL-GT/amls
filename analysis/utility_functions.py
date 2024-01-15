
'''
Module contains useful functions for the amls.analysis package
'''

# ---------------------------------- Imports --------------------------------- #

# Standard imports
from datetime import date, datetime
import os
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
from math import acos

# Add package and workspace directories to the path
import os
import sys
sim_pkg_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sim_pkg_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.insert(0, sim_pkg_path1)
sys.path.insert(0, sim_pkg_path2)

# Workspace package imports
from amls_impulse_contact import utility_functions as uf_cont


# --------------------------------- Functions -------------------------------- #

def create_contact_data(results_df: np.ndarray, \
                        points_B: np.ndarray) -> np.ndarray:
    '''
    Function takes in body-fixed collision locations along with body state
    information and creates a dataframe of the individual contact point 
    positions and velocities as an output
    '''

    # Create initial empty dataframe
    name_list_points = uf_cont.point_df_name(points_B.shape[0])
    points_df = pd.DataFrame(columns=name_list_points, index=results_df.index)
    points_df.t = results_df.t

    # Loop through UAV results df and generate contact data
    for i in range(len(results_df.t)): # For each uav position

        # Position and orientation
        uav_pos_W = results_df.iloc[i, 1:4].to_numpy().astype(float)
        uav_q = results_df.iloc[i, 4:8].to_numpy().astype(float)
        R_Q_W = Rot.from_quat(uav_q)

        # Linear and angular velocity
        uav_vel_W = results_df.iloc[i, 8:11].to_numpy().astype(float)
        uav_angvel_Q = results_df.iloc[i, 11:14].to_numpy().astype(float)

        for ii in range(points_B.shape[0]): # For each contact point

            # Compute contact point position
            point_ii_Q = points_B[ii].astype(float)
            point_ii_pos_W = uav_pos_W + R_Q_W.apply(point_ii_Q)
            points_df.iloc[i, (6*ii + 1):(6*ii + 4)] = point_ii_pos_W

            # Compute contact point velocity
            cross_term = np.cross(uav_angvel_Q, point_ii_Q)
            vel_ii_W = uav_vel_W + R_Q_W.apply(cross_term)

            if ii < points_B.shape[0]:
                points_df.iloc[i, (6*ii + 4):(6*ii + 7)] = vel_ii_W
            else:
                points_df.iloc[i, (6*ii + 4):] = vel_ii_W

    return points_df

def add_euler(results_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Function takes in a results dataframe with 13 states for either/both uav
    and gv. It adds euler angle orientation columns after the quaternion 
    orientations for both/either uav and gv
    '''

    num_cols = results_df.shape[1]

    # UAV Euler angle section
    # Compute uav euler angles from quaternions
    eul_conv_uav = lambda row: pd.Series(Rot.from_quat([row.qx_uav, row.qy_uav, 
        row.qz_uav, row.qw_uav]).as_euler('xyz', degrees=True))
    euler_uav_df = results_df.apply(eul_conv_uav, axis=1)
    euler_uav_df.columns = ['phi_uav', 'theta_uav', 'psi_uav']

    # Insert the euler angles into the results dataframe
    results_df.insert(8, 'phi_uav', euler_uav_df.phi_uav.values)
    results_df.insert(9, 'theta_uav', euler_uav_df.theta_uav.values)
    results_df.insert(10, 'psi_uav', euler_uav_df.psi_uav.values)

    # GV Euler angle section

    if num_cols > 17:

        # Compute gv euler angles from quaternions
        eul_conv_gv = lambda row: pd.Series(Rot.from_quat([row.qx_gv, row.qy_gv, 
            row.qz_gv, row.qw_gv]).as_euler('xyz', degrees=True))
        euler_gv_df = results_df.apply(eul_conv_gv, axis=1)
        euler_gv_df.columns = ['phi_gv', 'theta_gv', 'psi_gv']

        # Insert the euler angles into the results dataframe
        results_df.insert(24, 'phi_gv', euler_gv_df.phi_gv.values)
        results_df.insert(25, 'theta_gv', euler_gv_df.theta_gv.values)
        results_df.insert(26, 'psi_gv', euler_gv_df.psi_gv.values)

    return results_df

def add_euler_setpoint(setpoint_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Function takes in a setpoint dataframe with 13 states for the uav.
    It adds euler angle orientation columns after the quaternion orientations
    for the uav.
    '''

    # Compute uav setpoint euler angles from quaternions
    eul_conv = lambda row: pd.Series(Rot.from_quat([row.qx, row.qy, 
        row.qz, row.qw]).as_euler('xyz', degrees=True))
    euler_df = setpoint_df.apply(eul_conv, axis=1)
    euler_df.columns = ['phi', 'theta', 'psi']

    # Insert the euler angles into the results dataframe
    setpoint_df.insert(8, 'phi', euler_df.phi.values)
    setpoint_df.insert(9, 'theta', euler_df.theta.values)
    setpoint_df.insert(10, 'psi', euler_df.psi.values)

    return setpoint_df

def add_euler_quat(df_quat: pd.DataFrame) -> pd.DataFrame:
    '''
    Function takes in a dataframe with columns qx, qy, qz, qw, and adds 
    euler angles in 3 following columns
    '''

    eul_conv_uav = lambda row: pd.Series(Rot.from_quat([row[1], row[2], 
        row[3], row[4]]).as_euler('xyz', degrees=True))
    df_euler = df_quat.apply(eul_conv_uav, axis=1)
    df_euler.columns = ['phi', 'theta', 'psi']

    df_attitude = pd.concat([df_quat, df_euler], axis=1)

    return df_attitude

def add_euler_inplace(df: pd.DataFrame, qx_name: str = 'qx') -> None:
    '''
    Function takes an input dataframe containing a quaternion orientation and
    adds euler angles in columns directly following the quaternions
    '''

    df_cols = df.columns.values.tolist()
    qx_idx = df_cols.index(qx_name)
    eul_conv_uav = lambda row: pd.Series(Rot.from_quat([row[qx_idx], 
        row[qx_idx+1], row[qx_idx+2], row[qx_idx+3]]).as_euler('xyz'))
    df_euler = df.apply(eul_conv_uav, axis=1)
    df.insert(qx_idx+4, 'phi', df_euler.iloc[:, 0].values)
    df.insert(qx_idx+5, 'theta', df_euler.iloc[:, 1].values)
    df.insert(qx_idx+6, 'psi', df_euler.iloc[:, 2].values)

def result_dir_setup(batch_name: str, gv_filename: str, uav_filename: str) -> \
    tuple[str, list]:
    '''
    Function to perform results directory setup for batch amls runs. The 
    setup entails:
    - Determining the root file name for this batch run
    - Creating the directory for this batch run if it does not exist
    - Creating the summary text file list and filling out first few lines
    '''

    # Determine root name for this batch
    root_name, run_number = root_name_set(batch_name)

    # Create results directory for this batch run
    result_dir = os.path.dirname(__file__) + '/Simulation Results/' + root_name
    os.mkdir(result_dir)

    # Create summary list and first entries
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    date_str = today.strftime("%Y_%m_%d")
    date_time_msg = 'Run Date: ' + date_str + ', Time of Run: ' + current_time
    batch_msg = 'Batch name and run #: ' + batch_name + ' run %i' % run_number
    uav_msg = 'UAV parameter file: ' + uav_filename
    gv_msg = 'GV parameter file: ' + gv_filename
    txt_list = []
    txt_list.append(date_time_msg)
    txt_list.append(batch_msg)
    txt_list.append(uav_msg)
    txt_list.append(gv_msg)

    return root_name, txt_list

def root_name_set(batch_name: str) -> tuple[str, int]:
    '''
    Function to determine the root file name for this batch run. The root file
    name is "YYYY_MM_DD_[BATCH NAME]_#" where # is incremented each run based
    on the other results in the directory
    '''

    # Form initial component of root name string: YYYY_MM_DD_[BATCH NAME]
    today = date.today()
    date_str = today.strftime("%Y_%m_%d")
    root_name = date_str + '_' + batch_name

    # Get contents of results path directory
    result_dir = os.path.dirname(__file__) + '/Simulation Results/'
    result_dir_contents = os.listdir(result_dir)

    # Determine index number for this batch run
    if any(root_name in cont for cont in result_dir_contents):

        # Select the folder names that match this date
        matching_names = [cont for cont in result_dir_contents if 
                          root_name in cont]

        # Isolate the index numbers for these folder names
        index_str = [sub.replace(root_name + '_', '') for 
                     sub in matching_names]
        
        indexes = [eval(i) for i in index_str] # Convert to int

        batch_idx = max(indexes) + 1 # Increment by 1

    else: # This is the first result folder

        batch_idx = 0 # Start at 0

    # Set root name for this run
    root_name += '_' + str(batch_idx)

    return root_name, batch_idx

def body_fixed_z_adjust(pos_data: pd.DataFrame, z_adj: float, 
                        quat_data: pd.DataFrame) -> pd.Series:
    '''
    This function takes in inertial frame z data along with orientation data, 
    and applies a body-fixed offset to that z data
    '''

    # Create adjustment vector
    z_adj_Q = np.array([0, 0, z_adj])

    # Loop through all z data and apply the offset
    for i in range(pos_data.shape[0]):
        
        R = Rot.from_quat(quat_data.iloc[i, :].values) # Q -> W

        z_adj_W = R.apply(z_adj_Q)

        pos_data.iloc[i] += z_adj_W

    return pos_data

def px4_log_clean(raw_dir: str, clean_dir: str):
    '''
    This function takes in the path and filename of a px4 log in .csv format
    and cleans it. Isolating the desired columns, removing Nan values, and
    performing basic transformations to append some additional data. The
    cleaned data is then saved in a .csv of the desired name/directory. The 
    data from VehicleLocalPosition will be treated as primary in terms of
    timestamps. If data from other topics has different timestamps, a linear
    interpolation is used to match the times from VehicleLocalPosition. 
    
    The data columns in the cleaned .csv will be:
    NED Frame:
    - Position
    - Velocity
    - Acceleration
    - Quaternion Orientation
    - Euler Angles
    Rotated to UAV Initial Heading FRD Frame:
    - Position
    - Velocity
    - Acceleration
    - Quaternion Orientation
    - Euler Angles
    Other:
    - Linear x/y norm velocity
    - Heading Angle
    - Alpha angle of inclination from vertical axis
    - Actuator Commands
    '''

    # Import raw data log from given directory
    df_raw = pd.read_csv(raw_dir)

    # Isolate desired columns from raw dataframe
    desired_cols = ['__time', 
                    'vehicle_local_position/x', 
                    'vehicle_local_position/y', 
                    'vehicle_local_position/z',
                    'vehicle_local_position/vx', 
                    'vehicle_local_position/vy', 
                    'vehicle_local_position/vz', 
                    'vehicle_local_position/ax', 
                    'vehicle_local_position/ay',
                    'vehicle_local_position/az',
                    'vehicle_attitude/q.01', 
                    'vehicle_attitude/q.02', 
                    'vehicle_attitude/q.03', 
                    'vehicle_attitude/q.00',
                    'vehicle_local_position/heading',
                    'actuator_motors/control.00', 
                    'actuator_motors/control.01',
                    'actuator_motors/control.02', 
                    'actuator_motors/control.03']
    col_names = ['t', 'n', 'e', 'd', 'vn', 've', 'vd', 'an', 'ae', 'ad', 
                 'qx', 'qy', 'qz', 'qw', 'heading', 'w1', 'w2', 'w3', 'w4']
    df_clean = df_raw.loc[:, desired_cols]
    df_clean.columns = col_names

    # Fill non-primary columns with linear interpolation
    df_interp = df_clean.iloc[:, 10:].interpolate()
    df_clean.iloc[:, 10:] = df_interp

    # Drop rows with nan (in primary columns)
    df_clean.dropna(inplace=True)
    df_clean.reset_index(inplace=True, drop=True)

    # Add linear velocity to dataframe
    vel_norm = lambda row: pd.Series(LA.norm((row.vn, row.ve)))
    df_vel_mag = df_clean.apply(vel_norm, axis=1)
    df_clean.insert(15, 'v_lin', df_vel_mag.values)

    # Add euler angles to dataframe
    add_euler_inplace(df_clean)

    # Create ned to initial uav heading rotations
    heading_init = df_clean.loc[0, 'heading']
    R_rot_NED = Rot.from_euler('xyz', [0, 0, heading_init], degrees=True)
    R_NED_rot = R_rot_NED.inv()
        
    # Add rotated position to dataframe
    ned_conv_pos = lambda row: pd.Series(R_NED_rot.apply([row.n, row.e, row.d]))
    df_pos_rot = df_clean.apply(ned_conv_pos, axis=1)
    df_clean.insert(18, 'x', df_pos_rot.iloc[:, 0].values)
    df_clean.insert(19, 'y', df_pos_rot.iloc[:, 1].values)
    df_clean.insert(20, 'z', df_pos_rot.iloc[:, 2].values)

    # Add rotated velocity to dataframe
    ned_conv_vel = lambda row: pd.Series(R_NED_rot.apply([row.vn, row.ve, 
                                                        row.vd]))
    df_vel_rot = df_clean.apply(ned_conv_vel, axis=1)
    df_clean.insert(21, 'vx', df_vel_rot.iloc[:, 0].values)
    df_clean.insert(22, 'vy', df_vel_rot.iloc[:, 1].values)
    df_clean.insert(23, 'vz', df_vel_rot.iloc[:, 2].values)

    # Add rotated acceleration to dataframe
    ned_conv_accel = lambda row: pd.Series(R_NED_rot.apply([row.an, row.ae, 
                                                          row.ad]))
    df_accel_rot = df_clean.apply(ned_conv_accel, axis=1)
    df_clean.insert(24, 'ax', df_accel_rot.iloc[:, 0].values)
    df_clean.insert(25, 'ay', df_accel_rot.iloc[:, 1].values)
    df_clean.insert(26, 'az', df_accel_rot.iloc[:, 2].values)

    # Add quaternions and eulers with respect to rotated frame to dataframe
    df_clean.insert(27, 'qx_rot', '')
    df_clean.insert(28, 'qy_rot', '')
    df_clean.insert(29, 'qz_rot', '')
    df_clean.insert(30, 'qw_rot', '')
    df_clean.insert(31, 'phi_rot', '')
    df_clean.insert(32, 'theta_rot', '')
    df_clean.insert(33, 'psi_rot', '')
    df_clean.insert(35, 'alpha', '')
    for i in range(df_clean.shape[0]):
        row = df_clean.iloc[i, :]
        R_Q_NED = Rot.from_quat([row.qx, row.qy, row.qz, row.qw])
        # R_NED_Q = R_Q_NED.inv()
        R_Q_rot = R_NED_rot*R_Q_NED
        q_rot = R_Q_rot.as_quat()
        eul_rot = R_Q_rot.as_euler('xyz')
        df_clean.iloc[i, 27:31] = q_rot
        df_clean.iloc[i, 31:34] = eul_rot

        # Add alpha tilt angle from vertical
        zQ_Q = np.array([0, 0, 1])
        z_NED = np.array([0, 0, 1])
        zQ_NED = R_Q_NED.apply(zQ_Q)
        alpha = acos(z_NED@zQ_NED)
        df_clean.loc[i, 'alpha'] = alpha

    # Save cleaned dataframe as .csv to desired directory
    df_clean.to_csv(clean_dir, index=False)

def px4_dir_clean(px4_dir: str):
    '''
    This function takes in a directory and calls the px4 log cleaning 
    function on all of the rawlog files in it. The cleaned .csv are saved
    with the cleanlog_... file name.
    '''

    # Get the contents of the given directory
    result_dir_contents = os.listdir(px4_dir)

    # Loop through all directory contents containing rawlog
    raw_files = [file for file in result_dir_contents if 'rawlog' in file]
    for file in tqdm(raw_files, 'Cleaning PX4 Log Directory'):

        input_file = px4_dir + '/' + file

        output_name = 'cleanlog' + file[6:]
        output_file =  px4_dir + '/' + output_name

        print("Cleaning file: " + file)

        # Call the px4 cleaning function
        px4_log_clean(input_file, output_file)


























