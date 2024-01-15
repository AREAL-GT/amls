
import numpy as np

class PX4Parameters:
    '''
    Class used to organize parameters for the generic PX4 controller framework
    '''

    def __init__(self) -> None:

        # Position controller parameters
        self.pos_x_P: float = None # x-position (P)ID gain
        self.pos_y_P: float = None # y-position (P)ID gain
        self.pos_z_P: float = None # z-position (P)ID gain
        self.pos_x_I: float = None # x-position P(I)D gain
        self.pos_y_I: float = None # y-position P(I)D gain
        self.pos_z_I: float = None # z-position P(I)D gain
        self.pos_x_D: float = None # x-position PI(D) gain
        self.pos_y_D: float = None # y-position PI(D) gain
        self.pos_z_D: float = None # z-position PI(D) gain
        self.pos_err_int_lim: float = None # Position integral anti-windup

        # Velocity controller parameters
        self.vel_x_P: float = None # Linear x-velocity (P)ID gain
        self.vel_y_P: float = None # Linear y-velocity (P)ID gain
        self.vel_z_P: float = None # Linear z-velocity (P)ID gain
        self.vel_x_I: float = None # Linear x-velocity P(I)D gain
        self.vel_y_I: float = None # Linear y-velocity P(I)D gain
        self.vel_z_I: float = None # Linear z-velocity P(I)D gain
        self.vel_x_D: float = None # Linear x-velocity PI(D) gain
        self.vel_y_D: float = None # Linear y-velocity PI(D) gain
        self.vel_z_D: float = None # Linear z-velocity PI(D) gain
        self.vel_err_int_lim: float = None # Velocity integral anti-windup

        # Attitude controller parameters
        self.attitude_P: float = None # Attitude P gain
        self.attitude_time_const: float = None # Time constant for att control
        
        # Angular rate controller parameters
        self.ang_rate_x_P: float = None # Angular rate x (roll) (P)ID gain
        self.ang_rate_x_I: float = None # Angular rate x (roll) P(I)D gain
        self.ang_rate_x_D: float = None # Angular rate x (roll) PI(D) gain
        self.ang_rate_y_P: float = None # Angular rate y (pitch) (P)ID gain
        self.ang_rate_y_I: float = None # Angular rate y (pitch) P(I)D gain
        self.ang_rate_y_D: float = None # Angular rate y (pitch) PI(D) gain
        self.ang_rate_z_P: float = None # Angular rate z (yaw) (P)ID gain
        self.ang_rate_z_I: float = None # Angular rate z (yaw) P(I)D gain
        self.ang_rate_z_D: float = None # Angular rate z (yaw) PI(D) gain
        self.ang_rate_err_int_lim: float = None # Angular rate int anti-windup

        # Frequency control parameters
        self.pos_freq: float = None # Frequency for position controller
        self.vel_freq: float = None # Frequency for velocity controller
        self.att_freq: float = None # Frequency for attitude controller
        self.ang_rate_freq: float = None # Freq for angular rate controller

class GeometricParameters:
    '''
    Class used to organize parameters for the geomteric controller framework
    '''

    def __init__(self) -> None:

        # Frequency control parameters
        self.geom_freq: float = None

        # Thrust controller parameters
        self.kx: np.ndarray = None
        self.kv: np.ndarray = None
        self.ki: np.ndarray = None
        self.sigma: np.ndarray = None
        self.c1: np.ndarray = None

        # m1m2 attitude controller parameters
        self.kb: np.ndarray = None
        self.kw: np.ndarray = None
        self.kI: np.ndarray = None
        self.sigma2: float = None
        self.c2: np.ndarray = None

        # m3 yaw attitude controller parameters
        self.ky: float = None
        self.kwy: float = None
        self.kIy: float = None
        self.c3: float = None

# ----------------------- Geometric Tuning from Paper ------------------------ #
geom_ctl_param_def = GeometricParameters()
geom_ctl_param_def.geom_freq = 200
geom_ctl_param_def.kx = np.array([4, 4, 4])
geom_ctl_param_def.kv = np.array([2.8, 2.8, 2.8])
geom_ctl_param_def.ki = np.array([2, 2, 2])
geom_ctl_param_def.sigma = np.array([10, 10, 10])
geom_ctl_param_def.c1 = np.array([1, 1, 1])
geom_ctl_param_def.kb = np.array([16, 16, 16])
geom_ctl_param_def.kw = np.array([5.6, 5.6, 5.6])
geom_ctl_param_def.kI = np.array([10, 10, 10])
geom_ctl_param_def.c2 = np.array([1, 1, 1])
geom_ctl_param_def.ky = 1
geom_ctl_param_def.kwy = 1.4
geom_ctl_param_def.kIy = 10
geom_ctl_param_def.c3 = 1

# ----------------------- Geometric Tuning for x500v2 ------------------------ #
geom_ctl_param_x500 = GeometricParameters()
geom_ctl_param_x500.geom_freq = 200

# Thrust controller gains
geom_ctl_param_x500.kx = np.array([65.0, 65.0, 30.0]) # Main gains (10/9/23)
# geom_ctl_param_x500.kx = np.array([55., 55., 30.]) # 10m_s 10m
# geom_ctl_param_x500.kx = np.zeros(3)
# geom_ctl_param_x500.kx = 20.*np.ones(3)

geom_ctl_param_x500.kv = np.array([7.0, 7.0, 5.0]) # Main gains (10/9/23)
# geom_ctl_param_x500.kv = np.array([4., 4., 5.]) # 10m_s 10m
# geom_ctl_param_x500.kv = 2.*np.ones(3)
# geom_ctl_param_x500.kv = np.zeros(3)

geom_ctl_param_x500.ki = np.array([7.0, 8.5, 1.5]) # Main gains (10/9/23)
# geom_ctl_param_x500.ki = np.array([2., 3., 1.5]) # 10m/s 10m
# geom_ctl_param_x500.ki = 1.*np.ones(3)
# geom_ctl_param_x500.ki = np.zeros(3)

geom_ctl_param_x500.sigma = np.array([10, 10, 10])
geom_ctl_param_x500.c1 = np.array([1, 1, 1])

# M1M2 attitude controller gains
geom_ctl_param_x500.kb = np.array([5.0, 5.0, 5.0])
# geom_ctl_param_x500.kb = 5.0*np.ones(3)

geom_ctl_param_x500.kw = np.array([0.25, 0.25, 0.25])
# geom_ctl_param_x500.kw = 0.25*np.ones(3)

geom_ctl_param_x500.kI = np.array([0.0, 0.0, 0.0])
# geom_ctl_param_x500.kI = 0.0*np.ones(3)

geom_ctl_param_x500.c2 = np.array([1, 1, 1])
geom_ctl_param_x500.sigma2 = 10

# M3 yaw attitude controller gains
geom_ctl_param_x500.ky = 3.0
geom_ctl_param_x500.kwy = 1.5
geom_ctl_param_x500.kIy = 0.0
geom_ctl_param_x500.c3 = 1

# -------------------------- Generic PX4 Tuning #1 --------------------------- #
px4_ctl_param_iris = PX4Parameters()

px4_ctl_param_iris.pos_freq = 50
px4_ctl_param_iris.vel_freq = 50
px4_ctl_param_iris.att_freq = 250
px4_ctl_param_iris.ang_rate_freq = 1000

px4_ctl_param_iris.pos_x_P = 2.0
px4_ctl_param_iris.pos_y_P = 2.0
px4_ctl_param_iris.pos_z_P = 1.0

px4_ctl_param_iris.vel_err_int_lim = 25
px4_ctl_param_iris.vel_x_P = 5.0
px4_ctl_param_iris.vel_y_P = 4.0
px4_ctl_param_iris.vel_z_P = 10.0
px4_ctl_param_iris.vel_x_I = 0.01
px4_ctl_param_iris.vel_y_I = 0.01
px4_ctl_param_iris.vel_z_I = 0.025
px4_ctl_param_iris.vel_x_D = 0.1
px4_ctl_param_iris.vel_y_D = 0.075
px4_ctl_param_iris.vel_z_D = 0.1/2

px4_ctl_param_iris.attitude_P = 7.5
px4_ctl_param_iris.attitude_time_const = 1.0 # Equal to 1 in PX4 codebase

px4_ctl_param_iris.ang_rate_err_int_lim = 10
px4_ctl_param_iris.ang_rate_x_P = 1.0
px4_ctl_param_iris.ang_rate_x_I = 0.0
px4_ctl_param_iris.ang_rate_x_D = 0.01
px4_ctl_param_iris.ang_rate_y_P = 2.0
px4_ctl_param_iris.ang_rate_y_I = 0.0
px4_ctl_param_iris.ang_rate_y_D = 0.01
px4_ctl_param_iris.ang_rate_z_P = 0.5
px4_ctl_param_iris.ang_rate_z_I = 0.0
px4_ctl_param_iris.ang_rate_z_D = 0.0

# ------------------------ PX4 Tuning for x500 Model ------------------------- #
px4_ctl_param_x500 = PX4Parameters()

px4_ctl_param_x500.pos_freq = 50
px4_ctl_param_x500.vel_freq = 50
px4_ctl_param_x500.att_freq = 250
px4_ctl_param_x500.ang_rate_freq = 1000

# px4_ctl_param_x500.pos_err_int_lim = 10 # Tuning set 1m/s
# px4_ctl_param_x500.pos_x_P = 2.0
# px4_ctl_param_x500.pos_y_P = 2.0
# px4_ctl_param_x500.pos_z_P = 1.0
# px4_ctl_param_x500.pos_x_I = 0.05
# px4_ctl_param_x500.pos_y_I = 0.05
# px4_ctl_param_x500.pos_z_I = 0.01
# px4_ctl_param_x500.pos_x_D = 0.05
# px4_ctl_param_x500.pos_y_D = 0.05
# px4_ctl_param_x500.pos_z_D = 0.0

px4_ctl_param_x500.pos_err_int_lim = 10 # Tuning set ff
px4_ctl_param_x500.pos_x_P = 1.0
px4_ctl_param_x500.pos_y_P = 1.0
px4_ctl_param_x500.pos_z_P = 1.0
px4_ctl_param_x500.pos_x_I = 0.0
px4_ctl_param_x500.pos_y_I = 0.0
px4_ctl_param_x500.pos_z_I = 0.01
px4_ctl_param_x500.pos_x_D = 0.0
px4_ctl_param_x500.pos_y_D = 0.0
px4_ctl_param_x500.pos_z_D = 0.0

# px4_ctl_param_x500.pos_err_int_lim = 10 # Tuning set 5m/s 25m+ turns
# px4_ctl_param_x500.pos_x_P = 2.5
# px4_ctl_param_x500.pos_y_P = 2.5
# px4_ctl_param_x500.pos_z_P = 1.0
# px4_ctl_param_x500.pos_x_I = 0.175
# px4_ctl_param_x500.pos_y_I = 0.175
# px4_ctl_param_x500.pos_z_I = 0.01
# px4_ctl_param_x500.pos_x_D = 0.05
# px4_ctl_param_x500.pos_y_D = 0.05
# px4_ctl_param_x500.pos_z_D = 0.0

# px4_ctl_param_x500.pos_err_int_lim = 10 # Tuning set 5m/s 20m turns
# px4_ctl_param_x500.pos_x_P = 2.2
# px4_ctl_param_x500.pos_y_P = 2.2
# px4_ctl_param_x500.pos_z_P = 1.0
# px4_ctl_param_x500.pos_x_I = 0.175
# px4_ctl_param_x500.pos_y_I = 0.175
# px4_ctl_param_x500.pos_z_I = 0.01
# px4_ctl_param_x500.pos_x_D = 0.05
# px4_ctl_param_x500.pos_y_D = 0.05
# px4_ctl_param_x500.pos_z_D = 0.0

# px4_ctl_param_x500.pos_err_int_lim = 10 # Tuning set 5m/s 15m turns
# px4_ctl_param_x500.pos_x_P = 2.0
# px4_ctl_param_x500.pos_y_P = 2.0
# px4_ctl_param_x500.pos_z_P = 1.0
# px4_ctl_param_x500.pos_x_I = 0.175
# px4_ctl_param_x500.pos_y_I = 0.175
# px4_ctl_param_x500.pos_z_I = 0.01
# px4_ctl_param_x500.pos_x_D = 0.05
# px4_ctl_param_x500.pos_y_D = 0.05
# px4_ctl_param_x500.pos_z_D = 0.0

# px4_ctl_param_x500.pos_err_int_lim = 10 # Tuning set 10m/s straight
# px4_ctl_param_x500.pos_x_P = 2.5
# px4_ctl_param_x500.pos_y_P = 2.5
# px4_ctl_param_x500.pos_z_P = 1.0
# px4_ctl_param_x500.pos_x_I = 0.4
# px4_ctl_param_x500.pos_y_I = 0.4
# px4_ctl_param_x500.pos_z_I = 0.01
# px4_ctl_param_x500.pos_x_D = 0.05
# px4_ctl_param_x500.pos_y_D = 0.05
# px4_ctl_param_x500.pos_z_D = 0.0

# px4_ctl_param_x500.pos_err_int_lim = 10 # Tuning set 10m/s curve
# px4_ctl_param_x500.pos_x_P = 2.3
# px4_ctl_param_x500.pos_y_P = 2.3
# px4_ctl_param_x500.pos_z_P = 1.0
# px4_ctl_param_x500.pos_x_I = 0.3
# px4_ctl_param_x500.pos_y_I = 0.3
# px4_ctl_param_x500.pos_z_I = 0.01
# px4_ctl_param_x500.pos_x_D = 0.05
# px4_ctl_param_x500.pos_y_D = 0.05
# px4_ctl_param_x500.pos_z_D = 0.0

# px4_ctl_param_x500.pos_err_int_lim = 10 # Tuning set 15m/s const
# px4_ctl_param_x500.pos_x_P = 4.0
# px4_ctl_param_x500.pos_y_P = 4.0
# px4_ctl_param_x500.pos_z_P = 1.0
# px4_ctl_param_x500.pos_x_I = 0.375
# px4_ctl_param_x500.pos_y_I = 0.375
# px4_ctl_param_x500.pos_z_I = 0.01
# px4_ctl_param_x500.pos_x_D = 0.1
# px4_ctl_param_x500.pos_y_D = 0.1
# px4_ctl_param_x500.pos_z_D = 0.0

# px4_ctl_param_x500.pos_err_int_lim = 10 # Tuning set 20m/s
# px4_ctl_param_x500.pos_x_P = 5.05
# px4_ctl_param_x500.pos_y_P = 5.05
# px4_ctl_param_x500.pos_z_P = 1.0
# px4_ctl_param_x500.pos_x_I = 0.395
# px4_ctl_param_x500.pos_y_I = 0.395
# px4_ctl_param_x500.pos_z_I = 0.01
# px4_ctl_param_x500.pos_x_D = 0.1
# px4_ctl_param_x500.pos_y_D = 0.1
# px4_ctl_param_x500.pos_z_D = 0.0

# px4_ctl_param_x500.vel_err_int_lim = 25 # Normal tuning
# px4_ctl_param_x500.vel_x_P = 10.0
# px4_ctl_param_x500.vel_y_P = 10.0
# px4_ctl_param_x500.vel_z_P = 75.0
# px4_ctl_param_x500.vel_x_I = 0.01
# px4_ctl_param_x500.vel_y_I = 0.01
# px4_ctl_param_x500.vel_z_I = 0.025
# px4_ctl_param_x500.vel_x_D = 0.075
# px4_ctl_param_x500.vel_y_D = 0.075
# px4_ctl_param_x500.vel_z_D = 0.00

px4_ctl_param_x500.vel_err_int_lim = 25 # ff tuning
px4_ctl_param_x500.vel_x_P = 2.0
px4_ctl_param_x500.vel_y_P = 2.0
px4_ctl_param_x500.vel_z_P = 75.0
px4_ctl_param_x500.vel_x_I = 0.0
px4_ctl_param_x500.vel_y_I = 0.0
px4_ctl_param_x500.vel_z_I = 0.025
px4_ctl_param_x500.vel_x_D = 0.0
px4_ctl_param_x500.vel_y_D = 0.0
px4_ctl_param_x500.vel_z_D = 0.00

px4_ctl_param_x500.attitude_P = 15.0
px4_ctl_param_x500.attitude_time_const = 1.0 # Equal to 1 in PX4 codebase

px4_ctl_param_x500.ang_rate_err_int_lim = 10
px4_ctl_param_x500.ang_rate_x_P = 1.0
px4_ctl_param_x500.ang_rate_x_I = 0.0
px4_ctl_param_x500.ang_rate_x_D = 0.01
px4_ctl_param_x500.ang_rate_y_P = 1.0
px4_ctl_param_x500.ang_rate_y_I = 0.0
px4_ctl_param_x500.ang_rate_y_D = 0.01
px4_ctl_param_x500.ang_rate_z_P = 1.0
px4_ctl_param_x500.ang_rate_z_I = 0.0
px4_ctl_param_x500.ang_rate_z_D = 0.01















