import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pykalman import KalmanFilter

lowess = sm.nonparametric.lowess

filename1 = sys.argv[1]

cpu_data = pd.read_csv(filename1)

cpu_data.timestamp = pd.to_datetime(cpu_data.timestamp)

x = cpu_data['timestamp'] #X VALUE FOR LOWESS
y = cpu_data['temperature'] #Y VALUE FOR LOWESS

loess_smoothed = lowess(y, x, frac=0.0042) #CHECK THE FRAC
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha = 0.5)
plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-')


#KALMAN FILTERING
kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]

initial_state = kalman_data.iloc[0]

observation_covariance = np.diag([2, 3, 0.5, 40]) ** 2 # observed the plot of each data column and estimated the daviation
transition_covariance = np.diag([0.2, 0.2, 0.2, 0.2]) ** 2 # TODO: shouldn't be zero
transition = [[0.97,0.5,0.2,-0.001], [0.1,0.4,2.2,0], [0,0,0.95,0], [0,0,0,1]] # TODO: shouldn't (all) be zero


kf = KalmanFilter(
    initial_state_mean=initial_state,
    initial_state_covariance=observation_covariance,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance,
    transition_matrices=transition
)

kalman_smoothed, state_cov = kf.smooth(kalman_data)
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')
plt.savefig('cpu.svg')




















