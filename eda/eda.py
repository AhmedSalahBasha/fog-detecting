import pandas as pd
import numpy as np

from plots.plotting import plot_time_series


df_left_foot = pd.read_csv('../data/splitted_full_dataset/G04/trial_1/G04_FoG_trial_1_out_left_foot.csv', header=0, index_col=0)
df_right_foot = pd.read_csv('../data/splitted_full_dataset/G04/trial_1/G04_FoG_trial_1_out_right_foot.csv', header=0, index_col=0)
df_lower_left = pd.read_csv('../data/splitted_full_dataset/G04/trial_1/G04_FoG_trial_1_out_lower_left_foot.csv', header=0, index_col=0)
df_lower_right = pd.read_csv('../data/splitted_full_dataset/G04/trial_1/G04_FoG_trial_1_out_lower_right_foot.csv', header=0, index_col=0)

cols = ['Acc_1', 'Acc_2', 'Acc_3']

plot_time_series(df_left_foot, cols, 'Accelerometer Left Foot', 'acc_left_foot')
plot_time_series(df_right_foot, cols, 'Accelerometer Right Foot', 'acc_right_foot')
plot_time_series(df_lower_left, cols, 'Accelerometer Lower Left', 'acc_lower_left')
plot_time_series(df_lower_right, cols, 'Accelerometer Lower Right', 'acc_lower_right')







