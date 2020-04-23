import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('clean_data/P812_M050_B_FoG_trial_1_cleaned.csv', sep=',')
df_test = pd.read_csv('clean_data/P812_M050_B_FoG_trial_2_cleaned.csv', sep=',')

df = df.set_index('Time')
df_test = df_test.set_index('Time')
print(df.head(10))

print(df_test.head(10))

cols_names = ['acc_x_upperleft', 'acc_y_upperleft', 'acc_z_upperleft',
              'gyro_x_upperleft', 'gyro_y_upperleft', 'gyro_z_upperleft',
              'stride_length_upperleft',
              'acc_x_lowerleft', 'acc_y_lowerleft', 'acc_z_lowerleft',
              'gyro_x_lowerleft', 'gyro_y_lowerleft', 'gyro_z_lowerleft',
              'stride_length_lowerleft',
              'acc_x_upperright', 'acc_y_upperright', 'acc_z_upperright',
              'gyro_x_upperright', 'gyro_y_upperright', 'gyro_z_upperright',
              'stride_length_upperright',
              'acc_x_lowerright', 'acc_y_lowerright', 'acc_z_lowerright',
              'gyro_x_lowerright', 'gyro_y_lowerright', 'gyro_z_lowerright',
              'stride_length_lowerright',
              'target']
df.columns = cols_names
df_test.columns = cols_names

print('Train dataframe shape: ', df.shape)
print('Test dataframe shape: ', df_test.shape)

print('Train dataframe target value counts: ', df['target'].value_counts())

print('Test dataframe target value counts: ', df_test['target'].value_counts())


def rolling_window(input_df, win_size=400, label='target'):
    for col in input_df.columns:
        if col != 'target':
            col_name = col + '_mean'
            input_df[col + '_avg'] = input_df[col].rolling(win_size).mean()
            input_df[col + '_med'] = input_df[col].rolling(win_size).median()
            input_df[col + '_std'] = input_df[col].rolling(win_size).std()
            input_df[col + '_min'] = input_df[col].rolling(win_size).min()
            input_df[col + '_max'] = input_df[col].rolling(win_size).max()
    input_df.dropna(inplace=True)
    return input_df


df = rolling_window(df)
df_test = rolling_window(df_test)

print('Train-set shape: ', df.shape)
print('Test-set shape: ', df_test.shape)

df.to_csv('processed_data/P812_M050_B_FoG_trial_2_train.csv', sep=',', index=False)
df_test.to_csv('processed_data/P812_M050_B_FoG_trial_2_test.csv', sep=',', index=False)
