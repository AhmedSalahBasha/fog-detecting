import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Some meta data info:
    Label_1 and Label_2 are ground truth from two different clinicians
    acc_global are the transformed accelerometer values. They describe the acceleration in a global coordinate system. In this coordinate system the z-axis correspond with the earth gravitation vector.</p>
    Label values:
        0 = none
        1 = Shank Tremble
        2 = Shuffling
        3 = Festination
        4 = Schrittabfolge
        5 = Loss of Balance
        6 = Akinesia
'''

df_upper_left = pd.read_csv('data/P812_M050_2_B_FoG_trial_1_out_left_foot.csv', sep=',')
df_lower_left = pd.read_csv('data/P812_M050_2_B_FoG_trial_1_out_lower_left_foot.csv', sep=',')
df_upper_right = pd.read_csv('data/P812_M050_2_B_FoG_trial_1_out_right_foot.csv', sep=',')
df_lower_right = pd.read_csv('data/P812_M050_2_B_FoG_trial_1_out_lower_right_foot.csv', sep=',')

df_left = df_upper_left.set_index('Time').join(df_lower_left.set_index('Time'), lsuffix='_upperleft', rsuffix='_lowerleft')
df_right = df_upper_right.set_index('Time').join(df_lower_right.set_index('Time'), lsuffix='_upperright', rsuffix='_lowerright')
df = df_left.join(df_right, on='Time', how='inner')

df.index = pd.to_timedelta(df.index, unit='ms')
cols_drop = ['Acc_global_1_upperleft', 'Acc_global_2_upperleft', 'Acc_global_3_upperleft',
             'Pitch_upperleft', 'Roll_upperleft', 'Yaw_upperleft',
             'Movement_upperleft', 'Swing_upperleft', 'SwingFind_upperleft',
             'Acc_global_1_lowerleft', 'Acc_global_2_lowerleft', 'Acc_global_3_lowerleft',
             'Pitch_lowerleft', 'Roll_lowerleft', 'Yaw_lowerleft',
             'Movement_lowerleft', 'Swing_lowerleft', 'SwingFind_lowerleft',
             'Label1_lowerleft', 'Label2_lowerleft',
             'Acc_global_1_upperright', 'Acc_global_2_upperright', 'Acc_global_3_upperright',
             'Pitch_upperright', 'Roll_upperright', 'Yaw_upperright',
             'Movement_upperright', 'Swing_upperright', 'SwingFind_upperright',
             'Acc_global_1_lowerright', 'Acc_global_2_lowerright', 'Acc_global_3_lowerright',
             'Pitch_lowerright', 'Roll_lowerright', 'Yaw_lowerright',
             'Movement_lowerright', 'Swing_lowerright', 'SwingFind_lowerright',
             'Label1_lowerright', 'Label2_lowerright']
df.drop(cols_drop, axis=1, inplace=True)
print(df.info())

print("all labels: ", df['Label1_upperleft'].count())
print("count of equal labels: ", df['Label1_upperleft'][df['Label1_upperleft'] == df['Label2_upperleft']].count())
print("count of different labels: ", df['Label1_upperleft'][df['Label1_upperleft'] != df['Label2_upperleft']].count())


def create_new_target(row):
    if row['Label1_upperleft'] > 0 or row['Label2_upperleft'] > 0  :
        return 1
    else:
        return 0


df['target'] = df.apply(lambda row: create_new_target(row), axis=1)
df.drop(['Label1_upperleft', 'Label2_upperleft',
         'Label1_upperright', 'Label2_upperright'], axis=1, inplace=True)
print(df.head(10))

df['target'].value_counts()

df.to_csv('clean_data/P812_M050_B_FoG_trial_1_cleaned.csv', sep=',', index=True)

