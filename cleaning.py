import glob
import pandas as pd
import os
import shutil


def split_full_dataset():
    patients = ['G04', 'G06', 'G07', 'G08', 'G11', 'P379', 'P551', 'P812']
    os.makedirs('splitted_full_dateset/')

    for p in patients:
        if not os.path.exists('splitted_full_dateset/'+p):
            os.makedirs('splitted_full_dateset/'+p)
        for t in range(1, 3):
            dir_name = 'splitted_full_dateset/'+p+'/trial_'+str(t)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                files = glob.glob('full_dataset/**/'+p+'_*_trial_'+str(t)+'_out_*.csv', recursive=True)
                for f in files:
                    shutil.copy(f, dir_name)


def group_merged_dfs():
    patients = ['G04', 'G06', 'G07', 'G08', 'G11', 'P379', 'P551', 'P812']
    trials = ['trial_1', 'trial_2']
    files_sensors_position = ['left_foot', 'right_foot', 'lower_left_foot', 'lower_right_foot']
    cols = ['Time', 'Acc_1', 'Acc_2', 'Acc_3', 'Gyro_1', 'Gyro_2', 'Gyro_3', 'Label1', 'Label2']
    trials_merged_dfs = []
    for p in patients:
        for t in trials:
            position_files_dict = {}
            for s in files_sensors_position:
                position_files_list = glob.glob('splitted_full_dateset/'+p+'/'+t+'/*_out_' + s + '.csv')
                position_files_dict[s] = position_files_list
            for i in range(len(position_files_dict['left_foot'])):
                dfs = []
                for k, files_list in position_files_dict.items():
                    df = pd.read_csv(files_list[i], sep=',', usecols=cols)
                    dfs.append(df)
                merged_df = merge_dataframes(dfs)
                merged_df = apply_new_target(merged_df)
                trials_merged_dfs.append(merged_df)
    return trials_merged_dfs


def create_new_target(row):
    if row['Label1_left'] > 0 or row['Label2_left'] > 0:
        return 1
    else:
        return 0


def apply_new_target(df):
    df['Label'] = df.apply(lambda row: create_new_target(row), axis=1)
    df = df.drop(['Label1_left', 'Label2_left', 'Label1_right', 'Label2_right',
                  'Label1_lowerleft', 'Label2_lowerleft', 'Label1_lowerright', 'Label2_lowerright'], axis=1)
    return df


def merge_dataframes(dfs_list, key='Time'):
    df_upper = dfs_list[0].set_index('Time').join(dfs_list[1].set_index('Time'), lsuffix='_left', rsuffix='_right')
    df_lower = dfs_list[2].set_index('Time').join(dfs_list[3].set_index('Time'), lsuffix='_lowerleft', rsuffix='_lowerright')
    merged_df = df_upper.join(df_lower, on='Time', how='inner')
    merged_df.index = pd.to_timedelta(merged_df.index, unit='ms')
    return merged_df


