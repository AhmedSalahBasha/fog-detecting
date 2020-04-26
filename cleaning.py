import pandas as pd
import numpy as np
from functools import reduce


def read_data(files_path_list):
    dfs = []
    for file in files_path_list:
        df = pd.read_csv(file, sep=',')
        dfs.append(df)
    return dfs


def merge_dataframes(dfs_list, key='Time'):
    df_upper = dfs_list[0].set_index('Time').join(dfs_list[1].set_index('Time'), lsuffix='_left', rsuffix='_right')
    df_lower = dfs_list[2].set_index('Time').join(dfs_list[3].set_index('Time'), lsuffix='_lowerleft', rsuffix='_lowerright')
    merged_df = df_upper.join(df_lower, on='Time', how='inner')
    merged_df.index = pd.to_timedelta(merged_df.index, unit='ms')
    return merged_df


def drop_unimportant_cols(df, cols_list):
    df = df.drop(cols_list, axis=1)
    return df


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

