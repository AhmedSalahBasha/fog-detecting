from cleaning import cleaning
from preprocessing import rolling_window as rw
from preprocessing import features_selection as fs
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def apply_rolling_on_full_dataframe(win_size, step_size):
    full_dataset = pd.read_csv('data/processed_data/full_dataset.csv', sep=',')
    '''
    # Feature Selection
    SENSOR_TYPE = 'gyro'
    SENSOR_POS = 'feet'
    FEATURES_GROUP = 'freq'
    LEG = 'both'
    full_dataset = fs.sensors_features(full_dataset, pos=SENSOR_POS, group=FEATURES_GROUP, sensor=SENSOR_TYPE, leg=LEG)
    '''
    patients = list(full_dataset['patient'].unique())
    rolled_dfs = []
    for p in patients:
        print('========= Started with Patient: ' + str(p) + '============')
        p_df = full_dataset[full_dataset['patient'] == p]
        trials = list(p_df['trials'].unique())
        for t in trials:
            print('========= Started with Trial: ' + str(t) + '============')
            t_df = p_df[p_df['trials'] == t]
            '''
            # Removing outliers
            print("Dataframe shape BEFORE removing Outliers: ", t_df.shape)
            cleaned_df = removing_outliers(patient_df=t_df, lower_quantile=0.04, upper_quantile=0.96)
            print("Dataframe shape AFTER removing Outliers: ", cleaned_df.shape)
            '''
            rolled_df = rw.rolling_window(t_df, win_size, step_size)
            rolled_dfs.append(rolled_df)
    full_rolled_df = pd.concat(rolled_dfs, ignore_index=True)
    full_rolled_df.to_csv('full_rolled_dataset_winsize_'+str(win_size)+'.csv', sep=',', index=False)
    return full_rolled_df


def removing_outliers(patient_df, lower_quantile, upper_quantile):
    # identify upper and lower quartiles
    Q1 = patient_df.quantile(lower_quantile)
    Q3 = patient_df.quantile(upper_quantile)
    IQR = Q3 - Q1
    # filter dataset from outliers
    idx = ~((patient_df < (Q1 - 1.5 * IQR)) | (patient_df > (Q3 + 1.5 * IQR))).any(axis=1)
    patient_df_cleaned = patient_df.loc[idx]
    return patient_df_cleaned


def leave_one_patient_out(dataset, test_patient):
    test_df = dataset[dataset['patient'] == test_patient]
    train_df = pd.concat([dataset, test_df, test_df]).drop_duplicates(keep=False)
    return train_df, test_df


def get_full_dataframe():
    """

    :return:
    """
    dfs_list = cleaning.group_merged_dfs()
    full_dataset = pd.concat(dfs_list, ignore_index=True)
    full_dataset.to_csv('data/processed_data/full_dataset.csv', sep=',', index=False)
    return full_dataset


def create_rolled_train_test_dataframes(win_size, step_size):
    """
    It calls
    :param win_size: rolling window size
    :param step_size: rolling window step. It should be a percentage of win_size to overlap windows.
    :return: None
    """
    dfs_list = cleaning.group_merged_dfs()  # returning lists of 35 dfs
    test_dfs_list = [dfs_list[6]]     # Labels Count::  1-->37381  &  0-->7719
    dfs_list.pop(6)
    train_dfs_list = dfs_list

    '''
    #Try to choose the most balanced test set for one patient
    for i, df in enumerate(dfs_list):
        print("DataFrame Num. {0} \n {1}".format(i, df['Label'].value_counts()))
    '''

    test_set = _get_rolled_df(test_dfs_list, win_size, step_size)
    test_set.to_csv('data/processed_data/test_set_w300_s40.csv', sep=',', index=False)

    train_set = _get_rolled_df(train_dfs_list, win_size, step_size)
    train_set.to_csv('data/processed_data/train_set_w300_s40.csv', sep=',', index=False)

    full_rolled_set = pd.concat([train_set, test_set], ignore_index=True)
    full_rolled_set.to_csv('data/processed_data/full_rolled_set_w300_s40.csv', sep=',', index=False)


def _get_rolled_df(dfs_list, win_size, step_size):
    """

    :param dfs_list:
    :param win_size:
    :param step_size:
    :return:
    """
    if len(dfs_list) == 1:
        print("Started Test Dataframe")
        rolled_df = rw.rolling_window(dfs_list[0], win_size, step_size)
        return rolled_df
    elif len(dfs_list) > 1:
        rolled_dfs = []
        for i, df in enumerate(dfs_list):
            print("Started Training Dataframe Number:  ", i)
            rolled_df = rw.rolling_window(df, win_size, step_size)
            rolled_dfs.append(rolled_df)
        dataset_rolled = pd.concat(rolled_dfs, ignore_index=True)
        return dataset_rolled



def split_train_test_sets(fulldataset):
    X = fulldataset.drop(['Label'], axis=1)
    y = fulldataset['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=False)
    return X_train, X_test, y_train, y_test


def split_train_dev_test_sets(train_set, dev_set, test_set):
    X_train = train_set.drop(['Label'], axis=1)
    y_train = train_set['Label']

    X_dev = dev_set.drop(['Label'], axis=1)
    y_dev = dev_set['Label']

    X_test = test_set.drop(['Label'], axis=1)
    y_test = test_set['Label']

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def get_value_counts_of_array(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return np.asarray((unique, counts)).T

