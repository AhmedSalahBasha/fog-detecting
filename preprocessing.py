import cleaning
import rolling_window as rw
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_rolled_dataframe(win_size, step_size):
    """

    :param win_size:
    :param step_size:
    :return:
    """
    dfs_list = cleaning.group_merged_dfs()  # 35 dfs
    rolled_dfs = []
    for df in dfs_list:
        rolled_df = rw.rolling_window(df, win_size, step_size)
        rolled_dfs.append(rolled_df)
    full_dataset_rolled = pd.concat(rolled_dfs, ignore_index=True)
    full_dataset_rolled.to_csv('processed_data/freq_dom_features_rolled_dataset.csv', sep=',', index=False)
    return full_dataset_rolled


def create_rolled_train_dev_test_dataframes(win_size, step_size):
    """
    It calls
    :param win_size: rolling window size
    :param step_size: rolling window step. It should be a percentage of win_size to overlap windows.
    :return: None
    """
    dfs_list = cleaning.group_merged_dfs()  # returning lists of 35 dfs
    train_dfs_list = dfs_list[:20]
    dev_dfs_list = dfs_list[20:27]
    test_dfs_list = dfs_list[27:]

    train_set = _get_rolled_df(train_dfs_list, win_size, step_size)
    dev_set = _get_rolled_df(dev_dfs_list, win_size, step_size)
    test_set = _get_rolled_df(test_dfs_list, win_size, step_size)
    train_set.to_csv('processed_data/train_set.csv', sep=',', index=False)
    dev_set.to_csv('processed_data/dev_set.csv', sep=',', index=False)
    test_set.to_csv('processed_data/test_set.csv', sep=',', index=False)


def _get_rolled_df(dfs_list, win_size, step_size):
    """

    :param dfs_list:
    :param win_size:
    :param step_size:
    :return:
    """
    rolled_dfs = []
    for df in dfs_list:
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

