import pandas as pd
from sklearn.model_selection import train_test_split

import cleaning


def get_full_dataset():
    dfs_list = cleaning.group_merged_dfs()  # 35 dfs
    full_dataset = pd.concat(dfs_list, ignore_index=True)
    full_dataset.to_csv('data/full_dataset.csv', sep=',', index=False)
    '''
    train_dfs = dfs_list[0:23]
    train_df = pd.concat(train_dfs)
    test_dfs = dfs_list[23:]
    test_df = pd.concat(test_dfs)
    train_df.to_csv('data/train_df.csv', sep=',', index=False)
    test_df.to_csv('data/test_df.csv', sep=',', index=False)
    '''
    return full_dataset


def split_train_test_sets(fulldataset):
    X = fulldataset.drop(['Label'], axis=1).values
    y = fulldataset['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=False)
    return X_train, X_test, y_train, y_test

