import cleaning


def get_train_test_dfs(train_test_files_list):
    train_test_dfs = []
    for l in train_test_files_list:
        # read raw data
        dfs_list = cleaning.read_data(l)

        # drop unimportant columns
        unimportant_cols_drop = ['Acc_global_1', 'Acc_global_2', 'Acc_global_3', 'Pitch',
                                 'Roll', 'Yaw', 'Movement', 'Swing', 'SwingFind']
        dfs_list_for_merge = []
        for df in dfs_list:
            df = cleaning.drop_unimportant_cols(df, unimportant_cols_drop)
            dfs_list_for_merge.append(df)

        # merge dataframes
        merged_df = cleaning.merge_dataframes(dfs_list_for_merge)

        # create new target label
        cleaned_df = cleaning.apply_new_target(merged_df)
        train_test_dfs.append(cleaned_df)
    return train_test_dfs
