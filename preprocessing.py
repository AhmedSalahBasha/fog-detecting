

def get_train_df(train_test_dfs_list):
    return train_test_dfs_list[0]


def get_test_df(train_test_dfs_list):
    return train_test_dfs_list[1]


def get_cols_names_list(df):
    return list(df.columns)


# drop original columns and keep only the statistical columns generated from the rolling window
def drop_columns_except_target(input_df, columns_list):
    for col in columns_list:
        if col != 'Label':
            input_df = input_df.drop(col, axis=1)
    return input_df


def rolling_window(input_df, columns_list, win_size=400):
    for col in input_df.columns:
        if col != 'Label':
            col_name = col + '_mean'
            input_df[col + '_avg'] = input_df[col].rolling(win_size).mean()
            input_df[col + '_med'] = input_df[col].rolling(win_size).median()
            input_df[col + '_std'] = input_df[col].rolling(win_size).std()
            input_df[col + '_min'] = input_df[col].rolling(win_size).min()
            input_df[col + '_max'] = input_df[col].rolling(win_size).max()
    input_df = input_df.dropna()
    input_df = drop_columns_except_target(input_df, columns_list)
    return input_df

