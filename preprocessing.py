from numpy import percentile, fft, abs, argmax
import pandas as pd
from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters


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


def get_iqr(col):
    """
    Returns the interquartile range a time series
    :param data:
    :return:
    """
    iqr = percentile(col, 75) - percentile(col, 25)
    return iqr


def get_dominant_frequency(col, win_size):
    """
    Returns the dominant frequency of a time series with 32 samples per second
    :param data:
    :param win_size:
    :return:
    """
    w = fft.fft(col)
    freqs = fft.fftfreq(len(col))
    i = argmax(abs(w))
    dom_freq = freqs[i]
    dom_freq_hz = abs(dom_freq * win_size)
    return dom_freq_hz


def get_signal_power(col):
    sig_fft = fft.fft(col)
    sig_mag = argmax(abs(sig_fft))  # magnitude
    sig_power = sig_mag**2  # power
    return sig_power


def rolling_window(input_df, columns_list, win_size=400):
    for col in input_df.columns:
        if col != 'Label':
            input_df[col + '_avg'] = input_df[col].rolling(win_size).mean()     # mean
            input_df[col + '_med'] = input_df[col].rolling(win_size).median()   # median
            input_df[col + '_std'] = input_df[col].rolling(win_size).std()      # standard-deviation
            input_df[col + '_min'] = input_df[col].rolling(win_size).min()      # minimum
            input_df[col + '_max'] = input_df[col].rolling(win_size).max()      # maximum
            input_df[col + '_var'] = input_df[col].rolling(win_size).std()**2   # variance
            input_df[col + '_rng'] = input_df[col].rolling(win_size).max() - input_df[col].rolling(win_size).min()      # range
            input_df[col + '_iqr'] = input_df[col].rolling(win_size).apply(get_iqr)     # interquartile
            # input_df[col + '_exp_avg'] = input_df[col].expanding(2).mean()              # expand window mean
            # input_df[col + '_exp_med'] = input_df[col].expanding(2).median()            # expand window median
            # input_df[col + '_df'] = input_df[col].rolling(win_size).apply(get_dominant_frequency, args=(win_size,))      # dominant frequency
            # input_df[col + '_pow'] = input_df[col].rolling(win_size).apply(get_signal_power)    # power
    input_df = input_df.dropna()
    input_df = drop_columns_except_target(input_df, columns_list)
    return input_df


def tsfresh_generate_features(input_df):
    extraction_settings = ComprehensiveFCParameters()
    features_filtered = extract_features(input_df,
                                         column_id='Label',
                                         default_fc_parameters=extraction_settings)
    return features_filtered

