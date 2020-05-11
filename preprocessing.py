from numpy import percentile, fft, abs, argmax
import datetime
import pandas as pd
from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series
import tsfel


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


def get_max_frequency(col, fs):
    max_frq = tsfel.max_frequency(col, fs)
    return max_frq


def get_human_range_energy(col, fs):
    energy = tsfel.human_range_energy(col, fs)
    return energy


def get_total_energy(col, fs):
    total_energy = tsfel.total_energy(col, fs)
    return total_energy


def get_power_bandwidth(col, fs):
    power = tsfel.power_bandwidth(col, fs)
    return power


def get_max_power_spectrum(col, fs):
    max_power = tsfel.max_power_spectrum(col, fs)
    return max_power


def rolling_window(input_df, columns_list, win_size=400, step_size=50):
    for col in input_df.columns:
        if col != 'Label':
            input_df[col + '_avg'] = input_df[col].rolling(win_size).mean()[step_size - 1::step_size]     # mean
            input_df[col + '_med'] = input_df[col].rolling(win_size).median()[step_size - 1::step_size]   # median
            input_df[col + '_std'] = input_df[col].rolling(win_size).std()[step_size - 1::step_size]      # standard-deviation
            input_df[col + '_min'] = input_df[col].rolling(win_size).min()[step_size - 1::step_size]      # minimum
            input_df[col + '_max'] = input_df[col].rolling(win_size).max()[step_size - 1::step_size]      # maximum
            # input_df[col + '_rng'] = input_df[col].rolling(win_size).max() - input_df[col].rolling(win_size).min()      # range
            input_df[col + '_iqr'] = input_df[col].rolling(win_size).apply(get_iqr)[step_size - 1::step_size]     # interquartile
            input_df[col + '_hum_eng'] = input_df[col].rolling(win_size).apply(get_human_range_energy, args=(200,))[step_size - 1::step_size]  # human range energy
            input_df[col + '_total_eng'] = input_df[col].rolling(win_size).apply(get_total_energy, args=(200,))[step_size - 1::step_size]  # total energy
            # input_df[col + '_entropy'] = input_df[col].rolling(win_size).apply(tsfel.entropy)[step_size - 1::step_size]  # entropy
            input_df[col + '_var'] = input_df[col].rolling(win_size).apply(tsfel.calc_var)[step_size - 1::step_size]  # variance
            input_df[col + '_slope'] = input_df[col].rolling(win_size).apply(tsfel.slope)[step_size - 1::step_size]  # slope
            input_df[col + '_max_peaks'] = input_df[col].rolling(win_size).apply(tsfel.maxpeaks)[step_size - 1::step_size]  # slope
            input_df[col + '_abs_eng'] = input_df[col].rolling(win_size).apply(tsfel.abs_energy)[step_size - 1::step_size]  # absolute energy
            # input_df[col + '_dist'] = input_df[col].rolling(win_size).apply(tsfel.distance)[step_size - 1::step_size]  # distance
            input_df[col + '_max_freq'] = input_df[col].rolling(win_size).apply(get_max_frequency, args=(200,))[step_size - 1::step_size]     # max frequency
            input_df[col + '_pow_band'] = input_df[col].rolling(win_size).apply(get_power_bandwidth, args=(200,))[step_size - 1::step_size]  # power bandwidth
            input_df[col + '_max_pow_spec'] = input_df[col].rolling(win_size).apply(get_max_power_spectrum, args=(200,))[step_size - 1::step_size]  # max power spectrum
    input_df = input_df.dropna()
    input_df = drop_columns_except_target(input_df, columns_list)
    print(input_df.tail(10))
    return input_df


def tsfresh_generate_features(input_df):
    input_df['id'] = 1
    input_df = input_df.reset_index()
    extraction_settings = ComprehensiveFCParameters()
    features_filtered = extract_relevant_features(input_df,
                                                  y=input_df['Label'],
                                                  column_sort='Time',
                                                  column_id='id',
                                                  default_fc_parameters=extraction_settings)
    return features_filtered


def tsfel_extract_features(input_df):
    # If no argument is passed retrieves all available features
    cfg_file = tsfel.get_features_by_domain()
    # Receives a time series sampled at 50 Hz, divides into windows of size 250 (i.e. 5 seconds) and extracts all features
    X_train = tsfel.time_series_features_extractor(cfg_file,
                                                   input_df,
                                                   fs=200,
                                                   window_splitter=True,
                                                   window_size=400)
    return X_train


def tsfresh_generate_features_by_rolling(input_df, cols_names, win_size):
    output_features_df = input_df[cols_names].rolling(win_size).apply(tsfel_extract_features)
    return output_features_df


