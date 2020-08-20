import datetime

import numpy as np
from scipy import stats
import tsfel

from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters



def rolling_window(df, win_size, step_size):
    """
     Applying the rolling window technique that extracts features from the raw data
    :param input_df:
    :param win_size:
    :param step_size:
    :return:
    """
    input_df = df.copy()
    fs = 200     # Signal sampling frequency
    win_sec = win_size / fs
    columns_list = list(input_df.columns)
    for col in input_df.columns:
        if col == 'Label':
            input_df[col] = input_df[col].rolling(win_size).apply(_get_rolled_label)[step_size - 1::step_size]  # label
        elif col == 'patient':
            pass
        elif col == 'trials':
            pass
        else:
            print('### Started Column :  ' + col + ' At time:  ' + str(datetime.datetime.now()) + '######')
            # =============== STATISTICAL FEATURES ================
            input_df[col + '_avg'] = input_df[col].rolling(win_size).mean()[step_size - 1::step_size]     # mean
            input_df[col + '_med'] = input_df[col].rolling(win_size).median()[step_size - 1::step_size]   # median
            input_df[col + '_std'] = input_df[col].rolling(win_size).std()[step_size - 1::step_size]      # standard-deviation
            input_df[col + '_min'] = input_df[col].rolling(win_size).min()[step_size - 1::step_size]      # minimum
            input_df[col + '_max'] = input_df[col].rolling(win_size).max()[step_size - 1::step_size]      # maximum
            input_df[col + '_rms'] = input_df[col].rolling(win_size).apply(tsfel.rms)[step_size - 1::step_size]  # root mean squar
            #input_df[col + '_iqr'] = input_df[col].rolling(win_size).apply(_get_iqr)[step_size - 1::step_size]     # interquartile
            #input_df[col + '_iqr_rng'] = input_df[col].rolling(win_size).apply(tsfel.interq_range)[step_size - 1::step_size]  # interquartile range
            input_df[col + '_var'] = input_df[col].rolling(win_size).apply(tsfel.calc_var)[step_size - 1::step_size]  # variance
            # =============== FREQUENCY (SPECTRAL) DOMAIN FEATURES ===================
            input_df[col + '_fi'] = input_df[col].rolling(win_size).apply(_freeze_index,
                                                                          args=(fs, [3, 8], [0.5, 3], win_sec,))[step_size - 1::step_size]    # Freezing Index
            input_df[col + '_pi'] = input_df[col].rolling(win_size).apply(_power_index,
                                                                          args=(fs, [3, 8], [0.5, 3], win_sec,))[step_size - 1::step_size]  # Power Index
            input_df[col + '_fp'] = input_df[col].rolling(win_size).apply(_freeze_band_power,
                                                                          args=(fs, [3, 8], win_sec,))[step_size - 1::step_size]  # Freezing Band Power
            input_df[col + '_lp'] = input_df[col].rolling(win_size).apply(_locomotor_band_power,
                                                                         args=(fs, [0.5, 3], win_sec,))[step_size - 1::step_size]  # Locomotor Band Power
            '''
            input_df[col + '_spec_dist'] = input_df[col].rolling(win_size).apply(_get_spectral_distance,
                                                                               args=(fs,))[step_size - 1::step_size]  # spectral distance
            input_df[col + '_hum_eng'] = input_df[col].rolling(win_size).apply(_get_human_range_energy,
                                                                               args=(fs,))[step_size - 1::step_size]  # human range energy
            input_df[col + '_max_pow_spec'] = input_df[col].rolling(win_size).apply(_get_max_power_spectrum,
                                                                                    args=(fs,))[step_size - 1::step_size]  # max power spectrum
            input_df[col + '_max_freq'] = input_df[col].rolling(win_size).apply(_get_max_frequency,
                                                                                args=(fs,))[step_size - 1::step_size]     # max frequency
            input_df[col + '_spec_entropy'] = input_df[col].rolling(win_size).apply(_get_spectral_entropy, args=(fs,))[step_size - 1::step_size]  # total energy
            input_df[col + '_pow_band'] = input_df[col].rolling(win_size).apply(_get_power_bandwidth, args=(fs,))[step_size - 1::step_size]  # power bandwidth
            '''
            '''
            # =============== TEMPORAL FEATURES ======================
            input_df[col + '_slope'] = input_df[col].rolling(win_size).apply(tsfel.slope)[step_size - 1::step_size]  # slope
            input_df[col + '_max_peaks'] = input_df[col].rolling(win_size).apply(tsfel.maxpeaks)[step_size - 1::step_size]  # slope
            input_df[col + '_total_eng'] = input_df[col].rolling(win_size).apply(_get_total_energy,
                                                                                 args=(fs,))[step_size - 1::step_size]  # total energy
            input_df[col + '_abs_eng'] = input_df[col].rolling(win_size).apply(tsfel.abs_energy)[step_size - 1::step_size]  # absolute energy
            input_df[col + '_dist'] = input_df[col].rolling(win_size).apply(tsfel.distance)[step_size - 1::step_size]  # distance
            '''
    input_df = _drop_columns_except_target(input_df, columns_list)
    input_df['patient'] = df.iloc[0]['patient']
    input_df['trials'] = df.iloc[0]['trials']
    input_df = input_df.dropna()
    print('======== FINISHED DATAFRAME AT TIME:  ' + str(datetime.datetime.now()) + '==========')
    return input_df


def _drop_columns_except_target(input_df, columns_list):
    """

    :param input_df:
    :param columns_list:
    :return:
    """
    for col in columns_list:
        if col != 'Label':
            input_df = input_df.drop(col, axis=1)
    return input_df


def _get_rolled_label(col):
    """
    Get the mode (most common value) value for the label column within the window size while rolling
    :param col: dataframe columns (series)
    :return: the mode value
    """
    label = stats.mode(col)[0][0]
    return label


def _get_iqr(col):
    """

    :param col:
    :return:
    """
    iqr = np.percentile(col, 75) - np.percentile(col, 25)
    return iqr


def _get_max_frequency(col, fs):
    """

    :param col:
    :param fs:
    :return:
    """
    max_frq = tsfel.max_frequency(col, fs)
    return max_frq


def _get_human_range_energy(col, fs):
    """

    :param col:
    :param fs:
    :return:
    """
    energy = tsfel.human_range_energy(col, fs)
    return energy


def _get_spectral_distance(col, fs):
    """

    :param col:
    :param fs:
    :return:
    """
    spec_dist = tsfel.spectral_distance(col, fs)
    return spec_dist


def _get_total_energy(col, fs):
    """

    :param col:
    :param fs:
    :return:
    """
    total_energy = tsfel.total_energy(col, fs)
    return total_energy


def _get_power_bandwidth(col, fs):
    """

    :param col:
    :param fs:
    :return:
    """
    power = tsfel.power_bandwidth(col, fs)
    return power


def _get_max_power_spectrum(col, fs):
    """

    :param col:
    :param fs:
    :return:
    """
    max_power = tsfel.max_power_spectrum(col, fs)
    return max_power


def _get_spectral_entropy(col, fs):
    """

    :param col:
    :param fs:
    :return:
    """
    sepc_entropy = tsfel.spectral_entropy(col, fs)
    return sepc_entropy



def _freeze_index(data, sf, band1, band2, win_sec=None, relative=False):
    """
    The power in the “freeze” band (3-8Hz) divided by the power in the locomotor band (0.5-3Hz)
    :param data:
    :param sf:
    :param band1:
    :param band2:
    :param win_sec:
    :param relative:
    :return:
    """
    fi = _bandpower(data, sf, band1, win_sec, relative) / _bandpower(data, sf, band2, win_sec, relative)
    return fi


def _power_index(data, sf, band1, band2, win_sec=None, relative=False):
    """
    The sum of the power in the “freeze” band (3-8Hz) plus the power in the locomotor band (0.5-3Hz)
    :param data:
    :param sf:
    :param band1:
    :param band2:
    :param win_sec:
    :param relative:
    :return:
    """
    pow_indx = _bandpower(data, sf, band1, win_sec, relative) + _bandpower(data, sf, band2, win_sec, relative)
    return pow_indx


def _freeze_band_power(data, sf, freeze_band, win_sec=None, relative=False):
    """
    The sum of the power spectrum in the “freeze” band of frequencies (3-8Hz) divided by the sampling frequency
    :param data:
    :param sf:
    :param band1:
    :param win_sec:
    :param relative:
    :return:
    """
    freeze_power = _bandpower(data, sf, freeze_band, win_sec, relative)
    return freeze_power


def _locomotor_band_power(data, sf, locomotor_band, win_sec=None, relative=False):
    """
    The sum of the power spectrum in the locomotor band of frequencies (0.53Hz) divided by the sampling frequency
    :param data:
    :param sf:
    :param band2:
    :param win_sec:
    :param relative:
    :return:
    """
    locomotor_power = _bandpower(data, sf, locomotor_band, win_sec, relative)
    return locomotor_power


def _bandpower(data, sf, band, window_sec, relative):
    """Compute the average power of the signal x in a specific frequency band.
    Source: https://raphaelvallat.com/bandpower.html
    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp



# ---------------------- UN-USED FUNCTIONS ------------------------
def _get_signal_power(col):
    """

    :param col:
    :return:
    """
    sig_fft = np.fft.fft(col)
    sig_mag = np.argmax(abs(sig_fft))  # magnitude
    sig_power = sig_mag**2  # power
    return sig_power


def _get_dominant_frequency(col, win_size):
    """

    :param col:
    :param win_size:
    :return:
    """
    w = np.fft.fft(col)
    freqs = np.fft.fftfreq(len(col))
    i = np.argmax(abs(w))
    dom_freq = freqs[i]
    dom_freq_hz = abs(dom_freq * win_size)
    return dom_freq_hz


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
