import re
import pandas as pd


def sensors_features(df, pos, group, sensor):
    all_cols = df.columns
    re_lower = re.compile(".*_lower*")
    lower_cols = list(filter(re_lower.match, all_cols))
    upper_cols = [x for x in all_cols if x not in lower_cols]
    upper_cols.remove('Label')
    if pos == 'lower':
        if sensor == 'acc':
            lower_df = df.drop(upper_cols, axis=1)
            re_gyro = re.compile("^Gyro_*")
            gyro_cols = list(filter(re_gyro.match, list(lower_df.columns)))
            lower_df = lower_df.drop(gyro_cols, axis=1)
            lower_df = get_features_group(lower_df, group)
            return lower_df
        elif sensor == 'gyro':
            lower_df = df.drop(upper_cols, axis=1)
            re_acc = re.compile("^Acc_*")
            acc_cols = list(filter(re_acc.match, list(lower_df.columns)))
            lower_df = lower_df.drop(acc_cols, axis=1)
            lower_df = get_features_group(lower_df, group)
            return lower_df
        elif sensor == 'both':
            lower_df = df.drop(upper_cols, axis=1)
            lower_df = get_features_group(lower_df, group)
            return lower_df
    elif pos == 'upper':
        if sensor == 'acc':
            upper_df = df.drop(lower_cols, axis=1)
            re_gyro = re.compile("^Gyro_*")
            gyro_cols = list(filter(re_gyro.match, list(upper_df.columns)))
            upper_df = upper_df.drop(gyro_cols, axis=1)
            upper_df = get_features_group(upper_df, group)
            return upper_df
        elif sensor == 'gyro':
            upper_df = df.drop(lower_cols, axis=1)
            re_acc = re.compile("^Acc_*")
            acc_cols = list(filter(re_acc.match, list(upper_df.columns)))
            upper_df = upper_df.drop(acc_cols, axis=1)
            upper_df = get_features_group(upper_df, group)
            return upper_df
        elif sensor == 'both':
            upper_df = df.drop(upper_cols, axis=1)
            upper_df = get_features_group(upper_df, group)
            return upper_df


def get_features_group(df, group):
    all_cols = df.columns
    if group == 'stat':
        stat_regex = re.compile(".*_avg|.*_med|.*_std|.*_max$|.*_min|.*_iqr_rng|.*_rms|.*_var")
        stat_cols = list(filter(stat_regex.match, all_cols))
        stat_cols.append('Label')
        df = df[stat_cols]
        return df
    elif group == 'spec':
        spec_regex = re.compile(".*_hum_eng|.*_spec_entropy|.*_max_freq|.*_pow_band|.*_max_pow_spec|.*_fi")
        spec_cols = list(filter(spec_regex.match, all_cols))
        spec_cols.append('Label')
        df = df[spec_cols]
        return df
    elif group == 'temp':
        temp_regex = re.compile(".*_total_eng|.*_slope|.*_max_peaks|.*_abs_eng|.*_dist")
        temp_cols = list(filter(temp_regex.match, all_cols))
        temp_cols.append('Label')
        df = df[temp_cols]
        return df




