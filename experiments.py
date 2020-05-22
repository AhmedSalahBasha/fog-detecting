import re
import pandas as pd


def sensors_features(df, sensor, group):
    all_cols = df.columns
    r = re.compile(".*_lower*")
    lower_cols = list(filter(r.match, all_cols))
    upper_cols = [x for x in all_cols if x not in lower_cols]
    upper_cols.remove('Label')
    if sensor == 'lower':
        lower_df = df.drop(upper_cols, axis=1)
        lower_df = get_features_group(lower_df, group)
        return lower_df
    elif sensor == 'upper':
        upper_df = df.drop(lower_cols, axis=1)
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
        spec_regex = re.compile(".*_hum_eng|.*_spec_entropy|.*_max_freq|.*_pow_band|.*_max_pow_spec")
        spec_cols = list(filter(spec_regex.match, all_cols))
        spec_cols.append('Label')
        df = df[spec_cols]
        return df
    elif group == 'temp':
        temp_regex = re.compile(".*_total_eng|.*_slope|.*_max_peaks|.*_abs_eng|.*_dist|.*_")
        temp_cols = list(filter(temp_regex.match, all_cols))
        temp_cols.append('Label')
        df = df[temp_cols]
        return df




