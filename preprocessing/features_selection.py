import re


def sensors_features(df, pos, group, sensor, leg):
    """

    :param df:
    :param pos:
    :param group:
    :param sensor:
    :param leg:
    :return:
    """
    all_cols = df.columns
    lowerleg_cols = list(filter(re.compile(".*_lower*").match, all_cols))
    feet_cols = [x for x in all_cols if x not in lowerleg_cols]
    feet_cols.remove('Label')
    if pos == 'shank':
        if leg == 'left':
            rightleg_cols = list(filter(re.compile(".*right*").match, lowerleg_cols))
            lowerleg_df = df.drop(feet_cols, axis=1)
            lowerleg_df = lowerleg_df.drop(rightleg_cols, axis=1)
        elif leg == 'right':
            leftleg_cols = list(filter(re.compile(".*left*").match, lowerleg_cols))
            lowerleg_df = df.drop(feet_cols, axis=1)
            lowerleg_df = lowerleg_df.drop(leftleg_cols, axis=1)
        elif leg == 'both':
            lowerleg_df = df.drop(feet_cols, axis=1)
        if sensor == 'acc':
            re_gyro = re.compile("^Gyro_*")
            gyro_cols = list(filter(re_gyro.match, list(lowerleg_df.columns)))
            lowerleg_df = lowerleg_df.drop(gyro_cols, axis=1)
            if group != 'all':
                if type(group) is list:
                    lowerleg_df = _get_features_custom_group(lowerleg_df, group)
                else:
                    lowerleg_df = _get_features_group(lowerleg_df, group)
            return lowerleg_df
        elif sensor == 'gyro':
            re_acc = re.compile("^Acc_*")
            acc_cols = list(filter(re_acc.match, list(lowerleg_df.columns)))
            lowerleg_df = lowerleg_df.drop(acc_cols, axis=1)
            if group != 'all':
                if type(group) is list:
                    lowerleg_df = _get_features_custom_group(lowerleg_df, group)
                else:
                    lowerleg_df = _get_features_group(lowerleg_df, group)
            return lowerleg_df
        elif sensor == 'both':
            lowerleg_df = df.drop(feet_cols, axis=1)
            if group != 'all':
                if type(group) is list:
                    lowerleg_df = _get_features_custom_group(lowerleg_df, group)
                else:
                    lowerleg_df = _get_features_group(lowerleg_df, group)
            return lowerleg_df
    elif pos == 'feet':
        if leg == 'left':
            rightleg_cols = list(filter(re.compile(".*right*").match, feet_cols))
            feet_df = df.drop(lowerleg_cols, axis=1)
            feet_df = feet_df.drop(rightleg_cols, axis=1)
        elif leg == 'right':
            leftleg_cols = list(filter(re.compile(".*left*").match, feet_cols))
            feet_df = df.drop(lowerleg_cols, axis=1)
            feet_df = feet_df.drop(leftleg_cols, axis=1)
        elif leg == 'both':
            feet_df = df.drop(lowerleg_cols, axis=1)
        if sensor == 'acc':
            re_gyro = re.compile("^Gyro_*")
            gyro_cols = list(filter(re_gyro.match, list(feet_df.columns)))
            feet_df = feet_df.drop(gyro_cols, axis=1)
            if group != 'all':
                if type(group) is list:
                    feet_df = _get_features_custom_group(feet_df, group)
                else:
                    feet_df = _get_features_group(feet_df, group)
            return feet_df
        elif sensor == 'gyro':
            re_acc = re.compile("^Acc_*")
            acc_cols = list(filter(re_acc.match, list(feet_df.columns)))
            feet_df = feet_df.drop(acc_cols, axis=1)
            if group != 'all':
                if type(group) is list:
                    feet_df = _get_features_custom_group(feet_df, group)
                else:
                    feet_df = _get_features_group(feet_df, group)
            return feet_df
        elif sensor == 'both':
            if group != 'all':
                if type(group) is list:
                    feet_df = _get_features_custom_group(feet_df, group)
                else:
                    feet_df = _get_features_group(feet_df, group)
            return feet_df
    elif pos == 'all':
        all_df = df.copy()
        if sensor == 'acc':
            re_gyro = re.compile("^Gyro_*")
            gyro_cols = list(filter(re_gyro.match, list(all_df.columns)))
            all_df = all_df.drop(gyro_cols, axis=1)
            if group != 'all':
                if type(group) is list:
                    all_df = _get_features_custom_group(all_df, group)
                else:
                    all_df = _get_features_group(all_df, group)
            return all_df
        elif sensor == 'gyro':
            re_acc = re.compile("^Acc_*")
            acc_cols = list(filter(re_acc.match, list(all_df.columns)))
            all_df = all_df.drop(acc_cols, axis=1)
            if group != 'all':
                if type(group) is list:
                    all_df = _get_features_custom_group(all_df, group)
                else:
                    all_df = _get_features_group(all_df, group)
            return all_df
        elif sensor == 'both':
            if group != 'all':
                if type(group) is list:
                    all_df = _get_features_custom_group(all_df, group)
                else:
                    all_df = _get_features_group(all_df, group)
            return all_df


def _get_features_group(df, group):
    all_cols = df.columns
    if group == 'stat':
        stat_regex = re.compile(".*_avg|.*_med|.*_std|.*_max$|.*_min|.*_rms|.*_var")
        stat_cols = list(filter(stat_regex.match, all_cols))
        stat_cols.append('Label')
        df = df[stat_cols]
        return df
    elif group == 'freq':
        temp_regex = re.compile(".*_fi|.*_pi|.*_fp|.*_lp")
        temp_cols = list(filter(temp_regex.match, all_cols))
        temp_cols.append('Label')
        df = df[temp_cols]
        return df


def _get_features_custom_group(df, group_list):
    all_cols = df.columns
    cols = []
    for feature in group_list:
        if feature == 'max':
            regex = re.compile(".*_" + feature + '$')
        else:
            regex = re.compile(".*_"+feature)
        cols = cols + list(filter(regex.match, all_cols))
    cols.append('Label')
    df = df[cols]
    return df


def drop_features(df, features_list):
    new_df = df.copy()
    for f in features_list:
        regex = re.compile(".*"+f)
        cols = list(filter(regex.match, list(new_df.columns)))
        new_df = new_df.drop(cols, axis=1)
    return new_df




