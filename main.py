import preprocessing
import modelling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from keras.utils import to_categorical
import features_selection as fs
from keras.optimizers import SGD
from keras.optimizers import Adam

from plotting import plot_loss_accuracy, plot_loss_f1


# PRE-PROCESSING:: ATTENTION OF ACTIVATING THE FOLLOWING LINE - IT TAKES AROUND 12 HOURS
preprocessing.create_rolled_train_test_dataframes(win_size=300, step_size=40)

# Read processed data
train_set = pd.read_csv('processed_data/train_set.csv', sep=',')
dev_set = pd.read_csv('processed_data/dev_set.csv', sep=',')
test_set = pd.read_csv('processed_data/test_set.csv', sep=',')

new_test_set = pd.concat([test_set, dev_set], ignore_index=True)

# Feature Selection :: group --> (stat, spec, temp, freq, all)   # pos --> (upper, lower)   # sensor --> (acc, gyro, both)
train_df = fs.sensors_features(train_set, pos='upper', group='stat', sensor='acc')
test_df = fs.sensors_features(new_test_set, pos='upper', group='stat', sensor='acc')

X_train, y_train = train_df.drop('Label', axis=1), train_df['Label']
X_test, y_test = test_df.drop('Label', axis=1), test_df['Label']

# Print Label Value Counts
print('Training Label Value Counts: \n', train_df['Label'].value_counts())
print('Test Label Value Counts: \n', test_df['Label'].value_counts())

# Modelling
models = ['RF']
for m in models:
    print("Modelling model " + m + " start:  ", datetime.datetime.now())

    '''
    # Drop Unimportant Features -- Make sure to drop the same columns list from all dataframes
    drop_spec_features = ['_max_pow_spec', '_max_freq', '_spec_entropy']
    drop_stat_features = ['_iqr_rng', '_var', '_med', '_rms']
    train_df = fs.drop_features(train_df, drop_stat_features)
    dev_df = fs.drop_features(dev_df, drop_stat_features)
    test_df = fs.drop_features(test_df, drop_stat_features)
    '''


    # Calling models for training, testing and evaluation
    if m == 'SVM':
        model = modelling.call_svm_model()
        model.fit(X_train, y_train)
        model.predict(X_test)
        model.accuracy(y_test)
        model.f1_score(y_test)
        model.auc_score(y_test)
        model.conf_matrix(y_test)
        model.clf_report(y_test)
    elif m == 'RF':
        model = modelling.call_rf_model()
        model.fit(X_train, y_train)
        model.predict(X_test)
        model.accuracy(y_test)
        model.f1_score(y_test)
        model.auc_score(y_test)
        model.conf_matrix(y_test)
        model.clf_report(y_test)
    elif m == 'DT':
        model = modelling.call_dt_model()
    elif m == 'KNN':
        model = modelling.call_knn_model()
    elif m == 'KNN_DTW':
        model = modelling.call_knn_dtw_model()
    elif m == 'ANN':
        input_dim = X_train.shape[1]
        model = modelling.call_ann_model(input_dim)
    elif m == 'LSTM':
        TIME_STEPS = 200
        STEP = 40
        X_train, y_train = modelling.create_3d_dataset(X_train,
                                                       y_train,
                                                       time_steps=TIME_STEPS,
                                                       step=STEP)
        X_test, y_test = modelling.create_3d_dataset(X_test,
                                                     y_test,
                                                     time_steps=TIME_STEPS,
                                                     step=STEP)
        input_dim = (X_train.shape[1], X_train.shape[2])
        model = modelling.call_lstm_model(input_dim)
print("Modelling end:  ", datetime.datetime.now())

