import preprocessing
import modelling
import pandas as pd
import numpy as np
import datetime
import features_selection as fs
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from plotting import plot_loss_metric, plot_loss_accuracy, plot_metrics


# PRE-PROCESSING:: ATTENTION OF ACTIVATING THE FOLLOWING LINE - IT TAKES AROUND 12 HOURS
# preprocessing.create_rolled_train_test_dataframes(win_size=300, step_size=40)

# Read processed data
train_set = pd.read_csv('processed_data/train_set_w300_s40.csv', sep=',')
test_set = pd.read_csv('processed_data/test_set_w300_s40.csv', sep=',')

'''
# Re-Sampling 
X_train_resampled, y_train_resampled = SMOTE().fit_resample(train_set, train_set['Label'])
X_train_resampled = X_train_resampled.drop('Label', axis=1)
print(sorted(Counter(y_train_resampled).items()))
'''

'''
# Feature Selection :: #group --> (stat, spec, temp, freq, all)   #pos --> (feet, lower)   #sensor --> (acc, gyro, both)
train_set = fs.sensors_features(train_set, pos='feet', group='spec', sensor='acc')
test_set = fs.sensors_features(test_set, pos='feet', group='spec', sensor='acc')
'''

'''
# Drop Unimportant Features -- Make sure to drop the same columns list from all dataframes
drop_spec_features = ['_max_pow_spec', '_max_freq', '_spec_entropy']
drop_stat_features = ['_iqr_rng', '_var', '_med', '_rms']
train_df = fs.drop_features(train_df, drop_stat_features)
test_df = fs.drop_features(test_df, drop_stat_features)
'''

X_train, y_train = train_set.drop('Label', axis=1), train_set['Label']
X_test, y_test = test_set.drop('Label', axis=1), test_set['Label']

print("training set shape: ", X_train.shape)
print("training set label count: \n", y_train.value_counts())
print("testing set shape: ", X_test.shape)
print("testing set label count: \n", y_test.value_counts())

# Modelling
models = ['ANN']
for m in models:
    print("Model " + m + " started at:  ", datetime.datetime.now())

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
        model.fit(X_train, y_train)
        model.predict(X_test)
        model.accuracy(y_test)
        model.f1_score(y_test)
        model.auc_score(y_test)
        model.conf_matrix(y_test)
        model.clf_report(y_test)
    elif m == 'KNN':
        model = modelling.call_knn_model()
        model.fit(X_train, y_train)
        model.predict(X_test)
        model.accuracy(y_test)
        model.f1_score(y_test)
        model.auc_score(y_test)
        model.conf_matrix(y_test)
        model.clf_report(y_test)
    elif m == 'KNN_DTW':
        model = modelling.call_knn_dtw_model()
    elif m == 'ANN':
        input_dim = X_train.shape[1]
        num_hidden_layers = 5
        model = modelling.call_ann_model(input_dim, num_hidden_layers)
        X_train, X_test = model.features_scaling(X_train, X_test)
        model.fit(X_train, y_train, X_test, y_test, epochs=30, batch_size=64, verbose=2)
        plot_metrics(history=model.history, model_name='ANN')
        #plot_loss_metric(history=model.history, model_name="ANN")
    elif m == 'LSTM':
        TIME_STEPS = 5
        STEP = 1
        # X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
        X_train, y_train = modelling.create_3d_dataset(X_train, y_train, time_steps=TIME_STEPS, step=STEP)
        X_test, y_test = modelling.create_3d_dataset(X_test, y_test, time_steps=TIME_STEPS, step=STEP)
        print('Training Label Value Counts: \n', np.unique(y_train, return_counts=True))
        print('Test Label Value Counts: \n', np.unique(y_test, return_counts=True))
        input_dim = (X_train.shape[1], X_train.shape[2])
        model = modelling.call_lstm_model(input_dim)
        X_train, X_test = model.features_scaling(X_train, X_test, min_max=True)
        y_train, y_test = model.one_hot_labels(y_train, y_test)
        model.fit(X_train, y_train, X_test, y_test, epochs=30, batch_size=32, verbose=2)
        model.predict(X_test)
        plot_loss_accuracy(history=model.history, pic_name="loss_acc_lstm")
print("Modelling end:  ", datetime.datetime.now())

