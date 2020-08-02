import pandas as pd
import numpy as np
import datetime
from preprocessing import preprocessing
from modelling import modelling
import preprocessing.features_selection as fs
from plots import plotting
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter



# PRE-PROCESSING:: ATTENTION OF ACTIVATING THE FOLLOWING LINE - IT TAKES AROUND 12 HOURS
WIN_SIZE, STEP_SIZE = 400, 40
full_rolled_df = preprocessing.apply_rolling_on_full_dataframe(win_size=WIN_SIZE, step_size=STEP_SIZE)

patients = ['G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G11',
            'P231', 'P351', 'P379', 'P551', 'P623', 'P645', 'P812', 'P876', 'P940']

cols = ['test_patient', 'train_shape', 'test_shape', 'train_lable_count', 'test_label_count',
        'sensor_pos', 'sensor_type', 'features',
        'model_name', 'started_at', 'ended_at', 'duration', 'accuracy', 'f1_score', 'auc_score', 'conf_matrix', 'clf_report']
results_df = pd.DataFrame(columns=cols)
results_df['train_shape'] = results_df['train_shape'].astype(object)
results_df['test_shape'] = results_df['test_shape'].astype(object)
results_df['train_lable_count'] = results_df['train_lable_count'].astype(object)
results_df['test_label_count'] = results_df['test_label_count'].astype(object)
idx = 0
for p in range(len(patients)):
    PATIENT = patients[p]
    train_set, test_set = preprocessing.leave_one_patient_out(full_rolled_df, test_patient=PATIENT)

    # Feature Selection :: #group --> (stat, spec, temp, freq, all, list_of_features_names)   #pos --> (feet, lower)   #sensor --> (acc, gyro, both)
    SENSOR_POS = 'feet'
    SENSOR_TYPE = 'acc'
    FEATURES = ['avg', 'std', 'med', 'max', 'min', 'var', 'rms', 'fi', 'pi', 'fp', 'lp']
    train_set = fs.sensors_features(train_set, pos=SENSOR_POS, group=FEATURES, sensor=SENSOR_TYPE)
    test_set = fs.sensors_features(test_set, pos=SENSOR_POS, group=FEATURES, sensor=SENSOR_TYPE)

    print("training set shape: ", train_set.shape)
    print("training set label count: \n", train_set['Label'].value_counts())
    print("testing set shape: ", test_set.shape)
    print("testing set label count: \n", test_set['Label'].value_counts())

    # Modelling
    models = ['SVM', 'RF', 'GNB', 'KNN', 'KNN_DTW']
    for m in models:
        STARTED_AT = datetime.datetime.now()
        print("Model " + m + " started at:  ", str(STARTED_AT))

        X_train, y_train = train_set.drop('Label', axis=1), train_set['Label']
        X_test, y_test = test_set.drop('Label', axis=1), test_set['Label']

        results_df.at[idx, 'test_patient'] = PATIENT
        results_df.at[idx, 'train_shape'] = train_set.shape
        results_df.at[idx, 'test_shape'] = test_set.shape
        results_df.at[idx, 'train_lable_count'] = train_set['Label'].value_counts().to_dict()
        results_df.at[idx, 'test_label_count'] = test_set['Label'].value_counts().to_dict()
        results_df.at[idx, 'sensor_pos'] = SENSOR_POS
        results_df.at[idx, 'sensor_type'] = SENSOR_TYPE
        results_df.at[idx, 'features'] = FEATURES

        if m == 'SVM':
            #modelling.grid_search_svm(X_train, y_train)
            model = modelling.call_svm_model()
            model.fit(X_train, y_train)
            model.predict(X_test)
            accuracy = model.accuracy(y_test)
            f1_score = model.f1_score(y_test)
            auc_score = model.auc_score(y_test)
            matrix = model.conf_matrix(y_test)
            report = model.clf_report(y_test)
            results_df.at[idx, 'model_name'] = model.model_name
            results_df.at[idx, 'started_at'] = STARTED_AT
            results_df.at[idx, 'accuracy'] = accuracy
            results_df.at[idx, 'f1_score'] = f1_score
            results_df.at[idx, 'auc_score'] = auc_score
            results_df.at[idx, 'conf_matrix'] = matrix
            results_df.at[idx, 'clf_report'] = report
        elif m == 'RF':
            model = modelling.call_rf_model()
            model.fit(X_train, y_train)
            model.predict(X_test)
            accuracy = model.accuracy(y_test)
            f1_score = model.f1_score(y_test)
            auc_score = model.auc_score(y_test)
            matrix = model.conf_matrix(y_test)
            report = model.clf_report(y_test)
            results_df.at[idx, 'model_name'] = model.model_name
            results_df.at[idx, 'started_at'] = STARTED_AT
            results_df.at[idx, 'accuracy'] = accuracy
            results_df.at[idx, 'f1_score'] = f1_score
            results_df.at[idx, 'auc_score'] = auc_score
            results_df.at[idx, 'conf_matrix'] = matrix
            results_df.at[idx, 'clf_report'] = report
        elif m == 'GNB':
            model = modelling.call_gnb_model()
            model.fit(X_train, y_train)
            model.predict(X_test)
            accuracy = model.accuracy(y_test)
            f1_score = model.f1_score(y_test)
            auc_score = model.auc_score(y_test)
            matrix = model.conf_matrix(y_test)
            report = model.clf_report(y_test)
            results_df.at[idx, 'model_name'] = model.model_name
            results_df.at[idx, 'started_at'] = STARTED_AT
            results_df.at[idx, 'accuracy'] = accuracy
            results_df.at[idx, 'f1_score'] = f1_score
            results_df.at[idx, 'auc_score'] = auc_score
            results_df.at[idx, 'conf_matrix'] = matrix
            results_df.at[idx, 'clf_report'] = report
        elif m == 'KNN':
            model = modelling.call_knn_model()
            model.fit(X_train, y_train)
            model.predict(X_test)
            accuracy = model.accuracy(y_test)
            f1_score = model.f1_score(y_test)
            auc_score = model.auc_score(y_test)
            matrix = model.conf_matrix(y_test)
            report = model.clf_report(y_test)
            results_df.at[idx, 'model_name'] = model.model_name
            results_df.at[idx, 'started_at'] = STARTED_AT
            results_df.at[idx, 'accuracy'] = accuracy
            results_df.at[idx, 'f1_score'] = f1_score
            results_df.at[idx, 'auc_score'] = auc_score
            results_df.at[idx, 'conf_matrix'] = matrix
            results_df.at[idx, 'clf_report'] = report
        elif m == 'KNN_DTW':
            model = modelling.call_knn_dtw_model()
            model.fit(X_train, y_train)
            model.predict(X_test)
            accuracy = model.accuracy(y_test)
            f1_score = model.f1_score(y_test)
            auc_score = model.auc_score(y_test)
            matrix = model.conf_matrix(y_test)
            report = model.clf_report(y_test)
            results_df.at[idx, 'model_name'] = model.model_name
            results_df.at[idx, 'started_at'] = STARTED_AT
            results_df.at[idx, 'accuracy'] = accuracy
            results_df.at[idx, 'f1_score'] = f1_score
            results_df.at[idx, 'auc_score'] = auc_score
            results_df.at[idx, 'conf_matrix'] = matrix
            results_df.at[idx, 'clf_report'] = report
        elif m == 'ANN':
            input_dim = X_train.shape[1]
            NUM_HIDDEN_LAYERS = 5
            BATCH_SIZE = 64
            EPOCHS = 100
            model = modelling.call_ann_model(input_dim, NUM_HIDDEN_LAYERS)
            X_train, X_test = model.features_scaling(X_train, X_test)
            model.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
            model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
            y_pred = model.predict(X_test)
            plotting.plot_metrics(history=model.history, model_name=model.model_name)
            plotting.plot_cm(true_labels=y_test, predictions=y_pred, model_name=model.model_name)
        elif m == 'LSTM':
            TIME_STEPS = 3
            STEP = 1
            NUM_HIDDEN_LAYERS = 3
            BATCH_SIZE = 64
            EPOCHS = 30
            # X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
            X_train, y_train = modelling.create_3d_dataset(X_train, y_train, time_steps=TIME_STEPS, step=STEP)
            X_test, y_test = modelling.create_3d_dataset(X_test, y_test, time_steps=TIME_STEPS, step=STEP)
            print('Training Label Value Counts: \n', np.unique(y_train, return_counts=True))
            print('Test Label Value Counts: \n', np.unique(y_test, return_counts=True))
            input_dim = (X_train.shape[1], X_train.shape[2])
            model = modelling.call_lstm_model(input_dim, NUM_HIDDEN_LAYERS)
            X_train, X_test = model.features_scaling(X_train, X_test, min_max=True)
            y_train, y_test = model.one_hot_labels(y_train, y_test)
            model.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
            model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
            y_pred = model.predict(X_test)
            plotting.plot_metrics(history=model.history, model_name=model.model_name)
            plotting.plot_cm(true_labels=y_test.argmax(axis=1), predictions=y_pred.argmax(axis=1), model_name=model.model_name)
        ENDED_AT = datetime.datetime.now()
        print("======== Model end:  " + str(ENDED_AT) + " ============")
        results_df.at[idx, 'ended_at'] = ENDED_AT
        results_df.at[idx, 'duration'] = str(ENDED_AT - STARTED_AT)
        idx += 1

results_df.to_csv('results/results_ml_outliers_removed.csv', sep=',')







'''
# Re-Sampling 
X_train_resampled, y_train_resampled = SMOTE().fit_resample(train_set, train_set['Label'])
X_train_resampled = X_train_resampled.drop('Label', axis=1)
print(sorted(Counter(y_train_resampled).items()))


# Drop Unimportant Features -- Make sure to drop the same columns list from all dataframes
drop_spec_features = ['_max_pow_spec', '_max_freq', '_spec_entropy']
drop_stat_features = ['_iqr_rng', '_var', '_med', '_rms']
train_df = fs.drop_features(train_df, drop_stat_features)
test_df = fs.drop_features(test_df, drop_stat_features)
'''