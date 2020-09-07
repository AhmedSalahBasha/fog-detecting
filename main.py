import pandas as pd
import numpy as np
import datetime

from imblearn.over_sampling import SMOTE
from collections import Counter
from preprocessing import preprocessing
from modelling import modelling
import preprocessing.features_selection as fs
from plots import plotting
from keras.backend import clear_session


# PRE-PROCESSING:: ATTENTION OF ACTIVATING THE FOLLOWING LINE - IT TAKES AROUND 12 HOURS
WIN_SIZE, STEP_SIZE = 400, 40
#full_rolled_df = preprocessing.apply_rolling_on_full_dataframe(win_size=WIN_SIZE, step_size=STEP_SIZE)
full_rolled_df = pd.read_csv('data/processed_data/full_rolled_dataset_w400_s40.csv')

# Drop Unimportant Features
drop_features_list = ['_spec_dist', '_hum_eng', '_max_pow_spec', '_max_freq', '_spec_entropy', '_pow_band', '_slope', '_max_peaks', '_total_eng', '_abs_eng', '_dist']
full_rolled_df = fs.drop_features(full_rolled_df, drop_features_list)

patients = ['G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G11',
            'P231', 'P351', 'P379', 'P551', 'P623', 'P645', 'P812', 'P876', 'P940']

cols = ['test_patient', 'train_shape', 'test_shape', 'train_lable_count', 'test_label_count',
        'sensor_pos', 'sensor_type', 'features', 'model_name', 'started_at', 'ended_at', 'duration',
        'accuracy', 'f1_score', 'precision', 'recall', 'conf_matrix', 'clf_report', 'loss']
results_df = pd.DataFrame(columns=cols)
results_df['train_shape'] = results_df['train_shape'].astype(object)
results_df['test_shape'] = results_df['test_shape'].astype(object)
results_df['train_lable_count'] = results_df['train_lable_count'].astype(object)
results_df['test_label_count'] = results_df['test_label_count'].astype(object)
idx = 0
sensors_list = ['acc', 'gyro', 'both']
positions_list = ['shank', 'feet', 'all']
features_list = ['stat', 'freq', 'all']

for s in sensors_list:
    SENSOR_TYPE = s
    for sp in positions_list:
        SENSOR_POS = sp
        for f in features_list:
            FEATURES_GROUP = f
            for p in range(len(patients)):
                print("Testing Patient >>>>  ", patients[p])
                PATIENT = patients[p]
                train_set, test_set = preprocessing.leave_one_patient_out(full_rolled_df, test_patient=PATIENT)

                '''
                if len(test_set['Label'].value_counts().to_dict()) < 2:     # skip the patient if one of the two classes is missing
                    print("#################################")
                    print("Patient: ", PATIENT, " is being passed. Test Labels Count is: ", test_set['Label'].value_counts().to_dict())
                    print("#################################")
                    pass
                else:
                '''
                print("training set label count: \n", train_set['Label'].value_counts())
                print("testing set label count: \n", test_set['Label'].value_counts())

                # Feature Selection :: #group --> (stat, freq, all, list_of_features_names)   #pos --> (feet, shank, all)   #sensor --> (acc, gyro, both)
                train_set = fs.sensors_features(train_set, pos=SENSOR_POS, group=FEATURES_GROUP, sensor=SENSOR_TYPE)
                test_set = fs.sensors_features(test_set, pos=SENSOR_POS, group=FEATURES_GROUP, sensor=SENSOR_TYPE)

                # drop column patient or trials if they exist
                cols_to_drop = ['patient', 'trials']
                if set(cols_to_drop).issubset(train_set.columns):
                    train_set.drop(cols_to_drop, axis=1, inplace=True)
                if set(cols_to_drop).issubset(test_set.columns):
                    test_set.drop(cols_to_drop, axis=1, inplace=True)

                # Modelling
                models = ['SVM', 'RF', 'DT', 'GNB', 'KNN', 'KNN_DTW']
                for m in models:
                    clear_session() # for clearing the Model cache to avoid consuming memory over time
                    STARTED_AT = datetime.datetime.now()
                    print("# Model " + m + " started #")

                    X_train, y_train = train_set.drop('Label', axis=1), train_set['Label']
                    X_test, y_test = test_set.drop('Label', axis=1), test_set['Label']

                    results_df.at[idx, 'test_patient'] = PATIENT
                    results_df.at[idx, 'train_shape'] = train_set.shape
                    results_df.at[idx, 'test_shape'] = test_set.shape
                    results_df.at[idx, 'train_lable_count'] = train_set['Label'].value_counts().to_dict()
                    results_df.at[idx, 'test_label_count'] = test_set['Label'].value_counts().to_dict()
                    results_df.at[idx, 'sensor_pos'] = SENSOR_POS
                    results_df.at[idx, 'sensor_type'] = SENSOR_TYPE
                    results_df.at[idx, 'features'] = FEATURES_GROUP
                    if m == 'SVM':
                        model = modelling.call_svm_model()
                        model.fit(X_train, y_train)
                        model.predict(X_test)
                        accuracy = model.accuracy(y_test)
                        f1_score = model.f1_score(y_test)
                        precision = model.precision(y_test)
                        recall = model.recall(y_test)
                        matrix = model.conf_matrix(y_test)
                        clf_report = model.clf_report(y_test)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = accuracy
                        results_df.at[idx, 'f1_score'] = f1_score
                        results_df.at[idx, 'precision'] = precision
                        results_df.at[idx, 'recall'] = recall
                        results_df.at[idx, 'conf_matrix'] = matrix
                        results_df.at[idx, 'clf_report'] = clf_report
                    elif m == 'RF':
                        model = modelling.call_rf_model()
                        model.fit(X_train, y_train)
                        model.predict(X_test)
                        accuracy = model.accuracy(y_test)
                        f1_score = model.f1_score(y_test)
                        precision = model.precision(y_test)
                        recall = model.recall(y_test)
                        matrix = model.conf_matrix(y_test)
                        clf_report = model.clf_report(y_test)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = accuracy
                        results_df.at[idx, 'f1_score'] = f1_score
                        results_df.at[idx, 'precision'] = precision
                        results_df.at[idx, 'recall'] = recall
                        results_df.at[idx, 'conf_matrix'] = matrix
                        results_df.at[idx, 'clf_report'] = clf_report
                    elif m == 'DT':
                        model = modelling.call_dt_model()
                        model.fit(X_train, y_train)
                        model.predict(X_test)
                        accuracy = model.accuracy(y_test)
                        f1_score = model.f1_score(y_test)
                        precision = model.precision(y_test)
                        recall = model.recall(y_test)
                        matrix = model.conf_matrix(y_test)
                        clf_report = model.clf_report(y_test)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = accuracy
                        results_df.at[idx, 'f1_score'] = f1_score
                        results_df.at[idx, 'precision'] = precision
                        results_df.at[idx, 'recall'] = recall
                        results_df.at[idx, 'conf_matrix'] = matrix
                        results_df.at[idx, 'clf_report'] = clf_report
                    elif m == 'GNB':
                        model = modelling.call_gnb_model()
                        model.fit(X_train, y_train)
                        model.predict(X_test)
                        accuracy = model.accuracy(y_test)
                        f1_score = model.f1_score(y_test)
                        precision = model.precision(y_test)
                        recall = model.recall(y_test)
                        matrix = model.conf_matrix(y_test)
                        clf_report = model.clf_report(y_test)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = accuracy
                        results_df.at[idx, 'f1_score'] = f1_score
                        results_df.at[idx, 'precision'] = precision
                        results_df.at[idx, 'recall'] = recall
                        results_df.at[idx, 'conf_matrix'] = matrix
                        results_df.at[idx, 'clf_report'] = clf_report
                    elif m == 'KNN':
                        model = modelling.call_knn_model()
                        model.fit(X_train, y_train)
                        model.predict(X_test)
                        accuracy = model.accuracy(y_test)
                        f1_score = model.f1_score(y_test)
                        precision = model.precision(y_test)
                        recall = model.recall(y_test)
                        matrix = model.conf_matrix(y_test)
                        clf_report = model.clf_report(y_test)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = accuracy
                        results_df.at[idx, 'f1_score'] = f1_score
                        results_df.at[idx, 'precision'] = precision
                        results_df.at[idx, 'recall'] = recall
                        results_df.at[idx, 'conf_matrix'] = matrix
                        results_df.at[idx, 'clf_report'] = clf_report
                    elif m == 'KNN_DTW':
                        model = modelling.call_knn_dtw_model()
                        model.fit(X_train, y_train)
                        model.predict(X_test)
                        accuracy = model.accuracy(y_test)
                        f1_score = model.f1_score(y_test)
                        precision = model.precision(y_test)
                        recall = model.recall(y_test)
                        matrix = model.conf_matrix(y_test)
                        clf_report = model.clf_report(y_test)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = accuracy
                        results_df.at[idx, 'f1_score'] = f1_score
                        results_df.at[idx, 'precision'] = precision
                        results_df.at[idx, 'recall'] = recall
                        results_df.at[idx, 'conf_matrix'] = matrix
                        results_df.at[idx, 'clf_report'] = clf_report
                    elif m == 'ANN':
                        input_dim = X_train.shape[1]
                        NUM_HIDDEN_LAYERS = 5
                        BATCH_SIZE = 64
                        EPOCHS = 50
                        model = modelling.call_ann_model(input_dim, NUM_HIDDEN_LAYERS)
                        X_train, X_test = model.features_scaling(X_train, X_test)
                        model.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                        results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
                        y_pred = model.predict(X_test)
                        cm = model.conf_matrix(y_test)
                        plotting.plot_metrics(history=model.history, model_name=model.model_name+'_'+PATIENT)
                        plotting.plot_cm(cm, model_name=model.model_name+'_'+PATIENT)
                        results_df.at[idx, 'model_name'] = model.model_name
                        #results_df.at[idx, 'started_at'] = STARTED_AT
                        results_df.at[idx, 'conf_matrix'] = cm
                        precision = results.get('precision')
                        recall = results.get('recall')
                        results_df.at[idx, 'f1_score'] = (2 * precision * recall) / (precision + recall)
                        for key, value in results.items():
                            results_df.at[idx, key] = value
                    elif m == 'LSTM':
                        TIME_STEPS = 3
                        STEP = 1
                        NUM_HIDDEN_LAYERS = 3
                        BATCH_SIZE = 64
                        EPOCHS = 50
                        # X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
                        X_train, y_train = modelling.create_3d_dataset(X_train, y_train, time_steps=TIME_STEPS, step=STEP)
                        X_test, y_test = modelling.create_3d_dataset(X_test, y_test, time_steps=TIME_STEPS, step=STEP)
                        print('Training Label Value Counts: \n', np.unique(y_train, return_counts=True))
                        print('Test Label Value Counts: \n', np.unique(y_test, return_counts=True))
                        input_dim = (X_train.shape[1], X_train.shape[2])
                        model = modelling.call_lstm_model(input_dim, NUM_HIDDEN_LAYERS)
                        X_train, X_test = model.features_scaling(X_train, X_test, min_max=True)
                        y_train, y_test = model.one_hot_labels(y_train, y_test)
                        model.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                        results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
                        y_pred = model.predict(X_test)
                        cm = model.conf_matrix(y_test)
                        plotting.plot_metrics(history=model.history, model_name=model.model_name+'_'+PATIENT)
                        plotting.plot_cm(cm, model_name=model.model_name+'_'+PATIENT)
                        results_df.at[idx, 'model_name'] = model.model_name
                        #results_df.at[idx, 'started_at'] = STARTED_AT
                        results_df.at[idx, 'conf_matrix'] = cm
                        precision = results.get('precision')
                        recall = results.get('recall')
                        results_df.at[idx, 'f1_score'] = (2 * precision * recall) / (precision + recall)
                        for key, value in results.items():
                            results_df.at[idx, key] = value
                    ENDED_AT = datetime.datetime.now()
                    print("======== Model end:  " + str(ENDED_AT) + " ============")
                    #results_df.at[idx, 'ended_at'] = ENDED_AT
                    results_df.at[idx, 'duration'] = str(ENDED_AT - STARTED_AT)
                    idx += 1

            results_df.to_csv('results/ml_'+s+'_'+sp+'_'+f+'.csv', sep=',', index=False)




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