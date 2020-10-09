import pandas as pd
import numpy as np
import datetime
from modelling import modelling
import preprocessing.features_selection as fs
from plots import plotting
from tensorflow.keras.backend import clear_session


full_rolled_df = pd.read_csv('data/processed_data/full_rolled_dataset_w400_s40.csv')
# Drop Unimportant Features
drop_features_list = ['_spec_dist', '_hum_eng', '_max_pow_spec', '_max_freq', '_spec_entropy', '_pow_band', '_slope', '_max_peaks', '_total_eng', '_abs_eng', '_dist']
full_rolled_df = fs.drop_features(full_rolled_df, drop_features_list)

# Split dataset into train/test/dev based on patients
train = full_rolled_df[full_rolled_df['patient'].isin(['G04', 'G05', 'G08', 'P351', 'P551' 'P812', 'P876', 'P940', 'P645', 'G07', 'P231'])]
test = full_rolled_df[full_rolled_df['patient'].isin(['G06', 'G09', 'G11', 'P623', 'P379'])]

cols = ['train_shape', 'test_shape', 'train_lable_count', 'test_label_count',
        'sensor_pos', 'sensor_type', 'features', 'model_name', 'duration',
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
    for sp in positions_list:
        for f in features_list:
            SENSOR_TYPE = s
            SENSOR_POS = sp
            FEATURES_GROUP = f
            print('$$$ SENSOR_TYPE : ', SENSOR_TYPE)
            print('$$$ SENSOR_POSITION : ', SENSOR_POS)
            print('$$$ FEATURES_GROUP : ', FEATURES_GROUP)
            # Feature Selection
            train = fs.sensors_features(train, pos=SENSOR_POS, group=FEATURES_GROUP, sensor=SENSOR_TYPE)
            test = fs.sensors_features(test, pos=SENSOR_POS, group=FEATURES_GROUP, sensor=SENSOR_TYPE)

            # Train Test Split
            X_train, y_train = train.drop('Label', axis=1), train['Label']
            X_test, y_test = test.drop('Label', axis=1), test['Label']
            print("training set label count: \n", y_train.value_counts())
            print("validation set label count: \n", y_test.value_counts())

            # Modelling
            models = ['LSTM', 'ANN', 'SVM', 'RF', 'DT', 'GNB', 'KNN', 'KNN_DTW']
            for m in models:
                clear_session() # for clearing the Model cache to avoid consuming memory over time
                STARTED_AT = datetime.datetime.now()
                print("# Model " + m + " started #")

                # Train Test Split
                X_train, y_train = train.drop('Label', axis=1), train['Label']
                X_test, y_test = test.drop('Label', axis=1), test['Label']
                print("training set label count: \n", y_train.value_counts())
                print("validation set label count: \n", y_test.value_counts())

                results_df.at[idx, 'train_shape'] = train.shape
                results_df.at[idx, 'test_shape'] = test.shape
                results_df.at[idx, 'train_lable_count'] = y_train.value_counts().to_dict()
                results_df.at[idx, 'test_label_count'] = y_test['Label'].value_counts().to_dict()
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
                    BATCH_SIZE = 32
                    EPOCHS = 50
                    model = modelling.call_ann_model(input_dim, NUM_HIDDEN_LAYERS)
                    X_train, X_test = model.features_scaling(X_train, X_test)
                    model.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                    results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
                    y_pred = model.predict(X_test)
                    cm = model.conf_matrix(y_test)
                    results_df.at[idx, 'model_name'] = model.model_name
                    results_df.at[idx, 'conf_matrix'] = cm
                    clf_report = model.clf_report(y_test)
                    results_df.at[idx, 'clf_report'] = clf_report
                    for key, value in results.items():
                        results_df.at[idx, key] = value
                elif m == 'LSTM':
                    TIME_STEPS = 3
                    STEP = 1
                    NUM_HIDDEN_LAYERS = 3
                    BATCH_SIZE = 32
                    EPOCHS = 50
                    X_train, y_train = modelling.create_3d_dataset(X_train, y_train, time_steps=TIME_STEPS, step=STEP)
                    X_test, y_test = modelling.create_3d_dataset(X_test, y_test, time_steps=TIME_STEPS, step=STEP)
                    input_dim = (X_train.shape[1], X_train.shape[2])
                    model = modelling.call_lstm_model(input_dim, NUM_HIDDEN_LAYERS)
                    X_train, X_test = model.features_scaling(X_train, X_test, min_max=True)
                    y_train, y_test = model.one_hot_labels(y_train, y_test)
                    model.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                    results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
                    y_pred = model.predict(X_test)
                    cm = model.conf_matrix(y_test)
                    results_df.at[idx, 'model_name'] = model.model_name
                    results_df.at[idx, 'conf_matrix'] = cm
                    clf_report = model.clf_report(y_test)
                    results_df.at[idx, 'clf_report'] = clf_report
                    for key, value in results.items():
                        results_df.at[idx, key] = value

                ENDED_AT = datetime.datetime.now()
                print("======== Model end:  " + str(ENDED_AT) + " ============")
                results_df.at[idx, 'duration'] = str(ENDED_AT - STARTED_AT)
                idx += 1

        results_df.to_csv('results/sensor_'+s+'_position_'+sp+'_features_'+f+'.csv', sep=',', index=False)



