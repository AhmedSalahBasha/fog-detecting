import pandas as pd
import numpy as np
import datetime
from modelling import modelling
import preprocessing.features_selection as fs
from plots import plotting
from preprocessing import preprocessing as pre
from tensorflow.keras.backend import clear_session
import tensorflow as tf

tf.random.set_seed(123)
# tf.random.set_random_seed(123)
np.random.seed(123)

# full_rolled_df = pre.apply_rolling_on_full_dataframe(win_size=400, step_size=40)
full_rolled_df = pd.read_csv('data/processed_data/full_rolled_dataset_winsize_400.csv')

# Split dataset into train/test/dev based on patients
train_df = full_rolled_df[full_rolled_df['patient'].isin(['G04', 'G05', 'G08', 'P351', 'P551', 'P812', 'P876', 'P940', 'P645', 'G07', 'P231'])]
test_df = full_rolled_df[full_rolled_df['patient'].isin(['G06', 'G09', 'G11', 'P623', 'P379'])]

print('Full Dataset Shape: \n', full_rolled_df.shape)
print('Training Dataset Shape: \n', train_df.shape)
print('Testing Dataset Shape: \n', test_df.shape)

cols = ['train_shape', 'test_shape', 'train_lable_count', 'test_label_count',
        'sensor_pos', 'sensor_type', 'features', 'leg', 'model_name', 'duration',
        'f1_score', 'precision', 'recall', 'conf_matrix', 'clf_report', 'loss']
results_df = pd.DataFrame(columns=cols)
#results_df['train_shape'] = results_df['train_shape'].astype(object)
#results_df['test_shape'] = results_df['test_shape'].astype(object)
#results_df['train_lable_count'] = results_df['train_lable_count'].astype(object)
#results_df['test_label_count'] = results_df['test_label_count'].astype(object)

# Parameters
idx = 0
legs = ['left', 'right', 'both']
sensors_list = ['acc', 'gyro', 'both']
positions_list = ['shank', 'feet', 'all']
features_list = ['stat', 'freq', 'all']

for LEG in legs:
    for SENSOR_TYPE in sensors_list:
        for SENSOR_POS in positions_list:
            for FEATURES_GROUP in features_list:
                print('$$$ SENSOR_TYPE : ', SENSOR_TYPE)
                print('$$$ SENSOR_POSITION : ', SENSOR_POS)
                print('$$$ FEATURES_GROUP : ', FEATURES_GROUP)
                print('$$$ LEG : ', LEG)
                # Feature Selection
                train = fs.sensors_features(train_df, pos=SENSOR_POS, group=FEATURES_GROUP, sensor=SENSOR_TYPE, leg=LEG)
                test = fs.sensors_features(test_df, pos=SENSOR_POS, group=FEATURES_GROUP, sensor=SENSOR_TYPE, leg=LEG)

                # drop column patient or trials if they exist
                cols_to_drop = ['patient', 'trials']
                if set(cols_to_drop).issubset(train.columns):
                    train.drop(cols_to_drop, axis=1, inplace=True)
                if set(cols_to_drop).issubset(test.columns):
                    test.drop(cols_to_drop, axis=1, inplace=True)

                # Modelling
                models = ['LSTM']
                for m in models:
                    clear_session() # for clearing the Model cache to avoid consuming memory over time

                    # Train Test Split
                    X_train, y_train = train.drop('Label', axis=1), train['Label']
                    X_test, y_test = test.drop('Label', axis=1), test['Label']
                    print("Training set label count: \n", y_train.value_counts())
                    print("Testing set label count: \n", y_test.value_counts())
                    print("Training set shape: \n", X_train.shape)
                    print("Testing set shape: \n", X_test.shape)

                    STARTED_AT = datetime.datetime.now()
                    print("# Model " + m + " started #")
                    results_df.at[idx, 'train_shape'] = str(X_train.shape)
                    results_df.at[idx, 'test_shape'] = str(X_test.shape)
                    results_df.at[idx, 'train_lable_count'] = str(y_train.value_counts().to_dict())
                    results_df.at[idx, 'test_label_count'] = str(y_test.value_counts().to_dict())
                    results_df.at[idx, 'sensor_pos'] = SENSOR_POS
                    results_df.at[idx, 'sensor_type'] = SENSOR_TYPE
                    results_df.at[idx, 'features'] = FEATURES_GROUP
                    results_df.at[idx, 'leg'] = LEG
                    if m == 'SVM':
                        model = modelling.call_svm_model()
                        X_train_scaled, X_test_scaled = model.features_scaling(X_train, X_test)
                        model.fit(X_train_scaled, y_train)
                        model.predict(X_test_scaled)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = model.accuracy(y_test)
                        results_df.at[idx, 'f1_score'] = model.f1_score(y_test)
                        results_df.at[idx, 'precision'] = model.precision(y_test)
                        results_df.at[idx, 'recall'] = model.recall(y_test)
                        results_df.at[idx, 'conf_matrix'] = model.conf_matrix(y_test)
                        results_df.at[idx, 'clf_report'] = model.clf_report(y_test)
                    elif m == 'RF':
                        model = modelling.call_rf_model()
                        X_train_scaled, X_test_scaled = model.features_scaling(X_train, X_test)
                        model.fit(X_train_scaled, y_train)
                        model.predict(X_test_scaled)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = model.accuracy(y_test)
                        results_df.at[idx, 'f1_score'] = model.f1_score(y_test)
                        results_df.at[idx, 'precision'] = model.precision(y_test)
                        results_df.at[idx, 'recall'] = model.recall(y_test)
                        results_df.at[idx, 'conf_matrix'] = model.conf_matrix(y_test)
                        results_df.at[idx, 'clf_report'] = model.clf_report(y_test)
                    elif m == 'GNB':
                        model = modelling.call_gnb_model()
                        X_train_scaled, X_test_scaled = model.features_scaling(X_train, X_test)
                        model.fit(X_train_scaled, y_train)
                        model.predict(X_test_scaled)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = model.accuracy(y_test)
                        results_df.at[idx, 'f1_score'] = model.f1_score(y_test)
                        results_df.at[idx, 'precision'] = model.precision(y_test)
                        results_df.at[idx, 'recall'] = model.recall(y_test)
                        results_df.at[idx, 'conf_matrix'] = model.conf_matrix(y_test)
                        results_df.at[idx, 'clf_report'] = model.clf_report(y_test)
                    elif m == 'KNN':
                        model = modelling.call_knn_model()
                        X_train_scaled, X_test_scaled = model.features_scaling(X_train, X_test)
                        model.fit(X_train_scaled, y_train)
                        model.predict(X_test_scaled)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = model.accuracy(y_test)
                        results_df.at[idx, 'f1_score'] = model.f1_score(y_test)
                        results_df.at[idx, 'precision'] = model.precision(y_test)
                        results_df.at[idx, 'recall'] = model.recall(y_test)
                        results_df.at[idx, 'conf_matrix'] = model.conf_matrix(y_test)
                        results_df.at[idx, 'clf_report'] = model.clf_report(y_test)
                    elif m == 'KNN_DTW':
                        model = modelling.call_knn_dtw_model()
                        X_train_scaled, X_test_scaled = model.features_scaling(X_train, X_test)
                        model.fit(X_train_scaled, y_train)
                        model.predict(X_test_scaled)
                        results_df.at[idx, 'model_name'] = model.model_name
                        results_df.at[idx, 'accuracy'] = model.accuracy(y_test)
                        results_df.at[idx, 'f1_score'] = model.f1_score(y_test)
                        results_df.at[idx, 'precision'] = model.precision(y_test)
                        results_df.at[idx, 'recall'] = model.recall(y_test)
                        results_df.at[idx, 'conf_matrix'] = model.conf_matrix(y_test)
                        results_df.at[idx, 'clf_report'] = model.clf_report(y_test)
                    elif m == 'LSTM':
                        TIME_STEPS = 3
                        STEP = 1
                        NUM_HIDDEN_LAYERS = 3
                        BATCH_SIZE = 64
                        EPOCHS = 50
                        X_train_3d, y_train_3d = modelling.create_3d_dataset(X_train, y_train, time_steps=TIME_STEPS,
                                                                             step=STEP)
                        X_test_3d, y_test_3d = modelling.create_3d_dataset(X_test, y_test, time_steps=TIME_STEPS,
                                                                           step=STEP)
                        input_dim = (X_train_3d.shape[1], X_train_3d.shape[2])
                        print('Input dimension: ', input_dim)
                        print('X_train_3d shape: ', X_train_3d.shape)
                        print('X_test_3d shape: ', X_test_3d.shape)
                        model = modelling.call_lstm_model(input_dim, NUM_HIDDEN_LAYERS)
                        X_train_scaled, X_test_scaled = model.features_scaling_3d(X_train_3d, X_test_3d)
                        y_train_3d, y_test_3d = model.one_hot_labels(y_train_3d, y_test_3d)
                        model.fit(X_train_scaled, y_train_3d, X_test_scaled, y_test_3d, epochs=EPOCHS,
                                  batch_size=BATCH_SIZE, verbose=0)
                        results = model.evaluate(X_test_scaled, y_test_3d, batch_size=BATCH_SIZE)
                        y_pred = model.predict(X_test_scaled)
                        results_df.at[idx, 'conf_matrix'] = model.conf_matrix(y_test_3d)
                        results_df.at[idx, 'clf_report'] = model.clf_report(y_test_3d)
                        # tpr, fpr, auc = model.roc_auc(y_test_3d)
                        for key, value in results.items():
                            results_df.at[idx, key] = value

                    ENDED_AT = datetime.datetime.now()
                    print("======== Model end:  " + str(ENDED_AT) + " ============")
                    results_df.at[idx, 'duration'] = str(ENDED_AT - STARTED_AT)
                    idx += 1

results_df.to_csv('results/train_test_results_lstm.csv', sep=',', index=False)



