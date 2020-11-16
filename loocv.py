import pandas as pd
import numpy as np
import datetime

from collections import Counter
from preprocessing import preprocessing
from modelling import modelling
import preprocessing.features_selection as fs
from plots import plotting
from tensorflow.keras.backend import clear_session

import tensorflow as tf
tf.random.set_seed(123)
# tf.random.set_random_seed(123)
np.random.seed(123)


# PRE-PROCESSING:: ATTENTION OF ACTIVATING THE FOLLOWING LINE - IT TAKES AROUND 12 HOURS
WIN_SIZE, STEP_SIZE = 900, 90
# full_rolled_df = preprocessing.apply_rolling_on_full_dataframe(win_size=WIN_SIZE, step_size=STEP_SIZE)
full_rolled_df = pd.read_csv('data/processed_data/feet_gyro_rolled_dataset_winsize_900.csv')

cols = ['test_patient', 'train_shape', 'test_shape', 'train_lable_count', 'test_label_count',
        'accuracy', 'f1_score', 'precision', 'recall', 'conf_matrix', 'clf_report', 'loss']
results_df = pd.DataFrame(columns=cols)
results_df['train_shape'] = results_df['train_shape'].astype(object)
results_df['test_shape'] = results_df['test_shape'].astype(object)
results_df['train_lable_count'] = results_df['train_lable_count'].astype(object)
results_df['test_label_count'] = results_df['test_label_count'].astype(object)
idx = 0

patients = ['G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G11',
            'P231', 'P351', 'P379', 'P551', 'P623', 'P645', 'P812', 'P876', 'P940']

for p in patients:
    #clear_session()  # for clearing the Model cache to avoid consuming memory over time
    PATIENT = p
    print("Testing Patient >>>>  ", PATIENT)
    train_set, test_set = preprocessing.leave_one_patient_out(full_rolled_df, test_patient=PATIENT)

    print("training set label count: \n", train_set['Label'].value_counts())
    print("testing set label count: \n", test_set['Label'].value_counts())

    # drop column patient or trials if they exist
    cols_to_drop = ['patient', 'trials']
    if set(cols_to_drop).issubset(train_set.columns):
        train_set.drop(cols_to_drop, axis=1, inplace=True)
    if set(cols_to_drop).issubset(test_set.columns):
        test_set.drop(cols_to_drop, axis=1, inplace=True)

    X_train, y_train = train_set.drop('Label', axis=1), train_set['Label']
    X_test, y_test = test_set.drop('Label', axis=1), test_set['Label']

    results_df.at[idx, 'test_patient'] = PATIENT
    results_df.at[idx, 'train_shape'] = train_set.shape
    results_df.at[idx, 'test_shape'] = test_set.shape
    results_df.at[idx, 'train_lable_count'] = train_set['Label'].value_counts().to_dict()
    results_df.at[idx, 'test_label_count'] = test_set['Label'].value_counts().to_dict()

    # ================ LSTM Modeling ===============
    TIME_STEPS = 3
    STEP = 1
    NUM_HIDDEN_LAYERS = 3
    BATCH_SIZE = 64
    EPOCHS = 50
    X_train_3d, y_train_3d = modelling.create_3d_dataset(X_train, y_train, time_steps=TIME_STEPS, step=STEP)
    X_test_3d, y_test_3d = modelling.create_3d_dataset(X_test, y_test, time_steps=TIME_STEPS, step=STEP)
    input_dim = (X_train_3d.shape[1], X_train_3d.shape[2])
    model = modelling.call_lstm_model(input_dim, NUM_HIDDEN_LAYERS)
    X_train_scaled, X_test_scaled = model.features_scaling_3d(X_train_3d, X_test_3d)
    # y_train_3d, y_test_3d = model.one_hot_labels(y_train_3d, y_test_3d)
    model.fit(X_train_scaled, y_train_3d, X_test_scaled, y_test_3d, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    results = model.evaluate(X_test_scaled, y_test_3d, batch_size=BATCH_SIZE)
    y_pred = model.predict(X_test_scaled)
    results_df.at[idx, 'conf_matrix'] = model.conf_matrix(y_test_3d)
    results_df.at[idx, 'clf_report'] = model.clf_report(y_test_3d)
    tpr, fpr, auc = model.roc_auc(y_test_3d)
    # plotting.plot_metrics(history=model.history, image_name='LOOCV_metrics_'+PATIENT)
    # plotting.plot_roc_curve(roc_auc=auc, fpr=fpr, tpr=tpr, pic_name='LOOCV_ROC_'+PATIENT)
    for key, value in results.items():
        results_df.at[idx, key] = value
    print("======== Patient " + PATIENT + " ended at :  " + str(datetime.datetime.now()) + " ============")
    idx += 1

results_df.to_csv('results/loocv_results_v4.csv', sep=',', index=False)

