import pandas as pd
import numpy as np
from preprocessing import preprocessing
from modelling import modelling
from preprocessing import features_selection as fs
import tensorflow as tf
tf.random.set_seed(123)
# tf.random.set_random_seed(123)
np.random.seed(123)


full_rolled_df = pd.read_csv('data/processed_data/full_rolled_dataset_winsize_900.csv', sep=',')

# Feature Selection
SENSOR_TYPE = 'gyro'
SENSOR_POS = 'feet'
FEATURES_GROUP = 'all'
LEG = 'both'
full_rolled_df = fs.sensors_features(full_rolled_df, pos=SENSOR_POS, group=FEATURES_GROUP, sensor=SENSOR_TYPE, leg=LEG)

feat_dfs = []
patients = ['G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G11',
            'P231', 'P351', 'P379', 'P551', 'P623', 'P645', 'P812', 'P876', 'P940']
idx = 0
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

    # ================ RandomForest Modeling ===============
    model = modelling.call_rf_model()
    X_train_scaled, X_test_scaled = model.features_scaling(X_train, X_test)
    model.fit(X_train_scaled, y_train)
    model.predict(X_test_scaled)
    importances_df = model.features_importances(index=X_train.columns)
    feat_dfs.append(importances_df.T)
    print('-------------------------')
    idx += 1

feat_imp_df = pd.concat(feat_dfs, ignore_index=True)
feat_imp_df.to_csv('results/features_importances.csv', sep=',', index=False)

