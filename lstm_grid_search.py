import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import preprocessing.features_selection as fs
from modelling import modelling


def features_scaling(X_train):
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    return scaled_X_train


def one_hot_labels(y_train):
    y_train = to_categorical(y_train, num_classes=2)
    return y_train


def build_classifier(optimizer='adam', dropout_rate='0.1', init='uniform', num_hidden_layers=2):
    clf = Sequential()
    units = int(input_dim[1])
    clf.add(LSTM(units=units, input_shape=input_dim, return_sequences=True))
    clf.add(Dropout(rate=dropout_rate))
    for i in range(num_hidden_layers):
        clf.add(LSTM(units=units, return_sequences=True))
        clf.add(Dropout(rate=dropout_rate))
    clf.add(LSTM(units=int(units)))
    clf.add(Dropout(rate=dropout_rate))
    clf.add(Dense(units=units, activation='relu', kernel_initializer=init))
    clf.add(Dropout(rate=dropout_rate))
    clf.add(Dense(units=2, activation='softmax', kernel_initializer=init))
    clf.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return clf


full_rolled_df = pd.read_csv('data/processed_data/full_rolled_dataset_w400_s40.csv')
# Drop Unimportant Features
drop_features_list = ['_spec_dist', '_hum_eng', '_max_pow_spec', '_max_freq', '_spec_entropy', '_pow_band', '_slope', '_max_peaks', '_total_eng', '_abs_eng', '_dist']
full_rolled_df = fs.drop_features(full_rolled_df, drop_features_list)

# Split dataset into train/test/dev based on patients
X = full_rolled_df[full_rolled_df['patient'] == 'P379']

# Feature Selection
SENSOR_TYPE = 'gyro'
SENSOR_POS = 'feet'
FEATURES_GROUP = 'freq'
LEG = 'both'
X = fs.sensors_features(X, pos=SENSOR_POS, group=FEATURES_GROUP, sensor=SENSOR_TYPE, leg=LEG)

# drop column patient or trials if they exist
cols_to_drop = ['patient', 'trials']
if set(cols_to_drop).issubset(X.columns):
    X.drop(cols_to_drop, axis=1, inplace=True)

# Train Test Dev - Split
X_train, y_train = X.drop('Label', axis=1), X['Label']

TIME_STEPS = 3
STEP = 1
X_train, y_train = modelling.create_3d_dataset(X_train, y_train, time_steps=TIME_STEPS, step=STEP)
input_dim = (X_train.shape[1], X_train.shape[2])

X_train = features_scaling(X_train)
# y_train = one_hot_labels(y_train)
print('y_train shape:  ', X_train.shape)

np.random.seed(123)

classifier = KerasClassifier(build_fn=build_classifier, verbose=2)
parameters = {'batch_size': [32, 64, 128],
              'epochs': [100, 200, 300],
              'optimizer': ['adam', 'SGD', 'Adamax'],
              'dropout_rate': [0.3, 0.5, 0.7],
              'num_hidden_layers': [2, 3, 4],
              'init': ['normal', 'uniform']}
grid = GridSearchCV(estimator=classifier,
                    param_grid=parameters,
                    n_jobs=-1,
                    scoring='accuracy',
                    cv=5)
grid_results = grid.fit(X_train, y_train)
# summarize results
gs_file = open("grid_search_results.txt", "a")
best_str = "Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_)
print(best_str)
gs_file.write(best_str + '\n')
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    res_str = "%f (%f) with: %r" % (mean, stdev, param)
    print(res_str)
    gs_file.write(res_str + '\n')
gs_file.close()
