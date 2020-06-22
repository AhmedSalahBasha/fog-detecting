from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from keras.optimizers import SGD
from keras.metrics import AUC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import numpy as np
from scipy import stats

from dtaidistance import dtw

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
import keras_metrics as km
from keras import backend as K
from keras.metrics import Precision, Recall

from ml_models import SVM_Model, RF_Model, DT_Model, KNN_Model, KNN_DTW_Model
from dl_models import ANN_Model, LSTM_Model


def call_svm_model():
    model = SVM_Model(gamma=0.001, C=50, kernel='sigmoid')
    return model


def call_rf_model():
    model = RF_Model(n_estimators=400, max_depth=20, criterion='entropy', min_samples_split=4)
    return model


def call_dt_model():
    model = DT_Model(max_depth=10, criterion='gini', min_samples_split=2)
    return model

def call_knn_model():
    model = KNN_Model(n_neighbors=10, weights='distance', metric='minkowski')
    return model

def call_knn_dtw_model():
    KNN_DTW_Model(n_neighbors=10, weights='distance')

def call_ann_model(input_dim, num_hidden_layers):
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
    model = ANN_Model(input_dim=input_dim,
                      num_hidden_layers=num_hidden_layers,
                      hidden_layer_actv='relu',
                      output_layer_actv='sigmoid',
                      optimizer='adam',
                      dropout_rate=0.4,
                      metric=METRICS)
    return model

def call_lstm_model(input_dim):
    model = LSTM_Model(input_dim=input_dim,
                       num_hidden_layers=4,
                       hidden_layer_actv='relu',
                       output_layer_actv='softmax',
                       optimizer='adam',
                       dropout_rate=0.3,
                       metric='accuracy')
    return model


def create_3d_dataset(X, y, time_steps=1, step=1):
    """
    Creating 3-nd dataset as a standard input shape for RNN (LSTM) Model
    :param X: Features dataframe
    :param y: Label series
    :param time_steps: time steps to look back
    :param step: number of steps to reduce overlapping window
    :return: X as 3-nd numpy array and y
    """
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

def get_train_test_sets(train_df, test_df):
    # Preparing Training set and Test set from different datasets
    X_train = train_df.drop('Label', axis=1).values
    y_train = train_df['Label'].values
    X_test = test_df.drop('Label', axis=1).values
    y_test = test_df['Label']
    return X_train, y_train, X_test, y_test


def grid_search_ann_model(X_train, y_train):
    np.random.seed(123)

    def build_classifier(optimizer):
        clf = Sequential()
        output_dim = 210
        clf.add(Dense(units=output_dim, init='uniform', activation='relu', input_dim=434))
        clf.add(Dropout(rate=0.3))
        clf.add(Dense(units=output_dim, init='uniform', activation='relu'))
        clf.add(Dropout(rate=0.3))
        clf.add(Dense(units=output_dim, init='uniform', activation='relu'))
        clf.add(Dropout(rate=0.3))
        clf.add(Dense(units=output_dim, init='uniform', activation='relu'))
        clf.add(Dropout(rate=0.3))
        clf.add(Dense(units=1, init='uniform', activation='sigmoid'))
        clf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[f1])
        return clf

    classifier = KerasClassifier(build_fn=build_classifier)
    parameters = {'batch_size': [5, 10, 15],
                  'epochs': [10, 15, 20],
                  'optimizer': ['adam', 'Adadelta', 'Adamax']}
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=5)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print("Best Parameters: ", best_parameters)
    print("Best Accuracy: ", best_accuracy)


def grid_search_rf(classifier, X_train, y_train):
    grid_param = {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10],
        'min_samples_split': [2, 6],
        'min_samples_leaf': [1, 3, 5],
        'max_features': [None, "auto", "sqrt", "log2"]
    }
    gd_sr = GridSearchCV(estimator=classifier,
                         param_grid=grid_param,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(X_train, y_train)
    best_parameters = gd_sr.best_params_
    best_result = gd_sr.best_score_
    features_importance = gd_sr.best_estimator_.feature_importance()
    print("best_parameters: ", best_parameters)
    print("Best score: ", best_result)
    print("Features Importance:  ", features_importance)


def grid_search_svm(classifier, X_train, y_train):
    """
    function for tuning the SVM classifier
    :param X_train: X training dataframe
    :param y_train: y training dataframe
    :return: the best parameters for the classifier
    """
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'],
                         'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
                         },
                        {'kernel': ['sigmoid'],
                         'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
                         },
                        {'kernel': ['linear'],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
                         }
                        ]

    gd_sr = GridSearchCV(estimator=classifier,
                         param_grid=tuned_parameters,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(X_train, y_train)
    best_parameters = gd_sr.best_params_
    best_result = gd_sr.best_score_
    print("best_parameters: ", best_parameters)
    print("Best score: ", best_result)



def get_roc_curve(clf, X_test, y_test):
    # predict probabilities
    probs = clf.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    probs = probs[:, 0]
    # calculate roc curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
    roc_auc = metrics.roc_auc_score(y_test, probs)
    return roc_auc, fpr, tpr


def get_optimal_f1(clf, X_test, y_test):
    """
    fucntion to get the optimal f1 score
    :param clf: the classifier
    :param X_test: X test dataframe
    :param y_test: y test series
    :return: the best threshold and f1 score
    """
    best_thr = None
    best_f1 = None

    # predict probabilities
    proba_pred = clf.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    proba_pred = proba_pred[:, 0]

    for prob_thr in np.arange(0.001, 1.0, 0.001):
        labels_quantized = [0 if x <= prob_thr else 1 for x in proba_pred]
        f1 = metrics.f1_score(y_test, labels_quantized)
        prec = metrics.precision_score(y_test, labels_quantized)
        rec = metrics.recall_score(y_test, labels_quantized)
        # print("thr. %f -> f1 %f, precision %f, recall %f" % (prob_thr, f1, prec, rec))
        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_thr = prob_thr
    print("Best threshold %f || Best F1 Score %f" % (best_thr, best_f1))



def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

