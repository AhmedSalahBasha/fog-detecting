from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
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
from keras.wrappers.scikit_learn import KerasClassifier
import keras_metrics as km
from keras import backend as K
from keras.metrics import Precision, Recall


def get_train_test_sets(train_df, test_df):
    # Preparing Training set and Test set from different datasets
    X_train = train_df.drop('Label', axis=1).values
    y_train = train_df['Label'].values
    X_test = test_df.drop('Label', axis=1).values
    y_test = test_df['Label']
    return X_train, y_train, X_test, y_test


def apply_feature_scaling(X_train, X_test):
    sc = StandardScaler()
    scaled_X_train = sc.fit_transform(X_train)
    scaled_X_test = sc.transform(X_test)
    return scaled_X_train, scaled_X_test


def apply_lstm_feature_scaling(X_train, X_test):
    # sc = StandardScaler()
    mms = MinMaxScaler(feature_range=(0, 1))
    scaled_X_train = mms.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    scaled_X_test = mms.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return scaled_X_train, scaled_X_test


def create_lstm_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)


# fit and evaluate a model
def build_lstm_model(input_shape, num_hidden_layers=1, hidden_layer_actv='relu', output_layer_actv='softmax', optimizer='adam'):
    clf = Sequential()
    units = int(input_shape[1] / 2)
    clf.add(LSTM(units=units, input_shape=input_shape, return_sequences=True))
    clf.add(Dropout(rate=0.2))
    clf.add(LSTM(units=units, return_sequences=True))
    clf.add(Dropout(rate=0.2))
    clf.add(LSTM(units=units))
    clf.add(Dropout(rate=0.2))
    for i in range(num_hidden_layers):
        clf.add(Dense(units=units, activation=hidden_layer_actv))
        clf.add(Dropout(rate=0.2))
    clf.add(Dense(units=2, activation=output_layer_actv))
    clf.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return clf


def fit_lstm_model(clf, X_train, y_train, X_test, y_test, epochs=1, batch_size=1):
    history = clf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
    return history


def build_ann_model(input_dim, num_hidden_layers=1, hidden_layer_actv='relu', output_layer_actv='sigmoid', optimizer='adam'):
    clf = Sequential()
    # Adding the input layer and first hidden layer
    units = int(input_dim / 2)
    clf.add(Dense(units=units, init='uniform', activation=hidden_layer_actv, input_dim=input_dim))
    clf.add(Dropout(rate=0.2))
    for i in range(num_hidden_layers):
        # Adding hidden layer
        clf.add(Dense(units=units, init='uniform', activation=hidden_layer_actv))
        clf.add(Dropout(rate=0.2))
    # Adding the output layer
    clf.add(Dense(units=1, init='uniform', activation=output_layer_actv))
    clf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return clf


def fit_ann_model(clf, X_train, y_train, X_test, y_test, epochs=5, batch_size=1):
    history = clf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
    return history


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
        clf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
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


def evaluate_model(clf, X, y):
    # evaluate model
    _, accuracy = clf.evaluate(X, y, verbose=0)
    return accuracy


def build_fit_svm_model(X_train, y_train):
    clf = svm.SVC(kernel='sigmoid', C=50, gamma=0.001, probability=True)
    clf.fit(X_train, y_train)
    return clf


def build_fit_rf_model(X_train, y_train, n_estimators=100, max_depth=1, criterion='gini', min_samples_split=2):
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 random_state=0,
                                 criterion=criterion,
                                 min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)
    return clf


def build_fit_knn_dtw_model(X_train, y_train):
    # dtw._print_library_missing()
    # dist = dtw.distance_fast(X_train, y_train)
    clf = KNeighborsClassifier(n_neighbors=2, metric=dtw.distance_fast)
    clf.fit(X_train, y_train)
    return clf


def build_fit_knn_model(X_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=2, weights='distance')
    clf.fit(X_train, y_train)
    return clf


def build_fit_dt_model(X_train, y_train):
    clf = DecisionTreeClassifier(criterion="gini", max_depth=5)
    clf.fit(X_train, y_train)
    return clf


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


def get_model_accuracy(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc


def build_clf_report(clf, X_test, y_test, model_name):
    y_pred = clf.predict(X_test)
    y_pred = (y_pred > 0.5)
    # y_pred = np.argmax(y_pred, axis=1)
    clf_report = metrics.classification_report(y_test, y_pred)
    print("Classification Report For Model " + model_name + " : \n", clf_report)


def build_conf_matrix(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    # y_pred = (y_pred > 0.5)
    y_pred = np.argmax(y_pred, axis=1)
    cm = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
    return cm


def get_yhat_probs_classes(clf, X_test):
    # predict probabilities for test set
    yhat_classes = clf.predict(X_test, verbose=0)
    yhat_classes = (yhat_classes > 0.5)
    # predict crisp classes for test set
    # yhat_classes = clf.predict_classes(X_test, verbose=0)
    # reduce to 1d array
    # yhat_probs = yhat_probs[:, 0]
    # yhat_classes = yhat_classes[:, 0]
    return yhat_classes


def get_acc_pre_rec_f1(y_test, yhat_classes):
    # accuracy: (tp + tn) / (p + n)
    accuracy = metrics.accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = metrics.precision_score(y_test, yhat_classes, pos_label=1, average='binary')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)


def get_y_pred(clf, X_test):
    y_pred = clf.predict(X_test)
    y_pred = (y_pred > 0.5)
    return y_pred


def get_pre_rec(clf, X_test, y_test):
    # predict probabilities
    probs = clf.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    probs = probs[:, 0]
    # calculate precision-recall curve
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, probs)
    return precision, recall, thresholds


def get_y_proba(clf, X_test):
    y_proba = clf.predict_proba(X_test)
    #y_proba = y_proba[:, 0]
    return y_proba


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


def roc_auc_score(clf, X_test, y_test):
    y_proba = get_y_proba(clf, X_test)
    # y_pred = (y_proba > 0.5)
    y_pred = np.argmax(y_proba, axis=1)
    auc_score = metrics.roc_auc_score(y_test, y_pred)
    print(auc_score)


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

