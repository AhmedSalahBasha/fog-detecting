from tensorflow import keras
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn import svm
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import backend as K

from modelling.ml_models import SVM_Model, RF_Model, DT_Model, KNN_Model, KNN_DTW_Model, GNB_Model
from modelling.dl_models import ANN_Model, LSTM_Model


def call_svm_model():
    """
    This function calls the SupportVectorMachines model and initialize it with fixed hyperparameters values
    as a results of the experiments.
    :return: it returns the model object
    """
    model = SVM_Model(gamma=0.001, C=50, kernel='sigmoid')
    return model


def call_rf_model():
    """
    This function calls the RandomForest model and initialize it with fixed hyperparameters values
    as a results of the experiments.
    :return: it returns the model object
    """
    model = RF_Model(n_estimators=10, max_depth=3, criterion='entropy')
    return model


def call_dt_model():
    """
    This function calls the DecisionTree model and initialize it with fixed hyperparameters values
    as a results of the experiments.
    :return: it returns the model object
    """
    model = DT_Model(max_depth=5, criterion='gini')
    return model


def call_gnb_model():
    """
    This function calls the GaussianNaiveBayes model
    :return: it returns the model object
    """
    model = GNB_Model()
    return model


def call_knn_model():
    """
    This function calls the K-NearestNeighbor model and initialize it with fixed hyperparameters values
    as a results of the experiments.
    :return: it returns the model object
    """
    model = KNN_Model(n_neighbors=1, weights='distance', metric='minkowski')
    return model


def call_knn_dtw_model():
    """
    This function calls the K-NearestNeighbor with DynamicTimeWrapping model and initialize it with fixed
    hyperparameters values as a results of the experiments.
    :return: it returns the model object
    """
    model = KNN_DTW_Model(n_neighbors=1, weights='distance')
    return model


def call_ann_model(input_dim, num_hidden_layers):
    """
    This function calls the ArtificialNeuralNetwork model and initialize it with fixed
    hyperparameters values as a results of the experiments.
    Note: This model hasn't been used in this research
    :return: it returns the model object
    """
    model = ANN_Model(input_dim=input_dim,
                      num_hidden_layers=num_hidden_layers,
                      hidden_layer_actv='relu',
                      output_layer_actv='softmax',
                      optimizer='adam',
                      dropout_rate=0.6,
                      metric=[precision, recall, f1_score])
    return model


def call_lstm_model(input_dim, num_hidden_layers):
    """
    This function calls the Long Short-Term Memory (LSTM) model and initialize it with fixed
    hyperparameters values as a results of the experiments.
    :return: it returns the model object
    """
    model = LSTM_Model(input_dim=input_dim,
                       num_hidden_layers=num_hidden_layers,
                       hidden_layer_actv='relu',
                       output_layer_actv='softmax',
                       optimizer='adam',
                       dropout_rate=0.6,
                       metric=['accuracy', precision, recall, f1_score])
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
    """
    This function gets the training and testing dataframes and returns numpy arrays for training and testing
    :param train_df: the training set
    :param test_df: the testing set
    :return: it returns the X_train, y_train, X_test and y_test
    """
    # Preparing Training set and Test set from different datasets
    X_train = train_df.drop('Label', axis=1).values
    y_train = train_df['Label'].values
    X_test = test_df.drop('Label', axis=1).values
    y_test = test_df['Label']
    return X_train, y_train, X_test, y_test


def weight_classes(y_train):
    """
    This function was a try for weightning the classes
    :param y_train:
    :return:
    """
    if y_train is None:
        return None
    else:
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        class_weights = dict(enumerate(class_weights))
        print("Class Weights: \n", class_weights)
        return class_weights


def grid_search_rf(classifier, X_train, y_train):
    """
    This class for The Grid-Search technique for the RandomForest model
    :param classifier: the Random Forest classifier object
    :param X_train: the train set without label
    :param y_train: the train set label
    :return: it doesn't return anything but prints the result of the GridSearch
    """
    grid_param = {
        'n_estimators': [20, 50, 100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3, 5],
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


def grid_search_svm(X_train, y_train):
    """
    function for tuning the SVM classifier by applying the grid-search technique
    :param X_train: X training dataframe
    :param y_train: y training dataframe
    :return: it prints the result of the grid search method
    """
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf', 'sigmoid', 'poly'],
                         'gamma': [0.0001, 0.001, 0.01, 0.1],
                         'C': [0.01, 0.1, 10, 100, 1000]
                         }
                        ]
    classifier = svm.SVC()
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
    """
    This function for getting the required metrics for plotting the ROC curve
    :param clf: the classifier object
    :param X_test: the test set
    :param y_test: the label of the test set
    :return: returns the ROC-AUC score, FalsePositiveRate and TruePositiveRate
    """
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


# ===== Keras - Custom Metrics ======
def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred


def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

