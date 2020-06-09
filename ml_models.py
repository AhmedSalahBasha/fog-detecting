from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
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
from keras.layers import Bidirectional
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
import keras_metrics as km
from keras import backend as K
from keras.metrics import Precision, Recall

import preprocessing


class ML_Parent_Model:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X_test):
        self.y_pred = self.model.predict(X_test)

    def accuracy(self, y_test):
        acc = metrics.accuracy_score(y_test, self.y_pred)
        print('Accuracy Score: ', acc)

    def f1_score(self, y_test):
        f1 = metrics.f1_score(y_test, self.y_pred)
        print('F1 Score: ', f1)

    def auc_score(self, y_test):
        auc = metrics.roc_auc_score(y_test, self.y_pred)
        print('AUC Score: ', auc)

    def clf_report(self, y_test):
        y_pred = (self.y_pred > 0.5)
        # y_pred = np.argmax(y_pred, axis=1)
        print('Classification Report: \n', metrics.classification_report(y_test, y_pred))

    def conf_matrix(self, y_test):
        y_pred = (self.y_pred > 0.5)
        # y_pred = np.argmax(y_pred, axis=1)
        print('Confusion Matrix: \n', metrics.confusion_matrix(y_test, y_pred, labels=[0, 1]))


class SVM_Model(ML_Parent_Model):
    def __init__(self, gamma, C, kernel, probability=True):
        self.model_name = 'SupportVectorsMachine'
        self.gamma = gamma
        self.C = C
        self.kernel = kernel
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=probability)
        ML_Parent_Model.__init__(self, model=self.model)


class RF_Model(ML_Parent_Model):
    def __init__(self, n_estimators, max_depth, criterion, min_samples_split):
        self.model_name = 'RandomForest'
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 random_state=0,
                                 criterion=criterion,
                                 min_samples_split=min_samples_split)
        ML_Parent_Model.__init__(self, model=self.model)







train_set = pd.read_csv('processed_data/train_set.csv', sep=',')
dev_set = pd.read_csv('processed_data/dev_set.csv', sep=',')
test_set = pd.read_csv('processed_data/test_set.csv', sep=',')
X_train, y_train, X_dev, y_dev, X_test, y_test = preprocessing.split_train_dev_test_sets(train_set,
                                                                                             dev_set,
                                                                                             test_set)
rf = RF_Model(n_estimators=400,
              max_depth=10,
              criterion='gini',
              min_samples_split=4)
rf.fit(X_train, y_train)
rf.predict(X_dev)
rf.accuracy(y_dev)
rf.auc_score(y_dev)
rf.f1_score(y_dev)
rf.clf_report(y_dev)
rf.conf_matrix(y_dev)