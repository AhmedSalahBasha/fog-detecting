from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import sklearn.metrics as metrics
from scipy import stats
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


class DL_Parent_Model:
    def __init__(self, model=None):
        self.model = model

    def fit(self, X_train, y_train, X_test, y_test, epochs, batch_size, verbose):
        # callback = EarlyStopping(monitor='val_f1_score', patience=10)
        self.history = self.model.fit(X_train, y_train,
                                      validation_data=(X_test, y_test),
                                      epochs=epochs, batch_size=batch_size,
                                      verbose=verbose, shuffle=False)

    def evaluate(self, X, y, batch_size):
        results = self.model.evaluate(X, y, batch_size=batch_size, verbose=0)
        metrics_results = dict(zip(self.model.metrics_names, results))
        print(metrics_results)
        return metrics_results

    def predict(self, X_test):
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def one_hot_labels(self, y_train, y_test):
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
        return y_train, y_test

    def conf_matrix(self, y_test):
        self.cm = metrics.confusion_matrix(y_test.argmax(axis=1), self.y_pred.argmax(axis=1), labels=[0, 1])
        print(self.cm)
        return self.cm

    def clf_report(self, y_test):
        report = None
        try:
            report = metrics.classification_report(y_test.argmax(axis=1), self.y_pred.argmax(axis=1), output_dict=True)
        except ValueError:
            print('ERROR:: Classification Report Error ???')
            pass
        print('Classification Report: \n', report)
        return report


class ANN_Model(DL_Parent_Model):
    def __init__(self, input_dim, num_hidden_layers, hidden_layer_actv, output_layer_actv, optimizer, dropout_rate, metric):
        self.model_name = 'ANN'
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_actv = hidden_layer_actv
        self.output_layer_actv = output_layer_actv
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.metric = metric
        self.model = self.__initialize_ann_model()
        DL_Parent_Model.__init__(self, model=self.model)

    def __initialize_ann_model(self):
        clf = Sequential()
        units = int(self.input_dim)
        clf.add(Dense(units=units, activation=self.hidden_layer_actv, input_dim=self.input_dim))
        clf.add(Dropout(rate=self.dropout_rate))
        for i in range(self.num_hidden_layers):
            clf.add(Dense(units=units, activation=self.hidden_layer_actv))
            clf.add(Dropout(rate=self.dropout_rate))
        clf.add(Dense(units=2, activation=self.output_layer_actv))
        clf.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=[self.metric])
        return clf

    def features_scaling(self, X_train, X_test, min_max:bool=False):
        if min_max:
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)
        return scaled_X_train, scaled_X_test


class LSTM_Model(DL_Parent_Model):
    def __init__(self, input_dim, num_hidden_layers, hidden_layer_actv, output_layer_actv, optimizer, dropout_rate, metric):
        self.model_name = 'LSTM'
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_actv = hidden_layer_actv
        self.output_layer_actv = output_layer_actv
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.metric = metric
        self.model = self.__initialize_lstm_model()
        DL_Parent_Model.__init__(self, model=self.model)

    def __initialize_lstm_model(self):
        clf = Sequential()
        units = int(self.input_dim[1])
        clf.add(LSTM(units=units, input_shape=self.input_dim, return_sequences=True))
        clf.add(Dropout(rate=self.dropout_rate))
        for i in range(self.num_hidden_layers):
            clf.add(LSTM(units=units, return_sequences=True))
            clf.add(Dropout(rate=self.dropout_rate))
        clf.add(LSTM(units=int(units)))
        clf.add(Dropout(rate=self.dropout_rate))
        clf.add(Dense(units=units, activation=self.hidden_layer_actv, kernel_initializer='uniform'))
        clf.add(Dropout(rate=self.dropout_rate))
        clf.add(Dense(units=1, activation=self.output_layer_actv, kernel_initializer='uniform'))
        clf.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=[self.metric])
        return clf

    def features_scaling_3d(self, X_train, X_test, min_max:bool=False):
        if min_max:
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        scaled_X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        return scaled_X_train, scaled_X_test

    def roc_auc(self, y_test):
        fpr, tpr, thresholds = metrics.roc_curve(y_test.argmax(axis=1), self.y_pred.argmax(axis=1).ravel())
        auc = metrics.auc(fpr, tpr)
        return fpr, tpr, auc

