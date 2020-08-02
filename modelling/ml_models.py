import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dtaidistance import dtw


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
        return acc

    def f1_score(self, y_test):
        f1 = metrics.f1_score(y_test, self.y_pred)
        print('F1 Score: ', f1)
        return f1

    def auc_score(self, y_test):
        auc = 0
        try:
            auc = metrics.roc_auc_score(y_test, self.y_pred)
        except ValueError:
            pass
        print('AUC Score: ', auc)
        return auc

    def clf_report(self, y_test):
        y_pred = (self.y_pred > 0.5)
        # y_pred = np.argmax(y_pred, axis=1)
        report = None
        try:
            report = metrics.classification_report(y_test, y_pred, output_dict=True)
        except ValueError:
            pass
        print('Classification Report: \n', report)
        return report

    def conf_matrix(self, y_test):
        y_pred = (self.y_pred > 0.5)
        # y_pred = np.argmax(y_pred, axis=1)
        matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
        print('Confusion Matrix: \n', matrix)
        return matrix

    def features_scaling(self, X_train, X_test):
        sc = StandardScaler()
        scaled_X_train = sc.fit_transform(X_train)
        scaled_X_test = sc.transform(X_test)
        return scaled_X_train, scaled_X_test


class GNB_Model(ML_Parent_Model):
    def __init__(self):
        self.model_name = 'GaussianNaiveBayes'
        self.model = GaussianNB()
        ML_Parent_Model.__init__(self, model=self.model)


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

    def features_importances(self, index):
        importances_df = pd.DataFrame(self.model.feature_importances_,
                                      index=index,
                                      columns=['importance']).sort_values('importance', ascending=False)
        return importances_df


class KNN_Model(ML_Parent_Model):
    def __init__(self, n_neighbors, weights, metric):
        self.model_name = 'K-NearestNeighbors'
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                          weights=weights,
                                          metric=metric)
        ML_Parent_Model.__init__(self, model=self.model)


class KNN_DTW_Model(ML_Parent_Model):
    def __init__(self, n_neighbors, weights):
        self.model_name = 'K-NearestNeighbors_DynamicTimeWrapping'
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = dtw.distance_fast     # calculating distance using Dynamic Time Wrapping Algorithm
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                          weights=weights,
                                          metric=self.metric)
        ML_Parent_Model.__init__(self, model=self.model)


class DT_Model(ML_Parent_Model):
    def __init__(self, max_depth, criterion, min_samples_split):
        self.model_name = 'DecisionTree'
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.model = DecisionTreeClassifier(max_depth=max_depth,
                                            random_state=0,
                                            criterion=criterion,
                                            min_samples_split=min_samples_split)
        ML_Parent_Model.__init__(self, model=self.model)


