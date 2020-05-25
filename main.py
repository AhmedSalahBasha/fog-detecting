import cleaning_main
import preprocessing
import modelling
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
from keras.utils import to_categorical
import os
import experiments as ex

from plotting import plot_loss_accuracy, plot_conf_matrix, plot_pre_rec, plot_pre_rec_thresh, plot_roc_curve, plot_roc

'''
print("Pre-processing start:  ", datetime.datetime.now())
# Cleaning & Preparing
full_dataset = cleaning_main.get_full_dataset()
print(full_dataset.shape)
full_dataset_rolled = preprocessing.rolling_window(full_dataset, win_size=200, step_size=40)  # window 1 sec and 0.2 sec overlapping
full_dataset_rolled.to_csv('processed_data/full_dataset_rolled.csv', sep=',', index=False)
'''
full_dataset = pd.read_csv('data/full_dataset.csv', sep=',')
models = ['LSTM']
for m in models:
    print("Modelling model " + m + " start:  ", datetime.datetime.now())
    full_dataset_rolled = pd.read_csv('processed_data/full_dataset_rolled.csv', sep=',')
    full_dataset_rolled = ex.sensors_features(full_dataset_rolled, sensor='upper', group='spec')
    X_train, X_test, y_train, y_test = cleaning_main.split_train_test_sets(full_dataset_rolled)

    if m == 'SVM':
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_svm_model(X_train, y_train)
        # modelling.grid_search_svm(classifier, X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'RF':
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_rf_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'KNN':
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_knn_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'DT':
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_dt_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'KNN_DTW':
        # X_train, X_test, y_train, y_test = cleaning_main.split_train_test_sets(full_dataset)
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_knn_dtw_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'ANN':
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_ann_model(input_dim=X_train.shape[1], num_hidden_layers=10)
        history = modelling.fit_ann_model(classifier, X_train, y_train, X_test, y_test, epochs=50, batch_size=5)
        train_accuracy = modelling.evaluate_model(classifier, X_train, y_train)
        test_accuracy = modelling.evaluate_model(classifier, X_test, y_test)
        print(m + ' Training Accuracy: %.3f, Testing Accuracy: %.3f' % (train_accuracy, test_accuracy))
        plot_loss_accuracy(history, pic_name='loss_acc_' + m)
    elif m == 'LSTM':
        TIME_STEPS = 1
        STEP = 1
        X, y = modelling.create_lstm_dataset(full_dataset_rolled.loc[:, full_dataset_rolled.columns != 'Label'],
                                             full_dataset_rolled['Label'],
                                             time_steps=TIME_STEPS,
                                             step=STEP)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=False)
        X_train, X_test = modelling.apply_lstm_feature_scaling(X_train, X_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        input_shape = [X_train.shape[1], X_train.shape[2]]
        classifier = modelling.build_lstm_model(input_shape, optimizer='adam', num_hidden_layers=2)
        history = modelling.fit_lstm_model(classifier, X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
        train_accuracy = modelling.evaluate_model(classifier, X_train, y_train)
        test_accuracy = modelling.evaluate_model(classifier, X_test, y_test)
        print(m + ' Training Accuracy: %.3f, Testing Accuracy: %.3f' % (train_accuracy, test_accuracy))
        plot_loss_accuracy(history, pic_name='loss_acc_' + m)
print("Modelling end:  ", datetime.datetime.now())

