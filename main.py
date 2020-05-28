import cleaning_main
import cleaning
import preprocessing
import modelling
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
from keras.utils import to_categorical
import os
import experiments as ex

from plotting import plot_loss_accuracy, plot_conf_matrix, plot_pre_rec, plot_pre_rec_thresh, plot_roc_curve, plot_roc

# Pre-processing
full_dataset = pd.read_csv('data/full_dataset.csv', sep=',')
# ATTENTION OF ACTIVATING THE FOLLOWING LINE - IT TAKES AROUND 12 HOURS
# full_dataset_rolled = preprocessing.get_rolled_dataframe(win_size=200, step_size=40)
full_dataset_rolled = pd.read_csv('processed_data/new_full_dataset_rolled.csv', sep=',')

# Modelling
models = ['LSTM', 'SVM', 'ANN']
for m in models:
    print("Modelling model " + m + " start:  ", datetime.datetime.now())

    # group --> stat, spec, temp   # pos --> upper, lower   # sensor --> acc, gyro, both
    full_dataset_rolled = ex.sensors_features(full_dataset_rolled, pos='upper', group='spec', sensor='both')
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
        classifier = modelling.build_ann_model(input_dim=X_train.shape[1], num_hidden_layers=4)
        history = modelling.fit_ann_model(classifier, X_train, y_train, X_test, y_test, epochs=50, batch_size=8)
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

