import preprocessing
import modelling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from keras.utils import to_categorical
import features_selection as fs
from keras.optimizers import SGD
from keras.optimizers import Adam

from plotting import plot_loss_accuracy, plot_loss_f1


# PRE-PROCESSING:: ATTENTION OF ACTIVATING THE FOLLOWING LINE - IT TAKES AROUND 12 HOURS
# preprocessing.create_rolled_train_dev_test_dataframes(win_size=200, step_size=40)

# Read processed data
train_set = pd.read_csv('processed_data/train_set.csv', sep=',')
dev_set = pd.read_csv('processed_data/dev_set.csv', sep=',')
test_set = pd.read_csv('processed_data/test_set.csv', sep=',')

'''
# Print Label Value Counts
print('Training Label Value Counts: ', train_set['Label'].value_counts())
print('Dev Label Value Counts: ', dev_set['Label'].value_counts())
print('Teat Label Value Counts: ', test_set['Label'].value_counts())
'''

# Modelling
models = ['LSTM']
for m in models:
    print("Modelling model " + m + " start:  ", datetime.datetime.now())

    # Feature Selection :: group --> (stat, spec, temp, freq, all)   # pos --> (upper, lower)   # sensor --> (acc, gyro, both)
    train_df = fs.sensors_features(train_set, pos='upper', group='stat', sensor='acc')
    dev_df = fs.sensors_features(dev_set, pos='upper', group='stat', sensor='acc')
    test_df = fs.sensors_features(test_set, pos='upper', group='stat', sensor='acc')

    # Drop Unimportant Features -- Make sure to drop the same columns list from all dataframes
    drop_spec_features = ['_max_pow_spec', '_max_freq', '_spec_entropy']
    drop_stat_features = ['_iqr_rng', '_var', '_med', '_rms']
    train_df = fs.drop_features(train_df, drop_stat_features)
    dev_df = fs.drop_features(dev_df, drop_stat_features)
    test_df = fs.drop_features(test_df, drop_stat_features)

    # splitting to X and y for each train, dev and test dataframes
    X_train, y_train, X_dev, y_dev, X_test, y_test = preprocessing.split_train_dev_test_sets(train_df,
                                                                                             dev_df,
                                                                                             test_df)

    # Calling models for training, testing and evaluation
    if m == 'SVM':
        X_train, X_dev = modelling.apply_feature_scaling(X_train, X_dev)
        classifier = modelling.build_fit_svm_model(X_train, y_train)
        # modelling.grid_search_svm(classifier, X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_dev, y_dev)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_dev, y_dev, m)
    elif m == 'RF':
        scaled_X_train, scaled_X_dev = modelling.apply_feature_scaling(X_train, X_dev)
        classifier = modelling.build_fit_rf_model(scaled_X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, scaled_X_dev, y_dev)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, scaled_X_dev, y_dev, m)
        feature_importances = pd.DataFrame(classifier.feature_importances_,
                                           index=X_train.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        print(feature_importances)
    elif m == 'KNN':
        X_train, X_dev = modelling.apply_feature_scaling(X_train, X_dev)
        classifier = modelling.build_fit_knn_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_dev, y_dev)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_dev, y_dev, m)
    elif m == 'DT':
        scaled_X_train, scaled_X_dev = modelling.apply_feature_scaling(X_train, X_dev)
        classifier = modelling.build_fit_dt_model(scaled_X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, scaled_X_dev, y_dev)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, scaled_X_dev, y_dev, m)
        feature_importances = pd.DataFrame(classifier.feature_importances_,
                                           index=X_train.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        print(feature_importances)
    elif m == 'KNN_DTW':
        # X_train, X_dev, y_train, y_dev = cleaning_main.split_train_test_sets(full_dataset)
        X_train, X_dev = modelling.apply_feature_scaling(X_train, X_dev)
        classifier = modelling.build_fit_knn_dtw_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_dev, y_dev)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_dev, y_dev, m)
    elif m == 'ANN':
        X_train, X_dev = modelling.apply_feature_scaling(X_train, X_dev)
        classifier = modelling.build_ann_model(input_dim=X_train.shape[1], num_hidden_layers=4)
        history = modelling.fit_ann_model(classifier, X_train, y_train, X_dev, y_dev, epochs=50, batch_size=8)
        train_accuracy = modelling.evaluate_model(classifier, X_train, y_train)
        test_accuracy = modelling.evaluate_model(classifier, X_dev, y_dev)
        print(m + ' Training Accuracy: %.3f, Testing Accuracy: %.3f' % (train_accuracy, test_accuracy))
        plot_loss_accuracy(history, pic_name='loss_acc_' + m)
    elif m == 'LSTM':
        TIME_STEPS, STEP, BATCH_SIZE = 400, 80, 64
        NUM_HIDDEN_LAYERS, EPOCHS = 2, 300

        full_dataset = pd.read_csv('data/full_dataset.csv', sep=',')
        full_dataset = full_dataset.dropna()
        # full_dataset = pd.concat([train_df, dev_df], ignore_index=True)

        # Feature Selection :: group --> (stat, spec, temp, freq, all)   # pos --> (upper, lower)   # sensor --> (acc, gyro, both)
        # full_dataset = fs.sensors_features(full_dataset, pos='upper', group='all', sensor='acc')
        print("LSTM Input Dataset Shape:  ", full_dataset.shape)

        X, y = modelling.create_lstm_dataset(full_dataset.loc[:, full_dataset.columns != 'Label'],
                                             full_dataset['Label'],
                                             time_steps=TIME_STEPS,
                                             step=STEP)
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3, shuffle=False)
        # X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=0.4, shuffle=False)
        print("X_train shape: ", X_train.shape)
        print("X_dev shape: ", X_dev.shape)
        print("y_train shape: ", y_train.shape)
        print("y_dev shape: ", y_dev.shape)
        # Print Label Value Counts
        print('LSTM: Training Label Value Counts: ', preprocessing.get_value_counts_of_array(y_train))
        print('LSTM: Dev Label Value Counts: ', preprocessing.get_value_counts_of_array(y_dev))
        print('LSTM: Test Label Value Counts: ', preprocessing.get_value_counts_of_array(y_test))

        # Apply feature scaling
        X_train, X_dev = modelling.apply_lstm_feature_scaling(X_train, X_dev)
        y_train = to_categorical(y_train, num_classes=2)
        y_dev = to_categorical(y_dev, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
        input_shape = (X_train.shape[1], X_train.shape[2])
        opt = Adam(learning_rate=0.001)
        classifier = modelling.build_lstm_model(input_shape,
                                                optimizer=opt,
                                                num_hidden_layers=NUM_HIDDEN_LAYERS,
                                                output_layer_actv='softmax',
                                                loss_measure='categorical_crossentropy')
        history = modelling.fit_lstm_model(classifier, X_train, y_train, X_dev, y_dev, epochs=EPOCHS, batch_size=BATCH_SIZE)
        train_accuracy = modelling.evaluate_model(classifier, X_train, y_train)
        test_accuracy = modelling.evaluate_model(classifier, X_dev, y_dev)
        print(m + ' Training Accuracy: %.3f, Testing Accuracy: %.3f' % (train_accuracy, test_accuracy))
        plot_loss_f1(history, pic_name='loss_f1_' + m)
print("Modelling end:  ", datetime.datetime.now())

