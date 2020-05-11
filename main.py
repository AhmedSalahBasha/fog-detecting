import cleaning_main
import preprocessing
import modelling
import pandas as pd
import datetime
from keras.utils import to_categorical
import os

from plotting import plot_loss_accuracy, plot_conf_matrix, plot_pre_rec, plot_pre_rec_thresh, plot_roc_curve, plot_roc


print("Pre-processing start:  ", datetime.datetime.now())

train_test_files = [['sample_data/P812_M050_2_B_FoG_trial_1_out_left_foot.csv',
                     'sample_data/P812_M050_2_B_FoG_trial_1_out_right_foot.csv',
                     'sample_data/P812_M050_2_B_FoG_trial_1_out_lower_left_foot.csv',
                     'sample_data/P812_M050_2_B_FoG_trial_1_out_lower_right_foot.csv'],
                    ['sample_data/P812_M050_2_B_FoG_trial_2_out_left_foot.csv',
                     'sample_data/P812_M050_2_B_FoG_trial_2_out_right_foot.csv',
                     'sample_data/P812_M050_2_B_FoG_trial_2_out_lower_left_foot.csv',
                     'sample_data/P812_M050_2_B_FoG_trial_2_out_lower_right_foot.csv']]
# Cleaning
train_test_dfs = cleaning_main.get_train_test_dfs(train_test_files)

# Pre-processing
pre_train_df = preprocessing.get_train_df(train_test_dfs)
pre_test_df = preprocessing.get_test_df(train_test_dfs)
cols_names = preprocessing.get_cols_names_list(pre_train_df)

'''
train_df = preprocessing.rolling_window(pre_train_df, cols_names, win_size=400)
test_df = preprocessing.rolling_window(pre_test_df, cols_names, win_size=400)
train_df.to_csv('processed_data/train_new.csv', sep=',', index=False)
test_df.to_csv('processed_data/test_new.csv', sep=',', index=False)
'''
print("Pre-processing end:  ", datetime.datetime.now())

models = ['KNN_DTW', 'SVM', 'RF', 'KNN', 'DT']

for m in models:

    # Modelling
    print("Modelling model " + m + " start:  ", datetime.datetime.now())
    train_df = pd.read_csv('processed_data/train_new.csv', sep=',')
    test_df = pd.read_csv('processed_data/test_new.csv', sep=',')
    # X_train, y_train, X_test, y_test = modelling.get_train_test_sets(train_df, test_df)
    # classifier, history = None, None

    if m == 'SVM':
        X_train, y_train, X_test, y_test = modelling.get_train_test_sets(train_df, test_df)
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_svm_model(X_train, y_train)
        # modelling.grid_search_svm(classifier, X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'RF':
        X_train, y_train, X_test, y_test = modelling.get_train_test_sets(train_df, test_df)
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_rf_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'KNN':
        X_train, y_train, X_test, y_test = modelling.get_train_test_sets(train_df, test_df)
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_knn_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'KNN_DTW':
        X_train, y_train, X_test, y_test = modelling.get_train_test_sets(pre_train_df, pre_test_df)
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_knn_dtw_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'DT':
        X_train, y_train, X_test, y_test = modelling.get_train_test_sets(train_df, test_df)
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_fit_dt_model(X_train, y_train)
        accuracy = modelling.get_model_accuracy(classifier, X_test, y_test)
        print(m + " Accuracy = ", accuracy)
        modelling.build_clf_report(classifier, X_test, y_test, m)
    elif m == 'ANN':
        X_train, y_train, X_test, y_test = modelling.get_train_test_sets(pre_train_df, pre_test_df)
        X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
        classifier = modelling.build_ann_model(input_dim=X_train.shape[1], num_hidden_layers=5)
        modelling.grid_search_ann_model(X_train, y_train)
        '''
        history = modelling.fit_ann_model(classifier, X_train, y_train, X_test, y_test, epochs=10)
        plot_loss_accuracy(history, pic_name='loss_acc_' + m)
        train_accuracy = modelling.evaluate_model(classifier, X_train, y_train)
        test_accuracy = modelling.evaluate_model(classifier, X_test, y_test)
        print(m + ' Training Accuracy: %.3f, Testing Accuracy: %.3f' % (train_accuracy, test_accuracy))
        '''
    elif m == 'LSTM':
        # LSTM WORKS ON RAW DATA -NOT- PRE-PROCESSED DATA!
        TIME_STEPS = 200
        STEP = 10
        X_train, y_train = modelling.create_lstm_dataset(pre_train_df.loc[:, pre_train_df.columns != 'Label'],
                                                         pre_train_df['Label'],
                                                         time_steps=TIME_STEPS,
                                                         step=STEP)
        X_test, y_test = modelling.create_lstm_dataset(pre_test_df.loc[:, pre_test_df.columns != 'Label'],
                                                       pre_test_df['Label'],
                                                       time_steps=TIME_STEPS,
                                                       step=STEP)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        input_shape = [X_train.shape[1], X_train.shape[2]]
        classifier = modelling.build_lstm_model(input_shape, num_hidden_layers=5)
        history = modelling.fit_lstm_model(classifier, X_train, y_train, X_test, y_test, epochs=10)
        plot_loss_accuracy(history, pic_name='loss_acc_' + m)
        train_accuracy = modelling.evaluate_model(classifier, X_train, y_train)
        test_accuracy = modelling.evaluate_model(classifier, X_test, y_test)
        print(m + ' Training Accuracy: %.3f, Testing Accuracy: %.3f' % (train_accuracy, test_accuracy))
print("Modelling end:  ", datetime.datetime.now())

