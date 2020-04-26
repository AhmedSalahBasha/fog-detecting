import cleaning_main
import preprocessing
import modelling

from plotting import plot_loss_accuracy, plot_confusion_matrix, plot_clf_report


train_test_files = [['data/P812_M050_2_B_FoG_trial_1_out_left_foot.csv',
                     'data/P812_M050_2_B_FoG_trial_1_out_right_foot.csv',
                     'data/P812_M050_2_B_FoG_trial_1_out_lower_left_foot.csv',
                     'data/P812_M050_2_B_FoG_trial_1_out_lower_right_foot.csv'],
                    ['data/P812_M050_2_B_FoG_trial_2_out_left_foot.csv',
                     'data/P812_M050_2_B_FoG_trial_2_out_right_foot.csv',
                     'data/P812_M050_2_B_FoG_trial_2_out_lower_left_foot.csv',
                     'data/P812_M050_2_B_FoG_trial_2_out_lower_right_foot.csv']]
# Cleaning
train_test_dfs = cleaning_main.get_train_test_dfs(train_test_files)

# Pre-processing
pre_train_df = preprocessing.get_train_df(train_test_dfs)
pre_test_df = preprocessing.get_test_df(train_test_dfs)
cols_names = preprocessing.get_cols_names_list(pre_train_df)

train_df = preprocessing.rolling_window(pre_train_df, cols_names)
test_df = preprocessing.rolling_window(pre_test_df, cols_names)

# Modelling
X_train, y_train, X_test, y_test = modelling.get_train_test_sets(train_df, test_df)
X_train, X_test = modelling.apply_feature_scaling(X_train, X_test)
classifier = modelling.build_ann_model(X_train.shape[1], num_hidden_layers=2)
history = modelling.fit_ann_model(classifier, X_train, y_train, epochs=2)

# Plotting Model Output
conf_matrix = modelling.build_conf_matrix(classifier, X_test, y_test)
plot_confusion_matrix(conf_matrix, [0, 1])

clf_report = modelling.build_clf_report(classifier, X_test, y_test)
plot_clf_report(clf_report)
plot_loss_accuracy(history)

