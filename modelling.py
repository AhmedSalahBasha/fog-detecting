from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def get_train_test_sets(train_df, test_df):
    # Preparing Training set and Test set from different datasets
    X_train = train_df.drop('Label', axis=1).values
    y_train = train_df['Label'].values
    X_test = test_df.drop('Label', axis=1).values
    y_test = test_df['Label']
    return X_train, y_train, X_test, y_test


def apply_feature_scaling(X_train, X_test):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


def build_ann_model(input_dim, num_hidden_layers=4, hidden_layer_actv='relu', output_layer_actv='sigmoid', optimizer='adam'):
    # Building the ANN Model
    clf = Sequential()

    # Adding the input layer and first hidden layer
    output_dim = int(input_dim / 2)
    clf.add(Dense(output_dim=output_dim, init='uniform', activation=hidden_layer_actv, input_dim=input_dim))
    for i in range(num_hidden_layers):
        # Adding hidden layer
        clf.add(Dense(output_dim=output_dim, init='uniform', activation=hidden_layer_actv))

    # Adding the output layer
    clf.add(Dense(output_dim=1, init='uniform', activation=output_layer_actv))

    # Compiling ANN
    clf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return clf


def fit_ann_model(clf, X_train, y_train, epochs=5):
    # Fitting the classifier
    history = clf.fit(X_train, y_train, batch_size=1, epochs=epochs, verbose=0)
    return history


def build_clf_report(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred = (y_pred > 0.5)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    return clf_report


def build_conf_matrix(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    return cm

