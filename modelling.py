import pandas as pd
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# reading data
df = pd.read_csv('processed_data/P812_M050_B_FoG_trial_1_train.csv', sep=',')
df_test = pd.read_csv('processed_data/P812_M050_B_FoG_trial_2_test.csv', sep=',')

# Preparing Training set and Test set from different datasets
X_train = df.drop('target', axis=1).values
y_train = df['target'].values

X_test = df_test.drop('target', axis=1).values
y_test = df_test['target']

print("Training-Set Shape: ", X_train.shape)
print("Test-Set Shape: ", X_test.shape)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Building the ANN Model
def build_model(input_dim, num_hidden_layers=4, hidden_layer_actv='relu', output_layer_actv='sigmoid', optimizer='adam'):
    # Intializing ANN
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


classifier = build_model(input_dim=168, num_hidden_layers=5)

# Fitting the classifier
classifier.fit(X_train, y_train, batch_size=2, epochs=10)

# Part 3 - making predictions
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# creating the confusion matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

