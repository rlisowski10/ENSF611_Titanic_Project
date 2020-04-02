import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from keras.utils import np_utils
import time
import os
import matplotlib.pyplot as plt

# Read in titanic training data
titanic_df = pd.read_csv("_Data/train_with_ages.csv")
titanic_testing_df = pd.read_csv("_Data/test_with_ages.csv")
titanic_solutions_df = pd.read_csv("_Data/solution_set.csv")
titanic_submission_df = titanic_testing_df

# Prepare solutions
titanic_submission_df = titanic_submission_df.replace('"', '', regex=True)
titanic_solutions_df = titanic_solutions_df.replace('"', '', regex=True)
titanic_solutions_df = titanic_submission_df.merge(titanic_solutions_df, how='left', left_on=[
                                                   "Name", "Ticket"], right_on=["name", "ticket"])
titanic_solutions_df = titanic_solutions_df[['PassengerId', 'survived']]
# titanic_solutions_df.to_csv('../_Submission/010_temp_Submission_Solutions.csv', index=False)

# Set the fare to '0' for titanic_testing_df index '152'


def add_fare_values(fare):
    if fare >= 0:
        return fare
    else:
        return 0


titanic_testing_df['Fare'] = titanic_testing_df.apply(
    lambda row: add_fare_values(row['Fare']), axis=1)

# Calculate the number of relatives for each row


def add_relatives(sibsp, parch):
    return sibsp + parch


titanic_df['Relatives'] = titanic_df.apply(
    lambda row: add_relatives(row['SibSp'], row['Parch']), axis=1)
titanic_testing_df['Relatives'] = titanic_testing_df.apply(
    lambda row: add_relatives(row['SibSp'], row['Parch']), axis=1)

# Determine the per ticket price for each passenger.
titanic_combined_df = titanic_df.append(titanic_testing_df)
tickets = titanic_combined_df.groupby(
    ['Ticket'])['PassengerId'].count().reset_index()


def fare_per_person(row):
    fare_per_person = row['Fare'] / \
        tickets[tickets.Ticket == row['Ticket']]['PassengerId']
    return fare_per_person.values[0]


titanic_df['FarePerPerson'] = titanic_df.apply(fare_per_person, axis=1)
titanic_testing_df['FarePerPerson'] = titanic_testing_df.apply(
    fare_per_person, axis=1)

# Bin the ages


def bin_age(age):
    binned_age = 0

    if age > 0 and age <= 15:
        binned_age = 0
    elif age > 15 and age <= 25:
        binned_age = 1
    elif age > 25 and age <= 35:
        binned_age = 2
    elif age > 35 and age <= 45:
        binned_age = 3
    elif age > 45 and age <= 55:
        binned_age = 4
    elif age > 55:
        binned_age = 5

    return binned_age


titanic_df['AgeBinned'] = titanic_df.apply(
    lambda row: bin_age(row['Age']), axis=1)
titanic_testing_df['AgeBinned'] = titanic_testing_df.apply(
    lambda row: bin_age(row['Age']), axis=1)

# Convert Female to 0 and Male to 1
titanic_df = titanic_df.replace('female', 0)
titanic_df = titanic_df.replace('male', 1)

titanic_testing_df = titanic_testing_df.replace('female', 0)
titanic_testing_df = titanic_testing_df.replace('male', 1)

# Impute 2 missing values of embarked with most common value of 'S'
titanic_df.Embarked.fillna(titanic_df.Embarked.describe().top, inplace=True)
titanic_testing_df.Embarked.fillna(
    titanic_testing_df.Embarked.describe().top, inplace=True)

# Drop columns that data that will not likely be helpful
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Age', 'SibSp',
                              'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
titanic_testing_df = titanic_testing_df.drop(
    ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

# Convert pandas dataframe into numpy array
titanic_np = titanic_df.to_numpy()
titanic_testing_np = titanic_testing_df.to_numpy()
titanic_solutions_np = titanic_solutions_df.drop(
    ['PassengerId'], axis=1).to_numpy()
print(f'Shape of titanic_np: {titanic_np.shape}')
print(f'Shape of titanic_testing_np: {titanic_testing_np.shape}')
print(f'Shape of titanic_solutions_np: {titanic_solutions_np.shape}')

# Create separate numpy arrays for features and labels
titanic_features = titanic_np[:, 1:]
titanic_labels = titanic_np[:, 0:1]
titanic_testing_features = titanic_testing_np

print(f'Shape of titanic_features: {titanic_features.shape}')
print(f'Shape of titanic_labels: {titanic_labels.shape}')
print(f'Shape of titanic_testing_features: {titanic_testing_features.shape}')

# Split the data into 90% training and 10% testing
features_train, features_test, labels_train, labels_test = train_test_split(
    titanic_features, titanic_labels, test_size=0.10, random_state=10)

print(f'Shape of features_train: {features_train.shape}')
print(f'Shape of labels_train: {labels_train.shape}')
print(f'Shape of features_test: {features_test.shape}')
print(f'Shape of labels_test: {labels_test.shape}')

# Grid Search for Decision Tree Classifier
algorithm = "ANN"

# Tuning the Neural Network


def build_model(hp):
    n_features = features_train.shape[1]

    model = keras.models.Sequential()

    for i in range(hp.Int("n_layers", min_value=2, max_value=5, step=1)):
        model.add(
            Dropout(hp.Float("dropout", min_value=0.00, max_value=0.08, step=0.02)))
        model.add(Dense(hp.Int("dense_{i}_units", min_value=1024,
                               max_value=2304, step=256), input_dim=n_features, activation='relu'))

    model.add(Dense(n_classes, input_dim=n_features, activation=hp.Choice(
        'activation', values=['softmax', 'sigmoid'])))

    # Compile your model with accuracy as your metric.
    opt = Adam(hp.Float("learning_rate", min_value=0.0002, max_value=0.0007, step=0.0001))
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if algorithm == "ANN":
    # Convert the labels to one-hot encoding.
    n_classes=2
    labels_train=np_utils.to_categorical(titanic_labels, n_classes)
    print(f'Shape of one hot encoded labels_train: {labels_train.shape}')

    # Setup keras tuner.
    tuner=RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=6,
        executions_per_trial=1,
        directory=os.path.normpath('C:/_keras/' + str(int(time.time())))
    )

    tuner.search(
        x=titanic_features,
        y=labels_train,
        verbose=1,
        epochs=250,
        batch_size=64,
        validation_split=.10
    )

# Print the best tuner hyperparameters
print(tuner.results_summary())
model=tuner.get_best_models(num_models=1)

# Predict the test set labels.
labels_testing_predict=model[0].predict(titanic_testing_features)
labels_testing_predict=np.argmax(labels_testing_predict, axis=-1)

# Convert numpy labels array to dataframe
labels_testing_predict_df=pd.DataFrame(
    labels_testing_predict, index=None, columns=['Survived'])

# Copy the predicted survival labels to the submission dataframe and change column to int64
titanic_submission_df['Survived']=labels_testing_predict_df['Survived']
titanic_submission_df['Survived']=titanic_submission_df['Survived'].astype(
    'int64')

# Drop all columns not needed for submission.
titanic_submission_df_final=titanic_submission_df.drop(
    ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

# Write to CSV results file
titanic_submission_df_final.to_csv(
    '_Submission/014_Ryan_Submission_ANN.csv', index=False)

# Calculate the accuracy versus the Kaggle test solutions (doesn't affect previous csv export)
titanic_submission_df_final=titanic_submission_df_final.drop(
    ['PassengerId'], axis=1)
titanic_solutions_df=titanic_solutions_df.drop(['PassengerId'], axis=1)
titanic_solutions_df.columns=['Survived']

accuracy_test=accuracy_score(
    titanic_solutions_df, titanic_submission_df_final)
print(f"Accuracy for test dataset: {accuracy_test:.5f}")
