import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Read in titanic training data
titanic_df = pd.read_csv("_Data/train_with_ages.csv")
titanic_testing_df = pd.read_csv("_Data/test_with_ages.csv")
titanic_submission_df = titanic_testing_df

# Set the fare to '0' for titanic_testing_df index '152'


def add_fare_values(fare):
    if fare >= 0:
        return fare
    else:
        return 0


titanic_testing_df['Fare'] = titanic_testing_df.apply(
    lambda row: add_fare_values(row['Fare']), axis=1)

# Describe the data
titanic_df.describe()

# Drop columns that data that will not likely be helpful
titanic_df = titanic_df.drop(
    ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
titanic_testing_df = titanic_testing_df.drop(
    ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Convert Female to 0 and Male to 1
titanic_df = titanic_df.replace('female', 0)
titanic_df = titanic_df.replace('male', 1)

titanic_testing_df = titanic_testing_df.replace('female', 0)
titanic_testing_df = titanic_testing_df.replace('male', 1)

# Compare Age vs Survival
point_plot = sns.pointplot(x='Age', y='Survived', data=titanic_df, ci=None)
plt.xlabel('Age')
plt.ylabel('Survived')
plt.title('Age vs. Survival')

# Decrease number of labels along x-axis
for label in point_plot.get_xticklabels():
    if np.float(label.get_text()) % 10 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)

# Convert pandas dataframe into numpy array
titanic_np = titanic_df.to_numpy()
titanic_testing_np = titanic_testing_df.to_numpy()
print(f'Shape of titanic_np: {titanic_np.shape}')
print(f'Shape of titanic_testing_np: {titanic_testing_np.shape}')

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

# Perform a grid search for the chosen classifier


def grid_search(classifier, param_grid):
    classifier = GridSearchCV(classifier, param_grid, verbose=6, n_jobs=-1)
    classifier.fit(features_train, labels_train)
    classifier.best_params_
    print(classifier.best_params_)
    print(f'Best score: {classifier.best_score_}')

# Train and validate the model for the chosen classifier


def train_validate_model(model, algorithm):
    model = model.fit(features_train, labels_train)
    labels_predict = model.predict(features_test)

    accuracy = accuracy_score(labels_predict, labels_test)
    print(f"Accuracy for {algorithm}: {accuracy:.3f}")

    return model

# Predict Survived labels for testing dataset


def predict_testing_labels(model):
    labels_testing_predict = model.predict(titanic_testing_features)
    return labels_testing_predict


# Grid Search for Decision Tree Classifier
algorithm = "SVC"

if algorithm == "DT":
    dtc_grid_search = tree.DecisionTreeClassifier()
    param_grid = {'criterion': ['gini', 'entropy'], 'min_samples_split': [
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]}
    grid_search(dtc_grid_search, param_grid)
elif algorithm == "LogReg":
    log_reg_grid_search = LogisticRegression()
    param_grid = {'penalty': ['l1', 'l2'], 'C': [
        0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]}
    grid_search(log_reg_grid_search, param_grid)
elif algorithm == "KNN":
    KNN_grid_search = KNeighborsClassifier()
    param_grid = {'weights': ['uniform', 'distance'], 'n_neighbors': [
        1, 2, 6, 10, 12, 14, 16, 18, 20, 30, 40], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    grid_search(KNN_grid_search, param_grid)
elif algorithm == "SVC":
    SVC_grid_search = SVC()
    #param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [0.01, 0.2, 0.6, 1], 'degree': [1, 2, 3, 4], 'gamma': ['scale', 'auto']}
    param_grid = {'kernel': ['poly'], 'degree': [1, 2, 3, 4], 'gamma': ['scale']}
    grid_search(SVC_grid_search, param_grid)
