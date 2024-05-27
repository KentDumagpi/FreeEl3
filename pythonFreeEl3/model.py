import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset from sklearn
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(X.head(10))
print(y.head(10))

print(X.info())

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'max_iter': [200, 500, 1000]  # Increased max_iter values
}

# Initialize the GridSearchCV object for Logistic Regression
grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, n_jobs=-1, scoring='accuracy')

# Perform Grid Search for Logistic Regression
grid_search_lr.fit(X_train, y_train)

# Best parameters and best score for Logistic Regression
best_params_lr = grid_search_lr.best_params_
best_score_lr = grid_search_lr.best_score_
print("Best parameters for Logistic Regression: ", best_params_lr)
print("Best cross-validation accuracy for Logistic Regression: ", best_score_lr)

# Train the Logistic Regression model with best parameters
model_lr = LogisticRegression(**best_params_lr)
model_lr.fit(X_train, y_train)

# Predict on the test set with Logistic Regression
y_pred_lr = model_lr.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Test accuracy for Logistic Regression: ", accuracy_lr)
print("Classification report for Logistic Regression:\n", classification_report(y_test, y_pred_lr))
print("Confusion matrix for Logistic Regression:\n", confusion_matrix(y_test, y_pred_lr))

# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the GridSearchCV object for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, n_jobs=-1, scoring='accuracy')

# Perform Grid Search for Random Forest
grid_search_rf.fit(X_train, y_train)

# Best parameters and best score for Random Forest
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_
print("Best parameters for Random Forest: ", best_params_rf)
print("Best cross-validation accuracy for Random Forest: ", best_score_rf)

# Train the Random Forest model with best parameters
model_rf = RandomForestClassifier(**best_params_rf)
model_rf.fit(X_train, y_train)

# Predict on the test set with Random Forest
y_pred_rf = model_rf.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Test accuracy for Random Forest: ", accuracy_rf)
print("Classification report for Random Forest:\n", classification_report(y_test, y_pred_rf))
print("Confusion matrix for Random Forest:\n", confusion_matrix(y_test, y_pred_rf))
