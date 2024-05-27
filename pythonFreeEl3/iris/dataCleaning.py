import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Print first 10 rows of features and target
print("First 10 rows of features:")
print(X.head(10))

print("\nFirst 10 rows of target:")
print(y.head(10))

# Check for missing values
print("\nMissing values in features:")
print(X.isnull().sum())

print("\nMissing values in target:")
print(y.isnull().sum())

# Display info about the dataset
print("\nDataset information:")
print(X.info())
