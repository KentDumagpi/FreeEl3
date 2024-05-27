import pandas as pd
from sklearn.datasets import load_iris

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
