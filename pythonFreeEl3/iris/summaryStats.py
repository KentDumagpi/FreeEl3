import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris(as_frame=True)
X = iris.data

# Compute summary statistics
summary_stats = X.describe()
print(summary_stats)