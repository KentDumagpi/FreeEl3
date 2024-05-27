import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset from sklearn
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Set style
sns.set(style="whitegrid")

# Histograms for each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(X.columns):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=X[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(f'{feature} (cm)')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Pie chart for the distribution of species
species_counts = y.value_counts()
species_labels = iris.target_names

plt.figure(figsize=(8, 6))
plt.pie(species_counts, labels=species_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Distribution of Iris Species')
plt.show()

# Box plots for each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(X.columns):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x=y, y=X[feature], palette="pastel")
    plt.title(f'Box Plot of {feature}')
    plt.xlabel('Species')
    plt.ylabel(f'{feature} (cm)')
    plt.xticks(ticks=[0, 1, 2], labels=species_labels)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
