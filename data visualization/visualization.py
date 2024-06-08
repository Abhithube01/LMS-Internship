import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Explore the dataset
print(iris.info())
print(iris.describe())
print(iris.head())

# Check for missing values
print(iris.isnull().sum())

# Scatter plot: Sepal Length vs Sepal Width
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species')
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Pair plot
sns.pairplot(iris, hue='species')
plt.show()

# Box plot: Distribution of features
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris)
plt.title('Distribution of Iris Features')
plt.show()
