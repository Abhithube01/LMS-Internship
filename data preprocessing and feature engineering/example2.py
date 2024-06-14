import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Edward'],
    'Age': [25, 30, 35, 40, 120]  # Including an outlier
}

# Define the DataFrame
df = pd.DataFrame(data)

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)

# Calculate the Interquartile Range (IQR)
iqr = q3 - q1

# Determine upper and lower bounds for outliers
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

# Identify and print outliers
outliers = df[(df["Age"] < low) | (df["Age"] > up)]
print("Outliers:\n", outliers)