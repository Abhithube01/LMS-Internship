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

def load_application_train():
    data = pd.read_csv("application_train.csv")
    return data




#See the shape of bigger dataset
df = load_application_train()
print(df.shape)        # (307511, 122)





def load():
    data = pd.read_csv("titanic.csv")
    return data



#See the shape of smaller dataset
df = load()
print(df.shape)       # (891, 12)




#Let's begin with the smaller dataset, "titanic.csv"
#We will try to detect outliers in a numerical 'Age' column by using 
#boxplot.

sns.boxplot(x=df["Age"])
plt.show()  #IMAGE IS BELOW ('titan_age_outlier.png')