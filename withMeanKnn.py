#removed missing values
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

# load data from dataset
initial_data = pd.read_csv("D:\\ACADEMIC\\data\\data.csv")
#print(initial_data.head())

# drop id (col 1)
initial_data.drop(initial_data.columns[0], axis=1, inplace=True)

# drop unnamed column
for column in initial_data.columns:
    if "Unnamed" in column:
        initial_data.drop(column, axis = 1, inplace=True)

#print(initial_data.head())

# map class attribute
initial_data['diagnosis'] = initial_data['diagnosis'].map({'M': 1, 'B': 0})
#print(initial_data.head())

# check for imbalances
# sns.countplot(initial_data['diagnosis'], label="Sum")
# plt.show()

# check for nan values
col_labels = list(initial_data)

# for c in col_labels:
#     no_missing = initial_data[c].isnull().sum()
#     if no_missing > 0:
#         print(c)
#         print(no_missing)
#     else:
#         print(c)
#         print("No missing values")
#         print(' ')

# df = initial_data.describe()
# print(df.head())

# get rows with column value = 0. assume it is missing
# print((initial_data == 0).sum())

# replace 0 values with nan(null)
initial_data[['concavity_mean','concave points_mean','concavity_se','concave points_se','concavity_worst','concave points_worst']]=initial_data[['concavity_mean','concave points_mean','concavity_se','concave points_se','concavity_worst','concave points_worst']].replace(0, np.NaN)

# for c in col_labels:
#     no_missing = initial_data[c].isnull().sum()
#     if no_missing > 0:
#         print(c)
#         print(no_missing)
#     else:
#         print(c)
#         print("No missing values")
#         print(' ')

# replace rows with mean values
initial_data.fillna(initial_data.mean(), inplace=True)
# print((initial_data == 0).sum())

# plot for pre processed data (imbalance graph)
# sns.countplot(initial_data['diagnosis'],label="Sum")
# plt.show()

preprocessed_data = initial_data
# print((preprocessed_data == 0).sum())

X = preprocessed_data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean','fractal_dimension_mean',
                 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst','fractal_dimension_worst']]

Y = preprocessed_data['diagnosis']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.34, random_state = 0)

KNN = KNeighborsClassifier(n_neighbors=10)
modelKNN = KNN.fit(X_train, Y_train)

predictKNN = KNN.predict(X_test)
print(predictKNN)

print(accuracy_score(Y_test, predictKNN))
