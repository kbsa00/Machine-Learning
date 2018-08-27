import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Importing the Dataset
dataset = pd.read_csv('Data.csv')

# Creating a matrix. -1 means we are taking all of the colums except for the last one.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Categorical Data
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling

sc_x = StandardScaler()

# We always have to fit and transform when we are working with our trainingset. 
# Unlike our test set, we can just use transform
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print('Printing x train: ', x_train)
print('Printing x test: ', x_test)

