import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the data sets into training set and test set. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results 
y_pred = regressor.predict(x_test)

# Visualisig the Training set results
##plt.scatter(x_train, y_train, color ='red')
##plt.plot(x_train, regressor.predict(x_train), color = 'blue')
##plt.title('Salary Vs Experience (Training Set)')
#3plt.xlabel('Years of Experience')
##plt.ylabel('Salary')
##plt.show()

# Visiualising the test set results
plt.scatter(x_test, y_test, color ='red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
