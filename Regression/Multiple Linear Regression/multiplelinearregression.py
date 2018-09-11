import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

LabelEncoder_x = LabelEncoder()
x[:, 3] = LabelEncoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
x = onehotencoder.fit_transform(x).toarray()


# Avoiding the dummy variable trap
x = x[:, 1:]

# Splitting the test. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )

# Fitting multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set results 
y_pred = regressor.predict(x_test)

# Backward elemination
x = np.append(arr = np.ones((50, 1)).astype(int), values= x, axis = 1)
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()


x_opt = x[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog= y, exog= x_opt).fit()


x_opt = x[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog= y, exog= x_opt).fit()

x_opt = x[:,[0, 3]]
regressor_OLS = sm.OLS(endog= y, exog= x_opt).fit()


print(regressor_OLS.summary())