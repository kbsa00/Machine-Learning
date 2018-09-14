# Decision Tree Regression 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values 


# Fitting the Decision Tree Regression to the Dataset
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(x, y)

y_pred = regressor.predict(6.5)
print(y_pred)

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color= 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or bluff (Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()