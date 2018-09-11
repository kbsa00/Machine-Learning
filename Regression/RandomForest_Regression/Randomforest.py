import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Create your regressor.
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(x, y)


y_pred = regressor.predict(6.5)
print(y_pred)

#Visuializing
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y,  color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or bluff (Random forest Regression')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()
