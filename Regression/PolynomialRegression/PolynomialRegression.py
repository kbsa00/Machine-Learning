import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

## Importing the data. 
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset.
lin_reg = LinearRegression()
lin_reg.fit(x, y);

#Fitting Polynomial Regression to the dataset.
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


# Visualising the Linear Regression results 
'''plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() '''


# Visualising in the Polynomial regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 


# Predicting a new result with Linear 
print(lin_reg.predict(6.5))

# Predicting a new result wit polynomal
print(lin_reg2.predict(poly_reg.fit_transform(6.5)))