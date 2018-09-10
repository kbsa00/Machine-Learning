import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('RomPrice.csv')

x = dataset.iloc[:, 0].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

print(y_train)

x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

regression = LinearRegression()

# Do Fet..
regression.fit(x_train, y_train)

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regression.predict(x_test), color = 'blue')
plt.title('Antall rom for månedspris')
plt.xlabel('Antall Rom')
plt.ylabel('MNOK')
plt.show()