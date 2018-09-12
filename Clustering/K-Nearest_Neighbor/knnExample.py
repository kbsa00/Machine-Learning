import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Creating classifier and fitting it to the Traning set
classifier = KNeighborsClassifier(n_neighbors= 5, metric= 'minkowski', p=2)
classifier.fit(x_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)
