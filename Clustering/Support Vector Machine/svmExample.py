import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.partial_fit(x_train)
x_test = sc.transform(x_test)

#Creating the classifier SVM
classifier = SVC(kernel='linear', random_state= 0)
classifier.fit(x_test, y_test)
y_pred = classifier.predict(x_test)

#Making a cunfuison Matrixt
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Visualising the Test set
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() -1, stop = x_set[:, 0].max() + 1, step = 0.01),
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), 
alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)


plt.title('SVM (test data)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

plt.legend()
plt.show() 

