##### Problem: Train a logistic regression classifier to predict whether a flower is Iris Virginica or not..  ####


# Importing the libraries
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Importing the dataset.
iris = datasets.load_iris()

# Printing the keys of the dataset
# print(list(iris.keys()))

# Printing the data from the database
# print(iris['data'])

# Printint the target from the database
# print (iris.target)

# Print the data description
# print(iris.DESCR)

# Print the shape of the dataset
# print(iris.data.shape)

# This Logistic Regression Model is using only one feature to predict the label 
X = iris['data'][:, 3:]
# print(X)

Y = (iris['target'] == 2).astype(int)
# print(Y)

# Train a logistic regression classifier.
clf = LogisticRegression()
clf.fit(X,Y)
example = clf.predict([[2.6]])
# print(example)


# Using Matplotlib to plot the visualization
X_new = np.linspace(0,3, 1000).reshape(-1, 1)
Y_prob = clf.predict_proba(X_new)
plt.plot(X_new, Y_prob[:, -1], "g-", label= "virginica")
plt.show()
# print(X_new)