#elchanan bloom 207193855
#Chananel Zaguri 206275711
# -*- coding: utf-8 -*-
"""Log_Reg_From_Scratch_Iris.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oHqZRLJRJHrwysNEZBpsbrXz2tABKg_D

Code is based on an example my student Gilad Felsen gave me (he didn't write this for this class!!!)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""Using the Iris dataset to perform logistic regression from scratch in python. I already saved a nicely processed file on my Github."""

url = "https://github.com/rosenfa/nn/blob/master/iris2.csv?raw=true"
df_iris=pd.read_csv(url,  header=0, error_bad_lines=False)

df_iris
#print(df_iris.Species)

"""Adding a column of 1's to the X matrix for the bias terms and converting the dataframe to a numpy array (more standard)."""

X = np.asarray(df_iris.drop('Species',1))
X = np.append(np.ones([len(X),1]),X,1)
y = np.asarray(df_iris['Species'])
#X

"""Now make the theta vector. Note that X.shape returns two values and the 1 value is the number of columns (attributes + b)"""

print(X.shape)
theta = np.zeros(X.shape[1])
theta

"""Here we define our activation function; the sigmoid function 

$h_{\theta}(x) = g(\theta^{T}x)$

$z = \theta^{T}x$

$g(z) = \frac{1}{1+e^{(-z)}}$




X := data set

$\theta$ := vector of weights



h = hypothesis
"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.dot(X, theta)
a = sigmoid(z)

"""Next we will define our loss (Cost) function:


$J(\theta) = \frac{1}{m} * (-y * log(h) - (1-y)log(1-h)) $

---



Note: when y = 0 the first half of the equation is 0,  
and when y = 1, the second half of the equation is equal to 0.
"""

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

#gradient = np.dot(X.T, (a - y)) / y.shape[0]

"""We then subtract our values for theta by our chose learning rate * the gradient and loop for gradient descent."""

def predict_probs(X, theta):
    return sigmoid(np.dot(X, theta))


def predict(X, theta, threshold=0.5):
    if predict_probs(X, theta) >= threshold:
        return 1
    return 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

print(y_train.shape)

X_train.shape
theta

#fit the data 
cost_array = [] # keeping a list of the cost at each iteration to make sure it is constantly decreasing
iterations = 1000 #like the red arrow in slide 37
lr = 0.01

def dot(X_train,theta):
    grad = np.zeros(X_train.shape[0])
    for i in range(X_train.shape[0]):
        for j in range(0, X_train.shape[1]):
            grad[i] = grad[i] + X_train[i][j]*theta[j]
            # grad[i] = grad[i] - (y_actual[j]-y_pred[j])*x[j][i] #other way to write this
    return grad
for i in range(iterations):   
    z = dot(X_train, theta)
    #z = np.dot(X_train, theta)
    a = sigmoid(z)
    gradient = dot(X_train.T, (a - y_train)) / y_train.shape[0]
    #if (i>998):
    #  print(a)
    theta -= lr * gradient
    cost = loss(a, y_train)
    cost_array.append(cost)

print(X_train.shape)

"""Plotting the reducing in the error per the number of iterations of gradient descent:"""

plt.plot(cost_array)
#plotting the cost against the number of iterations to make sure our model is improving the whole time

print(theta)
print(y_train[0])
print(X_train[0])

"""Now We will test our model on our test data."""

correct = 0

for x,y in zip(X_test, y_test):
    p = predict(x, theta)
    if p == y:
        correct += 1

n = len(y_test)
accuracy = (correct)/n*100
print("accuracy: {}".format(accuracy) , "%")

"""It looks like our model is performing accurately!

Now we will use the sklearn built in functions to compare our model.
"""

from sklearn.linear_model import LogisticRegression
sk_model = LogisticRegression()     
sk_model.fit( X_train, y_train )

accuracy = sk_model.score(X_test, y_test)
print("accuracy = ", accuracy * 100, "%")

"""Note that while the results are the same, the weights weren't!"""

print('Coefficients: \n', sk_model.coef_)

print(theta)


y_train