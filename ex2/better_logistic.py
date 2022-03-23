# -*- coding: utf-8 -*-
"""better logistic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YMOWeJ7YZkwaMPsRrMqlwVx9YYWoIs0F

Adapted from the Homework of Otniel Elkayam and Guy Yechezkel from the Deep Learning Toar Sheni class last year with Dr. Elishai Ezra Tzur. This code is similar to the one we saw from last week, but uses pandas and a dictionary for the data processing.  The dictionary will be useful when we generalize the process.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

"""### sigmoid(z)

Here we define our activation function; the sigmoid function 

s = $g(\theta^{T}x)$

$z = \theta^{T}x$

$g(z) = \frac{1}{1+e^{(-z)}}$

X := data set

$\theta$ := vector of weights

Compute the sigmoid of z (A scalar or numpy array of any size) returns s

"""

def sigmoid(z):
    X = np.exp(z)
    return X/(1+X)

"""Some examples of using this function. Notice that we can give it an array of values.

Verify: sigmoid([0, 2]) = [ 0.5, 0.88079708]
"""

print(sigmoid([0,2]))
print(sigmoid(2))
print(sigmoid(np.array([2])))

"""### initialize_with_zeros(dim): w, b
Creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

Verify: initialize_with_zeros(2) = w = [[ 0.][ 0.]], b = 0  
"""

def initialize_with_zeros(dim):
    return np.zeros((dim, 1)), 0

print(initialize_with_zeros(2))

"""### propagate(w, b, X, Y): cost, dw, db
Implement the cost function and its gradient for the propagation
* w -- weights, a numpy array of size (number of features, 1)
* b -- bias, a scalar
* X -- data of size (num of features, number of examples)
* Y -- true "label" vector (containing 0 /1) of size (1, number of examples)
* cost -- negative log-likelihood cost for logistic regression
$J(\theta) = \frac{1}{m} * (-y * log(h) - (1-y)log(1-h)) $
* dw -- gradient of the loss with respect to w, thus same shape as w
* db -- gradient of the loss with respect to b, thus same shape as b

We divide each of these by m (number of records) as before:

$\frac{\delta J(\theta)}{\delta\theta_{j}} = \frac{1}{m} * X^{T}$

"""

def propagate(w, b, X, Y):
    assert X.shape[0] == w.shape[0]
    m = X.shape[1]
    assert X.shape[1] == Y.shape[1]
    
    Z = np.dot(w.T, X) + b #Activation function for z
    assert Z.shape == (1, m)
    A = sigmoid(Z)
    dZ = A - Y
    dw = 1/m * np.dot(X, dZ.T)
    db = 1/m * np.sum(dZ)
    #cost calculation for debugging
    cost = np.mean(-(Y*np.log(A) + (1-Y)*np.log(1-(A))))
    
    return cost, dw, db

"""Verify:

propagate(np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])) =

cost = 5.801545319394553, dw = [[ 0.99845601] [ 2.39507239]], db = 0.0014555781367842449
"""

ex_w= np.array([[1.],[2.]])
ex_X = np.array([[1.,2.,-1.],[3.,4.,-3.2]])
ex_Y = np.array([[1,0,1]])
print(ex_X.shape[0])
print(ex_X.shape[1])
print(propagate(ex_w, 2., ex_X, ex_Y))

"""### predict(w, b, X): Y_prediction
Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
* w -- weights, a numpy array of size (num_px, 1)
* b -- bias, a scalar
* X -- data of size (number of features, number of example records)
* Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in x

"""

def predict(w, b, X):
    Z = np.dot(w.T, X) + b 
    A = np.rint(sigmoid(Z))
    return A
    '''This rounds the values like:
    def predict(X, theta, threshold=0.5):
    if predict_probs(X, theta) >= threshold:
        return 1
        print(A)'''

""" 
Verify, if:
w = np.array([[0.1124579],[0.23106775]])

b = -0.3

X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])

then:

predict(w, b, X) = [[ 1. 1. 0.]]
"""

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
#print(X.shape)
#print(w.shape)
print(predict(w, b, X))

print(predict(np.array([[0.1124579],[0.23106775]]), -0.3, np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])))

"""If you haven't seen what a dictionary is in python, here is your chance!  It's a data structure to store data values in key:value pairs. There are many examples on the Internet like: https://www.w3schools.com/python/python_dictionaries.asp, but here is a short example here:"""

example_dict = {'one': 1, 'two': 2, 'three': 3}
print("The value for key 'one' is", example_dict.get('one'))
# Or, you can also do this
print("The key for value a is", example_dict['two'])

"""### optimize(w, b, X, Y, num_iterations, learning_rate): params, grads, costs
Optimizes w and b by running a gradient descent algorithm
* w -- weights, a numpy array of size (number of features, 1)
* b -- bias, a scalar
* X -- data of shape (number of features, number of examples)
* Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
* num_iterations -- number of iterations of the optimization loop
* learning_rate -- learning rate of the gradient descent update rule
* params -- dictionary containing the weights w and bias b
* grads -- dictionary containing the gradients of the weights and bias with respect to the cost
function
* costs -- list of all the costs computed during the optimization


"""

def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []
    
    for i in range(num_iterations):
        cost, dw, db = propagate(w, b, X, Y)
        costs.append(cost)
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
    grads = {'dw':dw, 'db':db}
        
    thetas = {'w':w, 'b':b}
    return thetas, grads, costs

"""Here's an example of running this code. Try to understand what you will see:
Verify:

optimize(np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]]),

num_iterations= 100, 
learning_rate = 0.009)
=

w = [[ 0.19033591] [ 0.12259159]],

b = 1.92535983008, dw = [[ 0.67752042] [ 1.41625495]], db = 0.219194504541
"""

theta, grads, costs = optimize(np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]]),
num_iterations=100, learning_rate = 0.009)
print(theta, grads)

"""### model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5): d
Builds the logistic regression model by calling the functions implemented above
* X_train -- training set represented by a numpy array of shape (number of features, m_train)
* Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
* X_test -- test set represented by a numpy array of shape (number of features, m_test)
* Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
* num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
* learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
* d -- dictionary containing information about the model. 
"""

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
    dim = X_train.shape[0]
#     print(':',Y_train.shape, Y_train.shape[1])
    w, b = initialize_with_zeros(dim)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    Y_hat1=predict(params['w'], params['b'], X_train)
    Y_hat2=predict(params['w'], params['b'], X_test)
    #print(("pred",Y_hat1),'real', Y_train)
    acc_train=1-np.mean(np.sqrt(np.square(Y_hat1 - Y_train)))
    acc_test=1-np.mean(np.sqrt(np.square(Y_hat2 - Y_test)))
    
    d = {'w':params['w'],
        'b':params['b'],
        'costs':costs,
        'Y_prediction_train':acc_train,
        'Y_prediction_test':acc_test,
        'num_iterations':num_iterations,
        'learning_rate':learning_rate}
    
    return d

X_train, Y_train, X_test,  Y_test = np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]]), np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]]), np.array([[1,1,0]])

print(X_train.shape,Y_train.shape)

md = model(X_train, Y_train, X_test,  Y_test,
num_iterations=100, learning_rate = 0.009)
print(md)

import pandas as pd
from sklearn.model_selection import train_test_split

url = 'https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true'
#url = 'https://github.com/rosenfa/nn/blob/master/class2/spam.csv?raw=true'
df=pd.read_csv(url,  header=0, error_bad_lines=False) 

features = df.drop(['Outcome'], axis = 1)
#features = df.drop(['class'], axis = 1)
features = ((features - features.mean())/features.std())

X = np.array(features)
Y = np.array(df['Outcome'])
#Y = np.array(df['class'])
Y = Y.reshape(len(Y),1)

X_train, X_test, Y_train,  Y_test = train_test_split(X, Y, random_state=0)
df

my_model = model(X_train.T, Y_train.T, X_test.T, Y_test.T, num_iterations = 1000, learning_rate = 0.01)

for i in range(10):
    print('Cost after iteration',i*100,':', my_model['costs'][i*100])
print('\nTrain accuracy:',my_model['Y_prediction_train'],'\nTest accuracy:', my_model['Y_prediction_test'])

plt.plot(my_model['costs'])

from sklearn.linear_model import LogisticRegression
sk_model = LogisticRegression()     
sk_model.fit(X_train, Y_train.ravel() ) #if you don't do "ravel" it tell you to add it because of the shape 
accuracy = sk_model.score(X_test, Y_test)
print("accuracy = ", accuracy * 100, "%")
#print(Y_train)

"""It seems there is room for our model to be better (77.6% accuracy vs. 80.2% for Sklearn).  Maybe there is a combination of iterations and learning rates for gradient descent that will be better?
Yes, I think there might be :)
Let's try:
num_iterations=5000 and learning_rate=0.005
"""

my_model = model(X_train.T, Y_train.T, X_test.T, Y_test.T, num_iterations = 5000, learning_rate = 0.005)

plt.plot(my_model['costs'])

print('\nTrain accuracy:',my_model['Y_prediction_train'],'\nTest accuracy:', my_model['Y_prediction_test'])

"""Where did we see the value of 80.20833333333334% accuracy before :)"""