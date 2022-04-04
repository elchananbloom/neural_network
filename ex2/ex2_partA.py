# -*- coding: utf-8 -*-
"""Neural net with 2 layers and MSE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vo0PekXT0NuBIP0STYUKWUZ9KCJMTUkI

Adapted from the homeworks with the simple logistic model from the Deep Learning Toar Sheni class last year with Dr. Elishai Ezra Tzur. We will now add one hidden layer with 3 nodes and see how that changes things.
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

Verify: sigmoid([0, 2]) = [ 0.5, 0.88079708]
"""


def sigmoid(z):
    """
    calculate sigmoid function to z
    :param z:- an array
    :return: sigmoid of z
    """
    X = np.exp(z)
    return X / (1 + X)


def sigmoid_der(z):
    """
    Calculate the derivative of sigmoid
    :param z: an array
    :return: derivative of sigmoid
    """
    A = sigmoid(z)
    return A * (1 - A)


# relu function
def relu(z):
    """
      calculate relu function to z
    :param z:  an array
    :return: relu of z
    """
    return np.maximum(0, z)

def tanh_der(z):
    """
    calculate tanh function to z
    :param z: an array
    :return: tanh of z
    """
    X = np.tanh(z)
    return 1 - X ** 2


# derivative relu function
relu_der = lambda x: np.array([(i > 0) * 1 for i in x])

"""Some examples of using this function. Notice that we can give it an array of values (not critical for us)"""

print(sigmoid([0, 2]))
print(sigmoid(2))
print(sigmoid(np.array([4])))

"""### initialize_with_random: w, b
We don't use inialize with zero as zero values can be bad as we discussed in class.
Instead we inialize with random numbers.
"""

""" Initialize w and b for the both layers according to the number of the features and number of neurons in the layers.
W should be initialized randomly to small values (otherwise, values at the activation functions could be at the flat part).
"""


def initialize_parameters(n_x, n_h, n_y):
    """
    parameters = Weights and b we use them in foreword propagation and backward  propagation
    :param n_x: number of nodes in the input layer
    :param n_h: number of nodes in the hidden layer
    :param n_y:  number of nodes in the output layer
    :return: dict with the Weights and b
    """
    return {
        "W1": np.random.randn(n_h, n_x) * 0.01,
        "b1": np.zeros([n_h, 1]),
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b2": np.zeros([n_y, 1]),
    }


# Toy example
print(initialize_parameters(4, 3, 1))

"""### forward propagate(X, thetas): 
retuns: A2 (the final value) and the cache of values
Implement the forward propagation
* parameters -- python dictionary containing your parameters (output of initialization function)
Note that thetas is now a cache of thetas (weights) 
* A2 -- The sigmoid output of the second activation
* cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
"""


def forward_propagation(X, parameters, func):
    """
    :param X: array input to the network
    :param parameters: dict All the Weights and b
    :param func: activation func - (tnah\sigmoid\relu)
    :return: A2, cache - use them in back propagation
    """

    # Hidden Layer
    Z1 = parameters["W1"].dot(X) + parameters["b1"]  # first stage of the network NOTE: if we increase nh the width of
    # Z1 will increase
    # added the activation func (tnah\sigmoid\relu) which we want to run it on
    A1 = func(Z1)  # calculate A1 with Activation func
    # Output Layer
    Z2 = parameters["W2"].dot(A1) + parameters["b2"]  # second stage
    A2 = sigmoid(Z2)
    # print(A2)
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


"""Back_propagation calcuates the weight updates using the derivative of the different activation functions. These equations are similar to those in the lecture."""


def backward_propagation(parameters, cache, X, Y, func):
    """
    this method hellp us to update the wights and b for the next iteration
    :param parameters: the original wights
    :param cache:  the wights before the update
    :param X: data
    :param Y: label
    :param func: activation func - (tnah\sigmoid\relu)
    :return: wights after the update
    """
    m = X.shape[1]  # Number of samples
    # Output Layer
    dA2 = -1 * (Y - cache["A2"])  # The derivative of MSE is -(Y-YP) (derivative of cost)
    dZ2 = dA2 * sigmoid_der(cache["Z2"])  # output derivative * node derivative
    dW2 = (1 / m) * np.dot(dZ2, cache[
        "A1"].T)  # for the input- A1 is the input to the second level, as X is the input to the first level
    db2 = (1 / m) * np.sum(dZ2)
    # db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    # Hidden Layer
    dA1 = np.dot(parameters["W2"].T, dA2)
    # added the derivative func (tnah_der\sigmoid_der\relu_der) which we want to run it on
    dZ1 = dA1 * func(cache["Z1"])  # change here for the targil
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1)
    # db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}


"""The cost function from the last example."""


def MSE_calculation(A, Y):
    """
    cost function
    :param A: data
    :param Y: label
    :return:  cost according to MSE
    """

    m = A.shape[1]
    E = Y - A
    cost = np.sum(E ** 2)
    return cost / (2 * m)


'''
def LogLoss_calculation(A,Y):
    cost = np.mean(-(Y*np.log(A) + (1-Y)*np.log(1-(A))))  
    return cost'''

"""Update the weights in the dictionary cache."""


def update_parameters(parameters, grads, learning_rate):
    """
    use the learning rate to update the params
    :param parameters: params before update
    :param grads: the derivative
    :param learning_rate:
    :return: params after the update
    """
    return {
        "W1": parameters["W1"] - learning_rate * grads["dW1"],
        "W2": parameters["W2"] - learning_rate * grads["dW2"],
        "b1": parameters["b1"] - learning_rate * grads["db1"],
        "b2": parameters["b2"] - learning_rate * grads["db2"],
    }


"""### nn_model(X, Y, num_iterations, learning_rate): d
Builds the logistic regression model by calling the functions implemented above
* X_train -- training set represented by a numpy array of shape (number of features, m_train)
* Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
* X_test -- test set represented by a numpy array of shape (number of features, m_test)
* Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
* num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
* learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
* d -- dictionary containing information about the model. 
"""


def nn_model(X, Y, iterations, lr, nh, func, func_der):
    """
    nh - number node in the hidden layer
    func - the activation method
    func_der - derivative of the activation method
 Builds the logistic regression model by calling the functions implemented above
* X_train -- training set represented by a numpy array of shape (number of features, m_train)
* Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
* X_test -- test set represented by a numpy array of shape (number of features, m_test)
* Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
* num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
* learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
* d -- dictionary containing information about the model.
    """
    n_x = X.shape[0]
    # number of nodes in the hidden layer
    n_h = nh  # change here for the targil
    n_y = 1
    parameters = initialize_parameters(n_x, n_h, n_y)
    print("Network shape ", X.shape[0], n_h, n_y)
    for i in range(iterations):
        A2, cache = forward_propagation(X, parameters, func)
        cost = MSE_calculation(A2, Y)
        # cost = LogLoss_calculation(A2,Y)
        grads = backward_propagation(parameters, cache, X, Y, func_der)
        parameters = update_parameters(parameters, grads, lr)
        costs.append(cost)
        # cost check
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    return parameters, costs


"""### predict(X, parameters): Y_prediction"""


def predict(X, parameters, func):
    """
    after we find the parameters we use them for check the result
    :param X: data
    :param parameters: wights
    :param func: activation
    :return: label
    """
    A2, cache = forward_propagation(X, parameters, func)
    return np.rint(A2)
    '''This round the values like:
    def predict(X, theta, threshold=0.5):
    if predict_probs(X, theta) >= threshold:
        return 1
        print(A)'''


def prediction_accuracy(y_pred, y_true):
    """
    find the accuracy of the network
    :param y_pred: predict
    :param y_true: actually
    :return:
    """
    return np.mean(y_pred == y_true)


import pandas as pd
from sklearn.model_selection import train_test_split

# creat 3 tables for the testing:
# relu
testing_relu_data = {
    'alpha': [0] * 6,
    '500 iter': [0] * 6,
    '1000 iter': [0] * 6,
    '1500 iter': [0] * 6,
    '2000 iter': [0] * 6

}
testing_relu_df = pd.DataFrame(testing_relu_data,
                               index=['1 nodes', '2 nodes', '3 nodes', '4 nodes', '5 nodes', '6 nodes'])

# tanh
testing_tanh_data = {
    'alpha': [0] * 6,
    '500 iter': [0] * 6,
    '1000 iter': [0] * 6,
    '1500 iter': [0] * 6,
    '2000 iter': [0] * 6

}
testing_tanh_df = pd.DataFrame(testing_tanh_data,
                               index=['1 nodes', '2 nodes', '3 nodes', '4 nodes', '5 nodes', '6 nodes'])

# sigmoid
testing_sigmoid_data = {
    'alpha': [0] * 6,
    '500 iter': [0] * 6,
    '1000 iter': [0] * 6,
    '1500 iter': [0] * 6,
    '2000 iter': [0] * 6

}
testing_sigmoid_df = pd.DataFrame(testing_sigmoid_data,
                                  index=['1 nodes', '2 nodes', '3 nodes', '4 nodes', '5 nodes', '6 nodes'])

# creat 3 tables for the training:
# tanh
training_tanh_data = {
    'alpha': [0] * 6,
    '500 iter': [0] * 6,
    '1000 iter': [0] * 6,
    '1500 iter': [0] * 6,
    '2000 iter': [0] * 6

}
training_tanh_df = pd.DataFrame(training_tanh_data,
                                index=['1 nodes', '2 nodes', '3 nodes', '4 nodes', '5 nodes', '6 nodes'])
# relu
training_relu_data = {
    'alpha': [0] * 6,
    '500 iter': [0] * 6,
    '1000 iter': [0] * 6,
    '1500 iter': [0] * 6,
    '2000 iter': [0] * 6

}
training_relu_df = pd.DataFrame(training_relu_data,
                                index=['1 nodes', '2 nodes', '3 nodes', '4 nodes', '5 nodes', '6 nodes'])
# sigmoid
training_sigmoid_data = {
    'alpha': [0] * 6,
    '500 iter': [0] * 6,
    '1000 iter': [0] * 6,
    '1500 iter': [0] * 6,
    '2000 iter': [0] * 6

}
training_sigmoid_df = pd.DataFrame(training_sigmoid_data,
                                   index=['1 nodes', '2 nodes', '3 nodes', '4 nodes', '5 nodes', '6 nodes'])

url = 'https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true'
# url = 'https://github.com/rosenfa/nn/blob/master/class2/spam.csv?raw=true'
#  read  csv file with panda
df = pd.read_csv(url, header=0, error_bad_lines=False)
# drop the Outcome
features = df.drop(['Outcome'], axis=1)
# normalize the data
features = ((features - features.mean()) / features.std())
X = np.array(features)
Y = np.array(df['Outcome'])
# split the data with sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
df

from sklearn.linear_model import LogisticRegression

sk_model = LogisticRegression()
sk_model.fit(X_train, Y_train)
accuracy = sk_model.score(X_test, Y_test)
print("accuracy = ", accuracy * 100, "%")
# print(Y_train)

X_train, X_test = X_train.T, X_test.T

num_iterations = 2000  # number of iterations  #change here for the targil
alpha = 1  # learning rate
costs = []
# parameters, costs = nn_model(X_train, Y_train,num_iterations,alpha)
# Y_train_predict = predict(X_train, parameters)
# train_acc = prediction_accuracy(Y_train_predict,Y_train)
# Y_test_predict = predict(X_test, parameters)
# test_acc = prediction_accuracy(Y_test_predict,Y_test)
# parameters["train_accuracy"] = train_acc
# parameters["test_accuracy"] = test_acc

# plt.plot(costs)

# print("Training acc : ", str(train_acc))
# print("Testing acc : ", str(test_acc))

# activation funcs
funcs = [np.tanh, sigmoid, relu]
# derivative funcs
funcs_der = [tanh_der, sigmoid_der, relu_der]
# training tables
df_train = [training_tanh_df, training_sigmoid_df, training_relu_df]
# testing tables
df_test = [testing_tanh_df, testing_sigmoid_df, testing_relu_df]
# run on the amount of activation funcs we have
for i in range(len(funcs)):
    # run 4 times for 500\1000\1500\2000 iterations
    for iterations in range(500, 2001, 500):

        for nh in range(1, 7):
            parameters, costs = nn_model(X_train, Y_train, iterations, alpha, nh, funcs[i], funcs_der[i])
            Y_train_predict = predict(X_train, parameters, funcs[i])
            train_acc = prediction_accuracy(Y_train_predict, Y_train)
            Y_test_predict = predict(X_test, parameters, funcs[i])
            test_acc = prediction_accuracy(Y_test_predict, Y_test)
            df_train[i].loc[f'{nh} nodes', f'{iterations} iter'] = train_acc
            df_test[i].loc[f'{nh} nodes', f'{iterations} iter'] = test_acc
    print(df_train[i])
    print(df_test[i])

    # plot the result
    plot_train = df_train[i].plot(title=f'training {funcs[i]}')
    ax1 = plt.gca()
    ax1.set_ylim([0.7, 0.9])
    plot_test = df_test[i].plot(title=f'testing {funcs[i]}')
    ax = plt.gca()
    ax.set_ylim([0.7, 0.9])
    plot_train.get_figure().savefig(f'output_train{i}.pdf', format='pdf')
    plot_test.get_figure().savefig(f'output_test{i}.pdf', format='pdf')
ax = plt.gca()
ax.set_ylim([0.7, 0.9])
plt.show()
