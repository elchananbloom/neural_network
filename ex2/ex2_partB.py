# part B we build another layer with
# 2 nodes
# 500 iter
# relu function
# because in part A thous param give as best result
# the reason why if we increase the
# number of node in the hidden layer the accuracy of the test decrease is overfiting

# NOTE -  we use the code from part A and add more leyer before the output layer we call that layer 12 so W12 are the
# wight in this layer


# number of nodes in the two hidden layer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

alpha = 1  # learning rate
costs = []
nh = 2


# # number of iteration
# iterations = 500


def relu(z):
    """
      calculate relu function to z
    :param z:  an array
    :return: relu of z
    """
    return np.maximum(0, z)


# derivative relu function
relu_der = lambda x: np.array([(i > 0) * 1 for i in x])


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
        "W12": np.random.randn(n_h, n_h) * 0.01,  # we add here new wight and b
        "b12": np.zeros([n_h, 1]),
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b2": np.zeros([n_y, 1]),
    }


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
    Z1 = parameters["W1"].dot(X) + parameters["b1"]
    A1 = func(Z1)

    # new Hidden Layer
    Z12 = parameters["W12"].dot(A1) + parameters["b12"]
    A12 = func(Z1)
    # Output Layer
    Z2 = parameters["W2"].dot(A12) + parameters["b2"]
    A2 = sigmoid(Z2)
    # print(A2)
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z12": Z12,
        "A12": A12,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


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
    # Assuming the Log_loss function is used -- like last time:
    # Output Layer--similar to the last time
    # dZ2 = cache["A2"] - Y #for the sigmoid layer
    # dW2 = (1 / m) * dZ2.dot(cache["A1"].T)
    # db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    # Output Layer
    dA2 = -1 * (Y - cache["A2"])  # The derivative of MSE is -(Y-YP) (derivative of cost)
    dZ2 = dA2 * sigmoid_der(cache["Z2"])  # output derivative * node derivative
    dW2 = (1 / m) * np.dot(dZ2, cache[
        "A12"].T)  # for the input- A1 is the input to the second level, as X is the input to the first level
    db2 = (1 / m) * np.sum(dZ2)
    # db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # new Hidden Layer
    dA12 = np.dot(parameters["W2"].T, dA2)
    dZ12 = dA12 * func(cache["Z12"])  # change here for the targil
    dW12 = (1 / m) * np.dot(dZ12, cache["A1"].T)
    db12 = (1 / m) * np.sum(dZ12)

    # hidden layer
    dA1 = np.dot(parameters["W12"].T, dA12)
    dZ1 = dA1 * func(cache["Z1"])  # change here for the targil
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1)

    return {"dW1": dW1, "dW12": dW12, "dW2": dW2, "db1": db1, "db12": db12, "db2": db2}


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
        "W12": parameters["W12"] - learning_rate * grads["dW12"],
        "W2": parameters["W2"] - learning_rate * grads["dW2"],
        "b1": parameters["b1"] - learning_rate * grads["db1"],
        "b12": parameters["b12"] - learning_rate * grads["db12"],
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
    n_x = X.shape[0]
    n_h = nh
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


training_relu_data = {
    '500 iter': [0],
    '1000 iter': [0],
    '1500 iter': [0],
    '2000 iter': [0]

}
training_two_layer_df = pd.DataFrame(training_relu_data,
                                     index=['2 nodes'])
testing_relu_data = {
    '500 iter': [0],
    '1000 iter': [0],
    '1500 iter': [0],
    '2000 iter': [0]

}
testing_two_layer_df = pd.DataFrame(testing_relu_data,
                                    index=['2 nodes'])

url = 'https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true'
# url = 'https://github.com/rosenfa/nn/blob/master/class2/spam.csv?raw=true'
df = pd.read_csv(url, header=0, error_bad_lines=False)
features = df.drop(['Outcome'], axis=1)
features = ((features - features.mean()) / features.std())
X = np.array(features)
Y = np.array(df['Outcome'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
df

X_train, X_test = X_train.T, X_test.T

for iterations in range(500, 2001, 500):
    parameters, costs = nn_model(X_train, Y_train, iterations, alpha, nh, relu, relu_der)
    Y_train_predict = predict(X_train, parameters, relu)
    train_acc = prediction_accuracy(Y_train_predict, Y_train)
    Y_test_predict = predict(X_test, parameters, relu)
    test_acc = prediction_accuracy(Y_test_predict, Y_test)
    training_two_layer_df.loc[f'{nh} nodes', f'{iterations} iter'] = train_acc
    testing_two_layer_df.loc[f'{nh} nodes', f'{iterations} iter'] = test_acc
    print("Training acc : ", str(train_acc))
    print("Testing acc : ", str(test_acc))


x_range = list(map(lambda x: x, range(500, 2001, 500)))
y_train = list(map(lambda x: training_two_layer_df.loc[f'{nh} nodes', f'{x} iter'], range(500, 2001, 500)))
y_test = list(map(lambda x: testing_two_layer_df.loc[f'{nh} nodes', f'{x} iter'], range(500, 2001, 500)))
plt.plot(x_range, y_train, label="train")
plt.plot(x_range, y_test, label="test")
plt.legend()
plt.title = " 2 hidden layers \naccuracy per iterations"
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.savefig('output_partB.png')
