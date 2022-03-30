#2 nodes 500 iter relu function

#nomber of nodes in the two hidden layer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

alpha = 1 #learning rate
costs = []
nh=2
#number of iteration
iterations=500
def relu(z):
    return np.maximum(0,z)

relu_der = lambda x : np.array([(i > 0) * 1 for i in x])

def sigmoid(z):
  X = np.exp(z)
  return X/(1+X)

def sigmoid_der(z):
  A = sigmoid(z)
  return A*(1-A)


""" Initialize w and b for the both layers according to the number of the features and number of neurons in the layers.
W should be initialized randomly to small values (otherwise, values at the activation functions could be at the flat part).
"""
def initialize_parameters (n_x, n_h, n_y):
    return {
    "W1":np.random.randn(n_h,n_x) * 0.01,
    "b1":np.zeros([n_h, 1]),
    "W12": np.random.randn(n_h, n_h) * 0.01,
    "b12": np.zeros([n_h, 1]),
    "W2":np.random.randn(n_y,n_h) * 0.01,
    "b2":np.zeros([n_y, 1]),
}

"""### forward propagate(X, thetas): 
retuns: A2 (the final value) and the cache of values
Implement the forward propagation
* parameters -- python dictionary containing your parameters (output of initialization function)
Note that thetas is now a cache of thetas (weights) 
* A2 -- The sigmoid output of the second activation
* cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
"""

def forward_propagation(X, parameters,func):
    #Hidden Layer
    Z1 = parameters["W1"].dot(X)+parameters["b1"]
    A1 = func(Z1) #change here for the targil

    # Hidden Layer
    Z12 = parameters["W12"].dot(A1) + parameters["b12"]
    A12 = func(Z1)  # change here for the targil
    #Output Layer
    Z2 = parameters["W2"].dot(A12)+parameters["b2"]
    A2 = sigmoid(Z2)
    #print(A2)
    cache = {
        "Z1":Z1,
        "A1":A1,
        "Z12": Z12,
        "A12": A12,
        "Z2":Z2,
        "A2":A2
    }
    return A2, cache

def backward_propagation(parameters, cache, X, Y,func):
    m = X.shape[1] # Number of samples
    #Assuming the Log_loss function is used -- like last time:
    #Output Layer--similar to the last time
    #dZ2 = cache["A2"] - Y #for the sigmoid layer
    #dW2 = (1 / m) * dZ2.dot(cache["A1"].T)
    #db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    #Output Layer
    dA2 =  -1 * (Y- cache["A2"]) #The derivative of MSE is -(Y-YP) (derivative of cost)
    dZ2 = dA2 * sigmoid_der(cache["Z2"]) #output derivative * node derivative
    dW2 = (1 / m) * np.dot(dZ2,cache["A12"].T ) #for the input- A1 is the input to the second level, as X is the input to the first level
    db2 = (1 / m) * np.sum(dZ2)
    #db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)



    #new Hidden Layer
    dA12 = np.dot(parameters["W2"].T, dA2)
    dZ12 =  dA12 * func(cache["Z12"])  #change here for the targil
    dW12 = (1 / m) * np.dot(dZ12, cache["A1"].T)
    db12 = (1 / m) * np.sum(dZ12)

    #hidden layer
    dA1 = np.dot(parameters["W12"].T, dA12)
    dZ1 = dA1 * func(cache["Z1"])  # change here for the targil
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1)

    return {"dW1":dW1,"dW12":dW12,"dW2":dW2,"db1":db1,"db12":db12,"db2":db2}

"""The cost function from the last example."""

def MSE_calculation(A,Y):
    m=A.shape[1]
    E=Y-A
    cost = np.sum(E**2)
    return cost/(2*m)

'''
def LogLoss_calculation(A,Y):
    cost = np.mean(-(Y*np.log(A) + (1-Y)*np.log(1-(A))))  
    return cost'''

"""Update the weights in the dictionary cache."""

def update_parameters(parameters, grads, learning_rate):
    return {
    "W1": parameters["W1"] - learning_rate*grads["dW1"],
    "W12": parameters["W12"] - learning_rate * grads["dW12"],
    "W2": parameters["W2"] - learning_rate*grads["dW2"],
    "b1": parameters["b1"] - learning_rate*grads["db1"],
    "b12": parameters["b12"] - learning_rate * grads["db12"],
    "b2": parameters["b2"] - learning_rate*grads["db2"],
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

def nn_model(X, Y, iterations,lr,nh,func,func_der):
    n_x=X.shape[0]
    n_h=nh  #change here for the targil
    n_y=1
    parameters = initialize_parameters(n_x,n_h,n_y)
    print("Network shape " , X.shape[0], n_h , n_y)
    for i in range(iterations):
        A2, cache = forward_propagation(X,parameters,func)
        cost = MSE_calculation(A2,Y)
        #cost = LogLoss_calculation(A2,Y)
        grads = backward_propagation(parameters,cache,X,Y,func_der)
        parameters = update_parameters(parameters,grads,lr)
        costs.append(cost)
        #cost check
        if i % 100 == 0:
            print (f"Cost after iteration {i}: {cost}")
    return parameters, costs

def predict(X, parameters,func):
    A2, cache = forward_propagation(X, parameters,func)
    return np.rint(A2)
    '''This round the values like:
    def predict(X, theta, threshold=0.5):
    if predict_probs(X, theta) >= threshold:
        return 1
        print(A)'''

def prediction_accuracy(y_pred,y_true):
    return np.mean(y_pred==y_true)

url = 'https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true'
#url = 'https://github.com/rosenfa/nn/blob/master/class2/spam.csv?raw=true'
df=pd.read_csv(url,  header=0, error_bad_lines=False)
features = df.drop(['Outcome'], axis = 1 )
features = ((features - features.mean())/features.std())
X = np.array(features)
Y = np.array(df['Outcome'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
df

X_train, X_test = X_train.T, X_test.T

parameters, costs = nn_model(X_train,Y_train,iterations,alpha,nh,relu,relu_der)
Y_train_predict = predict(X_train, parameters,relu)
train_acc = prediction_accuracy(Y_train_predict, Y_train)
Y_test_predict = predict(X_test, parameters,relu)
test_acc = prediction_accuracy(Y_test_predict, Y_test)

print("Training acc : ", str(train_acc))
print("Testing acc : ", str(test_acc))

plt.plot(costs)
plt.show()