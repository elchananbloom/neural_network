# -*- coding: utf-8 -*-
"""Keras_First_NN_HW with momentum.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VJUDFOIAu1gEWn8oqWHSerURyXFiW5Qz

## Using Keras to Build and Train Neural Networks

This exercise is based on Intel's course at: https://software.intel.com/content/dam/develop/public/us/en/downloads/intel-dl101-class5.zip and at: https://software.intel.com/content/www/us/en/develop/training/course-deep-learning.html. 

We will use a neural network to predict diabetes using the Pima Diabetes Dataset.  We will start by training a Random Forest to get a performance baseline.  Then we will use the Keras package to quickly build and train a neural network and compare the performance.  We will see how different network structures affect the performance, training time, and level of overfitting (or underfitting).

## UCI Pima Diabetes Dataset

* UCI ML Repositiory (http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)


### Attributes: (all numeric-valued)
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)

The UCI Pima Diabetes Dataset which has 8 numerical predictors and a binary outcome.
"""

# Commented out IPython magic to ensure Python compatibility.
#Preliminaries

from __future__ import absolute_import, division, print_function  # Python 2/3 compatibility

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

# %matplotlib inline

## Import Keras objects for Deep Learning

from keras.models  import Sequential
#from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
#from keras.optimizers import Adam, SGD, RMSprop

## Load in the data set (Internet Access needed)
url = 'https://github.com/rosenfa/nn/blob/master/pima-indians-diabetes.csv?raw=true'
diabetes_df=pd.read_csv(url,  header=0, error_bad_lines=False)

# Take a peek at the data -- if there are lots of "NaN" you may have internet connectivity issues
print(diabetes_df.shape)
diabetes_df.sample(5)

X = np.asarray(diabetes_df.drop('Outcome',1))
y = np.asarray(diabetes_df['Outcome'])

# Split the data to Train, and Test (75%, 25%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11111)

np.mean(y), np.mean(1-y)

"""Above, we see that about 35% of the patients in this dataset have diabetes, while 65% do not.  This means we can get an accuracy of 65% without any model - just declare that no one has diabetes. We will calculate the ROC-AUC score to evaluate performance of our model, and also look at the accuracy as well to see if we improved upon the 65% accuracy.
## Exercise: Get a baseline performance using Random Forest
To begin, and get a baseline for classifier performance:
1. Train a Random Forest model with 200 trees on the training data.
2. Calculate the accuracy and roc_auc_score of the predictions.
"""

## Train the RF Model
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X_train, y_train)

# Make predictions on the test set - both "hard" predictions, and the scores (percent of trees voting yes)
y_pred_class_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)


print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_rf)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_rf[:,1])))

def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} on PIMA diabetes problem'.format(model_name),
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])


plot_roc(y_test, y_pred_prob_rf[:, 1], 'RF')

"""## Build a Single Hidden Layer Neural Network

We will use the Sequential model to quickly build a neural network.  Our first network will be a single layer network.  We have 8 variables, so we set the input shape to 8.  Let's start by having a single hidden layer with 12 nodes.
"""

## First let's normalize the data
## This aids the training of neural nets by providing numerical stability
## Random Forest does not need this as it finds a split only, as opposed to performing matrix multiplications


normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

"""Keras has many shapes.  See: https://keras.io/api/layers/core_layers/dense/ for information about "dense" and sequential means a "basic" ordering (as opposed to more complicated ones we'll learn about starting next week)."""

# Define the Model 
# Input size is 8-dimensional
# 1 hidden layer, 12 hidden nodes, sigmoid activation
# Final layer has just one node with a sigmoid activation (standard for binary classification)

model_1 = Sequential([
    Dense(12, input_shape=(8,), activation="relu"),
    Dense(1, activation="sigmoid")
])

#  This is a nice tool to view the model you have created and count the parameters

model_1.summary()

"""### Comprehension question:
Why do we have 121 parameters?  Does that make sense?
Think about why there are 108 parameters in first layer when only 8 features exist and the network has 12 nodes (8*12 is NOT 108), and why the output layer has 13 values for only 12 nodes.


Let's fit our model for 200 epochs.
"""

# Fit(Train) the Model
from tensorflow.keras.optimizers import SGD
# Compile the model with Optimizer, Loss Function and Metrics
# Roc-Auc is not available in Keras as an off the shelf metric yet, so we will skip it here.

model_1.compile(SGD(lr = .003, momentum=0.8), "binary_crossentropy", metrics=["accuracy"])
run_hist_1 = model_1.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=50)
# the fit function returns the run history. 
# It is very convenient, as it contains information about the model fit, iterations etc.

## Like we did for the Random Forest, we generate two kinds of predictions
#  One is a hard decision, the other is a probabilitistic score.

y_pred_prob_nn_1 = model_1.predict(X_test_norm)
y_pred_class_nn_1 = np.rint(y_pred_prob_nn_1)

y_pred_prob_nn_1[:10]

# Let's check out the outputs to get a feel for how keras apis work.
y_pred_class_nn_1[:10]

# Print model performance and plot the roc curve
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_1)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_1)))

plot_roc(y_test, y_pred_prob_nn_1, 'NN')

"""There may be some variation in exact numbers due to randomness, but you should get results similar to the Random Forest - between 75% and 85% accuracy, between .8 and .9 for AUC.

Let's look at the `run_hist_1` object that was created, specifically its `history` attribute.
"""

run_hist_1.history.keys()

"""Let's plot the training loss and the validation loss over the different epochs and see how it looks."""

fig, ax = plt.subplots()
ax.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()

"""Looks like the losses are still going down on both the training set and the validation set.  This suggests that the model might benefit from further training.  Let's train the model a little more and see what happens. Note that it will pick up from where it left off. Train for 450 more epochs."""

## Note that when we call "fit" again, it picks up where it left off
run_hist_1b = model_1.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=450)

n = len(run_hist_1.history["loss"])
m = len(run_hist_1b.history['loss'])
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(range(n), run_hist_1.history["loss"],'r', marker='.', label="Train Loss - Run 1")
ax.plot(range(n, n+m), run_hist_1b.history["loss"], 'hotpink', marker='.', label="Train Loss - Run 2")

ax.plot(range(n), run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss - Run 1")
ax.plot(range(n, n+m), run_hist_1b.history["val_loss"], 'LightSkyBlue', marker='.',  label="Validation Loss - Run 2")

ax.legend()

"""Note that this graph begins where the other left off.  While the training loss is still going down, it looks like the validation loss has stabilized (or even gotten worse!).  This suggests that our network will not benefit from further training.  What is the appropriate number of epochs?"""

y_pred_prob_nn_2 = model_1.predict(X_test_norm)
y_pred_class_nn_2 = np.rint(y_pred_prob_nn_2)
# Print model performance and plot the roc curve
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_2)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_2)))

plot_roc(y_test, y_pred_prob_nn_2, 'NN')



"""## Exercise
Now it's your turn.  Do the following in the cells below:
- Build a model, model_2, with two hidden layers, each with 6 nodes
- Use the "relu" activation function for the hidden layers, and "sigmoid" for the final layer
- Use a learning rate of .003 and train for 100 epochs
- Graph the trajectory of the loss functions, accuracy on both train and test set
- Plot the roc curve for the predictions

You might want to look at the Keras documentation at: 
https://keras.io/guides/sequential_model/

Experiment with one network with 3 layers and save it as model_3. Did it work better?
Did using more or less epochs help?
Trying different learning rates for model_3.  Did that work better?
"""

# Type your code here for model_2 with layers 1,2 having 6 nodes and activation relu and layer 3 with activation sigmoid
# Define the Model 
# Input size is 8-dimensional
model_2 = Sequential([
    Dense(6, input_shape=(8,), activation="relu"),
    Dense(6, input_shape=(6,), activation="relu"),
    Dense(1, activation="sigmoid")
])
model_2.compile(SGD(lr = .003, momentum=0.8), "binary_crossentropy", metrics=["accuracy"])
run_hist_2 = model_2.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=100)




# Graph the trajectory of the loss functions and accuracy on both train and test set for model_2
fig1, ax1 = plt.subplots()
ax1.plot(run_hist_2.history["loss"],'r', marker='.', label="Train Loss")
ax1.plot(run_hist_2.history["val_loss"],'b', marker='.', label="Validation Loss")
ax1.legend()
plt.show()
# Type your code here to plot the loss, accuracy and ROC curve for model_2
y_pred_prob_nn_3 = model_2.predict(X_test_norm)
y_pred_class_nn_3 = np.rint(y_pred_prob_nn_3)
plot_roc(y_test, y_pred_prob_nn_3, 'NN')
# Type your code here for model_3 with layers 1,2,3 having activation relu and you pick the number of nodes in each layer (not the same)
# and layer 4 with activation sigmoid
# Define the Model
model_3 = Sequential([
    Dense(12, input_shape=(8,), activation="relu"),
    Dense(6, input_shape=(12,), activation="relu"),
    Dense(3, input_shape=(6,), activation="relu"),
    Dense(1, activation="sigmoid")
])
list_range=np.arange(0.001,0.009,0.001)
list_range+=np.arange(0.01,0.09,0.01)
list_range+=np.arange(0.1,0.9,0.1)
for i in list_range:
    model_3.compile(SGD(lr = i, momentum=0.8), "binary_crossentropy", metrics=["accuracy"])
    run_hist_3 = model_3.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=500)

    # Type your code here to plot the loss, accuracy and ROC curve for model_3
    fig2, ax2 = plt.subplots()
    ax2.plot(run_hist_3.history["loss"],'r', marker='.', label="Train Loss")
    ax2.plot(run_hist_3.history["val_loss"],'b', marker='.', label="Validation Loss")
    ax2.legend()
    plt.show()
    y_pred_prob_nn_4 = model_3.predict(X_test_norm)
    y_pred_class_nn_4 = np.rint(y_pred_prob_nn_4)
    plot_roc(y_test, y_pred_prob_nn_4, 'NN')
# Try using more or less epochs and different learning rates for model_3
