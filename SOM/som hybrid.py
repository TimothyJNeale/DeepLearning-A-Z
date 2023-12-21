# Build an SOM (Self Organising Map) to detect fraud in credit card applications

WORKING_DIRECTORY = 'SOM'
DATA_DIRECTORY ='data'
TRAINING_DATA = 'Credit_Card_Applications.csv'
MODEL_FILE = 'som_hybrid.keras'  

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# load environment variables from .env file
load_dotenv()

# Initialise logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG) # Supress debugging output from modules imported
#logging.disable(logging.CRITICAL) # Uncomment to disable all logging

########################################### START ################################################
logging.info('Start of program')

# Get the current directory
home = os.getcwd()
# Gwt the data directory
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(f'Data directory: {data_dir}')

# Get the training data
training_data_file = os.path.join(data_dir, TRAINING_DATA)
logging.info(f'Training data file: {training_data_file}')

# Get the working directory
working_dir = os.path.join(home, WORKING_DIRECTORY)
os.chdir(working_dir)
logging.info(f'Current directory: {os.getcwd()}')

#################################### DATA PREPROCESSING ########################################
logging.info('Data preprocessing section entered')

# Load the data
dataset = pd.read_csv(training_data_file)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature scaling
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#################################### TRAINING THE SOM ###########################################
logging.info('Training the SOM')

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#################################### VISUALISING THE RESULTS ####################################
logging.info('Visualising the results')
bone()
pcolor(som.distance_map().T) # distance_map() method returns all the mean interneuron distances in one 
                             # matrix and we take the transpose of it
colorbar()
markers = ['o', 's'] # o - circle, s - square
colors = ['r', 'g'] # r - red, g - green
for i, x in enumerate(X):
    w = som.winner(x) # returns the winning node for the customer x
    plot(w[0] + 0.5, # x coordinate of the winning node
         w[1] + 0.5, # y coordinate of the winning node
         markers[y[i]], # marker type
         markeredgecolor = colors[y[i]], # marker edge color
         markerfacecolor = 'None', # marker face color
         markersize = 10, # marker size
         markeredgewidth = 2) # marker edge width

show()

#################################### FINDING THE FRAUDS ##########################################
logging.info('Finding the frauds')

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,5)],mappings[(3,3)], mappings[(5,3)], mappings[(8,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)

logging.info(f'Frauds {frauds}')

################################## CREATE A SUPERVISED MODEL #####################################
logging.info('Building the dupervised model')

# Create the matrix of features
customers = dataset.iloc[:, 1:].values

# Create the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

logging.info(f'is_fraud: {is_fraud}')
logging.info(f'shape of vector: {is_fraud.shape}')
logging.info(f'Number of frauds: {sum(is_fraud)}')
logging.info(f'Number of non-frauds: {len(is_fraud) - sum(is_fraud)}')
logging.info(f'Percentage of frauds: {sum(is_fraud) / len(is_fraud)}')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Scale the training set
customers = sc.fit_transform(customers)

# Check if model file exists
if os.path.isfile(os.path.join(data_dir, MODEL_FILE)):
    logging.info(f'Model file {MODEL_FILE} exists')
    logging.info('Loading model')
    ann = tf.keras.models.load_model(os.path.join(data_dir, MODEL_FILE))
    logging.info('Model loaded')
else:
    # Model file does not exist, build the model

    #################################### BUILDING THE ANN ###########################################
    logging.info('ANN build section entered')

    # Initialise the ANN
    ann = tf.keras.models.Sequential()

    # Add the input layer and the first hidden layer
    # units = number of neurons in the layer
    # activation = activation function to use in the layer
    # input_dim = number of neurons in the input layer
    ann.add(tf.keras.layers.Dense(units=2, activation='relu', input_dim=15))


    # Add the output layer
    # units = number of neurons in the layer
    # activation = activation function to use in the layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    #################################### TRAINING THE ANN ###########################################
    logging.info('ANN training section entered')

    # Complie the ann
    # optimizer = algorithm used to find the optimal set of weights in the ANN
    # loss = loss function within the optimizer algorithm
    # metrics = criteria used to evaluate the model
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the ANN on the training set
    # batch_size = number of observations after which you want to update the weights
    # epochs = number of times you want to train the ANN on the whole training set
    ann.fit(customers, is_fraud, batch_size=1, epochs=2)

    # save the model
    ann.save(os.path.join(data_dir, MODEL_FILE))
    logging.info(f'Model saved as {MODEL_FILE}')

#################################### MAKING PREDICTIONS ###########################################
logging.info('Prediction section entered')

# Predict the test set results
y_pred = ann.predict(customers)
logging.info(f'y_pred: {y_pred}')

# Concatenate the customer ID with the prediction
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
logging.info(f'y_pred: {y_pred}')
logging.info(f'y_pred.shape: {y_pred.shape}')

# Sort the array in descending order of probability of fraud
y_pred = y_pred[y_pred[:, 1].argsort()[::-1]]
logging.info(f'y_pred: {y_pred}')

########################################### FINISH ###############################################
logging.info('End of program')

