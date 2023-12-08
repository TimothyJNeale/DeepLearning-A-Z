# Build an recurrent neural network 
# A LSTM (Long Short Term Memory) RNN is used
# Build a model based on 5 years of Google stock prices

WORKING_DIRECTORY = 'RNN'
DATA_DIRECTORY ='data'
TRAINING_DATA = 'Google_Stock_Price_Train.csv'
TEST_DATA = 'Google_Stock_Price_Test.csv'
MODEL_FILE = 'rnn1.keras'
PREDICTION_FILE_1 = 'cat_or_dog_1.jpg'
PREDICTION_FILE_2 = 'cat_or_dog_2.jpg'

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler


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

#Get the test data 
test_data_file = os.path.join(data_dir, TEST_DATA)
logging.info(f'Test data file: {test_data_file}')

# Get the training data
training_data_file = os.path.join(data_dir, TRAINING_DATA)
logging.info(f'Training data file: {training_data_file}')

# Get the working directory
working_dir = os.path.join(home, WORKING_DIRECTORY)
os.chdir(working_dir)
logging.info(f'Current directory: {os.getcwd()}')

# Check if model file exists
if os.path.isfile(os.path.join(data_dir, MODEL_FILE)):
    logging.info(f'Model file {MODEL_FILE} exists')
    logging.info('Loading model')
    rnn = tf.keras.models.load_model(os.path.join(data_dir, MODEL_FILE))
    logging.info('Model loaded')
else:
    # Model file does not exist, build the model
    logging.info(f'Model file {MODEL_FILE} does not exist')
    logging.info('Building model')

    ##################################### DATA PREPROCESSING ##########################################
    logging.info('Data preprocessing section entered')

    # Preprocess the training set
    dataset_train = pd.read_csv(training_data_file)
    logging.info(f'Training data shape: {dataset_train.shape}')

    training_set = dataset_train.iloc[:, 1:2].values
    logging.info(f'Training set shape: {training_set.shape}')
    logging.info(f'Training set type: {type(training_set)}')

    # Feature scaling
    sc = MinMaxScaler(feature_range=(0,1), copy=True)
    training_set_scaled = sc.fit_transform(training_set)
    logging.info(f'Training set shape: {training_set_scaled.shape}')
    logging.info(f'Training set type: {type(training_set_scaled)}')   

    #################################### BUILDING THE RNN ###########################################
    logging.info('RNN build section entered')

    # Creating the data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []

    for i in range(60, 1258):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    # Convert the lists to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    logging.info(f'X_train type: {type(X_train)}')
    logging.info(f'X_train shape: {X_train.shape}')
    logging.info(f'y_train type: {type(y_train)}')
    logging.info(f'y_train shape: {y_train.shape}')   
  
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    logging.info(f'X_train type: {type(X_train)}')
    logging.info(f'X_train shape: {X_train.shape}')


    # save the model
    # rnn.save(os.path.join(data_dir, MODEL_FILE))
    # logging.info(f'Model saved as {MODEL_FILE}')

####################################### MAKING PREDICTIONS ##########################################
logging.info('Prediction section entered')





############################################# FINISH ################################################
logging.info('End of program')

