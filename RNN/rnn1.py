# Build an recurrent neural network 

WORKING_DIRECTORY = 'RNN'
DATA_DIRECTORY ='data'
TRAINING_DATA_DIRECTORY = 'training_set'
TEST_DATA_DIRECTORY = 'test_set'
MODEL_FILE = 'rnn1.keras'
PREDICTION_FILE_1 = 'cat_or_dog_1.jpg'
PREDICTION_FILE_2 = 'cat_or_dog_2.jpg'

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import tensorflow as tf

# load environment variables from .env file
load_dotenv()

# Initialise logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG) # Supress debugging output from modules imported
#logging.disable(logging.CRITICAL) # Uncomment to disable all logging

########################################### START ################################################
logging.info('Start of program')
logging.info(f'Tensorflow version {tf.__version__}')

# Get the current directory
home = os.getcwd()
# Gwt the data directory
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(f'Data directory: {data_dir}')

#Get the test data directory
test_data_dir = os.path.join(data_dir, TEST_DATA_DIRECTORY)
logging.info(f'Test data directory: {test_data_dir}')

# Get the training data directory
training_data_dir = os.path.join(data_dir, TRAINING_DATA_DIRECTORY)
logging.info(f'Training data directory: {training_data_dir}')

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

    #################################### BUILDING THE RNN ###########################################
    logging.info('RNN build section entered')

    # Initialise the RNN
    rnn = tf.keras.models.Sequential()
  

    # save the model
    rnn.save(os.path.join(data_dir, MODEL_FILE))
    logging.info(f'Model saved as {MODEL_FILE}')

####################################### MAKING PREDICTIONS ##########################################
logging.info('Prediction section entered')





############################################# FINISH ################################################
logging.info('End of program')

