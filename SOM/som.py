# Build an SOM (Self Organising Map) to detect fraud in credit card applications

WORKING_DIRECTORY = 'SOM'
DATA_DIRECTORY ='data'
TRAINING_DATA = 'Google_Stock_Price_Train.csv'
TEST_DATA = 'Google_Stock_Price_Test.csv'
MODEL_FILE = 'som.keras'


# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

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

##################################### DATA PREPROCESSING ##########################################
logging.info('Data preprocessing section entered')



# Check if model file exists
if os.path.isfile(os.path.join(data_dir, MODEL_FILE)):
    logging.info(f'Model file {MODEL_FILE} exists')
    logging.info('Loading model')
    regressor = load_model(os.path.join(data_dir, MODEL_FILE))
    logging.info('Model loaded')
else:
    # Model file does not exist, build the model
    logging.info(f'Model file {MODEL_FILE} does not exist')
    logging.info('Building model')

    #################################### BUILDING THE SOM ###########################################
   

####################################### MAKING PREDICTIONS ##########################################
logging.info('Prediction section entered')


####################################### VISUALISING THE RESULTS ##########################################
logging.info('Visualisation section entered')


############################################# FINISH ################################################
logging.info('End of program')

