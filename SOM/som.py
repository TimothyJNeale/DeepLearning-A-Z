# Build an SOM (Self Organising Map) to detect fraud in credit card applications

WORKING_DIRECTORY = 'SOM'
DATA_DIRECTORY ='data'
TRAINING_DATA = 'Credit_Card_Applications.csv'

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

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


   

####################################### MAKING PREDICTIONS #######################################
logging.info('Prediction section entered')



########################################### FINISH ###############################################
logging.info('End of program')

