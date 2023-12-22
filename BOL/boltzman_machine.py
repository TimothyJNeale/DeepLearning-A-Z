# Build an Boltzman machinr=e to carry out recomendations on a movie dataset

WORKING_DIRECTORY = 'BOL'
DATA_DIRECTORY ='data'
TRAINING_DATA = 'Credit_Card_Applications.csv'
MODEL_FILE = 'bolt.keras'  

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

# Check if model file exists
if os.path.isfile(os.path.join(data_dir, MODEL_FILE)):
    logging.info(f'Model file {MODEL_FILE} exists')
    logging.info('Loading model')
    bol = tf.keras.models.load_model(os.path.join(data_dir, MODEL_FILE))
    logging.info('Model loaded')
else:
    # Model file does not exist, build the model
    logging.info(f'Model file {MODEL_FILE} does not exist')
    logging.info('Building model')

    ##################################### DATA PREPROCESSING ##########################################
    logging.info('Data preprocessing section entered')


    ############################### TRAINING THE BOLTZMAN MACHINE ####################################
    logging.info('Training the Boltzman Machine')


    

    ################################ TRAINING THE BOLTZMAN MACHINE #######################################
    logging.info('Boltzman Machine training section entered')



    # # Save the model
    # bol.save(os.path.join(data_dir, MODEL_FILE))
    # logging.info(f'Model saved as {MODEL_FILE}')

#################################### MAKING RECOMENDATIONS #######################################

logging.info('MaKing Recomendations')




########################################### FINISH ###############################################
logging.info('End of program')

