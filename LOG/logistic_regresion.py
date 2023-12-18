# Build an model to perform logistic regresion

WORKING_DIRECTORY = 'LOG'
DATA_DIRECTORY ='data'
DATA_FILE = ''
MODEL_FILE = 'customer_churn_prediction.keras'

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import pandas as pd

# load environment variables from .env file
load_dotenv()

# Initialise logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG) # Supress debugging output from modules imported
#logging.disable(logging.CRITICAL) # Uncomment to disable all logging

########################################### START ################################################
logging.info('Start of program')

# Get the data directory
home = os.getcwd()
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(data_dir)

working_dir = os.path.join(home, WORKING_DIRECTORY)
os.chdir(working_dir)
logging.info(os.getcwd())

##################################### DATA PREPROCESSING ##########################################
logging.info('Data preprocessing section entered')


#################################### BUILDING THE MODEL ###########################################
logging.info('LOG build section entered')



#################################### TRAINING THE ANN ###########################################
logging.info('LOG training section entered')



#################################### MAKING PREDICTIONS ###########################################
logging.info('Prediction section entered')



######################################### FINISH ##############################################
logging.info('End of program')

