# Build an model to perform logistic regresion on social network add data

WORKING_DIRECTORY = 'LOG'
DATA_DIRECTORY ='data'
DATA_FILE = 'Social_Network_Ads.csv'

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import matplotlib.pyplot as plt
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

# Import the dataset
dataset = pd.read_csv(os.path.join(data_dir, DATA_FILE))


#################################### TRAINING THE MODEL ###########################################
logging.info('LOG training section entered')





#################################### MAKING PREDICTIONS ###########################################
logging.info('Prediction section entered')



######################################### FINISH ##############################################
logging.info('End of program')

