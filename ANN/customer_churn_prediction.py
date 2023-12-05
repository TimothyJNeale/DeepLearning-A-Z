# Build an artifical neurak network to oredice custimer churn
# Data: Churn_Modelling.csv

WORKING_DIRECTORY = 'ANN'
DATA_DIRECTORY ='data'
DATA_FILE = 'Churn_Modelling.csv'

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

# load environment variables from .env file
load_dotenv()

# Initialise logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG) # Supress debugging output from modules imported
#logging.disable(logging.CRITICAL) # Uncomment to disable all logging

######################################### START ###############################################
logging.info('Start of program')

# Get the current DATA directory
home = os.getcwd()
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(data_dir)

working_dir = os.path.join(home, WORKING_DIRECTORY)
os.chdir(working_dir)
logging.info(os.getcwd())

########################################## MAIN ###############################################
logging.info('Main section entered')
logging.info(tf.__version__)

# import the dsts set
dataset = pd.read_csv(os.path.join(data_dir, DATA_FILE))
X = dataset.iloc[0, 3:-1].values # ignore the first three columns
y = dataset.iloc[0, -1].values

######################################### FINISH ##############################################
logging.info('End of program')