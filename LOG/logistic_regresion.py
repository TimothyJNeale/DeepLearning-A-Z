# Build an model to perform logistic regresion

WORKING_DIRECTORY = 'LOG'
DATA_DIRECTORY ='data'
DATA_FILE = 'Data.csv'
MODEL_FILE = 'customer_churn_prediction.keras'

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

logging.info(f'X {X}')
logging.info(f'y {y}')

# Missing data - replace with average values in the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
logging.info(f'X {X}')

# Encoding categorical data
# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
logging.info(f'X {X}')

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
logging.info(f'y {y}')




#################################### TRAINING THE MODEL ###########################################
logging.info('LOG training section entered')



#################################### MAKING PREDICTIONS ###########################################
logging.info('Prediction section entered')



######################################### FINISH ##############################################
logging.info('End of program')

