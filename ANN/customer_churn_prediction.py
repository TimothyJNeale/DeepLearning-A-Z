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

# import the data set
dataset = pd.read_csv(os.path.join(data_dir, DATA_FILE))
X = dataset.iloc[:, 3:-1].values # ignore the first three columns, all rows
y = dataset.iloc[:, -1].values # ignore all but the last column, all rows

# Encode categorical data
logging.info(type(X))
logging.info(X[0:6, :])

# Label encoding 'Gender' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
logging.info(type(X))
logging.info(X[0:6, :])

# One hot encoding 'Geography' column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
logging.info(type(X))
logging.info(X[0:6, :])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# 80% of the data will be used for training, 20% for testing
# random_state = 1 means the same random split will be used each time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Scale the training set
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


######################################### FINISH ##############################################
logging.info('End of program')