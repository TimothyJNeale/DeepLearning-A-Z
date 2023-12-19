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

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
logging.info(f'dataset {dataset}')

# how many NanN values are there in the dataset
logging.info(f'NanN values in dataset {dataset.isna().sum()}')

# drop any rows where there is a NanN in the dependant variable
dataset = dataset.dropna(subset=['Purchased'])
logging.info(f'dataset {dataset}')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

logging.info(f'X {X}')
logging.info(f'y {y}')

# Missing data - replace with average values in the column
missing_ages = dataset.iloc[:, 1].isna().sum()  # Assuming '1' is the index of the column with ages
logging.info(f'Missing ages {missing_ages}')
missing_salaries = dataset.iloc[:, 2].isna().sum()  # Assuming '2' is the index of the column with salaries
logging.info(f'Missing salaries {missing_salaries}')

if missing_ages > 0:
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:2])
    X[:, 1:2] = imputer.transform(X[:, 1:2])
    logging.info(f'X {X}')

if missing_salaries > 0:
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 2:3])
    X[:, 2:3] = imputer.transform(X[:, 2:3])
    logging.info(f'X {X}')

logging.info(f'X {X}')

# Encoding categorical data
# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
logging.info(f'X {X}')



#################################### TRAINING THE MODEL ###########################################
logging.info('LOG training section entered')

# Splitting the dataset into the Training set and Test set
# 80% training, 20% test
# random_state = 0 to get the same results as the course
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1 )
logging.info(f'X_train {X_train}')
logging.info(f'X_test {X_test}')
logging.info(f'y_train {y_train}')
logging.info(f'y_test {y_test}')

# Feature Scaling
sc = StandardScaler()
X_train[:, 2:] = sc.fit_transform(X_train[:, 2:])
X_test[:, 2:] = sc.transform(X_test[:, 2:])
logging.info(f'X_train {X_train}')
logging.info(f'X_test {X_test}')

# Training the Logistic Regression model on the Training set





#################################### MAKING PREDICTIONS ###########################################
logging.info('Prediction section entered')



######################################### FINISH ##############################################
logging.info('End of program')
