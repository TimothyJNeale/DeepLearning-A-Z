# Build an artifical neurak network to oredice custimer churn
# Data: Churn_Modelling.csv

WORKING_DIRECTORY = 'ANN'
DATA_DIRECTORY ='data'
DATA_FILE = 'Churn_Modelling.csv'
MODEL_FILE = 'customer_churn_prediction.keras'

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

########################################### START ################################################
logging.info('Start of program')

# Get the current DATA directory
home = os.getcwd()
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(data_dir)

working_dir = os.path.join(home, WORKING_DIRECTORY)
os.chdir(working_dir)
logging.info(os.getcwd())

##################################### DATA PREPROCESSING ##########################################
logging.info('Data preprocessing section entered')
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
logging.info(X.shape)

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


# Check if model file exists
if os.path.isfile(os.path.join(data_dir, MODEL_FILE)):
    logging.info(f'Model file {MODEL_FILE} exists')
    logging.info('Loading model')
    ann = tf.keras.models.load_model(os.path.join(data_dir, MODEL_FILE))
    logging.info('Model loaded')
else:
    # Model file does not exist, build the model

    #################################### BUILDING THE ANN ###########################################
    logging.info('ANN build section entered')

    # Initialise the ANN
    ann = tf.keras.models.Sequential()

    # Add the input layer and the first hidden layer
    # units = number of neurons in the layer
    # activation = activation function to use in the layer
    # input_dim = number of neurons in the input layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_dim=12))

    # Add the second hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    # Add the output layer
    # units = number of neurons in the layer
    # activation = activation function to use in the layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    #################################### TRAINING THE ANN ###########################################
    logging.info('ANN training section entered')

    # Complie the ann
    # optimizer = algorithm used to find the optimal set of weights in the ANN
    # loss = loss function within the optimizer algorithm
    # metrics = criteria used to evaluate the model
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the ANN on the training set
    # batch_size = number of observations after which you want to update the weights
    # epochs = number of times you want to train the ANN on the whole training set
    ann.fit(X_train, y_train, batch_size=32, epochs=100)

    # save the model
    ann.save(os.path.join(data_dir, MODEL_FILE))
    logging.info(f'Model saved as {MODEL_FILE}')

#################################### MAKING PREDICTIONS ###########################################
logging.info('Prediction section entered')

# Predict a single result
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40
# Tenure: 3
# Balance: $60000
# Number of Products: 2
# Has Credit Card: Yes
# Is Active Member: Yes
# Estimated Salary: $50000

# Scale the input data
X_sample = sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
logging.info(X_sample)

# Predict the result
y_pred = ann.predict(X_sample)
logging.info(y_pred[0][0])
logging.info(f'Customer will leave: {y_pred[0][0] > 0.5}')

# Predict the test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5) # convert probabilities to True/False

#  Make the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
logging.info(cm)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f'Accuracy score: {accuracy}')

######################################### FINISH ##############################################
logging.info('End of program')

