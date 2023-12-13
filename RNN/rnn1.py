# Build an recurrent neural network 
# A LSTM (Long Short Term Memory) RNN is used
# Build a model based on 5 years of Google stock prices

WORKING_DIRECTORY = 'RNN'
DATA_DIRECTORY ='data'
TRAINING_DATA = 'Google_Stock_Price_Train.csv'
TEST_DATA = 'Google_Stock_Price_Test.csv'
MODEL_FILE = 'regressor.keras'


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

# Load the training dataset
dataset_train = pd.read_csv(training_data_file)
logging.info(f'Training data shape: {dataset_train.shape}')
logging.info(f'Training data type: {type(dataset_train)}')

# Preprocess the training set
training_set = dataset_train.iloc[:, 1:2].values
logging.info(f'Training set shape: {training_set.shape}')
logging.info(f'Training set type: {type(training_set)}')

# Feature scaling
sc = MinMaxScaler(feature_range=(0,1), copy=True)
training_set_scaled = sc.fit_transform(training_set)
logging.info(f'Training set shape: {training_set_scaled.shape}')
logging.info(f'Training set type: {type(training_set_scaled)}')   

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

    #################################### BUILDING THE RNN ###########################################
    logging.info('RNN build section entered')

    # Creating the data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []

    for i in range(60, 1258):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    # Convert the lists to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    logging.info(f'X_train type: {type(X_train)}')
    logging.info(f'X_train shape: {X_train.shape}')
    logging.info(f'y_train type: {type(y_train)}')
    logging.info(f'y_train shape: {y_train.shape}')   
  
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    logging.info(f'X_train type: {type(X_train)}')
    logging.info(f'X_train shape: {X_train.shape}')
    logging.info(f'X_train.shape[1]: {X_train.shape[1]}')

    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(rate=0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(rate=0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(rate=0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(rate=0.2))

    # Adding the output layer
    regressor.add(Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs=100, batch_size=32)

    # save the model
    regressor.save(os.path.join(data_dir, MODEL_FILE))
    logging.info(f'Model saved as {MODEL_FILE}')

####################################### MAKING PREDICTIONS ##########################################
logging.info('Prediction section entered')

# Getting the real Google stock price from 2017
dataset_test = pd.read_csv(test_data_file)
logging.info(f'Test data shape: {dataset_test.shape}')

real_stock_price = dataset_test.iloc[:, 1:2].values
logging.info(f'Real stock price shape: {real_stock_price.shape}')
logging.info(f'Real stock price type: {type(real_stock_price)}')

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

# Get the inputs for the test data

# Get the last 60 days of the training data
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)

# Scale the inputs
inputs = sc.transform(inputs)

# Create the test data structure
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

# Convert the lists to numpy arrays
X_test = np.array(X_test)

# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the first 20 days of the test data
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

####################################### VISUALISING THE RESULTS ##########################################
logging.info('Visualisation section entered')

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

############################################# FINISH ################################################
logging.info('End of program')

