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
logging.info('Visualising the results')
bone()
pcolor(som.distance_map().T) # distance_map() method returns all the mean interneuron distances in one 
                             # matrix and we take the transpose of it
colorbar()
markers = ['o', 's'] # o - circle, s - square
colors = ['r', 'g'] # r - red, g - green
for i, x in enumerate(X):
    w = som.winner(x) # returns the winning node for the customer x
    plot(w[0] + 0.5, # x coordinate of the winning node
         w[1] + 0.5, # y coordinate of the winning node
         markers[y[i]], # marker type
         markeredgecolor = colors[y[i]], # marker edge color
         markerfacecolor = 'None', # marker face color
         markersize = 10, # marker size
         markeredgewidth = 2) # marker edge width

show()

#################################### FINDING THE FRAUDS ##########################################
logging.info('Finding the frauds')

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)


########################################### FINISH ###############################################
logging.info('End of program')

