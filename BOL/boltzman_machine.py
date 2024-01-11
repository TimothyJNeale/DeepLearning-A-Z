# Build an Boltzman machinr=e to carry out recomendations on a movie dataset

WORKING_DIRECTORY = 'BOL'
DATA_DIRECTORY ='data'
MOVIE_DATA = 'movies.dat'
USER_DATA = 'users.dat'
RATINGS_DATA = 'ratings.dat'
MODEL_FILE = 'boltzman_model.pt'  
TRAINING_DATA = 'u1.base'
TEST_DATA = 'u1.test'

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import pandas as pd
import torch    # Pytorch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

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
logging.info(f'Current directory: {home}')

# Get the data directory
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(f'Data directory: {data_dir}')

# Get the training data
movie_data_file = os.path.join(data_dir, MOVIE_DATA)
logging.info(f'Movie data file: {movie_data_file}')
user_data_file = os.path.join(data_dir, USER_DATA)
logging.info(f'User data file: {user_data_file}')
ratings_data_file = os.path.join(data_dir, RATINGS_DATA)
logging.info(f'User data file: {ratings_data_file}')

# Get the training data
training_data_file = os.path.join(data_dir, TRAINING_DATA)
logging.info(f'Training data file: {training_data_file}')

# Get the test data
test_data_file = os.path.join(data_dir, TEST_DATA)
logging.info(f'Test data file: {test_data_file}')

# Get the working directory
working_dir = os.path.join(home, WORKING_DIRECTORY)
os.chdir(working_dir)
logging.info(f'Current directory: {os.getcwd()}')



# Check if model file exists
if os.path.isfile(os.path.join(data_dir, MODEL_FILE)):
    logging.info(f'Model file {MODEL_FILE} exists')
    logging.info('Loading model')
    # use PyTorch to load the model
    bol = torch.load(os.path.join(data_dir, MODEL_FILE))

    logging.info('Model loaded')
else:
    # Model file does not exist, build the model
    logging.info(f'Model file {MODEL_FILE} does not exist')
    bol= None

    logging.info('Building model')

    ##################################### DATA PREPROCESSING ##########################################
    logging.info('Data preprocessing section entered')

    # import the data sets
    logging.info('Importing the data')
    movies = pd.read_csv(movie_data_file, sep='::', header=None, engine='python', encoding='latin-1')
    users = pd.read_csv(user_data_file, sep='::', header=None, engine='python', encoding='latin-1')
    ratings = pd.read_csv(ratings_data_file, sep='::', header=None, engine='python', encoding='latin-1')

    # Import the training and test sets
    logging.info('Importing the training and test sets')
    training_set = pd.read_csv(training_data_file, delimiter='\t', header=None)
    training_set = np.array(training_set, dtype='int')

    test_set = pd.read_csv(test_data_file, delimiter='\t', header=None)
    test_set = np.array(test_set, dtype='int')

    # Get the number of users and movies
    logging.info('Getting the number of users and movies')
    num_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
    num_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))
    logging.info(f'Number of users: {num_users}')
    logging.info(f'Number of movies: {num_movies}')

    # Convert the data into an array with users in lines and movies in columns
    logging.info('Converting the data into an array with users in lines and movies in columns')
    def convert(data):
        new_data = []
        for id_users in range(1, num_users + 1):
            id_movies = data[:, 1][data[:, 0] == id_users]
            id_ratings = data[:, 2][data[:, 0] == id_users]
            ratings = np.zeros(num_movies)
            ratings[id_movies - 1] = id_ratings
            new_data.append(list(ratings))
        return new_data
    
    training_set = convert(training_set)
    test_set = convert(test_set)

    # Convert the data into Torch tensors
    logging.info('Converting the data into Torch tensors')
    training_set = torch.FloatTensor(training_set)
    test_set = torch.FloatTensor(test_set)
    
    

    ############################### TRAINING THE BOLTZMAN MACHINE ####################################
    logging.info('Training the Boltzman Machine')



    ################################ TRAINING THE BOLTZMAN MACHINE #######################################
    logging.info('Boltzman Machine training section entered')

    # Save the model using PyTorch
    # logging.info('Saving model')
    # torch.save(bol, os.path.join(data_dir, MODEL_FILE))


#################################### MAKING RECOMENDATIONS #######################################

logging.info('Making Recomendations')




########################################### FINISH ###############################################
logging.info('End of program')

