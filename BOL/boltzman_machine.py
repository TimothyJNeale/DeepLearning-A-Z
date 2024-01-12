# Build an Boltzman machinr=e to carry out recomendations on a movie dataset

WORKING_DIRECTORY = 'BOL'
DATA_DIRECTORY ='data'
MOVIE_DATA = 'movies.dat'
USER_DATA = 'users.dat'
RATINGS_DATA = 'ratings.dat'
MODEL_FILE = 'boltzman_model.pt'  
TRAINING_DATA = 'u1.base'
TEST_DATA = 'u1.test'

NUMBER_OF_HIDDEN_NODES = 100
BATCH_SIZE = 100
NuMBER_OF_EPOCHS = 10

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

    # Convert the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
    logging.info('Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)')

    # Replace all the zeros with -1
    training_set[training_set == 0] = -1
    test_set[test_set == 0] = -1

    # Replace all the ratings of 1 and 2 with 0 (Not Liked)
    training_set[training_set == 1] = 0
    training_set[training_set == 2] = 0
    test_set[test_set == 1] = 0
    test_set[test_set == 2] = 0

    # Replace all the ratings of 3, 4 and 5 with 1 (Liked)
    training_set[training_set >= 3] = 1
    test_set[test_set >= 3] = 1

    ################################# CREATING THE ARCHITECTURE ######################################
    #bol = None
    logging.info('Creating the architecture section entered')

    # create the class
    class RBM():
        def __init__(self, nv, nh):
            self.W = torch.randn(nh, nv)
            self.a = torch.randn(1, nh)
            self.b = torch.randn(1, nv)
        
        def sample_h(self, x):
            wx = torch.mm(x, self.W.t())
            activation = wx + self.a.expand_as(wx)
            p_h_given_v = torch.sigmoid(activation)
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        
        def sample_v(self, y):
            wy = torch.mm(y, self.W)
            activation = wy + self.b.expand_as(wy)
            p_v_given_h = torch.sigmoid(activation)
            return p_v_given_h, torch.bernoulli(p_v_given_h)

        def train(self, v0, vk, ph0, phk):
            self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
            self.b += torch.sum((v0 - vk), 0)
            self.a += torch.sum((ph0 - phk), 0)

    # Create the Boltzman Machine
    logging.info('Creating the Boltzman Machine')
    bol = RBM(num_movies, NUMBER_OF_HIDDEN_NODES)

    ############################### TRAINING THE BOLTZMAN MACHINE ####################################
    logging.info('Training the Boltzman Machine')

    for epoch in range(1, NuMBER_OF_EPOCHS + 1):
        train_loss = 0
        s = 0.0
        for id_user in range(0, num_users - BATCH_SIZE, BATCH_SIZE):
            vk = training_set[id_user:id_user + BATCH_SIZE]
            v0 = training_set[id_user:id_user + BATCH_SIZE]
            ph0,_ = bol.sample_h(v0)
            for k in range(10):
                _,hk = bol.sample_h(vk)
                _,vk = bol.sample_v(hk)
                vk[v0<0] = v0[v0<0] # Freeze the movies that were not rated (-1)
            phk,_ = bol.sample_h(vk)
            bol.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
            s += 1.
        logging.info(f'epoch: {epoch} loss: {train_loss/s}')



    # Save the model using PyTorch
    logging.info('Saving model')
    torch.save(bol, os.path.join(data_dir, MODEL_FILE))


#################################### MAKING RECOMENDATIONS #######################################

logging.info('Making Recomendations')




########################################### FINISH ###############################################
logging.info('End of program')

