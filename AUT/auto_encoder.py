# Build an auto encoder to carry out recomendations on a movie dataset

WORKING_DIRECTORY = 'BOL'
DATA_DIRECTORY ='data'
MOVIE_DATA = 'movies.dat'
USER_DATA = 'users.dat'
RATINGS_DATA = 'ratings.dat' 
TRAINING_DATA = 'u1.base'
TEST_DATA = 'u1.test'

NUMBER_OF_HIDDEN_NODES1 = 20
NUMBER_OF_HIDDEN_NODES2 = 10
BATCH_SIZE = 100
NUMBER_OF_EPOCHS = 200

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

################################# CREATING THE ARCHITECTURE ######################################
#bol = None
logging.info('Creating the architecture section entered')

# create the auto encoder class
class SAE(nn.Module):
    def __init__(self, num_movies, num_hidden_nodes1, num_hidden_nodes2):
        super(SAE, self).__init__()
        #code
        self.fc1 = nn.Linear(num_movies, num_hidden_nodes1)
        self.fc2 = nn.Linear(num_hidden_nodes1, num_hidden_nodes2)
        #decode
        self.fc3 = nn.Linear(num_hidden_nodes2, num_hidden_nodes1)
        self.fc4 = nn.Linear(num_hidden_nodes1, num_movies)

        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        #code
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        #decode
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

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

##################################### CREATE THE AUTO ENCODER ####################################
logging.info('Creating the Auto Encoder')

sae = SAE(num_movies, NUMBER_OF_HIDDEN_NODES1, NUMBER_OF_HIDDEN_NODES2)
# Loss function
criterion = nn.MSELoss()
# optimizer
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

############################### TRAINING THE AUTO ENCODER ####################################
logging.info('Training the Auto Encoder')
for epoch in range(1, NUMBER_OF_EPOCHS + 1):
    train_loss = 0
    s = 0.
    for id_user in range(num_users):
        # Create a fake batch of size 1
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        # Check if user has rated any movies
        if torch.sum(target.data > 0) > 0:
            # Get the output
            output = sae(input)
            # Set the output to 0 for movies not rated
            target.require_grad = False
            output[target == 0] = 0
            # Compute the loss
            loss = criterion(output, target)
            mean_corrector = num_movies/float(torch.sum(target.data > 0) + 1e-10)
            # Compute the gradient
            loss.backward()
            # Update the loss
            train_loss += np.sqrt(loss.data * mean_corrector)
            # Update the number of users
            s += 1.
            # Update the weights
            optimizer.step()
    
    logging.info(f'Epoch: {str(epoch)} Loss: {str(train_loss/s)}')



####################################### RUN ON TEST DATA ##########################################
logging.info('Run on test data section entered')

test_loss = 0
s = 0.
for id_user in range(num_users):
    # Create a fake batch of size 1
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    # Check if user has rated any movies
    if torch.sum(target.data > 0) > 0:
        # Get the output
        output = sae(input)
        target.require_grad = False
        # Set the output to 0 for movies not rated
        output[target == 0] = 0
        # Compute the loss
        loss = criterion(output, target)
        mean_corrector = num_movies/float(torch.sum(target.data > 0) + 1e-10)
        # Update the loss
        test_loss += np.sqrt(loss.data * mean_corrector)
        # Update the number of users who rated at least one movie
        s += 1.
        # Update the weights
        optimizer.step()

logging.info(f'Test loss: {str(test_loss/s)}')



########################################### FINISH ###############################################
logging.info('End of program')

