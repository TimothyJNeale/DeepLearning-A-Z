# Build an convolutional neural network 

WORKING_DIRECTORY = 'CNN'
DATA_DIRECTORY ='data'
TRAINING_DATA_DIRECTORY = 'training_set'
TEST_DATA_DIRECTORY = 'test_set'
MODEL_FILE = 'cat_or_dog.keras'
PREDICTION_FILE_1 = 'cat_or_dog_1.jpg'
PREDICTION_FILE_2 = 'cat_or_dog_2.jpg'

# Import libraries
import os
from dotenv import load_dotenv
import logging

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# load environment variables from .env file
load_dotenv()

# Initialise logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG) # Supress debugging output from modules imported
#logging.disable(logging.CRITICAL) # Uncomment to disable all logging

########################################### START ################################################
logging.info('Start of program')
logging.info(f'Tensorflow version {tf.__version__}')

# Get the current directory
home = os.getcwd()
# Gwt the data directory
data_dir = os.path.join(home, DATA_DIRECTORY)
logging.info(f'Data directory: {data_dir}')

#Get the test data directory
test_data_dir = os.path.join(data_dir, TEST_DATA_DIRECTORY)
logging.info(f'Test data directory: {test_data_dir}')

# Get the training data directory
training_data_dir = os.path.join(data_dir, TRAINING_DATA_DIRECTORY)
logging.info(f'Training data directory: {training_data_dir}')

# Get the working directory
working_dir = os.path.join(home, WORKING_DIRECTORY)
os.chdir(working_dir)
logging.info(f'Current directory: {os.getcwd()}')

# Check if model file exists
if os.path.isfile(os.path.join(data_dir, MODEL_FILE)):
    logging.info(f'Model file {MODEL_FILE} exists')
    logging.info('Loading model')
    cnn = tf.keras.models.load_model(os.path.join(data_dir, MODEL_FILE))
    logging.info('Model loaded')
else:
    # Model file does not exist, build the model
    logging.info(f'Model file {MODEL_FILE} does not exist')
    logging.info('Building model')

    ##################################### DATA PREPROCESSING ##########################################
    logging.info('Data preprocessing section entered')

    # preprocess the training set
    # apply transformations to the images to prevent overfitting
    # rescale = rescale the pixel values to be between 0 and 1
    # shear_range = apply random transformations
    # zoom_range = apply random zoom
    # horizontal_flip = flip the images horizontally
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(training_data_dir, target_size=(64, 64), batch_size=32, class_mode='binary')

    # preprocess the test set
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(test_data_dir, target_size=(64, 64), batch_size=32, class_mode='binary')


    #################################### BUILDING THE CNN ###########################################
    logging.info('CNN build section entered')

    # Initialise the CNN
    cnn = tf.keras.models.Sequential()

    # Convolution
    # 32 = number of feature detectors
    # 3, 3 = number of rows and columns in the feature detector
    # input_shape = shape of the input image (3 = number of channels, 64 = number of rows, 64 = number of columns)
    # activation = activation function
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    # Pooling
    # pool_size = dimensions of the pooling filter
    # strides = number of pixels the filter moves each time
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Add a second convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Full connection
    # units = number of neurons in the hidden layer
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Output layer
    # units = number of neurons in the output layer
    # activation = activation function
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    #################################### TRAINING THE CNN ###########################################
    logging.info('CNN training section entered')

    # Compile the CNN
    # optimizer = algorithm used to find the optimal set of weights in the CNN
    # loss = loss function within the optimizer algorithm
    # metrics = criteria used to evaluate the model
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the CNN on the training set and evaluate it on the test set
    # steps_per_epoch = number of images in the training set
    # epochs = number of times you want to train the CNN on the whole training set
    # validation_data = test set on which the accuracy of the CNN will be evaluated after each epoch
    # validation_steps = number of images in the test set
    cnn.fit(x=training_set, validation_data=test_set, epochs=25)

    # save the model
    cnn.save(os.path.join(data_dir, MODEL_FILE))
    logging.info(f'Model saved as {MODEL_FILE}')

####################################### MAKING PREDICTIONS ##########################################
logging.info('Prediction section entered')

# Get the first test image
test_image = image.load_img(os.path.join(data_dir, PREDICTION_FILE_1), target_size=(64, 64))
test_image = image.img_to_array(test_image)
# Add an extra dimension to the image
# This is because the predict method expects a batch and batch is the first dimension
test_image = np.expand_dims(test_image, axis=0)
# Predict the result
result = cnn.predict(test_image/255.0)
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

logging.info(f'Prediction for {PREDICTION_FILE_1}: {prediction}')

# Get the second test image
test_image = image.load_img(os.path.join(data_dir, PREDICTION_FILE_2), target_size=(64, 64))
test_image = image.img_to_array(test_image)
# Add an extra dimension to the image
# This is because the predict method expects a batch and batch is the first dimension
test_image = np.expand_dims(test_image, axis=0)
# Predict the result
result = cnn.predict(test_image/255.0)
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

logging.info(f'Prediction for {PREDICTION_FILE_2}: {prediction}')




############################################# FINISH ################################################
logging.info('End of program')

