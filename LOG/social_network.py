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

from sklearn.metrics import confusion_matrix, accuracy_score

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
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#################################### MAKING PREDICTIONS ###########################################
logging.info('Prediction section entered')

# Predicting a new result
# Predict if a 30 year old male, with a salary of $87000 will buy the product
# 30,87000
# Scale the data
X_new = sc.transform([[30, 87000]])
# add the on hot variables
X_new = np.concatenate((np.array([[1.0, 0.0]]), X_new), axis=1)
logging.info(f'X_new {X_new}')
y_pred = classifier.predict(X_new)
logging.info(f'y_pred {y_pred}')

# Predict if a 39 year old female, with a salary of $122000 will buy the product
# 122,000

# Predicting the Test set results
X_new = sc.transform([[39, 122000]])
# add the on hot variables
X_new = np.concatenate((np.array([[0.0, 1.0]]), X_new), axis=1)
logging.info(f'X_new {X_new}')
y_pred = classifier.predict(X_new)
logging.info(f'y_pred {y_pred}')

y_pred = classifier.predict(X_test)
# logging.info(f'y_pred {y_pred}')
# logging.info(f'y_test {y_test}')
logging.info(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
logging.info(f'cm {cm}')
accuracy = accuracy_score(y_test, y_pred)
logging.info(f'accuracy {accuracy}')


#################################### VISUALISING THE RESULTS ###########################################
# Drop the gender columns and retrain the model
X_train = X_train[:, 2:]
X_test = X_test[:, 2:]
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Visualising the Training set results
from matplotlib.colors import ListedColormap


# Prepare the data for the training set plot
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))

# Create the subplot for the training set
plt.subplot(1, 2, 1)
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red','green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Prepare the data for the test set plot
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))

# Create the subplot for the test set
plt.subplot(1, 2, 2)
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red','green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Display the plots
plt.show()

######################################### FINISH ##############################################
logging.info('End of program')

