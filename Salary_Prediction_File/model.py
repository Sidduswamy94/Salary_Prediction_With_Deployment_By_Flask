# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter("ignore")

# Load the dataset.
dataset = pd.read_csv("hiring.csv")

dataset['experience'].fillna(0, inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# Create x and y variables.
X = dataset.iloc[:, :3]               # independent variable
y = dataset.iloc[:, -1]               # dependent variable

# Train the linear regression model
regressor = LinearRegression()        # save the model as "regressor"
regressor.fit(X, y)                   # Train the model with training sets

# Predict the model on given data
print('Before Loading:',regressor.predict([[4, 9, 7]]))

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print('After Loading:', model.predict([[4, 9, 7]]))
