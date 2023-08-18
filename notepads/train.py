import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

# Load the CSV file into a pandas DataFrame

data = pd.read_csv("data/auto-mpg.csv", sep=';', skipinitialspace=True)
data.dtypes

# Display the first few rows of the DataFrame
print(data.head())

y = data.loc[:,'mpg']
X = data.drop('mpg',axis=1)

#Train-test-split
X_train

#normalize data
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_train)




#Linear Regression
model = LinearRegression()
model.fit(X_scaled,y)

# save the model to disk
# Create a dictionary to hold the objects
saved_objects = {
    'model': model,
    'X_test': X_test
    'X_scaled': X_train_scaled
    'scaler': scaler}

# Save the dictionary with all objects to a file using pickle in the 'data/models' folder
saved_objects_path = 'data/models/saved_objects.pkl'
with open(saved_objects_path, 'wb') as objects_file:
    pickle.dump(saved_objects, objects_file)



