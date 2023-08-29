import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

# Load the CSV file into a pandas DataFrame

data = pd.read_csv("data/auto-mpg.csv", sep=';', skipinitialspace=True)
data.dtypes
print("data dtype")
print(data.dtypes)

# Display the first few rows of the DataFrame
print("head")
print(data.head())
print("shape", data.shape)
print("type", type(data))

y = data.loc[:,'mpg']
X = data.drop('mpg',axis=1)

# print("X", X.shape)
# print("y", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)

# print("X_train", X_train.shape)
# print("y_train", y_train.shape)


#normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_model = model.predict(X_test)
print("y_pred_model", y_pred_model)

# save the model to disk
# Create a dictionary to hold the objects
saved_objects = {
    'model': model,
    'X_test': X_test,
    'y_test': y_test,
    'data': data,
    'X_test_scaled': X_test_scaled}

# Save the dictionary with all objects to a file using pickle in the 'data/models' folder
saved_objects_path = 'data/models/saved_objects.pkl'
with open(saved_objects_path, 'wb') as objects_file:
    pickle.dump(saved_objects, objects_file)
