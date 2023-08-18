import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

# Load the saved objects using pickle from the 'data/models' folder
loaded_objects_path = 'data/models/saved_objects.pkl'
with open(loaded_objects_path, 'rb') as objects_file:
    loaded_objects = pickle.load(objects_file)

# Access the loaded objects
model = loaded_objects['model']
X = loaded_objects['X']
y = loaded_objects['y']
scaler = loaded_objects['scaler']
X_test = loaded_objects['X_test']


y_pred = model.predict(X_test_scaled)
print("y_pred", y_pred)


