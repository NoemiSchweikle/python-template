from flask import Flask, Response, request
import pandas as pd
import numpy as np
import os
from flask_cors import CORS
import pickle

# Access the loaded objects
loaded_objects_path = 'data/models/saved_objects.pkl'
with open(loaded_objects_path, 'rb') as objects_file:
    loaded_objects = pickle.load(objects_file)

model = loaded_objects['model']
X_train_scaled = loaded_objects['X_train_scaled']
y_test = loaded_objects['y_test']
data = loaded_objects['data']
X_test_scaled = loaded_objects['X_test_scaled']

#get parameters
zylinder = 10
ps = 40
gewicht = 20
beschleunigung = 100
baujahr = 80

    #make prediction
x_test =[[zylinder, ps, gewicht, beschleunigung, baujahr]]
print(x_test)
prediction = model.predict(x_test)
print('predicted miles per gallon:', int(prediction.item()))

print("predition", prediction)

