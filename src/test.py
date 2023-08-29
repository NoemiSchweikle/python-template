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
X_test_scaled = loaded_objects['X_test_scaled']
y_test = loaded_objects['y_test']
data = loaded_objects['data']
X_test = loaded_objects['X_test']

#get parameters
zylinder = 6
ps = 133
gewicht = 3410
beschleunigung = 15.8
baujahr = 78

    #make prediction
X_check =[[zylinder, ps, gewicht, beschleunigung, baujahr]]
print(X_check)
prediction = model.predict(X_check)
print('predicted miles per gallon:', int(prediction.item()))

print("predition", prediction)
