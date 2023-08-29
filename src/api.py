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

training_data_df = pd.read_csv("data/auto-mpg.csv", sep=';', skipinitialspace=True)
#print(training_data_df.head)

#print("shape trainings data", training_data_df.shape)

#start application
app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return {'hello visitor': 'welocme to this page'}

@app.route("/hello_world", methods = ['GET'])
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/training_data", methods = ['GET'])
def training_data():
    return Response(training_data_df.to_json(), mimetype='application/json')

# @app.route("/predict", methods = ['GET'])
# def predict_func():
#     #get parameters
#     zylinder = request.args.get ('zylinder')
#     ps = request.args.get('ps')
#     gewicht = request.args.get('gewicht')
#     beschleunigung = request.args.get('beschleunigung')
#     baujahr = request.args.get('baujahr')

#     #make prediction
#     prediction = model.predict([[zylinder, ps, gewicht, beschleunigung, baujahr]])
#     return {'predicted miles per gallon:': int(prediction.item())}

@app.route("/predict", methods = ['GET'])
def predict_func():
    #get parameters
    zylinder = request.args.get ('zylinder')
    ps = request.args.get('ps')
    gewicht = request.args.get('gewicht')
    beschleunigung = request.args.get('beschleunigung')
    baujahr = request.args.get('baujahr')

    #build a dataframe for an unknown instance
    instance = pd.DataFrame ({
        'zylinder': zylinder,
        'ps': ps,
        'gewicht' : gewicht,
        'beschleunigung' : beschleunigung,
        'baujahr' : baujahr},
        index=[0])

    #check GET parameters
    if (zylinder and ps and gewicht and beschleunigung and baujahr):
        #load model and make prediction
        prediction = model.predict(instance)
        return {'result': int(prediction.item())}
    else:
        return Response('One or more GET parameters is missing', mimetype = 'application/json')


# cwd = os.getcwd()  # Aktuelles Arbeitsverzeichnis abrufen
# print("Current Working Directory:", cwd)


