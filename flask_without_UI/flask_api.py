#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:11:00 2020

@author: varunsagotra
"""
### Test api using Postman

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import http

app = Flask(__name__)

# Load pickle model
pickle_in = open("currency_notes_authentication.pkl",'rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "welcome all"

@app.route('/predict', methods = ["GET"]) 
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy  = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness,curtosis,entropy]])
    return 'The predicted values is ' + str(prediction)

@app.route('/predict_file',methods = ["POST"])
def predict_note_authentication_via_file():
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))
    
    
    

if __name__ == '__main__':
    app.run()
# Note : UI App : browser link : http://127.0.0.1:5000/
# http://127.0.0.1:5000/predict?variance=2&skewness=1&curtosis=0&entropy=1   >>> Kindly change values to see desired result    