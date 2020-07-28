import pandas as pd
import numpy as np
from flask_cors import CORS,cross_origin
from flask import render_template,Flask,request,jsonify
import pickle

app = Flask(__name__)

@cross_origin()
@app.route('/')
def home():
    return render_template('index.html')
@cross_origin()
@app.route('/predict',methods=['POST'])
def predict():
    try:
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])

        filename = 'log_reg.pickle'
    ### Loading model for prediction
        loaded_model = pickle.load(open(filename,'rb'))
        scaler = pickle.load(open('scaler.pickle','rb'))
        prediction = loaded_model.predict(scaler.transform(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]))
        if prediction == [1]:
            prediction = 'diabetes'
        else:
            prediction = 'Normal'
    ## Showing result in UI
        if prediction == 'diabetes':
            return render_template('diabetes.html',prediction=prediction)
        else:
            return render_template('Normal.html',prediction=prediction)
    except Exception as e:
        print('The Exception message is',e)

if __name__ == '__main__':
    #app.run(host='127.0.0.1',port=8000,debug=True)
    app.run(debug=True)
