import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

# import the ridge regressor and standard scaler models
ridge_model = pickle.load(open(r'D:\Research Forecast\ETE ML Project Algerian Fire\models\ridge.pkl', 'rb'))
standard_scaler = pickle.load(open(r'D:\Research Forecast\ETE ML Project Algerian Fire\models\scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature= float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        RAIN = float(request.form.get('RAIN'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_data_scaled = standard_scaler.transform([[Temperature, RH, WS, RAIN, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)


        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)