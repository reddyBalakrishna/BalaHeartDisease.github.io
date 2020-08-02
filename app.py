from flask import *
import numpy as np
import pandas as pd
import pickle

filename = 'diabetes.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('predict.html')



@app.route('/result', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['pericarditis']
        trestbps= request.form['bloodpressure']
        chol = request.form['cholestrol']
        fbs = request.form['sugar']
        restecg = request.form['resting']
        thalach = request.form['heartrate']
        exang = request.form['exercise']
        oldpeak = float(request.form['oldpeaks'])
        slope = request.form['oldpeak']
        ca = request.form['carotid']
        thal = request.form['thal']
        
        
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        target = classifier.predict(data)
        
        return render_template('result.html',prediction=target)

@app.route('/result.html',methods=['POST'])
def result():
    return render_template('result.html')

if __name__ == '__main__':
	app.run(debug=True)
