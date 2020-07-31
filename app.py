from flask import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

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

dataset = pd.read_csv('D:/mini2/data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
  
dataset['ca']=pd.to_numeric(dataset['ca'],errors='coerce')
dataset['thal']=pd.to_numeric(dataset['thal'],errors='coerce')
dataset['slope']=pd.to_numeric(dataset['slope'],errors='coerce')
dataset['chol']=pd.to_numeric(dataset['chol'],errors='coerce')
dataset['restecg']=pd.to_numeric(dataset['restecg'],errors='coerce')
dataset['thalach']=pd.to_numeric(dataset['thalach'],errors='coerce')
dataset['fbs']=pd.to_numeric(dataset['fbs'],errors='coerce')
dataset['exang']=pd.to_numeric(dataset['exang'],errors='coerce')
dataset['trestbps']=pd.to_numeric(dataset['trestbps'],errors='coerce')
 
    
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)      
    
 
    
X = pd.DataFrame(dataset.iloc[:,:-1].values)
xt = DataFrameImputer().fit_transform(X)   
x_train,x_test,y_train,y_test = train_test_split(xt,y,test_size=0.2,random_state=0)
sc = StandardScaler()
x_train  = sc.fit_transform(x_train)

x_test= sc.transform(x_test)



classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)
y_pred= classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
v= accuracy_score(y_test,y_pred)

filename = 'diabetes.pkl'
pickle.dump(classifier, open(filename, 'wb'))

if __name__ == '__main__':
	app.run(debug=True)
