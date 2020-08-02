import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('D:/mini1/data1.csv')
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
    
 
from sklearn.model_selection import train_test_split     
X = pd.DataFrame(dataset.iloc[:,:-1].values)
xt = DataFrameImputer().fit_transform(X)   
x_train,x_test,y_train,y_test = train_test_split(xt,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train  = sc.fit_transform(x_train)
x_test= sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)
y_pred= classifier.predict(x_test)

filename = 'diabetes.pkl'
pickle.dump(classifier, open(filename, 'wb'))
