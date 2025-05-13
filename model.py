 # Multiple Linear Regression 
# Importing the libraries
 import numpy as np
 import matplotlib.pyplot as plt
 import pandas as pd
 import pickle #A module called pickle helps perform serialization and deserialization i
 n python. 
# Importing the dataset
 dataset = pd.read_csv('50_Startup.csv')
 X = dataset.iloc[:, :-1].values
 y = dataset.iloc[:, 3].values 
# Fitting Multiple Linear Regression to the Training set
 from sklearn.linear_model import LinearRegression
 regressor = LinearRegression()
 regressor.fit(X, y) 
# Saving model to disk
 pickle.dump(regressor, open('model.pkl','wb')) 
# Loading model to compare the results
 model = pickle.load(open('model.pkl','rb'))
 print(model.predict([[16000, 135000, 450000]]))
