## Price Prediction of cars on the basis of size, horsepower and variables that explain additional variance
## I used Python 3.6.0, numpy 1.14.0, pandas 0.22.0 and scikit-learn 0.19.1

import csv
import numpy as np
import pandas as pd

## Load DataFrame
data=pd.read_csv('Auto1-DS-TestData.csv')

## Data preprocessing
additional='engine-size'
data=data.replace('?', np.NaN) #replace question marks by NaN
data=data.dropna(axis=0,subset=['price','length','width','height','horsepower',additional]) #get rid of rows with NaNs
data['price']=data['price'].astype(float) #convert all input data into float and calculate variable size
data['size']=data['length'].values*data['width'].values*data['height'].values
data['horsepower']=data['horsepower'].astype(float)
data[additional]=data[additional].astype(float)

## Choose input and output variables
inputf=data[['size','horsepower',additional]].values
dependent=data['price'].values

## Split the Data
from sklearn.cross_validation import train_test_split
(trainset,testset,trainpred,testpred) = train_test_split(inputf, dependent, train_size=0.8, random_state=1)

## Do Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
LM = LinearRegression(fit_intercept=True, normalize=False)
LMfit = LM.fit(trainset,trainpred)
LMpred = LMfit.predict(testset)

## Model output
print("Mean squared error: %.2f" % mean_squared_error(testpred,LMpred))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(testpred,LMpred)))
print("Explained Variance: %.2f" % r2_score(testpred,LMpred))

## Do Cross_validation
from sklearn.model_selection import cross_val_score,cross_val_predict,LeaveOneOut
LMfit_cross = cross_val_score(LMfit,inputf, dependent, cv=LeaveOneOut())
LMpred_cross = cross_val_predict(LMfit,inputf, dependent, cv=LeaveOneOut())

# Model output with crossvalidation
print("Crossval Mean squared error: %.2f" % mean_squared_error(dependent,LMpred_cross))
print("Crossval Root mean squared error: %.2f" % np.sqrt(mean_squared_error(dependent,LMpred_cross)))
print("Crossval Explained Variance: %.2f" % r2_score(dependent,LMpred_cross))
