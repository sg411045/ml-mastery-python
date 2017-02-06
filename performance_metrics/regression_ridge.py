#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 08:14:18 2017

@author: sg
"""

# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

filename = '../datasets/housing-data.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']

dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]

kfold = KFold(n_splits=10, random_state=7)

model = Ridge(normalize=True)

print("Running Ridge Regression")
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())

rmse_scoring = 'neg_mean_squared_error'
results2 = cross_val_score(model, X, Y, cv=kfold, scoring=rmse_scoring)
print("MSE: %.3f (%.3f)") % (results2.mean(), results2.std())

r2_scoring = 'r2'
results2 = cross_val_score(model, X, Y, cv=kfold, scoring=r2_scoring)
print("R^2: %.3f (%.3f)") % (results2.mean(), results2.std())
