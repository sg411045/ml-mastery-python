#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:59:06 2017

@author: sg
"""

from pandas import read_csv
from pandas import set_option
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import pandas

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# seperate input and output
dataarray = data.values
X = dataarray[:,0:8]
y = dataarray[:,8]

# scaling of data
scaler =  MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)

# set print options
set_option('precision',3)
print(rescaledX[0:5,:])
mydataframe = pandas.DataFrame(rescaledX)
mydataframe.hist()
pyplot.show()

# standard scaler
standardScaledX = StandardScaler().fit(X)
rescaledXX = standardScaledX.transform(X)

print(rescaledXX[0:5,:])
mydataframe = pandas.DataFrame(rescaledXX)
mydataframe.hist()
pyplot.show()