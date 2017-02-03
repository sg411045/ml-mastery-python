#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:09:22 2017

@author: sg
"""
# Attribute information
#1. Number of times pregnant 
#2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test 
#3. Diastolic blood pressure (mm Hg) 
#4. Triceps skin fold thickness (mm) 
#5. 2-Hour serum insulin (mu U/ml) 
#6. Body mass index (weight in kg/(height in m)^2) 
#7. Diabetes pedigree function 
#8. Age (years) 

# Load CSV using Pandas
from pandas import read_csv
from pandas import set_option

from matplotlib import pyplot

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print(data.shape)
# print(data.dtypes)
# print(data.head(10))

set_option('display.width', 100)
set_option('precision', 3)

# print(data.describe())
print(data.groupby('class').size())
print(data.corr(method='pearson'))

print(data.skew())

# historgram plot
#data.hist()

# density plot
#data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)

# box plot
data.plot(kind='box', sym='g.', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()