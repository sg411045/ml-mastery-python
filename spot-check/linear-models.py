#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 15:46:53 2017

@author: sg
"""

# evaluating the algorithm using standard train test split
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

filename = '../datasets/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy LR: %.3f (%.3f)") % (results.mean(), results.std())

ldamodel = LinearDiscriminantAnalysis()
results2 = cross_val_score(ldamodel, X, Y, cv=kfold)
print("Accuracy LDA: %.3f (%.3f)") % (results2.mean(), results2.std())

knn = KNeighborsClassifier()
results3 = cross_val_score(knn, X, Y, cv=kfold)
print("Accuracy NB: %.3f (%.3f)") % (results3.mean(), results3.std())

nb = GaussianNB()
results4 = cross_val_score(nb, X, Y, cv=kfold)
print("Accuracy NB: %.3f (%.3f)") % (results4.mean(), results4.std())
