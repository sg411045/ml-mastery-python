# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:58:49 2016

@author: s.gopalakrishnan
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# baseline_model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='softmax'))
    #optimizer - adam or sgd
    sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix seed for random generator
seed = 7
numpy.random.seed(seed)

# load the dataset
dataframe = pandas.read_csv("../datasets/iris.csv", delimiter="," , header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(y)
encoder_y = encoder.transform(y)

# convert integers to dummy variables
dummy_y = np_utils.to_categorical(encoder_y)

 # Split-out validation dataset
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = train_test_split(X, dummy_y, test_size=validation_size, random_state=seed)


estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=500, batch_size=10, verbose=0)
estimator.fit(X_train, Y_train)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
predictions = estimator.predict(X_validation)
print(predictions)
print(encoder.inverse_transform(predictions))

results = cross_val_score(estimator, X_validation, Y_validation, cv=kfold)
print("Baseline:  %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))