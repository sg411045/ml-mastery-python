#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:48:29 2017

@author: sg
"""
# load csv with python

import csv
import numpy

filename = 'pima-indians-diabetes.csv'
raw_data = open(filename, 'rb')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)

data = numpy.array(x).astype('float')
print(data.shape)
