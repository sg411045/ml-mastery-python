#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:03:06 2017

@author: sg
"""

import pandas
from numpy import loadtxt
filename = 'pima-indians-diabetes.csv'

raw_data = open(filename, 'rb')
data = loadtxt(raw_data, delimiter=',')
print(data.shape)

