#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:04:55 2017

@author: sg
"""

import matplotlib.pyplot as plt
import numpy

x = numpy.array([1,2,3])
y = numpy.array([3,5,6])
plt.scatter(x,y)
plt.xlabel('some x axis')
plt.ylabel('some y axis')