# correlation matrix plot example
from pandas import read_csv
from matplotlib import pyplot
from pandas.tools.plotting import scatter_matrix
import numpy

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

#correlations = data.corr()
#
## plot the correlation matrix
#figure = pyplot.figure()
#ax = figure.add_subplot(111)
#cax = ax.matshow(correlations, vmin=-1, vmax=1)
#figure.colorbar(cax)
#ticks = numpy.arange(0,9,1)
##ax.set_xticks(ticks)
##ax.set_yticks(ticks)
##ax.set_xticklabels(names)
##ax.set_yticklabels(names)
#pyplot.show()
#

# scatter plot matrix
scatter_matrix(data)
pyplot.show()