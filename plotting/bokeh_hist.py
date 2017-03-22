# Python Project Template
# 1. Prepare Problem
# a) Load libraries
from pandas import read_csv
from matplotlib import pyplot
from pandas.tools.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from bokeh.charts import Histogram, Scatter, output_file, show

# b) Load dataset
filename = "../datasets/iris_with_headers.csv"
names = ["Sepal-Length", "Sepal-Width", "Petal-Length", "Petal-Width", "Class"]
dataset = read_csv(filename, header=0, names=names)

# 2. Summarize Data
# a) Descriptive statistics
print(dataset.shape)
print(dataset.groupby('Class').size())
print(dataset.head(60))
print(dataset.describe())

#
hist = Histogram(dataset, values='Class', color='Class', title="Distribution by Class", legend='top_right')
output_file("histogram_single.html", title="histogram_single.py example")
show(hist)

scatter = Scatter(dataset, x='Petal-Length', y='Petal-Width', color='Class', marker='Class',
                  title='Iris Dataset Color and Marker by Species',
                  legend=True)
output_file("iris_simple_scatter.html", title="iris_simple.py example")

show(scatter)