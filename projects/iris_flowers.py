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

from bokeh.charts import Histogram, output_file, show

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
print (dataset.hist())

# b) Data visualizations
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# historgram
# dataset.hist()
# pyplot.show()

#
hist = Histogram(dataset, values='Class', color='Class', title="Distribution by Class", legend='top_right')
output_file("histogram_single.html", title="histogram_single.py example")
show(hist)

# scatter_matrix
# scatter_matrix(dataset)
# pyplot.show()

# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

# 4. Evaluate Algorithms
# a) Split-out validation dataset
dataarray = dataset.values
X = dataarray[:,0:4]
y = dataarray[:,4]

validation_size = 0.20
seed = 7
X_train, X_validation, y_train, y_validation = train_test_split(X, y,
test_size=validation_size, random_state=seed)

# b) Test options and evaluation metric
# c) Spot Check Algorithms
models = []
models.append(("LR", LogisticRegression()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("DTC", DecisionTreeClassifier()))
models.append(("SVC", SVC()))

results = []
names = []
# d) Compare Algorithms
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = pyplot.figure()
fig.suptitle("Algorithm comparison")

ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)

pyplot.show()
# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles

# 6. Finalize Model
# a) Predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))

# b) Create standalone model on entire training dataset
# c) Save model for later use