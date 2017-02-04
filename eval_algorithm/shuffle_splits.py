# evaluating the algorithm using standard train test split
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression

filename = '../datasets/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

seed = 7
ss = ShuffleSplit(n_splits=10, test_size=0.30, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=ss)

print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
