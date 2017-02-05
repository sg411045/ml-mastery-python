# evaluating the algorithm using standard train test split
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

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
results = cross_val_score(model, X, Y, cv=kfold, scoring='neg_log_loss')
print("LogLoss: %.3f (%.3f)") % (results.mean(), results.std())

results2 = cross_val_score(model, X, Y, cv=kfold, scoring='roc_auc')
print("ROC: %.3f (%.3f)") % (results2.mean(), results2.std())
