# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)

from pandas import read_csv
from pandas import set_option
from sklearn.decomposition import PCA

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values

X = array[:,0:8]
Y = array[:,8]

pca = PCA(n_components = 3)
fit = pca.fit(X)

# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)
