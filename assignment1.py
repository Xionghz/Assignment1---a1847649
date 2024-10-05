
from sklearn.datasets import load_svmlight_file
#load dataset
X, y = load_svmlight_file('diabetes_scale')
X = X.toarray()
print(X.shape)
print(y.shape)
print(y)