from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
#load dataset
X, y = load_svmlight_file('diabetes_scale')
X = X.toarray()

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

  