from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
#load dataset
X, y = load_svmlight_file('diabetes_scale')
X = X.toarray()

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

#perceptron class
class Perceptron:
    def __init__(self, max_iter=1000, step_size=0.001):
        self.max_iter = max_iter
        self.step_size = step_size
        self.accuracy = []
        self.train_accuracy = []
        

    def fit(self, X, y):
        #initialize weight to -1
        self.w=np.zeros(X.shape[1])
        for i in range(self.w.shape[0]):
            self.w[i] = 1
        #initialize bias to 0
        self.b = 0
        #loop through the dataset for max_iter times
        for i in range(self.max_iter):
            #test if all samples are correctly classified
            #loop through each sample in the dataset
            for j in range(X.shape[0]):

                # if y*(wx+b) <= 0, update weights and bias
                if y[j] * (np.dot(X[j], self.w)+self.b) <= 0:


                    self.w += self.step_size * y[j] * X[j]
                    self.b += self.step_size * y[j]
            #calculate accuracy
            y_pred = perceptron.predict(X_test)
            correct = 0
            for j in range(len(y_test)):
                if y_test[j] == y_pred[j]:
                    correct += 1
            accuracy = correct / len(y_test)
            self.accuracy.append(accuracy)
            #calculate training accuracy
            y_pred = perceptron.predict(X_train)
            correct = 0
            for j in range(len(y_train)):
                if y_train[j] == y_pred[j]:
                    correct += 1
            accuracy = correct / len(y_train)
            self.train_accuracy.append(accuracy)
            
            
    
        return self

    def predict(self, X):
        predict_y =  np.sign(np.dot(X, self.w)+self.b)
        return predict_y
    

#train perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
print (perceptron.accuracy[-1])
#plot accuracys
plt.plot(perceptron.accuracy,label='test')
plt.plot(perceptron.train_accuracy,label='train')
plt.legend(['test','train'])
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Perceptron setp size 0.001')
plt.show()



 