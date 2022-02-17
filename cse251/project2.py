import math

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

# iris = datasets.load_iris()
# X = iris.data
# y = (iris.target != 0) * 1


wine = datasets.load_wine()
X = wine.data
y = wine.target

X_new, y_new = [], []
for i, d in enumerate(X):
    if y[i] == 2:
        continue
    # if y[i] == 0:
    #     y[i] == -1
    X_new.append(d)
    y_new.append(y[i])
X_new, y_new = np.array(X_new), np.array(y_new)
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y_new, test_size=0.2, random_state=8
)


logisticRegr = LogisticRegression(max_iter=4000)
logisticRegr.fit(X_train, y_train)
score = logisticRegr.score(X_test, y_test)
print("accuracy by the sklearn model", score)


class MyLogisticRegression:
    def __init__(self, lr=0.01, num_iter=10000):
        self.lr = lr
        self.num_iter = num_iter
        self.b = 0

    def __sigmoid(self, z):
        # z = z / 10000
        
        z = np.array(z, dtype=float128)
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        # print(h, y)
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        self.weights = np.ones(X.shape[1])
        for i in range(self.num_iter):
            intermediate = np.dot(X, self.weights) + self.b

            h = self.__sigmoid(intermediate)
            gradient = np.dot(X.T, (h - y)) / y.size
            db = np.sum(h - y)/y.size
            self.weights = self.weights - self.lr * gradient
            self.b = self.b - self.lr * db

            z = np.dot(X, self.weights)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
            print(loss)

    def predict_prob(self, X):
        return self.__sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        # predict_val = 1 / (1 + np.exp(-np.dot(X, self.weights) + self.b))
        # predict = np.where(predict_val > 0.5, 1, 0)
        # return predict
        # # Z = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        # # Y = np.where(Z > 0.5, 1, 0)
        # # return Y
        return self.__sigmoid(np.dot(X, self.weights) + self.b).round()


print(X_train.shape)
print(X_test.shape)
model = MyLogisticRegression(lr=0.2, num_iter=4000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print((preds == y_test).mean())
