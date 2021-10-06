import numpy as np
from urllib.request import urlopen
import scipy.optimize
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def parseDataURL(fname):
    for l in urlopen(fname):
        yield eval(l)


def parseData(fname):
    for l in open(fname):
        yield eval(l)


# Task 1
print("Task 1")
print("start reading data")
data = list(parseDataURL(r"https://cseweb.ucsd.edu/classes/fa21/cse258-b/data/beer_50000.json"))
print("review data have already been loaded")


x = [[len(d['review/text'])] for d in data]
y = [[1] if d['review/overall'] >= 4 else [0] for d in data]

# x_train, x_test = x[:25000], x[25000:]
# y_train, y_test = y[:25000], y[25000:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=50)

mod = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
mod.fit(x_train, y_train)
pred = mod.predict(x_test)

tp = sum(np.logical_and(pred, y_test))
fp = sum(np.logical_and(pred, np.logical_not(y_test)))
tn = sum(np.logical_and(np.logical_not(pred), np.logical_not(y_test)))
tp = sum(np.logical_and(np.logical_not(pred), y_test))

print(tp, fp, tn, tp)


