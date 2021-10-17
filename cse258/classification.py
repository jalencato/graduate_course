import numpy as np
from urllib.request import urlopen
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def parseDataURL(fname):
    for l in urlopen(fname):
        yield eval(l)


def parseData(fname):
    for l in open(fname):
        yield eval(l)


def flat_list(l):
    res = []
    for sub in l:
        for item in sub:
            res.append(item)
    return res


# Task 5
print("Task 5")
print("start reading data")
data = list(parseDataURL(r"https://cseweb.ucsd.edu/classes/fa21/cse258-b/data/beer_50000.json"))
print("review data have already been loaded")

x = [[len(d['review/text'])] for d in data]
y = [1 if d['review/overall'] >= 4 else 0 for d in data]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=50)
# x_train = x[:25000]
# y_train = y[:25000]
#
# x_test = x[25000:]
# y_test = y[25000:]

mod = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
mod.fit(x_train, y_train)
pred = mod.predict(x_test)
train_predictions = mod.predict(x_train)
test_predictions = mod.predict(x_test)
correct_train_predictions = train_predictions == y_train
correct_test_predictions = test_predictions == y_test
print("train accuracy: ", sum(correct_train_predictions) / len(correct_train_predictions))
print("test accuracy: ", sum(correct_test_predictions) / len(correct_test_predictions))
print(sum(np.logical_and([[1], [1]], [[1], [1]])))
print(pred[:2])


tp = sum(np.logical_and(pred, y_test))
fp = sum(np.logical_and(pred, np.logical_not(y_test)))
tn = sum(np.logical_and(np.logical_not(pred), np.logical_not(y_test)))
fn = sum(np.logical_and(np.logical_not(pred), y_test))
#
print("tp: ", tp, "fp: ", fp, "tn: ", tn, "fn: ", fn)

#Task 6
scores = mod.decision_function(x_test)
#
scores_labels = list(zip(scores, y_test))
scores_labels.sort(reverse = True)
#
sortedlabels = [x[1] for x in scores_labels]
#
line = [x for x in range(1, 10001)]
topkscore = [sum(sortedlabels[:i])/i for i in range(1, 10001)]

plt.xlabel("K")
plt.ylabel("precision@K")
plt.plot(line, topkscore)
plt.show()

#Task 7
scores = mod.decision_function(x_test)
#
print(len([1 for x in scores if x < 0]))
scores_labels = list(zip(abs(scores), y_test))
scores_labels.sort(reverse = True)
#
sortedlabels = [x[1] for x in scores_labels]
#
# precision at 10000
line = [x for x in range(1, 10001)]
topkscore = [sum(sortedlabels[:i])/i for i in range(1, 10001)]

plt.xlabel("K")
plt.ylabel("precision@K")
plt.plot(line, topkscore)
plt.show()

print("precision@1: ", sum(sortedlabels[:1])/1)
print("precision@100: ", sum(sortedlabels[:100])/100)
print("precision@10000: ", sum(sortedlabels[:10000])/10000)
