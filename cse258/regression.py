import numpy as np
from urllib.request import urlopen


def parseDataURL(fname):
    for l in urlopen(fname):
        yield eval(l)


def parseData(fname):
    for l in open(fname):
        yield eval(l)


# Task 1
print("Task 1")
print("start reading data")
data = list(parseData('data/fantasy_10000.json'))
print("review data have already been loaded")

ndatax = [len(d['review_text']) for d in data]
X = [[1, len(d['review_text'])] for d in data]
y = [d['rating'] for d in data]

theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
print('Coefficient: ', theta)


def mse_calc(x, y, theta):
    cnt, res = 0, 0
    while cnt < len(x):
        res += (y[cnt] - np.dot(x[cnt], theta)) ** 2
        cnt += 1
    res = res / len(x)
    return res


print("MSE: ", mse_calc(X, y, theta))
# Task 2
print("Task 2")
import dateutil.parser as duparser

t0 = duparser.parse(data[0]['date_added'])
t1 = duparser.parse(data[1]['date_added'])
# print(t0, t1)
# print(t0.weekday(), t0.year)
# print(t1.weekday(), t1.year)
maxyear, minyear = 0, 10000
for d in data:
    year = duparser.parse(d['date_added']).year
    maxyear = max(year, maxyear)
    minyear = min(year, minyear)


# print(maxyear, minyear)
# there are 7 weekdays so there are 6 dimensions
# t0: 100000 t1: 000010
# there are 12 year in total so there are 11 dimensions
# t0: 00000000000 t1: 00100000000
# one-hot spotting
def get_onehot(targets, classes):
    res = np.eye(classes, classes - 1)[np.array(targets).reshape(-1)]
    return list(res.reshape(list(targets.shape) + [classes - 1]))


print(get_onehot(np.array(duparser.parse(data[0]['date_added']).weekday() - 1), 7),
      get_onehot(np.array(duparser.parse(data[0]['date_added']).year - 2006), 12))
print(get_onehot(np.array(duparser.parse(data[1]['date_added']).weekday() - 1), 7),
      get_onehot(np.array(duparser.parse(data[1]['date_added']).year - 2006), 12))


# Task 3
# directly
X = [[1, len(d['review_text']), duparser.parse(d['date_added']).weekday(), duparser.parse(d['date_added']).year] for d
     in data]
y = [d['rating'] for d in data]
theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
print('Coefficient: ', theta)
print('MSE: ', mse_calc(X, y, theta))

# one-hot spotting
X = [[1, len(d['review_text'])] + get_onehot(np.array(duparser.parse(d['date_added']).weekday() - 1), 7) +
     get_onehot(np.array(duparser.parse(d['date_added']).year - 2006), 12) for d in data]
y = [d['rating'] for d in data]
theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
print('Coefficient: ', theta)
print('MSE: ', mse_calc(X, y, theta))

# Task 4
import random


def split_dataset(X, y, percent=0.5):
    combine = list(zip(X, y))
    random.shuffle(combine)
    a, b = zip(*combine)
    return a[int(len(a) * percent):], a[:int(len(X) * percent)], b[int(len(y) * percent):], b[:int(len(y) * percent)]


# direcly
X = [[1, len(d['review_text']), duparser.parse(d['date_added']).weekday(), duparser.parse(d['date_added']).year] for d
     in data]
y = [d['rating'] for d in data]
x_training, x_testing, y_training, y_testing = split_dataset(X, y)
theta, residuals, rank, s = np.linalg.lstsq(x_training, y_training, rcond=None)
print('Coefficient: ', theta)
print('MSE on training: ', mse_calc(x_training, y_training, theta))
print('MSE on testing: ', mse_calc(x_testing, y_testing, theta))

# one-hot
X = [[1, len(d['review_text'])] + get_onehot(np.array(duparser.parse(d['date_added']).weekday() - 1), 7) +
     get_onehot(np.array(duparser.parse(d['date_added']).year - 2006), 12) for d in data]
y = [d['rating'] for d in data]
x_training, x_testing, y_training, y_testing = split_dataset(X, y)
theta, residuals, rank, s = np.linalg.lstsq(x_training, y_training, rcond=None)
print('Coefficient: ', theta)
print('MSE on training: ', mse_calc(x_training, y_training, theta))
print('MSE on testing: ', mse_calc(x_testing, y_testing, theta))
