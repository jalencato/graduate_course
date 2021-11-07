
import warnings

warnings.filterwarnings("ignore")

import gzip
import scipy.optimize
import numpy as np
from collections import defaultdict
import scipy
import pandas as pd


def splitDataset(datapath):
    f = gzip.open(datapath, 'rt')
    data = pd.read_csv(f)
    train, valid = data[:400000], data[400001:]
    return data, train, valid


def prediction(user, item):
    return alpha + userBiases[user] + itemBiases[item]


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))


def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d['user_id'], d['recipe_id']) for index, d in train.iterrows()]
    cost = MSE(predictions, labels)
    print("Train MSE = " + str(cost))
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost


def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(train)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for index, d in train.iterrows():
        u,i = d['user_id'], d['recipe_id']
        pred = prediction(u, i)
        diff = pred - d['rating']
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    return np.array(dtheta)

#Task 9
lamb = 1
data, train, valid = splitDataset("data/trainInteractions.csv.gz")

ratingMean = train['rating'].mean()
alpha = ratingMean
labels = train['rating']

userBiases = defaultdict(float)
itemBiases = defaultdict(float)

users = list(set(train['user_id']))
items = list(set(train['recipe_id']))
nUsers = len(users)
nItems = len(items)

scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, lamb))

predictions = []
for index, d in valid.iterrows():
    u, i = d['user_id'], d['recipe_id']
    if u in userBiases and i in itemBiases:
        predictions.append(prediction(u, i))
    else:
        predictions.append(0)

print("Final MSE %.3f" % MSE(predictions, valid['rating']))

#Task 10