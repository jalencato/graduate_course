import collections
import warnings
from ast import literal_eval
from statistics import mean

warnings.filterwarnings("ignore")

import csv
import gzip
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
import scipy.optimize

def splitDataset(datapath):
    f = gzip.open(datapath, 'rt')
    data = pd.read_csv(f)
    train, valid = data, data[400001:]
    return data, train, valid


data, train, valid = splitDataset("data/trainInteractions.csv.gz")


def readCSV(path):
    f = gzip.open(path, 'rt')
    c = csv.reader(f)
    header = next(c)
    for l in c:
        d = dict(zip(header, l))
        yield d['user_id'], d['recipe_id'], d


#course method
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for index, row in tqdm(train.iterrows()):
    reviewsPerUser[row['user_id']].append(row['rating'])
    reviewsPerItem[row['recipe_id']].append(row['rating'])

# ratingMean = sum([d['rating'] for index, d in tqdm(train.iterrows())]) / len(train)
ratingMean = 4.5808
alpha = ratingMean

N = len(train)
nUsers = len(reviewsPerUser)
nItems = len(reviewsPerItem)
users = list(reviewsPerUser.keys())
items = list(reviewsPerItem.keys())
userBiases = defaultdict(float)
itemBiases = defaultdict(float)

print("Start loading...")
parameter = pd.read_csv('100_new.csv')

l = []
for index, k in parameter.iterrows():
    if index == 0:
        l.append(k['0'])
    else:
        l.append(literal_eval(k['0']))

s = []
alpha = float(l[0])
print(alpha)
for t in users:
    userBiases[t] = l[1][t]
    s.append(l[1][t])
for t in items:
    itemBiases[t] = l[2][t]
    s.append(l[2][t])


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in tqdm(zip(predictions, labels))]
    return sum(differences) / len(differences)


def prediction(user, item):
    userBias = 0
    itemBias = 0
    if user in userBiases:
        userBias = userBiases[user]
    if item in itemBiases:
        itemBias = itemBiases[item]
    if alpha + userBias + itemBias > 5:
        return 4.915
    elif alpha + userBias + itemBias < 0:
        return 0
    else:
        # return round(alpha + userBias + itemBias, 7)
        return alpha + userBias + itemBias


def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))


def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(row['user_id'], row['recipe_id']) for index, row in tqdm(train.iterrows())]
    cost = MSE(predictions, labels)
    print("Training MSE = " + str(cost))
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost


def derivative(theta, labels, lamb):
    unpack(theta)
    print(alpha)
    N = len(train)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for index, row in tqdm(train.iterrows()):
        pred = prediction(row['user_id'], row['recipe_id'])
        diff = pred - row['rating']
        dalpha += 2/N*diff
        dUserBiases[row['user_id']] += 2/N*diff
        dItemBiases[row['recipe_id']] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u_id] for u_id in users] + [dItemBiases[i_id] for i_id in items]
    return np.array(dtheta)


# alwaysPredictMean = [ratingMean for d in train.iterrows()]
alwaysPredictMeanTrain = [ratingMean]*500000
alwaysPredictMeanValid = [ratingMean]*100000


labels = [row['rating'] for index, row in tqdm(train.iterrows())]


num = pow(10, -5) * 0.8
scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems), derivative, args = (labels, num), maxiter=100)
# scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + s, derivative, args = (labels, num), maxiter=1)
# Predict from the model
# predictions = [prediction(row['user_id'], row['recipe_id']) for index, row in tqdm(valid.iterrows())]
# labelsValid = [row['rating'] for index, row in tqdm(valid.iterrows())]
# mse = MSE(predictions, labelsValid)
# print("Lambda: %.10f, MSE of validation set = %f" % (num, mse))

print("Sampling negative examples")
valid = [alpha, userBiases, itemBiases]
print("Start saving")
validFrame = pd.DataFrame(valid)
validFrame.to_csv('parameter.csv')

stream = open("predictions_Rating" + str(num) + ".txt", 'w')
for l in tqdm(open("data/stub_Rated.txt")):
    if l.startswith("user_id"):
        stream.write(l)
        continue
    str_u, str_i = l.strip().split('-')
    u, i = int(str_u), int(str_i)
    stream.write(str_u + '-' + str_i + ',' + str(prediction(u, i)) + '\n')
stream.close()