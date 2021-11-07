import collections
import warnings
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
    train, valid = data[:400000], data[400001:]
    return data, train, valid


data, train, valid = splitDataset("data/trainInteractions.csv.gz")

def readCSV(path):
    f = gzip.open(path, 'rt')
    c = csv.reader(f)
    header = next(c)
    for l in c:
        d = dict(zip(header,l))
        yield d['user_id'],d['recipe_id'],d

# allRatings = []
# userRatings = defaultdict(list)
#
# for user, recipe, d in readCSV("data/trainInteractions.csv.gz"):
#     r = int(d['rating'])
#     allRatings.append(r)
#     userRatings[user].append(r)
#
# globalAverage = sum(allRatings) / len(allRatings)
# userAverage = {}
# for u in userRatings:
#     userAverage[u] = sum(userRatings[u]) / len(userRatings[u])
#
# predictions = open("data/predictions_Rated.txt", 'w')
# for l in open("data/stub_Rated.txt"):
#     if l.startswith("user_id"):
#         #header
#         predictions.write(l)
#         continue
#     u,i = l.strip().split('-')
#     if u in userAverage:
#         predictions.write(u + '-' + i + ',' + str(userAverage[u]) + '\n')
#     else:
#         predictions.write(u + '-' + i + ',' + str(globalAverage) + '\n')
#
# predictions.close()


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
    print("MSE = " + str(cost))
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
alwaysPredictMeanTrain = [ratingMean]*400000
alwaysPredictMeanValid = [ratingMean]*100000


labels = [row['rating'] for index, row in tqdm(train.iterrows())]

# print(MSE(alwaysPredictMeanTrain, labels))
# 0.8987313599958769

scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems), derivative, args = (labels, 1))

predictions = [prediction(row['user_id'], row['recipe_id']) for index, row in tqdm(valid.iterrows())]

labelsValid = [row['rating'] for index, row in tqdm(valid.iterrows())]

print(MSE(predictions, labelsValid))

print("max user: %s , max value: %f" % (max(userBiases, key=userBiases.get), max(userBiases.values())))
print("max recipe: %s , max value: %f" % (max(itemBiases, key=itemBiases.get), max(itemBiases.values())))
print("min user: %s , min value: %f" % (min(userBiases, key=userBiases.get), min(userBiases.values())))
print("min recipe: %s , min value: %f" % (min(itemBiases, key=itemBiases.get), min(itemBiases.values())))

lambdaExpList = range(-7,2)
for exp in lambdaExpList:
    # Train the model
    l = pow(10, exp)
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems), derivative, args = (labels, l))

    # Predict from the model
    predictions = [prediction(row['user_id'], row['recipe_id']) for index, row in tqdm(valid.iterrows())]
    labelsValid = [row['rating'] for index, row in tqdm(valid.iterrows())]
    mse = MSE(predictions, labelsValid)
    print("Lambda: %.10f, MSE of validation set = %f" % (l, mse))
    # writeOutTestSetPred(l)

    stream = open("predictions_Rating" + str(l) + ".txt", 'w')
    for l in tqdm(open("data/stub_Rated.txt")):
        if l.startswith("user_id"):
            #header
            stream.write(l)
            continue
        str_u, str_i = l.strip().split('-')
        u, i = int(str_u), int(str_i)
        stream.write(str_u + '-' + str_i + ',' + str(prediction(u, i)) + '\n')
    stream.close()