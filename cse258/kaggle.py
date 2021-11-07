import warnings

warnings.filterwarnings("ignore")

import gzip
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


def splitDataset(datapath):
    f = gzip.open(datapath, 'rt')
    data = pd.read_csv(f)
    train, valid = data[:400000], data[400001:]
    return data, train, valid


data, train, valid = splitDataset("data/trainInteractions.csv.gz")


# Task 1
def sampleNegative(data, train, valid):
    NegValid, userRecipe = valid, {}
    print("Preprocessing Data userRecipe ...")

    for index, row in tqdm(data.iterrows()):
        if row['user_id'] not in userRecipe:
            userRecipe[row['user_id']] = {row['recipe_id']}
        else:
            userRecipe[row['user_id']].add(row['recipe_id'])

    print("confirm loading training data")
    for index, row in tqdm(valid.iterrows()):
        negValidRecipe = random.sample(set(data['recipe_id']).difference(userRecipe[row['user_id']]), 1)[0]
        NegValid = NegValid.append({'user_id': row['user_id'], 'recipe_id': negValidRecipe, 'date': 0, 'rating': -1},
                                   ignore_index=True)

    return NegValid, userRecipe


# print("Sampling negative examples")
# valid, _ = sampleNegative(data, train, valid)
# print("Start saving")
# validFrame = pd.DataFrame(valid)
# validFrame.to_csv('valid.csv')
# print("Start loading...")
# valid = pd.read_csv('valid.csv')
# print(valid)
# print('Training ...')
# recipeCount = defaultdict(int)
# totalCooked = 0
# for user, recipe in tqdm(train.iterrows()):
#     recipeCount[recipe['recipe_id']] += 1
#     totalCooked += 1
# mostPopular = [(recipeCount[x], x) for x in recipeCount]
# mostPopular.sort()
# mostPopular.reverse()

def baselineOnValidation():
    random.seed(5583)

    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalCooked / 2:
            break

    print('Evaluating ...')
    correct = 0
    for index, row in tqdm(valid.iterrows()):
        if row['recipe_id'] in return1:
            correct += (row['date'] != 0)
        else:
            correct += (row['date'] == 0)

    print('Accuracy on Validation set is %.3f' % (correct / len(valid)))


# baselineOnValidation()

# Task 2
def baselineOnValidationThreshold():
    random.seed(5583)

    acc = []
    thresholds = list(range(1, 11))

    for threshold in thresholds:
        return1 = set()
        count = 0
        for ic, i in mostPopular:
            count += ic
            return1.add(i)
            if count > totalCooked / threshold:
                break

        correct = 0
        for index, row in tqdm(valid.iterrows()):
            if row['recipe_id'] in return1:
                correct += (row['date'] != 0)
            else:
                correct += (row['date'] == 0)

        print('Evaluating on threshold %d ...' % threshold)
        acc.append(correct / len(valid))

    plt.plot(thresholds, acc, 'b-')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy on Validation Set for different thresholds')
    plt.show()
    print('Evaluating ...')
    print('max accuracy is ', max(acc))
#
#
# baselineOnValidationThreshold()
#
#
#Task 3
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


# userRecipe, recipeUser = {}, {}
userRecipe, recipeUser = defaultdict(set), defaultdict(set)
for index, row in tqdm(train.iterrows()):
    userRecipe[row['user_id']].add(row['recipe_id'])
    recipeUser[row['recipe_id']].add(row['user_id'])


thresholds = [1 / 2 ** i for i in range(1, 11)]
acc = []
# for threshold in thresholds:
#     print('Evaluating on threshold %.3f ...' % threshold)
#     correct = 0
#     for index, row in tqdm(valid.iterrows()):
#         userR = userRecipe[row['user_id']]
#         jac = []
#         m = -1
#         for recipe in userR:
#             if row['recipe_id'] not in recipeUser:
#                 m = max(0, m)
#             else:
#                 m = max(Jaccard(recipeUser[row['recipe_id']], recipeUser[recipe]), m)
#                 # jac.append(Jaccard(recipeUser[row['recipe_id']], recipeUser[recipe]))
#
#         if m > threshold:
#             correct += (row['date'] != 0)
#         else:
#             correct += (row['date'] == 0)
#
#     print('Evaluating on threshold %d ...' % threshold)
#     acc.append(correct / len(valid))
#
# plt.plot(thresholds, acc, 'b-')
# plt.xlabel('Threshold')
# plt.ylabel('Accuracy on Validation Set for different thresholds')
# plt.show()
# print('Evaluating ...')
#
# print('threshold %.3f, accuracy %.3f' % (thresholds[acc.index(max(acc))], max(acc)))

#Task 4
def ensemble():
    correct = 0
    for index, row in tqdm(valid.iterrows()):
        userR = userRecipe[row['user_id']]
        m = -1
        for recipe in userR:
            if row['recipe_id'] not in recipeUser:
                m = max(0, m)
            else:
                m = max(Jaccard(recipeUser[row['recipe_id']], recipeUser[recipe]), m)

        return1 = set()
        count = 0
        for ic, i in mostPopular:
            count += ic
            return1.add(i)
            if count > totalCooked / 2:
                break

        if m > 0.1 or row['recipe_id'] in return1:
            correct += (row['date'] != 0)
        else:
            correct += (row['date'] == 0)

    return correct / len(valid)

# print('accuracy %.3f' % ensemble())


#Task 5
predictions = open("kaggle.txt", 'w')
predictions.write('user_id-recipe_id,prediction\n')
def ensemble_kaggle():
    correct = 0
    for l in tqdm(open("data/stub_Made.txt")):
        u, r = l.strip().split('-')
        userR = userRecipe[int(u)]
        m = -1
        for recipe in userR:
            if int(r) not in recipeUser:
                m = max(0, m)
            else:
                m = max(Jaccard(recipeUser[int(r)], recipeUser[recipe]), m)

        return1 = set()
        count = 0

        for ic, i in mostPopular:
            count += ic
            return1.add(i)
            if count > totalCooked / 2:
                break

        if m > 0.001 or int(r) in return1:
            predictions.write(u + '-' + r + ",1\n")
        else:
            predictions.write(u + '-' + r + ",0\n")

    # acc.append(correct / len(valid))
    return correct / len(valid)

# print('accuracy %.3f' % ensemble_kaggle())
predictions.close()

#Task 6
import scipy
import scipy.optimize
import numpy as np
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


def prediction(user, item):
    return alpha + userBiases[user] + itemBiases[item]


def MSE(predictions, labels):
    differences = [(x-y)**2 for x, y in tqdm(zip(predictions,labels))]
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
    for index, d in tqdm(train.iterrows()):
        u, i = d['user_id'], d['recipe_id']
        pred = prediction(u, i)
        diff = pred - d['rating']
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
    for u in tqdm(userBiases):
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in tqdm(itemBiases):
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    return np.array(dtheta)


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, lamb))
print(userBiases)

predictions = []
for index, d in valid.iterrows():
    u, i = d['user_id'], d['recipe_id']
    if u in userBiases and i in itemBiases:
        predictions.append(prediction(u, i))
    else:
        predictions.append(0)

print("MSE %.3f" % MSE(predictions, valid['rating']))

# print("max user: %s , max value: %f" % (max(userBiases, key=userBiases.get), max(userBiases.values())))
# print("max recipe: %s , max value: %f" % (max(itemBiases, key=itemBiases.get), max(itemBiases.values())))
# print("min user: %s , min value: %f" % (min(userBiases, key=userBiases.get), min(userBiases.values())))
# print("min recipe: %s , min value: %f" % (min(itemBiases, key=itemBiases.get), min(itemBiases.values())))

lamb = 10**(-5)

def training(lamb):
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                                 derivative, args = (labels, lamb))

# training(lamb)

predictions = []
for index, d in valid.iterrows():
    u, i = d['user_id'], d['recipe_id']
    predictions.append(prediction(u, i))
    if u in userBiases and i in itemBiases:
        predictions.append(prediction(u, i))
    else:
        predictions.append(0)
        #     if u in userAverage:
#         predictions.write(u + '-' + i + ',' + str(userAverage[u]) + '\n')
#     else:
#         predictions.write(u + '-' + i + ',' + str(globalAverage) + '\n')
print(predictions)
print("MSE %.3f" % MSE(predictions, valid['rating']))

predictions = open("predictions_Rating.txt", 'w')
for l in tqdm(open("data/stub_Rated.txt")):
    if l.startswith("user_id"):
        #header
        predictions.write(l)
        continue
    u, i = l.strip().split('-')
    if u in userBiases and i in itemBiases:
        predictions.write(u + '-' + i + ',' + str(prediction(u, i)) + '\n')
    else:
        predictions.write(u + '-' + i + ',' + str(0) + '\n')

predictions.close()