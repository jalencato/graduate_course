import math
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

#
#
#Task 3


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


def Ochial(s1, s2):
    numer = len(s1.intersection(s2))
    denom = math.sqrt(len(s1) * len(s2))
    return numer/denom


def Simpson(s1, s2):
    numer = len(s1.intersection(s2))
    denom = min(len(s1), len(s2))
    return numer/denom


def Dice(s1, s2):
    numer = len(s1.intersection(s2)) * 2
    denom = len(s1) + len(s2)
    return numer/denom


userRecipe, recipeUser = defaultdict(set), defaultdict(set)
for index, row in tqdm(train.iterrows()):
    userRecipe[row['user_id']].add(row['recipe_id'])
    recipeUser[row['recipe_id']].add(row['user_id'])


thresholds = [1 / 2 ** i for i in range(1, 6)]
acc = []
for threshold in thresholds:
    print('Evaluating on threshold %.3f ...' % threshold)
    correct = 0
    for index, row in tqdm(valid.iterrows()):
        userR = userRecipe[row['user_id']]
        jac = []
        m = -1
        for recipe in userR:
            if row['recipe_id'] not in recipeUser:
                m = max(0, m)
            else:
                # m = max(Jaccard(recipeUser[row['recipe_id']], recipeUser[recipe]), m)
                # m = max(Ochial(recipeUser[row['recipe_id']], recipeUser[recipe]), m)
                m = max(Dice(recipeUser[row['recipe_id']], recipeUser[recipe]), m)

        if m > threshold:
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

print('threshold %.3f, accuracy %.3f' % (thresholds[acc.index(max(acc))], max(acc)))

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
            if count > totalCooked / 1.2:
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
# predictions.close()