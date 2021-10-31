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
    # train, valid = data[:4], data[5:9]
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
        # userRecipe[row['user_id']].append(row['recipe_id'])

    print("confirm loading training data")
    for index, row in tqdm(valid.iterrows()):
        negValidRecipe = random.sample(set(data['recipe_id']) - userRecipe[row['user_id']], 1)[0]
        NegValid = NegValid.append({'user_id': row['user_id'], 'recipe_id': negValidRecipe, 'date': 0, 'rating': -1},
                                   ignore_index=True)

    return NegValid, userRecipe


print("Sampling negative examples")
valid, _ = sampleNegative(data, train, valid)
print('Training ...')
recipeCount = defaultdict(int)
totalCooked = 0
for user, recipe in tqdm(train.iterrows()):
    recipeCount[recipe['recipe_id']] += 1
    totalCooked += 1
mostPopular = [(recipeCount[x], x) for x in recipeCount]
mostPopular.sort()
mostPopular.reverse()

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


baselineOnValidation()


# Task 2
def baselineOnValidationThreshold():
    random.seed(5583)

    acc = []
    thresholds = list(range(1, 21))

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

baselineOnValidationThreshold()

#Task 3
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


userRecipe, recipeUser = {}, {}
for index, row in tqdm(train.iterrows()):
    if row['user_id'] not in userRecipe:
        userRecipe[row['user_id']] = {row['recipe_id']}
    else:
        userRecipe[row['user_id']].add(row['recipe_id'])
    if row['recipe_id'] not in recipeUser:
        recipeUser[row['recipe_id']] = {row['user_id']}
    else:
        recipeUser[row['recipe_id']].add(row['user_id'])


thresholds = [1 / 2 ** i for i in range(1, 20)]
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
                m = max(Jaccard(recipeUser[row['recipe_id']], recipeUser[recipe]), m)
                # jac.append(Jaccard(recipeUser[row['recipe_id']], recipeUser[recipe]))

        if m > threshold:
            correct += (row['date'] != 0)
        else:
            correct += (row['date'] == 0)

    print('Evaluating on threshold %d ...' % threshold)
    acc.append(correct / len(valid))

# for threshold in thresholds:
#     correct = 0
#     for index, row in tqdm(valid.iterrows()):
#         userReads = userRecipe[row['userID']]
#         jac = []
#         for book in userReads:
#             if row['bookID'] not in recipeUser:
#                 jac.append(0)
#             else:
#                 jac.append(Jaccard(recipeUser[row['bookID']], recipeUser[book]))
#
#         if max(jac) > threshold:
#             correct += (row['read'] != 0)
#         else:
#             correct += (row['read'] == 0)
#
#     acc.append(correct / len(valid))

plt.plot(thresholds, acc, 'b-')
plt.xlabel('Threshold')
plt.ylabel('Accuracy on Validation Set for different thresholds')
plt.show()
print('Evaluating ...')

