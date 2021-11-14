import collections
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
    train, valid = data, data[400001:]
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


recipeCount = defaultdict(int)
totalCooked = 0
for index, row in tqdm(train.iterrows()):
    recipeCount[row['recipe_id']] += row['rating']
    # recipeCount[row['recipe_id']] += 1
    totalCooked += row['rating']
mostPopular = [(recipeCount[x], x) for x in recipeCount]
mostPopular.sort()
mostPopular.reverse()

recipeRatingCount = defaultdict(float)


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
for index, row in tqdm(data.iterrows()):
    userRecipe[row['user_id']].add(row['recipe_id'])
    recipeUser[row['recipe_id']].add(row['user_id'])


def parse(f):
    for l in open(f, 'r', encoding='utf8'):
        yield eval(l)


path = 'data/trainRecipes.json'
train_dataset = list(parse(path))
path = 'data/testRecipes.json'
test_dataset = list(parse(path))
itemPerRecipe_train, recipePerItem_train = collections.defaultdict(set), collections.defaultdict(set)
for d in train_dataset:
    for i in d['ingredients']:
        itemPerRecipe_train[d['recipe_id']].add(i)
        recipePerItem_train[i].add(d['recipe_id'])

itemPerRecipe_test, recipePerItem_test = collections.defaultdict(set), collections.defaultdict(set)
item = set()
for d in test_dataset:
    for i in d['ingredients']:
        item.add(i)
        itemPerRecipe_test[d['recipe_id']].add(i)
        recipePerItem_test[i].add(d['recipe_id'])


return1 = set()
count = 0

for ic, i in mostPopular:
    count += ic
    return1.add(i)
    # if count > totalCooked * 0.6455:
    if count > totalCooked*0.628:
        break
#Task 5
predictions = open("kaggle.txt", 'w')
predictions.write('user_id-recipe_id,prediction\n')
thresohold_up = 0.2
thresohold_down = -1
def ensemble_kaggle():
    for l in tqdm(open("data/stub_Made.txt")):
        u, r = l.strip().split('-')
        userR = userRecipe[int(u)]
        m = 0
        # method 1
        # for recipe in userR:
        #     if int(r) not in recipeUser:
        #         m = max(0, m)
        #     else:
        #         # m = max(Jaccard(recipeUser[int(r)], recipeUser[recipe]), m)
        #         m = max(Ochial(recipeUser[int(r)], recipeUser[recipe]), m)
        #         # m = max(Dice(recipeUser[int(r)], recipeUser[recipe]), m)
        #         if m > thresohold_up:
        #             break

        #method 2
        # us = set()
        # for recipe in userR:
        #     if int(r) not in recipeUser:
        #             m = max(0, m)
        #     else:
        #         us = us.union(recipeUser[recipe])
        #         m = max(Ochial(recipeUser[int(r)], recipeUser[recipe]), m)
        #         if m > thresohold_down:
        #             break

        # method 3

        # ingre_list = itemPerRecipe_test[int(r)]
        # append = set()
        # for i in ingre_list:
        #     max_similarity, max_item = -1, ''
        #     for it in item:
        #         if it in ingre_list:
        #             continue
        #         sim = Jaccard(recipePerItem_train[it], recipePerItem_train[i])
        #         if sim > max_similarity:
        #             max_similarity, max_item = sim, it
        #     append.add(max_item)
        #
        # for s in append:
        #     ingre_list.add(s)
        #
        # for d in tqdm(train_dataset):
        #     recipe = set(d['ingredients'])
        #     sim = Jaccard(recipe, ingre_list)
        #     m = max(sim, m)
        #     if m > thresohold:
        #         break

        if m > thresohold_down and int(r) in return1:
            predictions.write(u + '-' + r + ",1\n")
        else:
            predictions.write(u + '-' + r + ",0\n")

    # acc.append(correct / len(valid))
    return "finish training"

print(ensemble_kaggle())
# predictions.close()