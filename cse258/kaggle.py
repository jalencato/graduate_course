import gzip
from collections import defaultdict
import pandas as pd
from tqdm.notebook import tqdm
import random


def splitDataset(datapath):
    f = gzip.open(datapath, 'rt')
    data = pd.read_csv(f)
    train, valid = data[:400000], data[400001:]
    # train, valid = data[:4], data[5:9]
    return data, train, valid


data, train, valid = splitDataset("data/trainInteractions.csv.gz")
# print(train)

# Task 1
def sampleNegative(data, train, valid):
    NegValid = valid

    userRecipe = defaultdict(list)
    print("Preprocessing Data userRecipe ...")

    for index, row in tqdm(data.iterrows()):
        # print(index, row)
        if row['user_id'] not in userRecipe:
            userRecipe[row['user_id']] = {row['recipe_id']}
        else:
            userRecipe[row['user_id']].add(row['recipe_id'])

    print("confirm")
    for index, row in tqdm(valid.iterrows()):
        negValidRecipe = random.sample(set(data['recipe_id']) - userRecipe[row['user_id']], 1)[0]
        NegValid = NegValid.append({'user_id': row['user_id'], 'recipe_id': negValidRecipe, 'date': 0, 'rating': -1, 'valid':-1},
                                   ignore_index=True)

    return NegValid, userRecipe


valid,_ = sampleNegative(data, train, valid)


def baselineOnValidation():
    random.seed(5583)
    data, train, valid = splitDataset("data/trainInteractions.csv.gz")
    print("Sampling Negative samples ...")
    # valid, _ = sampleNegative(data, train, valid)

    print('Training ...')
    recipeCount = defaultdict(int)
    totalCooked = 0

    for user, recipe in tqdm(train.iterrows()):
        print("user: ", user)
        print("recipe: ", recipe)
        recipeCount[recipe['recipe_id']] += 1
        totalCooked += 1

    mostPopular = [(recipeCount[x], x) for x in recipeCount]
    mostPopular.sort()
    mostPopular.reverse()

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

    print('Accuracy on Validation set is %.3f' % (correct/len(valid)))

baselineOnValidation()