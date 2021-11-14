import collections
import warnings

warnings.filterwarnings("ignore")

import gzip
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


def splitDataset(datapath):
    f = gzip.open(datapath, 'rt')
    data = pd.read_csv(f)
    train, valid = data, data[400001:]
    return data, train, valid


data, train, valid = splitDataset("data/trainInteractions.csv.gz")



def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom


userRecipe, recipeUser = defaultdict(set), defaultdict(set)
for index, row in tqdm(data.iterrows()):
    userRecipe[row['user_id']].add(row['recipe_id'])
    recipeUser[row['recipe_id']].add(row['user_id'])

users_rate_count = {}
items_rate_count = {}
user_index = []
item_index = []
users_items = defaultdict(set)

for index, datum in tqdm(data.iterrows()):
    if datum['user_id'] not in users_rate_count:
        users_rate_count[datum['user_id']] = 1
    else:
        users_rate_count[datum['user_id']] += 1
    if datum['recipe_id'] not in items_rate_count:
        items_rate_count[datum['recipe_id']] = 1
    else:
        items_rate_count[datum['recipe_id']] += 1
    if datum['user_id'] not in user_index:
        user_index.append(datum['user_id'])
    if datum['recipe_id'] not in item_index:
        item_index.append(datum['recipe_id'])

for index, datum in tqdm(data.iterrows()):
    u_index = user_index.index(datum['user_id'])
    i_index = item_index.index(datum['recipe_id'])
    users_items[u_index].add(i_index)



#Task 5
predictions = open("kaggle.txt", 'w')
predictions.write('user_id-recipe_id,prediction\n')
def ensemble_kaggle():
    for l in tqdm(open("data/stub_Made.txt")):
        u, r = l.strip().split('-')


        predictions.write(u + '-' + r + ",1\n")

        predictions.write(u + '-' + r + ",0\n")

    # acc.append(correct / len(valid))
    return "finish training"

print(ensemble_kaggle())
