
# print("Start saving")
# validFrame = pd.DataFrame(valid)
# validFrame.to_csv('valid.csv')
# print("Start loading...")
# valid = pd.read_csv('valid.csv')

import pandas as pd
from tqdm import tqdm
import gzip


def splitDataset(datapath):
    f = gzip.open(datapath, 'rt')
    data = pd.read_csv(f)
    train, valid = data[:400000], data[400001:]
    return data, train, valid


data, train, _ = splitDataset("data/trainInteractions.csv.gz")

userRecipe = {}
print("Preprocessing Data userRecipe ...")

for index, row in tqdm(data.iterrows()):
    if row['user_id'] not in userRecipe:
        userRecipe[row['user_id']] = {row['recipe_id']}
    else:
        userRecipe[row['user_id']].add(row['recipe_id'])

res = []
valid = pd.read_csv('hw3_val.csv')
for index, row in tqdm(valid.iterrows()):
    d = [row['user'], row['item']]
    if d[1] not in userRecipe[d[0]]:
        res.append({'user_id': d[0], 'recipe_id': d[1], 'date': 0, 'rating': -1},
                                   ignore_index=True)
    else:

    # print(d)