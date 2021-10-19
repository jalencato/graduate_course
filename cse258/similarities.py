import gzip
import math
from collections import defaultdict
import json

path = 'data/goodreads_reviews_comics_graphic.json.gz'
f = gzip.open(path, 'rt', encoding="utf8")

dataset = []
for line in f:
    data = json.loads(line)
    data['rating'] = int(data['rating'])
    data['n_votes'] = int(data['n_votes'])
    data['n_comments'] = int(data['n_comments'])
    dataset.append(data)

# Task 1
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)
ratingDict = {}

for d in dataset:
    user, item = d['user_id'], d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user, item)] = d['n_votes']


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


def mostSimilaries(i, top=10):
    similarities = []
    users = usersPerItem[i]

    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim, i2))
    similarities.sort(reverse=True)
    return similarities[:top]


print(mostSimilaries(dataset[0]['book_id']))


# Task 2
target_user = 'dc3763cdb9b2cae805882878eebb6a32'

# a
maxinf = -1
for i in itemsPerUser[target_user]:
    if ratingDict[(target_user, i)] >= maxinf:
        maxinf = ratingDict[(target_user, i)]
        target_id = i


# strange to say the item of target user has only 1 item.


print(mostSimilaries(target_id))


# b
def mostSimilaries_user(i, top=10):
    similarities = []
    items = itemsPerUser[i]

    for i2 in itemsPerUser:
        if i2 == i: continue
        sim = Jaccard(items, itemsPerUser[i2])
        if not itemsPerUser[i2].issubset(items):
            similarities.append((sim, i2))
    similarities.sort(reverse=True)
    return similarities[:top]


recommend_user = mostSimilaries_user(target_user)
print(recommend_user)
recommend_user = [k[1] for k in recommend_user]
print(recommend_user)

recommend = set()
for user in recommend_user:
    for i in itemsPerUser[user]:
        if ratingDict[(user, i)] >= maxinf:
            maxinf = ratingDict[(user, i)]
            target_id = i
    recommend.add(target_id)
print(recommend)


# Task 3
userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u, i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)

for i in usersPerItem:
    rs = [ratingDict[(u, i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)


def Pearson_1(i1, i2):
    iBar1, iBar2 = itemAverages[i1], itemAverages[i2]
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer, denom1, denom2 = 0, 0, 0
    for u in inter:
        numer += (ratingDict[(u, i1)] - iBar1) * (ratingDict[(u, i2)] - iBar2)
    for u in inter:  # usersPerItem[i1]:
        denom1 += (ratingDict[(u, i1)] - iBar1) ** 2
        # for u in usersPerItem[i2]:
        denom2 += (ratingDict[(u, i2)] - iBar2) ** 2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


def Pearson_2(i1, i2):
    iBar1, iBar2 = itemAverages[i1], itemAverages[i2]
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer, denom1, denom2 = 0, 0, 0
    for u in inter:
        numer += (ratingDict[(u, i1)] - iBar1) * (ratingDict[(u, i2)] - iBar2)
    for u in usersPerItem[i1]:
        denom1 += (ratingDict[(u, i1)] - iBar1) ** 2
    for u in usersPerItem[i2]:
        denom2 += (ratingDict[(u, i2)] - iBar2) ** 2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


def mostSimilaries_Pearson(i, top=10, mode=1):
    similarities = []
    users = usersPerItem[i]

    for i2 in usersPerItem:
        if i2 == i: continue
        if mode == 1:
            sim = Pearson_1(i, i2)
        else:
            sim = Pearson_2(i, i2)
        similarities.append((sim, i2))
    similarities.sort(reverse=True)
    return similarities[:top]


print(mostSimilaries_Pearson(dataset[0]['book_id'], 10, 1))
print(mostSimilaries_Pearson(dataset[0]['book_id'], 10, 2))
