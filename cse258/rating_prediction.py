import gzip
import math
from collections import defaultdict
import json
import calendar
import datetime

from sklearn.model_selection import train_test_split

path = 'data/goodreads_reviews_comics_graphic.json.gz'
f = gzip.open(path, 'rt', encoding="utf8")

dataset = []
count = 0

for line in f:
    data = json.loads(line)
    # count += 1
    # if count >= 3:
    #     break
    data['rating'] = int(data['rating'])
    data['n_votes'] = int(data['n_votes'])
    data['n_comments'] = int(data['n_comments'])
    data['date_updated'] = data['date_updated'].split(' ')
    data['date_updated'][0] = list(calendar.day_abbr).index(data['date_updated'][0])
    data['date_updated'][1] = list(calendar.month_abbr).index(data['date_updated'][1])
    data['date_updated'][2] = int(data['date_updated'][2]) + (data['date_updated'][1] - 1) * 30
    if data['date_updated'][4] == '-0800':
        data['date_updated'][2] += 1.0/24
    dataset.append(data)

dataset, _ = train_test_split(dataset, train_size=10000, random_state=42)
#Task 4
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)
ratingDict, time_reviewed = {}, {}

minyear = 123123
for d in dataset:
    user, item = d['user_id'], d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user, item)] = d['rating']
    date_time = datetime.datetime.strptime(d['date_updated'][3], "%H:%M:%S")
    date_time -= datetime.datetime(1900, 1, 1)
    seconds = date_time.total_seconds()
    if minyear > int(d['date_updated'][5]):
        minyear = int(d['date_updated'][5])
    time_reviewed[(user, item)] = [int(d['date_updated'][5]), d['date_updated'][2] + int(seconds)/3600]

userAverages, itemAverages = {}, {}
minyear = int(minyear)

for u in itemsPerUser:
    rs = [ratingDict[(u, i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)

for i in usersPerItem:
    rs = [ratingDict[(u, i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    user, item = d['user_id'], d['book_id']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)

ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


def predictRating(user, item):
    ratings, similarities = [], []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item:
            continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item], usersPerItem[i2]))
    if sum(similarities) > 0:
        weightedRatings = [(x * y) for x, y in zip(ratings, similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


def MSE(predictions, labels):
    differences = [(x-y)**2 for x, y in zip(predictions, labels)]
    return sum(differences) / len(differences)


simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataset]
labels = [d['rating'] for d in dataset]
print(MSE(simPredictions, labels))

#Task 5


def time_factor(user, item, lambda_1='1', lambda_2='1'):
    time = time_reviewed[(user, item)]
    return (1/math.exp(1 + time[1])) + 0.1 * (1/math.exp(time[0] - minyear + 1))



def predictRating_time(user, item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item:
            continue
        ratings.append((d['rating'] - itemAverages[i2]) * time_factor(user, i2))
        similarities.append(Jaccard(usersPerItem[item], usersPerItem[i2]) * time_factor(user, i2))
    if sum(similarities) > 0:
        weightedRatings = [(x * y) for x, y in zip(ratings, similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


simPredictions_time = [predictRating_time(d['user_id'], d['book_id']) for d in dataset]
labels = [d['rating'] for d in dataset]
print(MSE(simPredictions_time, labels))