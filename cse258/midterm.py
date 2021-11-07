# Task 1
import collections
from sklearn import linear_model
import numpy as np
from collections import defaultdict


def parse(f):
    for l in open(f):
        yield eval(l)


path = 'data/trainRecipes.json'
f = open(path, 'rt', encoding="utf8")
dataset = list(parse(path))


train = dataset[:150000]
valid = dataset[150000:175000]
test = dataset[175000:]


# Task 1
def get_onehot(targets, classes):
    res = np.eye(classes, classes - 1)[np.array(targets).reshape(-1)]
    return list(res.reshape(list(targets.shape) + [classes - 1]))


def mse_calc(x, y, theta):
    cnt, res = 0, 0
    while cnt < len(x):
        res += (y[cnt] - np.dot(x[cnt], theta)) ** 2
        cnt += 1
    res = res / len(x)
    return res


def feat1a(d):
    return [len(d['steps']), len(d['ingredients'])]
    # return [len(d['ingredients'])]

# maxyear, minyear = -1, 100000
# for each in dataset:
#     date = each['submitted']
#     date = date.split('-')
#     year = int(date[0])
#     maxyear = maxyear if maxyear >= year else year
#     minyear = minyear if minyear <= year else year
#
# print(maxyear, minyear)
# 2018 1999


def feat1b(d):
    date = d['submitted']
    date = date.split('-')
    year, month, days = date[0], date[1], date[2]
    # 20 11 30
    # year, month = 2018, 12
    year, month, days = get_onehot(np.array(int(year) - 1999), 20), get_onehot(np.array(int(month) - 1),
                                    12), get_onehot(np.array(int(days) - 1), 31)
    return month + year


ingredient_list = collections.defaultdict(int)
for each in train:
    ingredients = each['ingredients']
    for ingre in ingredients:
        ingredient_list[ingre] += 1

ingredient_list = sorted(ingredient_list.items(), key=lambda v: v[1], reverse=True)[:100]
ingredient_name = [k for k, v in ingredient_list]


def feat1c(d):
    res = [0] * 50
    compare = ingredient_name[:50]
    for e in d['ingredients']:
        if e in compare:
            res[compare.index(e)] = 1
    return res


print(feat1a(dataset[0]))
print(feat1b(dataset[0]))
print(feat1c(dataset[0]))

def feat(d, a = True, b = True, c = True):
    X = [1]
    if a:
        X += (feat1a(d))
    if b:
        X += (feat1b(d))
    if c:
        X += (feat1c(d))
    return X

print(feat(dataset[0]))

def experiment(a = True, b = True, c = True, mod='linear_regression'):
    X_train = [feat(d, a, b, c) for d in train]
    y_train = [d['minutes'] for d in train]
    X_test = [feat(d, a, b, c) for d in test]
    y_test = [d['minutes'] for d in test]
    theta, residuals, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)
    print('Coefficient: ', theta)
    print('MSE: ', mse_calc(X_test, y_test, theta))


print(experiment(True, False, False))
print(experiment(False, True, False))
print(experiment(False, False, True))


#Task 2:
#a
print(experiment(True, True, True))
#b
print(experiment(False, True, True))
print(experiment(True, False, True))
print(experiment(True, True, False))


#Task 3:
def experiment_task3(a = True, b = True, c = True, mod='linear_regression'):
    if mod == 'linear_regression':
        train_t = [d for d in train if 0 <= d['minutes'] <= 800]
        test_t = [d for d in test if 15 <= d['minutes'] <= 480]
        X_train = [feat(d, a, b, c) for d in train_t]
        y_train = [d['minutes'] for d in train_t]
        X_test = [feat(d, a, b, c) for d in test]
        y_test = [d['minutes'] for d in test]
        theta, residuals, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)
        print('Coefficient: ', theta)
        print('MSE: ', mse_calc(X_test, y_test, theta))
    else:
        def feat_c(d):
            res = [0] * 50
            compare = ingredient_name[:50]
            for e in d['ingredients']:
                if e == 'butter':
                    continue
                if e in compare:
                    res[ingredient_name.index(e)] = 1
            return res

        X_train = [feat_c(d) for d in train]
        y_train = ['butter' in d['ingredients'] for d in train]
        X_test = [feat_c(d) for d in test]
        y_test = ['butter' in d['ingredients'] for d in test]
        mod = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
        mod.fit(X_train, y_train)
        pred = mod.predict(X_test)

        tp = sum(np.logical_and(pred, y_test))
        fp = sum(np.logical_and(pred, np.logical_not(y_test)))
        tn = sum(np.logical_and(np.logical_not(pred), np.logical_not(y_test)))
        fn = sum(np.logical_and(np.logical_not(pred), y_test))
        #
        print("tp: ", tp, "fp: ", fp, "tn: ", tn, "fn: ", fn)

        label_right, label_false = 0, 0
        for yi in y_test:
            if yi:
                label_right += 1
            else:
                label_false += 1
        BER = (fp/label_false + fn/label_right)/2
        print("The Balanced Label Rate is ", BER)


print(experiment_task3(True, True, True))
#
#
mod = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
print(experiment_task3(False, False, True, mod))


#Task 6
def task6_experment(C, N):
    def feat_c(d, N):
        res = [0] * N
        compare = ingredient_name[:N]
        for e in d['ingredients']:
            if e == 'butter':
                continue
            if e in compare:
                res[compare.index(e)] = 1
        return res

    X_train = [feat_c(d, N) for d in train]
    y_train = ['butter' in d['ingredients'] for d in train]
    X_valid = [feat_c(d, N) for d in valid]
    y_valid = ['butter' in d['ingredients'] for d in valid]
    X_test = [feat_c(d, N) for d in test]
    y_test = ['butter' in d['ingredients'] for d in test]
    mod = linear_model.LogisticRegression(C=C, class_weight='balanced')
    mod.fit(X_train, y_train)
    pred = mod.predict(X_test)
    train_predictions = mod.predict(X_train)
    valid_predictions = mod.predict(X_valid)
    test_predictions = mod.predict(X_test)
    correct_train_predictions = train_predictions == y_train
    correct_valid_predictions = valid_predictions == y_valid
    print("train accuracy: ", sum(correct_train_predictions) / len(correct_train_predictions))
    print("valid accuracy: ", sum(correct_valid_predictions) / len(correct_valid_predictions))
    tp = sum(np.logical_and(pred, y_test))
    fp = sum(np.logical_and(pred, np.logical_not(y_test)))
    tn = sum(np.logical_and(np.logical_not(pred), np.logical_not(y_test)))
    fn = sum(np.logical_and(np.logical_not(pred), y_test))
    #
    print("tp: ", tp, "fp: ", fp, "tn: ", tn, "fn: ", fn)

    label_right, label_false = 0, 0
    for yi in y_valid:
        if yi:
            label_right += 1
        else:
            label_false += 1
    BER = (fp / label_false + fn / label_right) / 2
    print("The Balanced Label Rate is ", BER)


#1, 100 is the best pair
for C in [0.25, 0.5, 1]:
    for N in [25, 50, 100]:
        print("Data for ", C, " ", N)
        task6_experment(C, N)

# Task 8
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


usersPerItem = collections.defaultdict(set)
itemPerUser = collections.defaultdict(set)
for d in dataset:
    user, item = d['recipe_id'], d['ingredients']
    for i in item:
        usersPerItem[user].add(i)
        itemPerUser[i].add(user)


def mostSimilaries(i, top=5):
    similarities = []
    users = usersPerItem[i]

    for i2 in usersPerItem:
        if i2 == i:
            continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim, i2))
    # similarities.sort(reverse=True)
    similarities.sort(key=lambda x: (-x[0], x[1]))
    return similarities[:top]


print(mostSimilaries(dataset[0]['recipe_id']))


#Task 9
target_user = 'butter'

# a
def mostSimilaries_user(i, top=5):
    similarities = []
    items = itemPerUser[i]

    for i2 in itemPerUser:
        if i2 == i:
            continue
        sim = Jaccard(items, itemPerUser[i2])
        # if not itemsPerUser[i2].issubset(items):
        similarities.append((sim, i2))
    similarities.sort(key=lambda x: (-x[0], x[1]))
    return similarities[:top]


# print(mostSimilaries_user(target_user))

#Task 10
user_mind = set(['cherries', 'vodka'])

itemPerRecipe, recipePerItem = collections.defaultdict(set), collections.defaultdict(set)
item = set()
for d in dataset:
    for i in d['ingredients']:
        item.add(i)
        itemPerRecipe[d['recipe_id']].add(i)
        recipePerItem[i].add(d['recipe_id'])

print('vodka' in item)
print('cherries' in item)
append = set()
for i in user_mind:
    max_similarity, max_item = -1, ''
    for it in item:
        if it in user_mind:
            continue
        sim = Jaccard(recipePerItem[it], recipePerItem[i])
        # print(sim, user)
        if sim > max_similarity:
            max_similarity, max_item = sim, it
    append.add(max_item)

for s in append:
    user_mind.add(s)
# user_mind.add('vodka')


max_sim, item_max = -1, ''
for d in dataset:
    recipe = set(d['ingredients'])
    sim = Jaccard(recipe, user_mind)
    if sim > max_sim:
        max_sim = sim
        item_max = d['recipe_id']
        ingre = d['ingredients']
print(max_sim, item_max, ingre)

user_mind = set(['cherries', 'vodka'])
max_sim, item_max = -1, ''
for d in dataset:
    recipe = set(d['ingredients'])
    sim = Jaccard(recipe, user_mind)
    if sim > max_sim:
        max_sim = sim
        item_max = d['recipe_id']
        ingre = d['ingredients']
print(max_sim, item_max, ingre)