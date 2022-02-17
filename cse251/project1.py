import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from sklearn.utils import check_random_state

# mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, cache=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=60000, test_size=10000
)
print(len(X_train))
# digits = datasets.load_digits()
#
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
#
# # Split data into 80% train and 20% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.2, shuffle=True
# )
# print(n_samples)#numpy.ndarray
#0-9

#select subset of the dataset
# prototype = [[0] * 64 for i in range(10)]
# size = [0] * 10
# for i, d in tqdm(enumerate(X_train)):
#     target = int(y_train[i])
#     prototype[target] += d
#     size[target] += 1
# for i in range(10):
#     prototype[i] /= size[i]

# primitive
X_train1, y_train1 = [], []
cnt, m = 0, 10000
count_num = [0] * 10
print("start loop")
for i, d in tqdm(enumerate(X_train)):
    target = int(y_train[i])
    if count_num[target] >= m:
        continue
    X_train1.append(d)
    y_train1.append(y_train[i])
    count_num[target] += 1
X_train1 = np.array(X_train1)
y_train1 = np.array(y_train1)
print("ending loop")


# #select by kmeans
label_arr = [[] for _ in range(10)]
for i, d in tqdm(enumerate(X_train)):
    target = int(y_train[i])
    label_arr[target].append(d)
for i in range(10):
    label_arr[i] = np.array(label_arr[i])
#
import os
os.environ['OMP_NUM_THREADS']="1"
X_train2, y_train2 = [], []
for i in range(10):
    kmeans_tmp = KMeans(n_clusters=m, init='k-means++', random_state=0).fit(label_arr[i])
    sample_tmp = kmeans_tmp.cluster_centers_
    for j in tqdm(range(len(sample_tmp))):
        X_train2.append(sample_tmp[j])
        y_train2.append(str(i))

X_train2 = np.array(X_train2)
y_train2 = np.array(y_train2)
#
#
# #use knn to select subset
label_train = [[] for _ in range(10)]
for i in range(10):
    cnt = 0
    while cnt < 500:
        label_train[i].append(label_arr[i][cnt])
        cnt += 1

split_xtrain, split_ytrain = [], []
for i in range(10):
    for d in label_train[i]:
        split_xtrain.append(d)
        split_ytrain.append(str(i))

knn_split = KNeighborsClassifier(n_neighbors=1)

knn_split.fit(split_xtrain, split_ytrain)

y_pred_split = knn_split.predict(X_train)

# iter_xtrain, iter_ytrain = [], []
# for i in range(len(split_xtrain)):


cnt = 0
X_train3, y_train3 = [], []
for i in range(len(X_train)):
    if cnt > m:
        break
    if y_pred_split[i] == y_train[i]:
        X_train3.append(X_train[i])
        y_train3.append(y_train[i])
    cnt += 1
X_train3, y_train3 = np.array(X_train3), np.array(y_train3)
#
#
#Apply by removing
X_train4, y_train4 = [], []
cnt4 = 0
for i in tqdm(range(len(X_train))):
    if cnt4 > m:
        break
    train_xtmp = list(X_train[:i]) + list(X_train[i+1:])
    train_ytmp = list(y_train[:i]) + list(y_train[i+1:])
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_xtmp, train_ytmp)
    t = [X_train[i]]
    res = knn.predict(t)
    if res == y_train[i]:
        cnt4 += 1
        X_train4.append(X_train[i])
        y_train4.append(y_train[i])

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train4, y_train4)

y_pred = knn.predict(X_test)

print("Accuracy for directly choose:", metrics.accuracy_score(y_test, y_pred))