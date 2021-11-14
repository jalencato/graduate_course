import gzip

import pandas as pd
import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier

# from sklearn.cross_validation import train_test_split


if __name__ == '__main__':

    print("Start read data...")

    time_1 = time.time()

    def splitDataset(datapath):
        f = gzip.open(datapath, 'rt')
        data = pd.read_csv(f)
        train, valid = data, data[400001:]
        return data, train, valid


    data, train, valid = splitDataset("../data/trainInteractions.csv.gz")
    data, train, valid = data.values, train.values, valid.values
    train_features, valid_features = [[d[0], d[1], d[3]] for d in train], [[d[0], d[1], d[3]] for d in valid]
    train_labels, valid_labels = [1 for d in train], [1 for d in valid]
    # features = data[::, 1::]
    # labels = data[::, 0]
    #
    # # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)
    #
    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))
    #
    print('Start training...')
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train_features, train_labels)
    time_3 = time.time()
    print('training cost %f seconds...' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = neigh.predict(valid_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))
    #
    score = neigh.score(valid_features, valid_labels)
    print("The accruacy score is %f" % score)