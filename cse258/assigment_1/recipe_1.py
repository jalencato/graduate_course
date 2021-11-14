import collections
import math
import warnings
from statistics import mean

warnings.filterwarnings("ignore")

import gzip
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import gzip
import random
import scipy
import tensorflow as tf
from collections import defaultdict
from implicit import bpr
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split

def splitDataset(datapath):
    f = gzip.open(datapath, 'rt')
    data = pd.read_csv(f)
    train, valid = data[:400001], data[400001:]
    return data, train, valid


data, train, valid = splitDataset("../data/trainInteractions.csv.gz")
itemsPerUser = defaultdict(list)
usersPerItem = defaultdict(list)
for index, row in tqdm(train.iterrows()):
    itemsPerUser[row['user_id']].append(row['recipe_id'])
    usersPerItem[row['recipe_id']].append(row['user_id'])

userIDs, itemIDs = {}, {}
interactions = []
for index, d in tqdm(data.iterrows()):
    u, i, r = d['user_id'], d['recipe_id'], d['rating']
    if not u in userIDs:
        userIDs[u] = len(userIDs)
    if not i in itemIDs:
        itemIDs[i] = len(itemIDs)
    interactions.append((u, i, r))

nUsers, nItems = len(userIDs), len(itemIDs)

items = list(itemIDs.keys())
class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        # print(u in list(self.gammaI))
        if u in userIDs and i in itemIDs:
            p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        else:
            p = 4.5808
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) + \
                            tf.nn.l2_loss(self.gammaU) + \
                            tf.nn.l2_loss(self.gammaI))

    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))


optimizer = tf.keras.optimizers.Adam(0.1)
modelBPR = BPRbatch(5, 0.00001)
def trainingStepBPR(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            u,i,_ = random.choice(interactions) # positive sample
            j = random.choice(items) # negative sample
            while j in itemsPerUser[u]:
                j = random.choice(items)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])

        loss = model(sampleU,sampleI,sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()


for i in range(100):
    obj = trainingStepBPR(modelBPR, interactions)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))

scoreList = []
predictions = open("kaggle.txt", 'w')
predictions.write('user_id-recipe_id,prediction\n')
cnt = 1
def ensemble():
    for l in tqdm(open("../data/stub_Made.txt")):
        u, r = l.strip().split('-')
        u_test, r_test = int(u), int(r)
        if u_test in userIDs and r_test in itemIDs:
            score = modelBPR.predict(userIDs[u_test], itemIDs[r_test])
        else:
            score = 4.5808
            cnt += 1
        scoreList.append(score)

        if True:
            predictions.write(u + '-' + r + ",1\n")
        else:
            predictions.write(u + '-' + r + ",0\n")

    return "finish training"

print(ensemble())
print(mean(scoreList))
print("cnt: ", cnt)