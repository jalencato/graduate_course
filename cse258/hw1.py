import numpy as np
from urllib.request import urlopen
import scipy.optimize


def parseDataURL(fname):
    for l in urlopen(fname):
        yield eval(l)

def parseData(fname):
    for l in open(fname):
        yield eval(l)

print("start reading data")
data = list(parseData('data/fantasy_10000.json'))
# data = list(parseData(r"https://cseweb.ucsd.edu/classes/fa21/cse258-b/data/fantasy_10000.json"))
print("review data have already been loaded")




# f_x =
# X = [feature(d) for d in data]
# y = [d['review/overall'] for d in data]
# theta, residuals, rank, s = np.linalg.lstsq(X, y)
