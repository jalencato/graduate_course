import numpy as np
from urllib.request import urlopen
import scipy.optimize

def parseData(fname):
    for l in urlopen(fname):
        yield eval(l)

# data = list(parseData('data/beer/beer_50000.json'))
data = list(parseData(r"https://cseweb.ucsd.edu/classes/fa21/cse258-b/data/beer_50000.json"))
