import numpy
from urllib.request import urlopen
import scipy.optimize
import random

def parseDataFromURL(fname):
    for l in urlopen(fname):
        yield eval(l)

def parseData(fname):
    for l in open(fname):
        yield eval(l)

print("Reading data...")
# Download from http://cseweb.ucsd.edu/classes/fa19/cse258-a/data/beer_50000.json"
data = list(parseData("data/beer/beer_50000.json"))
# data = list(parseData("http://cseweb.ucsd.edu/classes/fa19/cse258-a/data/beer_50000.json"))
print("done")