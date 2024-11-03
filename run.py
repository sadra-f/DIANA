import numpy as np
from Clustering.DIANA import DIANA
import pandas as pd

dataset = list(pd.read_csv("wine-clustering.csv").to_numpy())

def dist_func(x,y):
    temp = x - y
    return (np.sqrt(np.dot(temp.T, temp)))


c = DIANA(4, dist_func)
c.cluster(dataset)
print(c.labels)
# the result for 4 steps for wine-clustering dataset
# [1 1 3 3 2 3 3 3 1 1 3 3 3 1 3 3 3 1 3 2 2 2 1 1 2 2 3 3 1 1 3 3 1 3 1 1 1
#  1 1 2 2 1 1 2 1 1 1 1 1 3 1 3 3 3 1 1 1 3 3 0 2 0 2 4 4 2 0 0 2 2 1 4 0 1
#  1 4 4 0 2 0 4 2 2 0 0 0 0 0 2 2 0 0 0 4 4 1 2 4 2 4 2 0 4 4 2 4 0 0 4 2 0
#  4 2 4 4 4 0 4 4 0 2 0 4 4 4 4 4 0 4 2 2 0 0 2 2 2 2 0 2 2 2 2 0 0 2 2 4 2
#  2 0 0 0 4 2 2 2 0 1 2 2 0 2 0 2 2 0 2 2 2 2 0 0 2 2 2 2 2 0]