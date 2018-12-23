import timeit
import numpy as np
s ="""
from Utils import Graph
import pandas as pd
import numpy as np
import os


raw = pd.read_csv(os.path.join(os.getcwd().strip('Tests'), 'Data', 'cities.csv'))
pth = pd.read_csv(os.path.join(os.getcwd().strip('Tests'), 'Data', 'sample_submission.csv'))
pth = pth['Path'].values

tsp = Graph(raw['X'].values, raw['Y'].values, pth)
"""

# ben = timeit.Timer(stmt='tsp.calc_fitness()',setup=s).repeat(7,100)
# print(min(ben))

import numpy as np
import math as m
def nump(x,y):
    np.sqrt(np.power(x[:-1]-x[1:],2)+np.power(y[:-1]-y[1:],2)).sum()

def pys(x1,x2,y1,y2):
    sum([m.sqrt(m.pow(x-xx,2)+m.pow(y-yy,2)) for x,xx,y,yy in zip(x1,x2,y1,y2)])

def differ(x,y):
    np.sqrt(np.power(np.diff(x),2) + np.power(np.diff(y),2)).sum()

