import timeit

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

ben = timeit.Timer(stmt='tsp.calc_fitness()',setup=s).repeat(7,100)
print(min(ben))