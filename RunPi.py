import pandas as pd
import numpy as np
import os
from Anneal import anneal
from time import time

print(os.getcwd())

it = input('How Many Iterations should i run? (must be an int!)')
raw = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'cities.csv'))


a = anneal(raw['X'].values, raw['Y'].values)
# tt = a.fitness
# pl = a.fitness
# ls = []
# for i in range(int(it)):
#     st = a.calc_fitness()
#     print(st)
#     o,n = a.mutation()
#     pl -= (o-n)
#     ls.append(o-n)
#     print(o-n)
#     print(st-a.calc_fitness(),'\n')
#
#
# print(tt-sum(ls),pl)
#
# a.calc_fitness()


o = float(a.fitness)
st = time()
r = a.cool(iters=int(it))
print(time()-st)
print(f'Original Fitness = {o}')
print(f'New Fitness = {a.calc_fitness()}')
print(f'Savings = {o-a.calc_fitness()}')

a.tour_plot()

a.Submit_File(os.path.join(os.getcwd(), 'SubPi.csv'), msg='Test Matts Fitness', upload=False)