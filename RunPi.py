import pandas as pd
import numpy as np
import os
from Anneal import anneal
from time import time
np.random.seed(61)
print(os.getcwd())

it = input('How Many Iterations should i run? (must be an int!)\n')
raw = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'cities.csv'))
pth = pd.read_csv(os.path.join(os.getcwd(), 'SubPi.csv'))
a = anneal(raw['X'].values, raw['Y'].values) #,pth['Path'].values)

test = a.calc_fitness()
# a.trial()
# print(a.delta)
# a.mutation()
# print(test-a.calc_fitness())
#
#
#
# tt = a.fitness
# print(tt)
# ls = []
# for dex,i in enumerate(range(int(it))):
#     st = a.calc_fitness()
#     a.trial()
#     a.mutation()
#     ls.append(a.delta)
#     if np.floor(a.delta) != np.floor(st-a.calc_fitness()):
#         print(a.delta)
#         print(st-a.calc_fitness())
#         print(f'T>P = {a.t>=a.p}')
#         print(a.start,a.stop)
#         print('\n')
#         # print(np.min(a.oPen), np.min(a.nPen))
#         # print(np.max(a.oPen),np.max(a.nPen))
#
#
# print(tt-sum(ls),a.calc_fitness())
#
# a.calc_fitness()


o = float(a.fitness)
# st = time()

try:
    r = a.cool(iters=int(it))
except KeyboardInterrupt:
    pass



# # print(time()-st)
print(f'Original Fitness = {o}')
print(f'New Fitness = {a.calc_fitness()}')
print(f'Savings = {o-a.calc_fitness()}')

a.tour_plot()

a.Submit_File(os.path.join(os.getcwd(), 'SubPiMix.csv'), msg='Test Matts Fitness', upload=False)