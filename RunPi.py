import pandas as pd
import numpy as np
import os
from Anneal import anneal
from time import time

print(os.getcwd())

it = input('How Many Iterations should i run? (must be an int!)')
raw = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'cities.csv'))


a = anneal(raw['X'].values, raw['Y'].values)
o = float(a.fitness)
st = time()
r = a.cool(alpha=0.95,T=1,Tmin=0.0001,iters=int(it))
print(time()-st)
print(f'Original Fitness = {o}')
print(f'New Fitness = {a.calc_fitness()}')
print(f'Savings = {o-a.calc_fitness()}')

a.tour_plot()

a.Submit_File(os.path.join(os.getcwd(), 'SubPi.csv'), msg='Test Matts Fitness', upload=False)