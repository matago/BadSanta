from Utils import Graph
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
np.random.seed(50)
from time import time

class anneal(Graph):
    def mutation(self):
        if np.random.rand() >= 0.5:
            _start = np.random.randint(0,len(self.path))
            _stop = np.random.randint(_start,len(self.path))
            self.path[_start:_stop] = np.flip(self.path[_start:_stop],0)
        else:
            p1 = np.random.randint(0,self.path.shape[0])
            p2 = np.random.randint(0,self.path.shape[0])
            self.path = np.insert(self.path[self.path!=self.path[p1]]
                                  ,p2,self.path[p1])



    def cool(self,alpha=0.95,T=1,Tmin=0.0001):
        oldS = self.fitness
        oldP = self.path.copy()
        while T > Tmin:
            for i in range(10*self.n):
                self.mutation()
                newS = self.calc_fitness()
                if np.exp((oldS-newS)/T) > np.random.rand():
                    oldS = newS
                    oldP = self.path.copy()
                else:
                    self.path = oldP.copy()
            T *= alpha
        self.path = oldP
        return oldS

if __name__ == '__main__':
    samples = 1000
    raw = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'cities.csv')).head(samples)
    h = raw['CityId'].values[1:samples]


    a = anneal(raw['X'].values, raw['Y'].values,h)
    print(a.fitness)
    st = time()
    r = a.cool()
    print(time()-st)
    a.tour_plot()
    a.Submit_File(os.path.join(os.getcwd(), 'Sub.csv'), msg='Test Matts Fitness', upload=False)