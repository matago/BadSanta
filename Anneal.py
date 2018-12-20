from Utils import Graph
import numpy as np
import pandas as pd
import os
from time import time
np.random.seed(50)



class anneal(Graph):
    def mutation(self):
        locs = np.random.randint(1,self.path.shape[0]-1,(2))
        if np.random.rand() >= 0.5:
            self.path[min(locs):max(locs)] = np.flip(self.path[min(locs):max(locs)],0)
        else:
            self.path = np.insert(self.path[self.path!=self.path[locs[1]]]
                                  ,locs[0],self.path[locs[1]])

    def cool(self,alpha=0.95,T=1,Tmin=0.0001):
        oldS = self.fitness
        oldP = self.path.copy()
        stop,end = 0,0
        while T > Tmin:
            for i in range(100*self.n):
                self.mutation()
                newS = self.calc_fitness()
                if np.exp((oldS-newS)/T) > np.random.rand():
                    stop += newS == oldS
                    oldS = newS
                    oldP = self.path.copy()
                else:
                    stop += 1
                    self.path = oldP.copy()
            T *= alpha
            print(T,end)
        self.path = oldP
        return oldS

if __name__ == '__main__':
    samples = 50
    raw = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'cities.csv')) #.head(samples)
    # h = np.concatenate([raw['CityId'].values[:samples],[0]])


    a = anneal(raw['X'].values, raw['Y'].values)
    o = float(a.fitness)
    st = time()
    r = a.cool()
    print(time()-st)
    print(f'Original Fitness = {o}')
    print(f'New Fitness = {a.calc_fitness()}')
    print(f'Savings = {o-a.calc_fitness()}')
    a.tour_plot()
    a.Submit_File(os.path.join(os.getcwd(), 'Sub.csv'), msg='Test Matts Fitness', upload=False)