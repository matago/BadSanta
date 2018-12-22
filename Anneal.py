from Utils import Graph
import numpy as np
import pandas as pd
import os
import sys
from time import time,sleep
from tqdm import tqdm
np.random.seed(50)
import numexpr as ne


class anneal(Graph):
    def mutation(self):
        points = np.sort(np.random.randint(1, self.path.shape[0] - 1, (2)))
        while np.abs(np.diff(points)) > self.T*self.n:
            points = np.sort(np.random.randint(1, self.path.shape[0] - 1, (2)))
        self.start,self.stop = np.sort(points)
        take, put = points
        if self.T < np.random.rand():
            if np.random.rand() >= 1.0:
                cur = self.path[self.start-1:self.stop+1].copy()
                self.path[self.start:self.stop] = np.flip(self.path[self.start:self.stop],0)
                test = self.path[self.start-1:self.stop+1]
                return self.subTour(cur,0,False), self.subTour(test,0,False)
            else:
                cur = self.path[self.start-1:self.stop+2].copy()
                self.path = np.insert(self.path[self.path != self.path[take]]
                                      ,put,self.path[take])
                test = self.path[self.start - 1:self.stop + 2]
                return self.subTour(cur,1,False), self.subTour(test,1,False)
        else:
            if np.random.rand() >= 0.0:
                cur = self.subTour(self.path[self.start-1:self.start+1].copy(), 0, True) + \
                      self.subTour(self.path[self.stop - 1:self.stop + 1].copy(), 0, True)
                self.path[self.start:self.stop] = np.flip(self.path[self.start:self.stop], 0)
                test = self.subTour(self.path[self.start-1:self.start+1],0,True) + \
                       self.subTour(self.path[self.stop-1:self.stop+1],0,True)
                return cur,test
            else:
                cur = self.subTour(self.path[take:take +2],0,True) + \
                      self.subTour(self.path[put-1:put + 2], 0, True)
                self.path = np.insert(self.path[self.path != self.path[take]]
                                      ,put,self.path[take])
                test = self.subTour(self.path[take:take + 2],0,True) + \
                      self.subTour(self.path[put-1:put + 2], 0, True)
                return cur,test

    def subTour(self,tour,off,normal):
        if normal:
            # _fitness = ne.evaluate("sum(sqrt((x2-x1)**2+(y2-y1)**2))")
            _fitness = np.sqrt(np.power(self.x[tour[:-1]] - self.x[tour[1:]], 2)
                                  + np.power(self.y[tour[:-1]] - self.y[tour[1:]], 2)).sum()
        else:
            x1, x2 = self.x[tour[:-1]], self.x[tour[1:]]
            y1, y2 = self.y[tour[:-1]], self.y[tour[1:]]
            _penalty = np.clip(self.bitpen[self.start-1:self.stop+off] *
                               self.prime[tour[:-1]] * 2, 1, 1.1)
            _fitness = ne.evaluate("sum(_penalty*sqrt(((x2-x1)**2+(y2-y1)**2)))")

            # _fitness = np.sum(_penalty *
            #               np.sqrt(
            #                   np.power(self.x[tour[:-1]] - self.x[tour[1:]], 2)
            #                   + np.power(self.y[tour[:-1]] - self.y[tour[1:]], 2)
            #               ))
        return _fitness

    def length(self):
        x = 0
        t = 1
        while t > 0.0001:
            t*=.95
            x+=1
        return x

    def cool(self,alpha=0.95,T=1,Tmin=0.0001,iters=100):
        self.T = T
        oldSX,original = self.calc_fitness(),self.calc_fitness()
        oldP = self.path.copy()
        pbar = tqdm(total=self.length(),desc=f'Cooling at {int(oldSX):,}')
        while self.T > Tmin:
            for i in tqdm(range(iters)):
                oldS,newS = self.mutation()
                if np.exp((oldS-newS)/self.T) > np.random.rand():
                    oldP = self.path.copy()
                    oldSX -= (oldS-newS)
                else:
                    self.path = oldP.copy()
            self.T *= alpha
            pbar.update()
            newFit = self.calc_fitness()
            pbar.set_description(f'Cooling at {int(newFit):,}, Savings = {int(original-newFit):,}')
            # self.tour_plot()
        self.path = oldP
        # self.tour_plot()

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