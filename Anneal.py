from Utils import Graph
import numpy as np
import pandas as pd
import os
import sys
from time import time,sleep
from tqdm import tqdm
np.random.seed(5000)
import numexpr as ne


from profilehooks import profile

class anneal(Graph):
    def randGen(self):
        star =
        points = np.random.randint(1,self.n,(2))
        self.t, self.p = points
        self.start, self.stop = np.sort(points)

    def mutation(self):
        if self.kind == 1:
            self.path[self.start:self.stop] = self.path[self.start:self.stop][::-1]
        else:
            if self.t >= self.p:
                self.path[self.p:self.t + 1] = np.roll(self.path[self.p:self.t + 1], 1)
            else:
                self.path[self.t:self.p + 1] = np.roll(self.path[self.t:self.p + 1], -1)

    # @profile
    def trial(self):
        self.randGen()
        choose = self.T >= np.random.rand()
        if np.random.rand() >= .5:
            if self.start != self.stop:
                self.one,self.two = [self.start-1,self.stop-1,self.start-1,self.start], \
                                    [self.start,self.stop,self.stop-1,self.stop]
                if choose != True:
                    self.step = np.array([self.start,self.stop,self.start,self.stop])
                    # print(self.one)
                    # print(self.two)
                    # print(self.step)
                    # print(self.path)
                    static = self.bitpen[self.start:self.stop-1] == 1
                    # print(static)
                    self.a = self.path[self.start:self.stop][:-1][static]
                    self.b = self.path[self.start:self.stop][1:][static]
                    # print(self.a)
                    # print(self.b)

                    self.c = self.path[self.start:self.stop][1:][::-1][static]
                    self.d = self.path[self.start:self.stop][:-1][::-1][static]
                    # print(self.c)
                    # print(self.d)

                self.kind = 1
                self.subTour(choose)
            else:
                self.kind = 1
                self.delta = 0.0
        else:
            if self.t > self.p:
                self.one = [self.t - 1, self.t, self.p - 1] + [self.p - 1, self.t, self.t - 1]
                self.two = [self.t, self.t + 1, self.p] + [self.t, self.p, self.t + 1]
                if choose != True:
                    self.step = np.array([self.t,self.t+1,self.p] + [self.p,self.p+1,self.t+1])

                    static = self.bitpen[self.p:self.t] == 1

                    self.c = self.path[self.p-1:self.t][:-1][static][(self.p + 1) % 10 == 0:]
                    self.d = self.path[self.p-1:self.t][1:][static][(self.p + 1) % 10 == 0:]

                    static[-1:] = False

                    self.a = self.path[self.p:self.t + 1][:-1][static]
                    self.b = self.path[self.p:self.t + 1][1:][static]

                self.kind = 2
                self.subTour(choose)
            elif self.t == self.p:
                self.delta = 0
                self.kind = 2
            else:
                self.one = [self.t - 1, self.t, self.p] + [self.t - 1, self.p, self.t]
                self.two = [self.t, self.t + 1, self.p + 1] + [self.t + 1, self.t, self.p + 1]
                if choose != True:
                    self.step = np.array([self.t, self.t + 1, self.p+1] + [self.t, self.p, self.p + 1])

                    static = self.bitpen[self.t:self.p] == 1

                    self.a = self.path[self.t:self.p+1][:-1][static][(self.t+1)%10==0:]
                    self.b = self.path[self.t:self.p+1][1:][static][(self.t+1)%10==0:]

                    static[-1:] = False

                    self.c = self.path[self.t + 1:self.p+2][:-1][static]
                    self.d = self.path[self.t + 1:self.p+2][1:][static]

                self.kind = 2
                self.subTour(choose)

    def subTour(self,normal):
        _fitness = np.power(self.x[self.path[self.one]] - \
                                   self.x[self.path[self.two]], 2) +\
                           np.power(self.y[self.path[self.one]] - \
                                    self.y[self.path[self.two]], 2)
        if normal:
            _fitness = np.sqrt(_fitness)
            self.delta = _fitness[:int(len(_fitness)/2)].sum() - _fitness[int(len(_fitness)/2):].sum()
        else:
            _penalty = np.clip((self.step % 10 == 0) *
                               self.prime[self.path[self.one]] * 2, 1, 1.1)
            _fitness = np.sqrt(_fitness) * _penalty

            # _p1 = np.sqrt(np.power(self.x[self.a] - self.x[self.b],2) +
            #                        np.power(self.y[self.a] - self.y[self.b],2)) * \
            #       self.prime[self.a]*.1
            x1,x2 = self.x[self.a], self.x[self.b]
            y1,y2 = self.y[self.a], self.y[self.b]
            pen = self.prime[self.a]*.1
            _p1 = ne.evaluate('pen*sqrt((x1-x2)**2+(y1-y2)**2)')
            # _p2 = np.sqrt(np.power(self.x[self.c] - self.x[self.d], 2) +
            #                        np.power(self.y[self.c] - self.y[self.d], 2)) * \
            #       self.prime[self.c]*.1
            x1, x2 = self.x[self.c], self.x[self.d]
            y1, y2 = self.y[self.c], self.y[self.d]
            pen = self.prime[self.c] * .1

            _p2 = ne.evaluate('pen*sqrt((x1-x2)**2+(y1-y2)**2)')

            self.delta = _fitness[:int(len(_fitness) / 2)].sum() + _p1.sum() - \
                         _fitness[int(len(_fitness) / 2):].sum() - _p2.sum()

    def length(self):
        x = 0
        t = self.T
        while t > 0.0001:
            t *= .95
            x += 1
        return x

    def cool(self,alpha=0.95,T=1,Tmin=0.0001,iters=100):
        self.T = T
        self.total = self.length()
        original = self.calc_fitness()
        pbar = tqdm(total=self.total,desc=f'Cooling at {int(original):,}')
        while self.T > Tmin:
            for i in tqdm(range(iters)):
                self.trial()
                if np.exp((self.delta)/self.T) > np.random.rand() and self.delta != 0:
                    self.mutation()
            self.T *= alpha
            pbar.update()
            newFit = self.calc_fitness()
            pbar.set_description(f'Cooling at {int(newFit):,}, Savings = {int(original-newFit):,}')

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