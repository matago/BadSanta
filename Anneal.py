from Utils import Graph
import numpy as np
import pandas as pd
import os
from time import time
from tqdm import tqdm
np.random.seed(50)



class anneal(Graph):
    def mutation(self):
        locs = np.random.randint(1,self.path.shape[0]-1,(2))
        if np.random.rand() >= 0.5:
            self.path[min(locs):max(locs)] = np.flip(self.path[min(locs):max(locs)],0)
        else:
            self.path = np.insert(self.path[self.path!=self.path[locs[1]]]
                                  ,locs[0],self.path[locs[1]])

    def length(self):
        x = 0
        t = 1
        while t > 0.0001:
            t*=.95
            x+=1
        return x

    def cool(self,alpha=0.95,T=1,Tmin=0.0001,iters=100):
        oldS,original = self.fitness,self.fitness
        oldP = self.path.copy()
        stop,end = 0,0
        pbar = tqdm(total=self.length(),desc=f'Cooling at {int(oldS):,}')
        while T > Tmin:
            for i in tqdm(range(iters)):
                self.mutation()
                newS = self.calc_fitness()
                if np.exp((oldS-newS)/T) > np.random.rand():
                    stop += newS == oldS
                    oldS = newS
                    oldP = self.path.copy()
                    # tqdm.write(str(round(oldS,0)))
                else:
                    stop += 1
                    self.path = oldP.copy()
            T *= alpha
            pbar.update()
            pbar.set_description(f'Cooling at {int(oldS):,}, Savings = {int(original-oldS):,}')
            # print(T,end)
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