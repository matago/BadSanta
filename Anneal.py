from Utils import Graph
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
np.random.seed(50)

class anneal(Graph):
    def mutation(self):
        if np.random.rand() >= .5:
            _start = np.random.randint(1,len(self.path))
            _stop = np.random.randint(_start,len(self.path))
            self.path[_start:_stop] = np.flip(self.path[_start:_stop],0)
        else:
            arr = np.arange(0,len(self.path))
            p1 = np.random.randint(1,len(self.path)-1)
            p2 = np.random.randint(1,len(self.path)-1)


    def cool(self,alpha=0.95,T=1,Tmin=0.0001):
        oldS = self.fitness
        oldP = self.path.copy()
        while T > Tmin:
            for i in range(500):
                self.mutation()
                newS = self.calc_fitness()
                if np.exp((oldS-newS)/T) > np.random.rand():
                    oldS = newS
                    oldP = self.path.copy()
                    plt.plot(self.x[self.path],self.y[self.path],linestyle='-',marker='o')
                    plt.show()
                else:
                    self.path = oldP.copy()
            T *= alpha
        self.path = oldP
        print(self.path)
        return oldS

if __name__ == '__main__':
    raw = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'cities.csv')).head(15)
    a = anneal(raw['X'].values, raw['Y'].values,raw['CityId'].values,Pen=False)
    print(a.fitness)
    r = a.cool()
    print(r)
    a.tour_plot()
    a.Submit_File(os.path.join(os.getcwd(), 'Sub.csv'), msg='Test Matts Fitness', upload=False)