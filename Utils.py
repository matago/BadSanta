import numpy as np
import pandas as pd
import os
import time


class Graph:
    def __init__(self, vx, vy, seed_path=None):
        self.n = len(vx)

        #Set permutation to take place on cities 2-n
        if seed_path is None:
            self.path = np.concatenate([[0],np.random.permutation(np.arange(1,self.n))])
        else:
            self.path = seed_path
        print(self.path)
        self.x = vx
        self.y = vy

        self.prime = sieve(self.n)
        self._fitness = self.calc_fitness()

    def get_fitness(self):
        print("Getting Fitness Value")
        return self._fitness

    def calc_fitness(self):
        _z = -np.argmin(self.path)
        print(len(self.prime),len(self.path))
        _penalty = (np.clip(np.mod(np.arange(self.n), 9), 0, 1) ^ 1) * (self.prime[self.path])
        _penalty[0] = _penalty[1] = 0
        _penalty = np.clip(_penalty + 1.0, 1.0, 1.1)
        self._fitness = np.sum(
            np.sqrt(
                _penalty
                * (
                        np.power(self.x[np.roll(self.path, _z)] -
                                 self.x[np.roll(self.path, _z - 1)], 2)

                        + np.power(self.y[np.roll(self.path, _z)] -
                                    self.y[np.roll(self.path, _z - 1)], 2)
                )
            )
        )
        return self._fitness
    def Submit_File(self,file,msg='',upload=False):
        #Prepended 0 to the end on the path
        pd.DataFrame({'Path':np.concatenate([self.path,[0]])}).to_csv(file,index=False)
        if upload:
            command = f'kaggle competitions submit -c traveling-santa-2018-prime-paths -f {file} -m "{msg}"'
            os.system(command)

def sieve(n):
    flags = np.ones(n, dtype=int)
    flags[0] = flags[1] = 0
    for i in range(2, n):
        # We could use a lower upper bound for this loop, but I don't want to bother with
        # getting the rounding right on the sqrt handling.
        if flags[i] == 1:
            flags[i * i::i] = 0
    return flags ^ 1


if __name__ == '__main__':
    raw = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'cities.csv'))
    _t0 = time.time()
    tsp = Graph(raw['X'].values, raw['Y'].values)
    _t1 = time.time()
    print(_t1 - _t0)
    print(tsp._fitness)
    tsp.Submit_File(os.path.join(os.getcwd(),'Sub.csv'),msg='Test Matts Fitness',upload=False)