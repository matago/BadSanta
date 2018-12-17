import numpy as np
import pandas as pd
import os
import time

class Graph:
    def __init__(self, vx, vy, seed_path=None):
        self.n = len(vx)
        self.bitpen = (np.mod(np.arange(1, self.n+1), 10) == 0) * 1
        print(self.bitpen[0:21])
        # Set permutation to include cities 1:n, exclusive of Zero
        if seed_path is None:
            self.path = np.random.permutation(self.n-1) + 1
        else:
            self.path = seed_path

        self.x = vx
        self.y = vy

        self.prime = sieve(self.n)
        print(self.prime[0:21])
        self.fitness = self.calc_fitness()

    def calc_fitness(self):
        _penalty = np.clip((self.bitpen * self.prime[np.concatenate(([0], self.path))] + 1), 1.0, 1.1)
        print(_penalty[0:21])
        _fitness = np.sum(
            np.sqrt(
                _penalty
                * (
                        np.power(self.x[np.concatenate(([0], self.path))] -
                                 self.x[np.concatenate((self.path, [0]))], 2)

                        + np.power(self.y[np.concatenate(([0], self.path))] -
                                   self.y[np.concatenate((self.path, [0]))], 2)
                )
            )
        )
        return _fitness

    def Submit_File(self, file, msg='', upload=False):
        # Prepended 0 to the end on the path
        pd.DataFrame({'Path': np.concatenate(([0], self.path, [0]))}).to_csv(file, index=False)
        if upload:
            command = "kaggle competitions submit -c traveling-santa-2018-prime-paths -f {file} -m \"{msg}\""
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
    pth = pd.read_csv(os.path.join(os.getcwd(), 'Sub.csv'))
    pth = pth['Path'][1:-1].values
    _t0 = time.time()
    tsp = Graph(raw['X'].values, raw['Y'].values, pth)
    _t1 = time.time()
    print(_t1 - _t0)
    print(tsp.fitness)
    tsp.Submit_File(os.path.join(os.getcwd(), 'Sub.csv'), msg='Test Matts Fitness', upload=False)
