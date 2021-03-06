import numpy as np
import pandas as pd
import os, time
from bokeh.plotting import figure, output_file, show


class Graph:
    def __init__(self, vx, vy, seed_path=None):
        self.n = len(vx)

        self.bitpen = (np.mod(np.arange(1, self.n + 1), 10) == 0) * 1

        # Set permutation to include cities 1:n, exclusive of Zero
        if seed_path is None:
            self.path = np.random.permutation(self.n - 1) + 1
        else:
            self.path = seed_path

        self.x = vx
        self.y = vy

        self.prime = primesfrom2to(self.n + 2)

        self.fitness = self.calc_fitness()

    def calc_fitness(self):
        _penalty = np.clip(self.bitpen * np.isin(np.concatenate(([0], self.path)), self.prime, invert=True) * 2, 1, 1.1)

        _fitness = np.sum( _penalty * np.sqrt(
                        np.power(
                            self.x[np.concatenate(([0], self.path))] -
                            self.x[np.concatenate((self.path, [0]))]
                            , 2)

                        + np.power(
                            self.y[np.concatenate(([0], self.path))] -
                            self.y[np.concatenate((self.path, [0]))], 2)
                )
            )
        return _fitness

    def Submit_File(self, file, msg='', upload=False):
        # Prepended 0 to the end on the path
        pd.DataFrame({'Path': np.concatenate(([0], self.path, [0]))}).to_csv(file, index=False)
        if upload:
            command = f"kaggle competitions submit -c traveling-santa-2018-prime-paths -f {file} -m \"{msg}\""
            os.system(command)

    def tour_plot(self):
        output_file("line.html")
        p = figure(plot_width=1000, plot_height=1000)
        p.line(self.x, self.y, line_width=.05)
        p.circle(self.x[self.prime == 0], self.y[self.prime == 0], size=3, color='black')
        show(p)


def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    sieve[0] = False
    for i in range(int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[((k * k) // 3)::2 * k] = False
            sieve[(k * k + 4 * k - 2 * k * (i & 1)) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0] + 1) | 1)]


if __name__ == '__main__':
    raw = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'cities.csv'))
    pth = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'sample_submission.csv'))
    pth = pth['Path'][1:-1].values
    print(pth)
    _t0 = time.time()
    tsp = Graph(raw['X'].values, raw['Y'].values, pth)
    _t1 = time.time()
    print(_t1 - _t0)
    print(tsp.fitness)
    tsp.Submit_File(os.path.join(os.getcwd(), 'Data','sample_submission.csv'), msg='Test Matts Fitness', upload=False)
