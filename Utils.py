import numpy as np
import pandas as pd
import os


class Graph:

    def __int__(self, vx, vy, seed_path=None):
        self.n = len(vx)

        if seed_path is None:
            self.path = np.random.permutation(self.n)
        else:
            self.path = seed_path

        self.x = vx
        self.y = vy

        self.cost = self.fitness

        self.prime = sieve(self.n)

    def fitness(self):
        _from = np.roll(self.path, -np.argmin(self.path))
        _to = np.roll(self.path, -np.argmin(self.path) - 1)
        return np.sum(
            np.sqrt(
                np.power(self.x[_from] - self.x[_to], 2) + np.power(self.y[_from] - self.y[_to], 2)
                )
            )

    def prime_fitness(self):
        _from = np.roll(self.path, -np.argmin(self.path))
        _to = np.roll(self.path, -np.argmin(self.path) - 1)
        _penalty = (np.clip(np.mod(np.arange(self.n), 9), 0, 1) ^ 1) * (self.prime[self.path])
        _penalty[0] = 0
        _penalty = np.clip(_penalty + 1.1, 1.0, 1.1)
        return np.sum(
            np.sqrt(
                _penalty * (np.power(self.x[_from] - self.x[_to], 2) + np.power(self.y[_from] - self.y[_to], 2))
                )
            )


def sieve(n):
    flags = np.ones(n, dtype=int)
    flags[0] = flags[1] = 0
    for i in range(2, n):
        # We could use a lower upper bound for this loop, but I don't want to bother with
        # getting the rounding right on the sqrt handling.
        if flags[i] == 1:
            flags[i * i::i] = 0
    return flags ^ 1


sieve(100)
