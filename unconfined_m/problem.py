#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class Problem(object):
    def __init__(self,
                 sigma=25.0,
                 domain=(-500, 500, -500, 500, 0.0, 30.0),
                 xi=(201.67, -98.33)):
        self.sigma = sigma
        self.domain = domain
        self.xi = xi
        self.mu = 0.10
        self.K = 33.33

    def __repr__(self):
        return f'well problem with point source'

    def f(self, x):
        X = x[:, [0]]
        # R = 5.0e-4 * np.ones_like(X)
        f = - 4.0e4 * np.exp(-((x[:, [0]] - self.xi[0])**2 +
                              (x[:, [1]] - self.xi[1])**2) / (2*self.sigma**2)) / (2*np.pi*self.sigma**2)
        return f #+ R

    def bc(self, x, mode=0):
        """boundary/initial condition"""

        if mode == 0:  # initial condition
            u0 = 90.0 * np.ones_like(x[:, [0]])
            return u0

        elif mode == 1:  # Dirichlet boundary condition
            u_bdy1 = 90.0 * np.ones_like(x[:, [0]])
            return u_bdy1

        elif mode == 2:  # Neumann boundary condition
            u_bdy2 = 0.0 * np.ones_like(x[:, [0]])
            return u_bdy2


if __name__ == '__main__':
    from sampler import Sampler
    from dataset import Testset

    problem = Problem(sigma=1e-2)
    sampler = Sampler(problem,  'well.mat')
    xyt, xy0, xyt_bdy1, xyt_bdy2 = sampler()

    f = problem.f(xyt)
    print(f.shape)

    g = problem.bc(xy0, mode='initial')
    print(g.shape)
