#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
from options import Options
import os
import scipy.io
import math
from pyDOE import lhs


class Sampler(object):
    def __init__(self, problem, stage=1, tau=1.0):
        self.problem = problem
        self.sigma = self.problem.sigma
        self.xmin, self.xmax, self.ymin, self.ymax, self.tmin, self.tmax = self.problem.domain
        if stage == 1:
            self.tmax = tau
        elif stage == 2:
            self.tmin = tau

    def spatial(self, strategy='UNIFORM', nx=100, ny=100, n=None, filename=None):
        """Genration of spatial sampling points"""

        if strategy == 'UNIFORM':
            x_ = np.linspace(self.xmin, self.xmax, nx)
            y_ = np.linspace(self.ymin, self.ymax, ny)
            x, y = np.meshgrid(x_, y_)
            x, y = x.reshape(-1, 1), y.reshape(-1, 1)
            pts = np.c_[x, y]

            pos_bdy1 = np.where((pts[:, 0] == self.xmin) |
                                (pts[:, 0] == self.xmax))
            xy_bdy1 = pts[pos_bdy1]

            pos_bdy2 = np.where((pts[:, 1] == self.ymin) |
                                (pts[:, 1] == self.ymax))
            xy_bdy2 = pts[pos_bdy2]
            pos_more = np.where((xy_bdy2[:, 0] != self.xmin) &
                                (xy_bdy2[:, 0] != self.xmax))
            xy_bdy2 = xy_bdy2[pos_more]

            pos = np.where((pts[:, 0] != self.xmin) &
                           (pts[:, 0] != self.xmax) &
                           (pts[:, 1] != self.ymin) &
                           (pts[:, 1] != self.ymax))
            xy = pts[pos]

        elif strategy == 'LHS':
            if n is None:
                raise ValueError(
                    'Param n should be provided to generate interior points!')

            lb = np.array([self.xmin, self.ymin])
            ub = np.array([self.xmax, self.ymax])
            xy = lb + (ub - lb) * lhs(2, n)

            bdy2 = self.ymin + (self.ymax - self.ymin) * lhs(1, ny)
            y2_w = self.xmin * np.ones((ny, 1))
            y2_e = self.xmax * np.ones((ny, 1))
            xy_bdy2_w = np.c_[y2_w, bdy2]
            xy_bdy2_e = np.c_[y2_e, bdy2]
            xy_bdy2 = np.r_[xy_bdy2_w, xy_bdy2_e]

            bdy1 = self.xmin + (self.xmax - self.xmin) * lhs(1, nx)
            y1_s = self.ymin * np.ones((nx, 1))
            y1_n = self.ymax * np.ones((nx, 1))
            xy_bdy1_s = np.c_[bdy1, y1_s]
            xy_bdy1_n = np.c_[bdy1, y1_n]
            xy_bdy1 = np.r_[xy_bdy1_s, xy_bdy1_n]

        elif strategy == 'LR':
            if filename is None:
                raise ValueError('a spatial sampling file should be provided!')

            data = scipy.io.loadmat(filename)
            xy_bdy = data['bdy2']  # boundary data
            xy = data['xy2']  # interior data

            # boundary of first kind (dirichlet)
            pos_bdy1 = np.where((xy_bdy[:, 1] == self.ymin) |
                                (xy_bdy[:, 1] == self.ymax))
            xy_bdy1 = xy_bdy[pos_bdy1]

            # boundary of second kind (neumann)
            pos_bdy2 = np.where((xy_bdy[:, 0] == self.xmin) |
                                (xy_bdy[:, 0] == self.xmax))
            xy_bdy2 = xy_bdy[pos_bdy2]
            pos_more = np.where((xy_bdy2[:, 1] != self.ymin) &
                                (xy_bdy2[:, 1] != self.ymax))
            xy_bdy2 = xy_bdy2[pos_more]

        else:
            raise ValueError('the spatial strategy is undefined!')

        return xy, xy_bdy1, xy_bdy2

    def temporal(self, strategy='UNIFORM', n=100, ratio=None):
        """Generation of temporal sampling points"""

        if strategy == 'UNIFORM':
            t = np.linspace(self.tmin, self.tmax, n)

        elif strategy == 'LHS':
            t = self.tmin + (self.tmax - self.tmin) * lhs(1, n)

        elif strategy == 'LR':
            if ratio is None:
                raise ValueError('ratio should be provided!')
            length = self.tmax - self.tmin
            a = length * (ratio - 1) / (ratio**(n-1)-1)
            x = [a * ratio**k for k in range(n-1)]
            t = [self.tmin + sum(x[:k]) for k in range(n)]

        else:
            raise ValueError('the temporal strategy is undefined!')

        return t

    def spatial_temporal(self, xy, t):
        """Generation of spatial_temporal sampling points"""
        X, T = np.meshgrid(xy[:, [0]], t)
        Y, T = np.meshgrid(xy[:, [1]], t)
        return np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T

    def __call__(self,
                 spatial_strategy='UNIFORM',
                 temporal_strategy='UNIFORM',
                 nx=100, ny=100, nt=50, n=None,
                 ratio=None, filename=None):

        if spatial_strategy == 'UNIFORM':
            xy, xy_bdy1, xy_bdy2 = self.spatial(spatial_strategy, nx=nx, ny=ny)

        elif spatial_strategy == 'LHS':
            xy, xy_bdy1, xy_bdy2 = self.spatial(
                spatial_strategy, n=n, nx=nx, ny=ny)

        elif spatial_strategy == 'LR':
            xy, xy_bdy1, xy_bdy2 = self.spatial(
                spatial_strategy, filename=filename)

        if temporal_strategy == 'UNIFORM':
            t = self.temporal(temporal_strategy, n=nt)

        elif temporal_strategy == 'LHS':
            t = self.temporal(temporal_strategy, n=nt)

        elif temporal_strategy == 'LR':
            t = self.temporal(temporal_strategy, n=nt, ratio=ratio)

        xyt = self.spatial_temporal(xy, t[1:])
        xyt_bdy1 = self.spatial_temporal(xy_bdy1, t[1:])
        xyt_bdy2 = self.spatial_temporal(xy_bdy2, t[1:])
        xy0 = self.spatial_temporal(np.vstack((xy, xy_bdy1, xy_bdy2)), 0)
        return xyt, xy0, xyt_bdy1, xyt_bdy2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    args = Options().parse()
    from problem import Problem

    # args.problem         = Problem(sigma=args.sigma)
    problem = Problem()
    sampler = Sampler(problem, stage=2)

    # Generation of spatial sampling points

    # 1. uniform spatial strategy
    # xy, xy_bdy1, xy_bdy2 = sampler.spatial(strategy='UNIFORM', nx=20, ny=20)

    # 2. LHS spatial strategy
    # xy, xy_bdy1, xy_bdy2 = sampler.spatial(strategy='LHS', n=200, nx=20, ny=20)

    # 3. Locally refined spatial strategy
    xy, xy_bdy1, xy_bdy2 = sampler.spatial(
        strategy='LR', filename='./data/well.mat')

    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(xy[:, [0]], xy[:, [1]], s=0.5)
    axes[0].scatter(xy_bdy1[:, [0]], xy_bdy1[:, [1]], s=0.5, color='red')
    axes[0].scatter(xy_bdy2[:, [0]], xy_bdy2[:, [1]], s=0.5, color='orange')
    axes[0].set_aspect('equal', adjustable='box')

    # Generation of temporal sampling points

    # 1. uniform temporal strategy
    t = sampler.temporal(strategy='UNIFORM', n=10)

    # 2. LHS temporal strategy
    # t = sampler.temporal(strategy='LHS', n=5)

    # 3. Locally refined temporal strategy
    # t = sampler.temporal(strategy='LR', n=50, ratio=1.04)
    print(t)

    axes[1].scatter(t, np.zeros_like(t), s=0.5)
    axes[1].set_axis_off()
    plt.show()

    # Generation of spatial-temporal sampling points
    # xyt, xy0, xyt_bdy1, xyt_bdy2 = sampler(spatial_strategy='UNIFORM', nx=10, ny=10,
    #                                        temporal_strategy='UNIFORM', nt=5)

    # xyt, xy0, xyt_bdy1, xyt_bdy2 = sampler(spatial_strategy='LHS', n=50, nx=10, ny=10,
    #                                        temporal_strategy='LHS', nt=5)

    xyt, xy0, xyt_bdy1, xyt_bdy2 = sampler(spatial_strategy='LHS', n=50, nx=10, ny=10,
                                           temporal_strategy='LR', nt=50, ratio=1.04)

    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(xyt[:, 0], xyt[:, 1], xyt[:, 2])

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(xyt_bdy1[:, 0], xyt_bdy1[:, 1], xyt_bdy1[:, 2])

    ax2 = fig.add_subplot(223, projection='3d')
    ax2.scatter(xyt_bdy2[:, 0], xyt_bdy2[:, 1], xyt_bdy2[:, 2])

    ax3 = fig.add_subplot(224, projection='3d')
    ax3.scatter(xy0[:, 0], xy0[:, 1], xy0[:, 2])

    plt.show()
