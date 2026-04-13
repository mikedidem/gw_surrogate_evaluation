#!/usr/bin/env python
"""
model.py
--------
Physics Informed Neural Network for solving Poisson equation
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from problem import Problem
from options import Options


class Net(nn.Module):
    """
    Basic Network for PINNs
    """

    def __init__(self, args, stage=1):
        """
        Initialization for Net
        """
        super().__init__()
        self.args = args
        self.tau = args.tau  # watershed of two stages
        self.layers = args.layers  # layers of network
        self.scale = args.scale
        self.device = args.device
        self.constraint = args.constraint
        self.problem = args.problem
        self.stage = stage
        self.fcs = []
        self.params = []

        for i in range(len(self.layers) - 2):
            fc = nn.Linear(self.layers[i], self.layers[i+1])
            setattr(self, f'fc{i+1}', fc)
            self._init_weights(fc)
            self.fcs.append(fc)

            param = nn.Parameter(torch.randn(self.layers[i+1]))
            setattr(self, f'param{i+1}', param)
            self.params.append(param)

        fc = nn.Linear(self.layers[-2], self.layers[-1])
        setattr(self, f'fc{len(self.layers)-1}', fc)
        self._init_weights(fc)
        self.fcs.append(fc)

    def _init_weights(self, layer):
        init.xavier_normal_(layer.weight)
        init.constant_(layer.bias, 0.01)

    def forward(self, xyt, hstar=None):

        xmin, xmax, ymin, ymax, tmin, tmax = self.problem.domain
        if self.stage == 1:
            tmax = self.tau
        elif self.stage == 2:
            tmin = self.tau

        # Normalized
        lb = torch.from_numpy(
            np.array([xmin, ymin, tmin])).float().to(self.device)
        ub = torch.from_numpy(
            np.array([xmax, ymax, tmax])).float().to(self.device)
        X = 2.0 * (xyt - lb) / (ub - lb) - 1.0

        X = self.fcs[0](X)
        X = torch.mul(self.params[0], X) * self.scale
        X = torch.sin(X)

        for i in range(1, len(self.fcs)-1):
            res = self.fcs[i](X)
            res = torch.mul(self.params[i], res) * self.scale
            res = torch.sin(res)
            X = X + res

        u = self.fcs[-1](X)

        if self.constraint == 'SOFT':
            h = u

        elif self.constraint == 'HARD':
            if hstar is None:
                hstar = 90
            else:
                hstar = hstar

            y = xyt[:, [1]]
            t = xyt[:, [2]]
            d = ((t-tmin) * (ymax-y) * (y-ymin)) / \
                ((tmax-tmin) * (ymax-ymin)**2)
            h = hstar + d * u

        return h


class Net_Neumann(nn.Module):
    """Network for Neumann boundary"""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.stage = net.stage

    def forward(self, xyt_bdy2, hstar_x_bdy2=None):

        xyt_bdy2.requires_grad_(True)
        h = self.net(xyt_bdy2)

        w = torch.ones_like(xyt_bdy2[:, [0]])
        h_x_bdy2 = torch.autograd.grad(
            h, xyt_bdy2, w, create_graph=True)[0][:, [0]]

        xyt_bdy2.detach_()

        if self.stage == 1:
            return h_x_bdy2
        elif self.stage == 2:
            return hstar_x_bdy2 + h_x_bdy2


class Net_PDE(nn.Module):
    """Network for PDE"""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.problem = net.problem
        self.stage = net.stage
        self.device = net.device

    def forward(self, xyt, hstar_diff=None, out_diff=False):
        """
        Parameters:
        -----------
        xyt: (n, 3) tensor
            interior points

        hstar_diff: (n, 3) tensor
            if None, train the first stage
            if not None, train the second stage, 
            and diffusion of hstar should provided

        out_diff: (bool)
            If it is True, output the diffusion term. 
            It only used for the first stage.

        Note: You can only choose one of hstar_diff and out_diff.
        """
        xyt.requires_grad_(True)
        h = self.net(xyt)

        w = torch.ones_like(xyt[:, [0]])

        h_grad = torch.autograd.grad(h, xyt, w, create_graph=True)[0]
        h_x, h_y = h_grad[:, [0]], h_grad[:, [1]]

        h_xx = torch.autograd.grad(
            h_x, xyt, w, create_graph=True)[0][:, [0]]

        h_yy = torch.autograd.grad(
            h_y, xyt, w, create_graph=True)[0][:, [1]]

        h_t = h_grad[:, [2]]

        xyt.detach_()

        f = self.problem.f(xyt.cpu().numpy())
        f = torch.from_numpy(f).float().to(self.device)

        mu = self.problem.mu
        K = self.problem.K

        if hstar_diff is not None:
            return mu * h_t - K * (h * (h_xx + h_yy) + h_x * h_x + h_y * h_y) - f + hstar_diff
        else:
            if out_diff:
                return -K * (h * (h_xx + h_yy) + h_x * h_x + h_y * h_y)
            return mu * h_t - K * (h * (h_xx + h_yy) + h_x * h_x + h_y * h_y) - f


class PINN(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.constraint = net.constraint
        self.stage = net.stage
        self.net = net
        self.net_Neumann = Net_Neumann(net)
        self.net_PDE = Net_PDE(net)

    def forward(self, xyt, xyt_bdy2, xy0=None, xyt_bdy1=None,
                hstar=None, hstar_diff=None, hstar_x_bdy2=None,
                out_diff=False):

        h_x_bdy2 = self.net_Neumann(xyt_bdy2, hstar_x_bdy2=hstar_x_bdy2)
        res = self.net_PDE(xyt, hstar_diff=hstar_diff, out_diff=out_diff)

        if self.constraint == 'HARD':
            return res, h_x_bdy2

        elif self.constraint == 'SOFT':
            h0 = self.net(xy0)
            h_bdy1 = self.net(xyt_bdy1)
            return res, h0, h_bdy1, h_x_bdy2


if __name__ == '__main__':
    args = Options().parse()
    args.problem = Problem(sigma=args.sigma)

    net = Net(args)
    print(net)

    # net_Neumann = Net_Neumann(net)
    # net_pde = Net_PDE(net, problem)
    # pinn = PINN(net, problem)
    # params = list(net.parameters())

    # for name, value in net.named_parameters():
    #     print(name)
    # print(net.param1.shape)
