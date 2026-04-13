#!/usr/bin/env python
import torch
import torch.nn as nn

import time
import os
import argparse
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata
import pandas as pd

from problem import Problem
from dataset import Trainset, Validset
from sampler import Sampler
from model import Net, Net_PDE, PINN
from options import Options


class Tester():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.constraint = args.constraint
        self.stage = args.stage
        self.tau = args.tau
        self.problem = Problem(sigma=args.sigma)

        # Criterion
        self.criterion = nn.MSELoss()

        # Networks
        self.net = Net(self.args, stage=args.stage)
        self.net_pde = Net_PDE(self.net)
        self.pinn = PINN(self.net)

        name = f"{args.constraint}_{args.hidden_layers}x{args.hidden_neurons}_tau:{self.tau:.0f}_S:{args.spatial_strategy}_T:{args.temporal_strategy}_nt:{args.nt}_sigma:{args.sigma:.0f}"
        model_name = f"Stage{args.stage}_{name}"

        best_model = torch.load(
            f'checkpoints/{model_name}/best_model.pth.tar')
        self.pinn.load_state_dict(best_model['state_dict'])
        print(f'Best model loaded successfully!')

        # Dataset
        if self.stage == 1:
            """
            Save hstar preparing for the second stage
            """
            trainset = Trainset(self.problem,
                                spatial_strategy='LR', filename='../data/well.mat')
            xy, _, xy_bdy2 = trainset.spatial()
            tau = trainset.temporal(self.tau)

            xytau = trainset.spatial_temporal(xy, tau)
            xytau_bdy2 = trainset.spatial_temporal(xy_bdy2, tau)
            xytau_valid = Validset(self.problem, 100, 100, self.tau)()

            xytau = torch.from_numpy(xytau).float()
            xytau_bdy2 = torch.from_numpy(xytau_bdy2).float()

        elif self.stage == 2:
            model_name_stage1 = f"Stage1_{name}"

            ##########################################################################
            # Read information of hstar_valid, including hstar_valid, hstar_valid_diff,
            # which is used for validating the second stage
            ##########################################################################
            hstar = np.genfromtxt(os.path.join(
                'checkpoints', model_name_stage1, 'hstar_valid.csv')).reshape(-1, 1)
            hstar_diff = np.genfromtxt(os.path.join(
                'checkpoints', model_name_stage1, 'hstar_valid_diff.csv')).reshape(-1, 1)

            hstar = np.tile(hstar, (4, 1))
            hstar_diff = np.tile(hstar_diff, (4, 1))

            hstar = torch.from_numpy(hstar).float()
            hstar_diff = torch.from_numpy(hstar_diff).float()
