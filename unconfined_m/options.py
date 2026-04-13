#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn


class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no_cuda',
                            action='store_true',
                            default=False,
                            help='Disables CUDA training.')
        parser.add_argument('--cuda_index',
                            type=int,
                            default=0,
                            help='Cuda index you want to chooss.')
        parser.add_argument('--seed',
                            type=int,
                            default=200,
                            help='Seed')
        parser.add_argument('--scale',
                            type=float,
                            default=1.0,
                            help='Scale efficient in adaptive activation function')
        parser.add_argument('--hidden_layers',
                            type=int,
                            default=5,
                            help='number of hidden layers')
        parser.add_argument('--hidden_neurons',
                            type=int,
                            default=50,
                            help='number of neurons per hidden layer')
        parser.add_argument('--stage',
                            type=int,
                            default=1,
                            help='training stage')
        parser.add_argument('--tau',
                            type=float,
                            default=1.0,
                            help='watershed of two stages')
        parser.add_argument('--constraint',
                            type=str,
                            default='HARD',
                            help='constraint type (HARD, SOFT)')
        parser.add_argument('--spatial_strategy',
                            type=str,
                            default='LR',
                            help='spatial sampling strategy (UNIFORM, LHS, LR)')
        parser.add_argument('--temporal_strategy',
                            type=str,
                            default='LHS',
                            help='temporal sampling strategy (UNIFORM, LHS, LR) in current stage')
        parser.add_argument('--temporal_strategy_prev',
                            type=str,
                            default='UNIFORM',
                            help='temporal sampling strategy (UNIFORM, LHS, LR) in previous stage')
        parser.add_argument('--n',
                            type=int,
                            default=None,
                            help='number of interior spatial points (used in LHS)')
        parser.add_argument('--nx',
                            type=int,
                            default=None,
                            help='number of spatial points in x direction')
        parser.add_argument('--ny',
                            type=int,
                            default=None,
                            help='number of spatial points in y direction')
        parser.add_argument('--nt',
                            type=int,
                            default=None,
                            help='number of temporal points in current stage')
        parser.add_argument('--nt_prev',
                            type=int,
                            default=None,
                            help='number of temporal points in previous stage')
        parser.add_argument('--ratio',
                            type=float,
                            default=None,
                            help='ratio to generate temporal points')
        parser.add_argument('--filename',
                            type=str,
                            default=None,
                            help='filename to generate locally refined points')
        parser.add_argument('--sigma',
                            type=float,
                            default=30.0,
                            help='sigma in Gaussian function')
        parser.add_argument('--lam',
                            type=float,
                            default=100,
                            help='weight in loss function')
        parser.add_argument('--lr',
                            type=float,
                            default=0.001,
                            help='Initial learning rate')
        parser.add_argument('--epochs_Adam',
                            type=int,
                            default=3000,
                            help='Number of epochs for Adam optimizer to train')
        parser.add_argument('--epochs_LBFGS',
                            type=int,
                            default=1000,
                            help='Number of epochs for LBFGS optimizer to train')
        parser.add_argument('--resume',
                            type=str,
                            default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--w_data',
                            type=float,
                            default=1.0,
                            help='Weight for supervised MODFLOW data loss term')
        parser.add_argument('--data_bs',
                            type=int,
                            default=1024,
                            help='Batch size for MODFLOW supervised data loader')
        parser.add_argument('--data_pattern',
                            type=str,
                            default='./modflow/sdata/t*.txt',
                            help='Glob pattern for MODFLOW snapshot files')
        parser.add_argument('--data_val_frac',
                            type=float,
                            default=0.2,
                            help='Fraction of MODFLOW data held out for validation')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.device = torch.device(
            f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu')
        args.layers = [3] + args.hidden_layers * [args.hidden_neurons] + [1]

        return args


if __name__ == '__main__':
    args = Options().parse()
    print(args)
