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
                            help='RNG seed for weight init and collocation '
                                 'sampling; seeds torch and numpy in '
                                 'train.py/test.py. The default 200 is the '
                                 'value every recorded run used, so leaving it '
                                 'alone reproduces them -- but pass it '
                                 'explicitly so the run record shows it. Any '
                                 'seed other than 200 gets its own '
                                 'checkpoints/ directory (suffix _seed<N>), so '
                                 'two seeds cannot overwrite each other. The '
                                 '1234 used for the 80/20 supervised split in '
                                 'trainer.py is separate and deliberately '
                                 'fixed, so the held-out fifth is the same set '
                                 'across seeds.')
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
        parser.add_argument('--w_data',
                            type=float,
                            default=1.0,
                            help='weight on the supervised MODFLOW loss. '
                                 '0 = pure PINN: no data is loaded at all and '
                                 'the network is trained on physics alone.')
        parser.add_argument('--data_pattern',
                            type=str,
                            default='./modflow/sdata/t*.txt',
                            help='glob for supervised MODFLOW snapshots')
        parser.add_argument('--lbfgs_line_search',
                            action='store_true',
                            help="use strong-Wolfe line search in L-BFGS. "
                                 "Opt-in: omitting it reproduces the published "
                                 "runs. optim.LBFGS is otherwise built without "
                                 "line_search_fn, so it commits the full "
                                 "quasi-Newton step at lr=1 with no way to "
                                 "reject it -- and a sin(scale*Wx) network whose "
                                 "loss needs second derivatives amplifies a bad "
                                 "step by ~(layer gain)^2 per layer. Strong "
                                 "Wolfe backtracks a step that raises the loss "
                                 "BEFORE applying it.")
        parser.add_argument('--lbfgs_max_iter',
                            type=int,
                            default=20,
                            help='L-BFGS inner iterations per epoch. Default 20 '
                                 'matches the published runs; 5-10 takes smaller '
                                 'effective steps in a stiff landscape.')
        parser.add_argument('--lbfgs_div_retries',
                            type=int,
                            default=0,
                            help='how many times the L-BFGS divergence guard '
                                 '(valid > 1.2x best) may restore the last '
                                 'validated weights, clear the curvature '
                                 'history and continue, instead of stopping. '
                                 'Opt-in: 0 reproduces the break-immediately '
                                 'behaviour. The overflow guard already retries '
                                 'twice; this gives divergence the same '
                                 'treatment, which matters because whether a '
                                 'run recovers otherwise depends on which of '
                                 'the two failure modes it happens to hit.')
        parser.add_argument('--alpha_fixed',
                            type=float,
                            default=None,
                            help='pin the supervised-loss weight to this value '
                                 'instead of adapting it from the last-layer '
                                 'gradient-norm ratio. Opt-in: leaving it unset '
                                 'reproduces the adaptive behaviour exactly. Use '
                                 'it to hold the data weight constant across the '
                                 'arms of a controlled sweep -- the adaptive rule '
                                 'RAISES alpha when supervision is noisy, because '
                                 'noise inflates the data gradient norm, which '
                                 'both confounds a noise sweep and upweights the '
                                 'corrupted data. Clamped to [0.05, 0.4] like the '
                                 'adaptive value.')
        # ---- chronological train/validation/test protocol -----------------
        # Opt-in. Without --chrono every path below is skipped and the legacy
        # behaviour is unchanged, so the original experiment still reproduces.
        parser.add_argument('--chrono',
                            action='store_true',
                            default=False,
                            help='Enable the chronological protocol: '
                                 'supervised heads from --train_days, '
                                 'validation from --val_days with RMSE '
                                 'checkpoint selection, and a locked test set '
                                 'from --test_days. PDE collocation is NOT '
                                 'truncated. Writes to a separate output and '
                                 'checkpoint directory so legacy results are '
                                 'untouched.')
        parser.add_argument('--train_days',
                            type=str,
                            default='1-20',
                            help='days whose reference heads supervise '
                                 'training (default 1-20)')
        parser.add_argument('--val_days',
                            type=str,
                            default='21-25',
                            help='days whose reference heads are used for '
                                 'validation and checkpoint selection '
                                 '(default 21-25). Never added to the '
                                 'training loss.')
        parser.add_argument('--test_days',
                            type=str,
                            default='26-30',
                            help='days reserved for the locked final '
                                 'evaluation (default 26-30). Never touch '
                                 'gradients, normalization, checkpoint '
                                 'selection or scheduling.')
        parser.add_argument('--anchor_pattern',
                            type=str,
                            default='./modflow/sdata/t*.txt',
                            help='glob for the supervised observation/anchor '
                                 'snapshots (the existing spatial sampling)')
        parser.add_argument('--field_pattern',
                            type=str,
                            default='./modflow/t*.txt',
                            help='glob for the full reference field snapshots, '
                                 'used for validation and test scoring')
        parser.add_argument('--chrono_outdir',
                            type=str,
                            default='./outputs_chrono_b2',
                            help='output directory for the chronological '
                                 'experiment (kept separate from legacy runs)')

        # Arms that differ only in their DATA -- the noise sweep, the
        # collocation cases -- share every value encoded in model_name
        # and so share a checkpoint directory. This tag separates them.
        parser.add_argument('--run_tag', type=str, default='',
                            help='suffix appended to the checkpoint/output directory '
                                 'name, for arms that differ only in their input '
                                 'data (e.g. noise05). Empty leaves names unchanged. '
                                 'Must match between stage 1 and stage 2.')
        parser.add_argument('--lr',
                            type=float,
                            default=0.001,
                            help='Initial learning rate')
        parser.add_argument('--epochs_Adam',
                            type=int,
                            default=2000,
                            help='Number of epochs for Adam optimizer to train')
        parser.add_argument('--epochs_LBFGS',
                            type=int,
                            default=1000,
                            help='Number of epochs for LBFGS optimizer to train')
        parser.add_argument('--resume',
                            type=str,
                            default=None,
                            help='put the path to resuming file if needed')

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
