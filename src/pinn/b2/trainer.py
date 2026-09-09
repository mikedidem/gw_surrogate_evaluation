#!/usr/bin/env python
from options import Options
from utils import save_checkpoints, show_contours_2x2, mae, mse, rrmse

# show_surface() is commented out in utils.py. It is only a diagnostic plot, so
# fall back to a no-op rather than letting a missing figure stop training.
try:
    from utils import show_surface
except ImportError:
    def show_surface(*_a, **_k):
        return None
from model import Net, Net_Neumann, Net_PDE, PINN
from dataset import Trainset, Validset
from torch.utils.data import Subset
from sampler import Sampler
from problem import Problem
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from dataset import ModflowDataset
from chrono_split import (ChronoSplit, head_metrics, predict_heads, save_json,
                          _compact as compact_days)
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import argparse
import shutil


class _LBFGSNonFinite(RuntimeError):
    """L-BFGS walked the parameters into float32 overflow.

    Raised from inside the L-BFGS closure so the training loop can turn
    it into a clean stop. See train_LBFGS() for why it cannot be handled
    as a rejected step.
    """


class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.cuda_index = args.cuda_index
        self.problem = Problem(sigma=args.sigma)
        self.constraint = args.constraint
        self.stage = args.stage
        self.tau = args.tau

        # Model name. Underscores, not colons: colons are illegal in Windows
        # filenames and this string becomes a directory under checkpoints/.
        name = f"{args.constraint}_{args.hidden_layers}x{args.hidden_neurons}_tau_{self.tau:.0f}_sigma_{args.sigma:.0f}_S_{args.spatial_strategy}"

        self.model_name = f"Stage{self.stage}_{name}_T_{args.temporal_strategy}_nt_{args.nt}"
        if self.stage > 1:
            self.model_name_prev = f"Stage{self.stage-1}_{name}_T_{args.temporal_strategy_prev}_nt_{args.nt_prev}"

        # The seed is part of the run's identity, so it belongs in the
        # checkpoint path. Seed 200 keeps the bare directory name; any other
        # seed gets its own directory, so two seeds of one configuration cannot
        # overwrite each other. The suffix is applied to model_name_prev as
        # well, so a stage-2 run inherits from the stage 1 of its own seed.
        self.seed = getattr(args, 'seed', 200)
        if self.seed != 200:
            self.model_name = self.model_name + f"_seed{self.seed}"
            if self.stage > 1:
                self.model_name_prev = self.model_name_prev + f"_seed{self.seed}"

        # ---- chronological protocol (opt-in via --chrono) -----------------
        # A _CHRONO suffix keeps this experiment's checkpoints in their own
        # directory, so a legacy best_model.pth.tar is never overwritten. The
        # suffix is applied to model_name_prev as well, so a stage-2 chrono run
        # inherits from the stage-1 chrono run rather than from a legacy one.
        self.chrono = bool(getattr(args, 'chrono', False))
        self.split = None
        self.chrono_history = []
        self.chrono_best = None
        self.chrono_norm = None
        self._chrono_last = None
        self._chrono_epoch = 0
        self._chrono_pre_shifted = False
        if self.chrono:
            self.model_name = self.model_name + "_CHRONO"
            if self.stage > 1:
                self.model_name_prev = self.model_name_prev + "_CHRONO"
        # pure-PINN runs get their own checkpoint dir so they cannot be
        # confused with hybrid runs of the same configuration
        if getattr(args, 'w_data', 1.0) == 0:
            self.model_name = self.model_name + "_PURE"
            if self.stage > 1:
                self.model_name_prev = self.model_name_prev + "_PURE"

        # Free-form tag for arms that differ only in their DATA, not in
        # any value encoded above -- the noise sweep, the collocation
        # cases. Without it noise00/noise05/noise10 all resolve to ONE
        # checkpoint directory and overwrite each other; that already
        # happened once and was caught only by renaming directories by
        # hand between arms. Applied to model_name_prev too, so a stage-2
        # run inherits from the stage 1 of its OWN arm. Empty by default,
        # so every name already on disk is unchanged.
        run_tag = str(getattr(args, 'run_tag', '') or '').strip()
        if run_tag:
            self.model_name = self.model_name + f"__{run_tag}"
            if self.stage > 1:
                self.model_name_prev = (self.model_name_prev
                                        + f"__{run_tag}")

        # Build the split now, not at train time, so a bad day range or a
        # missing snapshot fails immediately and the protocol is printed before
        # anything runs.
        if self.chrono:
            self.chrono_outdir = os.path.join(
                getattr(args, 'chrono_outdir', './outputs_chrono_b2'),
                self.model_name)
            os.makedirs(self.chrono_outdir, exist_ok=True)
            self.split = ChronoSplit(
                self.problem, stage=self.stage, tau=self.tau,
                train_days=getattr(args, 'train_days', '1-20'),
                val_days=getattr(args, 'val_days', '21-25'),
                test_days=getattr(args, 'test_days', '26-30'),
                anchor_pattern=getattr(args, 'anchor_pattern',
                                       './modflow/sdata/t*.txt'),
                field_pattern=getattr(args, 'field_pattern',
                                      './modflow/t*.txt'))
            print()
            print(self.split.banner())
            print(f'chrono output dir      = {self.chrono_outdir}')
            print(f'chrono checkpoint dir  = checkpoints/{self.model_name}')
            print()

        # Networks
        self.net = Net(self.args, stage=self.stage)
        self.net_neumann = Net_Neumann(self.net)
        self.net_pde = Net_PDE(self.net)
        self.pinn = PINN(self.net)

        if self.stage > 1:
            self.net_prev = Net(self.args, stage=self.stage-1)
            self.net_neumann_prev = Net_Neumann(self.net_prev)
            self.net_pde_prev = Net_PDE(self.net_prev)
            self.pinn_prev = PINN(self.net_prev)

            if self.device == torch.device(type='cuda', index=self.cuda_index):
                self.net_prev.to(self.device)
                self.net_neumann_prev.to(self.device)
                self.net_pde_prev.to(self.device)
                self.pinn_prev.to(self.device)

            self.net_prev.eval()
            self.net_neumann_prev.eval()
            self.net_pde_prev.eval()
            self.pinn_prev.eval()

            # Loading best model
            best_model = torch.load(
                f'checkpoints/{self.model_name_prev}/best_model.pth.tar')
            self.pinn_prev.load_state_dict(best_model['state_dict'])

        # Criterion
        self.criterion = nn.MSELoss()

        # Resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                print(f'Resuming training, loading {args.resume} ...')
                self.pinn.load_state_dict(
                    torch.load(args.resume)['state_dict'])
            else:
                print('input resume error', args.resume)

        # Trainset
        self.trainset = Trainset(self.problem, stage=self.stage, tau=self.tau,
                                 spatial_strategy=args.spatial_strategy,
                                 temporal_strategy=args.temporal_strategy,
                                 n=args.n, nx=args.nx, ny=args.ny, nt=args.nt,
                                 ratio=args.ratio,
                                 # The collocation cloud is a recorded run
                                 # parameter, selected by --filename.
                                 filename=(getattr(args, 'filename', None)
                                           or './data/well.mat'))

        # Validset
        self.validsize = (100, 100)
        # Validation times must lie inside the model's OWN time window, which
        # is (tmin, tau] for stage 1 and [tau, tmax] for stage 2.
        #
        # These were keyed to the stage alone, which is correct for the HARD
        # two-stage schedule (tau = 1) but wrong for a SINGLE-stage run --
        # the only way SOFT can be run. With --stage 1 --tau 30 the model
        # spans 30 days and was validated on the first one: 3.3% of its
        # domain. Since is_best is decided on valid_loss, the saved checkpoint
        # was whichever weights happened to be best over that first day.
        #
        # For tau = 1 this reproduces the original values exactly, so every
        # HARD run is unaffected.
        _t_lo = self.tau if self.stage > 1 else self.problem.domain[4]
        _t_hi = self.problem.domain[5] if self.stage > 1 else self.tau
        if _t_hi >= 20.0:
            # window reaches the late transient: use the stage-2 times, so the
            # SOFT and HARD residuals are reported at the same instants
            time_stamps = [5, 10, 15, 20]
        else:
            time_stamps = [0.25 * _t_hi, 0.5 * _t_hi, 0.75 * _t_hi, _t_hi]
        self.validset = Validset(
            self.problem, self.validsize[0], self.validsize[1], time_stamps)

        if self.stage > 1:
            xy, _, xy_bdy2 = self.trainset.spatial()

            xytau = self.trainset.spatial_temporal(xy, self.tau)
            xytau_bdy2 = self.trainset.spatial_temporal(xy_bdy2, self.tau)

            xytau = torch.from_numpy(xytau).float()
            xytau_bdy2 = torch.from_numpy(xytau_bdy2).float()

            xyt = self.validset()
            xytau_valid = Validset(
                self.problem, self.validsize[0], self.validsize[1], self.tau)()

            if self.device == torch.device(type='cuda', index=self.cuda_index):
                xytau = xytau.to(self.device)
                xytau_bdy2 = xytau_bdy2.to(self.device)
                xytau_valid = xytau_valid.to(self.device)

            ##########################################################################
            # Generate information of hstar, including hstar, hstar_diff, hstar_x_bdy2,
            # which is used for training the second stage
            ##########################################################################
            # Base field h1 AND its spatial derivatives. The old code stored
            # only the value plus an additive correction (hstar_diff), which
            # assumed the base was near-constant -- true for a uniform benchmark,
            # false once a regional gradient is imposed.
            hstar, hstar_x, hstar_y, hstar_lap =                 self.net_pde_prev.base_field(xytau)
            hstar_x_bdy2 = self.net_neumann_prev(xytau_bdy2).detach()

            self.hstar     = hstar.repeat(args.nt-1, 1)
            self.hstar_x   = hstar_x.repeat(args.nt-1, 1)
            self.hstar_y   = hstar_y.repeat(args.nt-1, 1)
            self.hstar_lap = hstar_lap.repeat(args.nt-1, 1)
            self.hstar_x_bdy2 = hstar_x_bdy2.repeat(args.nt-1, 1)

            ##########################################################################
            # Read information of hstar_valid, including hstar_valid, hstar_valid_diff,
            # which is used for validating the second stage
            ##########################################################################
            hv, hv_x, hv_y, hv_lap = self.net_pde_prev.base_field(xytau_valid)

            self.hstar_valid     = hv.repeat(4, 1)
            self.hstar_valid_x   = hv_x.repeat(4, 1)
            self.hstar_valid_y   = hv_y.repeat(4, 1)
            self.hstar_valid_lap = hv_lap.repeat(4, 1)

    def train_info(self, optimizer, epoch, train_loss, valid_loss, tt):
        result = f'{optimizer:5s} '
        result += f'{epoch+1:5d}/{self.epochs_Adam+self.epochs_LBFGS:5d} '
        result += f'train_loss: {train_loss:.4e} '
        result += f'valid_loss: {valid_loss:.4e} '
        result += f'time: {time.time()-tt:5.2f} '
        if optimizer == 'Adam':
            result += f'lr: {self.lr_scheduler.get_last_lr()[0]:.2e}'
        print(result)

    def _next_data_batch(self):
        if self.data_loader is None:
            return None, None
        try:
            xyt_b, h_b = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data_loader)
            xyt_b, h_b = next(self._data_iter)
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xyt_b = xyt_b.to(self.device)
            h_b   = h_b.to(self.device)

        # ---- Align supervised data with the PINN coordinate system ----
        # MODFLOW writes model coords 0..1000; the network works on -500..500.
        # clone first: never edit a dataloader buffer in place.
        #
        # The chrono loader (chrono_split.load_days) has already applied the
        # shift, so shifting again would put every anchor at -1000..0 -- far
        # outside the domain, where the loss is finite but meaningless.
        xyt_b = xyt_b.clone()
        if not getattr(self, '_chrono_pre_shifted', False):
            xyt_b[:, 0] -= 500.0
            xyt_b[:, 1] -= 500.0
        xyt_b.requires_grad = False
        h_b.requires_grad = False
        return xyt_b, h_b

    @torch.no_grad()
    def _hstar_for_batch(self, xyt_batch):
        """
        For HARD Stage-2: compute h*(x,y) at t = tau for the batch’s (x,y).
        This preserves the spatially varying h*.
        """
        if self.stage != 2:
            return None
        # Build (x,y,t=tau) with same x,y as batch
        x = xyt_batch[:, [0]]
        y = xyt_batch[:, [1]]
        t = torch.full_like(x, fill_value=self.tau)
        xytau = torch.cat([x, y, t], dim=1)
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xytau = xytau.to(self.device)
        hstar_b = self.net_prev(xytau).detach()
        return hstar_b


    # =====================================================================
    # Chronological protocol. All of this is inert unless --chrono is given.
    # =====================================================================
    def _setup_chrono_data(self, data_bs):
        """Supervised loader from train days; validation tensors from val days.

        Replaces the pooled 80/20 random split for chrono runs. That split drew
        its held-out fifth from the same snapshots as training, so a validation
        point sat metres from a training point on the same day; here the two
        sets share no day at all.
        """
        # ---- supervised heads: existing anchor sampling, train days only ---
        tr_xyt, tr_h, tr_counts = self.split.train_anchors()
        self.chrono_train_counts = tr_counts

        # Same domain check the legacy path makes: a coordinate mismatch gives
        # a finite loss at points the network never otherwise evaluates.
        x_lo, x_hi, y_lo, y_hi, _, _ = self.problem.domain
        oob = int(((tr_xyt[:, 0] < x_lo) | (tr_xyt[:, 0] > x_hi) |
                   (tr_xyt[:, 1] < y_lo) | (tr_xyt[:, 1] > y_hi)).sum())
        if oob:
            raise ValueError(
                f"{oob} supervised anchor points fall outside the domain "
                f"x[{x_lo}, {x_hi}] y[{y_lo}, {y_hi}] after the -500 shift.")

        tr_xyt_t = torch.from_numpy(tr_xyt).float()
        tr_h_t = torch.from_numpy(tr_h).float()

        # The loader yields PINN coords already, so _next_data_batch must not
        # shift them a second time. That is what _chrono_pre_shifted signals.
        self.mod_ds = TensorDataset(tr_xyt_t, tr_h_t)
        self.mod_train_ds = self.mod_ds
        self._chrono_pre_shifted = True
        self.data_loader = DataLoader(self.mod_ds, batch_size=data_bs,
                                      shuffle=True, drop_last=False)
        self._data_iter = iter(self.data_loader)

        # ---- normalization statistics, from TRAINING heads only ------------
        self.chrono_norm = self.split.normalization(tr_h)

        # ---- validation heads: full reference field, val days --------------
        # Stage 1 of the original HARD curriculum covers only [0, tau] and is
        # an inherited pretraining model. It keeps the legacy in-window physics
        # criterion; the final stage loads held-out heads and selects on their
        # RMSE. This preserves the two-stage model while keeping validation and
        # test labels out of both stages' gradients.
        if self.split.val_days_used:
            # Held on CPU and moved chunk-by-chunk in predict_heads; 5 days x
            # 90k points is 450k rows and need not sit on the GPU.
            va_xyt, va_h, va_counts = self.split.val_field()
            self.chrono_val_xyt = torch.from_numpy(va_xyt).float()
            self.chrono_val_h = va_h
            self.chrono_val_counts = va_counts
        else:
            va_counts = {}
            self.chrono_val_xyt = None
            self.chrono_val_h = None
            self.chrono_val_counts = {}

        # The legacy supervised-validation tensors stay None: under chrono the
        # checkpoint is selected on validation RMSE over the val days, not on
        # a held-out fraction of the training snapshots.
        self.val_xyt_data = None
        self.val_h_data = None

        # ---- guard: no validation or test day may reach the training set ---
        train_t = set(np.unique(tr_xyt[:, 2]).tolist())
        forbidden = set(float(d) for d in
                        self.split.val_days + self.split.test_days)
        leak = sorted(train_t & forbidden)
        if leak:
            raise RuntimeError(
                f"LEAK: validation/test day(s) {leak} present in the "
                f"supervised training tensor.")

        print(f"[Supervised] CHRONO: {len(self.mod_ds):,} anchor points over "
              f"days {compact_days(self.split.train_days_used)} "
              f"({len(tr_counts)} snapshots x "
              f"{len(self.mod_ds)//max(len(tr_counts),1):,}), batch {data_bs}")
        if self.chrono_val_xyt is None:
            print('[Validation] curriculum pretraining stage: checkpoint '
                  'selected by the existing in-window physics validation; '
                  'held-out head validation is reserved for the final stage')
        else:
            print(f"[Validation] {self.chrono_val_xyt.shape[0]:,} "
                  f"reference-field points over days "
                  f"{compact_days(self.split.val_days_used)} -- scored under "
                  f"no_grad, never added to the training loss")
        print(f"[Test]       days {compact_days(self.split.test_days_used)} "
              f"LOCKED: not loaded until test() runs")

        # ---- persist split + normalization metadata ------------------------
        meta = self.split.metadata()
        meta['train_points'] = int(len(self.mod_ds))
        meta['train_points_per_day'] = {str(k): int(v)
                                        for k, v in tr_counts.items()}
        meta['val_points'] = (0 if self.chrono_val_xyt is None else
                              int(self.chrono_val_xyt.shape[0]))
        meta['val_points_per_day'] = {str(k): int(v)
                                      for k, v in va_counts.items()}
        meta['seed'] = self.seed
        meta['w_data'] = self.w_data
        meta['constraint'] = self.constraint
        meta['batch_size'] = data_bs
        save_json(os.path.join(self.chrono_outdir, 'split_metadata.json'), meta)
        save_json(os.path.join(self.chrono_outdir, 'normalization.json'),
                  self.chrono_norm)

    @torch.no_grad()
    def _chrono_validate(self, step, epoch):
        """Validation heads, days 21-25, in metres. Returns the metric dict.

        Decorated no_grad, and every prediction goes through predict_heads
        which is itself no_grad: no validation head can reach a gradient.
        """
        h_pred = predict_heads(self.net, self.chrono_val_xyt, self.device,
                               stage=self.stage, tau=self.tau,
                               net_prev=getattr(self, 'net_prev', None))
        m = head_metrics(self.chrono_val_h, h_pred)

        # per-day breakdown, useful for spotting a run that fits day 21 and
        # drifts by day 25
        per_day, off = {}, 0
        for d in self.split.val_days_used:
            n = self.chrono_val_counts[d]
            per_day[str(d)] = head_metrics(self.chrono_val_h[off:off + n],
                                           h_pred[off:off + n])
            off += n
        m['per_day'] = per_day
        m['epoch'] = int(epoch)
        m['step'] = int(step)

        for k in ('mae', 'mse', 'rmse', 'rrmse_pct', 'r2'):
            self.writer.add_scalar(f'chrono_valid_{k}', m[k], step)
        return m

    @torch.no_grad()
    def _chrono_test(self, best_model=None):
        """Locked final evaluation on the test days, in metres.

        Runs only after the best-validation checkpoint has been reloaded. These
        heads are read here for the first time in the whole run: they never
        entered a loss, a normalization statistic, checkpoint selection or the
        scheduler.
        """
        days = self.split.test_days_used
        if not days:
            print('[Test] no test day falls inside this model\'s time window; '
                  'nothing to score.')
            return None

        print(f'\n=== LOCKED TEST: days {compact_days(days)} '
              f'(reference field, metres) ===')

        diagnostic_prediction_dir = os.path.join(
            self.chrono_outdir, 'test_predictions')
        os.makedirs(diagnostic_prediction_dir, exist_ok=True)
        per_day, preds, trues = {}, [], []
        for d in days:
            xyt, h_true, _ = self.split.test_day_field(d)
            xyt_t = torch.from_numpy(xyt).float()
            h_pred = predict_heads(self.net, xyt_t, self.device,
                                   stage=self.stage, tau=self.tau,
                                   net_prev=getattr(self, 'net_prev', None))
            m = head_metrics(h_true, h_pred)
            per_day[str(d)] = m
            preds.append(h_pred)
            trues.append(h_true)

            # predictions in MODEL coords, so they line up with the MODFLOW
            # snapshot files they came from
            out = np.hstack([xyt[:, :2] + 500.0,
                             np.full((len(xyt), 1), float(d), dtype=np.float32),
                             h_true.reshape(-1, 1),
                             h_pred.reshape(-1, 1)])
            path = os.path.join(self.chrono_outdir,
                                f'test_predictions_t{d}.csv')
            np.savetxt(path, out, delimiter=',', fmt='%.6f',
                       header='x,y,t,h_modflow,h_pinn', comments='')

            # The shared groundwater diagnostics discover whitespace *.txt
            # snapshots and accept x y h fields. Keep the detailed comparison
            # CSV above, and also emit the prediction-only format expected by
            # evaluate_b2.py without requiring any conversion step.
            diagnostic_out = np.hstack([
                xyt[:, :2] + 500.0,
                h_pred.reshape(-1, 1),
            ])
            np.savetxt(
                os.path.join(diagnostic_prediction_dir, f't{d}.txt'),
                diagnostic_out, fmt='%.8f')

            print(f'  t={d:2d}: MAE={m["mae"]:.4f} m  RMSE={m["rmse"]:.4f} m  '
                  f'MSE={m["mse"]:.5f} m2  RRMSE={m["rrmse_pct"]:.4f} %  '
                  f'R2={m["r2"]:.6f}')

        pooled = head_metrics(np.vstack(trues), np.vstack(preds))
        avg = {k: float(np.mean([per_day[str(d)][k] for d in days]))
               for k in ('mae', 'mse', 'rmse', 'rrmse_pct', 'r2')}
        std = {k: float(np.std([per_day[str(d)][k] for d in days]))
               for k in ('mae', 'mse', 'rmse', 'rrmse_pct', 'r2')}

        print(f'\n  --- average over days {compact_days(days)} ---')
        print(f'  MAE   = {avg["mae"]:.4f} m   (sd {std["mae"]:.4f})')
        print(f'  MSE   = {avg["mse"]:.5f} m2  (sd {std["mse"]:.5f})')
        print(f'  RMSE  = {avg["rmse"]:.4f} m   (sd {std["rmse"]:.4f})')
        print(f'  RRMSE = {avg["rrmse_pct"]:.4f} %   (sd {std["rrmse_pct"]:.4f})')
        print(f'  R2    = {avg["r2"]:.6f}      (sd {std["r2"]:.6f})')
        print(f'  pooled over all test points: RMSE = {pooled["rmse"]:.4f} m, '
              f'R2 = {pooled["r2"]:.6f}')
        print('  ------------------------------------------\n')

        summary = {
            'protocol': 'chronological_train_val_test',
            'test_days': days,
            'units': 'metres',
            'per_day': per_day,
            'average': avg,
            'std_across_days': std,
            'pooled': pooled,
            'selected_epoch': (best_model or {}).get('epoch'),
            'selection_criterion': 'validation_rmse_metres',
            'best_valid_rmse_m': (best_model or {}).get('best_loss'),
            'checkpoint': f'checkpoints/{self.model_name}/best_model.pth.tar',
            'diagnostic_prediction_dir': diagnostic_prediction_dir,
            'normalization': (best_model or {}).get('normalization',
                                                    self.chrono_norm),
            'split': self.split.metadata(),
            'test_heads_used_in_training': False,
            'test_heads_used_in_normalization': False,
            'test_heads_used_in_checkpoint_selection': False,
        }
        save_json(os.path.join(self.chrono_outdir, 'test_summary.json'),
                  summary)
        print(f'[Test] wrote {self.chrono_outdir}/test_summary.json and '
              f'{len(days)} prediction file(s)')
        return summary

    def _chrono_record(self, epoch, step, train_loss, is_best):
        """Append to the history and, on an improvement, write the selection.

        best_validation.json is rewritten every time validation RMSE improves,
        so it always names the epoch whose weights are in best_model.pth.tar.
        """
        m = getattr(self, '_chrono_last', None)
        if m is None:
            return
        self.chrono_history.append({
            'epoch': int(epoch),
            'step': int(step),
            'phase': 'Adam' if epoch < self.epochs_Adam else 'LBFGS',
            'train_loss': float(train_loss),
            'valid_pde_mse': float(m.get('valid_pde_mse', float('nan'))),
            'valid_mae_m': m['mae'],
            'valid_mse_m2': m['mse'],
            'valid_rmse_m': m['rmse'],
            'valid_rrmse_pct': m['rrmse_pct'],
            'valid_r2': m['r2'],
            'is_best': bool(is_best),
        })
        if is_best:
            self.chrono_best = {
                'selected_epoch': int(epoch),
                'selected_step': int(step),
                'selection_criterion': 'validation_rmse_metres',
                'validation_days': self.split.val_days_used,
                'valid_mae_m': m['mae'],
                'valid_mse_m2': m['mse'],
                'valid_rmse_m': m['rmse'],
                'valid_rrmse_pct': m['rrmse_pct'],
                'valid_r2': m['r2'],
                'valid_per_day': m['per_day'],
                'valid_pde_mse': m.get('valid_pde_mse'),
                'train_loss_at_selection': float(train_loss),
                'checkpoint': f'checkpoints/{self.model_name}/best_model.pth.tar',
                'units': 'metres',
            }
            save_json(os.path.join(self.chrono_outdir,
                                   'best_validation.json'), self.chrono_best)
        self._chrono_save_history()

    def _chrono_save_history(self):
        path = os.path.join(self.chrono_outdir, 'history.csv')
        with open(path, 'w') as fh:
            fh.write('epoch,step,phase,train_loss,valid_pde_mse,'
                     'valid_mae_m,valid_mse_m2,valid_rmse_m,valid_rrmse_pct,'
                     'valid_r2,is_best\n')
            for r in self.chrono_history:
                fh.write('%d,%d,%s,%.8e,%.8e,%.8f,%.8f,%.8f,%.6f,%.8f,%d\n' % (
                    r['epoch'], r['step'], r['phase'], r['train_loss'],
                    r['valid_pde_mse'], r['valid_mae_m'], r['valid_mse_m2'],
                    r['valid_rmse_m'], r['valid_rrmse_pct'], r['valid_r2'],
                    int(r['is_best'])))
        return path

    def train(self):

        # Hyperparameters Setting
        self.epochs_Adam = self.args.epochs_Adam
        self.epochs_LBFGS = self.args.epochs_LBFGS
        self.lam = self.args.lam

        # Writer
        self.writer = SummaryWriter(comment=f'_{self.model_name}')

        # Optimizer
        self.lr = self.args.lr
        self.optimizer_Adam = optim.Adam(
            [param for param in self.pinn.parameters() if param.requires_grad == True],
            lr=self.lr)

        self.lr_scheduler = StepLR(self.optimizer_Adam,
                                   step_size=2000,
                                   gamma=0.1)

        # L-BFGS construction kwargs, shared by the initial build and by both
        # guard rebuilds so a retry cannot silently differ from the first try.
        #
        # --lbfgs_line_search supplies the trial-and-reject machinery whose
        # absence is documented at the non-finite guard below: without it
        # optim.LBFGS commits the full quasi-Newton step at lr=1 blind, and a
        # sin(scale*Wx) network whose loss needs SECOND derivatives amplifies a
        # bad step by ~(layer gain)^2 per layer. Strong-Wolfe enforces Armijo
        # sufficient decrease plus the curvature condition, so a step that
        # raises the loss is backtracked BEFORE it is applied. Opt-in: the
        # default reproduces the published runs exactly.
        self._lbfgs_kw = dict(
            max_iter=int(getattr(self.args, 'lbfgs_max_iter', 20) or 20))
        if getattr(self.args, 'lbfgs_line_search', False):
            self._lbfgs_kw['line_search_fn'] = 'strong_wolfe'
            print(f"[LBFGS] strong_wolfe line search ENABLED, "
                  f"max_iter={self._lbfgs_kw['max_iter']}")
        elif self._lbfgs_kw['max_iter'] != 20:
            print(f"[LBFGS] max_iter={self._lbfgs_kw['max_iter']}")

        self.optimizer_LBFGS = optim.LBFGS(
            [param for param in self.pinn.parameters() if param.requires_grad == True],
            **self._lbfgs_kw)

        print(f'{self.trainset}')

        ###########################################################
        # Generate trainset, including xyt, and (xyt_bdy2, hx_bdy2).
        # if SOFT, also (xy0, u0) and (xyt_bdy1, h_bdy1).
        ###########################################################
        xyt = self.trainset()
        xyt_bdy2, hx_bdy2 = self.trainset(mode=2)

        if self.constraint == 'SOFT':
            xy0, u0 = self.trainset(mode=0)
            xyt_bdy1, h_bdy1 = self.trainset(mode=1)

        best_loss = 1.0e10
        tt = time.time()

        self.pinn.train()
        step = 0


        # --------------------------
        # Supervised MODFLOW data
        # --------------------------
        self.w_data = getattr(self.args, "w_data", 1.0)
        data_bs     = getattr(self.args, "data_bs", 1024)
        data_pat    = getattr(self.args, "data_pattern", "./modflow/sdata/t*.txt")
        val_frac    = getattr(self.args, "data_val_frac", 0.2)

        # Adaptive gradient balancing, from the benchmark-1 trainer.
        # The data and PDE terms are not commensurable: on benchmark 2 the data
        # MSE starts near 21 (heads are 86-100 m) while the converged PDE
        # residual is ~7e-3, so a fixed weight of 1 would make the run a
        # supervised regression with a rounding-error physics term. alpha_ema
        # tracks the ratio of their gradient norms instead, clamped to
        # [0.05, 0.4]. Kept identical to benchmark 1 so the two are comparable.
        self.alpha_ema = 0.5          # start balanced; the EMA converges
        self.ema_momentum = 0.9

        # --alpha_fixed pins the weight instead of adapting it. It is opt-in:
        # when None, every path below is byte-identical to the adaptive rule.
        #
        # It exists because alpha_new = grad_pde / (grad_pde + grad_data), so
        # anything that inflates the data gradient DRIVES ALPHA UP. Observation
        # noise does exactly that, which means the adaptive rule puts *more*
        # weight on the supervised term precisely when the supervision is least
        # trustworthy. In the benchmark-2 noise sweep it settled at the 0.05
        # clamp floor for the clean and 5% arms but at 0.151 for the 10% arm --
        # a threefold difference that both confounded the sweep and is the
        # likely trigger of the non-finite L-BFGS objective that truncated that
        # run. Pin alpha across the arms of a controlled sweep.
        self.alpha_fixed = getattr(self.args, 'alpha_fixed', None)
        if self.alpha_fixed is not None:
            self.alpha_fixed = max(0.05, min(0.4, float(self.alpha_fixed)))
            self.alpha_ema = self.alpha_fixed
            print(f"[alpha] FIXED at {self.alpha_fixed:.4e} "
                  f"(adaptive gradient-norm balancing disabled)")

        # Create a DataLoader that yields mini-batches (x,y,t) -> h.
        # w_data == 0 is the pure-PINN arm: skip the data entirely rather than
        # loading it and multiplying by zero, so a pure run needs no sdata on
        # disk and cannot be accused of having seen the reference solution.
        if self.w_data == 0:
            print("[Supervised] w_data = 0 -> PURE PINN, no MODFLOW data loaded")
            self.mod_ds = None
            self.data_loader = None
            self._data_iter = None
            self.val_xyt_data = None
            self.val_h_data = None
        elif self.chrono:
            self._setup_chrono_data(data_bs)
        else:
            try:
                # stage/tau filter lives in the dataset, as in benchmark 1
                full_ds = ModflowDataset(pattern=data_pat,
                                         stage=self.stage, tau=self.tau)
                N = len(full_ds)

                # 80/20 random split of the supervised points, fixed seed, so
                # the held-out fifth is the same set on every run.
                seed = 1234
                torch.manual_seed(seed)
                np.random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                idx = torch.randperm(N)
                n_val = int(val_frac * N)
                val_idx, train_idx = idx[:n_val], idx[n_val:]

                self.mod_ds        = full_ds
                self.mod_train_ds  = Subset(full_ds, train_idx)
                self.mod_val_ds    = Subset(full_ds, val_idx)

                self.data_loader = DataLoader(self.mod_train_ds,
                                              batch_size=data_bs,
                                              shuffle=True, drop_last=False)
                self._data_iter  = iter(self.data_loader)

                val_loader = DataLoader(self.mod_val_ds,
                                        batch_size=len(self.mod_val_ds),
                                        shuffle=False)
                val_xyt, val_h = next(iter(val_loader))
                # shift to PINN coords, same as _next_data_batch
                val_xyt = val_xyt.clone()
                val_xyt[:, 0] -= 500.0
                val_xyt[:, 1] -= 500.0
                if self.device == torch.device(type='cuda',
                                               index=self.cuda_index):
                    val_xyt = val_xyt.to(self.device)
                    val_h   = val_h.to(self.device)
                self.val_xyt_data = val_xyt
                self.val_h_data   = val_h

                print(f"[Supervised] HYBRID: {N:,} points, "
                      f"{len(train_idx):,} train / {len(val_idx):,} val, "
                      f"batch {data_bs}")

                # A silent coordinate mismatch gives a finite loss at points the
                # network never otherwise evaluates, so check it rather than
                # discover it in the results.
                x_lo, x_hi, y_lo, y_hi, _, _ = self.problem.domain
                oob = ((val_xyt[:, 0] < x_lo) | (val_xyt[:, 0] > x_hi) |
                       (val_xyt[:, 1] < y_lo) | (val_xyt[:, 1] > y_hi)
                       ).sum().item()
                if oob:
                    raise ValueError(
                        f"{oob} supervised points fall outside the domain "
                        f"x[{x_lo}, {x_hi}] y[{y_lo}, {y_hi}] after the -500 "
                        f"shift -- are the sdata files in model coords?")
            except Exception as e:
                # Do NOT swallow this. A hybrid run that quietly loses its data
                # trains as a pure PINN and gets reported as a hybrid.
                print(f"[Supervised] FAILED to load MODFLOW data: {e}")
                raise


        ########################
        # Transfer them to GPU
        ########################
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xyt = xyt.to(self.device)
            xyt_bdy2, hx_bdy2 = xyt_bdy2.to(
                self.device), hx_bdy2.to(self.device)

            if self.constraint == 'HARD':
                if self.stage == 2:
                    self.hstar     = self.hstar.to(self.device)
                    self.hstar_x   = self.hstar_x.to(self.device)
                    self.hstar_y   = self.hstar_y.to(self.device)
                    self.hstar_lap = self.hstar_lap.to(self.device)
                    self.hstar_x_bdy2 = self.hstar_x_bdy2.to(self.device)

            elif self.constraint == 'SOFT':
                xy0, u0 = xy0.to(self.device), u0.to(self.device)
                xyt_bdy1, h_bdy1 = xyt_bdy1.to(
                    self.device), h_bdy1.to(self.device)

            self.net.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        self.xyt = xyt
        self.xyt_bdy2, self.hx_bdy2 = xyt_bdy2, hx_bdy2
        if self.constraint == 'SOFT':
            self.xy0, self.u0 = xy0, u0
            self.xyt_bdy1, self.h_bdy1 = xyt_bdy1, h_bdy1

        # Training
        # Stage1: Training Process using Adam Optimizer
        for epoch in range(self.epochs_Adam):
            train_loss = self.train_Adam(epoch)

            if (epoch + 1) % 100 == 0:
                step += 1
                self._chrono_epoch = epoch
                valid_loss = self.validate(step)
                self.train_info('Adam', epoch, train_loss, valid_loss, tt)
                tt = time.time()

                self.pinn.train()

                is_best = valid_loss < best_loss
                if is_best:
                    best_loss = valid_loss      # <- was never updated, so every
                                                #    checkpoint counted as "best"
                                                #    and best_model.pth.tar was
                                                #    really just the latest one
                state = {
                    'epoch': epoch,
                    'state_dict': self.pinn.state_dict(),
                    'best_loss': best_loss,
                    # Stamp the run parameters into the checkpoint. The
                    # directory name encodes most of them, but not the seed
                    # (a bare name means 200) nor the epoch budget, so a
                    # checkpoint copied out of its directory would otherwise
                    # carry no record of either.
                    'seed': self.seed,
                    'epochs_Adam': self.epochs_Adam,
                    'epochs_LBFGS': self.epochs_LBFGS,
                    'constraint': self.constraint,
                    'stage': self.stage,
                    'model_name': self.model_name,
                }
                if self.chrono:
                    # The protocol requires the normalization statistics and
                    # the split to travel WITH the checkpoint, so a checkpoint
                    # alone is enough to say what it was allowed to see.
                    state['chrono'] = True
                    state['normalization'] = self.chrono_norm
                    state['split'] = self.split.metadata()
                    state['selection_criterion'] = (
                        'in_window_physics_validation_mse'
                        if self.split.is_pretraining_stage
                        else 'validation_rmse_metres')
                    state['valid_metrics'] = getattr(self, '_chrono_last', None)
                save_checkpoints(state, is_best, save_dir=self.model_name)
                if self.chrono:
                    self._chrono_record(epoch, step, train_loss, is_best)

        # Seeded with infinity, not Adam's final loss. Comparing L-BFGS's first
        # loss against a different optimiser's last one is not a stagnation
        # measurement, and it can end the phase on its opening step.
        train_loss_old = float('inf')
        # Stage2: Training Process using LBFGS Optimizer
        lbfgs_resets = 0
        # STOP 2 (divergence) terminated on its first trip while STOP 0
        # (overflow) restores and retries twice. That asymmetry is not
        # principled: the same "restore the last validated weights and rebuild
        # the optimizer" recovery applies to both, and a rebuilt LBFGS scales
        # its first step by min(1, 1/|g|_1), which is exactly the conservative
        # restart a diverged step needs. On the benchmark-2 noise sweep the 5%
        # arm survived THREE failures because they happened to be non-finite
        # (retried) while the 10% arm was killed by its first divergence.
        # --lbfgs_div_retries N gives STOP 2 the same treatment. Default 0
        # reproduces the break-immediately behaviour exactly, so every run
        # already on disk is unaffected.
        lbfgs_div_resets = 0
        lbfgs_div_retries = int(getattr(self.args, 'lbfgs_div_retries', 0) or 0)
        for epoch in range(self.epochs_Adam, self.epochs_Adam + self.epochs_LBFGS):
            # ---- STOP 0: float32 overflow --------------------------
            # The parameters are already ruined when this fires (there
            # is no line search -- see the closure), so reload the last
            # VALIDATED weights and forget the curvature pairs that
            # produced the overflow. A rebuilt LBFGS scales its first
            # step by min(1, 1/|g|_1), which is conservative; that is
            # what makes a retry worth taking rather than just stopping.
            # Capped at 2, then a clean stop. A run that never overflows
            # never reaches any of this, so arms compared against each
            # other are unaffected.
            try:
                train_loss = self.train_LBFGS(epoch)
            except _LBFGSNonFinite as exc:
                ckpt_path = (f"checkpoints/{self.model_name}/"
                             "best_model.pth.tar")
                if not os.path.exists(ckpt_path):
                    print(f"\n[LBFGS] {exc} -- no checkpoint to "
                          "restore. Stopping.")
                    break
                print(f"\n[LBFGS] {exc} -- restoring {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.pinn.load_state_dict(ckpt["state_dict"])
                lbfgs_resets += 1
                if lbfgs_resets > 2:
                    print("[LBFGS] overflow after 2 resets. Stopping.")
                    break
                self.optimizer_LBFGS = optim.LBFGS(
                    [p for p in self.pinn.parameters()
                     if p.requires_grad],
                    **self._lbfgs_kw)
                # A stale train_loss would otherwise be compared against
                # the next epoch and could trip the stagnation break.
                train_loss_old = float('inf')
                print(f"[LBFGS] history cleared, "
                      f"retry {lbfgs_resets}/2.")
                continue

            # ---- STOP 1: physics convergence -------------------------------
            # L2 norm over all parameter gradients. If it is tiny we are at the
            # bottom and further LBFGS steps only chase noise.
            total_norm = 0.0
            for p in self.pinn.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            if total_norm < 1e-7:
                print(f"\n[LBFGS] Converged (Grad Norm {total_norm:.2e} < 1e-7). Stopping early.")
                break

            if (epoch+1) % 20 == 0:

                step += 1
                self._chrono_epoch = epoch
                valid_loss = self.validate(step)
                self.train_info('LBFGS', epoch, train_loss, valid_loss, tt)
                tt = time.time()

                # ---- STOP 2: divergence ------------------------------------
                # LBFGS drives the residual to ~0 at the training collocation
                # points while it can get WORSE everywhere else. Observed on
                # this benchmark: train fell 287x while valid rose 1.7x. The
                # stagnation break below cannot catch that -- a rising loss has
                # large step-to-step changes -- so bail out explicitly.
                if valid_loss > best_loss * 1.2:
                    ckpt_path = (f"checkpoints/{self.model_name}/"
                                 "best_model.pth.tar")
                    if (lbfgs_div_resets < lbfgs_div_retries
                            and os.path.exists(ckpt_path)):
                        lbfgs_div_resets += 1
                        print(f"\n[LBFGS] Divergence detected! Valid Loss "
                              f"{valid_loss:.4e} > 1.2x Best ({best_loss:.4e}) "
                              f"-- restoring {ckpt_path}")
                        ckpt = torch.load(ckpt_path, map_location=self.device)
                        self.pinn.load_state_dict(ckpt["state_dict"])
                        self.optimizer_LBFGS = optim.LBFGS(
                            [p for p in self.pinn.parameters()
                             if p.requires_grad],
                            **self._lbfgs_kw)
                        train_loss_old = float('inf')
                        print(f"[LBFGS] history cleared, divergence retry "
                              f"{lbfgs_div_resets}/{lbfgs_div_retries}.")
                        self.pinn.train()
                        continue
                    print(f"\n[LBFGS] Divergence detected! Valid Loss {valid_loss:.4e} "
                          f"> 1.2x Best ({best_loss:.4e}). Stopping"
                          + (f" after {lbfgs_div_resets} retries." if lbfgs_div_resets
                             else "."))
                    break

                self.pinn.train()

                is_best = valid_loss < best_loss
                if is_best:
                    best_loss = valid_loss      # <- was never updated, so every
                                                #    checkpoint counted as "best"
                                                #    and best_model.pth.tar was
                                                #    really just the latest one
                state = {
                    'epoch': epoch,
                    'state_dict': self.pinn.state_dict(),
                    'best_loss': best_loss,
                    # Stamp the run parameters into the checkpoint. The
                    # directory name encodes most of them, but not the seed
                    # (a bare name means 200) nor the epoch budget, so a
                    # checkpoint copied out of its directory would otherwise
                    # carry no record of either.
                    'seed': self.seed,
                    'epochs_Adam': self.epochs_Adam,
                    'epochs_LBFGS': self.epochs_LBFGS,
                    'constraint': self.constraint,
                    'stage': self.stage,
                    'model_name': self.model_name,
                }
                if self.chrono:
                    # The protocol requires the normalization statistics and
                    # the split to travel WITH the checkpoint, so a checkpoint
                    # alone is enough to say what it was allowed to see.
                    state['chrono'] = True
                    state['normalization'] = self.chrono_norm
                    state['split'] = self.split.metadata()
                    state['selection_criterion'] = (
                        'in_window_physics_validation_mse'
                        if self.split.is_pretraining_stage
                        else 'validation_rmse_metres')
                    state['valid_metrics'] = getattr(self, '_chrono_last', None)
                save_checkpoints(state, is_best, save_dir=self.model_name)
                if self.chrono:
                    self._chrono_record(epoch, step, train_loss, is_best)

            # Stagnation. The threshold was 1e-7, which sits INSIDE the normal
            # step size here: Adam finishes near 9e-04 and L-BFGS steps move the
            # loss by 1e-05 to 1e-06, so the test fired while the optimiser was
            # still descending and ended the phase after a handful of epochs --
            # silently, because this break printed nothing and the run still
            # reported success. 1e-12 is below any genuine step.
            delta = abs(train_loss - train_loss_old)
            if delta < 1.e-12:
                print()
                print(f"[LBFGS] Stagnated at epoch {epoch+1}: loss changed by "
                      f"{delta:.3e} (< 1e-12). Stopping.")
                break
            train_loss_old = train_loss

        self.writer.close()

        if self.chrono:
            hist = self._chrono_save_history()
            print('\n=== CHRONO TRAINING SUMMARY ===')
            print(f'history      -> {hist}')
            if self.chrono_best is not None:
                b = self.chrono_best
                print(f'selected epoch = {b["selected_epoch"]} '
                      f'(validation RMSE, days '
                      f'{compact_days(self.split.val_days_used)})')
                print(f'  valid MAE  = {b["valid_mae_m"]:.4f} m')
                print(f'  valid MSE  = {b["valid_mse_m2"]:.5f} m2')
                print(f'  valid RMSE = {b["valid_rmse_m"]:.4f} m')
                print(f'  valid R2   = {b["valid_r2"]:.6f}')
                print(f'best model   -> {b["checkpoint"]}')
                print(f'metrics      -> {self.chrono_outdir}/best_validation.json')
            elif self.split.is_pretraining_stage:
                print('curriculum stage 1 selected by in-window physics '
                      'validation; stage 2 will perform held-out head '
                      'validation on days 21-25')
                print(f'best model   -> checkpoints/{self.model_name}/'
                      'best_model.pth.tar')
            else:
                print('WARNING: no validation improvement was ever recorded.')
            print('test days remain unread until test.py runs.')
            print('===============================\n')

        print('Training finished successfully!!!\n')

    def train_Adam(self, epoch):
        """
        Training process using Adam optimizer
        """

        self.optimizer_Adam.zero_grad()

        # Forward and backward propogate
        if self.constraint == 'HARD':
            if self.stage == 1:
                res, hx_bdy2_pred = self.pinn(self.xyt,
                                              self.xyt_bdy2)
              
            else:
                res, hx_bdy2_pred = self.pinn(self.xyt,
                                              self.xyt_bdy2,
                                              hstar=self.hstar,
                                              hstar_x=self.hstar_x,
                                              hstar_y=self.hstar_y,
                                              hstar_lap=self.hstar_lap,
                                              hstar_x_bdy2=self.hstar_x_bdy2)
                
              

            loss = self.criterion(res, torch.zeros_like(res))
            loss_bdy2 = self.criterion(hx_bdy2_pred, self.hx_bdy2)

            loss_total = loss + self.lam * loss_bdy2

        elif self.constraint == 'SOFT':
            res, u0_pred, h_bdy1_pred, hx_bdy2_pred = self.pinn(self.xyt,
                                                                self.xyt_bdy2,
                                                                self.xy0,
                                                                self.xyt_bdy1)
            loss = self.criterion(res, torch.zeros_like(res))
            loss0 = self.criterion(u0_pred, self.u0)
            loss_bdy1 = self.criterion(h_bdy1_pred, self.h_bdy1)
            loss_bdy2 = self.criterion(hx_bdy2_pred, self.hx_bdy2)

            loss_total = loss + self.lam * (loss0 + loss_bdy1 + loss_bdy2)



        # --------------------------
        # Supervised batch (if any)
        # --------------------------
        loss_pde = loss                      # name it for the gradient balance
        loss_data = torch.tensor(0.0, device=self.device)
        hstar_b = None
        xyt_b, h_b = self._next_data_batch()
        if xyt_b is not None:
            if self.constraint == 'HARD' and self.stage == 2:
                # Written as the RESIDUAL (h - hstar) so the expression matches
                # what the stage-2 network actually produces: an increment on
                # top of a frozen hstar.
                #
                # NOTE: under MSE this is only a restatement, not a reweighting.
                # criterion is nn.MSELoss, so
                #     mean[((h_pred - hstar) - (h - hstar))^2] = mean[(h_pred - h)^2]
                # -- hstar cancels identically and the loss, its gradient and
                # every number derived from it are the same as the plain
                # criterion(h_pred_b, h_b) in the branch below.
                #
                # An earlier comment here claimed the subtraction stops the
                # network being charged for an inherited stage-1 offset it
                # cannot correct. The algebra forbids that: hstar is a constant
                # with respect to the weights, so subtracting it from both
                # arguments cannot change the objective. Reweighting the
                # supervision would take a different loss (e.g. a relative or
                # per-point-weighted error), not this rearrangement. Do not
                # paraphrase the old claim into the methods section.
                #
                # The form is kept only because it reads as the residual and to
                # leave the recorded runs bit-for-bit reproducible.
                with torch.no_grad():
                    hstar_b = self._hstar_for_batch(xyt_b)
                h_pred_b = self.net(xyt_b, hstar=hstar_b)
                loss_data = self.criterion(h_pred_b - hstar_b, h_b - hstar_b)
            else:
                # stage-1 HARD or any SOFT: plain forward
                h_pred_b = self.net(xyt_b)
                loss_data = self.criterion(h_pred_b, h_b)

            if not torch.isfinite(loss_data):
                print(f"Warning: Non-finite data loss at epoch {epoch}")
                print(f"  h_pred range: [{h_pred_b.min():.4f}, {h_pred_b.max():.4f}]")
                print(f"  h_true range: [{h_b.min():.4f}, {h_b.max():.4f}]")
                loss_data = torch.tensor(0.0, device=self.device)

            # ---- adaptive weighting by gradient norm ----
            # Compared at the output layer only: comparable across terms and
            # cheap enough to run every epoch.
            def last_layer_params(model):
                """Robustly find the last trainable module's parameters."""
                last = None
                for m in model.modules():
                    if any(p.requires_grad for p in m.parameters(recurse=False)):
                        last = m
                return (list(last.parameters()) if last is not None
                        else list(model.parameters()))

            def grad_norm(loss_term):
                try:
                    params = last_layer_params(self.net)
                    g = torch.autograd.grad(loss_term, params,
                                            retain_graph=True,
                                            create_graph=False,
                                            allow_unused=True)
                    return torch.sqrt(sum((gi**2).sum() for gi in g
                                          if gi is not None)).item()
                except Exception:
                    return 0.0

            grad_pde = grad_norm(loss_pde)
            grad_data = grad_norm(loss_data) if loss_data.item() > 0 else 0.0

            # frozen for the first 50 epochs, then EMA, clamped to [0.05, 0.4].
            # --alpha_fixed short-circuits the update and holds the pinned value.
            if self.alpha_fixed is not None:
                self.alpha_ema = self.alpha_fixed
            elif epoch > 50 and grad_data > 1e-8:
                alpha_new = grad_pde / (grad_pde + grad_data + 1e-8)
                self.alpha_ema = (self.ema_momentum * self.alpha_ema
                                  + (1 - self.ema_momentum) * alpha_new)
                self.alpha_ema = max(0.05, min(0.4, self.alpha_ema))
            loss_total = loss_total + self.alpha_ema * loss_data

            self.writer.add_scalar('train_loss_data', loss_data.item(), epoch)
            self.writer.add_scalar('alpha_ema', self.alpha_ema, epoch)

            if (epoch + 1) % 500 == 0:
                grad_bdy = (grad_norm(loss_bdy2)
                            if self.constraint == 'HARD' else 0.0)
                print(f"\n=== Gradient Diagnostics (Epoch {epoch+1}, last-layer) ===")
                print(f"  PDE grad norm:     {grad_pde:.4e}")
                if self.constraint == 'HARD':
                    print(f"  Boundary grad:     {grad_bdy:.4e}  (last-layer only)")
                if grad_data > 0:
                    print(f"  Data grad norm:    {grad_data:.4e}")
                    print(f"  Data loss:         {loss_data.item():.4e}")
                    print(f"  Adaptive alpha:    {self.alpha_ema:.4e}")
                print(f"=======================================\n")

        loss_total.backward()
        self.optimizer_Adam.step()
        self.lr_scheduler.step()
        train_loss = loss_total.item()

        self.writer.add_scalar('train_loss', train_loss, epoch)

        return train_loss
    

    


    def train_LBFGS(self, epoch):
        """
        Training process using LBFGS optimizer
        """
        # Sanity check on entering the LBFGS phase.
        #
        # Must run with gradients enabled: net_pde() differentiates through
        # the network with torch.autograd.grad, so under torch.no_grad() the
        # forward pass yields a tensor with no grad_fn and autograd raises
        # "element 0 of tensors does not require grad and does not have a
        # grad_fn". The result is detached for printing instead.
        #
        # Runs once, on the first LBFGS epoch. It is a pre-LBFGS check and is
        # not free, and calling _next_data_batch() every epoch would consume a
        # supervised batch that the training step below never sees.
        if epoch == self.epochs_Adam:
            xyt_chk = self.xyt[:2048]
            res = (self.net_pde(xyt_chk) if self.stage == 1
                   else self.net_pde(xyt_chk, hstar=self.hstar[:2048],
                                     hstar_x=self.hstar_x[:2048],
                                     hstar_y=self.hstar_y[:2048],
                                     hstar_lap=self.hstar_lap[:2048])).detach()
            print("pre-LBFGS PDE finite:", torch.isfinite(res).all().item(),
                  "| min/max:", res.min().item(), res.max().item(),
                  "| mean abs:", res.abs().mean().item())

            # also check the supervised batch if you use data loss
            with torch.no_grad():
                xyt_b, h_b = self._next_data_batch()
                if xyt_b is not None:
                    h_pred_b = self.net(xyt_b) if self.stage == 1 else self.net(xyt_b, self._hstar_for_batch(xyt_b))
                    ok = torch.isfinite(h_pred_b).all() and torch.isfinite(h_b).all()
                    print("pre-LBFGS data finite:", ok.item())



        # Forward and backward propogate

          # ---- FIX: cache a single supervised batch OUTSIDE the closure ----
        xyt_b, h_b = self._next_data_batch()  # may be (None, None)
        if (xyt_b is not None) and (self.constraint == 'HARD' and self.stage == 2):
            # also cache h* for this batch so closure is deterministic
            hstar_b = self._hstar_for_batch(xyt_b)
        else:
            hstar_b = None

        # Freeze the data weight for the whole LBFGS phase. LBFGS re-evaluates
        # the closure several times per step; a weight that moved between
        # evaluations would make the objective non-stationary and the line
        # search meaningless. Inherit whatever Adam converged to.
        alpha_lbfgs = getattr(self, 'alpha_ema', 0.1)
        alpha_lbfgs = max(0.05, min(0.4, alpha_lbfgs))
        if epoch == self.epochs_Adam and xyt_b is not None:
            src = ("pinned via --alpha_fixed"
                   if getattr(self, 'alpha_fixed', None) is not None
                   else "inherited from adaptive Adam value")
            print(f"[LBFGS] data weight frozen at alpha = {alpha_lbfgs:.4e} ({src})")

        last_loss = [0.0]
        def closure():
            if torch.is_grad_enabled():
                self.optimizer_LBFGS.zero_grad()

            if self.constraint == 'HARD':
                if self.stage == 1:
                    res, hx_bdy2_pred = self.pinn(self.xyt,
                                                  self.xyt_bdy2)
                   
                else:
                    res, hx_bdy2_pred = self.pinn(self.xyt,
                                                  self.xyt_bdy2,
                                                  hstar=self.hstar,
                                                  hstar_x=self.hstar_x,
                                                  hstar_y=self.hstar_y,
                                                  hstar_lap=self.hstar_lap,
                                                  hstar_x_bdy2=self.hstar_x_bdy2)
                  

                loss = self.criterion(res, torch.zeros_like(res))
                loss_bdy2 = self.criterion(hx_bdy2_pred, self.hx_bdy2)

                loss_total = loss + self.lam * loss_bdy2

            elif self.constraint == 'SOFT':
                res, u0_pred, h_bdy1_pred, hx_bdy2_pred = self.pinn(self.xyt,
                                                                    self.xyt_bdy2,
                                                                    self.xy0,
                                                                    self.xyt_bdy1)
                loss = self.criterion(res, torch.zeros_like(res))
                loss0 = self.criterion(u0_pred, self.u0)
                loss_bdy1 = self.criterion(h_bdy1_pred, self.h_bdy1)
                loss_bdy2 = self.criterion(hx_bdy2_pred, self.hx_bdy2)

                loss_total = loss + self.lam * (loss0 + loss_bdy1 + loss_bdy2)

         # Supervised batch (LBFGS can handle one mini-batch per step)
            #xyt_b, h_b = self._next_data_batch()
            if xyt_b is not None:
                if hstar_b is not None:
                   #hstar_b = self._hstar_for_batch(xyt_b)
                    h_pred_b = self.net(xyt_b, hstar=hstar_b)
                else:
                    h_pred_b = self.net(xyt_b)
                # As in train_Adam: under MSE the hstar subtraction cancels and
                # both arms compute the same loss. It is a restatement of the
                # objective, not a reweighting of it.
                if hstar_b is not None:
                    loss_data = self.criterion(h_pred_b - hstar_b,
                                               h_b - hstar_b)
                else:
                    loss_data = self.criterion(h_pred_b, h_b)
                # alpha is FROZEN through LBFGS: the closure is re-evaluated
                # several times per step, so a weight that moved between
                # evaluations would make the objective non-stationary and the
                # line search meaningless.
                if torch.isfinite(loss_data):
                    loss_total = loss_total + alpha_lbfgs * loss_data

            # Guard against non-finite values.
            #
            # optim.LBFGS is constructed WITHOUT line_search_fn, so there
            # is no trial-and-reject machinery: the step that produced
            # this iterate has already been applied and the parameters are
            # ruined by the time the closure sees it. Recovery therefore
            # cannot mean "reject the step" -- it can only mean reloading
            # the last validated checkpoint, which the training loop does.
            # Raise a sentinel it can catch rather than a bare
            # RuntimeError that takes the whole run down.
            #
            # Why overflow is reachable at all: activations are
            # sin(scale*Wx) and the residual needs SECOND derivatives,
            # and d2/dx2 sin(ax) ~ a**2. Across five layers the Laplacian
            # scales like the product of the layer gains squared, so one
            # unit step that inflates the weights by an order of magnitude
            # inflates lap by ~1e10. float32 tops out at 3.4e38.
            if not torch.isfinite(loss_total):
                raise _LBFGSNonFinite(
                    f"non-finite L-BFGS objective at epoch {epoch}")
        
            if loss_total.requires_grad:
                loss_total.backward()


            last_loss[0] = loss_total.item()  # save for logging
            return loss_total

        self.optimizer_LBFGS.step(closure)
        #train_loss = closure().item()
        train_loss = last_loss[0]           # <- no second closure call

        self.writer.add_scalar('train_loss', train_loss, epoch)

        return train_loss

    def validate(self, step):
        """Validate process"""

        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()

        xyt_valid = self.validset()
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xyt_valid = xyt_valid.to(self.device)

            if self.stage == 2:
                self.hstar_valid     = self.hstar_valid.to(self.device)
                self.hstar_valid_x   = self.hstar_valid_x.to(self.device)
                self.hstar_valid_y   = self.hstar_valid_y.to(self.device)
                self.hstar_valid_lap = self.hstar_valid_lap.to(self.device)

        if self.stage == 1:
            res = self.net_pde(xyt_valid)
        elif self.stage == 2:
            res = self.net_pde(xyt_valid, hstar=self.hstar_valid,
                               hstar_x=self.hstar_valid_x,
                               hstar_y=self.hstar_valid_y,
                               hstar_lap=self.hstar_valid_lap)

        loss = self.criterion(res, torch.zeros_like(res))
        valid_loss_pde = loss.item()
        self.writer.add_scalar('valid_loss_pde', valid_loss_pde, step)

        # ------------------------------------------------------------------
        # Chronological protocol: the checkpoint is selected on validation
        # RMSE over the validation days, in metres. The PDE residual above is
        # still logged, but it does not select -- a physics-only criterion and
        # a head-accuracy criterion pick different epochs, and the protocol
        # asks for the latter.
        # ------------------------------------------------------------------
        if self.chrono and getattr(self, 'chrono_val_xyt', None) is not None:
            m = self._chrono_validate(step, self._chrono_epoch)
            m['valid_pde_mse'] = valid_loss_pde
            self._chrono_last = m
            self.net.train()
            self.net_pde.train()
            self.net_neumann.train()
            return m['rmse']

        # ------------------------------------------------------------------
        # Supervised validation, on the held-out fifth of the MODFLOW points.
        #
        # The checkpoint is selected on whatever this returns, so it must match
        # what training actually minimises. If training optimises (PDE + alpha *
        # data) but selection watches PDE alone, the saved model is the best
        # physics fit rather than the best model -- the two diverge, silently.
        # A pure run has no supervised set and falls back to PDE only.
        # ------------------------------------------------------------------
        if getattr(self, 'val_xyt_data', None) is None:
            self.writer.add_scalar('valid_loss', valid_loss_pde, step)
            valid_loss = valid_loss_pde
        else:
            with torch.no_grad():
                xyt_data = self.val_xyt_data
                h_true   = self.val_h_data

                if self.stage == 2:
                    x = xyt_data[:, [0]]
                    y = xyt_data[:, [1]]
                    t_tau = torch.full_like(x, self.tau)
                    xyt_tau = torch.cat([x, y, t_tau], dim=1)
                    if self.device == torch.device(type='cuda',
                                                   index=self.cuda_index):
                        xyt_tau = xyt_tau.to(self.device)
                    hstar_val = self.net_prev(xyt_tau).detach()
                    h_pred_data = self.net(xyt_data, hstar=hstar_val)

                    # Written as the RESIDUAL to match how the training
                    # objective is written.
                    #
                    # NOTE: hstar_val cancels under MSE, so this equals the
                    # absolute-head MSE exactly -- see the longer note in
                    # train_Adam. It does NOT stop the model being charged for
                    # an inherited stage-1 offset; nothing in this expression
                    # can, because hstar_val is a constant here.
                    valid_loss_data = self.criterion(
                        h_pred_data - hstar_val, h_true - hstar_val).item()
                    # Consequence: 'valid_loss_data_absolute' below is the SAME
                    # quantity, not an independent absolute-error check, and
                    # the two curves coincide in TensorBoard by construction.
                    # Kept so existing runs stay comparable; do not read the
                    # pair as residual-vs-absolute evidence.
                    self.writer.add_scalar(
                        'valid_loss_data_absolute',
                        self.criterion(h_pred_data, h_true).item(), step)
                else:
                    h_pred_data = self.net(xyt_data)
                    valid_loss_data = self.criterion(h_pred_data,
                                                     h_true).item()

            self.writer.add_scalar('valid_loss_data', valid_loss_data, step)
            alpha_val = getattr(self, 'alpha_ema', 0.5)
            valid_loss = valid_loss_pde + alpha_val * valid_loss_data
            self.writer.add_scalar('valid_loss', valid_loss, step)

        # plot
        if self.stage == 1:
            h_valid = self.net(xyt_valid)
        elif self.stage == 2:
            h_valid = self.net(xyt_valid, self.hstar_valid)

        # Diagnostic figures must never kill a long training run.
        try:
            fig = show_surface(xyt_valid, h_valid, stage=self.stage)
            if fig is not None:
                self.writer.add_figure(tag='3D surface', figure=fig, global_step=step)

            fig = show_contours_2x2(self.validset, h_valid, stage=self.stage,
                                    well_type='unconfined_single_well')
            if fig is not None:
                self.writer.add_figure(tag='contour', figure=fig, global_step=step)
        except Exception as e:
            print(f'[validate] figure skipped: {e}')

        return valid_loss

    def test(self):
        print(f'{self.validset}')

        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()
        self.pinn.eval()

        if self.device == torch.device(type='cuda', index=self.cuda_index):
            self.net.to(self.device)
            self.net_neumann.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        best_model = torch.load(
            f'checkpoints/{self.model_name}/best_model.pth.tar')
        self.pinn.load_state_dict(best_model['state_dict'])

        if self.chrono:
            # Reload happened above; report WHICH epoch is now in the weights
            # so the locked test is provably scored on the selected checkpoint.
            print(f"[Test] reloaded best checkpoint from "
                  f"checkpoints/{self.model_name}/best_model.pth.tar")
            print(f"[Test] selected epoch = {best_model.get('epoch')} | "
                  f"selection criterion = "
                  f"{best_model.get('selection_criterion')} | "
                  f"best_loss (valid RMSE, m) = {best_model.get('best_loss')}")
            self._chrono_test(best_model)
            print('Testing finished successfully!!!\n')
            return

        xyt_valid = self.validset()
        if self.device == torch.device(type='cuda', index=self.cuda_index):
            xyt_valid = xyt_valid.to(self.device)
            if self.stage > 1:
                self.hstar_valid     = self.hstar_valid.to(self.device)
                self.hstar_valid_x   = self.hstar_valid_x.to(self.device)
                self.hstar_valid_y   = self.hstar_valid_y.to(self.device)
                self.hstar_valid_lap = self.hstar_valid_lap.to(self.device)

        if self.stage == 1:
            res_valid = self.net_pde(xyt_valid)
            h_valid = self.net(xyt_valid)

        elif self.stage > 1:
            res_valid = self.net_pde(xyt_valid, hstar=self.hstar_valid,
                                     hstar_x=self.hstar_valid_x,
                                     hstar_y=self.hstar_valid_y,
                                     hstar_lap=self.hstar_valid_lap)
            h_valid = self.net(xyt_valid, self.hstar_valid)

        # Compute test loss
        loss = self.criterion(res_valid, torch.zeros_like(res_valid)).item()
        print(f'Test loss: {loss:.4e}')

        # Plot 3D surface
        fig = show_surface(xyt_valid, h_valid, self.stage)

        # Plot contour map for comparing the results between MODFLOW and GWPINN
        fig = show_contours_2x2(self.validset, h_valid, self.stage,
                                well_type='unconfined_single_well')

        plt.show()

        # Report the o1 comparison and the t=26-30 metrics whenever the model's
        # time domain actually reaches the test window.
        #
        # Gated on the model's own time domain rather than on the stage. A
        # SOFT run is single-stage, since the two-stage handoff is a property
        # of the HARD construction h = hstar + d*u, and gating on stage would
        # skip it entirely. Stage 1 of a HARD run covers only t <= tau (= 1 d),
        # where scoring at t = 26-30 would be meaningless, so that case is
        # skipped.
        _tmax_model = self.problem.domain[5] if self.stage > 1 else self.tau
        if _tmax_model >= 26.0:
            df = pd.read_csv('modflow/o1.csv',
                             sep=r'\s+', names=['x', 'y', 't', 'h'])
            # o1.csv is written in MODFLOW's 0..1000 frame, like every other
            # data file; the network works in -500..500. Without this shift the
            # points are evaluated at their mirror-image locations and the test
            # reports nonsense while appearing to succeed.
            xy = df[['x', 'y']].values - 500.0
            t = df[['t']].values
            xyt = np.hstack([xy, t])
            xytau = np.hstack([xy, self.tau * np.ones_like(t)])

            xyt = torch.from_numpy(xyt).float()
            xytau = torch.from_numpy(xytau).float()

            if self.device == torch.device(type='cuda', index=self.cuda_index):
                xyt = xyt.to(self.device)
                xytau = xytau.to(self.device)

            # net_prev only exists for stage 2. A single-stage run -- which is
            # how SOFT must be run, since the two-stage handoff is a property
            # of the HARD construction h = hstar + d*u -- has no previous
            # stage to inherit from.
            if self.stage > 1:
                hstar = self.net_prev(xytau).detach()
                h_pred = self.net(xyt, hstar)
            else:
                h_pred = self.net(xyt)
            h_pred = h_pred.detach().cpu().numpy()

            h = df[['h']].values
            print(np.hstack((h, h_pred)))

            print(
                f'mae = {mae(h, h_pred):.3f}, rrmse = {rrmse(h, h_pred) * 100:.3f} %')

            # ================================================================
            # Test on the last 5 time steps (26-30) - temporal extrapolation.
            # Ported from trainer_working.py so the reported metric set is
            # identical: per-snapshot MAE / RRMSE / R2, then the averages, the
            # POOLED R2 (computed over all test points together, which is not
            # the same as the mean of the per-snapshot values), and the
            # standard deviations across snapshots.
            # ================================================================
            test_times = [26, 27, 28, 29, 30]
            all_mae = []
            all_rrmse = []
            all_r2 = []

            all_predictions_combined = []
            all_targets_combined = []

            print("\n=== Testing on Last 5 Time Steps (Extrapolation) ===")

            for t_val in test_times:
                try:
                    df = pd.read_csv(f'./modflow/t{t_val}.txt',
                                     sep=r'\s*,\s*|\s+', engine='python',
                                     header=None, names=['x', 'y', 'h'])
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()

                    if len(df) == 0:
                        print(f"  t={t_val}: No valid data, skipping")
                        continue

                    xy_shift = df[['x', 'y']].values - 500     # to [-500, 500]
                    t = np.full((len(df), 1), float(t_val))
                    xyt = np.hstack([xy_shift, t])
                    xytau = np.hstack([xy_shift, self.tau * np.ones_like(t)])

                    xyt = torch.from_numpy(xyt.astype(np.float32))
                    xytau = torch.from_numpy(xytau.astype(np.float32))

                    if self.device == torch.device(type='cuda',
                                                   index=self.cuda_index):
                        xyt = xyt.to(self.device)
                        xytau = xytau.to(self.device)

                    if self.stage > 1:
                        hstar = self.net_prev(xytau).detach()
                        h_pred = self.net(xyt, hstar).detach().cpu().numpy()
                    else:
                        h_pred = self.net(xyt).detach().cpu().numpy()
                    h_true = df[['h']].values

                    t_mae = mae(h_true, h_pred)
                    t_rrmse = rrmse(h_true, h_pred) * 100

                    ss_res = np.sum((h_true - h_pred) ** 2)
                    ss_tot = np.sum((h_true - np.mean(h_true)) ** 2)
                    t_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                    all_mae.append(t_mae)
                    all_rrmse.append(t_rrmse)
                    all_r2.append(t_r2)

                    all_predictions_combined.append(h_pred)
                    all_targets_combined.append(h_true)

                    print(f"  t={t_val:2d}: MAE={t_mae:.3f}m, "
                          f"RRMSE={t_rrmse:.3f}%, R\u00b2={t_r2:.6f}")

                except Exception as e:
                    print(f"  t={t_val}: Error loading/processing - {e}")

            if all_mae:
                all_predictions_combined = np.vstack(all_predictions_combined)
                all_targets_combined = np.vstack(all_targets_combined)

                ss_res_total = np.sum(
                    (all_targets_combined - all_predictions_combined) ** 2)
                ss_tot_total = np.sum(
                    (all_targets_combined - np.mean(all_targets_combined)) ** 2)
                r2_overall = (1 - (ss_res_total / ss_tot_total)
                              if ss_tot_total > 0 else 0.0)

                print(f"\n--- FINAL TEST RESULTS (t=26-30) ---")
                print(f"Average MAE:    {np.mean(all_mae):.3f} m")
                print(f"Average RRMSE:  {np.mean(all_rrmse):.3f} %")
                print(f"Average R\u00b2:     {np.mean(all_r2):.6f}")
                print(f"Overall R\u00b2:     {r2_overall:.6f} (combined)")
                print(f"Std MAE:        {np.std(all_mae):.3f} m")
                print(f"Std RRMSE:      {np.std(all_rrmse):.3f} %")
                print(f"Std R\u00b2:         {np.std(all_r2):.6f}")
                print(f"------------------------------------")
            else:
                print("WARNING: No test data found for t=26-30!")

        print('Testing finished successfully!!!\n')


if __name__ == '__main__':

    args = Options().parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.problem = Problem(sigma=args.sigma)
    print('****************************************************************')
    print(f'Unconfined aquifer, single pumping well (stage {args.stage})')
    print(f'domain={args.problem.domain}, tau={args.tau}')
    print(f'constraint={args.constraint}, sigma={args.sigma}')
    print(f'layers=args.layers')
    print('****************************************************************')
    trainer = Trainer(args)

    trainer.train()
