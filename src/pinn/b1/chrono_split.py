#!/usr/bin/env python
"""Chronological train / validation / test protocol for the benchmark-2 PINN.

Self-contained so the physics code (problem.py, model.py, sampler.py) is not
touched. trainer.py calls into this module only when --chrono is passed; without
that flag nothing here runs and the legacy behaviour is bit-for-bit unchanged.

The protocol
------------
    supervised train heads = days 1-20      (sdata/ anchors, 815 per snapshot)
    validation heads       = days 21-25     (full 300x300 reference field)
    test heads             = days 26-30     (full 300x300 reference field, LOCKED)
    PDE collocation        = full domain through day 30

Two distinct reference products live under modflow/:

  * ``modflow/sdata/t<d>.txt`` - 815 scattered observation points, the SAME
    locations at every day. This is the existing spatial observation/anchor
    sampling and it is preserved unchanged; it supplies the supervised head
    loss.
  * ``modflow/t<d>.txt`` - the full 90,000-node (300x300) reference field. Used
    for validation and test scoring, so that the metric selecting the
    checkpoint and the metric reported at test are the same quantity on the
    same support.

Only days 26-30 exist as a full field and nowhere in sdata/, so the locked test
set cannot leak into the supervised loader even by accident. The assertions in
``ChronoSplit`` check that anyway.

Initial condition
-----------------
Untouched. problem.bc(mode=0) is the analytic Dupuit profile h_initial(y); no
snapshot is required to impose it. The subdaily snapshots (t0.25, t0.5, t0.75)
are therefore NOT supervised targets under this protocol - the initial-condition
formulation does not need them, and day 1 is the earliest supervised day.

Normalization
-------------
The network predicts head directly in metres (h = hstar + d*u, hstar in metres)
and its input normalization is geometric - the min/max of the space-time domain,
not a data statistic. So there is no data-derived scaling to fit, and every
metric below is already in metres. The head statistics computed here are
derived from the TRAINING heads only and are stored with the checkpoint as
provenance: they record what the model was allowed to see, and make it checkable
after the fact that no validation or test head entered any normalization.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

# MODFLOW writes model coords 0..1000; the network works on -500..500.
COORD_SHIFT = 500.0

DEFAULT_TRAIN_DAYS = '1-20'
DEFAULT_VAL_DAYS = '21-25'
DEFAULT_TEST_DAYS = '26-30'


# ----------------------------------------------------------------------------
# day-range parsing and snapshot -> time mapping
# ----------------------------------------------------------------------------
def parse_days(spec):
    """'1-20' or '21,22,23' or '26-28,30' -> sorted list of ints."""
    days = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part[1:]:
            lo, hi = part.split('-', 1)
            days.extend(range(int(lo), int(hi) + 1))
        else:
            days.append(int(part))
    out = sorted(set(days))
    if not out:
        raise ValueError('empty day specification: %r' % (spec,))
    return out


def time_from_path(path):
    """t<time>.txt -> float. Same rule dataset.py uses, kept consistent."""
    m = re.search(r't([0-9]+(?:\.[0-9]+)?)', os.path.basename(path))
    if not m:
        raise ValueError('cannot parse time from filename: %s' % path)
    return float(m.group(1))


def snapshot_time_map(pattern):
    """{time -> path} for every snapshot matching pattern."""
    out = {}
    for p in sorted(glob.glob(pattern)):
        out[time_from_path(p)] = p
    return out


def _read_snapshot(path):
    df = pd.read_csv(path, sep=r'\s*,\s*|\s+', engine='python',
                     header=None, names=['x', 'y', 'h'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df = df[np.isfinite(df[['x', 'y', 'h']]).all(axis=1)]
    return df


def load_days(day_list, path_for_day, to_pinn_coords=True):
    """Stack the given days into (xyt, h) float32 arrays.

    Returns xyt in PINN coords (-500..500) and h in metres, plus the per-day
    row counts so callers can slice back out by day.
    """
    if not day_list:
        raise ValueError('load_days called with an empty day list')
    xs, hs, counts = [], [], {}
    for d in day_list:
        path = path_for_day(d)
        df = _read_snapshot(path)
        if len(df) == 0:
            raise ValueError('snapshot %s has no usable rows' % path)
        xy = df[['x', 'y']].values.astype(np.float32)
        if to_pinn_coords:
            xy = xy - COORD_SHIFT
        t = np.full((len(df), 1), float(d), dtype=np.float32)
        xs.append(np.hstack([xy, t]))
        hs.append(df[['h']].values.astype(np.float32))
        counts[d] = len(df)
    return np.vstack(xs), np.vstack(hs), counts


# ----------------------------------------------------------------------------
# metrics, all in physical head units (metres)
# ----------------------------------------------------------------------------
def head_metrics(h_true, h_pred):
    """MAE, MSE, RMSE, RRMSE (%), R2 in metres. h_* are (N,1) arrays."""
    h_true = np.asarray(h_true, dtype=np.float64).reshape(-1)
    h_pred = np.asarray(h_pred, dtype=np.float64).reshape(-1)
    err = h_true - h_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mean_true = float(np.mean(h_true))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((h_true - mean_true) ** 2))
    return {
        'mae': float(np.mean(np.abs(err))),
        'mse': mse,
        'rmse': rmse,
        # matches utils.rrmse (normalised by the mean, not the range), x100
        'rrmse_pct': float(rmse / abs(mean_true) * 100.0) if mean_true else float('nan'),
        'r2': float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        'n': int(h_true.size),
    }


# ----------------------------------------------------------------------------
# the split itself
# ----------------------------------------------------------------------------
class ChronoSplit(object):
    """Builds and guards the chronological split.

    stage/tau are the model's own time window. A day outside that window cannot
    be scored by this model -- the input normalization and the hard-constraint
    mask d are both built from the window -- so such days are reported and
    dropped rather than silently evaluated far outside the fitted range.
    """

    def __init__(self, problem, stage=1, tau=1.0,
                 train_days=DEFAULT_TRAIN_DAYS,
                 val_days=DEFAULT_VAL_DAYS,
                 test_days=DEFAULT_TEST_DAYS,
                 anchor_pattern='./modflow/sdata/t*.txt',
                 field_pattern='./modflow/t*.txt'):
        self.problem = problem
        self.stage = stage
        self.tau = tau
        self.anchor_pattern = anchor_pattern
        self.field_pattern = field_pattern

        self.train_days = parse_days(train_days)
        self.val_days = parse_days(val_days)
        self.test_days = parse_days(test_days)

        # --- the split must be disjoint. This is the whole point. ----------
        self._assert_disjoint()

        self.anchor_map = snapshot_time_map(anchor_pattern)
        self.field_map = snapshot_time_map(field_pattern)

        # --- availability --------------------------------------------------
        missing_train = [d for d in self.train_days if float(d) not in self.anchor_map]
        missing_val = [d for d in self.val_days if float(d) not in self.field_map]
        missing_test = [d for d in self.test_days if float(d) not in self.field_map]
        if missing_train:
            raise ValueError('no anchor snapshot for train day(s) %s under %r'
                             % (missing_train, anchor_pattern))
        if missing_val:
            raise ValueError('no reference field for validation day(s) %s under %r'
                             % (missing_val, field_pattern))
        if missing_test:
            raise ValueError('no reference field for test day(s) %s under %r'
                             % (missing_test, field_pattern))

        # --- restrict to the model's own time window -----------------------
        self.train_days_used, self.train_days_dropped = self._in_window(self.train_days)
        self.val_days_used, self.val_days_dropped = self._in_window(self.val_days)
        self.test_days_used, self.test_days_dropped = self._in_window(self.test_days)

        if not self.train_days_used:
            raise ValueError(
                'no supervised training day falls inside this model\'s time '
                'window (stage=%s, tau=%s). Days %s were all dropped.'
                % (stage, tau, self.train_days))

        # In the original HARD curriculum, stage 1 is a short pretraining model
        # over [tmin, tau]. It cannot reach days 21-25 and is selected by its
        # existing in-window physics validation. Stage 2 inherits that model,
        # spans (tau, tmax], and is the final model selected on held-out head
        # RMSE. This preserves the published two-stage formulation instead of
        # replacing it with a materially different single-stage network.
        self.is_pretraining_stage = (
            self.stage == 1 and not self.val_days_used
            and float(self.tau) < float(min(self.val_days))
        )
        if not self.val_days_used and not self.is_pretraining_stage:
            raise ValueError(
                'no validation day falls inside this model\'s time window '
                '(stage=%s, tau=%s): days %s are all outside (%g, %g]. '
                'Only stage-1 curriculum pretraining may omit held-out head '
                'validation; the final stage must cover days 21-25.'
                % (stage, tau, self.val_days,
                   self.problem.domain[4] if stage == 1 else tau,
                   tau if stage == 1 else self.problem.domain[5]))

        self._norm = None

    # -- guards ------------------------------------------------------------
    def _assert_disjoint(self):
        tr, va, te = set(self.train_days), set(self.val_days), set(self.test_days)
        for a, b, na, nb in ((tr, va, 'train', 'validation'),
                             (tr, te, 'train', 'test'),
                             (va, te, 'validation', 'test')):
            overlap = sorted(a & b)
            if overlap:
                raise ValueError('%s and %s day sets overlap on %s -- the '
                                 'split must be disjoint' % (na, nb, overlap))

    def _in_window(self, days):
        """Split days into (inside the model's window, outside)."""
        tmin, tmax = self.problem.domain[4], self.problem.domain[5]
        if self.stage == 1:
            lo, hi = tmin, self.tau
        else:
            lo, hi = self.tau, tmax
        used, dropped = [], []
        for d in days:
            t = float(d)
            # stage 1 keeps t <= tau, stage 2 keeps t > tau: the same rule
            # dataset.ModflowDataset applies, so the two agree.
            inside = (t <= hi) if self.stage == 1 else (lo < t <= hi)
            (used if inside else dropped).append(d)
        return used, dropped

    # -- data --------------------------------------------------------------
    def train_anchors(self):
        """Supervised heads: the existing 815-point anchor sampling, days 1-20."""
        return load_days(self.train_days_used,
                         lambda d: self.anchor_map[float(d)])

    def val_field(self):
        """Validation heads: full reference field, days 21-25."""
        if not self.val_days_used:
            raise ValueError('this curriculum-pretraining stage has no '
                             'held-out head-validation days')
        return load_days(self.val_days_used,
                         lambda d: self.field_map[float(d)])

    def test_field(self):
        """Locked test heads: full reference field, days 26-30."""
        return load_days(self.test_days_used,
                         lambda d: self.field_map[float(d)])

    def test_day_field(self, day):
        """One test day, for per-time reporting."""
        return load_days([day], lambda d: self.field_map[float(d)])

    # -- normalization provenance -----------------------------------------
    def normalization(self, h_train):
        """Head statistics from TRAINING heads only. See module docstring."""
        h = np.asarray(h_train, dtype=np.float64).reshape(-1)
        tmin, tmax = self.problem.domain[4], self.problem.domain[5]
        if self.stage == 1:
            t_lo, t_hi = tmin, self.tau
        else:
            t_lo, t_hi = self.tau, tmax
        self._norm = {
            'scheme': 'none_applied_to_head',
            'note': ('The network outputs head directly in metres '
                     '(h = hstar + d*u). Input normalization is geometric -- '
                     'the domain min/max below -- and carries no data '
                     'statistic. These head statistics come from the '
                     'supervised TRAINING days only and are recorded as '
                     'provenance; no validation or test head contributes to '
                     'any normalization.'),
            'head_stats_source': 'train_days_only',
            'head_mean_m': float(np.mean(h)),
            'head_std_m': float(np.std(h)),
            'head_min_m': float(np.min(h)),
            'head_max_m': float(np.max(h)),
            'head_n': int(h.size),
            'input_domain': {
                'x': [self.problem.domain[0], self.problem.domain[1]],
                'y': [self.problem.domain[2], self.problem.domain[3]],
                't': [t_lo, t_hi],
            },
            'coord_shift_applied_to_reference_data': -COORD_SHIFT,
            'units': 'metres',
        }
        return self._norm

    # -- reporting ---------------------------------------------------------
    def banner(self):
        pde_lo = self.problem.domain[4] if self.stage == 1 else self.tau
        pde_hi = self.tau if self.stage == 1 else self.problem.domain[5]

        def fmt(used, dropped, label):
            s = '%s = days %s' % (label, _compact(used)) if used else \
                '%s = (none in this model\'s window)' % label
            if dropped:
                s += '   [outside this model\'s window, not used: days %s]' % _compact(dropped)
            return s

        lines = [
            '================ TEMPORAL PROTOCOL ================',
            fmt(self.train_days_used, self.train_days_dropped,
                'supervised train heads'),
            fmt(self.val_days_used, self.val_days_dropped,
                'validation heads      '),
            fmt(self.test_days_used, self.test_days_dropped,
                'test heads            '),
            'PDE collocation        = full domain through day %g '
            '(this model: t in (%g, %g])' % (self.problem.domain[5], pde_lo, pde_hi),
            '---------------------------------------------------',
            'supervised head source = %s  (existing anchor sampling, preserved)'
            % self.anchor_pattern,
            'val/test head source   = %s  (full reference field)'
            % self.field_pattern,
            'initial condition      = analytic Dupuit profile (problem.bc mode 0);'
            ' subdaily snapshots are NOT supervised targets',
            ('checkpoint selection   = in-window physics validation MSE '
             '(curriculum pretraining only)' if self.is_pretraining_stage else
             'checkpoint selection   = validation RMSE (metres), days %s'
             % _compact(self.val_days_used)),
            'test heads             = LOCKED: excluded from loss, normalization,'
            ' checkpoint selection and scheduling',
            '===================================================',
        ]
        return '\n'.join(lines)

    def metadata(self):
        return {
            'protocol': 'chronological_train_val_test',
            'train_days_requested': self.train_days,
            'val_days_requested': self.val_days,
            'test_days_requested': self.test_days,
            'train_days_used': self.train_days_used,
            'val_days_used': self.val_days_used,
            'test_days_used': self.test_days_used,
            'train_days_dropped_outside_window': self.train_days_dropped,
            'val_days_dropped_outside_window': self.val_days_dropped,
            'test_days_dropped_outside_window': self.test_days_dropped,
            'disjoint': True,
            'stage': self.stage,
            'tau': self.tau,
            'model_time_window': (
                [self.problem.domain[4], self.tau] if self.stage == 1
                else [self.tau, self.problem.domain[5]]),
            'pde_collocation_window': (
                [self.problem.domain[4], self.tau] if self.stage == 1
                else [self.tau, self.problem.domain[5]]),
            'pde_collocation_truncated_at_day_20': False,
            'anchor_pattern': self.anchor_pattern,
            'field_pattern': self.field_pattern,
            'supervised_head_support': 'sdata anchors (existing sampling)',
            'val_test_head_support': 'full 300x300 reference field',
            'initial_condition': 'analytic Dupuit profile, problem.bc(mode=0)',
            'subdaily_snapshots_used_as_targets': False,
            'stage_role': ('curriculum_pretraining' if self.is_pretraining_stage
                           else 'final_validation_selected_model'),
            'selection_criterion': (
                'in_window_physics_validation_mse' if self.is_pretraining_stage
                else 'validation_rmse_metres'),
            'pipeline_train_days': self.train_days,
            'pipeline_pde_collocation_window': [self.problem.domain[4],
                                                self.problem.domain[5]],
        }


def _compact(days):
    """[1,2,3,5] -> '1-3,5'"""
    if not days:
        return '-'
    days = sorted(days)
    runs, start, prev = [], days[0], days[0]
    for d in days[1:]:
        if d == prev + 1:
            prev = d
            continue
        runs.append((start, prev))
        start = prev = d
    runs.append((start, prev))
    return ','.join('%g' % a if a == b else '%g-%g' % (a, b) for a, b in runs)


# ----------------------------------------------------------------------------
# chunked evaluation, so a 450k-point validation set does not blow up memory
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_heads(net, xyt, device, stage=1, tau=1.0, net_prev=None,
                  chunk=65536):
    """Head prediction in metres, under no_grad.

    no_grad is not a nicety here: it is what keeps validation and test heads out
    of the training graph. Nothing this returns can carry a gradient back into
    the model.
    """
    was_training = net.training
    net.eval()
    out = []
    for i in range(0, xyt.shape[0], chunk):
        xb = xyt[i:i + chunk].to(device)
        if stage > 1:
            xtau = torch.cat([xb[:, [0]], xb[:, [1]],
                              torch.full_like(xb[:, [0]], tau)], dim=1)
            hstar = net_prev(xtau).detach()
            hb = net(xb, hstar)
        else:
            hb = net(xb)
        out.append(hb.detach().cpu().numpy())
    if was_training:
        net.train()
    return np.vstack(out)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
