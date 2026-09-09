"""Hard Dirichlet enforcement for the gridded surrogates (CNN, ConvLSTM).

The physics-informed model differs from the data-driven models in two ways at
once: it enforces the boundary condition by construction, and it carries a PDE
residual in its loss. When it conserves mass and they do not, neither cause can
be assigned. A gridded surrogate carrying exactly one of the two isolates the
mechanism, which is what this module provides.

The constraint is a binary projection onto the prescribed head at the MODFLOW
constant-head cells:

    h = h_D     on CHD cells
    h = h_raw   elsewhere

On both benchmarks the CHD cells are the first and last grid rows.
``DirichletRowProjection`` applies this without altering adjacent or interior
rows. The distance and ramp transforms in ``boundary_mask`` support the
completed sensitivity runs; they are not the primary constrained baseline.

Normalisation
-------------
``GroundwaterDataset`` stores (H - mean)/std, so network output and targets both
live in normalised space, and applying a physical mask to a normalised field
would silently do nothing. Working it through:

    h_phys = h_bc + m * (u * std)
    h_norm = (h_phys - mean)/std = (h_bc - mean)/std + m * u
                                 =      hbc_norm     + m * u

so the constraint is exact in normalised space, with no denormalise and
renormalise round trip at runtime.

Usage
-----
    from constrained import DirichletRowProjection, chd_row_kwargs

    model = HeadCNN(
        constrained=True,
        bc=chd_row_kwargs('b2', y_grid, train_ds.mean, train_ds.std),
    ).to(device)

where the model's ``__init__`` does

    self.bc = DirichletRowProjection(**bc) if constrained else None

and its ``forward`` ends with

    return u if self.bc is None else self.bc(u)

``y_grid`` is the third return value of ``load_txt_to_grid()``. Optimizer, loss,
training loop and model capacity are unchanged. B2 fixes the south row at 90 m
and the north row at 100 m; the analytic interior Dupuit profile used by the
physics-informed trial function is not imposed on the CNN interior.
"""
import os

import numpy as np
import torch
import torch.nn as nn


def boundary_mask(y_coords, kind='parabolic', width=10):
    """m(y): 0 on the first and last grid rows, 1 in the interior. Shape (ny, 1).

    'parabolic'  m = (ymax-y)(y-ymin) / ((ymax-ymin)/2)^2
        The direct analogue of the PINN's d(y). Natural there: a coordinate
        network has a global smooth basis and sees y directly. On a gridded
        model it rescales the target over the WHOLE domain -- only 66 of 300
        rows have m > 0.95 -- and a translation-equivariant CNN with a 7-cell
        receptive field cannot see y to undo it.

    'ramp'       m = min(1, rows_from_nearest_boundary / width)
        Zero on the two Dirichlet rows, 1 over ~93% of the domain at
        width=10. Imposes the same boundary condition while leaving the
        interior learning problem identical to the unconstrained model.

    The choice is empirical. On B1 'parabolic' eliminated the boundary error
    (0.445 -> 0.000 m on the Dirichlet rows) at a cost of 130-200% in the
    interior; 'ramp' removes that cost. Both are reported.
    """
    y = np.asarray(y_coords, dtype=np.float64).reshape(-1, 1)
    if kind == 'parabolic':
        ymin, ymax = float(y.min()), float(y.max())
        m = (ymax - y) * (y - ymin) / (((ymax - ymin) / 2.0) ** 2)
    elif kind == 'ramp':
        n = y.shape[0]
        rows = np.minimum(np.arange(n), n - 1 - np.arange(n)).reshape(-1, 1)
        m = np.minimum(rows / float(width), 1.0)
    else:
        raise ValueError(f"unknown mask {kind!r}; expected 'parabolic' or 'ramp'")
    return np.clip(m, 0.0, None)


import paths

BENCHMARKS = {
    # h_south / h_north are the prescribed heads on the two Dirichlet rows.
    # The snapshot directory is a default; an explicit --data_path always wins.
    'b1': dict(h_south=90.0,  h_north=90.0,
               label='homogeneous K, h = 90 m both boundaries'),
    'b2': dict(h_south=90.0,  h_north=100.0,
               label='low-K lens, regional gradient 90 m south -> 100 m north'),
}


def data_path_for(benchmark):
    """Default MODFLOW snapshots for the named benchmark, as an absolute path."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; "
                         f"expected one of {sorted(BENCHMARKS)}")
    return str(paths.benchmark_dir(benchmark) / 't*.txt')


def h_bc_for(benchmark, y_coords):
    """Prescribed head on each grid row, for the named benchmark.

    Benchmark 1 holds 90 m on both Dirichlet rows, so h_bc is constant and the
    PINN's h* is likewise a constant 90.

    Benchmark 2 imposes a regional gradient, 90 m south and 100 m north. Steady
    unconfined flow with no recharge makes h^2 linear in y, so the analytic
    profile is the Dupuit parabola -- the same h_initial() the MODFLOW
    generator uses as `strt` and the PINN uses as h*. They must agree or the
    two are solving different problems.

    The fraction is referenced to the GRID ROWS, not to the nominal 0..1000
    domain. MODFLOW's constant-head cells sit at the first and last CELL
    CENTRES (y = 1.6667 and 998.3333), and evaluating the profile at y/1000
    there gives 90.0176 and 99.9842 instead of 90 and 100 -- a systematic
    1.6-1.8 cm offset that would show up in the boundary-error metric and look
    exactly like the constraint failing. Referencing to y[0] and y[-1] makes it
    exact at both boundaries and shifts the interior by at most 1.8 cm.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; "
                         f"expected one of {sorted(BENCHMARKS)}")
    cfg = BENCHMARKS[benchmark]
    y = np.asarray(y_coords, dtype=np.float64).ravel()
    hs, hn = cfg['h_south'], cfg['h_north']

    if hs == hn:
        return np.full_like(y, float(hs))

    frac = (y - y[0]) / (y[-1] - y[0])          # 0 at row 0, 1 at row -1
    return np.sqrt(hs ** 2 + (hn ** 2 - hs ** 2) * frac)


class BoundaryConstraint(nn.Module):
    """The constraint itself, as a layer applied to a model's raw output.

    Both HeadCNN and ConvLSTMModel emit (B, 1, ny, nx) in normalised units, so
    both call this. One implementation means the two arms cannot drift apart --
    which matters, because the experiment's whole claim is that they differ in
    exactly one thing.

    The mask buffers are registered persistent=False so they stay OUT of
    state_dict. A constrained and an unconstrained model therefore produce
    checkpoints with identical keys, and either loads into the other. (An
    earlier wrapper-based version renamed every weight with a `base.` prefix
    and injected the mask, so the two arms' checkpoints were incompatible.)
    """

    def __init__(self, y_coords, mean, std, h_bc=90.0, mask='parabolic',
                 mask_width=10):
        super().__init__()
        y = np.asarray(y_coords, dtype=np.float64).reshape(-1, 1)
        m = boundary_mask(y, kind=mask, width=mask_width)
        self.mask_kind = mask

        h_bc_col = (np.full_like(y, float(h_bc)) if np.ndim(h_bc) == 0
                    else np.asarray(h_bc, dtype=np.float64).reshape(-1, 1))
        if h_bc_col.shape != y.shape:
            raise ValueError(f"h_bc has {h_bc_col.shape[0]} rows but the grid "
                             f"has {y.shape[0]}")
        hbc_norm = (h_bc_col - mean) / (std + 1e-8)

        # (1, 1, ny, 1): broadcasts over batch and over x
        self.register_buffer('m', torch.tensor(m, dtype=torch.float32)[None, None],
                             persistent=False)
        self.register_buffer('hbc_norm',
                             torch.tensor(hbc_norm, dtype=torch.float32)[None, None],
                             persistent=False)
        self.mean, self.std = float(mean), float(std)

    def forward(self, u):
        return self.hbc_norm + self.m * u

    def describe(self):
        b = float(self.hbc_norm[0, 0, 0, 0]) * self.std + self.mean
        t = float(self.hbc_norm[0, 0, -1, 0]) * self.std + self.mean
        return (f"BoundaryConstraint[{self.mask_kind}]: "
                f"mask {float(self.m.min()):.3e} at the "
                f"boundary rows, {float(self.m.max()):.3f} mid-domain; "
                f"h_BC = {b:.4f} (row 0) .. {t:.4f} (row -1) m")


class DirichletRowProjection(nn.Module):
    """Hard projection on MODFLOW constant-head rows only.

    Unlike ``BoundaryConstraint``, this layer does not rescale any interior
    cell. It replaces the first and last output rows by their prescribed heads
    and leaves every other network output unchanged. The registered tensors
    broadcast over the batch and x dimensions.
    """

    def __init__(self, y_coords, mean, std, h_south, h_north):
        super().__init__()
        y = np.asarray(y_coords, dtype=np.float64).ravel()
        if y.size < 2:
            raise ValueError("Dirichlet row projection needs at least two rows")

        free = np.ones((y.size, 1), dtype=np.float32)
        free[[0, -1], 0] = 0.0

        fixed = np.zeros((y.size, 1), dtype=np.float32)
        fixed[0, 0] = (float(h_south) - mean) / (std + 1e-8)
        fixed[-1, 0] = (float(h_north) - mean) / (std + 1e-8)

        self.register_buffer(
            'free', torch.tensor(free)[None, None], persistent=False
        )
        self.register_buffer(
            'fixed', torch.tensor(fixed)[None, None], persistent=False
        )
        self.mean = float(mean)
        self.std = float(std)
        self.h_south = float(h_south)
        self.h_north = float(h_north)

    def forward(self, raw_head):
        if raw_head.ndim != 4:
            raise ValueError(
                "DirichletRowProjection expects (batch, channel, y, x), "
                f"got {tuple(raw_head.shape)}"
            )
        if raw_head.shape[-2] != self.free.shape[-2]:
            raise ValueError(
                f"prediction has {raw_head.shape[-2]} rows but the projection "
                f"was built for {self.free.shape[-2]}"
            )
        return self.free * raw_head + self.fixed

    def describe(self):
        return (
            "DirichletRowProjection[CHD rows only]: "
            f"row 0 = {self.h_south:.4f} m, row -1 = {self.h_north:.4f} m; "
            "all interior rows unchanged"
        )


class ConstrainedWrapper(nn.Module):
    """Wrap any model whose forward returns (B, 1, ny, nx) in normalised units.

    Model-agnostic on purpose: HeadCNN and ConvLSTMModel have the same output
    contract, so the CNN and ConvLSTM arms get an identical constraint rather
    than two implementations that might differ subtly.

    The wrapper adds NO parameters -- the constrained and unconstrained arms
    have identical capacity, so any difference between them is the constraint
    and nothing else.
    """

    def __init__(self, base, y_coords, mean, std, h_bc=90.0):
        super().__init__()
        self.base = base

        y = np.asarray(y_coords, dtype=np.float64).reshape(-1, 1)
        m = boundary_mask(y)

        h_bc_col = (np.full_like(y, float(h_bc)) if np.ndim(h_bc) == 0
                    else np.asarray(h_bc, dtype=np.float64).reshape(-1, 1))
        if h_bc_col.shape != y.shape:
            raise ValueError(f"h_bc has {h_bc_col.shape[0]} rows but the grid "
                             f"has {y.shape[0]}")
        hbc_norm = (h_bc_col - mean) / (std + 1e-8)

        # (1, 1, ny, 1) so both broadcast over batch and over x
        self.register_buffer('m', torch.tensor(m, dtype=torch.float32)[None, None])
        self.register_buffer('hbc_norm',
                             torch.tensor(hbc_norm, dtype=torch.float32)[None, None])
        self.mean, self.std = float(mean), float(std)

    def forward(self, x):
        return self.hbc_norm + self.m * self.base(x)

    def describe(self):
        b = float(self.hbc_norm[0, 0, 0, 0]) * self.std + self.mean
        t = float(self.hbc_norm[0, 0, -1, 0]) * self.std + self.mean
        return (f"ConstrainedWrapper: mask {float(self.m.min()):.3e} at the "
                f"boundary rows, {float(self.m.max()):.3f} mid-domain; "
                f"h_BC = {b:.4f} (row 0) .. {t:.4f} (row -1) m")


def bc_kwargs(benchmark, y_coords, mean, std, mask='parabolic',
              mask_width=10):
    """Everything BoundaryConstraint needs, for the named benchmark.

        model = HeadCNN(constrained=True,
                        bc=bc_kwargs('b2', y_grid, ds.mean, ds.std))
    """
    return dict(y_coords=y_coords, mean=mean, std=std,
                h_bc=h_bc_for(benchmark, y_coords),
                mask=mask, mask_width=mask_width)


def chd_row_kwargs(benchmark, y_coords, mean, std):
    """Arguments for exact projection on the benchmark's CHD rows."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; "
                         f"expected one of {sorted(BENCHMARKS)}")
    cfg = BENCHMARKS[benchmark]
    return dict(
        y_coords=y_coords,
        mean=mean,
        std=std,
        h_south=cfg['h_south'],
        h_north=cfg['h_north'],
    )


@torch.no_grad()
def boundary_error(model, dataset, device, h_bc=90.0):
    """max |h_pred - h_BC| on the two Dirichlet rows, in metres.

    For a constrained model this must return machine precision; anything
    larger means the mask is not doing what it claims, and nothing else in
    the comparison can be trusted.
    """
    model.eval()
    y = np.atleast_1d(np.asarray(h_bc, dtype=np.float64))
    bc_s = float(y[0]); bc_n = float(y[-1])
    worst_s = worst_n = 0.0
    for i in range(len(dataset)):
        x, _ = dataset[i]
        p = model(x.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        p = p * dataset.std + dataset.mean          # -> physical metres
        worst_s = max(worst_s, float(np.abs(p[0, :] - bc_s).max()))
        worst_n = max(worst_n, float(np.abs(p[-1, :] - bc_n).max()))
    return worst_s, worst_n


if __name__ == '__main__':
    # Self-test: the constraint must hold for an UNTRAINED network on random
    # input, for both output contracts.
    ny = nx = 300
    y = np.linspace(1.6667, 998.3333, ny)
    mean, std = 88.328, 1.162

    class _CNNLike(nn.Module):
        def __init__(self): super().__init__(); self.c = nn.Conv2d(1, 1, 3, padding=1)
        def forward(self, x): return self.c(x)

    class _ConvLSTMLike(nn.Module):
        """(B, seq, H, W) -> (B, 1, H, W), as ConvLSTMModel does."""
        def __init__(self, seq=3):
            super().__init__(); self.c = nn.Conv2d(seq, 1, 3, padding=1)
        def forward(self, x): return self.c(x)

    for name, base, shape in [('CNN     ', _CNNLike(), (2, 1, ny, nx)),
                              ('ConvLSTM', _ConvLSTMLike(3), (2, 3, ny, nx))]:
        w = ConstrainedWrapper(base, y, mean, std, h_bc=90.0)
        out = w(torch.randn(*shape)).detach().numpy() * std + mean
        e = max(np.abs(out[:, 0, 0, :] - 90).max(),
                np.abs(out[:, 0, -1, :] - 90).max())
        n_extra = sum(p.numel() for p in w.parameters()) - sum(p.numel() for p in base.parameters())
        print(f"{name}  max |h - 90| on Dirichlet rows = {e:.3e}   "
              f"added parameters = {n_extra}   "
              f"{'PASS' if e < 1e-4 and n_extra == 0 else 'FAIL'}")

    # benchmark 2 profile
    hbc2 = np.sqrt(90.0**2 + (100.0**2 - 90.0**2) * (y / 1000.0))
    w2 = ConstrainedWrapper(_CNNLike(), y, 95.0, 3.0, h_bc=hbc2)
    o2 = w2(torch.randn(1, 1, ny, nx)).detach().numpy() * 3.0 + 95.0
    print(f"benchmark-2 profile: row 0 = {o2[0,0,0,:].mean():.4f} "
          f"(want {hbc2[0]:.4f}), row -1 = {o2[0,0,-1,:].mean():.4f} "
          f"(want {hbc2[-1]:.4f})")
