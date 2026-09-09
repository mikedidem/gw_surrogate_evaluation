#!/usr/bin/env python
"""
Build modflow/o1.csv, the small hold-out sample that Trainer.test() prints for
stage 2.

Format expected by trainer.py:   x  y  t  h    whitespace separated, no header,
coordinates in MODFLOW's 0..1000 frame (test() applies the -500 shift itself).

The sample is stratified rather than uniform, so the printed comparison covers
the parts of the field that actually distinguish this benchmark: the steep
near-well cone, inside the low-K lens, the lens rim where grad K is largest,
and both Dirichlet boundaries carrying the regional gradient.

    python make_o1.py                # t = 30, ~32 points
    python make_o1.py --t 20 --n 60
"""
import argparse
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--t', type=float, default=30.0, help='snapshot time')
ap.add_argument('--n', type=int, default=32, help='approximate number of points')
ap.add_argument('--src', default=os.path.join(HERE, 'modflow'),
                help='folder holding t*.txt')
ap.add_argument('--out', default=os.path.join(HERE, 'modflow', 'o1.csv'))
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

ts = str(int(args.t)) if float(args.t).is_integer() else str(args.t)
src = os.path.join(args.src, f't{ts}.txt')
d = np.genfromtxt(src)
x, y, h = d[:, 0], d[:, 1], d[:, 2]

# centred coords, to define the zones
xc, yc = x - 500.0, y - 500.0
r = np.hypot(xc, yc)

# lens geometry — keep in step with problem.py
LX, LY, W = (-300.0, 300.0), (240.0, 390.0), 10.0
inside_lens = ((xc > LX[0]) & (xc < LX[1]) & (yc > LY[0]) & (yc < LY[1]))
near_rim = (~inside_lens
            & (xc > LX[0] - 3*W) & (xc < LX[1] + 3*W)
            & (yc > LY[0] - 3*W) & (yc < LY[1] + 3*W))

zones = {
    'near well  (r < 90 m)':      r < 90,
    'inside lens':                inside_lens,
    'lens rim':                   near_rim,
    'background':                 (r > 150) & ~inside_lens & ~near_rim & (np.abs(yc) < 450),
    'near north bdy':             yc > 450,
    'near south bdy':             yc < -450,
}

rng = np.random.default_rng(args.seed)
per = max(1, args.n // len(zones))
idx = []
for name, m in zones.items():
    pool = np.flatnonzero(m)
    take = min(per, len(pool))
    idx.append(rng.choice(pool, size=take, replace=False))
    print(f'  {name:24s} {take:3d} points   (pool {len(pool)})')
idx = np.unique(np.concatenate(idx))

out = np.c_[x[idx], y[idx], np.full(idx.size, args.t), h[idx]]
out = out[np.argsort(-out[:, 3])]          # deepest drawdown first, easier to read

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savetxt(args.out, out, fmt='%.4f %.4f %g %.4f')

print(f'\nwrote {args.out}')
print(f'  {len(out)} points at t = {args.t}')
print(f'  x  {out[:,0].min():.1f} .. {out[:,0].max():.1f}   (0..1000 frame)')
print(f'  h  {out[:,3].min():.4f} .. {out[:,3].max():.4f}')
