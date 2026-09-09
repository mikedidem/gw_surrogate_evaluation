# -*- coding: utf-8 -*-
"""
Benchmark 1, centred well: full reference fields and the supervised (sdata) set.

Mirrors gen_benchmark1.py exactly -- same discretisation, parameters, solver,
stress periods and export convention -- with two differences:

  * the well is at the domain centre (iwell = jwell = 150), which is the
    configuration the benchmark-1 CNN and ConvLSTM were trained on;
  * MODFLOW is run ONCE and exported twice, to the full 300 x 300 grid and to
    the locally refined observation points, so the two datasets cannot drift
    apart the way two separate runs could.

Y ORIENTATION -- deliberate, matching gen_benchmark1.py. ycent ascends while
flopy's H[row] starts at the north edge, so the written field is mirrored about
y = 500. For this benchmark that is a no-op: CHD is 90 m on both the north and
south edges, K is uniform and the well is centred, so the field is symmetric
about y = 500. The convention is kept because the CNN and ConvLSTM data were
exported this way. It is NOT a no-op for benchmark 2.

Outputs, under the B1 benchmark directory of the data root (see src/paths.py):
    t*.txt              33 full-grid reference fields, 90000 rows, `x y h`
    sdata/t*.txt        28 supervised files, t0.25..t25, TARGET_COUNT rows
    sdata/lrs_points.txt
    _mf/                the MODFLOW run itself

Requires MODFLOW-2005 on PATH, or MF2005_EXE naming the executable.
"""
import os
import sys
from pathlib import Path

import numpy as np
import flopy
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import paths  # noqa: E402

# Written under the data root; see src/paths.py. MODFLOW-2005 must be on
# PATH, or MF2005_EXE must name the executable.
BASE = str(paths.benchmark_dir("b1"))
MF_WS = os.path.join(BASE, "_mf")
FULL_DIR = BASE
SDATA_DIR = os.path.join(BASE, "sdata")
EXE = os.environ.get("MF2005_EXE", "mf2005")

WELL_SIGMA = 30.0           # Gaussian well half-width, m
WELL_TRUNC = 4.0            # distribute out to WELL_TRUNC * sigma
R_MIN, R_SLOPE, GRID_RES = 6.0, 0.08, 300
TARGET_COUNT = 815
SDATA_MAX_T = 25.0          # hold out the t = 26-30 test window

Lx = Ly = 1000.0
ncol = nrow = 300
nlay = 1
delr, delc = Lx / ncol, Ly / nrow
top, botm = 100.0, 0.0
K, Sy, Ss = 33.33, 0.10, 1e-6
h0, Qwell = 90.0, -40000.0

times = [0.25, 0.5, 0.75, 1.0] + list(range(2, 31))
perlen = [times[0]] + [times[i] - times[i - 1] for i in range(1, len(times))]
nper = len(perlen)

for d in (MF_WS, FULL_DIR, SDATA_DIR):
    os.makedirs(d, exist_ok=True)

ml = flopy.modflow.Modflow("b1_center", exe_name=EXE, model_ws=MF_WS)
flopy.modflow.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc,
                         top=top, botm=botm, nper=nper, perlen=perlen,
                         nstp=[5] + [3] * (nper - 1), tsmult=[1.0] * nper,
                         steady=[False] * nper)
flopy.modflow.ModflowBas(ml, ibound=np.ones((nlay, nrow, ncol), dtype=int), strt=h0)
flopy.modflow.ModflowLpf(ml, hk=K, vka=K, sy=Sy, ss=Ss, laytyp=1)

iwell, jwell = nrow // 2, ncol // 2

# The benchmark's well is a GAUSSIAN of sigma = 30 m, not a single cell: every
# benchmark-1 summary.json records well_sigma = 30.0 with well_rate = -40000,
# and the PINN's own source term is -4.0e4 * exp(-r^2/2sigma^2)/(2*pi*sigma^2).
# A one-cell WEL draws the centre down to ~77.5 m against the reference 83.0 m.
# Q is spread over cells by a normalised Gaussian weight, truncated at 4 sigma.
_xc = np.linspace(delr / 2.0, Lx - delr / 2.0, ncol)
_yc = np.linspace(delc / 2.0, Ly - delc / 2.0, nrow)
_X, _Y = np.meshgrid(_xc, _yc)
_r2 = (_X - _xc[jwell]) ** 2 + (_Y - _yc[iwell]) ** 2
_w = np.exp(-_r2 / (2.0 * WELL_SIGMA ** 2))
_w[_r2 > (WELL_TRUNC * WELL_SIGMA) ** 2] = 0.0
_w /= _w.sum()
_rows, _cols = np.nonzero(_w)
wel_data = [[0, int(i), int(j), float(Qwell * _w[i, j])] for i, j in zip(_rows, _cols)]
print(f"gaussian well: {len(wel_data)} cells, total Q = {sum(r[3] for r in wel_data):.1f} m3/d")
flopy.modflow.ModflowWel(ml, stress_period_data={0: wel_data})

chd_data = []
for j in range(ncol):
    chd_data.append([0, 0, j, h0, h0])
    chd_data.append([0, nrow - 1, j, h0, h0])
flopy.modflow.ModflowChd(ml, stress_period_data={0: chd_data})
flopy.modflow.ModflowPcg(ml, hclose=1e-5, rclose=5e-3, mxiter=400, iter1=150,
                         relax=0.97, damp=0.7)
flopy.modflow.ModflowOc(ml, stress_period_data={(k, 0): ['save head']
                                                for k in range(nper)})

mg = ml.modelgrid
WELL_CENTER = (float(mg.xcellcenters[iwell, jwell]),
               float(mg.ycellcenters[iwell, jwell]))
print("well cell (iwell, jwell) =", (iwell, jwell))
print("well centre, flopy model coords:", WELL_CENTER)

ml.write_input()
ok, _ = ml.run_model(silent=True)
if not ok:
    raise RuntimeError("MODFLOW failed")
print("MODFLOW run complete")


def generate_lrs(domain, well, rmin, rslope, grid_res, target):
    xmin, xmax, ymin, ymax = domain
    cx, cy = well

    def radius(p):
        return rmin + rslope * np.hypot(p[0] - cx, p[1] - cy)

    x = np.linspace(xmin, xmax, grid_res)
    y = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(x, y)
    cand = np.column_stack([X.ravel(), Y.ravel()])
    np.random.seed(42)
    np.random.shuffle(cand)

    pts, tree = [], None
    for p in cand:
        if not pts:
            pts.append(p)
            tree = cKDTree([p])
            continue
        if tree.query(p)[0] > radius(p):
            pts.append(p)
            tree = cKDTree(np.asarray(pts))
        if len(pts) >= target:
            break
    return np.asarray(pts)


xcent = np.linspace(delr / 2.0, Lx - delr / 2.0, ncol)
ycent = np.linspace(delc / 2.0, Ly - delc / 2.0, nrow)

Xg, Yg = np.meshgrid(xcent, ycent)
full_pts = np.column_stack([Xg.ravel(), Yg.ravel()])

# the LRS density keys off the exported well position, not flopy's internal one
lrs_pts = generate_lrs((0.0, Lx, 0.0, Ly), (xcent[jwell], ycent[iwell]),
                       R_MIN, R_SLOPE, GRID_RES, TARGET_COUNT)
lrs_pts[:, 0] = np.clip(lrs_pts[:, 0], xcent.min(), xcent.max())
lrs_pts[:, 1] = np.clip(lrs_pts[:, 1], ycent.min(), ycent.max())
np.savetxt(os.path.join(SDATA_DIR, "lrs_points.txt"), lrs_pts, fmt="%.4f",
           delimiter=",")
print(f"LRS points: {len(lrs_pts)}")

hds = flopy.utils.HeadFile(os.path.join(MF_WS, "b1_center.hds"))
ts = np.array(hds.get_times())
n_full = n_sd = 0
for t in times:
    H = hds.get_data(totim=ts[np.argmin(np.abs(ts - t))])[0]
    interp = RegularGridInterpolator((ycent, xcent), H, bounds_error=False,
                                     fill_value=np.nan, method="linear")
    t_str = str(int(t)) if float(t).is_integer() else str(t)

    h = interp(full_pts[:, [1, 0]])
    np.savetxt(os.path.join(FULL_DIR, f"t{t_str}.txt"),
               np.c_[full_pts, h], fmt="%.4f %.4f %.4f")
    n_full += 1

    if t <= SDATA_MAX_T:
        h = interp(lrs_pts[:, [1, 0]])
        np.savetxt(os.path.join(SDATA_DIR, f"t{t_str}.txt"),
                   np.c_[lrs_pts, h], fmt="%.4f %.4f %.4f")
        n_sd += 1

print(f"wrote {n_full} full-grid files -> {FULL_DIR}")
print(f"wrote {n_sd} sdata files      -> {SDATA_DIR}")
