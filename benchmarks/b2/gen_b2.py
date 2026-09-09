# -*- coding: utf-8 -*-
"""
Benchmark 2 generator — heterogeneous K + regional gradient.

Derived from updated_genmodflow.py (which produced hybrid_wellupdated_full,
the published baseline). Everything is identical to that baseline except:

  1. K is a low-conductivity lens embedded in a 33.33 m/d field.
     The lens touches no boundary, so all four edges keep the baseline K.
  2. The Dirichlet boundaries carry different heads (N = 100, S = 90),
     producing a regional gradient, north to south, that the pumping cone is
     superimposed on. The lens sits between the well and the upgradient
     boundary, so it throttles the well's principal supply.
  3. The initial head is the analytic Dupuit profile for that gradient,
     NOT a uniform 90 m.

Outputs, in 0-1000 model coordinates (trainer.py subtracts 500 at load).
Set OUTPUT_MODE to choose what gets written:

  "both"  (default)  full fields AND the LRS subset, from the same run
  "full"              full 300x300 fields only
  "lrs"               locally refined points only

  <OUT>/t*.txt          full 300x300 field   -> CNN / ConvLSTM + all evaluation
  <OUT>/sdata/t*.txt    LRS points, t <= 25  -> PINN supervised loss
  <OUT>/kfield.txt      x y K at cell centres -> diagnostics + verification

Note on orientation: every array below is indexed off flopy's modelgrid, which
is the same indexing MODFLOW returns heads in (row 0 = north). Nothing is
re-gridded, so the K field, the head field and the exported coordinates cannot
drift out of alignment.
"""

import os
import shutil
import sys
from pathlib import Path
import numpy as np
import flopy
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

# ==========================================================
# OUTPUT MODE
# ==========================================================
#   "both" -> full fields and the LRS subset from one run  (recommended:
#             one MODFLOW solve, so the two can never drift apart)
#   "full" -> full 300x300 fields only
#   "lrs"  -> locally refined points only
OUTPUT_MODE = os.environ.get("B2_OUTPUT_MODE", "lrs").lower()

# ==========================================================
# PATHS  — resolved against this file, not the shell's cwd,
#          so the script runs correctly from anywhere.
# ==========================================================
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import paths  # noqa: E402

OUT_BASE = str(paths.data_root() / "benchmarks")
SUBDIR   = os.environ.get("B2_SUBDIR", "b2")
OUT_DIR  = os.path.join(OUT_BASE, SUBDIR)
SDATA_DIR = os.path.join(OUT_DIR, "sdata")
os.makedirs(SDATA_DIR, exist_ok=True)

exe_path = os.environ.get("MF2005_EXE", "mf2005")
if not (os.path.exists(exe_path) or shutil.which(exe_path)):
    raise SystemExit(
        f"MODFLOW-2005 executable not found: {exe_path}. "
        f"Put it on PATH, or set MF2005_EXE to its location."
    )

if OUTPUT_MODE not in ("both", "full", "lrs"):
    raise SystemExit(f"OUTPUT_MODE must be 'both', 'full' or 'lrs' (got {OUTPUT_MODE!r})")
print(f"output mode: {OUTPUT_MODE}   ->  {OUT_DIR}")

# ==========================================================
# GRID / TIME  — identical to the published baseline
# ==========================================================
Lx, Ly = 1000.0, 1000.0
ncol, nrow, nlay = 300, 300, 1
delr, delc = Lx / ncol, Ly / nrow

top, botm = 100.0, 0.0
Sy = 0.10
Ss = 1e-6

Qwell = -40000.0          # total extraction, one well
sigma_well = 30.0         # MUST match the sigma used in PINN training

times  = [0.25, 0.5, 0.75, 1.0] + list(range(2, 31))
perlen = [times[0]] + [times[i] - times[i-1] for i in range(1, len(times))]
nper   = len(perlen)
nstp   = [5] + [3]*(nper-1)
tsmult = [1.0]*nper
steady = [False] * nper

TRAIN_TMAX = 25.0         # sdata covers t <= 25; t = 26-30 held out

# ==========================================================
# BENCHMARK 2 PARAMETERS
# ==========================================================
# --- regional gradient: heads on the two Dirichlet boundaries -------------
H_NORTH = 100.0           # y = Ly   (flopy row 0)      upgradient
H_SOUTH = 90.0            # y = 0    (flopy row nrow-1)  downgradient

# --- low-K lens, given in CENTRED coords (-500..+500) ---------------------
K_HI   = 33.33            # background, the baseline value
K_LO   = 3.333           # lens, 10:1 contrast
LENS_X = (-300.0, 300.0)  # 600 m wide  -> flow can divert around both ends
LENS_Y = ( 240.0, 390.0)  # 150 m thick -> 59 m clear of the 6-sigma well
                          #                support, 110 m clear of boundary
LENS_W = 10.0             # tanh edge scale; ~40 m transition, ~12 cells

# LRS sampling controls (unchanged from the baseline generator)
R_MIN, R_SLOPE, GRID_RES, TARGET_COUNT = 6.0, 0.08, 300, 1000


def k_of(xc, yc):
    """K on centred coordinates. Product of tanh ramps: smooth on every side
    and at every corner, so grad(K) is analytic and bounded everywhere."""
    sx = 0.5 * (np.tanh((xc - LENS_X[0]) / LENS_W)
                - np.tanh((xc - LENS_X[1]) / LENS_W))
    sy = 0.5 * (np.tanh((yc - LENS_Y[0]) / LENS_W)
                - np.tanh((yc - LENS_Y[1]) / LENS_W))
    return K_HI + (K_LO - K_HI) * sx * sy


def h_initial(y_model):
    """Analytic Dupuit profile for the regional gradient.

    Steady unconfined flow with no recharge gives h^2 linear in y, so this is
    the exact solution of the homogeneous problem and satisfies both Dirichlet
    conditions identically. Used as MODFLOW's strt AND as the PINN's hstar --
    they must be the same function or the two are solving different problems.
    """
    frac = y_model / Ly                      # 0 at south, 1 at north
    return np.sqrt(H_SOUTH**2 + (H_NORTH**2 - H_SOUTH**2) * frac)


# ==========================================================
# BUILD
# ==========================================================
modelname = "b2_model"
ml = flopy.modflow.Modflow(modelname, exe_name=exe_path, model_ws=OUT_DIR)

dis = flopy.modflow.ModflowDis(
    ml, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc,
    top=top, botm=botm, nper=nper, perlen=perlen, nstp=nstp,
    tsmult=tsmult, steady=steady,
)

mg = ml.modelgrid
xc = mg.xcellcenters          # (nrow, ncol), model coords 0..1000
yc = mg.ycellcenters          # row 0 is NORTH (y = 998.33)

xc_c, yc_c = xc - 500.0, yc - 500.0      # centred coords for the K field
K_arr = k_of(xc_c, yc_c)

print(f"K field: {K_arr.min():.3f} .. {K_arr.max():.3f} m/d"
      f"   ({100*(K_arr < 0.5*(K_HI+K_LO)).mean():.2f}% of cells inside lens)")

# initial head: regional profile, not uniform
strt = h_initial(yc)
print(f"strt: {strt.min():.3f} .. {strt.max():.3f} m")

ibound = np.ones((nlay, nrow, ncol), dtype=int)
bas = flopy.modflow.ModflowBas(ml, ibound=ibound,
                               strt=strt.reshape(nlay, nrow, ncol))

lpf = flopy.modflow.ModflowLpf(
    ml,
    hk=K_arr.reshape(nlay, nrow, ncol),
    vka=K_arr.reshape(nlay, nrow, ncol),
    sy=Sy, ss=Ss, laytyp=1,
)

# ---------- Gaussian-distributed well, centre of the domain ----------
iwell, jwell = nrow // 2, ncol // 2
xw, yw = float(xc[iwell, jwell]), float(yc[iwell, jwell])

r2 = (xc - xw)**2 + (yc - yw)**2
weights = np.exp(-r2 / (2.0 * sigma_well**2))
weights /= weights.sum()
q_per_cell = Qwell * weights

wel_spd = [[0, i, j, q_per_cell[i, j]]
           for i in range(nrow) for j in range(ncol)
           if abs(q_per_cell[i, j]) > 1e-6]

print(f"Well at ({xw:.2f}, {yw:.2f}), sigma={sigma_well} m, "
      f"{len(wel_spd)} cells, total Q = {sum(r[3] for r in wel_spd):.2f}")

wel = flopy.modflow.ModflowWel(
    ml, stress_period_data={k: wel_spd for k in range(nper)})

# ---------- CHD: different head on each boundary ----------
chd_data = []
for j in range(ncol):
    chd_data.append([0, 0,      j, H_NORTH, H_NORTH])   # row 0    = north
    chd_data.append([0, nrow-1, j, H_SOUTH, H_SOUTH])   # row n-1  = south
chd = flopy.modflow.ModflowChd(ml, stress_period_data={0: chd_data})

pcg = flopy.modflow.ModflowPcg(ml, hclose=1e-5, rclose=5e-3,
                               mxiter=400, iter1=150, relax=0.97, damp=0.7)
oc = flopy.modflow.ModflowOc(
    ml,
    stress_period_data={(k, nstp[k] - 1): ['save head'] for k in range(nper)},
)

# ==========================================================
# LRS POINTS (for the PINN supervised set)
# ==========================================================
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
            pts.append(p); tree = cKDTree([p]); continue
        d, _ = tree.query(p)
        if d > radius(p):
            pts.append(p); tree = cKDTree(np.asarray(pts))
        if len(pts) >= target:
            break
    return np.asarray(pts)


lrs_points = None
if OUTPUT_MODE in ("both", "lrs"):
    lrs_points = generate_lrs((0.0, Lx, 0.0, Ly), (xw, yw),
                              R_MIN, R_SLOPE, GRID_RES, TARGET_COUNT)

    # Clip points to the interpolation grid to avoid RGI bounds errors.
    # Restored from gen.py (the benchmark-1 generator), which has had these two
    # lines from the start. The sampler draws over the full 0..1000 domain but
    # RegularGridInterpolator is built on CELL CENTRES, 1.6667..998.3333, so
    # points on the outer 1.67 m fall outside and come back NaN -- 13 of them
    # here. Clipping snaps those onto the nearest cell centre, which is what
    # benchmark 1's supervised set does.
    x_min_grid, x_max_grid = xc.min(), xc.max()
    y_min_grid, y_max_grid = yc.min(), yc.max()
    lrs_points[:, 0] = np.clip(lrs_points[:, 0], x_min_grid, x_max_grid)
    lrs_points[:, 1] = np.clip(lrs_points[:, 1], y_min_grid, y_max_grid)

    print(f"LRS points: {len(lrs_points)}")
    np.savetxt(os.path.join(OUT_DIR, "lrs_points.txt"),
               lrs_points, fmt="%.4f", delimiter=",")

# K field for the diagnostics, on the same cell centres as the heads
np.savetxt(os.path.join(OUT_DIR, "kfield.txt"),
           np.c_[xc.ravel(), yc.ravel(), K_arr.ravel()],
           fmt="%.4f %.4f %.6f")

# ==========================================================
# RUN
# ==========================================================
ml.write_input()
ok, buff = ml.run_model(silent=False)
if not ok:
    raise RuntimeError("MODFLOW failed")

# ==========================================================
# EXPORT
# ==========================================================
hds = flopy.utils.HeadFile(os.path.join(OUT_DIR, f"{modelname}.hds"))
avail = np.array(hds.get_times())
if len(avail) != len(times) or not np.allclose(avail, times, rtol=0.0, atol=5e-5):
    raise RuntimeError(
        "saved MODFLOW times do not match requested snapshot times:\n"
        f"  requested={times}\n  available={avail.tolist()}"
    )

# ascending-y views, used only for interpolating onto the LRS points
y_asc = yc[::-1, 0]
x_ax = xc[0, :]

full_flat = np.c_[xc.ravel(), yc.ravel()]

n_full = n_lrs = 0
for t, saved_time in zip(times, avail):
    H = hds.get_data(totim=saved_time)[0]
    t_str = str(int(t)) if float(t).is_integer() else str(t)

    # --- full field: ravel directly, no re-gridding, no flip ---
    if OUTPUT_MODE in ("both", "full"):
        np.savetxt(os.path.join(OUT_DIR, f"t{t_str}.txt"),
                   np.c_[full_flat, H.ravel()], fmt="%.4f %.4f %.4f")
        n_full += 1

    # --- LRS subset for the PINN training window ---
    if OUTPUT_MODE in ("both", "lrs") and t <= TRAIN_TMAX:
        interp = RegularGridInterpolator(
            (y_asc, x_ax), H[::-1, :],
            bounds_error=False, fill_value=np.nan, method="linear")
        h_lrs = interp(lrs_points[:, [1, 0]])
        np.savetxt(os.path.join(SDATA_DIR, f"t{t_str}.txt"),
                   np.c_[lrs_points, h_lrs], fmt="%.4f %.4f %.4f")
        n_lrs += 1

    print(f"t={t:5.2f}  head {H.min():7.3f} .. {H.max():7.3f}")

# ==========================================================
# SANITY CHECKS
# ==========================================================
H0 = hds.get_data(totim=avail[0])[0]
Hf = hds.get_data(totim=avail[-1])[0]

dy = delc
def bflux(H, row_b, row_i, K_row):
    """Darcy inflow across one Dirichlet row, positive into the domain."""
    hb, hi = H[row_b, :], H[row_i, :]
    return float(np.sum(K_row * 0.5 * (hb + hi) * (hb - hi) / dy * delr))

qn0 = bflux(H0, 0, 1, K_arr[0, :])
qs0 = bflux(H0, nrow-1, nrow-2, K_arr[nrow-1, :])
qnf = bflux(Hf, 0, 1, K_arr[0, :])
qsf = bflux(Hf, nrow-1, nrow-2, K_arr[nrow-1, :])

print("\n--- checks ---")
print(f"drawdown vs initial, at t=30 : {float((h_initial(yc) - Hf).max()):.3f} m")
print(f"boundary flux  north  t={times[0]:.2f} : {qn0:+11.1f}   t=30 : {qnf:+11.1f} m3/d")
print(f"boundary flux  south  t={times[0]:.2f} : {qs0:+11.1f}   t=30 : {qsf:+11.1f} m3/d")
print(f"net inflow at t=30           : {qnf + qsf:+11.1f}  (well = {Qwell:.0f})")
print(f"K at every boundary row/col  : "
      f"N {K_arr[0].min():.2f}-{K_arr[0].max():.2f}  "
      f"S {K_arr[-1].min():.2f}-{K_arr[-1].max():.2f}  "
      f"W {K_arr[:,0].min():.2f}-{K_arr[:,0].max():.2f}  "
      f"E {K_arr[:,-1].min():.2f}-{K_arr[:,-1].max():.2f}   (all should be "
      f"{K_HI})")
print(f"\nmode '{OUTPUT_MODE}':  {n_full} full fields -> {OUT_DIR}")
print(f"                {n_lrs} LRS files    -> {SDATA_DIR}")
