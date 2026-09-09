import os
import glob, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import plot_test_cnn_vs_modflow, mae, mse, rrmse  # noqa: E402
from constrained import (  # noqa: E402
    DirichletRowProjection, boundary_error, chd_row_kwargs, h_bc_for,
    BENCHMARKS, data_path_for)

# =========================
# Utils
# =========================
def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_txt_to_grid(path):
    """
    Load MODFLOW txt file (x y h) → structured 2D grid
    """
    data = np.loadtxt(path)
    x, y, h = data[:, 0], data[:, 1], data[:, -1]

    x_unique = np.unique(x)
    y_unique = np.unique(y)

    nx, ny = len(x_unique), len(y_unique)

    xi = np.searchsorted(x_unique, x)
    yi = np.searchsorted(y_unique, y)

    H = np.full((ny, nx), np.nan)
    H[yi, xi] = h
    
    # Fill NaNs with nearest neighbor interpolation
    if np.any(np.isnan(H)):
        from scipy.interpolate import griddata
        mask = ~np.isnan(H)
        yy, xx = np.mgrid[0:ny, 0:nx]
        H = griddata((yy[mask], xx[mask]), H[mask], (yy, xx), method='nearest')

    return H, x_unique, y_unique


# =========================
# Dataset
# =========================
class GroundwaterDataset(Dataset):
    """
    (H_t-1) → (H_t) with temporal splits
    """
    def __init__(self, txt_files, start_idx=0, end_idx=None, mean=None, std=None,
                 grids=None):
        if grids is None:
            if end_idx is None:
                end_idx = len(txt_files)
            self.txt_files = list(txt_files[start_idx:end_idx])
            self.grids = np.asarray([
                load_txt_to_grid(path)[0] for path in self.txt_files
            ])
        else:
            self.txt_files = list(txt_files)
            self.grids = np.asarray(grids)
            if len(self.txt_files) != len(self.grids):
                raise ValueError("txt_files and grids must describe the same states")

        # Normalize (use provided stats or compute from data)
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean = np.mean(self.grids)
            self.std = np.std(self.grids)
        
        self.grids = (self.grids - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.grids) - 1

    def __getitem__(self, idx):
        x = self.grids[idx]
        y = self.grids[idx + 1]

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        return x, y


# =========================
# CNN Model (Regression)
# =========================
class HeadCNN(nn.Module):
    """Published CNN surrogate. `width` and `depth` exposed for the sweep that
    provides the more extensive hyperparameter search.

    The published model is width=32, depth=3 (the file's commented-out
    128/128/64 block was an earlier attempt at a wider version). Widening
    changes capacity but NOT receptive field: three 3x3 layers see 7 cells,
    23 m of a 1000 m domain, at any width. If accuracy does not improve with
    width, capacity and tuning are excluded and the limit is architectural.

    `depth` does grow the receptive field, by 2 cells per layer -- reaching
    the domain would need ~150 layers, which is why dilation is the practical
    route.
    """

    def __init__(self, in_channels=1, constrained=False, bc=None,
                 width=32, depth=3):
        super().__init__()
        layers, c = [], in_channels
        for _ in range(depth - 1):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.GELU()]
            c = width
        layers += [nn.Conv2d(c, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.depth = depth

        if constrained and bc is None:
            raise ValueError("constrained=True needs bc=dict(...)")
        self.bc = DirichletRowProjection(**bc) if constrained else None

    def receptive_field(self):
        return 1 + 2 * self.depth

    def forward(self, x):
        u = self.net(x)
        return u if self.bc is None else self.bc(u)


class DilatedHeadCNN(nn.Module):
    """Same idea as HeadCNN, with a receptive field that spans the domain.

    HeadCNN is three 3x3 convolutions: an output pixel sees 7 cells, 23 m of a
    1000 m domain. Groundwater flow is elliptic -- pumping at the centre moves
    the head everywhere, and Dirichlet boundaries hold it 500 m away -- so a
    23 m window cannot represent the response, and no amount of extra channels
    changes that. It also explains the ~0.8 m boundary error: an interior pixel
    has no way to know a boundary exists.

    Dilating instead grows the receptive field geometrically at constant
    kernel size:

        dilation 1,2,4,8,16,32,64,128  ->  511 cells = 1703 m  (covers it)

    Capacity is comparable to HeadCNN's 9,857 and the PINN's 10,451, so this
    isolates receptive field rather than confounding it with model size --
    which matters, because the comparison requires comparable accuracy and
    "bigger network" is not a diagnosis.
    """

    def __init__(self, in_channels=1, channels=24,
                 dilations=(1, 2, 4, 8, 16, 32, 64, 128),
                 constrained=False, bc=None):
        super().__init__()
        layers, c_in = [], in_channels
        for d in dilations:
            layers += [nn.Conv2d(c_in, channels, 3, padding=d, dilation=d),
                       nn.GELU()]
            c_in = channels
        layers += [nn.Conv2d(c_in, 1, 1)]
        self.net = nn.Sequential(*layers)
        self.dilations = tuple(dilations)

        if constrained and bc is None:
            raise ValueError("constrained=True needs bc=dict(...)")
        self.bc = DirichletRowProjection(**bc) if constrained else None

    def receptive_field(self):
        return 1 + sum(2 * d for d in self.dilations)

    def forward(self, x):
        u = self.net(x)
        return u if self.bc is None else self.bc(u)


# =========================
# Metrics
# =========================
def mae(pred, true):
    return torch.mean(torch.abs(pred - true)).item()

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()


# =========================
# Multi-Step Prediction
# =========================
def multi_step_predict(model, initial_state, n_steps, device):
    """
    Predict multiple time steps ahead
    """
    model.eval()
    predictions = []
    current = initial_state.to(device)
    
    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(current)
            predictions.append(pred.cpu())
            current = pred
    
    return predictions

def extract_time(fname):
        base = os.path.basename(fname)
        m = re.search(r"t([0-9]+(?:\.[0-9]+)?)", base)
        if not m:
            raise ValueError(f"Could not extract time from filename: {fname}")
        return float(m.group(1))


def benchmark_initial_head(benchmark, y_grid, nx):
    """Exact MODFLOW starting head for the selected benchmark."""
    if benchmark == "b1":
        return np.full((len(y_grid), nx), 90.0, dtype=np.float64)
    if benchmark != "b2":
        raise ValueError(f"unknown benchmark: {benchmark}")

    y_grid = np.asarray(y_grid, dtype=np.float64)
    dy = float(np.median(np.diff(y_grid)))
    y0 = float(y_grid[0] - 0.5 * dy)
    length = float(y_grid[-1] + 0.5 * dy - y0)
    eta = np.clip((y_grid - y0) / length, 0.0, 1.0)
    profile = np.sqrt(90.0 ** 2 + (100.0 ** 2 - 90.0 ** 2) * eta)
    return np.repeat(profile[:, None], nx, axis=1)


def daily_protocol(txt_files, benchmark):
    """Return t=0..30 daily states and the locked target-day split."""
    if isinstance(txt_files, (str, os.PathLike)):
        txt_files = glob.glob(os.fspath(txt_files))
    by_time = {extract_time(path): path for path in txt_files}
    required = [float(day) for day in range(1, 31)]
    missing = [time for time in required if time not in by_time]
    if missing:
        raise ValueError(
            f"{benchmark.upper()} daily protocol is missing snapshots: {missing}"
        )

    daily_files = [by_time[time] for time in required]
    _, x_grid, y_grid = load_txt_to_grid(daily_files[0])
    initial = benchmark_initial_head(benchmark, y_grid, len(x_grid))
    daily_grids = [initial] + [load_txt_to_grid(path)[0] for path in daily_files]
    state_files = [None] + daily_files

    if not np.allclose(initial[:, 0], initial[:, -1]):
        raise ValueError("day-zero head must be constant along each row")
    expected_rows = (
        (90.0, 90.0) if benchmark == "b1" else (90.01759, 99.98416)
    )
    if not np.isclose(initial[0, 0], expected_rows[0], atol=5e-5):
        raise ValueError(f"unexpected south day-zero head: {initial[0, 0]}")
    if not np.isclose(initial[-1, 0], expected_rows[1], atol=5e-5):
        raise ValueError(f"unexpected north day-zero head: {initial[-1, 0]}")

    return {
        "times": np.arange(0.0, 31.0, dtype=np.float64),
        "daily_files": daily_files,
        "state_files": state_files,
        "grids": np.asarray(daily_grids),
        "x_grid": x_grid,
        "y_grid": y_grid,
        "train": slice(0, 21),   # states t=0..20 -> targets 1..20
        "validation": slice(20, 26),  # states t=20..25 -> targets 21..25
        "test": slice(25, 31),   # states t=25..30 -> targets 26..30
    }


def default_output_dir(benchmark, constrained, arch, channels, width, depth):
    """Return a non-overwriting directory name that records the experiment."""
    out_dir = f"outputs_{benchmark}"
    out_dir = f"{out_dir}_corrected_daily_split" if benchmark == "b2" else f"{out_dir}_daily_split"
    if arch != "plain":
        out_dir = f"{out_dir}_{arch}{channels}"
    elif (width, depth) != (32, 3):
        out_dir = f"{out_dir}_w{width}d{depth}"
    return f"{out_dir}_{'chd_rows' if constrained else 'unconstrained'}"


def save_prediction_txt(field, target_file, out_path):
    """Save a south-to-north grid using the target file's coordinate order."""
    data = np.loadtxt(target_file)
    x, y = data[:, 0], data[:, 1]
    ux, uy = np.unique(x), np.unique(y)
    xi = np.searchsorted(ux, x)
    yi = np.searchsorted(uy, y)

    field = np.asarray(field).reshape(len(uy), len(ux))
    out = np.column_stack([x, y, field[yi, xi]])
    np.savetxt(
        out_path,
        out,
        fmt="%.6f",
        header="x y h",
        comments="# ",
    )


@torch.no_grad()
def save_operational_rollout(model, txt_files, mean, std, device, out_dir,
                             start_time=25.0,
                             target_times=(26.0, 27.0, 28.0, 29.0, 30.0)):
    """Recursively forecast locked targets from one observed initial field."""
    by_time = {extract_time(path): path for path in txt_files}
    required = (float(start_time),) + tuple(float(t) for t in target_times)
    missing = [t for t in required if t not in by_time]
    if missing:
        raise ValueError(f"operational rollout is missing snapshots: {missing}")

    initial, _, _ = load_txt_to_grid(by_time[float(start_time)])
    current = torch.tensor(
        (initial - mean) / (std + 1e-8), dtype=torch.float32
    )[None, None].to(device)

    os.makedirs(out_dir, exist_ok=True)
    rmse_by_time = {}
    model.eval()
    for target_time in target_times:
        target_time = float(target_time)
        prediction_norm = model(current)
        prediction = prediction_norm[0, 0].cpu().numpy() * std + mean

        target_file = by_time[target_time]
        basename = os.path.basename(target_file)
        out_path = os.path.join(out_dir, f"cnn_pred_{basename}")
        save_prediction_txt(prediction, target_file, out_path)

        reference, _, _ = load_txt_to_grid(target_file)
        rmse_m = float(np.sqrt(np.mean((prediction - reference) ** 2)))
        rmse_by_time[f"{target_time:g}"] = rmse_m
        print(f"  rollout t={target_time:g}: RMSE={rmse_m:.6f} m -> {out_path}")

        # Operational recurrence uses this prediction as the next input.
        current = prediction_norm

    return {
        'start_time': float(start_time),
        'target_times': [float(t) for t in target_times],
        'rmse_m_by_time': rmse_by_time,
        'mean_rmse_m': float(np.mean(list(rmse_by_time.values()))),
    }


# =========================
# Training
# =========================
def train(seed=42, epochs=100, lr=1e-3, batch_size=4, data_path=None,
          constrained=True, benchmark="b1", out_dir=None,
          arch="plain", channels=24, width=32, depth=3):
    # data_path=None -> the benchmark's own snapshots, resolved against
    # gw_mod/ so it works from any cwd. An explicit path always wins.
    if data_path is None:
        data_path = data_path_for(benchmark)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    

    txt_files = sorted(glob.glob(data_path), key=extract_time)

    if len(txt_files) == 0:
        raise ValueError(f"No files found at {data_path}")
    
    if benchmark in ("b1", "b2"):
        protocol = daily_protocol(txt_files, benchmark)
        grids = protocol["grids"]
        state_files = protocol["state_files"]
        train_slice = protocol["train"]
        val_slice = protocol["validation"]
        test_slice = protocol["test"]

        train_ds = GroundwaterDataset(
            state_files[train_slice], grids=grids[train_slice]
        )
        val_ds = GroundwaterDataset(
            state_files[val_slice], grids=grids[val_slice],
            mean=train_ds.mean, std=train_ds.std,
        )
        test_ds = GroundwaterDataset(
            state_files[test_slice], grids=grids[test_slice],
            mean=train_ds.mean, std=train_ds.std,
        )
        txt_files = protocol["daily_files"]
        y_grid = protocol["y_grid"]
        print(f"{benchmark.upper()} fixed-step protocol (subdaily snapshots excluded):")
        print("  training targets:   days 1-20  (20 transitions)")
        print("  validation targets: days 21-25 (5 transitions)")
        print("  test targets:       days 26-30 (5 transitions)")
        print("  split ratio:        66.7% / 16.7% / 16.7%")
        initial_name = "uniform" if benchmark == "b1" else "Dupuit"
        print(
            f"  day-zero {initial_name} rows: "
            f"south={grids[0, 0, 0]:.5f} m, north={grids[0, -1, 0]:.5f} m"
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # ---- the only difference between the two arms ----
    bc_kw = chd_row_kwargs(benchmark, y_grid, train_ds.mean, train_ds.std)
    if arch == 'plain':
        model = HeadCNN(constrained=constrained,
                        bc=bc_kw if constrained else None,
                        width=width, depth=depth).to(device)
    else:
        model = DilatedHeadCNN(channels=channels, constrained=constrained,
                               bc=bc_kw if constrained else None).to(device)

    cell = 1000.0 / len(y_grid)
    rf = model.receptive_field() if hasattr(model, 'receptive_field') else 7
    print(f"\nBENCHMARK {benchmark}: {BENCHMARKS[benchmark]['label']}")
    print(f"ARCH: {arch}  receptive field {rf} cells = {rf*cell:.0f} m "
          f"({100*rf*cell/1000:.0f}% of the domain)")
    print(f"  parameters {sum(p.numel() for p in model.parameters()):,}")
    if constrained:
        print("MODEL: constrained -- Dirichlet by construction")
        print(f"  {model.bc.describe()}")
    else:
        print("MODEL: unconstrained -- the published baseline")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                             factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # separate output directory so the two arms never overwrite each other
    if out_dir is None:
        out_dir = default_output_dir(
            benchmark, constrained, arch, channels, width, depth
        )
    os.makedirs(out_dir, exist_ok=True)
    print(f"  outputs -> {out_dir}/")
    checkpoint_path = os.path.join(out_dir, "cnn_best_model.pth")
    if epochs == 0:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"export-only evaluation needs an existing checkpoint: "
                f"{checkpoint_path}"
            )
        print("EXPORT ONLY: skipping training and using the saved checkpoint")
    
    best_val_loss = float('inf')
    history = {
        'train_mse_normalized': [],
        'validation_mse_normalized': [],
        'train_rmse_m': [],
        'validation_mse_m2': [],
        'validation_mae_m': [],
        'validation_rmse_m': [],
        'units': {
            'train_mse_normalized': 'dimensionless',
            'validation_mse_normalized': 'dimensionless',
            'train_rmse_m': 'm',
            'validation_mse_m2': 'm2',
            'validation_mae_m': 'm',
            'validation_rmse_m': 'm',
        },
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_mae_sum, val_rmse_sum = 0, 0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)

                val_loss += criterion(pred, yb).item()
                val_mae_sum += mae(pred, yb)
                val_rmse_sum += rmse(pred, yb)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae_norm = val_mae_sum / len(val_loader)
        avg_val_rmse_norm = val_rmse_sum / len(val_loader)
        train_rmse_m = np.sqrt(avg_train_loss) * train_ds.std
        val_mse_m2 = avg_val_loss * train_ds.std ** 2
        val_mae_m = avg_val_mae_norm * train_ds.std
        val_rmse_m = avg_val_rmse_norm * train_ds.std

        history['train_mse_normalized'].append(avg_train_loss)
        history['validation_mse_normalized'].append(avg_val_loss)
        history['train_rmse_m'].append(float(train_rmse_m))
        history['validation_mse_m2'].append(float(val_mse_m2))
        history['validation_mae_m'].append(float(val_mae_m))
        history['validation_rmse_m'].append(float(val_rmse_m))
        
        scheduler.step(avg_val_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train RMSE: {train_rmse_m:.4f} m | "
                f"Val MSE: {val_mse_m2:.6f} m2 | "
                f"Val MAE: {val_mae_m:.4f} m | "
                f"Val RMSE: {val_rmse_m:.4f} m"
            )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_mse_normalized': best_val_loss,
                'mean': train_ds.mean,
                'std': train_ds.std,
                'seed': seed,
                'benchmark': benchmark,
                'data_path': data_path,
                'arch': arch,
                'constrained': bool(constrained),
                'constraint': ('hard_dirichlet_chd_rows'
                               if constrained else None),
                'temporal_protocol': (
                    f'{benchmark}_daily_targets_train_1_20_val_21_25_test_26_30'
                ),
                'metric_policy': (
                    'normalized_mse_for_optimization; physical_units_for_reporting'
                ),
            }, checkpoint_path)

    # Load best model for testing
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")
    
    # Test set evaluation
    model.eval()
    with torch.no_grad():
        test_loss, test_mae_sum, test_rmse_sum = 0, 0, 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)

            test_loss += criterion(pred, yb).item()
            test_mae_sum += mae(pred, yb)
            test_rmse_sum += rmse(pred, yb)
    
    test_mse_normalized = test_loss / len(test_loader)
    test_mae_normalized = test_mae_sum / len(test_loader)
    test_rmse_normalized = test_rmse_sum / len(test_loader)
    test_results = {
        'test_mse_m2': float(test_mse_normalized * train_ds.std ** 2),
        'test_mae_m': float(test_mae_normalized * train_ds.std),
        'test_rmse_m': float(test_rmse_normalized * train_ds.std),
        'optimization_test_mse_normalized': float(test_mse_normalized),
        'normalization': {
            'method': 'global_training_z_score',
            'mean_m': float(train_ds.mean),
            'std_m': float(train_ds.std),
        },
        'units': {
            'test_mse_m2': 'm2',
            'test_mae_m': 'm',
            'test_rmse_m': 'm',
            'optimization_test_mse_normalized': 'dimensionless',
        },
        'seed': seed,
    }
    
    # ---- does the boundary condition actually hold? ----
    bs, bn = boundary_error(model, test_ds, device,
                            h_bc=h_bc_for(benchmark, y_grid))
    test_results['bc_err_south_max_m'] = bs
    test_results['bc_err_north_max_m'] = bn
    test_results['constrained'] = bool(constrained)
    test_results['benchmark'] = benchmark
    test_results['arch'] = arch
    test_results['params'] = sum(p.numel() for p in model.parameters())
    test_results['constraint'] = ('hard_dirichlet_chd_rows'
                                  if constrained else None)

    print("\n" + "="*50)
    print("BOUNDARY-CONDITION ERROR  max |h_pred - h_BC|  (m, test set)")
    print(f"  south row : {bs:.3e}")
    print(f"  north row : {bn:.3e}")
    if constrained and max(bs, bn) > 1e-3:
        print("  *** constraint NOT holding -- check normalisation ***")
    elif constrained:
        print("  exact by construction, as intended")
    print("="*50)

    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"Test MSE:  {test_results['test_mse_m2']:.6f} m2")
    print(f"Test MAE:  {test_results['test_mae_m']:.4f} m")
    print(f"Test RMSE: {test_results['test_rmse_m']:.4f} m")
    print("="*50)
    


    # =========================
    # SAVE TEST CNN PREDICTIONS AS MODFLOW-STYLE TXT FILES
    # =========================
    print("\nSaving individual TEST CNN predictions as text files...")

    save_dir = os.path.join(out_dir, "test_predictions")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in range(len(test_ds)):
            # Input is H(t-1)
            x, _ = test_ds[i]
            x = x.unsqueeze(0).to(device)

            # Predict normalized H(t)
            pred_norm = model(x).squeeze().cpu().numpy()

            # Denormalize to physical head values
            pred = pred_norm * test_ds.std + test_ds.mean

            # Load coordinates from corresponding MODFLOW target file (H(t))
            target_file = test_ds.txt_files[i + 1]
            basename = os.path.basename(target_file)
            out_path = os.path.join(save_dir, f"cnn_pred_{basename}")
            print(f"  one-step prediction {i}: {basename} -> {out_path}")
            save_prediction_txt(pred, target_file, out_path)

    print("Saved all TEST CNN predictions.")

    # Operational protocol: observe day 25 once, then recursively feed the
    # predictions to forecast days 26-30. Setting diagnostics mode to rollout
    # labels this protocol; it does not create the recurrence.
    print("\nSaving operational rollout predictions (day 25 -> days 26-30)...")
    rollout_dir = os.path.join(out_dir, "rollout_predictions")
    test_results['operational_rollout'] = save_operational_rollout(
        model=model,
        txt_files=txt_files,
        mean=train_ds.mean,
        std=train_ds.std,
        device=device,
        out_dir=rollout_dir,
    )
    print(
        "Operational rollout mean RMSE: "
        f"{test_results['operational_rollout']['mean_rmse_m']:.6f} m"
    )




    # Multi-step prediction on test set
    print("\nEvaluating multi-step prediction...")
    multi_step_errors = evaluate_multistep(
        model, test_ds, device, std_m=test_ds.std, steps=[1, 5]
    )
    test_results['multistep'] = multi_step_errors
    
    # Save results
    with open(os.path.join(out_dir, f"results_seed_{seed}.json"), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Preserve the original training history during export-only evaluation.
    if epochs > 0:
        with open(os.path.join(out_dir, f"history_seed_{seed}.json"), 'w') as f:
            json.dump(history, f, indent=2)
    
    print("Training finished.")
    return model, train_ds, val_ds, test_ds, test_results


def evaluate_multistep(model, dataset, device, std_m, steps=[1, 5, 10]):
    """
    Evaluate multi-step prediction error accumulation
    """
    model.eval()
    results = {}
    
    for n_steps in steps:
        errors = []
        # Use first few samples that allow n_steps prediction
        max_samples = min(10, len(dataset) - n_steps + 1)
        
        for i in range(max_samples):
            x, _ = dataset[i]
            x = x.unsqueeze(0).to(device)
            
            # Get ground truth
            true_states = []
            for j in range(i, min(i + n_steps, len(dataset))):
                _, y = dataset[j]
                true_states.append(y)
            
            # Predict
            predictions = multi_step_predict(model, x, len(true_states), device)
            
            # Calculate error
            for pred, true in zip(predictions, true_states):
                true = true.unsqueeze(0)
                # Move pred back to device to match true
                pred = pred.to(device)
                error_m = rmse(pred, true.to(device)) * std_m
                errors.append(error_m)
        
        results[f'{n_steps}_step'] = {
            'mean_rmse_m': float(np.mean(errors)),
            'std_rmse_m': float(np.std(errors)),
            'units': 'm',
        }
        print(
            f"{n_steps}-step RMSE: {np.mean(errors):.4f} +/- "
            f"{np.std(errors):.4f} m"
        )
    
    return results


# =========================
# Multiple Runs for Statistical Significance
# =========================
def run_multiple_seeds(n_runs=5, epochs=100, lr=1e-3, batch_size=4,
                       data_path=None, constrained=True,
                       benchmark="b1", out_dir=None, arch="plain",
                       channels=24, width=32, depth=3):
    """
    Run training with multiple seeds and aggregate results
    """
    if out_dir is None:
        out_dir = default_output_dir(
            benchmark, constrained, arch, channels, width, depth
        )
    os.makedirs(out_dir, exist_ok=True)
    all_results = []
    
    for run in range(n_runs):
        seed = 42 + run
        print(f"\n{'='*60}")
        print(f"RUN {run+1}/{n_runs} with seed={seed}")
        print(f"{'='*60}")
        
        model, train_ds, val_ds, test_ds, results = train(
            seed=seed, epochs=epochs, lr=lr, batch_size=batch_size,
            data_path=data_path, constrained=constrained, benchmark=benchmark,
            out_dir=out_dir, arch=arch, channels=channels,
            width=width, depth=depth
        )
        all_results.append(results)
    
    # Aggregate results
    metrics = ['test_mse_m2', 'test_mae_m', 'test_rmse_m']
    metric_units = {
        'test_mse_m2': 'm2',
        'test_mae_m': 'm',
        'test_rmse_m': 'm',
    }
    aggregated = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'units': metric_units[metric],
        }
    
    print("\n" + "="*60)
    print(f"AGGREGATED RESULTS ({n_runs} runs)")
    print("="*60)
    for metric in metrics:
        stats = aggregated[metric]
        print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"{stats['units']} (min={stats['min']:.4f}, "
              f"max={stats['max']:.4f})")
    print("="*60)
    
    # Save aggregated results
    with open(os.path.join(out_dir, "aggregated_results.json"), 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    return all_results, aggregated


# =========================
# Main
# =========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN surrogate for MODFLOW')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multiple'],
                       help='Run single training or multiple runs for statistical testing')
    parser.add_argument('--n_runs', type=int, default=5,
                       help='Number of runs with different seeds (for multiple mode)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path pattern for data files. Default: the '
                            "benchmark's own data (b1 -> cnn/data/, "
                            'b2 -> b2_corrected/).')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (for single mode)')
    parser.add_argument('--export-only', action='store_true',
                       help='load the existing checkpoint, skip training, and '
                            'regenerate one-step and operational rollout files')
    parser.add_argument('--constrained', dest='constrained', action='store_true',
                       default=True,
                       help='impose the Dirichlet BC by construction (default)')
    parser.add_argument('--unconstrained', dest='constrained', action='store_false',
                       help='the published baseline, BC learned from data')
    parser.add_argument('--arch', type=str, default='plain',
                       choices=['plain', 'dilated'],
                       help="plain = the published 3-layer CNN (7-cell "
                            "receptive field); dilated = 8 dilated layers "
                            "(511 cells, covers the domain)")
    parser.add_argument('--width', type=int, default=32,
                       help='hidden channels for --arch plain. Published = 32. '
                            'Widening changes capacity but NOT receptive field.')
    parser.add_argument('--depth', type=int, default=3,
                       help='conv layers for --arch plain. Published = 3 '
                            '(receptive field 1 + 2*depth cells).')
    parser.add_argument('--channels', type=int, default=24,
                       help='hidden channels for --arch dilated. 24 gives '
                            '36,721 parameters; 12 gives 9,289, matched to '
                            "the plain CNN's 9,857 and the PINN's 10,451, so "
                            'receptive field is isolated from capacity.')
    parser.add_argument('--benchmark', type=str, default='b1',
                       choices=sorted(BENCHMARKS),
                       help='b1 = homogeneous, 90 m both boundaries; '
                            'b2 = low-K lens with a 90 -> 100 m regional gradient')
    parser.add_argument('--out_dir', type=str, default=None,
                       help='output directory; the default records benchmark, '
                            'architecture, and CHD constraint status')
    parser.add_argument('--well_type', type=str, default='unconfined_single_well',
                       choices=['unconfined_single_well', 'unconfined_multiple_wells', 'unconfined_single_well'],
                       help='Well type for MODFLOW comparison')
  
    
    args = parser.parse_args()
    if args.export_only and args.mode != 'single':
        parser.error('--export-only is available only with --mode single')
    
    if args.mode == 'single':
        print("Running single training...")
        model, train_ds, val_ds, test_ds, results = train(
            seed=args.seed, epochs=0 if args.export_only else args.epochs,
            lr=args.lr,
            batch_size=args.batch_size, data_path=args.data_path,
            constrained=args.constrained, benchmark=args.benchmark,
            out_dir=args.out_dir, arch=args.arch, channels=args.channels,
            width=args.width, depth=args.depth
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
                
        # Simple visualization of first test sample
        _out = args.out_dir or default_output_dir(
            args.benchmark, args.constrained, args.arch, args.channels,
            args.width, args.depth
        )
        # utils.plot_test_cnn_vs_modflow defaults to outputs/test_predictions;
        # point it at THIS arm's directory or it reads the other arm's files,
        # or none at all.
        # A diagnostic figure must never discard a finished training run.
        # utils.plot_test_cnn_vs_modflow calls ax.clabel, which raises on some
        # matplotlib versions; by this point the model, the metrics and the
        # saved predictions are all on disk already.
        try:
            fig = plot_test_cnn_vs_modflow(
                test_ds, cnn_pred_dir=os.path.join(_out, "test_predictions"))
            fig.savefig(
                os.path.join(_out, "contours_test_cnn_vs_modflow.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)
        except Exception as e:
            print(f"[figure] contour plot skipped: {type(e).__name__}: {e}")

        
        # If we have at least 4 test samples, save predictions and create comparison plots
        if len(test_ds) >= 4:
            # Show available test timesteps
            print(f"\n{'='*50}")
            print(f"Available test set files (total: {len(test_ds.txt_files)}):")
            import re
            test_timesteps = []
            for i, fname in enumerate(test_ds.txt_files):
                match = re.search(r't([0-9.]+)\.txt', fname)
                if match:
                    t = float(match.group(1))
                    test_timesteps.append((i, t, fname))
                    if i < 10:  # Show first 10
                        print(f"  Index {i}: t={t} ({os.path.basename(fname)})")
            
            if len(test_timesteps) > 10:
                print(f"  ... ({len(test_timesteps) - 10} more files)")
            
            # Use indices 0,1,2,3 from test set (first 4 available)
            test_indices = [0, 1, 2, 3]
            selected_timesteps = [test_timesteps[i][1] for i in test_indices if i < len(test_timesteps)]
            print(f"\nVisualizing test indices {test_indices} -> timesteps: {selected_timesteps}")
            print(f"{'='*50}\n")
            
           
    else:
        print(f"Running {args.n_runs} training runs for statistical significance...")
        all_results, aggregated = run_multiple_seeds(
            n_runs=args.n_runs, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, data_path=args.data_path,
            constrained=args.constrained, benchmark=args.benchmark,
            out_dir=args.out_dir, arch=args.arch, channels=args.channels,
            width=args.width, depth=args.depth
        )
       



    
