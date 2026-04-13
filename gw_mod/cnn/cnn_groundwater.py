#!/usr/bin/env python
"""
CNN Surrogate Model for Groundwater Head Prediction
====================================================
Trains and evaluates a Convolutional Neural Network (CNN) surrogate that learns
the temporal evolution of hydraulic head fields from MODFLOW simulations.

Physics-based evaluation includes:
  - Autoregressive rollout metrics (MAE, RMSE, RRMSE, R²)
  - Nonlinear Boussinesq PDE residuals
  - Finite-volume mass balance error
  - Dirichlet boundary condition consistency

Usage
-----
Single training run:
    python cnn_groundwater.py --mode single --data_path ./data/t*.txt

Multiple runs (statistical significance):
    python cnn_groundwater.py --mode multiple --n_runs 5 --data_path ./data/t*.txt

Data format
-----------
Each input file is a whitespace-delimited text file with three columns:
    x  y  h
where (x, y) are spatial coordinates and h is the hydraulic head value.
Files should follow the naming convention t{timestep}.txt (e.g. t1.txt, t2.txt).
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import glob
import json
import os
import random
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D projections)
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Configuration
# =============================================================================
# Default data path — override via --data_path argument or by editing this value.
DEFAULT_DATA_PATH = "./data/t*.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_time(fname: str) -> float:
    """Extract the numeric timestep from a filename such as *t26.txt*."""
    base = os.path.basename(fname)
    m = re.search(r"t([0-9]+(?:\.[0-9]+)?)", base)
    if not m:
        raise ValueError(f"Could not extract time from filename: {fname}")
    return float(m.group(1))


def load_txt_to_grid(path: str):
    """
    Load a MODFLOW output text file (x y h) into a structured 2-D grid.

    Parameters
    ----------
    path : str
        Path to the whitespace-delimited text file.

    Returns
    -------
    H : ndarray, shape (ny, nx)
        Hydraulic head field on a regular grid.
    x_unique : ndarray
        Unique x-coordinates.
    y_unique : ndarray
        Unique y-coordinates.
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

    # Fill any NaN holes with nearest-neighbour interpolation
    if np.any(np.isnan(H)):
        mask = ~np.isnan(H)
        yy, xx = np.mgrid[0:ny, 0:nx]
        H = griddata((yy[mask], xx[mask]), H[mask], (yy, xx), method="nearest")

    return H, x_unique, y_unique


def denorm(x_norm, mean, std):
    """De-normalise a tensor or array: x_physical = x_norm * std + mean."""
    return x_norm * std + mean


# =============================================================================
# Dataset
# =============================================================================

class GroundwaterDataset(Dataset):
    """
    One-step-ahead dataset: each sample is the pair (H_t, H_{t+1}).

    Parameters
    ----------
    txt_files : list[str]
        Ordered list of MODFLOW output files representing consecutive stress periods.
    start_idx, end_idx : int
        Slice indices into *txt_files* that define this split.
    mean, std : float or None
        Normalisation statistics. If ``None`` they are computed from the data.
    """

    def __init__(self, txt_files, start_idx=0, end_idx=None, mean=None, std=None):
        self.txt_files = txt_files[start_idx:end_idx]

        self.grids = np.array([load_txt_to_grid(f)[0] for f in self.txt_files])

        if mean is not None and std is not None:
            self.mean, self.std = mean, std
        else:
            self.mean = float(np.mean(self.grids))
            self.std  = float(np.std(self.grids))

        self.grids = (self.grids - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.grids) - 1

    def __getitem__(self, idx):
        x = torch.tensor(self.grids[idx],     dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.grids[idx + 1], dtype=torch.float32).unsqueeze(0)
        return x, y


# =============================================================================
# Model
# =============================================================================

class HeadCNN(nn.Module):
    """
    Lightweight fully-convolutional network for one-step hydraulic head prediction.

    Architecture: three 3×3 Conv2d layers with GELU activations and same-padding,
    mapping a single head field H_t → H_{t+1}.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Scalar metrics
# =============================================================================

def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Mean absolute error (normalised space)."""
    return torch.mean(torch.abs(pred - true)).item()


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Root mean squared error (normalised space)."""
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()


# =============================================================================
# Multi-step prediction helper
# =============================================================================

def multi_step_predict(model, initial_state, n_steps, device):
    """
    Autoregressively predict *n_steps* ahead from *initial_state*.

    Returns
    -------
    list[torch.Tensor]
        Predicted frames, each with shape matching *initial_state*.
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


def evaluate_multistep(model, dataset, device, steps=(1, 5, 10)):
    """
    Evaluate RMSE for multi-step rollouts at each requested horizon.

    Horizons that exceed the number of available test samples are skipped
    automatically (avoids NaN warnings on small test sets).

    Returns
    -------
    dict
        ``{'1_step': {'mean_rmse': ..., 'std_rmse': ...}, ...}``
    """
    model.eval()
    results = {}
    max_possible = len(dataset) - 1  # largest valid horizon for this dataset

    for n_steps in steps:
        if n_steps > max_possible:
            print(f"{n_steps}-step rollout skipped "
                  f"(only {max_possible} test samples available)")
            continue

        errors = []
        max_samples = min(10, len(dataset) - n_steps + 1)

        for i in range(max_samples):
            x, _ = dataset[i]
            x = x.unsqueeze(0).to(device)

            true_states = [dataset[j][1] for j in range(i, min(i + n_steps, len(dataset)))]
            predictions = multi_step_predict(model, x, len(true_states), device)

            for pred, true in zip(predictions, true_states):
                errors.append(rmse(pred.to(device), true.unsqueeze(0).to(device)))

        results[f"{n_steps}_step"] = {
            "mean_rmse": float(np.mean(errors)),
            "std_rmse":  float(np.std(errors)),
        }
        print(f"{n_steps}-step RMSE: {np.mean(errors):.4f} ± {np.std(errors):.4f}")

    return results


# =============================================================================
# Rollout metrics (canonical evaluation)
# =============================================================================

@torch.no_grad()
def rollout_metrics_and_report(model, test_ds, device, seq_len, horizons, mean, std):
    """
    Compute MAE, RMSE, RRMSE, and R² for autoregressive rollout horizons.

    Reports results in PINN-comparable format (one line per horizon) followed by
    an aggregated summary.

    Parameters
    ----------
    model : HeadCNN
        Trained model.
    test_ds : GroundwaterDataset
        Test split dataset.
    device : torch.device
    seq_len : int
        Input sequence length (1 for this single-frame CNN).
    horizons : list[int]
        Rollout horizons to evaluate, e.g. [1, 2, 3, 4, 5].
    mean, std : torch.Tensor
        Training-set normalisation statistics (scalar tensors on *device*).
    """
    horizons = sorted(set(horizons))
    horizon_data = {h: {"preds": [], "targets": []} for h in horizons}
    N_test_grids = len(test_ds.grids)

    for h in horizons:
        max_start = N_test_grids - seq_len - h
        if max_start < 0:
            print(f"  Warning: insufficient test data for horizon {h} — skipping.")
            continue

        for start_idx in range(max_start + 1):
            xb, _ = test_ds[start_idx]
            xb_t = xb if isinstance(xb, torch.Tensor) else torch.from_numpy(xb)
            if xb_t.ndim == 3:
                xb_t = xb_t.unsqueeze(0)
            elif xb_t.ndim == 4 and xb_t.shape[0] != 1:
                xb_t = xb_t[:1]
            xb_t = xb_t.to(device)
            roll_in = xb_t.clone()

            for step in range(1, h + 1):
                pred_norm = model(roll_in)
                if step == h:
                    true_idx = start_idx + seq_len + (h - 1)
                    y_norm = (torch.tensor(test_ds.grids[true_idx], dtype=torch.float32)
                              .unsqueeze(0).unsqueeze(0).to(device))
                    pred_phys = denorm(pred_norm, mean, std).cpu().numpy()
                    y_phys    = denorm(y_norm,    mean, std).cpu().numpy()
                    horizon_data[h]["preds"].append(pred_phys)
                    horizon_data[h]["targets"].append(y_phys)
                if step < h:
                    roll_in = pred_norm

    print("\n=== Testing on Rollout Horizons (Extrapolation) ===")
    summary = {"mae": [], "rmse": [], "rrmse": [], "r2": []}

    for h in horizons:
        preds   = horizon_data[h]["preds"]
        targets = horizon_data[h]["targets"]
        if not preds:
            print(f"  h={h}: No data processed")
            continue

        preds   = np.concatenate(preds,   axis=0)
        targets = np.concatenate(targets, axis=0)

        h_mae   = np.mean(np.abs(preds - targets))
        h_mse   = np.mean((preds - targets) ** 2)
        h_rmse  = np.sqrt(h_mse)
        t_mean  = np.mean(targets)
        h_rrmse = (h_rmse / t_mean) * 100 if t_mean != 0 else 0.0

        ss_res = np.sum((targets - preds)          ** 2)
        ss_tot = np.sum((targets - t_mean)         ** 2)
        h_r2   = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        summary["mae"].append(h_mae)
        summary["rmse"].append(h_rmse)
        summary["rrmse"].append(h_rrmse)
        summary["r2"].append(h_r2)

        print(f"  h={h:<2}: MAE={h_mae:.3f} m, RMSE={h_rmse:.3f} m, "
              f"RRMSE={h_rrmse:.3f}%, R²={h_r2:.6f}")

    if summary["mae"]:
        print("\n--- FINAL TEST RESULTS (averaged over horizons) ---")
        for key, label, unit in [
            ("mae",   "Average MAE",   "m"),
            ("rmse",  "Average RMSE",  "m"),
            ("rrmse", "Average RRMSE", "%"),
            ("r2",    "Average R²",    ""),
        ]:
            vals = summary[key]
            print(f"{label:20s}: {np.mean(vals):.4f} {unit}  "
                  f"(std = {np.std(vals):.4f} {unit})")
        print("-" * 44)


# =============================================================================
# Training
# =============================================================================

def train(seed=42, epochs=100, lr=1e-3, batch_size=4, data_path=DEFAULT_DATA_PATH):
    """
    Train the CNN surrogate model.

    Parameters
    ----------
    seed : int
        Random seed.
    epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate for Adam.
    batch_size : int
        Mini-batch size.
    data_path : str
        Glob pattern for MODFLOW head files (e.g. ``./data/t*.txt``).

    Returns
    -------
    model, train_ds, val_ds, test_ds, test_results
    """
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    txt_files = sorted(glob.glob(data_path), key=extract_time)
    if not txt_files:
        raise ValueError(f"No files found matching: {data_path}")

    n_total = len(txt_files)
    print(f"Found {n_total} time steps")

    n_train = int(0.6 * n_total)
    n_val   = int(0.2 * n_total)
    print(f"Split: Train={n_train}, Val={n_val}, Test={n_total - n_train - n_val}")

    train_ds = GroundwaterDataset(txt_files, 0, n_train)
    val_ds   = GroundwaterDataset(txt_files, n_train, n_train + n_val,
                                  mean=train_ds.mean, std=train_ds.std)
    test_ds  = GroundwaterDataset(txt_files, n_train + n_val, n_total,
                                  mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

    model     = HeadCNN().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    os.makedirs("outputs", exist_ok=True)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = val_mae_sum = val_rmse_sum = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                val_loss     += criterion(pred, yb).item()
                val_mae_sum  += mae(pred, yb)
                val_rmse_sum += rmse(pred, yb)

        avg_train = train_loss   / len(train_loader)
        avg_val   = val_loss     / len(val_loader)
        avg_mae   = val_mae_sum  / len(val_loader)
        avg_rmse  = val_rmse_sum / len(val_loader)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_mae"].append(avg_mae)
        history["val_rmse"].append(avg_rmse)

        scheduler.step(avg_val)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train MSE: {avg_train:.4e} | "
                  f"Val MSE: {avg_val:.4e} | MAE: {avg_mae:.3f} | RMSE: {avg_rmse:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "mean": train_ds.mean,
                "std":  train_ds.std,
            }, "outputs/cnn_best_model.pth")

    # Load best checkpoint
    ckpt = torch.load("outputs/cnn_best_model.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nLoaded best model from epoch {ckpt['epoch']}")

    # Test set evaluation
    model.eval()
    with torch.no_grad():
        test_loss = test_mae_sum = test_rmse_sum = 0.0
        for xb, yb in test_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            test_loss     += criterion(pred, yb).item()
            test_mae_sum  += mae(pred, yb)
            test_rmse_sum += rmse(pred, yb)

    test_results = {
        "test_mse":  test_loss     / len(test_loader),
        "test_mae":  test_mae_sum  / len(test_loader),
        "test_rmse": test_rmse_sum / len(test_loader),
        "seed": seed,
    }
    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Test MSE:  {test_results['test_mse']:.4e}")
    print(f"Test MAE:  {test_results['test_mae']:.4f} m")
    print(f"Test RMSE: {test_results['test_rmse']:.4f} m")
    print("=" * 50)

    # Save test predictions as MODFLOW-style .txt files
    save_dir = "outputs/test_predictions"
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, _ = test_ds[i]
            x = x.unsqueeze(0).to(dev)
            pred_norm = model(x).squeeze().cpu().numpy()
            pred = pred_norm * test_ds.std + test_ds.mean
            target_file = test_ds.txt_files[i + 1]
            data = np.loadtxt(target_file)
            coords = data[:, :2]
            assert coords.shape[0] == pred.size, "Grid size mismatch"
            out = np.column_stack([coords[:, 0], coords[:, 1], pred.reshape(-1)])
            basename = os.path.basename(target_file)
            np.savetxt(
                os.path.join(save_dir, f"cnn_pred_{basename}"),
                out, fmt="%.6f", header="x y h", comments=""
            )
    print("Saved all test CNN predictions.")

    # Multi-step evaluation
    print("\nEvaluating multi-step prediction…")
    multi_step_errors = evaluate_multistep(model, test_ds, dev, steps=[1, 5, 10])
    test_results["multistep"] = multi_step_errors

    with open(f"outputs/results_seed_{seed}.json", "w") as f:
        json.dump(test_results, f, indent=2)
    with open(f"outputs/history_seed_{seed}.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Training finished.")
    return model, train_ds, val_ds, test_ds, test_results


# =============================================================================
# Multiple runs for statistical robustness
# =============================================================================

def run_multiple_seeds(n_runs=5, epochs=100, lr=1e-3, batch_size=4,
                       data_path=DEFAULT_DATA_PATH):
    """
    Train *n_runs* independent models with consecutive seeds and aggregate results.

    Returns
    -------
    all_results : list[dict]
    aggregated  : dict
    """
    all_results = []
    for run in range(n_runs):
        seed = 42 + run
        print(f"\n{'=' * 60}\nRUN {run + 1}/{n_runs}  seed={seed}\n{'=' * 60}")
        _, _, _, _, results = train(
            seed=seed, epochs=epochs, lr=lr,
            batch_size=batch_size, data_path=data_path
        )
        all_results.append(results)

    metrics = ["test_mse", "test_mae", "test_rmse"]
    aggregated = {
        m: {
            "mean": float(np.mean([r[m] for r in all_results])),
            "std":  float(np.std( [r[m] for r in all_results])),
            "min":  float(np.min( [r[m] for r in all_results])),
            "max":  float(np.max( [r[m] for r in all_results])),
        }
        for m in metrics
    }

    print(f"\n{'=' * 60}\nAGGREGATED RESULTS ({n_runs} runs)\n{'=' * 60}")
    for m in metrics:
        s = aggregated[m]
        print(f"{m.upper():12s}: {s['mean']:.4f} ± {s['std']:.4f} "
              f"(min={s['min']:.4f}, max={s['max']:.4f})")
    print("=" * 60)

    with open("outputs/aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    return all_results, aggregated


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN surrogate for MODFLOW groundwater head prediction"
    )
    parser.add_argument("--mode", choices=["single", "multiple"], default="single",
                        help="Single training run or multiple runs for statistics")
    parser.add_argument("--n_runs",     type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--data_path",  type=str,   default=DEFAULT_DATA_PATH,
                        help="Glob pattern for MODFLOW .txt files")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    if args.mode == "single":
        model, train_ds, val_ds, test_ds, results = train(
            seed=args.seed, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, data_path=args.data_path,
        )
        # ── Rollout evaluation ────────────────────────────────────────────────
        mean_t = torch.tensor(float(train_ds.mean), device=device, dtype=torch.float32)
        std_t  = torch.tensor(float(train_ds.std),  device=device, dtype=torch.float32)
        rollout_metrics_and_report(
            model=model, test_ds=test_ds, device=device,
            seq_len=1, horizons=[1, 2, 3, 4, 5],
            mean=mean_t, std=std_t,
        )
    else:
        run_multiple_seeds(
            n_runs=args.n_runs, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, data_path=args.data_path,
        )
