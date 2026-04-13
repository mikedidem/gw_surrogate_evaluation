#!/usr/bin/env python
"""
ConvLSTM Surrogate Model for Groundwater Head Prediction
=========================================================
Trains and evaluates a ConvLSTM surrogate that learns temporal sequences of
hydraulic head fields from MODFLOW simulations.

The model accepts a sliding window of *seq_len* consecutive stress periods and
predicts the next head field. Evaluation includes autoregressive rollout metrics
(MAE, RMSE, RRMSE, R²) matching the PINN reporting format.

Usage
-----
Single training run:
    python convlstm_groundwater.py --data_path ./data/t*.txt

Custom hyperparameters:
    python convlstm_groundwater.py --epochs 300 --seq_len 3 --lr 5e-4

Data format
-----------
Each file is a whitespace-delimited text file with three columns (x, y, h).
Files should follow the naming convention t{timestep}.txt (e.g. t1.txt, t2.txt).
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import glob
import os
import re
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_DATA_PATH = "./data/t*.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Data utilities
# =============================================================================

def extract_time(fname: str) -> float:
    """Extract numeric timestep from a filename such as *t26.txt*."""
    base = os.path.basename(fname)
    m = re.search(r"t([0-9]+(?:\.[0-9]+)?)", base)
    if not m:
        raise ValueError(f"Could not extract time from filename: {fname}")
    return float(m.group(1))


def load_txt_to_grid(path: str) -> np.ndarray:
    """
    Load a MODFLOW text file (x y h) into a structured 2-D head grid.

    Parameters
    ----------
    path : str
        Path to the whitespace-delimited text file.

    Returns
    -------
    H : ndarray, shape (ny, nx)
        Hydraulic head field on a regular grid.
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

    if np.any(np.isnan(H)):
        mask = ~np.isnan(H)
        yy, xx = np.mgrid[0:ny, 0:nx]
        H = griddata((yy[mask], xx[mask]), H[mask], (yy, xx), method="nearest")

    return H


def denorm(x_norm, mean, std):
    """De-normalise: x_physical = x_norm * std + mean."""
    return x_norm * std + mean


# =============================================================================
# Dataset
# =============================================================================

class GroundwaterDataset(Dataset):
    """
    Sliding-window dataset: each sample is (H_{t}, …, H_{t+seq_len-1}) → H_{t+seq_len}.

    Parameters
    ----------
    txt_files : list[str]
        Ordered list of MODFLOW output files for consecutive stress periods.
    start_idx, end_idx : int
        Slice indices into *txt_files* that define this split.
    mean, std : float or None
        Normalisation statistics. Computed from data if ``None``.
    seq_len : int
        Number of input frames per sample.
    """

    def __init__(self, txt_files, start_idx=0, end_idx=None,
                 mean=None, std=None, seq_len=3):
        self.seq_len   = seq_len
        self.txt_files = txt_files[start_idx:end_idx]

        self.grids = np.array([load_txt_to_grid(f) for f in self.txt_files])

        if mean is not None and std is not None:
            self.mean, self.std = float(mean), float(std)
        else:
            self.mean = float(np.mean(self.grids))
            self.std  = float(np.std(self.grids))

        self.grids = (self.grids - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.grids) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.grids[idx : idx + self.seq_len]          # (seq_len, H, W)
        y     = self.grids[idx + self.seq_len]                 # (H, W)
        x = torch.tensor(x_seq, dtype=torch.float32)          # (seq_len, H, W)
        y = torch.tensor(y,     dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        return x, y


def create_dataloaders(data_path, batch_size=4, seq_len=3,
                       train_ratio=0.6, val_ratio=0.2):
    """
    Build train / val / test DataLoaders with a temporal split.

    Returns
    -------
    train_loader, val_loader, test_loader, train_ds, test_ds
    """
    txt_files = sorted(glob.glob(data_path), key=extract_time)
    if not txt_files:
        raise ValueError(f"No files found matching: {data_path}")

    n_total = len(txt_files)
    n_train = int(train_ratio * n_total)
    n_val   = int(val_ratio   * n_total)
    print(f"Total timesteps: {n_total}  |  seq_len: {seq_len}")
    print(f"Split: Train={n_train}, Val={n_val}, Test={n_total - n_train - n_val}")

    train_ds = GroundwaterDataset(txt_files, 0, n_train, seq_len=seq_len)
    val_ds   = GroundwaterDataset(txt_files, n_train, n_train + n_val,
                                  mean=train_ds.mean, std=train_ds.std, seq_len=seq_len)
    test_ds  = GroundwaterDataset(txt_files, n_train + n_val, n_total,
                                  mean=train_ds.mean, std=train_ds.std, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

    print(f"Samples: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print(f"Normalisation: mean={train_ds.mean:.4f}, std={train_ds.std:.4f}")
    return train_loader, val_loader, test_loader, train_ds, test_ds


# =============================================================================
# Model
# =============================================================================

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.

    Parameters
    ----------
    input_channels, hidden_channels : int
    kernel_size, padding : int
    """

    def __init__(self, input_channels, hidden_channels, kernel_size, padding):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h_prev, c_prev):
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """
    Sequence encoder: processes *(B, T, C, H, W)* and returns the last hidden state.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, padding)

    def forward(self, x):
        if x.dim() == 4:                       # (B, T, H, W) → (B, T, 1, H, W)
            x = x.unsqueeze(2)
        B, T, C, H, W = x.size()
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return h                               # (B, hidden_channels, H, W)


class ConvLSTMModel(nn.Module):
    """
    Full ConvLSTM surrogate: encoder + spatial refinement head → *(B, 1, H, W)*.

    Architecture:
        ConvLSTM encoder → Conv7×7 + PReLU + Dropout(0.2) → Conv1×1
    """

    def __init__(self, input_channels=1, hidden_channels=32, seq_len=3):
        super().__init__()
        self.seq_len  = seq_len
        self.convlstm = ConvLSTM(input_channels, hidden_channels, kernel_size=5)
        self.conv1    = nn.Conv2d(hidden_channels, 64, kernel_size=7, padding=3)
        self.prelu1   = nn.PReLU()
        self.dropout  = nn.Dropout(0.2)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.convlstm(x)      # (B, hidden_channels, H, W)
        x = self.prelu1(self.conv1(x))
        x = self.dropout(x)
        return self.conv_out(x)   # (B, 1, H, W)


# =============================================================================
# Rollout evaluation (canonical)
# =============================================================================

@torch.no_grad()
def rollout_metrics_and_report(model, test_ds, device, seq_len, horizons, mean, std):
    """
    Compute MAE, RMSE, RRMSE, and R² for autoregressive rollout horizons.

    The model's sequence window is updated autoregressively: each predicted frame
    is appended to the window and the oldest frame is dropped.

    Parameters
    ----------
    model : ConvLSTMModel
    test_ds : GroundwaterDataset
    device : torch.device
    seq_len : int
        Input sequence length passed explicitly (may differ from test_ds.seq_len).
    horizons : list[int]
        Rollout horizons to evaluate, e.g. [1, 2, 3, 4, 5].
    mean, std : torch.Tensor
        Training-set normalisation statistics (scalar tensors on *device*).
    """
    horizons = sorted(set(horizons))
    horizon_data = {h: {"preds": [], "targets": []} for h in horizons}
    N = len(test_ds.grids)

    for h in horizons:
        max_start = N - seq_len - h
        if max_start < 0:
            print(f"  Warning: insufficient test data for horizon {h} — skipping.")
            continue

        for start in range(max_start + 1):
            xb, _ = test_ds[start]
            xb_t  = xb if isinstance(xb, torch.Tensor) else torch.from_numpy(xb)
            if xb_t.ndim == 3:
                xb_t = xb_t.unsqueeze(0)                     # (1, seq_len, H, W)
            xb_t     = xb_t.to(device)
            roll_in  = xb_t.clone()

            for step in range(1, h + 1):
                pred_norm = model(roll_in)
                if step == h:
                    true_idx   = start + seq_len + (h - 1)
                    y_norm     = (torch.tensor(test_ds.grids[true_idx], dtype=torch.float32)
                                  .unsqueeze(0).unsqueeze(0).to(device))
                    pred_phys  = denorm(pred_norm, mean, std).cpu().numpy()
                    y_phys     = denorm(y_norm,    mean, std).cpu().numpy()
                    horizon_data[h]["preds"].append(pred_phys)
                    horizon_data[h]["targets"].append(y_phys)
                if step < h:
                    # Slide window: drop oldest frame, append prediction
                    pred_frame = pred_norm[:, 0].unsqueeze(1)       # (1,1,H,W)
                    roll_in    = torch.cat([roll_in[:, 1:], pred_frame], dim=1)

    print("\n=== Rollout Evaluation (Extrapolation) ===")
    summary = {"mae": [], "rmse": [], "rrmse": [], "r2": []}

    for h in horizons:
        preds   = horizon_data[h]["preds"]
        targets = horizon_data[h]["targets"]
        if not preds:
            print(f"  h={h}: No data")
            continue

        preds   = np.concatenate(preds,   axis=0)
        targets = np.concatenate(targets, axis=0)

        h_mae   = np.mean(np.abs(preds - targets))
        h_rmse  = np.sqrt(np.mean((preds - targets) ** 2))
        t_mean  = np.mean(targets)
        h_rrmse = (h_rmse / t_mean) * 100 if t_mean != 0 else 0.0

        ss_res = np.sum((targets - preds)  ** 2)
        ss_tot = np.sum((targets - t_mean) ** 2)
        h_r2   = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        summary["mae"].append(h_mae)
        summary["rmse"].append(h_rmse)
        summary["rrmse"].append(h_rrmse)
        summary["r2"].append(h_r2)

        print(f"  h={h:<2}: MAE={h_mae:.3f} m, RMSE={h_rmse:.3f} m, "
              f"RRMSE={h_rrmse:.3f}%, R²={h_r2:.6f}")

    if summary["mae"]:
        print("\n--- FINAL RESULTS (averaged over horizons) ---")
        for key, label, unit in [
            ("mae",   "Average MAE",   "m"),
            ("rmse",  "Average RMSE",  "m"),
            ("rrmse", "Average RRMSE", "%"),
            ("r2",    "Average R²",    ""),
        ]:
            v = summary[key]
            print(f"{label:20s}: {np.mean(v):.4f} {unit}  (std = {np.std(v):.4f} {unit})")
        print("-" * 46)


# =============================================================================
# Training
# =============================================================================

def train(data_path=DEFAULT_DATA_PATH, epochs=200, lr=1e-4,
          batch_size=4, seq_len=1, seed=42):
    """
    Train the ConvLSTM surrogate.

    Parameters
    ----------
    data_path : str
        Glob pattern for MODFLOW head files.
    epochs : int
    lr : float
    batch_size : int
    seq_len : int
        Input sequence length.
    seed : int

    Returns
    -------
    model, train_ds, test_ds, model_path
    """
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    train_loader, val_loader, test_loader, train_ds, test_ds = create_dataloaders(
        data_path, batch_size=batch_size, seq_len=seq_len
    )

    model     = ConvLSTMModel(input_channels=1, hidden_channels=32, seq_len=seq_len).to(dev)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining ConvLSTM for {epochs} epochs | params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    os.makedirs("outputs", exist_ok=True)
    model_path    = f"outputs/convlstm_seq{seq_len}_best.pth"
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                val_loss += criterion(model(xb), yb).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "seq_len": seq_len,
                "train_mean": train_ds.mean,
                "train_std":  train_ds.std,
            }, model_path)

        if epoch % 10 == 0:
            tr_rmse = np.sqrt(avg_train) * train_ds.std
            vl_rmse = np.sqrt(avg_val)   * train_ds.std
            print(f"Epoch {epoch:03d} | Train MSE: {avg_train:.6f} (RMSE: {tr_rmse:.3f} m) | "
                  f"Val MSE: {avg_val:.6f} (RMSE: {vl_rmse:.3f} m)")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_path}")

    # Load best weights
    ckpt = torch.load(model_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Test set evaluation ──────────────────────────────────────────────────
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            p_m = pred.cpu().numpy() * train_ds.std + train_ds.mean
            y_m = yb.cpu().numpy()   * train_ds.std + train_ds.mean
            all_preds.append(p_m)
            all_targets.append(y_m)

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    global_mae  = np.mean(np.abs(all_targets - all_preds))
    global_rmse = np.sqrt(np.mean((all_targets - all_preds) ** 2))
    global_mean = np.mean(all_targets)
    global_rrmse = (global_rmse / global_mean) * 100
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - global_mean) ** 2)
    global_r2 = 1 - (ss_res / ss_tot)

    print("\n=== Test Set Performance ===")
    print(f"MAE:   {global_mae:.3f} m")
    print(f"RMSE:  {global_rmse:.3f} m")
    print(f"RRMSE: {global_rrmse:.3f} %")
    print(f"R²:    {global_r2:.6f}")
    print("=" * 30)

    # ── Rollout evaluation ───────────────────────────────────────────────────
    mean_t = torch.tensor(train_ds.mean, device=dev, dtype=torch.float32)
    std_t  = torch.tensor(train_ds.std,  device=dev, dtype=torch.float32)
    rollout_metrics_and_report(
        model, test_ds, dev,
        seq_len=seq_len, horizons=[1, 2, 3, 4, 5],
        mean=mean_t, std=std_t,
    )

    return model, train_ds, test_ds, model_path


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ConvLSTM surrogate for MODFLOW groundwater head prediction"
    )
    parser.add_argument("--data_path",  type=str,   default=DEFAULT_DATA_PATH,
                        help="Glob pattern for MODFLOW .txt files")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--seq_len",    type=int,   default=1,
                        help="Input sequence length (number of consecutive frames)")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
    )
