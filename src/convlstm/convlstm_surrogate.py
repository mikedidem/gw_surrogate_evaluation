import argparse
import glob
import json
import os
import platform
import random
import re
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths  # noqa: E402


TRAIN_END_DAY = 20
VALIDATION_DAYS = tuple(range(21, 26))
TEST_DAYS = tuple(range(26, 31))

BENCHMARKS = {
    "b1": {
        "label": "homogeneous K, 90 m on both CHD rows",
        "south_head": 90.0,
        "north_head": 90.0,
        "data_dir": str(paths.benchmark_dir("b1")),
        "output_root": str(paths.data_root()
                           / "outputs/convlstm_b1_temporal_residual_seq3"),
    },
    "b2": {
        "label": "heterogeneous low-K lens, 90 to 100 m regional gradient",
        "south_head": 90.0,
        "north_head": 100.0,
        "data_dir": str(paths.benchmark_dir("b2")),
        "output_root": str(paths.data_root()
                           / "outputs/convlstm_b2_temporal_residual_seq3"),
    },
}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def extract_time(path):
    match = re.search(r"t([0-9]+(?:\.[0-9]+)?)", Path(path).stem)
    if not match:
        raise ValueError(f"cannot parse time from {path}")
    return float(match.group(1))


def load_xyz_grid(path):
    data = np.loadtxt(path)
    x, y, values = data[:, 0], data[:, 1], data[:, -1]
    ux, uy = np.unique(x), np.unique(y)
    field = np.full((len(uy), len(ux)), np.nan, dtype=np.float32)
    field[np.searchsorted(uy, y), np.searchsorted(ux, x)] = values
    if np.any(~np.isfinite(field)):
        raise ValueError(f"incomplete grid in {path}")
    return ux, uy, field


def benchmark_initial_head(benchmark, y_coords, nx):
    """Exact MODFLOW starting field for B1 or B2."""
    if benchmark == "b1":
        return np.full((len(y_coords), nx), 90.0, dtype=np.float32)
    if benchmark != "b2":
        raise ValueError(f"unknown benchmark: {benchmark}")

    y_coords = np.asarray(y_coords, dtype=np.float64)
    dy = float(np.median(np.diff(y_coords)))
    y0 = float(y_coords[0] - 0.5 * dy)
    length = float(y_coords[-1] + 0.5 * dy - y0)
    eta = np.clip((y_coords - y0) / length, 0.0, 1.0)
    profile = np.sqrt(90.0**2 + (100.0**2 - 90.0**2) * eta)
    return np.repeat(profile[:, None], nx, axis=1).astype(np.float32)


def load_daily_states(data_dir, benchmark):
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {benchmark}")
    files = sorted(glob.glob(str(Path(data_dir) / "t*.txt")), key=extract_time)
    if len(files) != 33:
        raise ValueError(
            f"expected 33 saved {benchmark.upper()} snapshots, found {len(files)}"
        )

    by_time = {extract_time(path): path for path in files}
    daily_times = [float(day) for day in range(1, 31)]
    missing = [time for time in daily_times if time not in by_time]
    if missing:
        raise ValueError(
            f"daily {benchmark.upper()} protocol is missing snapshots: {missing}"
        )

    daily_files = [by_time[time] for time in daily_times]
    loaded = [load_xyz_grid(path) for path in daily_files]
    x_coords, y_coords = loaded[0][0], loaded[0][1]
    daily_grids = np.stack([item[2] for item in loaded]).astype(np.float32)
    initial = benchmark_initial_head(benchmark, y_coords, len(x_coords))
    grids = np.concatenate((initial[None], daily_grids), axis=0)
    times = np.arange(0.0, 31.0, dtype=np.float64)

    if grids.shape != (31, 300, 300):
        raise ValueError(f"unexpected daily {benchmark.upper()} shape: {grids.shape}")
    cfg = BENCHMARKS[benchmark]
    expected_initial_rows = (
        (90.0, 90.0) if benchmark == "b1" else (90.01759, 99.98416)
    )
    if not np.isclose(initial[0, 0], expected_initial_rows[0], atol=5e-5):
        raise ValueError(f"unexpected south day-zero head: {initial[0, 0]}")
    if not np.isclose(initial[-1, 0], expected_initial_rows[1], atol=5e-5):
        raise ValueError(f"unexpected north day-zero head: {initial[-1, 0]}")
    if not np.allclose(grids[1:, 0, :], cfg["south_head"], atol=0.05):
        raise ValueError("saved south CHD row does not match the benchmark")
    if not np.allclose(grids[1:, -1, :], cfg["north_head"], atol=0.05):
        raise ValueError("saved north CHD row does not match the benchmark")

    return times, grids, daily_files, x_coords, y_coords


class SequenceDataset(Dataset):
    def __init__(self, grids_norm, target_days, seq_len):
        self.grids_norm = grids_norm
        self.target_days = np.asarray(target_days, dtype=int)
        self.seq_len = int(seq_len)
        if np.any(self.target_days < self.seq_len):
            raise ValueError("target day lacks a complete observed input sequence")

    def __len__(self):
        return len(self.target_days)

    def __getitem__(self, item):
        target = int(self.target_days[item])
        inputs = self.grids_norm[target - self.seq_len:target]
        output = self.grids_norm[target][None]
        return (
            torch.from_numpy(inputs).float(),
            torch.from_numpy(output).float(),
            target,
        )


class ConvLSTMCell(nn.Module):
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
        gates = self.conv(torch.cat((x, h_prev), dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c_prev + i * torch.tanh(g)
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.cell = ConvLSTMCell(
            input_channels, hidden_channels, kernel_size, kernel_size // 2
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2)
        batch, steps, _, ny, nx = x.shape
        h = x.new_zeros((batch, self.cell.hidden_channels, ny, nx))
        c = torch.zeros_like(h)
        for step in range(steps):
            h, c = self.cell(x[:, step], h, c)
        return h


class TemporalResidualConvLSTM(nn.Module):
    def __init__(self, constrained, train_mean, train_std,
                 south_head, north_head):
        super().__init__()
        self.constrained = bool(constrained)
        self.convlstm = ConvLSTM(1, 32, 5)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        self.prelu1 = nn.PReLU()
        self.dropout = nn.Dropout(0.2)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        self.register_buffer(
            "south_norm",
            torch.tensor((south_head - train_mean) / train_std),
            persistent=False,
        )
        self.register_buffer(
            "north_norm",
            torch.tensor((north_head - train_mean) / train_std),
            persistent=False,
        )

    def forward(self, x):
        hidden = self.convlstm(x)
        correction = self.conv_out(self.dropout(self.prelu1(self.conv1(hidden))))
        latest = x[:, -1:] if x.dim() == 4 else x[:, -1]
        prediction = latest + correction
        if self.constrained:
            prediction = prediction.clone()
            prediction[:, :, 0, :] = self.south_norm
            prediction[:, :, -1, :] = self.north_norm
        return prediction


@torch.no_grad()
def validation_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    for inputs, targets, _ in loader:
        prediction = model(inputs.to(device))
        losses.append(criterion(prediction, targets.to(device)).item())
    return float(np.mean(losses))


def checkpoint_payload(model, optimizer, epoch, val_mse, args, mean, std,
                       train_targets):
    return {
        "epoch": int(epoch),
        "benchmark": args.benchmark,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_mse_normalized": float(val_mse),
        "seed": args.seed,
        "seq_len": args.seq_len,
        "train_mean": mean,
        "train_std": std,
        "architecture": "paper_core_temporal_residual_seq3",
        "constraint": "hard_dirichlet_chd_rows" if args.constrained else "none",
        "temporal_protocol": (
            f"{args.benchmark}_daily_train_through_20_val_21_25_test_26_30"
        ),
        "train_target_days": list(map(int, train_targets)),
        "validation_target_days": list(VALIDATION_DAYS),
        "test_target_days": list(TEST_DAYS),
        "metric_policy": (
            "normalized_mse_for_optimization; physical_units_for_reporting"
        ),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
    }


def field_metrics(prediction, truth):
    error = prediction - truth
    mse_m2 = float(np.mean(error**2))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "mse_m2": mse_m2,
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(mse_m2)),
        "rrmse_percent": 100.0 * float(np.sqrt(mse_m2)) / abs(float(truth.mean())),
        "r2": 1.0 - float(np.sum(error**2)) / denominator,
        "max_abs_m": float(np.max(np.abs(error))),
    }


def save_field(directory, target_day, prediction, target_file):
    directory.mkdir(parents=True, exist_ok=True)
    data = np.loadtxt(target_file)
    x, y = data[:, 0], data[:, 1]
    ux, uy = np.unique(x), np.unique(y)
    values = prediction[np.searchsorted(uy, y), np.searchsorted(ux, x)]
    table = np.column_stack((x, y, values))
    np.savetxt(
        directory / f"convlstm_pred_t{target_day}.txt",
        table,
        fmt="%.6f",
        header="x y h",
        comments="",
    )


@torch.no_grad()
def one_step_fields(model, grids_norm, target_days, seq_len, mean, std, device):
    model.eval()
    predictions = []
    for target in target_days:
        history = grids_norm[target - seq_len:target]
        tensor = torch.from_numpy(history[None]).float().to(device)
        prediction_norm = model(tensor)[0, 0].cpu().numpy()
        predictions.append(prediction_norm * std + mean)
    return predictions


@torch.no_grad()
def rollout_fields(model, grids_norm, target_days, seq_len, mean, std, device):
    model.eval()
    first_target = int(target_days[0])
    history = grids_norm[first_target - seq_len:first_target].copy()
    predictions = []
    for _target in target_days:
        tensor = torch.from_numpy(history[None]).float().to(device)
        prediction_norm = model(tensor)[0, 0].cpu().numpy()
        predictions.append(prediction_norm * std + mean)
        history = np.concatenate((history[1:], prediction_norm[None]), axis=0)
    return predictions


def summarize_predictions(predictions, grids, target_days):
    rows = [
        {"time_d": float(target), **field_metrics(pred, grids[target])}
        for pred, target in zip(predictions, target_days)
    ]
    return {
        "per_time": rows,
        "mean_mse_m2": float(np.mean([row["mse_m2"] for row in rows])),
        "mean_mae_m": float(np.mean([row["mae_m"] for row in rows])),
        "mean_rmse_m": float(np.mean([row["rmse_m"] for row in rows])),
        "mean_rrmse_percent": float(
            np.mean([row["rrmse_percent"] for row in rows])
        ),
        "mean_r2": float(np.mean([row["r2"] for row in rows])),
    }


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = BENCHMARKS[args.benchmark]
    data_dir = args.data_dir or cfg["data_dir"]
    output_root = args.output_root or cfg["output_root"]
    times, grids, daily_files, x_coords, y_coords = load_daily_states(
        data_dir, args.benchmark
    )
    del x_coords, y_coords

    train_mean = float(grids[:TRAIN_END_DAY + 1].mean())
    train_std = float(grids[:TRAIN_END_DAY + 1].std())
    grids_norm = (grids - train_mean) / (train_std + 1e-8)

    train_targets = tuple(range(args.seq_len, TRAIN_END_DAY + 1))
    train_ds = SequenceDataset(grids_norm, train_targets, args.seq_len)
    val_ds = SequenceDataset(grids_norm, VALIDATION_DAYS, args.seq_len)
    test_ds = SequenceDataset(grids_norm, TEST_DAYS, args.seq_len)
    del test_ds

    arm = "constrained_rows" if args.constrained else "unconstrained"
    out_dir = Path(output_root) / arm / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "convlstm_seq3_temporal_residual_best.pth"

    print(f"device: {device}")
    print(f"benchmark: {args.benchmark.upper()} - {cfg['label']}")
    print("Fixed daily protocol; subdaily snapshots excluded")
    print("  available fitting states: days 0-20")
    print(
        f"  training targets: days {train_targets[0]}-20 "
        f"({len(train_targets)} complete SEQ_LEN={args.seq_len} windows)"
    )
    print("  validation targets: days 21-25 (5 windows)")
    print("  test targets: days 26-30 (5 windows)")
    print("  no padding or invented pre-day-zero history")
    print(
        f"  day-zero rows: south={grids[0, 0, 0]:.5f} m, "
        f"north={grids[0, -1, 0]:.5f} m"
    )
    print(f"normalization: mean={train_mean:.6f} m, std={train_std:.6f} m")
    print(f"arm: {arm}")
    print(f"outputs: {out_dir}")

    model = TemporalResidualConvLSTM(
        args.constrained, train_mean, train_std,
        cfg["south_head"], cfg["north_head"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 206210:
        raise ValueError(f"unexpected parameter count: {parameter_count}")
    print(f"parameters: {parameter_count:,}")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    history = {
        "train_mse_normalized": [],
        "validation_mse_normalized": [],
        "train_rmse_m": [],
        "validation_rmse_m": [],
        "units": {
            "train_mse_normalized": "dimensionless",
            "validation_mse_normalized": "dimensionless",
            "train_rmse_m": "m",
            "validation_rmse_m": "m",
        },
    }

    if args.export_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    else:
        best_val_mse = validation_loss(model, val_loader, criterion, device)
        best_epoch = 0
        torch.save(
            checkpoint_payload(
                model, optimizer, 0, best_val_mse, args, train_mean,
                train_std, train_targets
            ),
            checkpoint_path,
        )
        print(f"epoch 000 | val RMSE={np.sqrt(best_val_mse) * train_std:.8f} m")

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                loss = criterion(prediction, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_mse = float(np.mean(train_losses))
            val_mse = validation_loss(model, val_loader, criterion, device)
            history["train_mse_normalized"].append(train_mse)
            history["validation_mse_normalized"].append(val_mse)
            history["train_rmse_m"].append(
                float(np.sqrt(train_mse) * train_std)
            )
            history["validation_rmse_m"].append(
                float(np.sqrt(val_mse) * train_std)
            )
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                torch.save(
                    checkpoint_payload(
                        model, optimizer, epoch, val_mse, args, train_mean,
                        train_std, train_targets
                    ),
                    checkpoint_path,
                )
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"epoch {epoch:03d} | "
                    f"train RMSE={np.sqrt(train_mse) * train_std:.6f} m | "
                    f"val RMSE={np.sqrt(val_mse) * train_std:.6f} m | "
                    f"best={best_epoch}"
                )

        (out_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        epochs = np.arange(1, args.epochs + 1)
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.plot(epochs, history["train_rmse_m"], label="training")
        axis.plot(epochs, history["validation_rmse_m"], label="validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("RMSE (m)")
        axis.set_title(
            f"{args.benchmark.upper()} temporal residual ConvLSTM: {arm}"
        )
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_history.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"selected epoch {checkpoint['epoch']}; validation RMSE="
        f"{np.sqrt(checkpoint['validation_mse_normalized']) * train_std:.8f} m"
    )

    one_step = one_step_fields(
        model, grids_norm, TEST_DAYS, args.seq_len, train_mean, train_std, device
    )
    rollout = rollout_fields(
        model, grids_norm, TEST_DAYS, args.seq_len, train_mean, train_std, device
    )
    persistence_one_step = [grids[target - 1] for target in TEST_DAYS]
    persistence_rollout = [grids[TEST_DAYS[0] - 1]] * len(TEST_DAYS)
    protocols = {
        "one_step": one_step,
        "rollout": rollout,
        "persistence_one_step": persistence_one_step,
        "persistence_rollout": persistence_rollout,
    }
    summaries = {
        name: summarize_predictions(predictions, grids, TEST_DAYS)
        for name, predictions in protocols.items()
    }
    summaries["one_step"]["persistence_skill"] = (
        1.0
        - summaries["one_step"]["mean_rmse_m"]
        / summaries["persistence_one_step"]["mean_rmse_m"]
    )
    summaries["rollout"]["persistence_skill"] = (
        1.0
        - summaries["rollout"]["mean_rmse_m"]
        / summaries["persistence_rollout"]["mean_rmse_m"]
    )

    for protocol, predictions in (("one_step", one_step), ("rollout", rollout)):
        directory = out_dir / f"{protocol}_predictions"
        for prediction, target in zip(predictions, TEST_DAYS):
            save_field(directory, target, prediction, daily_files[target - 1])

    boundary_max_error = max(
        max(
            float(np.max(np.abs(field[0] - cfg["south_head"])))
            for field in rollout
        ),
        max(
            float(np.max(np.abs(field[-1] - cfg["north_head"])))
            for field in rollout
        ),
    )
    result = {
        "benchmark": args.benchmark,
        "model": "temporal_residual_convlstm_seq3",
        "arm": arm,
        "temporal_protocol": (
            f"{args.benchmark}_daily_train_through_20_val_21_25_test_26_30"
        ),
        "seed": args.seed,
        "selected_epoch": int(checkpoint["epoch"]),
        "parameters": parameter_count,
        "seq_len": args.seq_len,
        "boundary_max_error_m": boundary_max_error,
        "normalization": {
            "method": "global_training_z_score",
            "states": "days_0_through_20",
            "mean_m": train_mean,
            "std_m": train_std,
        },
        "units": {
            "mse_m2": "m2",
            "mae_m": "m",
            "rmse_m": "m",
            "rrmse_percent": "percent",
            "r2": "dimensionless",
            "boundary_max_error_m": "m",
        },
        "summaries": summaries,
    }
    (out_dir / "evaluation_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(
        f"{args.benchmark.upper()} temporal residual ConvLSTM "
        f"SEQ_LEN={args.seq_len}: {arm}"
    )
    print("  Accuracy values below are denormalized physical quantities.")
    for name in protocols:
        summary = summaries[name]
        print(
            f"  {name:22s} MSE={summary['mean_mse_m2']:.8f} m2 | "
            f"MAE={summary['mean_mae_m']:.8f} m | "
            f"RMSE={summary['mean_rmse_m']:.8f} m | "
            f"RRMSE={summary['mean_rrmse_percent']:.6f}% | "
            f"R2={summary['mean_r2']:.6f}"
        )
    print(
        "  one-step persistence skill: "
        f"{summaries['one_step']['persistence_skill']:.6f}"
    )
    print(
        "  rollout persistence skill:  "
        f"{summaries['rollout']['persistence_skill']:.6f}"
    )
    print(f"  rollout boundary max error: {boundary_max_error:.8f} m")
    if args.constrained and boundary_max_error >= 1e-4:
        raise AssertionError("hard CHD-row constraint did not hold")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the B1 or B2 temporal residual ConvLSTM"
    )
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="b2")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="override the selected benchmark's default snapshot directory",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="override the selected benchmark's non-overwriting output root",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--constrained", dest="constrained", action="store_true", default=True
    )
    parser.add_argument(
        "--unconstrained", dest="constrained", action="store_false"
    )
    args = parser.parse_args()
    if args.seq_len != 3:
        parser.error("this locked experiment uses --seq-len 3")
    if args.export_only and args.epochs < 0:
        parser.error("--epochs must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
