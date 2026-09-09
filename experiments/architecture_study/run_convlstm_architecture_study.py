#!/usr/bin/env python
"""Run matched ConvLSTM architecture studies for groundwater B1 and B2."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paths  # noqa: E402


HERE = Path(__file__).resolve().parent
TRAIN_END_DAY = 20
COMMON_TRAIN_TARGETS = tuple(range(3, 21))
VALIDATION_DAYS = tuple(range(21, 26))
TEST_DAYS = tuple(range(26, 31))

BENCHMARKS = {
    "b1": {
        "label": "homogeneous K, 90 m on both CHD rows",
        "south_head": 90.0,
        "north_head": 90.0,
        "data_dir": str(paths.benchmark_dir("b1")),
    },
    "b2": {
        "label": "heterogeneous low-K lens, 90 to 100 m regional gradient",
        "south_head": 90.0,
        "north_head": 100.0,
        "data_dir": str(paths.benchmark_dir("b2")),
    },
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    seq_len: int
    recurrent_layers: int
    hidden_channels: int
    kernel_size: int
    dilation: int
    peephole: bool


VARIANTS = {
    "direct_seq1": VariantSpec(
        name="direct_seq1",
        description="single-layer direct-head ConvLSTM with one input field",
        seq_len=1,
        recurrent_layers=1,
        hidden_channels=32,
        kernel_size=5,
        dilation=1,
        peephole=False,
    ),
    "direct_seq3": VariantSpec(
        name="direct_seq3",
        description="single-layer direct-head ConvLSTM with three input fields",
        seq_len=3,
        recurrent_layers=1,
        hidden_channels=32,
        kernel_size=5,
        dilation=1,
        peephole=False,
    ),
    "peephole_seq3": VariantSpec(
        name="peephole_seq3",
        description="single-layer direct-head ConvLSTM with peephole gates",
        seq_len=3,
        recurrent_layers=1,
        hidden_channels=32,
        kernel_size=5,
        dilation=1,
        peephole=True,
    ),
    "dilated_seq3": VariantSpec(
        name="dilated_seq3",
        description="single-layer direct-head ConvLSTM with dilation four",
        seq_len=3,
        recurrent_layers=1,
        hidden_channels=32,
        kernel_size=5,
        dilation=4,
        peephole=False,
    ),
    "stacked_seq3": VariantSpec(
        name="stacked_seq3",
        description="two-layer direct-head ConvLSTM with three input fields",
        seq_len=3,
        recurrent_layers=2,
        hidden_channels=24,
        kernel_size=5,
        dilation=1,
        peephole=False,
    ),
}


def set_seed(seed: int) -> None:
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


def extract_time(path: str | Path) -> float:
    match = re.search(r"t([0-9]+(?:\.[0-9]+)?)", Path(path).stem)
    if not match:
        raise ValueError(f"cannot parse time from {path}")
    return float(match.group(1))


def load_xyz_grid(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    x, y, values = data[:, 0], data[:, 1], data[:, -1]
    ux, uy = np.unique(x), np.unique(y)
    field = np.full((len(uy), len(ux)), np.nan, dtype=np.float32)
    field[np.searchsorted(uy, y), np.searchsorted(ux, x)] = values
    if np.any(~np.isfinite(field)):
        raise ValueError(f"incomplete grid in {path}")
    return ux, uy, field


def benchmark_initial_head(
    benchmark: str, y_coords: np.ndarray, nx: int
) -> np.ndarray:
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


def load_daily_states(
    data_dir: str | Path, benchmark: str
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    files = sorted(
        glob.glob(str(Path(data_dir) / "t*.txt")), key=extract_time
    )
    if len(files) != 33:
        raise ValueError(
            f"expected 33 saved {benchmark.upper()} snapshots, found {len(files)}"
        )
    by_time = {extract_time(path): path for path in files}
    daily_times = [float(day) for day in range(1, 31)]
    missing = [time for time in daily_times if time not in by_time]
    if missing:
        raise ValueError(f"missing daily snapshots: {missing}")

    daily_files = [by_time[time] for time in daily_times]
    loaded = [load_xyz_grid(path) for path in daily_files]
    x_coords, y_coords = loaded[0][0], loaded[0][1]
    daily_grids = np.stack([item[2] for item in loaded]).astype(np.float32)
    initial = benchmark_initial_head(benchmark, y_coords, len(x_coords))
    grids = np.concatenate((initial[None], daily_grids), axis=0)
    times = np.arange(0.0, 31.0, dtype=np.float64)
    if grids.shape != (31, 300, 300):
        raise ValueError(f"unexpected daily shape: {grids.shape}")

    cfg = BENCHMARKS[benchmark]
    if not np.allclose(grids[1:, 0, :], cfg["south_head"], atol=0.05):
        raise ValueError("saved south CHD row does not match benchmark")
    if not np.allclose(grids[1:, -1, :], cfg["north_head"], atol=0.05):
        raise ValueError("saved north CHD row does not match benchmark")
    return times, grids, daily_files, x_coords, y_coords


class SequenceDataset(Dataset):
    def __init__(self, fields_norm: np.ndarray, target_days: tuple[int, ...], seq_len: int):
        self.fields_norm = fields_norm
        self.target_days = np.asarray(target_days, dtype=int)
        self.seq_len = int(seq_len)
        if np.any(self.target_days < self.seq_len):
            raise ValueError("target day lacks a complete input history")

    def __len__(self) -> int:
        return len(self.target_days)

    def __getitem__(self, item: int):
        target = int(self.target_days[item])
        inputs = self.fields_norm[target - self.seq_len:target]
        output = self.fields_norm[target][None]
        return (
            torch.from_numpy(inputs).float(),
            torch.from_numpy(output).float(),
            target,
        )


class DilatedConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        peephole: bool,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.peephole = bool(peephole)
        if self.peephole:
            shape = (1, hidden_channels, 1, 1)
            self.weight_ci = nn.Parameter(torch.zeros(shape))
            self.weight_cf = nn.Parameter(torch.zeros(shape))
            self.weight_co = nn.Parameter(torch.zeros(shape))

    def forward(self, x, h_prev, c_prev):
        gates = self.conv(torch.cat((x, h_prev), dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        if self.peephole:
            i = i + self.weight_ci * c_prev
            f = f + self.weight_cf * c_prev
        i, f = torch.sigmoid(i), torch.sigmoid(f)
        c = f * c_prev + i * torch.tanh(g)
        if self.peephole:
            o = o + self.weight_co * c
        o = torch.sigmoid(o)
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMLayer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        peephole: bool,
    ) -> None:
        super().__init__()
        self.cell = DilatedConvLSTMCell(
            input_channels, hidden_channels, kernel_size, dilation, peephole
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.dim() == 4:
            sequence = sequence.unsqueeze(2)
        batch, steps, _, ny, nx = sequence.shape
        h = sequence.new_zeros((batch, self.cell.hidden_channels, ny, nx))
        c = torch.zeros_like(h)
        outputs = []
        for step in range(steps):
            h, c = self.cell(sequence[:, step], h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class DirectHeadConvLSTM(nn.Module):
    def __init__(
        self,
        spec: VariantSpec,
        constrained: bool,
        field_mean: float,
        field_std: float,
        south_target: float,
        north_target: float,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.constrained = bool(constrained)
        layers = []
        input_channels = 1
        for _ in range(spec.recurrent_layers):
            layers.append(
                ConvLSTMLayer(
                    input_channels,
                    spec.hidden_channels,
                    spec.kernel_size,
                    spec.dilation,
                    spec.peephole,
                )
            )
            input_channels = spec.hidden_channels
        self.recurrent_layers = nn.ModuleList(layers)
        decoder_channels = 2 * spec.hidden_channels
        self.conv1 = nn.Conv2d(
            spec.hidden_channels, decoder_channels, kernel_size=7, padding=3
        )
        self.prelu1 = nn.PReLU()
        self.dropout = nn.Dropout(0.2)
        self.conv_out = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.register_buffer(
            "south_norm",
            torch.tensor((south_target - field_mean) / field_std),
            persistent=False,
        )
        self.register_buffer(
            "north_norm",
            torch.tensor((north_target - field_mean) / field_std),
            persistent=False,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden_sequence = sequence
        for layer in self.recurrent_layers:
            hidden_sequence = layer(hidden_sequence)
        hidden = hidden_sequence[:, -1]
        prediction = self.conv_out(
            self.dropout(self.prelu1(self.conv1(hidden)))
        )
        if self.constrained:
            prediction = prediction.clone()
            prediction[:, :, 0, :] = self.south_norm
            prediction[:, :, -1, :] = self.north_norm
        return prediction


def receptive_fields(spec: VariantSpec) -> tuple[int, int]:
    recurrent_radius = spec.dilation * (spec.kernel_size // 2)
    decoder_radius = 3
    latest_radius = spec.recurrent_layers * recurrent_radius
    oldest_radius = (
        spec.seq_len + spec.recurrent_layers - 1
    ) * recurrent_radius
    latest = 1 + 2 * (latest_radius + decoder_radius)
    oldest = 1 + 2 * (oldest_radius + decoder_radius)
    return latest, oldest


def experiment_slug(spec: VariantSpec) -> str:
    latest, maximum = receptive_fields(spec)
    return (
        f"{spec.name}_h{spec.hidden_channels}_k{spec.kernel_size}_"
        f"d{spec.dilation}_l{spec.recurrent_layers}_rf{latest}-{maximum}"
    )


def output_directory(
    output_root: Path,
    benchmark: str,
    arm: str,
    seed: int,
    spec: VariantSpec,
) -> Path:
    return output_root / benchmark / experiment_slug(spec) / arm / f"seed{seed}"


def diagnostics_directory(
    diagnostics_root: Path,
    benchmark: str,
    arm: str,
    seed: int,
    spec: VariantSpec,
    mode: str,
) -> Path:
    return (
        diagnostics_root
        / benchmark
        / experiment_slug(spec)
        / arm
        / f"seed{seed}"
        / mode
    )


def field_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
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


def summarize_predictions(
    predictions: list[np.ndarray], grids: np.ndarray, target_days: tuple[int, ...]
) -> dict:
    rows = [
        {"time_d": float(target), **field_metrics(pred, grids[target])}
        for pred, target in zip(predictions, target_days)
    ]
    return {
        "per_time": rows,
        "mean_mse_m2": float(np.mean([row["mse_m2"] for row in rows])),
        "mean_mae_m": float(np.mean([row["mae_m"] for row in rows])),
        "mean_rmse_m": float(np.mean([row["rmse_m"] for row in rows])),
        "mean_rrmse_percent": float(np.mean([row["rrmse_percent"] for row in rows])),
        "mean_r2": float(np.mean([row["r2"] for row in rows])),
    }


def save_field(
    directory: Path,
    target_day: int,
    prediction: np.ndarray,
    target_file: str | Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = np.loadtxt(target_file)
    x, y = data[:, 0], data[:, 1]
    ux, uy = np.unique(x), np.unique(y)
    values = prediction[np.searchsorted(uy, y), np.searchsorted(ux, x)]
    np.savetxt(
        directory / f"convlstm_pred_t{target_day}.txt",
        np.column_stack((x, y, values)),
        fmt="%.6f",
        header="x y h",
        comments="",
    )


@torch.no_grad()
def validation_loss(model, loader, criterion, device) -> float:
    model.eval()
    losses = []
    for inputs, targets, _ in loader:
        prediction = model(inputs.to(device))
        losses.append(criterion(prediction, targets.to(device)).item())
    return float(np.mean(losses))


@torch.no_grad()
def one_step_fields(
    model,
    fields_norm,
    target_days,
    seq_len,
    field_mean,
    field_std,
    device,
):
    model.eval()
    predictions = []
    for target in target_days:
        history = fields_norm[target - seq_len:target]
        tensor = torch.from_numpy(history[None]).float().to(device)
        prediction_norm = model(tensor)[0, 0].cpu().numpy()
        predictions.append(prediction_norm * field_std + field_mean)
    return predictions


@torch.no_grad()
def rollout_fields(
    model,
    fields_norm,
    target_days,
    seq_len,
    field_mean,
    field_std,
    device,
):
    model.eval()
    first_target = int(target_days[0])
    history = fields_norm[first_target - seq_len:first_target].copy()
    predictions = []
    for _target in target_days:
        tensor = torch.from_numpy(history[None]).float().to(device)
        prediction_norm = model(tensor)[0, 0].cpu().numpy()
        prediction_field = prediction_norm * field_std + field_mean
        predictions.append(prediction_field)
        history = np.concatenate((history[1:], prediction_norm[None]), axis=0)
    return predictions


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def checkpoint_payload(
    model,
    optimizer,
    epoch,
    val_mse,
    args,
    spec,
    benchmark,
    arm,
    field_mean,
    field_std,
):
    return {
        "epoch": int(epoch),
        "benchmark": benchmark,
        "variant": spec.name,
        "variant_spec": asdict(spec),
        "arm": arm,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_mse_normalized": float(val_mse),
        "seed": args.seed,
        "field_mean": field_mean,
        "field_std": field_std,
        "train_target_days": list(COMMON_TRAIN_TARGETS),
        "validation_target_days": list(VALIDATION_DAYS),
        "test_target_days": list(TEST_DAYS),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
    }


def run_diagnostics(
    args,
    benchmark: str,
    arm: str,
    spec: VariantSpec,
    run_dir: Path,
    reference_dir: Path,
) -> dict[str, str]:
    evaluate_script = args.diagnostics_script_dir / f"evaluate_{benchmark}.py"
    if not evaluate_script.is_file():
        raise FileNotFoundError(f"diagnostic script not found: {evaluate_script}")
    completed = {}
    model_name = (
        f"ConvLSTM {benchmark.upper()} {spec.name} {arm} seed{args.seed}"
    )
    for mode, prediction_subdir in (
        ("one-step", "one_step_predictions"),
        ("rollout", "rollout_predictions"),
    ):
        diagnostic_dir = diagnostics_directory(
            args.diagnostics_root,
            benchmark,
            arm,
            args.seed,
            spec,
            mode,
        )
        command = [
            sys.executable,
            str(evaluate_script),
            "--reference-dir",
            str(reference_dir),
            "--prediction-dir",
            str(run_dir / prediction_subdir),
            "--output-dir",
            str(diagnostic_dir),
            "--model-name",
            model_name,
            "--prediction-mode",
            mode,
        ]
        print("\nRunning:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        completed[mode] = str(diagnostic_dir)
    return completed


def run_one(args, benchmark: str, arm: str, base_spec: VariantSpec) -> None:
    spec = base_spec
    if args.hidden_channels is not None:
        spec = replace(spec, hidden_channels=args.hidden_channels)
    if args.dilation is not None:
        spec = replace(spec, dilation=args.dilation)

    latest_rf, maximum_rf = receptive_fields(spec)
    constrained = arm == "constrained"
    run_dir = output_directory(
        args.output_root, benchmark, arm, args.seed, spec
    )
    checkpoint_path = run_dir / "convlstm_direct_best.pth"
    result_path = run_dir / "evaluation_results.json"
    complete = checkpoint_path.is_file() and result_path.is_file()
    export_only = bool(args.export_only or (args.resume and complete))

    if checkpoint_path.exists() and not (args.overwrite or export_only or args.resume):
        raise FileExistsError(
            f"checkpoint already exists: {checkpoint_path}; use --export-only, "
            "--resume, or --overwrite"
        )
    if args.export_only and not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    if args.dry_run:
        model = DirectHeadConvLSTM(
            spec, constrained, 0.0, 1.0, 0.0, 0.0
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"DRY RUN | {benchmark.upper()} | {arm} | {spec.name} | "
            f"params={parameters:,} | RF latest/max={latest_rf}/{maximum_rf} cells"
        )
        return

    set_seed(args.seed)
    cfg = BENCHMARKS[benchmark]
    data_dir = Path(args.data_dir or cfg["data_dir"])
    _, grids, daily_files, _, _ = load_daily_states(data_dir, benchmark)

    fields = grids.copy()
    field_mean = float(fields[: TRAIN_END_DAY + 1].mean())
    field_std = float(fields[: TRAIN_END_DAY + 1].std())
    fields_norm = (fields - field_mean) / (field_std + 1e-8)
    south_target = float(cfg["south_head"])
    north_target = float(cfg["north_head"])

    train_ds = SequenceDataset(
        fields_norm, COMMON_TRAIN_TARGETS, spec.seq_len
    )
    val_ds = SequenceDataset(fields_norm, VALIDATION_DAYS, spec.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectHeadConvLSTM(
        spec,
        constrained,
        field_mean,
        field_std,
        south_target,
        north_target,
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "experiment_config.json"
    timestamp = datetime.now(timezone.utc).isoformat()
    previous_metadata = {}
    if config_path.is_file() and not args.overwrite:
        previous_metadata = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = {
        **previous_metadata,
        "created_utc": previous_metadata.get("created_utc", timestamp),
        "updated_utc": timestamp,
        "benchmark": benchmark,
        "arm": arm,
        "seed": args.seed,
        "variant": spec.name,
        "variant_spec": asdict(spec),
        "parameters": parameters,
        "receptive_field_latest_cells": latest_rf,
        "receptive_field_maximum_cells": maximum_rf,
        "receptive_field_latest_m": latest_rf * 1000.0 / 300.0,
        "receptive_field_maximum_m": maximum_rf * 1000.0 / 300.0,
        "train_target_days": list(COMMON_TRAIN_TARGETS),
        "validation_target_days": list(VALIDATION_DAYS),
        "test_target_days": list(TEST_DAYS),
        "normalization_states": "days_0_through_20",
        "output_dir": str(run_dir),
    }
    write_json(config_path, metadata)

    print(f"device: {device}")
    print(f"benchmark: {benchmark.upper()} - {cfg['label']}")
    print(f"variant: {spec.name} - {spec.description}")
    print(f"arm: {arm}")
    print("target representation: absolute normalized head")
    print(f"training targets: days 3-20 ({len(train_ds)} matched windows)")
    print("validation targets: days 21-25")
    print("test targets: days 26-30")
    print(f"normalization: mean={field_mean:.6f}, std={field_std:.6f}")
    print(f"parameters: {parameters:,}")
    print(f"receptive field latest/max: {latest_rf}/{maximum_rf} cells")
    print(f"outputs: {run_dir}")

    history = {
        "train_mse_normalized": [],
        "validation_mse_normalized": [],
        "train_rmse_m": [],
        "validation_rmse_m": [],
    }
    if not export_only:
        best_val_mse = float("inf")
        best_epoch = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                loss = criterion(prediction, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            train_mse = float(np.mean(train_losses))
            val_mse = validation_loss(model, val_loader, criterion, device)
            history["train_mse_normalized"].append(train_mse)
            history["validation_mse_normalized"].append(val_mse)
            history["train_rmse_m"].append(float(np.sqrt(train_mse) * field_std))
            history["validation_rmse_m"].append(float(np.sqrt(val_mse) * field_std))
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        epoch,
                        val_mse,
                        args,
                        spec,
                        benchmark,
                        arm,
                        field_mean,
                        field_std,
                    ),
                    checkpoint_path,
                )
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"epoch {epoch:03d} | train RMSE="
                    f"{np.sqrt(train_mse) * field_std:.6f} m | val RMSE="
                    f"{np.sqrt(val_mse) * field_std:.6f} m | best={best_epoch}"
                )

        write_json(run_dir / "training_history.json", history)
        epochs = np.arange(1, args.epochs + 1)
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.plot(epochs, history["train_rmse_m"], label="training")
        axis.plot(epochs, history["validation_rmse_m"], label="validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("RMSE (m)")
        axis.set_title(f"{benchmark.upper()} {spec.name}: {arm}")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "training_history.png", dpi=200, bbox_inches="tight")
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
        f"{np.sqrt(checkpoint['validation_mse_normalized']) * field_std:.8f} m"
    )

    one_step = one_step_fields(
        model,
        fields_norm,
        TEST_DAYS,
        spec.seq_len,
        field_mean,
        field_std,
        device,
    )
    rollout = rollout_fields(
        model,
        fields_norm,
        TEST_DAYS,
        spec.seq_len,
        field_mean,
        field_std,
        device,
    )
    summaries = {
        "one_step": summarize_predictions(one_step, grids, TEST_DAYS),
        "rollout": summarize_predictions(rollout, grids, TEST_DAYS),
    }
    for protocol, predictions in (("one_step", one_step), ("rollout", rollout)):
        directory = run_dir / f"{protocol}_predictions"
        for prediction, target in zip(predictions, TEST_DAYS):
            save_field(directory, target, prediction, daily_files[target - 1])

    boundary_max_error = max(
        max(float(np.max(np.abs(field[0] - cfg["south_head"]))) for field in rollout),
        max(float(np.max(np.abs(field[-1] - cfg["north_head"]))) for field in rollout),
    )
    result = {
        "benchmark": benchmark,
        "variant": spec.name,
        "variant_spec": asdict(spec),
        "arm": arm,
        "seed": args.seed,
        "selected_epoch": int(checkpoint["epoch"]),
        "parameters": parameters,
        "boundary_max_error_m": boundary_max_error,
        "summaries": summaries,
        "normalization": {
            "method": "global_training_z_score",
            "states": "days_0_through_20",
            "mean": field_mean,
            "std": field_std,
            "target": "absolute_head",
        },
    }
    write_json(result_path, result)

    for mode, summary in summaries.items():
        print(
            f"{mode:8s} | MAE={summary['mean_mae_m']:.6f} m | "
            f"RMSE={summary['mean_rmse_m']:.6f} m | "
            f"RRMSE={summary['mean_rrmse_percent']:.6f}% | "
            f"R2={summary['mean_r2']:.6f}"
        )
    print(f"rollout boundary max error: {boundary_max_error:.8f} m")
    if constrained and boundary_max_error >= 1e-4:
        raise AssertionError("hard CHD-row constraint did not hold")

    if args.diagnostics:
        metadata["diagnostic_dirs"] = run_diagnostics(
            args, benchmark, arm, spec, run_dir, data_dir
        )
        write_json(config_path, metadata)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched direct-head ConvLSTM architecture studies"
    )
    parser.add_argument("--benchmark", choices=("b1", "b2", "both"), default="b2")
    parser.add_argument(
        "--arm", choices=("constrained", "unconstrained", "both"), default="both"
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--dilation", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "outputs",
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=paths.diagnostic_results() / "convlstm_architecture_study",
    )
    parser.add_argument(
        "--diagnostics-script-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "src" / "diagnostics",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.hidden_channels is not None and args.hidden_channels < 1:
        parser.error("--hidden-channels must be positive")
    if args.dilation is not None and args.dilation < 1:
        parser.error("--dilation must be positive")
    if args.epochs < 1 and not args.export_only:
        parser.error("--epochs must be positive")
    if args.export_only and args.overwrite:
        parser.error("--export-only and --overwrite are mutually exclusive")
    args.output_root = args.output_root.resolve()
    args.diagnostics_root = args.diagnostics_root.resolve()
    args.diagnostics_script_dir = args.diagnostics_script_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    benchmarks = ("b1", "b2") if args.benchmark == "both" else (args.benchmark,)
    arms = ("unconstrained", "constrained") if args.arm == "both" else (args.arm,)
    spec = VARIANTS[args.variant]
    for benchmark in benchmarks:
        for arm in arms:
            run_one(args, benchmark, arm, spec)


if __name__ == "__main__":
    main()
