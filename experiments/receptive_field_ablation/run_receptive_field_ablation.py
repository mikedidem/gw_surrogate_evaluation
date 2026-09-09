#!/usr/bin/env python
"""Run matched CNN receptive-field ablations for groundwater B1 and B2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
CANONICAL_CNN_DIR = SRC / "cnn"
DIAGNOSTICS_DIR = SRC / "diagnostics"

for path in (SRC, CANONICAL_CNN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import paths  # noqa: E402
import cnn_surrogate as canonical  # noqa: E402
from constrained import DirichletRowProjection  # noqa: E402


def parse_dilations(value: str) -> tuple[int, ...]:
    try:
        dilations = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "dilations must be comma-separated positive integers"
        ) from exc
    if not dilations or any(value < 1 for value in dilations):
        raise argparse.ArgumentTypeError(
            "dilations must be comma-separated positive integers"
        )
    return dilations


class ReceptiveFieldCNN(nn.Module):
    """CNN with a user-selected sequence of 3x3 dilation factors."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 24,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        constrained: bool = False,
        bc: dict | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for dilation in dilations:
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                ]
            )
            current_channels = channels
        layers.append(nn.Conv2d(current_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.dilations = tuple(dilations)
        self.channels = int(channels)

        if constrained and bc is None:
            raise ValueError("constrained=True requires boundary metadata")
        self.bc = DirichletRowProjection(**bc) if constrained else None

    def receptive_field(self) -> int:
        return 1 + 2 * sum(self.dilations)

    def forward(self, head: torch.Tensor) -> torch.Tensor:
        prediction = self.net(head)
        return prediction if self.bc is None else self.bc(prediction)


def install_selected_architecture(
    dilations: tuple[int, ...],
) -> type[ReceptiveFieldCNN]:
    """Install the selected schedule into the canonical training backend."""

    class SelectedReceptiveFieldCNN(ReceptiveFieldCNN):
        def __init__(
            self,
            in_channels: int = 1,
            channels: int = 24,
            constrained: bool = False,
            bc: dict | None = None,
        ) -> None:
            super().__init__(
                in_channels=in_channels,
                channels=channels,
                dilations=dilations,
                constrained=constrained,
                bc=bc,
            )

    canonical.DilatedHeadCNN = SelectedReceptiveFieldCNN
    return SelectedReceptiveFieldCNN


def experiment_slug(dilations: tuple[int, ...], channels: int) -> str:
    receptive_field = 1 + 2 * sum(dilations)
    schedule = "-".join(str(value) for value in dilations)
    return f"rf{receptive_field}_c{channels}_d{schedule}"


def output_directory(
    output_root: Path,
    benchmark: str,
    arm: str,
    seed: int,
    dilations: tuple[int, ...],
    channels: int,
) -> Path:
    return (
        output_root
        / benchmark
        / experiment_slug(dilations, channels)
        / arm
        / f"seed{seed}"
    )


def diagnostics_directory(
    diagnostics_root: Path,
    benchmark: str,
    arm: str,
    seed: int,
    dilations: tuple[int, ...],
    channels: int,
    mode: str,
) -> Path:
    return (
        diagnostics_root
        / benchmark
        / experiment_slug(dilations, channels)
        / arm
        / f"seed{seed}"
        / mode
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_diagnostics(
    benchmark: str,
    arm: str,
    seed: int,
    channels: int,
    dilations: tuple[int, ...],
    run_dir: Path,
    diagnostics_root: Path,
) -> dict[str, str]:
    evaluate_script = DIAGNOSTICS_DIR / f"evaluate_{benchmark}.py"
    if not evaluate_script.is_file():
        raise FileNotFoundError(f"diagnostic script not found: {evaluate_script}")

    reference_dir = (
        GW_MOD_ROOT / "cnn" / "data"
        if benchmark == "b1"
        else GW_MOD_ROOT / "b2_corrected"
    )
    model_name = (
        f"CNN {benchmark.upper()} RF={1 + 2 * sum(dilations)} "
        f"channels={channels} {arm} seed{seed}"
    )
    completed: dict[str, str] = {}

    for mode, prediction_subdir in (
        ("one-step", "test_predictions"),
        ("rollout", "rollout_predictions"),
    ):
        prediction_dir = run_dir / prediction_subdir
        diagnostic_dir = diagnostics_directory(
            diagnostics_root,
            benchmark,
            arm,
            seed,
            dilations,
            channels,
            mode,
        )
        command = [
            sys.executable,
            str(evaluate_script),
            "--reference-dir",
            str(reference_dir),
            "--prediction-dir",
            str(prediction_dir),
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


def run_one(
    args: argparse.Namespace,
    benchmark: str,
    arm: str,
    model_class: type[ReceptiveFieldCNN],
) -> None:
    constrained = arm == "constrained"
    run_dir = output_directory(
        args.output_root,
        benchmark,
        arm,
        args.seed,
        args.dilations,
        args.channels,
    )
    checkpoint = run_dir / "cnn_best_model.pth"
    if checkpoint.exists() and not args.export_only and not args.overwrite:
        raise FileExistsError(
            f"checkpoint already exists: {checkpoint}\n"
            "Use --export-only to reuse it or --overwrite to retrain it."
        )

    receptive_field = 1 + 2 * sum(args.dilations)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "arm": arm,
        "constraint": "hard_dirichlet_chd_rows" if constrained else None,
        "architecture": "configurable_dilated_cnn",
        "kernel_size": 3,
        "dilations": list(args.dilations),
        "convolutional_layers": len(args.dilations),
        "channels": args.channels,
        "receptive_field_cells": receptive_field,
        "receptive_field_m": receptive_field * (1000.0 / 300.0),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "temporal_protocol": (
            f"{benchmark}_daily_targets_train_1_20_val_21_25_test_26_30"
        ),
        "normalization": "global_training_z_score",
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "experiment_config.json", metadata)

    print("\n" + "=" * 78)
    print(
        f"{benchmark.upper()} | {arm} | channels={args.channels} | "
        f"dilations={args.dilations}"
    )
    print(
        f"receptive field: {receptive_field} cells = "
        f"{metadata['receptive_field_m']:.1f} m"
    )
    print(f"outputs: {run_dir}")
    print("=" * 78)

    if args.dry_run:
        h_south, h_north = (90.0, 90.0) if benchmark == "b1" else (90.0, 100.0)
        boundary = {
            "y_coords": list(range(300)),
            "mean": 0.0,
            "std": 1.0,
            "h_south": h_south,
            "h_north": h_north,
        }
        model = model_class(
            channels=args.channels,
            constrained=constrained,
            bc=boundary if constrained else None,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"dry run parameter count: {parameter_count:,}")
        metadata["parameters"] = parameter_count
        write_json(run_dir / "experiment_config.json", metadata)
        return

    model, _, _, _, _ = canonical.train(
        seed=args.seed,
        epochs=0 if args.export_only else args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        data_path=None,
        constrained=constrained,
        benchmark=benchmark,
        out_dir=str(run_dir),
        arch="dilated",
        channels=args.channels,
    )
    metadata["parameters"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    saved_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata["selected_epoch"] = int(saved_checkpoint["epoch"])

    if args.diagnostics:
        metadata["diagnostic_dirs"] = run_diagnostics(
            benchmark=benchmark,
            arm=arm,
            seed=args.seed,
            channels=args.channels,
            dilations=args.dilations,
            run_dir=run_dir,
            diagnostics_root=args.diagnostics_root,
        )
    write_json(run_dir / "experiment_config.json", metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train configurable receptive-field CNNs using the locked B1/B2 "
            "daily protocol."
        )
    )
    parser.add_argument(
        "--benchmark", choices=("b1", "b2", "both"), default="b1"
    )
    parser.add_argument(
        "--arm",
        choices=("unconstrained", "constrained", "both"),
        default="both",
    )
    parser.add_argument(
        "--dilations",
        type=parse_dilations,
        default=parse_dilations("1,2,4,8,16,32,64,128"),
        help="comma-separated 3x3 dilation factors",
    )
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root", type=Path, default=HERE / "outputs"
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=paths.diagnostic_results() / "cnn_rf_ablation",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="run universal one-step and rollout diagnostics after training",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="reuse an existing checkpoint and regenerate predictions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow retraining into an existing experiment directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="construct models and write metadata without loading data or training",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.channels < 1:
        raise ValueError("--channels must be positive")
    if args.epochs < 1 and not args.export_only and not args.dry_run:
        raise ValueError("--epochs must be positive")

    args.output_root = args.output_root.resolve()
    args.diagnostics_root = args.diagnostics_root.resolve()
    model_class = install_selected_architecture(args.dilations)
    benchmarks = ("b1", "b2") if args.benchmark == "both" else (args.benchmark,)
    arms = (
        ("unconstrained", "constrained")
        if args.arm == "both"
        else (args.arm,)
    )
    for benchmark in benchmarks:
        for arm in arms:
            run_one(args, benchmark, arm, model_class)


if __name__ == "__main__":
    main()
