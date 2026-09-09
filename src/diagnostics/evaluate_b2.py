"""Run the conservation diagnostics on benchmark B2, heterogeneous K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths  # noqa: E402
from groundwater_diagnostics import (  # noqa: E402
    BenchmarkSpec, RunConfig, print_summary, run_diagnostics)


B2 = BenchmarkSpec(
    code="B2",
    name="heterogeneous 10:1 K lens with regional gradient",
    south_head=90.0,
    north_head=100.0,
    conductivity_constant=None,
    conductivity_file="kfield.txt",
    conductivity_contrast_range=(8.0, 12.0),
    well_file="b2_model.wel",
    temporal_scheme="numpy-gradient",
    spatial_operator="finite-volume",
)


def default_reference_dir() -> Path:
    return paths.benchmark_dir("b2")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Evaluate CNN, ConvLSTM, PINN, or MODFLOW head snapshots on Benchmark 2."
    )
    command.add_argument(
        "--reference-dir", type=Path,
        default=default_reference_dir(),
        help="B2 directory containing snapshots, kfield.txt, and b2_model.wel",
    )
    command.add_argument("--prediction-dir", type=Path, required=True)
    command.add_argument("--prediction-glob", default="*.txt")
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--model-name", required=True)
    command.add_argument("--prediction-mode", choices=("unknown", "one-step", "rollout", "direct"),
                         default="unknown")
    command.add_argument("--test-start", type=float, default=26.0)
    command.add_argument("--test-end", type=float, default=30.0)
    command.add_argument("--matrix-y-order", choices=("ascending", "descending"), default="ascending")
    command.add_argument("--grid-strides", default="1,2,3,4")
    command.add_argument("--no-plots", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    config = RunConfig(
        benchmark=B2,
        reference_dir=args.reference_dir,
        prediction_dir=args.prediction_dir,
        prediction_glob=args.prediction_glob,
        output_dir=args.output_dir,
        model_name=args.model_name,
        prediction_mode=args.prediction_mode,
        test_start=args.test_start,
        test_end=args.test_end,
        matrix_y_order=args.matrix_y_order,
        grid_strides=tuple(int(value) for value in args.grid_strides.split(",") if value.strip()),
        make_plots=not args.no_plots,
    )
    result = run_diagnostics(config)
    print_summary(result, config.output_dir)


if __name__ == "__main__":
    main()
