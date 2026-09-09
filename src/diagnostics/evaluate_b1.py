"""Run the conservation diagnostics on benchmark B1, homogeneous K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths  # noqa: E402
from groundwater_diagnostics import (  # noqa: E402
    BenchmarkSpec, RunConfig, print_summary, run_diagnostics)


B1 = BenchmarkSpec(
    code="B1",
    name="homogeneous single-well benchmark",
    south_head=90.0,
    north_head=90.0,
    conductivity_constant=33.33,
    conductivity_file=None,
    conductivity_contrast_range=(0.99, 1.01),
    well_file=None,
    temporal_scheme="backward",
    spatial_operator="expanded-central",
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Evaluate CNN, ConvLSTM, PINN, or MODFLOW head snapshots on Benchmark 1."
    )
    command.add_argument(
        "--reference-dir", type=Path,
        default=paths.benchmark_dir("b1"),
        help="B1 MODFLOW x-y-head snapshots",
    )
    command.add_argument("--prediction-dir", type=Path, required=True)
    command.add_argument("--prediction-glob", default="*.txt",
                         help="use *_pred.txt when prediction and ground-truth files share a directory")
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
        benchmark=B1,
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
