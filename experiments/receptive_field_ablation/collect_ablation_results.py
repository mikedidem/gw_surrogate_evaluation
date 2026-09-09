#!/usr/bin/env python
"""Collect CNN receptive-field training and diagnostic results into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def diagnostic_values(path: str | None) -> dict:
    if not path:
        return {}
    summary_path = Path(path) / "summary.json"
    if not summary_path.is_file():
        return {}
    payload = read_json(summary_path)["summary"]
    return {
        "pde_rmse_m_per_d": payload["pde_residual"]["model_mean_rmse_m_per_d"],
        "mass_error_percent": payload["mass_balance"]["mean_abs_percent_of_well"],
        "bc_mae_m": payload["boundary"]["mean_bc_mae_m"],
        "bc_max_m": payload["boundary"]["max_bc_error_m"],
    }


def collect(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for config_path in sorted(output_root.rglob("experiment_config.json")):
        config = read_json(config_path)
        run_dir = config_path.parent
        result_path = run_dir / f"results_seed_{config['seed']}.json"
        results = read_json(result_path) if result_path.is_file() else {}
        diagnostic_dirs = config.get("diagnostic_dirs", {})
        one_step = diagnostic_values(diagnostic_dirs.get("one-step"))
        rollout = diagnostic_values(diagnostic_dirs.get("rollout"))
        rows.append(
            {
                "benchmark": config["benchmark"],
                "arm": config["arm"],
                "seed": config["seed"],
                "channels": config["channels"],
                "dilations": "-".join(str(x) for x in config["dilations"]),
                "conv_layers": config["convolutional_layers"],
                "rf_cells": config["receptive_field_cells"],
                "rf_m": config["receptive_field_m"],
                "parameters": config.get("parameters"),
                "selected_epoch": config.get("selected_epoch"),
                "one_step_mae_m": results.get("test_mae_m"),
                "one_step_rmse_m": results.get("test_rmse_m"),
                "rollout_rmse_m": results.get("operational_rollout", {}).get(
                    "mean_rmse_m"
                ),
                "one_step_pde_rmse_m_per_d": one_step.get("pde_rmse_m_per_d"),
                "one_step_mass_error_percent": one_step.get("mass_error_percent"),
                "one_step_bc_mae_m": one_step.get("bc_mae_m"),
                "one_step_bc_max_m": one_step.get("bc_max_m"),
                "rollout_pde_rmse_m_per_d": rollout.get("pde_rmse_m_per_d"),
                "rollout_mass_error_percent": rollout.get("mass_error_percent"),
                "rollout_bc_mae_m": rollout.get("bc_mae_m"),
                "rollout_bc_max_m": rollout.get("bc_max_m"),
                "output_dir": str(run_dir),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument(
        "--csv", type=Path, default=HERE / "ablation_summary.csv"
    )
    args = parser.parse_args()
    rows = collect(args.output_root.resolve())
    if not rows:
        raise FileNotFoundError(f"no experiment metadata found under {args.output_root}")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected {len(rows)} experiments -> {args.csv.resolve()}")


if __name__ == "__main__":
    main()
