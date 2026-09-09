#!/usr/bin/env python
"""Collect ConvLSTM architecture and diagnostic results into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(rows: list[dict], key: str):
    if not rows or key not in rows[0]:
        return None
    return float(np.mean([float(row[key]) for row in rows]))


def diagnostic_values(path: str | None) -> dict:
    if not path:
        return {}
    summary_path = Path(path) / "summary.json"
    if not summary_path.is_file():
        return {}
    payload = read_json(summary_path)
    summary = payload["summary"]
    rows = payload.get("per_time", [])
    return {
        "diagnostic_mae_m": summary["accuracy"]["mean_mae_m"],
        "diagnostic_rmse_m": summary["accuracy"]["mean_rmse_m"],
        "diagnostic_r2": summary["accuracy"]["mean_r2"],
        "pde_rmse_m_per_d": summary["pde_residual"]["model_mean_rmse_m_per_d"],
        "pde_to_modflow_ratio": summary["pde_residual"]["model_to_modflow_rmse_ratio"],
        "mass_residual_m3_per_d": summary["mass_balance"]["mean_abs_residual_m3_per_d"],
        "mass_error_percent": summary["mass_balance"]["mean_abs_percent_of_well"],
        "modflow_mass_error_percent": summary["mass_balance"][
            "modflow_mean_abs_percent_of_well_same_operator"
        ],
        "bc_mae_m": summary["boundary"]["mean_bc_mae_m"],
        "bc_max_m": summary["boundary"]["max_bc_error_m"],
        "north_gradient_sign_match": summary["boundary"][
            "north_gradient_sign_match_fraction"
        ],
        "south_gradient_sign_match": summary["boundary"][
            "south_gradient_sign_match_fraction"
        ],
        "north_flux_m3_per_d": mean(rows, "north_inflow_m3_per_d"),
        "south_flux_m3_per_d": mean(rows, "south_inflow_m3_per_d"),
        "storage_rate_m3_per_d": mean(rows, "storage_rate_m3_per_d"),
    }


def collect(output_root: Path) -> list[dict]:
    rows = []
    for config_path in sorted(output_root.rglob("experiment_config.json")):
        config = read_json(config_path)
        run_dir = config_path.parent
        result_path = run_dir / "evaluation_results.json"
        result = read_json(result_path) if result_path.is_file() else {}
        spec = config["variant_spec"]
        summaries = result.get("summaries", {})
        one_accuracy = summaries.get("one_step", {})
        rollout_accuracy = summaries.get("rollout", {})
        diagnostic_dirs = config.get("diagnostic_dirs", {})
        one = diagnostic_values(diagnostic_dirs.get("one-step"))
        rollout = diagnostic_values(diagnostic_dirs.get("rollout"))

        row = {
            "benchmark": config["benchmark"],
            "variant": config["variant"],
            "arm": config["arm"],
            "seed": config["seed"],
            "seq_len": spec["seq_len"],
            "recurrent_layers": spec["recurrent_layers"],
            "hidden_channels": spec["hidden_channels"],
            "kernel_size": spec["kernel_size"],
            "dilation": spec["dilation"],
            "peephole": spec["peephole"],
            "parameters": config["parameters"],
            "rf_latest_cells": config["receptive_field_latest_cells"],
            "rf_maximum_cells": config["receptive_field_maximum_cells"],
            "rf_latest_m": config["receptive_field_latest_cells"] * 1000.0 / 300.0,
            "rf_maximum_m": config["receptive_field_maximum_cells"] * 1000.0 / 300.0,
            "selected_epoch": result.get("selected_epoch"),
            "one_step_mae_m": one_accuracy.get("mean_mae_m"),
            "one_step_rmse_m": one_accuracy.get("mean_rmse_m"),
            "one_step_r2": one_accuracy.get("mean_r2"),
            "rollout_mae_m": rollout_accuracy.get("mean_mae_m"),
            "rollout_rmse_m": rollout_accuracy.get("mean_rmse_m"),
            "rollout_r2": rollout_accuracy.get("mean_r2"),
            "one_step_pde_rmse_m_per_d": one.get("pde_rmse_m_per_d"),
            "one_step_pde_to_modflow_ratio": one.get("pde_to_modflow_ratio"),
            "one_step_mass_residual_m3_per_d": one.get("mass_residual_m3_per_d"),
            "one_step_mass_error_percent": one.get("mass_error_percent"),
            "one_step_bc_mae_m": one.get("bc_mae_m"),
            "one_step_bc_max_m": one.get("bc_max_m"),
            "one_step_north_flux_m3_per_d": one.get("north_flux_m3_per_d"),
            "one_step_south_flux_m3_per_d": one.get("south_flux_m3_per_d"),
            "one_step_storage_rate_m3_per_d": one.get("storage_rate_m3_per_d"),
            "rollout_pde_rmse_m_per_d": rollout.get("pde_rmse_m_per_d"),
            "rollout_pde_to_modflow_ratio": rollout.get("pde_to_modflow_ratio"),
            "rollout_mass_residual_m3_per_d": rollout.get("mass_residual_m3_per_d"),
            "rollout_mass_error_percent": rollout.get("mass_error_percent"),
            "rollout_bc_mae_m": rollout.get("bc_mae_m"),
            "rollout_bc_max_m": rollout.get("bc_max_m"),
            "rollout_north_gradient_sign_match": rollout.get("north_gradient_sign_match"),
            "rollout_south_gradient_sign_match": rollout.get("south_gradient_sign_match"),
            "rollout_north_flux_m3_per_d": rollout.get("north_flux_m3_per_d"),
            "rollout_south_flux_m3_per_d": rollout.get("south_flux_m3_per_d"),
            "rollout_storage_rate_m3_per_d": rollout.get("storage_rate_m3_per_d"),
            "modflow_mass_error_percent": rollout.get("modflow_mass_error_percent"),
            "output_dir": str(run_dir),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument(
        "--csv", type=Path, default=HERE / "convlstm_architecture_summary.csv"
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
