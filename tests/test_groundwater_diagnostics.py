from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "diagnostics"))

import paths  # noqa: E402
from evaluate_b1 import B1  # noqa: E402
from evaluate_b2 import B2  # noqa: E402
from groundwater_diagnostics import RunConfig, discover_snapshots, run_diagnostics  # noqa: E402


B1_REFERENCE = paths.benchmark_dir("b1")
B2_REFERENCE = paths.benchmark_dir("b2")

# The repository holds code only; the MODFLOW reference data is distributed
# separately. Point GW_DATA at it to run these checks. See src/paths.py.
requires_data = pytest.mark.skipif(
    not (B1_REFERENCE.is_dir() and B2_REFERENCE.is_dir()),
    reason=f"benchmark reference data not found under {paths.data_root()}; "
           f"set {paths.ENV_VAR}",
)


def _self_check(spec, reference: Path, output: Path) -> dict:
    return run_diagnostics(RunConfig(
        benchmark=spec,
        reference_dir=reference,
        prediction_dir=reference,
        output_dir=output,
        model_name=f"{spec.code} MODFLOW self-check",
        prediction_mode="direct",
        grid_strides=(1, 2),
        make_plots=False,
    ))


@requires_data
def test_snapshot_discovery() -> None:
    assert len(discover_snapshots(B1_REFERENCE)) == 33
    assert len(discover_snapshots(B2_REFERENCE)) == 33


@requires_data
def test_b1_modflow_self_check(tmp_path: Path) -> None:
    summary = _self_check(B1, B1_REFERENCE, tmp_path / "b1")["summary"]
    assert summary["benchmark"]["south_head_m"] == pytest.approx(90.0)
    assert summary["benchmark"]["north_head_m"] == pytest.approx(90.0)
    assert summary["benchmark"]["k_min_m_per_d"] == pytest.approx(33.33)
    assert B1.temporal_scheme == "backward"
    assert summary["evaluation"]["temporal_scheme"] == "mode-aware backward difference"
    assert summary["evaluation"]["spatial_operator"] == "expanded-central"
    assert summary["accuracy"]["mean_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert summary["pde_residual"]["model_to_modflow_rmse_ratio"] == pytest.approx(1.0)


@requires_data
def test_b2_modflow_self_check(tmp_path: Path) -> None:
    summary = _self_check(B2, B2_REFERENCE, tmp_path / "b2")["summary"]
    assert summary["benchmark"]["south_head_m"] == pytest.approx(90.0)
    assert summary["benchmark"]["north_head_m"] == pytest.approx(100.0)
    assert summary["benchmark"]["k_max_m_per_d"] / summary["benchmark"]["k_min_m_per_d"] \
        == pytest.approx(10.0, rel=0.01)
    assert B2.temporal_scheme == "numpy-gradient"
    assert summary["evaluation"]["temporal_scheme"] == "mode-aware backward difference"
    assert summary["evaluation"]["spatial_operator"] == "finite-volume"
    assert summary["accuracy"]["mean_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert summary["pde_residual"]["model_to_modflow_rmse_ratio"] == pytest.approx(1.0)
