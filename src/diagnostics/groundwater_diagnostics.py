"""Transparent accuracy and conservation diagnostics for groundwater heads."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


TIME_RE = re.compile(r"(?:^|[_-])t(?P<time>\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class BenchmarkSpec:
    code: str
    name: str
    south_head: float
    north_head: float
    conductivity_constant: float | None
    conductivity_file: str | None
    conductivity_contrast_range: tuple[float, float]
    well_file: str | None
    well_rate: float = -40_000.0
    well_sigma: float = 30.0
    # Explicit source centre, in model coordinates. Only consulted when
    # well_file is absent, where the source would otherwise be placed at the
    # grid mean. Benchmark 1 needs it: MODFLOW pumps from the cell centred at
    # (501.67, 501.67) while the grid mean is (500.0, 500.0), half a cell away
    # on the diagonal, and that mismatch shows up as a dipole in the residual.
    well_center: tuple[float, float] | None = None
    specific_yield: float = 0.1
    grid_shape: tuple[int, int] = (300, 300)
    temporal_scheme: str = "backward"
    spatial_operator: str = "expanded-central"


@dataclass
class RunConfig:
    benchmark: BenchmarkSpec
    reference_dir: Path
    prediction_dir: Path
    output_dir: Path
    model_name: str
    prediction_mode: str = "unknown"
    prediction_glob: str = "*.txt"
    test_start: float = 26.0
    test_end: float = 30.0
    well_zone_radius: float = 90.0
    k_rim_threshold: float = 0.01
    grid_strides: tuple[int, ...] = (1, 2, 3, 4)
    matrix_y_order: str = "ascending"
    make_plots: bool = True


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def ny(self) -> int:
        return len(self.y)

    @property
    def dx(self) -> float:
        return float(np.median(np.diff(self.x)))

    @property
    def dy(self) -> float:
        return float(np.median(np.diff(self.y)))

    @property
    def area(self) -> float:
        return self.dx * self.dy


@dataclass(frozen=True)
class BenchmarkData:
    spec: BenchmarkSpec
    grid: Grid
    references: dict[float, np.ndarray]
    conductivity: np.ndarray
    well_rates: np.ndarray
    well_x: float
    well_y: float


def extract_time(path: Path) -> float | None:
    match = TIME_RE.search(path.stem)
    return float(match.group("time")) if match else None


def discover_snapshots(directory: Path, pattern: str = "*.txt") -> dict[float, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {directory}")
    snapshots: dict[float, Path] = {}
    for path in sorted(directory.glob(pattern)):
        time = extract_time(path)
        if time is None:
            continue
        if time in snapshots:
            raise ValueError(
                f"multiple snapshots found for t={time:g}: "
                f"{snapshots[time].name}, {path.name}; narrow --prediction-glob"
            )
        snapshots[time] = path
    if not snapshots:
        raise FileNotFoundError(f"no time-stamped snapshots matching {pattern!r} in {directory}")
    return snapshots


def _read_array(path: Path) -> np.ndarray:
    try:
        data = np.loadtxt(path, comments="#", dtype=float)
    except ValueError:
        data = np.loadtxt(path, comments="#", dtype=float, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError(f"unsupported snapshot shape in {path}: {data.shape}")
    return data


def _read_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _read_array(path)
    if data.shape[1] < 3:
        raise ValueError(f"expected x y h columns in {path}, found shape {data.shape}")
    data = data[np.all(np.isfinite(data[:, [0, 1, -1]]), axis=1)]
    if not len(data):
        raise ValueError(f"no numeric x y h records in {path}")
    return data[:, 0], data[:, 1], data[:, -1]


def _grid_from_xyz(path: Path) -> tuple[Grid, np.ndarray]:
    x, y, values = _read_xyz(path)
    ux, uy = np.unique(x), np.unique(y)
    field = np.full((len(uy), len(ux)), np.nan, dtype=float)
    field[np.searchsorted(uy, y), np.searchsorted(ux, x)] = values
    if np.any(~np.isfinite(field)):
        raise ValueError(f"incomplete structured grid in {path}")
    return Grid(ux, uy), field


def _aligned_field(path: Path, target: Grid, matrix_y_order: str = "ascending") -> np.ndarray:
    data = _read_array(path)
    if data.shape == (target.ny, target.nx):
        field = data
        if matrix_y_order == "descending":
            field = field[::-1, :]
        elif matrix_y_order != "ascending":
            raise ValueError("matrix_y_order must be 'ascending' or 'descending'")
        return np.asarray(field, dtype=float)

    x, y, values = _read_xyz(path)
    ux, uy = np.unique(x), np.unique(y)
    if len(ux) != target.nx or len(uy) != target.ny:
        raise ValueError(f"grid in {path} is {len(ux)}x{len(uy)}, expected {target.nx}x{target.ny}")
    x_shift = float(np.mean(target.x) - np.mean(ux))
    y_shift = float(np.mean(target.y) - np.mean(uy))
    shifted_x, shifted_y = ux + x_shift, uy + y_shift
    if not np.allclose(shifted_x, target.x, atol=1e-3) or not np.allclose(
        shifted_y, target.y, atol=1e-3
    ):
        raise ValueError(f"coordinates in {path} do not align with the reference grid")
    field = np.full((target.ny, target.nx), np.nan, dtype=float)
    field[np.searchsorted(uy, y), np.searchsorted(ux, x)] = values
    if np.any(~np.isfinite(field)):
        raise ValueError(f"incomplete prediction grid in {path}")
    return field


def _load_well_rates(spec: BenchmarkSpec, reference_dir: Path, grid: Grid) -> np.ndarray:
    rates = np.zeros((grid.ny, grid.nx), dtype=float)
    path = reference_dir / spec.well_file if spec.well_file else None
    if path is not None and path.exists():
        numeric: list[list[str]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped:
                numeric.append(stripped.split())
        if len(numeric) < 3:
            raise ValueError(f"cannot parse MODFLOW WEL file: {path}")
        count = int(numeric[1][0])
        records = numeric[2:2 + count]
        if len(records) != count:
            raise ValueError(f"incomplete first stress period in WEL file: {path}")
        for record in records:
            row, col, rate = int(record[1]), int(record[2]), float(record[3])
            rates[grid.ny - row, col - 1] = rate
        return rates

    gx, gy = np.meshgrid(grid.x, grid.y)
    if spec.well_center is not None:
        x0, y0 = (float(spec.well_center[0]), float(spec.well_center[1]))
    else:
        x0, y0 = float(np.mean(grid.x)), float(np.mean(grid.y))
    weights = np.exp(-((gx - x0) ** 2 + (gy - y0) ** 2) / (2.0 * spec.well_sigma ** 2))
    return spec.well_rate * weights / weights.sum()


def load_benchmark(config: RunConfig) -> BenchmarkData:
    spec = config.benchmark
    reference_files = discover_snapshots(config.reference_dir)
    first_time = min(reference_files)
    grid, first_field = _grid_from_xyz(reference_files[first_time])
    references = {
        time: first_field if time == first_time else _aligned_field(path, grid)
        for time, path in sorted(reference_files.items())
    }

    if spec.conductivity_file:
        k_path = config.reference_dir / spec.conductivity_file
        if not k_path.exists():
            raise FileNotFoundError(f"{spec.code} conductivity file not found: {k_path}")
        conductivity = _aligned_field(k_path, grid)
    elif spec.conductivity_constant is not None:
        conductivity = np.full((grid.ny, grid.nx), spec.conductivity_constant, dtype=float)
    else:
        raise ValueError("benchmark must define a conductivity constant or file")

    south = np.array([field[0, :].mean() for field in references.values()])
    north = np.array([field[-1, :].mean() for field in references.values()])
    contrast = float(np.max(conductivity) / np.min(conductivity))
    problems: list[str] = []
    if not np.allclose(south, spec.south_head, atol=0.05):
        problems.append(f"south head is {np.median(south):.3f} m, expected {spec.south_head:g} m")
    if not np.allclose(north, spec.north_head, atol=0.05):
        problems.append(f"north head is {np.median(north):.3f} m, expected {spec.north_head:g} m")
    low, high = spec.conductivity_contrast_range
    if not low <= contrast <= high:
        problems.append(f"K contrast is {contrast:.2f}:1, expected {low:g}-{high:g}:1")
    expected_ny, expected_nx = spec.grid_shape
    if (grid.ny, grid.nx) != spec.grid_shape:
        problems.append(f"grid is {grid.ny}x{grid.nx}, expected {expected_ny}x{expected_nx}")
    if problems:
        raise ValueError(f"reference data does not match {spec.code}: " + "; ".join(problems))

    well_rates = _load_well_rates(spec, config.reference_dir, grid)
    if not math.isclose(float(well_rates.sum()), spec.well_rate, abs_tol=0.1):
        raise ValueError(
            f"well rates sum to {well_rates.sum():.3f}, expected {spec.well_rate:.3f} m3/d"
        )
    if spec.well_file and (config.reference_dir / spec.well_file).exists():
        well_iy, well_ix = np.unravel_index(np.argmax(np.abs(well_rates)), well_rates.shape)
        well_x, well_y = float(grid.x[well_ix]), float(grid.y[well_iy])
    elif spec.well_center is not None:
        well_x, well_y = (float(spec.well_center[0]), float(spec.well_center[1]))
    else:
        well_x, well_y = float(np.mean(grid.x)), float(np.mean(grid.y))
    return BenchmarkData(
        spec=spec,
        grid=grid,
        references=references,
        conductivity=conductivity,
        well_rates=well_rates,
        well_x=well_x,
        well_y=well_y,
    )


def _temporal_derivative(fields: np.ndarray, times: np.ndarray, scheme: str) -> np.ndarray:
    if len(times) < 2:
        raise ValueError("at least two consecutive prediction snapshots are required for dh/dt")
    if np.any(np.diff(times) <= 0):
        raise ValueError("prediction times must be strictly increasing")
    if scheme == "backward":
        derivative = np.empty_like(fields)
        derivative[0] = (fields[1] - fields[0]) / (times[1] - times[0])
        derivative[1:] = np.diff(fields, axis=0) / np.diff(times)[:, None, None]
        return derivative
    if scheme == "numpy-gradient":
        edge_order = 2 if len(times) >= 3 else 1
        return np.gradient(fields, times, axis=0, edge_order=edge_order)
    raise ValueError(f"unknown temporal scheme: {scheme}")


def _previous_reference(
    benchmark: BenchmarkData, time: float,
) -> tuple[float, np.ndarray] | None:
    reference_times = np.asarray(sorted(benchmark.references), dtype=float)
    position = int(np.searchsorted(reference_times, time))
    if position == 0:
        return None
    previous_time = float(reference_times[position - 1])
    return previous_time, benchmark.references[previous_time]


def _reference_derivative(
    benchmark: BenchmarkData, times: np.ndarray,
) -> np.ndarray:
    derivatives = []
    reference_times = np.asarray(sorted(benchmark.references), dtype=float)
    for time_value in times:
        time = float(time_value)
        current = benchmark.references[time]
        previous = _previous_reference(benchmark, time)
        if previous is not None:
            previous_time, previous_field = previous
            derivatives.append((current - previous_field) / (time - previous_time))
            continue
        position = int(np.searchsorted(reference_times, time))
        if position + 1 >= len(reference_times):
            raise ValueError(f"cannot form a reference derivative at t={time:g}")
        next_time = float(reference_times[position + 1])
        derivatives.append((benchmark.references[next_time] - current) / (next_time - time))
    return np.stack(derivatives)


def _prediction_derivative(
    predictions: np.ndarray,
    times: np.ndarray,
    benchmark: BenchmarkData,
    mode: str,
) -> tuple[np.ndarray, list[float], list[str]]:
    derivatives = []
    predecessor_times: list[float] = []
    predecessor_sources: list[str] = []

    for index, time_value in enumerate(times):
        time = float(time_value)
        if mode == "one-step":
            previous = _previous_reference(benchmark, time)
            if previous is None:
                raise ValueError(
                    f"one-step prediction at t={time:g} has no preceding reference snapshot"
                )
            previous_time, previous_field = previous
            derivatives.append((predictions[index] - previous_field) / (time - previous_time))
            predecessor_times.append(previous_time)
            predecessor_sources.append("observed-reference")
            continue

        if index > 0:
            previous_time = float(times[index - 1])
            derivatives.append(
                (predictions[index] - predictions[index - 1]) / (time - previous_time)
            )
            predecessor_times.append(previous_time)
            predecessor_sources.append("predicted")
            continue

        previous = _previous_reference(benchmark, time)
        if previous is not None:
            previous_time, previous_field = previous
            derivatives.append((predictions[index] - previous_field) / (time - previous_time))
            predecessor_times.append(previous_time)
            predecessor_sources.append("observed-reference-anchor")
            continue

        if len(times) < 2:
            raise ValueError(f"cannot form a prediction derivative at t={time:g}")
        next_time = float(times[1])
        derivatives.append((predictions[1] - predictions[0]) / (next_time - time))
        predecessor_times.append(next_time)
        predecessor_sources.append("predicted-forward")

    return np.stack(derivatives), predecessor_times, predecessor_sources


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = a + b
    return np.divide(2.0 * a * b, denominator, out=np.zeros_like(denominator), where=denominator != 0)


def _finite_volume_divergence(h: np.ndarray, k: np.ndarray, grid: Grid) -> np.ndarray:
    kx = _harmonic(k[:, :-1], k[:, 1:])
    ky = _harmonic(k[:-1, :], k[1:, :])
    qx = kx * 0.5 * (h[:, :-1] + h[:, 1:]) * np.diff(h, axis=1) / grid.dx
    qy = ky * 0.5 * (h[:-1, :] + h[1:, :]) * np.diff(h, axis=0) / grid.dy
    return (
        (qx[1:-1, 1:] - qx[1:-1, :-1]) / grid.dx
        + (qy[1:, 1:-1] - qy[:-1, 1:-1]) / grid.dy
    )


def _expanded_central_divergence(h: np.ndarray, k: np.ndarray, grid: Grid) -> np.ndarray:
    center = h[1:-1, 1:-1]
    hx = (h[1:-1, 2:] - h[1:-1, :-2]) / (2.0 * grid.dx)
    hy = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2.0 * grid.dy)
    hxx = (h[1:-1, 2:] - 2.0 * center + h[1:-1, :-2]) / grid.dx ** 2
    hyy = (h[2:, 1:-1] - 2.0 * center + h[:-2, 1:-1]) / grid.dy ** 2
    kx = (k[1:-1, 2:] - k[1:-1, :-2]) / (2.0 * grid.dx)
    ky = (k[2:, 1:-1] - k[:-2, 1:-1]) / (2.0 * grid.dy)
    return k[1:-1, 1:-1] * (center * (hxx + hyy) + hx ** 2 + hy ** 2) \
        + center * (kx * hx + ky * hy)


def _pde_residual(
    fields: np.ndarray,
    dhdt: np.ndarray,
    benchmark: BenchmarkData,
) -> np.ndarray:
    source_density = benchmark.well_rates / benchmark.grid.area
    residuals = []
    for h, ht in zip(fields, dhdt):
        if benchmark.spec.spatial_operator == "finite-volume":
            divergence = _finite_volume_divergence(h, benchmark.conductivity, benchmark.grid)
        elif benchmark.spec.spatial_operator == "expanded-central":
            divergence = _expanded_central_divergence(h, benchmark.conductivity, benchmark.grid)
        else:
            raise ValueError(f"unknown spatial operator: {benchmark.spec.spatial_operator}")
        residuals.append(
            benchmark.spec.specific_yield * ht[1:-1, 1:-1]
            - divergence
            - source_density[1:-1, 1:-1]
        )
    return np.stack(residuals)


def _zone_masks(benchmark: BenchmarkData, k: np.ndarray | None = None,
                grid: Grid | None = None, well_radius: float = 90.0,
                rim_threshold: float = 0.01) -> dict[str, np.ndarray]:
    grid = grid or benchmark.grid
    k = benchmark.conductivity if k is None else k
    gx, gy = np.meshgrid(grid.x, grid.y)
    well = np.hypot(gx - benchmark.well_x, gy - benchmark.well_y) <= well_radius
    masks: dict[str, np.ndarray] = {"well": well[1:-1, 1:-1]}
    contrast = float(k.max() / k.min())
    if contrast > 1.05:
        dky, dkx = np.gradient(k, grid.dy, grid.dx, edge_order=2)
        rim = np.hypot(dkx, dky) > rim_threshold
        lens = k < 0.5 * (float(k.min()) + float(k.max()))
        masks["lens_rim"] = rim[1:-1, 1:-1]
        masks["lens_core"] = lens[1:-1, 1:-1]
        masks["background"] = ~(masks["well"] | masks["lens_rim"])
    else:
        masks["background"] = ~masks["well"]
    masks["overall"] = np.ones_like(masks["well"], dtype=bool)
    return masks


def _residual_stats(residual: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, mask in masks.items():
        values = residual[mask]
        result[f"{name}_mean_abs_m_per_d"] = float(np.mean(np.abs(values)))
        result[f"{name}_rmse_m_per_d"] = float(np.sqrt(np.mean(values ** 2)))
        result[f"{name}_max_abs_m_per_d"] = float(np.max(np.abs(values)))
    return result


def _accuracy(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    mse = float(np.mean(error ** 2))
    denominator = float(np.sum((reference - reference.mean()) ** 2))
    return {
        "mse_m2": mse,
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": math.sqrt(mse),
        "rrmse_percent": 100.0 * math.sqrt(mse) / abs(float(reference.mean())),
        "r2": 1.0 - float(np.sum(error ** 2)) / denominator if denominator else float("nan"),
        "max_abs_m": float(np.max(np.abs(error))),
    }


def _boundary_fluxes(h: np.ndarray, benchmark: BenchmarkData) -> dict[str, float]:
    k, grid = benchmark.conductivity, benchmark.grid
    kn, ks = _harmonic(k[-1, :], k[-2, :]), _harmonic(k[0, :], k[1, :])
    kw, ke = _harmonic(k[:, 0], k[:, 1]), _harmonic(k[:, -1], k[:, -2])
    north = kn * 0.5 * (h[-1, :] + h[-2, :]) * (h[-1, :] - h[-2, :]) / grid.dy
    south = ks * 0.5 * (h[0, :] + h[1, :]) * (h[0, :] - h[1, :]) / grid.dy
    west = kw * 0.5 * (h[:, 0] + h[:, 1]) * (h[:, 0] - h[:, 1]) / grid.dx
    east = ke * 0.5 * (h[:, -1] + h[:, -2]) * (h[:, -1] - h[:, -2]) / grid.dx
    return {
        "north_inflow_m3_per_d": float(np.sum(north) * grid.dx),
        "south_inflow_m3_per_d": float(np.sum(south) * grid.dx),
        "west_derived_inflow_m3_per_d": float(np.sum(west) * grid.dy),
        "east_derived_inflow_m3_per_d": float(np.sum(east) * grid.dy),
        "west_no_flow_rmse_m2_per_d": float(np.sqrt(np.mean(west ** 2))),
        "east_no_flow_rmse_m2_per_d": float(np.sqrt(np.mean(east ** 2))),
    }


def _boundary_error(h: np.ndarray, benchmark: BenchmarkData) -> dict[str, float]:
    errors = np.concatenate((
        np.abs(h[0, :] - benchmark.spec.south_head),
        np.abs(h[-1, :] - benchmark.spec.north_head),
    ))
    return {
        "bc_mae_m": float(np.mean(errors)),
        "bc_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "bc_max_abs_m": float(np.max(errors)),
        "bc_p95_abs_m": float(np.percentile(errors, 95)),
    }


def _block_mean(array: np.ndarray, stride: int) -> np.ndarray:
    ny, nx = array.shape[-2:]
    if ny % stride or nx % stride:
        raise ValueError(f"stride {stride} does not divide grid {ny}x{nx}")
    shape = array.shape[:-2] + (ny // stride, stride, nx // stride, stride)
    return array.reshape(shape).mean(axis=(-3, -1))


def _block_sum(array: np.ndarray, stride: int) -> np.ndarray:
    ny, nx = array.shape[-2:]
    shape = array.shape[:-2] + (ny // stride, stride, nx // stride, stride)
    return array.reshape(shape).sum(axis=(-3, -1))


def _coarsened_grid(grid: Grid, stride: int) -> Grid:
    return Grid(
        grid.x.reshape(-1, stride).mean(axis=1),
        grid.y.reshape(-1, stride).mean(axis=1),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _make_plots(output_dir: Path, config: RunConfig, benchmark: BenchmarkData,
                eval_times: list[float], references: np.ndarray,
                predictions: np.ndarray, model_residual: np.ndarray,
                reference_residual: np.ndarray, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = benchmark.grid
    extent = [grid.x[0] - grid.dx / 2, grid.x[-1] + grid.dx / 2,
              grid.y[0] - grid.dy / 2, grid.y[-1] + grid.dy / 2]
    errors = np.abs(predictions - references)
    common_max = float(np.max(errors))
    if common_max == 0:
        common_max = 1e-12
    fig, axes = plt.subplots(2, len(eval_times), figsize=(3.4 * len(eval_times), 7), squeeze=False)
    for index, time in enumerate(eval_times):
        per_time_max = max(float(errors[index].max()), 1e-12)
        for row_index, vmax in ((0, common_max), (1, per_time_max)):
            image = axes[row_index, index].imshow(
                errors[index], origin="lower", extent=extent,
                cmap="magma_r", vmin=0, vmax=vmax,
            )
            fig.colorbar(image, ax=axes[row_index, index], shrink=0.74)
            axes[row_index, index].set_aspect("equal")
        axes[0, index].set_title(f"t={time:g} d\nmax={errors[index].max():.3f} m")
        axes[1, index].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("common scale\ny (m)")
    axes[1, 0].set_ylabel("per-time scale\ny (m)")
    fig.suptitle(f"{config.model_name}: absolute {benchmark.spec.code} head error (unclipped)")
    fig.tight_layout()
    fig.savefig(output_dir / "head_error_maps.png", dpi=180)
    plt.close(fig)

    limit = max(float(np.max(np.abs(model_residual))), float(np.max(np.abs(reference_residual))), 1e-12)
    residual_extent = [grid.x[1] - grid.dx / 2, grid.x[-2] + grid.dx / 2,
                       grid.y[1] - grid.dy / 2, grid.y[-2] + grid.dy / 2]
    fig, axes = plt.subplots(2, len(eval_times), figsize=(3.4 * len(eval_times), 7), squeeze=False)
    for index, time in enumerate(eval_times):
        for row_index, (values, label) in enumerate((
            (model_residual[index], config.model_name),
            (reference_residual[index], "MODFLOW baseline"),
        )):
            image = axes[row_index, index].imshow(
                values, origin="lower", extent=residual_extent,
                cmap="RdBu_r", vmin=-limit, vmax=limit,
            )
            rmse = np.sqrt(np.mean(values ** 2))
            axes[row_index, index].set_title(f"{label}, t={time:g}\nRMSE={rmse:.3g} m/d")
            axes[row_index, index].set_aspect("equal")
            fig.colorbar(image, ax=axes[row_index, index], shrink=0.74)
        axes[1, index].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("model\ny (m)")
    axes[1, 0].set_ylabel("MODFLOW\ny (m)")
    fig.suptitle(f"{benchmark.spec.code} governing-equation residual (shared, unclipped scale)")
    fig.tight_layout()
    fig.savefig(output_dir / "pde_residual_maps.png", dpi=180)
    plt.close(fig)

    times = np.asarray(eval_times)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(times, [row["rmse_m"] for row in rows], "o-", label="RMSE")
    axes[0, 0].plot(times, [row["mae_m"] for row in rows], "s-", label="MAE")
    axes[0, 0].set_ylabel("head error (m)")
    axes[0, 1].plot(times, [row["pde_overall_rmse_m_per_d"] for row in rows], "o-", label=config.model_name)
    axes[0, 1].plot(times, [row["modflow_pde_overall_rmse_m_per_d"] for row in rows], "s--", label="MODFLOW")
    axes[0, 1].set_ylabel("PDE residual RMSE (m/d)")
    axes[1, 0].plot(times, [row["north_inflow_m3_per_d"] for row in rows], "o-", label="north")
    axes[1, 0].plot(times, [row["south_inflow_m3_per_d"] for row in rows], "s-", label="south")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_ylabel("derived inflow (m3/d)")
    axes[1, 1].plot(times, [abs(row["mass_residual_percent_of_well"]) for row in rows], "o-")
    axes[1, 1].axhline(5, color="red", linestyle="--", label="5%")
    axes[1, 1].set_ylabel("mass-balance error (% of well)")
    for axis in axes.ravel():
        axis.set_xlabel("time (d)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"{config.model_name}: {benchmark.spec.code} diagnostic summary")
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostic_timeseries.png", dpi=180)
    plt.close(fig)


def run_diagnostics(config: RunConfig) -> dict[str, Any]:
    if config.test_start > config.test_end:
        raise ValueError("test_start must not exceed test_end")
    benchmark = load_benchmark(config)
    prediction_files = discover_snapshots(config.prediction_dir, config.prediction_glob)
    unknown_times = sorted(set(prediction_files) - set(benchmark.references))
    if unknown_times:
        raise ValueError(
            f"prediction times do not exist in {benchmark.spec.code}: "
            + ", ".join(f"t={time:g}" for time in unknown_times)
        )

    series_times = np.array(sorted(prediction_files), dtype=float)
    references = np.stack([benchmark.references[float(time)] for time in series_times])
    reference_snapshot_files = discover_snapshots(config.reference_dir)
    same_as_reference = all(
        prediction_files[float(time)].resolve()
        == reference_snapshot_files[float(time)].resolve()
        for time in series_times
    )
    if same_as_reference:
        predictions = references.copy()
    else:
        predictions = np.stack([
            _aligned_field(prediction_files[float(time)], benchmark.grid, config.matrix_y_order)
            for time in series_times
        ])
    eval_indices = np.flatnonzero(
        (series_times >= config.test_start) & (series_times <= config.test_end)
    )
    if not len(eval_indices):
        raise ValueError(
            f"no prediction snapshots in test window {config.test_start:g}-{config.test_end:g} d"
        )

    pred_dhdt, predecessor_times, predecessor_sources = _prediction_derivative(
        predictions, series_times, benchmark, config.prediction_mode
    )
    ref_dhdt = _reference_derivative(benchmark, series_times)
    pred_residual = _pde_residual(predictions, pred_dhdt, benchmark)
    ref_residual = _pde_residual(references, ref_dhdt, benchmark)
    masks = _zone_masks(
        benchmark, well_radius=config.well_zone_radius,
        rim_threshold=config.k_rim_threshold,
    )

    rows: list[dict[str, Any]] = []
    for horizon, index in enumerate(eval_indices, start=1):
        time = float(series_times[index])
        accuracy = _accuracy(references[index], predictions[index])
        model_stats = _residual_stats(pred_residual[index], masks)
        ref_stats = _residual_stats(ref_residual[index], masks)
        boundary = _boundary_error(predictions[index], benchmark)
        flux = _boundary_fluxes(predictions[index], benchmark)
        ref_flux = _boundary_fluxes(references[index], benchmark)
        storage = float(benchmark.spec.specific_yield * np.sum(pred_dhdt[index]) * benchmark.grid.area)
        ref_storage = float(benchmark.spec.specific_yield * np.sum(ref_dhdt[index]) * benchmark.grid.area)
        prescribed_inflow = flux["north_inflow_m3_per_d"] + flux["south_inflow_m3_per_d"]
        ref_prescribed_inflow = ref_flux["north_inflow_m3_per_d"] + ref_flux["south_inflow_m3_per_d"]
        mass_residual = storage - (prescribed_inflow + float(benchmark.well_rates.sum()))
        ref_mass_residual = ref_storage - (ref_prescribed_inflow + float(benchmark.well_rates.sum()))

        row: dict[str, Any] = {
            "horizon": horizon,
            "time_d": time,
            **accuracy,
            **boundary,
            **flux,
        }
        row.update({f"pde_{key}": value for key, value in model_stats.items()})
        row.update({f"modflow_pde_{key}": value for key, value in ref_stats.items()})
        row.update({
            "temporal_predecessor_time_d": predecessor_times[index],
            "temporal_predecessor_source": predecessor_sources[index],
            "modflow_north_inflow_m3_per_d": ref_flux["north_inflow_m3_per_d"],
            "modflow_south_inflow_m3_per_d": ref_flux["south_inflow_m3_per_d"],
            "north_gradient_sign_matches_modflow": int(
                np.sign(flux["north_inflow_m3_per_d"]) == np.sign(ref_flux["north_inflow_m3_per_d"])
            ),
            "south_gradient_sign_matches_modflow": int(
                np.sign(flux["south_inflow_m3_per_d"]) == np.sign(ref_flux["south_inflow_m3_per_d"])
            ),
            "storage_rate_m3_per_d": storage,
            "mass_residual_m3_per_d": mass_residual,
            "mass_residual_percent_of_well": 100.0 * mass_residual / abs(benchmark.spec.well_rate),
            "modflow_storage_rate_m3_per_d": ref_storage,
            "modflow_mass_residual_m3_per_d": ref_mass_residual,
            "modflow_mass_residual_percent_of_well": (
                100.0 * ref_mass_residual / abs(benchmark.spec.well_rate)
            ),
        })
        rows.append(row)

    pooled_reference = references[eval_indices].ravel()
    pooled_prediction = predictions[eval_indices].ravel()
    pooled_denominator = float(np.sum((pooled_reference - pooled_reference.mean()) ** 2))
    pde_model_mean = _mean(rows, "pde_overall_rmse_m_per_d")
    pde_ref_mean = _mean(rows, "modflow_pde_overall_rmse_m_per_d")
    summary = {
        "benchmark": {
            "code": benchmark.spec.code,
            "name": benchmark.spec.name,
            "grid": f"{benchmark.grid.ny}x{benchmark.grid.nx}",
            "dx_m": benchmark.grid.dx,
            "dy_m": benchmark.grid.dy,
            "south_head_m": benchmark.spec.south_head,
            "north_head_m": benchmark.spec.north_head,
            "k_min_m_per_d": float(benchmark.conductivity.min()),
            "k_max_m_per_d": float(benchmark.conductivity.max()),
            "well_rate_m3_per_d": float(benchmark.well_rates.sum()),
            "well_x_m": benchmark.well_x,
            "well_y_m": benchmark.well_y,
        },
        "model": {
            "name": config.model_name,
            "prediction_mode": config.prediction_mode,
            "prediction_dir": str(config.prediction_dir.resolve()),
            "prediction_glob": config.prediction_glob,
        },
        "evaluation": {
            "times": [float(series_times[index]) for index in eval_indices],
            "available_prediction_times": series_times.tolist(),
            "temporal_scheme": "mode-aware backward difference",
            "prediction_predecessor_sources": sorted(set(predecessor_sources)),
            "spatial_operator": benchmark.spec.spatial_operator,
        },
        "accuracy": {
            "mean_mse_m2": _mean(rows, "mse_m2"),
            "mean_mae_m": _mean(rows, "mae_m"),
            "mean_rmse_m": _mean(rows, "rmse_m"),
            "mean_rrmse_percent": _mean(rows, "rrmse_percent"),
            "mean_r2": _mean(rows, "r2"),
            "pooled_r2": 1.0 - float(np.sum((pooled_prediction - pooled_reference) ** 2)) / pooled_denominator,
            "max_abs_m": float(max(row["max_abs_m"] for row in rows)),
        },
        "pde_residual": {
            "model_mean_rmse_m_per_d": pde_model_mean,
            "modflow_mean_rmse_m_per_d": pde_ref_mean,
            "model_to_modflow_rmse_ratio": pde_model_mean / pde_ref_mean,
            "model_background_rmse_m_per_d": _mean(rows, "pde_background_rmse_m_per_d"),
            "model_well_rmse_m_per_d": _mean(rows, "pde_well_rmse_m_per_d"),
        },
        "boundary": {
            "mean_bc_mae_m": _mean(rows, "bc_mae_m"),
            "mean_bc_rmse_m": _mean(rows, "bc_rmse_m"),
            "max_bc_error_m": float(max(row["bc_max_abs_m"] for row in rows)),
            "north_gradient_sign_match_fraction": _mean(rows, "north_gradient_sign_matches_modflow"),
            "south_gradient_sign_match_fraction": _mean(rows, "south_gradient_sign_matches_modflow"),
            "mean_west_no_flow_rmse_m2_per_d": _mean(rows, "west_no_flow_rmse_m2_per_d"),
            "mean_east_no_flow_rmse_m2_per_d": _mean(rows, "east_no_flow_rmse_m2_per_d"),
        },
        "mass_balance": {
            "mean_abs_residual_m3_per_d": float(np.mean(np.abs([
                row["mass_residual_m3_per_d"] for row in rows
            ]))),
            "max_abs_residual_m3_per_d": float(max(abs(row["mass_residual_m3_per_d"]) for row in rows)),
            "mean_abs_percent_of_well": float(np.mean(np.abs([
                row["mass_residual_percent_of_well"] for row in rows
            ]))),
            "modflow_mean_abs_percent_of_well_same_operator": float(np.mean(np.abs([
                row["modflow_mass_residual_percent_of_well"] for row in rows
            ]))),
        },
    }

    sensitivity_rows: list[dict[str, Any]] = []
    for stride in config.grid_strides:
        if benchmark.grid.nx % stride or benchmark.grid.ny % stride:
            continue
        coarse_grid = _coarsened_grid(benchmark.grid, stride)
        coarse_benchmark = BenchmarkData(
            spec=benchmark.spec,
            grid=coarse_grid,
            references={},
            conductivity=_block_mean(benchmark.conductivity, stride),
            well_rates=_block_sum(benchmark.well_rates, stride),
            well_x=benchmark.well_x,
            well_y=benchmark.well_y,
        )
        coarse_predictions = _block_mean(predictions, stride)
        coarse_references = _block_mean(references, stride)
        coarse_pred_dt = _block_mean(pred_dhdt, stride)
        coarse_ref_dt = _block_mean(ref_dhdt, stride)
        model_r = _pde_residual(coarse_predictions, coarse_pred_dt, coarse_benchmark)
        reference_r = _pde_residual(coarse_references, coarse_ref_dt, coarse_benchmark)
        selected_model = model_r[eval_indices]
        selected_reference = reference_r[eval_indices]
        sensitivity_rows.append({
            "stride": stride,
            "grid_nx": coarse_grid.nx,
            "grid_ny": coarse_grid.ny,
            "dx_m": coarse_grid.dx,
            "model_mean_pde_rmse_m_per_d": float(np.mean(np.sqrt(np.mean(selected_model ** 2, axis=(1, 2))))),
            "modflow_mean_pde_rmse_m_per_d": float(np.mean(np.sqrt(np.mean(selected_reference ** 2, axis=(1, 2))))),
        })

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.output_dir / "per_time_metrics.csv", rows)
    _write_csv(config.output_dir / "grid_sensitivity.csv", sensitivity_rows)
    result = {
        "config": _jsonable(asdict(config)),
        "summary": summary,
        "per_time": rows,
        "grid_sensitivity": sensitivity_rows,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    if config.make_plots:
        selected = eval_indices.tolist()
        _make_plots(
            config.output_dir, config, benchmark,
            [float(series_times[index]) for index in selected],
            references[selected], predictions[selected],
            pred_residual[selected], ref_residual[selected], rows,
        )
    return result


def print_summary(result: dict[str, Any], output_dir: Path) -> None:
    summary = result["summary"]
    rows = result["per_time"]
    mode = summary["model"]["prediction_mode"]

    if mode == "rollout":
        heading = "=== Autoregressive Rollout Metrics ==="
        row_label = lambda row: f"h={int(row['horizon'])}"
        final_label = f"h=1-{len(rows)}"
    else:
        heading = f"=== {mode.title()} Metrics by Target Time ==="
        row_label = lambda row: f"t={row['time_d']:g} d"
        final_label = f"{len(rows)} snapshots"

    if mode == "rollout":
        test_mse = float(rows[0]["mse_m2"])
        test_rmse = float(rows[0]["rmse_m"])
        test_mae = float(rows[0]["mae_m"])
        test_r2 = float(rows[0]["r2"])
    else:
        test_mse = float(np.mean([float(row["mse_m2"]) for row in rows]))
        test_rmse = float(np.sqrt(test_mse))
        test_mae = float(np.mean([float(row["mae_m"]) for row in rows]))
        test_r2 = float(summary["accuracy"]["pooled_r2"])

    lines = [
        f"{summary['benchmark']['code']} diagnostics complete: {summary['model']['name']}",
        "",
        "=== Test Set Performance ===",
        f"MSE:  {test_mse:.6f} m2",
        f"RMSE: {test_rmse:.3f} m",
        f"MAE:  {test_mae:.3f} m",
        f"R2:   {test_r2:.6f}",
        "",
        heading,
    ]
    for row in rows:
        lines.append(
            f"  {row_label(row):<9}: "
            f"MAE={row['mae_m']:.3f} m, "
            f"RMSE={row['rmse_m']:.3f} m, "
            f"RRMSE={row['rrmse_percent']:.3f}%, "
            f"R2={row['r2']:.6f}"
        )

    values = {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in ("mae_m", "rmse_m", "rrmse_percent", "r2")
    }

    def mean_metric(key: str) -> float:
        metric_values = np.asarray([float(row[key]) for row in rows], dtype=float)
        finite = metric_values[np.isfinite(metric_values)]
        return float(finite.mean()) if finite.size else float("nan")

    def gradient_match(value: Any) -> str:
        return "yes" if bool(value) else "no"

    lines.extend([
        "",
        f"--- FINAL TEST RESULTS ({final_label}) ---",
        f"Average MAE:    {values['mae_m'].mean():.3f} m",
        f"Average RMSE:   {values['rmse_m'].mean():.3f} m",
        f"Average RRMSE:  {values['rrmse_percent'].mean():.3f} %",
        f"Average R2:     {values['r2'].mean():.6f}",
        f"Std MAE:        {values['mae_m'].std():.3f} m",
        f"Std RMSE:       {values['rmse_m'].std():.3f} m",
        f"Std RRMSE:      {values['rrmse_percent'].std():.3f} %",
        f"Std R2:         {values['r2'].std():.6f}",
        "------------------------------------",
        "",
        "=== PDE Residual Evaluation ===",
        (
            f"Grid: {summary['benchmark']['grid']}, "
            f"dx={summary['benchmark']['dx_m']:.2f} m, "
            f"dy={summary['benchmark']['dy_m']:.2f} m"
        ),
        "R = Sy*dh/dt - div(K*h*grad(h)) - source",
        "Residual statistics in the table are in m/d.",
        "",
        (
            f"{'Forecast':<10} | {'Time':>6} | {'Model RMSE':>10} | "
            f"{'MODFLOW':>10} | {'BG Mean|R|':>10} | {'BG RMSE':>9} | "
            f"{'Well RMSE':>9} | {'Max |R|':>9}"
        ),
        "-" * 101,
    ])

    for row in rows:
        lines.append(
            f"{row_label(row):<10} | {row['time_d']:>6.1f} | "
            f"{row['pde_overall_rmse_m_per_d']:>10.4f} | "
            f"{row['modflow_pde_overall_rmse_m_per_d']:>10.4f} | "
            f"{row['pde_background_mean_abs_m_per_d']:>10.4f} | "
            f"{row['pde_background_rmse_m_per_d']:>9.4f} | "
            f"{row['pde_well_rmse_m_per_d']:>9.4f} | "
            f"{row['pde_overall_max_abs_m_per_d']:>9.2f}"
        )

    lines.extend([
        "-" * 101,
        (
            f"{'MEAN':<10} | {'':>6} | "
            f"{summary['pde_residual']['model_mean_rmse_m_per_d']:>10.4f} | "
            f"{summary['pde_residual']['modflow_mean_rmse_m_per_d']:>10.4f} | "
            f"{mean_metric('pde_background_mean_abs_m_per_d'):>10.4f} | "
            f"{summary['pde_residual']['model_background_rmse_m_per_d']:>9.4f} | "
            f"{summary['pde_residual']['model_well_rmse_m_per_d']:>9.4f} | "
            f"{max(float(row['pde_overall_max_abs_m_per_d']) for row in rows):>9.2f}"
        ),
        "",
        "=== Boundary, Flux, Storage, and Water Balance by Forecast ===",
        "Sign convention: positive boundary flux=inflow, negative=outflow",
        "Storage rate, boundary flux, and R_MB are in m3/d.",
    ])

    for row in rows:
        prescribed_flux = (
            float(row["north_inflow_m3_per_d"])
            + float(row["south_inflow_m3_per_d"])
        )
        lines.extend([
            (
                f"  {row_label(row):<9} (t={row['time_d']:g} d, "
                f"from t={row['temporal_predecessor_time_d']:g} "
                f"{row['temporal_predecessor_source']}): "
                f"BC MAE/RMSE/max={row['bc_mae_m']:.4f}/"
                f"{row['bc_rmse_m']:.4f}/{row['bc_max_abs_m']:.4f} m"
            ),
            (
                f"             Storage={row['storage_rate_m3_per_d']:+,.1f} | "
                f"boundary inflow={prescribed_flux:+,.1f} | "
                f"R_MB={row['mass_residual_m3_per_d']:+,.1f} m3/d | "
                f"relative={row['mass_residual_percent_of_well']:+.2f}%"
            ),
            (
                f"             flux N/S={row['north_inflow_m3_per_d']:+,.1f}/"
                f"{row['south_inflow_m3_per_d']:+,.1f} m3/d | "
                f"MODFLOW N/S={row['modflow_north_inflow_m3_per_d']:+,.1f}/"
                f"{row['modflow_south_inflow_m3_per_d']:+,.1f} m3/d | "
                f"gradient match={gradient_match(row['north_gradient_sign_matches_modflow'])}/"
                f"{gradient_match(row['south_gradient_sign_matches_modflow'])}"
            ),
            (
                f"             derived flux W/E={row['west_derived_inflow_m3_per_d']:+,.1f}/"
                f"{row['east_derived_inflow_m3_per_d']:+,.1f} m3/d | "
                f"no-flow RMSE W/E={row['west_no_flow_rmse_m2_per_d']:.4f}/"
                f"{row['east_no_flow_rmse_m2_per_d']:.4f} m2/d | "
                f"MODFLOW R_MB={row['modflow_mass_residual_percent_of_well']:+.2f}%"
            ),
        ])

    lines.extend([
        "",
        "--- BOUNDARY CONDITIONS ---",
        f"Mean boundary MAE:       {summary['boundary']['mean_bc_mae_m']:.6f} m",
        f"Mean boundary RMSE:      {summary['boundary']['mean_bc_rmse_m']:.6f} m",
        f"Maximum boundary error: {summary['boundary']['max_bc_error_m']:.6f} m",
        (
            "Gradient sign agreement (N/S): "
            f"{100.0 * summary['boundary']['north_gradient_sign_match_fraction']:.1f}%/"
            f"{100.0 * summary['boundary']['south_gradient_sign_match_fraction']:.1f}%"
        ),
        "",
        "--- PDE RESIDUAL ---",
        f"Model overall RMSE:      {summary['pde_residual']['model_mean_rmse_m_per_d']:.4f} m/d",
        f"MODFLOW operator RMSE:   {summary['pde_residual']['modflow_mean_rmse_m_per_d']:.4f} m/d",
        f"Model/MODFLOW ratio:     {summary['pde_residual']['model_to_modflow_rmse_ratio']:.3f}",
        f"Well-zone RMSE:          {summary['pde_residual']['model_well_rmse_m_per_d']:.4f} m/d",
        *([
            f"Lens-rim RMSE:           {mean_metric('pde_lens_rim_rmse_m_per_d'):.4f} m/d",
            f"Lens-core RMSE:          {mean_metric('pde_lens_core_rmse_m_per_d'):.4f} m/d",
        ] if 'pde_lens_rim_rmse_m_per_d' in rows[0] else []),
        f"Background RMSE:         {summary['pde_residual']['model_background_rmse_m_per_d']:.4f} m/d",
        "",
        "--- BOUNDARY FLUXES ---",
        "Sign convention: positive=inflow, negative=outflow",
        (
            "Mean model north/south:  "
            f"{mean_metric('north_inflow_m3_per_d'):+,.1f}/"
            f"{mean_metric('south_inflow_m3_per_d'):+,.1f} m3/d"
        ),
        (
            "Mean MODFLOW north/south: "
            f"{mean_metric('modflow_north_inflow_m3_per_d'):+,.1f}/"
            f"{mean_metric('modflow_south_inflow_m3_per_d'):+,.1f} m3/d"
        ),
        (
            "Mean derived west/east:  "
            f"{mean_metric('west_derived_inflow_m3_per_d'):+,.1f}/"
            f"{mean_metric('east_derived_inflow_m3_per_d'):+,.1f} m3/d"
        ),
        (
            "Mean no-flow RMSE W/E:   "
            f"{summary['boundary']['mean_west_no_flow_rmse_m2_per_d']:.4f}/"
            f"{summary['boundary']['mean_east_no_flow_rmse_m2_per_d']:.4f} m2/d"
        ),
        "",
        "--- WATER BALANCE ---",
        f"Prescribed well rate:    {summary['benchmark']['well_rate_m3_per_d']:+,.1f} m3/d",
        f"Mean model storage rate: {mean_metric('storage_rate_m3_per_d'):+,.1f} m3/d",
        f"Mean MODFLOW storage:    {mean_metric('modflow_storage_rate_m3_per_d'):+,.1f} m3/d",
        f"Mean absolute residual:  {summary['mass_balance']['mean_abs_residual_m3_per_d']:,.1f} m3/d",
        f"Maximum absolute residual: {summary['mass_balance']['max_abs_residual_m3_per_d']:,.1f} m3/d",
        (
            "Mean relative mass-balance error: "
            f"{summary['mass_balance']['mean_abs_percent_of_well']:.2f}% "
            "(of |Q_well|)"
        ),
        (
            "MODFLOW same-operator baseline: "
            f"{summary['mass_balance']['modflow_mean_abs_percent_of_well_same_operator']:.2f}%"
        ),
        "",
        f"Full per-time metrics: {(output_dir / 'per_time_metrics.csv').resolve()}",
        f"Outputs: {output_dir.resolve()}",
    ])

    report = "\n".join(lines)
    print(report)
    (output_dir / "accuracy_report.txt").write_text(report + "\n", encoding="utf-8")
    (output_dir / "diagnostic_report.txt").write_text(report + "\n", encoding="utf-8")
