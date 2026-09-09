"""Filesystem locations for benchmark data and diagnostic exports.

The repository holds code only. The MODFLOW models, exported head snapshots,
trained checkpoints and diagnostic exports are large and are distributed
separately; see the Data availability section of the README.

Point ``GW_DATA`` at the directory holding that material::

    export GW_DATA=/path/to/gw_surrogate_data      # Linux, macOS
    set GW_DATA=D:\\gw_surrogate_data              # Windows

If ``GW_DATA`` is unset, ``<repository>/data`` is used, so a copy or symlink
placed there needs no configuration at all.

Expected layout under the data root::

    benchmarks/b1/          MODFLOW model and t*.txt snapshots
    benchmarks/b2/          as above, plus kfield.txt and b2_model.wel
    diagnostic_results/     one directory per evaluated arm
    predictions/            surrogate head fields, absolute metres
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ENV_VAR = "GW_DATA"


def data_root() -> Path:
    """Root of the separately distributed data tree."""
    override = os.environ.get(ENV_VAR)
    return Path(override).expanduser() if override else REPO / "data"


def benchmark_dir(benchmark: str) -> Path:
    """Reference directory for ``benchmark``, one of 'b1' or 'b2'."""
    if benchmark not in ("b1", "b2"):
        raise ValueError(f"benchmark must be 'b1' or 'b2', got {benchmark!r}")
    return data_root() / "benchmarks" / benchmark


def diagnostic_results() -> Path:
    """Directory holding one export per evaluated arm."""
    return data_root() / "diagnostic_results"


def predictions() -> Path:
    """Directory holding surrogate head fields, in absolute metres."""
    return data_root() / "predictions"


def require(path: Path, what: str) -> Path:
    """Return ``path``, or raise with the configuration hint if it is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. Set {ENV_VAR} to the data root, or "
            f"place the data under {REPO / 'data'}. See the README."
        )
    return path
