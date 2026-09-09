"""
Minimal stand-in for `pyDOE.lhs`, vendored so this tree needs no install.

Why vendored rather than pip-installed: `pip install pyDOE` now resolves to the
repackaged `pydoe`, which requires newer numpy and scipy and therefore upgrades
them *inside a running kernel*. On Colab that leaves a half-swapped numpy ABI
and the next scipy import dies with

    AttributeError: module 'numpy._core._multiarray_umath'
                    has no attribute '_blas_supports_fpe'

The only thing this project uses from the package is `lhs`, which is a few lines.
Vendoring it removes the dependency, and with it the upgrade and the restart.

`sampler.py` imports `pyDOE` first and falls back to `pydoe`. Because this module
sits in the tree root, and the trainer chdir's there and puts it on sys.path, it
wins that import -- so the classic behaviour below is what runs.

Behaviour matches classic `pyDOE._lhsclassic` with `criterion=None`: stratified
samples on [0, 1), one random permutation per column, drawn from **numpy's global
RNG**. That is what makes `np.random.seed(args.seed)` in test.py reproducible,
which the modern `pydoe` rewrite breaks by using its own default_rng.
"""
import numpy as np

__all__ = ["lhs"]


def lhs(n, samples=None, criterion=None, iterations=None):
    """Latin hypercube sample, shape (samples, n), values in [0, 1).

    Parameters mirror pyDOE. `criterion` and `iterations` are accepted for
    signature compatibility and ignored: this project only ever calls the
    default, unoptimised variant.
    """
    if samples is None:
        samples = n
    if criterion is not None:
        raise NotImplementedError(
            f"vendored lhs supports criterion=None only, got {criterion!r}")

    cut = np.linspace(0, 1, samples + 1)
    lo, hi = cut[:samples], cut[1:samples + 1]

    u = np.random.rand(samples, n)
    points = np.zeros_like(u)
    for j in range(n):
        points[:, j] = u[:, j] * (hi - lo) + lo

    out = np.zeros_like(points)
    for j in range(n):
        out[:, j] = points[np.random.permutation(samples), j]
    return out


if __name__ == "__main__":
    np.random.seed(0)
    a = lhs(3, samples=5)
    np.random.seed(0)
    b = lhs(3, samples=5)
    assert a.shape == (5, 3)
    assert np.array_equal(a, b), "not reproducible under np.random.seed"
    assert (a >= 0).all() and (a < 1).all()
    # one sample per stratum in every column
    for j in range(3):
        strata = np.floor(a[:, j] * 5).astype(int)
        assert sorted(strata) == list(range(5)), "strata not covered exactly once"
    print("vendored lhs OK:", a.shape, "reproducible, stratified")
